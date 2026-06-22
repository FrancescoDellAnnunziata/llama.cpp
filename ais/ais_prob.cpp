#include "llama.h"
#include "common.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include "httplib.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <mutex>
#include <memory>
#include <random>
#include <chrono>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using ordered_json = nlohmann::ordered_json;

// ==========================================
// RAII deleters for the llama C handles (free on scope exit / error return)
// ==========================================
struct LlamaModelDel   { void operator()(llama_model*   p) const { if (p) llama_model_free(p);   } };
struct LlamaCtxDel     { void operator()(llama_context* p) const { if (p) llama_free(p);          } };
struct LlamaSamplerDel { void operator()(llama_sampler* p) const { if (p) llama_sampler_free(p);  } };
using ModelPtr   = std::unique_ptr<llama_model,   LlamaModelDel>;
using CtxPtr     = std::unique_ptr<llama_context, LlamaCtxDel>;
using SamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDel>;

// ==========================================
// ENV helpers (DRY): one-shot getenv parsing with a default.
// NB: most call sites intentionally cache via `static const ... = []{...}()` so
// the env is read exactly once — keep those; these helpers replace the ad-hoc
// `getenv(k) ? atox(getenv(k)) : d` idiom (which calls getenv twice).
// ==========================================
static int    env_int  (const char* k, int    d) { const char* v = getenv(k); return v ? atoi(v) : d; }
static double env_float (const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }

// ==========================================
// CONFIG
// ==========================================
struct AISProbConfig {
    float surprise_threshold = 3.0f;
    int   anchor_size        = 64;
    int   recent_size        = 128;
    int   ctx_limit          = 32768;
};

static const int   MIN_DOC_SIZE       = 1000;
static const float NEIGHBOR_THRESHOLD = 8.0f;
static const int   NEIGHBOR_WIN       = 4;  // wider window protects XML tag structure around high-surprise identifiers

// ==========================================
// SURPRISE
// ==========================================
// True Shannon surprise: -log2(softmax(logits)[tok]). Costs an O(n_vocab)
// max-reduction PLUS an O(n_vocab) exp() accumulation per token. The exp() loop
// is the real hot spot (~256k exp/token × N tokens ≈ 695M exp on the bench).
// Kept for reference / A-B validation; no longer on the hot path.
[[maybe_unused]] static float compute_surprise(const float* logits, llama_token tok, int n_vocab) {
    float max_l = *std::max_element(logits, logits + n_vocab);
    double log_z = 0.0;
    for (int i = 0; i < n_vocab; i++)
        log_z += std::exp((double)(logits[i] - max_l));
    log_z = std::log(log_z) + (double)max_l;
    return -(logits[tok] - (float)log_z) / std::log(2.0f);
}

// Fast surprise (max-margin approximation): (max_logit - logits[tok]) / ln2.
// Drops the exp() accumulation — the dominant cost — keeping only the cheap
// max-reduction (float comparisons auto-vectorize, ~50× cheaper than exp()).
// It approximates true surprise from below: the gap is the distribution
// "spread" term (logsumexp - max), which is small for peaked distributions and
// far less position-dependent than -logits[tok] alone. Crucially it stays in
// BITS, so every threshold tuned against true surprise keeps its meaning: the
// 2.5-bit dense guard, the 2.5-bit floor, the 8.0-bit neighbor threshold, and
// the relative sigma-mk avg±k·σ bands.
// Horizontal max over the logit vector. std::max_element / a scalar std::max
// loop won't vectorize: IEEE float-max has a loop-carried dependency clang
// won't reassociate without -ffast-math, so it ran scalar (~19 cyc/elem, 3.2s
// over 256k×N on the bench). Explicit NEON lanes make it memory-bound (<0.2s).
static inline float logit_max(const float* logits, int n_vocab) {
#if defined(__ARM_NEON)
    int i = 0;
    float32x4_t m0 = vdupq_n_f32(-INFINITY), m1 = m0, m2 = m0, m3 = m0;
    for (; i + 16 <= n_vocab; i += 16) {
        m0 = vmaxq_f32(m0, vld1q_f32(logits + i));
        m1 = vmaxq_f32(m1, vld1q_f32(logits + i + 4));
        m2 = vmaxq_f32(m2, vld1q_f32(logits + i + 8));
        m3 = vmaxq_f32(m3, vld1q_f32(logits + i + 12));
    }
    float max_l = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    for (; i < n_vocab; ++i) max_l = std::max(max_l, logits[i]);
    return max_l;
#else
    return *std::max_element(logits, logits + n_vocab);
#endif
}

// Top-K surprise: logsumexp sui SOLI top-K logit. Misurato (PROJECT_GUIDE 5-bis):
// la partition function Z è dominata dalla testa (top-50 ≈ 99.995% di Z, errore
// ≤0.016 bit vs full) → ~esatta. A differenza del max-margin (top-1) resta FEDELE sui
// token high-surprise (dove il solo max sbaglia di ~1.3 bit e li butta sotto soglia).
// Costo: 1 scan + min-heap di K (no full exp loop). Richiede comunque i logit (lm_head).
static float topk_surprise(const float* logits, llama_token tok, int n_vocab, int K) {
    static thread_local std::vector<float> heap;  // min-heap dei top-K (riusato)
    heap.clear();
    for (int v = 0; v < n_vocab; v++) {
        float z = logits[v];
        if ((int)heap.size() < K) {
            heap.push_back(z);
            std::push_heap(heap.begin(), heap.end(), std::greater<float>());
        } else if (z > heap.front()) {            // front = il più piccolo dei top-K
            std::pop_heap(heap.begin(), heap.end(), std::greater<float>());
            heap.back() = z;
            std::push_heap(heap.begin(), heap.end(), std::greater<float>());
        }
    }
    float mx = -INFINITY;
    for (float z : heap) if (z > mx) mx = z;
    double se = 0.0;
    for (float z : heap) se += std::exp((double)(z - mx));
    return (float)(((double)mx + std::log(se) - (double)logits[tok]) / std::log(2.0));
}

// Nucleus / top-p surprise: -log2(softmax[id]) calcolato dalla SOLA testa (nucleo) della
// distribuzione. Z = Σ exp(l−max) sui SOLI logit entro GATE bit da max; la coda vale
// exp(−GATE)≈0 → si SALTA l'exp (non si calcola ciò che non conta). Una passata NEON per il
// max + una passata di confronto con exp RARO (solo il nucleo) → niente heap, niente full-exp.
// È la "surprise dopo la softmax" ristretta al top-p: faithful (coda trascurabile) e snella.
// GATE via AIS_COT_GATE (default 20 → exp(−20)≈2e-9). Con greedy id=argmax ⇒ surprise = −log2(p_max).
static inline double topp_surprise(const float* lg, llama_token id, int n_vocab) {
    static const float GATE = [](){ const char* v = getenv("AIS_COT_GATE"); return v ? (float)atof(v) : 20.0f; }();
    const float  mx  = logit_max(lg, n_vocab);
    const float  thr = mx - GATE;
    double z = 0.0;
    for (int i = 0; i < n_vocab; i++) if (lg[i] > thr) z += std::exp((double)(lg[i] - mx));
    return ((double)mx + std::log(z) - (double)lg[id]) / std::log(2.0);
}

// CoT-cut surprise (bit) del token campionato. Con greedy id=argmax il max-margin è degenere
// (=0), quindi serve il logsumexp. DEFAULT = nucleo (top-p, topp_surprise): salta la coda.
// AIS_COT_TOPK=K forza il vecchio metodo a heap (top-K) per A/B.
static inline double cot_surprise(const float* lg, llama_token id, int n_vocab) {
    static const int K = [](){ const char* v = getenv("AIS_COT_TOPK"); return v ? std::max(1, atoi(v)) : 0; }();
    if (K > 0) return (double) topk_surprise(lg, id, n_vocab, K);
    return topp_surprise(lg, id, n_vocab);
}

// Tetto al CoT (token di pensiero): forza il taglio quando il "thought" diventa troppo lungo,
// così resta SEMPRE budget per la RISPOSTA (un pensiero che esaurisce max_tokens non emette la
// risposta → item sbagliato + tempo sprecato). Accurato + veloce. Due forme combinabili:
//   AIS_COT_MAXTHINK=N        tetto ASSOLUTO in token di pensiero
//   AIS_COT_MAXTHINK_FRAC=f   tetto = f·max_tokens (AUTO-SCALA per cap/bench: garantisce (1−f) alla risposta)
// 0 = off. Se entrambi attivi → si usa il minore.
static int cot_think_cap(int max_tokens) {
    static const int    abs_t = env_int  ("AIS_COT_MAXTHINK",      0);
    static const double frac  = env_float("AIS_COT_MAXTHINK_FRAC", 0.0);
    int cap = (abs_t > 0) ? abs_t : 0;
    if (frac > 0.0) { int f = (int)(frac * max_tokens); cap = (cap > 0) ? std::min(cap, f) : f; }
    return cap;   // >0 = attivo (taglia a `cap` token di pensiero)
}

static float fast_surprise(const float* logits, llama_token tok, int n_vocab) {
    static const bool use_true = getenv("AIS_TRUE_SURPRISE") != nullptr; // A/B: true Shannon
    static const int  topk = [](){ const char* v = getenv("AIS_TOPK_SURPRISE"); return v ? atoi(v) : 0; }();
    if (use_true)  return compute_surprise(logits, tok, n_vocab);        // esatto (full Z)
    if (topk > 0)  return topk_surprise(logits, tok, n_vocab, topk);     // top-K (≈esatto, fedele)
    return (logit_max(logits, n_vocab) - logits[tok]) / std::log(2.0f);  // max-margin (default, lossy su high-surprise)
}

static void batch_add(llama_batch& b, llama_token id, llama_pos pos, bool logits) {
    b.token   [b.n_tokens]    = id;
    b.pos     [b.n_tokens]    = pos;
    b.n_seq_id[b.n_tokens]    = 1;
    b.seq_id  [b.n_tokens][0] = 0;
    b.logits  [b.n_tokens]    = logits ? 1 : 0;
    b.n_tokens++;
}

static std::string random_id() {
    // thread_local: the server only happens to serialize callers via `mtx` today;
    // making the RNG per-thread removes the latent data race if that ever changes.
    static thread_local std::mt19937 rng(std::random_device{}());
    static constexpr char hex[] = "0123456789abcdef";
    std::string s(8, '0');
    for (auto& c : s) c = hex[rng() % 16];
    return s;
}

// Scan only the freshly appended tail of `s` for `needle`, with a back-overlap of
// (needle_len-1) so a match straddling the previous append boundary isn't missed.
// O(appended) per call instead of O(s.size()) — avoids O(n²) over a long stream.
static bool tail_contains(const std::string& s, int appended, const char* needle, size_t needle_len) {
    const size_t app  = appended > 0 ? (size_t)appended : 0;
    const size_t back = needle_len > 1 ? needle_len - 1 : 0;
    const size_t from = (s.size() > app + back) ? s.size() - app - back : 0;
    return s.find(needle, from) != std::string::npos;
}

// AIS_IGNORE_EOS: benchmarking only — keep generating to max_tokens (ignore end-of-generation)
// so throughput (tok/s) is measured over a fixed token count, comparable across backends.
static bool ais_ignore_eos() { static const bool v = getenv("AIS_IGNORE_EOS") != nullptr; return v; }
static bool ais_is_eog(const llama_vocab* vocab, llama_token id) {
    return !ais_ignore_eos() && llama_vocab_is_eog(vocab, id);
}

// n_batch (logical batch). The per-token logits buffer llama reserves is
// n_outputs_max × n_vocab × 2 (logits+probs), and n_outputs_max ≤ n_batch — so
// lowering this shrinks AIS's Pass-1 memory overhead (~4GB→~1GB at 512). Caveat:
// it changes Metal float-accumulation in the KV → a different (not worse) greedy
// path, so it breaks exact-reproducibility vs the 2048 baseline. Default 2048.
static int ais_n_batch() {
    static int v = getenv("AIS_NBATCH") ? std::max(1, atoi(getenv("AIS_NBATCH"))) : 2048;
    return v;
}

// Map a --cache-type-k/v string to a ggml_type (subset llama.cpp accepts for KV).
// Defaults to f16 (the historical ais_prob behavior) on anything unrecognized.
static ggml_type kv_type_from_str(const std::string& s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, " Unsupported cache type '%s', falling back to f16\n", s.c_str());
    return GGML_TYPE_F16;
}

// ==========================================
// DEDUP PRE-GATE (adattivo) — rimuove n-gram ESATTI ripetuti PRIMA del forward-pass.
// Sicuro: tocca solo ripetizioni esatte (la 2a occorrenza non aggiunge nulla); i token
// unici (codici, risultati, needle) non si ripetono → non vengono mai tolti. Anchor e
// recent sempre tenuti. È prefix-deterministico (la scelta su i dipende solo da ≤i) →
// compatibile col delta path. Costa zero forward-pass: scala col contenuto del doc.
// ==========================================
static float quick_xml_density(const llama_vocab* vocab, const std::vector<llama_token>& raw) {
    long ang = 0; char p[64];
    for (llama_token t : raw) {
        int n = llama_token_to_piece(vocab, t, p, (int)sizeof(p) - 1, 0, true);
        for (int i = 0; i < n; i++) if (p[i] == '<' || p[i] == '>') { ang++; break; }
    }
    return raw.empty() ? 0.0f : (float)ang / raw.size();
}

static std::vector<llama_token> dedup_prefilter(const std::vector<llama_token>& raw,
                                                int anchor, int recent, int ngram) {
    const int N = (int)raw.size();
    std::vector<llama_token> out; out.reserve(N);
    std::unordered_set<uint64_t> seen;
    std::vector<llama_token> hist;
    for (int i = 0; i < N; i++) {
        if (i < anchor || i >= N - recent) { out.push_back(raw[i]); continue; }
        hist.push_back(raw[i]);
        bool drop = false;
        if ((int)hist.size() >= ngram) {
            uint64_t h = 14695981039346656037ULL;
            int s = (int)hist.size() - ngram;
            for (int k = 0; k < ngram; k++) { h ^= (uint64_t)hist[s + k]; h *= 1099511628211ULL; }
            if (seen.count(h)) drop = true; else seen.insert(h);
        }
        if (!drop) out.push_back(raw[i]);
        if (hist.size() > 4096) hist.erase(hist.begin(), hist.begin() + 2048);
    }
    return out;
}

// ==========================================
// LEXICAL SURPRISE (surface-form, ZERO forward-pass) + NEIGHBOR GATE.
// Idea: identifica i token "informativi" dalla FORMA (maiuscole, cifre, simboli, CamelCase,
// codici) e droppa SOLO i token a bassa sorpresa lessicale che NON sono al CONFINE di un
// token ad alta sorpresa (finestra ±win). In CODICE quasi ogni token è adiacente a simboli/
// identificatori → protetto (conservativo, sicuro); in PROSA/commenti i run di stop-word
// isolati cadono. Model-agnostic (solo testo del token) → "funziona con tutti i modelli".
// ==========================================
// Un token è "importante" (alta sorpresa lessicale)? In CODE MODE (syntax-aware) anche i
// token STRUTTURALI — qualsiasi simbolo/operatore/bracket/underscore — contano come
// importanti: in codice quasi ogni riga ha operatori → il neighbor-gate protegge keyword e
// identificatori snake_case che la sola sorpresa lessicale (cifre/maiuscole) mancherebbe.
static bool lex_is_important(const llama_vocab* vocab, llama_token id, float thr, bool code_mode) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, (int)sizeof(buf), 0, true);
    if (n <= 0) return false;
    bool up = false, dig = false, sym = false; int alnum = 0;
    for (int i = 0; i < n; i++) {
        const unsigned char c = (unsigned char) buf[i];   // unsigned: niente UB su byte UTF-8 ≥128
        if (isupper(c)) up = true;
        if (isdigit(c)) dig = true;
        if (ispunct(c)) sym = true;          // include _ ( ) [ ] { } . , : ; * + - / = < > ...
        if (isalnum(c)) alnum++;
    }
    if (code_mode && sym) return true;       // syntax-aware: token strutturale → protegge il codice
    if (up && dig)        return true;        // codici misti (VOLCANO-522)
    if (up && alnum > 3)  return true;        // acronimi / CamelCase
    float s = 1.0f;
    if (dig) s += 3.0f;
    if (up)  s += 1.5f;
    if (sym) s += 0.5f;
    if (alnum < 3) s -= 0.5f;                 // parole corte comuni (stop word)
    return s >= thr;
}

// Pre-gate a neighbor-window: tiene anchor/recent + i token "importanti" + tutto ciò che sta
// entro `win` da un importante; droppa il resto (filler isolato). Prima del forward-pass.
static std::vector<llama_token> lexgate_prefilter(const llama_vocab* vocab,
        const std::vector<llama_token>& raw, int anchor, int recent, int win, float thr, bool code_mode) {
    const int N = (int) raw.size();
    std::vector<char> keep(N, 0);
    for (int i = 0; i < N; i++) {
        if (lex_is_important(vocab, raw[i], thr, code_mode)) {   // importante → proteggi ±win (i confini)
            for (int j = std::max(0, i - win); j <= std::min(N - 1, i + win); j++) keep[j] = 1;
        }
    }
    std::vector<llama_token> out; out.reserve(N);
    for (int i = 0; i < N; i++)
        if (i < anchor || i >= N - recent || keep[i]) out.push_back(raw[i]);   // altrimenti: droppato
    return out;
}

// ==========================================
// DELTA STATE — persiste tra request HTTP consecutive
// ==========================================
struct DeltaState {
    std::vector<llama_token> original_toks; // raw prompt tokens (solo prompt, no generati)
    std::vector<bool>        tok_kept;       // tok_kept[i] = raw[i] inserito nel KV
    int                      n_kv    = 0;   // token nel KV cache dopo l'encode
    bool                     valid   = false;
};

// ==========================================
// HELPER: XML+FC protection, filtering, Pass 2
// Processa raw[raw_start..raw_start+slice_N-1].
// Rimuove KV da kv_start in poi, riscrive con i token tenuti.
// Ritorna n_past aggiornato, -1 su errore.
// ==========================================
static int ais_do_encode(
    llama_context*                  ctx,
    const llama_vocab*              vocab,
    int                             n_vocab,
    llama_batch&                    batch,
    const std::vector<llama_token>& raw,
    int                             raw_start,
    int                             slice_N,
    const std::vector<float>&       surprises,
    float                           threshold,
    int                             anchor_size,
    int                             recent_size,
    int                             kv_start,
    std::vector<bool>*              out_kept = nullptr,
    bool                            surgery_mode = false)
{
    const int n_batch = ais_n_batch();
    if (slice_N <= 0) { batch.n_tokens = 0; return kv_start; }  // nothing to encode (guards all /slice_N below)

    // ── XML+FC PROTECTION ──
    std::vector<bool> xml_protected(slice_N, false);
    {
        std::string text;
        text.reserve(slice_N * 5);
        std::vector<int> tcs(slice_N); // token char start
        char piece[512];
        for (int i = 0; i < slice_N; i++) {
            tcs[i] = (int)text.size();
            int len = llama_token_to_piece(vocab, raw[raw_start + i], piece, (int)sizeof(piece) - 1, 0, true);
            if (len > 0) text.append(piece, len);
        }
        auto prot = [&](int cs, int ce) {
            int lo = (int)(std::upper_bound(tcs.begin(), tcs.end(), cs) - tcs.begin()) - 1;
            if (lo < 0) lo = 0;
            int hi = (int)(std::lower_bound(tcs.begin(), tcs.end(), ce) - tcs.begin());
            for (int i = lo; i < hi && i < slice_N; i++) xml_protected[i] = true;
        };
        int tag_start = -1;
        for (int c = 0; c < (int)text.size(); c++) {
            if (text[c] == '<') { tag_start = c; }
            else if (text[c] == '>' && tag_start >= 0) {
                int te = c + 1;
                if (te - tag_start <= 120) prot(tag_start, te);
                tag_start = -1;
            } else if (text[c] == '\n' && tag_start >= 0) { tag_start = -1; }
        }
        // Pass B: contenuto breve tra tag (≤21 char) — tool names, argomenti corti
        for (int c = 0; c < (int)text.size() - 2; c++) {
            if (text[c] == '>') {
                int lim = std::min(c + 21, (int)text.size() - 1);
                for (int e = c + 1; e < lim; e++) {
                    if (text[e] == '<' && text[e + 1] == '/') { if (e > c + 1) prot(c + 1, e); break; }
                    if (text[e] == '<') break;
                }
            }
        }
        // Pass C: contenuto integrale dei tag critici di stato — tool results, error msgs.
        // Questi contengono conferme ("saved successfully", "error: ...") che il modello
        // deve leggere integralmente per non ripetere azioni già compiute.
        {
            static const char* RESULT_TAGS[] = {
                // risultati tool (conferme stato, output comandi)
                "<tool_result>", "<result>", "<error>", "<status>",
                "<output>", "<stdout>", "<stderr>",
                // argomenti write_to_file / replace_in_file (corpo file scritto):
                // se filtrati → modello non vede la scrittura completa → retry loop
                "<content>", "<new_content>", "<diff>",
                nullptr
            };
            for (int t = 0; RESULT_TAGS[t]; t++) {
                std::string open(RESULT_TAGS[t]);
                std::string close = "</" + open.substr(1);
                size_t pos = 0;
                while ((pos = text.find(open, pos)) != std::string::npos) {
                    size_t content_start = pos + open.size();
                    size_t close_pos = text.find(close, content_start);
                    if (close_pos != std::string::npos)
                        prot((int)content_start, (int)(close_pos + close.size()));
                    pos = (close_pos != std::string::npos) ? close_pos + close.size() : content_start + 1;
                }
            }
        }
    }
    int n_prot = (int)std::count(xml_protected.begin(), xml_protected.end(), true);
    fprintf(stderr, "XML+FC-protected: %d (%.1f%%)\n", n_prot, 100.0f * n_prot / slice_N);

    // ── NEIGHBOR WINDOW: adattivo a densità XML e distribuzione sorprese ──
    // Con NEIGHBOR_WIN=4 fisso e avg=16+ bit, ogni token scatta la finestra
    // → le finestre si sovrappongono e coprono tutto → 0% compressione.
    // Soluzione: WIN=0 per plain-text (pochi tag), WIN=4 solo per XML-denso.
    // Threshold neighbor adattiva: max(8, avg+0.5σ) evita che token mediocri
    // proteggano i loro vicini quando il livello base è già alto.
    float xml_density = (slice_N > 0) ? (float)n_prot / slice_N : 0.0f;
    int eff_neighbor_win = (xml_density >= 0.05f) ? NEIGHBOR_WIN
                         : (xml_density >= 0.01f) ? 1
                         :                          0;
    float s_loc_sum = 0.0f, s_loc_sum2 = 0.0f;
    for (int i = 0; i < slice_N; i++) { s_loc_sum += surprises[i]; s_loc_sum2 += surprises[i]*surprises[i]; }
    float s_loc_avg = s_loc_sum / std::max(1, slice_N);
    float s_loc_std = std::sqrt(std::max(0.0f, s_loc_sum2/slice_N - s_loc_avg*s_loc_avg));
    float eff_neigh_thr = std::max(NEIGHBOR_THRESHOLD, s_loc_avg + 0.5f * s_loc_std);
    fprintf(stderr, "Neighbor: win=%d thr=%.1f bit (xml=%.1f%%)\n",
            eff_neighbor_win, eff_neigh_thr, 100.0f * xml_density);

    // ── KEEP SET ──
    auto starts_word = [&](int i) -> bool {
        if (i <= 0) return true;
        char p[8];
        int len = llama_token_to_piece(vocab, raw[raw_start + i], p, (int)sizeof(p) - 1, 0, false);
        if (len <= 0) return true;
        return p[0] == ' ' || (uint8_t)p[0] == 0xE2;
    };
    // AIS_NO_PROTECT: disattiva protezione XML + neighbor (tiene solo anchor/recent +
    // soglia di sorpresa) → compressione massima. SPERIMENTALE: rompe i tag XML; sicuro
    // solo se il contenuto compresso è IRRILEVANTE per il task (es. tool inutilizzati).
    static const bool no_protect = getenv("AIS_NO_PROTECT") != nullptr;
    // keep_mask: O(n) bitmap (was std::set<int> → O(n log n) + per-insert alloc).
    // Scanning 0..slice_N to build `keep` yields the same ascending order the set gave.
    std::vector<char> keep_mask(slice_N, 0);
    for (int i = 0; i < slice_N; i++)
        if (i < anchor_size || i >= slice_N - recent_size || (!no_protect && xml_protected[i]) || surprises[i] >= threshold)
            keep_mask[i] = 1;
    if (eff_neighbor_win > 0 && !no_protect) {
        for (int i = 0; i < slice_N; i++) {
            if (surprises[i] >= eff_neigh_thr) {
                int lo = std::max(0, i - eff_neighbor_win), hi = std::min(slice_N, i + eff_neighbor_win + 1);
                while (lo > 0 && !starts_word(lo)) lo--;
                while (hi < slice_N && !starts_word(hi)) hi++;
                for (int k = lo; k < hi; k++) keep_mask[k] = 1;
            }
        }
    }
    { // word-completion forward pass (snapshot the current set so completions don't cascade)
        std::vector<int> snap;
        for (int i = 0; i < slice_N; i++) if (keep_mask[i]) snap.push_back(i);
        for (int k : snap) { int j = k + 1; while (j < slice_N && !starts_word(j)) { keep_mask[j] = 1; j++; } }
    }
    std::vector<int> keep;
    keep.reserve(slice_N);
    for (int i = 0; i < slice_N; i++) if (keep_mask[i]) keep.push_back(i);
    fprintf(stderr, "Compression: %.1f%% (%d → %d tok)\n",
            100.0f * (slice_N - (int)keep.size()) / slice_N, slice_N, (int)keep.size());

    if (out_kept) {
        out_kept->assign(slice_N, false);
        for (int idx : keep) (*out_kept)[idx] = true;
    }

    if (surgery_mode) {
        // ── KV SURGERY ──
        // Evict non-kept positions from the existing KV (filled during Pass 1).
        // Kept tokens stay at their original positions (kv_start+idx), preserving
        // RoPE offsets — positionally more accurate than Pass 2 compact reassignment.
        // The KV values of kept tokens implicitly encode context from removed tokens
        // (they attended to them during Pass 1), so quality is maintained.
        llama_memory_t mem = llama_get_memory(ctx);
        int run_start = -1;
        for (int i = 0; i <= slice_N; i++) {
            bool is_kept = (i < slice_N) && (keep_mask[i] != 0);
            if (!is_kept) {
                if (run_start < 0) run_start = kv_start + i;
            } else {
                if (run_start >= 0) {
                    llama_memory_seq_rm(mem, 0, run_start, kv_start + i);
                    run_start = -1;
                }
            }
        }
        batch.n_tokens = 0;
        return kv_start + slice_N;  // generation continues from original end position
    }

    // ── PASS 2 (compact re-encode, fallback) ──
    llama_memory_seq_rm(llama_get_memory(ctx), 0, kv_start, -1);
    batch.n_tokens = 0;
    int n_past = kv_start;
    for (int idx : keep) {
        batch_add(batch, raw[raw_start + idx], n_past, false);
        n_past++;
        if (batch.n_tokens >= n_batch) { if (llama_decode(ctx, batch) != 0) return -1; batch.n_tokens = 0; }
    }
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = 1;
        if (llama_decode(ctx, batch) != 0) return -1;
        batch.n_tokens = 0;
    }
    return n_past;
}

// ==========================================
// PASS 1 + SELEZIONE: prepara il KV cache
// Supporta delta encoding: se ds è valido e il prompt è un'estensione della
// sessione precedente, codifica solo i nuovi token (delta) senza ricomputare
// il prefisso già nel KV. Ritorna n_past, -1 su errore.
// ==========================================
static int ais_setup(
    llama_context*      ctx,
    const llama_vocab*  vocab,
    int                 n_vocab,
    llama_sampler*      smpl,
    llama_batch&        batch,
    const AISProbConfig& cfg_in,
    const std::string&  adapt_mode,
    float               adapt_param,
    const std::string&  prompt,
    DeltaState*         ds = nullptr)
{
    AISProbConfig cfg = cfg_in;
    const int n_batch = ais_n_batch();

    std::vector<llama_token> raw = ::common_tokenize(vocab, prompt, true, true);
    int N = (int)raw.size();
    if (N == 0) return 0;

    // ── DEDUP PRE-GATE (adattivo) ──────────────────────────────────────────
    // Salta i doc XML-densi (Cline: tool def unici, dedup rischioso/inutile) e
    // deduplica solo testo/log piatto (ripetizioni esatte). Riduce i token PRIMA
    // del forward-pass → meno KV, meno tempo, su carichi ridondanti. No-op sui doc densi.
    if (getenv("AIS_DEDUP")) {
        float xd = quick_xml_density(vocab, raw);
        if (xd < 0.02f) {
            std::vector<llama_token> surv = dedup_prefilter(raw, cfg.anchor_size, cfg.recent_size, 4);
            fprintf(stderr, "Dedup pre-gate: %d → %d tok (%.1f%% rimossi, xml=%.1f%%)\n",
                    N, (int)surv.size(), 100.0f * (N - (int)surv.size()) / N, 100.0f * xd);
            raw = std::move(surv); N = (int)raw.size();
        } else {
            fprintf(stderr, "Dedup pre-gate: saltato (xml-denso %.1f%%)\n", 100.0f * xd);
        }
    }

    fprintf(stderr, "Tokens: %d | mode=%s param=%.2f\n",
            N, adapt_mode.c_str(), adapt_mode == "fixed" ? cfg.surprise_threshold : adapt_param);

    // ── DELTA PATH ──────────────────────────────────────────────────────────
    // Se il prompt è un'estensione esatta della sessione precedente, codifica
    // solo il delta (nuovi token) usando il KV cache esistente come prefisso.
    // MIN_PARTIAL_PREFIX: accetta prefissi parziali purché sufficientemente lunghi.
    // Cline ha ~12k token statici + ~1k token dinamici → utile anche con P < maxP.
    const int MIN_PARTIAL_PREFIX = 2000;
    if (ds && ds->valid && !ds->tok_kept.empty()) {
        int P = 0, maxP = (int)ds->original_toks.size();
        while (P < maxP && P < N && raw[P] == ds->original_toks[P]) P++;

        fprintf(stderr, "Delta check: P=%d maxP=%d N=%d\n", P, maxP, N);

        if (P >= MIN_PARTIAL_PREFIX && N > P) {
            int D = N - P;

            // kv_prefix_count: token del prefisso effettivamente nel KV (per logging)
            // Con surgery mode, le posizioni KV sono originali [0..P-1], non compatte.
            int kv_prefix_count = 0;
            for (int i = 0; i < P && i < (int)ds->tok_kept.size(); i++)
                if (ds->tok_kept[i]) kv_prefix_count++;

            fprintf(stderr, "Delta: prefix=%d kv_pfx=%d delta=%d tok\n", P, kv_prefix_count, D);
            llama_sampler_reset(smpl);

            // Rimuove token generati (dopo il prefisso originale) e eventuale coda.
            // Surgery: posizioni nel KV sono originali → clear da P in poi.
            llama_memory_seq_rm(llama_get_memory(ctx), 0, P, -1);

            // Delta Pass 1 — delta tokens alle loro posizioni originali [P..P+D-1]
            std::vector<float> d_surp(D, 0.0f);
            bool ok = true;
            {
                // have=false: primo token delta ottiene sorpresa=0 (sempre tenuto)
                bool have = false;
                std::vector<float> prev;
                int offset = 0;
                while (offset < D && ok) {
                    int end = std::min(offset + n_batch, D), sz = end - offset;
                    batch.n_tokens = 0;
                    for (int i = 0; i < sz; i++)
                        batch_add(batch, raw[P + offset + i], P + offset + i, true);
                    if (llama_decode(ctx, batch)) { ok = false; break; }
                    if (have)
                        d_surp[offset] = fast_surprise(prev.data(), raw[P + offset], n_vocab);
                    for (int j = 0; j < sz - 1; j++) {
                        float* lgt = llama_get_logits_ith(ctx, j);
                        d_surp[offset + j + 1] = fast_surprise(lgt, raw[P + offset + j + 1], n_vocab);
                    }
                    float* last = llama_get_logits_ith(ctx, sz - 1);
                    prev.assign(last, last + n_vocab);
                    have = true; offset = end;
                }
            }

            if (ok) {
                // Surgery mode: ais_do_encode evicterà le posizioni non-kept direttamente
                // dal KV (niente clear+re-encode). Il delta rimane in KV a posizioni [P..P+D-1].

                float d_avg = std::accumulate(d_surp.begin(), d_surp.end(), 0.0f) / D;
                float d_var = 0.0f;
                for (float s : d_surp) d_var += (s - d_avg) * (s - d_avg);
                float d_std = std::sqrt(d_var / D);
                fprintf(stderr, "Delta: avg=%.2f bit\n", d_avg);

                // ── THRESHOLD DELTA ──────────────────────────────────────────────
                // Tre casi distinti basati su d_avg e d_std:
                //
                //  SMALL  (D <= 512):        nessun filtro — tool results, reasoning di
                //                            agenti sono critici; risparmio minimo.
                //
                //  MIXED  (d_std > 2.0 bit): distribuzione bimodale — contenuto misto
                //                            (es. Cline: tool args a 8+ bit + template a 0 bit).
                //                            Tagliamo solo i quasi-certi (< 0.5 bit).
                //
                //  LOW    (d_std <= 2.0 bit, d_avg <= 2.5): distribuzione uniforme bassa
                //                            — filler puro (log, template) → filtro aggressivo.
                //
                //  DENSE  (d_avg > 2.5 bit): contenuto ad alta entropia → floor 2.5 bit.
                //
                const int   SMALL_DELTA   = 1000;
                const float MIXED_STD_THR = 2.0f;
                float d_thr = 0.0f;

                if (D <= SMALL_DELTA) {
                    fprintf(stderr, "Delta: small (%d tok), no filter\n", D);
                } else if (d_avg > 2.5f) {
                    d_thr = std::max(2.5f, d_avg - adapt_param * d_std);
                    fprintf(stderr, "Delta: dense, thr=%.2f bit\n", d_thr);
                } else if (d_std > MIXED_STD_THR) {
                    // Distribuzione mista: token critici a bassa sorpresa (tool results,
                    // conferme "file scritto", reasoning interno agente). Tagliamo solo
                    // i token a sorpresa quasi-nulla (punteggiatura, spazi strutturali).
                    d_thr = 0.5f;
                    fprintf(stderr, "Delta: mixed (avg=%.2f std=%.2f), thr=0.50 bit\n",
                            d_avg, d_std);
                } else {
                    // Distribuzione uniforme bassa: log, template, filler ripetitivo.
                    // Nessun token critico basso-sorpresa — comprimiamo aggressivamente.
                    if (adapt_mode == "pct" || adapt_mode == "pct-mk") {
                        if (D >= 10) {
                            std::vector<float> s2 = d_surp; std::sort(s2.begin(), s2.end());
                            int idx = std::max(0, std::min(D - 1, (int)(D * (1.0f - adapt_param))));
                            d_thr = s2[idx];
                        }
                    } else {
                        d_thr = std::max(0.5f, d_avg + adapt_param * d_std);
                    }
                    fprintf(stderr, "Delta: low-entropy, thr=%.2f bit\n", d_thr);
                }

                std::vector<bool> delta_kept;
                int n_past = ais_do_encode(ctx, vocab, n_vocab, batch,
                                           raw, P, D, d_surp, d_thr,
                                           cfg.anchor_size, cfg.recent_size, P,
                                           &delta_kept, /*surgery_mode=*/true);
                if (n_past >= 0) {
                    // Aggiorna tok_kept: prefisso invariato + delta filtrato
                    std::vector<bool> new_kept(N, false);
                    int copy_len = std::min(P, (int)ds->tok_kept.size());
                    for (int i = 0; i < copy_len; i++) new_kept[i] = ds->tok_kept[i];
                    for (int i = 0; i < (int)delta_kept.size(); i++) new_kept[P + i] = delta_kept[i];
                    ds->original_toks = raw;
                    ds->tok_kept      = std::move(new_kept);
                    ds->n_kv          = n_past;
                    return n_past;
                }
            }
            fprintf(stderr, " Delta failed, full re-encode\n");
        }
    }

    // ── FULL ENCODE PATH ────────────────────────────────────────────────────
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    llama_sampler_reset(smpl);

    // ── GATE ADATTIVO (AIS_ADAPTIVE): comprimi dove conviene, altrove = vanilla ──
    // Decisione a COSTO ZERO (niente forward-pass). Se il doc NON è comprimibile
    // (XML denso tipo Cline, o corto) → prefill VANILLA (logit solo sull'ultimo token):
    // niente tassa lm_head, niente buffer extra → veloce e leggero come vanilla. Se è
    // comprimibile (ridondante o testo piano) → percorso AIS (sul deduplicato se serve).
    // Così "il motore guadagna dove può, e dove non può non perde nulla".
    if (getenv("AIS_ADAPTIVE")) {
        const float xd = quick_xml_density(vocab, raw);
        // XML denso (Cline) o doc corto → non comprimibile in SICUREZZA → VANILLA path:
        // il dedup sui tag rischierebbe la struttura e l'entropico rende troppo poco per
        // valere la tassa di scoring. Niente all-token lm_head, niente buffer → = vanilla.
        if (N < MIN_DOC_SIZE || xd >= 0.05f) {
            fprintf(stderr, "Adaptive: NON comprimibile (xml=%.1f%% N=%d) → vanilla path (no scoring)\n",
                    100.0f * xd, N);
            auto tvp = std::chrono::steady_clock::now();
            batch.n_tokens = 0;
            int offv = 0;
            while (offv < N) {
                int end = std::min(offv + n_batch, N);
                for (int i = offv; i < end; i++) batch_add(batch, raw[i], i, false);
                if (end == N) batch.logits[batch.n_tokens - 1] = 1;   // solo ultimo → per generare
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode error (vanilla path)\n"); return -1; }
                batch.n_tokens = 0; offv = end;
            }
            fprintf(stderr, " Vanilla-path prefill: %.0fms (no scoring tax)\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tvp).count());
            if (ds) { ds->original_toks = raw; ds->tok_kept.assign(N, true); ds->n_kv = N; ds->valid = true; }
            return N;
        }
        // testo a BASSO XML: il dedup è sicuro (solo ripetizioni esatte). Se conviene, usalo.
        std::vector<llama_token> dd = dedup_prefilter(raw, cfg.anchor_size, cfg.recent_size, 4);
        float ddpct = (N > 0) ? (float)(N - (int)dd.size()) / N : 0.0f;
        if (ddpct >= 0.10f) {
            fprintf(stderr, "Adaptive: comprimibile via dedup %.1f%% (%d→%d) → AIS\n",
                    100.0f * ddpct, N, (int)dd.size());
            raw = std::move(dd); N = (int)raw.size();
        } else {
            fprintf(stderr, "Adaptive: comprimibile (entropico, xml=%.1f%% dedup=%.1f%%) → AIS\n",
                    100.0f * xd, 100.0f * ddpct);
        }
    }

    // ── PASS 1 ──
    std::vector<float> surprises(N, 0.0f);
    std::vector<float> prev_logits(n_vocab, 0.0f);
    bool have_prev = false;

    auto t_pass1 = std::chrono::steady_clock::now();
    double decode_ms = 0.0, surprise_ms = 0.0;   // Pass-1 perf breakdown
    const bool validate_gather = getenv("AIS_VALIDATE_GATHER") != nullptr;
    const bool gather_score    = getenv("AIS_GATHER_SCORE")    != nullptr;
    // TODO#2 (PROJECT_GUIDE): dedicated [1×N] surprise readback. When AIS_SURPRISE_OUT is set
    // the graph emits a width-1 target-logit tensor → read it back via llama_get_ais_surprise_ith
    // (N floats) instead of llama_get_embeddings_ith(...)[0] (the [n_embd_out × N] embd buffer).
    const bool surprise_out    = getenv("AIS_SURPRISE_OUT")    != nullptr;
    double vg_max_err = 0.0, vg_sum_a = 0, vg_sum_b = 0, vg_sum_aa = 0, vg_sum_bb = 0, vg_sum_ab = 0;
    // B3 calibration: relate TRUE bit-surprise to the gather-dot -logit
    double vg_sum_true = 0, vg_sum_tt = 0, vg_sum_t_nt = 0, vg_sum_nt = 0, vg_sum_ntnt = 0;
    long   vg_n = 0;
    // ── TODO#1 EXPERIMENT (PROJECT_GUIDE §5-bis): "top-k max? senza calcolare su tutto" ──
    // The faithful surprise needs Z = logsumexp over ALL n_vocab logits; the gather-dot gives
    // logit[tok] (the numerator) for free but NOT Z. This probe asks: can Z be recovered from a
    // STATIC candidate subset of C vocab entries (→ an lm_head over only C columns, ~C/n_vocab of
    // the cost)? It builds the set ONCE from the head of the doc, then measures the surprise error
    // logsumexp_full − logsumexp_subset on later positions. Needs full logits → AIS_VALIDATE_GATHER.
    const int  zcand_C   = [](){ const char* v = getenv("AIS_ZCAND"); return v ? std::max(1, atoi(v)) : 0; }();
    const int  z_build_to = std::max(1, N / 4);   // build the static set from the first quarter of positions
    std::vector<double>      z_acc;               // accumulated logits over the build window (top-C source)
    std::vector<llama_token> z_set;               // the static candidate set, built once
    double z_err_sum = 0, z_err_max = 0, z_frac_sum = 0; long z_n = 0, z_cover = 0;
    if (zcand_C > 0 && validate_gather) z_acc.assign(n_vocab, 0.0);
    int offset = 0;
    while (offset < N) {
        int end      = std::min(offset + n_batch, N);
        int batch_sz = end - offset;
        batch.n_tokens = 0;
        for (int i = offset; i < end; i++)
            batch_add(batch, raw[i], i, true);
        auto t_d0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode error (pass 1)\n"); return -1;
        }
        auto t_d1 = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();
        if (gather_score) {
            // B2/B3: surprise from the gather-dot target logit (embd[0]) — the full
            // lm_head was SKIPPED in the graph. Pure -logit (raw logit units, not
            // bits): downstream thresholds must be scale-relative (B3). The first
            // token of each batch has no in-batch predictor → left 0 (always kept).
            if (surprise_out)
                for (int j = 0; j < batch_sz - 1; j++)
                    surprises[offset + j + 1] = -llama_get_ais_surprise_ith(ctx, j);
            else
                for (int j = 0; j < batch_sz - 1; j++)
                    surprises[offset + j + 1] = -llama_get_embeddings_ith(ctx, j)[0];
            surprise_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d1).count();
            have_prev = true; offset = end;
            continue;
        }
        if (have_prev && offset > 0)
            surprises[offset] = fast_surprise(prev_logits.data(), raw[offset], n_vocab);
        for (int j = 0; j < batch_sz - 1; j++) {
            float* lgt = llama_get_logits_ith(ctx, j);
            surprises[offset + j + 1] = fast_surprise(lgt, raw[offset + j + 1], n_vocab);
            if (validate_gather) {
                // full lm_head logit of the next token at position j (true value)
                float full = lgt[raw[offset + j + 1]];
                // gather-dot target logit emitted by the graph in embd[0]
                float tgt  = llama_get_embeddings_ith(ctx, j)[0];
                double e = std::fabs((double)full - (double)tgt);
                if (e > vg_max_err) vg_max_err = e;
                vg_sum_a += full; vg_sum_b += tgt; vg_sum_aa += (double)full*full;
                vg_sum_bb += (double)tgt*tgt; vg_sum_ab += (double)full*tgt;
                // true bit-surprise vs the gather-dot -logit (for the bit conversion)
                double ts = compute_surprise(lgt, raw[offset + j + 1], n_vocab);
                double nt = -(double)tgt;   // -logit (the gather-score raw surprise)
                vg_sum_true += ts; vg_sum_tt += ts*ts; vg_sum_t_nt += ts*nt;
                vg_sum_nt += nt; vg_sum_ntnt += nt*nt;
                vg_n++;

                // TODO#1 probe: static candidate-subset Z vs full Z.
                if (zcand_C > 0) {
                    const int p = offset + j + 1;       // global position of this prediction
                    if (z_set.empty()) {
                        // BUILD phase: accumulate logits; freeze the top-C set past the window.
                        for (int v = 0; v < n_vocab; v++) z_acc[v] += lgt[v];
                        if (p + 1 >= z_build_to) {
                            std::vector<int> idx(n_vocab);
                            for (int v = 0; v < n_vocab; v++) idx[v] = v;
                            std::partial_sort(idx.begin(), idx.begin() + zcand_C, idx.end(),
                                              [&](int a, int b){ return z_acc[a] > z_acc[b]; });
                            z_set.assign(idx.begin(), idx.begin() + zcand_C);
                        }
                    } else {
                        // COMPARE phase: full logsumexp vs subset logsumexp (surprise error in bits).
                        const float mx = logit_max(lgt, n_vocab);
                        double zf = 0.0;
                        for (int v = 0; v < n_vocab; v++) zf += std::exp((double)(lgt[v] - mx));
                        const double logZf = std::log(zf) + mx;
                        float mxc = -INFINITY;
                        for (llama_token v : z_set) if (lgt[v] > mxc) mxc = lgt[v];
                        double zc = 0.0;
                        for (llama_token v : z_set) zc += std::exp((double)(lgt[v] - mxc));
                        const double logZc = std::log(zc) + mxc;
                        const double err = std::fabs(logZf - logZc) / std::log(2.0);   // bits of surprise error
                        z_err_sum += err; if (err > z_err_max) z_err_max = err;
                        z_frac_sum += std::exp(logZc - logZf);          // Z_cand / Z_full
                        if (mxc >= mx - 1e-4f) z_cover++;               // the global argmax (head) is in the set
                        z_n++;
                    }
                }
            }
        }
        float* last = llama_get_logits_ith(ctx, batch_sz - 1);
        std::copy(last, last + n_vocab, prev_logits.begin());
        surprise_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_d1).count();
        have_prev = true;
        offset    = end;
    }
    if (validate_gather && vg_n > 0) {
        double ma=vg_sum_a/vg_n, mb=vg_sum_b/vg_n;
        double cov=vg_sum_ab/vg_n-ma*mb, va=vg_sum_aa/vg_n-ma*ma, vb=vg_sum_bb/vg_n-mb*mb;
        double pear=cov/((va>0&&vb>0)?std::sqrt(va*vb):1e-9);
        fprintf(stderr, "GATHER-DOT vs full lm_head over %ld positions: max|err|=%.4f  Pearson=%.6f  (full avg=%.3f, tgt avg=%.3f)\n",
                vg_n, vg_max_err, pear, ma, mb);
        // B3: fit true_bit_surprise ≈ slope*(-logit) + offset   (LN2=0.6931)
        double mt = vg_sum_true/vg_n, mn = vg_sum_nt/vg_n;
        double covtn = vg_sum_t_nt/vg_n - mt*mn;
        double vn = vg_sum_ntnt/vg_n - mn*mn, vt = vg_sum_tt/vg_n - mt*mt;
        double slope = (vn>0)? covtn/vn : 0.0;
        double offset = mt - slope*mn;
        double r2 = (vt>0&&vn>0)? (covtn*covtn)/(vt*vn) : 0.0;
        // offset assuming the theoretical slope 1/ln2 (bits = -logit/ln2 + B):
        double B_theory = mt - (1.0/0.69314718)*mn;
        fprintf(stderr, "BIT-FIT: true_avg=%.3f  -logit_avg=%.3f  | fit slope=%.4f offset=%.3f R^2=%.4f | B(slope=1/ln2)=%.3f\n",
                mt, mn, slope, offset, r2, B_theory);
    }
    if (zcand_C > 0 && validate_gather && z_n > 0) {
        // VERDICT: err mean ≪ 0.05 bit (the sigma-mk cut margin) AND head-in-set ≈100% ⇒ a static
        // C-set captures Z ⇒ the lm_head can run on C columns only ("top-k senza calcolare su tutto").
        // Large err / low coverage ⇒ the head shifts per position ⇒ a static set is NOT enough.
        fprintf(stderr, "ZCAND C=%d (static top-C from first %d pos) over %ld compare-pos: "
                "Z-surprise err mean=%.4f max=%.4f bit | Z_cand/Z_full mean=%.4f | head-in-set %.1f%%\n",
                zcand_C, z_build_to, z_n, z_err_sum/z_n, z_err_max, z_frac_sum/z_n, 100.0*z_cover/z_n);
    }
    fprintf(stderr, " Pass1: total=%.0fms decode=%.0fms surprise=%.0fms\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pass1).count(),
            decode_ms, surprise_ms);

    float s_sum = std::accumulate(surprises.begin(), surprises.end(), 0.0f);
    float s_avg = s_sum / N;
    float s_min = *std::min_element(surprises.begin(), surprises.end());
    float s_max = *std::max_element(surprises.begin(), surprises.end());
    fprintf(stderr, "Surprises: min=%.2f avg=%.2f max=%.2f bit\n", s_min, s_avg, s_max);
    if (const char* dp = getenv("AIS_DUMP_SURPRISE")) {  // per-token dump for scorer A/B
        FILE* df = fopen(dp, "w");
        if (df) { for (int i = 0; i < N; i++) fprintf(df, "%d %.4f\n", raw[i], surprises[i]); fclose(df); }
    }

    // ── THRESHOLD ADATTIVA ──
    if (adapt_mode == "pct" || adapt_mode == "pct-mk") {
        if (adapt_mode == "pct-mk" && N < MIN_DOC_SIZE) {
            cfg.surprise_threshold = 0.0f;
        } else {
            std::vector<float> sorted = surprises;
            std::sort(sorted.begin(), sorted.end());
            int idx = std::max(0, std::min(N - 1, (int)(N * (1.0f - adapt_param))));
            cfg.surprise_threshold = sorted[idx];
        }
    } else if (adapt_mode == "sigma" || adapt_mode == "sigma-mk") {
        float s_var = 0.0f;
        for (float s : surprises) s_var += (s - s_avg) * (s - s_avg);
        float s_std = std::sqrt(s_var / N);
        bool short_doc = (adapt_mode == "sigma-mk" && N < MIN_DOC_SIZE);
        if (short_doc) {
            cfg.surprise_threshold = 0.0f;
            fprintf(stderr, "SIGMA-MK: use Pass1 KV (short doc)\n");
        } else if (gather_score) {
            // B3: in gather-score the surprise is pure -logit (the lm_head was
            // skipped, so there is NO logsumexp normalizer). Measured: -logit is
            // only R^2~0.5 of true bit-surprise and the bit-offset is doc-dependent
            // (≈9.5 .. 18) → it CANNOT be converted to bits, so the bit-calibrated
            // constants (dense-guard 2.5, floors, neighbor 8.0) don't apply. Use a
            // scale-INVARIANT z-threshold (avg + k·σ); anchor/recent/XML protections
            // (scale-independent) still guard structure. This keeps high-surprise
            // outliers (codes, needles) reliably; it is a faster, lower-fidelity
            // filter than true-surprise sigma-mk, not a faithful replica.
            cfg.surprise_threshold = s_avg + adapt_param * s_std;
            fprintf(stderr, "GATHER-SCORE: scale-relative thr = avg + %.2f·σ = %.3f (-logit units)\n",
                    adapt_param, cfg.surprise_threshold);
        } else if (adapt_mode == "sigma-mk" && s_avg > 2.5f) {
            // Dense doc: soglia = max(2.5, s_avg - adapt_param*s_std).
            // adapt_param controlla aggressività: 1.0 → minimo (spesso colpisce il floor),
            // 0.5-0.7 → compressione moderata anche su doc alta-entropia (avg>12 bit).
            // Pass C protegge tool_result/error/output → sicuro anche con k < 1.0.
            const float FLOOR = 2.5f;
            cfg.surprise_threshold = std::max(FLOOR, s_avg - adapt_param * s_std);
            fprintf(stderr, "SIGMA-MK: dense doc, thr=%.2f bit\n",
                    cfg.surprise_threshold);
        } else {
            cfg.surprise_threshold = std::max(0.5f, s_avg + adapt_param * s_std);
            fprintf(stderr, "SIGMA%s: thr=%.2f bit\n",
                    adapt_mode == "sigma-mk" ? "-MK" : "", cfg.surprise_threshold);
        }
    }

    // Turn-aware recent window: se il prompt ha più turn (sessione multi-turn),
    // protegge integralmente gli ultimi 2 turn e lascia che il threshold sigma-mk
    // filtri i turn vecchi se hanno entropia bassa. Per un singolo turn rimane
    // cfg.recent_size (128 tok) — nessuna compressione forzata.
    int adj_recent_size = cfg.recent_size;
    if (N >= MIN_DOC_SIZE && cfg.surprise_threshold > 0.0f) {
        std::string full_text;
        full_text.reserve(N * 4);
        char piece[512];
        for (int i = 0; i < N; i++) {
            int len = llama_token_to_piece(vocab, raw[i], piece, sizeof(piece) - 1, 0, true);
            if (len > 0) full_text.append(piece, len);
        }
        const std::string TURN_MARKER = "<start_of_turn>";
        std::vector<size_t> turn_starts;
        size_t p = 0;
        while ((p = full_text.find(TURN_MARKER, p)) != std::string::npos) {
            turn_starts.push_back(p);
            p += TURN_MARKER.size();
        }
        if (turn_starts.size() >= 2) {
            // Protegge dal penultimo turn in poi
            size_t target_byte = turn_starts[turn_starts.size() - 2];
            size_t cum = 0;
            int tok_idx = N;
            for (int i = 0; i < N; i++) {
                if (cum >= target_byte) { tok_idx = i; break; }
                int len = llama_token_to_piece(vocab, raw[i], piece, sizeof(piece) - 1, 0, true);
                if (len > 0) cum += len;
            }
            adj_recent_size = std::max(cfg.recent_size, N - tok_idx);
            fprintf(stderr, "Turn-aware: %zu turns, protecting last 2 (%d tok)\n",
                    turn_starts.size(), adj_recent_size);
        }
    }

    // use_pass1_kv: doc corto sigma-mk — riusa KV da Pass 1 senza filtrare
    bool use_pass1_kv = (adapt_mode == "sigma-mk" && cfg.surprise_threshold == 0.0f);
    int n_past;
    std::vector<bool> full_kept;
    if (use_pass1_kv) {
        batch.n_tokens = 0;
        fprintf(stderr, "Pass1 KV reused — 0%% compression\n");
        n_past = N;
        if (ds) full_kept.assign(N, true);  // tutti tenuti (pass1 kv)
    } else {
        n_past = ais_do_encode(ctx, vocab, n_vocab, batch,
                               raw, 0, N, surprises, cfg.surprise_threshold,
                               cfg.anchor_size, adj_recent_size, 0,
                               ds ? &full_kept : nullptr, /*surgery_mode=*/true);
        if (n_past < 0) return -1;
    }

    if (ds) {
        ds->original_toks = raw;
        ds->tok_kept      = std::move(full_kept);
        ds->n_kv          = n_past;
        ds->valid         = true;
    }
    return n_past;
}

// ==========================================
// BATCH MODE: setup + genera tutto, ritorna stringa
// out_prompt_tok / out_completion_tok: contatori reali per usage
// ==========================================
static std::string ais_infer(
    llama_context*      ctx,
    const llama_vocab*  vocab,
    int                 n_vocab,
    llama_sampler*      smpl,
    llama_batch&        batch,
    const AISProbConfig& cfg,
    const std::string&  adapt_mode,
    float               adapt_param,
    const std::string&  prompt,
    int                 max_tokens,
    int*                out_prompt_tok     = nullptr,
    int*                out_completion_tok = nullptr,
    DeltaState*         ds                 = nullptr)
{
    int n_past = ais_setup(ctx, vocab, n_vocab, smpl, batch,
                           cfg, adapt_mode, adapt_param, prompt, ds);
    if (n_past < 0) return "[error: setup]";
    if (out_prompt_tok) *out_prompt_tok = n_past;

    std::string result;
    result.reserve(1024);
    int generated = 0;
    // ── Compressione online del CoT (taglio sulla sorpresa) — Gemma-4 ───────
    // Il modello PENSA, ma quando il "thought" entra in un tratto a bassa
    // informazione (sorpresa < TLOW per WIN token consecutivi) AIS chiude il
    // pensiero (inietta <channel|>) e passa alla risposta: tiene i token di
    // ragionamento significativi, scarta il flusso ripetitivo/di copiatura.
    static const bool cot_dump = getenv("AIS_COT_DUMP") != nullptr;
    static const bool cot_comp = getenv("AIS_COMPRESS_COT") != nullptr;
    auto envf = [](const char* k, double d){ const char* v = getenv(k); return v ? atof(v) : d; };
    const double TLOW = envf("AIS_COT_TLOW", 0.05);   // bit: sotto = token "scontato"
    const int    WIN  = (int)envf("AIS_COT_WIN", 24); // run di token bassi → taglio
    const int    MINT = (int)envf("AIS_COT_MIN", 48); // protegge il ragionamento iniziale
    const int    think_cap = cot_think_cap(max_tokens); // tetto: lascia budget per la risposta
    const bool   need_surp = cot_dump || cot_comp;
    const std::vector<llama_token> end_tag = common_tokenize(vocab, "<channel|>", false, true);
    bool in_thought = true;  // gemma-4 inizia col canale "thought"
    int  thought_toks = 0, low_run = 0;
    auto t_gen = std::chrono::steady_clock::now();
    for (int i = 0; i < max_tokens; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);
        // sorpresa del token scelto = -log2 p(id) = (logsumexp - logit[id])/ln2.
        // Calcolata solo durante il thought (poi non serve).
        if (need_surp && in_thought) {
            const float* lg = llama_get_logits_ith(ctx, -1);
            // PROBE top-K (diagnostica AIS_TOPK): quanto della partition function Z sta nei
            // top-K logit? Se i top-K catturano Z (dErr≈0) un lm_head a vocab-candidato sarebbe
            // fedele; altrimenti la coda conta. Tenuto ESATTO anche sotto AIS_COT_DUMP.
            static const int TOPK = [](){ const char* v=getenv("AIS_TOPK"); return v?atoi(v):0; }();
            double surp;
            if (cot_dump || TOPK > 0) {
                // full logsumexp — i diagnostici hanno bisogno di mx/se esatti
                float mx = -INFINITY;
                for (int v = 0; v < n_vocab; v++) if (lg[v] > mx) mx = lg[v];
                double se = 0.0;
                for (int v = 0; v < n_vocab; v++) se += std::exp((double)(lg[v] - mx));
                surp = ((double)mx + std::log(se) - (double)lg[id]) / std::log(2.0);
                if (cot_dump) fprintf(stderr, "COT %4d T surp=%.3f\n", generated, surp);
                if (TOPK > 0) {
                    std::vector<float> tmp(lg, lg + n_vocab);
                    int k = std::min(TOPK, n_vocab);
                    std::nth_element(tmp.begin(), tmp.begin()+k, tmp.end(), std::greater<float>());
                    double se_k = 0.0;
                    for (int j = 0; j < k; j++) se_k += std::exp((double)(tmp[j] - mx));
                    double surp_k = ((double)mx + std::log(se_k) - (double)lg[id]) / std::log(2.0);
                    // top-1 (solo il max, = fast_surprise / max-margin): logsumexp ≈ max
                    double surp_1 = ((double)mx - (double)lg[id]) / std::log(2.0);
                    fprintf(stderr, "TOPK %4d full=%.3f top1=%.3f top%d=%.3f e1=%.4f eK=%.4f Zfrac=%.5f\n",
                            generated, surp, surp_1, k, surp_k, surp - surp_1, surp - surp_k, se_k / se);
                }
            } else {
                // G1: top-K logsumexp (≈esatto a 0.016 bit) — niente doppio loop full-vocab
                surp = cot_surprise(lg, id, n_vocab);
            }
            thought_toks++;
            low_run = (surp < TLOW) ? low_run + 1 : 0;
            // taglio: tratto lungo a bassa informazione, dopo il minimo protetto
            if (cot_comp && thought_toks >= MINT && !end_tag.empty() &&
                (low_run >= WIN || (think_cap > 0 && thought_toks >= think_cap))) {
                for (llama_token et : end_tag) {
                    char b[128]; int bn = llama_token_to_piece(vocab, et, b, sizeof(b), 0, true);
                    if (bn > 0) result.append(b, bn);
                    batch_add(batch, et, n_past, true); n_past++; generated++;
                    if (llama_decode(ctx, batch) != 0) { i = max_tokens; break; }
                    batch.n_tokens = 0;
                }
                in_thought = false;
                fprintf(stderr, " CoT: taglio a %d tok di pensiero (low_run=%d)\n",
                        thought_toks, low_run);
                continue;  // scarta 'id' (token di pensiero scontato), genera la risposta
            }
        }
        if (ais_is_eog(vocab, id)) break;
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) break;
        result.append(buf, n);
        if (in_thought && tail_contains(result, n, "<channel|>", 10)) in_thought = false;
        batch_add(batch, id, n_past, true);
        n_past++;
        generated++;
        if (llama_decode(ctx, batch) != 0) break;
        batch.n_tokens = 0;
    }
    double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen).count();
    fprintf(stderr, " Gen: %d tok in %.0fms (%.1f tok/s)\n",
            generated, gen_ms, generated > 0 ? 1000.0 * generated / gen_ms : 0.0);
    if (out_completion_tok) *out_completion_tok = generated;
    return result;
}

// ============================================================================
// SnapKV-style PROBE (Milestone C, rotta A): cattura i pesi di attenzione
// (tensore kq_soft_max, path NON-flash) via cb_eval e calcola la RILEVANZA di
// ogni posizione = attenzione ricevuta dalla finestra-query (ultimi W token),
// sommata su teste e layer ad attenzione PIENA. Valida il segnale (rilevante vs
// filler) prima di costruire il fork del grafo (rotta B).
// ============================================================================
struct SnapState {
    bool on   = false;   // ROTTA A (probe): cattura kq_soft_max (flash off)
    bool fork = false;   // ROTTA B (cached-K fork Gemma-4): cattura ais_snap_rel (flash on)
    bool fa   = false;   // ROTTA C (hook GENERICO in build_attn_mha): ais_snap_rel per-layer, flash ON, OGNI modello
    int  W  = 16;
    std::vector<double> rel;  // rilevanza per posizione KV
    int  layers = 0;
    bool logged = false;
    // ── AIS_SNAP_PROFILE: diagnostica costo (readback rilevanza + sweep eviction) ──
    bool   capturing    = true;  // false durante la GENERAZIONE → il cb_eval esce subito
                                 // (la rilevanza serve solo nel prefill) = decode senza tassa
    bool   run_mode     = false; // running-eviction: per-chunk rel (host resets layers each chunk)
    bool   profile      = false;
    double rb_ms        = 0.0;   // tempo speso a leggere kq_soft_max dalla GPU
    size_t rb_bytes     = 0;     // byte trasferiti GPU→CPU per la rilevanza
    double evict_ms     = 0.0;   // tempo del sweep di seq_rm
    int    evict_calls  = 0;     // numero di chiamate seq_rm (= run contigui R)
};
static SnapState g_snap;

static bool snap_eval_cb(struct ggml_tensor* t, bool ask, void* /*ud*/) {
    if (!g_snap.capturing) return false;   // generazione: nessuna cattura → zero lavoro per nodo
    if (!t->name[0]) return false;
    // ROTTA B (fork cached-K): cattura il tensore di rilevanza già pronto [n_kv].
    if (g_snap.fork) {
        const bool is_rel = strstr(t->name, "ais_snap_rel") != nullptr;
        if (ask) return is_rel;
        if (!is_rel || t->type != GGML_TYPE_F32) return true;
        const int nkv = (int) t->ne[0];
        if (nkv < (int) g_snap.rel.size()) return true;   // tieni la rel con n_kv MAX (ultimo chunk)
        std::vector<float> buf(nkv);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) nkv * sizeof(float));
        g_snap.rel.assign(buf.begin(), buf.end());
        if (!g_snap.logged) { fprintf(stderr, "SnapKV cattura rel: %s n_kv=%d\n", t->name, nkv); g_snap.logged = true; }
        return true;
    }
    // ROTTA C (hook generico flash-ON): "ais_snap_rel" è emesso PER-LAYER (non pre-sommato)
    // → ACCUMULA sui layer. Tieni il chunk con n_kv MAX (l'ultimo = KV piena): n_kv cresce
    // monotòno nel prefill chunked → reset+accumula quando arriva un n_kv più grande.
    if (g_snap.fa) {
        const bool is_rel = strstr(t->name, "ais_snap_rel") != nullptr;
        if (ask) return is_rel;
        if (!is_rel || t->type != GGML_TYPE_F32) return true;
        const int nkv = (int) t->ne[0];
        if (g_snap.run_mode) {
            // running eviction: host resets g_snap.layers before each chunk. Size rel to THIS
            // chunk's nkv on its first captured layer, then accumulate over layers. The max-nkv
            // heuristic below can't be used here: n_kv shrinks after mid-prefill eviction.
            std::vector<float> rbuf(nkv);
            ggml_backend_tensor_get(t, rbuf.data(), 0, (size_t) nkv * sizeof(float));
            if (g_snap.layers == 0) g_snap.rel.assign(nkv, 0.0);
            if ((int) g_snap.rel.size() == nkv)
                for (int k = 0; k < nkv; k++) g_snap.rel[k] += rbuf[k];
            g_snap.layers++;
            return true;
        }
        if (nkv < (int) g_snap.rel.size()) return true;            // layer SWA o chunk più piccolo → ignora
        if (nkv > (int) g_snap.rel.size()) g_snap.rel.assign(nkv, 0.0);   // nuovo chunk più grande → reset
        auto t_rb = std::chrono::steady_clock::now();
        std::vector<float> buf(nkv);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) nkv * sizeof(float));
        for (int k = 0; k < nkv; k++) g_snap.rel[k] += buf[k];
        if (g_snap.profile) { g_snap.rb_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_rb).count(); g_snap.rb_bytes += (size_t)nkv*sizeof(float); }
        g_snap.layers++;
        if (!g_snap.logged) { fprintf(stderr, "SnapKV FA cattura rel per-layer: %s n_kv=%d\n", t->name, nkv); g_snap.logged = true; }
        return true;
    }
    if (!g_snap.on) return false;
    const bool is_kq = strstr(t->name, "kq_soft_max") != nullptr;
    if (ask) return is_kq;                       // sì, voglio i dati di kq_soft_max
    if (!is_kq || t->type != GGML_TYPE_F32) return true;
    const int n_kv = (int)t->ne[0], n_tok = (int)t->ne[1], n_head = (int)t->ne[2];
    if (!g_snap.logged) {
        fprintf(stderr, "SnapKV cattura: %s ne=[%d,%d,%d]\n", t->name, n_kv, n_tok, n_head);
        g_snap.logged = true;
    }
    // accumula solo i layer ad attenzione PIENA (n_kv == N); salta SWA (finestra ridotta)
    if (n_kv < (int)g_snap.rel.size()) return true;
    if (n_kv > (int)g_snap.rel.size()) g_snap.rel.assign(n_kv, 0.0);
    const int W = std::min(g_snap.W, n_tok);
    auto t_rb = std::chrono::steady_clock::now();
    // Leggi SOLO le ultime W righe-query per testa (non tutte le n_tok): nel layout
    // [n_kv, n_tok, n_head] le W query servono sono contigue in coda a ogni slab-testa
    // [n_kv × n_tok] → 1 read di [n_kv × W] per testa. Trasferimento ~ n_tok/W volte
    // più piccolo del read pieno (es. 512/24 ≈ 21×) + meno lavoro CPU.
    std::vector<float> buf((size_t)n_kv * W);
    for (int h = 0; h < n_head; h++) {
        const size_t off = ((size_t)h * n_tok + (n_tok - W)) * n_kv;   // prima riga della finestra
        ggml_backend_tensor_get(t, buf.data(), off * sizeof(float), (size_t)n_kv * W * sizeof(float));
        for (int q = 0; q < W; q++) {
            const float* row = buf.data() + (size_t)q * n_kv;
            for (int k = 0; k < n_kv; k++) g_snap.rel[k] += row[k];
        }
        if (g_snap.profile) g_snap.rb_bytes += (size_t)n_kv * W * sizeof(float);
    }
    if (g_snap.profile) g_snap.rb_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_rb).count();
    g_snap.layers++;
    return true;
}

// ============================================================================
// SnapKV cached-K — inferenza server (Milestone C-2) CON DELTA-PATH:
//  - 1ª richiesta (o prefisso cambiato): prefill CHUNKED completo (flash on; il fork
//    calcola la rilevanza, cb_eval la cattura) → eviction query-aware → salva lo stato.
//  - richieste successive con prefisso condiviso: RIUSA la KV compressa del prefisso
//    (i token tenuti restano alle posizioni originali) e prefilla SOLO il delta (token
//    nuovi) → veloce. Il prefisso resta compresso (niente ri-eviction). = il win long-run.
// Poi generazione con CoT-cut. lm_head solo sull'ultimo token di ogni chunk.
// ============================================================================
// CROSS-TURN DIFF-REUSE: trova il PIÙ LUNGO blocco di token contiguo comune tra il
// prompt vecchio e il nuovo (es. il corpo di un file ri-mandato con una piccola modifica).
// Anchor su 64-gram (hash) + estensione left/right. O(N) atteso. Ritorna (bo,bn,L) =
// inizio nel vecchio, inizio nel nuovo, lunghezza. La KV di quel blocco si RIUSA
// (spostata via seq_add) e si prefilla SOLO ciò che cambia ai due lati → niente
// ri-prefill del corpo invariato. Il blocco riusato ha attenzione al vecchio contorno
// (drift lieve, accettato per diff piccoli: come il caching a blocchi).
// ============================================================================
static bool find_common_block(const std::vector<llama_token>& a, const std::vector<llama_token>& b,
                              int& bo, int& bn, int& L, int minblk) {
    const int B = 64;
    const int Na = (int)a.size(), Nb = (int)b.size();
    if (Na < B || Nb < B) return false;
    auto h64 = [&](const std::vector<llama_token>& t, int s) {
        uint64_t h = 1469598103934665603ULL;
        for (int k = 0; k < B; k++) { h ^= (uint64_t)(uint32_t)t[s + k]; h *= 1099511628211ULL; }
        return h;
    };
    std::unordered_map<uint64_t, int> idx; idx.reserve(Na);
    for (int i = 0; i + B <= Na; i++) idx.emplace(h64(a, i), i);   // prima occorrenza del 64-gram
    int best = 0, bbo = -1, bbn = -1;
    for (int j = 0; j + B <= Nb; ) {
        auto it = idx.find(h64(b, j));
        if (it == idx.end()) { j++; continue; }
        const int oi = it->second;
        // Verify the B anchor tokens: the map key is only a 64-bit hash, so a
        // collision would otherwise reuse a KV block that isn't actually equal.
        bool anchor_eq = true;
        for (int k = 0; k < B; k++) if (a[oi + k] != b[j + k]) { anchor_eq = false; break; }
        if (!anchor_eq) { j++; continue; }
        int l = 0; while (oi - 1 - l >= 0 && j - 1 - l >= 0 && a[oi - 1 - l] == b[j - 1 - l]) l++;
        int r = 0; while (oi + B + r < Na && j + B + r < Nb && a[oi + B + r] == b[j + B + r]) r++;
        const int len = B + l + r;
        if (len > best) { best = len; bbo = oi - l; bbn = j - l; }
        j += B + r;   // salta oltre il match
    }
    if (best >= minblk) { bo = bbo; bn = bbn; L = best; return true; }
    return false;
}

// ── FAST EVICTION: applica la maschera keep in UNA passata di celle (O(n_kv)) invece di R
// chiamate seq_rm a run contigui (ognuna scandisce tutte le celle → O(R·n_kv)). La maschera è
// la STESSA → keep-set IDENTICO, è puro speedup. Ritorna #scansioni celle effettuate (1 = fast).
// AIS_SNAPKV_FASTEVICT=0 forza il path run-wise (A/B vs il vecchio comportamento).
static int snap_fast_evict(llama_memory_t mem, const std::vector<bool>& keep, int p0 = 0) {
    static const bool fast = [](){ const char* v = getenv("AIS_SNAPKV_FASTEVICT"); return !v || atoi(v) != 0; }();
    const int N = (int) keep.size();
    if (fast) {
        std::vector<int8_t> mask(N);
        for (int i = 0; i < N; i++) mask[i] = keep[i] ? 1 : 0;
        llama_memory_seq_rm_mask(mem, 0, p0, mask.data(), (uint32_t) N);
        return 1;                               // una sola passata di celle
    }
    int run = -1, runs = 0;                     // fallback storico: un seq_rm per run contiguo
    for (int i = 0; i <= N; i++) {
        const bool ev = (i < N) && !keep[i];
        if (ev && run < 0) run = i;
        else if (!ev && run >= 0) { llama_memory_seq_rm(mem, 0, p0 + run, p0 + i); run = -1; runs++; }
    }
    return runs;
}

// FAST TOP-K (gate): seleziona i `target` indici a rilevanza più alta in O(N) con nth_element
// invece dell'O(N log N) del sort completo. Set-IDENTICO al sort: soglia = target-esimo valore,
// si tengono i > soglia e si riempie sui pari-soglia in ordine d'indice (stesso tie-break del
// sort stabile per indice). Marca keep[] (non sovrascrive i già-keep: anchor/recent/speciali).
// AIS_SNAPKV_FASTTOPK=1 lo attiva (default OFF → sort, per riproducibilità bit-identica storica).
static void snap_fast_topk(const std::vector<double>& rel, int N, int target, std::vector<bool>& keep) {
    int already = (int) std::count(keep.begin(), keep.end(), true);
    if (target <= already) return;
    std::vector<double> v; v.reserve(N);
    for (int i = 0; i < N; i++) if (!keep[i]) v.push_back(rel[i]);
    if ((int) v.size() <= target - already) {            // tutti i candidati entrano
        for (int i = 0; i < N; i++) if (!keep[i]) keep[i] = true;
        return;
    }
    const int k = target - already;                       // candidati da aggiungere
    std::nth_element(v.begin(), v.begin() + (k - 1), v.end(), std::greater<double>());
    const double thr = v[k - 1];                          // k-esimo valore più alto tra i candidati
    int added = 0;
    for (int i = 0; i < N && added < k; i++) if (!keep[i] && rel[i] > thr) { keep[i] = true; added++; }
    for (int i = 0; i < N && added < k; i++) if (!keep[i] && rel[i] == thr) { keep[i] = true; added++; } // pari-soglia, ordine d'indice
}

// ── RUNNING EVICTION (AIS_SNAPKV_RUNEVICT, experimental) ─────────────────────────────────────
// Evict DURING the chunked prefill of [lo,N) instead of once at the end, so later chunks attend
// over a smaller KV (attention is O(ctx²) → cuts PREFILL FLOPs on long prompts and big deltas).
// Relies on the append-only allocator (same env, core side) so physical slot == token position
// holds across mid-prefill evictions — the SnapKV relevance hook is slot-indexed.
//
// DESIGN — "conservative accelerator, query-aware authority":
//   • intermediate checkpoints DROP ONLY CLEAR LOSERS: tokens whose ACCUMULATED relevance (summed
//     over all chunks so far = "attended by nobody") is below RUNFLOOR×peak. Dense content → none
//     below floor → nothing dropped (do no harm). This just sheds obvious filler to speed prefill.
//   • the FINAL checkpoint is the AUTHORITY: it ranks survivors by the TRUE final-window relevance
//     and keeps the real target ratio. This defines the cached mask → multi-turn reuse stays
//     query-aware (a weak running signal never decides what gets cached).
//
// Positions [0,lo) are an already-present (compressed) prefix and are NEVER evicted (reuse "do no
// harm"). Fills keep_mask[lo..N); returns kept-count in [lo,N), or -1 to bail to a clean prefill.
static int snap_runevict_prefill(llama_context* ctx, llama_memory_t mem, const llama_vocab* vocab,
                                 llama_batch& batch, const std::vector<llama_token>& toks,
                                 int lo, int N, float keep_ratio, std::vector<bool>& keep_mask) {
    const int    CH       = 512;
    const int    ANCHOR   = 8, RECENT = 64;
    const int    EVERY    = std::max(CH, env_int("AIS_SNAPKV_RUNEVICT_EVERY", 4096));        // checkpoint stride
    const double RKEEP    = std::min(0.99, std::max(0.10, env_float("AIS_SNAPKV_RUNEVICT_KEEP", 0.85)));  // intermediate keep (generous)

    auto is_special = [&](int pos) {
        const llama_token_attr at = llama_vocab_get_attr(vocab, toks[pos]);
        return (at & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0;
    };

    std::vector<char>   live(N, 0);     // live[pos] for pos in [lo,N): prefilled & not evicted (pos == slot)
    std::vector<double> acc (N, 0.0);   // accumulated relevance per position (do-no-harm running signal)
    g_snap.run_mode  = true;
    g_snap.capturing = true;
    g_snap.rel.clear();
    llama_set_eval_callback(ctx, snap_eval_cb, nullptr);
    auto bail = [&]() -> int { g_snap.run_mode = false; llama_set_eval_callback(ctx, nullptr, nullptr); return -1; };

    int next_ckpt = lo + EVERY;
    for (int s = lo; s < N; s += CH) {
        const int e = std::min(N, s + CH);
        g_snap.layers = 0;                                          // reset accumulation for THIS chunk
        batch.n_tokens = 0;
        for (int i = s; i < e; i++) batch_add(batch, toks[i], i, i == N - 1);   // lm_head only on the final token
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "RUNEVICT bail: decode fail @[%d,%d)\n", s, e); return bail(); }
        batch.n_tokens = 0;
        for (int i = s; i < e; i++) live[i] = 1;

        const bool last = (e == N);
        if (g_snap.layers == 0) { if (last) break; else continue; }  // no relevance captured (all-SWA?) → can't evict
        if ((int) g_snap.rel.size() < e) {                          // slot/pos consistency check
            fprintf(stderr, "RUNEVICT bail: rel.size=%zu < e=%d (layers=%d) @[%d,%d)\n",
                    g_snap.rel.size(), e, g_snap.layers, s, e);
            return bail();
        }
        for (int p = lo; p < e; p++) if (live[p]) acc[p] += g_snap.rel[p];   // accumulate the do-no-harm signal

        if (e < next_ckpt && !last) continue;
        next_ckpt += EVERY;

        // candidates = live positions in [lo,e) that are NOT protected (anchor / recent end / special)
        std::vector<int> cand;
        int live_cnt = 0;
        for (int p = lo; p < e; p++) {
            if (!live[p]) continue;
            live_cnt++;
            if (p < lo + ANCHOR || p >= e - RECENT || is_special(p)) continue;
            cand.push_back(p);
        }
        if (cand.empty()) continue;
        const int protected_cnt = live_cnt - (int) cand.size();

        if (!last) {
            // CONSERVATIVE: keep the top RKEEP fraction of live by ACCUMULATED relevance (a token
            // attended by ANY past window survives → do no harm). Generous budget caps the drop at
            // (1−RKEEP) of live, so it sheds only the clear bottom; the final query-aware pass is the
            // authority. NB: a floor relative to the peak is NOT safe — attention is power-law
            // concentrated, so even a tiny floor would drop the majority.
            std::sort(cand.begin(), cand.end(), [&](int a, int b){ return acc[a] > acc[b]; });
            const int target = std::max(protected_cnt, (int)(RKEEP * live_cnt));
            const int keep_c = std::max(0, target - protected_cnt);
            if (keep_c >= (int) cand.size()) continue;              // nothing below the budget
            for (int r = keep_c; r < (int) cand.size(); r++) live[cand[r]] = 0;
        } else {
            // FINAL = query-aware authority: keep the target ratio of survivors by final-window rel.
            std::sort(cand.begin(), cand.end(), [&](int a, int b){ return g_snap.rel[a] > g_snap.rel[b]; });
            const int target = std::max(protected_cnt, (int)((double) keep_ratio * live_cnt));
            const int keep_c = std::max(0, target - protected_cnt);
            if (keep_c >= (int) cand.size()) continue;              // nothing to drop
            for (int r = keep_c; r < (int) cand.size(); r++) live[cand[r]] = 0;
        }

        std::vector<int8_t> maskE(e, 1);                            // keep prefix [0,lo) + protected; evict losers
        for (int p = lo; p < e; p++) maskE[p] = live[p] ? 1 : 0;
        llama_memory_seq_rm_mask(mem, 0, 0, maskE.data(), (uint32_t) e);
        int now = 0; for (int p = lo; p < e; p++) now += live[p];
        fprintf(stderr, "RUNEVICT ckpt @%d: %d→%d live%s\n", e, live_cnt, now,
                last ? " (final, query-aware)" : " (conservative)");
    }

    g_snap.run_mode = false;
    llama_set_eval_callback(ctx, nullptr, nullptr);
    keep_mask.assign(N, false);
    int kept = 0;
    for (int p = lo; p < N; p++) if (live[p]) { keep_mask[p] = true; kept++; }
    return kept;
}

static std::string ais_snapkv_infer(
    llama_context* ctx, const llama_vocab* vocab,
    llama_sampler* smpl, llama_batch& batch,
    const std::string& prompt, int max_tokens, float keep_ratio,
    DeltaState* ds = nullptr, int* out_pt = nullptr, int* out_ct = nullptr,
    bool setup_only = false, int* out_n_past = nullptr)
{
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<llama_token> toks = ::common_tokenize(vocab, prompt, true, true);
    // ── DEDUP PRE-GATE (AIS_SNAPKV_DEDUP): toglie gli n-gram ESATTI ripetuti PRIMA del
    // prefill → quei token non ricevono MAI attenzione/FFN (skip computazione "gratis",
    // solo hashing). Complementare all'eviction SnapKV (che invece PAGA il prefill e poi
    // evicta per rilevanza): qui il ridondante lessicale non entra proprio. Prefix-
    // deterministico → compatibile col delta path. Salta i prompt XML-densi (Cline).
    static const bool snap_dedup = getenv("AIS_SNAPKV_DEDUP") != nullptr;
    bool dedup_strong = false;   // dedup ha già fatto la compressione (lossless) → eviction gentile
    if (snap_dedup && (int)toks.size() > 256) {
        const float xd = quick_xml_density(vocab, toks);
        if (xd < 0.20f) {
            const int before = (int)toks.size();
            std::vector<llama_token> dd = dedup_prefilter(toks, 8, 64, 8);
            if ((int)dd.size() < before) {
                const float removed = (float)(before - (int)dd.size()) / before;
                fprintf(stderr, "SnapKV dedup pre-gate: %d → %d tok (%.1f%% saltati dal prefill, gratis)\n",
                        before, (int)dd.size(), 100.0f * removed);
                toks = std::move(dd);
                // Se il dedup ha già tolto molto (ridondante), il residuo è quasi-unico →
                // NON sovra-comprimere con l'eviction per rilevanza (perde dettagli esatti:
                // misurato 7741→7411). Dedup e eviction servono regimi diversi (ridondante vs denso).
                dedup_strong = (removed >= 0.50f);
            }
        } else {
            fprintf(stderr, "SnapKV dedup pre-gate: saltato (xml-denso %.0f%%)\n", 100.0f * xd);
        }
    }
    // ── LEXGATE (AIS_SNAPKV_LEXGATE): droppa il filler ISOLATO (bassa sorpresa lessicale e
    // NON al confine di un token importante) PRIMA del forward. Sicuro sul codice (token
    // strutturali adiacenti a simboli/identificatori → protetti); comprime prosa/commenti.
    // Model-agnostic. Consigliato per coding (sempre on). Prefix-deterministico entro ±win.
    static const bool snap_lexgate = getenv("AIS_SNAPKV_LEXGATE") != nullptr;
    if (snap_lexgate && (int)toks.size() > 128) {
        static const int   LW = [](){ const char* v = getenv("AIS_LEXGATE_WIN"); return v ? atoi(v) : 3; }();
        static const float LT = [](){ const char* v = getenv("AIS_LEXGATE_THR"); return v ? (float)atof(v) : 2.0f; }();
        static const bool  LC = getenv("AIS_LEXGATE_CODE") != nullptr;   // syntax-aware (protegge il codice)
        const int before = (int)toks.size();
        std::vector<llama_token> lg = lexgate_prefilter(vocab, toks, 8, 64, LW, LT, LC);
        if ((int)lg.size() < before) {
            fprintf(stderr, "SnapKV lexgate%s: %d → %d tok (%.1f%% filler isolato saltato, gratis | win=%d thr=%.1f)\n",
                    LC ? "[code]" : "", before, (int)lg.size(), 100.0f * (before - (int)lg.size()) / before, LW, LT);
            toks = std::move(lg);
            if ((float)(before - (int)toks.size()) / before >= 0.50f) dedup_strong = true;   // molto tolto → eviction gentile
        }
    }
    const int N = (int)toks.size();
    if (N <= 0) { if (out_pt) *out_pt = 0; if (out_ct) *out_ct = 0; return ""; }
    const int CH = 512;

    // Prefill OTTIMIZZATO. pf_plain: prefilla [lo,hi) senza cattura (cb_eval OFF, no lm_head).
    // pf_final: prefilla [lo,N) e attiva la cattura rilevanza SOLO sulle ULTIME W posizioni
    // (la vera finestra-query del prompt, NON la coda dell'ultimo chunk da 512 — che poteva
    // essere minuscola e MANCARE la query → eviction sbagliata). lm_head solo sul token finale.
    // Effetto: niente compute/sync sprecati sui chunk intermedi (più vicino a vanilla) + finestra
    // rilevanza corretta a prescindere dai confini dei chunk. Ritornano false su decode fail.
    auto pf_plain = [&](int lo, int hi) -> bool {
        for (int s = lo; s < hi; s += CH) {
            const int e = std::min(hi, s + CH);
            llama_set_eval_callback(ctx, nullptr, nullptr);
            batch.n_tokens = 0;
            for (int i = s; i < e; i++) batch_add(batch, toks[i], i, false);
            if (llama_decode(ctx, batch) != 0) return false;
        }
        batch.n_tokens = 0; return true;
    };
    auto pf_final = [&](int lo) -> bool {
        const int Wf    = std::min(N - lo, std::max(g_snap.W, 1));   // ultime Wf = finestra-query
        const int split = N - Wf;
        if (!pf_plain(lo, split)) return false;                      // [lo,split): nessuna cattura
        llama_set_eval_callback(ctx, snap_eval_cb, nullptr);         // [split,N): cattura la rilevanza
        for (int s = split; s < N; s += CH) {
            const int e = std::min(N, s + CH);
            batch.n_tokens = 0;
            for (int i = s; i < e; i++) batch_add(batch, toks[i], i, i == N - 1);   // lm_head solo sul finale
            if (llama_decode(ctx, batch) != 0) return false;
        }
        batch.n_tokens = 0; return true;
    };
    // ROUTER (KV-bound gate): prefill [lo,N) SENZA cattura rilevanza (cb_eval OFF ovunque),
    // logits solo sul finale = prefill VANILLA-equivalente (niente side-mul_mat [W×n_kv], niente
    // sync per-split, niente eviction dopo). Usato quando il decode NON sarà KV-bound (contesto
    // corto): la KV compressa non accelererebbe il decode, quindi la rilevanza/eviction è solo tassa.
    auto pf_nocap = [&](int lo) -> bool {
        llama_set_eval_callback(ctx, nullptr, nullptr);
        for (int s = lo; s < N; s += CH) {
            const int e = std::min(N, s + CH);
            batch.n_tokens = 0;
            for (int i = s; i < e; i++) batch_add(batch, toks[i], i, i == N - 1);   // lm_head solo sul finale
            if (llama_decode(ctx, batch) != 0) return false;
        }
        batch.n_tokens = 0; return true;
    };

    // prefisso condiviso con la richiesta precedente
    int P = 0;
    if (ds && ds->valid && !ds->original_toks.empty()) {
        const int maxP = (int)ds->original_toks.size();
        while (P < maxP && P < N && toks[P] == ds->original_toks[P]) P++;
    }
    const bool reuse = (ds && ds->valid && P >= 64 && P < N && (int)ds->tok_kept.size() >= P);

    // RUNNING EVICTION applies only to a first-turn BIG single prefill (not reuse/diff): that's
    // where prefill is the cost and there's no cached prefix to protect. Needs head-room for
    // generation since the append-only allocator can't reuse evicted holes (N + gen ≤ n_ctx).
    // value-based (not mere presence): AIS_SNAPKV_RUNEVICT=0 explicitly disables, =1 enables.
    static const bool runevict_env = [](){ const char* v = getenv("AIS_SNAPKV_RUNEVICT"); return v && atoi(v) != 0; }();
    const bool runevict_ok = runevict_env && !reuse && N >= 4 * CH
                             && N + std::max(512, max_tokens) <= (int) llama_n_ctx(ctx);

    // ── CROSS-TURN DIFF-REUSE (AIS_SNAPKV_DIFF): riusa il PIÙ LUNGO blocco comune (corpo
    // file invariato) anche se cambia ai DUE lati (modifica + domanda nuova), dove il
    // prefix-reuse fallisce. Si tiene SOLO il blocco, lo si sposta a posizione (seq_add →
    // re-rope), e si prefilla solo prefix-change + suffix-change. ──
    static const bool diff_en = getenv("AIS_SNAPKV_DIFF") != nullptr;
    static const int  DIFF_MIN = [](){ const char* v = getenv("AIS_SNAPKV_DIFFMIN"); return v ? atoi(v) : 256; }();
    // seq_add (shift di posizione + re-rope) richiede RoPE 1D: i modelli M-RoPE (vision,
    // es. Qwen-VL) hanno n_pos_per_embd>1 → seq_add asserisce. Gate sul tipo di rope.
    static const bool diff_supported = [&]{
        const llama_rope_type rt = llama_model_rope_type(llama_get_model(ctx));
        const bool ok = rt != LLAMA_ROPE_TYPE_MROPE && rt != LLAMA_ROPE_TYPE_IMROPE && rt != LLAMA_ROPE_TYPE_VISION;
        if (diff_en && !ok) fprintf(stderr, "SnapKV diff-reuse: NON supportato su questo modello (M-RoPE) → disattivo\n");
        return ok;
    }();
    int bo = -1, bn = -1, L = 0;
    bool diff_ok = false;
    if (diff_en && diff_supported && ds && ds->valid && !ds->original_toks.empty()
        && (int)ds->tok_kept.size() >= (int)ds->original_toks.size()
        && find_common_block(ds->original_toks, toks, bo, bn, L, DIFF_MIN)) {
        if (bn + L >= N) L = N - bn - 1;                 // lascia ≥1 token finale fresco (logits per la gen)
        diff_ok = (L >= DIFF_MIN && bn >= 0 && bo >= 0 && bo + L <= (int)ds->original_toks.size() && bn + L < N);
    }

    auto t_pf = std::chrono::steady_clock::now();
    int kept = N;

    if (diff_ok) {
        // tieni SOLO il blocco comune [bo,bo+L); butta prefix-change, suffix-change e gen del turno prima
        llama_memory_seq_rm(mem, 0, 0, bo);
        llama_memory_seq_rm(mem, 0, bo + L, -1);
        const int shift = bn - bo;
        if (shift != 0) llama_memory_seq_add(mem, 0, bo, bo + L, shift);   // sposta il blocco a [bn,bn+L) + re-rope
        // prefilla SOLO ciò che cambia: prima del blocco [0,bn), poi dopo [bn+L,N) (logits sull'ultimo)
        bool ok = pf_plain(0, bn) && pf_final(bn + L);
        batch.n_tokens = 0;
        if (ok) {
            std::vector<bool> nk(N, true);
            for (int j = 0; j < L; j++) nk[bn + j] = ds->tok_kept[bo + j];   // blocco: eredita la maschera (eviction turno-1)
            kept = (int)std::count(nk.begin(), nk.end(), true);
            ds->original_toks = toks; ds->tok_kept = std::move(nk); ds->n_kv = kept; ds->valid = true;
            const int prefilled = bn + (N - (bn + L));
            fprintf(stderr, "SnapKV diff-reuse: blocco comune L=%d riusato | prefill solo %d/%d tok (pre %d + post %d) | %.0fms\n",
                    L, prefilled, N, bn, N - (bn + L),
                    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_pf).count());
        } else {
            // FALLBACK SICURO: il riuso mid-blocco non è gestibile da questa cache (append-order
            // e/o ISWA sliding-window) → pulisci e fai un full prefill normale. Mai crashare.
            fprintf(stderr, "SnapKV diff-reuse: non gestibile su questo modello (cache append/ISWA) → full prefill\n");
            llama_memory_clear(mem, true);
            if (!pf_final(0)) { fprintf(stderr, "snapkv: fallback decode fail\n"); return "[error: decode]"; }
            kept = N;
            ds->original_toks = toks; ds->tok_kept.assign(N, true); ds->n_kv = N; ds->valid = true;
        }
    } else if (reuse) {
        // ── DELTA: riusa il prefisso compresso [0,P), prefilla solo [P,N) ──
        llama_memory_seq_rm(mem, 0, P, -1);   // via tutto da P; con append-only head→used_max_p1=P (delta a [P,..))
        const int D = N - P;
        int kv_pfx = 0;
        for (int i = 0; i < P; i++) if (ds->tok_kept[i]) kv_pfx++;

        // ── DELTA RUNNING EVICTION: se il delta è GRANDE (log/output incollato) la sua attenzione
        // interna O(D²) domina → evict DURANTE il suo prefill (cut FLOPs + comprime). Il prefisso
        // [0,P) resta intatto (già compresso query-aware nel suo turno → reuse "do no harm"). Stesso
        // schema del primo turno: conservative intermedio + finale query-aware (autorità della maschera).
        const bool delta_runevict = runevict_env && D >= 4 * CH
                                    && N + std::max(512, max_tokens) <= (int) llama_n_ctx(ctx);
        bool delta_done = false;
        if (delta_runevict) {
            g_snap.logged = false;
            std::vector<bool> dkm;
            const int rk = snap_runevict_prefill(ctx, mem, vocab, batch, toks, /*lo=*/P, N, keep_ratio, dkm);
            if (rk >= 0) {
                std::vector<bool> nk(N, false);
                for (int i = 0; i < P; i++) nk[i] = ds->tok_kept[i];   // prefisso: maschera invariata
                for (int i = P; i < N; i++) nk[i] = dkm[i];
                kept = kv_pfx + rk;
                ds->original_toks = toks; ds->tok_kept = std::move(nk); ds->n_kv = kept;
                fprintf(stderr, "SnapKV delta RUNEVICT: prefisso=%d(kv %d) + delta=%d→%d → %d tok | %.0fms\n",
                        P, kv_pfx, D, rk, kept,
                        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_pf).count());
                delta_done = true;
            } else {
                fprintf(stderr, "SnapKV delta RUNEVICT bail → normal delta prefill\n");
                llama_memory_seq_rm(mem, 0, P, -1);   // clear any partially-evicted delta → clean redo below
            }
        }
        if (delta_done) { /* handled above */ }
        else {
        if (!pf_final(P)) { fprintf(stderr, "snapkv: delta decode fail\n"); return "[error: decode]"; }
        std::vector<bool> nk(N, true);
        for (int i = 0; i < P; i++) nk[i] = ds->tok_kept[i];      // prefisso: maschera invariata
        // ── AUTO-MASS PER-TURNO: comprimi anche il DELTA se grande+ridondante (es. log/output
        // incollato). Posizione-indicizzata: g_snap.rel[i] = rilevanza della posizione assoluta i.
        // Tiene recent + speciali + top-rilevanza per massa (auto-MASS dalla ridondanza del delta).
        static const bool AUTO = getenv("AIS_SNAPKV_AUTO") != nullptr;
        int kept_delta = D;
        // Senza append-only il prefisso è compattato → delta alle celle [kv_pfx, kv_pfx+D); con
        // append-only slot==posizione → delta alle celle [P, P+D). Indicizza g_snap.rel di conseguenza.
        const int dbase = runevict_env ? P : kv_pfx;
        if (AUTO && D >= 256 && (int)g_snap.rel.size() >= dbase + D) {
            const int NG = 16; std::unordered_set<uint64_t> seen; long rep = 0, tot = 0;  // n-gram lungo: non conta i ripetuti STRUTTURALI brevi dell'XML
            for (int i = P; i + NG <= N; i++) { uint64_t h = 1469598103934665603ULL;
                for (int j = 0; j < NG; j++) { h ^= (uint64_t)(uint32_t)toks[i + j]; h *= 1099511628211ULL; }
                tot++; if (!seen.insert(h).second) rep++; }
            const double red = tot ? (double)rep / tot : 0.0;
            const double mass = std::max(0.70, std::min(0.97, 0.95 - 0.45 * red));
            const int drecent = 64;
            std::vector<bool> dk(D, false);
            for (int j = std::max(0, D - drecent); j < D; j++) dk[j] = true;     // query/risposta recente
            for (int j = 0; j < D; j++) {
                const llama_token_attr at = llama_vocab_get_attr(vocab, toks[P + j]);
                if ((at & LLAMA_TOKEN_ATTR_CONTROL) || (at & LLAMA_TOKEN_ATTR_USER_DEFINED)) dk[j] = true;
            }
            std::vector<int> di(D); for (int j = 0; j < D; j++) di[j] = j;
            std::sort(di.begin(), di.end(), [&](int a, int b){ return g_snap.rel[dbase + a] > g_snap.rel[dbase + b]; });
            double dtot = 0.0; for (int j = 0; j < D; j++) if (!dk[j]) dtot += g_snap.rel[dbase + j];
            double acc = 0.0;
            for (int r = 0; r < D && acc < mass * dtot; r++) { int j = di[r]; if (!dk[j]) { dk[j] = true; acc += g_snap.rel[dbase + j]; } }
            snap_fast_evict(mem, dk, P);                                         // evict delta non tenuti (single-pass, offset P)
            kept_delta = (int)std::count(dk.begin(), dk.end(), true);
            for (int j = 0; j < D; j++) nk[P + j] = dk[j];
            fprintf(stderr, "SnapKV delta auto-MASS: red=%.3f mass=%.2f → delta %d→%d\n", red, mass, D, kept_delta);
        }
        kept = kv_pfx + kept_delta;
        ds->original_toks = toks; ds->tok_kept = std::move(nk); ds->n_kv = kept;
        fprintf(stderr, "SnapKV delta: prefisso=%d (kv compresso=%d) + delta=%d (tenuti %d) → %d tok | %.0fms\n",
                P, kv_pfx, D, kept_delta, kept,
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_pf).count());
        }   // chiude l'else del delta-runevict (path normale pf_final + auto-MASS)
    } else if (runevict_ok) {
        // ── RUNNING EVICTION: evict during prefill (cuts prefill FLOPs on long single prompts) ──
        llama_memory_clear(mem, true);
        g_snap.logged = false;
        std::vector<bool> km;
        const int rk = snap_runevict_prefill(ctx, mem, vocab, batch, toks, /*lo=*/0, N, keep_ratio, km);
        if (rk < 0) {
            // bail → clean full prefill fallback (no compression), never corrupts
            fprintf(stderr, "RUNEVICT: bailed (validation/decode) → full prefill fallback\n");
            llama_memory_clear(mem, true);
            g_snap.rel.clear(); g_snap.logged = false;
            if (!pf_final(0)) { fprintf(stderr, "snapkv: runevict fallback decode fail\n"); return "[error: decode]"; }
            kept = N;
            if (ds) { ds->original_toks = toks; ds->tok_kept.assign(N, true); ds->n_kv = N; ds->valid = true; }
        } else {
            kept = rk;
            if (ds) { ds->original_toks = toks; ds->tok_kept = std::move(km); ds->n_kv = kept; ds->valid = true; }
            fprintf(stderr, "SnapKV RUNEVICT: %d→%d tok (%.0f%% compresso, running) | %.0fms\n",
                    N, kept, 100.0f * (N - kept) / N,
                    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_pf).count());
        }
    } else {
        // ── PRIMO TURNO: prefill completo + eviction SnapKV ──
        llama_memory_clear(mem, true);
        g_snap.rel.clear(); g_snap.logged = false;
        g_snap.rb_ms = g_snap.evict_ms = 0.0; g_snap.rb_bytes = 0; g_snap.evict_calls = 0;  // profile per-richiesta
        // ── ROUTER — LENGTH AXIS (KV-bound gate) ────────────────────────────────────────────────
        // L'eviction per-rilevanza (e la sua cattura) ripaga SOLO quando il decode è KV-bound =
        // contesto abbastanza lungo (E2B: ~flat fino a 2k, poi 36→20→14 tok/s a 2k→6k). Sotto la
        // soglia il decode non rilegge una KV grande → comprimerla NON accelera, ma cattura+eviction
        // COSTANO → si SALTANO (prefill vanilla-equivalente via pf_nocap). I gain PRE-forward
        // (dedup/lexgate, già applicati a `toks`) e multi-turno (delta, ramo sopra) NON dipendono da
        // qui. Attivo solo nel path AUTO (il router). Soglia: AIS_SNAPKV_KVBOUND (default 2500),
        // 0 = eviction sempre. NB: model-dependent → tarare sulla curva tok/s-vs-KV del modello.
        static const bool AUTO     = getenv("AIS_SNAPKV_AUTO") != nullptr;
        static const int  KVBOUND  = [](){ const char* v = getenv("AIS_SNAPKV_KVBOUND"); return v ? atoi(v) : 2500; }();
        const bool kvbound_skip = AUTO && KVBOUND > 0 && N < KVBOUND;
        if (kvbound_skip) {
            if (!pf_nocap(0)) { fprintf(stderr, "snapkv router: decode fail\n"); return "[error: decode]"; }
            kept = N;
            fprintf(stderr, " SnapKV router: N=%d < KV-bound %d → decode non KV-bound → niente rilevanza/eviction (prefill vanilla-equ.) | %.0fms\n",
                    N, KVBOUND, std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_pf).count());
            if (ds) { ds->original_toks = toks; ds->tok_kept.assign(N, true); ds->n_kv = N; ds->valid = true; }
        } else {
        if (!pf_final(0)) { fprintf(stderr, "snapkv: decode fail\n"); return "[error: decode]"; }
        batch.n_tokens = 0;
        const double pf_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_pf).count();
        std::vector<bool> keep(N, true);
        static const double MASS_ENV = [](){ const char* v = getenv("AIS_SNAPKV_MASS"); return v ? atof(v) : 0.0; }();
        double MASS = MASS_ENV;
        if (AUTO) {
            // AUTO-MASS (idea ais_lex, zero forward-pass): la ridondanza n-gram del prompt
            // sceglie il MASS. Contenuto UNICO/denso (codice) → ridondanza bassa → MASS alto
            // (tieni di più, recupero importante); RIDONDANTE (log/filler) → MASS basso (comprimi).
            // Misurato: codice red≈0.01→0.94 | Cline≈0.14→0.89 | filler≈0.94→0.70.
            const int NG = 16; std::unordered_set<uint64_t> seen; long rep = 0, tot = 0;  // n-gram lungo: non conta i ripetuti STRUTTURALI brevi dell'XML
            for (int i = 0; i + NG <= N; i++) {
                uint64_t h = 1469598103934665603ULL;
                for (int j = 0; j < NG; j++) { h ^= (uint64_t)(uint32_t)toks[i + j]; h *= 1099511628211ULL; }
                tot++; if (!seen.insert(h).second) rep++;
            }
            const double red = tot ? (double)rep / tot : 0.0;
            MASS = std::max(0.70, std::min(0.97, 0.95 - 0.45 * red));
            fprintf(stderr, "SnapKV auto-MASS: ridondanza8=%.3f → MASS=%.2f\n", red, MASS);
        }
        if (dedup_strong && N > 0) {
            fprintf(stderr, " SnapKV: dedup forte → eviction per-rilevanza SALTATA (residuo quasi-unico, %d tok)\n", N);
        } else if ((int)g_snap.rel.size() >= N && (MASS > 0.0 || keep_ratio < 0.999f)) {
            // FAST TOP-K (gate): col solo keep_ratio (count fisso) basta nth_element O(N) → si
            // SALTA il sort O(N log N). MASS accumula per massa → serve l'ordine → sort pieno.
            static const bool fast_topk = [](){ const char* v = getenv("AIS_SNAPKV_FASTTOPK"); return v && atoi(v) != 0; }();
            const bool use_fast_topk = fast_topk && MASS <= 0.0;
            std::vector<int> idx;
            if (!use_fast_topk) {
                idx.resize(N); for (int i = 0; i < N; i++) idx[i] = i;
                std::sort(idx.begin(), idx.end(), [](int a, int b){ return g_snap.rel[a] > g_snap.rel[b]; });
            }
            const int anchor = 8, recent = 64;
            keep.assign(N, false);
            for (int i = 0; i < anchor && i < N; i++) keep[i] = true;            // anchor (include il <bos> sink)
            for (int i = std::max(0, N - recent); i < N; i++) keep[i] = true;     // turni/query recenti interi
            // PROTEGGI i token SPECIALI/strutturali (control + user-defined: <bos>, <start_of_turn>,
            // <end_of_turn>, marker di canale/ruolo...): evictarli romperebbe la struttura del
            // prompt. Sono pochi; restano esclusi anche dalla massa (keep=true → non contati).
            int n_spec = 0;
            for (int i = 0; i < N; i++) {
                if (keep[i]) continue;
                const llama_token_attr at = llama_vocab_get_attr(vocab, toks[i]);
                if ((at & LLAMA_TOKEN_ATTR_CONTROL) || (at & LLAMA_TOKEN_ATTR_USER_DEFINED)) { keep[i] = true; n_spec++; }
            }
            if (n_spec) fprintf(stderr, "SnapKV: protetti %d token speciali (struttura)\n", n_spec);
            kept = (int)std::count(keep.begin(), keep.end(), true);
            if (MASS > 0.0) {
                // KEEP ADATTIVO (Step 2): tieni i top-rilevanza finché coprono MASS della massa
                // di attenzione del "middle" (anchor/recent esclusi → SINK-SKIP del <bos>).
                // Rilevanza concentrata → pochi token; piatta (tutto serve) → molti. Auto-sicuro.
                double tot = 0.0; for (int i = 0; i < N; i++) if (!keep[i]) tot += g_snap.rel[i];
                double acc = 0.0;
                for (int r = 0; r < N && acc < MASS * tot; r++) { int i = idx[r]; if (!keep[i]) { keep[i] = true; acc += g_snap.rel[i]; kept++; } }
                // FLOOR min-keep (turno-1): l'XML strutturale "sembra" ridondante e l'attenzione è
                // concentrata → la massa potrebbe tenere troppo poco ed evictare le ISTRUZIONI di
                // sistema (→ il modello divaga). Garantisce ≥ MINKEEP·N (default 0.5).
                static const double MINKEEP = [](){ const char* v = getenv("AIS_SNAPKV_MINKEEP"); return v ? atof(v) : 0.5; }();
                const int floor_k = (int)(MINKEEP * N);
                for (int r = 0; r < N && kept < floor_k; r++) { int i = idx[r]; if (!keep[i]) { keep[i] = true; kept++; } }
                fprintf(stderr, "SnapKV keep adattivo (mass=%.2f, floor=%.2f) → %d/%d tenuti\n", MASS, MINKEEP, kept, N);
            } else {
                const int target = std::max(anchor + recent, (int)(N * keep_ratio));
                if (use_fast_topk) {
                    snap_fast_topk(g_snap.rel, N, target, keep);          // O(N), set-identico al sort
                    kept = (int)std::count(keep.begin(), keep.end(), true);
                } else {
                    for (int r = 0; r < N && kept < target; r++) { int i = idx[r]; if (!keep[i]) { keep[i] = true; kept++; } }
                }
            }
            // ── REL-FLOOR — compressione SICURA a OGNI lunghezza, anche CORTA ──────────────
            // Difetto sui prompt CORTI/densi (MMLU/AIME): il budget (MASS o keep_ratio) FORZA
            // l'eviction anche quando NON c'è nulla di davvero ridondante → butta token che
            // servono (opzioni di risposta, dati) = "falsa" il bench. Fix guidato dal CONTENUTO,
            // non dalla LUNGHEZZA (niente gate 2k): non evictare MAI un token del middle la cui
            // rilevanza-alla-query supera REL_FLOOR×(picco del middle). Prompt denso (anche corto)
            // → nessun token sotto soglia → si tiene ~tutto = niente degrado; filler vero (rel≈0)
            // → resta droppato = comprime dove c'è da comprimere. Solo ALZA i kept (mai abbassa)
            // → conservativo, monotòno. Default 0 = OFF (Cline maturo + sweep a keep-fisso invariati).
            static const double RELFLOOR = [](){ const char* v = getenv("AIS_SNAPKV_RELFLOOR"); return v ? atof(v) : 0.0; }();
            if (RELFLOOR > 0.0 && N > anchor + recent) {
                double mid_max = 0.0;
                for (int i = anchor; i < N - recent; i++) mid_max = std::max(mid_max, (double)g_snap.rel[i]);
                const double rthr = RELFLOOR * mid_max;
                int saved = 0;
                for (int i = anchor; i < N - recent; i++)
                    if (!keep[i] && (double)g_snap.rel[i] >= rthr) { keep[i] = true; kept++; saved++; }
                if (saved) fprintf(stderr, " SnapKV rel-floor %.2f×picco → riprotetti %d token rilevanti → %d/%d tenuti\n", RELFLOOR, saved, kept, N);
            }
            // ── REDUNDANCY-GATE — compressione SICURA a OGNI lunghezza (scelta 12/06) ──────
            // Evicta SOLO token lessicalmente RIDONDANTI: la prima occorrenza di ogni n-gram è
            // UNICA → protetta, le RIPETIZIONI sono droppabili. I candidati all'eviction qui sono
            // già i bassa-rilevanza (MASS/ratio sopra), quindi droppa = bassa-rel ∩ ridondante.
            // Prompt DENSO (zero ripetizioni, anche CORTO) → tenuto INTERO = niente degrado;
            // prompt con filler (anche corto) → comprime. Guidato dal CONTENUTO, non da N (niente
            // gate 2k). Solo ALZA i kept (mai abbassa). Default off (Cline maturo + sweep invariati).
            static const bool REDGATE = getenv("AIS_SNAPKV_REDGATE") != nullptr;
            if (REDGATE && N > anchor + recent) {
                static const int RNG = [](){ const char* v = getenv("AIS_SNAPKV_REDNG"); return v ? atoi(v) : 4; }();
                std::unordered_set<uint64_t> seen;
                std::vector<bool> redundant(N, false);
                for (int i = 0; i + RNG <= N; i++) {
                    uint64_t h = 1469598103934665603ULL;
                    for (int j = 0; j < RNG; j++) { h ^= (uint64_t)(uint32_t)toks[i + j]; h *= 1099511628211ULL; }
                    if (!seen.insert(h).second) redundant[i] = true;   // n-gram già visto → token i droppabile
                }
                int saved = 0;
                for (int i = anchor; i < N - recent; i++)
                    if (!keep[i] && !redundant[i]) { keep[i] = true; kept++; saved++; }   // UNICO → riproteggi
                if (saved) fprintf(stderr, "SnapKV redgate(NG=%d): riprotetti %d token UNICI → %d/%d tenuti\n", RNG, saved, kept, N);
            }
            // ── LIVE EVICTION DUMP (viz) ──
            // AIS_EVICT_DUMP=path → one JSONL record with, for every token, its detokenized
            // piece + the REAL streaming SnapKV decision (kept flag) + attention relevance +
            // special-token flag. Drives the real-prompt eviction GIF (true model decisions).
            if (const char* evict_dump = getenv("AIS_EVICT_DUMP")) {
                // unique_ptr deleter: the JSON-escaping below allocates (std::string) and can
                // throw bad_alloc — RAII guarantees the handle is closed on every exit path.
                std::unique_ptr<FILE, int(*)(FILE*)> efg(fopen(evict_dump, "a"), &fclose);
                if (FILE* ef = efg.get()) {
                    fprintf(ef, "{\"slice_N\":%d,\"kept\":%d,\"mass\":%.3f,\"tokens\":[", N, kept, MASS);
                    char pc[512];
                    const bool have_rel = (int)g_snap.rel.size() >= N;
                    for (int i = 0; i < N; i++) {
                        int len = llama_token_to_piece(vocab, toks[i], pc, (int)sizeof(pc) - 1, 0, true);
                        std::string s(pc, len > 0 ? len : 0), esc;
                        for (char c : s) {
                            switch (c) {
                                case '"':  esc += "\\\""; break;
                                case '\\': esc += "\\\\"; break;
                                case '\n': esc += "\\n";  break;
                                case '\r': esc += "\\r";  break;
                                case '\t': esc += "\\t";  break;
                                default:
                                    if ((uint8_t)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); esc += b; }
                                    else esc += c;
                            }
                        }
                        const llama_token_attr at = llama_vocab_get_attr(vocab, toks[i]);
                        const int spec = (at & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) ? 1 : 0;
                        fprintf(ef, "%s{\"t\":\"%s\",\"k\":%d,\"s\":%.4f,\"x\":%d}",
                                i ? "," : "", esc.c_str(), keep[i] ? 1 : 0,
                                have_rel ? g_snap.rel[i] : 0.0f, spec);
                    }
                    fprintf(ef, "]}\n");
                }
            }
            auto t_ev = std::chrono::steady_clock::now();
            const int runs = snap_fast_evict(mem, keep);   // O(n_kv) singola passata (vs O(R·n_kv))
            if (g_snap.profile) {
                g_snap.evict_ms    += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_ev).count();
                g_snap.evict_calls += runs;
            }
            fprintf(stderr, " SnapKV: %d→%d tok (%.0f%% compresso) | prefill %.0fms | rel n_kv=%zu\n",
                    N, kept, 100.0f * (N - kept) / N, pf_ms, g_snap.rel.size());
            if (g_snap.profile)
                fprintf(stderr, "PROFILE: readback %.1fms / %.1fMB | eviction %.2fms / %d scan (%s, N=%d → %ld cell-scan) | full-readback evitato %.0f×\n",
                        g_snap.rb_ms, g_snap.rb_bytes / 1e6, g_snap.evict_ms, runs,
                        runs <= 1 ? "mask single-pass O(N)" : "run-wise O(R·N)", N, (long)runs * N,
                        g_snap.W > 0 ? 512.0 / g_snap.W : 1.0);
        } else {
            fprintf(stderr, " SnapKV: rel n_kv=%zu (N=%d)/keep=%.2f → no eviction (prefill %.0fms)\n",
                    g_snap.rel.size(), N, keep_ratio, pf_ms);
        }
        if (ds) { ds->original_toks = toks; ds->tok_kept = keep; ds->n_kv = kept; ds->valid = true; }
        }   // chiude l'else del KV-bound gate (router)
    }
    if (out_pt) *out_pt = kept;

    // ── STREAMING SETUP-ONLY ─────────────────────────────────────────────────
    // Il path streaming (content_provider) ha il SUO loop di generazione: qui facciamo
    // SOLO il setup SnapKV (dedup/lexgate + prefill + cattura rilevanza + eviction già
    // fatti sopra) e torniamo la KV compressa pronta. n_past = N (l'eviction è "surgery":
    // tiene le posizioni ORIGINALI, quindi la generazione prosegue da N anche se in KV
    // restano solo `kept` celle). Spegniamo la cattura come fa il loop batch (decode veloce);
    // il chiamante streaming la ri-arma a fine risposta. out_pt resta `kept` (token compressi).
    if (setup_only) {
        g_snap.capturing = false;
        llama_set_eval_callback(ctx, nullptr, nullptr);
        if (out_n_past) *out_n_past = N;
        if (out_ct)     *out_ct     = 0;
        return "";
    }

    // generazione + CoT-cut (come ais_infer): col contesto compresso il modello tende a
    // dilungarsi nel "thought"; tronchiamo il flusso a bassa informazione e passiamo alla
    // risposta. Sempre attivo in SnapKV (necessario per non rimanere nel pensiero).
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto envf = [](const char* k, double d){ const char* v = getenv(k); return v ? atof(v) : d; };
    const double TLOW = envf("AIS_COT_TLOW", 0.05);
    const int    WIN  = (int)envf("AIS_COT_WIN", 24);
    const int    MINT = (int)envf("AIS_COT_MIN", 48);
    // AIS_COT_OFF: disattiva il CoT-cut (ablation: eviction-solo vs eviction+CoT-cut).
    // DEFAULT = attivo: in snapkv è di fatto richiesto (con contesto compresso il modello
    // tende a non uscire dal "thought" → content vuoto). Chiude la divergenza doc/codice.
    static const bool cot_off = getenv("AIS_COT_OFF") != nullptr;
    // Il CoT-cut usa i marker di canale Gemma/harmony (<channel|>) → ha senso SOLO sul
    // fork Gemma-4 (modello "thinking"). In SnapKV GENERICO (qualsiasi modello) il tag
    // è privo di significato e il parser di canale non si applica → disattivato di
    // default; AIS_COT_FORCE per riabilitarlo esplicitamente.
    // CoT-cut DEFAULT-ON: also honour AIS_COMPRESS_COT (set by OMNI for thinking/gemma models),
    // so the non-streaming batch path matches the streaming path instead of needing the fork route.
    const bool cot_enabled = !cot_off && (g_snap.fork || getenv("AIS_COMPRESS_COT") != nullptr
                                          || getenv("AIS_COT_FORCE") != nullptr);
    const int think_cap = cot_think_cap(max_tokens);   // tetto pensiero → spazio garantito per la risposta
    const std::vector<llama_token> end_tag = ::common_tokenize(vocab, "<channel|>", false, true);
    // ── REASONING EVICTION (AIS_REASON_EVICT): a fine "pensiero" evicta dalla KV i token
    // di ragionamento a BASSA sorpresa (filler: "hmm, let me think, so...") tenendo quelli
    // SOSTANZIALI (alta sorpresa) + la conclusione recente. → la RISPOSTA decodifica contro
    // una KV più piccola = più veloce nel regime KV-bound (contesto grande, vedi SNAPKV_NOTES
    // §kv-bound). RIUSA la sorpresa già calcolata per il CoT-cut → zero computazione extra.
    static const bool reason_evict = getenv("AIS_REASON_EVICT") != nullptr;
    static const double reason_keep = [](){ const char* v = getenv("AIS_REASON_KEEP"); return v ? atof(v) : 0.5; }();
    std::vector<std::pair<int,float>> th;   // (posizione KV, sorpresa) dei token di pensiero
    // Quando reason_evict è OFF la sorpresa serve solo al CoT-cut → calcolala SOLO da
    // (MINT−WIN) in poi (prima il taglio non può scattare) = meno scan full-vocab.
    const int surp_from = reason_evict ? 0 : std::max(0, MINT - WIN);
    bool in_thought = cot_enabled; int thought_toks = 0, low_run = 0;
    std::string result; result.reserve(1024);
    int gen = 0, n_past = N;
    // GEN: la rilevanza è già catturata nel prefill. AZZERA il cb_eval sullo scheduler →
    // il decode torna a calcolare il grafo in un colpo (niente sync per-nodo) = veloce come
    // vanilla. (Il flag capturing resta come rete di sicurezza.) Ripristinato dopo il loop.
    g_snap.capturing = false;
    llama_set_eval_callback(ctx, nullptr, nullptr);
    auto t_gen = std::chrono::steady_clock::now();
    for (int i = 0; i < max_tokens; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);
        double cur_surp = -1.0;
        if (cot_enabled && in_thought && !end_tag.empty()) {
            if (thought_toks >= surp_from) {           // sorpresa solo quando può servire (riduzione)
                const float* lg = llama_get_logits_ith(ctx, -1);
                cur_surp = cot_surprise(lg, id, n_vocab);   // G1: top-K logsumexp (≈esatto)
                low_run  = (cur_surp < TLOW) ? low_run + 1 : 0;
            }
            thought_toks++;
            if (thought_toks >= MINT && (low_run >= WIN || (think_cap > 0 && thought_toks >= think_cap))) {
                // ── REASONING EVICTION: prima di rispondere, togli dalla KV il pensiero filler ──
                if (reason_evict && (int) th.size() > 2 * WIN) {
                    const int M = (int) th.size();
                    std::vector<char> keepT(M, 0);
                    for (int j = std::max(0, M - WIN); j < M; j++) keepT[j] = 1;       // conclusione recente
                    std::vector<int> ord(M); for (int j = 0; j < M; j++) ord[j] = j;
                    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return th[a].second > th[b].second; });
                    int kept_t = 0; for (int j = 0; j < M; j++) kept_t += keepT[j];
                    const int target = std::max((int)(reason_keep * M), WIN);          // alta-sorpresa fino a KEEP·M
                    for (int r = 0; r < M && kept_t < target; r++) { int j = ord[r]; if (!keepT[j]) { keepT[j] = 1; kept_t++; } }
                    int evicted = 0, run0 = -1;                                         // evict (run contigui di posizioni)
                    for (int j = 0; j <= M; j++) {
                        const bool ev = (j < M) && !keepT[j];
                        if (ev && run0 < 0) run0 = j;
                        else if (!ev && run0 >= 0) { llama_memory_seq_rm(mem, 0, th[run0].first, th[j-1].first + 1); evicted += j - run0; run0 = -1; }
                    }
                    if (evicted) fprintf(stderr, "Reason-evict: pensiero %d→%d tok in KV (−%d filler, keep=%.2f)\n", M, M - evicted, evicted, reason_keep);
                }
                for (llama_token et : end_tag) {
                    char b[128]; int bn = llama_token_to_piece(vocab, et, b, sizeof(b), 0, true);
                    if (bn > 0) result.append(b, bn);
                    batch.n_tokens = 0; batch_add(batch, et, n_past, true); n_past++; gen++;
                    if (llama_decode(ctx, batch) != 0) { i = max_tokens; break; }
                }
                in_thought = false;
                fprintf(stderr, " SnapKV CoT: taglio a %d tok di pensiero\n", thought_toks);
                continue;
            }
        }
        if (ais_is_eog(vocab, id)) break;
        char buf[128]; int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) break;
        result.append(buf, n);
        if (in_thought && tail_contains(result, n, "<channel|>", 10)) in_thought = false;
        if (reason_evict && in_thought && cur_surp >= 0.0) th.push_back({ n_past, (float) cur_surp });   // token di pensiero
        batch.n_tokens = 0; batch_add(batch, id, n_past, true); n_past++; gen++;
        if (llama_decode(ctx, batch) != 0) break; batch.n_tokens = 0;
    }
    const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen).count();
    g_snap.capturing = true;                              // ripristina per il prefill successivo
    llama_set_eval_callback(ctx, snap_eval_cb, nullptr);  // ri-arma la cattura per il prossimo prefill
    fprintf(stderr, " SnapKV Gen: %d tok in %.0fms (%.1f tok/s)\n", gen, gen_ms, gen > 0 ? 1000.0 * gen / gen_ms : 0.0);
    if (out_ct) *out_ct = gen;
    return result;
}

// Rileva una richiesta di CODING dal PROMPT (prima della generazione → niente trap) e, SE OPT-IN
// (AIS_COT_NOCODE=1), disattiva il CoT-cut per quella richiesta. DEFAULT = OFF: misurato che
// togliere il taglio sul coding PEGGIORA i modelli thinking (budget-starvation: gemma-E2B 100%→33%,
// gemma-26B 66.7%→0%) — il taglio è code-SAFE (mai tronca la risposta) E necessario perché il modello
// risponda entro il budget. Quindi di default il taglio resta ATTIVO anche sul coding. Opt-in solo
// se vuoi che il modello ragioni sul codice senza alcun taglio (accettando il rischio budget).
static bool prompt_is_coding(const std::string& s) {
    if (!getenv("AIS_COT_NOCODE") || atoi(getenv("AIS_COT_NOCODE")) == 0) return false;  // OPT-IN (default OFF)
    if (s.find("```") != std::string::npos) return true;                                 // code fence
    int ang = 0; for (char c : s) if (c == '<' || c == '>') ang++;                       // XML/tool-def (Cline)
    if (s.size() > 200 && (long)ang * 1000 / (long)s.size() > 25) return true;           // >2.5% parentesi angolari (64-bit: no overflow su prompt MB)
    static const char* kw[] = {"def ", "function ", "class ", "import ", "#include", "public ",
        "const ", "return ", "write a function", "implement ", "fix the bug", "refactor", "debug",
        "stack trace", "Traceback", "</", "/>", "() {", ");", "```", "code block", "the function"};
    for (auto k : kw) if (s.find(k) != std::string::npos) return true;
    return false;
}

// ==========================================
// MAIN
// ==========================================
int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "model.gguf";
    std::string adapt_mode  = "fixed";
    float       adapt_param = 0.0f;
    AISProbConfig cfg;

    // Tolerant numeric parsing: a non-numeric CLI argument falls back to the
    // default instead of throwing an uncaught exception that aborts the process.
    auto safe_stof = [](const char* s, float d) { try { return std::stof(s); } catch (...) {
        fprintf(stderr, " Invalid number '%s', using %.3f\n", s, d); return d; } };
    auto safe_stoi = [](const char* s, int d)   { try { return std::stoi(s); } catch (...) {
        fprintf(stderr, " Invalid integer '%s', using %d\n", s, d); return d; } };

    if (argc > 3) {
        adapt_mode  = argv[3];
        adapt_param = safe_stof(argv[2], adapt_param);
    } else if (argc > 2) {
        std::string a2 = argv[2];
        if (a2 != "--server" && a2 != "--max-tokens" && a2 != "--ctx")
            cfg.surprise_threshold = safe_stof(a2.c_str(), cfg.surprise_threshold);
    }

    int         server_port = -1;
    int         max_tokens  = 4096;
    std::string server_host = "0.0.0.0";
    // KV cache + flash-attn. Defaults preserve historical ais_prob behavior
    // (f16 KV, flash AUTO) so existing validated runs are unchanged; the bench
    // opts into q8_0 + flash-on to match llama-server's vanilla config.
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    auto parse_fa = [](const std::string& v) {
        if (v == "on"  || v == "1" || v == "true")  return LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (v == "off" || v == "0" || v == "false") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--server"       && i + 1 < argc) server_port   = safe_stoi(argv[++i], server_port);
        if (a == "--host"         && i + 1 < argc) server_host   = argv[++i];
        if (a == "--max-tokens"   && i + 1 < argc) max_tokens    = safe_stoi(argv[++i], max_tokens);
        if (a == "--ctx"          && i + 1 < argc) cfg.ctx_limit = safe_stoi(argv[++i], cfg.ctx_limit);
        if (a == "--cache-type-k" && i + 1 < argc) type_k        = kv_type_from_str(argv[++i]);
        if (a == "--cache-type-v" && i + 1 < argc) type_v        = kv_type_from_str(argv[++i]);
        if (a == "--flash-attn"   && i + 1 < argc) fa            = parse_fa(argv[++i]);
    }

    llama_backend_init();
    struct BackendGuard { ~BackendGuard() { llama_backend_free(); } } backend_guard;  // freed last

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    ModelPtr model_guard(llama_model_load_from_file(model_path, mparams));
    llama_model* model = model_guard.get();
    if (!model) { fprintf(stderr, "Failed to load model: %s\n", model_path); return 1; }

    const auto* vocab   = llama_model_get_vocab(model);
    const int   n_vocab = llama_vocab_n_tokens(vocab);

    // ════════════════════════════════════════════════════════════════════════
    // SnapKV BEST — la "forma finale": combina TUTTI i gain validati con un solo
    // flag, auto-selezionando la rotta di rilevanza migliore per il modello:
    //   • Gemma-4 → fork tuned (AIS_SNAPKV, flash ON, ais_snap_rel pre-sommato)
    //   • altri   → hook generico (AIS_SNAPKV_FA, flash ON, build_attn_mha)
    // + auto-MASS (compressione guidata dalla ridondanza) + dedup pre-gate (salta dal
    // prefill i ripetuti esatti, gratis) + KV q8_0 (½ memoria) + delta multi-turno +
    // CoT-cut (riduzione ragionamento). Ogni sub-flag esplicito ha la precedenza.
    // ════════════════════════════════════════════════════════════════════════
    // AIS_SNAPKV_CODE = BEST + lexgate SYNTAX-AWARE (gate per il coding: comprime prosa/
    // commenti, protegge codice/identificatori/valori). Implica BEST.
    // ════════════════════════════════════════════════════════════════════════
    // AIS_OMNI — LA MODALITÀ FINALE (una flag, tutto incluso, chat + Cline, STREAMING + ogni
    // modello). = SnapKV (auto-route: gemma→fork tuned, ogni altro modello→hook FA generico flash-ON,
    // "compressione attaccata su FA") + dedup pre-gate + delta multi-turno + auto-MASS +
    // gate KV-bound (nessuna tassa sui prompt corti) + CoT-cut SOLO sui modelli thinking (gemma).
    // KV f16 (AIS_SNAPKV_KVQ8=1 → ½ memoria). NIENTE lexgate → prosa/chat intatta.
    //   AIS_OMNI_CODE = OMNI + lexgate syntax-aware (massima compressione su codice/commenti).
    // È il successore consigliato di BEST (che resta come alias). Ora gira anche in streaming.
    // ════════════════════════════════════════════════════════════════════════
    const bool omni      = getenv("AIS_OMNI") != nullptr || getenv("AIS_OMNI_CODE") != nullptr;
    const bool omni_code = getenv("AIS_OMNI_CODE") != nullptr;
    const bool best_code = getenv("AIS_SNAPKV_CODE") != nullptr || omni_code;
    if (getenv("AIS_SNAPKV_BEST") || best_code || omni) {
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        const bool is_gemma = strstr(arch, "gemma") != nullptr;
        auto set_def = [](const char* k, const char* v){ if (!getenv(k)) setenv(k, v, 1); };
        // UNIFICATO SU FA: l'hook generico flash-ON funziona su OGNI modello, gemma incluso
        // (validato in streaming: parità col fork su compressione/correttezza/velocità). Niente
        // più fork per-modello nel routing → un solo percorso, più semplice e robusto agli update
        // di llama.cpp. Il fork gemma4-iswa.cpp resta dormiente (attivabile solo via AIS_SNAPKV).
        set_def("AIS_SNAPKV_FA", "1");                  // rotta C (FA) per TUTTI i modelli
        set_def("AIS_SNAPKV_AUTO",  "1");
        set_def("AIS_SNAPKV_DEDUP", "1");
        set_def("AIS_SNAPKV_W",     "24");
        set_def("AIS_SNAPKV_KEEP",  "0.4");
        // RUNNING EVICTION (AIS_SNAPKV_RUNEVICT) is intentionally NOT defaulted on. Measured
        // (gemma-E2B, bench_final --think + standalone smoke, see ais/bench/RESULTS_kv_eviction.txt):
        // when it sticks it helps (multi-turn 29.10s→27.93s, dense ptok 5917→3108), but on SWA models
        // its slot/position check can BAIL mid-prefill (only full-attn layers capture relevance) and
        // fall back to a full re-prefill — wasting the partial pass → SLOWER. Correctness is always
        // preserved (clean fallback), but the speed is not, so it stays opt-in.
        // CoT-cut sui modelli THINKING (gemma): comprime il ragionamento → +veloce, qualità piena.
        // NON tocca il CODICE: il taglio opera SOLO nella fase di pensiero (in_thought). Con thinking
        // OFF (AIS_NO_THINK) o nella fase di RISPOSTA, in_thought=false → la risposta (codice incluso)
        // non viene mai tagliata. Su altri modelli il marker di canale non si applica. AIS_COT_OFF=1 disattiva.
        if (omni && is_gemma && !getenv("AIS_COT_OFF")) set_def("AIS_COMPRESS_COT", "1");
        if (best_code) {                                // preset coding (OMNI_CODE o SNAPKV_CODE)
            set_def("AIS_SNAPKV_LEXGATE", "1");
            set_def("AIS_LEXGATE_CODE",   "1");         // syntax-aware → protegge il codice
            set_def("AIS_LEXGATE_WIN",    "4");
        }
        // ROUTER — KV PRECISION: default f16 (velocità vanilla sul caso comune piccolo/interattivo:
        // la q8_0 dequant è ~7% di tassa di prefill, e il suo unico vantaggio — ½ memoria — conta
        // solo a contesto LUNGO/memory-bound). Opt-in q8_0 per i deployment memory-bound:
        // AIS_SNAPKV_KVQ8=1 (o --cache-type-k/v q8_0). [Cambio di default: BEST prima forzava q8_0;
        // AIS_BEST_KV_F16 resta accettato come no-op per compatibilità.]
        if (getenv("AIS_SNAPKV_KVQ8") && type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
            type_k = type_v = GGML_TYPE_Q8_0;           // ½ memoria KV (paga ~7% prefill: solo se memory-bound)
        }
        const char* label = omni_code ? "OMNI+CODE" : (omni ? "OMNI" : (best_code ? "CODE" : "BEST"));
        fprintf(stderr, "AIS %s: arch=%s → rotta FA(C) + AUTO + DEDUP%s%s + W24 + KEEP0.4 + KV=%s (KVQ8=%s)\n",
                label, arch[0] ? arch : "?",
                (omni && is_gemma && getenv("AIS_COMPRESS_COT")) ? " + CoT-cut[thinking-only]" : "",
                best_code ? " + LEXGATE[code]" : "", ggml_type_name(type_k),
                getenv("AIS_SNAPKV_KVQ8") ? "on" : "off(f16)");
    }

    auto cparams            = llama_context_default_params();
    cparams.n_ctx           = cfg.ctx_limit;
    cparams.n_batch         = ais_n_batch();
    cparams.type_k          = type_k;
    cparams.type_v          = type_v;
    cparams.flash_attn_type = fa;
    // SnapKV PROBE (rotta A): serve kq_soft_max materializzato → flash OFF + cb_eval.
    if (getenv("AIS_SNAPKV_PROBE")) {
        g_snap.on = true;
        if (const char* w = getenv("AIS_SNAPKV_W")) g_snap.W = atoi(w);
        cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.cb_eval           = snap_eval_cb;
        cparams.cb_eval_user_data = nullptr;
        // ubatch 512: ad ogni chunk kq = [n_kv × 512 × n_head] (transitorio, memoria-safe
        // a N grande). L'ULTIMO ubatch ha n_kv massimo (KV piena) → il callback tiene solo
        // quello (la finestra-query globale è nelle sue ultime W righe).
        cparams.n_batch  = 512;
        cparams.n_ubatch = 512;
        fprintf(stderr, "SnapKV PROBE attivo (W=%d, flash OFF, ubatch=512)\n", g_snap.W);
    }
    // SnapKV GENERIC (model-agnostic): rilevanza dal tensore CORE `kq_soft_max` via
    // cb_eval (path non-flash, emesso da build_attn_mha per OGNI architettura) → NON
    // serve il fork del grafo. È la "rotta A" cablata anche nel SERVER (ais_snapkv_infer
    // legge g_snap.rel come per il fork). Costo: flash OFF + ubatch 512 (più lento del
    // fork Gemma-4, ma funziona ovunque). Mutuamente esclusivo con PROBE/FORK.
    const bool snap_generic = getenv("AIS_SNAPKV_GENERIC") != nullptr
                              && !getenv("AIS_SNAPKV_PROBE") && !getenv("AIS_SNAPKV");
    if (snap_generic) {
        g_snap.on = true;
        g_snap.profile = getenv("AIS_SNAP_PROFILE") != nullptr;
        if (const char* w = getenv("AIS_SNAPKV_W")) g_snap.W = atoi(w);
        cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.cb_eval           = snap_eval_cb;
        cparams.cb_eval_user_data = nullptr;
        cparams.n_batch  = 512;
        cparams.n_ubatch = 512;
        fprintf(stderr, "SnapKV GENERIC (kq_soft_max probe, model-agnostic, flash OFF, ubatch=512, W=%d)\n", g_snap.W);
    }
    // SnapKV FA (rotta C): hook GENERICO in build_attn_mha (src/llama-graph.cpp) → la
    // rilevanza [n_kv] è calcolata DENTRO il grafo, per-layer, con FLASH ON (lo slice
    // [W×n_kv] è a parte; l'attenzione principale resta flash). Vale per QUALSIASI modello
    // che passa per build_attn (Qwen3, Llama, ...), senza forkare il singolo .cpp. Il
    // grafo emette "ais_snap_rel" se l'env AIS_SNAPKV_FA è settato (gating lato core).
    const bool snap_fa = getenv("AIS_SNAPKV_FA") != nullptr
                         && !getenv("AIS_SNAPKV_PROBE") && !getenv("AIS_SNAPKV") && !snap_generic;
    if (snap_fa) {
        g_snap.fa      = true;
        g_snap.profile = getenv("AIS_SNAP_PROFILE") != nullptr;
        if (const char* w = getenv("AIS_SNAPKV_W")) g_snap.W = atoi(w);
        cparams.cb_eval           = snap_eval_cb;          // cattura "ais_snap_rel" per-layer
        cparams.cb_eval_user_data = nullptr;
        // flash resta com'è (default ON/auto) → onora --cache-type-k/v e --flash-attn.
        fprintf(stderr, "SnapKV FA (hook generico flash-ON, model-agnostic, KV=%s, W=%d)\n",
                ggml_type_name(type_k), g_snap.W);
    }
    fprintf(stderr, "KV cache: k=%s v=%s | flash_attn=%s\n",
            ggml_type_name(type_k), ggml_type_name(type_v),
            llama_flash_attn_type_name(fa));
    // Milestone B: embeddings(pooling NONE) makes the gemma4 graph emit per-token
    // "target logits" via the gather-dot (in embd[0]).
    //   AIS_VALIDATE_GATHER → keep full lm_head too, to compare (no speedup).
    //   AIS_GATHER_SCORE    → skip full lm_head during scoring (the speedup).
    if (getenv("AIS_VALIDATE_GATHER") || getenv("AIS_GATHER_SCORE")) {
        // TODO#2: with AIS_SURPRISE_OUT the dedicated [1×N] tensor carries the target logits,
        // so we DON'T enable embeddings/pooling at all — no [n_embd_out×N] readback nor buffer.
        // (validate still needs embeddings on: it reads the gather-dot via llama_get_embeddings.)
        const bool surprise_out = getenv("AIS_SURPRISE_OUT") && getenv("AIS_GATHER_SCORE") && !getenv("AIS_VALIDATE_GATHER");
        if (!surprise_out) {
            cparams.embeddings   = true;
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        }
        fprintf(stderr, "gather-dot mode (embeddings=%s, %s)\n", surprise_out ? "off/dedicated" : "on,pooling=none",
                getenv("AIS_GATHER_SCORE") ? "skip-lm_head scoring" : "validate");
    }
    // SnapKV FORK (Milestone C rotta B): embeddings on → il grafo gemma4 emette la
    // RILEVANZA per posizione (in embd[0]). Ubatch SINGOLO così Kcur = tutte le N chiavi.
    // Flash resta ON (lo slice [W×N] è calcolato a parte, economico).
    if (getenv("AIS_SNAPKV") && !getenv("AIS_SNAPKV_PROBE")) {
        cparams.cb_eval           = snap_eval_cb;          // cattura "ais_snap_rel"
        cparams.cb_eval_user_data = nullptr;
        g_snap.fork = true;
        if (const char* w = getenv("AIS_SNAPKV_W")) g_snap.W = atoi(w);
        // KV onora --cache-type-k/v (q8_0 ok: la K cache viene dequantizzata a f16 nel grafo
        // per il mul_mat dello slice). lm_head ATTIVO (no embeddings) → logits per generare.
        fprintf(stderr, "SnapKV FORK cached-K (chunked, flash ON, KV=%s)\n", ggml_type_name(type_k));
    }
    CtxPtr ctx_guard(llama_init_from_model(model, cparams));
    llama_context* ctx = ctx_guard.get();
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    auto sparams = llama_sampler_chain_default_params();
    SamplerPtr smpl_guard(llama_sampler_chain_init(sparams));
    llama_sampler* smpl = smpl_guard.get();
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    struct BatchGuard {
        llama_batch b;
        explicit BatchGuard(llama_batch batch) : b(batch) {}
        ~BatchGuard() { llama_batch_free(b); }
    } batch_guard(llama_batch_init(cparams.n_batch, 0, 1));
    llama_batch& batch = batch_guard.b;

    // ==========================================
    // SERVER MODE
    // ==========================================
    if (server_port > 0) {
        auto tmpls = common_chat_templates_init(model, "", "", "");
        std::mutex mtx;
        DeltaState ds;

        auto make_chat = [&](const ordered_json& body) -> common_chat_params {
            auto msgs = common_chat_msgs_parse_oaicompat(body["messages"]);
            common_chat_templates_inputs inputs;
            inputs.messages              = msgs;
            inputs.add_generation_prompt = true;
            inputs.use_jinja             = true;
            // Costruisce il parser per ESTRARRE il "thought" di Gemma-4 (canale separato
            // → reasoning_content). Con NONE il thought finirebbe nel content coi marker.
            inputs.reasoning_format      = COMMON_REASONING_FORMAT_AUTO;
            // AIS_NO_THINK: non genera affatto il canale "thought" (che per Cline stripiamo
            // comunque) → generazione molto più rapida, stesso codice consegnato.
            static const bool no_think = getenv("AIS_NO_THINK") != nullptr;
            if (no_think) inputs.enable_thinking = false;
            return common_chat_templates_apply(tmpls.get(), inputs);
        };
        // Parser dell'output: estrae SOLO il canale finale (scarta il "thought" di
        // Gemma-4) come fa llama-server. Carica l'arena PEG serializzata dal template.
        auto make_parser = [](const common_chat_params& cp) -> common_chat_parser_params {
            common_chat_parser_params pp;
            pp.format            = cp.format;
            pp.generation_prompt = cp.generation_prompt;
            pp.reasoning_format  = COMMON_REASONING_FORMAT_AUTO;  // thinking → reasoning_content
            if (!cp.parser.empty()) pp.parser.load(cp.parser);
            return pp;
        };
        auto final_content = [](const std::string& raw, const common_chat_parser_params& pp) -> std::string {
            common_chat_msg m = common_chat_parse(raw, /*is_partial=*/false, pp);
            // se il parse non estrae nulla (formato sconosciuto) torna il testo grezzo
            if (m.content.empty() && m.reasoning_content.empty() && !raw.empty()) return raw;
            return m.content;
        };

        httplib::Server svr;
        svr.set_read_timeout(600);
        svr.set_write_timeout(600);

        // ── POST /v1/chat/completions ──────────────────────────────────
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body      = ordered_json::parse(req.body);
                bool streaming = body.value("stream", false);
                int  n_tokens  = body.value("max_tokens", max_tokens);
                common_chat_params cp = make_chat(body);
                std::string prompt = cp.prompt;

                fprintf(stderr, "\n--- request (stream=%s) ---\n",
                        streaming ? "true" : "false");

                if (!streaming) {
                    // ── BATCH ──────────────────────────────────────────
                    std::lock_guard<std::mutex> lock(mtx);
                    int pt = 0, ct = 0;
                    std::string response;
                    if (g_snap.fork || g_snap.on || g_snap.fa) {   // fork Gemma-4 / probe flash-off / hook generico flash-on
                        static const float keep = getenv("AIS_SNAPKV_KEEP") ? atof(getenv("AIS_SNAPKV_KEEP")) : 0.5f;
                        response = ais_snapkv_infer(ctx, vocab, smpl, batch, prompt, n_tokens, keep, &ds, &pt, &ct);
                    } else {
                        response = ais_infer(
                            ctx, vocab, n_vocab, smpl, batch,
                            cfg, adapt_mode, adapt_param, prompt, n_tokens,
                            &pt, &ct, &ds);
                    }
                    response = final_content(response, make_parser(cp));  // strip thought/marker
                    fprintf(stderr, "--- done (pt=%d ct=%d) ---\n", pt, ct);

                    ordered_json resp = {
                        {"id",      "chatcmpl-" + random_id()},
                        {"object",  "chat.completion"},
                        {"created", (int64_t)std::time(nullptr)},
                        {"model",   "ais-prob"},
                        {"choices", ordered_json::array({{
                            {"index",         0},
                            {"message",       {{"role","assistant"},{"content",response}}},
                            {"finish_reason", "stop"}
                        }})},
                        {"usage", {{"prompt_tokens",pt},{"completion_tokens",ct},{"total_tokens",pt+ct}}}
                    };
                    res.set_content(resp.dump(), "application/json");
                    return;
                }

                // ── STREAMING SSE ──────────────────────────────────────
                // Acquisiamo il lock e lo manteniamo per tutta la streaming session
                // tramite shared_ptr catturato nella lambda del content_provider.
                auto lock_ptr = std::make_shared<std::unique_lock<std::mutex>>(mtx);

                // Pass 1 + selezione (setup KV cache). Se SnapKV è attivo (fork/hook/generic),
                // usa la STESSA compressione del path batch via ais_snapkv_infer in modalità
                // setup-only → KV compressa + delta multi-turno anche in STREAMING (prima girava
                // solo non-streaming → Cline non ne beneficiava). n_past = posizione originale N.
                int n_past, snap_kept = 0;
                const bool snap_active = (g_snap.fork || g_snap.on || g_snap.fa);
                if (snap_active) {
                    static const float snap_keep = getenv("AIS_SNAPKV_KEEP") ? atof(getenv("AIS_SNAPKV_KEEP")) : 0.5f;
                    llama_sampler_reset(smpl);
                    ais_snapkv_infer(ctx, vocab, smpl, batch, prompt, n_tokens, snap_keep,
                                     &ds, &snap_kept, nullptr, /*setup_only=*/true, &n_past);
                } else {
                    n_past = ais_setup(ctx, vocab, n_vocab, smpl, batch,
                                       cfg, adapt_mode, adapt_param, prompt, &ds);
                }
                if (n_past < 0) {
                    lock_ptr->unlock();
                    res.status = 500;
                    res.set_content(R"({"error":"setup failed"})", "application/json");
                    return;
                }

                // Stato condiviso con il content_provider
                struct StreamState {
                    int  n_past;
                    int  prompt_tokens;   // n_past dopo ais_setup (token compressi)
                    int  token_idx   = 0;
                    int  max_tokens;
                    bool role_sent   = false;
                    bool finished    = false;
                    std::string id;
                    int64_t     created;
                    std::shared_ptr<std::unique_lock<std::mutex>> lock;
                    std::string raw;   // testo grezzo accumulato (con eventuale thought)
                    std::string sent;  // content (canale finale) già inviato
                    common_chat_parser_params pp;  // parser del canale finale
                    bool in_thought = true;        // compressione CoT: fase di pensiero (init sotto)
                    int  thought_toks = 0, low_run = 0;
                    bool snap = false;             // SnapKV attivo → ri-armare la cattura a fine risposta
                };
                auto ss           = std::make_shared<StreamState>();
                ss->n_past        = n_past;
                ss->prompt_tokens = snap_active ? snap_kept : n_past;   // SnapKV: riporta i token COMPRESSI
                ss->snap          = snap_active;
                // CoT-cut SOLO se c'è davvero una fase di pensiero: con AIS_NO_THINK (o modelli non
                // thinking) parte già nella RISPOSTA → in_thought=false → la risposta (anche CODICE)
                // non viene mai tagliata. Con thinking ON il marker di canale chiude il pensiero.
                // AUTO-CODING: se il prompt è una richiesta di CODICE → niente CoT-cut (il modello
                // ragiona sul codice senza tagli). Per-richiesta, nessun flag al server start.
                const bool coding_req = prompt_is_coding(prompt);
                ss->in_thought    = (getenv("AIS_NO_THINK") == nullptr) && !coding_req;
                if (coding_req && getenv("AIS_COMPRESS_COT"))
                    fprintf(stderr, "CoT-cut OFF: coding rilevato nel prompt (no taglio)\n");
                ss->max_tokens    = n_tokens;
                ss->id        = "chatcmpl-" + random_id();
                ss->created   = (int64_t)std::time(nullptr);
                ss->lock      = std::move(lock_ptr);
                ss->pp        = make_parser(cp);

                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                // content_provider: chiamato ripetutamente da httplib,
                // genera un token per chiamata e lo invia come SSE chunk.
                res.set_content_provider(
                    "text/event-stream; charset=utf-8",
                    [ss, ctx, vocab, smpl, &batch, n_vocab, &ds]
                    (size_t /*offset*/, httplib::DataSink& sink) -> bool {
                      // try/catch around the whole provider: common_chat_parse() runs per token
                      // and can throw — without this the request mutex (held via ss->lock) would
                      // never be released and EVERY later request would deadlock.
                      try {
                        auto send = [&](const std::string& s) -> bool {
                            return sink.write(s.data(), s.size());
                        };

                        // Prima chiamata: invia il chunk con role=assistant
                        if (!ss->role_sent) {
                            ss->role_sent = true;
                            ordered_json first = {
                                {"id",      ss->id},
                                {"object",  "chat.completion.chunk"},
                                {"created", ss->created},
                                {"model",   "ais-prob"},
                                {"choices", ordered_json::array({{
                                    {"index",         0},
                                    {"delta",         {{"role","assistant"},{"content",""}}},
                                    {"finish_reason", nullptr}
                                }})}
                            };
                            send("data: " + first.dump() + "\n\n");
                            return true;
                        }

                        // Fine: invia chunk usage + [DONE], rilascia lock
                        if (ss->finished || ss->token_idx >= ss->max_tokens) {
                            int pt = ss->prompt_tokens;
                            int ct = ss->token_idx;
                            // flush finale: parse non-partial → ultimo delta del canale finale
                            {
                                common_chat_msg mf = common_chat_parse(ss->raw, /*is_partial=*/false, ss->pp);
                                if (mf.content.size() > ss->sent.size() &&
                                    mf.content.compare(0, ss->sent.size(), ss->sent) == 0) {
                                    ordered_json fchunk = {
                                        {"id", ss->id}, {"object","chat.completion.chunk"},
                                        {"created", ss->created}, {"model","ais-prob"},
                                        {"choices", ordered_json::array({{
                                            {"index",0},
                                            {"delta",{{"content", mf.content.substr(ss->sent.size())}}},
                                            {"finish_reason", nullptr}
                                        }})}
                                    };
                                    send("data: " + fchunk.dump() + "\n\n");
                                    ss->sent = mf.content;
                                }
                            }
                            ordered_json usage_chunk = {
                                {"id",      ss->id},
                                {"object",  "chat.completion.chunk"},
                                {"created", ss->created},
                                {"model",   "ais-prob"},
                                {"choices", ordered_json::array()},
                                {"usage",   {{"prompt_tokens",pt},{"completion_tokens",ct},{"total_tokens",pt+ct}}}
                            };
                            send("data: " + usage_chunk.dump() + "\n\n");
                            send("data: [DONE]\n\n");
                            if (ss->snap) {   // ri-arma la cattura SnapKV per il prossimo prefill
                                g_snap.capturing = true;
                                llama_set_eval_callback(ctx, snap_eval_cb, nullptr);
                            }
                            ss->lock->unlock();
                            sink.done();
                            return false;
                        }

                        // Genera un token
                        llama_token id = llama_sampler_sample(smpl, ctx, -1);
                        llama_sampler_accept(smpl, id);

                        // Compressione CoT online (stessa logica di ais_infer): se il
                        // pensiero entra in un tratto a bassa informazione, chiudi il
                        // thought (inietta <channel|>) e passa alla risposta.
                        static const bool   cot_comp = getenv("AIS_COMPRESS_COT") != nullptr;
                        static const double TLOW = [](){ const char* v=getenv("AIS_COT_TLOW"); return v?atof(v):0.05; }();
                        static const int    WIN  = [](){ const char* v=getenv("AIS_COT_WIN");  return v?atoi(v):24;   }();
                        static const int    MINT = [](){ const char* v=getenv("AIS_COT_MIN");  return v?atoi(v):48;   }();
                        static const std::vector<llama_token> end_tag = common_tokenize(vocab, "<channel|>", false, true);
                        if (cot_comp && ss->in_thought) {
                            const float* lg = llama_get_logits_ith(ctx, -1);
                            double surp = cot_surprise(lg, id, n_vocab);   // G1: top-K logsumexp
                            ss->thought_toks++;
                            ss->low_run = (surp < TLOW) ? ss->low_run + 1 : 0;
                            const int think_cap = cot_think_cap(ss->max_tokens);
                            if (ss->thought_toks >= MINT && !end_tag.empty() &&
                                (ss->low_run >= WIN || (think_cap > 0 && ss->thought_toks >= think_cap))) {
                                for (llama_token et : end_tag) {
                                    char b[128]; int bn = llama_token_to_piece(vocab, et, b, sizeof(b), 0, true);
                                    if (bn > 0) ss->raw.append(b, bn);
                                    batch.n_tokens = 0; batch_add(batch, et, ss->n_past, true); ss->n_past++;
                                    if (llama_decode(ctx, batch) != 0) { ss->finished = true; break; }
                                }
                                batch.n_tokens = 0;
                                ss->in_thought = false;
                                fprintf(stderr, " CoT: taglio a %d tok di pensiero (low_run=%d)\n",
                                        ss->thought_toks, ss->low_run);
                                ss->token_idx++;
                                return true;  // scarta 'id', prossimo giro genera la risposta
                            }
                        }

                        if (ais_is_eog(vocab, id)) {
                            ss->finished = true;
                            return true;  // manda [DONE] al prossimo giro
                        }

                        char buf[128];
                        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
                        if (n > 0) ss->raw.append(buf, n);
                        if (ss->in_thought && tail_contains(ss->raw, n, "<channel|>", 10)) ss->in_thought = false;
                        // parse incrementale: invia SOLO il delta del canale finale.
                        // Durante il "thought" di Gemma-4 il content resta vuoto → niente leak.
                        common_chat_msg m = common_chat_parse(ss->raw, /*is_partial=*/true, ss->pp);
                        if (m.content.size() > ss->sent.size() &&
                            m.content.compare(0, ss->sent.size(), ss->sent) == 0) {
                            std::string piece = m.content.substr(ss->sent.size());
                            ss->sent = m.content;
                            ordered_json chunk = {
                                {"id",      ss->id},
                                {"object",  "chat.completion.chunk"},
                                {"created", ss->created},
                                {"model",   "ais-prob"},
                                {"choices", ordered_json::array({{
                                    {"index",         0},
                                    {"delta",         {{"content", piece}}},
                                    {"finish_reason", nullptr}
                                }})}
                            };
                            if (!send("data: " + chunk.dump() + "\n\n")) {
                                ss->finished = true;
                                if (ss->snap) {   // ri-arma la cattura anche se il client si disconnette
                                    g_snap.capturing = true;
                                    llama_set_eval_callback(ctx, snap_eval_cb, nullptr);
                                }
                                ss->lock->unlock();
                                sink.done();
                                return false;  // client disconnected
                            }
                        }

                        // Prepara prossimo token
                        batch.n_tokens = 0;
                        batch_add(batch, id, ss->n_past, true);
                        ss->n_past++;
                        if (llama_decode(ctx, batch) != 0) {
                            ss->finished = true;
                        }
                        batch.n_tokens = 0;
                        ss->token_idx++;
                        return true;
                      } catch (const std::exception& e) {
                        fprintf(stderr, "stream provider error: %s\n", e.what());
                        if (ss->snap) {   // ri-arma la cattura SnapKV anche su errore
                            g_snap.capturing = true;
                            llama_set_eval_callback(ctx, snap_eval_cb, nullptr);
                        }
                        if (ss->lock) ss->lock->unlock();   // mai lasciare il mutex bloccato
                        sink.done();
                        return false;
                      }
                    }
                );

            } catch (const std::exception& e) {
                ordered_json err = {{"error",{{"message",e.what()},{"type","invalid_request_error"}}}};
                res.status = 400;
                res.set_content(err.dump(), "application/json");
            }
        });

        // ── POST /v1/completions (raw text) ────────────────────────────
        svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mtx);
            try {
                auto body     = ordered_json::parse(req.body);
                std::string prompt = body.value("prompt", "");
                int  n_tokens = body.value("max_tokens", max_tokens);
                std::string response = ais_infer(
                    ctx, vocab, n_vocab, smpl, batch,
                    cfg, adapt_mode, adapt_param, prompt, n_tokens);
                ordered_json resp = {
                    {"id",      "cmpl-" + random_id()},
                    {"object",  "text_completion"},
                    {"created", (int64_t)std::time(nullptr)},
                    {"model",   "ais-prob"},
                    {"choices", ordered_json::array({{
                        {"text", response}, {"index", 0}, {"finish_reason","stop"}
                    }})}
                };
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(ordered_json{{"error",e.what()}}.dump(), "application/json");
            }
        });

        // ── CORS headers su tutte le risposte ─────────────────────────
        svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin",  "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        });
        svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin",  "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.status = 204;
        });

        // ── GET / — Chat UI ────────────────────────────────────────────
        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            static const std::string html = R"HTML(<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIS Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0d0d0d; color: #e8e8e8; height: 100dvh; display: flex; flex-direction: column; }
  #header { padding: 12px 20px; background: #161616; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; gap: 10px; }
  #header h1 { font-size: 1rem; font-weight: 600; color: #13eaad; }
  #status { font-size: 0.75rem; color: #666; margin-left: auto; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .user  { background: #1a3a2a; color: #e8e8e8; align-self: flex-end; border-bottom-right-radius: 4px; }
  .asst  { background: #1a1a1a; color: #e8e8e8; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid #2a2a2a; }
  .asst.thinking::after { content: "▋"; animation: blink 0.7s infinite; color: #13eaad; }
  @keyframes blink { 50% { opacity: 0; } }
  #footer { padding: 16px; background: #161616; border-top: 1px solid #2a2a2a; }
  #form { display: flex; gap: 10px; }
  #input { flex: 1; background: #222; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; color: #e8e8e8; font-size: 0.95rem; resize: none; min-height: 44px; max-height: 160px; outline: none; }
  #input:focus { border-color: #13eaad55; }
  #send { background: #13eaad; color: #000; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 600; white-space: nowrap; }
  #send:disabled { background: #333; color: #666; cursor: not-allowed; }
  #clear { background: transparent; border: 1px solid #333; color: #888; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
  #clear:hover { border-color: #555; color: #aaa; }
  #info { font-size: 0.72rem; color: #444; margin-top: 6px; text-align: right; }
</style>
</head>
<body>
<div id="header">
  <h1>AIS Chat</h1>
  <span id="status">connessione...</span>
</div>
<div id="messages"></div>
<div id="footer">
  <div id="form">
    <textarea id="input" placeholder="Scrivi un messaggio... (Invio = invia, Shift+Invio = nuova riga)" rows="1"></textarea>
    <button id="clear" title="Pulisci chat">Pulisci</button>
    <button id="send">Invia</button>
  </div>
  <div id="info" id="info"></div>
</div>
<script>
const API = window.location.origin + "/v1/chat/completions";
let history = [];
let busy = false;

async function checkStatus() {
  try {
    const r = await fetch(window.location.origin + "/health");
    const d = await r.json();
    document.getElementById("status").textContent = d.status === "ok" ? "online" : "" + d.status;
    document.getElementById("status").style.color = "#13eaad";
  } catch { document.getElementById("status").textContent = "offline"; }
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "asst");
  el.textContent = text;
  document.getElementById("messages").appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
  return el;
}

async function send() {
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true;
  document.getElementById("send").disabled = true;
  input.value = "";
  input.style.height = "auto";

  addMsg("user", text);
  history.push({ role: "user", content: text });

  const aEl = addMsg("assistant", "");
  aEl.classList.add("thinking");
  let reply = "";
  const t0 = Date.now();

  try {
    const resp = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, stream: true, max_tokens: 4096 })
    });

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") continue;
        try {
          const j = JSON.parse(data);
          const delta = j.choices?.[0]?.delta?.content;
          if (delta) { reply += delta; aEl.textContent = reply; aEl.scrollIntoView({ behavior: "smooth" }); }
          const usage = j.usage;
          if (usage) {
            const ms = Date.now() - t0;
            document.getElementById("info").textContent =
              `prompt: ${usage.prompt_tokens} tok | risposta: ${usage.completion_tokens} tok | ${ms}ms`;
          }
        } catch {}
      }
    }
  } catch(e) { reply = "Errore: " + e.message; aEl.textContent = reply; }

  aEl.classList.remove("thinking");
  if (!reply) aEl.textContent = "(nessuna risposta)";
  history.push({ role: "assistant", content: reply });
  busy = false;
  document.getElementById("send").disabled = false;
  input.focus();
}

document.getElementById("send").onclick = send;
document.getElementById("clear").onclick = () => {
  history = [];
  document.getElementById("messages").innerHTML = "";
  document.getElementById("info").textContent = "";
};
document.getElementById("input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  setTimeout(() => {
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, 0);
});

checkStatus();
setInterval(checkStatus, 15000);
</script>
</body>
</html>)HTML";
            res.set_content(html, "text/html; charset=utf-8");
        });

        // ── GET /v1/models ─────────────────────────────────────────────
        svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
            ordered_json resp = {
                {"object","list"},
                {"data", ordered_json::array({{
                    {"id","ais-prob"},{"object","model"},{"owned_by","local"}
                }})}
            };
            res.set_content(resp.dump(), "application/json");
        });

        // ── GET /health ────────────────────────────────────────────────
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status":"ok"})", "application/json");
        });

        fprintf(stderr,
            "\nAIS Server avviato\n"
            "   Modello  : %s\n"
            "   Modalità : %s (param=%.2f)\n"
            "   URL      : http://%s:%d\n"
            "   Streaming: SSE token-by-token\n\n",
            model_path, adapt_mode.c_str(), adapt_param,
            server_host.c_str(), server_port);

        svr.listen(server_host.c_str(), server_port);

        return 0;   // model/ctx/sampler/batch/backend freed by RAII guards
    }

    // ==========================================
    // CLI MODE
    // ==========================================
    fprintf(stderr, "AIS_PROB CLI (mode=%s param=%.2f)\n",
            adapt_mode.c_str(), adapt_mode == "fixed" ? cfg.surprise_threshold : adapt_param);

    std::string input_buffer, line;
    while (std::getline(std::cin, line))
        input_buffer += line + "\n";

    // ── SnapKV PROBE: prefill + dump rilevanza per token (no generazione) ──
    if (g_snap.on && !input_buffer.empty()) {
        std::vector<llama_token> toks = common_tokenize(vocab, input_buffer, true, false);
        int N = (int)toks.size();
        fprintf(stderr, "SnapKV PROBE: %d token (W=%d) → prefill (chunk 512)\n", N, g_snap.W);
        auto t_pf = std::chrono::steady_clock::now();
        const int CH = 512;
        for (int s = 0; s < N; s += CH) {
            batch.n_tokens = 0;
            int e = std::min(N, s + CH);
            for (int i = s; i < e; i++) batch_add(batch, toks[i], i, i == N - 1);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode fallito\n"); break; }
        }
        batch.n_tokens = 0;
        fprintf(stderr, " prefill(flash off): %.1fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf).count());
        if ((int)g_snap.rel.size() >= N && N > 0) {   // n_kv può essere paddato > N (posizioni reali = [0,N))
            std::vector<int> idx(N);
            for (int i = 0; i < N; i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [](int a, int b){ return g_snap.rel[a] > g_snap.rel[b]; });
            auto piece = [&](int i){ char b[64]; int n = llama_token_to_piece(vocab, toks[i], b, sizeof(b), 0, true); return std::string(b, n > 0 ? n : 0); };
            fprintf(stderr, "layer pieni catturati: %d\n", g_snap.layers);
            printf("\n=== TOP 20 RILEVANTI (attesi: query + token a cui guarda) ===\n");
            for (int r = 0; r < std::min(20, N); r++) { int i = idx[r];     printf("  #%2d pos=%4d rel=%.3f  %s\n", r + 1, i, g_snap.rel[i], piece(i).c_str()); }
            printf("\n=== BOTTOM 20 (attesi: filler irrilevante) ===\n");
            for (int r = 0; r < std::min(20, N); r++) { int i = idx[N-1-r]; printf("  pos=%4d rel=%.3f  %s\n", i, g_snap.rel[i], piece(i).c_str()); }

            // ── EVICTION query-aware + generazione (test end-to-end) ──
            // AIS_SNAPKV_KEEP=<ratio>: tieni anchor + recent (la query) + top-rilevanti,
            // evicta il resto dalla KV (seq_rm) e genera → la risposta deve reggere.
            if (const char* ke = getenv("AIS_SNAPKV_KEEP")) {
                const float ratio = atof(ke);
                const int anchor = 4, recent = 32;
                const int target = std::max(anchor + recent, (int)(N * ratio));
                std::vector<bool> keep(N, false);
                for (int i = 0; i < anchor && i < N; i++) keep[i] = true;
                for (int i = std::max(0, N - recent); i < N; i++) keep[i] = true;
                int kept = (int)std::count(keep.begin(), keep.end(), true);
                for (int r = 0; r < N && kept < target; r++) { int i = idx[r]; if (!keep[i]) { keep[i] = true; kept++; } }
                llama_memory_t mem = llama_get_memory(ctx);
                auto t_ev = std::chrono::steady_clock::now();
                const int scans = snap_fast_evict(mem, keep);     // single-pass O(N) (vs run-wise O(R·N))
                const double ev_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t_ev).count();
                fprintf(stderr, " SnapKV eviction: tenuti %d/%d → %.1f%% compresso | %.2fms / %d scan (%s, %ld cell-scan)\n",
                        kept, N, 100.0f * (N - kept) / N, ev_ms, scans,
                        scans <= 1 ? "mask O(N)" : "run-wise O(R·N)", (long)scans * N);
                std::string ans; int n_past = N;
                for (int t = 0; t < 48; t++) {
                    llama_token id = llama_sampler_sample(smpl, ctx, -1);
                    llama_sampler_accept(smpl, id);
                    if (ais_is_eog(vocab, id)) break;
                    char b[128]; int n = llama_token_to_piece(vocab, id, b, sizeof(b), 0, true);
                    if (n > 0) ans.append(b, n);
                    batch.n_tokens = 0; batch_add(batch, id, n_past, true); n_past++;
                    if (llama_decode(ctx, batch) != 0) break; batch.n_tokens = 0;
                }
                printf("\n=== RISPOSTA dopo eviction query-aware (tenuto %.0f%%) ===\n%s\n", ratio * 100, ans.c_str());
            }
        } else {
            fprintf(stderr, "rel.size=%zu != N=%d (prefill in chunk? usa prompt < n_batch)\n", g_snap.rel.size(), N);
        }
    } else if (getenv("AIS_SNAPKV") && !input_buffer.empty()) {
        // ── SnapKV FORK cached-K (rotta B): prefill CHUNKED (flash on, no OOM); la rilevanza
        // arriva da "ais_snap_rel" via cb_eval (g_snap.rel, tenuta col n_kv massimo). ──
        std::vector<llama_token> toks = common_tokenize(vocab, input_buffer, true, false);
        int N = (int)toks.size();
        fprintf(stderr, "SnapKV FORK cached-K: %d token (W=%d) → prefill chunked (512)\n", N, g_snap.W);
        auto t0 = std::chrono::steady_clock::now();
        const int CH = 512;
        for (int s = 0; s < N; s += CH) {
            batch.n_tokens = 0;
            int e = std::min(N, s + CH);
            for (int i = s; i < e; i++) batch_add(batch, toks[i], i, true);  // multi-output → salta lm_head
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode fallito\n"); break; }
        }
        batch.n_tokens = 0;
        fprintf(stderr, " prefill(flash ON, chunked): %.1fs | rel n_kv=%zu\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_snap.rel.size());
        if ((int)g_snap.rel.size() >= N && N > 0) {
            std::vector<int> idx(N); for (int i = 0; i < N; i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [](int a, int b){ return g_snap.rel[a] > g_snap.rel[b]; });
            auto piece = [&](int i){ char b[64]; int n = llama_token_to_piece(vocab, toks[i], b, sizeof(b), 0, true); return std::string(b, n > 0 ? n : 0); };
            printf("\n=== FORK cached-K: TOP 15 RILEVANTI ===\n");
            for (int r = 0; r < std::min(15, N); r++) { int i = idx[r];     printf("  #%2d pos=%4d rel=%.3f  %s\n", r + 1, i, g_snap.rel[i], piece(i).c_str()); }
            printf("\n=== FORK cached-K: BOTTOM 15 ===\n");
            for (int r = 0; r < std::min(15, N); r++) { int i = idx[N-1-r]; printf("  pos=%4d rel=%.3f  %s\n", i, g_snap.rel[i], piece(i).c_str()); }

            // ── eviction query-aware + generazione (validazione qualità a scala) ──
            if (const char* ke = getenv("AIS_SNAPKV_KEEP")) {
                const float ratio = atof(ke);
                const int anchor = 8, recent = 64;
                const int target = std::max(anchor + recent, (int)(N * ratio));
                std::vector<bool> keep(N, false);
                for (int i = 0; i < anchor && i < N; i++) keep[i] = true;
                for (int i = std::max(0, N - recent); i < N; i++) keep[i] = true;
                int kept = (int)std::count(keep.begin(), keep.end(), true);
                for (int r = 0; r < N && kept < target; r++) { int i = idx[r]; if (!keep[i]) { keep[i] = true; kept++; } }
                llama_memory_t mem = llama_get_memory(ctx);
                int run = -1;
                for (int i = 0; i <= N; i++) {
                    bool ev = (i < N) && !keep[i];
                    if (ev && run < 0) run = i;
                    else if (!ev && run >= 0) { llama_memory_seq_rm(mem, 0, run, i); run = -1; }
                }
                fprintf(stderr, " eviction: tenuti %d/%d → %.1f%% compresso\n", kept, N, 100.0f * (N - kept) / N);
                std::string ans; int n_past = N;
                for (int t = 0; t < 64; t++) {
                    llama_token id = llama_sampler_sample(smpl, ctx, -1);
                    llama_sampler_accept(smpl, id);
                    if (ais_is_eog(vocab, id)) break;
                    char b[128]; int n = llama_token_to_piece(vocab, id, b, sizeof(b), 0, true);
                    if (n > 0) ans.append(b, n);
                    batch.n_tokens = 0; batch_add(batch, id, n_past, true); n_past++;
                    if (llama_decode(ctx, batch) != 0) break; batch.n_tokens = 0;
                }
                printf("\n=== RISPOSTA dopo eviction SnapKV (tenuto %.0f%%) ===\n%s\n", ratio * 100, ans.c_str());
            }
        } else {
            fprintf(stderr, "rel.size=%zu < N=%d (cattura ais_snap_rel fallita?)\n", g_snap.rel.size(), N);
        }
    } else if (!input_buffer.empty()) {
        std::string response = ais_infer(
            ctx, vocab, n_vocab, smpl, batch,
            cfg, adapt_mode, adapt_param, input_buffer, max_tokens);
        std::cout << response << std::endl;
    }

    return 0;   // model/ctx/sampler/batch/backend freed by RAII guards
}
