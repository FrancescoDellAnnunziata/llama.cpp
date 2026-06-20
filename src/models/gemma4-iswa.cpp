#include "models.h"
#include "llama-kv-cache.h"       // SnapKV cached-K: get_k(ctx, il) sul contesto base
#include "llama-kv-cache-iswa.h"  // SnapKV cached-K: mctx->get_base() (attenzione piena)

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * ggml_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

llm_build_gemma4_iswa::llm_build_gemma4_iswa(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    // ── AIS SnapKV (Milestone C): rilevanza = attenzione delle ultime W query a
    // tutte le chiavi, sommata su teste e layer pieni. Richiede ubatch SINGOLO
    // (Kcur = tutte le N chiavi) + flash ON (l'attenzione principale resta economica;
    // qui calcoliamo solo lo slice [W×N], ~MB). Output via t_embd (come il gather-dot).
    // cb_eval != nullptr: l'host azzera il callback nei chunk non-finali del prefill → la
    // rilevanza si calcola solo sull'ULTIMO chunk (KV piena); gli altri erano scartati.
    const bool do_snap = getenv("AIS_SNAPKV") != nullptr && n_tokens > 1 && cparams.cb_eval != nullptr;
    const int  snap_W  = getenv("AIS_SNAPKV_W") ? atoi(getenv("AIS_SNAPKV_W")) : 32;
    ggml_tensor * snap_rel = nullptr;  // [n_kv] accumulato sui layer pieni

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);

            // SnapKV cached-K (Milestone C-2): DOPO build_attn (cpy_k ha scritto il chunk in
            // cache) → prendi la K in CACHE (TUTTE le N posizioni) e calcola l'attenzione
            // delle ultime W query a tutte le chiavi. Funziona con prefill CHUNKED (niente
            // ubatch singolo, niente OOM). Solo layer pieni (!is_swa) → get_base().
            if (do_snap && !hparams.is_swa(il)) {
                const int W = snap_W < (int) n_tokens ? snap_W : (int) n_tokens;
                ggml_tensor * kc = inp_attn->mctx->get_base()->get_k(ctx0, il);   // [n_eh, n_head_kv, n_kv, ns]
                if (kc->type != GGML_TYPE_F16 && kc->type != GGML_TYPE_F32)
                    kc = ggml_cast(ctx0, kc, GGML_TYPE_F16);   // dequant K cache (es. q8_0→f16) per il mul_mat
                ggml_tensor * kp = ggml_cont(ctx0, ggml_permute(ctx0, kc, 0, 2, 1, 3)); // [n_eh, n_kv, n_head_kv, ns]
                ggml_tensor * qp = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3));// [n_eh, n_tokens, n_head]
                ggml_tensor * qW = ggml_cont(ctx0, ggml_view_3d(ctx0, qp, qp->ne[0], W, qp->ne[2],
                                                qp->nb[1], qp->nb[2], (n_tokens - W) * qp->nb[1]));
                ggml_tensor * kq = ggml_mul_mat(ctx0, kp, qW);           // [n_kv, W, n_head] (GQA broadcast)
                ggml_tensor * sm = ggml_soft_max_ext(ctx0, kq, nullptr, hparams.f_attention_scale, 0.0f);
                const int64_t nkv = sm->ne[0];
                ggml_tensor * fl = ggml_reshape_2d(ctx0, ggml_cont(ctx0, sm), nkv, sm->ne[1] * sm->ne[2]);
                ggml_tensor * sv = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, fl))); // [1, n_kv]
                ggml_tensor * rl = ggml_reshape_1d(ctx0, sv, nkv);
                snap_rel = snap_rel ? ggml_add(ctx0, snap_rel, rl) : rl;
            }
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    nullptr, // up_exps
                    nullptr, // gate_exps
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    nullptr, // up_exps_s
                    nullptr, // gate_exps_s
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = ggml_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    ggml_tensor * hidden = cur;   // [n_embd, n_outputs] post-final-norm, PRE-lm_head
    res->t_embd = cur;

    // AIS scoring mode = embeddings on (pooling NONE) + multi-token batch. In that
    // mode we emit per-token TARGET logits via the gather-dot below and SKIP the
    // full 256k lm_head (the speedup). Generation (single token) and validation
    // keep the full lm_head. AIS_VALIDATE_GATHER forces both (to compare).
    static const bool ais_keep_full   = getenv("AIS_VALIDATE_GATHER") != nullptr;
    // TODO#2: AIS_SURPRISE_OUT drives the gather-dot via a DEDICATED [1×N] tensor, so it does
    // not need embeddings(pooling) on — letting the host skip the [n_embd_out×N] readback AND
    // its buffer entirely. Otherwise the trigger stays the embeddings flag (validate / B2 path).
    static const bool ais_surprise_out = getenv("AIS_SURPRISE_OUT") != nullptr;
    const bool ais_scoring = (cparams.embeddings || ais_surprise_out) && res->t_inp_tokens && hidden->ne[1] > 1;

    // lm_head (full 256k projection) — conditional
    if (!ais_scoring || ais_keep_full) {
        ggml_tensor * logits = build_lora_mm(model.output, hidden);
        if (hparams.f_final_logit_softcapping) {
            logits = ggml_scale(ctx0, logits, 1.0f / hparams.f_final_logit_softcapping);
            logits = ggml_tanh(ctx0, logits);
            logits = ggml_scale(ctx0, logits, hparams.f_final_logit_softcapping);
        }
        cb(logits, "result_output", -1);
        res->t_logits = logits;
        ggml_build_forward_expand(gf, logits);
    }

    // ── AIS custom: "target-logit" gather-dot (Milestone B) ─────────────────
    // When embeddings mode is on (cparams.embeddings, pooling NONE) and this is a
    // multi-token batch, REPURPOSE the embeddings output to carry, per output
    // position p, the logit of the NEXT input token:
    //     target[p] = hidden[:,p] · W_out[:, token_{p+1}]   (then final softcap)
    // That is exactly what AIS needs for per-token surprise, and it avoids
    // materializing the full 256k-wide logits row (the lm_head readback tax).
    // B1: we KEEP the full lm_head above so ais_prob can validate target==full
    //     logits[tok]. B2 will SKIP the full lm_head when this path is active.
    // ASSUMPTION: full-output scoring batch (n_outputs == n_tokens, contiguous),
    //   so output p ↔ input token p and target = t_inp_tokens[p+1]; the last
    //   position has no in-batch next token → target left 0 (caller ignores it).
    // PORTING to another model: replicate this block right after that model's
    //   lm_head, reusing its post-final-norm `hidden`, its `model.output`, and
    //   its own final-logit transform (softcap/none). Everything else is generic.
    // SnapKV cached-K: la rilevanza [n_kv] è un tensore a parte, nominato + espanso, letto
    // via cb_eval (NON via t_embd: con prefill chunked n_outputs≠n_kv). embeddings resta on
    // solo per saltare l'lm_head (t_embd = hidden, ignorato). Per chunk con n_kv massimo
    // (ultimo chunk) ais_prob tiene quella rilevanza (= query globale → tutte le N chiavi).
    if (do_snap && snap_rel) {
        cb(snap_rel, "ais_snap_rel", -1);
        ggml_build_forward_expand(gf, snap_rel);
    }

    if (ais_scoring && !do_snap) {
        const int64_t n_out = hidden->ne[1];
        const int64_t n_eo  = hparams.n_embd_out();
        ggml_tensor * nxt = ggml_view_1d(ctx0, res->t_inp_tokens, n_out - 1,
                                         ggml_element_size(res->t_inp_tokens));      // token_{p+1}
        ggml_tensor * Wr  = ggml_get_rows(ctx0, model.output, nxt);                  // [n_embd, n_out-1] (dequant)
        ggml_tensor * h0  = ggml_view_2d(ctx0, hidden, hidden->ne[0], n_out - 1,
                                         hidden->nb[1], 0);                           // [n_embd, n_out-1]
        ggml_tensor * tl  = ggml_sum_rows(ctx0, ggml_mul(ctx0, h0, Wr));             // [1, n_out-1]
        if (hparams.f_final_logit_softcapping) {
            tl = ggml_scale(ctx0, tl, 1.0f / hparams.f_final_logit_softcapping);
            tl = ggml_tanh(ctx0, tl);
            tl = ggml_scale(ctx0, tl, hparams.f_final_logit_softcapping);
        }
        tl = ggml_pad(ctx0, tl, 0, 1, 0, 0);          // [1, n_out]    (last col = 0)
        // TODO#2 (PROJECT_GUIDE): a DEDICATED [1, n_out] output instead of padding to the full
        // [n_eo, n_out] embeddings width. The host then reads back only N floats (one target
        // logit per position) instead of the [n_embd_out × N] embd buffer (~138MB @ 13.5k tok),
        // sending the gather-dot "surprise region" to ~0. Gated by AIS_SURPRISE_OUT (A/B).
        if (ais_surprise_out) {
            cb(tl, "ais_surprise", -1);               // [1, n_out] — row p = target logit of token_{p+1}
            res->t_surprise = tl;                     // dedicated narrow readback (see llama-context)
            res->t_embd     = nullptr;                // disable the wide embeddings readback entirely
            ggml_build_forward_expand(gf, tl);
        } else {
            tl = ggml_pad(ctx0, tl, n_eo - 1, 0, 0, 0);   // [n_eo, n_out] row0 = target logit
            cb(tl, "ais_target_logits", -1);
            res->t_embd = tl;
            ggml_build_forward_expand(gf, tl);
        }
    }
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llm_build_gemma4_iswa::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows  (ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llm_build_gemma4_iswa::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    per_layer_proj = ggml_mul_mat   (ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}
