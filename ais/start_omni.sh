#!/usr/bin/env bash
# ============================================================================
# start_omni.sh — launch AIS OMNI, the final unified mode.
#
#   ONE flag. Streaming-native (Cline benefits). Works on EVERY model via the FA hook
#   (no per-model fork). Bundles: SnapKV eviction + dedup pre-gate + multi-turn delta
#   reuse + auto-MASS + KV-bound gate (no tax on short prompts) + CoT-cut (thinking models).
#   Chat-safe (no prose-lossy lexgate). Quality = vanilla; faster + lighter where it counts.
#
# USAGE:
#   bash ais/start_omni.sh [MODEL.gguf] [PORT] [CTX]
#   bash ais/start_omni.sh                                  # first .gguf in models/, :8080, ctx 16384
#   CODE=1 bash ais/start_omni.sh model.gguf                # OMNI_CODE (max coding compression)
#   KVQ8=1 bash ais/start_omni.sh model.gguf                # ½ KV memory (memory-bound)
#
# Then point any OpenAI client (Cline/Continue) at: http://localhost:<PORT>/v1  (key: anything)
# ============================================================================
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BIN="$LLAMA_DIR/build/bin"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  MODEL=$(ls -1 "$LLAMA_DIR"/models/*.gguf 2>/dev/null | grep -viE 'vocab' | head -1 || true)
fi
PORT="${2:-8080}"
CTX="${3:-16384}"

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "❌ model not found. Pass a path: bash ais/start_omni.sh /path/to/model.gguf"; exit 1
fi
if [[ ! -x "$BIN/ais_prob" ]]; then
  echo "❌ ais_prob not built. Run:"
  echo "   cmake --build $LLAMA_DIR/build --target ais_prob -j\$(sysctl -n hw.logicalcpu)"; exit 1
fi

# free the port if busy
pid=$(lsof -t -i:"$PORT" 2>/dev/null || true); [[ -n "$pid" ]] && { echo "port $PORT busy, freeing…"; kill $pid 2>/dev/null || true; sleep 1; }

FLAG="AIS_OMNI"; [[ "${CODE:-0}" == "1" ]] && FLAG="AIS_OMNI_CODE"
EXTRA=(); [[ "${KVQ8:-0}" == "1" ]] && EXTRA+=("AIS_SNAPKV_KVQ8=1")

echo "🟢 AIS OMNI → :$PORT  (model: $(basename "$MODEL"), ctx=$CTX${CODE:+, CODE}${KVQ8:+, KVQ8})"
echo "   Client (Cline/Continue): OpenAI provider → http://localhost:$PORT/v1  (API key: anything)"
echo ""
exec env "$FLAG=1" "${EXTRA[@]}" \
  "$BIN/ais_prob" "$MODEL" 0.7 sigma-mk \
  --server "$PORT" --host 0.0.0.0 --ctx "$CTX"
