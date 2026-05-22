#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DATASET_JSONL="${DATASET_JSONL:-datasets/dialsim/friends/train.jsonl}"
SLICE_JSONL="${SLICE_JSONL:-reports/dialsim_answer_side_slice_minimal_20260513.jsonl}"
ROWS_JSONL="${ROWS_JSONL:-datasets/dialsim/friends/answer_side_slice_minimal_20260513.rows.jsonl}"

LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11534/v1}"
EMBED_HOST="${EMBED_HOST:-http://127.0.0.1:11534}"

RUN_TAG="${RUN_TAG:-answer_side_slice_ab}"

if [[ ! -f "$ROWS_JSONL" ]]; then
  python tools/materialize_dialsim_subset_rows.py "$DATASET_JSONL" "$SLICE_JSONL" --out "$ROWS_JSONL"
fi

python tools/run_dialsim.py \
  --script friends \
  --rows-jsonl "$ROWS_JSONL" \
  --sort-policy episode_session \
  --reset-policy none \
  --questions-per-session 1 \
  --llm-model "$LLM_MODEL" \
  --embedding-model "$EMBED_MODEL" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --embedding-host "$EMBED_HOST" \
  --run-tag "$RUN_TAG" \
  "$@"
