#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DATASET_JSONL="${DATASET_JSONL:-datasets/dialsim/friends/train.jsonl}"
SLICE_JSONL="${SLICE_JSONL:-reports/dialsim_answer_side_slice_minimal_20260513.jsonl}"
CASE_DIR="${CASE_DIR:-datasets/dialsim/friends/answer_side_slice_cases_20260513}"

LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11534/v1}"
EMBED_HOST="${EMBED_HOST:-http://127.0.0.1:11534}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-answer_side_case}"

if [[ ! -f "${CASE_DIR}/manifest.json" ]]; then
  python tools/materialize_dialsim_case_prefixes.py "$DATASET_JSONL" "$SLICE_JSONL" --out-dir "$CASE_DIR"
fi

python - <<'PY' "$CASE_DIR/manifest.json" "$LLM_MODEL" "$EMBED_MODEL" "$OPENAI_BASE_URL" "$EMBED_HOST" "$RUN_TAG_PREFIX" "$@"
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
llm_model = sys.argv[2]
embed_model = sys.argv[3]
openai_base_url = sys.argv[4]
embed_host = sys.argv[5]
run_tag_prefix = sys.argv[6]
extra_args = sys.argv[7:]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for item in manifest:
    case_id = str(item["case_id"])
    rows_jsonl = str(item["rows_jsonl"])
    cmd = [
        "python",
        "tools/run_dialsim.py",
        "--script",
        "friends",
        "--rows-jsonl",
        rows_jsonl,
        "--sort-policy",
        "date",
        "--reset-policy",
        "none",
        "--questions-per-session",
        "1",
        "--llm-model",
        llm_model,
        "--embedding-model",
        embed_model,
        "--openai-base-url",
        openai_base_url,
        "--embedding-host",
        embed_host,
        "--run-tag",
        f"{run_tag_prefix}_{case_id}",
    ]
    cmd.extend(extra_args)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
PY
