#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv_ruler/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv_ruler/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "${OLLAMA_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/ollama_bin/bin/ollama" ]]; then
    OLLAMA_BIN="${ROOT_DIR}/ollama_bin/bin/ollama"
  elif command -v ollama >/dev/null 2>&1; then
    OLLAMA_BIN="$(command -v ollama)"
  else
    OLLAMA_BIN="ollama"
  fi
fi

DEFAULT_LLM_MODEL="${DEFAULT_LLM_MODEL:-qwen3:8b}"
DEFAULT_EMBED_MODEL="${DEFAULT_EMBED_MODEL:-nomic-embed-text}"
DEFAULT_OPENAI_BASE_URL="${DEFAULT_OPENAI_BASE_URL:-http://127.0.0.1:11534/v1}"
DEFAULT_EMBED_HOST="${DEFAULT_EMBED_HOST:-http://127.0.0.1:11534}"
DEFAULT_RUN_TAG="${DEFAULT_RUN_TAG:-replay_$(date +%Y%m%d)}"
DEFAULT_DIALSIM_SMOKE_SESSIONS="${DEFAULT_DIALSIM_SMOKE_SESSIONS:-10}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_benchmark_replays.sh <target> [extra runner args...]

Targets:
  recover-data             Rebuild local replay datasets from cached benchmark artifacts.
  prepare-data             Try online dataset preparation via official/HF sources.
  ollama-status            Check whether the configured local Ollama host responds.
  ollama-serve             Start the local Ollama server on the configured host.
  ollama-serve-bg          Start the local Ollama server in the background and wait until ready.
  ollama-logs              Tail the log file for the configured local Ollama host.
  ollama-stop              Stop the background Ollama process started by this script.
  ollama-pull-defaults     Pull the default LLM and embedding models.
  dialsim-friends-local-ready
                           Start local Ollama if needed and pull the default models.
  dialsim-friends-local-bootstrap
                           Prepare local Ollama and run a small DialSim smoke.
  dialsim-friends-local-smoke
                           Replay DialSim friends on a small local smoke slice.
  qmsum-local              Replay QMSum from local recovered files.
  qmsum-full-local         Replay QMSum from the official local full test file.
  qmsum-hf                 Replay QMSum from HF dataset fladhak/qmsum.
  friendsqa-hf             Replay FriendsQA from HF dataset SahilRS/friends-qa.
  molweni-local            Replay Molweni from local official files.
  dialsim-friends-local    Replay DialSim friends from local recovered rows.
  dialsim-friends-hf       Replay DialSim friends from default online loader.
  locomo-conv48            Replay LoCoMo conv-48 with planner enabled.

Environment overrides:
  PYTHON_BIN
  OLLAMA_BIN
  DEFAULT_LLM_MODEL
  DEFAULT_EMBED_MODEL
  DEFAULT_OPENAI_BASE_URL
  DEFAULT_EMBED_HOST
  DEFAULT_RUN_TAG
  DEFAULT_DIALSIM_SMOKE_SESSIONS
  DRY_RUN=1                Print the command without executing it.

Examples:
  tools/run_benchmark_replays.sh recover-data
  tools/run_benchmark_replays.sh ollama-status
  tools/run_benchmark_replays.sh ollama-serve
  tools/run_benchmark_replays.sh ollama-serve-bg
  tools/run_benchmark_replays.sh ollama-logs
  tools/run_benchmark_replays.sh ollama-stop
  tools/run_benchmark_replays.sh ollama-pull-defaults
  tools/run_benchmark_replays.sh dialsim-friends-local-ready
  tools/run_benchmark_replays.sh dialsim-friends-local-bootstrap
  tools/run_benchmark_replays.sh dialsim-friends-local-smoke
  tools/run_benchmark_replays.sh qmsum-local --max-queries 20
  DRY_RUN=1 tools/run_benchmark_replays.sh locomo-conv48
EOF
}

strip_openai_v1_suffix() {
  local url="$1"
  printf '%s' "${url%/v1}"
}

strip_url_scheme() {
  local url="$1"
  url="${url#http://}"
  url="${url#https://}"
  printf '%s' "${url%/}"
}

ensure_http_url() {
  local value="$1"
  if [[ "$value" == http://* || "$value" == https://* ]]; then
    printf '%s' "${value%/}"
    return 0
  fi
  printf 'http://%s' "${value%/}"
}

resolve_ollama_serve_host() {
  local explicit_host="${OLLAMA_HOST:-}"
  if [[ -n "$explicit_host" ]]; then
    strip_url_scheme "$explicit_host"
    return 0
  fi
  if [[ -n "${DEFAULT_EMBED_HOST:-}" ]]; then
    strip_url_scheme "$DEFAULT_EMBED_HOST"
    return 0
  fi
  strip_url_scheme "$(strip_openai_v1_suffix "${DEFAULT_OPENAI_BASE_URL:-http://127.0.0.1:11534/v1}")"
}

wait_for_ollama() {
  local http_host="$1"
  local attempts="${2:-30}"
  local delay_seconds="${3:-1}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if curl -fsS --max-time 5 "${http_host%/}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_seconds"
  done
  return 1
}

die() {
  echo "error: $*" >&2
  exit 1
}

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

ensure_local_file() {
  local path="$1"
  local dataset_key="$2"
  if [[ -f "$path" ]]; then
    return 0
  fi
  run_cmd "$PYTHON_BIN" tools/recover_cached_benchmark_datasets.py --dataset "$dataset_key"
  [[ -f "$path" ]] || die "missing required local dataset after recovery: $path"
}

target="${1:-}"
if [[ -z "$target" || "$target" == "-h" || "$target" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "$target" in
  recover-data)
    run_cmd "$PYTHON_BIN" tools/recover_cached_benchmark_datasets.py "$@"
    ;;

  prepare-data)
    run_cmd "$PYTHON_BIN" tools/prepare_benchmark_datasets.py "$@"
    ;;

  ollama-status)
    ollama_http_host="$(ensure_http_url "$(resolve_ollama_serve_host)")"
    echo "+ curl -fsS ${ollama_http_host}/api/tags"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      exit 0
    fi
    curl -fsS "${ollama_http_host}/api/tags"
    ;;

  ollama-serve)
    run_cmd env OLLAMA_HOST="$(resolve_ollama_serve_host)" "$OLLAMA_BIN" serve "$@"
    ;;

  ollama-serve-bg)
    ollama_serve_host="$(resolve_ollama_serve_host)"
    ollama_http_host="$(ensure_http_url "$ollama_serve_host")"
    safe_host_suffix="$(printf '%s' "$ollama_serve_host" | tr ':/' '__')"
    log_path="reports/ollama_${safe_host_suffix}.log"
    pid_path="reports/ollama_${safe_host_suffix}.pid"

    if wait_for_ollama "$ollama_http_host" 1 0; then
      echo "ollama already responding at ${ollama_http_host}"
      exit 0
    fi

    mkdir -p reports
    echo "+ env OLLAMA_HOST=${ollama_serve_host} ${OLLAMA_BIN} serve > ${log_path} 2>&1 &"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      exit 0
    fi

    env OLLAMA_HOST="$ollama_serve_host" "$OLLAMA_BIN" serve >"$log_path" 2>&1 &
    ollama_pid=$!
    printf '%s\n' "$ollama_pid" > "$pid_path"

    if ! wait_for_ollama "$ollama_http_host"; then
      die "ollama did not become ready at ${ollama_http_host}; inspect ${log_path}"
    fi

    echo "ollama ready at ${ollama_http_host} (pid ${ollama_pid}, log ${log_path})"
    ;;

  ollama-logs)
    ollama_serve_host="$(resolve_ollama_serve_host)"
    safe_host_suffix="$(printf '%s' "$ollama_serve_host" | tr ':/' '__')"
    log_path="reports/ollama_${safe_host_suffix}.log"
    echo "+ tail -n 200 -f ${log_path}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      exit 0
    fi
    if [[ ! -f "$log_path" ]]; then
      die "missing log file: $log_path"
    fi
    tail -n 200 -f "$log_path"
    ;;

  ollama-stop)
    ollama_serve_host="$(resolve_ollama_serve_host)"
    safe_host_suffix="$(printf '%s' "$ollama_serve_host" | tr ':/' '__')"
    pid_path="reports/ollama_${safe_host_suffix}.pid"
    echo "+ kill \$(cat ${pid_path})"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      exit 0
    fi
    if [[ ! -f "$pid_path" ]]; then
      die "missing pid file: $pid_path"
    fi
    ollama_pid="$(tr -d '[:space:]' < "$pid_path")"
    if [[ -z "$ollama_pid" ]]; then
      die "empty pid file: $pid_path"
    fi
    kill "$ollama_pid"
    ;;

  ollama-pull-defaults)
    embed_host="${DEFAULT_EMBED_HOST:-http://127.0.0.1:11534}"
    llm_host="$(strip_openai_v1_suffix "${DEFAULT_OPENAI_BASE_URL:-http://127.0.0.1:11534/v1}")"
    run_cmd env OLLAMA_HOST="$embed_host" "$OLLAMA_BIN" pull "$DEFAULT_EMBED_MODEL"
    run_cmd env OLLAMA_HOST="$llm_host" "$OLLAMA_BIN" pull "$DEFAULT_LLM_MODEL"
    ;;

  dialsim-friends-local-ready)
    run_cmd ./tools/run_benchmark_replays.sh ollama-serve-bg
    run_cmd ./tools/run_benchmark_replays.sh ollama-pull-defaults
    ;;

  dialsim-friends-local-bootstrap)
    run_cmd ./tools/run_benchmark_replays.sh dialsim-friends-local-ready
    run_cmd ./tools/run_benchmark_replays.sh dialsim-friends-local-smoke "$@"
    ;;

  qmsum-local)
    ensure_local_file "datasets/qmsum/test.jsonl" "qmsum"
    run_cmd "$PYTHON_BIN" tools/run_qmsum.py \
      --data-path datasets/qmsum \
      --split test \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  qmsum-full-local)
    if [[ ! -f "datasets/qmsum_full/qmsum/test.jsonl" ]]; then
      run_cmd "$PYTHON_BIN" tools/prepare_benchmark_datasets.py --dataset qmsum --qmsum-split test --output-root datasets/qmsum_full
    fi
    [[ -f "datasets/qmsum_full/qmsum/test.jsonl" ]] || die "missing official local QMSum test file"
    run_cmd "$PYTHON_BIN" tools/run_qmsum.py \
      --data-path datasets/qmsum_full/qmsum \
      --split test \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  qmsum-hf)
    run_cmd "$PYTHON_BIN" tools/run_qmsum.py \
      --dataset fladhak/qmsum \
      --split test \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  friendsqa-hf)
    run_cmd "$PYTHON_BIN" tools/run_friendsqa.py \
      --dataset SahilRS/friends-qa \
      --split train \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  molweni-local)
    run_cmd "$PYTHON_BIN" tools/run_molweni.py \
      --data-dir "third_party/Molweni/MRC(withDiscourse)" \
      --split test \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  dialsim-friends-local)
    ensure_local_file "datasets/dialsim/friends/train.jsonl" "dialsim-friends"
    run_cmd "$PYTHON_BIN" tools/run_dialsim.py \
      --script friends \
      --rows-jsonl datasets/dialsim/friends/train.jsonl \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  dialsim-friends-local-smoke)
    ensure_local_file "datasets/dialsim/friends/train.jsonl" "dialsim-friends"
    run_cmd "$PYTHON_BIN" tools/run_dialsim.py \
      --script friends \
      --rows-jsonl datasets/dialsim/friends/train.jsonl \
      --max-sessions "$DEFAULT_DIALSIM_SMOKE_SESSIONS" \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  dialsim-friends-hf)
    run_cmd "$PYTHON_BIN" tools/run_dialsim.py \
      --script friends \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --openai-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --run-tag "$DEFAULT_RUN_TAG" \
      "$@"
    ;;

  locomo-conv48)
    report_path="reports/locomo_conv48_${DEFAULT_RUN_TAG}.json"
    run_cmd "$PYTHON_BIN" statemem/run_locomo \
      --sample-id conv-48 \
      --llm-model "$DEFAULT_LLM_MODEL" \
      --planner-model "$DEFAULT_LLM_MODEL" \
      --embedding-model "$DEFAULT_EMBED_MODEL" \
      --llm-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --planner-base-url "$DEFAULT_OPENAI_BASE_URL" \
      --embedding-host "$DEFAULT_EMBED_HOST" \
      --keep-ollama \
      --report-path "$report_path" \
      "$@"
    ;;

  *)
    die "unknown target: $target"
    ;;
esac
