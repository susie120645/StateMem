#!/usr/bin/env python3
"""
Run NVIDIA RULER synthetic benchmark with Remem (MemoryOS) as the inference backend.

This script:
1) uses RULER's `prepare.py` to generate synthetic jsonl samples
2) ingests the (long) context into MemoryOS as durable memory (LTM)
3) answers the question by retrieving relevant chunks + calling the LLM
4) runs RULER's `evaluate.py` to produce `summary.csv`

Example:
  .venv_ruler/bin/python tools/run_ruler.py \\
    --taskset synthetic_noqa \\
    --seq-lengths 4096,8192,16384,32768 \\
    --num-samples 10 \\
    --llm-model qwen3:8b \\
    --openai-base-url http://127.0.0.1:11534/v1 \\
    --embedding-host http://127.0.0.1:11534 \\
    --model-name remem-qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RULER_SCRIPTS_DIR = REPO_ROOT / "third_party" / "RULER" / "scripts"
RULER_CONFIG_TASKS = RULER_SCRIPTS_DIR / "config_tasks.sh"
RULER_BENCHMARK_ROOT = RULER_SCRIPTS_DIR / "benchmark_root"


def _strip_openai_v1_suffix(url: str) -> str:
    candidate = str(url or "").strip().rstrip("/")
    if candidate.endswith("/v1"):
        return candidate[: -len("/v1")]
    return candidate


def _is_localhost_url(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(url or "")).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost"}


def _ollama_has_model(ollama_host: str, model_name: str) -> bool:
    """
    Best-effort check against Ollama's `/api/tags`.
    Accepts model name with or without a tag (e.g. `nomic-embed-text` vs `nomic-embed-text:latest`).
    """
    if not ollama_host or not model_name:
        return False

    base = str(model_name).strip().lower().split(":", 1)[0]
    try:
        import requests

        response = requests.get(f"{str(ollama_host).rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False

    models = payload.get("models", []) if isinstance(payload, dict) else []
    for item in models:
        name = str((item or {}).get("name") or "").strip().lower()
        if not name:
            continue
        if name == str(model_name).strip().lower():
            return True
        if name.split(":", 1)[0] == base:
            return True
    return False


def _parse_int_list(value: str) -> List[int]:
    items: List[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _parse_tasksets_from_shell_config(path: Path) -> Dict[str, List[str]]:
    """
    Parse bash arrays like:
      synthetic=( "a" "b" )
    from RULER's `config_tasks.sh`.
    """
    content = path.read_text(encoding="utf-8").splitlines()
    array_start = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=\(\s*$")
    array_end = re.compile(r"^\s*\)\s*$")

    tasksets: Dict[str, List[str]] = {}
    current_name: Optional[str] = None
    current_items: List[str] = []

    for raw_line in content:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if current_name is None:
            match = array_start.match(line)
            if match:
                current_name = match.group(1)
                current_items = []
            continue

        if array_end.match(line):
            tasksets[current_name] = list(current_items)
            current_name = None
            current_items = []
            continue

        # remove inline comment
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        line = line.rstrip(",").strip()
        if not line:
            continue

        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1]
        item = line.strip()
        if item:
            current_items.append(item)

    return tasksets


def _split_ruler_input(prompt: str) -> Tuple[str, str]:
    """
    Heuristic split of a RULER `input` string into (context, query).

    - Prefer the last occurrence of '\\nQuestion:'.
    - Otherwise, treat the last non-empty line ending with '?' as query.
    - Fallback: (prompt, '').
    """
    text = str(prompt or "").strip()
    if not text:
        return "", ""

    marker = "\nQuestion:"
    idx = text.rfind(marker)
    if idx != -1:
        context = text[:idx].strip()
        query = text[idx + len(marker) :].strip()
        return context, query

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[-1].endswith("?"):
        query = lines[-1]
        # recover context without trimming internal newlines too aggressively
        cut = text.rfind(lines[-1])
        if cut != -1:
            context = text[:cut].rstrip()
            return context, query

    return text, ""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _existing_pred_indices(pred_path: Path) -> Set[int]:
    if not pred_path.exists():
        return set()
    indices: Set[int] = set()
    try:
        for item in _iter_jsonl(pred_path):
            idx = item.get("index")
            if isinstance(idx, int):
                indices.add(idx)
    except Exception:
        return set()
    return indices


def _run_subprocess(cmd: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} (cwd={cwd})")


@dataclass(frozen=True)
class RememConfig:
    user_id: str
    assistant_id: str
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    data_storage_path: str
    embedding_model_name: str
    embedding_host: str
    short_term_capacity: int
    long_term_knowledge_capacity: int
    keyword_ranker_path: str
    persist_access_updates: bool
    enable_graph_one_hop: bool
    enable_storage_backups: bool


def _build_memoryos(cfg: RememConfig):
    # Local import so that the script can be imported without initializing MemoryOS.
    from memoryos.memoryos import Memoryos

    return Memoryos(
        user_id=cfg.user_id,
        assistant_id=cfg.assistant_id,
        openai_api_key=cfg.openai_api_key,
        openai_base_url=cfg.openai_base_url,
        llm_model=cfg.llm_model,
        data_storage_path=cfg.data_storage_path,
        embedding_model_name=cfg.embedding_model_name,
        embedding_model_kwargs={
            # embeddings endpoint host (without /v1)
            "embedding_host": cfg.embedding_host,
        },
        short_term_capacity=cfg.short_term_capacity,
        long_term_knowledge_capacity=cfg.long_term_knowledge_capacity,
        keyword_ranker_path=cfg.keyword_ranker_path if cfg.keyword_ranker_path else None,
        enable_graph_one_hop=cfg.enable_graph_one_hop,
        enable_storage_backups=cfg.enable_storage_backups,
        enable_async_nli_verify=False,
        enable_async_entity_enrichment=False,
        persist_access_updates=cfg.persist_access_updates,
    )


def _answer_from_pages(
    *,
    client: Any,
    model: str,
    query: str,
    pages: List[Dict[str, Any]],
    max_tokens: int,
) -> str:
    if not pages:
        return "I don't know."

    blocks: List[str] = []
    for page in pages:
        citation_id = page.get("citation_id") or page.get("page_id") or ""
        raw_content = str(page.get("raw_content") or page.get("summary") or "").strip()
        if not raw_content:
            continue
        blocks.append(f"[{citation_id}]\n{raw_content}")

    prompt = (
        "Answer the question using ONLY the memory context below.\n"
        "Return ONLY the final answer (no analysis, no extra words).\n"
        "If the context is insufficient, return exactly: I don't know.\n\n"
        f"Question:\n{query}\n\n"
        "Memory context:\n"
        + "\n".join(blocks)
    )

    try:
        return str(
            client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max(1, int(max_tokens)),
            )
            or ""
        ).strip()
    except Exception as exc:
        return f"Error: {exc}"


def _predict_task_with_remem(
    *,
    memoryos: Any,
    task: str,
    task_file: Path,
    pred_file: Path,
    subset: str,
    max_results: int,
    token_budget: int,
    answer_max_tokens: int,
    salience_seed: float,
) -> None:
    done = _existing_pred_indices(pred_file)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0

    with pred_file.open("a", encoding="utf-8", buffering=1) as fout:
        for sample in _iter_jsonl(task_file):
            total += 1
            index = sample.get("index")
            if not isinstance(index, int):
                continue
            if index in done:
                continue

            input_text = str(sample.get("input") or "")
            context, query = _split_ruler_input(input_text)
            if not query:
                query = input_text

            # Reset per-sample memory to keep RULER iid semantics.
            memoryos.clear_all()

            doc_id = f"ruler_{task}_{index}"
            ingest_started = time.time()
            try:
                memoryos.add_memory(
                    context,
                    "",
                    meta_data={
                        "doc_id": doc_id,
                        "page_id": doc_id,
                        "topic": "RULER",
                        "flags": {"important": True},
                        "salience_seed": float(salience_seed),
                    },
                )
                ingest_error = ""
            except Exception as exc:
                ingest_error = str(exc)

            ingest_ms = round((time.time() - ingest_started) * 1000.0, 3)

            retrieve_started = time.time()
            try:
                retrieval_results = memoryos.retriever.retrieve_context(
                    query,
                    top_k=int(max_results),
                    token_budget=int(token_budget),
                )
                pages = retrieval_results.get("retrieved_pages", []) or []
                source_stats = retrieval_results.get("source_stats", {}) or {}
                hit_ids = retrieval_results.get("hit_ids", []) or []
                retrieve_error = ""
            except Exception as exc:
                pages = []
                source_stats = {}
                hit_ids = []
                retrieve_error = str(exc)
            retrieve_ms = round((time.time() - retrieve_started) * 1000.0, 3)

            answer_started = time.time()
            pred_text = _answer_from_pages(
                client=memoryos.client,
                model=memoryos.llm_model,
                query=query,
                pages=pages,
                max_tokens=answer_max_tokens,
            )
            answer_ms = round((time.time() - answer_started) * 1000.0, 3)

            others = sample.get("others")
            if not isinstance(others, dict):
                others = {}
            others = dict(others)
            others["remem"] = {
                "task": task,
                "subset": subset,
                "doc_id": doc_id,
                "ingest_ms": ingest_ms,
                "retrieve_ms": retrieve_ms,
                "answer_ms": answer_ms,
                "ingest_error": ingest_error,
                "retrieve_error": retrieve_error,
                "source_stats": source_stats,
                "hit_ids": hit_ids,
                "retrieved_count": len(pages),
                "token_budget": int(token_budget),
                "max_results": int(max_results),
            }

            outputs = sample.get("outputs")
            if not isinstance(outputs, list):
                fallback_output = sample.get("output")
                outputs = [str(fallback_output)] if fallback_output not in {None, ""} else [""]

            out = {
                "index": index,
                "pred": pred_text,
                "input": input_text,
                "outputs": outputs,
                "others": others,
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if written % 5 == 0:
                print(f"[pred] task={task} wrote={written} last_index={index} ingest_ms={ingest_ms} retrieve_ms={retrieve_ms} answer_ms={answer_ms}")

    print(f"[pred] task={task} done: wrote={written} skipped={len(done)} total_seen={total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RULER synthetic benchmark with Remem (MemoryOS).")
    parser.add_argument("--taskset", default="synthetic_noqa", help="Taskset name in RULER config_tasks.sh")
    parser.add_argument("--seq-lengths", default="4096,8192,16384,32768")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--model-name", default="remem-qwen3-8b")
    parser.add_argument("--benchmark-data", default="synthetic", help="RULER benchmark yaml/constants name (usually synthetic).")

    # LLM / embeddings
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--openai-api-key", default="ollama")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:11534/v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-host", default="")

    # Remem retrieval/generation knobs
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=2200)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--salience-seed", type=float, default=0.95)

    # Storage
    parser.add_argument("--data-storage-root", default=str(REPO_ROOT / "fast_eval_storage" / "ruler_remem"))
    parser.add_argument("--short-term-capacity", type=int, default=5)
    parser.add_argument("--long-term-capacity", type=int, default=6000)
    parser.add_argument("--keyword-ranker-path", default=str(REPO_ROOT / "models" / "keyword_ranker.json"))
    parser.add_argument("--persist-access-updates", action="store_true", default=False)
    parser.add_argument("--disable-graph-one-hop", action="store_true", default=False)
    parser.add_argument("--enable-storage-backups", action="store_true", default=False)

    # Pipeline control
    parser.add_argument("--skip-prepare", action="store_true", default=False)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    args = parser.parse_args()

    if not RULER_SCRIPTS_DIR.exists():
        raise FileNotFoundError(f"RULER scripts dir not found: {RULER_SCRIPTS_DIR}")
    if not RULER_CONFIG_TASKS.exists():
        raise FileNotFoundError(f"RULER config not found: {RULER_CONFIG_TASKS}")

    tasksets = _parse_tasksets_from_shell_config(RULER_CONFIG_TASKS)
    if args.taskset not in tasksets:
        raise ValueError(f"Unknown taskset={args.taskset!r}. Available: {sorted(tasksets.keys())}")

    seq_lengths = _parse_int_list(args.seq_lengths)
    tasks = tasksets[args.taskset]
    if not tasks:
        raise ValueError(f"Empty task list for taskset={args.taskset!r}")

    openai_base_url = str(args.openai_base_url).strip()
    embedding_host = str(args.embedding_host).strip() or _strip_openai_v1_suffix(openai_base_url)

    # Preflight checks for local Ollama setups to prevent silent all-zero embeddings.
    if _is_localhost_url(embedding_host) and not _ollama_has_model(embedding_host, str(args.embedding_model)):
        raise RuntimeError(
            "Embedding model is not available on the selected Ollama host.\n"
            f"- embedding_host: {embedding_host}\n"
            f"- embedding_model: {args.embedding_model}\n"
            "Fix: pull the embedding model on that Ollama instance, e.g.\n"
            f"  OLLAMA_HOST={embedding_host} ./ollama_bin/bin/ollama pull {args.embedding_model}\n"
            "Or point `--embedding-host/--embedding-model` to an Ollama instance that already has an embedding model."
        )

    llm_host = _strip_openai_v1_suffix(openai_base_url)
    if _is_localhost_url(llm_host) and not _ollama_has_model(llm_host, str(args.llm_model)):
        raise RuntimeError(
            "LLM model is not available on the selected Ollama host.\n"
            f"- llm_host: {llm_host}\n"
            f"- llm_model: {args.llm_model}\n"
            "Fix: pull the model on that Ollama instance, e.g.\n"
            f"  OLLAMA_HOST={llm_host} ./ollama_bin/bin/ollama pull {args.llm_model}\n"
        )

    run_id = f"{int(time.time())}"
    data_storage_path = str(Path(args.data_storage_root).resolve() / args.model_name / f"{args.taskset}_{run_id}")
    Path(data_storage_path).mkdir(parents=True, exist_ok=True)

    remem_cfg = RememConfig(
        user_id=f"ruler_{args.model_name}_{run_id}",
        assistant_id="ruler_runner",
        openai_api_key=str(args.openai_api_key),
        openai_base_url=openai_base_url,
        llm_model=str(args.llm_model),
        data_storage_path=data_storage_path,
        embedding_model_name=str(args.embedding_model),
        embedding_host=embedding_host,
        short_term_capacity=int(args.short_term_capacity),
        long_term_knowledge_capacity=int(args.long_term_capacity),
        keyword_ranker_path=str(args.keyword_ranker_path) if args.keyword_ranker_path else "",
        persist_access_updates=bool(args.persist_access_updates),
        enable_graph_one_hop=not bool(args.disable_graph_one_hop),
        enable_storage_backups=bool(args.enable_storage_backups),
    )

    memoryos = _build_memoryos(remem_cfg)

    for max_seq_length in seq_lengths:
        results_dir = RULER_BENCHMARK_ROOT / args.model_name / args.taskset / str(max_seq_length)
        data_dir = results_dir / "data"
        pred_dir = results_dir / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            if not args.skip_prepare:
                _run_subprocess(
                    [
                        sys.executable,
                        "data/prepare.py",
                        "--save_dir",
                        str(data_dir),
                        "--benchmark",
                        str(args.benchmark_data),
                        "--task",
                        str(task),
                        "--tokenizer_path",
                        "cl100k_base",
                        "--tokenizer_type",
                        "openai",
                        "--max_seq_length",
                        str(max_seq_length),
                        "--model_template_type",
                        "base",
                        "--num_samples",
                        str(int(args.num_samples)),
                        "--subset",
                        str(args.subset),
                    ],
                    cwd=RULER_SCRIPTS_DIR,
                )

            task_file = data_dir / task / f"{args.subset}.jsonl"
            if not task_file.exists():
                raise FileNotFoundError(f"Task data not found: {task_file}")

            pred_file = pred_dir / f"{task}.jsonl"
            _predict_task_with_remem(
                memoryos=memoryos,
                task=task,
                task_file=task_file,
                pred_file=pred_file,
                subset=str(args.subset),
                max_results=int(args.max_results),
                token_budget=int(args.token_budget),
                answer_max_tokens=int(args.answer_max_tokens),
                salience_seed=float(args.salience_seed),
            )

        if not args.skip_eval:
            _run_subprocess(
                [
                    sys.executable,
                    "eval/evaluate.py",
                    "--data_dir",
                    str(pred_dir),
                    "--benchmark",
                    str(args.benchmark_data),
                ],
                cwd=RULER_SCRIPTS_DIR,
            )

    try:
        memoryos.shutdown(wait_for_background_tasks=False)
    except Exception:
        pass

    print(f"Done. Outputs under: {RULER_BENCHMARK_ROOT / args.model_name / args.taskset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
