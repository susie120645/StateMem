#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory
from ollama import Client


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
MEMORYOS_ROOT = REPO_ROOT.parent
DEFAULT_DATASET = MEMORYOS_ROOT / "eval" / "locomo10.json"
RUN_LOCOMO_PATH = REPO_ROOT / "statemem" / "run_locomo"

if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from statemem.evaluation.scorer import build_evaluation_result, summarize_score_metrics  # noqa: E402


TOKEN_FIELDS = (
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
UNKNOWN_MARKERS = (
    "unknown",
    "not mentioned",
    "not provided",
    "does not mention",
    "doesn't mention",
    "not specified",
    "no evidence",
)
INGEST_PROGRESS_LOG_INTERVAL = 10


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("mem0_locomo_full")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "run"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def retry_call(
    label: str,
    func,
    logger: logging.Logger,
    retries: int = 3,
    base_sleep_sec: float = 1.0,
):
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            delay = base_sleep_sec * attempt
            logger.warning(
                "%s failed on attempt %s/%s: %s. Retrying in %.1fs",
                label,
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{label} failed without an exception")


def load_locomo_helpers():
    loader = importlib.machinery.SourceFileLoader("statemem_run_locomo", str(RUN_LOCOMO_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to load LoCoMo helpers from {RUN_LOCOMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid LoCoMo dataset: {dataset_path}")
    return data


def get_gold_answer(qa: Dict[str, Any]) -> str:
    answer = qa.get("answer")
    if answer not in {None, ""}:
        return str(answer).strip()
    adversarial_answer = qa.get("adversarial_answer")
    if adversarial_answer not in {None, ""}:
        return str(adversarial_answer).strip()
    return ""


def session_sort_key(name: str) -> int:
    match = re.fullmatch(r"session_(\d+)", str(name or ""))
    return int(match.group(1)) if match else 0


def iter_sessions(conversation: Dict[str, Any]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    session_names = sorted(
        [key for key in conversation.keys() if re.fullmatch(r"session_\d+", key)],
        key=session_sort_key,
    )
    return [
        (
            session_name,
            conversation.get(f"{session_name}_date_time", "Unknown date"),
            conversation[session_name],
        )
        for session_name in session_names
    ]


def build_turn_message(
    turn: Dict[str, Any],
    session_name: str,
    session_date_time: str,
    speaker_a: str,
) -> Dict[str, str]:
    role = "user" if turn.get("speaker") == speaker_a else "assistant"
    dia_id = turn.get("dia_id", "NA")
    speaker = turn.get("speaker", "Unknown")
    text = str(turn.get("text", "")).strip()
    content = f"[{session_name} | {session_date_time} | {dia_id} | {speaker}] {text}"
    return {"role": role, "content": content}


def empty_usage_counter() -> Dict[str, int]:
    counter = {field: 0 for field in TOKEN_FIELDS}
    counter.update({"call_count": 0, "chat_calls": 0, "embed_calls": 0})
    return counter


def empty_usage_state() -> Dict[str, Any]:
    return {"overall": empty_usage_counter(), "by_host": {}}


def safe_response_field(response: Any, field: str) -> int:
    value = None
    if hasattr(response, field):
        value = getattr(response, field)
    elif isinstance(response, dict):
        value = response.get(field)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class OllamaUsageTracker:
    def __init__(self) -> None:
        self.state = empty_usage_state()

    def record(self, host: str, op: str, response: Any) -> None:
        counters = [self.state["overall"]]
        host_counters = self.state["by_host"].setdefault(host, empty_usage_counter())
        counters.append(host_counters)

        for counter in counters:
            for field in TOKEN_FIELDS:
                counter[field] += safe_response_field(response, field)
            counter["call_count"] += 1
            if op == "chat":
                counter["chat_calls"] += 1
            elif op == "embed":
                counter["embed_calls"] += 1

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    @staticmethod
    def diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        diff_state = empty_usage_state()
        for field in diff_state["overall"]:
            diff_state["overall"][field] = int(after["overall"].get(field, 0)) - int(before["overall"].get(field, 0))

        for host in set(before.get("by_host", {})) | set(after.get("by_host", {})):
            before_counter = before.get("by_host", {}).get(host, empty_usage_counter())
            after_counter = after.get("by_host", {}).get(host, empty_usage_counter())
            host_diff = empty_usage_counter()
            for field in host_diff:
                host_diff[field] = int(after_counter.get(field, 0)) - int(before_counter.get(field, 0))
            diff_state["by_host"][host] = host_diff
        return diff_state


def patch_ollama_clients(tracker: OllamaUsageTracker) -> None:
    original_chat = Client.chat
    original_embed = Client.embed

    def wrapped_chat(self, *args: Any, **kwargs: Any) -> Any:
        response = original_chat(self, *args, **kwargs)
        tracker.record(str(self._client.base_url), "chat", response)
        return response

    def wrapped_embed(self, *args: Any, **kwargs: Any) -> Any:
        response = original_embed(self, *args, **kwargs)
        tracker.record(str(self._client.base_url), "embed", response)
        return response

    Client.chat = wrapped_chat
    Client.embed = wrapped_embed


def summarize_counter(counter: Dict[str, int], denominator: int) -> Dict[str, float]:
    divisor = float(denominator) if denominator else 0.0
    prompt_tokens = int(counter.get("prompt_eval_count", 0))
    completion_tokens = int(counter.get("eval_count", 0))
    total_tokens = prompt_tokens + completion_tokens
    return {
        "call_count": int(counter.get("call_count", 0)),
        "chat_calls": int(counter.get("chat_calls", 0)),
        "embed_calls": int(counter.get("embed_calls", 0)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(prompt_tokens / divisor, 4) if divisor else 0.0,
        "avg_completion_tokens": round(completion_tokens / divisor, 4) if divisor else 0.0,
        "avg_total_tokens": round(total_tokens / divisor, 4) if divisor else 0.0,
        "total_duration_sec": round(int(counter.get("total_duration", 0)) / 1e9, 4),
        "load_duration_sec": round(int(counter.get("load_duration", 0)) / 1e9, 4),
        "prompt_eval_duration_sec": round(int(counter.get("prompt_eval_duration", 0)) / 1e9, 4),
        "eval_duration_sec": round(int(counter.get("eval_duration", 0)) / 1e9, 4),
    }


def summarize_usage(entries: Sequence[Dict[str, Any]], denominator: int) -> Dict[str, Any]:
    aggregate = empty_usage_state()
    for entry in entries:
        overall = entry.get("overall", {})
        for field in aggregate["overall"]:
            aggregate["overall"][field] += int(overall.get(field, 0))
        for host, counter in (entry.get("by_host", {}) or {}).items():
            host_counter = aggregate["by_host"].setdefault(host, empty_usage_counter())
            for field in host_counter:
                host_counter[field] += int(counter.get(field, 0))

    return {
        "overall": summarize_counter(aggregate["overall"], denominator),
        "by_host": {
            host: summarize_counter(counter, denominator)
            for host, counter in sorted(aggregate["by_host"].items())
        },
    }


def metric_stats(values: Sequence[float]) -> Dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "sum": 0.0,
        }
    return {
        "mean": round(mean(numbers), 4),
        "median": round(median(numbers), 4),
        "min": round(min(numbers), 4),
        "max": round(max(numbers), 4),
        "count": len(numbers),
        "sum": round(sum(numbers), 4),
    }


def should_log_turn_progress(turn_number: int, total_turns: int) -> bool:
    if turn_number <= 3:
        return True
    if total_turns and turn_number >= total_turns:
        return True
    return turn_number % INGEST_PROGRESS_LOG_INTERVAL == 0


def extract_text_from_response(response: Any) -> str:
    if isinstance(response, dict):
        return str((response.get("message") or {}).get("content") or "").strip()
    message = getattr(response, "message", None)
    return str(getattr(message, "content", "") or "").strip()


def make_memory(args: argparse.Namespace, chroma_path: Path, history_db_path: Path, collection_name: str) -> Memory:
    config = {
        "version": "v1.0",
        "llm": {
            "provider": "ollama",
            "config": {
                "model": args.llm_model,
                "temperature": 0.0,
                "max_tokens": args.mem0_max_tokens,
                "top_p": 0.1,
                "top_k": 20,
                "ollama_base_url": args.llm_ollama_base_url,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": args.embedding_model,
                "embedding_dims": args.embedding_dims,
                "ollama_base_url": args.embed_ollama_base_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                "path": str(chroma_path),
            },
        },
        "history_db_path": str(history_db_path),
    }
    return Memory.from_config(config)


def build_answer_prompt(question: str, memories: List[Dict[str, Any]]) -> str:
    if memories:
        evidence_lines = []
        for idx, item in enumerate(memories, start=1):
            metadata = item.get("metadata", {}) or {}
            session_name = metadata.get("session_name", "session")
            session_date_time = metadata.get("session_date_time", "Unknown date")
            dia_id = metadata.get("dia_id", "NA")
            speaker = metadata.get("speaker", "Unknown")
            evidence_lines.append(
                f"{idx}. [{session_name} | {session_date_time} | {dia_id} | {speaker}] "
                f"{item.get('memory', '').strip()} (score={item.get('score', 0):.4f})"
            )
        evidence_block = "\n".join(evidence_lines)
    else:
        evidence_block = "No evidence retrieved."
    return (
        f"Memory evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n"
        "Answer using only the evidence above."
    )


def normalize_text(text: object) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())


def is_unknown_like(text: object) -> bool:
    normalized = normalize_text(text)
    return not normalized or any(marker in normalized for marker in UNKNOWN_MARKERS)


def answer_support_ratio(answer: object, evidence_text: object) -> float:
    answer_tokens = normalize_text(answer).split()
    evidence_tokens = set(normalize_text(evidence_text).split())
    if not answer_tokens or not evidence_tokens:
        return 0.0
    overlap = sum(1 for token in answer_tokens if token in evidence_tokens)
    return overlap / float(len(answer_tokens))


def heuristic_support_precision(answer: object, memories: Sequence[Dict[str, Any]]) -> float:
    if is_unknown_like(answer):
        return 0.0
    if not memories:
        return 0.0
    support_hits = 0
    for memory in memories:
        if answer_support_ratio(answer, memory.get("memory", "")) >= 0.6:
            support_hits += 1
    return round(support_hits / float(len(memories)), 4)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def extract_retrieved_evidence_ids(memories: Sequence[Dict[str, Any]]) -> List[str]:
    evidence_ids: List[str] = []
    for memory in memories:
        metadata = memory.get("metadata", {}) or {}
        dia_id = metadata.get("dia_id")
        if dia_id:
            evidence_ids.append(str(dia_id))
    return unique_preserve_order(evidence_ids)


def recall_at_k(gold_ids: Sequence[str], predicted_ids: Sequence[str], k: int) -> float:
    gold = {str(item).strip() for item in gold_ids or [] if str(item).strip()}
    predicted = {str(item).strip() for item in predicted_ids[:k] if str(item).strip()}
    if not gold:
        return 1.0
    return 1.0 if gold.intersection(predicted) else 0.0


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_output = REPO_ROOT / "reports" / f"mem0_locomo_full_{timestamp}.json"
    default_log = REPO_ROOT / "logs" / f"mem0_locomo_full_{timestamp}.log"
    parser = argparse.ArgumentParser(description="Run full LoCoMo10 with mem0 and extended metrics.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--embedding-dims", type=int, default=768)
    parser.add_argument("--llm-ollama-base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--embed-ollama-base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--mem0-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--answer-timeout-sec", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--operation-retries", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--log-file", default=str(default_log))
    parser.add_argument("--run-root", default="")
    return parser.parse_args()


def build_report(
    args: argparse.Namespace,
    run_started_at: str,
    status: str,
    ingestion_records: List[Dict[str, Any]],
    question_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    category_counts: Dict[int, int] = {}
    sample_counts: Dict[str, int] = {}
    for record in question_records:
        category = int(record.get("category") or 0)
        category_counts[category] = category_counts.get(category, 0) + 1
        sample_id = str(record.get("sample_id") or "")
        sample_counts[sample_id] = sample_counts.get(sample_id, 0) + 1

    overall_metrics = {
        "em": [float(record["em"]) for record in question_records],
        "f1": [float(record["f1"]) for record in question_records],
        "recall_at_1": [float(record["recall_at_1"]) for record in question_records],
        "recall_at_10": [float(record["recall_at_10"]) for record in question_records],
        "latency_sec": [float(record["latency_sec"]) for record in question_records],
        "search_latency_sec": [float(record["search_latency_sec"]) for record in question_records],
        "answer_latency_sec": [float(record["answer_latency_sec"]) for record in question_records],
    }

    summary = {name: metric_stats(values) for name, values in overall_metrics.items()}
    summary["grounding"] = summarize_score_metrics([record["evaluation"] for record in question_records])
    summary["token_usage"] = {
        "qa": summarize_usage([record["token_usage"] for record in question_records], len(question_records)),
        "ingest": summarize_usage([record["token_usage"] for record in ingestion_records], len(ingestion_records)),
        "end_to_end": summarize_usage(
            [record["token_usage"] for record in ingestion_records] + [record["token_usage"] for record in question_records],
            len(question_records),
        ),
    }

    by_category: Dict[str, Dict[str, Any]] = {}
    for category, count in sorted(category_counts.items()):
        category_records = [record for record in question_records if int(record.get("category") or 0) == category]
        category_summary = {
            "em": metric_stats([float(record["em"]) for record in category_records]),
            "f1": metric_stats([float(record["f1"]) for record in category_records]),
            "recall_at_1": metric_stats([float(record["recall_at_1"]) for record in category_records]),
            "recall_at_10": metric_stats([float(record["recall_at_10"]) for record in category_records]),
            "latency_sec": metric_stats([float(record["latency_sec"]) for record in category_records]),
            "search_latency_sec": metric_stats([float(record["search_latency_sec"]) for record in category_records]),
            "answer_latency_sec": metric_stats([float(record["answer_latency_sec"]) for record in category_records]),
            "grounding": summarize_score_metrics([record["evaluation"] for record in category_records]),
            "token_usage": summarize_usage([record["token_usage"] for record in category_records], count),
        }
        by_category[str(category)] = category_summary

    ingest_summary = {
        "samples": len({record["sample_id"] for record in ingestion_records}),
        "turn_records": len(ingestion_records),
        "latency_sec": metric_stats([float(record["latency_sec"]) for record in ingestion_records]),
        "token_usage": summarize_usage([record["token_usage"] for record in ingestion_records], len(ingestion_records)),
    }

    return {
        "status": status,
        "run_started_at": run_started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": str(args.dataset),
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "llm_ollama_base_url": args.llm_ollama_base_url,
            "embed_ollama_base_url": args.embed_ollama_base_url,
            "search_limit": args.search_limit,
            "answer_max_tokens": args.answer_max_tokens,
            "ratio": args.ratio,
        },
        "total_questions": len(question_records),
        "sample_distribution": sample_counts,
        "category_distribution": {str(key): value for key, value in sorted(category_counts.items())},
        "summary": {
            "overall": summary,
            "by_category": by_category,
            "ingest": ingest_summary,
        },
        "ingestion_records": ingestion_records,
        "records": question_records,
    }


def main() -> int:
    args = parse_args()
    logger = setup_logger(Path(args.log_file))
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    output_path = Path(args.output).resolve()
    run_root = Path(args.run_root).resolve() if args.run_root else output_path.with_suffix("")
    run_root.mkdir(parents=True, exist_ok=True)

    helpers = load_locomo_helpers()
    tracker = OllamaUsageTracker()
    patch_ollama_clients(tracker)
    answer_client = Client(host=args.llm_ollama_base_url)

    dataset = load_dataset(dataset_path)
    if args.ratio < 1.0:
        sample_count = max(1, int(len(dataset) * args.ratio))
        dataset = dataset[:sample_count]

    run_started_at = datetime.now(timezone.utc).isoformat()
    ingestion_records: List[Dict[str, Any]] = []
    question_records: List[Dict[str, Any]] = []

    logger.info("Loaded %s samples from %s", len(dataset), dataset_path)
    write_json(
        output_path,
        build_report(
            args,
            run_started_at=run_started_at,
            status="running",
            ingestion_records=ingestion_records,
            question_records=question_records,
        ),
    )
    logger.info("Wrote initial running report to %s", output_path)

    for sample_index, sample in enumerate(dataset, start=1):
        sample_id = str(sample.get("sample_id") or f"sample_{sample_index}")
        sample_dir = run_root / sanitize_name(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        collection_name = f"mem0_{sanitize_name(sample_id)}_{int(time.time())}"
        history_db_path = sample_dir / "history.db"
        chroma_path = sample_dir / "chroma"
        user_id = f"mem0_locomo_{sanitize_name(sample_id)}_{int(time.time())}"

        memory = make_memory(args, chroma_path, history_db_path, collection_name)
        conversation = sample["conversation"]
        speaker_a = conversation["speaker_a"]
        session_items = iter_sessions(conversation)
        total_turns = sum(len(turns) for _, _, turns in session_items)
        sample_turn_counter = 0
        sample_ingest_started = time.time()

        logger.info(
            "Ingesting sample %s (%s/%s, sessions=%s, turns=%s)",
            sample_id,
            sample_index,
            len(dataset),
            len(session_items),
            total_turns,
        )
        for session_index, (session_name, session_date_time, turns) in enumerate(session_items, start=1):
            logger.info(
                "Sample %s entering session %s/%s (%s turns)",
                sample_id,
                session_index,
                len(session_items),
                len(turns),
            )
            for turn_index, turn in enumerate(turns, start=1):
                usage_before = tracker.snapshot()
                ingest_started = time.time()
                message = build_turn_message(turn, session_name, session_date_time, speaker_a)
                ingest_error = ""
                next_turn = sample_turn_counter + 1
                if should_log_turn_progress(next_turn, total_turns):
                    logger.info(
                        "Sample %s ingest turn %s/%s starting (session=%s turn=%s dia_id=%s)",
                        sample_id,
                        next_turn,
                        total_turns,
                        session_name,
                        turn_index,
                        turn.get("dia_id"),
                    )
                try:
                    retry_call(
                        label=f"ingest {sample_id} {turn.get('dia_id')}",
                        func=lambda: memory.add(
                            [message],
                            user_id=user_id,
                            metadata={
                                "sample_id": sample_id,
                                "session_name": session_name,
                                "session_date_time": session_date_time,
                                "dia_id": turn.get("dia_id"),
                                "speaker": turn.get("speaker"),
                                "turn_index": turn_index,
                            },
                        ),
                        logger=logger,
                        retries=args.operation_retries,
                    )
                except Exception as exc:  # noqa: BLE001
                    ingest_error = str(exc)
                    logger.error(
                        "Failed to ingest sample=%s session=%s dia_id=%s: %s",
                        sample_id,
                        session_name,
                        turn.get("dia_id"),
                        exc,
                    )
                usage_after = tracker.snapshot()
                sample_turn_counter = next_turn
                if should_log_turn_progress(sample_turn_counter, total_turns):
                    logger.info(
                        "Sample %s ingest turn %s/%s finished in %.2fs (elapsed %.1fs)",
                        sample_id,
                        sample_turn_counter,
                        total_turns,
                        time.time() - ingest_started,
                        time.time() - sample_ingest_started,
                    )
                ingestion_records.append(
                    {
                        "sample_id": sample_id,
                        "session_name": session_name,
                        "session_date_time": session_date_time,
                        "dia_id": turn.get("dia_id"),
                        "speaker": turn.get("speaker"),
                        "turn_index": turn_index,
                        "latency_sec": round(time.time() - ingest_started, 4),
                        "token_usage": OllamaUsageTracker.diff(usage_before, usage_after),
                        "error": ingest_error,
                    }
                )
        write_json(
            output_path,
            build_report(
                args,
                run_started_at=run_started_at,
                status="running",
                ingestion_records=ingestion_records,
                question_records=question_records,
            ),
        )
        logger.info(
            "Wrote ingestion checkpoint after sample %s/%s to %s",
            sample_index,
            len(dataset),
            output_path,
        )

        qas = sample.get("qa", [])
        logger.info("Answering %s questions for sample %s", len(qas), sample_id)
        for question_index, qa in enumerate(qas, start=1):
            gold_answer = get_gold_answer(qa)
            if not gold_answer:
                continue

            usage_before = tracker.snapshot()
            search_started = time.time()
            search_error = ""
            answer_error = ""
            try:
                search_result = retry_call(
                    label=f"search {sample_id} q{question_index}",
                    func=lambda: memory.search(
                        qa["question"],
                        user_id=user_id,
                        limit=args.search_limit,
                        rerank=True,
                    ),
                    logger=logger,
                    retries=args.operation_retries,
                )
                retrieved_memories = search_result.get("results", []) or []
            except Exception as exc:  # noqa: BLE001
                search_error = str(exc)
                logger.error("Search failed for sample=%s q=%s: %s", sample_id, question_index, exc)
                retrieved_memories = []
            search_latency_sec = time.time() - search_started

            answer_started = time.time()
            try:
                answer_response = retry_call(
                    label=f"answer {sample_id} q{question_index}",
                    func=lambda: answer_client.chat(
                        model=args.llm_model,
                        messages=[
                            {"role": "system", "content": helpers.LOCOMO_SYSTEM_PROMPT},
                            {"role": "user", "content": build_answer_prompt(qa["question"], retrieved_memories)},
                        ],
                        options={
                            "temperature": 0.0,
                            "top_p": 0.1,
                            "top_k": 20,
                            "num_predict": args.answer_max_tokens,
                        },
                    ),
                    logger=logger,
                    retries=args.operation_retries,
                )
            except Exception as exc:  # noqa: BLE001
                answer_error = str(exc)
                logger.error("Answer generation failed for sample=%s q=%s: %s", sample_id, question_index, exc)
                answer_response = {"message": {"content": ""}}
            answer_latency_sec = time.time() - answer_started
            usage_after = tracker.snapshot()

            prediction = extract_text_from_response(answer_response)
            retrieved_evidence_ids = extract_retrieved_evidence_ids(retrieved_memories)
            support_precision = heuristic_support_precision(prediction, retrieved_memories)
            em = helpers.exact_match(prediction, gold_answer)
            f1 = helpers.token_f1(prediction, gold_answer)
            evaluation = build_evaluation_result(
                gold_answer=gold_answer,
                predicted_answer=prediction,
                gold_evidence_ids=qa.get("evidence", []),
                predicted_evidence_ids=retrieved_evidence_ids,
                citation_support_precision=support_precision,
                citation_ids=retrieved_evidence_ids,
                hit_ids=retrieved_evidence_ids,
                answer_em=em,
                answer_f1=f1,
            )

            question_records.append(
                {
                    "sample_id": sample_id,
                    "question_index": question_index,
                    "question": qa["question"],
                    "gold_answer": gold_answer,
                    "prediction": prediction,
                    "category": qa.get("category"),
                    "gold_evidence_ids": qa.get("evidence", []),
                    "retrieved_evidence_ids": retrieved_evidence_ids,
                    "em": em,
                    "f1": f1,
                    "recall_at_1": recall_at_k(qa.get("evidence", []), retrieved_evidence_ids, 1),
                    "recall_at_10": recall_at_k(qa.get("evidence", []), retrieved_evidence_ids, 10),
                    "latency_sec": round(search_latency_sec + answer_latency_sec, 4),
                    "search_latency_sec": round(search_latency_sec, 4),
                    "answer_latency_sec": round(answer_latency_sec, 4),
                    "support_precision": support_precision,
                    "token_usage": OllamaUsageTracker.diff(usage_before, usage_after),
                    "evaluation": evaluation,
                    "retrieved_memories": retrieved_memories,
                    "search_error": search_error,
                    "answer_error": answer_error,
                }
            )

            if len(question_records) % args.checkpoint_every == 0:
                logger.info(
                    "Checkpoint %s questions: EM=%.4f F1=%.4f R@1=%.4f R@10=%.4f",
                    len(question_records),
                    mean(float(item["em"]) for item in question_records),
                    mean(float(item["f1"]) for item in question_records),
                    mean(float(item["recall_at_1"]) for item in question_records),
                    mean(float(item["recall_at_10"]) for item in question_records),
                )
                checkpoint = build_report(
                    args,
                    run_started_at=run_started_at,
                    status="running",
                    ingestion_records=ingestion_records,
                    question_records=question_records,
                )
                write_json(output_path, checkpoint)

        del memory
        gc.collect()

    final_report = build_report(
        args,
        run_started_at=run_started_at,
        status="completed",
        ingestion_records=ingestion_records,
        question_records=question_records,
    )
    write_json(output_path, final_report)

    overall = final_report["summary"]["overall"]
    logger.info(
        "Completed %s questions: EM=%.4f F1=%.4f R@1=%.4f R@10=%.4f",
        len(question_records),
        overall["em"]["mean"],
        overall["f1"]["mean"],
        overall["recall_at_1"]["mean"],
        overall["recall_at_10"]["mean"],
    )
    logger.info("Grounding buckets: %s", overall["grounding"]["judgement_buckets"])
    logger.info("Results saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
