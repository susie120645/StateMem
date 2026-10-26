#!/usr/bin/env python3
import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import requests

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
MEMORYOS_ROOT = REPO_ROOT.parent
DEFAULT_DATASET = MEMORYOS_ROOT / "eval" / "locomo10.json"
RUN_LOCOMO_PATH = REPO_ROOT / "statemem" / "run_locomo"


def load_locomo_helpers():
    loader = importlib.machinery.SourceFileLoader("statemem_run_locomo", str(RUN_LOCOMO_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to load LoCoMo helpers from {RUN_LOCOMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_sample(dataset_path: Path, sample_id: str) -> Dict[str, Any]:
    data = json.loads(dataset_path.read_text())
    for item in data:
        if item.get("sample_id") == sample_id:
            return item
    raise ValueError(f"Sample {sample_id!r} not found in {dataset_path}")


def get_gold_answer(qa: Dict[str, Any]) -> str:
    answer = qa.get("answer")
    if answer not in {None, ""}:
        return str(answer).strip()
    adversarial_answer = qa.get("adversarial_answer")
    if adversarial_answer not in {None, ""}:
        return str(adversarial_answer).strip()
    return ""


def session_sort_key(name: str) -> int:
    match = re.fullmatch(r"session_(\d+)", name)
    if not match:
        return 0
    return int(match.group(1))


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


def build_messages(
    turns: List[Dict[str, Any]],
    session_name: str,
    session_date_time: str,
    speaker_a: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in turns:
        role = "user" if turn.get("speaker") == speaker_a else "assistant"
        dia_id = turn.get("dia_id", "NA")
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "").strip()
        content = f"[{session_name} | {session_date_time} | {dia_id} | {speaker}] {text}"
        messages.append({"role": role, "content": content})
    return messages


def make_memory(args: argparse.Namespace, chroma_path: Path, history_db_path: Path) -> Memory:
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
                "ollama_base_url": args.ollama_base_url,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": args.embedding_model,
                "embedding_dims": args.embedding_dims,
                "ollama_base_url": args.ollama_base_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": args.collection_name,
                "path": str(chroma_path),
            },
        },
        "history_db_path": str(history_db_path),
    }
    return Memory.from_config(config)


def chat_with_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int,
    num_predict: int,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 20,
                "num_predict": num_predict,
            },
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") or {}
    return str(message.get("content") or "").strip()


def build_answer_prompt(question: str, memories: List[Dict[str, Any]]) -> str:
    if memories:
        evidence_lines = [
            f"{idx}. {item.get('memory', '').strip()} (score={item.get('score', 0):.4f})"
            for idx, item in enumerate(memories, start=1)
        ]
        evidence_block = "\n".join(evidence_lines)
    else:
        evidence_block = "No evidence retrieved."
    return (
        f"Memory evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n"
        "Answer using only the evidence above."
    )


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (REPO_ROOT / "reports" / f"mem0_locomo_{args.sample_id}_{timestamp}.json").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mem0 locally on a LoCoMo sample.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sample-id", default="conv-48")
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--search-limit", type=int, default=8)
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--embedding-dims", type=int, default=768)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--mem0-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-timeout-sec", type=int, default=180)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path = build_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_path.with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)

    args.collection_name = f"mem0_{args.sample_id}_{int(time.time())}"
    chroma_path = run_dir / "chroma"
    history_db_path = run_dir / "history.db"
    user_id = f"mem0_locomo_{args.sample_id}_{int(time.time())}"

    helpers = load_locomo_helpers()
    sample = load_sample(dataset_path, args.sample_id)
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    sessions = iter_sessions(conversation)
    qas = sample["qa"]
    if args.question_limit > 0:
        qas = qas[: args.question_limit]

    started_at = datetime.now(timezone.utc)
    memory = make_memory(args, chroma_path, history_db_path)

    ingestion_records: List[Dict[str, Any]] = []
    ingest_started = time.time()
    for index, (session_name, session_date_time, turns) in enumerate(sessions, start=1):
        messages = build_messages(turns, session_name, session_date_time, speaker_a)
        session_start = time.time()
        memory.add(
            messages,
            user_id=user_id,
            metadata={
                "sample_id": args.sample_id,
                "session_name": session_name,
                "session_date_time": session_date_time,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            },
        )
        ingestion_records.append(
            {
                "session_name": session_name,
                "session_date_time": session_date_time,
                "turn_count": len(turns),
                "latency_sec": round(time.time() - session_start, 3),
                "session_index": index,
            }
        )
        print(f"[ingest] {session_name} ({index}/{len(sessions)})")
    ingest_latency_sec = time.time() - ingest_started

    qa_records: List[Dict[str, Any]] = []
    qa_started = time.time()
    for index, qa in enumerate(qas, start=1):
        question = qa["question"]
        gold_answer = get_gold_answer(qa)
        if not gold_answer:
            print(f"[skip] {index}/{len(qas)} missing gold answer")
            continue
        evidence = qa.get("evidence", [])

        search_start = time.time()
        search_result = memory.search(
            question,
            user_id=user_id,
            limit=args.search_limit,
            rerank=True,
        )
        search_latency = time.time() - search_start
        memories = search_result.get("results", [])

        answer_start = time.time()
        prediction = chat_with_ollama(
            base_url=args.ollama_base_url,
            model=args.llm_model,
            system_prompt=helpers.LOCOMO_SYSTEM_PROMPT,
            user_prompt=build_answer_prompt(question, memories),
            timeout_sec=args.answer_timeout_sec,
            num_predict=args.answer_max_tokens,
        )
        answer_latency = time.time() - answer_start

        em = helpers.exact_match(prediction, gold_answer)
        f1 = helpers.token_f1(prediction, gold_answer)

        qa_records.append(
            {
                "question_index": index,
                "question": question,
                "prediction": prediction,
                "gold_answer": gold_answer,
                "evidence": evidence,
                "category": qa.get("category"),
                "em": em,
                "f1": f1,
                "search_latency_sec": round(search_latency, 3),
                "answer_latency_sec": round(answer_latency, 3),
                "retrieved_memories": memories,
            }
        )
        print(
            f"[qa] {index}/{len(qas)} em={em:.1f} f1={f1:.3f} "
            f"search={search_latency:.2f}s answer={answer_latency:.2f}s"
        )
    qa_latency_sec = time.time() - qa_started

    report = {
        "dataset": str(dataset_path),
        "sample_id": args.sample_id,
        "question_count": len(qa_records),
        "ollama_base_url": args.ollama_base_url,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "search_limit": args.search_limit,
        "user_id": user_id,
        "speakers": {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ingestion": {
            "session_count": len(ingestion_records),
            "latency_sec": round(ingest_latency_sec, 3),
            "records": ingestion_records,
        },
        "overall": {
            "question_count": len(qa_records),
            "em": round(mean(record["em"] for record in qa_records), 4) if qa_records else 0.0,
            "f1": round(mean(record["f1"] for record in qa_records), 4) if qa_records else 0.0,
            "qa_latency_sec": round(qa_latency_sec, 3),
            "avg_search_latency_sec": round(mean(record["search_latency_sec"] for record in qa_records), 3)
            if qa_records
            else 0.0,
            "avg_answer_latency_sec": round(mean(record["answer_latency_sec"] for record in qa_records), 3)
            if qa_records
            else 0.0,
            "avg_retrieved_memories": round(mean(len(record["retrieved_memories"]) for record in qa_records), 3)
            if qa_records
            else 0.0,
        },
        "records": qa_records,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
