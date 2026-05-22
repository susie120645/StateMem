#!/usr/bin/env python3
"""
Run Molweni MRC with Remem (MemoryOS) as retrieval + generation backend.

This runner targets the official Molweni MRC data format under:
  third_party/Molweni/MRC(withDiscourse)/

Each dialogue is indexed into MemoryOS using EDU windows, then QA is evaluated
with SQuAD-style EM/F1. When answer offsets are available, the runner also
tracks whether a retrieved page covers the gold EDU containing the answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoryos.adaptive_answer_skill import (
    apply_adaptive_answer_action,
    build_adaptive_answer_state,
    select_adaptive_answer_action,
    should_retry_adaptive_answer,
)
from memoryos.answer_routing import (
    build_answer_mode_instruction,
    normalize_generated_answer,
    resolve_effective_answer_format,
    resolve_effective_answer_mode,
)
from memoryos.query_routing import infer_retrieval_query_kind


DEFAULT_DATA_DIR = REPO_ROOT / "third_party" / "Molweni" / "MRC(withDiscourse)"


MOLWENI_PROMPT = """You answer questions about a multi-party dialogue.
Use only the retrieved dialogue context.
Return the exact answer span copied verbatim from the context whenever possible.
Do not explain your reasoning.
If the answer is not in the context, output "unknown".

Question:
{question}

Dialogue Context:
{context}

Answer:"""


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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = " ".join(value.split())
    return value


def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1_score(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_prediction(prediction: Any, gold_answers: Sequence[str]) -> Dict[str, float]:
    if not gold_answers:
        return {"exact_match": 0.0, "f1": 0.0}
    return {
        "exact_match": round(max(exact_match_score(prediction, answer) for answer in gold_answers), 4),
        "f1": round(max(token_f1_score(prediction, answer) for answer in gold_answers), 4),
    }


@dataclass(frozen=True)
class MolweniQaInstance:
    dialogue_id: str
    qa_id: str
    question: str
    answers: Tuple[str, ...]
    answer_starts: Tuple[int, ...]
    gold_edu_ids: Tuple[int, ...]
    is_impossible: bool = False


@dataclass(frozen=True)
class MolweniDialogue:
    dialogue_id: str
    context: str
    edus: Tuple[Dict[str, Any], ...]
    qas: Tuple[MolweniQaInstance, ...]


def _load_split(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    data = payload.get("data", {})
    dialogues = data.get("dialogues", []) if isinstance(data, dict) else []
    return [dialogue for dialogue in dialogues if isinstance(dialogue, dict)]


def _edu_segments(edus: Sequence[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for edu in edus:
        segment = f"{edu.get('speaker', 'Speaker')}: {edu.get('text', '')}".strip()
        if pieces:
            cursor += 1
        start = cursor
        pieces.append(segment)
        cursor += len(segment)
        spans.append((start, cursor))
    return " ".join(pieces), spans


def _map_answer_spans_to_edu_ids(edus: Sequence[Dict[str, Any]], answers: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
    _, spans = _edu_segments(edus)
    gold_edu_ids: List[int] = []
    for answer in answers:
        try:
            start = int(answer.get("answer_start"))
        except Exception:
            continue
        text = str(answer.get("text") or "").strip()
        if not text:
            continue
        end = start + len(text)
        for index, (edu_start, edu_end) in enumerate(spans):
            if end <= edu_start or start >= edu_end:
                continue
            gold_edu_ids.append(index)
    return tuple(sorted(set(gold_edu_ids)))


def load_molweni_dialogues(data_dir: str, split: str = "test") -> List[MolweniDialogue]:
    path = Path(str(data_dir)).expanduser().resolve() / f"{split}.json"
    raw_dialogues = _load_split(path)
    dialogues: List[MolweniDialogue] = []
    for index, raw in enumerate(raw_dialogues):
        edus_raw = raw.get("edus", [])
        if not isinstance(edus_raw, list) or not edus_raw:
            continue
        edus: List[Dict[str, Any]] = []
        for edu_index, edu in enumerate(edus_raw):
            if not isinstance(edu, dict):
                continue
            text = str(edu.get("text") or "").strip()
            if not text:
                continue
            edus.append(
                {
                    "edu_id": edu_index,
                    "speaker": str(edu.get("speaker") or "Speaker").strip() or "Speaker",
                    "text": text,
                }
            )
        if not edus:
            continue
        context = str(raw.get("context") or "").strip()
        if not context:
            context, _ = _edu_segments(edus)
        dialogue_id = str(raw.get("dialogue_id") or raw.get("id") or f"{split}_{index}")
        qas: List[MolweniQaInstance] = []
        for qa_index, qa in enumerate(raw.get("qas", []) or []):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            answers_raw = qa.get("answers", []) or []
            if isinstance(answers_raw, dict):
                answers_raw = [answers_raw]
            answers: List[str] = []
            answer_starts: List[int] = []
            for answer in answers_raw:
                if not isinstance(answer, dict):
                    continue
                text = str(answer.get("text") or "").strip()
                if text:
                    answers.append(text)
                try:
                    answer_starts.append(int(answer.get("answer_start")))
                except Exception:
                    pass
            answers = list(dict.fromkeys(answers))
            answer_starts = list(dict.fromkeys(answer_starts))
            qas.append(
                MolweniQaInstance(
                    dialogue_id=dialogue_id,
                    qa_id=str(qa.get("id") or f"{dialogue_id}_q{qa_index}"),
                    question=question,
                    answers=tuple(answers),
                    answer_starts=tuple(answer_starts),
                    gold_edu_ids=_map_answer_spans_to_edu_ids(edus, answers_raw),
                    is_impossible=bool(qa.get("is_impossible", False)),
                )
            )
        if qas:
            dialogues.append(
                MolweniDialogue(
                    dialogue_id=dialogue_id,
                    context=context,
                    edus=tuple(edus),
                    qas=tuple(qas),
                )
            )
    return dialogues


def _build_chunks(
    edus: Sequence[Dict[str, Any]],
    *,
    dialogue_id: str,
    window_size: int,
    window_stride: int,
) -> List[Dict[str, Any]]:
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(edus):
        end = min(len(edus), start + window_size)
        window = edus[start:end]
        text = "\n".join(f"{item['speaker']}: {item['text']}" for item in window).strip()
        if text:
            chunks.append(
                {
                    "chunk_id": f"{dialogue_id}_chunk_{chunk_index}",
                    "dialogue_id": dialogue_id,
                    "edu_start": start,
                    "edu_end": end - 1,
                    "edu_ids": [int(item["edu_id"]) for item in window],
                    "text": text,
                }
            )
            chunk_index += 1
        if end >= len(edus):
            break
        start += window_stride
    return chunks


def _context_candidates_from_pages(pages: Sequence[Dict[str, Any]]) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for page in pages or []:
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if not raw:
            continue
        for line in raw.replace("\\n", "\n").splitlines():
            line = line.strip()
            if not line or line.startswith("[Dialogue:"):
                continue
            if ":" in line:
                _, tail = line.split(":", 1)
                tail = tail.strip()
                if tail and tail not in seen:
                    seen.add(tail)
                    candidates.append(tail)
            if line and line not in seen:
                seen.add(line)
                candidates.append(line)
    return candidates


def _align_prediction_to_context(prediction: str, pages: Sequence[Dict[str, Any]]) -> str:
    text = str(prediction or "").strip()
    if not text or "unknown" in text.lower():
        return text
    candidates = _context_candidates_from_pages(pages)
    if not candidates:
        return text
    best_candidate = text
    best_score = 0.0
    for candidate in candidates:
        score = token_f1_score(text, candidate)
        if normalize_answer(text) and normalize_answer(text) in normalize_answer(candidate):
            score += 0.15
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate if best_score >= 0.35 else text


def compute_support_hit(retrieved_pages: Sequence[Dict[str, Any]], gold_edu_ids: Sequence[int]) -> Optional[bool]:
    if not gold_edu_ids:
        return None
    gold = {int(item) for item in gold_edu_ids}
    for page in retrieved_pages or []:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        edu_ids = metadata.get("edu_ids")
        if not isinstance(edu_ids, list):
            continue
        page_set = set()
        for item in edu_ids:
            try:
                page_set.add(int(item))
            except Exception:
                continue
        if gold & page_set:
            return True
    return False


def _retrieval_query_kind_for_question(question: str) -> str:
    return infer_retrieval_query_kind(question, task_hint="dialogue_qa")


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
    lightweight_extraction: bool
    keyword_ranker_path: str
    enable_graph_one_hop: bool
    enable_storage_backups: bool
    persist_access_updates: bool


def _build_memoryos(cfg: RememConfig):
    from memoryos.memoryos import Memoryos

    return Memoryos(
        user_id=cfg.user_id,
        assistant_id=cfg.assistant_id,
        openai_api_key=cfg.openai_api_key,
        openai_base_url=cfg.openai_base_url,
        llm_model=cfg.llm_model,
        data_storage_path=cfg.data_storage_path,
        embedding_model_name=cfg.embedding_model_name,
        embedding_model_kwargs={"embedding_host": cfg.embedding_host},
        short_term_capacity=cfg.short_term_capacity,
        long_term_knowledge_capacity=cfg.long_term_knowledge_capacity,
        lightweight_extraction=cfg.lightweight_extraction,
        keyword_ranker_path=cfg.keyword_ranker_path if cfg.keyword_ranker_path else None,
        enable_graph_one_hop=cfg.enable_graph_one_hop,
        enable_storage_backups=cfg.enable_storage_backups,
        enable_async_nli_verify=False,
        enable_async_entity_enrichment=False,
        persist_access_updates=cfg.persist_access_updates,
    )


def _answer_question(
    *,
    client: Any,
    model: str,
    question: str,
    pages: Sequence[Dict[str, Any]],
    max_tokens: int,
    answer_mode: str = "",
    answer_format: str = "",
    extra_guidance: Sequence[str] = (),
) -> str:
    blocks: List[str] = []
    for page in pages or []:
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if raw:
            blocks.append(raw)
    context = "\n\n".join(blocks).strip() if blocks else "No dialogue context was retrieved."
    mode_instruction = build_answer_mode_instruction(
        answer_mode or "extractive",
        task_hint="dialogue_qa",
        answer_format=answer_format,
    )
    prompt = (
        MOLWENI_PROMPT
        + "\n\nAdditional answer constraint:\n"
        + mode_instruction
        + (
            "\n\nAdaptive guidance:\n" + "\n".join(f"- {line}" for line in extra_guidance if str(line).strip())
            if any(str(line).strip() for line in extra_guidance)
            else ""
        )
    ).format(question=question.strip(), context=context)
    answer = str(
        client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
        )
        or ""
    ).strip()
    answer = normalize_generated_answer(
        answer,
        answer_mode,
        task_hint="dialogue_qa",
        answer_format=answer_format,
    )
    return _align_prediction_to_context(answer, pages)


def _build_report(args: argparse.Namespace, results: Sequence[Dict[str, Any]], dialogues_processed: int, questions_processed: int) -> Dict[str, Any]:
    qa_rows = [item for item in results if item.get("qa_id")]
    support_rows = [item for item in qa_rows if item.get("support_hit") is not None]
    return {
        "dataset": "Molweni",
        "split": str(args.split),
        "model": str(args.model_name),
        "llm_model": str(args.llm_model),
        "embedding_model": str(args.embedding_model),
        "dialogues": dialogues_processed,
        "questions": questions_processed,
        "exact_match": round(_mean([float(item["exact_match"]) for item in qa_rows]), 4) if qa_rows else 0.0,
        "f1": round(_mean([float(item["f1"]) for item in qa_rows]), 4) if qa_rows else 0.0,
        "support_hit_rate": round(_mean([1.0 if item["support_hit"] else 0.0 for item in support_rows]), 4)
        if support_rows
        else None,
        "results": list(results),
    }


def _write_outputs(out_csv: Path, out_json: Path, report: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dialogue_id",
        "qa_id",
        "question",
        "gold_answers",
        "prediction",
        "exact_match",
        "f1",
        "dialogue_chunk_count",
        "ingest_ms",
        "retrieve_ms",
        "answer_ms",
        "total_ms",
        "ingest_error",
        "retrieve_error",
        "answer_error",
        "retrieved_count",
        "support_hit",
        "source_stats",
        "hit_ids",
        "effective_answer_mode",
        "effective_answer_format",
        "routing_plan",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["gold_answers"] = json.dumps(row.get("gold_answers", []), ensure_ascii=False)
            row["source_stats"] = json.dumps(row.get("source_stats", {}), ensure_ascii=False)
            row["hit_ids"] = json.dumps(row.get("hit_ids", []), ensure_ascii=False)
            row["routing_plan"] = json.dumps(row.get("routing_plan", {}), ensure_ascii=False)
            writer.writerow({name: row.get(name) for name in fieldnames})
    out_json.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Molweni MRC with Remem (MemoryOS).")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-dialogues", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--answer-max-tokens", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=200)

    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--openai-api-key", default="ollama")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:11534/v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-host", default="")

    parser.add_argument("--model-name", default="remem-qwen3-8b")
    parser.add_argument("--data-storage-root", default=str(REPO_ROOT / "fast_eval_storage" / "molweni_remem"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--run-tag", default="")

    parser.add_argument("--short-term-capacity", type=int, default=18)
    parser.add_argument("--long-term-capacity", type=int, default=6000)
    parser.add_argument("--keyword-ranker-path", default=str(REPO_ROOT / "models" / "keyword_ranker.json"))
    parser.add_argument("--disable-lightweight-extraction", action="store_true", default=False)
    parser.add_argument("--enable-graph-one-hop", action="store_true", default=False)
    parser.add_argument("--persist-access-updates", action="store_true", default=False)
    parser.add_argument("--enable-storage-backups", action="store_true", default=False)
    args = parser.parse_args()

    openai_base_url = str(args.openai_base_url).strip()
    embedding_host = str(args.embedding_host).strip() or _strip_openai_v1_suffix(openai_base_url)
    if _is_localhost_url(embedding_host) and not _ollama_has_model(embedding_host, str(args.embedding_model)):
        raise RuntimeError(
            "Embedding model is not available on the selected Ollama host.\n"
            f"- embedding_host: {embedding_host}\n"
            f"- embedding_model: {args.embedding_model}\n"
            "Fix: pull the embedding model on that Ollama instance, e.g.\n"
            f"  OLLAMA_HOST={embedding_host} ./ollama_bin/bin/ollama pull {args.embedding_model}\n"
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

    split = "dev" if str(args.split) == "dev" else str(args.split)
    dialogues = load_molweni_dialogues(str(args.data_dir), split=split)

    run_id = f"{int(time.time())}"
    tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"molweni_{args.model_name}{tag}_{run_id}.csv"
    out_json = out_dir / f"molweni_{args.model_name}{tag}_{run_id}.json"

    data_storage_path = str(Path(args.data_storage_root).resolve() / args.model_name / f"{split}_{run_id}")
    Path(data_storage_path).mkdir(parents=True, exist_ok=True)

    remem_cfg = RememConfig(
        user_id=f"molweni_{args.model_name}_{run_id}",
        assistant_id="molweni_runner",
        openai_api_key=str(args.openai_api_key),
        openai_base_url=openai_base_url,
        llm_model=str(args.llm_model),
        data_storage_path=data_storage_path,
        embedding_model_name=str(args.embedding_model),
        embedding_host=embedding_host,
        short_term_capacity=int(args.short_term_capacity),
        long_term_knowledge_capacity=int(args.long_term_capacity),
        lightweight_extraction=not bool(args.disable_lightweight_extraction),
        keyword_ranker_path=str(args.keyword_ranker_path) if args.keyword_ranker_path else "",
        enable_graph_one_hop=bool(args.enable_graph_one_hop),
        enable_storage_backups=bool(args.enable_storage_backups),
        persist_access_updates=bool(args.persist_access_updates),
    )
    memoryos = _build_memoryos(remem_cfg)

    results: List[Dict[str, Any]] = []
    dialogues_processed = 0
    questions_processed = 0
    for dialogue in dialogues:
        if args.max_dialogues and dialogues_processed >= int(args.max_dialogues):
            break
        memoryos.clear_all()
        chunks = _build_chunks(
            dialogue.edus,
            dialogue_id=dialogue.dialogue_id,
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
        )
        if not chunks:
            continue

        ingest_started = time.time()
        ingest_error = ""
        for chunk in chunks:
            try:
                memoryos.add_memory(
                    f"[Dialogue: {dialogue.dialogue_id} | EDUs: {chunk['edu_start']}-{chunk['edu_end']}]\n{chunk['text']}",
                    "",
                    meta_data={
                        "doc_id": chunk["chunk_id"],
                        "page_id": chunk["chunk_id"],
                        "dialogue_id": dialogue.dialogue_id,
                        "edu_start": chunk["edu_start"],
                        "edu_end": chunk["edu_end"],
                        "edu_ids": chunk["edu_ids"],
                        "topic": "Molweni Dialogue Chunk",
                        "salience_seed": 1.0,
                    },
                )
            except Exception as exc:
                ingest_error = str(exc)
                break
        ingest_ms = round((time.time() - ingest_started) * 1000.0, 3)

        if ingest_error:
            results.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "qa_id": None,
                    "question": None,
                    "gold_answers": [],
                    "prediction": "",
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "dialogue_chunk_count": len(chunks),
                    "ingest_ms": ingest_ms,
                    "retrieve_ms": 0.0,
                    "answer_ms": 0.0,
                    "total_ms": ingest_ms,
                    "ingest_error": ingest_error,
                    "retrieve_error": "",
                    "answer_error": "",
                    "retrieved_count": 0,
                    "source_stats": {},
                    "hit_ids": [],
                    "support_hit": None,
                }
            )
            dialogues_processed += 1
            continue

        for qa in dialogue.qas:
            if args.max_questions and questions_processed >= int(args.max_questions):
                break
            routing_plan = memoryos.build_task_routing_plan(
                qa.question,
                task_hint="dialogue_qa",
                preferred_query_kind=_retrieval_query_kind_for_question(qa.question),
                extra_signals={"benchmark": "molweni"},
            )
            effective_answer_mode = resolve_effective_answer_mode(routing_plan)
            effective_answer_format = resolve_effective_answer_format(routing_plan)
            retrieve_started = time.time()
            try:
                retrieval_results = memoryos.retrieve_context(
                    qa.question,
                    max_results=int(args.max_results),
                    token_budget=int(args.token_budget),
                    task_hint="dialogue_qa",
                    preferred_query_kind=_retrieval_query_kind_for_question(qa.question),
                    preferred_answer_mode=routing_plan.answer_mode,
                    benchmark="molweni",
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

            adaptive_state = build_adaptive_answer_state(
                qa.question,
                retrieval_results if not retrieve_error else {"retrieved_pages": [], "source_stats": {}},
                trace_context={
                    "task_hint": "dialogue_qa",
                    "routing_confidence": routing_plan.confidence,
                    "token_budget": int(args.token_budget),
                    "task_routing_plan": routing_plan.to_payload(),
                },
                retrieval_mode=routing_plan.retrieval_mode,
                answer_mode=effective_answer_mode,
                answer_format=effective_answer_format,
            )
            adaptive_action = select_adaptive_answer_action(adaptive_state)
            adaptive_overrides = apply_adaptive_answer_action(
                adaptive_state,
                adaptive_action,
                routing_plan.to_payload(),
            )
            chosen_answer_mode = str(adaptive_overrides.get("effective_answer_mode") or effective_answer_mode)
            chosen_answer_format = str(adaptive_overrides.get("effective_answer_format") or effective_answer_format)
            adaptive_guidance = list(adaptive_overrides.get("instruction_fragments") or [])

            answer_started = time.time()
            try:
                prediction = _answer_question(
                    client=memoryos.client,
                    model=memoryos.llm_model,
                    question=qa.question,
                    pages=pages,
                    max_tokens=int(args.answer_max_tokens),
                    answer_mode=chosen_answer_mode,
                    answer_format=chosen_answer_format,
                    extra_guidance=adaptive_guidance,
                )
                adaptive_retry_count = 0
                if should_retry_adaptive_answer(
                    prediction,
                    state=adaptive_state,
                    action=adaptive_action,
                    answer_mode=chosen_answer_mode,
                    answer_format=chosen_answer_format,
                ):
                    retry_pages = pages
                    retry_guidance = list(adaptive_guidance)
                    retry_strategy = str(adaptive_overrides.get("retry_strategy") or "none")
                    if retry_strategy == "retry_with_shorter_context":
                        retry_pages = list(pages[: max(1, min(3, len(pages)))])
                        retry_guidance.append("Re-answer using only the most relevant retrieved evidence.")
                    elif retry_strategy == "retry_with_stricter_format":
                        retry_guidance.append("Re-answer and follow the required answer format exactly.")
                    retry_prediction = _answer_question(
                        client=memoryos.client,
                        model=memoryos.llm_model,
                        question=qa.question,
                        pages=retry_pages,
                        max_tokens=int(args.answer_max_tokens),
                        answer_mode=chosen_answer_mode,
                        answer_format=chosen_answer_format,
                        extra_guidance=retry_guidance,
                    )
                    adaptive_retry_count = 1
                    if retry_prediction.strip():
                        prediction = retry_prediction
                else:
                    adaptive_retry_count = 0
                answer_error = ""
            except Exception as exc:
                prediction = ""
                adaptive_retry_count = 0
                answer_error = str(exc)
            answer_ms = round((time.time() - answer_started) * 1000.0, 3)

            metrics = score_prediction(prediction, qa.answers)
            support_hit = compute_support_hit(pages, qa.gold_edu_ids)
            total_ms = round(ingest_ms + retrieve_ms + answer_ms, 3)
            results.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "qa_id": qa.qa_id,
                    "question": qa.question,
                    "gold_answers": list(qa.answers),
                    "prediction": prediction,
                    "exact_match": metrics["exact_match"],
                    "f1": metrics["f1"],
                    "dialogue_chunk_count": len(chunks),
                    "ingest_ms": ingest_ms,
                    "retrieve_ms": retrieve_ms,
                    "answer_ms": answer_ms,
                    "total_ms": total_ms,
                    "ingest_error": ingest_error,
                    "retrieve_error": retrieve_error,
                    "answer_error": answer_error,
                    "retrieved_count": len(pages),
                    "source_stats": source_stats,
                    "hit_ids": hit_ids,
                    "support_hit": support_hit,
                    "routing_plan": routing_plan.to_payload(),
                    "effective_answer_mode": effective_answer_mode,
                    "effective_answer_format": effective_answer_format,
                    "adaptive_answer_action": adaptive_action.to_payload(),
                    "adaptive_answer_overrides": adaptive_overrides,
                    "adaptive_retry_count": adaptive_retry_count,
                }
            )
            questions_processed += 1
            if args.progress_every and questions_processed % int(args.progress_every) == 0:
                partial_em = _mean([float(item["exact_match"]) for item in results if item.get("qa_id")])
                partial_f1 = _mean([float(item["f1"]) for item in results if item.get("qa_id")])
                print(
                    f"[progress] dialogues={dialogues_processed + 1} questions={questions_processed} "
                    f"EM={partial_em:.4f} F1={partial_f1:.4f}"
                )
            if args.checkpoint_every and questions_processed % int(args.checkpoint_every) == 0:
                report = _build_report(args, results, dialogues_processed, questions_processed)
                _write_outputs(out_csv, out_json, report, results)

        dialogues_processed += 1
        if args.max_questions and questions_processed >= int(args.max_questions):
            break

    memoryos.shutdown()
    report = _build_report(args, results, dialogues_processed, questions_processed)
    _write_outputs(out_csv, out_json, report, results)

    print("Done.")
    print(f"- questions: {questions_processed}")
    print(f"- exact_match: {report['exact_match']:.4f}")
    print(f"- f1: {report['f1']:.4f}")
    if report["support_hit_rate"] is not None:
        print(f"- support_hit_rate: {report['support_hit_rate']:.4f}")
    print(f"- csv: {out_csv}")
    print(f"- json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
