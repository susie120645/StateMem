#!/usr/bin/env python3
"""
Run QMSum with Remem (MemoryOS) as retrieval + generation backend.

This runner expects the official QMSum data files to be available locally.
It supports the common layouts used by the official repo, such as:

  data/ALL/test.jsonl
  data/jsonl/test.jsonl
  data/test/*.json

Each meeting is evaluated independently: the transcript is chunked into
speaker-labeled windows, indexed into MemoryOS, and then used to answer
query-focused summarization prompts. Output metrics include ROUGE-1/2/L F1
and, when gold `relevant_text_span` annotations are present, retrieval support
coverage over transcript spans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
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


QMSUM_PROMPT = """You answer query-focused meeting summarization tasks.
Use only the retrieved meeting context.
Write a concise answer that directly addresses the query.
Prefer concrete names, decisions, reasons, action items, and timeline details when present.
Do not mention retrieval, sources, or uncertainty unless the context is clearly insufficient.

Query Type: {query_type}
Query:
{query}

Meeting Context:
{context}

Answer:"""


QMSUM_GENERAL_PROMPT = """You answer query-focused meeting summarization tasks.
Use only the retrieved meeting context.
Write a compact meeting summary for the query.
Focus on major decisions, conclusions, disagreements, and action items.
Avoid generic lead-in phrases.
Use 3 to 6 short sentences.

Query:
{query}

Meeting Context:
{context}

Answer:"""


QMSUM_SPECIFIC_PROMPT = """You answer query-focused meeting summarization tasks.
Use only the retrieved meeting context.
Answer the question directly in 1 to 3 short sentences.
Copy names, dates, numbers, and decisions verbatim when possible.
Do not add background that is not needed for the answer.

Question:
{query}

Meeting Context:
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


def _normalize_text(text: Any) -> str:
    value = " ".join(str(text or "").split()).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _tokenize_words(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _f1_from_counts(pred_counts: Counter, ref_counts: Counter) -> float:
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum((pred_counts & ref_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, sum(pred_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_n_f1(prediction: Any, reference: Any, n: int) -> float:
    return _f1_from_counts(_ngram_counts(_tokenize_words(prediction), n), _ngram_counts(_tokenize_words(reference), n))


def _lcs_len(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> int:
    if not pred_tokens or not ref_tokens:
        return 0
    previous = [0] * (len(ref_tokens) + 1)
    current = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous, current = current, [0] * (len(ref_tokens) + 1)
    return previous[-1]


def rouge_l_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = _tokenize_words(prediction)
    ref_tokens = _tokenize_words(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_summary(prediction: Any, reference: Any) -> Dict[str, float]:
    return {
        "rouge1_f1": round(rouge_n_f1(prediction, reference, 1), 4),
        "rouge2_f1": round(rouge_n_f1(prediction, reference, 2), 4),
        "rougeL_f1": round(rouge_l_f1(prediction, reference), 4),
    }


def _iter_json_records(path: Path) -> Iterable[Tuple[Dict[str, Any], str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item, f"{path.stem}_{index}"
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if any(key in payload for key in ("meeting_transcripts", "general_query_list", "specific_query_list")):
                yield payload, path.stem
                return
            for key, value in payload.items():
                if isinstance(value, dict):
                    yield value, str(key)
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            yield item, f"{key}_{index}"
                return
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    yield item, f"{path.stem}_{index}"


def _resolve_data_files(data_path: str, split: str) -> List[Path]:
    base = Path(str(data_path)).expanduser().resolve()
    if base.is_file():
        return [base]

    candidates = [
        base / f"{split}.jsonl",
        base / f"{split}.json",
        base / "jsonl" / f"{split}.jsonl",
        base / "jsonl" / f"{split}.json",
        base / "all" / f"{split}.jsonl",
        base / "all" / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return [candidate]

    split_dir = base / split
    if split_dir.exists() and split_dir.is_dir():
        files = sorted([*split_dir.glob("*.json"), *split_dir.glob("*.jsonl")])
        if files:
            return files

    nested_matches = sorted(base.glob(f"**/{split}.jsonl")) + sorted(base.glob(f"**/{split}.json"))
    if nested_matches:
        return nested_matches

    raise FileNotFoundError(
        f"Could not find QMSum split '{split}' under {base}. "
        f"Expected files like {base / f'{split}.jsonl'} or {base / split}/*.json."
    )


def load_qmsum_records(data_path: str, split: str = "test") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for file_path in _resolve_data_files(data_path, split):
        for item, fallback_id in _iter_json_records(file_path):
            record = dict(item)
            record["_source_path"] = str(file_path)
            record["_fallback_id"] = fallback_id
            records.append(record)
    return records


def load_qmsum_records_from_hf(dataset_name: str, split: str = "test") -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    records: List[Dict[str, Any]] = []
    for index, item in enumerate(ds):
        if isinstance(item, dict):
            record = dict(item)
            record["_source_path"] = f"hf://{dataset_name}/{split}"
            record["_fallback_id"] = f"{dataset_name}_{split}_{index}"
            records.append(record)
    return records


def _transcript_signature(record: Dict[str, Any]) -> str:
    transcript = record.get("meeting_transcripts") or record.get("transcript") or record.get("meeting_transcript") or []
    if isinstance(transcript, list):
        parts: List[str] = []
        for turn in transcript:
            if isinstance(turn, dict):
                speaker = str(turn.get("speaker") or turn.get("name") or turn.get("role") or "").strip()
                content = str(turn.get("content") or turn.get("text") or turn.get("utterance") or "").strip()
                if content:
                    parts.append(f"{speaker}\t{content}")
        normalized = "\n".join(parts).strip()
        if normalized:
            return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return hashlib.sha1(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryInstance:
    meeting_id: str
    query_id: str
    query_type: str
    query: str
    answer: str
    relevant_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class MeetingBundle:
    meeting_id: str
    record: Dict[str, Any]
    queries: Tuple[QueryInstance, ...]


def _coerce_answer(payload: Dict[str, Any]) -> str:
    for key in ("answer", "summary", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_relevant_spans(value: Any) -> Tuple[Tuple[int, int], ...]:
    spans: List[Tuple[int, int]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    start = int(item[0])
                    end = int(item[1])
                except Exception:
                    continue
                if end < start:
                    start, end = end, start
                spans.append((start, end))
            elif isinstance(item, dict):
                try:
                    start = int(item.get("start"))
                    end = int(item.get("end"))
                except Exception:
                    continue
                if end < start:
                    start, end = end, start
                spans.append((start, end))
    return tuple(spans)


def _meeting_id(record: Dict[str, Any]) -> str:
    for key in ("meeting_id", "id", "meeting_key", "topic_list"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(record.get("_fallback_id") or "meeting")


def _collect_query_instances(
    record: Dict[str, Any],
    *,
    include_general: bool = True,
    include_specific: bool = True,
) -> List[QueryInstance]:
    meeting_id = _meeting_id(record)
    queries: List[QueryInstance] = []
    flat_query = str(record.get("query") or "").strip()
    flat_answer = _coerce_answer(record)
    if flat_query and flat_answer:
        flat_type = "general" if "whole meeting" in flat_query.lower() or "summarize" in flat_query.lower() else "specific"
        if (flat_type == "general" and include_general) or (flat_type == "specific" and include_specific):
            queries.append(
                QueryInstance(
                    meeting_id=meeting_id,
                    query_id=f"{meeting_id}_{flat_type}_0",
                    query_type=flat_type,
                    query=flat_query,
                    answer=flat_answer,
                    relevant_spans=_coerce_relevant_spans(record.get("relevant_text_span")),
                )
            )
        return queries

    groups: List[Tuple[str, Any]] = []
    if include_general:
        groups.append(("general", record.get("general_query_list")))
    if include_specific:
        groups.append(("specific", record.get("specific_query_list")))

    for query_type, payload in groups:
        if not isinstance(payload, list):
            continue
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            answer = _coerce_answer(item)
            if not query or not answer:
                continue
            queries.append(
                QueryInstance(
                    meeting_id=meeting_id,
                    query_id=f"{meeting_id}_{query_type}_{index}",
                    query_type=query_type,
                    query=query,
                    answer=answer,
                    relevant_spans=_coerce_relevant_spans(item.get("relevant_text_span")),
                )
            )
    return queries


def _normalize_transcript_turn(turn: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(turn, dict):
        return None
    speaker = str(turn.get("speaker") or turn.get("name") or turn.get("role") or "Speaker").strip()
    content = str(turn.get("content") or turn.get("text") or turn.get("utterance") or "").strip()
    if not content:
        return None
    return speaker, content


def _build_transcript_chunks(
    record: Dict[str, Any],
    *,
    window_size: int,
    window_stride: int,
) -> List[Dict[str, Any]]:
    transcript = record.get("meeting_transcripts") or record.get("transcript") or record.get("meeting_transcript") or []
    normalized: List[Tuple[str, str]] = []
    for turn in transcript:
        parsed = _normalize_transcript_turn(turn)
        if parsed is not None:
            normalized.append(parsed)

    if not normalized:
        return []

    meeting_id = _meeting_id(record)
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    chunks: List[Dict[str, Any]] = []

    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(len(normalized), start + window_size)
        lines = [f"{speaker}: {content}" for speaker, content in normalized[start:end]]
        chunk_text = "\n".join(lines).strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{meeting_id}_chunk_{chunk_index}",
                    "meeting_id": meeting_id,
                    "span_start": start,
                    "span_end": end - 1,
                    "text": chunk_text,
                    "turn_count": end - start,
                }
            )
            chunk_index += 1
        if end >= len(normalized):
            break
        start += window_stride
    return chunks


def _group_records_into_meetings(
    records: Sequence[Dict[str, Any]],
    *,
    include_general: bool,
    include_specific: bool,
) -> List[MeetingBundle]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        queries = _collect_query_instances(record, include_general=include_general, include_specific=include_specific)
        if not queries:
            continue
        signature = _transcript_signature(record)
        bucket = grouped.get(signature)
        if bucket is None:
            bucket_meeting_id = _meeting_id(record)
            if not isinstance(bucket_meeting_id, str) or not bucket_meeting_id.strip() or bucket_meeting_id.startswith("fladhak/qmsum"):
                bucket_meeting_id = f"meeting_{signature[:12]}"
            bucket = {
                "meeting_id": bucket_meeting_id,
                "record": dict(record),
                "queries": [],
                "query_ids": set(),
            }
            grouped[signature] = bucket
        for query in queries:
            query_id = query.query_id or f"{bucket['meeting_id']}_q{index}_{len(bucket['queries'])}"
            if query_id in bucket["query_ids"]:
                query_id = f"{query_id}_{len(bucket['queries'])}"
            bucket["query_ids"].add(query_id)
            bucket["queries"].append(
                QueryInstance(
                    meeting_id=str(bucket["meeting_id"]),
                    query_id=query_id,
                    query_type=query.query_type,
                    query=query.query,
                    answer=query.answer,
                    relevant_spans=query.relevant_spans,
                )
            )
    bundles: List[MeetingBundle] = []
    for bucket in grouped.values():
        bundles.append(
            MeetingBundle(
                meeting_id=str(bucket["meeting_id"]),
                record=dict(bucket["record"]),
                queries=tuple(bucket["queries"]),
            )
        )
    bundles.sort(key=lambda item: item.meeting_id)
    return bundles


def _spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def compute_support_coverage(retrieved_pages: Sequence[Dict[str, Any]], relevant_spans: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    if not relevant_spans:
        return {"support_any_hit": None, "support_span_recall": None, "support_hit_count": None}

    retrieved_spans: List[Tuple[int, int]] = []
    for page in retrieved_pages or []:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        start = metadata.get("span_start")
        end = metadata.get("span_end")
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        retrieved_spans.append((start_i, end_i))

    if not retrieved_spans:
        return {"support_any_hit": False, "support_span_recall": 0.0, "support_hit_count": 0}

    hit_count = 0
    for gold_span in relevant_spans:
        if any(_spans_overlap(gold_span, candidate) for candidate in retrieved_spans):
            hit_count += 1
    recall = hit_count / max(1, len(relevant_spans))
    return {
        "support_any_hit": hit_count > 0,
        "support_span_recall": round(recall, 4),
        "support_hit_count": hit_count,
    }


def _retrieval_query_kind_for_query(query_type: str) -> str:
    return infer_retrieval_query_kind("", query_type=query_type, task_hint="meeting_summary") or "local"


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


def _answer_query(
    *,
    client: Any,
    model: str,
    query: str,
    query_type: str,
    pages: Sequence[Dict[str, Any]],
    max_tokens: int,
    answer_mode: str = "",
    answer_format: str = "",
    extra_guidance: Sequence[str] = (),
) -> str:
    def _query_terms(text: str) -> List[str]:
        return [token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if token not in {
            "the","a","an","and","or","to","of","in","on","for","with","what","which","who","when","where","why","how","did","does","was","were","is","are",
        }]

    def _snippet_lines(page: Dict[str, Any], terms: Sequence[str], limit: int = 2) -> List[str]:
        raw = str(page.get("raw_content") or page.get("summary") or "").replace("\\n", "\n")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return []
        scored: List[Tuple[float, str]] = []
        for line in lines:
            lowered = line.lower()
            overlap = sum(1 for term in terms if term in lowered)
            score = float(overlap)
            if ":" in line and not line.startswith("[Meeting:"):
                score += 0.1
            if len(line.split()) <= 28:
                score += 0.05
            scored.append((score, line))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = []
        seen = set()
        for score, line in scored:
            if score <= 0 and chosen:
                break
            if line in seen:
                continue
            seen.add(line)
            chosen.append(line)
            if len(chosen) >= limit:
                break
        return chosen

    query_terms = _query_terms(query)
    blocks: List[str] = []
    for page in pages or []:
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if not raw:
            continue
        if query_type == "general":
            blocks.append(raw)
            continue
        snippets = _snippet_lines(page, query_terms, limit=2)
        if snippets:
            blocks.append("\n".join(snippets))
        else:
            blocks.append(raw)
    context = "\n\n".join(blocks).strip() if blocks else "No meeting context was retrieved."
    prompt_template = QMSUM_GENERAL_PROMPT if query_type == "general" else QMSUM_SPECIFIC_PROMPT
    mode_instruction = build_answer_mode_instruction(
        answer_mode or "abstractive",
        task_hint="meeting_summary",
        answer_format=answer_format,
    )
    prompt = (
        prompt_template
        + "\n\nAdditional answer constraint:\n"
        + mode_instruction
        + (
            "\n\nAdaptive guidance:\n" + "\n".join(f"- {line}" for line in extra_guidance if str(line).strip())
            if any(str(line).strip() for line in extra_guidance)
            else ""
        )
    ).format(query=query.strip(), context=context)
    return str(
        client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
        )
        or ""
    ).strip()


def _build_report(args: argparse.Namespace, results: Sequence[Dict[str, Any]], meetings_processed: int, queries_processed: int) -> Dict[str, Any]:
    rouge1_values = [float(item.get("rouge1_f1", 0.0) or 0.0) for item in results if item.get("query_id")]
    rouge2_values = [float(item.get("rouge2_f1", 0.0) or 0.0) for item in results if item.get("query_id")]
    rougeL_values = [float(item.get("rougeL_f1", 0.0) or 0.0) for item in results if item.get("query_id")]
    support_values = [
        float(item.get("support_span_recall", 0.0))
        for item in results
        if item.get("support_span_recall") is not None
    ]
    support_hits = [
        bool(item.get("support_any_hit"))
        for item in results
        if item.get("support_any_hit") is not None
    ]
    by_query_type: Dict[str, Dict[str, float]] = {}
    for query_type in sorted({item.get("query_type") for item in results if item.get("query_type")}):
        rows = [item for item in results if item.get("query_type") == query_type]
        by_query_type[query_type] = {
            "count": len(rows),
            "rouge1_f1": round(_mean([float(row["rouge1_f1"]) for row in rows]), 4),
            "rouge2_f1": round(_mean([float(row["rouge2_f1"]) for row in rows]), 4),
            "rougeL_f1": round(_mean([float(row["rougeL_f1"]) for row in rows]), 4),
        }
    return {
        "dataset": str(args.dataset or "QMSum"),
        "split": str(args.split),
        "model": str(args.model_name),
        "llm_model": str(args.llm_model),
        "embedding_model": str(args.embedding_model),
        "meetings": meetings_processed,
        "queries": queries_processed,
        "avg_rouge1_f1": round(_mean(rouge1_values), 4),
        "avg_rouge2_f1": round(_mean(rouge2_values), 4),
        "avg_rougeL_f1": round(_mean(rougeL_values), 4),
        "support_any_hit_rate": round(_mean([1.0 if value else 0.0 for value in support_hits]), 4) if support_hits else None,
        "avg_support_span_recall": round(_mean(support_values), 4) if support_values else None,
        "by_query_type": by_query_type,
        "results": list(results),
    }


def _write_outputs(out_csv: Path, out_json: Path, report: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "meeting_id",
        "query_id",
        "query_type",
        "query",
        "reference",
        "prediction",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "meeting_chunk_count",
        "ingest_ms",
        "retrieve_ms",
        "answer_ms",
        "total_ms",
        "ingest_error",
        "retrieve_error",
        "answer_error",
        "retrieved_count",
        "support_any_hit",
        "support_span_recall",
        "support_hit_count",
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
            row["source_stats"] = json.dumps(row.get("source_stats", {}), ensure_ascii=False)
            row["hit_ids"] = json.dumps(row.get("hit_ids", []), ensure_ascii=False)
            row["routing_plan"] = json.dumps(row.get("routing_plan", {}), ensure_ascii=False)
            writer.writerow({name: row.get(name) for name in fieldnames})
    out_json.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QMSum with Remem (MemoryOS).")
    parser.add_argument("--data-path", default="", help="Path to an official QMSum file or data directory.")
    parser.add_argument("--dataset", default="", help="Optional Hugging Face dataset id when using `datasets.load_dataset`.")
    parser.add_argument("--split", default="test", help="Dataset split to load when --data-path is a directory.")
    parser.add_argument("--max-meetings", type=int, default=0, help="Limit number of meetings (0 = all).")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit total number of queries (0 = all).")
    parser.add_argument(
        "--query-types",
        default="general,specific",
        help="Comma-separated subset of query types to run: general,specific",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--max-results-general", type=int, default=12)
    parser.add_argument("--max-results-specific", type=int, default=8)
    parser.add_argument("--token-budget-general", type=int, default=3600)
    parser.add_argument("--token-budget-specific", type=int, default=2200)
    parser.add_argument("--answer-max-tokens", type=int, default=160)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--openai-api-key", default="ollama")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:11534/v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-host", default="")

    parser.add_argument("--model-name", default="remem-qwen3-8b")
    parser.add_argument("--data-storage-root", default=str(REPO_ROOT / "fast_eval_storage" / "qmsum_remem"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--run-tag", default="")

    parser.add_argument("--short-term-capacity", type=int, default=24)
    parser.add_argument("--long-term-capacity", type=int, default=8000)
    parser.add_argument("--keyword-ranker-path", default=str(REPO_ROOT / "models" / "keyword_ranker.json"))
    parser.add_argument("--disable-lightweight-extraction", action="store_true", default=False)
    parser.add_argument("--enable-graph-one-hop", action="store_true", default=False)
    parser.add_argument("--persist-access-updates", action="store_true", default=False)
    parser.add_argument("--enable-storage-backups", action="store_true", default=False)

    args = parser.parse_args()

    if not str(args.data_path).strip() and not str(args.dataset).strip():
        raise SystemExit("Specify either --data-path or --dataset.")

    query_types = {part.strip().lower() for part in str(args.query_types).split(",") if part.strip()}
    include_general = "general" in query_types
    include_specific = "specific" in query_types
    if not include_general and not include_specific:
        raise SystemExit("--query-types must include at least one of: general,specific")

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

    if str(args.dataset).strip():
        records = load_qmsum_records_from_hf(str(args.dataset).strip(), split=str(args.split))
    else:
        records = load_qmsum_records(str(args.data_path), split=str(args.split))
    bundles = _group_records_into_meetings(
        records,
        include_general=include_general,
        include_specific=include_specific,
    )

    run_id = f"{int(time.time())}"
    tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"qmsum_{args.model_name}{tag}_{run_id}.csv"
    out_json = out_dir / f"qmsum_{args.model_name}{tag}_{run_id}.json"

    data_storage_path = str(Path(args.data_storage_root).resolve() / args.model_name / f"{args.split}_{run_id}")
    Path(data_storage_path).mkdir(parents=True, exist_ok=True)

    remem_cfg = RememConfig(
        user_id=f"qmsum_{args.model_name}_{run_id}",
        assistant_id="qmsum_runner",
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
    meetings_processed = 0
    queries_processed = 0

    for bundle in bundles:
        if args.max_meetings and meetings_processed >= int(args.max_meetings):
            break
        queries = list(bundle.queries)
        if not queries:
            continue

        memoryos.clear_all()
        meeting_id = bundle.meeting_id
        chunks = _build_transcript_chunks(
            bundle.record,
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
                    f"[Meeting: {meeting_id} | Turns: {chunk['span_start']}-{chunk['span_end']}]\n{chunk['text']}",
                    "",
                    meta_data={
                        "doc_id": chunk["chunk_id"],
                        "page_id": chunk["chunk_id"],
                        "meeting_id": meeting_id,
                        "span_start": chunk["span_start"],
                        "span_end": chunk["span_end"],
                        "turn_count": chunk["turn_count"],
                        "topic": "QMSum Meeting Chunk",
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
                    "meeting_id": meeting_id,
                    "query_id": None,
                    "query_type": None,
                    "query": None,
                    "reference": None,
                    "prediction": "",
                    "rouge1_f1": 0.0,
                    "rouge2_f1": 0.0,
                    "rougeL_f1": 0.0,
                    "meeting_chunk_count": len(chunks),
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
                    "support_any_hit": None,
                    "support_span_recall": None,
                    "support_hit_count": None,
                }
            )
            meetings_processed += 1
            continue

        for query_instance in queries:
            if args.max_queries and queries_processed >= int(args.max_queries):
                break
            top_k = int(args.max_results_general if query_instance.query_type == "general" else args.max_results_specific)
            token_budget = int(
                args.token_budget_general if query_instance.query_type == "general" else args.token_budget_specific
            )
            routing_plan = memoryos.build_task_routing_plan(
                query_instance.query,
                query_type=query_instance.query_type,
                task_hint="meeting_summary",
                preferred_query_kind=_retrieval_query_kind_for_query(query_instance.query_type),
                extra_signals={"benchmark": "qmsum"},
            )
            effective_answer_mode = resolve_effective_answer_mode(routing_plan)
            effective_answer_format = resolve_effective_answer_format(routing_plan)

            retrieve_started = time.time()
            try:
                retrieval_results = memoryos.retrieve_context(
                    query_instance.query,
                    max_results=top_k,
                    token_budget=token_budget,
                    query_type=query_instance.query_type,
                    task_hint="meeting_summary",
                    preferred_query_kind=_retrieval_query_kind_for_query(query_instance.query_type),
                    preferred_answer_mode=routing_plan.answer_mode,
                    benchmark="qmsum",
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
                query_instance.query,
                retrieval_results if not retrieve_error else {"retrieved_pages": [], "source_stats": {}},
                trace_context={
                    "query_type": query_instance.query_type,
                    "task_hint": "meeting_summary",
                    "routing_confidence": routing_plan.confidence,
                    "token_budget": token_budget,
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
                prediction = _answer_query(
                    client=memoryos.client,
                    model=memoryos.llm_model,
                    query=query_instance.query,
                    query_type=query_instance.query_type,
                    pages=pages,
                    max_tokens=int(args.answer_max_tokens),
                    answer_mode=chosen_answer_mode,
                    answer_format=chosen_answer_format,
                    extra_guidance=adaptive_guidance,
                )
                prediction = normalize_generated_answer(
                    prediction,
                    chosen_answer_mode,
                    task_hint="meeting_summary",
                    answer_format=chosen_answer_format,
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
                    retry_prediction = _answer_query(
                        client=memoryos.client,
                        model=memoryos.llm_model,
                        query=query_instance.query,
                        query_type=query_instance.query_type,
                        pages=retry_pages,
                        max_tokens=int(args.answer_max_tokens),
                        answer_mode=chosen_answer_mode,
                        answer_format=chosen_answer_format,
                        extra_guidance=retry_guidance,
                    )
                    retry_prediction = normalize_generated_answer(
                        retry_prediction,
                        chosen_answer_mode,
                        task_hint="meeting_summary",
                        answer_format=chosen_answer_format,
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

            metrics = score_summary(prediction, query_instance.answer)
            support = compute_support_coverage(pages, query_instance.relevant_spans)
            total_ms = round(ingest_ms + retrieve_ms + answer_ms, 3)

            results.append(
                {
                    "meeting_id": meeting_id,
                    "query_id": query_instance.query_id,
                    "query_type": query_instance.query_type,
                    "query": query_instance.query,
                    "reference": query_instance.answer,
                    "prediction": prediction,
                    "rouge1_f1": metrics["rouge1_f1"],
                    "rouge2_f1": metrics["rouge2_f1"],
                    "rougeL_f1": metrics["rougeL_f1"],
                    "meeting_chunk_count": len(chunks),
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
                    "support_any_hit": support["support_any_hit"],
                    "support_span_recall": support["support_span_recall"],
                    "support_hit_count": support["support_hit_count"],
                    "routing_plan": routing_plan.to_payload(),
                    "effective_answer_mode": effective_answer_mode,
                    "effective_answer_format": effective_answer_format,
                    "adaptive_answer_action": adaptive_action.to_payload(),
                    "adaptive_answer_overrides": adaptive_overrides,
                    "adaptive_retry_count": adaptive_retry_count,
                }
            )
            queries_processed += 1
            if args.progress_every and queries_processed % int(args.progress_every) == 0:
                partial_r1 = _mean([float(item["rouge1_f1"]) for item in results if item.get("query_id")])
                partial_r2 = _mean([float(item["rouge2_f1"]) for item in results if item.get("query_id")])
                partial_rl = _mean([float(item["rougeL_f1"]) for item in results if item.get("query_id")])
                print(
                    f"[progress] meetings={meetings_processed + 1} queries={queries_processed} "
                    f"R1={partial_r1:.4f} R2={partial_r2:.4f} RL={partial_rl:.4f}"
                )
            if args.checkpoint_every and queries_processed % int(args.checkpoint_every) == 0:
                report = _build_report(args, results, meetings_processed, queries_processed)
                _write_outputs(out_csv, out_json, report, results)

        meetings_processed += 1
        if args.max_queries and queries_processed >= int(args.max_queries):
            break

    memoryos.shutdown()
    report = _build_report(args, results, meetings_processed, queries_processed)
    _write_outputs(out_csv, out_json, report, results)

    print("Done.")
    print(f"- queries: {queries_processed}")
    print(f"- avg_rouge1_f1: {report['avg_rouge1_f1']:.4f}")
    print(f"- avg_rouge2_f1: {report['avg_rouge2_f1']:.4f}")
    print(f"- avg_rougeL_f1: {report['avg_rougeL_f1']:.4f}")
    if report["support_any_hit_rate"] is not None:
        print(f"- support_any_hit_rate: {report['support_any_hit_rate']:.4f}")
        print(f"- avg_support_span_recall: {report['avg_support_span_recall']:.4f}")
    print(f"- csv: {out_csv}")
    print(f"- json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
