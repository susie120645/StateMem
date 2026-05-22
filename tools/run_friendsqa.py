#!/usr/bin/env python3
"""
Run FriendsQA with Remem (MemoryOS) as retrieval + generation backend.

This runner supports official-style local JSON files as well as Hugging Face
datasets-compatible splits when available. Each dialogue is evaluated
independently: the dialogue transcript is chunked into speaker-labeled windows,
indexed into MemoryOS, and then used to answer span-style questions.

Output metrics:
  - Exact Match (EM)
  - Token-level F1 (SQuAD-style normalization)
  - Retrieval support hit rate over gold answer utterance ids when available
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


FRIENDSQA_PROMPT = """You answer questions about a multi-party dialogue.
Use only the retrieved dialogue context.
Return the exact answer span copied verbatim from the dialogue context.
Prefer the shortest verbatim span that fully answers the question.
If the answer is a person's name, location, object, or short phrase, output only that phrase.
Do not explain your reasoning.
If the context does not contain the answer, output "unknown".

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


def _iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        for key in ("dialogs", "dialogues", "examples", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _resolve_data_files(data_path: str, split: str) -> List[Path]:
    base = Path(str(data_path)).expanduser().resolve()
    if base.is_file():
        return [base]

    aliases = [split]
    lowered = split.lower()
    if lowered == "validation":
        aliases.extend(["dev", "valid", "val"])
    elif lowered == "train":
        aliases.extend(["trn"])
    elif lowered == "test":
        aliases.extend(["tst"])

    candidates: List[Path] = []
    for alias in aliases:
        candidates.extend(
            [
                base / f"{alias}.json",
                base / f"{alias}.jsonl",
                base / f"friendsqa_{alias}.json",
                base / f"friendsqa_{alias}.jsonl",
                base / alias / "data.json",
                base / alias / f"{alias}.json",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return [candidate]

    nested_matches: List[Path] = []
    for alias in aliases:
        nested_matches.extend(sorted(base.glob(f"**/{alias}.json")))
        nested_matches.extend(sorted(base.glob(f"**/{alias}.jsonl")))
        nested_matches.extend(sorted(base.glob(f"**/friendsqa_{alias}.json")))
        nested_matches.extend(sorted(base.glob(f"**/friendsqa_{alias}.jsonl")))
    if nested_matches:
        return nested_matches

    raise FileNotFoundError(
        f"Could not find FriendsQA split '{split}' under {base}. "
        f"Expected files like {base / f'{split}.json'} or {base / f'friendsqa_{split}.json'}."
    )


def _normalize_speaker(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " & ".join(parts) if parts else "Speaker"
    text = str(value or "").strip()
    return text or "Speaker"


def _context_signature(context: str) -> str:
    normalized = " ".join(str(context or "").split()).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _normalize_utterance(item: Any, index: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"utterance_id": index, "speaker": "Speaker", "text": text}

    if not isinstance(item, dict):
        return None

    speaker = _normalize_speaker(item.get("speaker") or item.get("speakers") or item.get("role") or item.get("name"))
    text = str(
        item.get("utterance")
        or item.get("content")
        or item.get("text")
        or item.get("sentence")
        or item.get("script")
        or ""
    ).strip()
    if not text:
        return None
    utterance_id = item.get("utterance_id", item.get("uid", item.get("id", index)))
    try:
        utterance_id = int(utterance_id)
    except Exception:
        utterance_id = index
    return {
        "utterance_id": utterance_id,
        "speaker": speaker,
        "text": text,
    }


def _split_flat_context_to_utterances(context: str) -> List[Dict[str, Any]]:
    normalized_context = str(context or "").replace("\\n", "\n")
    lines = [line.strip() for line in normalized_context.splitlines() if line.strip()]
    utterances: List[Dict[str, Any]] = []
    current_speaker = "Narration"
    current_parts: List[str] = []

    def _flush() -> None:
        nonlocal current_parts
        text = " ".join(current_parts).strip()
        if text:
            utterances.append(
                {
                    "utterance_id": len(utterances),
                    "speaker": current_speaker,
                    "text": text,
                }
            )
        current_parts = []

    for raw_line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z0-9_ .'-]{0,40}):\s*(.*)$", raw_line)
        if match:
            _flush()
            current_speaker = match.group(1).strip()
            tail = match.group(2).strip()
            current_parts = [tail] if tail else []
            continue
        current_parts.append(raw_line)
    _flush()

    if utterances:
        return utterances
    compact = " ".join(lines).strip()
    if not compact:
        return []
    return [{"utterance_id": 0, "speaker": "Narration", "text": compact}]


def _build_flat_dialogue_record(group_rows: Sequence[Dict[str, Any]], fallback_id: str) -> Optional[DialogueRecord]:
    if not group_rows:
        return None
    context = str(group_rows[0].get("context") or "").strip()
    utterances = _split_flat_context_to_utterances(context)
    if not utterances:
        return None
    dialogue_id = str(group_rows[0].get("id") or group_rows[0].get("dialogue_id") or fallback_id)
    qas: List[DialogueQaInstance] = []
    for index, row in enumerate(group_rows):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        qas.append(
            DialogueQaInstance(
                dialogue_id=dialogue_id,
                qa_id=str(row.get("qa_id") or row.get("id") or f"{dialogue_id}_q{index}"),
                question=question,
                answers=(answer,),
                gold_utterance_ids=(),
            )
        )
    if not qas:
        return None
    return DialogueRecord(dialogue_id=dialogue_id, utterances=tuple(utterances), qas=tuple(qas))


def _extract_answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if not isinstance(answer, dict):
        return ""
    for key in ("answer_text", "text", "answer", "span_text"):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_answer_utterance_ids(answer: Any) -> List[int]:
    ids: List[int] = []
    if isinstance(answer, dict):
        for key in ("utterance_id", "utterance_idx", "speaker_id"):
            value = answer.get(key)
            if value is None:
                continue
            try:
                ids.append(int(value))
            except Exception:
                pass
        for key in ("utterance_ids", "utterance_idxs"):
            value = answer.get(key)
            if isinstance(value, list):
                for item in value:
                    try:
                        ids.append(int(item))
                    except Exception:
                        continue
    return ids


@dataclass(frozen=True)
class DialogueQaInstance:
    dialogue_id: str
    qa_id: str
    question: str
    answers: Tuple[str, ...]
    gold_utterance_ids: Tuple[int, ...]


@dataclass(frozen=True)
class DialogueRecord:
    dialogue_id: str
    utterances: Tuple[Dict[str, Any], ...]
    qas: Tuple[DialogueQaInstance, ...]


def _parse_dialogue_units(raw_record: Dict[str, Any], fallback_id: str) -> List[DialogueRecord]:
    records: List[DialogueRecord] = []
    flat_question = str(raw_record.get("question") or "").strip()
    flat_context = str(raw_record.get("context") or "").strip()
    flat_answer = str(raw_record.get("answer") or "").strip()
    if flat_question and flat_context and flat_answer:
        utterances = _split_flat_context_to_utterances(flat_context)
        if not utterances:
            return []
        dialogue_id = str(raw_record.get("id") or raw_record.get("dialogue_id") or fallback_id)
        return [
            DialogueRecord(
                dialogue_id=dialogue_id,
                utterances=tuple(utterances),
                qas=(
                    DialogueQaInstance(
                        dialogue_id=dialogue_id,
                        qa_id=f"{dialogue_id}_q0",
                        question=flat_question,
                        answers=(flat_answer,),
                        gold_utterance_ids=(),
                    ),
                ),
            )
        ]

    paragraphs = raw_record.get("paragraphs")
    dialogue_title = str(raw_record.get("id") or raw_record.get("title") or raw_record.get("dialogue_id") or fallback_id)

    if isinstance(paragraphs, list):
        units = paragraphs
    else:
        units = [raw_record]

    for unit_index, unit in enumerate(units):
        if not isinstance(unit, dict):
            continue
        utterance_payload = (
            unit.get("utterances")
            or unit.get("dialogue")
            or unit.get("dialog")
            or unit.get("conversation")
            or unit.get("context")
            or []
        )
        if isinstance(utterance_payload, str):
            utterance_payload = [{"speaker": "Speaker", "utterance": utterance_payload}]
        if not isinstance(utterance_payload, list):
            continue

        utterances: List[Dict[str, Any]] = []
        for index, item in enumerate(utterance_payload):
            parsed = _normalize_utterance(item, index)
            if parsed is not None:
                utterances.append(parsed)
        if not utterances:
            continue

        qas_payload = unit.get("qas") or unit.get("qa") or unit.get("questions") or []
        if not isinstance(qas_payload, list):
            continue

        dialogue_id = str(unit.get("id") or unit.get("dialogue_id") or f"{dialogue_title}_{unit_index}")
        qas: List[DialogueQaInstance] = []
        for qa_index, qa in enumerate(qas_payload):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or qa.get("query") or "").strip()
            if not question:
                continue
            qa_id = str(qa.get("id") or qa.get("qid") or f"{dialogue_id}_q{qa_index}")
            answers_payload = qa.get("answers") or qa.get("answer") or []
            if isinstance(answers_payload, dict):
                answers_payload = [answers_payload]
            if isinstance(answers_payload, str):
                answers_payload = [answers_payload]
            if not isinstance(answers_payload, list):
                continue

            answers: List[str] = []
            gold_utterance_ids: List[int] = []
            for answer in answers_payload:
                text = _extract_answer_text(answer)
                if text:
                    answers.append(text)
                gold_utterance_ids.extend(_extract_answer_utterance_ids(answer))
            answers = list(dict.fromkeys(answers))
            gold_utterance_ids = list(dict.fromkeys(gold_utterance_ids))
            if not answers:
                continue
            qas.append(
                DialogueQaInstance(
                    dialogue_id=dialogue_id,
                    qa_id=qa_id,
                    question=question,
                    answers=tuple(answers),
                    gold_utterance_ids=tuple(gold_utterance_ids),
                )
            )
        if qas:
            records.append(DialogueRecord(dialogue_id=dialogue_id, utterances=tuple(utterances), qas=tuple(qas)))
    return records


def load_friendsqa_dialogues(data_path: str, split: str = "validation") -> List[DialogueRecord]:
    records: List[DialogueRecord] = []
    flat_groups: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in _resolve_data_files(data_path, split):
        for index, raw_record in enumerate(_iter_json_records(file_path)):
            fallback_id = f"{file_path.stem}_{index}"
            if (
                isinstance(raw_record, dict)
                and str(raw_record.get("question") or "").strip()
                and str(raw_record.get("context") or "").strip()
                and str(raw_record.get("answer") or "").strip()
                and not raw_record.get("paragraphs")
            ):
                flat_groups.setdefault(_context_signature(str(raw_record.get("context") or "")), []).append(dict(raw_record))
                continue
            records.extend(_parse_dialogue_units(raw_record, fallback_id))
    for signature, rows in flat_groups.items():
        grouped = _build_flat_dialogue_record(rows, f"flat_{signature[:10]}")
        if grouped is not None:
            records.append(grouped)
    return records


def load_friendsqa_dialogues_from_hf(dataset_name: str, split: str) -> List[DialogueRecord]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    records: List[DialogueRecord] = []
    flat_groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, item in enumerate(ds):
        if isinstance(item, dict):
            row = dict(item)
            if (
                str(row.get("question") or "").strip()
                and str(row.get("context") or "").strip()
                and str(row.get("answer") or "").strip()
                and not row.get("paragraphs")
            ):
                flat_groups.setdefault(_context_signature(str(row.get("context") or "")), []).append(row)
                continue
            records.extend(_parse_dialogue_units(row, f"{dataset_name}_{index}"))
    for signature, rows in flat_groups.items():
        grouped = _build_flat_dialogue_record(rows, f"{dataset_name}_{signature[:10]}")
        if grouped is not None:
            records.append(grouped)
    return records


def _build_dialogue_chunks(
    utterances: Sequence[Dict[str, Any]],
    *,
    window_size: int,
    window_stride: int,
    dialogue_id: str,
) -> List[Dict[str, Any]]:
    if not utterances:
        return []
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(utterances):
        end = min(len(utterances), start + window_size)
        window = utterances[start:end]
        text = "\n".join(f"{item['speaker']}: {item['text']}" for item in window).strip()
        if text:
            chunks.append(
                {
                    "chunk_id": f"{dialogue_id}_chunk_{chunk_index}",
                    "dialogue_id": dialogue_id,
                    "span_start": start,
                    "span_end": end - 1,
                    "utterance_ids": [int(item["utterance_id"]) for item in window],
                    "text": text,
                }
            )
            chunk_index += 1
        if end >= len(utterances):
            break
        start += window_stride
    return chunks


def compute_support_hit(retrieved_pages: Sequence[Dict[str, Any]], gold_utterance_ids: Sequence[int]) -> Optional[bool]:
    if not gold_utterance_ids:
        return None
    gold_set = {int(item) for item in gold_utterance_ids}
    for page in retrieved_pages or []:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        utterance_ids = metadata.get("utterance_ids")
        if not isinstance(utterance_ids, list):
            continue
        page_set = set()
        for item in utterance_ids:
            try:
                page_set.add(int(item))
            except Exception:
                continue
        if gold_set & page_set:
                return True
    return False


def _retrieval_query_kind_for_question(question: str) -> str:
    return infer_retrieval_query_kind(question, task_hint="dialogue_qa")


def _context_candidates_from_pages(pages: Sequence[Dict[str, Any]]) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for page in pages or []:
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if not raw:
            continue
        for line in raw.replace("\\n", "\n").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Dialogue:"):
                continue
            if line.startswith("Q:"):
                line = line[2:].strip()
            if line.startswith("A:"):
                line = line[2:].strip()
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


def _question_expected_shape(question: str) -> str:
    lowered = str(question or "").strip().lower()
    if lowered.startswith("who"):
        return "person"
    if lowered.startswith("where"):
        return "location"
    if lowered.startswith("when"):
        return "time"
    if lowered.startswith("why"):
        return "reason"
    if lowered.startswith("how"):
        return "manner"
    return "fact"


def _expand_candidate_fragments(candidates: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        parts = [candidate]
        parts.extend(
            fragment.strip(" []()\"'")
            for fragment in re.split(r"[.;!?]|\s+-\s+|\s+but\s+|\s+because\s+", candidate)
            if fragment.strip()
        )
        parts.extend(
            match.group(1).strip()
            for match in re.finditer(r"['\"]([^'\"]{2,160})['\"]", candidate)
        )
        for part in parts:
            normalized = " ".join(str(part).split()).strip()
            if not normalized:
                continue
            key = normalize_answer(normalized)
            if not key or key in seen:
                continue
            seen.add(key)
            expanded.append(normalized)
    return expanded


def _align_prediction_to_context(question: str, prediction: str, pages: Sequence[Dict[str, Any]]) -> str:
    text = str(prediction or "").strip()
    if not text or "unknown" in text.lower():
        return text
    candidates = _expand_candidate_fragments(_context_candidates_from_pages(pages))
    if not candidates:
        return text
    best_candidate = text
    best_score = 0.0
    expected_shape = _question_expected_shape(question)
    for candidate in candidates:
        score = token_f1_score(text, candidate)
        if normalize_answer(text) and normalize_answer(text) in normalize_answer(candidate):
            score += 0.15
        token_count = len(candidate.split())
        if token_count <= 8:
            score += 0.03
        elif token_count >= 24:
            score -= 0.04
        if expected_shape == "person" and re.fullmatch(r"[A-Z][A-Za-z'`.-]*(?:\s+[A-Z][A-Za-z'`.-]*){0,3}", candidate):
            score += 0.05
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate if best_score >= 0.35 else text


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
        FRIENDSQA_PROMPT
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
    return _align_prediction_to_context(question, answer, pages)


def _build_report(args: argparse.Namespace, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    qa_rows = [item for item in results if item.get("qa_id")]
    support_rows = [item for item in qa_rows if item.get("support_hit") is not None]
    return {
        "dataset": str(args.dataset or "FriendsQA"),
        "split": str(args.split),
        "model": str(args.model_name),
        "llm_model": str(args.llm_model),
        "embedding_model": str(args.embedding_model),
        "dialogues": sum(1 for item in results if item.get("qa_id") is None or item.get("qa_id")),
        "questions": len(qa_rows),
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
    parser = argparse.ArgumentParser(description="Run FriendsQA with Remem (MemoryOS).")
    parser.add_argument("--data-path", default="", help="Path to a local FriendsQA JSON/JSONL file or directory.")
    parser.add_argument("--dataset", default="", help="Optional Hugging Face dataset id when using `datasets.load_dataset`.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-dialogues", type=int, default=0, help="Limit number of dialogues (0 = all).")
    parser.add_argument("--max-questions", type=int, default=0, help="Limit total number of QA pairs (0 = all).")
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
    parser.add_argument("--data-storage-root", default=str(REPO_ROOT / "fast_eval_storage" / "friendsqa_remem"))
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

    if not str(args.data_path).strip() and not str(args.dataset).strip():
        raise SystemExit("Specify either --data-path or --dataset.")

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
        dialogues = load_friendsqa_dialogues_from_hf(str(args.dataset).strip(), split=str(args.split))
    else:
        dialogues = load_friendsqa_dialogues(str(args.data_path), split=str(args.split))

    run_id = f"{int(time.time())}"
    tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"friendsqa_{args.model_name}{tag}_{run_id}.csv"
    out_json = out_dir / f"friendsqa_{args.model_name}{tag}_{run_id}.json"

    data_storage_path = str(Path(args.data_storage_root).resolve() / args.model_name / f"{args.split}_{run_id}")
    Path(data_storage_path).mkdir(parents=True, exist_ok=True)

    remem_cfg = RememConfig(
        user_id=f"friendsqa_{args.model_name}_{run_id}",
        assistant_id="friendsqa_runner",
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
        chunks = _build_dialogue_chunks(
            dialogue.utterances,
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            dialogue_id=dialogue.dialogue_id,
        )
        if not chunks:
            continue

        ingest_started = time.time()
        ingest_error = ""
        for chunk in chunks:
            try:
                memoryos.add_memory(
                    f"[Dialogue: {dialogue.dialogue_id} | Turns: {chunk['span_start']}-{chunk['span_end']}]\n{chunk['text']}",
                    "",
                    meta_data={
                        "doc_id": chunk["chunk_id"],
                        "page_id": chunk["chunk_id"],
                        "dialogue_id": dialogue.dialogue_id,
                        "span_start": chunk["span_start"],
                        "span_end": chunk["span_end"],
                        "utterance_ids": chunk["utterance_ids"],
                        "topic": "FriendsQA Dialogue Chunk",
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
                extra_signals={"benchmark": "friendsqa"},
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
                    benchmark="friendsqa",
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
            support_hit = compute_support_hit(pages, qa.gold_utterance_ids)
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
                report = _build_report(args, results)
                _write_outputs(out_csv, out_json, report, results)

        dialogues_processed += 1
        if args.max_questions and questions_processed >= int(args.max_questions):
            break

    memoryos.shutdown()

    report = _build_report(args, results)
    report["dialogues"] = dialogues_processed
    report["questions"] = questions_processed
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
