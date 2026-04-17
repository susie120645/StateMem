from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_ANSWER_WEIGHT = 0.65
GROUNDING_BUCKETS = (
    "correct_and_grounded",
    "correct_but_bad_citation",
    "wrong_but_grounded",
    "wrong_and_ungrounded",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _round4(value: float) -> float:
    return round(float(value), 4)


def unique_strings(values: Optional[Iterable[Any]]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalize_evidence_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_em(prediction: Any, answer: Any) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(answer) else 0.0


def compute_f1(prediction: Any, answer: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(count, gold_counts[token]) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_answer_scores(
    prediction: Any,
    gold_answer: Any,
    semantic_score: Optional[float] = None,
    answer_em: Optional[float] = None,
    answer_f1: Optional[float] = None,
) -> Dict[str, Any]:
    answer_em = compute_em(prediction, gold_answer) if answer_em is None else _clamp(float(answer_em))
    answer_f1 = compute_f1(prediction, gold_answer) if answer_f1 is None else _clamp(float(answer_f1))
    semantic_value = answer_f1 if semantic_score is None else _clamp(float(semantic_score))
    answer_score = max(answer_em, (answer_f1 * 0.5) + (semantic_value * 0.5))
    return {
        "answer_em": _round4(answer_em),
        "answer_f1": _round4(answer_f1),
        "answer_semantic": _round4(semantic_value),
        "answer_score": _round4(answer_score),
        "answer_semantic_method": "f1_fallback" if semantic_score is None else "provided",
    }


def build_citation_scores(
    gold_evidence_ids: Optional[Sequence[Any]],
    predicted_evidence_ids: Optional[Sequence[Any]],
    citation_support_precision: float = 0.0,
) -> Dict[str, Any]:
    gold_unique = unique_strings(gold_evidence_ids)
    predicted_unique = unique_strings(predicted_evidence_ids)

    gold_normalized = {normalize_evidence_id(value) for value in gold_unique if normalize_evidence_id(value)}
    predicted_normalized = {normalize_evidence_id(value) for value in predicted_unique if normalize_evidence_id(value)}
    overlap_normalized = gold_normalized & predicted_normalized
    matched_ids = [value for value in predicted_unique if normalize_evidence_id(value) in overlap_normalized]

    gold_precision = len(overlap_normalized) / len(predicted_normalized) if predicted_normalized else 0.0
    if gold_normalized:
        gold_recall = len(overlap_normalized) / len(gold_normalized)
    else:
        gold_recall = 1.0 if not predicted_normalized else 0.0

    support_precision = _clamp(citation_support_precision)
    citation_score = (
        (0.5 * support_precision)
        + (0.25 * gold_precision)
        + (0.25 * gold_recall)
    )
    return {
        "citation_support_precision": _round4(support_precision),
        "citation_gold_precision": _round4(gold_precision),
        "citation_gold_recall": _round4(gold_recall),
        "citation_score": _round4(citation_score),
        "matched_evidence_ids": matched_ids,
    }


def classify_grounding(answer_score: float, citation_score: float) -> str:
    normalized_answer = _clamp(answer_score)
    normalized_citation = _clamp(citation_score)
    if normalized_answer >= 0.8 and normalized_citation >= 0.6:
        return "correct_and_grounded"
    if normalized_answer >= 0.8:
        return "correct_but_bad_citation"
    if normalized_citation >= 0.6:
        return "wrong_but_grounded"
    return "wrong_and_ungrounded"


def build_evaluation_result(
    gold_answer: Any,
    predicted_answer: Any,
    gold_evidence_ids: Optional[Sequence[Any]],
    predicted_evidence_ids: Optional[Sequence[Any]],
    citation_support_precision: float = 0.0,
    citation_ids: Optional[Sequence[Any]] = None,
    hit_ids: Optional[Sequence[Any]] = None,
    semantic_score: Optional[float] = None,
    answer_em: Optional[float] = None,
    answer_f1: Optional[float] = None,
    answer_weight: float = DEFAULT_ANSWER_WEIGHT,
) -> Dict[str, Any]:
    answer_scores = build_answer_scores(
        predicted_answer,
        gold_answer,
        semantic_score=semantic_score,
        answer_em=answer_em,
        answer_f1=answer_f1,
    )
    citation_scores = build_citation_scores(
        gold_evidence_ids=gold_evidence_ids,
        predicted_evidence_ids=predicted_evidence_ids,
        citation_support_precision=citation_support_precision,
    )
    answer_weight = _clamp(answer_weight)
    citation_weight = _clamp(1.0 - answer_weight)
    final_score = (
        (answer_weight * float(answer_scores["answer_score"]))
        + (citation_weight * float(citation_scores["citation_score"]))
    )
    bucket = classify_grounding(answer_scores["answer_score"], citation_scores["citation_score"])

    return {
        "gold": {
            "answer": str(gold_answer or ""),
            "evidence_ids": unique_strings(gold_evidence_ids),
        },
        "pred": {
            "answer": str(predicted_answer or ""),
            "citations": unique_strings(citation_ids),
            "evidence_ids": unique_strings(predicted_evidence_ids),
            "hit_ids": unique_strings(hit_ids),
        },
        "scores": {
            **answer_scores,
            **citation_scores,
            "final": _round4(final_score),
            "bucket": bucket,
            "answer_weight": _round4(answer_weight),
            "citation_weight": _round4(citation_weight),
        },
    }


def support_precision_from_nli_preview(preview: Optional[Dict[str, Any]]) -> float:
    preview = preview or {}
    entailment = len(preview.get("entailment", []) or [])
    contradiction = len(preview.get("contradiction", []) or [])
    neutral = len(preview.get("neutral", []) or [])
    evaluated = entailment + contradiction + neutral
    if evaluated == 0:
        return 0.0
    return _round4(entailment / float(evaluated))


def summarize_score_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = (
        "answer_em",
        "answer_f1",
        "answer_semantic",
        "answer_score",
        "citation_support_precision",
        "citation_gold_precision",
        "citation_gold_recall",
        "citation_score",
        "final",
    )
    summary = {key: 0.0 for key in metric_keys}
    bucket_counts = {bucket: 0 for bucket in GROUNDING_BUCKETS}
    if not records:
        summary["judgement_buckets"] = bucket_counts
        return summary

    for record in records:
        scores = record.get("scores", {}) if isinstance(record, dict) else {}
        for key in metric_keys:
            value = scores.get(key, 0.0)
            if isinstance(value, (int, float)):
                summary[key] += float(value)
        bucket = str(scores.get("bucket", "")).strip()
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for key in metric_keys:
        summary[key] = _round4(summary[key] / len(records))
    summary["judgement_buckets"] = bucket_counts
    return summary
