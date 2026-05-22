#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _label(text: Any) -> str:
    match = re.search(r"\(([A-Ea-e])\)", str(text or ""))
    return match.group(1).upper() if match else ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _support(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("choice_support") or {}
    return payload if isinstance(payload, dict) else {}


def _has_reason(record: Dict[str, Any], reason: str) -> bool:
    return reason in (_support(record).get("reason_codes") or [])


def _priority_score(record: Dict[str, Any]) -> float:
    support = _support(record)
    margin = _safe_float(support.get("support_margin"))
    time_overlap = _safe_float(support.get("question_time_overlap"))
    entity_overlap = _safe_float(support.get("question_entity_overlap"))
    family_share = _safe_float(support.get("top_family_share"))
    relation_overlap = list(support.get("relation_overlap") or [])

    score = 0.0
    score += 2.0 * min(1.0, margin)
    score += 1.0 * min(1.0, time_overlap)
    score += 1.0 * min(1.0, entity_overlap)
    score += 0.5 * min(1.0, family_share)
    score += 0.25 * min(3, len(relation_overlap))

    if _has_reason(record, "strong_margin"):
        score += 0.5
    if _has_reason(record, "question_relation_aligned"):
        score += 0.25
    if _has_reason(record, "support_source_single_family"):
        score += 0.15
    if {"dating", "family", "roommate"} <= set(relation_overlap):
        score += 0.35
    return round(score, 4)


def _priority_tier(record: Dict[str, Any]) -> str:
    support = _support(record)
    margin = _safe_float(support.get("support_margin"))
    time_overlap = _safe_float(support.get("question_time_overlap"))
    entity_overlap = _safe_float(support.get("question_entity_overlap"))
    relation_overlap = set(support.get("relation_overlap") or [])

    if (
        margin >= 0.8
        and time_overlap >= 0.5
        and entity_overlap >= 0.5
        and len(relation_overlap) >= 2
    ):
        return "P1"
    if (
        margin >= 0.6
        and time_overlap >= 0.5
        and entity_overlap >= 0.45
        and len(relation_overlap) >= 1
    ):
        return "P2"
    return "P3"


def select_strict_relation_candidates(rows: List[Dict[str, Any]], margin_threshold: float) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if _label(row.get("pred")) == "E"
        and row.get("result") == "Wrong"
        and _has_reason(row, "best_choice_non_abstain")
        and _has_reason(row, "question_time_anchor_aligned")
        and _has_reason(row, "question_entity_anchor_aligned")
        and _has_reason(row, "question_relation_aligned")
        and _has_reason(row, "support_source_single_family")
        and _safe_float(_support(row).get("support_margin")) >= margin_threshold
    ]


def build_review_queue(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for row in rows:
        support = _support(row)
        priority_tier = _priority_tier(row)
        priority_score = _priority_score(row)
        queue.append(
            {
                "queue_reason": "strict_relation_non_abstain_support",
                "priority_tier": priority_tier,
                "priority_score": priority_score,
                "episode": row.get("episode", ""),
                "session": row.get("session", ""),
                "date": row.get("date", ""),
                "qtype": row.get("qtype", ""),
                "question": row.get("question", ""),
                "true_label": row.get("true_label", ""),
                "current_pred": row.get("pred", ""),
                "support_best_choice": support.get("best_choice", ""),
                "support_margin": support.get("support_margin", 0.0),
                "best_choice_score": support.get("best_choice_score", 0.0),
                "best_non_abstain_choice": support.get("best_non_abstain_choice", ""),
                "best_non_abstain_score": support.get("best_non_abstain_score", 0.0),
                "question_time_overlap": support.get("question_time_overlap", 0.0),
                "question_entity_overlap": support.get("question_entity_overlap", 0.0),
                "relation_overlap": support.get("relation_overlap", []),
                "relation_missing": support.get("relation_missing", []),
                "reason_codes": support.get("reason_codes", []),
                "top_family_share": support.get("top_family_share", 0.0),
                "family_ids": support.get("family_ids", []),
                "hit_ids": row.get("hit_ids", []),
                "source_stats": row.get("source_stats", {}),
            }
        )
    queue.sort(
        key=lambda item: (
            str(item.get("priority_tier", "")),
            -_safe_float(item.get("priority_score")),
            -_safe_float(item.get("support_margin")),
            -_safe_float(item.get("question_time_overlap")),
            -_safe_float(item.get("question_entity_overlap")),
            str(item.get("episode", "")),
            _safe_float(item.get("session")),
        )
    )
    return queue


def write_review_queue(queue: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for item in queue:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    output_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_review_queue_by_tier(queue: List[Dict[str, Any]], *, priority_tier: str = "") -> List[Dict[str, Any]]:
    normalized = str(priority_tier or "").strip().upper()
    if not normalized:
        return list(queue)
    return [item for item in queue if str(item.get("priority_tier", "")).strip().upper() == normalized]


def filter_review_queue_by_relation_signature(
    queue: List[Dict[str, Any]],
    *,
    required_relations: Iterable[str],
) -> List[Dict[str, Any]]:
    required = {str(item).strip().lower() for item in required_relations if str(item).strip()}
    if not required:
        return list(queue)
    filtered: List[Dict[str, Any]] = []
    for item in queue:
        overlap = {str(rel).strip().lower() for rel in (item.get("relation_overlap") or []) if str(rel).strip()}
        if required <= overlap:
            filtered.append(item)
    return filtered


def classify_review_item(item: Dict[str, Any]) -> str:
    question = str(item.get("question", "")).lower()
    relation_overlap = {str(rel).strip().lower() for rel in (item.get("relation_overlap") or []) if str(rel).strip()}
    current_pred = _label(item.get("current_pred"))
    true_label = _label(item.get("true_label"))
    support_best_choice = _label(item.get("support_best_choice"))

    if (
        "went out with that person" in question
        and "who was bunking with that person's roommate" in question
        and {"dating", "family"} <= relation_overlap
    ):
        return "relation_template_mismatch"

    if "who was the roommate of the client" in question or "who was the roommate of the client by the time" in question:
        return "relation_reference_ambiguity"

    if (
        current_pred == "E"
        and true_label in {"A", "B", "C", "D"}
        and support_best_choice == true_label
    ):
        return "clean_answer_side_failure"

    return "other"


def relabel_review_queue(queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relabeled: List[Dict[str, Any]] = []
    for item in queue:
        relabeled.append(
            {
                **item,
                "diagnosis_bucket": classify_review_item(item),
            }
        )
    return relabeled


def extract_clean_answer_side_failures(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for row in rows:
        support = _support(row)
        candidate = {
            "episode": row.get("episode", ""),
            "session": row.get("session", ""),
            "date": row.get("date", ""),
            "qtype": row.get("qtype", ""),
            "question": row.get("question", ""),
            "true_label": row.get("true_label", ""),
            "current_pred": row.get("pred", ""),
            "support_best_choice": support.get("best_choice", ""),
            "support_margin": support.get("support_margin", 0.0),
            "citations": support.get("citations", []),
            "reason_codes": support.get("reason_codes", []),
            "options": row.get("options", []),
            "hit_ids": row.get("hit_ids", []),
            "relation_overlap": support.get("relation_overlap", []),
        }
        if classify_review_item(candidate) != "clean_answer_side_failure":
            continue
        extracted.append(
            {
                "diagnosis_bucket": "clean_answer_side_failure",
                **candidate,
            }
        )
    return extracted


def _subset_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    qtypes = Counter(r.get("qtype", "") for r in rows)
    reasons = Counter()
    relation_overlap = Counter()
    relation_missing = Counter()
    for row in rows:
        support = _support(row)
        reasons.update(support.get("reason_codes") or [])
        relation_overlap.update(support.get("relation_overlap") or [])
        relation_missing.update(support.get("relation_missing") or [])

    return {
        "count": len(rows),
        "qtypes": dict(qtypes),
        "reason_counts": dict(reasons.most_common()),
        "relation_overlap_counts": dict(relation_overlap.most_common()),
        "relation_missing_counts": dict(relation_missing.most_common()),
        "avg_support_margin": round(_avg(_safe_float(_support(r).get("support_margin")) for r in rows), 4),
        "avg_best_choice_score": round(_avg(_safe_float(_support(r).get("best_choice_score")) for r in rows), 4),
        "avg_best_non_abstain_score": round(_avg(_safe_float(_support(r).get("best_non_abstain_score")) for r in rows), 4),
        "avg_question_time_overlap": round(_avg(_safe_float(_support(r).get("question_time_overlap")) for r in rows), 4),
        "avg_question_entity_overlap": round(_avg(_safe_float(_support(r).get("question_entity_overlap")) for r in rows), 4),
        "avg_top_family_share": round(_avg(_safe_float(_support(r).get("top_family_share")) for r in rows), 4),
    }


def analyze_report(report_path: Path, margin_threshold: float) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(payload.get("results", []) or [])

    pred_e = [row for row in rows if _label(row.get("pred")) == "E"]
    pred_non_e = [row for row in rows if _label(row.get("pred")) in {"A", "B", "C", "D"}]
    wrong_pred_e = [row for row in pred_e if row.get("result") == "Wrong"]

    non_abstain_alt = [
        row for row in pred_e if _label(_support(row).get("best_choice")) in {"A", "B", "C", "D"}
    ]
    recoverable = [
        row
        for row in non_abstain_alt
        if _label(_support(row).get("best_choice")) == _label(row.get("true_label"))
    ]
    false_alt = [row for row in non_abstain_alt if row not in recoverable]

    support_e_for_pred_non_e = [
        row for row in pred_non_e if _label(_support(row).get("best_choice")) == "E"
    ]
    corrective_e = [
        row for row in support_e_for_pred_non_e if _label(row.get("true_label")) == "E"
    ]

    strict_relation_candidates = select_strict_relation_candidates(rows, margin_threshold)
    strict_relation_correct = [
        row
        for row in strict_relation_candidates
        if _label(_support(row).get("best_choice")) == _label(row.get("true_label"))
    ]
    strict_relation_priority_counts = Counter(_priority_tier(row) for row in strict_relation_candidates)

    return {
        "report_path": str(report_path),
        "accuracy": payload.get("accuracy"),
        "total": len(rows),
        "pred_e_count": len(pred_e),
        "wrong_pred_e_count": len(wrong_pred_e),
        "non_abstain_alt_count": len(non_abstain_alt),
        "recoverable_count": len(recoverable),
        "false_alt_count": len(false_alt),
        "alt_precision_on_pred_e": round(len(recoverable) / len(non_abstain_alt), 4) if non_abstain_alt else 0.0,
        "alt_coverage_on_wrong_pred_e": round(len(recoverable) / len(wrong_pred_e), 4) if wrong_pred_e else 0.0,
        "support_e_for_pred_non_e_count": len(support_e_for_pred_non_e),
        "corrective_e_count": len(corrective_e),
        "support_e_precision": round(len(corrective_e) / len(support_e_for_pred_non_e), 4) if support_e_for_pred_non_e else 0.0,
        "strict_relation_margin_threshold": margin_threshold,
        "strict_relation_candidate_count": len(strict_relation_candidates),
        "strict_relation_correct_count": len(strict_relation_correct),
        "strict_relation_precision": round(len(strict_relation_correct) / len(strict_relation_candidates), 4) if strict_relation_candidates else 0.0,
        "strict_relation_priority_counts": dict(strict_relation_priority_counts),
        "recoverable_metrics": _subset_metrics(recoverable),
        "false_alt_metrics": _subset_metrics(false_alt),
        "strict_relation_examples": [
            {
                "priority_tier": _priority_tier(row),
                "priority_score": _priority_score(row),
                "qtype": row.get("qtype", ""),
                "true_label": row.get("true_label", ""),
                "pred": row.get("pred", ""),
                "support_best_choice": _support(row).get("best_choice", ""),
                "support_margin": _support(row).get("support_margin", 0.0),
                "question_time_overlap": _support(row).get("question_time_overlap", 0.0),
                "question_entity_overlap": _support(row).get("question_entity_overlap", 0.0),
                "relation_overlap": _support(row).get("relation_overlap", []),
                "relation_missing": _support(row).get("relation_missing", []),
                "question": row.get("question", ""),
            }
            for row in strict_relation_candidates[:20]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze baseline-safe DialSim choice-support logs.")
    parser.add_argument("report", help="Path to a DialSim report JSON file.")
    parser.add_argument("--margin-threshold", type=float, default=0.6)
    parser.add_argument(
        "--export-review-queue",
        default="",
        help="Optional path to export the strict relation review queue (.json or .jsonl).",
    )
    parser.add_argument(
        "--export-p1-only",
        default="",
        help="Optional path to export only the P1 subset of the strict relation review queue (.json or .jsonl).",
    )
    parser.add_argument(
        "--export-relation-signature",
        default="",
        help="Optional path to export only rows whose relation_overlap contains all relations from --relation-signature.",
    )
    parser.add_argument(
        "--relation-signature",
        default="dating,family,roommate",
        help="Comma-separated relation signature for --export-relation-signature.",
    )
    parser.add_argument(
        "--export-relabeled-queue",
        default="",
        help="Optional path to export the strict relation queue after automatic bucket relabeling (.json or .jsonl).",
    )
    parser.add_argument(
        "--export-clean-answer-side",
        default="",
        help="Optional path to export clean answer-side failures from the full report (.json or .jsonl).",
    )
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    summary = analyze_report(report_path, float(args.margin_threshold))
    queue: List[Dict[str, Any]] = []
    if (
        str(args.export_review_queue).strip()
        or str(args.export_p1_only).strip()
        or str(args.export_relation_signature).strip()
        or str(args.export_relabeled_queue).strip()
    ):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = list(payload.get("results", []) or [])
        queue = build_review_queue(select_strict_relation_candidates(rows, float(args.margin_threshold)))
    else:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = list(payload.get("results", []) or [])
    if str(args.export_review_queue).strip():
        output_path = Path(args.export_review_queue).expanduser().resolve()
        write_review_queue(queue, output_path)
        summary["review_queue_path"] = str(output_path)
        summary["review_queue_count"] = len(queue)
    if str(args.export_p1_only).strip():
        p1_queue = filter_review_queue_by_tier(queue, priority_tier="P1")
        output_path = Path(args.export_p1_only).expanduser().resolve()
        write_review_queue(p1_queue, output_path)
        summary["review_queue_p1_path"] = str(output_path)
        summary["review_queue_p1_count"] = len(p1_queue)
    if str(args.export_relation_signature).strip():
        required_relations = [item.strip() for item in str(args.relation_signature).split(",") if item.strip()]
        relation_queue = filter_review_queue_by_relation_signature(
            queue,
            required_relations=required_relations,
        )
        output_path = Path(args.export_relation_signature).expanduser().resolve()
        write_review_queue(relation_queue, output_path)
        summary["review_queue_relation_signature"] = required_relations
        summary["review_queue_relation_path"] = str(output_path)
        summary["review_queue_relation_count"] = len(relation_queue)
    if str(args.export_relabeled_queue).strip():
        relabeled_queue = relabel_review_queue(queue)
        output_path = Path(args.export_relabeled_queue).expanduser().resolve()
        write_review_queue(relabeled_queue, output_path)
        summary["relabeled_queue_path"] = str(output_path)
        summary["relabeled_queue_count"] = len(relabeled_queue)
        bucket_counts = Counter(item.get("diagnosis_bucket", "other") for item in relabeled_queue)
        summary["relabeled_queue_bucket_counts"] = dict(bucket_counts)
    if str(args.export_clean_answer_side).strip():
        clean_queue = extract_clean_answer_side_failures(rows)
        output_path = Path(args.export_clean_answer_side).expanduser().resolve()
        write_review_queue(clean_queue, output_path)
        summary["clean_answer_side_path"] = str(output_path)
        summary["clean_answer_side_count"] = len(clean_queue)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
