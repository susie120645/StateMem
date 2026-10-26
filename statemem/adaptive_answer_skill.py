from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _page_text(page: Dict[str, Any]) -> str:
    return str(page.get("raw_content") or page.get("summary") or "").strip()


def _page_score(page: Dict[str, Any]) -> float:
    return _safe_float(page.get("final_score", page.get("score", 0.0)), 0.0)


def _has_options(query_text: str) -> bool:
    text = str(query_text or "")
    lowered = text.lower()
    return "options:" in lowered or all(marker in text for marker in ("(A)", "(B)", "(C)"))


def _answer_looks_abstained(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    return text in {
        "i don't know.",
        "i do not know.",
        "don't know",
        "do not know",
        "unknown",
        "not enough information",
    } or "don't know" in text


@dataclass(frozen=True)
class AdaptiveAnswerState:
    query_text: str
    query_type: str
    task_hint: str
    retrieval_mode: str
    answer_mode: str
    answer_format: str
    routing_confidence: float
    retrieved_count: int
    source_stats: Dict[str, int]
    top_score: float
    score_gap: float
    avg_score: float
    token_budget: int
    context_char_count: int
    has_options: bool
    evidence_density: float
    uncertainty_flags: Dict[str, bool]

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveAnswerAction:
    answer_strategy: str
    retry_strategy: str
    abstain_policy: str
    format_policy: str

    def to_payload(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveAnswerOutcome:
    metric_value: float
    format_ok: bool
    support_ok: Optional[bool]
    abstained: bool
    latency_ms: float

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


def build_adaptive_answer_state(
    query: str,
    retrieval_results: Dict[str, Any],
    *,
    trace_context: Optional[Dict[str, Any]] = None,
    retrieval_mode: str = "",
    answer_mode: str = "",
    answer_format: str = "",
) -> AdaptiveAnswerState:
    pages = list(retrieval_results.get("retrieved_pages", []) or [])
    scores = [_page_score(page) for page in pages]
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    score_gap = max(0.0, top_score - second_score)
    context_char_count = sum(len(_page_text(page)) for page in pages)
    source_stats = {
        str(key): _safe_int(value, 0)
        for key, value in (retrieval_results.get("source_stats", {}) or {}).items()
        if isinstance(key, str)
    }
    query_type = str((trace_context or {}).get("query_type", "") or "")
    task_hint = str((trace_context or {}).get("task_hint", "") or "")
    routing_confidence = _safe_float((trace_context or {}).get("routing_confidence", 0.0), 0.0)
    token_budget = _safe_int((trace_context or {}).get("token_budget", 0), 0)
    retrieved_count = len(pages)
    evidence_density = min(
        1.0,
        max(
            0.0,
            0.45 * top_score
            + 0.35 * avg_score
            + 0.20 * min(retrieved_count, 4) / 4.0,
        ),
    )
    uncertainty_flags = {
        "no_pages": retrieved_count == 0,
        "low_routing_confidence": routing_confidence < 0.45,
        "weak_evidence": top_score < 0.30 and avg_score < 0.25,
        "diffuse_evidence": retrieved_count >= 4 and score_gap < 0.05,
        "long_context": context_char_count > 4000,
        "has_options": _has_options(query),
    }
    return AdaptiveAnswerState(
        query_text=str(query or "").strip(),
        query_type=query_type.strip(),
        task_hint=task_hint.strip(),
        retrieval_mode=str(retrieval_mode or "").strip().lower(),
        answer_mode=str(answer_mode or "").strip().lower(),
        answer_format=str(answer_format or "").strip().lower(),
        routing_confidence=routing_confidence,
        retrieved_count=retrieved_count,
        source_stats=source_stats,
        top_score=round(top_score, 4),
        score_gap=round(score_gap, 4),
        avg_score=round(avg_score, 4),
        token_budget=token_budget,
        context_char_count=context_char_count,
        has_options=_has_options(query),
        evidence_density=round(evidence_density, 4),
        uncertainty_flags=uncertainty_flags,
    )


def select_adaptive_answer_action(
    state: AdaptiveAnswerState,
    policy_snapshot: Optional[Dict[str, Any]] = None,
) -> AdaptiveAnswerAction:
    del policy_snapshot
    answer_strategy = "direct"
    retry_strategy = "none"
    abstain_policy = "allow"
    format_policy = "default"
    strongly_typed_structured_format = state.answer_format in {
        "yes_no",
        "items",
        "choice",
        "count",
        "rank",
        "comparison",
    }

    if state.answer_mode == "structured" and strongly_typed_structured_format:
        answer_strategy = "structured"
    elif state.answer_mode in {"binary", "list"}:
        answer_strategy = "direct"

    if state.answer_mode in {"binary", "list", "structured"} or state.answer_format in {
        "yes_no",
        "items",
        "choice",
        "count",
        "rank",
        "comparison",
    }:
        format_policy = "strict"

    if state.answer_mode == "extractive" and state.retrieval_mode == "local" and state.evidence_density >= 0.45:
        answer_strategy = "extractive"
        if state.answer_format == "span":
            format_policy = "minimal"

    if state.answer_mode == "abstractive" and state.retrieval_mode == "global":
        answer_strategy = "abstractive"

    if state.retrieval_mode == "hybrid" and state.uncertainty_flags.get("diffuse_evidence"):
        answer_strategy = "evidence_first"
        retry_strategy = "retry_with_shorter_context"

    strong_evidence = (
        state.retrieved_count >= 2
        and state.top_score >= 0.78
        and state.evidence_density >= 0.60
        and (state.score_gap >= 0.10 or state.retrieved_count <= 2)
        and not state.uncertainty_flags.get("low_routing_confidence")
        and not state.uncertainty_flags.get("weak_evidence")
        and not state.uncertainty_flags.get("diffuse_evidence")
    )
    if strong_evidence:
        abstain_policy = "discourage"
        if retry_strategy == "none":
            retry_strategy = "retry_once"

    if state.uncertainty_flags.get("weak_evidence") or state.uncertainty_flags.get("no_pages"):
        abstain_policy = "allow"
        if retry_strategy == "retry_once":
            retry_strategy = "none"

    if format_policy == "strict" and retry_strategy == "none":
        retry_strategy = "retry_with_stricter_format"

    return AdaptiveAnswerAction(
        answer_strategy=answer_strategy,
        retry_strategy=retry_strategy,
        abstain_policy=abstain_policy,
        format_policy=format_policy,
    )


def apply_adaptive_answer_action(
    state: AdaptiveAnswerState,
    action: AdaptiveAnswerAction,
    routing_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del routing_plan
    effective_answer_mode = state.answer_mode or "abstractive"
    effective_answer_format = state.answer_format or "freeform"
    strategy_hint = ""

    if action.answer_strategy == "extractive":
        effective_answer_mode = "extractive"
        if effective_answer_format in {"", "freeform", "open"}:
            effective_answer_format = "span"
    elif action.answer_strategy == "abstractive":
        effective_answer_mode = "abstractive"
    elif action.answer_strategy == "structured":
        effective_answer_mode = "structured"
        if effective_answer_format in {"", "freeform", "open"}:
            effective_answer_format = "structured"
        if effective_answer_format == "choice":
            strategy_hint = (
                "Evaluate each option against the retrieved evidence and choose the single best-supported option."
            )
        elif effective_answer_format == "count":
            strategy_hint = "Compute the count from the retrieved evidence and return only the number."
        elif effective_answer_format == "rank":
            strategy_hint = "Order the items by the retrieved evidence and return them in rank order."
        elif effective_answer_format == "comparison":
            strategy_hint = "Compare the candidates using the retrieved evidence and return the most supported comparison."
        elif effective_answer_format == "yes_no":
            strategy_hint = "Decide whether the evidence supports yes or no and return only that answer."
    elif action.answer_strategy == "evidence_first":
        strategy_hint = (
            "First identify the most relevant supporting evidence from the retrieved context, "
            "then provide the final answer."
        )

    if effective_answer_mode == "binary" and effective_answer_format in {"", "freeform", "open"}:
        effective_answer_format = "yes_no"
    elif effective_answer_mode == "list" and effective_answer_format in {"", "freeform", "open"}:
        effective_answer_format = "items"
    elif effective_answer_mode == "structured" and effective_answer_format in {"", "freeform", "open"}:
        effective_answer_format = "structured"

    instruction_fragments = []
    if action.abstain_policy == "discourage":
        instruction_fragments.append(
            "Only use \"I don't know.\" if the retrieved context clearly lacks the needed evidence."
        )
    elif action.abstain_policy == "disallow":
        instruction_fragments.append(
            "Do not answer with \"I don't know.\"; provide the best supported answer from the retrieved context."
        )

    if action.format_policy == "strict":
        instruction_fragments.append("Follow the required answer format exactly.")
    elif action.format_policy == "minimal":
        instruction_fragments.append("Keep the final answer as short as possible.")

    if strategy_hint:
        instruction_fragments.insert(0, strategy_hint)

    return {
        "effective_answer_mode": effective_answer_mode,
        "effective_answer_format": effective_answer_format,
        "retry_strategy": action.retry_strategy,
        "abstain_policy": action.abstain_policy,
        "format_policy": action.format_policy,
        "instruction_fragments": instruction_fragments,
    }


def should_retry_adaptive_answer(
    answer_text: str,
    *,
    state: AdaptiveAnswerState,
    action: AdaptiveAnswerAction,
    answer_mode: str,
    answer_format: str,
) -> bool:
    if action.retry_strategy == "none":
        return False
    if action.retry_strategy in {"retry_once", "retry_with_shorter_context"}:
        return _answer_looks_abstained(answer_text) and action.abstain_policy in {"discourage", "disallow"}
    if action.retry_strategy == "retry_with_stricter_format":
        normalized = str(answer_text or "").strip()
        if not normalized:
            return True
        normalized_mode = str(answer_mode or "").strip().lower()
        normalized_format = str(answer_format or "").strip().lower()
        if normalized_mode == "binary":
            return normalized.lower() not in {"yes", "no"}
        if normalized_format == "choice":
            return not bool(re.match(r"^\(?[A-Z]\)?(?:[.)])?(?:\s|$)", normalized))
        if normalized_format == "count":
            return not bool(re.fullmatch(r"-?\d+", normalized))
        if normalized_format == "rank":
            return "," not in normalized and "\n" not in normalized
        if normalized_mode in {"list", "structured"}:
            return len(normalized) == 0
    return False


def record_adaptive_answer_outcome(
    state: AdaptiveAnswerState,
    action: AdaptiveAnswerAction,
    outcome: AdaptiveAnswerOutcome,
    log_path: str,
) -> None:
    if not str(log_path or "").strip():
        return
    target = Path(str(log_path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state.to_payload(),
        "action": action.to_payload(),
        "outcome": outcome.to_payload(),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")
