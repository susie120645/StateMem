import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


GLOBAL_QUERY_PATTERNS = (
    re.compile(r"^\s*summar(?:ize|y)\b", re.I),
    re.compile(r"\b(?:overall|big picture|broader context|background|profile|overview)\b", re.I),
    re.compile(r"\b(?:main topics?|main decisions?|key decisions?|action items?)\b", re.I),
    re.compile(r"\bwhole (?:meeting|conversation|dialogue|discussion)\b", re.I),
)

LOCAL_QUERY_PATTERNS = (
    re.compile(r"^\s*(?:who|when|where|which)\b", re.I),
    re.compile(r"^\s*how\s+(?:many|much|long|old)\b", re.I),
    re.compile(r"\b(?:exact|specific|verbatim|span)\b", re.I),
    re.compile(r"\b(?:date|time|year|month|day|city|country|location|name|number)\b", re.I),
)

PROFILE_QUERY_PATTERNS = (
    re.compile(r"\b(?:preferences?|habits?|routine|personality|relationship|family context)\b", re.I),
    re.compile(r"\bwhat is .+ like\b", re.I),
)

BINARY_QUERY_PATTERNS = (
    re.compile(r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|should|would|will)\b", re.I),
)

LIST_QUERY_PATTERNS = (
    re.compile(r"\b(?:which|what)\s+(?:games|books|countries|locations|places|items|names|ways|methods)\b", re.I),
    re.compile(r"\b(?:list|name all|what are the names)\b", re.I),
)

STRUCTURED_QUERY_PATTERNS = (
    re.compile(r"\b(?:choose|option|candidate|rank|ranking|order|compare|comparison)\b", re.I),
    re.compile(r"\b(?:how many times|count|number of)\b", re.I),
)

CHOICE_QUERY_PATTERNS = (
    re.compile(r"\b(?:choose|choice|option|candidate)\b", re.I),
    re.compile(r"\b(?:answer|select)\s+(?:option|choice)\b", re.I),
)

COUNT_QUERY_PATTERNS = (
    re.compile(r"^\s*how\s+many\b", re.I),
    re.compile(r"\b(?:count|number of|how many times)\b", re.I),
)

RANK_QUERY_PATTERNS = (
    re.compile(r"\b(?:rank|ranking|order|ordered|sort)\b", re.I),
    re.compile(r"\b(?:first|second|third|last)\b", re.I),
)

COMPARISON_QUERY_PATTERNS = (
    re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b", re.I),
    re.compile(r"\b(?:earlier|later|before|after|more|less|higher|lower)\b", re.I),
)

RETRIEVAL_MODES = {"global", "local", "hybrid", "ood"}
ANSWER_MODES = {"extractive", "abstractive", "binary", "list", "structured", "ood"}


@dataclass(frozen=True)
class TaskRoutingFallback:
    retrieval_mode: str = "hybrid"
    answer_mode: str = "abstractive"

    def to_payload(self) -> Dict[str, str]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "answer_mode": self.answer_mode,
        }


@dataclass(frozen=True)
class TaskRoutingPlan:
    retrieval_mode: str
    answer_mode: str
    answer_format: str
    confidence: float
    fallback: TaskRoutingFallback = field(default_factory=TaskRoutingFallback)
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "answer_mode": self.answer_mode,
            "answer_format": self.answer_format,
            "confidence": round(float(self.confidence), 4),
            "fallback": self.fallback.to_payload(),
            "signals": dict(self.signals),
        }


def _normalize_mode(value: str, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else ""


def _clean_query_text(query: str) -> str:
    return " ".join(str(query or "").split()).strip()


def _matches_any(text: str, patterns) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _has_choice_markers(text: str) -> bool:
    normalized = str(text or "")
    lowered = normalized.lower()
    return "options:" in lowered or all(marker in normalized for marker in ("(A)", "(B)", "(C)"))


def infer_retrieval_mode(
    query: str,
    *,
    query_type: str = "",
    task_hint: str = "",
    preferred_kind: str = "",
) -> str:
    explicit = _normalize_mode(preferred_kind, RETRIEVAL_MODES)
    if explicit:
        return explicit

    normalized_query_type = str(query_type or "").strip().lower()
    if normalized_query_type == "general":
        return "global"
    if normalized_query_type == "specific":
        return "local"

    normalized_task_hint = str(task_hint or "").strip().lower()
    text = _clean_query_text(query)
    if not text:
        return "ood" if normalized_task_hint else ""

    has_global = _matches_any(text, GLOBAL_QUERY_PATTERNS) or _matches_any(text, PROFILE_QUERY_PATTERNS)
    has_local = _matches_any(text, LOCAL_QUERY_PATTERNS)

    if normalized_task_hint in {"dialogue_qa", "span_qa", "extractive_qa", "mrc"}:
        if has_global and has_local:
            return "hybrid"
        if has_global:
            return "global"
        return "local"

    if normalized_task_hint in {"meeting_summary", "meeting_qa"}:
        if has_global and has_local:
            return "hybrid"
        if has_global:
            return "global"
        if has_local:
            return "local"
        return "global"

    if has_global and has_local:
        return "hybrid"
    if has_global:
        return "global"
    if has_local:
        return "local"
    return "ood"


def infer_answer_mode(
    query: str,
    *,
    query_type: str = "",
    task_hint: str = "",
    preferred_mode: str = "",
) -> str:
    explicit = _normalize_mode(preferred_mode, ANSWER_MODES)
    if explicit:
        return explicit

    normalized_query_type = str(query_type or "").strip().lower()
    normalized_task_hint = str(task_hint or "").strip().lower()
    text = _clean_query_text(query)

    if normalized_query_type == "general":
        return "abstractive"

    if normalized_task_hint in {"meeting_summary", "meeting_qa"}:
        if normalized_query_type == "specific" and (
            _matches_any(text, STRUCTURED_QUERY_PATTERNS)
            or _matches_any(text, COUNT_QUERY_PATTERNS)
        ):
            return "structured"
        return "abstractive"

    if _matches_any(text, BINARY_QUERY_PATTERNS):
        return "binary"
    if _matches_any(text, LIST_QUERY_PATTERNS):
        return "list"
    if _matches_any(text, STRUCTURED_QUERY_PATTERNS):
        return "structured"

    if normalized_task_hint in {"dialogue_qa", "span_qa", "extractive_qa", "mrc"}:
        return "extractive"

    if _matches_any(text, GLOBAL_QUERY_PATTERNS):
        return "abstractive"
    if _matches_any(text, LOCAL_QUERY_PATTERNS):
        return "extractive"
    return "ood"


def infer_answer_format(
    query: str,
    *,
    answer_mode: str = "",
    query_type: str = "",
    task_hint: str = "",
) -> str:
    normalized_answer_mode = str(answer_mode or "").strip().lower()
    normalized_query_type = str(query_type or "").strip().lower()
    normalized_task_hint = str(task_hint or "").strip().lower()
    text = _clean_query_text(query)

    if normalized_answer_mode == "binary":
        return "yes_no"
    if normalized_answer_mode == "list":
        return "items"
    if normalized_answer_mode == "extractive":
        return "span"
    if normalized_answer_mode == "abstractive":
        if normalized_query_type == "general" or normalized_task_hint in {"meeting_summary", "meeting_qa"}:
            return "summary"
        return "freeform"
    if normalized_answer_mode == "structured":
        if _has_choice_markers(text):
            return "choice"
        if _matches_any(text, CHOICE_QUERY_PATTERNS):
            return "choice"
        if _matches_any(text, COUNT_QUERY_PATTERNS):
            return "count"
        if _matches_any(text, RANK_QUERY_PATTERNS):
            return "rank"
        if _matches_any(text, COMPARISON_QUERY_PATTERNS):
            return "comparison"
        return "structured"
    return "open"


def infer_task_routing_plan(
    query: str,
    *,
    query_type: str = "",
    task_hint: str = "",
    preferred_retrieval_mode: str = "",
    preferred_answer_mode: str = "",
    extra_signals: Optional[Dict[str, Any]] = None,
) -> TaskRoutingPlan:
    retrieval_mode = infer_retrieval_mode(
        query,
        query_type=query_type,
        task_hint=task_hint,
        preferred_kind=preferred_retrieval_mode,
    )
    answer_mode = infer_answer_mode(
        query,
        query_type=query_type,
        task_hint=task_hint,
        preferred_mode=preferred_answer_mode,
    )
    answer_format = infer_answer_format(
        query,
        answer_mode=answer_mode,
        query_type=query_type,
        task_hint=task_hint,
    )

    confidence = 0.35
    if preferred_retrieval_mode or preferred_answer_mode:
        confidence = 0.95
    elif query_type:
        confidence = 0.88
    elif task_hint:
        confidence = 0.74

    if retrieval_mode == "hybrid":
        confidence = min(confidence, 0.58)
    elif retrieval_mode == "ood":
        confidence = min(confidence, 0.32)

    if answer_mode == "ood":
        confidence = min(confidence, 0.32)
    elif answer_mode in {"binary", "list", "structured"}:
        confidence = max(confidence, 0.7)
    elif answer_mode == "extractive" and retrieval_mode == "local":
        confidence = max(confidence, 0.8 if task_hint else 0.68)
    elif answer_mode == "abstractive" and retrieval_mode == "global":
        confidence = max(confidence, 0.8 if task_hint or query_type == "general" else 0.68)

    if confidence < 0.45:
        fallback = TaskRoutingFallback(retrieval_mode="hybrid", answer_mode="abstractive")
    else:
        fallback = TaskRoutingFallback(
            retrieval_mode="hybrid" if retrieval_mode != "hybrid" else retrieval_mode,
            answer_mode="abstractive" if answer_mode != "abstractive" else answer_mode,
        )

    signals: Dict[str, Any] = dict(extra_signals or {})
    if query_type:
        signals["query_type"] = str(query_type).strip()
    if task_hint:
        signals["task_hint"] = str(task_hint).strip()
    if retrieval_mode:
        signals["retrieval_mode"] = retrieval_mode
    if answer_mode:
        signals["answer_mode"] = answer_mode
    if answer_format:
        signals["answer_format"] = answer_format
    if answer_mode == "extractive":
        signals.setdefault("evidence_need", "single_span")
        signals.setdefault("reasoning_need", "low")
    elif answer_mode == "abstractive":
        signals.setdefault("evidence_need", "multi_span")
        signals.setdefault("reasoning_need", "medium")
    elif answer_mode in {"binary", "list", "structured"}:
        signals.setdefault("evidence_need", "targeted")
        signals.setdefault("reasoning_need", "medium")

    return TaskRoutingPlan(
        retrieval_mode=retrieval_mode or "ood",
        answer_mode=answer_mode or "ood",
        answer_format=answer_format or "open",
        confidence=max(0.0, min(1.0, float(confidence))),
        fallback=fallback,
        signals=signals,
    )


def resolve_effective_retrieval_mode(
    plan: Any,
    *,
    min_confidence: float = 0.45,
    default_mode: str = "hybrid",
) -> str:
    payload: Dict[str, Any]
    if isinstance(plan, TaskRoutingPlan):
        payload = plan.to_payload()
    elif isinstance(plan, dict):
        payload = dict(plan)
    else:
        payload = {}

    retrieval_mode = str(payload.get("retrieval_mode", "") or "").strip().lower()
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    fallback = payload.get("fallback", {})
    fallback_mode = ""
    if isinstance(fallback, dict):
        fallback_mode = str(fallback.get("retrieval_mode", "") or "").strip().lower()

    if confidence < float(min_confidence):
        return fallback_mode or default_mode
    if retrieval_mode in {"", "ood"}:
        return fallback_mode or default_mode
    return retrieval_mode


def infer_retrieval_query_kind(
    query: str,
    *,
    query_type: str = "",
    task_hint: str = "",
    preferred_kind: str = "",
) -> str:
    retrieval_mode = infer_retrieval_mode(
        query,
        query_type=query_type,
        task_hint=task_hint,
        preferred_kind=preferred_kind,
    )
    if retrieval_mode in {"global", "local"}:
        return retrieval_mode
    return ""


def build_retrieval_trace_context(
    query: str,
    *,
    query_type: str = "",
    task_hint: str = "",
    preferred_kind: str = "",
    preferred_answer_mode: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trace_context: Dict[str, Any] = dict(extra or {})
    query_text = str(query or "").strip()
    if query_text:
        trace_context["query_text"] = query_text
    normalized_query_type = str(query_type or "").strip()
    if normalized_query_type:
        trace_context["query_type"] = normalized_query_type
    normalized_task_hint = str(task_hint or "").strip()
    if normalized_task_hint:
        trace_context["task_hint"] = normalized_task_hint
    routing_plan = infer_task_routing_plan(
        query_text,
        query_type=normalized_query_type,
        task_hint=normalized_task_hint,
        preferred_retrieval_mode=preferred_kind,
        preferred_answer_mode=preferred_answer_mode,
        extra_signals=trace_context.get("signals") if isinstance(trace_context.get("signals"), dict) else None,
    )
    trace_context["task_routing_plan"] = routing_plan.to_payload()
    trace_context["retrieval_mode"] = routing_plan.retrieval_mode
    trace_context["answer_mode"] = routing_plan.answer_mode
    trace_context["answer_format"] = routing_plan.answer_format
    trace_context["routing_confidence"] = round(float(routing_plan.confidence), 4)
    trace_context["effective_retrieval_mode"] = resolve_effective_retrieval_mode(routing_plan)
    query_kind = routing_plan.retrieval_mode if routing_plan.retrieval_mode in {"global", "local"} else ""
    if query_kind:
        trace_context["query_kind"] = query_kind
    return trace_context
