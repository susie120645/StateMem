import json
import re
from typing import Any, Dict, Optional

try:
    from .query_routing import TaskRoutingPlan
except (ImportError, ValueError):
    from query_routing import TaskRoutingPlan


def extract_task_routing_plan_payload(trace_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(trace_context, dict):
        return {}
    payload = trace_context.get("task_routing_plan")
    return dict(payload) if isinstance(payload, dict) else {}


def resolve_effective_answer_mode(
    plan: Any,
    *,
    min_confidence: float = 0.45,
    default_mode: str = "abstractive",
) -> str:
    payload: Dict[str, Any]
    if isinstance(plan, TaskRoutingPlan):
        payload = plan.to_payload()
    elif isinstance(plan, dict):
        payload = dict(plan)
    else:
        payload = {}

    answer_mode = str(payload.get("answer_mode", "") or "").strip().lower()
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    fallback = payload.get("fallback", {})
    fallback_mode = ""
    if isinstance(fallback, dict):
        fallback_mode = str(fallback.get("answer_mode", "") or "").strip().lower()

    if confidence < float(min_confidence):
        return fallback_mode or default_mode
    if answer_mode in {"", "ood"}:
        return fallback_mode or default_mode
    return answer_mode


def resolve_effective_answer_format(
    plan: Any,
    *,
    default_format: str = "freeform",
) -> str:
    payload: Dict[str, Any]
    if isinstance(plan, TaskRoutingPlan):
        payload = plan.to_payload()
    elif isinstance(plan, dict):
        payload = dict(plan)
    else:
        payload = {}

    answer_format = str(payload.get("answer_format", "") or "").strip().lower()
    if answer_format:
        return answer_format

    answer_mode = resolve_effective_answer_mode(payload)
    if answer_mode == "binary":
        return "yes_no"
    if answer_mode == "list":
        return "items"
    if answer_mode == "extractive":
        return "span"
    if answer_mode == "structured":
        return "structured"
    if answer_mode == "abstractive":
        return "freeform"
    return default_format


def build_answer_mode_instruction(answer_mode: str, *, task_hint: str = "", answer_format: str = "") -> str:
    normalized_mode = str(answer_mode or "").strip().lower()
    normalized_task_hint = str(task_hint or "").strip().lower()
    normalized_format = str(answer_format or "").strip().lower()
    hybrid_prefix = ""
    if normalized_task_hint in {"hybrid", "hybrid_qa"}:
        hybrid_prefix = "Use broad background to identify the right context, then rely on the most specific supporting details for the final answer. "

    if normalized_mode == "extractive":
        return (
            hybrid_prefix
            + "Return the shortest answer copied verbatim from the retrieved context. "
            "If the answer contains multiple items, return a concise comma-separated list using exact terms from the context."
        )
    if normalized_mode == "binary":
        return hybrid_prefix + "Answer with yes or no only, unless a one-phrase clarification is strictly necessary."
    if normalized_mode == "list":
        return hybrid_prefix + "Return a deduplicated comma-separated list using the exact item names found in the context."
    if normalized_mode == "structured":
        if normalized_format == "choice":
            return hybrid_prefix + "Return only the final choice label or exact option key, such as A, B, C, or D."
        if normalized_format == "count":
            return hybrid_prefix + "Return only the final count as a single number."
        if normalized_format == "rank":
            return hybrid_prefix + "Return the ranked items in order as a concise comma-separated list."
        if normalized_format == "comparison":
            return hybrid_prefix + "Return a compact structured comparison in plain text, such as earlier=lunch; later=dinner."
        return hybrid_prefix + "Return a compact structured answer in plain text that matches the task, such as count=3, choice=A, or rank: item1, item2."
    if normalized_mode == "abstractive":
        if normalized_task_hint in {"meeting_summary", "meeting_qa"}:
            return hybrid_prefix + "Write a concise answer that synthesizes the retrieved context without adding unsupported details."
        return hybrid_prefix + "Write a concise answer grounded only in the retrieved context."
    return hybrid_prefix + "Use a conservative evidence-grounded answer. If the context is insufficient, say you do not know."


def _strip_answer_wrapper(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
    cleaned = re.sub(r"^\s*(?:final answer|answer|response)\s*:\s*", "", cleaned, flags=re.I)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _normalize_binary_answer(text: str) -> str:
    lowered = text.lower()
    if re.match(r"^(yes|yeah|yep|true)\b", lowered):
        return "yes"
    if re.match(r"^(no|nope|false)\b", lowered):
        return "no"
    return text


def _normalize_list_answer(text: str) -> str:
    raw_items = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line.strip())
        if not stripped:
            continue
        raw_items.extend(part.strip() for part in re.split(r"\s*[;,]\s*", stripped) if part.strip())
    if not raw_items:
        raw_items = [part.strip() for part in re.split(r"\s*[;,]\s*", text) if part.strip()]
    seen = set()
    items = []
    for item in raw_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return ", ".join(items) if items else text


def _normalize_choice_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = re.search(r'(?i)"?(?:choice|answer|option)"?\s*[:=]\s*"?([A-Z])\b', cleaned)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*(?:option|choice)\s*([A-Z])\b", cleaned, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*\(?([A-Z])\)?(?:[.:)\s]|$)", cleaned)
    if match:
        return match.group(1).upper()
    return cleaned


def _normalize_count_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = re.search(r"(?i)(?:count|number|total)\s*[:=]\s*(-?\d+)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"-?\d+", cleaned)
    if match:
        return match.group(0)
    return cleaned


def _normalize_rank_answer(text: str) -> str:
    raw_items = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line.strip())
        if not stripped:
            continue
        raw_items.extend(part.strip() for part in re.split(r"\s*(?:,|>|->|→|;)\s*", stripped) if part.strip())
    if not raw_items:
        raw_items = [part.strip() for part in re.split(r"\s*(?:,|>|->|→|;)\s*", text) if part.strip()]
    seen = set()
    items = []
    for item in raw_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return ", ".join(items) if items else text


def _normalize_structured_answer(text: str, answer_format: str = "") -> str:
    compact = text.strip()
    if not compact:
        return ""
    if compact.startswith("{") and compact.endswith("}"):
        try:
            payload = json.loads(compact)
            if isinstance(payload, dict):
                if answer_format == "choice":
                    choice_value = payload.get("choice") or payload.get("answer") or payload.get("option")
                    if choice_value is not None:
                        return _normalize_choice_answer(str(choice_value))
                if answer_format == "count":
                    count_value = payload.get("count")
                    if count_value is None:
                        count_value = payload.get("number", payload.get("value"))
                    if count_value is not None:
                        return _normalize_count_answer(str(count_value))
                if answer_format == "rank":
                    for key in ("rank", "ranking", "order", "items"):
                        rank_value = payload.get(key)
                        if isinstance(rank_value, list):
                            return _normalize_rank_answer(", ".join(str(item) for item in rank_value))
                return "; ".join(f"{key}={value}" for key, value in payload.items())
            if isinstance(payload, list) and answer_format == "rank":
                return _normalize_rank_answer(", ".join(str(item) for item in payload))
        except Exception:
            pass
    if answer_format == "choice":
        return _normalize_choice_answer(compact)
    if answer_format == "count":
        return _normalize_count_answer(compact)
    if answer_format == "rank":
        return _normalize_rank_answer(compact)
    lines = [re.sub(r"\s+", " ", line.strip()) for line in compact.splitlines() if line.strip()]
    return "; ".join(lines) if lines else compact


def normalize_generated_answer(answer: Any, answer_mode: str, *, task_hint: str = "", answer_format: str = "") -> str:
    del task_hint
    cleaned = _strip_answer_wrapper(str(answer or ""))
    if not cleaned:
        return ""
    normalized_mode = str(answer_mode or "").strip().lower()
    normalized_format = str(answer_format or "").strip().lower()
    if normalized_mode == "binary":
        return _normalize_binary_answer(cleaned)
    if normalized_mode == "list":
        return _normalize_list_answer(cleaned)
    if normalized_mode == "structured":
        return _normalize_structured_answer(cleaned, normalized_format)
    if normalized_mode == "extractive":
        return cleaned.splitlines()[0].strip().strip("\"'“”")
    return cleaned
