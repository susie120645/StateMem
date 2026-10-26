import re
from typing import Any, List, Optional, Sequence


STRUCTURAL_ENTITY_TERMS = {
    "summary",
    "session summary",
    "event summary",
    "conversation summary",
    "transcript",
    "session transcript",
    "transcript window",
    "overview",
    "session overview",
    "window",
    "observations",
    "observation",
    "resolved relative times",
    "resolved event facts",
    "dialogue ids",
    "dialogue id",
    "query hint",
    "image caption",
    "image url",
    "topic",
    "sample",
    "date",
    "time",
    "exact time",
    "resolved time",
    "user asked",
    "assistant replied",
}

STRUCTURAL_ENTITY_TOKENS = {
    "summary",
    "session",
    "event",
    "conversation",
    "dialogue",
    "dialogues",
    "transcript",
    "overview",
    "window",
    "observation",
    "observations",
    "resolved",
    "relative",
    "times",
    "facts",
    "fact",
    "query",
    "hint",
    "image",
    "caption",
    "url",
    "sample",
    "date",
    "time",
    "exact",
    "topic",
    "user",
    "assistant",
    "asked",
    "replied",
    "ids",
    "id",
}


def normalize_entity_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip(" \t\r\n\"'`[](){}<>.,;:!?")


def looks_like_memory_reference(value: Any) -> bool:
    candidate = normalize_entity_candidate(value)
    if not candidate:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9-]+(?:_[A-Za-z0-9-]+)+", candidate) and re.search(r"\d", candidate))


def is_structural_entity_label(value: Any) -> bool:
    candidate = normalize_entity_candidate(value)
    lowered = candidate.lower()
    if not lowered:
        return True
    if lowered in STRUCTURAL_ENTITY_TERMS:
        return True
    if re.fullmatch(r"(?:locomo\s+)?session\s+\d+(?:\s+(?:overview|transcript|window)(?:\s+\d+)?)?(?:\s+sample)?", lowered):
        return True
    if re.fullmatch(r"d\d+:\d+(?:\s*-\s*d?\d+:\d+)?", lowered):
        return True

    tokens = re.findall(r"[a-z0-9]+", lowered)
    if not tokens:
        return True

    non_numeric = [token for token in tokens if not token.isdigit()]
    if not non_numeric:
        return True

    substantive_tokens = [token for token in non_numeric if token not in STRUCTURAL_ENTITY_TOKENS]
    heading_tokens = [token for token in non_numeric if token in STRUCTURAL_ENTITY_TOKENS]

    if not substantive_tokens and heading_tokens:
        return True
    if substantive_tokens == ["locomo"] and heading_tokens:
        return True
    if len(substantive_tokens) <= 1 and len(heading_tokens) >= 2 and len(non_numeric) >= 3:
        return True
    return False


def filter_entity_values(values: Sequence[Any], limit: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        candidate = normalize_entity_candidate(value)
        if len(candidate) < 2 or is_structural_entity_label(candidate) or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
        if limit is not None and len(result) >= limit:
            break
    return result


def filter_semantic_edge_values(values: Sequence[Any], limit: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        candidate = normalize_entity_candidate(value)
        if len(candidate) < 2 or candidate in seen:
            continue
        if not looks_like_memory_reference(candidate) and is_structural_entity_label(candidate):
            continue
        seen.add(candidate)
        result.append(candidate)
        if limit is not None and len(result) >= limit:
            break
    return result
