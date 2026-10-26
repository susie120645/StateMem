from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

try:
    from .choice_answering import parse_choice_question
    from .entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate
    from .utils import extract_temporal_canonical_forms
except (ImportError, ValueError):
    from choice_answering import parse_choice_question
    from entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate
    from utils import extract_temporal_canonical_forms


_RELATION_PATTERNS = {
    "dating": (
        re.compile(r"\bdate\b", re.I),
        re.compile(r"\bdating\b", re.I),
        re.compile(r"\bgoing out\b", re.I),
        re.compile(r"\bwent out with\b", re.I),
        re.compile(r"\bgirlfriend\b", re.I),
        re.compile(r"\bboyfriend\b", re.I),
        re.compile(r"\bpartner\b", re.I),
    ),
    "roommate": (
        re.compile(r"\broommate\b", re.I),
        re.compile(r"\bbunking with\b", re.I),
        re.compile(r"\bliving with\b", re.I),
        re.compile(r"\blive with\b", re.I),
    ),
    "neighbor": (
        re.compile(r"\bnext door\b", re.I),
        re.compile(r"\bneighbor\b", re.I),
    ),
    "family": (
        re.compile(r"\bfamily\b", re.I),
        re.compile(r"\bmother\b", re.I),
        re.compile(r"\bfather\b", re.I),
        re.compile(r"\bsister\b", re.I),
        re.compile(r"\bbrother\b", re.I),
        re.compile(r"\bson\b", re.I),
        re.compile(r"\bdaughter\b", re.I),
        re.compile(r"\bkid\b", re.I),
        re.compile(r"\bchild\b", re.I),
        re.compile(r"\bparent\b", re.I),
    ),
    "work": (
        re.compile(r"\bwork(?:s|ed|ing)?\b", re.I),
        re.compile(r"\bjob\b", re.I),
        re.compile(r"\bcompany\b", re.I),
        re.compile(r"\boffice\b", re.I),
        re.compile(r"\bcareer\b", re.I),
    ),
    "object": (
        re.compile(r"\bring\b", re.I),
        re.compile(r"\bgift\b", re.I),
        re.compile(r"\bpresent\b", re.I),
        re.compile(r"\bcreature\b", re.I),
        re.compile(r"\bpet\b", re.I),
    ),
    "age": (
        re.compile(r"\bhow old\b", re.I),
        re.compile(r"\bage\b", re.I),
        re.compile(r"\b\d{1,3}\s+years old\b", re.I),
        re.compile(r"\bonly\s+\d{1,3}\b", re.I),
        re.compile(r"\bturned\s+\d{1,3}\b", re.I),
    ),
}

_PRONOUN_SLOT_PATTERNS = (
    re.compile(
        r"\b(that|this|his|her|their)\s+"
        r"(person|roommate|neighbor|daughter|son|mother|father|brother|sister|kid|child|partner|date)\b",
        re.I,
    ),
    re.compile(r"\b(the one|that one)\b", re.I),
)

_ENTITY_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Za-z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
)
_POSSESSIVE_ENTITY_RE = re.compile(r"\b((?:Mr|Mrs|Ms|Dr)\.?\s+[A-Za-z][a-z]+|[A-Z][a-z]+)'s\b")
_QUESTION_WORDS = {
    "who", "what", "when", "where", "which", "why", "how", "statemember", "hey", "so",
    "just", "oh", "god", "my", "do", "did", "does", "is", "are", "was", "were",
}
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_WEEKDAY_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
}
_BLOCKED_ENTITY_WORDS = {
    "i", "you", "we", "they", "he", "she", "it", "me", "him", "her", "them",
    "there", "what", "when", "where", "why", "who", "how",
    "episode", "session", "date", "question", "answer", "q", "a",
    "hi", "hello", "hey", "oh", "god", "well", "no", "yes", "really", "okay",
    "come", "must", "listen", "statemember", "that", "this", "those", "these",
    "he's", "she's", "it's", "that's", "what's", "there's", "you're", "i'm", "we're", "they're",
}
_DIALOGUE_TURN_RE = re.compile(r"(?<!\[)([A-Z][A-Za-z .'\-]{1,40}):\s*")
_BLOCKED_SPEAKERS = _BLOCKED_ENTITY_WORDS | {"q", "a", "all", "date", "episode", "session"}
_ROOMMATE_SELF_RE = re.compile(r"\bmy roommate\b", re.I)
_SELF_DATE_RE = re.compile(
    r"\b(?:i got a date with|i'm going out with|i am going out with|i went out with|my girlfriend(?:,?\s+uh,?\s+)?|my boyfriend(?:,?\s+uh,?\s+)?)\s+([A-Z][A-Za-z. ]+)",
    re.I,
)
_THAT_GUY_DATE_RE = re.compile(r"\b([A-Z][A-Za-z. ]+),\s+that (?:guy|girl) I went out with\b", re.I)
_THIRD_PARTY_RELATION_RE = re.compile(
    r"\b([A-Z][A-Za-z. ]+?)\s+(?:was|is|went out with|going out with|dated)\s+([A-Z][A-Za-z. ]+)\b",
    re.I,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_useful_fact_value(value: str) -> bool:
    candidate = _clean_text(value)
    lowered = candidate.lower()
    if not candidate:
        return False
    if lowered in _BLOCKED_ENTITY_WORDS:
        return False
    if is_structural_entity_label(candidate):
        return False
    if len(candidate) == 1 and not candidate.isdigit():
        return False
    if re.fullmatch(r"[A-Za-z]+'?[a-z]*", candidate) and lowered in _BLOCKED_ENTITY_WORDS:
        return False
    return True


def _extract_entities(text: str) -> List[str]:
    candidates = []
    for match in _POSSESSIVE_ENTITY_RE.findall(str(text or "")):
        candidate = normalize_entity_candidate(match)
        if candidate:
            candidates.append(candidate)
    for match in _ENTITY_RE.findall(str(text or "")):
        candidate = normalize_entity_candidate(match)
        if not candidate:
            continue
        if candidate.lower() in _QUESTION_WORDS:
            continue
        if candidate.lower() in _MONTH_WORDS:
            continue
        if candidate.lower() in _WEEKDAY_WORDS:
            continue
        if candidate.lower() in _BLOCKED_ENTITY_WORDS:
            continue
        if is_structural_entity_label(candidate):
            continue
        candidates.append(candidate)
    return filter_entity_values(candidates, limit=24)


def _extract_relation_terms(text: str) -> List[str]:
    compact = str(text or "")
    relations: List[str] = []
    for relation, patterns in _RELATION_PATTERNS.items():
        if any(pattern.search(compact) for pattern in patterns):
            relations.append(relation)
    return relations


def _extract_pronoun_slots(text: str) -> List[str]:
    slots: List[str] = []
    for pattern in _PRONOUN_SLOT_PATTERNS:
        for match in pattern.finditer(str(text or "")):
            slot = _clean_text(match.group(0))
            if slot and slot not in slots:
                slots.append(slot)
    return slots


def _infer_question_shape(entities: Sequence[str], temporal_forms: Sequence[str], relation_terms: Sequence[str], pronoun_slots: Sequence[str]) -> str:
    if pronoun_slots and relation_terms:
        return "two_hop_relation_lookup"
    if entities and temporal_forms and relation_terms:
        return "entity_time_relation_lookup"
    if entities and relation_terms:
        return "entity_relation_lookup"
    if temporal_forms and relation_terms:
        return "time_relation_lookup"
    if entities and temporal_forms:
        return "entity_time_lookup"
    return "open"


def canonicalize_query(query: str, *, task_hint: str = "", answer_format: str = "") -> Dict[str, Any]:
    parsed = parse_choice_question(query)
    stem = _clean_text(parsed.stem or query)
    entities = _extract_entities(stem)
    temporal_forms = list(extract_temporal_canonical_forms(stem))
    relation_terms = _extract_relation_terms(stem)
    pronoun_slots = _extract_pronoun_slots(stem)
    question_shape = _infer_question_shape(entities, temporal_forms, relation_terms, pronoun_slots)
    return {
        "query_text": stem,
        "task_hint": str(task_hint or ""),
        "answer_format": str(answer_format or ""),
        "entities": entities,
        "temporal_forms": temporal_forms,
        "relation_terms": relation_terms,
        "pronoun_slots": pronoun_slots,
        "question_shape": question_shape,
        "choice_options": [
            {"label": option.label, "marker": option.marker, "text": option.text}
            for option in parsed.options
        ],
    }


def _normalize_fact_tuple(candidate: Dict[str, Any], *, fallback_citation_id: str, fallback_family: str, fallback_session: str) -> Dict[str, Any]:
    temporal_forms = candidate.get("temporal_forms") or candidate.get("time") or []
    if isinstance(temporal_forms, str):
        temporal_forms = [temporal_forms]
    relation_terms = candidate.get("relation_terms") or candidate.get("relations") or []
    if isinstance(relation_terms, str):
        relation_terms = [relation_terms]
    return {
        "fact_id": str(candidate.get("fact_id") or f"{fallback_citation_id}:fact"),
        "citation_id": str(candidate.get("citation_id") or fallback_citation_id),
        "subject": _clean_text(candidate.get("subject")),
        "relation_terms": [str(value) for value in relation_terms if str(value).strip()],
        "object": _clean_text(candidate.get("object")),
        "temporal_forms": [str(value) for value in temporal_forms if str(value).strip()],
        "source_family": str(candidate.get("source_family") or fallback_family),
        "source_session": str(candidate.get("source_session") or fallback_session),
        "evidence_text": _clean_text(candidate.get("evidence_text") or candidate.get("text")),
        "fact_quality": str(candidate.get("fact_quality") or "metadata"),
    }


def _page_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
    metadata = page.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _page_archive_index(page: Dict[str, Any]) -> Dict[str, Any]:
    archive_index = page.get("archive_index", {})
    return archive_index if isinstance(archive_index, dict) else {}


def _page_entities(page: Dict[str, Any], text: str) -> List[str]:
    entities = page.get("entities")
    if isinstance(entities, list) and entities:
        return filter_entity_values(entities, limit=24)
    metadata = _page_metadata(page)
    entities = metadata.get("entities")
    if isinstance(entities, list) and entities:
        return filter_entity_values(entities, limit=24)
    return _extract_entities(text)


def _page_temporal_forms(page: Dict[str, Any], text: str) -> List[str]:
    metadata = _page_metadata(page)
    archive_index = _page_archive_index(page)
    values = metadata.get("temporal_forms")
    if isinstance(values, list) and values:
        return [str(value) for value in values if str(value).strip()]
    values = archive_index.get("temporal_forms")
    if isinstance(values, list) and values:
        return [str(value) for value in values if str(value).strip()]
    return list(extract_temporal_canonical_forms(text))


def _page_relation_terms(page: Dict[str, Any], text: str) -> List[str]:
    metadata = _page_metadata(page)
    archive_index = _page_archive_index(page)
    values = metadata.get("relation_terms")
    if isinstance(values, list) and values:
        return [str(value) for value in values if str(value).strip()]
    values = archive_index.get("relation_terms")
    if isinstance(values, list) and values:
        return [str(value) for value in values if str(value).strip()]
    return _extract_relation_terms(text)


def _page_source_family(page: Dict[str, Any]) -> str:
    metadata = _page_metadata(page)
    archive_index = _page_archive_index(page)
    return str(
        archive_index.get("source_family")
        or metadata.get("source_family")
        or metadata.get("session_group_id")
        or metadata.get("conversation_id")
        or metadata.get("session_id")
        or page.get("doc_id")
        or page.get("citation_id")
        or page.get("page_id")
        or ""
    )


def _page_source_session(page: Dict[str, Any]) -> str:
    metadata = _page_metadata(page)
    archive_index = _page_archive_index(page)
    return str(archive_index.get("source_session") or metadata.get("session_id") or "")


def _iter_fact_segments(page: Dict[str, Any]) -> List[str]:
    raw_content = str(page.get("raw_content", "") or "")
    summary = str(page.get("summary", "") or "")
    segments: List[str] = []
    for line in raw_content.splitlines():
        line = line.strip()
        if line.startswith("A:"):
            segments.append(line[2:].strip())
    if summary:
        segments.append(summary)
    for part in re.split(r"[\n.!?]+", raw_content):
        part = _clean_text(part)
        if part:
            segments.append(part)
    deduped: List[str] = []
    seen = set()
    for segment in segments:
        lowered = segment.lower()
        if lowered.startswith("q:") or "[episode:" in lowered:
            continue
        if segment not in seen:
            seen.add(segment)
            deduped.append(segment)
    return deduped[:8]


def _iter_dialogue_turns(page: Dict[str, Any]) -> List[tuple[str, str]]:
    raw_content = str(page.get("raw_content", "") or "")
    if not raw_content:
        return []

    cleaned = raw_content.replace("\n\n\nA:Q:", "\n").replace("A:Q:", "\n")
    cleaned = re.sub(r"Q:\s*\[Episode:[^\]]+\]\s*", "", cleaned)
    matches = list(_DIALOGUE_TURN_RE.finditer(cleaned))
    if not matches:
        return []

    turns: List[tuple[str, str]] = []
    for index, match in enumerate(matches):
        speaker = normalize_entity_candidate(match.group(1))
        if not speaker or speaker.lower() in _BLOCKED_SPEAKERS:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        utterance = _clean_text(cleaned[start:end])
        if not utterance:
            continue
        turns.append((speaker, utterance))
    return turns[:40]


def _candidate_name_from_text(text: str) -> str:
    entities = _extract_entities(text)
    return entities[0] if entities else ""


def _build_fact(
    *,
    citation_id: str,
    index: int,
    subject: str,
    relation: str,
    obj: str,
    temporal_forms: Sequence[str],
    source_family: str,
    source_session: str,
    evidence_text: str,
    fact_quality: str,
) -> Dict[str, Any]:
    return {
        "fact_id": f"{citation_id}:{index}:{relation}",
        "citation_id": citation_id,
        "subject": subject,
        "relation_terms": [relation],
        "object": obj,
        "temporal_forms": list(temporal_forms)[:2],
        "source_family": source_family,
        "source_session": source_session,
        "evidence_text": evidence_text,
        "fact_quality": fact_quality,
    }


def _fact_quality_bonus(fact_quality: str) -> float:
    quality = str(fact_quality or "")
    if quality.startswith("pattern_"):
        return 0.08
    if quality.startswith("heuristic_dialogue"):
        return 0.02
    if quality == "fallback":
        return -0.05
    return 0.0


def _fact_entity_set(fact: Dict[str, Any]) -> set[str]:
    entities = {
        str(fact.get("subject") or "").strip().lower(),
        str(fact.get("object") or "").strip().lower(),
    }
    return {value for value in entities if value}


def _other_fact_entity(fact: Dict[str, Any], anchor_entities: set[str]) -> str:
    for key in ("subject", "object"):
        candidate = _clean_text(fact.get(key))
        if candidate and candidate.lower() not in anchor_entities and _is_useful_fact_value(candidate):
            return candidate
    return ""


def _resolve_two_hop_chain(
    canonical_query: Dict[str, Any],
    scored_facts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    anchor_entities = {value.lower() for value in canonical_query.get("entities", [])}
    roommate_facts = []
    dating_facts = []
    for fact in scored_facts:
        relations = set(fact.get("relation_terms", []))
        if "roommate" in relations:
            roommate_facts.append(fact)
        if "dating" in relations:
            dating_facts.append(fact)

    pivot_candidates = []
    for fact in roommate_facts:
        fact_entities = _fact_entity_set(fact)
        if anchor_entities and not (fact_entities & anchor_entities):
            continue
        pivot = _other_fact_entity(fact, anchor_entities)
        if pivot:
            pivot_candidates.append((pivot, fact))

    best = {
        "score": -1.0,
        "pivot": "",
        "resolved_candidate": "",
        "roommate_fact": None,
        "dating_fact": None,
        "status": "missing_roommate_anchor",
        "pivot_count": 0,
    }
    for pivot, roommate_fact in pivot_candidates:
        pivot_lower = pivot.lower()
        for dating_fact in dating_facts:
            dating_entities = _fact_entity_set(dating_fact)
            if pivot_lower not in dating_entities:
                continue
            resolved = _other_fact_entity(dating_fact, {pivot_lower})
            if not _is_useful_fact_value(resolved):
                continue
            chain_score = float(roommate_fact.get("score", 0.0)) + float(dating_fact.get("score", 0.0))
            if chain_score > best["score"]:
                best = {
                    "score": chain_score,
                    "pivot": pivot,
                    "resolved_candidate": resolved,
                    "roommate_fact": roommate_fact,
                    "dating_fact": dating_fact,
                    "status": "resolved",
                    "pivot_count": len(pivot_candidates),
                }

    if pivot_candidates and best["status"] != "resolved":
        best["status"] = "missing_dating_link"
        best["pivot_count"] = len(pivot_candidates)
    return best


def extract_fact_tuples(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    citation_id = str(page.get("citation_id") or page.get("page_id") or "")
    source_family = _page_source_family(page)
    source_session = _page_source_session(page)
    metadata = _page_metadata(page)
    raw_fact_tuples = metadata.get("fact_tuples")
    if isinstance(raw_fact_tuples, list) and raw_fact_tuples:
        normalized = [
            _normalize_fact_tuple(
                item,
                fallback_citation_id=citation_id,
                fallback_family=source_family,
                fallback_session=source_session,
            )
            for item in raw_fact_tuples
            if isinstance(item, dict)
        ]
        return [item for item in normalized if item.get("relation_terms") or item.get("subject") or item.get("object")]

    text = "\n".join(
        part for part in [page.get("summary", ""), page.get("raw_content", "")] if str(part or "").strip()
    )
    entities = _page_entities(page, text)
    temporal_forms = _page_temporal_forms(page, text)
    relation_terms = _page_relation_terms(page, text)
    facts: List[Dict[str, Any]] = []
    seen = set()

    dialogue_turns = _iter_dialogue_turns(page)
    turn_iterable: List[tuple[str, str, str]] = [
        (speaker, utterance, "dialogue") for speaker, utterance in dialogue_turns
    ]
    if not turn_iterable:
        turn_iterable = [("", segment, "segment") for segment in _iter_fact_segments(page)]

    for index, (speaker, segment, source_kind) in enumerate(turn_iterable):
        segment_entities = _extract_entities(segment)
        segment_relations = _extract_relation_terms(segment)
        segment_temporal = list(extract_temporal_canonical_forms(segment)) or temporal_forms
        if not segment_relations:
            continue

        if speaker and "roommate" in segment_relations and _ROOMMATE_SELF_RE.search(segment):
            for candidate in segment_entities:
                if candidate.lower() != speaker.lower() and _is_useful_fact_value(candidate):
                    facts.append(
                        _build_fact(
                            citation_id=citation_id,
                            index=index,
                            subject=speaker,
                            relation="roommate",
                            obj=candidate,
                            temporal_forms=segment_temporal,
                            source_family=source_family,
                            source_session=source_session,
                            evidence_text=segment,
                            fact_quality=f"pattern_{source_kind}",
                        )
                    )
                    break

        if speaker and "dating" in segment_relations:
            indirect_match = _THAT_GUY_DATE_RE.search(segment)
            if indirect_match:
                candidate = _candidate_name_from_text(indirect_match.group(1))
                if _is_useful_fact_value(candidate):
                    facts.append(
                        _build_fact(
                            citation_id=citation_id,
                            index=index,
                            subject=speaker,
                            relation="dating",
                            obj=candidate,
                            temporal_forms=segment_temporal,
                            source_family=source_family,
                            source_session=source_session,
                            evidence_text=segment,
                            fact_quality=f"pattern_{source_kind}",
                        )
                    )
            date_match = _SELF_DATE_RE.search(segment)
            if date_match:
                candidate = _candidate_name_from_text(date_match.group(1))
                if _is_useful_fact_value(candidate):
                    facts.append(
                        _build_fact(
                            citation_id=citation_id,
                            index=index,
                            subject=speaker,
                            relation="dating",
                            obj=candidate,
                            temporal_forms=segment_temporal,
                            source_family=source_family,
                            source_session=source_session,
                            evidence_text=segment,
                            fact_quality=f"pattern_{source_kind}",
                        )
                    )

        third_party_match = _THIRD_PARTY_RELATION_RE.search(segment)
        if third_party_match:
            subject_candidate = _candidate_name_from_text(third_party_match.group(1))
            object_candidate = _candidate_name_from_text(third_party_match.group(2))
            if _is_useful_fact_value(subject_candidate) and _is_useful_fact_value(object_candidate):
                facts.append(
                    _build_fact(
                        citation_id=citation_id,
                        index=index,
                        subject=subject_candidate,
                        relation="dating",
                        obj=object_candidate,
                        temporal_forms=segment_temporal,
                        source_family=source_family,
                        source_session=source_session,
                        evidence_text=segment,
                        fact_quality=f"pattern_{source_kind}",
                    )
                )

        if "age" in segment_relations and (segment_entities or speaker):
            age_match = re.search(r"\b(\d{1,3})\b", segment)
            if age_match:
                subject = segment_entities[0] if segment_entities else speaker
                facts.append(
                    _build_fact(
                        citation_id=citation_id,
                        index=index,
                        subject=subject,
                        relation="age",
                        obj=age_match.group(1),
                        temporal_forms=segment_temporal,
                        source_family=source_family,
                        source_session=source_session,
                        evidence_text=segment,
                        fact_quality=f"heuristic_{source_kind}",
                    )
                )
                continue

        subject = ""
        obj = ""
        if len(segment_entities) >= 2:
            subject = segment_entities[0]
            obj = segment_entities[1]
        elif speaker and segment_entities:
            first = segment_entities[0]
            if first.lower() == speaker.lower() and len(segment_entities) >= 2:
                subject = first
                obj = segment_entities[1]
            else:
                subject = speaker
                obj = first
        if not _is_useful_fact_value(subject) or not _is_useful_fact_value(obj):
            continue
        for relation in segment_relations[:2]:
            key = (subject, relation, obj, tuple(segment_temporal[:2]), citation_id)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                _build_fact(
                    citation_id=citation_id,
                    index=index,
                    subject=subject,
                    relation=relation,
                    obj=obj,
                    temporal_forms=segment_temporal,
                    source_family=source_family,
                    source_session=source_session,
                    evidence_text=segment,
                    fact_quality=f"heuristic_{source_kind}",
                )
            )

    if not facts and len(entities) >= 2 and relation_terms:
        subject = entities[0]
        obj = entities[1]
        for index, relation in enumerate(relation_terms[:2]):
            facts.append(
                _build_fact(
                    citation_id=citation_id,
                    index=index,
                    subject=subject,
                    relation=relation,
                    obj=obj,
                    temporal_forms=temporal_forms,
                    source_family=source_family,
                    source_session=source_session,
                    evidence_text=_clean_text(page.get("summary") or page.get("raw_content")),
                    fact_quality="fallback",
                )
            )

    return facts[:12]


def _score_fact(canonical_query: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, float]:
    query_entities = {value.lower() for value in canonical_query.get("entities", [])}
    query_temporal = set(canonical_query.get("temporal_forms", []))
    query_relations = set(canonical_query.get("relation_terms", []))
    fact_entities = {str(fact.get("subject", "")).lower(), str(fact.get("object", "")).lower()}
    fact_entities.discard("")
    fact_temporal = set(fact.get("temporal_forms", []))
    fact_relations = set(fact.get("relation_terms", []))

    entity_match = 0.0
    if query_entities and fact_entities:
        entity_match = len(query_entities & fact_entities) / float(len(query_entities))

    temporal_match = 0.0
    if query_temporal and fact_temporal:
        temporal_match = len(query_temporal & fact_temporal) / float(len(query_temporal))

    relation_match = 0.0
    if query_relations and fact_relations:
        relation_match = len(query_relations & fact_relations) / float(len(query_relations))

    score = (0.4 * entity_match) + (0.35 * relation_match) + (0.25 * temporal_match)
    score += _fact_quality_bonus(str(fact.get("fact_quality") or ""))
    if query_temporal and temporal_match <= 0.0:
        score -= 0.08
    if query_relations and relation_match <= 0.0:
        score -= 0.08
    if canonical_query.get("question_shape") == "two_hop_relation_lookup":
        if "roommate" in fact_relations:
            score += 0.08
        if "dating" in fact_relations:
            score += 0.04
    return {
        "entity_match": round(entity_match, 4),
        "temporal_match": round(temporal_match, 4),
        "relation_match": round(relation_match, 4),
        "score": round(max(score, 0.0), 4),
    }


def _infer_reasoning_mode(canonical_query: Dict[str, Any]) -> str:
    shape = str(canonical_query.get("question_shape") or "")
    if shape == "two_hop_relation_lookup":
        return "two_hop_relation"
    if canonical_query.get("temporal_forms") and canonical_query.get("relation_terms"):
        return "time_filtered_lookup"
    if canonical_query.get("relation_terms"):
        return "direct_lookup"
    return "evidence_review"


def build_canonical_evidence_trace(
    query: str,
    pages: Sequence[Dict[str, Any]],
    *,
    task_hint: str = "",
    answer_format: str = "",
) -> Dict[str, Any]:
    canonical_query = canonicalize_query(query, task_hint=task_hint, answer_format=answer_format)
    fact_tuples: List[Dict[str, Any]] = []
    for page in pages or []:
        if isinstance(page, dict):
            fact_tuples.extend(extract_fact_tuples(page))

    scored_facts = []
    for fact in fact_tuples:
        metrics = _score_fact(canonical_query, fact)
        scored_fact = dict(fact)
        scored_fact.update(metrics)
        scored_facts.append(scored_fact)
    scored_facts.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    top_facts: List[Dict[str, Any]] = []
    query_relations = [str(value) for value in canonical_query.get("relation_terms", []) if str(value).strip()]
    chain_resolution = {}
    if canonical_query.get("question_shape") == "two_hop_relation_lookup" and query_relations:
        chain_resolution = _resolve_two_hop_chain(canonical_query, scored_facts)
        seen_fact_ids = set()
        for fact in (chain_resolution.get("roommate_fact"), chain_resolution.get("dating_fact")):
            if not isinstance(fact, dict):
                continue
            fact_id = str(fact.get("fact_id") or "")
            if not fact_id or fact_id in seen_fact_ids:
                continue
            top_facts.append(fact)
            seen_fact_ids.add(fact_id)
        for relation in query_relations:
            for fact in scored_facts:
                if relation not in fact.get("relation_terms", []):
                    continue
                fact_id = str(fact.get("fact_id") or "")
                if fact_id in seen_fact_ids:
                    continue
                top_facts.append(fact)
                seen_fact_ids.add(fact_id)
                break
        for fact in scored_facts:
            if len(top_facts) >= 3:
                break
            fact_id = str(fact.get("fact_id") or "")
            if fact_id in seen_fact_ids:
                continue
            top_facts.append(fact)
            seen_fact_ids.add(fact_id)
    else:
        top_facts = scored_facts[:3]
    resolved_candidates: List[str] = []
    chain_candidate = _clean_text(chain_resolution.get("resolved_candidate", "")) if chain_resolution else ""
    if chain_candidate:
        resolved_candidates.append(chain_candidate)
    include_top_fact_candidates = canonical_query.get("question_shape") != "two_hop_relation_lookup" or bool(chain_candidate)
    if include_top_fact_candidates:
        for fact in top_facts:
            candidate = _clean_text(fact.get("object"))
            if candidate and candidate not in resolved_candidates:
                resolved_candidates.append(candidate)

    top_confidence = float(top_facts[0].get("score", 0.0)) if top_facts else 0.0
    if chain_candidate:
        top_confidence = max(top_confidence, float(chain_resolution.get("score", 0.0)) / 2.0)
    relation_hit = any(float(fact.get("relation_match", 0.0)) > 0.0 for fact in top_facts)
    temporal_needed = bool(canonical_query.get("temporal_forms"))
    temporal_hit = any(float(fact.get("temporal_match", 0.0)) > 0.0 for fact in top_facts)
    relation_coverage = {
        relation: any(relation in fact.get("relation_terms", []) for fact in top_facts)
        for relation in query_relations
    }
    abstain_recommended = bool(
        (top_confidence < 0.55)
        or (canonical_query.get("relation_terms") and not relation_hit)
        or (temporal_needed and not temporal_hit)
        or (canonical_query.get("question_shape") == "two_hop_relation_lookup" and not all(relation_coverage.values()))
    )

    return {
        "canonical_query": canonical_query,
        "fact_tuples": scored_facts,
        "reasoning_mode": _infer_reasoning_mode(canonical_query),
        "top_fact_ids": [str(fact.get("fact_id") or "") for fact in top_facts if str(fact.get("fact_id") or "")],
        "resolved_candidates": resolved_candidates,
        "confidence": round(top_confidence, 4),
        "abstain_recommended": abstain_recommended,
        "relation_coverage": relation_coverage,
        "chain_resolution": {
            "pivot": _clean_text(chain_resolution.get("pivot", "")) if chain_resolution else "",
            "resolved_candidate": chain_candidate,
            "score": round(float(chain_resolution.get("score", 0.0)), 4) if chain_resolution else 0.0,
            "status": str(chain_resolution.get("status", "")) if chain_resolution else "",
            "pivot_count": int(chain_resolution.get("pivot_count", 0) or 0) if chain_resolution else 0,
        },
        "fact_count": len(scored_facts),
    }
