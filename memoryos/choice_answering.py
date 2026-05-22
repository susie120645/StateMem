from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from .answer_routing import normalize_generated_answer
except (ImportError, ValueError):
    from answer_routing import normalize_generated_answer


_OPTION_LINE_RE = re.compile(r"^\s*\(?([A-Z])\)?(?:[.)])?\s+(.*\S)\s*$")
_ABSTAIN_OPTION_PATTERNS = (
    re.compile(r"\bi\s+don't\s+know\b", re.I),
    re.compile(r"\bdo\s+not\s+know\b", re.I),
    re.compile(r"\bnone\s+of\s+the\s+above\b", re.I),
    re.compile(r"\bnot\s+enough\s+information\b", re.I),
    re.compile(r"\binsufficient\s+information\b", re.I),
    re.compile(r"\bunknown\b", re.I),
)
_TIME_ANCHOR_PATTERNS = (
    re.compile(r"\b(?:when|before|after|earlier|later|during|as of|back in)\b", re.I),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)
_ANCHOR_WORD_PATTERNS = (
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)
_FAMILY_SUFFIX_PATTERNS = (
    re.compile(r"(?:[_-](?:c|chunk|span)\d+)+$", re.I),
    re.compile(r"(?:[_-]s\d+)+$", re.I),
)
_RELATION_PATTERNS = {
    "roommate": re.compile(r"\broommate|bunking with|lives with\b", re.I),
    "neighbor": re.compile(r"\bnext door|neighbor|lives next door\b", re.I),
    "dating": re.compile(r"\bdate|dating|went out with|boyfriend|girlfriend\b", re.I),
    "work": re.compile(r"\bwork(?:ing)?|job|worked|works at|doing for work\b", re.I),
    "family": re.compile(r"\bmother|father|mom|dad|son|daughter|brother|sister|kid|child\b", re.I),
    "gift_object": re.compile(r"\bring|gift|present|brought|gave|handed\b", re.I),
    "residence": re.compile(r"\blive|lives|living|apartment|house|home\b", re.I),
}
_OVERLAP_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "there",
    "have", "been", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "would", "could", "should", "about", "after", "before", "back", "again", "none",
    "above", "dont", "know", "your", "just", "really", "they", "them", "then", "than",
    "back", "time", "chat", "question", "answer", "option", "options",
}


@dataclass(frozen=True)
class ChoiceOption:
    label: str
    marker: str
    text: str


@dataclass(frozen=True)
class ParsedChoiceQuestion:
    stem: str
    options: Sequence[ChoiceOption]

    def marker_for(self, label: str) -> str:
        normalized = str(label or "").strip().upper()
        for option in self.options:
            if option.label == normalized:
                return option.marker
        return f"({normalized})" if normalized else ""


def parse_choice_question(question_text: str) -> ParsedChoiceQuestion:
    lines = [line.rstrip() for line in str(question_text or "").splitlines() if line.strip()]
    stem_lines: List[str] = []
    options: List[ChoiceOption] = []
    saw_option = False

    for line in lines:
        match = _OPTION_LINE_RE.match(line)
        if match:
            saw_option = True
            label = match.group(1).upper()
            options.append(
                ChoiceOption(
                    label=label,
                    marker=f"({label})",
                    text=match.group(2).strip(),
                )
            )
            continue

        if not saw_option:
            stem_lines.append(line.strip())
        elif options:
            last = options[-1]
            options[-1] = ChoiceOption(
                label=last.label,
                marker=last.marker,
                text=f"{last.text} {line.strip()}".strip(),
            )

    if options:
        return ParsedChoiceQuestion(stem="\n".join(stem_lines).strip(), options=tuple(options))

    normalized = str(question_text or "")
    split_match = re.search(r"(?is)\boptions\s*:\s*(.+)$", normalized)
    if not split_match:
        return ParsedChoiceQuestion(stem=" ".join(stem_lines).strip() or normalized.strip(), options=tuple())

    stem = normalized[: split_match.start()].strip()
    raw_options = split_match.group(1).strip()
    items = [item.strip() for item in re.split(r"\s*[|;]\s*", raw_options) if item.strip()]
    parsed_options = [
        ChoiceOption(label=chr(ord("A") + index), marker=f"({chr(ord('A') + index)})", text=item)
        for index, item in enumerate(items[:26])
    ]
    return ParsedChoiceQuestion(stem=stem, options=tuple(parsed_options))


def _resolve_choice_label(value: Any, parsed: ParsedChoiceQuestion) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""

    normalized = normalize_generated_answer(candidate, "structured", answer_format="choice").strip().upper()
    if len(normalized) == 1:
        for option in parsed.options:
            if option.label == normalized:
                return normalized

    lowered = candidate.lower()
    for option in parsed.options:
        if lowered == option.marker.lower() or lowered == option.text.lower():
            return option.label
    return ""


def _extract_best_choice(payload: Dict[str, Any], parsed: ParsedChoiceQuestion) -> str:
    for key in ("best_choice", "choice", "answer", "best_option", "selected_option"):
        label = _resolve_choice_label(payload.get(key), parsed)
        if label:
            return label

    option_scores = payload.get("option_scores")
    if isinstance(option_scores, dict):
        ranked: List[tuple[float, str]] = []
        for raw_key, raw_value in option_scores.items():
            label = _resolve_choice_label(raw_key, parsed)
            if not label and isinstance(raw_value, dict):
                label = _resolve_choice_label(
                    raw_value.get("label") or raw_value.get("choice") or raw_value.get("answer"),
                    parsed,
                )
            if not label:
                continue
            score = 0.0
            if isinstance(raw_value, dict):
                for score_key in ("score", "confidence", "support"):
                    try:
                        score = float(raw_value.get(score_key, 0.0))
                        break
                    except Exception:
                        continue
            else:
                try:
                    score = float(raw_value)
                except Exception:
                    score = 0.0
            ranked.append((score, label))
        if ranked:
            ranked.sort(reverse=True)
            return ranked[0][1]

    return ""


def _extract_option_scores(payload: Dict[str, Any], parsed: ParsedChoiceQuestion) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    option_scores = payload.get("option_scores")
    if not isinstance(option_scores, dict):
        return scores
    for raw_key, raw_value in option_scores.items():
        label = _resolve_choice_label(raw_key, parsed)
        if not label and isinstance(raw_value, dict):
            label = _resolve_choice_label(
                raw_value.get("label") or raw_value.get("choice") or raw_value.get("answer"),
                parsed,
            )
        if not label:
            continue
        score = 0.0
        if isinstance(raw_value, dict):
            for score_key in ("score", "confidence", "support"):
                try:
                    score = float(raw_value.get(score_key, 0.0))
                    break
                except Exception:
                    continue
        else:
            try:
                score = float(raw_value)
            except Exception:
                score = 0.0
        scores[label] = score
    return scores


def _extract_json_object_slice(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    start = cleaned.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(cleaned[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return cleaned[start:]


def _content_terms(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9']+", str(text or ""))
        if len(token) >= 3
    }
    return {token for token in tokens if token not in _OVERLAP_STOPWORDS}


def _option_overlap_ratio(option_text: str, context_text: str) -> float:
    option_terms = _content_terms(option_text)
    if not option_terms:
        return 0.0
    context_terms = _content_terms(context_text)
    if not context_terms:
        return 0.0
    overlap = len(option_terms & context_terms)
    return overlap / max(1, len(option_terms))


def _extract_anchor_terms(text: str) -> set[str]:
    terms: set[str] = set()
    normalized = str(text or "")
    for pattern in _TIME_ANCHOR_PATTERNS:
        for match in pattern.findall(normalized):
            if isinstance(match, tuple):
                match = next((item for item in match if item), "")
            if match:
                terms.add(str(match).lower())
    for match in _ANCHOR_WORD_PATTERNS:
        for item in match.findall(normalized):
            cleaned = " ".join(str(item).split()).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in _OVERLAP_STOPWORDS:
                continue
            terms.add(lowered)
    for token in re.findall(r"[A-Za-z0-9']+", normalized):
        lowered = token.lower()
        if len(lowered) >= 3 and lowered not in _OVERLAP_STOPWORDS:
            terms.add(lowered)
    return terms


def _anchor_overlap_ratio(anchor_terms: set[str], context_text: str) -> float:
    if not anchor_terms:
        return 0.0
    context_terms = _content_terms(context_text)
    if not context_terms:
        return 0.0
    overlap = len(anchor_terms & context_terms)
    return overlap / max(1, len(anchor_terms))


def _citation_family(citation_id: str) -> str:
    family = str(citation_id or "").strip()
    if not family:
        return ""
    for pattern in _FAMILY_SUFFIX_PATTERNS:
        family = pattern.sub("", family)
    family = re.sub(r"([_-](?:c|chunk|span|s)\d+)$", "", family, flags=re.I)
    return family.strip()


def _source_family_stats(citation_ids: Sequence[str]) -> Dict[str, Any]:
    families = [_citation_family(citation_id) for citation_id in citation_ids if str(citation_id or "").strip()]
    families = [family for family in families if family]
    if not families:
        return {
            "family_count": 0,
            "top_family": "",
            "top_family_share": 0.0,
            "family_ids": [],
        }
    counts: Dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_family, top_count = ranked[0]
    total = len(families)
    return {
        "family_count": len(counts),
        "top_family": top_family,
        "top_family_share": round(top_count / total, 4),
        "family_ids": [family for family, _ in ranked],
    }


def _relation_hits(text: str) -> set[str]:
    normalized = str(text or "")
    return {
        name
        for name, pattern in _RELATION_PATTERNS.items()
        if pattern.search(normalized)
    }


def _fallback_extract_choice_payload(raw_response: str) -> Dict[str, Any]:
    cleaned = _extract_json_object_slice(raw_response)
    if not cleaned:
        return {}
    payload: Dict[str, Any] = {}
    best_choice = re.search(r'"best_choice"\s*:\s*"([A-E])"', cleaned, re.I)
    if best_choice:
        payload["best_choice"] = best_choice.group(1).upper()
    abstain = re.search(r'"abstain_recommended"\s*:\s*(true|false)', cleaned, re.I)
    if abstain:
        payload["abstain_recommended"] = abstain.group(1).lower() == "true"
    option_scores: Dict[str, float] = {}
    scores_match = re.search(r'"option_scores"\s*:\s*\{(.*)', cleaned, re.I | re.S)
    if scores_match:
        tail = scores_match.group(1)
        for label, score_text in re.findall(r'"([A-E])"\s*:\s*([0-9]*\.?[0-9]+)', tail, re.I):
            try:
                option_scores[label.upper()] = float(score_text)
            except Exception:
                continue
    if option_scores:
        payload["option_scores"] = option_scores
    return payload


def _abstain_labels(parsed: ParsedChoiceQuestion) -> set[str]:
    labels = set()
    for option in parsed.options:
        if any(pattern.search(option.text) for pattern in _ABSTAIN_OPTION_PATTERNS):
            labels.add(option.label)
    return labels


def _build_choice_prompt(
    *,
    parsed_question: ParsedChoiceQuestion,
    context_blocks: Sequence[str],
    allowed_ids: Sequence[str],
    extra_guidance: Sequence[str] = (),
    task_instruction: str = "",
) -> str:
    option_lines = "\n".join(f"{option.marker} {option.text}" for option in parsed_question.options)
    guidance_block = "\n".join(f"- {line}" for line in extra_guidance if str(line).strip())
    return (
        "You are selecting the single best answer option using only the retrieved context.\n"
        "Return compact JSON with keys best_choice, option_scores, abstain_recommended, citations.\n"
        f"best_choice must be exactly one label from: {', '.join(option.label for option in parsed_question.options)}.\n"
        "option_scores must map each label to a support score between 0 and 1.\n"
        "abstain_recommended must be true only when the retrieved context does not support any factual option strongly enough.\n"
        "If one option explicitly means insufficient evidence or unknown, choose it only when the retrieved context does not support any factual option.\n"
        + (f"Task instruction:\n{str(task_instruction).strip()}\n" if str(task_instruction).strip() else "")
        + (f"Allowed citation ids: {list(allowed_ids)}\n" if allowed_ids else "")
        + f"Question:\n{parsed_question.stem}\n\n"
        + f"Options:\n{option_lines}\n\n"
        + (f"Adaptive guidance:\n{guidance_block}\n\n" if guidance_block else "")
        + "Retrieved context:\n"
        + "\n".join(context_blocks)
    )


def _build_choice_context_blocks(
    pages: Sequence[Dict[str, Any]],
    *,
    allowed_citation_ids: Optional[Sequence[str]] = None,
) -> tuple[List[str], List[str]]:
    context_blocks: List[str] = []
    allowed_ids = [str(item).strip() for item in (allowed_citation_ids or []) if str(item).strip()]
    for index, page in enumerate(pages or []):
        citation_id = str(page.get("citation_id") or page.get("page_id") or f"ctx_{index+1}").strip()
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if not raw:
            continue
        context_blocks.append(f"[{citation_id}]\n{raw}")
        if citation_id and citation_id not in allowed_ids:
            allowed_ids.append(citation_id)
    return context_blocks, allowed_ids


def _run_choice_model(
    *,
    client: Any,
    model: str,
    parsed_question: ParsedChoiceQuestion,
    pages: Sequence[Dict[str, Any]],
    max_tokens: int,
    extra_guidance: Sequence[str] = (),
    allowed_citation_ids: Optional[Sequence[str]] = None,
    task_instruction: str = "",
) -> tuple[Dict[str, Any], str, List[str]]:
    context_blocks, allowed_ids = _build_choice_context_blocks(
        pages,
        allowed_citation_ids=allowed_citation_ids,
    )
    if not context_blocks:
        return {}, "", allowed_ids

    prompt = _build_choice_prompt(
        parsed_question=parsed_question,
        context_blocks=context_blocks,
        allowed_ids=allowed_ids,
        extra_guidance=extra_guidance,
        task_instruction=task_instruction,
    )

    raw_response = ""
    payload: Dict[str, Any] = {}
    try:
        raw_response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw_response or "{}")
    except Exception:
        payload = _fallback_extract_choice_payload(raw_response)
    return payload, raw_response, allowed_ids


def _build_reason_payload(
    *,
    parsed_question: ParsedChoiceQuestion,
    pages: Sequence[Dict[str, Any]],
    option_scores: Dict[str, float],
    best_choice: str,
    abstain_labels: set[str],
    abstain_recommended: bool,
    support_margin: float,
    citations: Sequence[str],
    question_text: str,
) -> Dict[str, Any]:
    reason_codes: List[str] = []
    if abstain_labels:
        reason_codes.append("abstain_option_present")
    if not option_scores:
        reason_codes.append("scores_missing")
    elif len(option_scores) < len(parsed_question.options):
        reason_codes.append("partial_scores")
    else:
        reason_codes.append("scores_complete")

    if best_choice in abstain_labels:
        reason_codes.append("best_choice_abstain")
    elif best_choice:
        reason_codes.append("best_choice_non_abstain")

    if support_margin >= 0.75:
        reason_codes.append("strong_margin")
    elif support_margin >= 0.35:
        reason_codes.append("moderate_margin")
    elif support_margin > 0.0:
        reason_codes.append("weak_margin")
    else:
        reason_codes.append("zero_margin")

    if citations:
        reason_codes.append("citations_present")
    else:
        reason_codes.append("citations_missing")

    page_count = len(pages or [])
    context_text = "\n".join(
        str(page.get("raw_content") or page.get("summary") or "").strip()
        for page in (pages or [])
    ).strip()
    context_chars = len(context_text)
    if page_count <= 2 or context_chars < 500:
        reason_codes.append("context_sparse")
    elif page_count >= 8 or context_chars >= 2500:
        reason_codes.append("context_dense")
    else:
        reason_codes.append("context_mid")

    question_stem_text = parsed_question.stem or str(question_text or "")
    if any(pattern.search(question_stem_text) for pattern in _TIME_ANCHOR_PATTERNS):
        reason_codes.append("time_anchor_in_question")

    question_anchor_terms = _extract_anchor_terms(question_stem_text)
    question_time_terms = {
        term
        for term in question_anchor_terms
        if any(pattern.search(term) for pattern in _TIME_ANCHOR_PATTERNS)
    }
    question_entity_terms = question_anchor_terms - question_time_terms
    question_time_overlap = _anchor_overlap_ratio(question_time_terms, context_text)
    question_entity_overlap = _anchor_overlap_ratio(question_entity_terms, context_text)
    if question_time_terms:
        if question_time_overlap >= 0.5:
            reason_codes.append("question_time_anchor_aligned")
        else:
            reason_codes.append("question_time_anchor_missing")
    if question_entity_terms:
        if question_entity_overlap >= 0.5:
            reason_codes.append("question_entity_anchor_aligned")
        else:
            reason_codes.append("question_entity_anchor_missing")

    question_relations = _relation_hits(question_stem_text)
    context_relations = _relation_hits(context_text)
    relation_overlap = sorted(question_relations & context_relations)
    relation_missing = sorted(question_relations - context_relations)
    if question_relations:
        if relation_overlap:
            reason_codes.append("question_relation_aligned")
        if relation_missing:
            reason_codes.append("question_relation_missing")

    ranked_non_abstain = sorted(
        ((score, label) for label, score in option_scores.items() if label not in abstain_labels),
        reverse=True,
    )
    best_non_abstain_score = ranked_non_abstain[0][0] if ranked_non_abstain else 0.0
    best_non_abstain_label = ranked_non_abstain[0][1] if ranked_non_abstain else ""
    abstain_score = max((option_scores.get(label, 0.0) for label in abstain_labels), default=0.0)
    best_choice_score = option_scores.get(best_choice, 0.0)

    if best_choice in abstain_labels and best_non_abstain_label:
        if best_non_abstain_score >= max(0.35, abstain_score - 0.1):
            reason_codes.append("non_abstain_alternative_available")
    elif best_choice and abstain_labels and abstain_score >= max(0.35, best_choice_score - 0.1):
        reason_codes.append("abstain_competitor_close")

    best_option_text = ""
    for option in parsed_question.options:
        if option.label == best_choice:
            best_option_text = option.text
            break
    best_option_overlap = _option_overlap_ratio(best_option_text, context_text)
    if best_option_text:
        if best_option_overlap >= 0.5:
            reason_codes.append("best_option_lexical_overlap")
        elif best_option_overlap == 0.0:
            reason_codes.append("best_option_no_overlap")

    best_non_abstain_text = ""
    for option in parsed_question.options:
        if option.label == best_non_abstain_label:
            best_non_abstain_text = option.text
            break
    best_non_abstain_overlap = _option_overlap_ratio(best_non_abstain_text, context_text)
    if best_non_abstain_text:
        if best_non_abstain_overlap >= 0.5:
            reason_codes.append("best_non_abstain_lexical_overlap")
        elif best_non_abstain_overlap == 0.0:
            reason_codes.append("best_non_abstain_no_overlap")

    citation_families = _source_family_stats(citations or [])
    if citation_families["family_count"] >= 2 and citation_families["top_family_share"] >= 0.66:
        reason_codes.append("support_source_concentrated")
    elif citation_families["family_count"] >= 2:
        reason_codes.append("support_source_scattered")
    elif citation_families["family_count"] == 1:
        reason_codes.append("support_source_single_family")

    if abstain_recommended:
        reason_codes.append("abstain_recommended")

    return {
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "best_choice_score": round(float(best_choice_score), 4),
        "best_non_abstain_choice": parsed_question.marker_for(best_non_abstain_label) if best_non_abstain_label else "",
        "best_non_abstain_score": round(float(best_non_abstain_score), 4),
        "best_non_abstain_overlap": round(float(best_non_abstain_overlap), 4),
        "abstain_score": round(float(abstain_score), 4),
        "best_option_overlap": round(float(best_option_overlap), 4),
        "question_time_overlap": round(float(question_time_overlap), 4),
        "question_entity_overlap": round(float(question_entity_overlap), 4),
        "question_relations": sorted(question_relations),
        "context_relations": sorted(context_relations),
        "relation_overlap": relation_overlap,
        "relation_missing": relation_missing,
        **citation_families,
        "page_count": page_count,
        "context_chars": context_chars,
    }


def answer_choice_with_context(
    *,
    client: Any,
    model: str,
    question_text: str,
    pages: Sequence[Dict[str, Any]],
    max_tokens: int = 220,
    extra_guidance: Sequence[str] = (),
    allowed_citation_ids: Optional[Sequence[str]] = None,
    task_instruction: str = "",
    prefer_supported_choice: bool = False,
) -> Dict[str, Any]:
    parsed_question = parse_choice_question(question_text)
    if len(parsed_question.options) < 2:
        return {}

    payload, raw_response, allowed_ids = _run_choice_model(
        client=client,
        model=model,
        parsed_question=parsed_question,
        pages=pages,
        max_tokens=max_tokens,
        extra_guidance=extra_guidance,
        allowed_citation_ids=allowed_citation_ids,
        task_instruction=task_instruction,
    )

    selected_label = _extract_best_choice(payload, parsed_question)
    if not selected_label:
        selected_label = _resolve_choice_label(raw_response, parsed_question)
    if not selected_label:
        return {}

    option_scores = _extract_option_scores(payload, parsed_question)
    abstain_labels = _abstain_labels(parsed_question)
    if prefer_supported_choice and selected_label in abstain_labels and option_scores:
        abstain_score = option_scores.get(selected_label, 0.0)
        supported = [
            (score, label)
            for label, score in option_scores.items()
            if label not in abstain_labels
        ]
        if supported:
            supported.sort(reverse=True)
            best_score, best_label = supported[0]
            if best_score >= max(0.35, abstain_score - 0.05):
                selected_label = best_label

    citations = []
    if allowed_ids:
        citations = [
            citation_id
            for citation_id in payload.get("citations", [])
            if isinstance(citation_id, str) and citation_id in allowed_ids
        ]
    thought_process = str(payload.get("rationale") or payload.get("thought_process") or "").strip()
    if not thought_process:
        thought_process = "Compared the available options against the retrieved evidence."

    return {
        "thought_process": thought_process,
        "final_answer": parsed_question.marker_for(selected_label),
        "citations": citations,
        "raw_model_output": raw_response,
        "choice_label": selected_label,
        "option_scores": option_scores,
        "choice_options": [
            {"label": option.label, "marker": option.marker, "text": option.text}
            for option in parsed_question.options
        ],
    }


def score_choice_support_with_context(
    *,
    client: Any,
    model: str,
    question_text: str,
    pages: Sequence[Dict[str, Any]],
    max_tokens: int = 220,
    allowed_citation_ids: Optional[Sequence[str]] = None,
    task_instruction: str = "",
) -> Dict[str, Any]:
    parsed_question = parse_choice_question(question_text)
    if len(parsed_question.options) < 2:
        return {}

    payload, raw_response, allowed_ids = _run_choice_model(
        client=client,
        model=model,
        parsed_question=parsed_question,
        pages=pages,
        max_tokens=max_tokens,
        allowed_citation_ids=allowed_citation_ids,
        task_instruction=task_instruction,
    )
    option_scores = _extract_option_scores(payload, parsed_question)
    best_choice = _extract_best_choice(payload, parsed_question)
    if not best_choice:
        best_choice = _resolve_choice_label(raw_response, parsed_question)
    abstain_labels = _abstain_labels(parsed_question)
    abstain_recommended = bool(payload.get("abstain_recommended"))
    if not abstain_recommended and best_choice in abstain_labels:
        abstain_recommended = True

    ranked = sorted(
        ((score, label) for label, score in option_scores.items()),
        reverse=True,
    )
    support_margin = 0.0
    if len(ranked) >= 2:
        support_margin = max(0.0, ranked[0][0] - ranked[1][0])

    citations = []
    if allowed_ids:
        citations = [
            citation_id
            for citation_id in payload.get("citations", [])
            if isinstance(citation_id, str) and citation_id in allowed_ids
        ]

    reason_payload = _build_reason_payload(
        parsed_question=parsed_question,
        pages=pages,
        option_scores=option_scores,
        best_choice=best_choice,
        abstain_labels=abstain_labels,
        abstain_recommended=abstain_recommended,
        support_margin=support_margin,
        citations=citations,
        question_text=question_text,
    )

    return {
        "best_choice": parsed_question.marker_for(best_choice) if best_choice else "",
        "option_scores": option_scores,
        "support_margin": round(float(support_margin), 4),
        "abstain_recommended": abstain_recommended,
        "citations": citations,
        "raw_model_output": raw_response[:1000],
        **reason_payload,
        "choice_options": [
            {"label": option.label, "marker": option.marker, "text": option.text}
            for option in parsed_question.options
        ],
    }


def maybe_override_abstaining_choice_answer(
    prediction_text: str,
    choice_support: Dict[str, Any],
    *,
    min_margin: float = 0.7,
) -> Dict[str, Any]:
    current_label = _resolve_choice_label(str(prediction_text or ""), ParsedChoiceQuestion(stem="", options=tuple(
        ChoiceOption(label=chr(ord("A") + index), marker=f"({chr(ord('A') + index)})", text="")
        for index in range(5)
    )))
    best_choice_marker = str(choice_support.get("best_choice") or "").strip()
    best_choice_label = _resolve_choice_label(best_choice_marker, ParsedChoiceQuestion(stem="", options=tuple(
        ChoiceOption(label=chr(ord("A") + index), marker=f"({chr(ord('A') + index)})", text="")
        for index in range(5)
    )))
    try:
        support_margin = float(choice_support.get("support_margin", 0.0) or 0.0)
    except Exception:
        support_margin = 0.0

    applied = False
    reason = ""
    final_prediction = str(prediction_text or "")

    if current_label == "E" and best_choice_label and best_choice_label != "E":
        if not bool(choice_support.get("abstain_recommended")) and support_margin >= float(min_margin):
            final_prediction = best_choice_marker or f"({best_choice_label})"
            applied = True
            reason = "non_abstain_choice_support"

    return {
        "prediction_text": final_prediction,
        "applied": applied,
        "reason": reason,
        "from_label": current_label,
        "to_label": best_choice_label if applied else current_label,
        "support_margin": round(support_margin, 4),
    }
