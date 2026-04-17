import ast
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

try:
    from ..entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate
except (ImportError, ValueError):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    from entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate


ENTITY_TYPE_SLOT_MAP = {
    "gpe": "LOCATION",
    "location": "LOCATION",
    "location/place": "LOCATION",
    "place": "LOCATION",
    "city": "LOCATION",
    "country": "LOCATION",
    "state": "LOCATION",
    "province": "LOCATION",
    "date": "TIME",
    "time": "TIME",
    "datetime": "TIME",
    "duration": "TIME",
    "year": "TIME",
    "month": "TIME",
    "product": "PRODUCT",
    "brand": "PRODUCT",
    "model": "PRODUCT",
    "device": "PRODUCT",
    "food": "FOOD",
    "dish": "FOOD",
    "meal": "FOOD",
    "drink": "FOOD",
    "beverage": "FOOD",
}

LOCATION_HINT_TOKENS = {
    "airport",
    "avenue",
    "beach",
    "boulevard",
    "city",
    "country",
    "county",
    "district",
    "island",
    "lake",
    "mount",
    "mountain",
    "park",
    "plaza",
    "province",
    "river",
    "road",
    "state",
    "station",
    "street",
    "town",
    "university",
    "village",
}

PRODUCT_HINT_TOKENS = {
    "app",
    "brand",
    "camera",
    "device",
    "gadget",
    "headphones",
    "laptop",
    "model",
    "phone",
    "product",
    "software",
    "tablet",
    "tool",
    "watch",
}

FOOD_HINT_TOKENS = {
    "beverage",
    "breakfast",
    "burger",
    "cake",
    "cafe",
    "coffee",
    "dessert",
    "dinner",
    "dish",
    "drink",
    "food",
    "lunch",
    "meal",
    "pasta",
    "pizza",
    "restaurant",
    "salad",
    "sandwich",
    "snack",
    "soup",
    "tea",
}

GAP_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "meet",
    "of",
    "on",
    "the",
    "their",
    "they",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

GRAPH_1HOP_QUESTION_ENTITY_STOP_WORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "would",
    "could",
    "should",
}

GRAPH_1HOP_SPECIFICITY_PATTERNS = (
    re.compile(r"\bwhat kind\b", re.IGNORECASE),
    re.compile(r"\bwhat type\b", re.IGNORECASE),
    re.compile(r"\bwhat project\b", re.IGNORECASE),
    re.compile(r"\bwhat game\b", re.IGNORECASE),
    re.compile(r"\bwhat journal\b", re.IGNORECASE),
    re.compile(r"\bwhat gift\b", re.IGNORECASE),
    re.compile(r"\bwhat country\b", re.IGNORECASE),
    re.compile(r"\bwhich country\b", re.IGNORECASE),
    re.compile(r"\bwhich city\b", re.IGNORECASE),
    re.compile(r"\bwhich year\b", re.IGNORECASE),
    re.compile(r"\bwhat are the names\b", re.IGNORECASE),
    re.compile(r"\bhow long\b", re.IGNORECASE),
)

GRAPH_1HOP_RELATION_PATTERNS = (
    re.compile(r"\bboth\b", re.IGNORECASE),
    re.compile(r"\btogether\b", re.IGNORECASE),
    re.compile(r"\bbetween\b", re.IGNORECASE),
    re.compile(r"\bplan to\b", re.IGNORECASE),
    re.compile(r"\bworking on\b", re.IGNORECASE),
    re.compile(r"\bmet\b", re.IGNORECASE),
    re.compile(r"\bgift(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bfrom the same\b", re.IGNORECASE),
)

GRAPH_1HOP_NEGATIVE_PATTERNS = (
    re.compile(r"^\s*(?:would|could|should)\b", re.IGNORECASE),
    re.compile(r"\blikely\b", re.IGNORECASE),
    re.compile(r"\bconsidered\b", re.IGNORECASE),
)

TIME_PATTERNS = (
    re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", re.IGNORECASE),
    re.compile(r"\b(?:in|during|on)\s+\d{4}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:today|yesterday|tomorrow|tonight|this morning|this evening|"
        r"last night|last week|last month|last year|next week|next month|next year|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
)

LOCATION_PATTERNS = (
    re.compile(
        r"\b(?:located|based|born|grew up|lives|live|living|stayed|stay|works|work|"
        r"married|met|traveled|travel|moved|move|visited|visit|arrived|arrive|"
        r"headed|head)\s+(?:in|at|from|to|near)\s+[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,4}"
        r"(?:,\s*[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2})?",
    ),
    re.compile(
        r"\b(?:in|at|from|to|near)\s+[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,4}"
        r"(?:,\s*[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2})?",
    ),
)


class Retriever:
    def __init__(self, short_term_memory, long_term_memory, forget_memory, queue_capacity: int = 25):
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.forget_memory = forget_memory
        self.queue_capacity = queue_capacity
        self.tracer = None

    def _is_structural_entity_label(self, candidate: str) -> bool:
        return is_structural_entity_label(candidate)

    def _filter_entities(self, values: List[Any]) -> List[str]:
        return filter_entity_values([normalize_entity_candidate(value) for value in values or []])

    def _estimate_page_tokens(self, page: Dict[str, Any]) -> int:
        content = page.get("raw_content") or f"{page.get('user_input', '')}\n{page.get('agent_response', '')}"
        return max(1, len(content) // 4)

    def _calc_final_score(self, page: Dict[str, Any]) -> float:
        relevance = float(page.get("score", page.get("relevance", 0.0)))
        utility = float(page.get("G", 0.0))
        confidence = float(page.get("C", 0.6))
        temperature = float(page.get("T", 0.0))
        salience_seed = float(page.get("salience_seed", 0.5))
        t_norm = max(0.05, min(1.0, temperature / 100.0))
        memory_factor = (1.0 + math.log1p(max(0.0, utility))) * confidence * t_norm * (0.5 + salience_seed)
        estimated_tokens = self._estimate_page_tokens(page)
        length_factor = max(0.3, min(1.0, math.sqrt(320.0 / max(320.0, float(estimated_tokens)))))
        slot_multiplier = float(page.get("slot_rerank_multiplier", 1.0))
        gap_multiplier = float(page.get("gap_rerank_multiplier", 1.0))
        return round(relevance * memory_factor * length_factor * slot_multiplier * gap_multiplier, 6)

    def _extract_entities(self, text: str) -> List[str]:
        extractor = getattr(self, "entity_extractor", None)
        if callable(extractor):
            try:
                entities = extractor(text)
                return self._filter_entities(entities)
            except Exception:
                return []
        entities = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{1,}\b", text or "")
        return self._filter_entities(list(dict.fromkeys(entities)))

    def _parse_raw_content(self, raw_content: str, fallback_summary: str = "") -> Dict[str, str]:
        if "\nA:" in raw_content:
            user_part, answer_part = raw_content.split("\nA:", 1)
            return {
                "user_input": user_part.replace("Q:", "", 1).strip(),
                "agent_response": answer_part.strip(),
            }
        return {"user_input": fallback_summary or raw_content, "agent_response": raw_content}

    def _detect_expected_slot(self, query: str) -> Optional[str]:
        lowered = (query or "").strip().lower()
        if not lowered:
            return None
        if lowered.startswith("where") or any(
            phrase in lowered
            for phrase in ("what city", "which city", "what country", "which country", "what place", "which place")
        ):
            return "LOCATION"
        if lowered.startswith("when") or any(
            phrase in lowered
            for phrase in ("what date", "which date", "what year", "which year", "what time", "how long")
        ):
            return "TIME"
        if any(phrase in lowered for phrase in ("what product", "which product", "what brand", "which brand", "what model", "which model")):
            return "PRODUCT"
        if any(phrase in lowered for phrase in ("what food", "which food", "what dish", "which dish", "what meal", "which meal", "what drink", "which drink")):
            return "FOOD"
        if re.search(r"\b(?:eat|ate|drink|drank|ordered|cook|cooked)\b", lowered):
            return "FOOD"
        return None

    def _page_text(self, page: Dict[str, Any]) -> str:
        return "\n".join(
            str(part or "")
            for part in (
                page.get("user_input", ""),
                page.get("agent_response", ""),
                page.get("summary", ""),
                page.get("raw_content", ""),
            )
        )

    def _collect_raw_entities(self, page: Dict[str, Any]) -> List[Any]:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        raw_entities: List[Any] = []
        for values in (
            page.get("_raw_entities"),
            page.get("entities"),
            metadata.get("entities_raw"),
            metadata.get("entities"),
        ):
            if isinstance(values, list):
                raw_entities.extend(values)
        return raw_entities

    def _coerce_entity_payload(self, value: Any) -> Dict[str, str]:
        label = ""
        entity_type = ""
        if isinstance(value, dict):
            label = str(value.get("name") or value.get("label") or value.get("text") or value.get("value") or "")
            entity_type = str(value.get("type") or value.get("entity_type") or value.get("slot") or "")
        elif isinstance(value, str):
            stripped = value.strip()
            label = stripped
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = ast.literal_eval(stripped)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    label = str(parsed.get("name") or parsed.get("label") or parsed.get("text") or parsed.get("value") or stripped)
                    entity_type = str(parsed.get("type") or parsed.get("entity_type") or parsed.get("slot") or "")
        return {
            "label": normalize_entity_candidate(label),
            "type": normalize_entity_candidate(entity_type),
        }

    def _infer_slots_from_entity(self, value: Any) -> Set[str]:
        payload = self._coerce_entity_payload(value)
        label = payload["label"]
        lowered_type = payload["type"].lower()
        slots: Set[str] = set()
        for candidate, slot_name in ENTITY_TYPE_SLOT_MAP.items():
            if lowered_type == candidate or candidate in lowered_type:
                slots.add(slot_name)
        if label:
            lowered_label = label.lower()
            label_tokens = set(re.findall(r"[a-z0-9]+", lowered_label))
            looks_like_time = any(pattern.search(label) for pattern in TIME_PATTERNS)
            looks_like_location = bool(re.search(r"[A-Za-z].*,\s*[A-Za-z]", label)) or bool(label_tokens & LOCATION_HINT_TOKENS)
            if looks_like_location and not looks_like_time:
                slots.add("LOCATION")
            if looks_like_time:
                slots.add("TIME")
            if label_tokens & PRODUCT_HINT_TOKENS:
                slots.add("PRODUCT")
            if label_tokens & FOOD_HINT_TOKENS:
                slots.add("FOOD")
        return slots

    def _infer_slots_from_text(self, text: str) -> Set[str]:
        lowered = (text or "").lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        slots: Set[str] = set()
        if any(pattern.search(text or "") for pattern in TIME_PATTERNS):
            slots.add("TIME")
        if any(pattern.search(text or "") for pattern in LOCATION_PATTERNS):
            slots.add("LOCATION")
        if tokens & PRODUCT_HINT_TOKENS:
            slots.add("PRODUCT")
        if tokens & FOOD_HINT_TOKENS:
            slots.add("FOOD")
        return slots

    def _infer_page_slots(self, page: Dict[str, Any]) -> Set[str]:
        slots = self._infer_slots_from_text(self._page_text(page))
        for entity in self._collect_raw_entities(page):
            slots.update(self._infer_slots_from_entity(entity))
        return slots

    def _build_slot_rerank(self, expected_slot: Optional[str], page: Dict[str, Any]) -> Dict[str, Any]:
        observed_slots = self._infer_page_slots(page)
        multiplier = 1.0
        reason = "slot_neutral"

        if expected_slot == "LOCATION":
            if "LOCATION" in observed_slots:
                multiplier = 1.5
                reason = "location_match"
            elif "TIME" in observed_slots:
                multiplier = 0.7
                reason = "time_only_distractor"
        elif expected_slot == "TIME":
            if "TIME" in observed_slots:
                multiplier = 1.35
                reason = "time_match"
            elif "LOCATION" in observed_slots and "TIME" not in observed_slots:
                multiplier = 0.8
                reason = "location_only_distractor"
        elif expected_slot == "PRODUCT":
            if "PRODUCT" in observed_slots:
                multiplier = 1.35
                reason = "product_match"
        elif expected_slot == "FOOD":
            if "FOOD" in observed_slots:
                multiplier = 1.35
                reason = "food_match"
            elif "PRODUCT" in observed_slots and "FOOD" not in observed_slots:
                multiplier = 0.85
                reason = "product_only_distractor"
        elif expected_slot is None:
            reason = "no_expected_slot"

        return {
            "expected_slot": expected_slot,
            "observed_slots": sorted(observed_slots),
            "multiplier": multiplier,
            "reason": reason,
        }

    def _extract_gap_query_terms(self, query: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (query or "").lower())
            if token not in GAP_QUERY_STOP_WORDS
        }

    def _session_key_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        session_group_id = metadata.get("session_group_id")
        if session_group_id:
            return str(session_group_id)
        session_id = metadata.get("session_id")
        conversation_id = metadata.get("conversation_id")
        if session_id is not None and conversation_id:
            return f"{conversation_id}:{session_id}"
        if session_id is not None:
            return f"session:{session_id}"
        return None

    def _gap_session_key(self, page: Dict[str, Any]) -> Optional[str]:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        return self._session_key_from_metadata(metadata)

    def _gap_query_overlap(self, query_terms: Set[str], page: Dict[str, Any]) -> int:
        if not query_terms:
            return 0
        text_terms = set(re.findall(r"[a-z0-9]+", self._page_text(page).lower()))
        return len(query_terms & text_terms)

    def _entry_query_overlap(self, query_terms: Set[str], entry: Dict[str, Any]) -> int:
        if not query_terms:
            return 0
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        text = "\n".join(
            str(part or "")
            for part in (
                entry.get("summary", ""),
                entry.get("raw_content", ""),
                metadata.get("topic", ""),
            )
        )
        text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(query_terms & text_terms)

    def _page_rank_score(self, page: Dict[str, Any]) -> float:
        return float(page.get("final_score", page.get("score", 0.0)))

    def _resolve_session_group_id(self, page: Dict[str, Any]) -> Optional[str]:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        session_group_id = metadata.get("session_group_id")
        if session_group_id:
            return str(session_group_id)

        for value in (
            page.get("citation_id"),
            page.get("page_id"),
            page.get("doc_id"),
            metadata.get("page_id"),
            metadata.get("doc_id"),
        ):
            text = str(value or "")
            if not text:
                continue
            match = re.match(r"(.+?)_(?:overview|transcript|window_\d+)$", text)
            if match:
                return match.group(1)
        return None

    def _window_index(self, page: Dict[str, Any]) -> Optional[int]:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        for value in (
            metadata.get("window_index"),
            page.get("window_index"),
            page.get("citation_id"),
            page.get("page_id"),
            page.get("doc_id"),
        ):
            if value in {None, ""}:
                continue
            if isinstance(value, int):
                return value if value > 0 else None
            text = str(value)
            if text.isdigit():
                parsed = int(text)
                return parsed if parsed > 0 else None
            match = re.search(r"_window_(\d+)$", text)
            if match:
                parsed = int(match.group(1))
                return parsed if parsed > 0 else None
        return None

    def _window_hit_is_extremely_high(self, page: Dict[str, Any], best_score: float) -> bool:
        if best_score <= 0.0:
            return False
        page_score = self._page_rank_score(page)
        raw_score = float(page.get("score", page_score))
        return raw_score >= 0.72 or (raw_score >= 0.58 and page_score >= (best_score * 0.9))

    def _find_best_session_window_entry(
        self,
        session_key: str,
        session_group_id: Optional[str],
        query_terms: Set[str],
        exclude_ids: Set[str],
    ) -> Optional[Dict[str, Any]]:
        best_entry = None
        best_rank = (-1, -1.0, -1.0)

        get_all_nodes = getattr(self.long_term_memory, "get_all_nodes", None)
        if not callable(get_all_nodes):
            return None

        for entry in get_all_nodes():
            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
            if metadata.get("chunk_scope") != "turn_window":
                continue
            if session_group_id:
                if str(metadata.get("session_group_id") or "") != session_group_id:
                    continue
            elif self._session_key_from_metadata(metadata) != session_key:
                continue

            entry_id = entry.get("id")
            if not entry_id or entry_id in exclude_ids:
                continue

            overlap = self._entry_query_overlap(query_terms, entry)
            rank = (
                overlap,
                float(entry.get("T", 0.0)),
                float(entry.get("G", 0.0)),
            )
            if rank > best_rank:
                best_rank = rank
                best_entry = entry

        return best_entry

    def _build_session_window_expansion_page(
        self,
        entry: Dict[str, Any],
        trigger_page: Dict[str, Any],
        reason: str,
        query_terms: Set[str],
        is_anchor: bool = False,
    ) -> Dict[str, Any]:
        trigger_score = float(trigger_page.get("score", trigger_page.get("final_score", 0.0)))
        trigger_scope = (
            trigger_page.get("metadata", {}).get("chunk_scope")
            if isinstance(trigger_page.get("metadata", {}), dict)
            else None
        )
        overlap = self._entry_query_overlap(query_terms, entry)
        ratio = 0.96 if is_anchor else 0.9
        if trigger_scope == "session_overview":
            ratio = 0.92 if is_anchor else 0.82
        seeded_score = max(0.42 if is_anchor else 0.36, trigger_score * ratio)
        seeded_score += min(0.12, 0.03 * overlap)
        page = self._format_node_to_page(
            entry,
            source="SESSION_WINDOW_EXPAND",
            score=round(min(0.98, seeded_score), 4),
        )
        page["session_window_expand"] = {
            "reason": reason,
            "trigger_page_id": trigger_page.get("citation_id") or trigger_page.get("page_id"),
            "trigger_scope": trigger_scope,
            "query_overlap": overlap,
            "is_anchor": is_anchor,
        }
        return page

    def _supplement_session_adjacent_windows(
        self,
        query: str,
        pages: List[Dict[str, Any]],
        top_k: int,
        seen_ids: Set[str],
    ) -> int:
        if not pages:
            return 0

        query_terms = self._extract_gap_query_terms(query)
        prelim_band = sorted(
            pages,
            key=self._page_rank_score,
            reverse=True,
        )[: max(int(top_k or 0), 8)]
        if not prelim_band:
            return 0

        overall_best_score = self._page_rank_score(prelim_band[0])
        session_buckets: Dict[str, Dict[str, Any]] = {}
        for page in prelim_band:
            session_key = self._gap_session_key(page)
            if not session_key:
                continue
            bucket = session_buckets.setdefault(session_key, {"pages": [], "overviews": [], "windows": []})
            bucket["pages"].append(page)
            chunk_scope = page.get("metadata", {}).get("chunk_scope") if isinstance(page.get("metadata", {}), dict) else None
            if chunk_scope == "session_overview":
                bucket["overviews"].append(page)
            elif chunk_scope == "turn_window":
                bucket["windows"].append(page)

        added_count = 0
        ranked_sessions = sorted(
            session_buckets.items(),
            key=lambda item: max(self._page_rank_score(page) for page in item[1]["pages"]),
            reverse=True,
        )
        for session_key, bucket in ranked_sessions[: max(2, min(4, int(top_k or 0) or 4))]:
            overview_pages = bucket["overviews"]
            window_pages = bucket["windows"]
            best_window = (
                max(
                    window_pages,
                    key=lambda page: (self._page_rank_score(page), self._gap_query_overlap(query_terms, page)),
                )
                if window_pages
                else None
            )
            has_overview_hit = bool(overview_pages)
            high_score_window_hit = bool(best_window and self._window_hit_is_extremely_high(best_window, overall_best_score))
            if not has_overview_hit and not high_score_window_hit:
                continue

            trigger_page = (
                max(overview_pages, key=self._page_rank_score)
                if has_overview_hit
                else best_window
            )
            if trigger_page is None:
                continue

            session_group_id = self._resolve_session_group_id(best_window or trigger_page)
            anchor_page = best_window
            if anchor_page is None:
                anchor_entry = self._find_best_session_window_entry(
                    session_key,
                    session_group_id,
                    query_terms,
                    seen_ids,
                )
                if not anchor_entry:
                    continue
                anchor_page = self._build_session_window_expansion_page(
                    anchor_entry,
                    trigger_page=trigger_page,
                    reason="overview_hit_anchor_window",
                    query_terms=query_terms,
                    is_anchor=True,
                )
                anchor_id = anchor_page.get("citation_id") or anchor_page.get("page_id")
                if anchor_id and anchor_id not in seen_ids:
                    pages.append(anchor_page)
                    seen_ids.add(anchor_id)
                    added_count += 1

            anchor_index = self._window_index(anchor_page)
            if anchor_index is None:
                continue

            session_group_id = session_group_id or self._resolve_session_group_id(anchor_page)
            if not session_group_id:
                continue

            expansion_reason = "high_score_window_adjacent" if high_score_window_hit else "overview_session_adjacent"
            for neighbor_index in (anchor_index - 1, anchor_index + 1):
                if neighbor_index <= 0:
                    continue
                neighbor_page_id = f"{session_group_id}_window_{neighbor_index}"
                if neighbor_page_id in seen_ids:
                    continue
                neighbor_entry = self.long_term_memory.get_entry_by_id(neighbor_page_id)
                if not neighbor_entry:
                    continue
                neighbor_page = self._build_session_window_expansion_page(
                    neighbor_entry,
                    trigger_page=best_window or trigger_page,
                    reason=expansion_reason,
                    query_terms=query_terms,
                )
                pages.append(neighbor_page)
                seen_ids.add(neighbor_page_id)
                added_count += 1

        return added_count

    def _build_gap_profiles(self, query: str, pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        query_terms = self._extract_gap_query_terms(query)
        profiles: Dict[str, Dict[str, Any]] = {}
        for page in pages:
            metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
            chunk_scope = metadata.get("chunk_scope")
            if chunk_scope not in {"session_overview", "turn_window"}:
                continue
            session_key = self._gap_session_key(page)
            if not session_key:
                continue
            profile = profiles.setdefault(session_key, {"overviews": [], "windows": []})
            record = {
                "page": page,
                "page_id": page.get("citation_id") or page.get("page_id") or str(id(page)),
                "score": float(page.get("final_score", page.get("score", 0.0))),
                "overlap": self._gap_query_overlap(query_terms, page),
            }
            if chunk_scope == "session_overview":
                profile["overviews"].append(record)
            elif chunk_scope == "turn_window":
                profile["windows"].append(record)
        return profiles

    def _build_gap_rerank(
        self,
        page: Dict[str, Any],
        gap_profiles: Dict[str, Dict[str, Any]],
        expected_slot: Optional[str],
    ) -> Dict[str, Any]:
        metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
        chunk_scope = metadata.get("chunk_scope")
        if chunk_scope not in {"session_overview", "turn_window"}:
            return {"multiplier": 1.0, "reason": "gap_not_applicable"}

        session_key = self._gap_session_key(page)
        profile = gap_profiles.get(session_key or "")
        if not profile or not profile.get("overviews") or not profile.get("windows"):
            return {
                "session_key": session_key,
                "multiplier": 1.0,
                "reason": "gap_no_competing_granularity",
            }

        current_page_id = page.get("citation_id") or page.get("page_id") or str(id(page))
        overviews = profile["overviews"]
        windows = profile["windows"]
        current_record = None
        pool = overviews if chunk_scope == "session_overview" else windows
        for record in pool:
            if record["page_id"] == current_page_id:
                current_record = record
                break
        if current_record is None:
            return {
                "session_key": session_key,
                "multiplier": 1.0,
                "reason": "gap_page_not_in_profile",
            }

        best_overview = max(overviews, key=lambda item: (item["overlap"], item["score"]))
        best_window = max(windows, key=lambda item: (item["overlap"], item["score"]))
        if chunk_scope == "session_overview":
            window_support = best_window["overlap"] >= max(1, current_record["overlap"])
            score_support = best_window["score"] >= (current_record["score"] * 0.72)
            if not (window_support or score_support):
                return {
                    "session_key": session_key,
                    "multiplier": 1.0,
                    "reason": "gap_no_query_aligned_window",
                }
            multiplier = 0.82 if expected_slot == "TIME" else 0.72
            if best_window["overlap"] > current_record["overlap"]:
                multiplier -= 0.04
            if len(windows) >= 2:
                multiplier -= 0.02
            multiplier = max(0.64, multiplier)
            return {
                "session_key": session_key,
                "multiplier": round(multiplier, 6),
                "reason": "overview_penalized_by_fine_grain",
                "peer_page_id": best_window["page_id"],
                "peer_count": len(windows),
                "peer_overlap": best_window["overlap"],
            }

        overview_support = current_record["overlap"] >= max(1, best_overview["overlap"])
        score_support = current_record["score"] >= (best_overview["score"] * 0.72)
        if not (overview_support or score_support):
            return {
                "session_key": session_key,
                "multiplier": 1.0,
                "reason": "gap_window_not_competitive",
            }
        multiplier = 1.04 if expected_slot == "TIME" else 1.1
        if current_record["overlap"] > best_overview["overlap"] or current_record["score"] >= (best_overview["score"] * 0.9):
            multiplier += 0.02
        multiplier = min(1.16, multiplier)
        return {
            "session_key": session_key,
            "multiplier": round(multiplier, 6),
            "reason": "fine_grain_preferred_over_overview",
            "peer_page_id": best_overview["page_id"],
            "peer_count": len(windows),
            "peer_overlap": best_overview["overlap"],
        }

    def _score_pages(self, query: str, pages: List[Dict[str, Any]], top_k: int = 10) -> None:
        expected_slot = self._detect_expected_slot(query)
        for page in pages:
            slot_rerank = self._build_slot_rerank(expected_slot, page)
            page["slot_rerank"] = slot_rerank
            page["slot_rerank_multiplier"] = slot_rerank["multiplier"]
            page["gap_rerank"] = {"multiplier": 1.0, "reason": "gap_not_evaluated"}
            page["gap_rerank_multiplier"] = 1.0
            page["final_score"] = self._calc_final_score(page)

        prelim_band = sorted(
            pages,
            key=lambda item: item.get("final_score", 0.0),
            reverse=True,
        )[: max(int(top_k or 0), 8)]
        gap_profiles = self._build_gap_profiles(query, prelim_band)
        for page in pages:
            gap_rerank = self._build_gap_rerank(page, gap_profiles, expected_slot)
            page["gap_rerank"] = gap_rerank
            page["gap_rerank_multiplier"] = float(gap_rerank.get("multiplier", 1.0))
            page["final_score"] = self._calc_final_score(page)

    def _decorate_legacy_page(self, page: Dict[str, Any], location: str) -> Dict[str, Any]:
        metadata = page.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        page["metadata"] = metadata
        page["meta_data"] = metadata
        page["location"] = location
        page["content"] = page.get("raw_content") or f"Q: {page.get('user_input', '')}\nA: {page.get('agent_response', '')}"
        page["gcl_scores"] = {
            "G": float(page.get("G", 0.0)),
            "C": float(page.get("C", 0.6)),
            "T": float(page.get("T", 0.0)),
            "G0": float(page.get("salience_seed", metadata.get("salience_seed", metadata.get("G_base", 0.5)))),
        }
        return page

    def _format_cold_record_to_page(self, record: Dict[str, Any], source: str, score: Optional[float] = None) -> Dict[str, Any]:
        parsed = self._parse_raw_content(record.get("raw_content", ""), record.get("summary", ""))
        page = {
            "page_id": record.get("page_id"),
            "citation_id": record.get("page_id"),
            "doc_id": record.get("doc_id"),
            "user_input": parsed["user_input"] or record.get("user_input", ""),
            "agent_response": parsed["agent_response"] or record.get("agent_response", ""),
            "timestamp": record.get("timestamp"),
            "metadata": {
                **(record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}),
                "entities": self._filter_entities(record.get("entities", [])),
                "semantic_edges": record.get("semantic_edges", []),
                "node_type": "cold_record",
                "storage_tier": record.get("storage_tier", "cold"),
            },
            "summary": record.get("summary", ""),
            "raw_content": record.get("raw_content", ""),
            "G": record.get("G", 0.0),
            "C": record.get("C", 0.5),
            "T": record.get("T", 0.0),
            "salience_seed": record.get("salience_seed", 0.35),
            "score": score if score is not None else record.get("score", 0.0),
            "_source": source,
            "_raw_entities": record.get("entities", []) or (
                record.get("metadata", {}).get("entities", []) if isinstance(record.get("metadata"), dict) else []
            ),
        }
        return self._decorate_legacy_page(page, location="forget_pool")

    def _format_node_to_page(
        self,
        node: Dict[str, Any],
        source: str,
        score: Optional[float] = None,
        prefer_primary_context: bool = False,
    ) -> Dict[str, Any]:
        page_node = node
        if node.get("pointer_to"):
            page_node = self.long_term_memory.get_primary_context_node(node) or node
        elif prefer_primary_context:
            page_node = self.long_term_memory.get_primary_context_node(node) or node

        parsed = self._parse_raw_content(page_node.get("raw_content", ""), page_node.get("summary", ""))
        page = {
            "page_id": page_node.get("id"),
            "citation_id": page_node.get("id"),
            "doc_id": page_node.get("doc_id"),
            "user_input": parsed["user_input"],
            "agent_response": parsed["agent_response"],
            "timestamp": page_node.get("timestamp"),
            "metadata": {
                **(page_node.get("metadata", {}) if isinstance(page_node.get("metadata"), dict) else {}),
                "entities": self._filter_entities(page_node.get("entities", [])),
                "semantic_edges": page_node.get("semantic_edges", []),
                "node_type": page_node.get("node_type"),
                "storage_tier": page_node.get("storage_tier"),
            },
            "summary": page_node.get("summary", ""),
            "raw_content": page_node.get("raw_content", ""),
            "G": page_node.get("G", 0.0),
            "C": page_node.get("C", 0.6),
            "T": page_node.get("T", 0.0),
            "salience_seed": page_node.get("salience_seed", 0.5),
            "score": score if score is not None else node.get("score", 0.0),
            "_source": source,
            "_raw_entities": page_node.get("entities", []) or (
                page_node.get("metadata", {}).get("entities", []) if isinstance(page_node.get("metadata"), dict) else []
            ),
        }
        return self._decorate_legacy_page(page, location="long_term")

    def _auto_merge(self, ltm_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for hit in ltm_hits:
            candidate = hit
            if hit.get("pointer_to"):
                candidate = self.long_term_memory.get_primary_context_node(hit) or hit

            candidate_id = candidate.get("id")
            if not candidate_id:
                continue
            current = merged.get(candidate_id)
            if current is None or hit.get("score", 0.0) > current.get("score", 0.0):
                merged[candidate_id] = {
                    **candidate,
                    "score": hit.get("score", 0.0),
                    "semantic_score": hit.get("semantic_score", 0.0),
                    "keyword_score": hit.get("keyword_score", 0.0),
                }
        return list(merged.values())

    def _collect_expansion_entities(self, query: str, pages: List[Dict[str, Any]]) -> List[str]:
        entities = []
        for token in self._extract_entities(query):
            if token not in entities:
                entities.append(token)
        for page in pages:
            raw_entities = page.get("entities", []) or page.get("metadata", {}).get("entities", [])
            for entity in self._filter_entities(raw_entities):
                if isinstance(entity, str) and entity not in entities:
                    entities.append(entity)
        return entities[:12]

    def _is_detail_question(self, query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        if lowered.startswith(("when", "where", "who", "why", "how")):
            return True
        if lowered.startswith("would ") or " likely " in f" {lowered} " or "considered" in lowered:
            return False
        if lowered.startswith(("what ", "which ")):
            return True
        detail_markers = (
            "what kind",
            "what type",
            "what journal",
            "what game",
            "what new",
            "what was the",
            "what are the names",
            "which country",
            "which year",
            "which locations",
            "how long",
        )
        return any(marker in lowered for marker in detail_markers)

    def _question_supports_graph_one_hop(self, query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        if any(pattern.search(query or "") for pattern in GRAPH_1HOP_NEGATIVE_PATTERNS):
            return False
        if not self._is_detail_question(query):
            return False

        query_terms = self._extract_gap_query_terms(query)
        expected_slot = self._detect_expected_slot(query)
        question_entities = [
            entity
            for entity in self._extract_entities(query)
            if entity.lower() not in GRAPH_1HOP_QUESTION_ENTITY_STOP_WORDS
        ]

        question_score = 0
        if expected_slot is not None:
            question_score += 1
        if len(query_terms) >= 4:
            question_score += 1
        if question_entities:
            question_score += 1
        if len(question_entities) >= 2:
            question_score += 1
        if any(pattern.search(query or "") for pattern in GRAPH_1HOP_SPECIFICITY_PATTERNS):
            question_score += 1
        if any(pattern.search(query or "") for pattern in GRAPH_1HOP_RELATION_PATTERNS):
            question_score += 1
        return question_score >= 2

    def _has_strong_query_aligned_turn_window(
        self,
        query_terms: Set[str],
        top_pages: List[Dict[str, Any]],
        top1_score: float,
    ) -> bool:
        overview_overlap = max(
            (
                self._gap_query_overlap(query_terms, page)
                for page in top_pages
                if (
                    page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
                ).get("chunk_scope")
                == "session_overview"
            ),
            default=0,
        )
        min_window_overlap = 1 if overview_overlap <= 1 else 2
        score_floor = max(0.34, top1_score * 0.72)

        for page in top_pages:
            metadata = page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}
            if metadata.get("chunk_scope") != "turn_window":
                continue
            page_score = self._page_rank_score(page)
            if page_score < score_floor:
                continue
            overlap = self._gap_query_overlap(query_terms, page)
            slot_match = float(page.get("slot_rerank_multiplier", 1.0)) > 1.0
            if overlap >= min_window_overlap or slot_match:
                return True
        return False

    def _should_trigger_graph_one_hop(
        self,
        query: str,
        ranked_pages: List[Dict[str, Any]],
        top_k: int,
    ) -> bool:
        if not self._question_supports_graph_one_hop(query):
            return False

        top_pages = ranked_pages[: max(int(top_k or 0), 4)]
        if not top_pages:
            return False

        expected_slot = self._detect_expected_slot(query)
        query_terms = self._extract_gap_query_terms(query)
        top1_score = self._page_rank_score(top_pages[0])
        strong_hits = sum(1 for page in top_pages if self._page_rank_score(page) >= 0.34)
        slot_match = (
            expected_slot is None
            or any(float(page.get("slot_rerank_multiplier", 1.0)) > 1.0 for page in top_pages[:3])
        )
        overview_pages = [
            page
            for page in top_pages
            if ((page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}).get("chunk_scope") == "session_overview")
        ]
        turn_window_pages = [
            page
            for page in top_pages
            if ((page.get("metadata", {}) if isinstance(page.get("metadata", {}), dict) else {}).get("chunk_scope") == "turn_window")
        ]
        has_strong_turn_window = self._has_strong_query_aligned_turn_window(query_terms, top_pages, top1_score)
        overview_dominant = bool(overview_pages) and len(overview_pages) >= max(1, len(turn_window_pages))

        session_counts: Dict[str, int] = {}
        for page in top_pages:
            session_key = self._gap_session_key(page)
            if not session_key:
                continue
            session_counts[session_key] = session_counts.get(session_key, 0) + 1
        dominant_session_ok = (
            not session_counts
            or len(session_counts) == 1
            or max(session_counts.values()) >= 2
        )
        overlap_ok = max((self._gap_query_overlap(query_terms, page) for page in top_pages), default=0) >= 2

        insufficiency_score = 0
        if top1_score < 0.42:
            insufficiency_score += 2
        if strong_hits < 2:
            insufficiency_score += 2
        if expected_slot is not None and not slot_match:
            insufficiency_score += 1
        if self._is_detail_question(query) and not has_strong_turn_window:
            insufficiency_score += 2
            if overview_dominant:
                insufficiency_score += 1
        if not dominant_session_ok:
            insufficiency_score += 1
        if not overlap_ok:
            insufficiency_score += 1
        return insufficiency_score >= 3

    def _accept_graph_one_hop_pages(
        self,
        query: str,
        pages: List[Dict[str, Any]],
        candidate_pages: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidate_pages:
            return []

        baseline_ranked = sorted(pages, key=self._page_rank_score, reverse=True)
        if baseline_ranked:
            weakest_topk_score = self._page_rank_score(
                baseline_ranked[min(max(int(top_k or 0), 1), len(baseline_ranked)) - 1]
            )
        else:
            weakest_topk_score = 0.0

        combined_pages = list(pages) + list(candidate_pages)
        self._score_pages(query, combined_pages, top_k=top_k)
        ranked_combined = sorted(combined_pages, key=self._page_rank_score, reverse=True)
        candidate_ids = {
            page.get("citation_id") or page.get("page_id")
            for page in candidate_pages
            if page.get("citation_id") or page.get("page_id")
        }

        accepted: List[Dict[str, Any]] = []
        for rank, page in enumerate(ranked_combined, start=1):
            citation_id = page.get("citation_id") or page.get("page_id")
            if citation_id not in candidate_ids:
                continue
            graph_meta = page.get("graph_1hop", {}) if isinstance(page.get("graph_1hop"), dict) else {}
            entity_overlap = int(graph_meta.get("entity_overlap", 0) or 0)
            query_overlap = int(graph_meta.get("query_overlap", 0) or 0)
            page_score = self._page_rank_score(page)
            if entity_overlap < 1 and query_overlap < 2:
                continue
            rank_cap = max(int(top_k or 0), 3)
            if entity_overlap >= 3 or (entity_overlap >= 2 and query_overlap >= 2) or query_overlap >= 4:
                rank_cap += 2
            elif entity_overlap >= 2 or query_overlap >= 3:
                rank_cap += 1
            if rank > rank_cap:
                continue
            if page_score < 0.30 and page_score < weakest_topk_score + 0.04:
                continue
            accepted.append(page)
            if len(accepted) >= 2:
                break
        return accepted

    def _build_graph_one_hop_page(
        self,
        entry: Dict[str, Any],
        expansion_entities: List[str],
        query_terms: Set[str],
    ) -> Dict[str, Any]:
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        related_entities = self._filter_entities(
            entry.get("entities", []) or metadata.get("entities", [])
        )
        related_entity_set = {entity.lower() for entity in related_entities}
        expansion_entity_set = {entity.lower() for entity in expansion_entities}
        entity_overlap = len(related_entity_set & expansion_entity_set)
        query_overlap = self._entry_query_overlap(query_terms, entry)

        seeded_score = 0.24
        seeded_score += min(0.2, 0.05 * entity_overlap)
        seeded_score += min(0.12, 0.03 * query_overlap)
        if metadata.get("chunk_scope") == "turn_window":
            seeded_score += 0.04
        elif metadata.get("chunk_scope") == "session_overview":
            seeded_score += 0.02

        page = self._format_node_to_page(
            entry,
            source="GRAPH_1HOP",
            score=round(min(0.78, seeded_score), 4),
        )
        page["graph_1hop"] = {
            "matched_entities": [entity for entity in related_entities if entity.lower() in expansion_entity_set],
            "entity_overlap": entity_overlap,
            "query_overlap": query_overlap,
        }
        return page

    def _collect_graph_one_hop_pages(
        self,
        query: str,
        pages: List[Dict[str, Any]],
        seen_ids: Set[str],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        get_related = getattr(self.long_term_memory, "get_related_by_entities", None)
        if not callable(get_related):
            return []

        expansion_entities = self._collect_expansion_entities(query, pages)
        if not expansion_entities:
            return []

        query_terms = self._extract_gap_query_terms(query)
        related_entries = get_related(
            expansion_entities,
            max_results=max_results,
            exclude_ids=seen_ids,
        )

        related_pages: List[Dict[str, Any]] = []
        for entry in related_entries:
            entry_id = entry.get("id")
            if not entry_id or entry_id in seen_ids:
                continue
            related_pages.append(
                self._build_graph_one_hop_page(
                    entry,
                    expansion_entities=expansion_entities,
                    query_terms=query_terms,
                )
            )
        return related_pages

    def _apply_token_budget(self, pages: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
        if token_budget <= 0:
            return pages
        selected = []
        used_tokens = 0
        for page in pages:
            estimated_tokens = self._estimate_page_tokens(page)
            if selected and used_tokens + estimated_tokens > token_budget:
                continue
            selected.append(page)
            used_tokens += estimated_tokens
        return selected

    def retrieve_context(self, query: str, top_k: Any = 10, token_budget: int = 2200, **kwargs) -> Dict[str, Any]:
        trace_context = kwargs.get("trace_context")
        results = {
            "retrieved_pages": [],
            "hit_ids": [],
            "salvage_count": 0,
            "source_stats": {
                "STM": 0,
                "LTM": 0,
                "AUTO_MERGED": 0,
                "GRAPH_1HOP": 0,
                "FORGET_COLD": 0,
            },
        }
        if not query:
            return results

        if isinstance(top_k, str):
            top_k = kwargs.get("top_k", 10)
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 10

        seen_ids: Set[str] = set()
        pages: List[Dict[str, Any]] = []

        stm_hits = self.short_term_memory.search(query, k=min(5, top_k))
        for hit in stm_hits:
            page_id = hit.get("page_id")
            if not page_id or page_id in seen_ids:
                continue
            score = 0.55 + (
                0.05
                * len(set(query.lower().split()) & set(f"{hit.get('user_input', '')} {hit.get('agent_response', '')}".lower().split()))
            )
            page = {
                "page_id": page_id,
                "citation_id": page_id,
                "doc_id": hit.get("doc_id", page_id),
                "user_input": hit.get("user_input", ""),
                "agent_response": hit.get("agent_response", ""),
                "timestamp": hit.get("timestamp"),
                "metadata": hit.get("metadata", {}),
                "summary": hit.get("metadata", {}).get("topic", ""),
                "raw_content": f"Q: {hit.get('user_input', '')}\nA: {hit.get('agent_response', '')}",
                "G": hit.get("G", 0.0),
                "C": hit.get("C", 0.6),
                "T": hit.get("T", 100.0),
                "salience_seed": hit.get("gcl_scores", {}).get("G0", 0.5),
                "score": round(score, 4),
                "_source": "STM_HOT",
                "_raw_entities": hit.get("metadata", {}).get("entities", []) if isinstance(hit.get("metadata"), dict) else [],
            }
            page = self._decorate_legacy_page(page, location="short_term")
            pages.append(page)
            seen_ids.add(page_id)
            results["source_stats"]["STM"] += 1

        ltm_hits = self.long_term_memory.search_knowledge(query, top_k=max(top_k * 2, 8))
        merged_hits = self._auto_merge(ltm_hits)
        for hit in merged_hits:
            page_id = hit.get("id")
            if not page_id or page_id in seen_ids:
                continue
            page = self._format_node_to_page(hit, source="LTM_HYBRID", score=hit.get("score", 0.0))
            pages.append(page)
            seen_ids.add(page_id)
            results["source_stats"]["LTM"] += 1
        results["source_stats"]["AUTO_MERGED"] = max(0, len(ltm_hits) - len(merged_hits))

        cold_hits = self.forget_memory.search(query, top_k=max(top_k, 4), exclude_ids=seen_ids)
        for hit in cold_hits:
            page_id = hit.get("page_id")
            if not page_id or page_id in seen_ids:
                continue
            page = self._format_cold_record_to_page(hit, source="FORGET_COLD", score=hit.get("score", 0.0))
            pages.append(page)
            seen_ids.add(page_id)
            results["source_stats"]["FORGET_COLD"] += 1

        self._score_pages(query, pages, top_k=top_k)
        ranked_pages = sorted(pages, key=self._page_rank_score, reverse=True)
        if self._should_trigger_graph_one_hop(query, ranked_pages, top_k):
            raw_graph_pages = self._collect_graph_one_hop_pages(
                query,
                ranked_pages[: max(int(top_k or 0), 4)],
                seen_ids=seen_ids,
                max_results=3,
            )
            accepted_graph_pages = self._accept_graph_one_hop_pages(query, pages, raw_graph_pages, top_k)
            if accepted_graph_pages:
                pages.extend(accepted_graph_pages)
                for page in accepted_graph_pages:
                    citation_id = page.get("citation_id") or page.get("page_id")
                    if citation_id:
                        seen_ids.add(citation_id)
                results["source_stats"]["GRAPH_1HOP"] += len(accepted_graph_pages)
                self._score_pages(query, pages, top_k=top_k)
        pages.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        pages = self._apply_token_budget(pages[: max(top_k * 2, top_k)], token_budget)
        results["salvage_count"] = sum(1 for page in pages if page.get("_source") == "FORGET_COLD")

        access_updates = []
        ltm_page_ids = []
        for page in pages:
            citation_id = page.get("citation_id")
            if citation_id and self.long_term_memory.node_exists(citation_id):
                ltm_page_ids.append(citation_id)
                if self.tracer is not None:
                    entry = self.long_term_memory.get_entry_by_id(citation_id)
                    access_updates.append(
                        {
                            "citation_id": citation_id,
                            "before": self.tracer.snapshot_record(entry, default_location="long_term"),
                        }
                    )
        if ltm_page_ids:
            self.long_term_memory.record_access(ltm_page_ids)
            if self.tracer is not None:
                for update in access_updates:
                    entry = self.long_term_memory.get_entry_by_id(update["citation_id"])
                    update["after"] = self.tracer.snapshot_record(entry, default_location="long_term")

        results["retrieved_pages"] = pages[:top_k]
        results["hit_ids"] = [page.get("citation_id") or page.get("page_id") for page in results["retrieved_pages"]]
        if self.tracer is not None:
            hits = []
            for index, page in enumerate(results["retrieved_pages"], start=1):
                snapshot = self.tracer.snapshot_record(page, default_location=page.get("location", "unknown"))
                snapshot["rank"] = index
                hits.append(snapshot)
            self.tracer.record_retrieval_event(
                {
                    "query": query,
                    "trace_context": dict(trace_context or {}),
                    "memory_ids": [hit.get("memory_id") for hit in hits if hit.get("memory_id")],
                    "hit_ids": list(results["hit_ids"]),
                    "source_stats": dict(results.get("source_stats", {})),
                    "salvage_count": results.get("salvage_count", 0),
                    "hits": hits,
                    "access_updates": access_updates,
                }
            )
        return results

    def get_context_summary(self, retrieval_results: Dict[str, Any]) -> str:
        pages = retrieval_results.get("retrieved_pages", [])
        if not pages:
            return "No relevant memories found."

        lines = ["### Memory Context"]
        for page in pages:
            lines.append(f"[{page.get('citation_id')}] {page.get('user_input', '')}")
            lines.append(page.get("agent_response", ""))
            lines.append("")
        return "\n".join(lines)
