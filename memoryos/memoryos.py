import json
import hashlib
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from .adaptive_answer_skill import (
        AdaptiveAnswerOutcome,
        apply_adaptive_answer_action,
        build_adaptive_answer_state,
        record_adaptive_answer_outcome,
        select_adaptive_answer_action,
        should_retry_adaptive_answer,
    )
    from .choice_answering import answer_choice_with_context
    from .canonical_evidence import build_canonical_evidence_trace
    from .answer_routing import (
        build_answer_mode_instruction,
        normalize_generated_answer,
        resolve_effective_answer_format,
        resolve_effective_answer_mode,
    )
    from .entity_filters import (
        filter_entity_values,
        filter_semantic_edge_values,
        is_structural_entity_label,
        normalize_entity_candidate,
    )
    from .forget_memory import ForgetMemory
    from .long_term import LongTermMemory
    from .query_routing import (
        TaskRoutingPlan,
        build_retrieval_trace_context,
        infer_task_routing_plan,
        resolve_effective_retrieval_mode,
    )
    from .services.lifecycle_service import LifecycleService
    from .services.retrieval_service import Retriever
    from .short_term import ShortTermMemory
    from .services.ingestion_service import Updater
    from .tracing import TraceRecorder
    from .utils import (
        OpenAIClient,
        ensure_directory_exists,
        fallback_extract_keywords,
        generate_id,
        get_timestamp,
        mask_system_tags,
        normalize_keyword_values,
    )
except (ImportError, ValueError):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from adaptive_answer_skill import (
        AdaptiveAnswerOutcome,
        apply_adaptive_answer_action,
        build_adaptive_answer_state,
        record_adaptive_answer_outcome,
        select_adaptive_answer_action,
        should_retry_adaptive_answer,
    )
    from choice_answering import answer_choice_with_context
    from canonical_evidence import build_canonical_evidence_trace
    from answer_routing import (
        build_answer_mode_instruction,
        normalize_generated_answer,
        resolve_effective_answer_format,
        resolve_effective_answer_mode,
    )
    from entity_filters import (
        filter_entity_values,
        filter_semantic_edge_values,
        is_structural_entity_label,
        normalize_entity_candidate,
    )
    from forget_memory import ForgetMemory
    from long_term import LongTermMemory
    from query_routing import (
        TaskRoutingPlan,
        build_retrieval_trace_context,
        infer_task_routing_plan,
        resolve_effective_retrieval_mode,
    )
    from services.lifecycle_service import LifecycleService
    from services.retrieval_service import Retriever
    from short_term import ShortTermMemory
    from services.ingestion_service import Updater
    from tracing import TraceRecorder
    from utils import (
        OpenAIClient,
        ensure_directory_exists,
        fallback_extract_keywords,
        generate_id,
        get_timestamp,
        mask_system_tags,
        normalize_keyword_values,
    )


DEFAULT_ASSISTANT_ID = "default_assistant_profile"


class MemoryQueryResult(dict):
    """Dict result with string-like compatibility for legacy callers."""

    def __init__(self, *args, answer_text: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_text = answer_text or self.get("final_answer") or self.get("context_summary", "")

    def __str__(self) -> str:
        return self.answer_text

    def __getitem__(self, key):
        if isinstance(key, (slice, int)):
            return self.answer_text[key]
        return super().__getitem__(key)

    def lower(self) -> str:
        return self.answer_text.lower()


class Memoryos:
    def __init__(
        self,
        user_id: str,
        openai_api_key: str,
        data_storage_path: str,
        openai_base_url: str = None,
        planner_base_url: str = None,
        assistant_id: str = DEFAULT_ASSISTANT_ID,
        short_term_capacity: int = 15,
        long_term_knowledge_capacity: int = 5000,
        retrieval_queue_capacity: int = 30,
        llm_model: str = "qwen2.5:7b",
        planner_model: Optional[str] = None,
        embedding_model_name: str = "nomic-embed-text",
        embedding_model_kwargs: dict = None,
        trace_recorder: Optional[TraceRecorder] = None,
        lightweight_extraction: bool = False,
        keyword_ranker_path: Optional[str] = None,
        enable_graph_one_hop: bool = True,
        enable_async_nli_verify: bool = False,
        enable_async_entity_enrichment: bool = True,
        enable_storage_backups: bool = True,
        persist_access_updates: bool = True,
    ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.planner_model = planner_model or llm_model
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.trace_recorder = trace_recorder
        self.lightweight_extraction = bool(lightweight_extraction)
        self.enable_graph_one_hop = bool(enable_graph_one_hop)
        self.enable_async_nli_verify = bool(enable_async_nli_verify)
        self.enable_async_entity_enrichment = bool(enable_async_entity_enrichment)
        self.enable_storage_backups = bool(enable_storage_backups)
        self.persist_access_updates = bool(persist_access_updates)
        self.current_session_hit_ids: List[str] = []
        self._entity_cache: Dict[str, List[str]] = {}
        self._keyword_cache: Dict[str, List[str]] = {}
        self.keyword_ranker = None
        self._ingest_count = 0
        self._retrieval_count = 0
        self.last_ingest_result: Dict[str, Any] = {}
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)
        if planner_base_url and str(planner_base_url).strip() != str(openai_base_url or "").strip():
            self.planner_client = OpenAIClient(api_key=openai_api_key, base_url=planner_base_url)
        else:
            self.planner_client = self.client

        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_long_term_dir = os.path.join(self.user_data_dir, "long_term")
        user_forget_path = os.path.join(self.user_data_dir, "forget_pool.jsonl")

        ensure_directory_exists(user_short_term_path)
        ensure_directory_exists(user_forget_path)

        self.short_term_memory = ShortTermMemory(
            file_path=user_short_term_path,
            max_capacity=short_term_capacity,
            enable_backups=self.enable_storage_backups,
        )
        self.forget_memory = ForgetMemory(
            file_path=user_forget_path,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            enable_backups=self.enable_storage_backups,
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_dir,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            enable_backups=self.enable_storage_backups,
        )
        self.long_term_memory = self.user_long_term_memory
        self.short_term_memory.keyword_extractor = self._extract_keywords
        self.long_term_memory.entity_extractor = self._extract_entities_local
        self.long_term_memory.keyword_extractor = self._extract_keywords
        self.mid_term_memory = self.short_term_memory
        self.updater = Updater(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.user_long_term_memory,
            forget_memory=self.forget_memory,
            client=self.client,
            llm_model=self.llm_model,
            stm_capacity=short_term_capacity,
        )
        self.updater.memoryos = self
        self.updater.tracer = self.trace_recorder
        self.forget_memory.entity_extractor = self._extract_entities_local
        self.forget_memory.keyword_extractor = self._extract_keywords
        self.retriever = Retriever(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.user_long_term_memory,
            forget_memory=self.forget_memory,
            queue_capacity=retrieval_queue_capacity,
            enable_graph_one_hop=self.enable_graph_one_hop,
            persist_access_updates=self.persist_access_updates,
        )
        self.retriever.entity_extractor = self._extract_entities_local
        self.retriever.tracer = self.trace_recorder
        self.lifecycle = LifecycleService(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.user_long_term_memory,
            forget_memory=self.forget_memory,
            client=self.client,
            llm_model=self.llm_model,
        )
        self.lifecycle.tracer = self.trace_recorder
        if self.trace_recorder is not None:
            self.trace_recorder.update_run_metadata(
                user_id=self.user_id,
                data_storage_path=self.data_storage_path,
                llm_model=self.llm_model,
                planner_model=self.planner_model,
                embedding_model_name=self.embedding_model_name,
            )
        self.original_add_memory = self.add_memory

        if keyword_ranker_path:
            try:
                try:
                    from .keyword_ranker import LinearKeywordRanker
                except (ImportError, ValueError):
                    from keyword_ranker import LinearKeywordRanker

                if os.path.exists(str(keyword_ranker_path)):
                    self.keyword_ranker = LinearKeywordRanker.load(str(keyword_ranker_path))
            except Exception:
                self.keyword_ranker = None

    def get_llm_priority_scores(self, memories: List[Dict]) -> Dict[str, float]:
        """Legacy compatibility hook. Hot path no longer blocks on this."""
        return {}

    def _prepare_entity_extraction_text(self, text: str, max_chars: int = 1800) -> str:
        compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
        if len(compact) <= max_chars:
            return compact
        head = compact[: int(max_chars * 0.65)]
        tail = compact[-int(max_chars * 0.2) :]
        return f"{head}\n...\n{tail}"

    def _prepare_keyword_extraction_text(self, text: str, max_chars: int = 1600) -> str:
        compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
        if len(compact) <= max_chars:
            return compact
        head = compact[: int(max_chars * 0.7)]
        tail = compact[-int(max_chars * 0.15) :]
        return f"{head}\n...\n{tail}"

    def _parse_entity_payload(self, raw_response: str) -> Dict[str, Any]:
        if not raw_response or not isinstance(raw_response, str):
            return {}
        if raw_response.startswith("Error:") or raw_response.startswith("Simulated response:"):
            return {}
        try:
            parsed = json.loads(raw_response)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            start = raw_response.find("{")
            end = raw_response.rfind("}")
            if start == -1 or end <= start:
                return {}
            try:
                parsed = json.loads(raw_response[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

    def _normalize_llm_keywords(self, payload: Dict[str, Any], limit: int = 12) -> List[str]:
        raw_keywords = payload.get("keywords", [])
        if not isinstance(raw_keywords, list):
            return []
        return normalize_keyword_values(raw_keywords, limit=limit)

    def _is_structural_entity_label(self, candidate: str) -> bool:
        return is_structural_entity_label(candidate)

    def _normalize_llm_entities(self, payload: Dict[str, Any]) -> List[str]:
        raw_entities = payload.get("entities", [])
        if not isinstance(raw_entities, list):
            return []

        blocked = {
            "i",
            "you",
            "we",
            "they",
            "he",
            "she",
            "it",
            "this",
            "that",
            "the",
            "a",
            "an",
            "user",
            "assistant",
            "system",
            "memory",
            "project",
        }
        candidates: List[str] = []
        for item in raw_entities:
            if isinstance(item, dict):
                candidate = item.get("canonical") or item.get("name") or item.get("surface")
            else:
                candidate = item
            if not isinstance(candidate, str):
                continue
            candidate = normalize_entity_candidate(candidate)
            if len(candidate) < 2:
                continue
            if candidate.lower() in blocked:
                continue
            candidates.append(candidate)
        return filter_entity_values(candidates, limit=24)

    def _extract_entities(self, text: str) -> List[str]:
        prepared_text = self._prepare_entity_extraction_text(text)
        if not prepared_text:
            return []

        cache_key = hashlib.sha1(prepared_text.encode("utf-8")).hexdigest()
        cached = self._entity_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if getattr(self, "lightweight_extraction", False):
            entities = filter_entity_values(re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{1,}\b", prepared_text), limit=24)
            self._entity_cache[cache_key] = list(entities)
            return list(entities)

        prompt = (
            "You are an entity extraction component for a conversational memory system.\n"
            "Task: extract reusable named entities and technical identifiers that are helpful for retrieval.\n"
            'Return strict JSON only in this format: {"entities": ["entity1", "entity2"]}.\n'
            "Keep only explicit named entities or concrete identifiers such as people, organizations, products, "
            "projects, repositories, models, libraries, classes, functions, APIs, file paths, locations, dates, "
            "and named events.\n"
            "Prefer exact surface forms copied from the text.\n"
            "Ignore pronouns, greetings, vague descriptions, generic nouns, structural field labels, and filler.\n"
            "Do not output labels like Summary, Session Summary, Event Summary, Transcript, Transcript Window, "
            "Observations, Resolved Relative Times, Resolved Event Facts, Query Hint, Image Caption, User, "
            "Assistant, or System.\n"
            "If an item is uncertain, omit it.\n"
            'Return at most 24 entities sorted by retrieval usefulness. If nothing is useful, return {"entities": []}.\n\n'
            f"Text:\n{prepared_text}"
        )

        entities: List[str] = []
        for response_format in ({"type": "json_object"}, None):
            raw_response = self.client.chat_completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
                response_format=response_format,
            )
            entities = self._normalize_llm_entities(self._parse_entity_payload(raw_response))
            if entities:
                break

        self._entity_cache[cache_key] = list(entities)
        return list(entities)

    def _extract_entities_local(self, text: str) -> List[str]:
        prepared_text = self._prepare_entity_extraction_text(text)
        if not prepared_text:
            return []

        cache_key = f"local:{hashlib.sha1(prepared_text.encode('utf-8')).hexdigest()}"
        cached = self._entity_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        raw_candidates: List[str] = []

        raw_candidates.extend(
            match.group(0)
            for match in re.finditer(r"\b\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}\b", prepared_text)
        )
        raw_candidates.extend(
            match.group(0)
            for match in re.finditer(r"\b[A-Z][a-z]+\s+\d{4}\b", prepared_text)
        )
        raw_candidates.extend(
            match.group(1)
            for match in re.finditer(r"['\"]([^'\"]{2,80})['\"]", prepared_text)
            if 1 <= len(match.group(1).split()) <= 6
        )
        raw_candidates.extend(
            match.group(0)
            for match in re.finditer(
                r"\b[A-Z][\w'’.-]*(?:\s+(?:[A-Z][\w'’.-]*|of|the|and|de|di|da|la|le|van|von)){0,4}\b",
                prepared_text,
            )
        )
        raw_candidates.extend(
            token
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_:\-./]{2,}\b", prepared_text)
            if any(ch.isdigit() for ch in token) or any(ch in token for ch in "._:/-")
        )

        entities = filter_entity_values(raw_candidates, limit=24)
        self._entity_cache[cache_key] = list(entities)
        return list(entities)

    def _extract_keyword_phrases_local(self, text: str, max_keywords: int = 6) -> List[str]:
        compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
        if not compact:
            return []

        phrase_blocked = {
            "a",
            "an",
            "and",
            "answer",
            "answers",
            "assistant",
            "about",
            "after",
            "before",
            "by",
            "during",
            "for",
            "from",
            "how",
            "in",
            "into",
            "is",
            "it",
            "kind",
            "many",
            "memory",
            "name",
            "names",
            "of",
            "on",
            "question",
            "response",
            "session",
            "summary",
            "the",
            "their",
            "them",
            "they",
            "this",
            "to",
            "type",
            "unknown",
            "user",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "would",
        }

        raw_phrases: List[str] = []
        raw_phrases.extend(
            candidate.lower()
            for candidate in self._extract_entities_local(compact)
            if " " in candidate and len(candidate.split()) <= 5
        )

        lowered_tokens = re.findall(r"[A-Za-z0-9]+", compact.lower())
        for n in (2, 3):
            for index in range(len(lowered_tokens) - n + 1):
                gram = lowered_tokens[index : index + n]
                if any(token in phrase_blocked for token in gram):
                    continue
                if not any(len(token) >= 5 for token in gram):
                    continue
                raw_phrases.append(" ".join(gram))

        return normalize_keyword_values(raw_phrases, limit=max_keywords)

    def _extract_keywords(self, text: str, max_keywords: int = 12) -> List[str]:
        prepared_text = self._prepare_keyword_extraction_text(text)
        if not prepared_text:
            return []

        cache = getattr(self, "_keyword_cache", None)
        if cache is None:
            cache = {}
            self._keyword_cache = cache

        cache_key = hashlib.sha1(prepared_text.encode("utf-8")).hexdigest()
        cached = cache.get(cache_key)
        if cached is not None:
            return list(cached)

        ranker = getattr(self, "keyword_ranker", None)
        if ranker is not None:
            try:
                ranked = ranker.extract(prepared_text, max_keywords=max_keywords)
                ranked = normalize_keyword_values(ranked, limit=max_keywords)
                if ranked:
                    cache[cache_key] = list(ranked)
                    return list(ranked)
            except Exception:
                pass

        # Retrieval-side keywording stays deterministic: hybrid recall always uses
        # lexical terms derived locally instead of an extra LLM keyword pass.
        phrase_keywords = self._extract_keyword_phrases_local(prepared_text, max_keywords=min(6, max_keywords))
        token_keywords = fallback_extract_keywords(prepared_text, limit=max_keywords)
        keywords = normalize_keyword_values(phrase_keywords + token_keywords, limit=max_keywords)
        cache[cache_key] = list(keywords)
        return list(keywords)

    def _infer_topic(self, combined: str) -> str:
        topic = "General"
        if re.search(r"\b(error|traceback|bug|exception|stack)\b", combined, re.I):
            topic = "Debugging"
        elif re.search(r"\b(architecture|memory|rag|retriev|vector|graph|agent)\b", combined, re.I):
            topic = "Architecture"
        elif re.search(r"\b(prefer|like|favorite|always|never|dislike)\b", combined, re.I):
            topic = "Preference"
        return topic

    def _infer_flags(self, combined: str) -> Dict[str, Any]:
        flags: Dict[str, Any] = {}
        if re.search(r"\b(prefer|favorite|like|dislike|always|never)\b", combined, re.I):
            flags["preference"] = True
        if re.search(r"\b(important|key|critical|remember|must retain|worth remembering)\b", combined, re.I):
            flags["important"] = True
        return flags

    def _extract_metadata(
        self,
        user_input: str,
        agent_response: str,
        force_local_entities: bool = False,
    ) -> Dict[str, Any]:
        safe_input = mask_system_tags(user_input)
        safe_response = mask_system_tags(agent_response)
        combined = f"{safe_input}\n{safe_response}".strip()
        entities = self._extract_entities_local(combined) if force_local_entities else self._extract_entities(combined)
        keywords = self._extract_keywords(combined, max_keywords=16)
        topic = self._infer_topic(combined)
        flags = self._infer_flags(combined)
        semantic_edges = list(dict.fromkeys(entities + keywords[:4]))
        return {
            "topic": topic,
            "entities": entities,
            "semantic_edges": semantic_edges,
            "keywords": keywords,
            "flags": flags,
        }

    def _merge_memory_metadata(self, extracted: Dict[str, Any], external_metadata: Dict[str, Any]) -> Dict[str, Any]:
        merged_metadata = {**extracted, **external_metadata}
        merged_flags = {}
        if isinstance(extracted.get("flags"), dict):
            merged_flags.update(extracted.get("flags", {}))
        if isinstance(external_metadata.get("flags"), dict):
            merged_flags.update(external_metadata.get("flags", {}))
        merged_metadata["flags"] = merged_flags
        merged_metadata["entities"] = list(
            dict.fromkeys(extracted.get("entities", []) + external_metadata.get("entities", []))
        )
        merged_metadata["semantic_edges"] = list(
            dict.fromkeys(extracted.get("semantic_edges", []) + external_metadata.get("semantic_edges", []))
        )
        shared_keywords = normalize_keyword_values(external_metadata.get("keywords", []), limit=32)
        if not shared_keywords:
            shared_keywords = normalize_keyword_values(extracted.get("keywords", []), limit=32)
        merged_metadata["keywords"] = list(shared_keywords)
        return merged_metadata

    def _apply_enriched_metadata(self, page_id: str, doc_id: str, metadata: Dict[str, Any]) -> None:
        clean_entities = filter_entity_values(metadata.get("entities", []), limit=24)
        clean_edges = filter_semantic_edge_values(metadata.get("semantic_edges", []), limit=48)
        clean_keywords = normalize_keyword_values(metadata.get("keywords", []), limit=32)

        stm_page = self.short_term_memory.get_page_by_id(page_id)
        if stm_page is not None:
            page_meta = stm_page.get("metadata", {})
            if not isinstance(page_meta, dict):
                page_meta = {}
            page_meta.update(metadata)
            page_meta["entities"] = list(clean_entities)
            page_meta["semantic_edges"] = list(clean_edges)
            page_meta["keywords"] = list(clean_keywords)
            page_meta["async_entity_enriched"] = True
            stm_page["metadata"] = page_meta
            stm_page["keywords"] = list(clean_keywords)
            self.short_term_memory.save()

        ltm_changed = False
        for entry in self.long_term_memory.get_all_nodes():
            if entry.get("doc_id") not in {doc_id, page_id} and entry.get("id") not in {doc_id, page_id}:
                continue
            entry_meta = entry.get("metadata", {})
            if not isinstance(entry_meta, dict):
                entry_meta = {}
            entry_meta.update(metadata)
            entry_meta["entities"] = list(clean_entities)
            entry_meta["semantic_edges"] = list(clean_edges)
            entry_meta["keywords"] = list(clean_keywords)
            entry_meta["async_entity_enriched"] = True
            entry["metadata"] = entry_meta
            entry["entities"] = list(clean_entities)
            entry["semantic_edges"] = list(clean_edges)
            entry["keywords"] = list(clean_keywords)
            ltm_changed = True
        if ltm_changed:
            self.long_term_memory._rebuild_indexes()
            self.long_term_memory.save()

    def _async_enrich_memory_entities(
        self,
        page_id: str,
        doc_id: str,
        user_input: str,
        agent_response: str,
        external_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not page_id:
            return
        extracted = self._extract_metadata(user_input, agent_response, force_local_entities=False)
        merged_metadata = self._merge_memory_metadata(
            extracted,
            self._normalize_external_metadata(external_metadata),
        )
        self._apply_enriched_metadata(page_id, doc_id or page_id, merged_metadata)

    def _calculate_smart_g0(self, user_input: str, metadata: Dict[str, Any]) -> float:
        boost_terms = ["memory", "agent", "architecture", "important", "remember", "偏好", "架构", "关键"]
        base = 0.55 if any(term in user_input.lower() for term in boost_terms) else 0.35
        entity_bonus = min(0.25, 0.03 * len(metadata.get("entities", [])))
        return round(min(0.95, base + entity_bonus), 2)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _normalize_external_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}

        normalized = dict(metadata)
        if "G_base" in normalized and "salience_seed" not in normalized:
            normalized["salience_seed"] = normalized.get("G_base")
        if "entity_list" in normalized and "entities" not in normalized:
            normalized["entities"] = normalized.get("entity_list", [])
        return normalized

    def _decorate_memory_page(self, page: Dict[str, Any], location: str) -> Dict[str, Any]:
        metadata = page.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        page["metadata"] = metadata
        page["meta_data"] = metadata
        page["location"] = location
        page["_memory_tier"] = location
        page["content"] = page.get("raw_content") or f"Q: {page.get('user_input', '')}\nA: {page.get('agent_response', '')}"
        page["gcl_scores"] = {
            "G": float(page.get("G", 0.0)),
            "C": float(page.get("C", 0.6)),
            "T": float(page.get("T", 0.0)),
            "G0": float(page.get("salience_seed", metadata.get("salience_seed", metadata.get("G_base", 0.5)))),
        }
        return page

    def _derive_feedback_score(self, final_answer: str, citations: List[str], hit_ids: List[str]) -> float:
        answer = (final_answer or "").strip().lower()
        if not hit_ids:
            return 0.0
        if answer == "i don't know.":
            return 0.45 if citations else 0.3
        if citations:
            return 0.9
        return 0.7

    def _merge_hybrid_retrieval_results(
        self,
        retrieval_results: List[Dict[str, Any]],
        *,
        max_results: int,
        token_budget: int,
    ) -> Dict[str, Any]:
        merged_pages: Dict[str, Dict[str, Any]] = {}
        ordered_ids: List[str] = []
        branch_rank: Dict[str, Dict[str, Any]] = {}
        source_stats: Dict[str, int] = {}
        salvage_count = 0

        for result in retrieval_results:
            profile = result.get("retrieval_profile", {}) if isinstance(result.get("retrieval_profile"), dict) else {}
            branch = str(profile.get("query_kind", "") or "").strip().lower()
            for page in result.get("retrieved_pages", []) or []:
                page_id = str(page.get("citation_id") or page.get("page_id") or page.get("doc_id") or "")
                if not page_id:
                    continue
                page_copy = dict(page)
                branches = set(page_copy.get("_hybrid_branches", []) or [])
                if branch:
                    branches.add(branch)
                page_copy["_hybrid_branches"] = sorted(branches)
                existing = merged_pages.get(page_id)
                if existing is None:
                    merged_pages[page_id] = page_copy
                    ordered_ids.append(page_id)
                else:
                    current_score = float(existing.get("final_score", existing.get("score", 0.0)) or 0.0)
                    new_score = float(page_copy.get("final_score", page_copy.get("score", 0.0)) or 0.0)
                    better = page_copy if new_score >= current_score else existing
                    merged = dict(existing)
                    merged.update({key: value for key, value in page_copy.items() if value not in (None, "", [], {})})
                    merged.update(better)
                    merged["_hybrid_branches"] = sorted(
                        {
                            str(item).strip().lower()
                            for item in (existing.get("_hybrid_branches", []) or []) + (page_copy.get("_hybrid_branches", []) or [])
                            if str(item).strip()
                        }
                    )
                    merged_pages[page_id] = merged

                page_score = float(page_copy.get("final_score", page_copy.get("score", 0.0)) or 0.0)
                if branch and (
                    branch not in branch_rank
                    or page_score > float(branch_rank[branch].get("final_score", branch_rank[branch].get("score", 0.0)) or 0.0)
                ):
                    branch_rank[branch] = merged_pages[page_id]

            for key, value in (result.get("source_stats", {}) or {}).items():
                if isinstance(value, (int, float)):
                    source_stats[key] = source_stats.get(key, 0) + int(value)
            salvage_count += int(result.get("salvage_count", 0) or 0)

        pages = sorted(
            (merged_pages[page_id] for page_id in ordered_ids),
            key=lambda item: float(item.get("final_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for branch in ("global", "local"):
            page = branch_rank.get(branch)
            if not page:
                continue
            page_id = str(page.get("citation_id") or page.get("page_id") or page.get("doc_id") or "")
            if page_id and page_id not in selected_ids:
                selected.append(page)
                selected_ids.add(page_id)

        for page in pages:
            page_id = str(page.get("citation_id") or page.get("page_id") or page.get("doc_id") or "")
            if not page_id or page_id in selected_ids:
                continue
            selected.append(page)
            selected_ids.add(page_id)

        selected.sort(
            key=lambda item: float(item.get("final_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        selected = self.retriever._apply_token_budget(selected, token_budget)[: max(1, int(max_results or 0))]

        return {
            "retrieved_pages": selected,
            "hit_ids": [
                page.get("citation_id") or page.get("page_id")
                for page in selected
                if page.get("citation_id") or page.get("page_id")
            ],
            "salvage_count": salvage_count,
            "source_stats": source_stats,
            "retrieval_profile": {
                "query_kind": "hybrid",
                "branches": [profile.get("query_kind") for profile in [
                    result.get("retrieval_profile", {}) if isinstance(result.get("retrieval_profile"), dict) else {}
                    for result in retrieval_results
                ] if profile.get("query_kind")],
            },
        }

    def _run_retrieval(
        self,
        query: str,
        *,
        max_results: int,
        token_budget: int,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        routing_plan = {}
        if isinstance(trace_context, dict):
            payload = trace_context.get("task_routing_plan")
            if isinstance(payload, dict):
                routing_plan = payload
        if not routing_plan:
            return self.retriever.retrieve_context(
                query,
                top_k=max_results,
                token_budget=token_budget,
                trace_context=trace_context,
            )
        retrieval_mode = resolve_effective_retrieval_mode(routing_plan)
        explicit_query_kind = str((trace_context or {}).get("query_kind", "") or "").strip().lower() if isinstance(trace_context, dict) else ""

        if retrieval_mode == "hybrid" and explicit_query_kind not in {"global", "local"}:
            branches = []
            for branch in ("global", "local"):
                branch_context = dict(trace_context or {})
                branch_context["query_kind"] = branch
                branch_context["hybrid_branch"] = branch
                branches.append(
                    self.retriever.retrieve_context(
                        query,
                        top_k=max_results,
                        token_budget=token_budget,
                        trace_context=branch_context,
                    )
                )
            return self._merge_hybrid_retrieval_results(
                branches,
                max_results=max_results,
                token_budget=token_budget,
            )

        return self.retriever.retrieve_context(
            query,
            top_k=max_results,
            token_budget=token_budget,
            trace_context=trace_context,
        )

    def _generate_answer_bundle(
        self,
        query: str,
        retrieval_results: Dict[str, Any],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pages = retrieval_results.get("retrieved_pages", [])
        task_hint = str((trace_context or {}).get("task_hint", "") or "") if isinstance(trace_context, dict) else ""
        routing_plan_payload = {}
        if isinstance(trace_context, dict):
            payload = trace_context.get("task_routing_plan")
            if isinstance(payload, dict):
                routing_plan_payload = dict(payload)
        answer_mode = resolve_effective_answer_mode(routing_plan_payload)
        answer_format = resolve_effective_answer_format(routing_plan_payload)
        canonical_evidence = build_canonical_evidence_trace(
            query,
            pages,
            task_hint=task_hint,
            answer_format=answer_format,
        )
        if not pages:
            return {
                "thought_process": "No supporting memory was retrieved.",
                "final_answer": "I don't know.",
                "citations": [],
                "canonical_evidence": canonical_evidence,
            }

        allowed_ids = []
        context_blocks = []
        for page in pages:
            citation_id = page.get("citation_id") or page.get("page_id")
            if not citation_id:
                continue
            allowed_ids.append(citation_id)
            context_blocks.append(
                f"[{citation_id}]\n{page.get('raw_content', '') or page.get('summary', '')}"
            )

        retrieval_mode = resolve_effective_retrieval_mode(routing_plan_payload)
        adaptive_state = build_adaptive_answer_state(
            query,
            retrieval_results,
            trace_context=trace_context,
            retrieval_mode=retrieval_mode,
            answer_mode=answer_mode,
            answer_format=answer_format,
        )
        adaptive_action = select_adaptive_answer_action(adaptive_state)
        adaptive_overrides = apply_adaptive_answer_action(
            adaptive_state,
            adaptive_action,
            routing_plan_payload,
        )
        effective_answer_mode = str(adaptive_overrides.get("effective_answer_mode") or answer_mode or "abstractive")
        effective_answer_format = str(adaptive_overrides.get("effective_answer_format") or answer_format or "freeform")
        retry_strategy = str(adaptive_overrides.get("retry_strategy") or "none")
        abstain_policy = str(adaptive_overrides.get("abstain_policy") or "allow")
        instruction_fragments = [
            str(fragment).strip()
            for fragment in (adaptive_overrides.get("instruction_fragments") or [])
            if str(fragment).strip()
        ]
        mode_task_hint = "hybrid" if retrieval_mode == "hybrid" else task_hint

        def _answer_looks_abstained(text: str) -> bool:
            lowered = str(text or "").strip().lower()
            if not lowered:
                return True
            return lowered in {
                "i don't know.",
                "i do not know.",
                "don't know",
                "do not know",
                "unknown",
                "not enough information",
            } or "don't know" in lowered

        def _build_prompt(prompt_pages: List[Dict[str, Any]], *, retry_note: str = "") -> tuple[str, List[str]]:
            prompt_allowed_ids = []
            prompt_context_blocks = []
            for page in prompt_pages:
                citation_id = page.get("citation_id") or page.get("page_id")
                if not citation_id:
                    continue
                prompt_allowed_ids.append(citation_id)
                prompt_context_blocks.append(
                    f"[{citation_id}]\n{page.get('raw_content', '') or page.get('summary', '')}"
                )
            answer_instruction = build_answer_mode_instruction(
                effective_answer_mode,
                task_hint=mode_task_hint,
                answer_format=effective_answer_format,
            )
            if abstain_policy == "allow":
                abstain_line = (
                    'If the context is insufficient, final_answer must be exactly "I don\'t know." and citations must be [].'
                )
            elif abstain_policy == "discourage":
                abstain_line = (
                    'Only use "I don\'t know." if the retrieved context clearly lacks the needed evidence; otherwise provide the best supported answer.'
                )
            else:
                abstain_line = (
                    'Do not answer with "I don\'t know."; provide the strongest supported answer from the retrieved context.'
                )
            guidance = instruction_fragments[:]
            if retry_note:
                guidance.append(str(retry_note).strip())
            guidance_block = "\n".join(f"- {line}" for line in guidance if line)
            prompt = (
                "You are answering using only the provided memory context.\n"
                "Return strict JSON with keys thought_process, final_answer, citations.\n"
                f"{abstain_line}\n"
                f"Answer mode: {effective_answer_mode or 'abstractive'}.\n"
                f"Answer format: {effective_answer_format or 'freeform'}.\n"
                f"Answer instruction: {answer_instruction}\n"
                + (f"Adaptive guidance:\n{guidance_block}\n" if guidance_block else "")
                + f"Allowed citation ids: {prompt_allowed_ids}\n\n"
                + f"Question:\n{query}\n\n"
                + "Memory context:\n"
                + f"{chr(10).join(prompt_context_blocks)}"
            )
            return prompt, prompt_allowed_ids

        def _run_answer_pass(prompt_pages: List[Dict[str, Any]], *, retry_note: str = "") -> Dict[str, Any]:
            prompt, prompt_allowed_ids = _build_prompt(prompt_pages, retry_note=retry_note)
            guidance = instruction_fragments[:]
            if retry_note:
                guidance.append(str(retry_note).strip())

            if effective_answer_format == "choice" and bool(getattr(self, "enable_choice_answer_head", False)):
                choice_result = answer_choice_with_context(
                    client=self.client,
                    model=self.llm_model,
                    question_text=query,
                    pages=prompt_pages,
                    max_tokens=160,
                    extra_guidance=guidance,
                    allowed_citation_ids=prompt_allowed_ids,
                    task_instruction=build_answer_mode_instruction(
                        effective_answer_mode,
                        task_hint=mode_task_hint,
                        answer_format=effective_answer_format,
                    ),
                    prefer_supported_choice=abstain_policy in {"discourage", "disallow"},
                )
                if choice_result.get("final_answer"):
                    return choice_result

            parsed: Dict[str, Any] = {}
            raw_response = ""
            try:
                raw_response = self.client.chat_completion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=320,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(raw_response)
            except Exception:
                parsed = {}

            citations = [
                citation_id
                for citation_id in parsed.get("citations", [])
                if isinstance(citation_id, str) and citation_id in prompt_allowed_ids
            ]
            final_answer = parsed.get("final_answer")
            if not isinstance(final_answer, str) or not final_answer.strip():
                final_answer = retrieval_results.get("context_summary", "") or "I don't know."
            final_answer = normalize_generated_answer(
                final_answer,
                effective_answer_mode,
                task_hint=task_hint,
                answer_format=effective_answer_format,
            ) or "I don't know."
            thought_process = parsed.get("thought_process")
            if not isinstance(thought_process, str) or not thought_process.strip():
                thought_process = "Grounded in retrieved memory context."
            return {
                "thought_process": thought_process,
                "final_answer": final_answer,
                "citations": citations,
                "raw_model_output": raw_response,
            }

        answer_pass = _run_answer_pass(pages)
        retry_count = 0
        if should_retry_adaptive_answer(
            answer_pass.get("final_answer", ""),
            state=adaptive_state,
            action=adaptive_action,
            answer_mode=effective_answer_mode,
            answer_format=effective_answer_format,
        ):
            retry_pages = pages
            retry_note = "Re-check the answer and stay grounded in the retrieved context."
            if retry_strategy == "retry_with_shorter_context":
                retry_pages = pages[: max(1, min(3, len(pages)))]
                retry_note = "The first answer may have been too diffuse. Re-answer using only the most relevant evidence."
            elif retry_strategy == "retry_with_stricter_format":
                retry_note = "Re-answer and follow the required answer format exactly."
            retry_pass = _run_answer_pass(retry_pages, retry_note=retry_note)
            retry_count = 1
            if (
                _answer_looks_abstained(answer_pass.get("final_answer", ""))
                and not _answer_looks_abstained(retry_pass.get("final_answer", ""))
            ) or not _answer_looks_abstained(retry_pass.get("final_answer", "")):
                answer_pass = retry_pass

        citations = answer_pass.get("citations", [])
        final_answer = answer_pass.get("final_answer", "I don't know.")
        thought_process = answer_pass.get("thought_process", "Grounded in retrieved memory context.")
        raw_response = answer_pass.get("raw_model_output", "")

        if citations and getattr(self, "enable_async_nli_verify", False):
            self.lifecycle.schedule_nli_verify_citations(final_answer, citations, trace_context=trace_context)

        adaptive_log_path = str(getattr(self, "adaptive_answer_log_path", "") or "").strip()
        if adaptive_log_path:
            record_adaptive_answer_outcome(
                adaptive_state,
                adaptive_action,
                AdaptiveAnswerOutcome(
                    metric_value=0.0,
                    format_ok=not _answer_looks_abstained(final_answer) or abstain_policy == "allow",
                    support_ok=None,
                    abstained=_answer_looks_abstained(final_answer),
                    latency_ms=0.0,
                ),
                adaptive_log_path,
            )

        return {
            "thought_process": thought_process,
            "final_answer": final_answer,
            "citations": citations,
            "raw_model_output": raw_response,
            "canonical_evidence": canonical_evidence,
            "adaptive_answer_state": adaptive_state.to_payload(),
            "adaptive_answer_action": adaptive_action.to_payload(),
            "adaptive_answer_overrides": adaptive_overrides,
            "adaptive_retry_count": retry_count,
        }

    def add_memory(
        self,
        user_input: str,
        agent_response: str,
        timestamp: str = None,
        meta_data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        timestamp = timestamp or get_timestamp()
        extracted = self._extract_metadata(user_input, agent_response, force_local_entities=True)
        external_metadata = self._normalize_external_metadata(meta_data or metadata)
        merged_metadata = self._merge_memory_metadata(extracted, external_metadata)
        g0 = self._calculate_smart_g0(user_input, merged_metadata)
        if external_metadata.get("salience_seed") is not None:
            try:
                g0 = max(g0, float(external_metadata["salience_seed"]))
            except Exception:
                pass
        self.last_ingest_result = self.updater.ingest_new_memory(
            user_input,
            agent_response,
            g0,
            metadata=merged_metadata,
            timestamp=timestamp,
        ) or {}
        if (
            getattr(self, "enable_async_entity_enrichment", False)
            and not getattr(self, "lightweight_extraction", False)
            and self.client is not None
        ):
            page_id = self.last_ingest_result.get("page_id") or merged_metadata.get("page_id")
            doc_id = self.last_ingest_result.get("doc_id") or merged_metadata.get("doc_id") or page_id
            self.lifecycle.schedule_metadata_enrichment(
                self._async_enrich_memory_entities,
                page_id,
                doc_id,
                user_input,
                agent_response,
                external_metadata,
            )
        self._ingest_count += 1
        if self._ingest_count % 10 == 0:
            self.lifecycle.schedule_soft_decay_memory()

    def get_response(
        self,
        query: str,
        max_results: int = 10,
        token_budget: int = 2200,
        trace_context: Optional[Dict[str, Any]] = None,
        *,
        query_type: str = "",
        task_hint: str = "",
        preferred_query_kind: str = "",
        preferred_answer_mode: str = "",
        benchmark: str = "",
    ) -> MemoryQueryResult:
        trace_context = self.build_retrieval_trace_context(
            query,
            trace_context=trace_context,
            query_type=query_type,
            task_hint=task_hint,
            preferred_query_kind=preferred_query_kind,
            preferred_answer_mode=preferred_answer_mode,
            benchmark=benchmark,
        )
        if isinstance(trace_context, dict):
            trace_context.setdefault("token_budget", int(token_budget))
            trace_context.setdefault("max_results", int(max_results))
        retrieval_results = self._run_retrieval(
            query,
            max_results=max_results,
            token_budget=token_budget,
            trace_context=trace_context,
        )
        retrieval_results["context_summary"] = self.retriever.get_context_summary(retrieval_results)
        retrieval_results.update(self._generate_answer_bundle(query, retrieval_results, trace_context=trace_context))
        self.current_session_hit_ids = retrieval_results.get("hit_ids", [])
        background_targets = retrieval_results.get("citations", []) or self.current_session_hit_ids[:3]
        base_feedback_score = self._derive_feedback_score(
            retrieval_results.get("final_answer", ""),
            retrieval_results.get("citations", []),
            self.current_session_hit_ids,
        )
        reflection = self.lifecycle.build_interaction_reflection(
            query=query,
            final_answer=retrieval_results.get("final_answer", ""),
            retrieval_results=retrieval_results,
            base_feedback_score=base_feedback_score,
        )
        retrieval_results["reflection"] = reflection
        feedback_score = self._clamp01(reflection.get("feedback_score", base_feedback_score))
        self._retrieval_count += 1
        if background_targets and feedback_score > 0.0:
            self.lifecycle.schedule_post_inference_memory_flow(
                self,
                background_targets,
                feedback_score=feedback_score,
                reflection=reflection,
                trace_context=trace_context,
            )
        if self.current_session_hit_ids:
            self.lifecycle.schedule_pointer_compress_candidates(
                self.current_session_hit_ids[:3],
                trace_context=trace_context,
            )
        if self._retrieval_count % 20 == 0:
            self.lifecycle.schedule_soft_decay_memory()
        return MemoryQueryResult(retrieval_results, answer_text=retrieval_results.get("final_answer", ""))

    def retrieve_context(
        self,
        query: str,
        max_results: int = 10,
        token_budget: int = 2200,
        trace_context: Optional[Dict[str, Any]] = None,
        *,
        query_type: str = "",
        task_hint: str = "",
        preferred_query_kind: str = "",
        preferred_answer_mode: str = "",
        benchmark: str = "",
    ) -> MemoryQueryResult:
        trace_context = self.build_retrieval_trace_context(
            query,
            trace_context=trace_context,
            query_type=query_type,
            task_hint=task_hint,
            preferred_query_kind=preferred_query_kind,
            preferred_answer_mode=preferred_answer_mode,
            benchmark=benchmark,
        )
        if isinstance(trace_context, dict):
            trace_context.setdefault("token_budget", int(token_budget))
            trace_context.setdefault("max_results", int(max_results))
        retrieval_results = self._run_retrieval(
            query,
            max_results=max_results,
            token_budget=token_budget,
            trace_context=trace_context,
        )
        retrieval_results["context_summary"] = self.retriever.get_context_summary(retrieval_results)
        return MemoryQueryResult(retrieval_results, answer_text=retrieval_results.get("context_summary", ""))

    def build_retrieval_trace_context(
        self,
        query: str,
        *,
        trace_context: Optional[Dict[str, Any]] = None,
        query_type: str = "",
        task_hint: str = "",
        preferred_query_kind: str = "",
        preferred_answer_mode: str = "",
        benchmark: str = "",
    ) -> Optional[Dict[str, Any]]:
        base_context = dict(trace_context or {})
        if benchmark and "benchmark" not in base_context:
            base_context["benchmark"] = str(benchmark)

        resolved_query_type = str(query_type or base_context.get("query_type") or "").strip()
        resolved_task_hint = str(task_hint or base_context.get("task_hint") or "").strip()
        resolved_preferred_kind = str(
            preferred_query_kind or base_context.get("query_kind") or ""
        ).strip()
        resolved_preferred_answer_mode = str(
            preferred_answer_mode or base_context.get("answer_mode") or ""
        ).strip()

        route_enabled = bool(
            resolved_query_type or resolved_task_hint or resolved_preferred_kind or resolved_preferred_answer_mode
        )
        if not route_enabled:
            return base_context or None

        return build_retrieval_trace_context(
            query,
            query_type=resolved_query_type,
            task_hint=resolved_task_hint,
            preferred_kind=resolved_preferred_kind,
            preferred_answer_mode=resolved_preferred_answer_mode,
            extra=base_context,
        )

    def build_task_routing_plan(
        self,
        query: str,
        *,
        query_type: str = "",
        task_hint: str = "",
        preferred_query_kind: str = "",
        preferred_answer_mode: str = "",
        extra_signals: Optional[Dict[str, Any]] = None,
    ) -> TaskRoutingPlan:
        return infer_task_routing_plan(
            query,
            query_type=query_type,
            task_hint=task_hint,
            preferred_retrieval_mode=preferred_query_kind,
            preferred_answer_mode=preferred_answer_mode,
            extra_signals=extra_signals,
        )

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        if not memory_id:
            return None

        stm_page = self.short_term_memory.get_page_by_id(memory_id)
        if stm_page:
            page = dict(stm_page)
            page.setdefault("citation_id", page.get("page_id"))
            page.setdefault("doc_id", page.get("doc_id", page.get("page_id")))
            page.setdefault("summary", page.get("metadata", {}).get("topic", ""))
            page.setdefault("raw_content", f"Q: {page.get('user_input', '')}\nA: {page.get('agent_response', '')}".strip())
            page.setdefault("salience_seed", page.get("gcl_scores", {}).get("G0", 0.5))
            page.setdefault("T", page.get("T", 100.0))
            return self._decorate_memory_page(page, location="short_term")

        ltm_entry = self.long_term_memory.get_entry_by_id(memory_id)
        if ltm_entry:
            page = self.retriever._format_node_to_page(ltm_entry, source="LTM_DIRECT")
            return self._decorate_memory_page(page, location="long_term")

        forget_page = self.forget_memory.get_page_by_id(memory_id)
        if forget_page:
            page = self.retriever._format_cold_record_to_page(forget_page, source="FORGET_DIRECT")
            return self._decorate_memory_page(page, location="forget_pool")

        return None

    def node_exists(self, node_id: str) -> bool:
        return self.lifecycle.node_exists(node_id)

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.lifecycle.get_memory_stats()

    def get_background_status(self) -> Dict[str, Any]:
        return self.lifecycle.get_background_status()

    def get_last_ingest_result(self) -> Dict[str, Any]:
        return dict(self.last_ingest_result or {})

    def get_user_profile(self) -> str:
        return self.user_long_term_memory.get_raw_user_profile(self.user_id)

    def nli_verify_citations(self, generated_answer: str, citation_ids: List[str]) -> Dict[str, Any]:
        return self.lifecycle.nli_verify_citations(generated_answer, citation_ids)

    def soft_decay_memory(self, half_life_days: int = 7):
        self.lifecycle.soft_decay_memory(half_life_days=half_life_days)

    def pointer_compress(self, chunk_id: str) -> Optional[str]:
        return self.lifecycle.pointer_compress(chunk_id)

    def post_inference_memory_flow(
        self,
        page_ids: List[str],
        feedback_score: float = 1.0,
        background: bool = True,
        reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if background:
            self.lifecycle.schedule_post_inference_memory_flow(
                self,
                page_ids,
                feedback_score=feedback_score,
                reflection=reflection,
            )
            return {
                "status": "scheduled",
                "page_ids": list(dict.fromkeys(page_ids or [])),
                "background_status": self.get_background_status(),
            }
        return self.lifecycle.post_inference_memory_flow(
            self,
            page_ids,
            feedback_score=feedback_score,
            reflection=reflection,
        )

    def clear_all(self):
        self.lifecycle.clear_all()

    def shutdown(self, wait_for_background_tasks: bool = False):
        self.lifecycle.shutdown(wait=wait_for_background_tasks)
