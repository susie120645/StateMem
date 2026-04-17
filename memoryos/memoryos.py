import json
import hashlib
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from .entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate
    from .forget_memory import ForgetMemory
    from .long_term import LongTermMemory
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

    from entity_filters import filter_entity_values, is_structural_entity_label, normalize_entity_candidate
    from forget_memory import ForgetMemory
    from long_term import LongTermMemory
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
        assistant_id: str = DEFAULT_ASSISTANT_ID,
        short_term_capacity: int = 15,
        long_term_knowledge_capacity: int = 5000,
        retrieval_queue_capacity: int = 30,
        llm_model: str = "qwen2.5:7b",
        embedding_model_name: str = "nomic-embed-text",
        embedding_model_kwargs: dict = None,
        trace_recorder: Optional[TraceRecorder] = None,
    ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.trace_recorder = trace_recorder
        self.current_session_hit_ids: List[str] = []
        self._entity_cache: Dict[str, List[str]] = {}
        self._keyword_cache: Dict[str, List[str]] = {}
        self._ingest_count = 0
        self._retrieval_count = 0
        self.last_ingest_result: Dict[str, Any] = {}
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)

        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_long_term_dir = os.path.join(self.user_data_dir, "long_term")
        user_forget_path = os.path.join(self.user_data_dir, "forget_pool.jsonl")

        ensure_directory_exists(user_short_term_path)
        ensure_directory_exists(user_forget_path)

        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.forget_memory = ForgetMemory(
            file_path=user_forget_path,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_dir,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
        )
        self.long_term_memory = self.user_long_term_memory
        self.short_term_memory.keyword_extractor = self._extract_keywords
        self.long_term_memory.entity_extractor = self._extract_entities
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
        self.forget_memory.entity_extractor = self._extract_entities
        self.forget_memory.keyword_extractor = self._extract_keywords
        self.retriever = Retriever(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.user_long_term_memory,
            forget_memory=self.forget_memory,
            queue_capacity=retrieval_queue_capacity,
        )
        self.retriever.entity_extractor = self._extract_entities
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
                embedding_model_name=self.embedding_model_name,
            )
        self.original_add_memory = self.add_memory

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

        prompt = (
            "You are a keyword extraction component for a memory retrieval system.\n"
            "Task: identify the most salient retrieval keywords from the text.\n"
            "Focus on nouns, verbs, named entities, tasks, products, locations, dates, and key concepts.\n"
            "Return strict JSON only in this format: {\"keywords\": [\"keyword1\", \"keyword2\"]}.\n"
            "Use short phrases copied from the text when possible.\n"
            "If the text is English, prefer English keywords.\n"
            "Ignore pronouns, greetings, filler words, structural labels, and generic metadata fields.\n"
            "Do not output labels like Summary, Session Summary, Event Summary, Transcript, Query, Question, Answer, User, Assistant, or System.\n"
            f"Return at most {max_keywords} keywords sorted by importance. If nothing is useful, return {{\"keywords\": []}}.\n\n"
            f"Text:\n{prepared_text}"
        )

        keywords: List[str] = []
        for response_format in ({"type": "json_object"}, None):
            raw_response = self.client.chat_completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
                response_format=response_format,
            )
            keywords = self._normalize_llm_keywords(
                self._parse_entity_payload(raw_response),
                limit=max_keywords,
            )
            if keywords:
                break

        if not keywords:
            keywords = fallback_extract_keywords(prepared_text, limit=max_keywords)

        cache[cache_key] = list(keywords)
        return list(keywords)

    def _extract_metadata(self, user_input: str, agent_response: str) -> Dict[str, Any]:
        safe_input = mask_system_tags(user_input)
        safe_response = mask_system_tags(agent_response)
        combined = f"{safe_input}\n{safe_response}".strip()
        entities = self._extract_entities(combined)

        topic = "General"
        if re.search(r"\b(error|traceback|bug|exception|stack)\b", combined, re.I):
            topic = "Debugging"
        elif re.search(r"\b(architecture|memory|rag|retriev|vector|graph|agent)\b", combined, re.I):
            topic = "Architecture"
        elif re.search(r"\b(prefer|like|always|never)\b", combined, re.I):
            topic = "Preference"

        semantic_edges = entities[:]
        return {
            "topic": topic,
            "entities": entities,
            "semantic_edges": semantic_edges,
            "flags": {},
        }

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

    def _generate_answer_bundle(
        self,
        query: str,
        retrieval_results: Dict[str, Any],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pages = retrieval_results.get("retrieved_pages", [])
        if not pages:
            return {
                "thought_process": "No supporting memory was retrieved.",
                "final_answer": "I don't know.",
                "citations": [],
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

        prompt = (
            "You are answering using only the provided memory context.\n"
            "Return strict JSON with keys thought_process, final_answer, citations.\n"
            "If the context is insufficient, final_answer must be exactly \"I don't know.\" and citations must be [].\n"
            f"Allowed citation ids: {allowed_ids}\n\n"
            f"Question:\n{query}\n\n"
            "Memory context:\n"
            f"{chr(10).join(context_blocks)}"
        )

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
            if isinstance(citation_id, str) and citation_id in allowed_ids
        ]
        final_answer = parsed.get("final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            final_answer = retrieval_results.get("context_summary", "") or "I don't know."
        thought_process = parsed.get("thought_process")
        if not isinstance(thought_process, str) or not thought_process.strip():
            thought_process = "Grounded in retrieved memory context."

        if citations:
            self.lifecycle.schedule_nli_verify_citations(final_answer, citations, trace_context=trace_context)

        return {
            "thought_process": thought_process,
            "final_answer": final_answer,
            "citations": citations,
            "raw_model_output": raw_response,
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
        extracted = self._extract_metadata(user_input, agent_response)
        external_metadata = self._normalize_external_metadata(meta_data or metadata)
        combined = f"{user_input}\n{agent_response}".strip()
        shared_keywords = normalize_keyword_values(external_metadata.get("keywords", []), limit=32)
        if not shared_keywords:
            shared_keywords = self._extract_keywords(combined, max_keywords=16)
        merged_metadata = {**extracted, **external_metadata}
        merged_metadata["entities"] = list(
            dict.fromkeys(extracted.get("entities", []) + external_metadata.get("entities", []))
        )
        merged_metadata["semantic_edges"] = list(
            dict.fromkeys(extracted.get("semantic_edges", []) + external_metadata.get("semantic_edges", []))
        )
        merged_metadata["keywords"] = list(shared_keywords)
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
        self._ingest_count += 1
        if self._ingest_count % 10 == 0:
            self.lifecycle.schedule_soft_decay_memory()

    def get_response(
        self,
        query: str,
        max_results: int = 10,
        token_budget: int = 2200,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> MemoryQueryResult:
        retrieval_results = self.retriever.retrieve_context(
            query,
            top_k=max_results,
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
    ) -> MemoryQueryResult:
        retrieval_results = self.retriever.retrieve_context(
            query,
            top_k=max_results,
            token_budget=token_budget,
            trace_context=trace_context,
        )
        retrieval_results["context_summary"] = self.retriever.get_context_summary(retrieval_results)
        return MemoryQueryResult(retrieval_results, answer_text=retrieval_results.get("context_summary", ""))

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
