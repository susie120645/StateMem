import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    from .entity_filters import filter_entity_values, filter_semantic_edge_values
    from .utils import (
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        normalize_keyword_values,
        normalize_vector,
    )
except ImportError:
    from entity_filters import filter_entity_values, filter_semantic_edge_values
    from utils import (
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        normalize_keyword_values,
        normalize_vector,
    )


class ForgetMemory:
    """Append-only cold storage with lightweight hybrid retrieval."""

    DEFAULT_COLD_CONFIDENCE = 0.5
    DEFAULT_COLD_TEMPERATURE = 18.0

    def __init__(
        self,
        file_path: str,
        embedding_model_name: str = "nomic-embed-text",
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self._id_offset_index: Dict[str, int] = {}
        self._record_cache: Dict[str, Dict[str, Any]] = {}
        self._ensure_file_exists()
        self._build_indexes()
        print(
            f"ForgetMemory: Initialized with {len(self._record_cache)} cold records.",
            file=sys.stderr,
        )

    def _ensure_file_exists(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8"):
                pass

    def _build_indexes(self) -> None:
        self._id_offset_index = {}
        self._record_cache = {}
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, "rb") as handle:
                offset = 0
                for line in handle:
                    try:
                        item = json.loads(line.decode("utf-8"))
                        record = self._normalize_loaded_item(item)
                        page_id = record.get("page_id")
                        if page_id:
                            self._id_offset_index[page_id] = offset
                            self._record_cache[page_id] = record
                    except Exception:
                        pass
                    offset += len(line)
        except Exception as exc:
            logging.error("ForgetMemory index error: %s", exc)

    def _extract_entities(self, text: str) -> List[str]:
        extractor = getattr(self, "entity_extractor", None)
        if callable(extractor):
            try:
                entities = extractor(text)
                return filter_entity_values(entities, limit=24)
            except Exception:
                return []
        entities = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{1,}\b", text or "")
        return filter_entity_values(list(dict.fromkeys(entities)), limit=24)

    def _extract_keywords(self, text: str) -> List[str]:
        extractor = getattr(self, "keyword_extractor", None)
        if callable(extractor):
            try:
                keywords = normalize_keyword_values(extractor(text), limit=48)
                if keywords:
                    return keywords
            except Exception:
                pass
        return fallback_extract_keywords(text, limit=48)

    def _summarize_text(self, text: str, max_chars: int = 220) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _build_embedding(self, text: str) -> List[float]:
        vector = get_embedding(
            text,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs,
        )
        if not vector:
            return []
        return normalize_vector(np.array(vector, dtype=np.float32)).tolist()

    def _normalize_loaded_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if "record_version" in item or "page_id" in item:
            page_id = item.get("page_id") or item.get("pid") or "unknown"
            user_input = item.get("user_input") or item.get("act") or ""
            agent_response = item.get("agent_response", "")
            if not agent_response and item.get("ans") not in {None, "NULL"}:
                agent_response = item.get("ans", "")
            raw_content = item.get("raw_content") or f"Q: {user_input}\nA: {agent_response}".strip()
            summary = item.get("summary") or self._summarize_text(raw_content)
            entities = item.get("entities") or item.get("ent") or self._extract_entities(f"{user_input}\n{agent_response}")
            keywords = normalize_keyword_values(item.get("keywords") or [], limit=48)
            if not keywords:
                keywords = fallback_extract_keywords(f"{summary}\n{raw_content}", limit=48)
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"legacy_metadata": metadata}
            if not entities:
                entities = metadata.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            entities = filter_entity_values(entities, limit=24)
            metadata["entities"] = filter_entity_values(metadata.get("entities", []), limit=24)
            metadata["semantic_edges"] = filter_semantic_edge_values(metadata.get("semantic_edges", []))
            metadata["keywords"] = list(keywords)
            legacy_topic = metadata.get("topic", "")
            legacy_action = metadata.get("action", "")
            if (not item.get("summary")) and (legacy_topic or legacy_action):
                action_text = legacy_action
                if isinstance(action_text, list):
                    action_text = ", ".join(str(part) for part in action_text[:4])
                summary = self._summarize_text(
                    " | ".join(part for part in [legacy_topic, user_input, str(action_text)] if part)
                )
            embedding = item.get("embedding", []) if isinstance(item.get("embedding", []), list) else []
            if not embedding:
                embed_text = self._summarize_text(f"{summary}\n{user_input}\n{agent_response}", max_chars=1200)
                embedding = self._build_embedding(embed_text)
            return {
                "record_version": int(item.get("record_version", 2)),
                "page_id": page_id,
                "doc_id": item.get("doc_id") or page_id,
                "user_input": user_input,
                "agent_response": agent_response,
                "raw_content": raw_content,
                "summary": summary,
                "timestamp": item.get("timestamp") or item.get("ts") or get_timestamp(),
                "archived_at": item.get("archived_at") or item.get("f_at") or get_timestamp(),
                "entities": entities,
                "keywords": [kw for kw in keywords if isinstance(kw, str)],
                "embedding": embedding,
                "G": float(item.get("G", item.get("g0", 0.0))),
                "C": float(item.get("C", self.DEFAULT_COLD_CONFIDENCE)),
                "T": float(item.get("T", self.DEFAULT_COLD_TEMPERATURE)),
                "salience_seed": float(item.get("salience_seed", item.get("g0", 0.35))),
                "storage_tier": "cold",
                "source_kind": item.get("source_kind", "forget_pool"),
                "metadata": metadata,
            }

        page_id = item.get("pid") or "unknown"
        user_input = item.get("act", "")
        agent_response = "" if item.get("ans") in {None, "NULL"} else item.get("ans", "")
        raw_content = f"Q: {user_input}\nA: {agent_response}".strip()
        embedding = item.get("embedding", []) if isinstance(item.get("embedding", []), list) else []
        if not embedding:
            embedding = self._build_embedding(self._summarize_text(raw_content, max_chars=1200))
        return {
            "record_version": 1,
            "page_id": page_id,
            "doc_id": page_id,
            "user_input": user_input,
            "agent_response": agent_response,
            "raw_content": raw_content,
            "summary": self._summarize_text(raw_content),
            "timestamp": item.get("ts") or get_timestamp(),
            "archived_at": item.get("f_at") or get_timestamp(),
            "entities": filter_entity_values(item.get("ent", []), limit=24),
            "keywords": fallback_extract_keywords(raw_content, limit=48),
            "embedding": embedding,
            "G": float(item.get("g0", 0.0)),
            "C": self.DEFAULT_COLD_CONFIDENCE,
            "T": self.DEFAULT_COLD_TEMPERATURE,
            "salience_seed": float(item.get("g0", 0.35)),
            "storage_tier": "cold",
            "source_kind": "forget_pool_legacy",
            "metadata": {"legacy_record": True},
        }

    def _build_cold_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        user_input = str(item.get("user_input", "")).strip()
        agent_response = str(item.get("agent_response", "")).strip()
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"legacy_metadata": metadata}

        combined = f"{user_input}\n{agent_response}".strip()
        entities = metadata.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        if not entities:
            entities = self._extract_entities(combined)
        entities = filter_entity_values(entities, limit=24)
        semantic_edges = metadata.get("semantic_edges", [])
        if not isinstance(semantic_edges, list):
            semantic_edges = []
        if not semantic_edges:
            semantic_edges = entities[:]
        semantic_edges = filter_semantic_edge_values(semantic_edges, limit=48)
        metadata["entities"] = entities
        metadata["semantic_edges"] = semantic_edges

        summary_parts = []
        if user_input:
            summary_parts.append(user_input)
        if agent_response:
            summary_parts.append(agent_response[:240])
        summary = self._summarize_text(" | ".join(summary_parts) or combined)
        keywords = normalize_keyword_values(item.get("keywords") or metadata.get("keywords") or [], limit=48)
        if not keywords:
            keywords = self._extract_keywords(f"{summary}\n{combined}")
        metadata["keywords"] = list(keywords)
        embedding_text = self._summarize_text(f"{summary}\n{combined}", max_chars=1200)
        raw_content = f"Q: {user_input}\nA: {agent_response}".strip()
        page_id = item.get("page_id") or item.get("pid") or generate_id("fgt")
        gcl_scores = item.get("gcl_scores", {})
        if not isinstance(gcl_scores, dict):
            gcl_scores = {}
        salience_seed = gcl_scores.get("G0")
        if salience_seed is None:
            salience_seed = metadata.get("salience_seed", metadata.get("G_base", 0.35))
        utility_score = item.get("G", metadata.get("G_base", 0.0))

        return {
            "record_version": 2,
            "page_id": page_id,
            "doc_id": item.get("doc_id") or page_id,
            "user_input": user_input,
            "agent_response": agent_response,
            "raw_content": raw_content,
            "summary": summary,
            "timestamp": item.get("timestamp") or get_timestamp(),
            "archived_at": get_timestamp(),
            "entities": entities,
            "semantic_edges": semantic_edges,
            "keywords": keywords,
            "embedding": self._build_embedding(embedding_text),
            "G": float(utility_score),
            "C": float(item.get("C", self.DEFAULT_COLD_CONFIDENCE)),
            "T": min(self.DEFAULT_COLD_TEMPERATURE, float(item.get("T", self.DEFAULT_COLD_TEMPERATURE))),
            "salience_seed": float(salience_seed),
            "storage_tier": "cold",
            "source_kind": "forget_pool",
            "metadata": {
                **metadata,
                "archived_from": "short_term",
                "original_page_id": item.get("page_id"),
            },
        }

    def add_page(self, item: Dict[str, Any]) -> None:
        try:
            record = self._build_cold_record(item)
            offset = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
            with open(self.file_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            page_id = record.get("page_id")
            if page_id:
                self._id_offset_index[page_id] = offset
                self._record_cache[page_id] = record
        except Exception as exc:
            logging.error("ForgetMemory write error: %s", exc)

    def add(self, user_input: str, agent_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Legacy compatibility alias used by the evaluation layer.
        """
        self.add_page(
            {
                "page_id": None,
                "doc_id": None,
                "user_input": user_input,
                "agent_response": agent_response,
                "metadata": metadata or {},
            }
        )

    def add_discard(self, user_input: str, agent_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Compatibility alias for evaluation flows that route low-value inputs
        directly into the forget pool.
        """
        self.add(user_input=user_input, agent_response=agent_response, metadata=metadata)

    def get_page_by_id(self, page_id: str) -> Optional[Dict[str, Any]]:
        record = self._record_cache.get(page_id)
        if record is not None:
            return dict(record)

        offset = self._id_offset_index.get(page_id)
        if offset is None:
            return None
        try:
            with open(self.file_path, "rb") as handle:
                handle.seek(offset)
                line = handle.readline().decode("utf-8")
            record = self._normalize_loaded_item(json.loads(line))
            self._record_cache[page_id] = record
            return dict(record)
        except Exception as exc:
            logging.error("ForgetMemory seek error: %s", exc)
            return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        exclude_ids = exclude_ids or set()
        query_embedding = np.array(
            get_embedding(query, model_name=self.embedding_model_name, **self.embedding_model_kwargs),
            dtype=np.float32,
        )
        if query_embedding.size == 0:
            query_embedding = np.zeros(768, dtype=np.float32)

        query_terms = set(self._extract_keywords(query))
        query_entities = {ent.lower() for ent in self._extract_entities(query)}
        query_lower = query.lower().strip()
        results = []

        for page_id, record in self._record_cache.items():
            if page_id in exclude_ids:
                continue

            item_embedding = np.array(record.get("embedding", []), dtype=np.float32)
            semantic_score = 0.0
            if item_embedding.size and query_embedding.size:
                denom = float(np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding))
                if denom > 0:
                    semantic_score = float(np.dot(query_embedding, item_embedding) / denom)

            record_terms = set(record.get("keywords", []))
            keyword_score = 0.0
            if query_terms and record_terms:
                keyword_score = len(query_terms & record_terms) / float(len(query_terms))

            record_entities = {ent.lower() for ent in record.get("entities", []) if isinstance(ent, str)}
            entity_score = 0.0
            if query_entities and record_entities:
                entity_score = len(query_entities & record_entities) / float(len(query_entities))

            haystack = " ".join(
                [
                    record.get("summary", ""),
                    record.get("user_input", ""),
                    record.get("agent_response", ""),
                ]
            ).lower()
            exact_bonus = 0.08 if query_lower and query_lower in haystack else 0.0
            surface_score = 0.0
            if query_terms:
                haystack_terms = record_terms or set(fallback_extract_keywords(haystack, limit=48))
                if haystack_terms:
                    surface_score = len(query_terms & haystack_terms) / float(len(query_terms))

            relevance = (
                (semantic_score * 0.58)
                + (keyword_score * 0.18)
                + (entity_score * 0.12)
                + (surface_score * 0.12)
                + exact_bonus
            )
            if relevance < threshold:
                continue

            candidate = dict(record)
            candidate["semantic_score"] = round(semantic_score, 4)
            candidate["keyword_score"] = round(keyword_score, 4)
            candidate["entity_score"] = round(entity_score, 4)
            candidate["surface_score"] = round(surface_score, 4)
            candidate["score"] = round(relevance, 4)
            results.append(candidate)

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]

    def entity_salvage(self, query: str) -> List[Dict[str, Any]]:
        return self.search(query, top_k=10)

    def get_stats(self) -> Dict[str, Any]:
        size_bytes = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        with_embeddings = sum(1 for record in self._record_cache.values() if record.get("embedding"))
        return {
            "lean_records": len(self._record_cache),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "embedded_records": with_embeddings,
            "storage_tier": "cold",
        }

    def clear(self) -> None:
        self._id_offset_index = {}
        self._record_cache = {}
        with open(self.file_path, "w", encoding="utf-8"):
            pass
