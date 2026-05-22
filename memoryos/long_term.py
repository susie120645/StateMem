import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    from .entity_filters import filter_entity_values, filter_semantic_edge_values
    from .utils import (
        OpenAIClient,
        ensure_directory_exists,
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        extract_temporal_canonical_forms,
        normalize_keyword_values,
        normalize_vector,
        write_json_with_backup,
    )
except ImportError:
    from entity_filters import filter_entity_values, filter_semantic_edge_values
    from utils import (
        OpenAIClient,
        ensure_directory_exists,
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        extract_temporal_canonical_forms,
        normalize_keyword_values,
        normalize_vector,
        write_json_with_backup,
    )


class LongTermMemory:
    """Graph-vector hybrid long-term memory with backward-compatible JSON storage."""

    HOT_TEMPERATURE_THRESHOLD = 25.0

    def __init__(
        self,
        file_path: str,
        knowledge_capacity: int = 5000,
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_model_kwargs: Optional[dict] = None,
        enable_backups: bool = True,
    ):
        self.file_path = self._resolve_storage_path(file_path)
        self.knowledge_capacity = knowledge_capacity
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.enable_backups = bool(enable_backups)
        self.data: Dict[str, Any] = {
            "version": 2,
            "user_profile": "None",
            "knowledge_base": [],
            "storage_meta": {
                "active_temperature_threshold": self.HOT_TEMPERATURE_THRESHOLD,
            },
        }
        self._node_map: Dict[str, Dict[str, Any]] = {}
        self._entity_index: Dict[str, Set[str]] = {}
        self._doc_children: Dict[str, List[str]] = {}
        self._pointer_index: Dict[str, Set[str]] = {}
        self.load()

    def _resolve_storage_path(self, path: str) -> str:
        if os.path.isdir(path):
            return os.path.join(path, "memory_graph.json")
        if not os.path.exists(path) and not os.path.splitext(path)[1]:
            os.makedirs(path, exist_ok=True)
            return os.path.join(path, "memory_graph.json")
        return path

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                self.data["version"] = loaded_data.get("version", 1)
                self.data["user_profile"] = loaded_data.get("user_profile", "None")
                self.data["knowledge_base"] = loaded_data.get("knowledge_base", [])
                self.data["storage_meta"] = loaded_data.get(
                    "storage_meta",
                    {"active_temperature_threshold": self.HOT_TEMPERATURE_THRESHOLD},
                )
            except Exception as exc:
                print(f"Error loading long-term memory: {exc}", file=sys.stderr)
        else:
            print(f"LongTermMemory: creating new store at {self.file_path}", file=sys.stderr)
            self.save()

        self._migrate_legacy_entries()
        self._rebuild_indexes()

    def save(self):
        ensure_directory_exists(self.file_path)
        try:
            write_json_with_backup(
                self.file_path,
                self.data,
                ensure_ascii=False,
                indent=2,
                create_backup=self.enable_backups,
            )
        except Exception as exc:
            print(f"Error saving long-term memory: {exc}", file=sys.stderr)

    def _migrate_legacy_entries(self):
        migrated = []
        changed = False
        for entry in self.data.get("knowledge_base", []):
            normalized = self._normalize_entry(entry)
            if normalized != entry:
                changed = True
            migrated.append(normalized)
        self.data["knowledge_base"] = migrated
        if changed:
            self.save()

    def _rebuild_indexes(self):
        self._node_map = {}
        self._entity_index = {}
        self._doc_children = {}
        self._pointer_index = {}

        for entry in self.data.get("knowledge_base", []):
            node_id = entry.get("id")
            if not node_id:
                continue

            self._node_map[node_id] = entry

            doc_id = entry.get("doc_id") or node_id
            if entry.get("node_type") == "chunk":
                self._doc_children.setdefault(doc_id, []).append(node_id)

            pointer_to = entry.get("pointer_to")
            if pointer_to:
                self._pointer_index.setdefault(pointer_to, set()).add(node_id)

            for entity in entry.get("entities", []):
                if not isinstance(entity, str):
                    continue
                self._entity_index.setdefault(entity.lower(), set()).add(node_id)

    def _normalize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        raw_metadata = entry.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {"legacy_metadata": raw_metadata}

        node_id = entry.get("id") or entry.get("chunk_id") or generate_id("amn")
        doc_id = entry.get("doc_id") or raw_metadata.get("doc_id") or node_id
        pointer_to = entry.get("pointer_to") or raw_metadata.get("pointer_to")

        raw_content = (
            entry.get("raw_content")
            or entry.get("content")
            or raw_metadata.get("raw_content")
            or ""
        )
        summary = (
            entry.get("summary")
            or entry.get("sketch")
            or entry.get("index_data")
            or raw_metadata.get("summary")
            or self._summarize_text(raw_content)
        )

        entities = filter_entity_values(
            entry.get("entities")
            or raw_metadata.get("entities")
            or self._extract_entities(f"{raw_content} {summary}")
        )
        semantic_edges = filter_semantic_edge_values(
            entry.get("semantic_edges")
            or raw_metadata.get("semantic_edges")
            or raw_metadata.get("pre_linked_ids")
            or []
        )
        keywords = normalize_keyword_values(
            entry.get("keywords") or raw_metadata.get("keywords") or [],
            limit=32,
        )
        if not keywords:
            keywords = fallback_extract_keywords(f"{summary} {raw_content}", limit=32)
        parent_id = entry.get("parent_id") or raw_metadata.get("parent_id")
        timestamp = entry.get("timestamp") or raw_metadata.get("timestamp") or get_timestamp()
        last_access_time = entry.get("last_access_time") or timestamp
        temperature = float(entry.get("T", raw_metadata.get("T", 100.0)))
        storage_tier = entry.get("storage_tier") or entry.get("location")
        if storage_tier not in {"hot", "cold", "pointer"}:
            storage_tier = "hot" if temperature >= self.HOT_TEMPERATURE_THRESHOLD else "cold"
        if pointer_to:
            storage_tier = "pointer"

        node_type = entry.get("node_type")
        if node_type not in {"document", "chunk", "summary"}:
            if pointer_to:
                node_type = "summary"
            elif parent_id:
                node_type = "chunk"
            else:
                node_type = "document"

        embedding = entry.get("embedding", [])
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        if not isinstance(embedding, list):
            embedding = []

        normalized = {
            "id": node_id,
            "doc_id": doc_id,
            "chunk_id": entry.get("chunk_id", node_id),
            "node_type": node_type,
            "raw_content": raw_content,
            "summary": summary,
            "embedding": embedding,
            "G": float(entry.get("G", raw_metadata.get("G", 0.0))),
            "C": float(entry.get("C", raw_metadata.get("C", 0.6))),
            "T": temperature,
            "salience_seed": float(entry.get("salience_seed", raw_metadata.get("salience_seed", 0.5))),
            "entities": entities,
            "semantic_edges": semantic_edges,
            "keywords": keywords,
            "parent_id": parent_id,
            "child_ids": self._unique_strings(entry.get("child_ids", [])),
            "pointer_to": pointer_to,
            "storage_tier": storage_tier,
            "timestamp": timestamp,
            "last_access_time": last_access_time,
            "metadata": raw_metadata,
            "source_kind": entry.get("source_kind", raw_metadata.get("source_kind", "conversation")),
            "retrieved_count": int(entry.get("retrieved_count", raw_metadata.get("retrieved_count", 0))),
        }

        normalized["metadata"].update(
            {
                "doc_id": normalized["doc_id"],
                "parent_id": normalized["parent_id"],
                "pointer_to": normalized["pointer_to"],
            }
        )
        if not isinstance(normalized["metadata"].get("temporal_forms"), list):
            normalized["metadata"]["temporal_forms"] = extract_temporal_canonical_forms(
                f"{normalized.get('summary', '')}\n{normalized.get('raw_content', '')}"
            )
        return normalized

    def _extract_keywords(self, text: str) -> List[str]:
        extractor = getattr(self, "keyword_extractor", None)
        if callable(extractor):
            try:
                keywords = normalize_keyword_values(extractor(text), limit=32)
                if keywords:
                    return keywords
            except Exception:
                pass
        return fallback_extract_keywords(text, limit=32)

    def _extract_entities(self, text: str) -> List[str]:
        extractor = getattr(self, "entity_extractor", None)
        if callable(extractor):
            try:
                entities = extractor(text)
                return filter_entity_values(entities, limit=24)
            except Exception:
                return []
        entities = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{1,}\b", text or "")
        return filter_entity_values(self._unique_strings(entities), limit=24)

    def _unique_strings(self, values: List[Any]) -> List[str]:
        result = []
        seen = set()
        for value in values or []:
            if not isinstance(value, str):
                continue
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _summarize_text(self, text: str, max_chars: int = 220) -> str:
        if not text:
            return ""
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _chunk_text(self, text: str, max_chars: int = 900, overlap: int = 160) -> List[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        if len(compact) <= max_chars:
            return [compact]

        chunks = []
        start = 0
        while start < len(compact):
            end = min(len(compact), start + max_chars)
            chunk = compact[start:end]
            if end < len(compact):
                boundary = max(chunk.rfind(". "), chunk.rfind(" "), int(max_chars * 0.65))
                if boundary > 0:
                    end = start + boundary
                    chunk = compact[start:end]
            chunks.append(chunk.strip())
            if end >= len(compact):
                break
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    def _build_node_embedding(self, text: str) -> List[float]:
        vector = get_embedding(
            text,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs,
        )
        if vector is None:
            return []
        return normalize_vector(np.array(vector, dtype=np.float32)).tolist()

    def _upsert_entry(self, entry: Dict[str, Any]):
        normalized = self._normalize_entry(entry)
        node_id = normalized["id"]
        self._node_map[node_id] = normalized

        replaced = False
        for idx, current in enumerate(self.data["knowledge_base"]):
            if current.get("id") == node_id:
                self.data["knowledge_base"][idx] = normalized
                replaced = True
                break
        if not replaced:
            self.data["knowledge_base"].append(normalized)

        if len(self.data["knowledge_base"]) > self.knowledge_capacity:
            overflow = len(self.data["knowledge_base"]) - self.knowledge_capacity
            self.data["knowledge_base"] = self.data["knowledge_base"][overflow:]

        self._rebuild_indexes()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        return list(self.data.get("knowledge_base", []))

    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self._node_map.get(entry_id)

    def node_exists(self, node_id: str) -> bool:
        return node_id in self._node_map

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        return [self._node_map[node_id] for node_id in self._doc_children.get(doc_id, []) if node_id in self._node_map]

    def get_primary_context_node(self, entry_id_or_node: Any) -> Optional[Dict[str, Any]]:
        entry = entry_id_or_node
        if isinstance(entry_id_or_node, str):
            entry = self.get_entry_by_id(entry_id_or_node)
        if not entry:
            return None

        if entry.get("pointer_to"):
            target = self.get_entry_by_id(entry["pointer_to"])
            if target:
                return target

        if entry.get("node_type") == "chunk" and entry.get("parent_id"):
            parent = self.get_entry_by_id(entry["parent_id"])
            if parent:
                return parent

        return entry

    def record_access(self, node_ids: List[str], temperature_boost: float = 8.0, persist: bool = True):
        changed = False
        now = get_timestamp()
        for node_id in node_ids:
            entry = self.get_entry_by_id(node_id)
            if not entry:
                continue
            entry["retrieved_count"] = int(entry.get("retrieved_count", 0)) + 1
            entry["last_access_time"] = now
            entry["T"] = min(100.0, float(entry.get("T", 0.0)) + temperature_boost)
            entry["storage_tier"] = "hot" if entry["T"] >= self.HOT_TEMPERATURE_THRESHOLD else "cold"
            changed = True
        if changed and persist:
            self.save()

    def search_knowledge(
        self,
        query: str,
        top_k: int = 8,
        threshold: float = 0.0,
        include_cold: bool = True,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        if query_embedding is None:
            query_embedding = np.array(
                get_embedding(query, self.embedding_model_name, **self.embedding_model_kwargs),
                dtype=np.float32,
            )
        else:
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.size == 0:
            query_embedding = np.zeros(0, dtype=np.float32)

        query_terms = set(self._extract_keywords(query))
        query_entities = {ent.lower() for ent in self._extract_entities(query)}
        query_lower = query.lower().strip()
        results = []
        searchable_types = {"document", "chunk", "summary"}

        for entry in self.data.get("knowledge_base", []):
            if entry.get("node_type") not in searchable_types:
                continue
            if not include_cold and entry.get("storage_tier") == "cold":
                continue

            item_embedding = np.array(entry.get("embedding", []), dtype=np.float32)
            semantic_score = 0.0
            if item_embedding.size and query_embedding.size and item_embedding.shape == query_embedding.shape:
                denom = float(np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding))
                if denom > 0:
                    semantic_score = float(np.dot(query_embedding, item_embedding) / denom)

            entry_terms = set(entry.get("keywords", []))
            keyword_score = 0.0
            if query_terms and entry_terms:
                keyword_score = len(query_terms & entry_terms) / float(len(query_terms))

            entry_entities = {ent.lower() for ent in entry.get("entities", []) if isinstance(ent, str)}
            entity_score = 0.0
            if query_entities and entry_entities:
                entity_score = len(query_entities & entry_entities) / float(len(query_entities))

            haystack = " ".join(
                [
                    str(entry.get("summary", "")),
                    str(entry.get("raw_content", "")),
                ]
            ).lower()
            exact_bonus = 0.04 if query_lower and query_lower in haystack else 0.0
            surface_score = 0.0
            if query_terms:
                haystack_terms = entry_terms or set(fallback_extract_keywords(haystack, limit=48))
                if haystack_terms:
                    surface_score = len(query_terms & haystack_terms) / float(len(query_terms))

            relevance = (
                (semantic_score * 0.56)
                + (keyword_score * 0.18)
                + (entity_score * 0.12)
                + (surface_score * 0.14)
                + exact_bonus
            )
            if relevance < threshold:
                continue

            candidate = dict(entry)
            candidate["semantic_score"] = round(semantic_score, 4)
            candidate["keyword_score"] = round(keyword_score, 4)
            candidate["entity_score"] = round(entity_score, 4)
            candidate["surface_score"] = round(surface_score, 4)
            candidate["exact_bonus"] = round(exact_bonus, 4)
            candidate["score"] = round(relevance, 4)
            results.append(candidate)

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]

    def search_temporal(
        self,
        query: str,
        top_k: int = 6,
        threshold: float = 0.0,
        include_cold: bool = True,
    ) -> List[Dict[str, Any]]:
        query_forms = set(extract_temporal_canonical_forms(query))
        query_entities = {ent.lower() for ent in self._extract_entities(query)}
        query_terms = set(self._extract_keywords(query))
        temporal_hint = bool(query_forms)
        if not temporal_hint and "when" not in str(query or "").lower() and "year" not in str(query or "").lower() and "how long" not in str(query or "").lower():
            return []

        results: List[Dict[str, Any]] = []
        searchable_types = {"document", "chunk", "summary"}
        for entry in self.data.get("knowledge_base", []):
            if entry.get("node_type") not in searchable_types:
                continue
            if not include_cold and entry.get("storage_tier") == "cold":
                continue

            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
            entry_forms = set(metadata.get("temporal_forms", []) or extract_temporal_canonical_forms(f"{entry.get('summary', '')}\n{entry.get('raw_content', '')}"))
            if not entry_forms:
                continue

            temporal_score = 0.0
            if query_forms:
                temporal_score = len(query_forms & entry_forms) / float(max(1, len(query_forms)))
            elif temporal_hint:
                temporal_score = 0.6

            entry_entities = {ent.lower() for ent in entry.get("entities", []) if isinstance(ent, str)}
            entity_score = 0.0
            if query_entities and entry_entities:
                entity_score = len(query_entities & entry_entities) / float(len(query_entities))

            entry_terms = set(entry.get("keywords", []))
            keyword_score = 0.0
            if query_terms and entry_terms:
                keyword_score = len(query_terms & entry_terms) / float(len(query_terms))

            if temporal_score <= 0.0 and entity_score <= 0.0:
                continue

            relevance = (0.55 * temporal_score) + (0.25 * entity_score) + (0.20 * keyword_score)
            if relevance < threshold:
                continue

            candidate = dict(entry)
            candidate["temporal_score"] = round(temporal_score, 4)
            candidate["entity_score"] = round(entity_score, 4)
            candidate["keyword_score"] = round(keyword_score, 4)
            candidate["score"] = round(relevance, 4)
            results.append(candidate)

        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]

    def get_related_by_entities(
        self,
        entities: List[str],
        max_results: int = 5,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        related_ids: List[str] = []

        for entity in entities or []:
            node_ids = self._entity_index.get(entity.lower(), set())
            for node_id in node_ids:
                if node_id not in exclude_ids and node_id not in related_ids:
                    related_ids.append(node_id)
                if len(related_ids) >= max_results:
                    break
            if len(related_ids) >= max_results:
                break

        if len(related_ids) < max_results:
            for entry in self.data.get("knowledge_base", []):
                if entry.get("id") in exclude_ids or entry.get("id") in related_ids:
                    continue
                if any(edge in exclude_ids for edge in entry.get("semantic_edges", [])):
                    related_ids.append(entry["id"])
                elif any(edge in (entities or []) for edge in entry.get("semantic_edges", [])):
                    related_ids.append(entry["id"])
                if len(related_ids) >= max_results:
                    break

        return [self._node_map[node_id] for node_id in related_ids if node_id in self._node_map][:max_results]

    def apply_soft_decay(self, half_life_days: int = 7, min_T: float = 2.0):
        now = datetime.utcnow()
        changed = False
        for entry in self.data.get("knowledge_base", []):
            ref_ts = entry.get("last_access_time") or entry.get("timestamp")
            if not ref_ts:
                continue
            try:
                ref_time = datetime.fromisoformat(ref_ts)
            except Exception:
                try:
                    ref_time = datetime.strptime(ref_ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

            days_passed = max(0.0, (now - ref_time).total_seconds() / 86400.0)
            decay_factor = 0.5 ** (days_passed / float(max(1, half_life_days)))
            new_temperature = max(min_T, float(entry.get("T", 0.0)) * decay_factor)
            if abs(new_temperature - float(entry.get("T", 0.0))) > 1e-6:
                entry["T"] = round(new_temperature, 4)
                entry["storage_tier"] = "hot" if new_temperature >= self.HOT_TEMPERATURE_THRESHOLD else "cold"
                changed = True
        if changed:
            self.save()

    def add_knowledge(self, sync_data: Dict[str, Any]):
        doc_id = sync_data.get("doc_id") or sync_data.get("id") or generate_id("doc")
        raw_content = sync_data.get("raw_content", "")
        summary = sync_data.get("summary") or sync_data.get("index_data") or self._summarize_text(raw_content)
        metadata = sync_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"legacy_metadata": metadata}

        nodes: List[Dict[str, Any]] = []
        chunks = self._chunk_text(raw_content) if raw_content else []
        stored_keywords = normalize_keyword_values(
            sync_data.get("keywords") or metadata.get("keywords") or [],
            limit=32,
        )
        document_keywords = stored_keywords or self._extract_keywords(f"{summary} {raw_content}")
        metadata["keywords"] = list(document_keywords)
        metadata["temporal_forms"] = list(
            metadata.get("temporal_forms") or extract_temporal_canonical_forms(f"{summary}\n{raw_content}")
        )

        stored_entities = metadata.get("entities")
        if not isinstance(stored_entities, list):
            stored_entities = []
        normalized_entities = filter_entity_values(stored_entities or self._extract_entities(raw_content))

        if len(chunks) <= 1:
            node_id = sync_data.get("id") or doc_id
            if self.node_exists(node_id):
                return
            node = {
                "id": node_id,
                "doc_id": doc_id,
                "chunk_id": sync_data.get("chunk_id", node_id),
                "node_type": "document",
                "raw_content": raw_content,
                "summary": summary,
                "embedding": self._build_node_embedding(f"{summary} {json.dumps(metadata, ensure_ascii=False)}"),
                "G": sync_data.get("G", 0.0),
                "C": sync_data.get("C", 0.6),
                "T": sync_data.get("T", 100.0),
                "salience_seed": sync_data.get("salience_seed", metadata.get("salience_seed", 0.5)),
                "entities": normalized_entities,
                "semantic_edges": filter_semantic_edge_values(metadata.get("semantic_edges", [])),
                "keywords": document_keywords,
                "parent_id": metadata.get("parent_id"),
                "child_ids": [],
                "pointer_to": None,
                "storage_tier": "hot",
                "timestamp": sync_data.get("timestamp") or get_timestamp(),
                "last_access_time": sync_data.get("timestamp") or get_timestamp(),
                "metadata": metadata,
                "source_kind": sync_data.get("source_kind", "conversation"),
                "retrieved_count": 0,
            }
            nodes.append(node)
        else:
            parent_id = sync_data.get("id") or doc_id
            if self.node_exists(parent_id):
                return
            child_ids = []
            entities = normalized_entities
            semantic_edges = filter_semantic_edge_values(metadata.get("semantic_edges", []))
            for idx, chunk in enumerate(chunks, start=1):
                child_id = f"{parent_id}_c{idx}"
                child_ids.append(child_id)
                nodes.append(
                    {
                        "id": child_id,
                        "doc_id": doc_id,
                        "chunk_id": child_id,
                        "node_type": "chunk",
                        "raw_content": chunk,
                        "summary": self._summarize_text(chunk, max_chars=160),
                        "embedding": self._build_node_embedding(chunk),
                        "G": sync_data.get("G", 0.0),
                        "C": sync_data.get("C", 0.6),
                        "T": sync_data.get("T", 100.0),
                        "salience_seed": sync_data.get("salience_seed", metadata.get("salience_seed", 0.5)),
                        "entities": entities,
                        "semantic_edges": semantic_edges,
                        "keywords": document_keywords,
                        "parent_id": parent_id,
                        "child_ids": [],
                        "pointer_to": None,
                        "storage_tier": "hot",
                        "timestamp": sync_data.get("timestamp") or get_timestamp(),
                        "last_access_time": sync_data.get("timestamp") or get_timestamp(),
                        "metadata": metadata,
                        "source_kind": sync_data.get("source_kind", "conversation"),
                        "retrieved_count": 0,
                    }
                )

            nodes.append(
                {
                    "id": parent_id,
                    "doc_id": doc_id,
                    "chunk_id": parent_id,
                    "node_type": "document",
                    "raw_content": raw_content,
                    "summary": summary,
                    "embedding": self._build_node_embedding(summary),
                    "G": sync_data.get("G", 0.0),
                    "C": sync_data.get("C", 0.6),
                    "T": sync_data.get("T", 100.0),
                    "salience_seed": sync_data.get("salience_seed", metadata.get("salience_seed", 0.5)),
                    "entities": entities,
                    "semantic_edges": semantic_edges,
                    "keywords": document_keywords,
                    "parent_id": None,
                    "child_ids": child_ids,
                    "pointer_to": None,
                    "storage_tier": "hot",
                    "timestamp": sync_data.get("timestamp") or get_timestamp(),
                    "last_access_time": sync_data.get("timestamp") or get_timestamp(),
                    "metadata": metadata,
                    "source_kind": sync_data.get("source_kind", "conversation"),
                    "retrieved_count": 0,
                }
            )

        for node in nodes:
            self._upsert_entry(node)
        self.save()

    def add_from_page(self, page: Dict[str, Any]) -> Optional[str]:
        """
        Compatibility helper for post-inference flow: promote a retrieved page
        into durable long-term storage.
        """
        if not isinstance(page, dict):
            return None

        node_id = page.get("citation_id") or page.get("page_id") or page.get("id") or generate_id("doc")
        if self.node_exists(node_id):
            return node_id

        metadata = page.get("metadata") or page.get("meta_data") or {}
        if not isinstance(metadata, dict):
            metadata = {"legacy_metadata": metadata}
        entities = filter_entity_values(metadata.get("entities", page.get("entities", [])))
        semantic_edges = filter_semantic_edge_values(metadata.get("semantic_edges", page.get("semantic_edges", [])))
        keywords = normalize_keyword_values(page.get("keywords") or metadata.get("keywords", []), limit=32)

        raw_content = page.get("raw_content")
        if not raw_content:
            raw_content = f"Q: {page.get('user_input', '')}\nA: {page.get('agent_response', '')}".strip()

        sync_data = {
            "id": node_id,
            "doc_id": page.get("doc_id") or node_id,
            "chunk_id": page.get("chunk_id", node_id),
            "timestamp": page.get("timestamp") or get_timestamp(),
            "metadata": {
                **metadata,
                "entities": entities,
                "semantic_edges": semantic_edges,
                "promoted_from": page.get("location") or page.get("_source") or "retrieval",
            },
            "summary": page.get("summary") or self._summarize_text(raw_content),
            "index_data": page.get("summary") or raw_content,
            "keywords": keywords,
            "raw_content": raw_content,
            "G": page.get("G", metadata.get("G_base", 0.0)),
            "C": page.get("C", metadata.get("C_base", 0.6)),
            "T": page.get("T", 42.0),
            "salience_seed": page.get("salience_seed", metadata.get("salience_seed", metadata.get("G_base", 0.5))),
            "source_kind": page.get("source_kind", "post_inference"),
        }
        self.add_knowledge(sync_data)
        return node_id

    def update_index_status(self, page_id: str, status: str):
        entry = self.get_entry_by_id(page_id)
        if not entry:
            return
        if status in {"hot", "cold", "pointer"}:
            entry["storage_tier"] = status
        else:
            entry["storage_tier"] = "hot" if status == "active" else "cold"
        self.save()

    def get_raw_user_profile(self, user_id: str = None) -> str:
        return self.data.get("user_profile", "None")

    def update_user_profile(self, user_id: str, new_profile_text: str, merge: bool = False):
        if not new_profile_text or new_profile_text == "None":
            return
        if merge:
            current = self.data.get("user_profile", "")
            self.data["user_profile"] = f"{current}\n{new_profile_text}".strip()
        else:
            self.data["user_profile"] = new_profile_text
        self.save()

    def adjust_chunk_metrics(self, chunk_id: str, delta_G: float = 0.0, delta_C: float = 0.0, delta_T: float = 0.0):
        entry = self.get_entry_by_id(chunk_id)
        if not entry:
            return False
        entry["G"] = max(0.0, float(entry.get("G", 0.0)) + float(delta_G))
        entry["C"] = min(1.0, max(0.0, float(entry.get("C", 0.0)) + float(delta_C)))
        entry["T"] = min(100.0, max(0.0, float(entry.get("T", 0.0)) + float(delta_T)))
        entry["storage_tier"] = "hot" if entry["T"] >= self.HOT_TEMPERATURE_THRESHOLD else "cold"
        self.save()
        return True

    def pointer_compress(
        self,
        chunk_id: str,
        summary_model: str = "qwen2.5:7b",
        client: Optional[OpenAIClient] = None,
    ) -> Optional[str]:
        entry = self.get_entry_by_id(chunk_id)
        if not entry:
            return None

        context_entry = self.get_primary_context_node(entry) or entry
        if not context_entry.get("raw_content"):
            return None

        pointer_id = f"{context_entry['id']}_ptr"
        if self.node_exists(pointer_id):
            return pointer_id

        llm_client = client or OpenAIClient(api_key=None, base_url=None)
        prompt = (
            "请将以下文本压缩成忠实摘要，突出关键事实、实体和结论。"
            "不要编造未出现的信息。\n\n"
            f"{context_entry.get('raw_content', '')}\n\n摘要："
        )
        try:
            summary = llm_client.chat_completion(
                model=summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            ).strip()
        except Exception:
            summary = ""

        if not summary:
            summary = self._summarize_text(context_entry.get("raw_content", ""), max_chars=220)

        pointer_entry = {
            "id": pointer_id,
            "doc_id": context_entry.get("doc_id", context_entry["id"]),
            "chunk_id": pointer_id,
            "node_type": "summary",
            "raw_content": summary,
            "summary": summary,
            "embedding": self._build_node_embedding(summary),
            "G": context_entry.get("G", 0.0),
            "C": context_entry.get("C", 0.6),
            "T": context_entry.get("T", 100.0),
            "salience_seed": context_entry.get("salience_seed", 0.5),
            "entities": filter_entity_values(context_entry.get("entities", [])),
            "semantic_edges": filter_semantic_edge_values(
                context_entry.get("semantic_edges", []) + [context_entry["id"]]
            ),
            "keywords": normalize_keyword_values(context_entry.get("keywords", []), limit=32) or self._extract_keywords(summary),
            "parent_id": context_entry.get("parent_id"),
            "child_ids": [],
            "pointer_to": context_entry["id"],
            "storage_tier": "pointer",
            "timestamp": get_timestamp(),
            "last_access_time": get_timestamp(),
            "metadata": {"pointer_to": context_entry["id"], "compression": "pointer_summary"},
            "source_kind": context_entry.get("source_kind", "conversation"),
            "retrieved_count": 0,
        }
        self._upsert_entry(pointer_entry)
        self.save()
        return pointer_id

    def clear(self):
        self.data["knowledge_base"] = []
        self._rebuild_indexes()
        self.save()
