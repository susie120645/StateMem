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
        create_backup_snapshot,
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        extract_temporal_canonical_forms,
        normalize_keyword_values,
        normalize_vector,
        write_lines_with_backup,
    )
except ImportError:
    from entity_filters import filter_entity_values, filter_semantic_edge_values
    from utils import (
        create_backup_snapshot,
        fallback_extract_keywords,
        generate_id,
        get_embedding,
        get_timestamp,
        extract_temporal_canonical_forms,
        normalize_keyword_values,
        normalize_vector,
        write_lines_with_backup,
    )


class ForgetMemory:
    """Append-only cold storage with lightweight hybrid retrieval."""

    DEFAULT_COLD_CONFIDENCE = 0.5
    DEFAULT_COLD_TEMPERATURE = 18.0
    RELATION_PATTERNS = {
        "dating": [
            r"\bdate\b",
            r"\bdating\b",
            r"\bgoing out\b",
            r"\bwent out\b",
            r"\bgirlfriend\b",
            r"\bboyfriend\b",
            r"\bpartner\b",
            r"\bfianc[ée]\b",
        ],
        "roommate": [
            r"\broommate\b",
            r"\bbunking with\b",
            r"\bliving with\b",
            r"\blive with\b",
        ],
        "neighbor": [
            r"\bnext door\b",
            r"\bneighbor\b",
            r"\bliving next door\b",
        ],
        "family": [
            r"\bfamily\b",
            r"\bmother\b",
            r"\bfather\b",
            r"\bsister\b",
            r"\bbrother\b",
            r"\bson\b",
            r"\bdaughter\b",
            r"\bkid\b",
            r"\bchild\b",
            r"\bparent\b",
        ],
        "work": [
            r"\bwork(?:s|ed|ing)? at\b",
            r"\bjob\b",
            r"\bcompany\b",
            r"\boffice\b",
        ],
        "pet": [
            r"\bpet\b",
            r"\bdog\b",
            r"\bcat\b",
        ],
    }

    def __init__(
        self,
        file_path: str,
        embedding_model_name: str = "nomic-embed-text",
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
        enable_backups: bool = True,
    ):
        self.file_path = file_path
        self.archive_index_path = self._default_archive_index_path(file_path)
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.enable_backups = bool(enable_backups)
        self._id_offset_index: Dict[str, int] = {}
        self._record_cache: Dict[str, Dict[str, Any]] = {}
        self._archive_index_cache: Dict[str, Dict[str, Any]] = {}
        self._ensure_file_exists()
        self._ensure_archive_index_exists()
        self._build_indexes()
        self._build_archive_index()
        print(
            (
                "ForgetMemory: Initialized with "
                f"{len(self._record_cache)} cold records and "
                f"{len(self._archive_index_cache)} archive index entries."
            ),
            file=sys.stderr,
        )

    def _default_archive_index_path(self, file_path: str) -> str:
        if file_path.endswith(".jsonl"):
            return file_path[:-6] + "_archive_index.jsonl"
        return file_path + ".archive_index.jsonl"

    def _ensure_file_exists(self) -> None:
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8"):
                pass

    def _ensure_archive_index_exists(self) -> None:
        directory = os.path.dirname(self.archive_index_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(self.archive_index_path):
            with open(self.archive_index_path, "w", encoding="utf-8"):
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

    def _load_archive_index_cache(self) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(self.archive_index_path):
            return cache

        try:
            with open(self.archive_index_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    page_id = entry.get("page_id")
                    if page_id:
                        cache[page_id] = entry
        except Exception as exc:
            logging.error("ForgetMemory archive index load error: %s", exc)
        return cache

    def _build_archive_index(self) -> None:
        cached = self._load_archive_index_cache()
        if cached.keys() == self._record_cache.keys():
            self._archive_index_cache = cached
            return

        rebuilt = {
            page_id: self._build_archive_index_entry(record)
            for page_id, record in self._record_cache.items()
        }
        self._archive_index_cache = rebuilt
        lines = [json.dumps(entry, ensure_ascii=False) for entry in rebuilt.values()]
        write_lines_with_backup(
            self.archive_index_path,
            lines,
            create_backup=self.enable_backups,
        )

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

    def _extract_relation_terms(self, text: str) -> List[str]:
        compact = str(text or "").lower()
        if not compact:
            return []

        relations: List[str] = []
        for relation, patterns in self.RELATION_PATTERNS.items():
            if any(re.search(pattern, compact) for pattern in patterns):
                relations.append(relation)
        return relations

    def _build_archive_index_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        text = "\n".join(
            part
            for part in [
                record.get("summary", ""),
                record.get("raw_content", ""),
                record.get("user_input", ""),
                record.get("agent_response", ""),
            ]
            if part
        )
        temporal_forms = metadata.get("temporal_forms")
        if not isinstance(temporal_forms, list):
            temporal_forms = extract_temporal_canonical_forms(text)
        relation_terms = metadata.get("relation_terms")
        if not isinstance(relation_terms, list):
            relation_terms = self._extract_relation_terms(text)
        source_family = (
            metadata.get("source_family")
            or metadata.get("session_group_id")
            or metadata.get("conversation_id")
            or metadata.get("session_id")
            or record.get("doc_id")
            or record.get("page_id")
        )
        return {
            "record_version": 1,
            "page_id": record.get("page_id"),
            "doc_id": record.get("doc_id") or record.get("page_id"),
            "archived_at": record.get("archived_at") or get_timestamp(),
            "entities": filter_entity_values(record.get("entities", []), limit=24),
            "keywords": normalize_keyword_values(record.get("keywords", []), limit=48),
            "temporal_forms": [value for value in temporal_forms if isinstance(value, str)],
            "relation_terms": [value for value in relation_terms if isinstance(value, str)],
            "source_family": str(source_family or ""),
            "source_session": str(metadata.get("session_id") or ""),
        }

    def _persist_archive_index_entry(self, entry: Dict[str, Any]) -> None:
        with open(self.archive_index_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
            metadata["temporal_forms"] = list(
                metadata.get("temporal_forms") or extract_temporal_canonical_forms(f"{summary}\n{raw_content}")
            )
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
        raw_content = f"Q: {user_input}\nA: {agent_response}".strip()
        metadata["temporal_forms"] = list(
            metadata.get("temporal_forms") or extract_temporal_canonical_forms(f"{summary}\n{raw_content}")
        )
        embedding_text = self._summarize_text(f"{summary}\n{combined}", max_chars=1200)
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
            create_backup_snapshot(self.file_path, enabled=self.enable_backups)
            offset = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
            with open(self.file_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            page_id = record.get("page_id")
            if page_id:
                self._id_offset_index[page_id] = offset
                self._record_cache[page_id] = record
                archive_entry = self._build_archive_index_entry(record)
                self._archive_index_cache[page_id] = archive_entry
                self._persist_archive_index_entry(archive_entry)
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
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        exclude_ids = exclude_ids or set()
        if query_embedding is None:
            query_embedding = np.array(
                get_embedding(query, model_name=self.embedding_model_name, **self.embedding_model_kwargs),
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

        for page_id, record in self._record_cache.items():
            if page_id in exclude_ids:
                continue

            item_embedding = np.array(record.get("embedding", []), dtype=np.float32)
            semantic_score = 0.0
            if item_embedding.size and query_embedding.size and item_embedding.shape == query_embedding.shape:
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

    def search_archive(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        exclude_ids = exclude_ids or set()
        query_forms = set(extract_temporal_canonical_forms(query))
        query_terms = set(self._extract_keywords(query))
        query_entities = {ent.lower() for ent in self._extract_entities(query)}
        query_relations = set(self._extract_relation_terms(query))

        results: List[Dict[str, Any]] = []
        for page_id, index_entry in self._archive_index_cache.items():
            if page_id in exclude_ids:
                continue
            record = self._record_cache.get(page_id)
            if record is None:
                continue

            record_terms = set(index_entry.get("keywords", []))
            record_entities = {
                ent.lower()
                for ent in index_entry.get("entities", [])
                if isinstance(ent, str)
            }
            record_forms = {
                form
                for form in index_entry.get("temporal_forms", [])
                if isinstance(form, str)
            }
            record_relations = {
                relation
                for relation in index_entry.get("relation_terms", [])
                if isinstance(relation, str)
            }

            entity_score = 0.0
            if query_entities and record_entities:
                entity_score = len(query_entities & record_entities) / float(len(query_entities))

            temporal_score = 0.0
            if query_forms and record_forms:
                temporal_score = len(query_forms & record_forms) / float(max(1, len(query_forms)))

            relation_score = 0.0
            if query_relations and record_relations:
                relation_score = len(query_relations & record_relations) / float(max(1, len(query_relations)))

            keyword_score = 0.0
            if query_terms and record_terms:
                keyword_score = len(query_terms & record_terms) / float(len(query_terms))

            if max(entity_score, temporal_score, relation_score, keyword_score) <= 0.0:
                continue

            relevance = (
                (0.34 * entity_score)
                + (0.26 * temporal_score)
                + (0.24 * relation_score)
                + (0.16 * keyword_score)
            )
            if query_forms and temporal_score <= 0.0:
                relevance -= 0.08
            if query_relations and relation_score <= 0.0:
                relevance -= 0.08
            if relevance < threshold:
                continue

            candidate = dict(record)
            candidate["archive_score"] = round(relevance, 4)
            candidate["score"] = round(max(candidate.get("score", 0.0), relevance), 4)
            candidate["archive_match"] = {
                "entity_overlap": round(entity_score, 4),
                "temporal_overlap": round(temporal_score, 4),
                "relation_overlap": round(relation_score, 4),
                "keyword_overlap": round(keyword_score, 4),
                "relation_terms": sorted(query_relations & record_relations),
                "temporal_forms": sorted(query_forms & record_forms),
                "source_family": index_entry.get("source_family", ""),
                "source_session": index_entry.get("source_session", ""),
            }
            results.append(candidate)

        results.sort(
            key=lambda item: (
                item.get("archive_score", 0.0),
                item.get("score", 0.0),
                item.get("G", 0.0),
            ),
            reverse=True,
        )
        return results[:top_k]

    def search_temporal(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        query_forms = set(extract_temporal_canonical_forms(query))
        query_entities = {ent.lower() for ent in self._extract_entities(query)}
        query_terms = set(self._extract_keywords(query))
        temporal_hint = bool(query_forms)
        if not temporal_hint and "when" not in str(query or "").lower() and "year" not in str(query or "").lower() and "how long" not in str(query or "").lower():
            return []

        results: List[Dict[str, Any]] = []
        for page_id, record in self._record_cache.items():
            if page_id in exclude_ids:
                continue

            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
            record_forms = set(metadata.get("temporal_forms", []) or extract_temporal_canonical_forms(f"{record.get('summary', '')}\n{record.get('raw_content', '')}"))
            if not record_forms:
                continue

            temporal_score = 0.0
            if query_forms:
                temporal_score = len(query_forms & record_forms) / float(max(1, len(query_forms)))
            elif temporal_hint:
                temporal_score = 0.6

            record_entities = {ent.lower() for ent in record.get("entities", []) if isinstance(ent, str)}
            entity_score = 0.0
            if query_entities and record_entities:
                entity_score = len(query_entities & record_entities) / float(len(query_entities))

            record_terms = set(record.get("keywords", []))
            keyword_score = 0.0
            if query_terms and record_terms:
                keyword_score = len(query_terms & record_terms) / float(len(query_terms))

            if temporal_score <= 0.0 and entity_score <= 0.0:
                continue

            relevance = (0.55 * temporal_score) + (0.25 * entity_score) + (0.20 * keyword_score)
            if relevance < threshold:
                continue

            candidate = dict(record)
            candidate["temporal_score"] = round(temporal_score, 4)
            candidate["entity_score"] = round(entity_score, 4)
            candidate["keyword_score"] = round(keyword_score, 4)
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
            "archive_index_entries": len(self._archive_index_cache),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "embedded_records": with_embeddings,
            "storage_tier": "cold",
        }

    def clear(self) -> None:
        self._id_offset_index = {}
        self._record_cache = {}
        self._archive_index_cache = {}
        write_lines_with_backup(self.file_path, [], create_backup=self.enable_backups)
        write_lines_with_backup(self.archive_index_path, [], create_backup=self.enable_backups)
