import json
import logging
from typing import Any, Dict, Optional

try:
    from ..forget_memory import ForgetMemory
    from ..long_term import LongTermMemory
    from ..short_term import ShortTermMemory
    from ..utils import generate_id, get_timestamp, normalize_keyword_values
except ImportError:
    from forget_memory import ForgetMemory
    from long_term import LongTermMemory
    from short_term import ShortTermMemory
    from utils import generate_id, get_timestamp, normalize_keyword_values


def mem_log(stage: str, flow: str, meta: dict, content: str):
    short_content = " ".join((content or "").split())
    if len(short_content) > 80:
        short_content = short_content[:77] + "..."
    logging.info("[%s] %s | %s | %s", stage, flow, json.dumps(meta, ensure_ascii=False), short_content)


class Updater:
    """
    Hot-path writer:
    - Always accepts new memory.
    - Writes to STM for recency and to LTM for durable graph-vector retrieval.
    - Trims STM as a recency buffer instead of using a synchronous discard funnel.
    """

    DEFAULT_C0 = 0.6
    DEFAULT_T0 = 100.0

    def __init__(
        self,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        forget_memory: ForgetMemory,
        client,
        llm_model: str = "qwen2.5:7b",
        stm_capacity: int = 15,
        forget_threshold: float = 0.35,
        long_term_threshold: float = 0.65,
    ):
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.forget_memory = forget_memory
        self.client = client
        self.llm_model = llm_model
        self.stm_capacity = stm_capacity
        self.forget_threshold = forget_threshold
        self.long_term_threshold = long_term_threshold
        self.memoryos = None
        self.tracer = None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _archive_overflow_before_write(self) -> None:
        """
        Preserve evicted STM records by archiving them before the next write.
        """
        while len(self.short_term_memory.get_all()) >= self.stm_capacity:
            oldest = self.short_term_memory.get_all()[0]
            page_id = oldest.get("page_id")
            try:
                self.forget_memory.add_page(oldest)
            except Exception as exc:
                logging.warning("ForgetMemory archive failed for %s: %s", page_id, exc)
            finally:
                if page_id:
                    self.short_term_memory.remove_page(page_id)
                elif self.short_term_memory.get_all():
                    self.short_term_memory.memory = self.short_term_memory.get_all()[1:]
                    self.short_term_memory.save()

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            metadata = {}

        entities = metadata.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        semantic_edges = metadata.get("semantic_edges", [])
        if not isinstance(semantic_edges, list):
            semantic_edges = []
        pre_linked_ids = metadata.get("pre_linked_ids", [])
        if isinstance(pre_linked_ids, list):
            semantic_edges = list(dict.fromkeys(semantic_edges + [pid for pid in pre_linked_ids if isinstance(pid, str)]))

        keywords = normalize_keyword_values(metadata.get("keywords", []), limit=32)
        metadata["entities"] = [ent for ent in entities if isinstance(ent, str)]
        metadata["semantic_edges"] = semantic_edges
        metadata["keywords"] = keywords
        return metadata

    def _build_summary(self, user_input: str, agent_response: str, metadata: Dict[str, Any]) -> str:
        topic = metadata.get("topic", "General")
        entities = ", ".join(metadata.get("entities", [])[:4])
        pieces = [f"Topic: {topic}"]
        if entities:
            pieces.append(f"Entities: {entities}")
        if user_input:
            pieces.append(f"User asked: {' '.join(user_input.split())[:120]}")
        if agent_response:
            pieces.append(f"Assistant replied: {' '.join(agent_response.split())[:120]}")
        return " | ".join(pieces)

    def _decide_initial_route(self, salience_seed: float, metadata: Dict[str, Any], raw_content: str) -> Dict[str, Any]:
        flags = metadata.get("flags", {}) if isinstance(metadata.get("flags"), dict) else {}
        entity_bonus = min(0.12, 0.03 * len(metadata.get("entities", [])))
        content_bonus = 0.08 if len(raw_content) >= 260 else 0.0
        important_bonus = 0.10 if flags.get("important") or flags.get("preference") else 0.0
        route_score = self._clamp01(float(salience_seed) + entity_bonus + content_bonus + important_bonus)

        if route_score >= self.long_term_threshold:
            durable_target = "long_term"
            destinations = ["short_term", "long_term"]
        else:
            durable_target = "short_term"
            destinations = ["short_term"]

        return {
            "route_score": round(route_score, 4),
            "salience_seed": round(float(salience_seed), 4),
            "durable_target": durable_target,
            "destinations": destinations,
        }

    def _trim_short_term_memory(self):
        all_items = self.short_term_memory.get_all()
        if len(all_items) <= self.stm_capacity:
            return

        overflow = len(all_items) - self.stm_capacity
        evicted_items = all_items[:overflow]
        retained_items = all_items[overflow:]
        self.short_term_memory.memory = retained_items
        self.short_term_memory.save()

        for item in evicted_items:
            try:
                self.forget_memory.add_page(item)
            except Exception as exc:
                logging.warning("ForgetMemory archive failed for %s: %s", item.get("page_id"), exc)

    def process_short_term_to_routing(self, metadata: Dict = None):
        """
        Backward-compatible entrypoint.
        The new architecture treats STM as a recency cache only.
        """
        self._trim_short_term_memory()

    def process_short_term_to_mid_term(self, metadata: Dict = None):
        """
        Legacy alias kept for existing tests and scripts.
        """
        self.process_short_term_to_routing(metadata=metadata)

    def ingest_new_memory(
        self,
        user_input: str,
        agent_response: str,
        g0_score: float,
        metadata: Dict = None,
        timestamp: str = None,
    ):
        timestamp = timestamp or get_timestamp()
        metadata = self._normalize_metadata(metadata)
        doc_id = metadata.get("doc_id") or generate_id("doc")
        page_id = metadata.get("page_id") or doc_id

        try:
            salience_seed = float(g0_score) if g0_score is not None else 0.5
        except Exception:
            salience_seed = 0.5

        raw_content = f"Q: {user_input}\nA: {agent_response}".strip()
        route_info = self._decide_initial_route(salience_seed, metadata, raw_content)
        metadata["initial_route"] = route_info["durable_target"]
        metadata["initial_destinations"] = list(route_info["destinations"])
        metadata["initial_route_score"] = route_info["route_score"]

        qa_unit = {
            "page_id": page_id,
            "doc_id": doc_id,
            "chunk_id": page_id,
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "keywords": list(metadata.get("keywords", [])),
            "metadata": metadata,
            "G": 0.0,
            "C": self.DEFAULT_C0,
            "T": self.DEFAULT_T0,
            "gcl_scores": {"G": 0.0, "C": self.DEFAULT_C0, "T": self.DEFAULT_T0, "G0": salience_seed},
        }

        self._archive_overflow_before_write()
        self.short_term_memory.add_qa_pair(qa_unit)
        self._trim_short_term_memory()

        summary = self._build_summary(user_input, agent_response, metadata)
        if "long_term" in route_info["destinations"]:
            sync_data = {
                "id": doc_id,
                "doc_id": doc_id,
                "chunk_id": page_id,
                "timestamp": timestamp,
                "metadata": {
                    **metadata,
                    "doc_id": doc_id,
                    "source": "hot_ingest",
                    "salience_seed": salience_seed,
                },
                "summary": summary,
                "index_data": summary,
                "keywords": list(metadata.get("keywords", [])),
                "raw_content": raw_content,
                "G": 0.0,
                "C": self.DEFAULT_C0,
                "T": self.DEFAULT_T0,
                "salience_seed": salience_seed,
                "source_kind": "conversation",
            }
            self.long_term_memory.add_knowledge(sync_data)
        mem_log(
            "INGEST",
            "HOT_PATH_WRITE",
            {
                "doc_id": doc_id,
                "C0": self.DEFAULT_C0,
                "T0": self.DEFAULT_T0,
                "route_score": route_info["route_score"],
                "durable_target": route_info["durable_target"],
            },
            user_input,
        )
        if self.tracer is not None:
            destinations = []
            memory_ids = []

            short_term_entry = self.short_term_memory.get_page_by_id(page_id)
            if short_term_entry:
                snapshot = self.tracer.snapshot_record(short_term_entry, default_location="short_term")
                snapshot["destination"] = "short_term"
                destinations.append(snapshot)
                if snapshot.get("memory_id"):
                    memory_ids.append(snapshot["memory_id"])

            long_term_entry = self.long_term_memory.get_entry_by_id(doc_id) or self.long_term_memory.get_entry_by_id(page_id)
            if long_term_entry:
                snapshot = self.tracer.snapshot_record(long_term_entry, default_location="long_term")
                snapshot["destination"] = "long_term"
                destinations.append(snapshot)
                if snapshot.get("memory_id"):
                    memory_ids.append(snapshot["memory_id"])

            self.tracer.record_ingest_event(
                {
                    "action": "ingest_new_memory",
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "memory_ids": list(dict.fromkeys(memory_ids or [page_id])),
                    "durable_target": route_info["durable_target"],
                    "destinations": list(route_info["destinations"]),
                    "route_score": route_info["route_score"],
                    "reason": {
                        "salience_seed": round(float(salience_seed), 4),
                        "entity_count": len(metadata.get("entities", [])),
                        "topic": metadata.get("topic"),
                        "flags": metadata.get("flags", {}),
                    },
                    "records": destinations,
                }
            )
        return route_info
