import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compact_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _round_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except Exception:
        return None


class TraceRecorder:
    def __init__(self, output_path: str, run_metadata: Optional[Dict[str, Any]] = None):
        self.output_path = str(Path(output_path).resolve())
        self.run_metadata = dict(run_metadata or {})
        self.run_metadata.setdefault("started_at", datetime.utcnow().isoformat())
        self._lock = threading.Lock()
        self._seq = 0
        self.ingest_events: List[Dict[str, Any]] = []
        self.retrieval_events: List[Dict[str, Any]] = []
        self.question_events: List[Dict[str, Any]] = []
        self.background_events: List[Dict[str, Any]] = []
        self.notes: List[Dict[str, Any]] = []

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _stamp(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "seq": self._next_seq(),
            "event_type": event_type,
            "event_time": datetime.utcnow().isoformat(),
            **payload,
        }

    def update_run_metadata(self, **kwargs: Any) -> None:
        with self._lock:
            self.run_metadata.update(kwargs)

    def snapshot_record(self, record: Optional[Dict[str, Any]], default_location: str = "unknown") -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {"memory_id": None, "location": "missing", "exists": False}

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = record.get("meta_data", {}) if isinstance(record.get("meta_data", {}), dict) else {}

        memory_id = (
            record.get("page_id")
            or record.get("citation_id")
            or record.get("id")
            or record.get("chunk_id")
            or record.get("doc_id")
        )
        return {
            "memory_id": memory_id,
            "page_id": record.get("page_id") or record.get("id") or record.get("citation_id"),
            "citation_id": record.get("citation_id") or record.get("page_id") or record.get("id"),
            "doc_id": record.get("doc_id"),
            "location": record.get("location") or default_location,
            "source": record.get("_source") or record.get("source_kind"),
            "storage_tier": record.get("storage_tier"),
            "node_type": record.get("node_type"),
            "chunk_scope": metadata.get("chunk_scope"),
            "conversation_id": metadata.get("conversation_id"),
            "session_id": metadata.get("session_id"),
            "window_index": metadata.get("window_index"),
            "window_dia_ids": metadata.get("window_dia_ids", []),
            "topic": metadata.get("topic"),
            "G": _round_float(record.get("G")),
            "C": _round_float(record.get("C")),
            "T": _round_float(record.get("T")),
            "salience_seed": _round_float(record.get("salience_seed")),
            "score": _round_float(record.get("score")),
            "final_score": _round_float(record.get("final_score")),
            "retrieved_count": record.get("retrieved_count"),
            "summary_preview": _compact_text(
                record.get("summary")
                or record.get("raw_content")
                or record.get("agent_response")
                or record.get("user_input")
            ),
            "user_input_preview": _compact_text(record.get("user_input")),
            "agent_response_preview": _compact_text(record.get("agent_response")),
            "exists": True,
        }

    def record_note(self, message: str, trace_context: Optional[Dict[str, Any]] = None) -> None:
        event = self._stamp(
            "note",
            {
                "message": str(message),
                "trace_context": dict(trace_context or {}),
            },
        )
        with self._lock:
            self.notes.append(event)

    def record_ingest_event(self, payload: Dict[str, Any]) -> None:
        event = self._stamp("ingest", payload)
        with self._lock:
            self.ingest_events.append(event)

    def record_retrieval_event(self, payload: Dict[str, Any]) -> None:
        event = self._stamp("retrieval", payload)
        with self._lock:
            self.retrieval_events.append(event)

    def record_question_event(self, payload: Dict[str, Any]) -> None:
        event = self._stamp("question", payload)
        with self._lock:
            self.question_events.append(event)

    def record_background_event(self, payload: Dict[str, Any]) -> None:
        event = self._stamp("background", payload)
        with self._lock:
            self.background_events.append(event)

    def _memory_timeline_entry(self, event: Dict[str, Any], memory_id: str) -> Dict[str, Any]:
        event_type = event.get("event_type")
        entry = {
            "seq": event.get("seq"),
            "event_time": event.get("event_time"),
            "event_type": event_type,
        }
        if event_type == "ingest":
            entry.update(
                {
                    "action": event.get("action"),
                    "durable_target": event.get("durable_target"),
                    "destinations": event.get("destinations", []),
                    "reason": event.get("reason"),
                }
            )
        elif event_type == "retrieval":
            entry.update(
                {
                    "query": event.get("query"),
                    "query_kind": (event.get("trace_context") or {}).get("query_kind"),
                    "rank": next(
                        (
                            hit.get("rank")
                            for hit in event.get("hits", [])
                            if hit.get("memory_id") == memory_id
                        ),
                        None,
                    ),
                    "final_score": next(
                        (
                            hit.get("final_score")
                            for hit in event.get("hits", [])
                            if hit.get("memory_id") == memory_id
                        ),
                        None,
                    ),
                }
            )
        elif event_type == "question":
            entry.update(
                {
                    "question_index": event.get("question_index"),
                    "question": event.get("question"),
                    "prediction": event.get("prediction"),
                }
            )
        elif event_type == "background":
            entry.update(
                {
                    "task_name": event.get("task_name"),
                    "trace_context": event.get("trace_context", {}),
                    "changes": [
                        change
                        for change in event.get("changes", [])
                        if change.get("memory_id") == memory_id or change.get("citation_id") == memory_id
                    ],
                }
            )
        return entry

    def _build_memory_timelines(self) -> Dict[str, List[Dict[str, Any]]]:
        timelines: Dict[str, List[Dict[str, Any]]] = {}
        for collection in (
            self.ingest_events,
            self.retrieval_events,
            self.question_events,
            self.background_events,
        ):
            for event in collection:
                memory_ids = list(dict.fromkeys(event.get("memory_ids", [])))
                for memory_id in memory_ids:
                    if not memory_id:
                        continue
                    timelines.setdefault(memory_id, []).append(self._memory_timeline_entry(event, memory_id))
        return timelines

    def finalize(self, extra: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            payload = {
                "run_metadata": {
                    **self.run_metadata,
                    "finished_at": datetime.utcnow().isoformat(),
                    "trace_path": self.output_path,
                },
                "summary": {
                    "ingest_events": len(self.ingest_events),
                    "retrieval_events": len(self.retrieval_events),
                    "question_events": len(self.question_events),
                    "background_events": len(self.background_events),
                    "notes": len(self.notes),
                },
                "ingest_events": list(self.ingest_events),
                "retrieval_events": list(self.retrieval_events),
                "question_events": list(self.question_events),
                "background_events": list(self.background_events),
                "notes": list(self.notes),
                "memory_timelines": self._build_memory_timelines(),
            }
            if extra:
                payload["extra"] = extra

        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return self.output_path
