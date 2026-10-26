import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from ..schemas import MemoryStats, MetricDistribution
except ImportError:
    from schemas import MemoryStats, MetricDistribution


class LifecycleService:
    def __init__(self, short_term_memory, long_term_memory, forget_memory, client, llm_model: str):
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.forget_memory = forget_memory
        self.client = client
        self.llm_model = llm_model
        self.tracer = None
        self._task_queue: "queue.Queue[Tuple[str, Callable, tuple, dict]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run_background_tasks,
            name="statemem-cold-path",
            daemon=True,
        )
        self._worker.start()

    def _run_background_tasks(self) -> None:
        while not self._stop_event.is_set():
            try:
                task_name, fn, args, kwargs = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logging.warning("Async cold-path task %s failed: %s", task_name, exc)
            finally:
                self._task_queue.task_done()

    def _enqueue(self, task_name: str, fn: Callable, *args, **kwargs) -> None:
        self._task_queue.put((task_name, fn, args, kwargs))

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _normalize_answer(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _extract_evidence_text(record: Dict[str, Any]) -> str:
        if not isinstance(record, dict):
            return ""

        raw_content = str(record.get("raw_content") or record.get("content") or record.get("summary") or "").strip()
        if raw_content:
            return raw_content

        user_input = str(record.get("user_input") or "").strip()
        agent_response = str(record.get("agent_response") or "").strip()
        if user_input or agent_response:
            return f"Q: {user_input}\nA: {agent_response}".strip()
        return ""

    def _lookup_evidence_record(self, citation_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        if not citation_id:
            return None, "missing"

        entry = self.long_term_memory.get_entry_by_id(citation_id)
        if entry:
            return entry, "long_term"

        page = self.short_term_memory.get_page_by_id(citation_id)
        if page:
            return page, "short_term"

        if hasattr(self.forget_memory, "get_page_by_id"):
            page = self.forget_memory.get_page_by_id(citation_id)
            if page:
                return page, "forget_pool"
        return None, "missing"

    def _snapshot_memory_state(self, memory_id: str, statemem=None) -> Dict[str, Any]:
        if self.tracer is None:
            return {}
        if not memory_id:
            return {"memory_id": None, "location": "missing", "exists": False}

        record = None
        location = "missing"
        if statemem is not None and hasattr(statemem, "get_memory_by_id"):
            try:
                record = statemem.get_memory_by_id(memory_id)
                if isinstance(record, dict):
                    location = record.get("location", "unknown")
            except Exception:
                record = None
        if not record:
            record, location = self._lookup_evidence_record(memory_id)
        return self.tracer.snapshot_record(record, default_location=location)

    @staticmethod
    def _dedupe_ids(values: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for value in values or []:
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _resolve_verdict(results: Dict[str, Any], citation_id: str) -> str:
        for verdict in ("entailment", "contradiction", "neutral", "skipped"):
            if citation_id in (results.get(verdict) or []):
                return verdict
        return "unknown"

    def _record_background_trace(
        self,
        task_name: str,
        trace_context: Optional[Dict[str, Any]],
        memory_ids: List[str],
        changes: List[Dict[str, Any]],
        **extra: Any,
    ) -> None:
        if self.tracer is None:
            return
        payload = {
            "task_name": task_name,
            "trace_context": dict(trace_context or {}),
            "memory_ids": self._dedupe_ids(memory_ids),
            "changes": changes,
        }
        payload.update(extra)
        self.tracer.record_background_event(payload)

    def _judge_nli_verdict(self, premise: str, hypothesis: str) -> str:
        if not premise or not hypothesis or self.client is None:
            return "skipped"

        nli_prompt = (
            "You are doing strict NLI verification.\n"
            "Return exactly one label: entailment, contradiction, or neutral.\n\n"
            f"Premise:\n{premise}\n\n"
            f"Hypothesis:\n{hypothesis}\n"
        )
        try:
            response = self.client.chat_completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": nli_prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            response_lower = (response or "").lower()
        except Exception:
            return "skipped"

        if "contradict" in response_lower:
            return "contradiction"
        if "entail" in response_lower:
            return "entailment"
        if "neutral" in response_lower:
            return "neutral"
        return "neutral"

    def preview_nli_citations(
        self,
        generated_answer: str,
        citation_ids: List[str],
        max_citations: int = 3,
    ) -> Dict[str, Any]:
        preview = {
            "checked_ids": [],
            "entailment": [],
            "contradiction": [],
            "neutral": [],
            "skipped": [],
            "alignment_score": 0.5,
            "contradiction_ratio": 0.0,
            "coverage_ratio": 0.0,
        }
        if not generated_answer or not citation_ids:
            return preview

        checked = 0
        for citation_id in citation_ids:
            if not citation_id or checked >= max_citations:
                continue
            checked += 1
            preview["checked_ids"].append(citation_id)
            record, _ = self._lookup_evidence_record(citation_id)
            premise = self._extract_evidence_text(record or {})
            if not premise:
                preview["skipped"].append(citation_id)
                continue

            verdict = self._judge_nli_verdict(premise, generated_answer)
            preview.setdefault(verdict, [])
            preview[verdict].append(citation_id)

        evaluated = (
            len(preview["entailment"])
            + len(preview["contradiction"])
            + len(preview["neutral"])
        )
        total_checked = len(preview["checked_ids"])
        if evaluated == 0:
            return preview

        entailment_ratio = len(preview["entailment"]) / float(evaluated)
        contradiction_ratio = len(preview["contradiction"]) / float(evaluated)
        neutral_ratio = len(preview["neutral"]) / float(evaluated)
        alignment_score = self._clamp((1.0 * entailment_ratio) + (0.45 * neutral_ratio) - (0.8 * contradiction_ratio))
        preview["alignment_score"] = round(alignment_score, 4)
        preview["contradiction_ratio"] = round(contradiction_ratio, 4)
        preview["coverage_ratio"] = round(evaluated / float(max(1, total_checked)), 4)
        return preview

    def build_interaction_reflection(
        self,
        query: str,
        final_answer: str,
        retrieval_results: Dict[str, Any],
        base_feedback_score: float = 0.0,
    ) -> Dict[str, Any]:
        pages = retrieval_results.get("retrieved_pages", []) or []
        citations = retrieval_results.get("citations", []) or []
        hit_ids = retrieval_results.get("hit_ids", []) or []
        source_stats = retrieval_results.get("source_stats", {}) or {}
        salvage_count = int(retrieval_results.get("salvage_count", 0) or 0)
        normalized_answer = self._normalize_answer(final_answer)
        unknown_answer = normalized_answer in {"i don't know.", "i don't know", "unknown", "unknown."}

        page_count = len(pages)
        hit_count = len(hit_ids)
        citation_count = len(citations)
        hit_strength = min(1.0, hit_count / 3.0)
        citation_ratio = min(1.0, citation_count / float(max(1, min(3, hit_count or page_count or 1))))
        source_diversity = sum(
            1 for key in ("STM", "LTM", "GRAPH_1HOP", "FORGET_COLD") if int(source_stats.get(key, 0) or 0) > 0
        ) / 4.0
        salvage_ratio = min(1.0, salvage_count / float(max(1, page_count or 1)))
        nli_targets = citations or ([] if unknown_answer else hit_ids[:2])
        nli_reflection = self.preview_nli_citations(final_answer, nli_targets, max_citations=3)
        nli_alignment = float(nli_reflection.get("alignment_score", 0.5))
        nli_contradiction = float(nli_reflection.get("contradiction_ratio", 0.0))
        nli_coverage = float(nli_reflection.get("coverage_ratio", 0.0))

        groundedness = self._clamp(
            0.18
            + (0.24 * hit_strength)
            + (0.16 * citation_ratio)
            + (0.12 * source_diversity)
            + (0.22 * nli_alignment)
            + (0.10 if citation_count and not unknown_answer else 0.0)
            - (0.12 if unknown_answer and hit_count else 0.0)
            - (0.18 * nli_contradiction)
        )
        coverage = self._clamp(
            0.12
            + (0.28 * min(1.0, page_count / 4.0))
            + (0.20 * source_diversity)
            + (0.15 if not unknown_answer else 0.0)
            + (0.15 * salvage_ratio)
            + (0.10 * nli_coverage)
        )
        utility = self._clamp(
            0.15
            + (0.40 if not unknown_answer else 0.05)
            + (0.10 * citation_ratio)
            + (0.10 * source_diversity)
            + (0.20 * float(base_feedback_score))
            + (0.10 * nli_alignment)
            - (0.10 * nli_contradiction)
        )
        memory_score = self._clamp(
            (0.38 * groundedness)
            + (0.25 * coverage)
            + (0.25 * utility)
            + (0.05 * salvage_ratio)
            + (0.07 * nli_alignment)
        )
        feedback_score = self._clamp((0.40 * float(base_feedback_score)) + (0.60 * memory_score))

        if nli_contradiction >= 0.34 and nli_alignment < 0.4:
            recommended_destination = "forget_pool"
        elif nli_alignment >= 0.74 and memory_score >= 0.70:
            recommended_destination = "long_term"
        elif memory_score <= 0.32:
            recommended_destination = "forget_pool"
        else:
            recommended_destination = "short_term"

        summary_bits = []
        if citation_count:
            summary_bits.append(f"{citation_count} cited memories")
        elif hit_count:
            summary_bits.append(f"{hit_count} retrieved hits without citations")
        else:
            summary_bits.append("no usable memory hits")
        if source_diversity >= 0.5:
            summary_bits.append("multi-source evidence")
        if salvage_count:
            summary_bits.append(f"{salvage_count} cold-memory salvage")
        if unknown_answer:
            summary_bits.append("answer remained unresolved")
        if nli_reflection.get("checked_ids"):
            summary_bits.append(
                "nli "
                f"e={len(nli_reflection['entailment'])}/"
                f"c={len(nli_reflection['contradiction'])}/"
                f"n={len(nli_reflection['neutral'])}"
            )

        return {
            "query": query,
            "answer": {
                "groundedness": round(groundedness, 4),
                "coverage": round(coverage, 4),
                "utility": round(utility, 4),
            },
            "nli": nli_reflection,
            "memory": {
                "memory_score": round(memory_score, 4),
                "recommended_destination": recommended_destination,
                "summary": "; ".join(summary_bits) or "stable evidence support",
            },
            "base_feedback_score": round(float(base_feedback_score), 4),
            "feedback_score": round(feedback_score, 4),
        }

    def _compute_route_score(
        self,
        updated_g: float,
        updated_c: float,
        temperature: float,
        feedback_score: float,
        reflection: Optional[Dict[str, Any]] = None,
    ) -> float:
        reflection = reflection or {}
        answer_reflection = reflection.get("answer", {}) if isinstance(reflection, dict) else {}
        memory_reflection = reflection.get("memory", {}) if isinstance(reflection, dict) else {}
        nli_reflection = reflection.get("nli", {}) if isinstance(reflection, dict) else {}
        groundedness = float(answer_reflection.get("groundedness", feedback_score))
        coverage = float(answer_reflection.get("coverage", feedback_score))
        utility = float(answer_reflection.get("utility", feedback_score))
        nli_alignment = float(nli_reflection.get("alignment_score", 0.5))
        nli_contradiction = float(nli_reflection.get("contradiction_ratio", 0.0))
        temperature_norm = self._clamp(float(temperature) / 100.0)
        route_score = self._clamp(
            (0.24 * updated_g)
            + (0.20 * updated_c)
            + (0.15 * groundedness)
            + (0.10 * coverage)
            + (0.10 * utility)
            + (0.10 * float(feedback_score))
            + (0.08 * nli_alignment)
            - (0.12 * nli_contradiction)
            + (0.05 * temperature_norm)
        )

        recommended_destination = str(memory_reflection.get("recommended_destination", "")).strip().lower()
        if recommended_destination == "long_term":
            route_score = self._clamp(route_score + 0.08)
        elif recommended_destination == "forget_pool":
            route_score = self._clamp(route_score - 0.08)
        return route_score

    def _select_destination(self, route_score: float, raw_len: int, location: str) -> str:
        if route_score >= 0.74 or (route_score >= 0.62 and raw_len >= 220):
            return "long_term"
        if route_score <= 0.32:
            return "forget_pool"
        if location == "forget_pool" and route_score >= 0.58:
            return "long_term"
        return "short_term"

    def _archive_to_forget_if_needed(self, memory_id: str, page: Dict[str, Any]) -> None:
        if not memory_id or self.forget_memory.get_page_by_id(memory_id):
            return
        try:
            self.forget_memory.add_page(page)
        except Exception as exc:
            logging.warning("ForgetMemory archive failed during reroute for %s: %s", memory_id, exc)

    def get_background_status(self) -> Dict[str, Any]:
        return {
            "pending_tasks": self._task_queue.qsize(),
            "worker_alive": self._worker.is_alive(),
            "stop_requested": self._stop_event.is_set(),
        }

    def wait_for_background_tasks(self) -> None:
        self._task_queue.join()

    def get_memory_stats(self) -> Dict[str, Any]:
        st_count = len(self.short_term_memory.get_all())
        nodes = self.long_term_memory.get_all_nodes()
        forget_info = self.forget_memory.get_stats() if hasattr(self.forget_memory, "get_stats") else {}

        g_values = [float(item.get("G", 0.0)) for item in nodes]
        c_values = [float(item.get("C", 0.0)) for item in nodes]
        t_values = [float(item.get("T", 0.0)) for item in nodes]
        hot_count = sum(1 for item in nodes if item.get("storage_tier") == "hot")
        cold_count = sum(1 for item in nodes if item.get("storage_tier") == "cold")
        pointer_count = sum(1 for item in nodes if item.get("storage_tier") == "pointer")

        stats = MemoryStats(
            st_count=st_count,
            ltm_count=len(nodes),
            hot_count=hot_count,
            cold_count=cold_count,
            pointer_count=pointer_count,
            forget_count=forget_info.get("lean_records", 0),
            g_distribution=MetricDistribution.from_values(g_values),
            c_distribution=MetricDistribution.from_values(c_values),
            t_distribution=MetricDistribution.from_values(t_values),
        )
        return stats.to_dict()

    def node_exists(self, node_id: str) -> bool:
        return self.long_term_memory.node_exists(node_id)

    def nli_verify_citations(self, generated_answer: str, citation_ids: List[str]) -> Dict[str, Any]:
        results = {
            "entailment": [],
            "contradiction": [],
            "neutral": [],
            "skipped": [],
        }
        if not citation_ids:
            return results

        for citation_id in citation_ids:
            record, location = self._lookup_evidence_record(citation_id)
            if not record:
                results["skipped"].append(citation_id)
                continue

            premise = self._extract_evidence_text(record)
            if not premise:
                results["skipped"].append(citation_id)
                continue
            verdict = self._judge_nli_verdict(premise, generated_answer)

            if verdict == "entailment":
                if location == "long_term":
                    self.long_term_memory.adjust_chunk_metrics(citation_id, delta_G=1.0, delta_C=0.08, delta_T=6.0)
                results["entailment"].append(citation_id)
            elif verdict == "contradiction":
                if location == "long_term":
                    self.long_term_memory.adjust_chunk_metrics(citation_id, delta_G=-0.2, delta_C=-0.2, delta_T=-8.0)
                results["contradiction"].append(citation_id)
            elif verdict == "neutral":
                if location == "long_term":
                    self.long_term_memory.adjust_chunk_metrics(citation_id, delta_T=-1.0)
                results["neutral"].append(citation_id)
            else:
                results["skipped"].append(citation_id)

        return results

    def _run_nli_verify_citations_traced(
        self,
        generated_answer: str,
        citation_ids: List[str],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        unique_ids = self._dedupe_ids(citation_ids)
        before = {citation_id: self._snapshot_memory_state(citation_id) for citation_id in unique_ids}
        results = self.nli_verify_citations(generated_answer, citation_ids)
        changes = []
        for citation_id in unique_ids:
            changes.append(
                {
                    "memory_id": citation_id,
                    "citation_id": citation_id,
                    "verdict": self._resolve_verdict(results, citation_id),
                    "before": before.get(citation_id),
                    "after": self._snapshot_memory_state(citation_id),
                }
            )
        self._record_background_trace(
            "nli_verify_citations",
            trace_context,
            unique_ids,
            changes,
            generated_answer=generated_answer,
            results=results,
        )
        return results

    def schedule_nli_verify_citations(
        self,
        generated_answer: str,
        citation_ids: List[str],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not generated_answer or not citation_ids:
            return
        if self.tracer is not None:
            self._enqueue(
                "nli_verify_citations",
                self._run_nli_verify_citations_traced,
                generated_answer,
                citation_ids,
                trace_context=trace_context,
            )
            return
        self._enqueue("nli_verify_citations", self.nli_verify_citations, generated_answer, citation_ids)

    def schedule_metadata_enrichment(self, fn: Callable, *args, **kwargs) -> None:
        self._enqueue("metadata_enrichment", fn, *args, **kwargs)

    def _retain_short_term_as_evidence(self, page_id: str, updated_g: float, updated_c: float) -> bool:
        page = self.short_term_memory.get_page_by_id(page_id)
        if not page:
            return False

        page["G"] = updated_g
        page["C"] = updated_c
        page["T"] = max(float(page.get("T", 100.0)), 42.0)
        metadata = page.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            page["metadata"] = metadata
        metadata["retained_as_evidence"] = True
        metadata["post_inference_reviewed"] = True

        agent_response = str(page.get("agent_response", ""))
        if len(agent_response) > 280:
            page["agent_response"] = agent_response[:277] + "..."

        gcl_scores = page.get("gcl_scores", {})
        if not isinstance(gcl_scores, dict):
            gcl_scores = {}
        gcl_scores.update(
            {
                "G": updated_g,
                "C": updated_c,
                "T": page["T"],
                "G0": gcl_scores.get("G0", metadata.get("salience_seed", 0.5)),
            }
        )
        page["gcl_scores"] = gcl_scores
        self.short_term_memory.save()
        return True

    def post_inference_memory_flow(
        self,
        statemem,
        page_ids: List[str],
        feedback_score: float = 1.0,
        reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = {
            "processed": [],
            "missing": [],
            "feedback_score": float(feedback_score),
            "reflection": reflection or {},
        }
        seen = set()

        for memory_id in page_ids or []:
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)

            page = statemem.get_memory_by_id(memory_id)
            if not page:
                results["missing"].append(memory_id)
                continue

            location = page.get("location", "unknown")
            raw_content = page.get("raw_content") or page.get("content") or ""
            raw_len = len(raw_content)
            metadata = page.get("meta_data", {})
            if not isinstance(metadata, dict):
                metadata = {}

            base_g = float(page.get("G", metadata.get("G_base", 0.0)))
            base_c = float(page.get("C", metadata.get("C_base", 0.6)))
            seed_g = float(page.get("salience_seed", metadata.get("salience_seed", metadata.get("G_base", 0.5))))
            updated_g = self._clamp(max(base_g, seed_g) + (0.12 * float(feedback_score)) + (0.08 if raw_len >= 280 else 0.0))
            updated_c = self._clamp((base_c * 0.75) + (float(feedback_score) * 0.25))
            citation_id = page.get("citation_id") or page.get("page_id") or memory_id
            temperature = float(page.get("T", metadata.get("T", 0.0)))
            route_score = self._compute_route_score(
                updated_g=updated_g,
                updated_c=updated_c,
                temperature=temperature,
                feedback_score=feedback_score,
                reflection=reflection,
            )
            target_destination = self._select_destination(route_score, raw_len=raw_len, location=location)

            if target_destination == "long_term":
                decision = "COMPRESS" if location != "long_term" else "REINFORCE"
            elif target_destination == "forget_pool":
                decision = "EVICT"
            elif location == "short_term":
                decision = "RETAIN_EVIDENCE"
            else:
                decision = "REINFORCE"

            action = "noop"
            if decision == "COMPRESS":
                promoted_id = self.long_term_memory.add_from_page(page) or citation_id
                if promoted_id and self.long_term_memory.node_exists(promoted_id):
                    self.long_term_memory.adjust_chunk_metrics(
                        promoted_id,
                        delta_G=max(0.0, updated_g - base_g),
                        delta_C=updated_c - base_c,
                        delta_T=12.0,
                    )
                if location == "short_term":
                    self._retain_short_term_as_evidence(memory_id, updated_g, updated_c)
                    action = "stm_promoted_to_ltm_retained"
                elif location == "forget_pool":
                    action = "forget_promoted_to_ltm"
                else:
                    action = "ltm_reinforced"
            elif decision == "EVICT":
                if location == "short_term":
                    self.short_term_memory.evict(memory_id, to_forget_pool=True, forget_memory=self.forget_memory)
                    action = "stm_evicted_to_forget"
                elif citation_id and self.long_term_memory.node_exists(citation_id):
                    self.long_term_memory.adjust_chunk_metrics(citation_id, delta_G=-0.1, delta_C=-0.1, delta_T=-12.0)
                    action = "ltm_cooled"
            elif decision == "RETAIN_EVIDENCE":
                if self._retain_short_term_as_evidence(memory_id, updated_g, updated_c):
                    action = "stm_retained_as_evidence"
            else:
                if citation_id and self.long_term_memory.node_exists(citation_id):
                    self.long_term_memory.adjust_chunk_metrics(
                        citation_id,
                        delta_G=max(0.0, updated_g - base_g) * 0.5,
                        delta_C=max(0.0, updated_c - base_c) * 0.25,
                        delta_T=4.0 if feedback_score >= 0.6 else 1.0,
                    )
                    action = "ltm_reinforced"
                elif location == "short_term":
                    if self._retain_short_term_as_evidence(memory_id, updated_g, updated_c):
                        action = "stm_refreshed"

            results["processed"].append(
                {
                    "page_id": memory_id,
                    "location": location,
                    "decision": decision,
                    "action": action,
                    "target_destination": target_destination,
                    "route_score": round(route_score, 4),
                    "updated_G": round(updated_g, 4),
                    "updated_C": round(updated_c, 4),
                }
            )

        return results

    def _run_post_inference_memory_flow_traced(
        self,
        statemem,
        page_ids: List[str],
        feedback_score: float = 1.0,
        reflection: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        unique_ids = self._dedupe_ids(page_ids)
        before = {memory_id: self._snapshot_memory_state(memory_id, statemem=statemem) for memory_id in unique_ids}
        results = self.post_inference_memory_flow(
            statemem,
            page_ids,
            feedback_score=feedback_score,
            reflection=reflection,
        )
        processed = {item.get("page_id"): item for item in results.get("processed", [])}
        changes = []
        for memory_id in unique_ids:
            change = {
                "memory_id": memory_id,
                "citation_id": memory_id,
                "before": before.get(memory_id),
                "after": self._snapshot_memory_state(memory_id, statemem=statemem),
            }
            if memory_id in processed:
                change.update(processed[memory_id])
            elif memory_id in (results.get("missing") or []):
                change["status"] = "missing"
            changes.append(change)
        self._record_background_trace(
            "post_inference_memory_flow",
            trace_context,
            unique_ids,
            changes,
            feedback_score=float(feedback_score),
            reflection=results.get("reflection", {}),
        )
        return results

    def schedule_post_inference_memory_flow(
        self,
        statemem,
        page_ids: List[str],
        feedback_score: float = 1.0,
        reflection: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not page_ids:
            return
        if self.tracer is not None:
            self._enqueue(
                "post_inference_memory_flow",
                self._run_post_inference_memory_flow_traced,
                statemem,
                page_ids,
                feedback_score,
                reflection,
                trace_context=trace_context,
            )
            return
        self._enqueue(
            "post_inference_memory_flow",
            self.post_inference_memory_flow,
            statemem,
            page_ids,
            feedback_score,
            reflection,
        )

    def soft_decay_memory(self, half_life_days: int = 7):
        self.long_term_memory.apply_soft_decay(half_life_days=half_life_days)

    def schedule_soft_decay_memory(self, half_life_days: int = 7) -> None:
        self._enqueue("soft_decay_memory", self.soft_decay_memory, half_life_days)

    def pointer_compress(self, chunk_id: str) -> Optional[str]:
        return self.long_term_memory.pointer_compress(
            chunk_id,
            summary_model=self.llm_model,
            client=self.client,
        )

    def _run_pointer_compress_traced(
        self,
        chunk_id: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        before = self._snapshot_memory_state(chunk_id)
        pointer_id = self.pointer_compress(chunk_id)
        changes = [
            {
                "memory_id": chunk_id,
                "citation_id": chunk_id,
                "before": before,
                "after": self._snapshot_memory_state(chunk_id),
                "action": "source_context",
            }
        ]
        memory_ids = [chunk_id]
        if pointer_id:
            changes.append(
                {
                    "memory_id": pointer_id,
                    "citation_id": pointer_id,
                    "before": {"memory_id": pointer_id, "location": "missing", "exists": False},
                    "after": self._snapshot_memory_state(pointer_id),
                    "action": "pointer_created",
                    "pointer_to": chunk_id,
                }
            )
            memory_ids.append(pointer_id)
        self._record_background_trace(
            "pointer_compress",
            trace_context,
            memory_ids,
            changes,
            chunk_id=chunk_id,
            pointer_id=pointer_id,
        )
        return pointer_id

    def schedule_pointer_compress(
        self,
        chunk_id: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not chunk_id:
            return
        if self.tracer is not None:
            self._enqueue("pointer_compress", self._run_pointer_compress_traced, chunk_id, trace_context=trace_context)
            return
        self._enqueue("pointer_compress", self.pointer_compress, chunk_id)

    def schedule_pointer_compress_candidates(
        self,
        node_ids: List[str],
        min_raw_chars: int = 900,
        min_temperature: float = 35.0,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        seen = set()
        for node_id in node_ids:
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            entry = self.long_term_memory.get_entry_by_id(node_id)
            if not entry:
                continue
            context_entry = self.long_term_memory.get_primary_context_node(entry) or entry
            if context_entry.get("pointer_to"):
                continue
            if len(str(context_entry.get("raw_content", ""))) < min_raw_chars:
                continue
            if float(context_entry.get("T", 0.0)) < min_temperature:
                continue
            self.schedule_pointer_compress(context_entry.get("id"), trace_context=trace_context)

    def clear_all(self):
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.forget_memory.clear()

    def shutdown(self, wait: bool = False) -> None:
        if wait:
            self.wait_for_background_tasks()
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
