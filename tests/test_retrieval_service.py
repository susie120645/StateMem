import unittest

from memoryos.services.retrieval_service import Retriever


class _StubShortTermMemory:
    def search(self, query, k=5):
        return []


class _StubLongTermMemory:
    def __init__(self, hits, extra_entries=None, related_entries=None):
        self._hits = list(hits)
        self._entries = {}
        self._related_entries = list(related_entries or [])
        self.related_calls = 0
        for entry in [*self._hits, *(extra_entries or []), *self._related_entries]:
            entry_id = entry.get("id")
            if entry_id:
                self._entries[entry_id] = dict(entry)

    def search_knowledge(self, query, top_k=8):
        return list(self._hits)

    def get_primary_context_node(self, node):
        return node

    def get_related_by_entities(self, entities, max_results=5, exclude_ids=None):
        self.related_calls += 1
        exclude_ids = exclude_ids or set()
        return [
            dict(entry)
            for entry in self._related_entries
            if entry.get("id") not in exclude_ids
        ][:max_results]

    def node_exists(self, node_id):
        return node_id in self._entries

    def record_access(self, page_ids):
        return None

    def get_entry_by_id(self, node_id):
        return self._entries.get(node_id)

    def get_all_nodes(self):
        return list(self._entries.values())


class _StubForgetMemory:
    def search(self, query, top_k=4, exclude_ids=None):
        return []


class RetrievalServiceSlotAwareRerankTest(unittest.TestCase):
    def _make_locomo_window_entry(self, session_group_id, window_index, raw_content, score=0.0):
        return {
            "id": f"{session_group_id}_window_{window_index}",
            "doc_id": f"{session_group_id}_window_{window_index}",
            "raw_content": raw_content,
            "summary": f"Window {window_index}",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "LoCoMo Turn Window",
                "chunk_scope": "turn_window",
                "session_id": 26,
                "session_group_id": session_group_id,
                "conversation_id": "conv-48",
                "window_index": window_index,
            },
            "entities": [],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": score,
        }

    def _make_scored_page(self, page_id, score, entity_overlap=0, query_overlap=0):
        page = {
            "page_id": page_id,
            "citation_id": page_id,
            "doc_id": page_id,
            "user_input": f"Question for {page_id}",
            "agent_response": f"Answer for {page_id}",
            "timestamp": "2026-03-20 00:00:00",
            "metadata": {},
            "summary": page_id,
            "raw_content": f"Q: {page_id}\nA: content for {page_id}",
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": score,
            "_source": "LTM_HYBRID",
            "_raw_entities": [],
        }
        if entity_overlap or query_overlap:
            page["graph_1hop"] = {
                "matched_entities": ["Deborah"] if entity_overlap else [],
                "entity_overlap": entity_overlap,
                "query_overlap": query_overlap,
            }
        return page

    def test_location_text_heuristic_ignores_lowercase_non_location_phrases(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(),
        )

        slots = retriever._infer_slots_from_text(
            "Trying to find time for everything is stressful, so I made a schedule to prioritize work."
        )

        self.assertNotIn("LOCATION", slots)

    def test_gap_prefers_turn_window_over_same_session_overview(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory(
                [
                    {
                        "id": "page_overview",
                        "doc_id": "doc_overview",
                        "raw_content": (
                            "Q: LoCoMo Session 26 Overview\n"
                            "A: Deborah and Jolene plan a coffee date when they meet in a new cafe."
                        ),
                        "summary": "Coffee Date",
                        "timestamp": "2026-03-19 09:00:00",
                        "metadata": {
                            "topic": "LoCoMo Session Overview",
                            "chunk_scope": "session_overview",
                            "session_id": 26,
                            "session_group_id": "conv-48_session_26",
                            "conversation_id": "conv-48",
                        },
                        "entities": [],
                        "semantic_edges": [],
                        "G": 0.6,
                        "C": 0.9,
                        "T": 100.0,
                        "salience_seed": 0.6,
                        "score": 0.86,
                    },
                    {
                        "id": "page_window",
                        "doc_id": "doc_window",
                        "raw_content": (
                            "Q: LoCoMo Session 26 Window 2\n"
                            "A: Deborah and Jolene plan to try coffee and fresh pastries when they meet in a new cafe."
                        ),
                        "summary": "Coffee and fresh pastries",
                        "timestamp": "2026-03-19 09:01:00",
                        "metadata": {
                            "topic": "LoCoMo Turn Window",
                            "chunk_scope": "turn_window",
                            "session_id": 26,
                            "session_group_id": "conv-48_session_26",
                            "conversation_id": "conv-48",
                        },
                        "entities": [],
                        "semantic_edges": [],
                        "G": 0.6,
                        "C": 0.9,
                        "T": 100.0,
                        "salience_seed": 0.6,
                        "score": 0.78,
                    },
                ]
            ),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=2,
        )

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_window")
        self.assertGreater(result["retrieved_pages"][0]["gap_rerank"]["multiplier"], 1.0)
        self.assertEqual(result["retrieved_pages"][0]["gap_rerank"]["reason"], "fine_grain_preferred_over_overview")
        self.assertEqual(result["retrieved_pages"][1]["citation_id"], "page_overview")
        self.assertLess(result["retrieved_pages"][1]["gap_rerank"]["multiplier"], 1.0)
        self.assertEqual(result["retrieved_pages"][1]["gap_rerank"]["reason"], "overview_penalized_by_fine_grain")

    def test_where_query_boosts_location_page_and_penalizes_time_only_distractor(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory(
                [
                    {
                        "id": "page_location",
                        "doc_id": "doc_location",
                        "raw_content": "Q: Scott Derrickson residence\nA: Residence note captured for the biography archive.",
                        "summary": "Residence note",
                        "timestamp": "2026-03-19 09:00:00",
                        "metadata": {"topic": "Biography"},
                        "entities": [{"name": "Los Angeles, California", "type": "GPE"}],
                        "semantic_edges": [],
                        "G": 0.6,
                        "C": 0.9,
                        "T": 100.0,
                        "salience_seed": 0.6,
                        "score": 0.5,
                    },
                    {
                        "id": "page_time",
                        "doc_id": "doc_time",
                        "raw_content": "Q: When was Scott Derrickson born?\nA: Scott Derrickson was born on July 16, 1966.",
                        "summary": "Birth date",
                        "timestamp": "2026-03-19 09:01:00",
                        "metadata": {"topic": "Biography"},
                        "entities": ["July 16, 1966"],
                        "semantic_edges": [],
                        "G": 0.6,
                        "C": 0.9,
                        "T": 100.0,
                        "salience_seed": 0.6,
                        "score": 0.72,
                    },
                ]
            ),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("Where does Scott Derrickson live?", top_k=2)

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_location")
        self.assertEqual(result["retrieved_pages"][0]["slot_rerank"]["multiplier"], 1.5)
        self.assertEqual(result["retrieved_pages"][0]["slot_rerank"]["reason"], "location_match")
        self.assertEqual(result["retrieved_pages"][1]["citation_id"], "page_time")
        self.assertEqual(result["retrieved_pages"][1]["slot_rerank"]["multiplier"], 0.7)
        self.assertEqual(result["retrieved_pages"][1]["slot_rerank"]["reason"], "time_only_distractor")

    def test_graph_one_hop_gate_triggers_for_weak_overview_only(self):
        session_group_id = "conv-48_session_26"
        overview_hit = {
            "id": f"{session_group_id}_overview",
            "doc_id": f"{session_group_id}_overview",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview\n"
                "A: Deborah and Jolene plan a coffee date when they meet in a new cafe."
            ),
            "summary": "Coffee date overview",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "LoCoMo Session Overview",
                "chunk_scope": "session_overview",
                "session_id": 26,
                "session_group_id": session_group_id,
                "conversation_id": "conv-48",
            },
            "entities": [],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.88,
        }
        related_window = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee when they meet in a new cafe.",
        )
        long_term_memory = _StubLongTermMemory(
            [overview_hit],
            related_entries=[related_window],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=4,
            token_budget=5000,
        )
        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]

        self.assertIn(f"{session_group_id}_window_2", retrieved_ids)
        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 1)
        self.assertEqual(long_term_memory.related_calls, 1)

    def test_graph_one_hop_question_gate_prefers_specific_fact_queries(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(),
        )

        self.assertTrue(
            retriever._question_supports_graph_one_hop(
                "What do Deborah and Jolene plan to try when they meet in a new cafe?"
            )
        )
        self.assertFalse(
            retriever._question_supports_graph_one_hop(
                "Would Deborah likely enjoy meeting Jolene in a new cafe?"
            )
        )

    def test_graph_one_hop_gate_triggers_when_top_band_is_overview_dominant(self):
        session_group_id = "conv-48_session_26"
        overview_hit = {
            "id": f"{session_group_id}_overview",
            "doc_id": f"{session_group_id}_overview",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview\n"
                "A: Deborah and Jolene plan a coffee date when they meet in a new cafe."
            ),
            "summary": "Coffee date overview",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "LoCoMo Session Overview",
                "chunk_scope": "session_overview",
                "session_id": 26,
                "session_group_id": session_group_id,
                "conversation_id": "conv-48",
            },
            "entities": [],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.88,
        }
        second_overview_hit = {
            "id": f"{session_group_id}_overview_followup",
            "doc_id": f"{session_group_id}_overview_followup",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview Follow-up\n"
                "A: Deborah and Jolene talk about meeting in a new cafe and trying drinks together."
            ),
            "summary": "Cafe meetup overview",
            "timestamp": "2026-03-19 09:02:00",
            "metadata": {
                "topic": "LoCoMo Session Overview",
                "chunk_scope": "session_overview",
                "session_id": 26,
                "session_group_id": session_group_id,
                "conversation_id": "conv-48",
            },
            "entities": [],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.81,
        }
        related_window = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee when they meet in a new cafe.",
        )
        long_term_memory = _StubLongTermMemory(
            [overview_hit, second_overview_hit],
            related_entries=[related_window],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=4,
            token_budget=5000,
        )
        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]

        self.assertIn(f"{session_group_id}_window_2", retrieved_ids)
        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 1)
        self.assertEqual(long_term_memory.related_calls, 1)

    def test_graph_one_hop_gate_skips_when_strong_turn_window_already_exists(self):
        session_group_id = "conv-48_session_26"
        window_2 = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee when they meet in a new cafe.",
            score=0.84,
        )
        long_term_memory = _StubLongTermMemory(
            [window_2],
            related_entries=[window_2],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=3,
            token_budget=5000,
        )

        self.assertEqual(result["source_stats"]["LTM"], 1)
        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 0)
        self.assertEqual([page["citation_id"] for page in result["retrieved_pages"]], [f"{session_group_id}_window_2"])
        self.assertEqual(long_term_memory.related_calls, 0)

    def test_graph_one_hop_gate_skips_for_hypothetical_question(self):
        session_group_id = "conv-48_session_26"
        overview_hit = {
            "id": f"{session_group_id}_overview",
            "doc_id": f"{session_group_id}_overview",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview\n"
                "A: Deborah and Jolene plan a coffee date when they meet in a new cafe."
            ),
            "summary": "Coffee date overview",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "LoCoMo Session Overview",
                "chunk_scope": "session_overview",
                "session_id": 26,
                "session_group_id": session_group_id,
                "conversation_id": "conv-48",
            },
            "entities": [],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.88,
        }
        related_window = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee when they meet in a new cafe.",
        )
        long_term_memory = _StubLongTermMemory(
            [overview_hit],
            related_entries=[related_window],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "Would Deborah likely enjoy meeting Jolene in a new cafe?",
            top_k=4,
            token_budget=5000,
        )

        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 0)
        self.assertEqual(long_term_memory.related_calls, 0)

    def test_graph_one_hop_accepts_strong_candidate_just_outside_top_k(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(),
        )
        pages = [
            self._make_scored_page("base_1", 0.62),
            self._make_scored_page("base_2", 0.58),
            self._make_scored_page("base_3", 0.54),
            self._make_scored_page("base_4", 0.50),
        ]
        candidate = self._make_scored_page("graph_strong", 0.49, entity_overlap=3, query_overlap=2)

        accepted = retriever._accept_graph_one_hop_pages(
            "What project was Deborah working on?",
            pages,
            [candidate],
            top_k=4,
        )

        self.assertEqual([page["citation_id"] for page in accepted], ["graph_strong"])

    def test_graph_one_hop_rejects_weak_candidate_outside_top_k(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(),
        )
        pages = [
            self._make_scored_page("base_1", 0.62),
            self._make_scored_page("base_2", 0.58),
            self._make_scored_page("base_3", 0.54),
            self._make_scored_page("base_4", 0.50),
        ]
        candidate = self._make_scored_page("graph_weak", 0.49, entity_overlap=1, query_overlap=2)

        accepted = retriever._accept_graph_one_hop_pages(
            "What project was Deborah working on?",
            pages,
            [candidate],
            top_k=4,
        )

        self.assertEqual(accepted, [])


if __name__ == "__main__":
    unittest.main()
