import unittest

from statemem.services.retrieval_service import Retriever


class _StubShortTermMemory:
    def search(self, query, k=5):
        return []


class _StubLongTermMemory:
    def __init__(self, hits, extra_entries=None, related_entries=None, temporal_hits=None, index_related_entries=True):
        self._hits = list(hits)
        self._temporal_hits = list(temporal_hits or [])
        self._entries = {}
        self._doc_children = {}
        self._related_entries = list(related_entries or [])
        self.related_calls = 0
        self.search_knowledge_calls = []
        self.search_temporal_calls = []
        indexed_entries = [*self._hits, *(extra_entries or []), *self._temporal_hits]
        if index_related_entries:
            indexed_entries.extend(self._related_entries)
        for entry in indexed_entries:
            entry_id = entry.get("id")
            if entry_id:
                self._entries[entry_id] = dict(entry)
                doc_id = str(entry.get("doc_id") or "").strip()
                if doc_id:
                    self._doc_children.setdefault(doc_id, []).append(entry_id)

    def search_knowledge(self, query, top_k=8, **kwargs):
        self.search_knowledge_calls.append({"query": query, "top_k": top_k})
        return list(self._hits)

    def search_temporal(self, query, top_k=6):
        self.search_temporal_calls.append({"query": query, "top_k": top_k})
        return list(self._temporal_hits)[:top_k]

    def get_primary_context_node(self, node):
        if not isinstance(node, dict):
            return node
        pointer_to = node.get("pointer_to")
        if pointer_to and pointer_to in self._entries:
            return self._entries[pointer_to]
        parent_id = node.get("parent_id")
        if node.get("node_type") == "chunk" and parent_id and parent_id in self._entries:
            return self._entries[parent_id]
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

    def record_access(self, page_ids, **kwargs):
        return None

    def get_entry_by_id(self, node_id):
        return self._entries.get(node_id)

    def get_all_nodes(self):
        return list(self._entries.values())

    def get_chunks_by_doc_id(self, doc_id):
        return [dict(self._entries[node_id]) for node_id in self._doc_children.get(doc_id, []) if node_id in self._entries]


class _StubForgetMemory:
    def __init__(self, archive_hits=None, search_hits=None, temporal_hits=None):
        self._archive_hits = list(archive_hits or [])
        self._search_hits = list(search_hits or [])
        self._temporal_hits = list(temporal_hits or [])

    def search_archive(self, query, top_k=4, exclude_ids=None, **kwargs):
        exclude_ids = exclude_ids or set()
        return [
            dict(hit)
            for hit in self._archive_hits
            if hit.get("page_id") not in exclude_ids
        ][:top_k]

    def search(self, query, top_k=4, exclude_ids=None, **kwargs):
        exclude_ids = exclude_ids or set()
        return [
            dict(hit)
            for hit in self._search_hits
            if hit.get("page_id") not in exclude_ids
        ][:top_k]

    def search_temporal(self, query, top_k=4, exclude_ids=None, **kwargs):
        exclude_ids = exclude_ids or set()
        return [
            dict(hit)
            for hit in self._temporal_hits
            if hit.get("page_id") not in exclude_ids
        ][:top_k]


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
            index_related_entries=False,
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

    def test_graph_one_hop_can_be_disabled_at_retriever_level(self):
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
            index_related_entries=False,
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
            enable_graph_one_hop=False,
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=4,
            token_budget=5000,
        )
        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]

        self.assertNotIn(f"{session_group_id}_window_2", retrieved_ids)
        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 0)
        self.assertEqual(long_term_memory.related_calls, 0)

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
        self.assertTrue(
            retriever._question_supports_graph_one_hop(
                "Is Deborah married?"
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
            index_related_entries=False,
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

    def test_session_window_supplement_adds_anchor_and_neighbors_for_overview_hit(self):
        session_group_id = "conv-48_session_26"
        overview_hit = {
            "id": f"{session_group_id}_overview",
            "doc_id": f"{session_group_id}_overview",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview\n"
                "A: Deborah and Jolene talk about meeting in a new cafe and trying something together."
            ),
            "summary": "Cafe meetup overview",
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
        window_1 = self._make_locomo_window_entry(
            session_group_id,
            1,
            "Q: LoCoMo Session 26 Window 1\nA: Deborah and Jolene discuss where to meet for coffee.",
        )
        window_2 = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee and pastries when they meet in a new cafe.",
        )
        window_3 = self._make_locomo_window_entry(
            session_group_id,
            3,
            "Q: LoCoMo Session 26 Window 3\nA: Deborah and Jolene decide to meet on Saturday morning.",
        )
        long_term_memory = _StubLongTermMemory(
            [overview_hit],
            extra_entries=[window_1, window_2, window_3],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
            enable_graph_one_hop=False,
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=4,
            token_budget=5000,
        )
        page_lookup = {page["citation_id"]: page for page in result["retrieved_pages"]}

        self.assertIn(f"{session_group_id}_window_2", page_lookup)
        self.assertIn(f"{session_group_id}_window_1", page_lookup)
        self.assertIn(f"{session_group_id}_window_3", page_lookup)
        self.assertEqual(
            page_lookup[f"{session_group_id}_window_2"]["session_window_expand"]["reason"],
            "overview_hit_anchor_window",
        )
        self.assertEqual(
            page_lookup[f"{session_group_id}_window_1"]["session_window_expand"]["reason"],
            "overview_session_adjacent",
        )
        self.assertEqual(result["source_stats"]["GRAPH_1HOP"], 0)

    def test_session_window_supplement_skips_irrelevant_neighbors_for_overview_hit(self):
        session_group_id = "conv-48_session_26"
        overview_hit = {
            "id": f"{session_group_id}_overview",
            "doc_id": f"{session_group_id}_overview",
            "raw_content": (
                "Q: LoCoMo Session 26 Overview\n"
                "A: Deborah and Jolene talk about trying coffee in a new cafe."
            ),
            "summary": "Cafe meetup overview",
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
        window_1 = self._make_locomo_window_entry(
            session_group_id,
            1,
            "Q: LoCoMo Session 26 Window 1\nA: The weather was sunny and everyone talked about schedules.",
        )
        window_2 = self._make_locomo_window_entry(
            session_group_id,
            2,
            "Q: LoCoMo Session 26 Window 2\nA: Deborah and Jolene plan to try coffee and pastries in a new cafe.",
        )
        window_3 = self._make_locomo_window_entry(
            session_group_id,
            3,
            "Q: LoCoMo Session 26 Window 3\nA: Later they discussed traffic and the train timetable.",
        )
        long_term_memory = _StubLongTermMemory(
            [overview_hit],
            extra_entries=[window_1, window_2, window_3],
        )
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
            enable_graph_one_hop=False,
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=4,
            token_budget=5000,
        )

        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]
        self.assertIn(f"{session_group_id}_window_2", retrieved_ids)
        self.assertNotIn(f"{session_group_id}_window_1", retrieved_ids)
        self.assertNotIn(f"{session_group_id}_window_3", retrieved_ids)

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

    def test_auto_merge_preserves_window_dia_ids_from_pointer_source(self):
        parent_entry = {
            "id": "doc_parent",
            "doc_id": "doc_parent",
            "raw_content": "Q: Travel log\nA: Deborah traveled to Thailand and Brazil.",
            "summary": "Travel log",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "chunk_scope": "session_transcript",
            },
            "entities": ["Deborah"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
        }
        pointer_hit = {
            "id": "doc_parent_ptr",
            "doc_id": "doc_parent",
            "pointer_to": "doc_parent",
            "raw_content": "Travel summary pointer",
            "summary": "Travel summary pointer",
            "timestamp": "2026-03-19 09:01:00",
            "metadata": {
                "window_dia_ids": ["D6:10", "D23:18"],
            },
            "entities": ["Deborah"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.91,
        }
        long_term_memory = _StubLongTermMemory([pointer_hit], extra_entries=[parent_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("Which countries has Deborah traveled to?", top_k=2)

        page = result["retrieved_pages"][0]
        self.assertEqual(page["citation_id"], "doc_parent")
        self.assertEqual(page["metadata"]["window_dia_ids"], ["D6:10", "D23:18"])

    def test_temporal_recall_branch_adds_temporal_hits_for_time_questions(self):
        temporal_entry = {
            "id": "page_temporal",
            "doc_id": "page_temporal",
            "raw_content": "Q: Family memory\nA: Jolene's mother passed away in 2022.",
            "summary": "Family memory",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "temporal_forms": ["year:2022"],
            },
            "entities": ["Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.91,
            "temporal_score": 1.0,
        }
        long_term_memory = _StubLongTermMemory([], temporal_hits=[temporal_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("When did Jolene's mother pass away?", top_k=2)

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_temporal")
        self.assertEqual(result["retrieved_pages"][0]["temporal_recall"]["source"], "ltm")

    def test_auto_merge_preserves_window_dia_ids_when_pointer_resolves_to_parent(self):
        pointer_hit = {
            "id": "page_parent_ptr",
            "doc_id": "page_parent",
            "pointer_to": "page_parent",
            "raw_content": "Pointer summary",
            "summary": "Pointer summary",
            "metadata": {"window_dia_ids": ["D6:10", "D23:18"]},
            "entities": ["Deborah"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.92,
        }
        parent_entry = {
            "id": "page_parent",
            "doc_id": "page_parent",
            "raw_content": "Parent content",
            "summary": "Parent content",
            "metadata": {"chunk_scope": "session_transcript"},
            "entities": ["Deborah"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
        }
        long_term_memory = _StubLongTermMemory([pointer_hit], extra_entries=[parent_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("Which countries has Deborah traveled to?", top_k=2)

        page = result["retrieved_pages"][0]
        self.assertEqual(page["citation_id"], "page_parent")
        self.assertEqual(page["metadata"]["window_dia_ids"], ["D6:10", "D23:18"])

    def test_temporal_recall_branch_adds_temporal_hits(self):
        temporal_entry = {
            "id": "page_temporal",
            "doc_id": "page_temporal",
            "raw_content": "Jolene's mom passed away in 2022.",
            "summary": "Family timeline",
            "metadata": {"temporal_forms": ["year:2022"]},
            "entities": ["Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.91,
            "temporal_score": 1.0,
        }
        long_term_memory = _StubLongTermMemory([], temporal_hits=[temporal_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("When did Jolene's mother pass away?", top_k=2)

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_temporal")
        self.assertEqual(result["retrieved_pages"][0]["temporal_recall"]["source"], "ltm")

    def test_temporal_recall_branch_adds_hits_for_non_when_time_constrained_query(self):
        temporal_entry = {
            "id": "page_temporal",
            "doc_id": "page_temporal",
            "raw_content": "Jolene was working on an electricity engineering project in January 2023.",
            "summary": "Project timeline",
            "metadata": {"temporal_forms": ["month:2023-01"]},
            "entities": ["Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.91,
            "temporal_score": 1.0,
        }
        long_term_memory = _StubLongTermMemory([], temporal_hits=[temporal_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What kind of project was Jolene working on in the beginning of January 2023?",
            top_k=2,
        )

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_temporal")
        self.assertEqual(result["retrieved_pages"][0]["temporal_recall"]["source"], "ltm")

    def test_session_diversity_keeps_secondary_session_in_top_k(self):
        same_session_pages = []
        for index, score in enumerate((0.93, 0.92, 0.91), start=1):
            page = self._make_scored_page(f"session_a_{index}", score)
            page["metadata"] = {
                "session_group_id": "conv-48_session_26",
                "conversation_id": "conv-48",
                "session_id": 26,
            }
            same_session_pages.append(page)

        other_session_page = self._make_scored_page("session_b_1", 0.88)
        other_session_page["metadata"] = {
            "session_group_id": "conv-48_session_27",
            "conversation_id": "conv-48",
            "session_id": 27,
        }

        hits = []
        for page in [*same_session_pages, other_session_page]:
            hits.append(
                {
                    "id": page["citation_id"],
                    "doc_id": page["doc_id"],
                    "raw_content": page["raw_content"],
                    "summary": page["summary"],
                    "timestamp": page["timestamp"],
                    "metadata": page["metadata"],
                    "entities": [],
                    "semantic_edges": [],
                    "G": page["G"],
                    "C": page["C"],
                    "T": page["T"],
                    "salience_seed": page["salience_seed"],
                    "score": page["score"],
                }
            )

        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory(hits),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context("Which memory note matters here?", top_k=3, token_budget=5000)

        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]
        retrieved_sessions = [
            page.get("metadata", {}).get("session_group_id")
            for page in result["retrieved_pages"]
        ]

        self.assertIn("session_b_1", retrieved_ids)
        self.assertEqual(retrieved_sessions.count("conv-48_session_26"), 2)

    def test_retrieve_context_expands_ltm_candidate_pool_size(self):
        long_term_memory = _StubLongTermMemory([])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        retriever.retrieve_context("Memory note summary", top_k=3, token_budget=2000)

        self.assertEqual(long_term_memory.search_knowledge_calls[0]["top_k"], 16)

    def test_temporal_recall_uses_expanded_temporal_candidate_pool_size(self):
        temporal_entry = {
            "id": "page_temporal",
            "doc_id": "page_temporal",
            "raw_content": "Jolene's mom passed away in 2022.",
            "summary": "Family timeline",
            "metadata": {"temporal_forms": ["year:2022"]},
            "entities": ["Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.91,
            "temporal_score": 1.0,
        }
        long_term_memory = _StubLongTermMemory([], temporal_hits=[temporal_entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term_memory,
            forget_memory=_StubForgetMemory(),
        )

        retriever.retrieve_context("When did Jolene's mother pass away?", top_k=3, token_budget=2000)

        self.assertEqual(long_term_memory.search_temporal_calls[0]["top_k"], 9)

    def test_granularity_diversity_caps_overviews_when_fine_grained_hits_exist(self):
        hits = [
            {
                "id": "overview_1",
                "doc_id": "overview_1",
                "raw_content": "Q: Session overview\nA: High-level summary one.",
                "summary": "Overview one",
                "timestamp": "2026-03-19 09:00:00",
                "metadata": {"chunk_scope": "session_overview", "session_group_id": "conv-48_session_26"},
                "entities": [],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.95,
            },
            {
                "id": "overview_2",
                "doc_id": "overview_2",
                "raw_content": "Q: Session overview\nA: High-level summary two.",
                "summary": "Overview two",
                "timestamp": "2026-03-19 09:01:00",
                "metadata": {"chunk_scope": "session_overview", "session_group_id": "conv-48_session_27"},
                "entities": [],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.94,
            },
            {
                "id": "window_1",
                "doc_id": "window_1",
                "raw_content": "Q: LoCoMo Session 26 Window 1\nA: Deborah planned to try coffee in a new cafe.",
                "summary": "Window one",
                "timestamp": "2026-03-19 09:02:00",
                "metadata": {"chunk_scope": "turn_window", "session_group_id": "conv-48_session_26", "window_index": 1},
                "entities": [],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.93,
            },
            {
                "id": "transcript_1",
                "doc_id": "transcript_1",
                "raw_content": "Q: Session transcript\nA: Detailed transcript evidence about coffee planning.",
                "summary": "Transcript one",
                "timestamp": "2026-03-19 09:03:00",
                "metadata": {"chunk_scope": "session_transcript", "session_group_id": "conv-48_session_28"},
                "entities": [],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.92,
            },
        ]
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory(hits),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            top_k=3,
            token_budget=5000,
        )

        retrieved_ids = [page["citation_id"] for page in result["retrieved_pages"]]
        self.assertIn("window_1", retrieved_ids)
        self.assertIn("transcript_1", retrieved_ids)
        overview_ids = {
            page["citation_id"]
            for page in result["retrieved_pages"]
            if page.get("metadata", {}).get("chunk_scope") == "session_overview"
        }
        self.assertEqual(len(overview_ids), 1)

    def test_detail_query_expands_ltm_candidate_pool(self):
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(),
        )

        retriever.retrieve_context(
            "Which game did Jolene recommend to Deborah for being thrilling and interesting?",
            top_k=3,
            token_budget=5000,
        )

        self.assertEqual(retriever.long_term_memory.search_knowledge_calls[0]["top_k"], 20)

    def test_detail_query_candidate_bias_prefers_turn_window_with_evidence(self):
        hits = [
            {
                "id": "overview_1",
                "doc_id": "overview_1",
                "raw_content": "Q: Session overview\nA: Jolene talked about hobbies, travel, and books.",
                "summary": "Overview one",
                "timestamp": "2026-03-19 09:00:00",
                "metadata": {"chunk_scope": "session_overview", "session_group_id": "conv-48_session_12"},
                "entities": ["Jolene", "Deborah"],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.92,
            },
            {
                "id": "window_1",
                "doc_id": "window_1",
                "raw_content": (
                    "Q: LoCoMo Session 12 Window 1\n"
                    "A: Jolene recommended the game Dead Space to Deborah because it was thrilling and interesting."
                ),
                "summary": "Window one",
                "timestamp": "2026-03-19 09:02:00",
                "metadata": {
                    "chunk_scope": "turn_window",
                    "session_group_id": "conv-48_session_12",
                    "window_index": 1,
                    "window_dia_ids": ["D12:4"],
                },
                "entities": ["Jolene", "Deborah", "Dead Space"],
                "semantic_edges": [],
                "G": 0.6,
                "C": 0.9,
                "T": 100.0,
                "salience_seed": 0.6,
                "score": 0.8,
            },
        ]
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory(hits),
            forget_memory=_StubForgetMemory(),
            enable_graph_one_hop=False,
        )

        result = retriever.retrieve_context(
            "Which game did Jolene recommend to Deborah for being thrilling and interesting?",
            top_k=1,
            token_budget=5000,
        )

        page = result["retrieved_pages"][0]
        self.assertEqual(page["citation_id"], "window_1")
        self.assertEqual(page["metadata"]["window_dia_ids"], ["D12:4"])

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

    def test_dependency_query_kind_adjusts_ltm_candidate_pool_size(self):
        entry = {
            "id": "page_overview",
            "doc_id": "doc_overview",
            "raw_content": "Q: Deborah context\nA: Deborah keeps a stable profile note about travel and painting.",
            "summary": "Profile note",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {"topic": "Profile", "chunk_scope": "session_overview"},
            "entities": ["Deborah"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.5,
        }
        long_term = _StubLongTermMemory([entry])
        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=long_term,
            forget_memory=_StubForgetMemory(),
        )

        global_result = retriever.retrieve_context(
            "Explain Deborah context",
            top_k=3,
            trace_context={"query_kind": "global"},
        )
        local_result = retriever.retrieve_context(
            "Explain Deborah context",
            top_k=3,
            trace_context={"query_kind": "local"},
        )

        self.assertEqual(global_result["retrieval_profile"]["query_kind"], "global")
        self.assertEqual(local_result["retrieval_profile"]["query_kind"], "local")
        self.assertLess(long_term.search_knowledge_calls[0]["top_k"], long_term.search_knowledge_calls[1]["top_k"])

    def test_dependency_query_kind_biases_overview_and_window_ranking(self):
        overview = {
            "id": "page_overview",
            "doc_id": "doc_overview",
            "raw_content": "Q: Deborah context\nA: Deborah and Jolene have an ongoing note about their shared routines.",
            "summary": "Session overview",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "Overview",
                "chunk_scope": "session_overview",
                "session_group_id": "conv-9_session_4",
            },
            "entities": ["Deborah", "Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.6,
        }
        window = {
            "id": "page_window",
            "doc_id": "doc_window",
            "raw_content": "Q: Deborah context\nA: Deborah and Jolene have an ongoing note about their shared routines.",
            "summary": "Turn window",
            "timestamp": "2026-03-19 09:01:00",
            "metadata": {
                "topic": "Window",
                "chunk_scope": "turn_window",
                "session_group_id": "conv-9_session_4",
            },
            "entities": ["Deborah", "Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.6,
        }

        global_retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([overview, window]),
            forget_memory=_StubForgetMemory(),
        )
        local_retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([overview, window]),
            forget_memory=_StubForgetMemory(),
        )

        global_result = global_retriever.retrieve_context(
            "Explain Deborah context",
            top_k=2,
            trace_context={"query_kind": "global"},
        )
        local_result = local_retriever.retrieve_context(
            "Explain Deborah context",
            top_k=2,
            trace_context={"query_kind": "local"},
        )

        self.assertEqual(global_result["retrieved_pages"][0]["citation_id"], "page_overview")
        self.assertEqual(local_result["retrieved_pages"][0]["citation_id"], "page_window")
        self.assertGreater(
            global_result["retrieved_pages"][0]["dependency_channel_multiplier"],
            global_result["retrieved_pages"][1]["dependency_channel_multiplier"],
        )
        self.assertGreater(
            local_result["retrieved_pages"][0]["dependency_channel_multiplier"],
            local_result["retrieved_pages"][1]["dependency_channel_multiplier"],
        )

    def test_dependency_query_kind_infers_global_from_summary_query(self):
        overview = {
            "id": "page_overview",
            "doc_id": "doc_overview",
            "raw_content": "Q: Deborah context\nA: Deborah and Jolene have an ongoing note about their shared routines.",
            "summary": "Session overview",
            "timestamp": "2026-03-19 09:00:00",
            "metadata": {
                "topic": "Overview",
                "chunk_scope": "session_overview",
                "session_group_id": "conv-9_session_4",
            },
            "entities": ["Deborah", "Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.6,
        }
        window = {
            "id": "page_window",
            "doc_id": "doc_window",
            "raw_content": "Q: Deborah context\nA: Deborah and Jolene have an ongoing note about their shared routines.",
            "summary": "Turn window",
            "timestamp": "2026-03-19 09:01:00",
            "metadata": {
                "topic": "Window",
                "chunk_scope": "turn_window",
                "session_group_id": "conv-9_session_4",
            },
            "entities": ["Deborah", "Jolene"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.6,
        }

        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([overview, window]),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "Summarize the overall meeting decisions and action items",
            top_k=2,
            trace_context={"task_hint": "meeting_summary"},
        )

        self.assertEqual(result["retrieval_profile"]["query_kind"], "global")
        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "page_overview")

    def test_archive_index_cold_hit_is_inserted_before_generic_cold_search(self):
        archive_hit = {
            "page_id": "cold_archive",
            "doc_id": "cold_archive",
            "user_input": "Who was Julie dating in October 1995?",
            "agent_response": "Julie was dating Ross.",
            "raw_content": "Q: Who was Julie dating in October 1995?\nA: Julie was dating Ross.",
            "summary": "Julie was dating Ross in October 1995.",
            "timestamp": "2026-03-20 00:00:00",
            "archived_at": "2026-03-20 00:00:00",
            "entities": ["Julie", "Ross"],
            "keywords": ["julie", "dating", "october", "1995", "ross"],
            "semantic_edges": [],
            "G": 0.5,
            "C": 0.5,
            "T": 18.0,
            "salience_seed": 0.35,
            "score": 0.92,
            "archive_match": {
                "entity_overlap": 1.0,
                "temporal_overlap": 1.0,
                "relation_overlap": 1.0,
                "keyword_overlap": 0.6,
                "relation_terms": ["dating"],
                "temporal_forms": ["1995-10"],
                "source_family": "friends_s11",
                "source_session": "11",
            },
            "metadata": {"temporal_forms": ["1995-10"]},
        }
        generic_cold_hit = {
            "page_id": "cold_generic",
            "doc_id": "cold_generic",
            "user_input": "Who was Rachel dating in January 1994?",
            "agent_response": "Rachel was dating Paolo.",
            "raw_content": "Q: Who was Rachel dating in January 1994?\nA: Rachel was dating Paolo.",
            "summary": "Rachel was dating Paolo.",
            "timestamp": "2026-03-20 00:00:00",
            "archived_at": "2026-03-20 00:00:00",
            "entities": ["Rachel", "Paolo"],
            "keywords": ["rachel", "dating", "paolo"],
            "semantic_edges": [],
            "G": 0.5,
            "C": 0.5,
            "T": 18.0,
            "salience_seed": 0.35,
            "score": 0.51,
            "metadata": {"temporal_forms": ["1994-01"]},
        }

        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([]),
            forget_memory=_StubForgetMemory(
                archive_hits=[archive_hit],
                search_hits=[generic_cold_hit],
            ),
        )

        result = retriever.retrieve_context(
            "So, who was Julie's date on the night of October 1995?",
            top_k=2,
        )

        self.assertEqual(result["retrieved_pages"][0]["citation_id"], "cold_archive")
        self.assertEqual(result["retrieved_pages"][0]["_cold_channel"], "archive_index")
        self.assertEqual(result["retrieved_pages"][0]["archive_index"]["relation_terms"], ["dating"])

    def test_two_hop_relation_query_adds_same_doc_roommate_sibling(self):
        dating_hit = {
            "id": "episode_s1_c1",
            "doc_id": "episode_s1",
            "raw_content": (
                "Q: Episode recap\n"
                "A: Joey asks Chandler about Janice, and Chandler says he broke up with her on October 22, 1994."
            ),
            "summary": "Chandler broke up with Janice.",
            "timestamp": "2026-03-20 00:00:00",
            "metadata": {"topic": "Episode recap"},
            "entities": ["Joey", "Chandler", "Janice"],
            "semantic_edges": [],
            "G": 0.6,
            "C": 0.9,
            "T": 100.0,
            "salience_seed": 0.6,
            "score": 0.88,
        }
        roommate_sibling = {
            "id": "episode_s1_c2",
            "doc_id": "episode_s1",
            "raw_content": (
                "Q: Episode recap\n"
                "A: Joey introduces Chandler and says Chandler is his roommate."
            ),
            "summary": "Joey says Chandler is his roommate.",
            "timestamp": "2026-03-20 00:00:00",
            "metadata": {"topic": "Episode recap"},
            "entities": ["Joey", "Chandler"],
            "semantic_edges": [],
            "G": 0.55,
            "C": 0.9,
            "T": 88.0,
            "salience_seed": 0.5,
            "score": 0.12,
        }

        retriever = Retriever(
            short_term_memory=_StubShortTermMemory(),
            long_term_memory=_StubLongTermMemory([dating_hit], extra_entries=[roommate_sibling]),
            forget_memory=_StubForgetMemory(),
        )

        result = retriever.retrieve_context(
            "Remember when Joey had a roommate back on October 20, 1994? So, who was going out with that roommate by October 22, 1994?",
            top_k=3,
        )

        sibling_pages = [
            page for page in result["retrieved_pages"] if page.get("citation_id") == "episode_s1_c2"
        ]
        self.assertEqual(len(sibling_pages), 1)
        sibling_page = sibling_pages[0]
        self.assertEqual(sibling_page["_source"], "DOC_SIBLING_EXPAND")
        self.assertEqual(sibling_page["doc_sibling_expand"]["matched_relations"], ["roommate"])
        self.assertEqual(result["retrieval_profile"]["doc_sibling_hits"], 1)
        self.assertGreaterEqual(result["source_stats"]["LTM"], 2)


if __name__ == "__main__":
    unittest.main()
