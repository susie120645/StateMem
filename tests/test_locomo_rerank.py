import importlib.machinery
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "statemem" / "run_locomo"
LOADER = importlib.machinery.SourceFileLoader("statemem_run_locomo", str(MODULE_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
RUN_LOCOMO = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(RUN_LOCOMO)


class LocomoSlotAwareRerankTest(unittest.TestCase):
    def test_gap_penalizes_overview_when_same_session_window_is_available(self):
        retrieval_results = {
            "retrieved_pages": [
                {
                    "page_id": "page_overview",
                    "citation_id": "page_overview",
                    "user_input": "LoCoMo Session 26 Overview",
                    "summary": "Coffee Date",
                    "raw_content": "Deborah and Jolene plan to try coffee when they meet in a new cafe.",
                    "metadata": {
                        "chunk_scope": "session_overview",
                        "session_id": 26,
                        "session_group_id": "conv-48_session_26",
                    },
                    "final_score": 1.45,
                },
                {
                    "page_id": "page_window",
                    "citation_id": "page_window",
                    "user_input": "LoCoMo Session 26 Window 2",
                    "summary": "Coffee and fresh pastries",
                    "raw_content": "Deborah and Jolene plan to try coffee and fresh pastries when they meet in a new cafe.",
                    "metadata": {
                        "chunk_scope": "turn_window",
                        "session_id": 26,
                        "session_group_id": "conv-48_session_26",
                    },
                    "final_score": 1.1,
                },
            ],
            "hit_ids": ["page_overview", "page_window"],
        }

        reranked = RUN_LOCOMO.rerank_retrieval_results(
            "What do Deborah and Jolene plan to try when they meet in a new cafe?",
            retrieval_results,
        )
        pages = reranked["retrieved_pages"]

        self.assertEqual(pages[0]["citation_id"], "page_window")
        self.assertGreater(pages[0]["locomo_gap_adjustment"], 0.0)
        self.assertEqual(pages[1]["citation_id"], "page_overview")
        self.assertLess(pages[1]["locomo_gap_adjustment"], 0.0)

    def test_slot_rerank_signal_survives_second_stage_reranking(self):
        retrieval_results = {
            "retrieved_pages": [
                {
                    "page_id": "page_location",
                    "citation_id": "page_location",
                    "user_input": "Scott Derrickson residence",
                    "summary": "Residence note",
                    "raw_content": "Scott Derrickson residence note for Los Angeles archive.",
                    "metadata": {"chunk_scope": "turn_window"},
                    "final_score": 1.0,
                    "slot_rerank_multiplier": 1.5,
                    "slot_rerank": {
                        "expected_slot": "LOCATION",
                        "observed_slots": ["LOCATION"],
                        "multiplier": 1.5,
                        "reason": "location_match",
                    },
                },
                {
                    "page_id": "page_time",
                    "citation_id": "page_time",
                    "user_input": "Scott Derrickson biography",
                    "summary": "Birth date",
                    "raw_content": "Scott Derrickson biography note for July 16, 1966.",
                    "metadata": {"chunk_scope": "turn_window"},
                    "final_score": 1.35,
                    "slot_rerank_multiplier": 0.7,
                    "slot_rerank": {
                        "expected_slot": "LOCATION",
                        "observed_slots": ["TIME"],
                        "multiplier": 0.7,
                        "reason": "time_only_distractor",
                    },
                },
            ],
            "hit_ids": ["page_location", "page_time"],
        }

        reranked = RUN_LOCOMO.rerank_retrieval_results("Where does Scott Derrickson live?", retrieval_results)
        pages = reranked["retrieved_pages"]

        self.assertEqual(pages[0]["citation_id"], "page_location")
        self.assertGreater(pages[0]["locomo_slot_adjustment"], 0.0)
        self.assertLess(pages[1]["locomo_slot_adjustment"], 0.0)

    def test_expanded_queries_add_semantic_hints_without_leaking_answer_tokens(self):
        queries = RUN_LOCOMO.build_expanded_queries("What are Jolene's favorite books?")
        joined = " || ".join(queries)

        self.assertTrue(queries)
        self.assertNotIn("Sapiens", joined)
        self.assertNotIn("Avalanche", joined)
        self.assertNotIn("Neal Stephenson", joined)

    def test_country_expanded_queries_do_not_leak_destination_names(self):
        queries = RUN_LOCOMO.build_expanded_queries("Which countries has Deborah traveled to?")
        joined = " || ".join(queries)

        self.assertTrue(queries)
        self.assertNotIn("Bali", joined)
        self.assertNotIn("Rio de Janeiro", joined)


class LocomoReportMetricsTest(unittest.TestCase):
    def test_summarize_salvage_metrics_counts_hits_and_question_rate(self):
        summary = RUN_LOCOMO.summarize_salvage_metrics(
            [
                {"salvage_count": 2, "source_stats": {"FORGET_COLD": 3}},
                {"source_stats": {"FORGET_COLD": 1}},
                {"salvage_count": 0, "source_stats": {"FORGET_COLD": 0}},
            ]
        )

        self.assertEqual(summary["forget_pool_salvage_count"], 3)
        self.assertEqual(summary["questions_with_forget_pool_salvage"], 2)
        self.assertEqual(summary["forget_pool_salvage_rate"], 0.6667)
        self.assertEqual(summary["avg_forget_pool_salvage_count"], 1.0)

    def test_merge_retrieval_results_keeps_higher_scored_duplicate_page(self):
        primary = {
            "retrieved_pages": [
                {
                    "page_id": "dup_page",
                    "citation_id": "dup_page",
                    "raw_content": "primary",
                    "final_score": 0.9,
                }
            ],
            "hit_ids": ["dup_page"],
            "source_stats": {"LTM": 1},
            "salvage_count": 0,
        }
        extra = {
            "retrieved_pages": [
                {
                    "page_id": "dup_page",
                    "citation_id": "dup_page",
                    "raw_content": "expanded",
                    "final_score": 1.6,
                    "_source": "GRAPH_1HOP",
                }
            ],
            "hit_ids": ["dup_page"],
            "source_stats": {"GRAPH_1HOP": 1},
            "salvage_count": 1,
        }

        merged = RUN_LOCOMO.merge_retrieval_results(primary, [extra])
        self.assertEqual(len(merged["retrieved_pages"]), 1)
        self.assertEqual(merged["retrieved_pages"][0]["raw_content"], "expanded")
        self.assertEqual(merged["source_stats"]["LTM"], 1)
        self.assertEqual(merged["source_stats"]["GRAPH_1HOP"], 1)
        self.assertEqual(merged["salvage_count"], 1)


class _StubAnswerClient:
    def __init__(self, response="park, beach", planner_response=None, verifier_response=None):
        self.response = response
        self.planner_response = planner_response
        self.verifier_response = verifier_response
        self.models = []

    def chat_completion(self, **kwargs):
        self.models.append(kwargs.get("model"))
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        if "retrieval planner" in prompt.lower() and self.planner_response is not None:
            return self.planner_response
        if "verifier evidence:" in prompt.lower() and self.verifier_response is not None:
            return self.verifier_response
        return self.response


class _StubAnswerMemo:
    def __init__(
        self,
        response="park, beach",
        agent_response="Deborah practices yoga at the park and the beach.",
        planner_response=None,
        verifier_response=None,
    ):
        self.client = _StubAnswerClient(
            response=response,
            planner_response=planner_response,
            verifier_response=verifier_response,
        )
        self.planner_client = self.client
        self.llm_model = "stub-model"
        self.planner_model = "stub-model"
        self.queries = []
        self.agent_response = agent_response

    def retrieve_context(self, query, max_results=10, token_budget=2200, trace_context=None):
        self.queries.append(query)
        return {
            "retrieved_pages": [
                {
                    "page_id": "page_yoga",
                    "citation_id": "page_yoga",
                    "user_input": "Deborah yoga notes",
                    "agent_response": self.agent_response,
                    "raw_content": self.agent_response,
                    "summary": "Yoga locations",
                    "metadata": {"chunk_scope": "turn_window"},
                    "score": 0.9,
                }
            ],
            "hit_ids": ["page_yoga"],
            "salvage_count": 0,
            "source_stats": {"STM": 1},
            "context_summary": "Yoga locations",
        }

    def get_response(self, *args, **kwargs):
        raise AssertionError("answer_question should use retrieve_context, not get_response")


class LocomoAnswerQuestionFlowTest(unittest.TestCase):
    def test_question_expects_list_detects_names_and_games_variants(self):
        self.assertTrue(RUN_LOCOMO.question_expects_list("What are the names of Jolene's snakes?"))
        self.assertTrue(RUN_LOCOMO.question_expects_list("Which games have Jolene and her partner played together?"))

    def test_sanitize_model_answer_output_strips_think_wrapper(self):
        cleaned = RUN_LOCOMO.sanitize_model_answer_output("<think>analyze</think>\nFinal Answer: 8 years old")

        self.assertEqual(cleaned, "8 years old")

    def test_plan_retrieval_queries_uses_planner_model_override(self):
        memo = _StubAnswerMemo(
            planner_response='{"need":"find yoga locations","answer_mode":"list","global_dependencies":[],"local_dependencies":[],"queries":[]}'
        )
        memo.planner_model = "small-planner"

        RUN_LOCOMO.plan_retrieval_queries(memo, "Which locations does Deborah practice her yoga at?")

        self.assertEqual(memo.client.models[0], "small-planner")

    def test_planner_dependency_categories_synthesize_queries(self):
        memo = _StubAnswerMemo(
            planner_response=(
                '{"need":"resolve marriage status","answer_mode":"binary",'
                '"global_dependencies":["stable relationship and family context"],'
                '"local_dependencies":["married husband wife direct status evidence"],'
                '"queries":[]}'
            )
        )

        _, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "Is Deborah married?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
        )

        self.assertTrue(retrieval_results["retrieval_plan"]["global_queries"])
        self.assertTrue(retrieval_results["retrieval_plan"]["local_queries"])

    def test_planner_queries_are_added_before_heuristic_expansion(self):
        memo = _StubAnswerMemo(
            planner_response=(
                '{"need":"find yoga locations","answer_mode":"list",'
                '"global_dependencies":["Deborah profile and recurring yoga context"],'
                '"local_dependencies":["yoga practice locations"],'
                '"queries":["Deborah yoga places practice locations"]}'
            )
        )

        _, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "Which locations does Deborah practice her yoga at?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=40,
            evidence_pages=2,
        )

        self.assertGreaterEqual(len(memo.queries), 3)
        self.assertIn("Deborah yoga places practice locations", memo.queries)
        self.assertIn("Deborah Deborah profile and recurring yoga context", memo.queries)
        self.assertEqual(retrieval_results["retrieval_plan"]["answer_mode"], "list")
        self.assertEqual(
            retrieval_results["retrieval_plan"]["global_dependencies"],
            ["Deborah profile and recurring yoga context"],
        )
        self.assertEqual(
            retrieval_results["retrieval_plan"]["local_dependencies"],
            ["yoga practice locations"],
        )

    def test_answer_question_uses_lightweight_retrieval_for_expanded_queries(self):
        memo = _StubAnswerMemo()

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "Which locations does Deborah practice her yoga at?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=40,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "park, beach")
        self.assertGreaterEqual(len(memo.queries), 2)
        self.assertEqual(memo.queries[0], "Which locations does Deborah practice her yoga at?")
        self.assertEqual(retrieval_results["citations"], ["page_yoga"])

    def test_answer_aware_citation_selection_prefers_event_specific_temporal_evidence(self):
        retrieval_results = {
            "retrieved_pages": [
                {
                    "page_id": "page_session",
                    "citation_id": "page_session",
                    "raw_content": "Sample: conv-48 Date: 30 August, 2023 Session: 12",
                    "summary": "Session metadata",
                    "locomo_rerank_score": 5.0,
                },
                {
                    "page_id": "page_event",
                    "citation_id": "page_event",
                    "raw_content": "Jolene's mom gifted her a pendant in 2010 during a difficult year.",
                    "summary": "Pendant memory",
                    "locomo_rerank_score": 3.0,
                },
            ]
        }

        citations = RUN_LOCOMO.select_top_citations(
            "When did Jolene's mom gift her a pendant?",
            retrieval_results,
            answer="in 2010",
            max_citations=2,
        )

        self.assertEqual(citations[0], "page_event")

    def test_binary_question_uses_inference_path_and_returns_yes(self):
        memo = _StubAnswerMemo(
            response="No",
            agent_response="Deborah said the time she spends with her husband helps them bond and make memories.",
        )

        prediction, _ = RUN_LOCOMO.answer_question(
            memo,
            "Is Deborah married?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "Yes")

    def test_answer_question_recovers_blank_answer_from_evidence(self):
        memo = _StubAnswerMemo(
            response="",
            agent_response="Deborah: Max, Deborah's mother's cat, is 8 years old. (D22:27)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "How old is Max?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "8 years old")
        self.assertEqual(retrieval_results["citations"], ["page_yoga"])

    def test_shape_adaptive_answer_verifier_leaves_unknown_when_disabled(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            agent_response="Deborah: Max, Deborah's mother's cat, is 8 years old. (D22:27)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "How old is Max?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "Unknown")
        self.assertFalse(retrieval_results["answer_verifier"]["applied"])
        self.assertEqual(retrieval_results["answer_verifier"]["reason"], "disabled")

    def test_shape_adaptive_answer_verifier_recovers_unknown_age_answer(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            agent_response="Deborah: Max, Deborah's mother's cat, is 8 years old. (D22:27)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "How old is Max?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
            enable_shape_adaptive_answer_verifier=True,
        )

        self.assertEqual(prediction, "8 years old")
        self.assertTrue(retrieval_results["answer_verifier"]["applied"])
        self.assertEqual(retrieval_results["answer_verifier"]["question_shape"], "age_lookup")

    def test_shape_adaptive_answer_verifier_uses_focused_verifier_for_entity_questions(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            verifier_response="Rachel",
            agent_response="Joey: Paolo went out with Rachel in November 1994. (D22:27)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "Who did Paolo go out with in November 1994?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
            enable_shape_adaptive_answer_verifier=True,
        )

        self.assertEqual(prediction, "Rachel")
        self.assertTrue(retrieval_results["answer_verifier"]["applied"])
        self.assertIn("focused_verifier", retrieval_results["answer_verifier"]["sources"])

    def test_shape_adaptive_answer_verifier_rejects_unsupported_focused_candidate(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            verifier_response="Monica",
            agent_response="Joey: Paolo went out with Rachel in November 1994. (D22:27)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "Who did Paolo go out with in November 1994?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
            enable_shape_adaptive_answer_verifier=True,
        )

        self.assertEqual(prediction, "Unknown")
        self.assertFalse(retrieval_results["answer_verifier"]["applied"])

    def test_shape_adaptive_answer_verifier_keeps_supported_fact_answer(self):
        memo = _StubAnswerMemo(
            response="comforted",
            verifier_response="Comforting.",
            agent_response="Jolene says spending time with Seraphim is comforting and calming.",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "How does Jolene feel when spending time with Seraphim?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
            enable_shape_adaptive_answer_verifier=True,
        )

        self.assertEqual(prediction, "comforted")
        self.assertFalse(retrieval_results["answer_verifier"]["applied"])
        self.assertEqual(
            retrieval_results["answer_verifier"]["reason"],
            "non_unknown_override_disabled_for_shape",
        )

    def test_shape_adaptive_answer_verifier_preserves_fact_recovery_path(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            verifier_response="electricity engineering project",
            agent_response="Jolene finished an electricity engineering project last week. (D1:2)",
        )

        prediction, retrieval_results = RUN_LOCOMO.answer_question(
            memo,
            "What kind of project was Jolene working on in the beginning of January 2023?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=20,
            evidence_pages=2,
            enable_shape_adaptive_answer_verifier=True,
        )

        self.assertEqual(prediction, "electricity engineering project")
        self.assertFalse(retrieval_results["answer_verifier"]["applied"])
        self.assertEqual(retrieval_results["answer_verifier"]["question_shape"], "fact_lookup")

    def test_answer_question_prefers_multifact_heuristic_for_supported_game_list(self):
        memo = _StubAnswerMemo(
            response="It takes two",
            agent_response=(
                "Jolene recommended Zelda BOTW for Switch, Animal Crossing: New Horizons, "
                "and Overcooked 2 for Deborah."
            ),
        )

        prediction, _ = RUN_LOCOMO.answer_question(
            memo,
            "What games does Jolene recommend for Deborah?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=40,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "Zelda BOTW for Switch, Animal Crossing: New Horizons, Overcooked 2")

    def test_answer_question_uses_generalized_susie_origin_rule(self):
        memo = _StubAnswerMemo(
            response="Unknown",
            agent_response="I adopted Susie two years ago when I was feeling lonely and needed companionship.",
        )

        prediction, _ = RUN_LOCOMO.answer_question(
            memo,
            "How did Deborah come to have her pet, Susie?",
            max_results=3,
            token_budget=1200,
            answer_max_tokens=40,
            evidence_pages=2,
        )

        self.assertEqual(prediction, "She adopted her two years ago when feeling lonely")

    def test_apply_planner_channel_quota_preserves_global_page_for_inference_questions(self):
        retrieval_results = {
            "retrieved_pages": [
                {"citation_id": "local_1", "locomo_rerank_score": 9.0, "_planner_query_tiers": ["local"]},
                {"citation_id": "local_2", "locomo_rerank_score": 8.5, "_planner_query_tiers": ["local"]},
                {"citation_id": "global_1", "locomo_rerank_score": 8.0, "_planner_query_tiers": ["global"]},
                {"citation_id": "other_1", "locomo_rerank_score": 7.0, "_planner_query_tiers": []},
            ],
            "hit_ids": ["local_1", "local_2", "global_1", "other_1"],
        }

        trimmed = RUN_LOCOMO.apply_planner_channel_quota(
            retrieval_results,
            max_results=3,
            inference_mode=True,
            binary_mode=False,
            list_mode=False,
            temporal_mode=False,
        )

        kept = [page["citation_id"] for page in trimmed["retrieved_pages"]]
        self.assertIn("global_1", kept)
        self.assertIn("local_1", kept)

    def test_apply_planner_channel_quota_prefers_local_pages_for_fact_questions(self):
        retrieval_results = {
            "retrieved_pages": [
                {"citation_id": "local_1", "locomo_rerank_score": 9.0, "_planner_query_tiers": ["local"]},
                {"citation_id": "local_2", "locomo_rerank_score": 8.5, "_planner_query_tiers": ["local"]},
                {"citation_id": "global_1", "locomo_rerank_score": 8.0, "_planner_query_tiers": ["global"]},
                {"citation_id": "global_2", "locomo_rerank_score": 7.5, "_planner_query_tiers": ["global"]},
            ],
            "hit_ids": ["local_1", "local_2", "global_1", "global_2"],
        }

        trimmed = RUN_LOCOMO.apply_planner_channel_quota(
            retrieval_results,
            max_results=3,
            inference_mode=False,
            binary_mode=False,
            list_mode=True,
            temporal_mode=False,
        )

        kept = [page["citation_id"] for page in trimmed["retrieved_pages"]]
        self.assertIn("local_1", kept)
        self.assertIn("local_2", kept)
        self.assertIn("global_1", kept)

    def test_apply_slot_filtering_soft_penalizes_obvious_time_distractor_for_location_question(self):
        retrieval_results = {
            "retrieved_pages": [
                {
                    "citation_id": "page_location",
                    "locomo_rerank_score": 5.0,
                    "slot_rerank": {
                        "expected_slot": "LOCATION",
                        "observed_slots": ["LOCATION"],
                        "reason": "location_match",
                    },
                    "_locomo_query_overlap": 1,
                },
                {
                    "citation_id": "page_time",
                    "locomo_rerank_score": 4.9,
                    "slot_rerank": {
                        "expected_slot": "LOCATION",
                        "observed_slots": ["TIME"],
                        "reason": "time_only_distractor",
                    },
                    "_locomo_query_overlap": 0,
                },
                {
                    "citation_id": "page_other",
                    "locomo_rerank_score": 4.8,
                    "slot_rerank": {
                        "expected_slot": "LOCATION",
                        "observed_slots": [],
                        "reason": "slot_neutral",
                    },
                    "_locomo_query_overlap": 0,
                },
            ],
            "hit_ids": ["page_location", "page_time", "page_other"],
        }

        filtered = RUN_LOCOMO.apply_slot_filtering("Where does Scott Derrickson live?", retrieval_results)

        kept = [page["citation_id"] for page in filtered["retrieved_pages"]]
        self.assertIn("page_location", kept)
        self.assertIn("page_time", kept)
        self.assertTrue(filtered["slot_filtering"]["applied"])
        page_lookup = {page["citation_id"]: page for page in filtered["retrieved_pages"]}
        self.assertLess(page_lookup["page_time"]["locomo_rerank_score"], page_lookup["page_other"]["locomo_rerank_score"])
        self.assertLess(page_lookup["page_time"]["slot_filter_adjustment"], 0.0)


if __name__ == "__main__":
    unittest.main()
