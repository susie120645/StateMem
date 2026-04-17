import importlib.machinery
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "memoryos" / "run_locomo"
LOADER = importlib.machinery.SourceFileLoader("memoryos_run_locomo", str(MODULE_PATH))
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


class _StubAnswerClient:
    def chat_completion(self, **kwargs):
        return "park, beach"


class _StubAnswerMemo:
    def __init__(self):
        self.client = _StubAnswerClient()
        self.llm_model = "stub-model"
        self.queries = []

    def retrieve_context(self, query, max_results=10, token_budget=2200, trace_context=None):
        self.queries.append(query)
        return {
            "retrieved_pages": [
                {
                    "page_id": "page_yoga",
                    "citation_id": "page_yoga",
                    "user_input": "Deborah yoga notes",
                    "agent_response": "Deborah practices yoga at the park and the beach.",
                    "raw_content": "Deborah practices yoga at the park and the beach.",
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
        self.assertEqual(len(memo.queries), 2)
        self.assertEqual(retrieval_results["citations"], ["page_yoga"])


if __name__ == "__main__":
    unittest.main()
