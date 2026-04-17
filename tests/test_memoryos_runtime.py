import unittest

from memoryos.memoryos import Memoryos


class _StubRetriever:
    def retrieve_context(self, query, top_k=10, token_budget=2200, trace_context=None):
        return {
            "retrieved_pages": [{"citation_id": "doc_ltm", "page_id": "doc_ltm"}],
            "hit_ids": ["doc_ltm"],
            "salvage_count": 0,
            "source_stats": {"LTM": 1},
        }

    def get_context_summary(self, retrieval_results):
        return "### Memory Context"


class _StubLifecycle:
    def __init__(self):
        self.post_inference_calls = []
        self.pointer_compress_calls = []
        self.soft_decay_calls = 0

    def build_interaction_reflection(self, query, final_answer, retrieval_results, base_feedback_score=0.0):
        return {"feedback_score": 0.8}

    def schedule_post_inference_memory_flow(self, memoryos, page_ids, feedback_score=1.0, reflection=None, trace_context=None):
        self.post_inference_calls.append(
            {
                "page_ids": list(page_ids),
                "feedback_score": feedback_score,
                "reflection": dict(reflection or {}),
            }
        )

    def schedule_pointer_compress_candidates(self, node_ids, min_raw_chars=900, min_temperature=35.0, trace_context=None):
        self.pointer_compress_calls.append(list(node_ids))

    def schedule_soft_decay_memory(self, half_life_days=7):
        self.soft_decay_calls += 1


class MemoryosRuntimeFlowTest(unittest.TestCase):
    def test_get_response_auto_schedules_pointer_compress_candidates(self):
        memo = Memoryos.__new__(Memoryos)
        memo.retriever = _StubRetriever()
        memo.lifecycle = _StubLifecycle()
        memo.current_session_hit_ids = []
        memo._retrieval_count = 0
        memo._generate_answer_bundle = lambda query, retrieval_results, trace_context=None: {
            "final_answer": "Los Angeles",
            "citations": ["doc_ltm"],
            "thought_process": "Grounded in retrieved memory context.",
        }
        memo._derive_feedback_score = lambda final_answer, citations, hit_ids: 0.7

        result = memo.get_response("Where does the user live?")

        self.assertEqual(result["final_answer"], "Los Angeles")
        self.assertEqual(memo.current_session_hit_ids, ["doc_ltm"])
        self.assertEqual(len(memo.lifecycle.post_inference_calls), 1)
        self.assertEqual(memo.lifecycle.pointer_compress_calls, [["doc_ltm"]])


if __name__ == "__main__":
    unittest.main()
