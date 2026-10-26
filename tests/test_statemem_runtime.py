import unittest

from statemem.statemem import statemem


class _StubRetriever:
    def __init__(self):
        self.calls = []

    def retrieve_context(self, query, top_k=10, token_budget=2200, trace_context=None):
        query_kind = (trace_context or {}).get("query_kind")
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "token_budget": token_budget,
                "trace_context": dict(trace_context or {}),
            }
        )
        if query_kind == "global":
            return {
                "retrieved_pages": [{"citation_id": "doc_global", "page_id": "doc_global", "final_score": 0.8}],
                "hit_ids": ["doc_global"],
                "salvage_count": 0,
                "source_stats": {"LTM": 1},
                "retrieval_profile": {"query_kind": "global"},
            }
        if query_kind == "local":
            return {
                "retrieved_pages": [{"citation_id": "doc_local", "page_id": "doc_local", "final_score": 0.9}],
                "hit_ids": ["doc_local"],
                "salvage_count": 0,
                "source_stats": {"LTM": 1},
                "retrieval_profile": {"query_kind": "local"},
            }
        return {
            "retrieved_pages": [{"citation_id": "doc_ltm", "page_id": "doc_ltm"}],
            "hit_ids": ["doc_ltm"],
            "salvage_count": 0,
            "source_stats": {"LTM": 1},
            "retrieval_profile": {"query_kind": "default"},
        }

    def get_context_summary(self, retrieval_results):
        return "### Memory Context"

    def _apply_token_budget(self, pages, token_budget):
        return list(pages)


class _StubLifecycle:
    def __init__(self):
        self.post_inference_calls = []
        self.pointer_compress_calls = []
        self.soft_decay_calls = 0

    def build_interaction_reflection(self, query, final_answer, retrieval_results, base_feedback_score=0.0):
        return {"feedback_score": 0.8}

    def schedule_post_inference_memory_flow(self, statemem, page_ids, feedback_score=1.0, reflection=None, trace_context=None):
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


class _StubClient:
    def __init__(self, raw_response: str):
        self.raw_response = raw_response

    def chat_completion(self, *args, **kwargs):
        return self.raw_response


class _StubSequenceClient:
    def __init__(self, raw_responses):
        self.raw_responses = list(raw_responses)
        self.calls = 0

    def chat_completion(self, *args, **kwargs):
        if self.calls >= len(self.raw_responses):
            return self.raw_responses[-1]
        value = self.raw_responses[self.calls]
        self.calls += 1
        return value


class MemoryosRuntimeFlowTest(unittest.TestCase):
    def test_get_response_auto_schedules_pointer_compress_candidates(self):
        memo = statemem.__new__(statemem)
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

    def test_retrieve_context_builds_default_trace_context(self):
        memo = statemem.__new__(statemem)
        memo.retriever = _StubRetriever()

        result = memo.retrieve_context(
            "Summarize the overall meeting decisions.",
            max_results=4,
            token_budget=1800,
            query_type="general",
            task_hint="meeting_summary",
            benchmark="qmsum",
        )

        self.assertEqual(result["context_summary"], "### Memory Context")
        self.assertEqual(memo.retriever.calls[0]["top_k"], 4)
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["query_kind"], "global")
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["task_hint"], "meeting_summary")
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["benchmark"], "qmsum")
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["task_routing_plan"]["answer_mode"], "abstractive")
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["task_routing_plan"]["answer_format"], "summary")

    def test_build_retrieval_trace_context_preserves_plain_trace_without_routing_hints(self):
        memo = statemem.__new__(statemem)

        trace_context = memo.build_retrieval_trace_context(
            "When did Jolene buy her pet snake?",
            trace_context={"sample_id": "conv-48"},
        )

        self.assertEqual(trace_context, {"sample_id": "conv-48"})

    def test_build_task_routing_plan_returns_dataclass(self):
        memo = statemem.__new__(statemem)

        plan = memo.build_task_routing_plan(
            "Who booked the restaurant?",
            task_hint="dialogue_qa",
        )

        self.assertEqual(plan.retrieval_mode, "local")
        self.assertEqual(plan.answer_mode, "extractive")
        self.assertEqual(plan.answer_format, "span")
        self.assertGreater(plan.confidence, 0.7)

    def test_retrieve_context_executes_hybrid_retrieval_path(self):
        memo = statemem.__new__(statemem)
        memo.retriever = _StubRetriever()

        result = memo.retrieve_context(
            "Summarize the overall meeting background and exact decision details",
            max_results=4,
            token_budget=1800,
            preferred_query_kind="hybrid",
            benchmark="manual",
        )

        self.assertEqual(result["retrieval_profile"]["query_kind"], "hybrid")
        self.assertEqual(len(memo.retriever.calls), 2)
        self.assertEqual(memo.retriever.calls[0]["trace_context"]["query_kind"], "global")
        self.assertEqual(memo.retriever.calls[1]["trace_context"]["query_kind"], "local")
        self.assertEqual(result["hit_ids"], ["doc_local", "doc_global"])

    def test_generate_answer_bundle_normalizes_binary_answers(self):
        memo = statemem.__new__(statemem)
        memo.client = _StubClient(
            '{"thought_process":"grounded","final_answer":"Answer: Yes, she is.","citations":["doc_ltm"]}'
        )
        memo.llm_model = "stub"
        memo.lifecycle = _StubLifecycle()

        bundle = memo._generate_answer_bundle(
            "Is Deborah married?",
            {
                "retrieved_pages": [
                    {
                        "citation_id": "doc_ltm",
                        "page_id": "doc_ltm",
                        "raw_content": "Deborah is married.",
                        "summary": "Marriage note",
                    }
                ]
            },
            trace_context={
                "task_routing_plan": {
                    "answer_mode": "binary",
                    "confidence": 0.9,
                    "fallback": {"answer_mode": "abstractive", "retrieval_mode": "hybrid"},
                }
            },
        )

        self.assertEqual(bundle["final_answer"], "yes")

    def test_generate_answer_bundle_normalizes_structured_count_answers(self):
        memo = statemem.__new__(statemem)
        memo.client = _StubClient(
            '{"thought_process":"grounded","final_answer":"count=3","citations":["doc_ltm"]}'
        )
        memo.llm_model = "stub"
        memo.lifecycle = _StubLifecycle()

        bundle = memo._generate_answer_bundle(
            "How many action items were assigned?",
            {
                "retrieved_pages": [
                    {
                        "citation_id": "doc_ltm",
                        "page_id": "doc_ltm",
                        "raw_content": "Three action items were assigned.",
                        "summary": "Meeting note",
                    }
                ]
            },
            trace_context={
                "task_routing_plan": {
                    "answer_mode": "structured",
                    "answer_format": "count",
                    "confidence": 0.9,
                    "fallback": {"answer_mode": "abstractive", "retrieval_mode": "hybrid"},
                }
            },
        )

        self.assertEqual(bundle["final_answer"], "3")

    def test_generate_answer_bundle_retries_after_abstention_under_strong_evidence(self):
        memo = statemem.__new__(statemem)
        memo.client = _StubSequenceClient(
            [
                '{"thought_process":"grounded","final_answer":"I don\'t know.","citations":["doc_ltm"]}',
                '{"thought_process":"grounded","final_answer":"Monica","citations":["doc_ltm"]}',
            ]
        )
        memo.llm_model = "stub"
        memo.lifecycle = _StubLifecycle()

        bundle = memo._generate_answer_bundle(
            "Who booked the restaurant?",
            {
                "retrieved_pages": [
                    {
                        "citation_id": "doc_ltm",
                        "page_id": "doc_ltm",
                        "raw_content": "Monica booked the restaurant.",
                        "summary": "Restaurant note",
                        "final_score": 0.95,
                    },
                    {
                        "citation_id": "doc_ltm_2",
                        "page_id": "doc_ltm_2",
                        "raw_content": "Rachel thanked Monica later.",
                        "summary": "Follow-up note",
                        "final_score": 0.78,
                    },
                ]
            },
            trace_context={
                "task_hint": "dialogue_qa",
                "routing_confidence": 0.9,
                "task_routing_plan": {
                    "retrieval_mode": "local",
                    "answer_mode": "extractive",
                    "answer_format": "span",
                    "confidence": 0.9,
                    "fallback": {"answer_mode": "abstractive", "retrieval_mode": "hybrid"},
                },
            },
        )

        self.assertEqual(bundle["final_answer"], "Monica")
        self.assertEqual(bundle["adaptive_retry_count"], 1)
        self.assertEqual(bundle["adaptive_answer_action"]["abstain_policy"], "discourage")

    def test_generate_answer_bundle_uses_generic_choice_head(self):
        memo = statemem.__new__(statemem)
        memo.client = _StubClient(
            '{"best_choice":"A","rationale":"The retrieved note directly supports Monica.","citations":["doc_ltm"]}'
        )
        memo.llm_model = "stub"
        memo.lifecycle = _StubLifecycle()
        memo.enable_choice_answer_head = True

        bundle = memo._generate_answer_bundle(
            "Who booked the restaurant?\n(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey",
            {
                "retrieved_pages": [
                    {
                        "citation_id": "doc_ltm",
                        "page_id": "doc_ltm",
                        "raw_content": "Monica booked the restaurant.",
                        "summary": "Restaurant note",
                        "final_score": 0.95,
                    }
                ]
            },
            trace_context={
                "task_hint": "dialogue_qa",
                "routing_confidence": 0.9,
                "task_routing_plan": {
                    "retrieval_mode": "local",
                    "answer_mode": "structured",
                    "answer_format": "choice",
                    "confidence": 0.9,
                    "fallback": {"answer_mode": "abstractive", "retrieval_mode": "hybrid"},
                },
            },
        )

        self.assertEqual(bundle["final_answer"], "(A)")
        self.assertEqual(bundle["citations"], ["doc_ltm"])
        self.assertEqual(bundle["adaptive_answer_overrides"]["effective_answer_format"], "choice")
        self.assertIn("canonical_evidence", bundle)
        self.assertEqual(len(bundle["canonical_evidence"]["canonical_query"]["choice_options"]), 4)


if __name__ == "__main__":
    unittest.main()
