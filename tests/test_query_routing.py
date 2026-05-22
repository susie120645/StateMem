import unittest

from memoryos.query_routing import (
    build_retrieval_trace_context,
    infer_answer_mode,
    infer_answer_format,
    infer_retrieval_query_kind,
    infer_task_routing_plan,
)


class QueryRoutingTest(unittest.TestCase):
    def test_infers_global_for_summary_queries(self):
        self.assertEqual(
            infer_retrieval_query_kind("Summarize the overall meeting decisions and action items"),
            "global",
        )

    def test_infers_local_for_detail_questions(self):
        self.assertEqual(
            infer_retrieval_query_kind("When did Jolene buy her pet snake?", task_hint="dialogue_qa"),
            "local",
        )

    def test_query_type_overrides_text_heuristics(self):
        self.assertEqual(
            infer_retrieval_query_kind("What happened next?", query_type="general", task_hint="meeting_summary"),
            "global",
        )

    def test_build_trace_context_carries_metadata(self):
        trace_context = build_retrieval_trace_context(
            "What was the main point?",
            query_type="specific",
            task_hint="meeting_summary",
            extra={"benchmark": "qmsum"},
        )

        self.assertEqual(trace_context["query_kind"], "local")
        self.assertEqual(trace_context["query_type"], "specific")
        self.assertEqual(trace_context["task_hint"], "meeting_summary")
        self.assertEqual(trace_context["benchmark"], "qmsum")

    def test_infers_answer_modes(self):
        self.assertEqual(
            infer_answer_mode("Who booked the restaurant?", task_hint="dialogue_qa"),
            "extractive",
        )
        self.assertEqual(
            infer_answer_mode("Summarize the overall meeting decisions", task_hint="meeting_summary"),
            "abstractive",
        )
        self.assertEqual(
            infer_answer_mode("Is Deborah married?"),
            "binary",
        )
        self.assertEqual(
            infer_answer_format("How many action items were assigned?", answer_mode="structured", task_hint="meeting_summary"),
            "count",
        )
        self.assertEqual(
            infer_answer_format("Choose the correct option for the venue.", answer_mode="structured"),
            "choice",
        )
        self.assertEqual(
            infer_answer_format(
                "Who booked the restaurant?\n(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey\n(E) I don't know.",
                answer_mode="structured",
            ),
            "choice",
        )

    def test_builds_task_routing_plan_with_fallback(self):
        plan = infer_task_routing_plan(
            "Summarize the overall meeting decisions and action items",
            query_type="general",
            task_hint="meeting_summary",
            extra_signals={"benchmark": "qmsum"},
        )

        self.assertEqual(plan.retrieval_mode, "global")
        self.assertEqual(plan.answer_mode, "abstractive")
        self.assertEqual(plan.answer_format, "summary")
        self.assertGreaterEqual(plan.confidence, 0.8)
        self.assertEqual(plan.fallback.retrieval_mode, "hybrid")
        self.assertEqual(plan.signals["benchmark"], "qmsum")
        self.assertEqual(plan.signals["query_type"], "general")

    def test_trace_context_includes_task_routing_plan(self):
        trace_context = build_retrieval_trace_context(
            "Who booked the restaurant?",
            task_hint="dialogue_qa",
            extra={"benchmark": "friendsqa"},
        )

        self.assertEqual(trace_context["retrieval_mode"], "local")
        self.assertEqual(trace_context["answer_mode"], "extractive")
        self.assertEqual(trace_context["answer_format"], "span")
        self.assertIn("task_routing_plan", trace_context)
        self.assertEqual(trace_context["task_routing_plan"]["fallback"]["answer_mode"], "abstractive")
        self.assertEqual(trace_context["task_routing_plan"]["answer_format"], "span")


if __name__ == "__main__":
    unittest.main()
