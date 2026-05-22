import unittest

from memoryos.answer_routing import (
    build_answer_mode_instruction,
    normalize_generated_answer,
    resolve_effective_answer_format,
    resolve_effective_answer_mode,
)
from memoryos.query_routing import infer_task_routing_plan, resolve_effective_retrieval_mode


class AnswerRoutingTest(unittest.TestCase):
    def test_resolve_effective_answer_mode_uses_primary_mode_when_confident(self):
        plan = infer_task_routing_plan(
            "Who booked the restaurant?",
            task_hint="dialogue_qa",
        )

        self.assertEqual(resolve_effective_answer_mode(plan), "extractive")

    def test_resolve_effective_answer_mode_falls_back_when_low_confidence(self):
        plan = infer_task_routing_plan(
            "Talk about this.",
        )

        self.assertEqual(resolve_effective_answer_mode(plan), "abstractive")
        self.assertEqual(resolve_effective_retrieval_mode(plan), "hybrid")

    def test_build_answer_mode_instruction_covers_main_modes(self):
        self.assertIn("verbatim", build_answer_mode_instruction("extractive"))
        self.assertIn("yes or no", build_answer_mode_instruction("binary"))
        self.assertIn("comma-separated list", build_answer_mode_instruction("list"))
        self.assertIn("structured answer", build_answer_mode_instruction("structured"))
        self.assertIn("single number", build_answer_mode_instruction("structured", answer_format="count"))
        self.assertIn("choice label", build_answer_mode_instruction("structured", answer_format="choice"))
        self.assertIn("concise", build_answer_mode_instruction("abstractive", task_hint="meeting_summary"))
        self.assertIn("broad background", build_answer_mode_instruction("abstractive", task_hint="hybrid"))

    def test_normalize_generated_answer_by_mode(self):
        self.assertEqual(normalize_generated_answer("Answer: Yes, she is.", "binary"), "yes")
        self.assertEqual(
            normalize_generated_answer("- Monica\n- Rachel\n- Monica", "list"),
            "Monica, Rachel",
        )
        self.assertEqual(
            normalize_generated_answer('{"choice":"A","count":3}', "structured"),
            "choice=A; count=3",
        )
        self.assertEqual(
            normalize_generated_answer("Final Answer: \"Monica\"", "extractive"),
            "Monica",
        )
        self.assertEqual(
            normalize_generated_answer("Option C: lunch", "structured", answer_format="choice"),
            "C",
        )
        self.assertEqual(
            normalize_generated_answer("count=12", "structured", answer_format="count"),
            "12",
        )
        self.assertEqual(
            normalize_generated_answer("1. Monica\n2. Rachel", "structured", answer_format="rank"),
            "Monica, Rachel",
        )

    def test_resolve_effective_answer_format_uses_plan_payload(self):
        plan = infer_task_routing_plan(
            "How many restaurants were discussed?",
            query_type="specific",
            task_hint="meeting_summary",
        )

        self.assertEqual(resolve_effective_answer_format(plan), "count")


if __name__ == "__main__":
    unittest.main()
