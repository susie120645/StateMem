import unittest

from statemem.adaptive_answer_skill import (
    AdaptiveAnswerAction,
    apply_adaptive_answer_action,
    build_adaptive_answer_state,
    select_adaptive_answer_action,
    should_retry_adaptive_answer,
)


class AdaptiveAnswerSkillTest(unittest.TestCase):
    def test_build_state_captures_generic_retrieval_signals(self):
        state = build_adaptive_answer_state(
            "Who booked the restaurant?",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.92},
                    {"citation_id": "doc2", "raw_content": "Rachel thanked Monica later.", "final_score": 0.71},
                ],
                "source_stats": {"LTM": 2},
            },
            trace_context={"query_type": "", "task_hint": "dialogue_qa", "routing_confidence": 0.9, "token_budget": 1800},
            retrieval_mode="local",
            answer_mode="extractive",
            answer_format="span",
        )

        self.assertEqual(state.retrieval_mode, "local")
        self.assertEqual(state.answer_mode, "extractive")
        self.assertEqual(state.answer_format, "span")
        self.assertEqual(state.retrieved_count, 2)
        self.assertGreater(state.top_score, 0.9)
        self.assertGreater(state.evidence_density, 0.4)
        self.assertFalse(state.uncertainty_flags["no_pages"])

    def test_select_action_stays_generic_but_discourages_abstention_when_evidence_is_strong(self):
        state = build_adaptive_answer_state(
            "Who booked the restaurant?",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.95},
                    {"citation_id": "doc2", "raw_content": "The others agreed.", "final_score": 0.76},
                ],
                "source_stats": {"LTM": 2},
            },
            trace_context={"task_hint": "dialogue_qa", "routing_confidence": 0.9},
            retrieval_mode="local",
            answer_mode="extractive",
            answer_format="span",
        )

        action = select_adaptive_answer_action(state)
        overrides = apply_adaptive_answer_action(state, action)

        self.assertEqual(action.answer_strategy, "extractive")
        self.assertEqual(action.abstain_policy, "discourage")
        self.assertEqual(action.retry_strategy, "retry_once")
        self.assertEqual(overrides["effective_answer_mode"], "extractive")
        self.assertEqual(overrides["effective_answer_format"], "span")

    def test_retry_rule_triggers_on_abstention_under_strong_evidence(self):
        state = build_adaptive_answer_state(
            "Who booked the restaurant?",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.93},
                    {"citation_id": "doc2", "raw_content": "Rachel thanked Monica later.", "final_score": 0.70},
                ],
                "source_stats": {"LTM": 2},
            },
            trace_context={"task_hint": "dialogue_qa", "routing_confidence": 0.9},
            retrieval_mode="local",
            answer_mode="extractive",
            answer_format="span",
        )
        action = select_adaptive_answer_action(state)

        self.assertTrue(
            should_retry_adaptive_answer(
                "I don't know.",
                state=state,
                action=action,
                answer_mode="extractive",
                answer_format="span",
            )
        )

    def test_untyped_structured_mode_stays_direct_but_preserves_strict_format(self):
        state = build_adaptive_answer_state(
            "Select the best supported option.",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.91},
                    {"citation_id": "doc2", "raw_content": "Rachel thanked Monica later.", "final_score": 0.83},
                ],
                "source_stats": {"LTM": 2},
            },
            trace_context={"task_hint": "dialogue_qa", "routing_confidence": 0.9},
            retrieval_mode="local",
            answer_mode="structured",
            answer_format="structured",
        )

        action = select_adaptive_answer_action(state)

        self.assertEqual(action.answer_strategy, "direct")
        self.assertEqual(action.format_policy, "strict")

    def test_typed_structured_mode_uses_structured_strategy(self):
        state = build_adaptive_answer_state(
            "Choose the best option.",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.91},
                    {"citation_id": "doc2", "raw_content": "Rachel thanked Monica later.", "final_score": 0.83},
                ],
                "source_stats": {"LTM": 2},
            },
            trace_context={"task_hint": "dialogue_qa", "routing_confidence": 0.9},
            retrieval_mode="local",
            answer_mode="structured",
            answer_format="choice",
        )

        action = select_adaptive_answer_action(state)
        overrides = apply_adaptive_answer_action(state, action)

        self.assertEqual(action.answer_strategy, "structured")
        self.assertEqual(overrides["effective_answer_format"], "choice")
        self.assertIn("single best-supported option", " ".join(overrides["instruction_fragments"]))

    def test_choice_retry_accepts_parseable_option_marker(self):
        state = build_adaptive_answer_state(
            "Choose the best option.",
            {
                "retrieved_pages": [
                    {"citation_id": "doc1", "raw_content": "Monica booked the restaurant.", "final_score": 0.91},
                ],
                "source_stats": {"LTM": 1},
            },
            trace_context={"task_hint": "dialogue_qa", "routing_confidence": 0.9},
            retrieval_mode="local",
            answer_mode="structured",
            answer_format="choice",
        )

        action = AdaptiveAnswerAction(
            answer_strategy="structured",
            retry_strategy="retry_with_stricter_format",
            abstain_policy="allow",
            format_policy="strict",
        )

        self.assertFalse(
            should_retry_adaptive_answer(
                "(A)",
                state=state,
                action=action,
                answer_mode="structured",
                answer_format="choice",
            )
        )


if __name__ == "__main__":
    unittest.main()
