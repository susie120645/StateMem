import unittest

from statemem.choice_answering import (
    answer_choice_with_context,
    parse_choice_question,
    maybe_override_abstaining_choice_answer,
    score_choice_support_with_context,
)


class _StubChoiceClient:
    def __init__(self, raw_response: str):
        self.raw_response = raw_response

    def chat_completion(self, *args, **kwargs):
        return self.raw_response


class ChoiceAnsweringTest(unittest.TestCase):
    def test_parse_choice_question_reads_labeled_options(self):
        parsed = parse_choice_question(
            "Who booked the restaurant?\n(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey"
        )

        self.assertEqual(parsed.stem, "Who booked the restaurant?")
        self.assertEqual([option.label for option in parsed.options], ["A", "B", "C", "D"])
        self.assertEqual(parsed.options[0].text, "Monica")

    def test_answer_choice_with_context_returns_original_marker(self):
        result = answer_choice_with_context(
            client=_StubChoiceClient('{"best_choice":"A","rationale":"The evidence directly states Monica booked it.","citations":["doc1"]}'),
            model="stub",
            question_text="Who booked the restaurant?\n(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey",
            pages=[
                {
                    "citation_id": "doc1",
                    "raw_content": "Monica booked the restaurant.",
                }
            ],
            allowed_citation_ids=["doc1"],
        )

        self.assertEqual(result["final_answer"], "(A)")
        self.assertEqual(result["citations"], ["doc1"])

    def test_prefer_supported_choice_can_override_abstain_option(self):
        result = answer_choice_with_context(
            client=_StubChoiceClient(
                '{"best_choice":"E","option_scores":{"A":0.62,"B":0.11,"C":0.09,"D":0.08,"E":0.64},"rationale":"A is supported but E is slightly safer."}'
            ),
            model="stub",
            question_text=(
                "Who booked the restaurant?\n"
                "(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey\n(E) I don't know. (None of the above)"
            ),
            pages=[
                {
                    "citation_id": "doc1",
                    "raw_content": "Monica booked the restaurant.",
                }
            ],
            prefer_supported_choice=True,
        )

        self.assertEqual(result["final_answer"], "(A)")

    def test_score_choice_support_returns_margin_and_abstain_flag(self):
        result = score_choice_support_with_context(
            client=_StubChoiceClient(
                '{"best_choice":"E","option_scores":{"A":0.51,"B":0.22,"C":0.12,"D":0.08,"E":0.67},"abstain_recommended":true,"citations":["doc1"]}'
            ),
            model="stub",
            question_text=(
                "Who booked the restaurant?\n"
                "(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey\n(E) I don't know. (None of the above)"
            ),
            pages=[
                {
                    "citation_id": "doc1",
                    "raw_content": "Monica booked the restaurant.",
                }
            ],
            allowed_citation_ids=["doc1"],
        )

        self.assertEqual(result["best_choice"], "(E)")
        self.assertTrue(result["abstain_recommended"])
        self.assertAlmostEqual(result["support_margin"], 0.16, places=4)
        self.assertEqual(result["citations"], ["doc1"])
        self.assertEqual(result["best_non_abstain_choice"], "(A)")
        self.assertIn("abstain_option_present", result["reason_codes"])
        self.assertIn("best_choice_abstain", result["reason_codes"])
        self.assertIn("abstain_recommended", result["reason_codes"])
        self.assertIn("support_source_single_family", result["reason_codes"])

    def test_score_choice_support_marks_non_abstain_alternative(self):
        result = score_choice_support_with_context(
            client=_StubChoiceClient(
                '{"best_choice":"E","option_scores":{"A":0.58,"B":0.22,"C":0.10,"D":0.08,"E":0.61},"abstain_recommended":true}'
            ),
            model="stub",
            question_text=(
                "Back in 1994, who booked the restaurant?\n"
                "(A) Monica\n(B) Rachel\n(C) Phoebe\n(D) Joey\n(E) I don't know. (None of the above)"
            ),
            pages=[
                {
                    "citation_id": "doc1",
                    "raw_content": "Monica booked the restaurant in 1994.",
                }
            ],
        )

        self.assertIn("non_abstain_alternative_available", result["reason_codes"])
        self.assertIn("time_anchor_in_question", result["reason_codes"])
        self.assertIn("best_non_abstain_lexical_overlap", result["reason_codes"])
        self.assertIn("question_time_anchor_aligned", result["reason_codes"])
        self.assertIn("question_entity_anchor_aligned", result["reason_codes"])

    def test_score_choice_support_marks_relation_alignment(self):
        result = score_choice_support_with_context(
            client=_StubChoiceClient(
                '{"best_choice":"A","option_scores":{"A":0.91,"B":0.05,"C":0.02,"D":0.01,"E":0.01},"abstain_recommended":false}'
            ),
            model="stub",
            question_text=(
                "Back in 1994, who was Monica's roommate?\n"
                "(A) Rachel\n(B) Joey\n(C) Phoebe\n(D) Ross\n(E) I don't know. (None of the above)"
            ),
            pages=[
                {
                    "citation_id": "doc1",
                    "raw_content": "Rachel was Monica's roommate in 1994.",
                }
            ],
        )

        self.assertIn("question_relation_aligned", result["reason_codes"])
        self.assertIn("roommate", result["question_relations"])
        self.assertIn("roommate", result["context_relations"])
        self.assertIn("roommate", result["relation_overlap"])

    def test_override_abstaining_choice_answer_uses_strong_support(self):
        result = maybe_override_abstaining_choice_answer(
            "(E) I don't know. (None of the above)",
            {
                "best_choice": "(C)",
                "abstain_recommended": False,
                "support_margin": 0.9,
            },
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["prediction_text"], "(C)")
        self.assertEqual(result["to_label"], "C")

    def test_override_abstaining_choice_answer_respects_abstain_recommendation(self):
        result = maybe_override_abstaining_choice_answer(
            "(E) I don't know. (None of the above)",
            {
                "best_choice": "(C)",
                "abstain_recommended": True,
                "support_margin": 0.9,
            },
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["prediction_text"], "(E) I don't know. (None of the above)")


if __name__ == "__main__":
    unittest.main()
