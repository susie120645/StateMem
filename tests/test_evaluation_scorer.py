import unittest

from statemem.evaluation.scorer import build_evaluation_result, summarize_score_metrics


class EvaluationScorerTest(unittest.TestCase):
    def test_exact_answer_and_matching_citations_score_full_credit(self):
        result = build_evaluation_result(
            gold_answer="1999",
            predicted_answer="1999",
            gold_evidence_ids=["doc:A", "doc:B"],
            predicted_evidence_ids=["doc:A", "doc:B"],
            citation_support_precision=1.0,
            citation_ids=["page_1", "page_2"],
            hit_ids=["page_1", "page_2"],
        )

        self.assertEqual(result["scores"]["answer_em"], 1.0)
        self.assertEqual(result["scores"]["citation_score"], 1.0)
        self.assertEqual(result["scores"]["final"], 1.0)
        self.assertEqual(result["scores"]["bucket"], "correct_and_grounded")

    def test_exact_answer_with_bad_citations_is_separated(self):
        result = build_evaluation_result(
            gold_answer="1999",
            predicted_answer="1999",
            gold_evidence_ids=["doc:A"],
            predicted_evidence_ids=["doc:Z"],
            citation_support_precision=0.0,
            citation_ids=["page_9"],
            hit_ids=["page_9"],
        )

        self.assertEqual(result["scores"]["answer_em"], 1.0)
        self.assertEqual(result["scores"]["citation_score"], 0.0)
        self.assertEqual(result["scores"]["final"], 0.65)
        self.assertEqual(result["scores"]["bucket"], "correct_but_bad_citation")

    def test_wrong_answer_can_still_be_grounded(self):
        result = build_evaluation_result(
            gold_answer="1999",
            predicted_answer="2001",
            gold_evidence_ids=["doc:A"],
            predicted_evidence_ids=["doc:A"],
            citation_support_precision=1.0,
            citation_ids=["page_1"],
            hit_ids=["page_1"],
        )

        self.assertEqual(result["scores"]["answer_score"], 0.0)
        self.assertEqual(result["scores"]["citation_score"], 1.0)
        self.assertEqual(result["scores"]["final"], 0.35)
        self.assertEqual(result["scores"]["bucket"], "wrong_but_grounded")

    def test_summary_aggregates_scores_and_buckets(self):
        records = [
            build_evaluation_result(
                gold_answer="1999",
                predicted_answer="1999",
                gold_evidence_ids=["doc:A"],
                predicted_evidence_ids=["doc:A"],
                citation_support_precision=1.0,
            ),
            build_evaluation_result(
                gold_answer="1999",
                predicted_answer="2001",
                gold_evidence_ids=["doc:A"],
                predicted_evidence_ids=[],
                citation_support_precision=0.0,
            ),
        ]

        summary = summarize_score_metrics(records)

        self.assertEqual(summary["answer_em"], 0.5)
        self.assertEqual(summary["citation_score"], 0.5)
        self.assertEqual(summary["final"], 0.5)
        self.assertEqual(summary["judgement_buckets"]["correct_and_grounded"], 1)
        self.assertEqual(summary["judgement_buckets"]["wrong_and_ungrounded"], 1)


if __name__ == "__main__":
    unittest.main()
