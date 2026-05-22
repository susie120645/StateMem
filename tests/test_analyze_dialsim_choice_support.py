import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_dialsim_choice_support import (
    analyze_report,
    build_review_queue,
    classify_review_item,
    extract_clean_answer_side_failures,
    filter_review_queue_by_relation_signature,
    filter_review_queue_by_tier,
    relabel_review_queue,
    select_strict_relation_candidates,
)


class AnalyzeDialSimChoiceSupportTest(unittest.TestCase):
    def test_selects_and_exports_strict_relation_candidates(self):
        payload = {
            "accuracy": 0.5,
            "results": [
                {
                    "episode": "ep1",
                    "session": 1,
                    "date": "1994-09-22",
                    "qtype": "hard_qs_past_past",
                    "question": "Who was the roommate?",
                    "true_label": "(B)",
                    "pred": "(E) I don't know. (None of the above)",
                    "result": "Wrong",
                    "hit_ids": ["doc1"],
                    "source_stats": {"LTM": 1},
                    "choice_support": {
                        "best_choice": "(B)",
                        "support_margin": 0.8,
                        "best_choice_score": 0.9,
                        "best_non_abstain_choice": "(B)",
                        "best_non_abstain_score": 0.9,
                        "question_time_overlap": 0.7,
                        "question_entity_overlap": 0.8,
                        "relation_overlap": ["roommate"],
                        "relation_missing": [],
                        "top_family_share": 1.0,
                        "family_ids": ["doc1"],
                        "reason_codes": [
                            "best_choice_non_abstain",
                            "question_time_anchor_aligned",
                            "question_entity_anchor_aligned",
                            "question_relation_aligned",
                            "support_source_single_family",
                        ],
                    },
                },
                {
                    "episode": "ep2",
                    "session": 2,
                    "date": "1994-10-01",
                    "qtype": "hard_qs_past",
                    "question": "Who lived next door?",
                    "true_label": "(A)",
                    "pred": "(E) I don't know. (None of the above)",
                    "result": "Wrong",
                    "hit_ids": ["doc2"],
                    "source_stats": {"LTM": 1},
                    "choice_support": {
                        "best_choice": "(D)",
                        "support_margin": 0.9,
                        "reason_codes": [
                            "best_choice_non_abstain",
                            "question_time_anchor_aligned",
                            "question_entity_anchor_missing",
                        ],
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = analyze_report(report_path, 0.6)
            self.assertEqual(summary["strict_relation_candidate_count"], 1)
            self.assertEqual(summary["strict_relation_correct_count"], 1)
            self.assertEqual(summary["strict_relation_priority_counts"], {"P2": 1})

            rows = payload["results"]
            queue = build_review_queue(select_strict_relation_candidates(rows, 0.6))
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["support_best_choice"], "(B)")
            self.assertEqual(queue[0]["queue_reason"], "strict_relation_non_abstain_support")
            self.assertEqual(queue[0]["priority_tier"], "P2")
            self.assertGreater(queue[0]["priority_score"], 0.0)
            self.assertEqual(filter_review_queue_by_tier(queue, priority_tier="P1"), [])
            self.assertEqual(len(filter_review_queue_by_tier(queue, priority_tier="P2")), 1)
            self.assertEqual(
                len(filter_review_queue_by_relation_signature(queue, required_relations=["roommate"])),
                1,
            )
            self.assertEqual(
                filter_review_queue_by_relation_signature(queue, required_relations=["dating"]),
                [],
            )

    def test_classifies_template_mismatch_and_clean_answer_side(self):
        template_item = {
            "question": "Hey, remember when Paolo went out with that person back in January 1994? So, who was bunking with that person's roommate as of September 1994?",
            "relation_overlap": ["dating", "family", "roommate"],
            "current_pred": "(E) I don't know. (None of the above)",
            "true_label": "(B)",
            "support_best_choice": "(B)",
        }
        self.assertEqual(classify_review_item(template_item), "relation_template_mismatch")

        clean_item = {
            "question": "Hey, do you remember how old Mr. geller's kid was back in September 1994?",
            "relation_overlap": ["family"],
            "current_pred": "(E) I don't know. (None of the above)",
            "true_label": "(D)",
            "support_best_choice": "(D)",
        }
        self.assertEqual(classify_review_item(clean_item), "clean_answer_side_failure")

        ambiguous_item = {
            "question": "Hey, so you remember how Phoebe had that client back on June 11, 1995? Who was the roommate of the client by the time September 1994 rolled around?",
            "relation_overlap": ["roommate"],
            "current_pred": "(E) I don't know. (None of the above)",
            "true_label": "(B)",
            "support_best_choice": "(C)",
        }
        self.assertEqual(classify_review_item(ambiguous_item), "relation_reference_ambiguity")

    def test_relabels_queue_and_extracts_clean_answer_side(self):
        queue = [
            {
                "question": "Hey, remember when Paolo went out with that person back in January 1994? So, who was bunking with that person's roommate as of September 1994?",
                "relation_overlap": ["dating", "family", "roommate"],
                "current_pred": "(E) I don't know. (None of the above)",
                "true_label": "(B)",
                "support_best_choice": "(B)",
            },
            {
                "question": "Hey, do you remember how old Mr. geller's kid was back in September 1994?",
                "relation_overlap": ["family"],
                "current_pred": "(E) I don't know. (None of the above)",
                "true_label": "(D)",
                "support_best_choice": "(D)",
            },
        ]
        relabeled = relabel_review_queue(queue)
        self.assertEqual(relabeled[0]["diagnosis_bucket"], "relation_template_mismatch")
        self.assertEqual(relabeled[1]["diagnosis_bucket"], "clean_answer_side_failure")

        rows = [
            {
                "episode": "ep1",
                "session": 1,
                "date": "1994-09-22",
                "qtype": "hard",
                "question": "Hey, remember when Paolo went out with that person back in January 1994? So, who was bunking with that person's roommate as of September 1994?",
                "true_label": "(B)",
                "pred": "(E) I don't know. (None of the above)",
                "options": ["A", "B", "C", "D", "E"],
                "hit_ids": ["doc1"],
                "choice_support": {
                    "best_choice": "(B)",
                    "support_margin": 1.0,
                    "citations": ["doc1"],
                    "reason_codes": ["best_choice_non_abstain"],
                    "relation_overlap": ["dating", "family", "roommate"],
                },
            },
            {
                "episode": "ep2",
                "session": 2,
                "date": "1994-09-22",
                "qtype": "hard",
                "question": "Hey, do you remember how old Mr. geller's kid was back in September 1994?",
                "true_label": "(D)",
                "pred": "(E) I don't know. (None of the above)",
                "options": ["A", "B", "C", "D", "E"],
                "hit_ids": ["doc2"],
                "choice_support": {
                    "best_choice": "(D)",
                    "support_margin": 1.0,
                    "citations": ["doc2"],
                    "reason_codes": ["best_choice_non_abstain"],
                    "relation_overlap": ["family"],
                },
            },
        ]
        clean = extract_clean_answer_side_failures(rows)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]["question"], "Hey, do you remember how old Mr. geller's kid was back in September 1994?")


if __name__ == "__main__":
    unittest.main()
