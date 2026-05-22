import unittest

from memoryos.canonical_evidence import (
    build_canonical_evidence_trace,
    canonicalize_query,
    extract_fact_tuples,
)


class CanonicalEvidenceTest(unittest.TestCase):
    def test_canonicalize_query_extracts_entities_time_relations_and_options(self):
        payload = canonicalize_query(
            "Who was Julie's date in October 1995?\n(A) Ross\n(B) Barry\n(C) Paolo"
        )

        self.assertEqual(payload["entities"], ["Julie"])
        self.assertIn("month:1995-10", payload["temporal_forms"])
        self.assertEqual(payload["relation_terms"], ["dating"])
        self.assertEqual(payload["question_shape"], "entity_time_relation_lookup")
        self.assertEqual(len(payload["choice_options"]), 3)

    def test_extract_fact_tuples_prefers_answer_segment(self):
        facts = extract_fact_tuples(
            {
                "citation_id": "doc_ltm",
                "page_id": "doc_ltm",
                "raw_content": "Q: Who was Julie dating in October 1995?\nA: Julie was dating Ross.",
                "summary": "Julie was dating Ross in October 1995.",
                "archive_index": {
                    "source_family": "friends_s11",
                    "source_session": "11",
                    "temporal_forms": ["1995-10"],
                    "relation_terms": ["dating"],
                },
            }
        )

        self.assertTrue(facts)
        self.assertEqual(facts[0]["subject"], "Julie")
        self.assertEqual(facts[0]["object"], "Ross")
        self.assertEqual(facts[0]["relation_terms"], ["dating"])
        self.assertEqual(facts[0]["source_family"], "friends_s11")

    def test_canonicalize_query_handles_possessive_titled_entity(self):
        payload = canonicalize_query("How old was Mr. geller's kid back in September 1994?")

        self.assertTrue(any(entity.lower().startswith("mr") for entity in payload["entities"]))
        self.assertIn("family", payload["relation_terms"])
        self.assertIn("age", payload["relation_terms"])

    def test_extract_fact_tuples_captures_age_value(self):
        facts = extract_fact_tuples(
            {
                "citation_id": "doc_age",
                "page_id": "doc_age",
                "raw_content": "Ross: I'm divorced! I'm only 26 and I'm divorced!",
                "summary": "Mr. Geller's son was 26 years old.",
                "archive_index": {
                    "source_family": "friends_s4",
                    "source_session": "4",
                    "temporal_forms": ["month:1994-09", "year:1994"],
                    "relation_terms": ["family", "age"],
                },
            }
        )

        self.assertTrue(
            any(
                fact["relation_terms"] == ["age"]
                and fact["object"] == "26"
                and fact["subject"] == "Ross"
                for fact in facts
            )
        )

    def test_extract_fact_tuples_uses_speaker_for_single_named_entity_relation(self):
        facts = extract_fact_tuples(
            {
                "citation_id": "doc_roommate",
                "page_id": "doc_roommate",
                "raw_content": "Ross: This is Chandler! My roommate and lead singer of our band!",
                "summary": "Ross introduces Chandler as his roommate.",
                "archive_index": {
                    "source_family": "friends_s7",
                    "source_session": "7",
                    "temporal_forms": ["year:1994"],
                    "relation_terms": ["roommate"],
                },
            }
        )

        self.assertTrue(
            any(
                fact["relation_terms"] == ["roommate"]
                and fact["subject"] == "Ross"
                and fact["object"] == "Chandler"
                for fact in facts
            )
        )

    def test_extract_fact_tuples_captures_date_with_pattern(self):
        facts = extract_fact_tuples(
            {
                "citation_id": "doc_date",
                "page_id": "doc_date",
                "raw_content": "Joey: Great story! But, I gotta go, I got a date with Andrea.",
                "summary": "Joey has a date with Andrea.",
                "archive_index": {
                    "source_family": "friends_s10",
                    "source_session": "10",
                    "temporal_forms": ["year:1994"],
                    "relation_terms": ["dating"],
                },
            }
        )

        self.assertTrue(
            any(
                fact["relation_terms"] == ["dating"]
                and fact["subject"] == "Joey"
                and fact["object"] == "Andrea"
                and fact["fact_quality"].startswith("pattern_")
                for fact in facts
            )
        )

    def test_extract_fact_tuples_captures_that_guy_i_went_out_with_pattern(self):
        facts = extract_fact_tuples(
            {
                "citation_id": "doc_indirect_date",
                "page_id": "doc_indirect_date",
                "raw_content": "Phoebe: Oh ok, Ryan, that guy I went out with, who's in the Navy.",
                "summary": "Phoebe mentions Ryan, the guy she went out with.",
                "archive_index": {
                    "source_family": "friends_s1",
                    "source_session": "1",
                    "temporal_forms": ["year:1994"],
                    "relation_terms": ["dating"],
                },
            }
        )

        self.assertTrue(
            any(
                fact["relation_terms"] == ["dating"]
                and fact["subject"] == "Phoebe"
                and fact["object"] == "Ryan"
                and fact["fact_quality"].startswith("pattern_")
                for fact in facts
            )
        )

    def test_build_trace_marks_two_hop_queries(self):
        trace = build_canonical_evidence_trace(
            "Who was going out with that roommate by October 22, 1994?\n(A) Ross\n(B) Joey\n(C) Monica",
            [
                {
                    "citation_id": "doc_one",
                    "page_id": "doc_one",
                    "raw_content": "Rachel was roommates with Monica. Paolo was dating Rachel in October 1994.",
                    "summary": "Rachel and Monica were roommates; Paolo dated Rachel in October 1994.",
                    "archive_index": {
                        "source_family": "friends_s4",
                        "source_session": "4",
                        "temporal_forms": ["1994-10"],
                        "relation_terms": ["roommate", "dating"],
                    },
                }
            ],
            task_hint="dialogue_qa",
            answer_format="choice",
        )

        self.assertEqual(trace["canonical_query"]["question_shape"], "two_hop_relation_lookup")
        self.assertEqual(trace["reasoning_mode"], "two_hop_relation")
        self.assertGreaterEqual(trace["fact_count"], 1)
        self.assertIn("roommate", trace["relation_coverage"])
        self.assertIn("dating", trace["relation_coverage"])

    def test_build_trace_resolves_two_hop_candidate(self):
        trace = build_canonical_evidence_trace(
            "Remember when Joey had a roommate back on October 20, 1994? Who was going out with that roommate by October 22, 1994?\n(A) Paolo\n(B) Ross\n(C) Joey",
            [
                {
                    "citation_id": "doc_roommate",
                    "page_id": "doc_roommate",
                    "metadata": {
                        "fact_tuples": [
                            {
                                "subject": "Joey",
                                "relation_terms": ["roommate"],
                                "object": "Rachel",
                                "temporal_forms": ["year:1994"],
                            }
                        ]
                    },
                },
                {
                    "citation_id": "doc_dating",
                    "page_id": "doc_dating",
                    "metadata": {
                        "fact_tuples": [
                            {
                                "subject": "Paolo",
                                "relation_terms": ["dating"],
                                "object": "Rachel",
                                "temporal_forms": ["year:1994"],
                            }
                        ]
                    },
                },
            ],
            task_hint="dialogue_qa",
            answer_format="choice",
        )

        self.assertEqual(trace["chain_resolution"]["pivot"], "Rachel")
        self.assertEqual(trace["chain_resolution"]["resolved_candidate"], "Paolo")
        self.assertEqual(trace["chain_resolution"]["status"], "resolved")
        self.assertEqual(trace["resolved_candidates"][0], "Paolo")

    def test_build_trace_hides_misleading_candidates_when_two_hop_chain_missing(self):
        trace = build_canonical_evidence_trace(
            "Remember when Joey had a roommate back on October 20, 1994? Who was going out with that roommate by October 22, 1994?\n(A) Paolo\n(B) Ross\n(C) Joey",
            [
                {
                    "citation_id": "doc_dating_only",
                    "page_id": "doc_dating_only",
                    "metadata": {
                        "fact_tuples": [
                            {
                                "subject": "Paolo",
                                "relation_terms": ["dating"],
                                "object": "Rachel",
                                "temporal_forms": ["year:1994"],
                            }
                        ]
                    },
                }
            ],
            task_hint="dialogue_qa",
            answer_format="choice",
        )

        self.assertEqual(trace["chain_resolution"]["status"], "missing_roommate_anchor")
        self.assertEqual(trace["resolved_candidates"], [])


if __name__ == "__main__":
    unittest.main()
