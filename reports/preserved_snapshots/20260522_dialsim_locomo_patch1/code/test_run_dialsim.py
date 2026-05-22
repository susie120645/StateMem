import unittest

from tools.run_dialsim import (
    _answer_choice_with_canonical_candidate,
    _build_shape_adaptive_calibration,
    _classify_canonical_retrieval_gap,
    _build_canonical_prompt_guidance,
    _build_canonical_retrieval_retry_query,
    _canonical_retrieval_score,
    _maybe_apply_citation_backed_choice_fallback,
    _maybe_apply_contrastive_option_verifier,
    _maybe_override_abstaining_choice_answer_with_canonical,
    _resolve_canonical_choice_candidate,
    _resolve_candidate_choice_label,
    _shape_calibration_base_thresholds,
    _should_retry_canonical_retrieval,
)


class RunDialSimTest(unittest.TestCase):
    def test_resolve_candidate_choice_label_matches_exact_surface(self):
        label, match_type = _resolve_candidate_choice_label(
            "Mr. Geller",
            ["Ross", "Rachel", "Mr Geller", "Monica", "Phoebe"],
        )

        self.assertEqual(label, "(C)")
        self.assertEqual(match_type, "exact")

    def test_canonical_override_applies_for_resolved_two_hop_candidate(self):
        result = _maybe_override_abstaining_choice_answer_with_canonical(
            "(E) I don't know. (None of the above)",
            {
                "confidence": 0.93,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {
                    "status": "resolved",
                    "resolved_candidate": "Paolo",
                    "score": 1.84,
                },
                "resolved_candidates": ["Paolo"],
            },
            qtype="hard_qs_past_past",
            options=["Ross", "Paolo", "Monica", "Joey", "I don't know. (None of the above)"],
            min_confidence=0.78,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["prediction_text"], "(B)")
        self.assertEqual(result["to_label"], "B")
        self.assertEqual(result["candidate"], "Paolo")

    def test_canonical_choice_head_applies_for_strong_time_filtered_candidate(self):
        result = _answer_choice_with_canonical_candidate(
            {
                "confidence": 0.92,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "time_filtered_lookup"},
                "resolved_candidates": ["26"],
            },
            qtype="hard_qs_past",
            options=["24", "25", "26", "27", "I don't know. (None of the above)"],
            min_confidence=0.9,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["prediction_text"], "(C)")
        self.assertEqual(result["to_label"], "C")

    def test_canonical_choice_head_respects_support_margin_threshold(self):
        result = _answer_choice_with_canonical_candidate(
            {
                "confidence": 0.92,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {
                    "status": "resolved",
                    "resolved_candidate": "Paolo",
                    "score": 0.45,
                },
                "resolved_candidates": ["Paolo"],
            },
            qtype="hard_qs_past_past",
            options=["Ross", "Paolo", "Monica", "Joey", "I don't know. (None of the above)"],
            min_confidence=0.8,
            min_support_margin=0.8,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "canonical_support_margin_below_threshold")

    def test_canonical_candidate_resolution_rejects_ambiguous_candidates(self):
        result = _resolve_canonical_choice_candidate(
            {
                "confidence": 0.94,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "entity_relation_lookup"},
                "resolved_candidates": ["Ross", "Paolo"],
            },
            qtype="hard_qs_past",
            options=["Ross", "Paolo", "Monica", "Joey", "I don't know. (None of the above)"],
            min_confidence=0.9,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertEqual(result["reason"], "canonical_candidate_ambiguous")
        self.assertEqual(result["matched_label"], "")

    def test_build_canonical_retrieval_retry_query_includes_structured_facets(self):
        query = _build_canonical_retrieval_retry_query(
            "Hey, remember when Paolo went out with that person back in January 1994? So, who was bunking with that person's roommate as of September 1994?",
            qtype="hard_qs_past_past",
        )

        self.assertIn("Question shape: two_hop_relation_lookup", query)
        self.assertIn("Key entities:", query)
        self.assertIn("Required relations:", query)
        self.assertIn("Time anchors:", query)

    def test_classify_canonical_retrieval_gap_marks_two_hop_missing_anchor(self):
        gap = _classify_canonical_retrieval_gap(
            {
                "reasoning_mode": "two_hop_relation",
                "canonical_query": {
                    "question_shape": "two_hop_relation_lookup",
                    "entities": ["Paolo"],
                    "relation_terms": ["dating", "roommate"],
                },
                "chain_resolution": {"status": "missing_roommate_anchor", "pivot": ""},
                "relation_coverage": {"dating": False, "roommate": False},
            },
            question_text="Remember when Paolo went out with that person back in January 1994? Who was that person's roommate in September 1994?",
        )

        self.assertEqual(gap["gap_type"], "two_hop_missing_anchor")
        self.assertEqual(gap["question_shape"], "two_hop_relation_lookup")

    def test_build_canonical_retrieval_retry_query_targets_two_hop_anchor_gap(self):
        query = _build_canonical_retrieval_retry_query(
            "Remember when Paolo went out with that person back in January 1994? Who was that person's roommate in September 1994?",
            qtype="hard_qs_past_past",
            canonical_evidence={
                "reasoning_mode": "two_hop_relation",
                "canonical_query": {
                    "query_text": "Remember when Paolo went out with that person back in January 1994? Who was that person's roommate in September 1994?",
                    "question_shape": "two_hop_relation_lookup",
                    "entities": ["Paolo"],
                    "relation_terms": ["dating", "roommate"],
                    "pronoun_slots": ["that person"],
                },
                "chain_resolution": {"status": "missing_roommate_anchor", "pivot": ""},
                "relation_coverage": {"dating": False, "roommate": False},
            },
        )

        self.assertIn("Repair focus: resolve the intermediate person", query)
        self.assertIn("Hop 1: find who is connected to Paolo through dating", query)
        self.assertIn("Hop 2: after resolving that person, retrieve roommate-related evidence", query)

    def test_build_canonical_retrieval_retry_query_targets_missing_relation_gap(self):
        query = _build_canonical_retrieval_retry_query(
            "So, who was Rachel's date on the night of January 7, 1996?",
            qtype="hard_qs_past",
            canonical_evidence={
                "reasoning_mode": "time_filtered_lookup",
                "canonical_query": {
                    "query_text": "Who was Rachel's date on the night of January 7, 1996?",
                    "question_shape": "entity_time_relation_lookup",
                    "entities": ["Rachel"],
                    "relation_terms": ["dating"],
                    "temporal_forms": ["year:1996"],
                    "pronoun_slots": [],
                },
                "relation_coverage": {"dating": False},
                "abstain_recommended": True,
            },
        )

        self.assertIn("Missing relations: dating", query)
        self.assertIn("Target entities: Rachel", query)
        self.assertIn("Prefer direct relation statements over broad conversational context.", query)

    def test_build_canonical_prompt_guidance_marks_two_hop_candidate(self):
        guidance = _build_canonical_prompt_guidance(
            {
                "confidence": 0.93,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {
                    "status": "resolved",
                    "resolved_candidate": "Paolo",
                    "score": 1.84,
                },
                "resolved_candidates": ["Paolo"],
            },
            qtype="hard_qs_past_past",
            options=["Ross", "Paolo", "Monica", "Joey", "I don't know. (None of the above)"],
            min_confidence=0.9,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertEqual(guidance[0], "This is a two-hop relation lookup.")
        self.assertIn("option (B)", " ".join(guidance))

    def test_canonical_retrieval_score_rewards_resolved_two_hop_evidence(self):
        resolved = _canonical_retrieval_score(
            {
                "confidence": 0.84,
                "abstain_recommended": False,
                "resolved_candidates": ["Paolo"],
                "relation_coverage": {"dating": True, "roommate": True},
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {"status": "resolved"},
            }
        )
        weak = _canonical_retrieval_score(
            {
                "confidence": 0.84,
                "abstain_recommended": True,
                "resolved_candidates": [],
                "relation_coverage": {"dating": False, "roommate": False},
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {"status": "missing_dating_link"},
            }
        )

        self.assertGreater(resolved, weak)

    def test_should_retry_canonical_retrieval_triggers_for_weak_two_hop_chain(self):
        self.assertTrue(
            _should_retry_canonical_retrieval(
                {
                    "confidence": 0.74,
                    "abstain_recommended": True,
                    "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                    "chain_resolution": {"status": "missing_dating_link"},
                },
                qtype="hard_qs_past_past",
                enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
                min_confidence=0.82,
            )
        )

    def test_should_retry_canonical_retrieval_skips_strong_non_targeted_qtype(self):
        self.assertFalse(
            _should_retry_canonical_retrieval(
                {
                    "confidence": 0.95,
                    "abstain_recommended": False,
                    "canonical_query": {"question_shape": "time_filtered_lookup"},
                },
                qtype="easy_qs_ans_w_time",
                enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
                min_confidence=0.82,
            )
        )

    def test_canonical_override_rejects_unresolved_two_hop_chain(self):
        result = _maybe_override_abstaining_choice_answer_with_canonical(
            "(E) I don't know. (None of the above)",
            {
                "confidence": 0.95,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "chain_resolution": {
                    "status": "missing_dating_link",
                    "resolved_candidate": "",
                    "score": 0.0,
                },
                "resolved_candidates": [],
            },
            qtype="hard_qs_past_past",
            options=["Ross", "Paolo", "Monica", "Joey", "I don't know. (None of the above)"],
            min_confidence=0.78,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "canonical_chain_unresolved")

    def test_canonical_override_rejects_non_targeted_qtype(self):
        result = _maybe_override_abstaining_choice_answer_with_canonical(
            "(E) I don't know. (None of the above)",
            {
                "confidence": 0.9,
                "abstain_recommended": False,
                "canonical_query": {"question_shape": "time_filtered_lookup"},
                "resolved_candidates": ["26"],
            },
            qtype="easy_qs_ans_w_time",
            options=["24", "25", "26", "27", "I don't know. (None of the above)"],
            min_confidence=0.78,
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "qtype_not_targeted")

    def test_contrastive_option_verifier_applies_for_strong_targeted_support(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(C)",
                "support_margin": 0.82,
                "best_non_abstain_score": 0.7,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 1.0,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": True,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["prediction_text"], "(C)")
        self.assertEqual(result["to_label"], "C")

    def test_contrastive_option_verifier_rejects_weak_time_anchor(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(B)",
                "support_margin": 0.9,
                "best_non_abstain_score": 0.8,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 0.2,
                "question_entity_overlap": 1.0,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": False,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "question_time_overlap_below_threshold")

    def test_contrastive_option_verifier_rejects_missing_option_overlap(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 0.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 1.0,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": True,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "zero_score_non_exact_option_match")

    def test_contrastive_option_verifier_rejects_low_option_score_when_requested(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 1.0,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": True,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
            min_non_abstain_score=0.2,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "best_non_abstain_score_below_threshold")

    def test_contrastive_option_verifier_rejects_zero_score_partial_overlap_for_one_hop_qtype(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 0.5,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 0.8,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": True,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "zero_score_non_exact_option_match")

    def test_contrastive_option_verifier_allows_two_hop_qtype_without_option_overlap(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 0.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 1.0,
                "relation_overlap": ["dating"],
                "question_relations": ["dating"],
                "abstain_recommended": True,
            },
            qtype="hard_qs_past_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["to_label"], "D")

    def test_contrastive_option_verifier_rejects_non_abstain_best_choice_without_strong_score(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_choice": "(D)",
                "best_choice_score": 0.7,
                "best_non_abstain_choice": "(D)",
                "support_margin": 0.6,
                "best_non_abstain_score": 0.7,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 0.7,
                "relation_overlap": ["roommate"],
                "question_relations": ["roommate"],
                "abstain_recommended": False,
            },
            qtype="hard_qs_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "best_choice_non_abstain_not_confident_enough")

    def test_contrastive_option_verifier_rejects_shape_mismatch_for_two_hop_qtype(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 0.75,
                "relation_overlap": ["dating", "family"],
                "question_relations": ["dating", "family"],
                "abstain_recommended": True,
            },
            canonical_evidence={
                "reasoning_mode": "time_filtered_lookup",
                "canonical_query": {"question_shape": "entity_time_relation_lookup"},
                "relation_coverage": {"dating": True},
                "chain_resolution": {"status": "", "resolved_candidate": ""},
            },
            qtype="hard_qs_past_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "canonical_shape_mismatch_for_two_hop_qtype")

    def test_contrastive_option_verifier_rejects_unresolved_two_hop_chain_despite_relation_coverage(self):
        result = _maybe_apply_contrastive_option_verifier(
            "(E) I don't know. (None of the above)",
            {
                "best_non_abstain_choice": "(D)",
                "support_margin": 1.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 1.0,
                "question_time_overlap": 0.5,
                "question_entity_overlap": 0.63,
                "relation_overlap": ["dating", "roommate"],
                "question_relations": ["dating", "roommate"],
                "abstain_recommended": True,
            },
            canonical_evidence={
                "reasoning_mode": "two_hop_relation",
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "relation_coverage": {"dating": True, "roommate": True},
                "chain_resolution": {"status": "missing_roommate_anchor", "resolved_candidate": ""},
            },
            qtype="hard_qs_past_past",
            enabled_qtypes=["hard_qs_past", "hard_qs_past_past"],
            min_margin=0.6,
            min_time_overlap=0.5,
            min_entity_overlap=0.3,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "canonical_chain_unresolved_despite_relation_coverage")

    def test_citation_backed_choice_fallback_applies_for_entity_time_lookup_hard_past(self):
        result = _maybe_apply_citation_backed_choice_fallback(
            "(E) I don't know. (None of the above)",
            {
                "best_choice": "(D)",
                "best_choice_score": 0.8,
                "support_margin": 0.7,
                "best_option_overlap": 1.0,
                "question_time_overlap": 1.0,
                "question_entity_overlap": 0.3333,
                "family_count": 1,
                "top_family_share": 1.0,
                "citations": ["episode_s5_c3"],
            },
            canonical_evidence={
                "canonical_query": {"question_shape": "entity_time_lookup"},
            },
            qtype="hard_qs_past",
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["to_label"], "D")
        self.assertEqual(result["reason"], "citation_backed_choice_fallback")

    def test_citation_backed_choice_fallback_applies_for_easy_time_question(self):
        result = _maybe_apply_citation_backed_choice_fallback(
            "(E) I don't know. (None of the above)",
            {
                "best_choice": "(D)",
                "best_choice_score": 0.7,
                "support_margin": 0.6,
                "best_option_overlap": 0.75,
                "question_time_overlap": 0.5,
                "question_entity_overlap": 0.6429,
                "family_count": 1,
                "top_family_share": 1.0,
                "citations": ["episode_s2_c2"],
                "question_relations": ["gift_object"],
                "relation_overlap": ["gift_object"],
            },
            canonical_evidence={
                "canonical_query": {"question_shape": "entity_time_relation_lookup"},
            },
            qtype="easy_qs_ans_w_time",
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["to_label"], "D")
        self.assertEqual(result["reason"], "citation_backed_choice_fallback")

    def test_shape_adaptive_calibration_lowers_thresholds_for_two_hop_hard_qtype(self):
        calibration = _build_shape_adaptive_calibration(
            {
                "confidence": 0.81,
                "abstain_recommended": False,
                "reasoning_mode": "two_hop_relation",
                "canonical_query": {"question_shape": "two_hop_relation_lookup"},
                "relation_coverage": {"dating": True, "roommate": True},
                "chain_resolution": {"status": "resolved", "score": 1.6},
            },
            qtype="hard_qs_past_past",
            enabled=True,
            base_thresholds=_shape_calibration_base_thresholds(
                retrieval_retry_min_confidence=0.82,
                retrieval_retry_score_margin=0.05,
                choice_head_min_confidence=0.9,
                choice_override_min_confidence=0.78,
            ),
        )

        self.assertTrue(calibration.enabled)
        self.assertLess(calibration.choice_head_min_confidence, 0.9)
        self.assertLess(calibration.choice_override_min_confidence, 0.78)
        self.assertLess(calibration.retrieval_retry_min_confidence, 0.82)
        self.assertGreaterEqual(calibration.choice_head_min_support_margin, 0.8)
        self.assertIn("question_shape:two_hop_relation_lookup", calibration.applied_layers)

    def test_shape_adaptive_calibration_profile_overrides_builtin_defaults(self):
        calibration = _build_shape_adaptive_calibration(
            {
                "confidence": 0.84,
                "abstain_recommended": False,
                "reasoning_mode": "direct_lookup",
                "canonical_query": {"question_shape": "entity_relation_lookup"},
                "relation_coverage": {"dating": True},
                "chain_resolution": {"status": "", "score": 0.0},
            },
            qtype="hard_qs_past",
            enabled=True,
            base_thresholds=_shape_calibration_base_thresholds(
                retrieval_retry_min_confidence=0.82,
                retrieval_retry_score_margin=0.05,
                choice_head_min_confidence=0.9,
                choice_override_min_confidence=0.78,
            ),
            calibration_profile={
                "qtype": {
                    "hard_qs_past": {
                        "choice_head_min_confidence": 0.66,
                        "choice_override_min_confidence": 0.61,
                    }
                }
            },
            calibration_profile_name="unit_test_profile",
        )

        self.assertEqual(calibration.source, "unit_test_profile")
        self.assertEqual(calibration.choice_head_min_confidence, 0.63)
        self.assertEqual(calibration.choice_override_min_confidence, 0.58)


if __name__ == "__main__":
    unittest.main()
