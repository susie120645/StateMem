import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_qmsum.py"
SPEC = importlib.util.spec_from_file_location("run_qmsum", MODULE_PATH)
RUN_QMSUM = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUN_QMSUM
SPEC.loader.exec_module(RUN_QMSUM)


class QMSumRunnerHelpersTest(unittest.TestCase):
    def test_collect_query_instances_supports_general_and_specific(self):
        record = {
            "meeting_id": "meeting_alpha",
            "general_query_list": [
                {"query": "What budget decision did the team make?", "answer": "They approved the increased budget."}
            ],
            "specific_query_list": [
                {
                    "query": "What did Alice say about launch timing?",
                    "answer": "Alice said the launch should slip by two weeks.",
                    "relevant_text_span": [[2, 4], ["6", "6"]],
                }
            ],
        }

        queries = RUN_QMSUM._collect_query_instances(record)

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0].query_type, "general")
        self.assertEqual(queries[1].query_type, "specific")
        self.assertEqual(queries[1].relevant_spans, ((2, 4), (6, 6)))

    def test_build_transcript_chunks_uses_window_and_stride(self):
        record = {
            "meeting_id": "meeting_beta",
            "meeting_transcripts": [
                {"speaker": "A", "content": "Turn 1"},
                {"speaker": "B", "content": "Turn 2"},
                {"speaker": "A", "content": "Turn 3"},
                {"speaker": "B", "content": "Turn 4"},
                {"speaker": "A", "content": "Turn 5"},
            ],
        }

        chunks = RUN_QMSUM._build_transcript_chunks(record, window_size=3, window_stride=2)

        self.assertEqual([(chunk["span_start"], chunk["span_end"]) for chunk in chunks], [(0, 2), (2, 4)])
        self.assertIn("A: Turn 1", chunks[0]["text"])
        self.assertIn("A: Turn 5", chunks[1]["text"])

    def test_score_summary_prefers_closer_match(self):
        exact = RUN_QMSUM.score_summary(
            "They approved the increased budget and delayed launch by two weeks.",
            "They approved the increased budget and delayed launch by two weeks.",
        )
        loose = RUN_QMSUM.score_summary(
            "They discussed several topics.",
            "They approved the increased budget and delayed launch by two weeks.",
        )

        self.assertGreater(exact["rouge1_f1"], loose["rouge1_f1"])
        self.assertGreater(exact["rougeL_f1"], loose["rougeL_f1"])
        self.assertEqual(exact["rouge1_f1"], 1.0)
        self.assertEqual(exact["rougeL_f1"], 1.0)

    def test_compute_support_coverage_tracks_gold_span_hits(self):
        pages = [
            {"metadata": {"span_start": 3, "span_end": 5}},
            {"metadata": {"span_start": 9, "span_end": 10}},
        ]

        support = RUN_QMSUM.compute_support_coverage(pages, [(0, 1), (4, 4), (10, 12)])

        self.assertTrue(support["support_any_hit"])
        self.assertEqual(support["support_hit_count"], 2)
        self.assertAlmostEqual(support["support_span_recall"], 0.6667, places=4)

    def test_retrieval_query_kind_maps_general_and_specific(self):
        self.assertEqual(RUN_QMSUM._retrieval_query_kind_for_query("general"), "global")
        self.assertEqual(RUN_QMSUM._retrieval_query_kind_for_query("specific"), "local")
        self.assertEqual(RUN_QMSUM._retrieval_query_kind_for_query("unknown"), "local")

    def test_load_qmsum_records_reads_jsonl_split_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            path = base / "test.jsonl"
            path.write_text(
                '{"meeting_id":"m1","meeting_transcripts":[{"speaker":"A","content":"Hi"}],"general_query_list":[{"query":"Q","answer":"A"}]}\n',
                encoding="utf-8",
            )

            records = RUN_QMSUM.load_qmsum_records(str(base), split="test")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["meeting_id"], "m1")

    def test_collect_query_instances_supports_flat_query_answer_schema(self):
        record = {
            "id": 7,
            "query": "Summarize the whole meeting.",
            "answer": "They approved the schedule and assigned follow-up tasks.",
            "transcript": [{"speaker": "A", "content": "Hello"}],
        }

        queries = RUN_QMSUM._collect_query_instances(record)

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].query_type, "general")
        self.assertEqual(queries[0].answer, "They approved the schedule and assigned follow-up tasks.")


if __name__ == "__main__":
    unittest.main()
