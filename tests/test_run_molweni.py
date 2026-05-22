import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_molweni.py"
SPEC = importlib.util.spec_from_file_location("run_molweni", MODULE_PATH)
RUN_MOLWENI = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUN_MOLWENI
SPEC.loader.exec_module(RUN_MOLWENI)


class MolweniRunnerHelpersTest(unittest.TestCase):
    def test_map_answer_spans_to_edu_ids(self):
        edus = [
            {"edu_id": 0, "speaker": "a", "text": "hello there"},
            {"edu_id": 1, "speaker": "b", "text": "the answer is terminal"},
        ]
        answer = {"text": "terminal", "answer_start": len("a: hello there ") + len("b: the answer is ")}

        gold = RUN_MOLWENI._map_answer_spans_to_edu_ids(edus, [answer])

        self.assertEqual(gold, (1,))

    def test_load_molweni_dialogues_reads_official_style_file(self):
        payload = {
            "data": {
                "title": "test",
                "dialogues": [
                    {
                        "edus": [
                            {"speaker": "a", "text": "hello there"},
                            {"speaker": "b", "text": "the answer is terminal"},
                        ],
                        "context": "a: hello there b: the answer is terminal",
                        "qas": [
                            {
                                "question": "What is the answer?",
                                "id": "q1",
                                "answers": [{"text": "terminal", "answer_start": 33}],
                                "is_impossible": False,
                            }
                        ],
                        "relations": [],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dialogues = RUN_MOLWENI.load_molweni_dialogues(tmpdir, split="test")

        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0].qas[0].answers, ("terminal",))
        self.assertEqual(dialogues[0].qas[0].gold_edu_ids, (1,))

    def test_build_chunks_uses_window_and_stride(self):
        edus = [
            {"edu_id": 0, "speaker": "a", "text": "one"},
            {"edu_id": 1, "speaker": "b", "text": "two"},
            {"edu_id": 2, "speaker": "a", "text": "three"},
            {"edu_id": 3, "speaker": "b", "text": "four"},
        ]

        chunks = RUN_MOLWENI._build_chunks(edus, dialogue_id="dlg", window_size=2, window_stride=1)

        self.assertEqual([(c["edu_start"], c["edu_end"]) for c in chunks], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(chunks[0]["edu_ids"], [0, 1])

    def test_score_prediction_prefers_best_alias(self):
        metrics = RUN_MOLWENI.score_prediction("terminal", ["the terminal", "terminal"])

        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_compute_support_hit_uses_edu_ids(self):
        pages = [
            {"metadata": {"edu_ids": [0, 1]}},
            {"metadata": {"edu_ids": [3]}},
        ]

        self.assertTrue(RUN_MOLWENI.compute_support_hit(pages, [3]))
        self.assertFalse(RUN_MOLWENI.compute_support_hit(pages, [4]))
        self.assertIsNone(RUN_MOLWENI.compute_support_hit(pages, []))

    def test_retrieval_query_kind_is_local(self):
        self.assertEqual(RUN_MOLWENI._retrieval_query_kind_for_question("What is the answer?"), "local")


if __name__ == "__main__":
    unittest.main()
