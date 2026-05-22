import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_friendsqa.py"
SPEC = importlib.util.spec_from_file_location("run_friendsqa", MODULE_PATH)
RUN_FRIENDSQA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUN_FRIENDSQA
SPEC.loader.exec_module(RUN_FRIENDSQA)


class FriendsQARunnerHelpersTest(unittest.TestCase):
    def test_load_dialogues_reads_official_style_json(self):
        payload = {
            "data": [
                {
                    "title": "episode_1",
                    "paragraphs": [
                        {
                            "utterances": [
                                {"speakers": ["Monica"], "utterance": "I booked the restaurant."},
                                {"speakers": ["Rachel"], "utterance": "Great, what time is dinner?"},
                            ],
                            "qas": [
                                {
                                    "id": "q1",
                                    "question": "Who booked the restaurant?",
                                    "answers": [
                                        {"answer_text": "Monica", "utterance_id": 0},
                                        {"answer_text": "Monica"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dialogues = RUN_FRIENDSQA.load_friendsqa_dialogues(str(path), split="validation")

        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0].dialogue_id, "episode_1_0")
        self.assertEqual(dialogues[0].qas[0].answers, ("Monica",))
        self.assertEqual(dialogues[0].qas[0].gold_utterance_ids, (0,))

    def test_load_dialogues_reads_flat_hf_style_record(self):
        payload = [
            {
                "question": "What does Chandler say?",
                "context": "Ross: Hi there.\\nChandler: Angela's the screamer.",
                "answer": "Angela's the screamer.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dialogues = RUN_FRIENDSQA.load_friendsqa_dialogues(str(path), split="train")

        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0].qas[0].question, "What does Chandler say?")
        self.assertEqual(dialogues[0].qas[0].answers, ("Angela's the screamer.",))
        self.assertEqual(len(dialogues[0].utterances), 2)

    def test_build_dialogue_chunks_uses_window_and_stride(self):
        utterances = [
            {"utterance_id": 0, "speaker": "A", "text": "Turn 1"},
            {"utterance_id": 1, "speaker": "B", "text": "Turn 2"},
            {"utterance_id": 2, "speaker": "A", "text": "Turn 3"},
            {"utterance_id": 3, "speaker": "B", "text": "Turn 4"},
            {"utterance_id": 4, "speaker": "A", "text": "Turn 5"},
        ]

        chunks = RUN_FRIENDSQA._build_dialogue_chunks(utterances, window_size=3, window_stride=2, dialogue_id="dlg")

        self.assertEqual([(chunk["span_start"], chunk["span_end"]) for chunk in chunks], [(0, 2), (2, 4)])
        self.assertEqual(chunks[0]["utterance_ids"], [0, 1, 2])
        self.assertIn("A: Turn 5", chunks[1]["text"])

    def test_score_prediction_uses_best_gold_alias(self):
        metrics = RUN_FRIENDSQA.score_prediction("Monica", ["Rachel", "Monica Geller", "Monica"])

        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_compute_support_hit_uses_gold_utterance_ids(self):
        pages = [
            {"metadata": {"utterance_ids": [3, 4, 5]}},
            {"metadata": {"utterance_ids": [8, 9]}},
        ]

        self.assertTrue(RUN_FRIENDSQA.compute_support_hit(pages, [1, 4]))
        self.assertFalse(RUN_FRIENDSQA.compute_support_hit(pages, [10]))
        self.assertIsNone(RUN_FRIENDSQA.compute_support_hit(pages, []))

    def test_retrieval_query_kind_is_local(self):
        self.assertEqual(RUN_FRIENDSQA._retrieval_query_kind_for_question("Who booked the restaurant?"), "local")
        self.assertEqual(RUN_FRIENDSQA._retrieval_query_kind_for_question("Why was Ross upset?"), "local")

    def test_normalize_answer_matches_squad_style(self):
        self.assertEqual(RUN_FRIENDSQA.normalize_answer("The, Monica!"), "monica")
        self.assertAlmostEqual(RUN_FRIENDSQA.token_f1_score("Monica", "the Monica"), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
