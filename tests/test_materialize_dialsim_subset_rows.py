import unittest

from tools.materialize_dialsim_subset_rows import build_subset_rows


class MaterializeDialSimSubsetRowsTest(unittest.TestCase):
    def test_build_subset_rows(self):
        dataset_rows = [
            {
                "Episode": "ep1",
                "Session": 1,
                "Date": "1994-09-22",
                "Script": "script1",
                "hard_qs_past_questions": ["Who was Ross dating?"],
                "hard_qs_past_options": [["Julie", "Rachel", "Carol", "Monica", "I don't know. (None of the above)"]],
                "hard_qs_past_answers": ["Julie"],
                "easy_qs_ans_w_time_questions": ["What shiny thing did Rachel lose?"],
                "easy_qs_ans_w_time_options": [["A ring", "A dress", "A card", "A ticket", "I don't know. (None of the above)"]],
                "easy_qs_ans_w_time_answers": ["A ring"],
                "easy_qs_ans_w_time_idxes": [3],
            }
        ]
        slice_rows = [
            {
                "episode": "ep1",
                "session": 1,
                "qtype": "hard_qs_past",
                "question": "Who was Ross dating?",
            },
            {
                "episode": "ep1",
                "session": 1,
                "qtype": "easy_qs_ans_w_time",
                "question": "What shiny thing did Rachel lose?",
            },
        ]
        subset = build_subset_rows(dataset_rows, slice_rows)
        self.assertEqual(len(subset), 1)
        row = subset[0]
        self.assertEqual(row["Episode"], "ep1")
        self.assertEqual(row["hard_qs_past_questions"], ["Who was Ross dating?"])
        self.assertEqual(row["hard_qs_past_answers"], ["Julie"])
        self.assertEqual(row["easy_qs_ans_w_time_questions"], ["What shiny thing did Rachel lose?"])
        self.assertEqual(row["easy_qs_ans_w_time_idxes"], [3])


if __name__ == "__main__":
    unittest.main()
