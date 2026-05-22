import unittest

from tools.materialize_dialsim_case_prefixes import build_case_prefixes


class MaterializeDialSimCasePrefixesTest(unittest.TestCase):
    def test_build_case_prefixes(self):
        dataset_rows = [
            {
                "Episode": "ep1",
                "Session": 1,
                "Date": "September 22, 1994",
                "Script": "script1",
                "hard_qs_past_questions": ["Who was Ross dating?"],
                "hard_qs_past_options": [["Julie", "Rachel", "Carol", "Monica", "I don't know. (None of the above)"]],
                "hard_qs_past_answers": ["Julie"],
            },
            {
                "Episode": "ep1",
                "Session": 2,
                "Date": "September 23, 1994",
                "Script": "script2",
                "easy_qs_ans_w_time_questions": ["What shiny thing did Rachel lose?"],
                "easy_qs_ans_w_time_options": [["A ring", "A dress", "A card", "A ticket", "I don't know. (None of the above)"]],
                "easy_qs_ans_w_time_answers": ["A ring"],
                "easy_qs_ans_w_time_idxes": [3],
            },
        ]
        slice_rows = [
            {
                "episode": "ep1",
                "session": 2,
                "date": "September 23, 1994",
                "qtype": "easy_qs_ans_w_time",
                "question": "What shiny thing did Rachel lose?",
                "true_label": "(A)",
                "support_best_choice": "(A)",
                "template_family": "object_recall",
            }
        ]
        cases = build_case_prefixes(dataset_rows, slice_rows)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["row_count"], 2)
        self.assertEqual(case["prefix_rows"][0]["Episode"], "ep1")
        self.assertNotIn("hard_qs_past_questions", case["prefix_rows"][0])
        self.assertEqual(case["prefix_rows"][1]["easy_qs_ans_w_time_questions"], ["What shiny thing did Rachel lose?"])
        self.assertEqual(case["prefix_rows"][1]["easy_qs_ans_w_time_idxes"], [3])


if __name__ == "__main__":
    unittest.main()
