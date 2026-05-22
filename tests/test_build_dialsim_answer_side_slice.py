import unittest

from tools.build_dialsim_answer_side_slice import build_minimal_slice, template_family


class BuildDialSimAnswerSideSliceTest(unittest.TestCase):
    def test_template_family(self):
        self.assertEqual(
            template_family("So, who was Julie's date on the night of November 1995?"),
            "direct_dating_lookup",
        )
        self.assertEqual(
            template_family("Hey, just out of curiosity, who was bunking with Rachel in 1994?"),
            "roommate_lookup",
        )
        self.assertEqual(
            template_family("Hey, do you remember how old Mr. geller's kid was back in September 1994?"),
            "attribute_recall_age",
        )

    def test_builds_balanced_slice(self):
        rows = [
            {
                "question": "So, who was Julie's date on the night of November 1995?",
                "qtype": "hard_qs_past",
                "episode": "ep1",
                "session": 1,
                "support_margin": 1.0,
                "citations": ["doc1"],
            },
            {
                "question": "So, who was Julie's date on the night of October 1995?",
                "qtype": "hard_qs_past",
                "episode": "ep2",
                "session": 2,
                "support_margin": 0.9,
                "citations": ["doc2"],
            },
            {
                "question": "Hey, do you remember who Ross's girlfriend was back on February 20, 1997?",
                "qtype": "hard_qs_past",
                "episode": "ep3",
                "session": 3,
                "support_margin": 0.8,
                "citations": [],
            },
            {
                "question": "Hey, just out of curiosity, who was bunking with Rachel in 1994?",
                "qtype": "hard_qs_past",
                "episode": "ep4",
                "session": 4,
                "support_margin": 1.0,
                "citations": ["doc4"],
            },
            {
                "question": "Hey, just out of curiosity, who was bunking with Joey on October 20, 1994?",
                "qtype": "hard_qs_past",
                "episode": "ep5",
                "session": 5,
                "support_margin": 1.0,
                "citations": [],
            },
        ]
        selected = build_minimal_slice(rows, max_per_template=2)
        self.assertEqual(len(selected), 4)
        dating = [row for row in selected if row["template_family"] == "direct_dating_lookup"]
        roommate = [row for row in selected if row["template_family"] == "roommate_lookup"]
        self.assertEqual(len(dating), 2)
        self.assertEqual(len(roommate), 2)
        self.assertEqual(dating[0]["episode"], "ep1")
        self.assertEqual(dating[1]["episode"], "ep2")


if __name__ == "__main__":
    unittest.main()
