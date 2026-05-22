import tempfile
import unittest
from pathlib import Path

from memoryos.forget_memory import ForgetMemory


class ForgetMemoryArchiveIndexTest(unittest.TestCase):
    def test_archive_search_prefers_relation_and_time_aligned_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            forget_path = Path(temp_dir) / "forget_pool.jsonl"
            memory = ForgetMemory(file_path=str(forget_path))
            memory._build_embedding = lambda text: [1.0]

            memory.add_page(
                {
                    "page_id": "aligned",
                    "doc_id": "aligned",
                    "user_input": "Who was Julie dating in October 1995?",
                    "agent_response": "Julie was dating Ross.",
                    "metadata": {
                        "entities": ["Julie", "Ross"],
                        "temporal_forms": ["1995-10"],
                    },
                }
            )
            memory.add_page(
                {
                    "page_id": "distractor",
                    "doc_id": "distractor",
                    "user_input": "Who was Rachel dating in January 1994?",
                    "agent_response": "Rachel was dating Paolo.",
                    "metadata": {
                        "entities": ["Rachel", "Paolo"],
                        "temporal_forms": ["1994-01"],
                    },
                }
            )

            hits = memory.search_archive(
                "So, who was Julie's date on the night of October 1995?",
                top_k=2,
            )

            self.assertEqual(hits[0]["page_id"], "aligned")
            self.assertGreater(hits[0]["archive_match"]["relation_overlap"], 0.0)
            self.assertGreater(hits[0]["archive_match"]["keyword_overlap"], 0.0)
            self.assertEqual(hits[0]["archive_match"]["relation_terms"], ["dating"])
            self.assertEqual(memory.get_stats()["archive_index_entries"], 2)


if __name__ == "__main__":
    unittest.main()
