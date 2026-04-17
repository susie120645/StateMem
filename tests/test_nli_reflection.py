import copy
import unittest

from memoryos.services.lifecycle_service import LifecycleService


class DummyClient:
    def __init__(self, verdict_map):
        self.verdict_map = verdict_map

    def chat_completion(self, model, messages, temperature=0.0, max_tokens=8):
        prompt = messages[0]["content"]
        for marker, verdict in self.verdict_map.items():
            if marker in prompt:
                return verdict
        return "neutral"


class DummyForgetMemory:
    def __init__(self):
        self.records = {}

    def get_page_by_id(self, page_id):
        return self.records.get(page_id)

    def add_page(self, page):
        page_id = page.get("page_id") or page.get("citation_id")
        if page_id:
            self.records[page_id] = copy.deepcopy(page)

    def get_stats(self):
        return {"lean_records": len(self.records)}

    def clear(self):
        self.records = {}


class DummyShortTermMemory:
    def __init__(self, pages=None):
        self.pages = {page["page_id"]: copy.deepcopy(page) for page in (pages or [])}

    def get_page_by_id(self, page_id):
        page = self.pages.get(page_id)
        return copy.deepcopy(page) if page else None

    def get_all(self):
        return [copy.deepcopy(page) for page in self.pages.values()]

    def evict(self, page_id, to_forget_pool=False, forget_memory=None):
        page = self.pages.pop(page_id, None)
        if page and to_forget_pool and forget_memory is not None:
            forget_memory.add_page(page)
        return copy.deepcopy(page) if page else None

    def save(self):
        return None

    def clear(self):
        self.pages = {}


class DummyLongTermMemory:
    HOT_TEMPERATURE_THRESHOLD = 35.0

    def __init__(self, entries=None):
        self.entries = {entry["id"]: copy.deepcopy(entry) for entry in (entries or [])}

    def get_entry_by_id(self, entry_id):
        entry = self.entries.get(entry_id)
        return copy.deepcopy(entry) if entry else None

    def node_exists(self, node_id):
        return node_id in self.entries

    def get_all_nodes(self):
        return [copy.deepcopy(entry) for entry in self.entries.values()]

    def add_from_page(self, page):
        node_id = page.get("citation_id") or page.get("page_id")
        if not node_id:
            return None
        if node_id not in self.entries:
            self.entries[node_id] = {
                "id": node_id,
                "raw_content": page.get("raw_content", ""),
                "summary": page.get("summary", ""),
                "G": float(page.get("G", 0.0)),
                "C": float(page.get("C", 0.6)),
                "T": float(page.get("T", 42.0)),
                "storage_tier": "hot",
            }
        return node_id

    def adjust_chunk_metrics(self, chunk_id, delta_G=0.0, delta_C=0.0, delta_T=0.0):
        entry = self.entries.get(chunk_id)
        if not entry:
            return False
        entry["G"] = max(0.0, float(entry.get("G", 0.0)) + float(delta_G))
        entry["C"] = min(1.0, max(0.0, float(entry.get("C", 0.0)) + float(delta_C)))
        entry["T"] = min(100.0, max(0.0, float(entry.get("T", 0.0)) + float(delta_T)))
        entry["storage_tier"] = "hot" if entry["T"] >= self.HOT_TEMPERATURE_THRESHOLD else "cold"
        return True

    def clear(self):
        self.entries = {}


class DummyMemoryos:
    def __init__(self, pages):
        self.pages = {page["page_id"]: copy.deepcopy(page) for page in pages}

    def get_memory_by_id(self, memory_id):
        page = self.pages.get(memory_id)
        return copy.deepcopy(page) if page else None


class NLIReflectionRoutingTest(unittest.TestCase):
    def build_service(self, verdict_map, stm_pages=None, ltm_entries=None):
        self.forget_memory = DummyForgetMemory()
        self.short_term_memory = DummyShortTermMemory(stm_pages)
        self.long_term_memory = DummyLongTermMemory(ltm_entries)
        self.lifecycle = LifecycleService(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            forget_memory=self.forget_memory,
            client=DummyClient(verdict_map),
            llm_model="dummy",
        )
        return self.lifecycle

    def tearDown(self):
        if hasattr(self, "lifecycle"):
            self.lifecycle.shutdown()

    def test_contradiction_reflection_demotes_memory_to_forget_pool(self):
        page = {
            "page_id": "doc_bad",
            "citation_id": "doc_bad",
            "location": "long_term",
            "raw_content": "The user dislikes concise answers and prefers detailed responses.",
            "summary": "Preference memory",
            "metadata": {},
            "meta_data": {},
            "G": 0.05,
            "C": 0.2,
            "T": 18.0,
            "salience_seed": 0.1,
        }
        self.build_service(
            verdict_map={"dislikes concise answers": "contradiction"},
            ltm_entries=[
                {
                    "id": "doc_bad",
                    "raw_content": page["raw_content"],
                    "summary": page["summary"],
                    "G": page["G"],
                    "C": page["C"],
                    "T": page["T"],
                    "storage_tier": "cold",
                }
            ],
        )
        reflection = self.lifecycle.build_interaction_reflection(
            query="How should I answer the user?",
            final_answer="The user prefers concise answers.",
            retrieval_results={
                "retrieved_pages": [page],
                "citations": ["doc_bad"],
                "hit_ids": ["doc_bad"],
                "source_stats": {"LTM": 1},
                "salvage_count": 0,
            },
            base_feedback_score=0.7,
        )
        self.assertEqual(reflection["nli"]["contradiction"], ["doc_bad"])
        self.assertEqual(reflection["memory"]["recommended_destination"], "forget_pool")

        memoryos = DummyMemoryos([page])
        result = self.lifecycle.post_inference_memory_flow(
            memoryos,
            ["doc_bad"],
            feedback_score=reflection["feedback_score"],
            reflection=reflection,
        )
        processed = result["processed"][0]
        self.assertEqual(processed["target_destination"], "forget_pool")
        self.assertEqual(processed["action"], "ltm_demoted_to_cold_archive")
        self.assertIn("doc_bad", self.forget_memory.records)

    def test_entailment_reflection_promotes_short_term_memory_to_long_term(self):
        raw_content = (
            "The user prefers concise, direct answers with bullet-free formatting whenever possible. "
            "They repeatedly ask for short responses and dislike redundant explanation. "
            "This preference should be remembered for future sessions."
        )
        page = {
            "page_id": "doc_good",
            "citation_id": "doc_good",
            "location": "short_term",
            "raw_content": raw_content,
            "summary": "Concise response preference",
            "metadata": {},
            "meta_data": {},
            "G": 0.42,
            "C": 0.72,
            "T": 54.0,
            "salience_seed": 0.84,
            "agent_response": raw_content,
        }
        self.build_service(
            verdict_map={"prefers concise, direct answers": "entailment"},
            stm_pages=[page],
        )
        reflection = self.lifecycle.build_interaction_reflection(
            query="How should I answer the user?",
            final_answer="The user prefers concise, direct answers.",
            retrieval_results={
                "retrieved_pages": [page],
                "citations": ["doc_good"],
                "hit_ids": ["doc_good"],
                "source_stats": {"STM": 1},
                "salvage_count": 0,
            },
            base_feedback_score=0.9,
        )
        self.assertEqual(reflection["nli"]["entailment"], ["doc_good"])
        self.assertEqual(reflection["memory"]["recommended_destination"], "long_term")

        memoryos = DummyMemoryos([page])
        result = self.lifecycle.post_inference_memory_flow(
            memoryos,
            ["doc_good"],
            feedback_score=reflection["feedback_score"],
            reflection=reflection,
        )
        processed = result["processed"][0]
        self.assertEqual(processed["target_destination"], "long_term")
        self.assertEqual(processed["action"], "stm_archived_to_forget_and_ltm_reinforced")
        self.assertTrue(self.long_term_memory.node_exists("doc_good"))


if __name__ == "__main__":
    unittest.main()
