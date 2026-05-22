import json
import os
import tempfile
import unittest
from unittest.mock import patch

from memoryos.forget_memory import ForgetMemory
from memoryos.long_term import LongTermMemory
from memoryos.memoryos import Memoryos
from memoryos.short_term import ShortTermMemory


class _CountingEntityClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_completion(self, model, messages, temperature=0.0, max_tokens=220, response_format=None):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class _CountingMemoryClient:
    def __init__(self, keyword_payload, entity_payload):
        self.keyword_payload = keyword_payload
        self.entity_payload = entity_payload
        self.calls = 0
        self.keyword_calls = 0
        self.entity_calls = 0

    def chat_completion(self, model, messages, temperature=0.0, max_tokens=180, response_format=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        if '"entities"' in prompt and "entity extraction component" in prompt:
            self.entity_calls += 1
            return json.dumps(self.entity_payload, ensure_ascii=False)
        self.keyword_calls += 1
        return json.dumps(self.keyword_payload, ensure_ascii=False)


class _ExplodingClient:
    def chat_completion(self, *args, **kwargs):
        raise AssertionError("LLM should not be called here")


class KeywordExtractionFlowTest(unittest.TestCase):
    def test_memoryos_keyword_extractor_uses_local_fallback_and_cache(self):
        memo = Memoryos.__new__(Memoryos)
        memo.client = _ExplodingClient()
        memo.llm_model = "stub-model"
        memo._keyword_cache = {}
        memo._entity_cache = {}

        first = memo._extract_keywords("Deborah and Jolene are planning a coffee project.")
        second = memo._extract_keywords("Deborah and Jolene are planning a coffee project.")

        self.assertEqual(first, second)
        self.assertIn("coffee project", first)
        self.assertIn("coffee", first)
        self.assertIn("project", first)
        self.assertIn("jolene", first)

    def test_long_term_uses_injected_keyword_extractor_for_store_and_query(self):
        calls = []

        def extractor(text):
            calls.append(text)
            lowered = text.lower()
            keywords = []
            if "coffee project" in lowered:
                keywords.append("coffee project")
            if "jolene" in lowered:
                keywords.append("jolene")
            if "project" in lowered and "coffee project" not in keywords:
                keywords.append("project")
            return keywords

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "long_term")
            with patch("memoryos.long_term.get_embedding", return_value=[0.0] * 768):
                long_term = LongTermMemory(file_path=store_path)
                long_term.keyword_extractor = extractor
                long_term.add_knowledge(
                    {
                        "id": "doc_1",
                        "doc_id": "doc_1",
                        "summary": "Coffee Project",
                        "raw_content": "Q: Deborah project\nA: Deborah and Jolene are planning a coffee project.",
                        "metadata": {"entities": ["Deborah", "Jolene"]},
                    }
                )

                entry = long_term.get_entry_by_id("doc_1")
                self.assertEqual(entry["keywords"], ["coffee project", "jolene"])

                results = long_term.search_knowledge("What project is Jolene planning?", top_k=1)

        self.assertEqual(results[0]["id"], "doc_1")
        self.assertEqual(len(calls), 2)

    def test_long_term_search_knowledge_uses_entity_score_in_hybrid_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "long_term")
            with patch("memoryos.long_term.get_embedding", return_value=[0.0] * 768):
                long_term = LongTermMemory(file_path=store_path)
                long_term.entity_extractor = lambda text: ["Jolene"] if "Jolene" in text else []
                long_term.keyword_extractor = lambda text: ["robotics project"] if "robotics project" in text.lower() else []
                long_term.add_knowledge(
                    {
                        "id": "doc_match",
                        "doc_id": "doc_match",
                        "summary": "Career note",
                        "raw_content": "Q: Career update\nA: Jolene finished a robotics project.",
                        "metadata": {"entities": ["Jolene"], "keywords": ["robotics project"]},
                    }
                )
                long_term.add_knowledge(
                    {
                        "id": "doc_other",
                        "doc_id": "doc_other",
                        "summary": "Another note",
                        "raw_content": "Q: Career update\nA: Deborah finished a robotics project.",
                        "metadata": {"entities": ["Deborah"], "keywords": ["robotics project"]},
                    }
                )

                results = long_term.search_knowledge("What robotics project did Jolene finish?", top_k=2)

        self.assertEqual(results[0]["id"], "doc_match")
        self.assertGreater(results[0]["entity_score"], results[1]["entity_score"])

    def test_long_term_search_knowledge_uses_exact_match_bonus_to_break_lexical_ties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "long_term")
            with patch("memoryos.long_term.get_embedding", return_value=[0.0] * 768):
                long_term = LongTermMemory(file_path=store_path)
                long_term.entity_extractor = lambda text: []
                long_term.keyword_extractor = lambda text: [
                    token
                    for token in ("aurora", "beacon")
                    if token in text.lower()
                ]
                long_term.add_knowledge(
                    {
                        "id": "doc_exact",
                        "doc_id": "doc_exact",
                        "summary": "Codename record",
                        "raw_content": "Q: Project codename\nA: The codename was Aurora Beacon.",
                        "metadata": {"keywords": ["aurora", "beacon"]},
                    }
                )
                long_term.add_knowledge(
                    {
                        "id": "doc_loose",
                        "doc_id": "doc_loose",
                        "summary": "Codename note",
                        "raw_content": "Q: Project codename\nA: The codename used aurora themes with a beacon signal.",
                        "metadata": {"keywords": ["aurora", "beacon"]},
                    }
                )

                results = long_term.search_knowledge("Aurora Beacon", top_k=2)

        self.assertEqual(results[0]["id"], "doc_exact")
        self.assertGreater(results[0]["exact_bonus"], results[1]["exact_bonus"])
        self.assertEqual(results[0]["surface_score"], results[1]["surface_score"])

    def test_forget_memory_search_only_extracts_query_keywords_once(self):
        calls = []

        def extractor(text):
            calls.append(text)
            lowered = text.lower()
            keywords = []
            if "coffee" in lowered:
                keywords.append("coffee")
            if "jolene" in lowered:
                keywords.append("jolene")
            if "plan" in lowered or "planning" in lowered:
                keywords.append("plan")
            return keywords

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "forget_pool.jsonl")
            with patch("memoryos.forget_memory.get_embedding", return_value=[0.0] * 768):
                forget_memory = ForgetMemory(file_path=file_path)
                forget_memory.keyword_extractor = extractor
                forget_memory.add_page(
                    {
                        "page_id": "cold_1",
                        "doc_id": "cold_1",
                        "user_input": "Deborah and Jolene planned coffee.",
                        "agent_response": "They planned to meet at the cafe.",
                        "metadata": {},
                    }
                )
                self.assertEqual(len(calls), 1)

                results = forget_memory.search("What coffee did Jolene plan?", top_k=1)

        self.assertEqual(results[0]["page_id"], "cold_1")
        self.assertEqual(len(calls), 2)


    def test_short_term_search_reuses_saved_keywords_and_only_extracts_query_once(self):
        calls = []

        def extractor(text):
            calls.append(text)
            lowered = text.lower()
            keywords = []
            if "coffee project" in lowered or "project" in lowered:
                keywords.append("coffee project")
            if "jolene" in lowered:
                keywords.append("jolene")
            return keywords

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "short_term.json")
            short_term = ShortTermMemory(file_path=file_path, max_capacity=5)
            short_term.keyword_extractor = extractor
            short_term.add_page(
                {
                    "page_id": "stm_1",
                    "user_input": "Deborah and Jolene are planning a coffee project.",
                    "agent_response": "They want to finish the coffee project next week.",
                    "keywords": ["coffee project", "jolene"],
                    "metadata": {"keywords": ["coffee project", "jolene"]},
                }
            )

            results = short_term.search("What project is Jolene planning?", k=1)

        self.assertEqual(results[0]["page_id"], "stm_1")
        self.assertEqual(len(calls), 1)

    def test_short_term_save_creates_backup_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "short_term.json")
            short_term = ShortTermMemory(file_path=file_path, max_capacity=5)
            short_term.add_page(
                {
                    "page_id": "stm_1",
                    "user_input": "First note",
                    "agent_response": "Stored in STM.",
                    "metadata": {},
                }
            )
            short_term.add_page(
                {
                    "page_id": "stm_2",
                    "user_input": "Second note",
                    "agent_response": "Overwrites STM store.",
                    "metadata": {},
                }
            )

            with open(f"{file_path}.bak", "r", encoding="utf-8") as handle:
                backup_payload = json.load(handle)

        self.assertEqual([item["page_id"] for item in backup_payload], ["stm_1"])

    def test_long_term_save_creates_backup_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "long_term")
            with patch("memoryos.long_term.get_embedding", return_value=[0.0] * 768):
                long_term = LongTermMemory(file_path=store_path)
                long_term.add_knowledge(
                    {
                        "id": "doc_backup",
                        "doc_id": "doc_backup",
                        "summary": "First LTM note",
                        "raw_content": "Q: backup\nA: first version",
                        "metadata": {"entities": ["Backup"]},
                    }
                )

            backup_path = os.path.join(store_path, "memory_graph.json.bak")
            with open(backup_path, "r", encoding="utf-8") as handle:
                backup_payload = json.load(handle)

        self.assertEqual(backup_payload["knowledge_base"], [])

    def test_forget_memory_append_creates_backup_before_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "forget_pool.jsonl")
            with patch("memoryos.forget_memory.get_embedding", return_value=[0.0] * 768):
                forget_memory = ForgetMemory(file_path=file_path)
                forget_memory.add_page(
                    {
                        "page_id": "cold_1",
                        "doc_id": "cold_1",
                        "user_input": "First cold note",
                        "agent_response": "Stored in cold pool.",
                        "metadata": {},
                    }
                )
                forget_memory.add_page(
                    {
                        "page_id": "cold_2",
                        "doc_id": "cold_2",
                        "user_input": "Second cold note",
                        "agent_response": "Appended to cold pool.",
                        "metadata": {},
                    }
                )

            with open(f"{file_path}.bak", "r", encoding="utf-8") as handle:
                backup_lines = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual([item["page_id"] for item in backup_lines], ["cold_1"])

    def test_entities_use_llm_extraction_and_cache_same_text(self):
        memo = Memoryos.__new__(Memoryos)
        memo.client = _CountingEntityClient({"entities": ["Jolene", "qwen2.5:7b", "Paris"]})
        memo.llm_model = "stub-model"
        memo._entity_cache = {}

        first = memo._extract_entities(
            "Jolene tested qwen2.5:7b in Paris and saved notes at /tmp/memoryos-mcp/logs."
        )
        second = memo._extract_entities(
            "Jolene tested qwen2.5:7b in Paris and saved notes at /tmp/memoryos-mcp/logs."
        )

        self.assertEqual(first, ["Jolene", "qwen2.5:7b", "Paris"])
        self.assertEqual(second, ["Jolene", "qwen2.5:7b", "Paris"])
        self.assertEqual(memo.client.calls, 1)

    def test_local_entity_extractor_captures_multi_token_titles_dates_and_identifiers(self):
        memo = Memoryos.__new__(Memoryos)
        memo._entity_cache = {}

        entities = memo._extract_entities_local(
            'Jolene revisited Rio de Janeiro on 30 August 2023 and replayed "Walking Dead" with qwen2.5:7b notes.'
        )

        self.assertIn("Rio de Janeiro", entities)
        self.assertIn("30 August 2023", entities)
        self.assertIn("Walking Dead", entities)
        self.assertIn("qwen2.5:7b", entities)

    def test_keywords_are_extracted_locally_and_reused_across_stm_ltm_and_forget(self):
        client = _CountingMemoryClient(
            keyword_payload={"keywords": ["Coffee Project", "Jolene"]},
            entity_payload={"entities": ["Jolene"]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("memoryos.long_term.get_embedding", return_value=[0.0] * 768), patch(
                "memoryos.forget_memory.get_embedding", return_value=[0.0] * 768
            ):
                memo = Memoryos(
                    user_id="keyword-user",
                    openai_api_key="ollama",
                    data_storage_path=tmpdir,
                    llm_model="stub-model",
                    enable_async_entity_enrichment=False,
                )
                memo.client = client

                memo.add_memory(
                    user_input="Important coffee project update from Jolene",
                    agent_response="Jolene said the coffee project is moving quickly.",
                    meta_data={"salience_seed": 0.95},
                )

                stm_page = memo.short_term_memory.get_all()[0]
                self.assertIn("coffee", stm_page["keywords"])
                self.assertIn("project", stm_page["keywords"])
                self.assertIn("jolene", stm_page["keywords"])

                ltm_entry = memo.long_term_memory.get_entry_by_id(stm_page["doc_id"])
                self.assertIsNotNone(ltm_entry)
                self.assertIn("coffee", ltm_entry["keywords"])
                self.assertIn("project", ltm_entry["keywords"])
                self.assertIn("jolene", ltm_entry["keywords"])

                memo.forget_memory.add_page(stm_page)
                cold_page = memo.forget_memory.get_page_by_id(stm_page["page_id"])

        self.assertIn("coffee", cold_page["keywords"])
        self.assertIn("project", cold_page["keywords"])
        self.assertIn("jolene", cold_page["keywords"])
        self.assertEqual(client.keyword_calls, 0)
        self.assertEqual(client.entity_calls, 0)
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
