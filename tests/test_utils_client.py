import os
import unittest
from unittest.mock import patch

from memoryos import utils


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, calls, payload):
        self.calls = calls
        self.payload = payload
        self.trust_env = True

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.payload)


class OpenAIClientThinkSettingTest(unittest.TestCase):
    def test_qwen3_defaults_to_no_think_for_local_ollama(self):
        calls = []
        payload = {
            "message": {"content": "4"},
            "prompt_eval_count": 12,
            "eval_count": 1,
        }
        client = utils.OpenAIClient(api_key="ollama", base_url="http://127.0.0.1:11434/v1", timeout=5)
        client.client = object()

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_THINK", None)
            os.environ.pop("OLLAMA_THINK_MODE", None)
            with patch("memoryos.utils.requests.Session", side_effect=lambda: _FakeSession(calls, payload)):
                answer = client.chat_completion(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "只回答数字：2+2=?"}],
                    temperature=0.0,
                    max_tokens=16,
                )

        self.assertEqual(answer, "4")
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:11434/api/chat")
        self.assertIs(calls[0]["json"]["think"], False)

    def test_explicit_env_think_setting_overrides_default(self):
        calls = []
        payload = {
            "message": {"content": "4"},
            "prompt_eval_count": 12,
            "eval_count": 1,
        }
        client = utils.OpenAIClient(api_key="ollama", base_url="http://127.0.0.1:11434/v1", timeout=5)
        client.client = object()

        with patch.dict(os.environ, {"OLLAMA_THINK": "true"}, clear=False):
            with patch("memoryos.utils.requests.Session", side_effect=lambda: _FakeSession(calls, payload)):
                answer = client.chat_completion(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "只回答数字：2+2=?"}],
                    temperature=0.0,
                    max_tokens=16,
                )

        self.assertEqual(answer, "4")
        self.assertIs(calls[0]["json"]["think"], True)


if __name__ == "__main__":
    unittest.main()
