import os
import sys
import re
import json
import uuid  # ✅ 明确引入 uuid，解决 "name 'uuid' is not defined" 报错
import logging
import hashlib
import time
import math
import requests 
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import uuid

# ✅ 强制设置环境变量以解决 Protobuf 版本冲突问题
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None 
    print("Warning: openai module not found.")

# 全局配置
OLLAMA_HOST = "http://localhost:11434"
LOCAL_OLLAMA_HOSTS = {"127.0.0.1", "localhost"}
EMBEDDING_USAGE_EVENTS: List[Dict[str, Any]] = []
# 屏蔽词列表：防止这些系统内部标识符进入长时记忆（LTM）
SYSTEM_RESERVED_KEYWORDS = ["fastchunk", "memoryos", "stm_add", "ltm_raw", "chunk_id", "masiqi"]

# ==============================================================================
# 🛠️ 基础辅助函数
# ==============================================================================

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_id(prefix: str = "page") -> str:
    """
    生成唯一的资源 ID。
    """
    uid = str(uuid.uuid4().hex)[:16]
    if prefix:
        return f"{prefix}_{uid}"
    return uid

def ensure_directory_exists(file_path: str):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    if v is None: return np.zeros(768, dtype=np.float32) 
    try:
        norm = np.linalg.norm(v)
        if norm == 0: return v
        return v / norm
    except Exception:
        return np.zeros(768, dtype=np.float32)

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _get_url_host(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").strip().lower()
    except Exception:
        return ""


def is_local_ollama_url(url: str) -> bool:
    return _get_url_host(url) in LOCAL_OLLAMA_HOSTS


def ensure_local_no_proxy(url: str = "") -> None:
    target = str(url or OLLAMA_HOST or "").strip()
    if not target or not is_local_ollama_url(target):
        return

    required_hosts = ["127.0.0.1", "localhost"]
    for env_name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(env_name, "")
        values = [item.strip() for item in current.split(",") if item.strip()]
        changed = False
        for host in required_hosts:
            if host not in values:
                values.append(host)
                changed = True
        if changed or not current:
            os.environ[env_name] = ",".join(values)


def clear_embedding_usage_history() -> None:
    EMBEDDING_USAGE_EVENTS.clear()


def snapshot_embedding_usage_history() -> List[Dict[str, Any]]:
    return [dict(item) for item in EMBEDDING_USAGE_EVENTS]


def summarize_usage_events(events: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for event in events or []:
        if not isinstance(event, dict):
            continue
        summary["calls"] += 1
        usage = event.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                summary[key] += int(value)
    return summary


def parse_ollama_think_setting() -> Any:
    raw_value = os.environ.get("OLLAMA_THINK")
    if raw_value is None:
        raw_value = os.environ.get("OLLAMA_THINK_MODE")
    if raw_value is None:
        return None

    value = str(raw_value).strip().lower()
    if value in {"", "auto", "default"}:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"low", "medium", "high"}:
        return value
    return raw_value


def normalize_keyword_candidate(value: Any) -> str:
    if value is None:
        return ""
    candidate = str(value)
    candidate = candidate.replace("\n", " ").replace("\r", " ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = candidate.strip(" ,.;:!?\"'`[]{}()<>|\/，。；：！？【】（）《》")
    if not candidate:
        return ""
    return candidate.lower()


def normalize_keyword_values(values: List[Any], limit: int = 16) -> List[str]:
    blocked = {
        "a",
        "an",
        "and",
        "answer",
        "answers",
        "assistant",
        "chat",
        "conversation",
        "dialogue",
        "i",
        "it",
        "memory",
        "question",
        "questions",
        "response",
        "responses",
        "session",
        "summary",
        "system",
        "text",
        "the",
        "they",
        "this",
        "transcript",
        "user",
        "we",
        "you",
        "问题",
        "回答",
        "对话",
        "总结",
        "文本",
        "用户",
        "系统",
        "助手",
    }
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        if isinstance(value, dict):
            candidate = (
                value.get("keyword")
                or value.get("term")
                or value.get("text")
                or value.get("label")
                or value.get("name")
            )
        else:
            candidate = value
        candidate = normalize_keyword_candidate(candidate)
        if len(candidate) < 2 or candidate in blocked or candidate in seen:
            continue
        if re.fullmatch(r"[_\-./\\]+", candidate):
            continue
        if len(candidate) > 64:
            continue
        seen.add(candidate)
        normalized.append(candidate)
        if len(normalized) >= limit:
            break
    return normalized


def fallback_extract_keywords(text: str, limit: int = 16) -> List[str]:
    compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
    if not compact:
        return []

    stop_words = {
        "about",
        "after",
        "answer",
        "assistant",
        "based",
        "could",
        "from",
        "have",
        "into",
        "just",
        "memory",
        "memories",
        "only",
        "possible",
        "provided",
        "question",
        "summary",
        "their",
        "them",
        "they",
        "unknown",
        "user",
        "were",
        "what",
        "when",
        "which",
        "with",
        "would",
    }
    ascii_tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", compact.lower())
        if token not in stop_words
    ]
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,12}", compact)
    return normalize_keyword_values(list(dict.fromkeys(ascii_tokens + cjk_tokens)), limit=limit)

# ---------------------------------------------------------
# ✅ 核心增强: 系统标签脱敏 (防止 LLM 学习系统 ID)
# ---------------------------------------------------------
def mask_system_tags(text: str) -> str:
    """
    将文本中的 FastChunk_71 等系统标识符替换为通用占位符，
    防止 LLM 将其提取为实体或核心知识点。
    """
    if not text: return text
    # 匹配 FastChunk_数字 或相似格式
    masked_text = re.sub(r'FastChunk_\d+', '[CONTEXT_BLOCK]', text)
    masked_text = re.sub(r'memoryos-mcp', '[SYSTEM_PATH]', masked_text)
    return masked_text

# ---------------------------------------------------------
# ✅ 核心增强: 实体清洗 (防止 LTM 污染)
# ---------------------------------------------------------
def filter_system_metadata(data: Union[Dict, List, str]) -> Any:
    """
    递归检查数据对象，剔除所有包含系统保留字（如 FastChunk）的条目。
    用于在 STM -> LTM 转化阶段进行最后的安全性过滤。
    """
    if isinstance(data, dict):
        return {k: filter_system_metadata(v) for k, v in data.items() 
                if not any(word in str(k).lower() for word in SYSTEM_RESERVED_KEYWORDS)}
    elif isinstance(data, list):
        return [filter_system_metadata(i) for i in data 
                if not any(word in str(i).lower() for word in SYSTEM_RESERVED_KEYWORDS)]
    elif isinstance(data, str):
        if any(word in data.lower() for word in SYSTEM_RESERVED_KEYWORDS):
            return "[REDACTED_SYS_META]"
        return data
    return data

# ---------------------------------------------------------
# ✅ 修复: compute_time_decay (增强参数兼容性)
# ---------------------------------------------------------
def compute_time_decay(timestamp: str, *args, **kwargs) -> float:
    """
    计算记忆随时间的衰减率。
    """
    half_life = 24.0  # 默认半衰期为 24 小时
    for arg in args:
        if isinstance(arg, (int, float)):
            half_life = float(arg)
    if 'half_life_hours' in kwargs:
        half_life = kwargs['half_life_hours']
        
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        delta = datetime.now() - dt
        hours_passed = delta.total_seconds() / 3600.0
        if hours_passed < 0: hours_passed = 0
        decay = math.exp(- (0.693 / half_life) * hours_passed)
        return round(decay, 4)
    except Exception:
        return 1.0

def check_conversation_continuity(last_qa: Dict, current_qa: Dict, *args, threshold_seconds: int = 300, **kwargs) -> bool:
    if not last_qa or not current_qa: return False
    try:
        t1 = datetime.strptime(last_qa.get('timestamp'), "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(current_qa.get('timestamp'), "%Y-%m-%d %H:%M:%S")
        delta = (t2 - t1).total_seconds()
        return abs(delta) <= threshold_seconds
    except Exception:
        return False

def run_parallel_tasks(tasks: List[Callable], max_workers: int = 5) -> List[Any]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None: results.append(res)
            except Exception as e:
                logging.error(f"Error in parallel task: {e}")
    return results

# ==============================================================================
# 🤖 OpenAI Client (适配 Ollama & 自动脱敏)
# ==============================================================================

class OpenAIClient:
    def __init__(self, api_key: str = None, base_url: str = None, timeout: float = 120.0):
        self.api_key = api_key or "ollama"
        self.base_url = base_url or f"{OLLAMA_HOST}/v1"
        self.timeout = timeout
        self.last_response_meta: Dict[str, Any] = {}
        self._usage_history: List[Dict[str, Any]] = []
        ensure_local_no_proxy(self.base_url)
        if OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            self.client = None

    def clear_usage_history(self) -> None:
        self.last_response_meta = {}
        self._usage_history = []

    def snapshot_usage_history(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._usage_history]

    def _record_response_meta(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response: Any = None,
        error: str = "",
    ) -> None:
        usage_payload = getattr(response, "usage", None) if response is not None else None
        if usage_payload is None and isinstance(response, dict):
            usage_payload = response.get("usage")
        usage: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage_payload, key, None)
            if value is None and isinstance(usage_payload, dict):
                value = usage_payload.get(key)
            if isinstance(value, (int, float)):
                usage[key] = int(value)

        meta = {
            "timestamp": get_timestamp(),
            "model": model,
            "message_count": len(messages or []),
            "prompt_chars": sum(len(str(message.get("content", ""))) for message in (messages or [])),
            "usage": usage,
        }
        if error:
            meta["error"] = str(error)
        self.last_response_meta = meta
        self._usage_history.append(meta)

    def chat_completion(self, messages: List[Dict], model: str = "qwen2.5:7b", temperature: float = 0.7, max_tokens: int = 1000, response_format = None) -> str:
        if self.client is None:
            self._record_response_meta(model=model, messages=messages, error="openai_module_not_installed")
            return "Simulated response: OpenAI module not installed."
        
        # ✅ 输入端脱敏：防止 LLM 学习和利用上下文块 ID
        processed_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if "content" in new_msg:
                new_msg["content"] = mask_system_tags(new_msg["content"])
            processed_messages.append(new_msg)

        try:
            think_setting = parse_ollama_think_setting()
            use_native_ollama_chat = think_setting is not None and str(self.base_url).rstrip("/").endswith("/v1")

            if use_native_ollama_chat:
                native_base_url = str(self.base_url).rstrip("/")
                if native_base_url.endswith("/v1"):
                    native_base_url = native_base_url[:-3]
                payload = {
                    "model": model,
                    "messages": processed_messages,
                    "stream": False,
                    "think": think_setting,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                if response_format and response_format.get("type") == "json_object":
                    payload["format"] = "json"

                session = requests.Session()
                session.trust_env = False
                response = session.post(
                    f"{native_base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                usage_response = {
                    "usage": {
                        "prompt_tokens": int(response_json.get("prompt_eval_count", 0) or 0),
                        "completion_tokens": int(response_json.get("eval_count", 0) or 0),
                        "total_tokens": int(response_json.get("prompt_eval_count", 0) or 0)
                        + int(response_json.get("eval_count", 0) or 0),
                    }
                }
                self._record_response_meta(model=model, messages=processed_messages, response=usage_response)
                content = str(response_json.get("message", {}).get("content", "") or "")
            else:
                kwargs = {
                    "model": model,
                    "messages": processed_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if think_setting is not None:
                    kwargs["extra_body"] = {"think": think_setting}
                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                self._record_response_meta(model=model, messages=processed_messages, response=response)
                content = response.choices[0].message.content or ""

            # ✅ 输出端清洗：剔除非法系统元数据
            if response_format and response_format.get("type") == "json_object":
                try:
                    js_data = json.loads(content)
                    cleaned_data = filter_system_metadata(js_data)
                    return json.dumps(cleaned_data, ensure_ascii=False)
                except:
                    pass

            return content
        except Exception as e:
            self._record_response_meta(model=model, messages=processed_messages, error=str(e))
            logging.error(f"❌ OpenAI API Error: {e}")
            return f"Error: {str(e)}"

# ---------------------------------------------------------
# 🧠 Embedding Utilities
# ---------------------------------------------------------

def get_embedding(text: str, model_name: str = "nomic-embed-text", **kwargs) -> List[float]:
    """
    调用本地 Ollama 生成向量。
    """
    if not text: return [0.0] * 768
    
    # 预处理文本，防止系统 ID 扭曲向量空间分布
    text = mask_system_tags(text)
    
    # 自动重定向旧模型名
    if 'bge-m3' in model_name.lower(): model_name = 'nomic-embed-text'
    
    try:
        url = f"{OLLAMA_HOST}/v1/embeddings"
        payload = {"model": model_name, "input": text}
        ensure_local_no_proxy(url)
        with requests.Session() as session:
            if is_local_ollama_url(url):
                session.trust_env = False
            response = session.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            EMBEDDING_USAGE_EVENTS.append(
                {
                    "timestamp": get_timestamp(),
                    "model": model_name,
                    "prompt_chars": len(text),
                    "usage": usage if isinstance(usage, dict) else {},
                }
            )
            embedding = None
            if isinstance(data, dict):
                rows = data.get("data", [])
                if isinstance(rows, list) and rows:
                    first = rows[0] if isinstance(rows[0], dict) else {}
                    embedding = first.get("embedding")
                if embedding is None:
                    embedding = data.get('embedding')
            if embedding:
                vec = np.array(embedding, dtype=np.float32)
                return normalize_vector(vec).tolist()
        return [0.0] * 768
    except Exception as e:
        logging.error(f"❌ Embedding Generation Error: {e}")
        return [0.0] * 768

# ---------------------------------------------------------
# 占位函数：供各组件统一调用，防止导入循环
# ---------------------------------------------------------
def generate_page_meta_info(*args, **kwargs) -> Dict[str, Any]:
    return {"summary": "Standard memory unit", "keywords": [], "G": 0.5}

def gpt_user_profile_analysis(*args, **kwargs): return "Updating profile..."
def gpt_knowledge_extraction(*args, **kwargs): return {}