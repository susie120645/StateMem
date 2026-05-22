import os
import sys
import re
import json
import uuid  # ✅ 明确引入 uuid，解决 "name 'uuid' is not defined" 报错
import logging
import hashlib
import time
import math
import shutil
import tempfile
import requests 
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Iterable, Optional, Union, Callable
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


def get_backup_path(file_path: str, suffix: str = ".bak") -> str:
    return f"{file_path}{suffix}"


def create_backup_snapshot(file_path: str, enabled: bool = True, min_size_bytes: int = 1) -> Optional[str]:
    if not enabled or not file_path or not os.path.exists(file_path):
        return None

    try:
        if os.path.getsize(file_path) < max(0, int(min_size_bytes)):
            return None
    except OSError:
        return None

    backup_path = get_backup_path(file_path)
    ensure_directory_exists(backup_path)
    shutil.copy2(file_path, backup_path)
    return backup_path


def _write_atomic(file_path: str, writer: Callable[[Any], None]) -> None:
    ensure_directory_exists(file_path)
    directory = os.path.dirname(file_path) or "."
    suffix = os.path.splitext(file_path)[1] or ".tmp"
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_json_with_backup(
    file_path: str,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    create_backup: bool = False,
) -> Optional[str]:
    backup_path = create_backup_snapshot(file_path, enabled=create_backup)
    _write_atomic(
        file_path,
        lambda handle: json.dump(payload, handle, ensure_ascii=ensure_ascii, indent=indent),
    )
    return backup_path


def write_lines_with_backup(
    file_path: str,
    lines: Iterable[str],
    *,
    create_backup: bool = False,
) -> Optional[str]:
    backup_path = create_backup_snapshot(file_path, enabled=create_backup)

    def _writer(handle: Any) -> None:
        handle.writelines(lines)

    _write_atomic(file_path, _writer)
    return backup_path

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


def resolve_ollama_think_setting(model: str, base_url: str = "", explicit_setting: Any = None) -> Any:
    if explicit_setting is not None:
        return explicit_setting
    if not is_local_ollama_url(base_url or OLLAMA_HOST):
        return None
    model_name = str(model or "").strip().lower()
    if "qwen3" in model_name:
        return False
    return None


def normalize_keyword_candidate(value: Any) -> str:
    if value is None:
        return ""
    candidate = str(value)
    candidate = candidate.replace("\n", " ").replace("\r", " ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = candidate.strip(" ,.;:!?\"'`[]{}()<>|/，。；：！？【】（）《》")
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


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def strip_ordinal_suffixes(text: str) -> str:
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", str(text or ""), flags=re.I)


def parse_number_token(token: object) -> Union[int, None]:
    text = str(token or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    return words.get(text)


def extract_temporal_canonical_forms(text: object) -> List[str]:
    cleaned = strip_ordinal_suffixes(str(text or ""))
    lowered = cleaned.lower()
    forms = set()

    for match in re.finditer(r"\b(\d{1,2})\s+([A-Za-z]+)\s*,?\s*(\d{4})\b", cleaned):
        month_num = MONTH_NAME_TO_NUMBER.get(match.group(2).lower())
        if not month_num:
            continue
        day = int(match.group(1))
        year = int(match.group(3))
        try:
            dt = datetime(year, month_num, day)
        except ValueError:
            continue
        forms.add(f"date:{dt.strftime('%Y-%m-%d')}")
        forms.add(f"year:{year}")
        forms.add(f"month:{dt.strftime('%Y-%m')}")

    for match in re.finditer(r"\b([A-Za-z]+)\s+(\d{4})\b", cleaned):
        month_num = MONTH_NAME_TO_NUMBER.get(match.group(1).lower())
        if month_num:
            forms.add(f"month:{int(match.group(2)):04d}-{month_num:02d}")
            forms.add(f"year:{match.group(2)}")

    for match in re.finditer(r"\b(?:since about|start year|year|in)\s+(\d{4})\b", lowered):
        forms.add(f"year:{match.group(1)}")

    if not forms:
        years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", lowered)
        if len(set(years)) == 1:
            forms.add(f"year:{years[0]}")

    for match in re.finditer(r"\b(?:a|one)\s+week\s+before\s+(\d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4})\b", cleaned, re.I):
        ref_forms = extract_temporal_canonical_forms(match.group(1))
        ref_date = next((item for item in ref_forms if item.startswith("date:")), None)
        if ref_date:
            ref_dt = datetime.strptime(ref_date.split(":", 1)[1], "%Y-%m-%d")
            anchor = ref_dt - timedelta(days=7)
            forms.add(f"week_of:{anchor.strftime('%Y-%m-%d')}")

    for match in re.finditer(r"\bweek of (\d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4})\b", cleaned, re.I):
        ref_forms = extract_temporal_canonical_forms(match.group(1))
        ref_date = next((item for item in ref_forms if item.startswith("date:")), None)
        if ref_date:
            ref_dt = datetime.strptime(ref_date.split(":", 1)[1], "%Y-%m-%d")
            forms.add(f"week_of:{ref_dt.strftime('%Y-%m-%d')}")

    for match in re.finditer(r"\blast week of ([A-Za-z]+)\s+(\d{4})\b", lowered):
        month_num = MONTH_NAME_TO_NUMBER.get(match.group(1).lower())
        if month_num:
            forms.add(f"last_week_month:{match.group(2)}-{month_num:02d}")

    for match in re.finditer(
        r"\b(?:for\s+)?(?:about |around |almost )?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b",
        lowered,
    ):
        value = parse_number_token(match.group(1))
        if value is not None:
            forms.add(f"duration_years:{value}")

    return sorted(forms)

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
    def __init__(self, api_key: str = None, base_url: str = None, timeout: float = None):
        self.api_key = api_key or "ollama"
        self.base_url = base_url or f"{OLLAMA_HOST}/v1"
        if timeout is None:
            raw_timeout = os.environ.get("OLLAMA_CLIENT_TIMEOUT") or os.environ.get("OPENAI_CLIENT_TIMEOUT")
            try:
                timeout = float(raw_timeout) if raw_timeout is not None else 120.0
            except Exception:
                timeout = 120.0
        self.timeout = float(timeout)
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
            explicit_think_setting = parse_ollama_think_setting()
            think_setting = resolve_ollama_think_setting(
                model=model,
                base_url=self.base_url,
                explicit_setting=explicit_think_setting,
            )
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

_EMBEDDING_DIM_CACHE: Dict[str, int] = {}


def _embedding_cache_key(host: str, model_name: str) -> str:
    return f"{str(host or '').rstrip('/')}::{str(model_name or '').strip()}"


def _get_cached_embedding_dim(host: str, model_name: str) -> Optional[int]:
    dim = _EMBEDDING_DIM_CACHE.get(_embedding_cache_key(host, model_name))
    return dim if isinstance(dim, int) and dim > 0 else None


def _set_cached_embedding_dim(host: str, model_name: str, dim: int) -> None:
    try:
        dim = int(dim)
    except Exception:
        return
    if dim <= 0:
        return
    _EMBEDDING_DIM_CACHE[_embedding_cache_key(host, model_name)] = dim


def _zero_embedding(host: str, model_name: str, fallback_dim: int = 768) -> List[float]:
    cached_dim = _get_cached_embedding_dim(host, model_name)
    dim = cached_dim if cached_dim else int(fallback_dim)
    return [0.0] * max(0, dim)


def _extract_embedding_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return str(getattr(response, "text", "") or "")

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return str(payload)


def _is_context_length_error(message: str) -> bool:
    msg = str(message or "").lower()
    return (
        "maximum context length" in msg
        or "context length" in msg
        or "too many tokens" in msg
        or "max tokens" in msg
    )


def get_embedding(text: str, model_name: str = "nomic-embed-text", **kwargs) -> List[float]:
    """
    调用 Ollama / OpenAI-compatible embeddings 生成向量。
    """
    embedding_host = str(
        kwargs.get("embedding_host")
        or kwargs.get("ollama_host")
        or kwargs.get("base_url")
        or OLLAMA_HOST
        or ""
    ).strip()
    if not embedding_host:
        embedding_host = OLLAMA_HOST

    if not text:
        return _zero_embedding(embedding_host, model_name)
    
    # 预处理文本，防止系统 ID 扭曲向量空间分布
    text = mask_system_tags(text)
    
    # 自动重定向旧模型名
    if 'bge-m3' in model_name.lower(): model_name = 'nomic-embed-text'
    
    try:
        embedding_host = embedding_host.rstrip("/")
        url = f"{embedding_host}/v1/embeddings"
        timeout = kwargs.get("timeout", 30)
        try:
            timeout = float(timeout)
        except Exception:
            timeout = 30.0

        max_retries = kwargs.get("max_retries", 6)
        try:
            max_retries = int(max_retries)
        except Exception:
            max_retries = 6
        max_retries = max(1, max_retries)

        max_input_chars = kwargs.get("max_input_chars")
        if max_input_chars is None:
            lower_name = str(model_name or "").lower()
            max_input_chars = 1200 if "minilm" in lower_name else 6000
        try:
            max_input_chars = int(max_input_chars)
        except Exception:
            max_input_chars = 6000
        max_input_chars = max(1, max_input_chars)

        min_input_chars = kwargs.get("min_input_chars", 200)
        try:
            min_input_chars = int(min_input_chars)
        except Exception:
            min_input_chars = 200
        min_input_chars = max(32, min_input_chars)

        input_text = str(text or "").strip()
        if len(input_text) > max_input_chars:
            input_text = input_text[:max_input_chars]

        ensure_local_no_proxy(url)

        for _ in range(max_retries):
            payload = {"model": model_name, "input": input_text}
            with requests.Session() as session:
                if is_local_ollama_url(url):
                    session.trust_env = False
                response = session.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {}) if isinstance(data, dict) else {}

                embedding = None
                if isinstance(data, dict):
                    rows = data.get("data", [])
                    if isinstance(rows, list) and rows:
                        first = rows[0] if isinstance(rows[0], dict) else {}
                        embedding = first.get("embedding")
                    if embedding is None:
                        embedding = data.get("embedding")

                if embedding:
                    vec = np.array(embedding, dtype=np.float32)
                    if vec.size:
                        _set_cached_embedding_dim(embedding_host, model_name, int(vec.size))
                    EMBEDDING_USAGE_EVENTS.append(
                        {
                            "timestamp": get_timestamp(),
                            "model": model_name,
                            "prompt_chars": len(input_text),
                            "usage": usage if isinstance(usage, dict) else {},
                        }
                    )
                    return normalize_vector(vec).tolist()
                break

            message = _extract_embedding_error_message(response)
            if response.status_code in (400, 413) and _is_context_length_error(message) and len(input_text) > min_input_chars:
                input_text = input_text[: max(min_input_chars, len(input_text) // 2)]
                continue
            break

        return _zero_embedding(embedding_host, model_name)
    except Exception as e:
        logging.error(f"❌ Embedding Generation Error: {e}")
        return _zero_embedding(embedding_host, model_name)

# ---------------------------------------------------------
# 占位函数：供各组件统一调用，防止导入循环
# ---------------------------------------------------------
def generate_page_meta_info(*args, **kwargs) -> Dict[str, Any]:
    return {"summary": "Standard memory unit", "keywords": [], "G": 0.5}

def gpt_user_profile_analysis(*args, **kwargs): return "Updating profile..."
def gpt_knowledge_extraction(*args, **kwargs): return {}
