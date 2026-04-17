import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

try:
    from .utils import (
        ensure_directory_exists,
        fallback_extract_keywords,
        get_timestamp,
        normalize_keyword_values,
    )
except ImportError:
    from utils import ensure_directory_exists, fallback_extract_keywords, get_timestamp, normalize_keyword_values


class ShortTermMemory:
    def __init__(self, file_path: str, max_capacity: int = 20):
        """
        [STM: 短期/工作记忆池]
        存储完整的对话原文。当容量满时，由 Updater 触发 G2 评分分流至 LTM 或 ForgetMemory。
        """
        self.max_capacity = max_capacity
        self.file_path = file_path
        self.memory: List[Dict[str, Any]] = []
        ensure_directory_exists(self.file_path)
        self.load()

    def _extract_keywords(self, text: str) -> List[str]:
        extractor = getattr(self, "keyword_extractor", None)
        if callable(extractor):
            try:
                keywords = normalize_keyword_values(extractor(text), limit=32)
                if keywords:
                    return keywords
            except Exception:
                pass
        return fallback_extract_keywords(text, limit=32)

    def _prepare_page_keywords(self, qa_pair: Dict[str, Any]) -> None:
        metadata = qa_pair.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        qa_pair["metadata"] = metadata

        keywords = normalize_keyword_values(
            qa_pair.get("keywords") or metadata.get("keywords") or [],
            limit=32,
        )
        if not keywords:
            combined = f"{qa_pair.get('user_input', '')}\n{qa_pair.get('agent_response', '')}".strip()
            keywords = self._extract_keywords(combined)
        qa_pair["keywords"] = list(keywords)
        metadata["keywords"] = list(keywords)

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    def add_qa_pair(self, qa_pair: Dict[str, Any]):
        """
        别名接口：配合 memoryos.py 的调用逻辑。
        """
        self.add_page(qa_pair)

    def add_page(self, qa_pair: Dict[str, Any]):
        """
        Improved add_page method to ensure data integrity and prevent duplicates.
        """
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            qa_pair["timestamp"] = get_timestamp()

        if any(p.get('page_id') == qa_pair.get('page_id') for p in self.memory):
            logging.warning(f"Duplicate page_id detected: {qa_pair.get('page_id')}")
            return

        self._prepare_page_keywords(qa_pair)
        self.memory.append(qa_pair)
        if len(self.memory) > self.max_capacity:
            self.memory.pop(0)
        self.save()

    def remove_page(self, page_id: str):
        """物理移除记忆"""
        initial_len = len(self.memory)
        self.memory = [p for p in self.memory if p.get("page_id") != page_id]
        if len(self.memory) < initial_len:
            self.save()

    def evict(self, page_id: str, to_forget_pool: bool = False, forget_memory=None) -> Optional[Dict[str, Any]]:
        """
        Backward-compatible eviction entrypoint used by evaluation and
        background flow logic.
        """
        page = self.get_page_by_id(page_id)
        if not page:
            return None

        archived = dict(page)
        self.remove_page(page_id)
        if to_forget_pool and forget_memory is not None:
            try:
                forget_memory.add_page(archived)
            except Exception as exc:
                logging.warning("ForgetMemory archive failed during STM eviction for %s: %s", page_id, exc)
        return archived

    def get_all(self) -> List[Dict[str, Any]]:
        """
        别名接口：配合 updater.py 的调用逻辑。
        """
        return self.memory

    def get_all_pages(self) -> List[Dict[str, Any]]:
        return self.memory

    def get_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        兼容性别名，用于 Retriever 从 LTM 索引反查原文。
        """
        return self.get_page_by_id(page_id)

    def get_page_by_id(self, page_id: str) -> Optional[Dict[str, Any]]:
        """精确查找页面"""
        for p in self.memory:
            if p.get("page_id") == page_id:
                return p
        return None

    def is_full(self) -> bool:
        return len(self.memory) >= self.max_capacity

    def calculate_g0(self, user_input: str, agent_response: str) -> float:
        """
        [G0 粗筛算法]
        对应图片逻辑：引入实体密度 (Entity Density) 权重。
        """
        content = f"{user_input} {agent_response}"
        content_lower = content.lower()

        if re.search(r"fact:|supporting\s+evidence|diary|remember|重要|关键", content_lower):
            semantic_base = 0.85
        elif re.search(r"context:|background|note|背景", content_lower):
            semantic_base = 0.55
        else:
            semantic_base = 0.30

        words = content.split()
        if not words:
            return 0.1

        entities = [
            w
            for i, w in enumerate(words)
            if len(w) > 0 and ((i > 0 and w[0].isupper()) or any(c.isdigit() for c in w))
        ]

        density_ratio = len(entities) / len(words)
        density_score = 1 / (1 + math.exp(-10 * (density_ratio - 0.15)))
        length_penalty = min(len(content) / 500, 1.0) * 0.2

        g0 = (semantic_base * 0.4) + (density_score * 0.4) + length_penalty
        return round(max(0.1, min(g0, 0.98)), 2)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        [G1 粗筛检索] 优先使用已保存 keywords，其次才回退到表层词命中。
        """
        if not query:
            return []

        query_keywords = set(self._extract_keywords(query))
        query_terms = self._tokenize_text(query)
        scored_results = []

        for item in self.memory:
            content = f"{item.get('user_input', '')} {item.get('agent_response', '')}"
            content_terms = self._tokenize_text(content)
            record_keywords = set(
                normalize_keyword_values(
                    item.get('keywords')
                    or (item.get('metadata', {}) if isinstance(item.get('metadata', {}), dict) else {}).get('keywords', []),
                    limit=32,
                )
            )

            keyword_score = 0.0
            if query_keywords and record_keywords:
                keyword_score = len(query_keywords & record_keywords) / float(len(query_keywords))

            surface_score = 0.0
            if query_terms and content_terms:
                surface_score = len(query_terms & content_terms) / float(len(query_terms))

            score = (keyword_score * 0.7) + (surface_score * 0.3)
            if score <= 0.0:
                continue

            candidate = dict(item)
            candidate['score'] = round(score, 4)
            candidate['keyword_score'] = round(keyword_score, 4)
            candidate['surface_score'] = round(surface_score, 4)
            scored_results.append((score, candidate))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results[:k]]

    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [STM] Save error: {e}")

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.memory = data
            except Exception as e:
                print(f"⚠️ [STM] Load error: {e}")
                self.memory = []

    def clear(self):
        self.memory = []
        self.save()
