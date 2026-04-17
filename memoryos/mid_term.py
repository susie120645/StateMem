import json
import numpy as np
from collections import defaultdict
import heapq
import time
import os

try:
    from .utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )
except ImportError:
    from utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )

def ensure_list(vec):
    """确保向量是列表格式，处理 numpy 或 list 的兼容性，防止 JSON 序列化失败"""
    if vec is None: return []
    if isinstance(vec, np.ndarray):
        return vec.tolist()
    return vec

def compute_segment_heat(session, alpha=1.0, beta=1.0, gamma=1.0, tau_hours=24):
    """
    计算 Session 的 A-MEM 热度（代谢指标）
    """
    n_visit = session.get("N_visit", 0)
    details = session.get("details", [])
    l_interaction = session.get("L_interaction", len(details))
    
    r_recency = 1.0 
    if session.get("last_visit_time"):
        r_recency = compute_time_decay(session["last_visit_time"], get_timestamp(), tau_hours)
    
    # 热度公式：αN + β(L/5) + γR
    return alpha * n_visit + beta * (l_interaction / 5.0) + gamma * r_recency

class MidTermMemory:
    def __init__(self, file_path: str, client: OpenAIClient, max_capacity=2000, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 embedding_model_kwargs: dict = None):
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        
        self.client = client
        self.max_capacity = max_capacity
        self.embedding_model_name = embedding_model_name
        
        self.sessions = {} 
        self.access_frequency = defaultdict(int) 
        self.heap = [] 
        self.thresholds = {"G_admission": 0.3, "L_budget": 2048}
        
        self.load()

    # ==========================================
    # --- 核心检索接口 (含修复切片报错逻辑) ---
    # ==========================================

    def search_sessions(self, query: str = None, query_text: str = None, 
                        top_k: int = 3, top_k_sessions: int = None):
        """
        核心检索接口：同时兼容 Retriever.py 的不同参数命名。
        ✅ 修复：强制转换 target_k 为 int，防止 slice indices 报错。
        """
        target_query = query_text if query_text is not None else query
        
        # 🛡️ 强制类型检查，防止字符串切片
        try:
            raw_k = top_k_sessions if top_k_sessions is not None else top_k
            target_k = int(raw_k) if raw_k is not None else 3
        except (TypeError, ValueError):
            target_k = 3
        
        if not target_query: return []

        # 向量化 Query
        query_emb = get_embedding(target_query)
        if query_emb is None: return []
        query_vec = np.array(normalize_vector(query_emb))

        results = []
        for sid, sess in self.sessions.items():
            # 防御：如果没有向量则在线补齐
            if "summary_embedding" not in sess or not sess["summary_embedding"]:
                emb = get_embedding(sess.get("summary", "Empty Summary"))
                sess["summary_embedding"] = ensure_list(normalize_vector(emb))
            
            sess_vec = np.array(sess["summary_embedding"])
            # 计算相似度
            score = float(np.dot(sess_vec, query_vec))
            
            # 更新代谢热度指标
            self.access_frequency[sid] += 1
            sess["N_visit"] = sess.get("N_visit", 0) + 1
            sess["H_segment"] = compute_segment_heat(sess)

            results.append({
                "session_id": sid, 
                "score": score,
                "summary": sess.get("summary", ""), 
                "details": sess.get("details", [])
            })
            
        # 按得分降序排列
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:target_k]

    def search(self, query: str, top_k: int = 5):
        """标准映射接口"""
        return self.search_sessions(query=query, top_k=top_k)

    # ==========================================
    # --- 代谢与维护逻辑 ---
    # ==========================================

    def rebuild_heap(self):
        """重新构建代谢堆"""
        temp_heap = []
        for sid, sess in self.sessions.items():
            heat = sess.get("H_segment", 1.0)
            temp_heap.append((-float(heat), sid))
        heapq.heapify(temp_heap)
        self.heap = temp_heap

    def update_page_connections(self, prev_page_id, next_page_id):
        """建立 Page 间的链式引用关系"""
        if prev_page_id:
            p = self.get_page_by_id(prev_page_id)
            if p: p["next_page"] = next_page_id
        if next_page_id:
            p = self.get_page_by_id(next_page_id)
            if p: p["pre_page"] = prev_page_id

    def get_page_by_id(self, page_id):
        for sess in self.sessions.values():
            for p in sess.get("details", []):
                if p.get("page_id") == page_id: return p
        return None

    def insert_pages_into_session(self, summary_for_new_pages, keywords_for_new_pages, pages_to_insert, 
                                  similarity_threshold=0.6, keyword_similarity_alpha=1.0):
        if not self.sessions:
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

        raw_vec = get_embedding(summary_for_new_pages)
        if raw_vec is None: return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)
        new_vec = np.array(normalize_vector(raw_vec))

        best_sid, best_score = None, -1
        for sid, sess in self.sessions.items():
            if "summary_embedding" not in sess or not sess["summary_embedding"]:
                emb = get_embedding(sess["summary"])
                sess["summary_embedding"] = ensure_list(normalize_vector(emb))
            
            sess_vec = np.array(sess["summary_embedding"])
            sem_sim = float(np.dot(sess_vec, new_vec))
            
            exist_k = set(sess.get("summary_keywords", []) or [])
            new_k = set(keywords_for_new_pages or [])
            k_sim = len(exist_k & new_k) / len(exist_k | new_k) if (exist_k | new_k) else 0
            
            score = sem_sim + keyword_similarity_alpha * k_sim
            if score > best_score:
                best_score, best_sid = score, sid

        if best_sid and best_score >= similarity_threshold:
            target = self.sessions[best_sid]
            for p in pages_to_insert:
                p_text = f"User: {p.get('user_input','')} AI: {p.get('agent_response','')}"
                p["page_embedding"] = ensure_list(normalize_vector(get_embedding(p_text)))
                p["metrics"] = {"G_base": 1.0, "C_base": 0.8, "L_cost": len(p_text)}
                p["page_id"] = p.get("page_id", generate_id("page"))
                target["details"].append(p)
            
            target["L_interaction"] = len(target["details"])
            target["last_visit_time"] = get_timestamp()
            target["H_segment"] = compute_segment_heat(target)
            
            self.rebuild_heap()
            self.save()
            return best_sid
        else:
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

    def add_session(self, summary, details, summary_keywords=None):
        sid = generate_id("session")
        s_vec = ensure_list(normalize_vector(get_embedding(summary)))
        
        processed_pages = []
        for p in details:
            p_text = f"User: {p.get('user_input','')} AI: {p.get('agent_response','')}"
            processed_pages.append({
                **p,
                "page_id": p.get("page_id", generate_id("page")),
                "page_embedding": ensure_list(normalize_vector(get_embedding(p_text))),
                "metrics": {"G_base": 1.0, "C_base": 0.8, "L_cost": len(p_text)}
            })
            
        session_obj = {
            "id": sid, 
            "summary": summary, 
            "summary_keywords": summary_keywords or [],
            "summary_embedding": s_vec, 
            "details": processed_pages,
            "L_interaction": len(processed_pages), 
            "N_visit": 1,
            "last_visit_time": get_timestamp(), 
            "H_segment": 0.0
        }
        
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        self.sessions[sid] = session_obj
        self.access_frequency[sid] = 1
        
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
            
        self.rebuild_heap()
        self.save()
        return sid

    def save(self):
        data = {
            "sessions": self.sessions, 
            "access_frequency": dict(self.access_frequency)
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.access_frequency = defaultdict(int, data.get("access_frequency", {}))
                self.rebuild_heap()
        except Exception as e:
            print(f"⚠️ Load MTM Failed: {e}")

    def evict_lfu(self):
        if not self.access_frequency: return
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        self.sessions.pop(lfu_sid, None)
        self.access_frequency.pop(lfu_sid, None)
        print(f"♻️ Evicted LFU Session: {lfu_sid}")