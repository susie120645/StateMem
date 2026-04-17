import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

@dataclass
class EvalMetadata:
    G_base: float = 0.5
    C_base: float = 1.0
    L_base: float = 0.1
    access_count: int = 0
    conflict_count: int = 0
    last_access_time: float = field(default_factory=time.time)

class EvaluationController:
    def __init__(self, memory_os_instance):
        self.mos = memory_os_instance
        self.sp = self.mos.short_term_memory
        self.fp = self.mos.forget_memory

    def on_new_input(self, user_input: str, agent_response: str) -> str:
        # G0 Heuristic
        g0_score = 0.3
        text = user_input.lower()
        if len(text) > 10: g0_score += 0.2
        if "python" in text or "code" in text: g0_score += 0.3
        
        print(f"   [Evaluation] G0 Score: {g0_score:.2f}")
        meta = asdict(EvalMetadata(G_base=g0_score))
        
        if g0_score > 0.4:
            # Call original memoryos add
            self.mos.original_add_memory(user_input, agent_response, meta_data=meta)
            return "ACCEPTED_SP"
        else:
            # Call forget pool
            self.fp.add(user_input=user_input, agent_response=agent_response, metadata=meta)
            return "DISCARDED_TO_FP"

    def on_retrieve(self, query: str, top_k: int = 5):
        print(f"   [Evaluation] Calculating G1 for query: '{query[:20]}...'")
        # Reuse existing retriever
        raw_results = self.mos.retriever.retrieve_context(query, self.mos.user_id)
        
        # Simulating G1 re-ranking simply for demo
        raw_pages = raw_results.get("retrieved_pages", [])
        if raw_pages:
            print(f"   -> [Evaluation] Re-ranking {len(raw_pages)} candidates...")
            for p in raw_pages:
                p['_debug_G1'] = 0.9 # Fake score
        
        return raw_results
