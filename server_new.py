#!/usr/bin/env python3
import argparse
import builtins
import json
import os
import sys
from typing import Any, Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    original_stdout_fd = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = os.fdopen(original_stdout_fd, "w", encoding="utf-8", buffering=1)
except Exception:
    pass


def clean_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    sys.stderr.write(" ".join(map(str, args)) + kwargs.get("end", "\n"))
    sys.stderr.flush()


builtins.print = clean_print

CURRENT_ROOT = os.path.dirname(os.path.abspath(__file__))
if CURRENT_ROOT not in sys.path:
    sys.path.insert(0, CURRENT_ROOT)

try:
    from mcp.server.fastmcp import FastMCP
    from memoryos.memoryos import Memoryos
except ImportError as exc:
    clean_print(f"Import error: {exc}")
    sys.exit(1)


memoryos_instance: Optional[Memoryos] = None
mcp = FastMCP("MemoryOS-Cluster")


@mcp.tool()
def add_memory(user_input: str, agent_response: str = "ok") -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    try:
        memoryos_instance.add_memory(user_input, agent_response)
        return {
            "status": "success",
            "st_size": len(memoryos_instance.short_term_memory.get_all()),
            "ingest_result": memoryos_instance.get_last_ingest_result(),
            "memory_stats": memoryos_instance.get_memory_stats(),
        }
    except Exception as exc:
        clean_print(f"Add error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def retrieve_memory(query: str, max_results: int = 5, token_budget: int = 2200) -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    try:
        res = memoryos_instance.get_response(query, max_results=max_results, token_budget=token_budget)
        return {
            "status": "success",
            "retrieved_pages": res.get("retrieved_pages", []),
            "source_stats": res.get("source_stats", {}),
            "hit_ids": res.get("hit_ids", []),
            "context_summary": res.get("context_summary", ""),
            "final_answer": res.get("final_answer", ""),
            "citations": res.get("citations", []),
            "thought_process": res.get("thought_process", ""),
            "reflection": res.get("reflection", {}),
            "background_status": memoryos_instance.get_background_status(),
            "st_count": len(memoryos_instance.short_term_memory.get_all()),
        }
    except Exception as exc:
        clean_print(f"Retrieve error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def nli_verify(generated_answer: str, citation_ids: list) -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    missing_nodes = [node_id for node_id in citation_ids if not memoryos_instance.node_exists(node_id)]
    if missing_nodes:
        return {
            "status": "error",
            "message": f"Missing citation nodes: {missing_nodes}",
        }

    try:
        result = memoryos_instance.nli_verify_citations(generated_answer, citation_ids)
        return {"status": "success", "result": result}
    except Exception as exc:
        clean_print(f"NLI verification error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def apply_soft_decay(half_life_days: int = 7) -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    try:
        memoryos_instance.soft_decay_memory(half_life_days=half_life_days)
        return {
            "status": "success",
            "message": "Soft decay applied successfully.",
            "memory_stats": memoryos_instance.get_memory_stats(),
        }
    except Exception as exc:
        clean_print(f"Soft decay error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def pointer_compress(chunk_id: str) -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    try:
        pointer_id = memoryos_instance.pointer_compress(chunk_id)
        if not pointer_id:
            return {"status": "error", "message": "Pointer compression failed"}
        return {"status": "success", "pointer_id": pointer_id}
    except Exception as exc:
        clean_print(f"Pointer compress error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def process_memory_flow(page_ids: List[str], feedback_score: float = 1.0, background: bool = True) -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}

    try:
        result = memoryos_instance.post_inference_memory_flow(
            page_ids=page_ids,
            feedback_score=feedback_score,
            background=background,
        )
        if background:
            return {"status": "success", **result}
        return {"status": "success", "result": result, "background_status": memoryos_instance.get_background_status()}
    except Exception as exc:
        clean_print(f"Process memory flow error: {exc}")
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def get_background_status() -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}
    return {"status": "success", "background_status": memoryos_instance.get_background_status()}


@mcp.tool()
def get_user_profile() -> Dict[str, Any]:
    global memoryos_instance
    if memoryos_instance is None:
        return {"status": "error", "message": "Instance not ready"}
    try:
        return {"status": "success", "user_profile": memoryos_instance.get_user_profile()}
    except Exception as exc:
        clean_print(f"Get user profile error: {exc}")
        return {"status": "error", "message": str(exc)}


def load_config(config_path: str) -> Dict[str, Any]:
    default_config = {
        "user_id": "user",
        "assistant_id": "local_ai",
        "openai_api_key": "ollama",
        "openai_base_url": "http://localhost:11434/v1",
        "llm_model": "qwen2.5:7b",
        "data_storage_path": "./memoryos_data",
        "embedding_model_name": "nomic-embed-text",
        "short_term_capacity": 20,
        "long_term_knowledge_capacity": 5000,
    }
    if not config_path or not os.path.exists(config_path):
        return default_config

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    default_config.update(loaded)
    return default_config


def main():
    global memoryos_instance

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        memoryos_instance = Memoryos(
            user_id=config["user_id"],
            assistant_id=config.get("assistant_id", "local_ai"),
            openai_api_key=config.get("openai_api_key", "ollama"),
            openai_base_url=config.get("openai_base_url"),
            llm_model=config.get("llm_model", "qwen2.5:7b"),
            data_storage_path=config["data_storage_path"],
            embedding_model_name=config.get("embedding_model_name", "nomic-embed-text"),
            short_term_capacity=config.get("short_term_capacity", 20),
            long_term_knowledge_capacity=config.get("long_term_knowledge_capacity", 5000),
        )
        clean_print(f"MemoryOS ready (user={config['user_id']})")
        mcp.run(transport="stdio")
    except Exception as exc:
        import traceback

        clean_print(f"Boot error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
