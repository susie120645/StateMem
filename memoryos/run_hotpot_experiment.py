#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PACKAGE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from memoryos.memoryos import Memoryos
    from memoryos.evaluation.scorer import (
        build_evaluation_result,
        summarize_score_metrics,
        support_precision_from_nli_preview,
    )
except ImportError:
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    from memoryos import Memoryos
    from evaluation.scorer import (
        build_evaluation_result,
        summarize_score_metrics,
        support_precision_from_nli_preview,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused HotpotQA experiment against MemoryOS.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data" / "hotpot" / "hotpot_dev_distractor_v1.json"),
        help="Path to a HotpotQA JSON file.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Dataset start index.")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to evaluate. 0 means all remaining.")
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "hotpot" / "hotpot_experiment_report.json"),
        help="Where to write the experiment report JSON.",
    )
    parser.add_argument(
        "--storage-root",
        default=str(PROJECT_ROOT / "hotpot_eval_storage"),
        help="Root directory for per-sample MemoryOS storage.",
    )
    parser.add_argument("--ollama-base", default="http://localhost:11434/v1", help="OpenAI-compatible Ollama base URL.")
    parser.add_argument("--llm-model", default="qwen2.5:7b", help="LLM model name.")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model name.")
    parser.add_argument("--short-term-capacity", type=int, default=12, help="Short-term memory capacity.")
    parser.add_argument("--max-results", type=int, default=8, help="Retriever top-k for get_response.")
    parser.add_argument("--token-budget", type=int, default=2200, help="Retriever token budget for get_response.")
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=0,
        help="Limit how many context titles from each sample are ingested. 0 means ingest all context items.",
    )
    parser.add_argument("--report-every", type=int, default=1, help="Print progress every N samples.")
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid HotpotQA dataset: {path}")
    return data


def slice_samples(dataset: List[Dict[str, Any]], start_index: int, count: int) -> List[Dict[str, Any]]:
    if start_index < 0 or start_index >= len(dataset):
        raise IndexError(f"start-index {start_index} out of range for dataset size {len(dataset)}")
    subset = dataset[start_index:]
    if count > 0:
        subset = subset[:count]
    return subset


def build_context_block(title: str, sentences: List[str]) -> str:
    joined = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    return f"Title: {title}\nContent: {joined}".strip()


def _page_lookup(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for page in pages or []:
        citation_id = page.get("citation_id") or page.get("page_id")
        if citation_id:
            lookup[str(citation_id)] = page
    return lookup


def _page_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
    metadata = page.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata
    legacy_metadata = page.get("meta_data", {})
    return legacy_metadata if isinstance(legacy_metadata, dict) else {}


def canonical_hotpot_evidence_id(title: Any) -> str:
    return f"title::{str(title or '').strip()}"


def extract_hotpot_title(page: Dict[str, Any]) -> str:
    metadata = _page_metadata(page or {})
    entities = metadata.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            value = str(entity or "").strip()
            if value:
                return value
    text = "\n".join(
        [
            str(page.get("raw_content", "")),
            str(page.get("agent_response", "")),
            str(page.get("summary", "")),
        ]
    )
    match = re.search(r"(?m)^Title:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


def get_hotpot_gold_evidence_ids(item: Dict[str, Any]) -> List[str]:
    evidence_ids: List[str] = []
    for supporting_fact in item.get("supporting_facts", []) or []:
        if not isinstance(supporting_fact, list) or not supporting_fact:
            continue
        title = str(supporting_fact[0] or "").strip()
        if title:
            evidence_ids.append(canonical_hotpot_evidence_id(title))
    return evidence_ids


def get_hotpot_predicted_evidence_ids(citations: List[str], pages: List[Dict[str, Any]]) -> List[str]:
    evidence_ids: List[str] = []
    page_lookup = _page_lookup(pages)
    for citation_id in citations or []:
        title = extract_hotpot_title(page_lookup.get(str(citation_id), {}))
        if title:
            evidence_ids.append(canonical_hotpot_evidence_id(title))
    return evidence_ids


def ingest_hotpot_context(memo: Memoryos, item: Dict[str, Any], max_contexts: int = 0) -> int:
    contexts = item.get("context", []) or []
    if max_contexts > 0:
        contexts = contexts[:max_contexts]

    ingested = 0
    for title, sentences in contexts:
        content = build_context_block(str(title), list(sentences or []))
        memo.add_memory(
            user_input=f"HotpotQA context from {title}",
            agent_response=content,
            metadata={
                "topic": "HotpotQA",
                "entities": [str(title)],
                "flags": {"important": True},
            },
        )
        ingested += 1

    memo.updater.process_short_term_to_routing()
    return ingested


def run_single_sample(item: Dict[str, Any], dataset_index: int, args: argparse.Namespace) -> Dict[str, Any]:
    sample_storage = Path(args.storage_root).resolve()
    sample_storage.mkdir(parents=True, exist_ok=True)
    user_id = f"hotpot_{dataset_index}_{int(time.time())}"

    memo = Memoryos(
        user_id=user_id,
        openai_api_key="ollama",
        openai_base_url=args.ollama_base,
        data_storage_path=str(sample_storage),
        llm_model=args.llm_model,
        embedding_model_name=args.embedding_model,
        short_term_capacity=args.short_term_capacity,
    )

    try:
        ingested_contexts = ingest_hotpot_context(memo, item, max_contexts=args.max_contexts)
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))

        start_time = time.time()
        result = memo.get_response(
            question,
            max_results=args.max_results,
            token_budget=args.token_budget,
        )
        latency_sec = time.time() - start_time
        prediction = str(result)
        citations = result.get("citations", []) or []
        hit_ids = result.get("hit_ids", []) or []
        retrieved_pages = result.get("retrieved_pages", []) or []
        citation_preview = memo.lifecycle.preview_nli_citations(
            prediction,
            citations,
            max_citations=max(1, len(citations)),
        )
        evaluation = build_evaluation_result(
            gold_answer=answer,
            predicted_answer=prediction,
            gold_evidence_ids=get_hotpot_gold_evidence_ids(item),
            predicted_evidence_ids=get_hotpot_predicted_evidence_ids(citations, retrieved_pages),
            citation_support_precision=support_precision_from_nli_preview(citation_preview),
            citation_ids=citations,
            hit_ids=hit_ids,
        )

        record = {
            "dataset_index": dataset_index,
            "sample_id": item.get("_id", f"sample_{dataset_index}"),
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "em": evaluation["scores"]["answer_em"],
            "f1": evaluation["scores"]["answer_f1"],
            "latency_sec": round(latency_sec, 4),
            "ingested_contexts": ingested_contexts,
            "retrieved_pages": len(retrieved_pages),
            "citations": citations,
            "hit_ids": hit_ids,
            "reflection": result.get("reflection", {}) or {},
            "citation_preview": citation_preview,
            "gold": evaluation["gold"],
            "pred": evaluation["pred"],
            "scores": evaluation["scores"],
        }
        return record
    finally:
        memo.shutdown(wait_for_background_tasks=True)


def write_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def build_report(
    dataset_path: Path,
    args: argparse.Namespace,
    samples: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score_summary = summarize_score_metrics(records)
    return {
        "dataset": str(dataset_path),
        "start_index": args.start_index,
        "count": len(samples),
        "ollama_base": args.ollama_base,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "overall": {
            "em": score_summary["answer_em"],
            "f1": score_summary["answer_f1"],
            "latency_sec": round(mean(record["latency_sec"] for record in records), 4) if records else 0.0,
            **score_summary,
        },
        "records": records,
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output_path).resolve()

    dataset = load_dataset(dataset_path)
    samples = slice_samples(dataset, args.start_index, args.count)

    print("========================================")
    print("HotpotQA Experiment")
    print("========================================")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(samples)} (from index {args.start_index})")
    print(f"Storage root: {Path(args.storage_root).resolve()}")
    print(f"Ollama base: {args.ollama_base}")
    print(f"LLM model: {args.llm_model}")
    print(f"Embedding model: {args.embedding_model}")

    records: List[Dict[str, Any]] = []
    for offset, item in enumerate(samples, start=args.start_index):
        record = run_single_sample(item=item, dataset_index=offset, args=args)
        records.append(record)
        write_report(output_path, build_report(dataset_path, args, samples, records))

        if len(records) % max(1, args.report_every) == 0:
            current_em = mean(record_item["em"] for record_item in records)
            current_f1 = mean(record_item["f1"] for record_item in records)
            print(
                f"[{offset}] EM={record['em']:.0f} F1={record['f1']:.4f} "
                f"avg_EM={current_em:.4f} avg_F1={current_f1:.4f} "
                f"latency={record['latency_sec']:.2f}s"
            )

    report = build_report(dataset_path, args, samples, records)
    write_report(output_path, report)

    print("\n========================================")
    print("HotpotQA Summary")
    print("========================================")
    print(f"EM: {report['overall']['em']:.4f}")
    print(f"F1: {report['overall']['f1']:.4f}")
    print(f"Avg Latency: {report['overall']['latency_sec']:.4f}s")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
