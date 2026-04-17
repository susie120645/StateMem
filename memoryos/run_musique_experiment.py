#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


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
    parser = argparse.ArgumentParser(description="Run a focused MuSiQue experiment against MemoryOS.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data" / "musique" / "musique_ans_v1.0_dev.jsonl"),
        help="Path to a MuSiQue JSONL file.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Dataset start index.")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to evaluate. 0 means all remaining.")
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "musique" / "musique_experiment_report.json"),
        help="Where to write the experiment report JSON.",
    )
    parser.add_argument(
        "--storage-root",
        default=str(PROJECT_ROOT / "musique_eval_storage"),
        help="Root directory for per-sample MemoryOS storage.",
    )
    parser.add_argument("--ollama-base", default="http://localhost:11434/v1", help="OpenAI-compatible Ollama base URL.")
    parser.add_argument("--llm-model", default="qwen2.5:7b", help="LLM model name.")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model name.")
    parser.add_argument("--short-term-capacity", type=int, default=12, help="Short-term memory capacity.")
    parser.add_argument("--max-results", type=int, default=8, help="Retriever top-k for get_response.")
    parser.add_argument("--token-budget", type=int, default=2200, help="Retriever token budget for get_response.")
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=0,
        help="Limit how many paragraphs are ingested per sample. 0 means all paragraphs.",
    )
    parser.add_argument(
        "--supporting-only",
        action="store_true",
        help="Ingest only paragraphs marked as supporting.",
    )
    parser.add_argument("--report-every", type=int, default=1, help="Print progress every N samples.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"MuSiQue dataset is empty: {path}")
    return records


def slice_samples(dataset: List[Dict[str, Any]], start_index: int, count: int) -> List[Dict[str, Any]]:
    if start_index < 0 or start_index >= len(dataset):
        raise IndexError(f"start-index {start_index} out of range for dataset size {len(dataset)}")
    subset = dataset[start_index:]
    if count > 0:
        subset = subset[:count]
    return subset


def iter_paragraphs(item: Dict[str, Any], supporting_only: bool, max_paragraphs: int) -> Iterable[Dict[str, Any]]:
    paragraphs = item.get("paragraphs", []) or []
    emitted = 0
    for paragraph in paragraphs:
        if supporting_only and not paragraph.get("is_supporting"):
            continue
        yield paragraph
        emitted += 1
        if max_paragraphs > 0 and emitted >= max_paragraphs:
            break


def build_paragraph_content(paragraph: Dict[str, Any]) -> str:
    title = str(paragraph.get("title", "")).strip()
    text = str(paragraph.get("paragraph_text", "")).strip()
    idx = paragraph.get("idx")
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if idx is not None:
        parts.append(f"Paragraph Index: {idx}")
    if text:
        parts.append(f"Paragraph: {text}")
    return "\n".join(parts).strip()


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


def canonical_musique_evidence_id(paragraph_idx: Any) -> str:
    return f"paragraph::{str(paragraph_idx).strip()}"


def extract_musique_paragraph_idx(page: Dict[str, Any]) -> str:
    metadata = _page_metadata(page or {})
    paragraph_idx = metadata.get("paragraph_idx")
    if paragraph_idx not in {None, ""}:
        return str(paragraph_idx).strip()
    text = "\n".join(
        [
            str(page.get("raw_content", "")),
            str(page.get("agent_response", "")),
            str(page.get("summary", "")),
        ]
    )
    match = re.search(r"(?m)^Paragraph Index:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


def get_musique_gold_evidence_ids(item: Dict[str, Any]) -> List[str]:
    evidence_ids: List[str] = []
    for paragraph in item.get("paragraphs", []) or []:
        if paragraph.get("is_supporting") and paragraph.get("idx") not in {None, ""}:
            evidence_ids.append(canonical_musique_evidence_id(paragraph.get("idx")))
    return evidence_ids


def get_musique_predicted_evidence_ids(citations: List[str], pages: List[Dict[str, Any]]) -> List[str]:
    evidence_ids: List[str] = []
    page_lookup = _page_lookup(pages)
    for citation_id in citations or []:
        paragraph_idx = extract_musique_paragraph_idx(page_lookup.get(str(citation_id), {}))
        if paragraph_idx:
            evidence_ids.append(canonical_musique_evidence_id(paragraph_idx))
    return evidence_ids


def ingest_musique_paragraphs(memo: Memoryos, item: Dict[str, Any], args: argparse.Namespace) -> int:
    ingested = 0
    for paragraph in iter_paragraphs(item, supporting_only=args.supporting_only, max_paragraphs=args.max_paragraphs):
        title = str(paragraph.get("title", "MuSiQue Context"))
        content = build_paragraph_content(paragraph)
        memo.add_memory(
            user_input=f"MuSiQue paragraph from {title}",
            agent_response=content,
            metadata={
                "topic": "MuSiQue",
                "entities": [title],
                "flags": {"important": True},
                "paragraph_idx": paragraph.get("idx"),
                "is_supporting": bool(paragraph.get("is_supporting")),
            },
        )
        ingested += 1

    memo.updater.process_short_term_to_routing()
    return ingested


def run_single_sample(item: Dict[str, Any], dataset_index: int, args: argparse.Namespace) -> Dict[str, Any]:
    storage_root = Path(args.storage_root).resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    user_id = f"musique_{dataset_index}_{int(time.time())}"

    memo = Memoryos(
        user_id=user_id,
        openai_api_key="ollama",
        openai_base_url=args.ollama_base,
        data_storage_path=str(storage_root),
        llm_model=args.llm_model,
        embedding_model_name=args.embedding_model,
        short_term_capacity=args.short_term_capacity,
    )

    try:
        ingested_paragraphs = ingest_musique_paragraphs(memo, item, args)
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
            gold_evidence_ids=get_musique_gold_evidence_ids(item),
            predicted_evidence_ids=get_musique_predicted_evidence_ids(citations, retrieved_pages),
            citation_support_precision=support_precision_from_nli_preview(citation_preview),
            citation_ids=citations,
            hit_ids=hit_ids,
        )

        return {
            "dataset_index": dataset_index,
            "sample_id": item.get("id", f"sample_{dataset_index}"),
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "answerable": bool(item.get("answerable", True)),
            "em": evaluation["scores"]["answer_em"],
            "f1": evaluation["scores"]["answer_f1"],
            "latency_sec": round(latency_sec, 4),
            "ingested_paragraphs": ingested_paragraphs,
            "supporting_paragraphs": sum(1 for paragraph in item.get("paragraphs", []) if paragraph.get("is_supporting")),
            "retrieved_pages": len(retrieved_pages),
            "citations": citations,
            "hit_ids": hit_ids,
            "reflection": result.get("reflection", {}) or {},
            "question_decomposition": item.get("question_decomposition", []) or [],
            "citation_preview": citation_preview,
            "gold": evaluation["gold"],
            "pred": evaluation["pred"],
            "scores": evaluation["scores"],
        }
    finally:
        memo.shutdown(wait_for_background_tasks=True)


def build_report(dataset_path: Path, args: argparse.Namespace, samples: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_summary = summarize_score_metrics(records)
    return {
        "dataset": str(dataset_path),
        "start_index": args.start_index,
        "count": len(samples),
        "ollama_base": args.ollama_base,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "supporting_only": args.supporting_only,
        "max_paragraphs": args.max_paragraphs,
        "overall": {
            "em": score_summary["answer_em"],
            "f1": score_summary["answer_f1"],
            "latency_sec": round(mean(record["latency_sec"] for record in records), 4) if records else 0.0,
            **score_summary,
        },
        "records": records,
    }


def write_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output_path).resolve()

    dataset = load_jsonl(dataset_path)
    samples = slice_samples(dataset, args.start_index, args.count)

    print("========================================")
    print("MuSiQue Experiment")
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
    print("MuSiQue Summary")
    print("========================================")
    print(f"EM: {report['overall']['em']:.4f}")
    print(f"F1: {report['overall']['f1']:.4f}")
    print(f"Avg Latency: {report['overall']['latency_sec']:.4f}s")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
