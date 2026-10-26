#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets"


def _extract_run_id(path: Path) -> int:
    match = re.search(r"(\d{10})", path.stem)
    return int(match.group(1)) if match else -1


def _latest_path(paths: Iterable[Path]) -> Optional[Path]:
    candidates = sorted(paths, key=_extract_run_id, reverse=True)
    return candidates[0] if candidates else None


def _largest_result_report(paths: Iterable[Path]) -> Optional[Path]:
    best: Optional[Tuple[int, int, Path]] = None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result_count = len(payload.get("results") or [])
        except Exception:
            continue
        candidate = (result_count, _extract_run_id(path), path)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def _parse_transcript_lines(text: str) -> List[Tuple[str, str]]:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^Q:\s*", "", cleaned)
    cleaned = re.sub(r"^\[Meeting:[^\]]+\]\s*", "", cleaned)
    lines: List[Tuple[str, str]] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        speaker, content = line.split(":", 1)
        speaker = speaker.strip()
        content = content.strip()
        if speaker and content:
            lines.append((speaker, content))
    return lines


def _recover_qmsum(output_root: Path) -> Dict[str, object]:
    report_path = _largest_result_report(REPO_ROOT.glob("reports/qmsum_*.json"))
    memory_graph_paths = sorted(
        REPO_ROOT.glob("fast_eval_storage/qmsum_statemem/statemem-qwen3-8b/test_*/users/*/long_term/memory_graph.json"),
        key=lambda path: _extract_run_id(path.parent.parent.parent.parent),
        reverse=True,
    )
    if report_path is None or not memory_graph_paths:
        raise FileNotFoundError("No cached QMSum report/storage artifacts found.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = [row for row in (report.get("results") or []) if row.get("meeting_id") and row.get("query")]

    turns_by_meeting: Dict[str, Dict[int, Tuple[str, str]]] = defaultdict(dict)
    memory_sources: Dict[str, str] = {}
    for memory_graph_path in memory_graph_paths:
        memory_graph = json.loads(memory_graph_path.read_text(encoding="utf-8"))
        for node in memory_graph.get("knowledge_base", []) or []:
            metadata = node.get("metadata") or {}
            meeting_id = str(metadata.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            memory_sources.setdefault(meeting_id, str(memory_graph_path))
            try:
                span_start = int(metadata.get("span_start", 0))
            except Exception:
                span_start = 0
            for offset, turn in enumerate(_parse_transcript_lines(node.get("raw_content") or "")):
                turns_by_meeting[meeting_id].setdefault(span_start + offset, turn)

    grouped_queries: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: {"general": [], "specific": []})
    for row in results:
        meeting_id = str(row.get("meeting_id") or "").strip()
        if not meeting_id:
            continue
        query_type = str(row.get("query_type") or "specific").strip().lower()
        if query_type not in {"general", "specific"}:
            query_type = "specific"
        grouped_queries[meeting_id][query_type].append(
            {
                "query": str(row.get("query") or "").strip(),
                "answer": str(row.get("reference") or "").strip(),
            }
        )

    output_path = output_root / "qmsum" / "test.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meeting_count = 0
    query_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for meeting_id in sorted(grouped_queries):
            transcript_turns = turns_by_meeting.get(meeting_id, {})
            meeting_transcripts = [
                {"speaker": speaker, "content": content}
                for _, (speaker, content) in sorted(transcript_turns.items())
            ]
            if not meeting_transcripts:
                continue
            payload = {
                "meeting_id": meeting_id,
                "meeting_transcripts": meeting_transcripts,
                "general_query_list": grouped_queries[meeting_id]["general"],
                "specific_query_list": grouped_queries[meeting_id]["specific"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            meeting_count += 1
            query_count += len(grouped_queries[meeting_id]["general"]) + len(grouped_queries[meeting_id]["specific"])

    return {
        "path": str(output_path),
        "meetings": meeting_count,
        "queries": query_count,
        "report_source": str(report_path),
        "memory_sources": memory_sources,
    }


def _recover_dialsim_friends(output_root: Path) -> Dict[str, object]:
    report_path = _largest_result_report(REPO_ROOT.glob("reports/dialsim_friends_*.json"))
    report_run_id = _extract_run_id(report_path) if report_path is not None else -1
    storage_root = None
    if report_run_id >= 0:
        candidate = REPO_ROOT / "fast_eval_storage" / "dialsim_statemem" / "statemem-qwen3-8b" / f"friends_{report_run_id}"
        if candidate.exists():
            storage_root = candidate
    if storage_root is None:
        storage_root = _latest_path(REPO_ROOT.glob("fast_eval_storage/dialsim_statemem/statemem-qwen3-8b/friends_*"))
    if report_path is None or storage_root is None:
        raise FileNotFoundError("No cached DialSim friends report/storage pair found.")

    memory_graph_paths = list(storage_root.glob("users/*/long_term/memory_graph.json"))
    if not memory_graph_paths:
        raise FileNotFoundError(f"No cached DialSim memory graph found under {storage_root}")
    memory_graph_path = memory_graph_paths[0]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results") or []

    memory_graph = json.loads(memory_graph_path.read_text(encoding="utf-8"))
    chunks_by_doc: Dict[str, List[Tuple[int, str, Dict[str, object]]]] = defaultdict(list)
    for node in memory_graph.get("knowledge_base", []) or []:
        metadata = node.get("metadata") or {}
        doc_id = str(metadata.get("doc_id") or node.get("doc_id") or "").strip()
        if not doc_id:
            continue
        chunk_id = str(node.get("chunk_id") or node.get("id") or "").strip()
        match = re.search(r"_c(\d+)$", chunk_id)
        chunk_index = int(match.group(1)) if match else 0
        chunks_by_doc[doc_id].append((chunk_index, str(node.get("raw_content") or ""), metadata))

    scripts_by_session: Dict[Tuple[str, int, str], str] = {}
    for doc_id, chunks in chunks_by_doc.items():
        del doc_id
        chunks.sort(key=lambda item: item[0])
        metadata = chunks[0][2] if chunks else {}
        episode = str(metadata.get("episode") or "").strip()
        try:
            session = int(metadata.get("session", 0) or 0)
        except Exception:
            session = 0
        date = str(metadata.get("date") or "").strip()
        if not episode or not session:
            continue
        full_text = "".join(chunk_text for _, chunk_text, _ in chunks)
        full_text = re.sub(r"^Q:\s*", "", full_text)
        full_text = re.sub(
            r"^\[Episode:\s*.*?,\s*Session:\s*\d+,\s*Date:\s*.*?\]\s*",
            "",
            full_text,
            count=1,
            flags=re.S,
        )
        scripts_by_session[(episode, session, date)] = full_text.strip()

    easy_prefixes = {
        "easy_qs_ans_w_time",
        "easy_qs_ans_wo_time",
        "easy_qs_before_event_unans",
        "easy_qs_dont_know_unans",
        "easy_qs_dont_know_unans_time",
    }
    rows_by_session: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for row in results:
        episode = str(row.get("episode") or "").strip()
        try:
            session = int(row.get("session", 0) or 0)
        except Exception:
            session = 0
        date = str(row.get("date") or "").strip()
        key = (episode, session, date)
        script_text = scripts_by_session.get(key, "")
        if not episode or not session or not script_text:
            continue
        bucket = rows_by_session.setdefault(
            key,
            {
                "Episode": episode,
                "Session": session,
                "Date": date,
                "Script": script_text,
            },
        )
        qtype = str(row.get("qtype") or "").strip()
        if not qtype:
            continue
        bucket.setdefault(f"{qtype}_questions", []).append({"default": str(row.get("question") or "").strip()})
        bucket.setdefault(f"{qtype}_options", []).append(list(row.get("options") or []))
        bucket.setdefault(f"{qtype}_answers", []).append(str(row.get("answer_text") or "").strip())
        if qtype in easy_prefixes:
            idx_list = bucket.setdefault(f"{qtype}_idxes", [])
            if isinstance(idx_list, list):
                idx_list.append(len(idx_list))

    output_path = output_root / "dialsim" / "friends" / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(rows_by_session):
            handle.write(json.dumps(rows_by_session[key], ensure_ascii=False))
            handle.write("\n")

    return {
        "path": str(output_path),
        "sessions": len(rows_by_session),
        "questions": len(results),
        "report_source": str(report_path),
        "memory_source": str(memory_graph_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover local benchmark datasets from cached statemem benchmark artifacts.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Repeatable. Supported: qmsum, dialsim-friends. Default: both.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = [str(item).strip().lower() for item in args.dataset if str(item).strip()] or ["qmsum", "dialsim-friends"]
    invalid = [item for item in selection if item not in {"qmsum", "dialsim-friends"}]
    if invalid:
        raise SystemExit(f"Unsupported --dataset values: {', '.join(sorted(invalid))}")

    output_root = Path(str(args.output_root)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {}
    if "qmsum" in selection:
        manifest["qmsum"] = _recover_qmsum(output_root)
    if "dialsim-friends" in selection:
        manifest["dialsim-friends"] = _recover_dialsim_friends(output_root)

    manifest_path = output_root / "benchmark_manifest_recovered.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Recovered datasets:")
    for name, payload in manifest.items():
        print(f"- {name}: {json.dumps(payload, ensure_ascii=False)}")
    print(f"- manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
