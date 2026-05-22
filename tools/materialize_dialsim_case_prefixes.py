#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.materialize_dialsim_subset_rows import EASY_PREFIXES, HARD_PREFIXES, _extract_default_question, load_jsonl


ALL_PREFIXES = HARD_PREFIXES + EASY_PREFIXES


def _parse_dialsim_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(rows)
    rows.sort(
        key=lambda row: (
            _parse_dialsim_date(row.get("Date")) or datetime.min,
            str(row.get("Episode") or ""),
            int(row.get("Session") or 0),
        )
    )
    return rows


def _safe_slug(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip()).strip("_").lower()
    return normalized[:80] or "case"


def strip_questions(row: Dict[str, Any]) -> Dict[str, Any]:
    kept = {
        "Episode": row.get("Episode", ""),
        "Session": row.get("Session", 0),
        "Date": row.get("Date", ""),
        "Script": row.get("Script", ""),
    }
    return kept


def keep_single_question(row: Dict[str, Any], *, qtype: str, question_text: str) -> Dict[str, Any]:
    kept = strip_questions(row)
    questions = row.get(f"{qtype}_questions")
    options = row.get(f"{qtype}_options")
    answers = row.get(f"{qtype}_answers")
    idxes = row.get(f"{qtype}_idxes")
    if not isinstance(questions, list) or not isinstance(options, list) or not isinstance(answers, list):
        raise ValueError(f"Malformed row for {row.get('Episode')} session {row.get('Session')} qtype {qtype}")

    match_index = None
    for idx, question_entry in enumerate(questions):
        if _extract_default_question(question_entry) == question_text:
            match_index = idx
            break
    if match_index is None:
        raise ValueError(f"Question not found: {question_text}")

    kept[f"{qtype}_questions"] = [questions[match_index]]
    kept[f"{qtype}_options"] = [options[match_index]]
    kept[f"{qtype}_answers"] = [answers[match_index]]
    if qtype in EASY_PREFIXES:
        if not isinstance(idxes, list):
            raise ValueError(f"Missing idxes for easy qtype {qtype}")
        kept[f"{qtype}_idxes"] = [idxes[match_index]]
    return kept


def build_case_prefixes(
    dataset_rows: Sequence[Dict[str, Any]],
    slice_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_rows = sort_rows(dataset_rows)
    case_entries: List[Dict[str, Any]] = []

    for case_index, slice_row in enumerate(slice_rows, 1):
        target_episode = str(slice_row.get("episode", ""))
        target_session = int(slice_row.get("session", 0) or 0)
        target_qtype = str(slice_row.get("qtype", ""))
        target_question = str(slice_row.get("question", ""))

        target_row_index = None
        for idx, row in enumerate(ordered_rows):
            if str(row.get("Episode", "")) == target_episode and int(row.get("Session", 0) or 0) == target_session:
                target_row_index = idx
                break
        if target_row_index is None:
            raise KeyError(f"Missing target row for {target_episode} / {target_session}")

        prefix_rows: List[Dict[str, Any]] = []
        for row in ordered_rows[:target_row_index]:
            prefix_rows.append(strip_questions(row))
        prefix_rows.append(
            keep_single_question(
                ordered_rows[target_row_index],
                qtype=target_qtype,
                question_text=target_question,
            )
        )

        slug = _safe_slug(f"{case_index:02d}_{target_episode}_s{target_session}_{target_qtype}")
        case_entries.append(
            {
                "case_id": slug,
                "case_index": case_index,
                "episode": target_episode,
                "session": target_session,
                "date": slice_row.get("date", ""),
                "qtype": target_qtype,
                "question": target_question,
                "true_label": slice_row.get("true_label", ""),
                "support_best_choice": slice_row.get("support_best_choice", ""),
                "template_family": slice_row.get("template_family", ""),
                "row_count": len(prefix_rows),
                "prefix_rows": prefix_rows,
            }
        )

    return case_entries


def write_case_files(case_entries: Sequence[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for case in case_entries:
        case_id = str(case["case_id"])
        path = out_dir / f"{case_id}.rows.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in case["prefix_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest.append(
            {
                key: value
                for key, value in case.items()
                if key != "prefix_rows"
            }
            | {"rows_jsonl": str(path)}
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize prefix-preserving DialSim case files.")
    parser.add_argument("dataset_jsonl", help="Path to the source DialSim dataset JSONL.")
    parser.add_argument("slice_jsonl", help="Path to the selected slice JSONL.")
    parser.add_argument(
        "--out-dir",
        default="datasets/dialsim/friends/answer_side_slice_cases_20260513",
        help="Directory for per-case prefix files and manifest.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    slice_path = Path(args.slice_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    dataset_rows = load_jsonl(dataset_path)
    slice_rows = load_jsonl(slice_path)
    case_entries = build_case_prefixes(dataset_rows, slice_rows)
    manifest_path = write_case_files(case_entries, out_dir)

    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "slice_path": str(slice_path),
                "out_dir": str(out_dir),
                "manifest_path": str(manifest_path),
                "case_count": len(case_entries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
