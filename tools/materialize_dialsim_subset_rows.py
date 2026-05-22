#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


HARD_PREFIXES = [
    "hard_qs_past",
    "hard_qs_cur",
    "hard_qs_fu",
    "hard_qs_past_past",
    "hard_qs_cur_past",
]
EASY_PREFIXES = [
    "easy_qs_ans_w_time",
    "easy_qs_ans_wo_time",
    "easy_qs_before_event_unans",
    "easy_qs_dont_know_unans",
    "easy_qs_dont_know_unans_time",
]


def _extract_default_question(question_entry: Any) -> str:
    if isinstance(question_entry, dict):
        value = question_entry.get("default")
        if isinstance(value, str) and value.strip():
            return value.strip()
        for value in question_entry.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(question_entry, str):
        return question_entry.strip()
    return str(question_entry or "").strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_subset_rows(
    dataset_rows: Iterable[Dict[str, Any]],
    slice_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    dataset_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in dataset_rows:
        key = (str(row.get("Episode", "")), int(row.get("Session", 0) or 0))
        dataset_index[key] = row

    requested: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        key = (str(row.get("episode", "")), int(row.get("session", 0) or 0))
        requested[key].append(row)

    subset_rows: List[Dict[str, Any]] = []
    for key, targets in requested.items():
        if key not in dataset_index:
            raise KeyError(f"Missing dataset row for {key}")
        source = dataset_index[key]
        subset: Dict[str, Any] = {
            "Episode": source.get("Episode", ""),
            "Session": source.get("Session", 0),
            "Date": source.get("Date", ""),
            "Script": source.get("Script", ""),
        }

        for target in targets:
            qtype = str(target.get("qtype", ""))
            questions = source.get(f"{qtype}_questions")
            options = source.get(f"{qtype}_options")
            answers = source.get(f"{qtype}_answers")
            idxes = source.get(f"{qtype}_idxes")
            if not isinstance(questions, list) or not isinstance(options, list) or not isinstance(answers, list):
                raise ValueError(f"Malformed source row for {key} / {qtype}")

            match_index = None
            target_question = str(target.get("question", "")).strip()
            for idx, question_entry in enumerate(questions):
                if _extract_default_question(question_entry) == target_question:
                    match_index = idx
                    break
            if match_index is None:
                raise ValueError(f"Question not found for {key} / {qtype}: {target_question}")

            subset.setdefault(f"{qtype}_questions", []).append(questions[match_index])
            subset.setdefault(f"{qtype}_options", []).append(options[match_index])
            subset.setdefault(f"{qtype}_answers", []).append(answers[match_index])
            if qtype in EASY_PREFIXES:
                if not isinstance(idxes, list):
                    raise ValueError(f"Missing idxes for easy qtype {qtype} in {key}")
                subset.setdefault(f"{qtype}_idxes", []).append(idxes[match_index])

        subset_rows.append(subset)

    subset_rows.sort(key=lambda row: (str(row.get("Episode", "")), int(row.get("Session", 0) or 0)))
    return subset_rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize replayable DialSim subset rows from a slice JSONL.")
    parser.add_argument("dataset_jsonl", help="Path to the source DialSim dataset JSONL.")
    parser.add_argument("slice_jsonl", help="Path to the selected slice JSONL.")
    parser.add_argument(
        "--out",
        default="datasets/dialsim/friends/answer_side_slice_minimal_20260513.rows.jsonl",
        help="Output JSONL path for replayable subset rows.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    slice_path = Path(args.slice_jsonl).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    dataset_rows = load_jsonl(dataset_path)
    slice_rows = load_jsonl(slice_path)
    subset_rows = build_subset_rows(dataset_rows, slice_rows)
    write_jsonl(subset_rows, out_path)

    qtype_counts = Counter()
    for row in subset_rows:
        for prefix in HARD_PREFIXES + EASY_PREFIXES:
            questions = row.get(f"{prefix}_questions")
            if isinstance(questions, list):
                qtype_counts[prefix] += len(questions)

    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "slice_path": str(slice_path),
                "out_path": str(out_path),
                "row_count": len(subset_rows),
                "question_count": sum(qtype_counts.values()),
                "qtype_counts": dict(qtype_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
