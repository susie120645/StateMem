#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def template_family(question: str) -> str:
    normalized = str(question or "").lower()
    if "sparkly thing" in normalized or "what's that thing" in normalized:
        return "object_recall"
    if "how old" in normalized:
        return "attribute_recall_age"
    if "living next door" in normalized:
        return "neighbor_lookup"
    if "remember when joey had a roommate" in normalized and "who was going out with that roommate" in normalized:
        return "chained_roommate_to_dating"
    if "bunking with" in normalized:
        return "roommate_lookup"
    if (
        "girlfriend was back" in normalized
        or "who was the guy" in normalized
        or "date on the night" in normalized
    ):
        return "direct_dating_lookup"
    return "other"


def citation_status(row: Dict[str, Any]) -> str:
    citations = list(row.get("citations") or [])
    if not citations:
        return "missing"
    if len(citations) == 1:
        return "single"
    return "multi"


def _citation_rank(status: str) -> int:
    if status == "single":
        return 0
    if status == "multi":
        return 1
    return 2


def prepare_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["template_family"] = template_family(str(row.get("question", "")))
        enriched["citation_status"] = citation_status(row)
        prepared.append(enriched)
    return prepared


def build_minimal_slice(rows: Iterable[Dict[str, Any]], *, max_per_template: int = 2) -> List[Dict[str, Any]]:
    prepared = prepare_rows(rows)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in prepared:
        buckets.setdefault(str(row.get("template_family", "other")), []).append(row)

    selected: List[Dict[str, Any]] = []
    for template in sorted(buckets):
        candidates = list(buckets[template])
        candidates.sort(
            key=lambda row: (
                _citation_rank(str(row.get("citation_status", ""))),
                -float(row.get("support_margin", 0.0) or 0.0),
                str(row.get("qtype", "")),
                str(row.get("episode", "")),
                int(row.get("session", 0) or 0),
            )
        )
        selected.extend(candidates[:max_per_template])

    selected.sort(
        key=lambda row: (
            str(row.get("template_family", "")),
            str(row.get("qtype", "")),
            str(row.get("episode", "")),
            int(row.get("session", 0) or 0),
        )
    )
    return selected


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "template_family",
                "qtype",
                "episode",
                "session",
                "date",
                "true_label",
                "support_best_choice",
                "support_margin",
                "citation_status",
                "question",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "template_family": row.get("template_family", ""),
                    "qtype": row.get("qtype", ""),
                    "episode": row.get("episode", ""),
                    "session": row.get("session", ""),
                    "date": row.get("date", ""),
                    "true_label": row.get("true_label", ""),
                    "support_best_choice": row.get("support_best_choice", ""),
                    "support_margin": row.get("support_margin", 0.0),
                    "citation_status": row.get("citation_status", ""),
                    "question": row.get("question", ""),
                }
            )


def write_markdown(rows: Iterable[Dict[str, Any]], path: Path, *, source_path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    template_counts = Counter(str(row.get("template_family", "")) for row in rows)
    qtype_counts = Counter(str(row.get("qtype", "")) for row in rows)
    citation_counts = Counter(str(row.get("citation_status", "")) for row in rows)

    lines: List[str] = []
    lines.append("# DialSim Minimal Answer-Side A/B Slice\n")
    lines.append(f"- Source queue: `{source_path}`")
    lines.append(f"- Slice size: `{len(rows)}`")
    lines.append(f"- Template counts: `{dict(template_counts)}`")
    lines.append(f"- QType counts: `{dict(qtype_counts)}`")
    lines.append(f"- Citation counts: `{dict(citation_counts)}`")
    lines.append("")
    lines.append("## Selection rule\n")
    lines.append("- Stratified by `template_family`")
    lines.append("- Keep at most `2` rows per template")
    lines.append("- Prefer cited rows, then higher support margin")
    lines.append("")
    lines.append("## Cases\n")
    lines.append("| Template | QType | Gold | Support | Margin | Cite | Episode | Question |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row.get('template_family', '')}` | `{row.get('qtype', '')}` | "
            f"`{row.get('true_label', '')}` | `{row.get('support_best_choice', '')}` | "
            f"`{row.get('support_margin', 0.0)}` | `{row.get('citation_status', '')}` | "
            f"`{row.get('episode', '')}` | {row.get('question', '')} |"
        )
    lines.append("")
    lines.append("## Intended use\n")
    lines.append("- Use this slice for fast answer-side A/B checks.")
    lines.append("- If a method cannot improve this slice, it is unlikely to improve the full clean answer-side queue.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a minimal DialSim answer-side A/B slice.")
    parser.add_argument("input_jsonl", help="Path to the clean answer-side JSONL queue.")
    parser.add_argument("--max-per-template", type=int, default=2)
    parser.add_argument("--out-prefix", default="reports/dialsim_answer_side_slice_minimal_20260513")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    rows = load_rows(input_path)
    selected = build_minimal_slice(rows, max_per_template=max(1, int(args.max_per_template)))

    out_prefix = Path(args.out_prefix).expanduser().resolve()
    write_jsonl(selected, out_prefix.with_suffix(".jsonl"))
    write_csv(selected, out_prefix.with_suffix(".csv"))
    write_markdown(selected, out_prefix.with_suffix(".md"), source_path=input_path)

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "selected_count": len(selected),
                "out_jsonl": str(out_prefix.with_suffix(".jsonl")),
                "out_csv": str(out_prefix.with_suffix(".csv")),
                "out_md": str(out_prefix.with_suffix(".md")),
                "template_counts": dict(Counter(str(row.get("template_family", "")) for row in selected)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
