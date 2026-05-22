#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _label(text: Any) -> str:
    match = re.search(r"\(([A-Ea-e])\)", str(text or ""))
    return match.group(1).upper() if match else ""


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_report(report_dir: Path, case_id: str, prefix: str) -> Path | None:
    pattern = f"dialsim_friends_remem-qwen3-8b_{prefix}_{case_id}_"
    matches = sorted(report_dir.glob(f"{pattern}*.json"))
    return matches[-1] if matches else None


def summarize_case(report_path: Path) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = (payload.get("results") or [{}])[0]
    support = result.get("choice_support") or {}
    return {
        "report_path": str(report_path),
        "accuracy": payload.get("accuracy"),
        "true_label": result.get("true_label", ""),
        "pred": result.get("pred", ""),
        "pred_label": _label(result.get("pred")),
        "result": result.get("result", ""),
        "support_best_choice": support.get("best_choice", ""),
        "support_best_label": _label(support.get("best_choice")),
        "support_margin": support.get("support_margin", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a prefix-preserving DialSim case batch.")
    parser.add_argument("manifest_json", help="Path to the case manifest.json.")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--run-tag-prefix", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_json).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    run_tag_prefix = str(args.run_tag_prefix).strip()

    manifest = load_manifest(manifest_path)
    rows: List[Dict[str, Any]] = []
    for item in manifest:
        case_id = str(item["case_id"])
        report_path = find_report(report_dir, case_id, run_tag_prefix)
        summary = {
            "case_id": case_id,
            "template_family": item.get("template_family", ""),
            "qtype": item.get("qtype", ""),
            "question": item.get("question", ""),
            "true_label": item.get("true_label", ""),
            "report_found": bool(report_path),
        }
        if report_path is not None:
            summary.update(summarize_case(report_path))
        rows.append(summary)

    found = [row for row in rows if row.get("report_found")]
    finished = len(found)
    correct = sum(1 for row in found if row.get("result") == "Correct")
    pred_counts = Counter(row.get("pred_label", "") for row in found)
    support_counts = Counter(row.get("support_best_label", "") for row in found)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "run_tag_prefix": run_tag_prefix,
                "total_cases": len(rows),
                "finished_cases": finished,
                "finished_accuracy": (correct / finished) if finished else 0.0,
                "pred_label_counts": dict(pred_counts),
                "support_best_label_counts": dict(support_counts),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
