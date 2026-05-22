#!/usr/bin/env python3
"""
Summarize latency stats from Remem-backed RULER prediction jsonl files.

Reads `third_party/RULER/scripts/benchmark_root/<model>/<taskset>/<seq_len>/pred/*.jsonl`
and aggregates `others.remem.{ingest_ms,retrieve_ms,answer_ms}`.

Example:
  .venv_ruler/bin/python tools/summarize_ruler_remem_latency.py \
    --model remem-qwen3-8b \
    --taskset synthetic_noqa \
    --out reports/ruler_remem-qwen3-8b_latency_20260501.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "third_party" / "RULER" / "scripts" / "benchmark_root"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _p(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Remem latency from RULER pred jsonl.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--taskset", required=True)
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    root = BENCHMARK_ROOT / str(args.model) / str(args.taskset)
    if not root.exists():
        raise SystemExit(f"Missing directory: {root}")

    rows: List[Dict[str, Any]] = []
    for length_dir in sorted(root.iterdir()):
        if not length_dir.is_dir():
            continue
        try:
            seq_len = int(length_dir.name)
        except Exception:
            continue
        pred_dir = length_dir / "pred"
        if not pred_dir.exists():
            continue
        for jsonl_path in sorted(pred_dir.glob("*.jsonl")):
            task = jsonl_path.stem
            for item in _iter_jsonl(jsonl_path):
                others = item.get("others") if isinstance(item, dict) else None
                remem = (others or {}).get("remem") if isinstance(others, dict) else None
                if not isinstance(remem, dict):
                    continue
                ingest_ms = remem.get("ingest_ms")
                retrieve_ms = remem.get("retrieve_ms")
                answer_ms = remem.get("answer_ms")
                if ingest_ms is None and retrieve_ms is None and answer_ms is None:
                    continue
                rows.append(
                    {
                        "model": str(args.model),
                        "taskset": str(args.taskset),
                        "seq_len": seq_len,
                        "task": task,
                        "index": item.get("index"),
                        "ingest_ms": float(ingest_ms) if ingest_ms is not None else None,
                        "retrieve_ms": float(retrieve_ms) if retrieve_ms is not None else None,
                        "answer_ms": float(answer_ms) if answer_ms is not None else None,
                        "total_ms": (
                            (float(ingest_ms) if ingest_ms is not None else 0.0)
                            + (float(retrieve_ms) if retrieve_ms is not None else 0.0)
                            + (float(answer_ms) if answer_ms is not None else 0.0)
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No latency rows found. Are you using a Remem-backed runner that writes `others.remem`?")

    def agg(group: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": int(len(group)),
        }
        for key in ["ingest_ms", "retrieve_ms", "answer_ms", "total_ms"]:
            series = group[key].dropna().astype(float)
            out[f"{key}_mean"] = float(series.mean()) if not series.empty else float("nan")
            out[f"{key}_p50"] = _p(series, 0.50)
            out[f"{key}_p90"] = _p(series, 0.90)
            out[f"{key}_p95"] = _p(series, 0.95)
        return out

    grouped = []
    for keys, group in df.groupby(["model", "taskset", "seq_len"], sort=True):
        model, taskset, seq_len = keys
        stats = agg(group)
        grouped.append({"model": model, "taskset": taskset, "seq_len": int(seq_len), **stats})
    by_len = pd.DataFrame(grouped).sort_values(["model", "taskset", "seq_len"])

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_len.to_csv(out_path, index=False)

    # Also write a per-task breakdown next to it.
    per_task = []
    for keys, group in df.groupby(["model", "taskset", "seq_len", "task"], sort=True):
        model, taskset, seq_len, task = keys
        stats = agg(group)
        per_task.append({"model": model, "taskset": taskset, "seq_len": int(seq_len), "task": task, **stats})
    per_task_df = pd.DataFrame(per_task).sort_values(["model", "taskset", "seq_len", "task"])
    per_task_path = out_path.with_name(out_path.stem.replace("_latency", "_latency_by_task") + out_path.suffix)
    per_task_df.to_csv(per_task_path, index=False)

    print("Wrote:")
    print(" -", out_path)
    print(" -", per_task_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

