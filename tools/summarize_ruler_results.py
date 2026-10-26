#!/usr/bin/env python3
"""
Summarize NVIDIA RULER outputs under `third_party/RULER/scripts/benchmark_root`.

This script converts per-length `pred/summary.csv` files (RULER's 3-row format)
into:
  1) long `current_results.csv`
  2) per-(taskset, seq_len) `rollup.csv`
  3) per-task `task_metric_summary.csv`
  4) wide `taskset_len_wide.csv` (score/nulls/null_rate columns per task)

Example:
  .venv_ruler/bin/python tools/summarize_ruler_results.py \
    --model statemem-qwen3-8b \
    --tasksets synthetic_noqa \
    --out-prefix reports/ruler_statemem-qwen3-8b
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "third_party" / "RULER" / "scripts" / "benchmark_root"


@dataclass(frozen=True)
class SummaryRow:
    model: str
    taskset: str
    seq_len: int
    task: str
    score: float
    nulls: str
    num_samples: int
    summary_path: str


def _parse_nulls_fraction(text: str) -> Tuple[int, int]:
    value = str(text or "").strip()
    if "/" not in value:
        return 0, 0
    left, right = value.split("/", 1)
    try:
        return int(left), int(right)
    except Exception:
        return 0, 0


def _read_ruler_summary(path: Path) -> Tuple[List[str], List[float], List[str]]:
    """
    RULER `summary.csv` structure is:
      row0: column indices
      row1: Tasks, <task1>, <task2>, ...
      row2: Score, <score1>, <score2>, ...
      row3: Nulls, <nulls1>, <nulls2>, ...
    """
    df = pd.read_csv(path)
    if df.shape[0] < 3 or df.shape[1] < 2:
        raise ValueError(f"Unexpected RULER summary format: {path}")

    # Row labels are in column 0 (e.g. "Tasks", "Score", "Nulls")
    row_name_col = df.columns[0]
    row_names = [str(x).strip() for x in df[row_name_col].tolist()]

    def _row_index(name: str) -> int:
        for idx, value in enumerate(row_names):
            if value.lower() == name.lower():
                return idx
        raise KeyError(f"Missing row {name!r} in {path}")

    tasks_row = _row_index("Tasks")
    scores_row = _row_index("Score")
    nulls_row = _row_index("Nulls")

    tasks: List[str] = []
    scores: List[float] = []
    nulls: List[str] = []

    for col in df.columns[1:]:
        task = str(df.at[tasks_row, col]).strip()
        if not task or task.lower() == "nan":
            continue
        score_raw = df.at[scores_row, col]
        null_raw = df.at[nulls_row, col]
        try:
            score = float(score_raw)
        except Exception:
            score = float("nan")
        tasks.append(task)
        scores.append(score)
        nulls.append(str(null_raw).strip())

    return tasks, scores, nulls


def _iter_summaries(model: str, tasksets: Sequence[str]) -> Iterable[Tuple[str, int, Path]]:
    for taskset in tasksets:
        root = BENCHMARK_ROOT / model / taskset
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            try:
                seq_len = int(child.name)
            except Exception:
                continue
            summary_path = child / "pred" / "summary.csv"
            if summary_path.exists():
                yield taskset, seq_len, summary_path


def _collect_rows(model: str, tasksets: Sequence[str]) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    for taskset, seq_len, summary_path in _iter_summaries(model, tasksets):
        tasks, scores, nulls = _read_ruler_summary(summary_path)
        num_samples = 0
        if nulls:
            _, denom = _parse_nulls_fraction(nulls[0])
            num_samples = denom
        for task, score, null_text in zip(tasks, scores, nulls):
            rows.append(
                SummaryRow(
                    model=model,
                    taskset=taskset,
                    seq_len=seq_len,
                    task=task,
                    score=float(score),
                    nulls=null_text,
                    num_samples=int(num_samples) if num_samples else 0,
                    summary_path=str(summary_path),
                )
            )
    return rows


def _write_current_results(path: Path, rows: List[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "taskset", "seq_len", "task", "score", "nulls", "num_samples", "summary_path"])
        for row in sorted(rows, key=lambda r: (r.model, r.taskset, r.seq_len, r.task)):
            writer.writerow([row.model, row.taskset, row.seq_len, row.task, row.score, row.nulls, row.num_samples, row.summary_path])


def _write_taskset_len_wide(path: Path, rows: List[SummaryRow]) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        path.write_text("", encoding="utf-8")
        return

    # Determine task universe for stable columns.
    all_tasks = sorted(set(df["task"].tolist()))
    base_cols = ["model", "taskset", "seq_len", "num_samples"]
    meta = (
        df.groupby(["model", "taskset", "seq_len"], as_index=False)
        .agg(num_samples=("num_samples", "max"), n_tasks=("task", "nunique"))
        .sort_values(["model", "taskset", "seq_len"])
    )

    wide = meta.copy()
    for task in all_tasks:
        part = df[df["task"] == task][["model", "taskset", "seq_len", "score", "nulls"]].copy()
        part = part.rename(columns={"score": f"score__{task}", "nulls": f"nulls__{task}"})
        wide = wide.merge(part, on=["model", "taskset", "seq_len"], how="left")

        # null_rate derived from "a/b"
        if f"nulls__{task}" in wide.columns:
            def _rate(x: object) -> Optional[float]:
                num, denom = _parse_nulls_fraction(str(x))
                if denom <= 0:
                    return None
                return num / denom

            wide[f"null_rate__{task}"] = wide[f"nulls__{task}"].map(_rate)

    # Reorder columns: base + n_tasks + per-task triplets
    columns: List[str] = ["model", "taskset", "seq_len", "num_samples", "n_tasks"]
    for task in all_tasks:
        columns.extend([f"score__{task}", f"nulls__{task}", f"null_rate__{task}"])
    wide = wide.reindex(columns=[c for c in columns if c in wide.columns])

    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(path, index=False)


def _write_rollup(path: Path, rows: List[SummaryRow]) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        path.write_text("", encoding="utf-8")
        return

    def _niah_mean(group: pd.DataFrame) -> Optional[float]:
        subset = group[group["task"].str.startswith("niah_")]
        if subset.empty:
            return None
        return float(subset["score"].mean())

    def _task_score(group: pd.DataFrame, task: str) -> Optional[float]:
        row = group[group["task"] == task]
        if row.empty:
            return None
        return float(row["score"].iloc[0])

    out_rows: List[Dict[str, object]] = []
    for (model, taskset, seq_len), group in df.groupby(["model", "taskset", "seq_len"]):
        out_rows.append(
            {
                "model": model,
                "taskset": taskset,
                "seq_len": int(seq_len),
                "mean_score": round(float(group["score"].mean()), 2),
                "n_tasks": int(group["task"].nunique()),
                "num_samples": int(group["num_samples"].max() or 0),
                "niah_mean": round(_niah_mean(group), 2) if _niah_mean(group) is not None else None,
                "vt": _task_score(group, "vt"),
                "cwe": _task_score(group, "cwe"),
                "fwe": _task_score(group, "fwe"),
                "qa_1": _task_score(group, "qa_1"),
                "qa_2": _task_score(group, "qa_2"),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["model", "taskset", "seq_len"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _write_task_metric_summary(path: Path, rows: List[SummaryRow]) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        path.write_text("", encoding="utf-8")
        return

    seq_lens = sorted(set(int(x) for x in df["seq_len"].tolist()))
    out_rows: List[Dict[str, object]] = []

    for (model, taskset, task), group in df.groupby(["model", "taskset", "task"]):
        scores = [float(x) for x in group["score"].tolist()]
        out: Dict[str, object] = {
            "model": model,
            "taskset": taskset,
            "task": task,
            "num_samples": int(group["num_samples"].max() or 0),
            "n_lengths": int(group["seq_len"].nunique()),
            "mean_score": round(float(pd.Series(scores).mean()), 2),
            "min_score": float(pd.Series(scores).min()),
            "max_score": float(pd.Series(scores).max()),
        }

        by_len = {int(r.seq_len): r for r in (SummaryRow(**rec) for rec in group.to_dict(orient="records"))}
        for seq_len in seq_lens:
            row = by_len.get(seq_len)
            out[f"score_{seq_len}"] = float(row.score) if row is not None else None
            nulls = row.nulls if row is not None else ""
            out[f"nulls_{seq_len}"] = nulls
            num, denom = _parse_nulls_fraction(nulls)
            out[f"null_rate_{seq_len}"] = (num / denom) if denom > 0 else None

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows).sort_values(["model", "taskset", "task"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RULER benchmark_root outputs into analysis-friendly CSVs.")
    parser.add_argument("--model", required=True, help="benchmark_root model directory name (e.g. qwen3-8b-ollama, statemem-qwen3-8b)")
    parser.add_argument("--tasksets", default="synthetic_noqa", help="comma-separated taskset names under the model directory")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix (e.g. reports/ruler_statemem-qwen3-8b)")
    parser.add_argument("--date", default="", help="Optional date suffix like 20260501 (matches existing report naming).")
    args = parser.parse_args()

    tasksets = [t.strip() for t in str(args.tasksets).split(",") if t.strip()]
    rows = _collect_rows(str(args.model), tasksets)
    if not rows:
        raise SystemExit(f"No RULER summaries found under {BENCHMARK_ROOT}/{args.model} for tasksets={tasksets}")

    prefix = Path(str(args.out_prefix))
    date = str(args.date or "").strip()
    suffix = f"_{date}" if date else ""

    current_path = prefix.with_name(prefix.name + f"_current_results{suffix}.csv")
    rollup_path = prefix.with_name(prefix.name + f"_rollup{suffix}.csv")
    metric_path = prefix.with_name(prefix.name + f"_task_metric_summary{suffix}.csv")
    wide_path = prefix.with_name(prefix.name + f"_taskset_len_wide{suffix}.csv")

    _write_current_results(current_path, rows)
    _write_rollup(rollup_path, rows)
    _write_task_metric_summary(metric_path, rows)
    _write_taskset_len_wide(wide_path, rows)

    print("Wrote:")
    print(" -", current_path)
    print(" -", rollup_path)
    print(" -", metric_path)
    print(" -", wide_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
