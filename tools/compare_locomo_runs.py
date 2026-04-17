#!/usr/bin/env python3
"""Compare LoCoMo evaluation outputs from AgenticMemory and MemoryOS."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


UNKNOWN_LIKE = {
    "unknown",
    "unknown.",
    "not mentioned in the conversation",
    "not specified",
}


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("`", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def is_unknown_like(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in UNKNOWN_LIKE or normalized.startswith("the context does not") or normalized.startswith(
        "the memory does not"
    ) or normalized.startswith("no information provided")


def load_agenticmemory(path: Path) -> Tuple[dict, List[dict]]:
    data = json.loads(path.read_text())
    rows = []
    for index, row in enumerate(data["individual_results"]):
        metrics = row["metrics"]
        rows.append(
            {
                "_ordinal": index,
                "qid": row.get("sample_id"),
                "question": row["question"],
                "category": int(row["category"]),
                "gold_answer": row["reference"],
                "prediction": row["prediction"],
                "em": float(metrics["exact_match"]),
                "f1": float(metrics["f1"]),
            }
        )
    return data, rows


def load_memoryos(path: Path) -> Tuple[dict, List[dict]]:
    data = json.loads(path.read_text())
    rows = []
    for index, row in enumerate(data["records"]):
        rows.append(
            {
                "_ordinal": index,
                "qid": row.get("question_index"),
                "question": row["question"],
                "category": int(row["category"]),
                "gold_answer": row["gold_answer"],
                "prediction": row["prediction"],
                "em": float(row["em"]),
                "f1": float(row["f1"]),
            }
        )
    return data, rows


def index_rows(rows: List[dict]) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[normalize_question(row["question"])].append(row)
    duplicate_sizes = {key: len(value) for key, value in grouped.items() if len(value) > 1}
    return grouped, duplicate_sizes


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("`", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def row_signature(row: dict) -> Tuple[int, str]:
    return row["category"], normalize_answer(row["gold_answer"])


def classify_pair(agent: dict, memory: dict) -> str:
    agent_unknown = is_unknown_like(agent["prediction"])
    memory_unknown = is_unknown_like(memory["prediction"])
    if memory["f1"] > agent["f1"]:
        if agent_unknown and not memory_unknown:
            return "agenticmemory_unknown"
        if memory["em"] == 1 and agent["f1"] == 0:
            return "agenticmemory_wrong_fact"
        return "memoryos_better_match"
    if agent["f1"] > memory["f1"]:
        if memory_unknown and not agent_unknown:
            return "memoryos_unknown"
        if agent["em"] == 1 and memory["f1"] == 0:
            return "memoryos_wrong_fact"
        return "agenticmemory_better_match"
    if agent["em"] == 0 and memory["em"] == 0 and agent["f1"] > 0:
        return "both_partial_match"
    if agent["f1"] == 0 and memory["f1"] == 0:
        return "both_wrong"
    return "tie"


def compare_rows(agent_rows: List[dict], memory_rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict], dict]:
    agent_index, agent_dupes = index_rows(agent_rows)
    memory_index, memory_dupes = index_rows(memory_rows)

    common_keys = sorted(set(agent_index) & set(memory_index))
    compared = []
    unmatched_agent = []
    unmatched_memory = []
    skipped_duplicates = []

    for key in common_keys:
        agent_group = agent_index[key]
        memory_group = memory_index[key]
        agent_by_signature: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
        memory_by_signature: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
        for row in agent_group:
            agent_by_signature[row_signature(row)].append(row)
        for row in memory_group:
            memory_by_signature[row_signature(row)].append(row)

        for signature in sorted(set(agent_by_signature) | set(memory_by_signature)):
            agent_signature_rows = sorted(agent_by_signature.get(signature, []), key=lambda row: row["_ordinal"])
            memory_signature_rows = sorted(memory_by_signature.get(signature, []), key=lambda row: row["_ordinal"])

            pair_count = min(len(agent_signature_rows), len(memory_signature_rows))
            for index in range(pair_count):
                agent = agent_signature_rows[index]
                memory = memory_signature_rows[index]
                winner = "tie"
                if memory["f1"] > agent["f1"]:
                    winner = "memoryos"
                elif agent["f1"] > memory["f1"]:
                    winner = "agenticmemory"
                elif memory["em"] > agent["em"]:
                    winner = "memoryos"
                elif agent["em"] > memory["em"]:
                    winner = "agenticmemory"

                compared.append(
                    {
                        "normalized_question": key,
                        "question": memory["question"],
                        "category": memory["category"],
                        "gold_answer": memory["gold_answer"],
                        "agenticmemory_prediction": agent["prediction"],
                        "agenticmemory_em": agent["em"],
                        "agenticmemory_f1": agent["f1"],
                        "memoryos_prediction": memory["prediction"],
                        "memoryos_em": memory["em"],
                        "memoryos_f1": memory["f1"],
                        "winner": winner,
                        "delta_f1_memoryos_minus_agenticmemory": memory["f1"] - agent["f1"],
                        "delta_em_memoryos_minus_agenticmemory": memory["em"] - agent["em"],
                        "note": classify_pair(agent, memory),
                    }
                )

            if len(agent_signature_rows) > pair_count:
                unmatched_agent.extend(agent_signature_rows[pair_count:])
            if len(memory_signature_rows) > pair_count:
                unmatched_memory.extend(memory_signature_rows[pair_count:])

            if len(agent_signature_rows) != len(memory_signature_rows):
                skipped_duplicates.append(
                    {
                        "normalized_question": key,
                        "signature": signature,
                        "agent_count": len(agent_signature_rows),
                        "memory_count": len(memory_signature_rows),
                    }
                )

    compared_keys = {row["normalized_question"] for row in compared}
    for key, rows in agent_index.items():
        if key not in compared_keys:
            unmatched_agent.extend(rows)
    for key, rows in memory_index.items():
        if key not in compared_keys:
            unmatched_memory.extend(rows)

    metadata = {
        "agent_duplicate_keys": agent_dupes,
        "memory_duplicate_keys": memory_dupes,
        "skipped_duplicates": skipped_duplicates,
    }
    return compared, unmatched_agent, unmatched_memory, metadata


def summarize_categories(compared: List[dict]) -> List[dict]:
    by_category: Dict[int, List[dict]] = defaultdict(list)
    for row in compared:
        by_category[row["category"]].append(row)

    summary = []
    for category in sorted(by_category):
        rows = by_category[category]
        summary.append(
            {
                "category": category,
                "count": len(rows),
                "agenticmemory_em": mean(row["agenticmemory_em"] for row in rows),
                "agenticmemory_f1": mean(row["agenticmemory_f1"] for row in rows),
                "memoryos_em": mean(row["memoryos_em"] for row in rows),
                "memoryos_f1": mean(row["memoryos_f1"] for row in rows),
                "memoryos_win_count": sum(row["winner"] == "memoryos" for row in rows),
                "agenticmemory_win_count": sum(row["winner"] == "agenticmemory" for row in rows),
                "tie_count": sum(row["winner"] == "tie" for row in rows),
            }
        )
    return summary


def full_run_summary_agenticmemory(data: dict, rows: List[dict]) -> dict:
    overall = data["aggregate_metrics"]["overall"]
    return {
        "questions": len(rows),
        "em": float(overall["exact_match"]["mean"]),
        "f1": float(overall["f1"]["mean"]),
        "zero_f1": sum(row["f1"] == 0 for row in rows),
        "unknown_like": sum(is_unknown_like(row["prediction"]) for row in rows),
    }


def full_run_summary_memoryos(data: dict, rows: List[dict]) -> dict:
    overall = data["overall"]
    return {
        "questions": len(rows),
        "em": float(overall["em"]),
        "f1": float(overall["f1"]),
        "zero_f1": sum(row["f1"] == 0 for row in rows),
        "unknown_like": sum(is_unknown_like(row["prediction"]) for row in rows),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "category",
        "question",
        "gold_answer",
        "agenticmemory_prediction",
        "agenticmemory_em",
        "agenticmemory_f1",
        "memoryos_prediction",
        "memoryos_em",
        "memoryos_f1",
        "winner",
        "delta_f1_memoryos_minus_agenticmemory",
        "delta_em_memoryos_minus_agenticmemory",
        "note",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(
    path: Path,
    agent_path: Path,
    memory_path: Path,
    agent_summary: dict,
    memory_summary: dict,
    compared: List[dict],
    category_summary: List[dict],
    unmatched_agent: List[dict],
    unmatched_memory: List[dict],
    metadata: dict,
) -> None:
    memory_wins = sorted(
        (row for row in compared if row["winner"] == "memoryos"),
        key=lambda row: (row["delta_f1_memoryos_minus_agenticmemory"], row["delta_em_memoryos_minus_agenticmemory"]),
        reverse=True,
    )
    agent_wins = sorted(
        (row for row in compared if row["winner"] == "agenticmemory"),
        key=lambda row: (row["delta_f1_memoryos_minus_agenticmemory"], row["delta_em_memoryos_minus_agenticmemory"]),
    )
    lines = [
        "# LoCoMo Run Comparison",
        "",
        f"- AgenticMemory: `{agent_path}`",
        f"- MemoryOS: `{memory_path}`",
        "",
        "## Full Run Summary",
        "",
        "| Run | Questions | EM | F1 | Zero-F1 | Unknown-like |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| AgenticMemory | {agent_summary['questions']} | {format_float(agent_summary['em'])} | {format_float(agent_summary['f1'])} | {agent_summary['zero_f1']} | {agent_summary['unknown_like']} |",
        f"| MemoryOS | {memory_summary['questions']} | {format_float(memory_summary['em'])} | {format_float(memory_summary['f1'])} | {memory_summary['zero_f1']} | {memory_summary['unknown_like']} |",
        "",
        "## Common Questions",
        "",
        f"- Compared questions: {len(compared)}",
        f"- MemoryOS wins: {sum(row['winner'] == 'memoryos' for row in compared)}",
        f"- AgenticMemory wins: {sum(row['winner'] == 'agenticmemory' for row in compared)}",
        f"- Ties: {sum(row['winner'] == 'tie' for row in compared)}",
        "",
        "| Category | Count | AgenticMemory EM | AgenticMemory F1 | MemoryOS EM | MemoryOS F1 | MemoryOS Wins | AgenticMemory Wins | Ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_summary:
        lines.append(
            f"| {row['category']} | {row['count']} | {format_float(row['agenticmemory_em'])} | {format_float(row['agenticmemory_f1'])} | {format_float(row['memoryos_em'])} | {format_float(row['memoryos_f1'])} | {row['memoryos_win_count']} | {row['agenticmemory_win_count']} | {row['tie_count']} |"
        )

    lines.extend(
        [
            "",
            "## Top MemoryOS Wins",
            "",
            "| Category | Delta F1 | Question | Gold | AgenticMemory | MemoryOS | Note |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in memory_wins[:15]:
        lines.append(
            f"| {row['category']} | {format_float(row['delta_f1_memoryos_minus_agenticmemory'])} | {row['question']} | {row['gold_answer']} | {row['agenticmemory_prediction']} | {row['memoryos_prediction']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Top AgenticMemory Wins",
            "",
            "| Category | Delta F1 | Question | Gold | AgenticMemory | MemoryOS | Note |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in agent_wins[:15]:
        lines.append(
            f"| {row['category']} | {format_float(-row['delta_f1_memoryos_minus_agenticmemory'])} | {row['question']} | {row['gold_answer']} | {row['agenticmemory_prediction']} | {row['memoryos_prediction']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Unmatched Questions",
            "",
            f"- AgenticMemory only: {len(unmatched_agent)}",
            f"- MemoryOS only: {len(unmatched_memory)}",
            f"- Duplicate normalized question keys skipped: {len(metadata['skipped_duplicates'])}",
        ]
    )
    if unmatched_agent:
        lines.extend(["", "### AgenticMemory Only", ""])
        for row in unmatched_agent[:15]:
            lines.append(f"- `{row['question']}`")
    if unmatched_memory:
        lines.extend(["", "### MemoryOS Only", ""])
        for row in unmatched_memory[:15]:
            lines.append(f"- `{row['question']}`")
    if metadata["skipped_duplicates"]:
        lines.extend(["", "### Duplicate Keys", ""])
        for row in metadata["skipped_duplicates"][:15]:
            lines.append(
                f"- `{row['normalized_question']}`: agent={row['agent_count']}, memory={row['memory_count']}"
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AgenticMemory and MemoryOS LoCoMo outputs.")
    parser.add_argument("--agenticmemory", required=True, type=Path, help="Path to AgenticMemory results JSON")
    parser.add_argument("--memoryos", required=True, type=Path, help="Path to MemoryOS report JSON")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output path prefix without suffix")
    args = parser.parse_args()

    agent_data, agent_rows = load_agenticmemory(args.agenticmemory)
    memory_data, memory_rows = load_memoryos(args.memoryos)
    compared, unmatched_agent, unmatched_memory, metadata = compare_rows(agent_rows, memory_rows)
    category_summary = summarize_categories(compared)
    agent_summary = full_run_summary_agenticmemory(agent_data, agent_rows)
    memory_summary = full_run_summary_memoryos(memory_data, memory_rows)

    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".summary.json")

    write_csv(csv_path, compared)
    write_markdown(
        md_path,
        args.agenticmemory,
        args.memoryos,
        agent_summary,
        memory_summary,
        compared,
        category_summary,
        unmatched_agent,
        unmatched_memory,
        metadata,
    )

    summary_payload = {
        "agenticmemory": agent_summary,
        "memoryos": memory_summary,
        "compared_questions": len(compared),
        "memoryos_wins": sum(row["winner"] == "memoryos" for row in compared),
        "agenticmemory_wins": sum(row["winner"] == "agenticmemory" for row in compared),
        "ties": sum(row["winner"] == "tie" for row in compared),
        "category_summary": category_summary,
        "unmatched_agenticmemory": [row["question"] for row in unmatched_agent],
        "unmatched_memoryos": [row["question"] for row in unmatched_memory],
        "duplicate_keys": metadata["skipped_duplicates"],
    }
    json_path.write_text(json.dumps(summary_payload, indent=2))

    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "summary": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
