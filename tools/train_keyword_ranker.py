#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CURRENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_ROOT not in sys.path:
    sys.path.insert(0, CURRENT_ROOT)

from statemem.keyword_ranker import LinearKeywordRanker, prepare_keyword_text, train_linear_ranker  # noqa: E402
from statemem.utils import normalize_keyword_values  # noqa: E402


def iter_records(path: str) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    yield item
        return

    if path.endswith(".json") or path.endswith(".json.bak"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return

        if isinstance(payload, dict) and isinstance(payload.get("knowledge_base"), list):
            for item in payload.get("knowledge_base", []):
                if isinstance(item, dict):
                    yield item
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return


def extract_training_pair(record: Dict[str, Any]) -> Optional[Tuple[str, List[str]]]:
    raw_keywords = record.get("keywords")
    if not isinstance(raw_keywords, list):
        meta = record.get("metadata")
        if isinstance(meta, dict):
            raw_keywords = meta.get("keywords")
    if not isinstance(raw_keywords, list):
        return None

    labels = normalize_keyword_values(raw_keywords, limit=64)
    if not labels:
        return None

    raw_text = record.get("raw_content")
    if not isinstance(raw_text, str) or not raw_text.strip():
        user_input = record.get("user_input") if isinstance(record.get("user_input"), str) else ""
        agent_response = record.get("agent_response") if isinstance(record.get("agent_response"), str) else ""
        raw_text = f"{user_input}\n{agent_response}".strip() or record.get("summary") or ""
    if not isinstance(raw_text, str):
        return None

    prepared = prepare_keyword_text(raw_text)
    if not prepared:
        return None
    return prepared, labels


def load_training_pairs(globs: Sequence[str], *, max_records: int) -> List[Tuple[str, List[str]]]:
    pairs: List[Tuple[str, List[str]]] = []
    expanded: List[str] = []
    for pattern in globs:
        expanded.extend(glob.glob(pattern, recursive=True))
    expanded = sorted(set(expanded))

    for path in expanded:
        for record in iter_records(path):
            pair = extract_training_pair(record)
            if pair is None:
                continue
            pairs.append(pair)
            if max_records and len(pairs) >= int(max_records):
                return pairs
    return pairs


def evaluate(model: LinearKeywordRanker, pairs: Sequence[Tuple[str, Sequence[str]]], *, k: int) -> Dict[str, float]:
    if not pairs:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0}

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for text, labels in pairs:
        label_set = set(normalize_keyword_values(list(labels or []), limit=64))
        if not label_set:
            continue
        pred = model.extract(text, max_keywords=int(k))
        hit = sum(1 for term in pred if term in label_set)
        denom_p = max(1, int(k))
        denom_r = max(1, len(label_set))
        precision = float(hit) / float(denom_p)
        recall = float(hit) / float(denom_r)
        f1 = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall) / (precision + recall)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    n = float(len(pairs))
    return {
        "precision_at_k": precision_sum / n,
        "recall_at_k": recall_sum / n,
        "f1_at_k": f1_sum / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        action="append",
        default=[
            "eval_storage/**/forget_pool.jsonl",
            "eval_storage/**/long_term/memory_graph.json",
        ],
        help="Repeatable glob patterns for training data sources.",
    )
    parser.add_argument("--output", default="models/keyword_ranker.json")
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--phrase-limit", type=int, default=96)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--eval-k", type=int, default=12)
    parser.add_argument("--train-split", type=float, default=0.9)
    args = parser.parse_args()

    pairs = load_training_pairs(args.glob, max_records=args.max_records)
    if not pairs:
        raise SystemExit("No training pairs found. Check --glob patterns.")

    rng = random.Random(int(args.seed))
    rng.shuffle(pairs)
    split = int(max(1, min(len(pairs) - 1, int(len(pairs) * float(args.train_split)))))
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]

    model = train_linear_ranker(
        train_pairs,
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        candidate_limit=int(args.candidate_limit),
        phrase_limit=int(args.phrase_limit),
        negative_ratio=int(args.negative_ratio),
    )

    metrics = evaluate(model, eval_pairs, k=int(args.eval_k))
    print(
        "Eval metrics:",
        json.dumps(metrics, ensure_ascii=False),
        f"(n_train={len(train_pairs)}, n_eval={len(eval_pairs)})",
    )

    model.save(args.output)
    print(f"Saved keyword ranker -> {args.output}")
    print("To enable in statemem, set config.json: keyword_ranker_path =", args.output)


if __name__ == "__main__":
    main()

