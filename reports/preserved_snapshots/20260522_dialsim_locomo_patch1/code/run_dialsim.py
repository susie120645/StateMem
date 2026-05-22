#!/usr/bin/env python3
"""
Run DialSim (LongDialQA) with Remem (MemoryOS) as retrieval + generation backend.

This runner uses the Hugging Face Dataset Viewer `/rows` API to fetch the DialSim
parquet datasets without requiring `pyarrow` locally.

Default evaluation:
  - 1 randomly sampled question per session (matching DialSim's randomized setup)
  - answer format: multi_choice_structured (model must output one of (A)-(E))
  - retrieval: Remem `retrieve_context(top_k, token_budget)`

Example (friends, 50 sessions smoke):
  .venv_ruler/bin/python tools/run_dialsim.py \\
    --script friends \\
    --max-sessions 50 \\
    --llm-model qwen3:8b \\
    --openai-base-url http://127.0.0.1:11534/v1 \\
    --embedding-model nomic-embed-text

Example (full friends):
  .venv_ruler/bin/python tools/run_dialsim.py --script friends
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoryos.adaptive_answer_skill import (
    apply_adaptive_answer_action,
    build_adaptive_answer_state,
    select_adaptive_answer_action,
    should_retry_adaptive_answer,
)
from memoryos.canonical_evidence import build_canonical_evidence_trace, canonicalize_query
from memoryos.choice_answering import maybe_override_abstaining_choice_answer, score_choice_support_with_context
from memoryos.answer_routing import resolve_effective_answer_format, resolve_effective_answer_mode
from memoryos.choice_answering import answer_choice_with_context


DIALSIM_PROMPT_PATH = REPO_ROOT / "third_party" / "DialSim" / "prompt" / "RAG_qa_prompt_multi_choice_structured.txt"
_CANONICAL_TARGET_QTYPES = {"hard_qs_past", "hard_qs_past_past"}
_SHAPE_CALIBRATION_FIELDS = (
    "retrieval_retry_min_confidence",
    "retrieval_retry_score_margin",
    "choice_head_min_confidence",
    "choice_head_min_support_margin",
    "choice_override_min_confidence",
    "choice_override_min_support_margin",
)
_BUILTIN_SHAPE_CALIBRATION_PROFILE: Dict[str, Any] = {
    "default": {},
    "qtype_prefix": {
        "hard": {
            "retrieval_retry_min_confidence": 0.78,
            "retrieval_retry_score_margin": 0.03,
            "choice_head_min_confidence": 0.82,
            "choice_head_min_support_margin": 0.0,
            "choice_override_min_confidence": 0.76,
            "choice_override_min_support_margin": 0.0,
        },
        "easy": {
            "retrieval_retry_min_confidence": 0.90,
            "retrieval_retry_score_margin": 0.06,
            "choice_head_min_confidence": 0.94,
            "choice_head_min_support_margin": 0.0,
            "choice_override_min_confidence": 0.90,
            "choice_override_min_support_margin": 0.0,
        },
    },
    "question_shape": {
        "two_hop_relation_lookup": {
            "retrieval_retry_min_confidence": 0.72,
            "retrieval_retry_score_margin": 0.02,
            "choice_head_min_confidence": 0.74,
            "choice_head_min_support_margin": 0.8,
            "choice_override_min_confidence": 0.70,
            "choice_override_min_support_margin": 0.8,
        },
        "entity_time_relation_lookup": {
            "retrieval_retry_min_confidence": 0.78,
            "retrieval_retry_score_margin": 0.03,
            "choice_head_min_confidence": 0.80,
            "choice_head_min_support_margin": 0.4,
            "choice_override_min_confidence": 0.76,
            "choice_override_min_support_margin": 0.4,
        },
        "entity_relation_lookup": {
            "retrieval_retry_min_confidence": 0.80,
            "retrieval_retry_score_margin": 0.04,
            "choice_head_min_confidence": 0.84,
            "choice_head_min_support_margin": 0.25,
            "choice_override_min_confidence": 0.80,
            "choice_override_min_support_margin": 0.25,
        },
        "time_relation_lookup": {
            "retrieval_retry_min_confidence": 0.80,
            "retrieval_retry_score_margin": 0.04,
            "choice_head_min_confidence": 0.84,
            "choice_head_min_support_margin": 0.25,
            "choice_override_min_confidence": 0.80,
            "choice_override_min_support_margin": 0.25,
        },
        "entity_time_lookup": {
            "retrieval_retry_min_confidence": 0.86,
            "retrieval_retry_score_margin": 0.05,
            "choice_head_min_confidence": 0.90,
            "choice_head_min_support_margin": 0.0,
            "choice_override_min_confidence": 0.88,
            "choice_override_min_support_margin": 0.0,
        },
        "open": {
            "retrieval_retry_min_confidence": 0.92,
            "retrieval_retry_score_margin": 0.06,
            "choice_head_min_confidence": 0.95,
            "choice_head_min_support_margin": 0.0,
            "choice_override_min_confidence": 0.92,
            "choice_override_min_support_margin": 0.0,
        },
    },
    "reasoning_mode": {
        "two_hop_relation": {
            "retrieval_retry_min_confidence": 0.70,
            "retrieval_retry_score_margin": 0.02,
            "choice_head_min_confidence": 0.72,
            "choice_head_min_support_margin": 0.8,
            "choice_override_min_confidence": 0.68,
            "choice_override_min_support_margin": 0.8,
        },
        "time_filtered_lookup": {
            "retrieval_retry_min_confidence": 0.78,
            "retrieval_retry_score_margin": 0.03,
            "choice_head_min_confidence": 0.80,
            "choice_head_min_support_margin": 0.4,
            "choice_override_min_confidence": 0.76,
            "choice_override_min_support_margin": 0.4,
        },
        "direct_lookup": {
            "retrieval_retry_min_confidence": 0.82,
            "retrieval_retry_score_margin": 0.04,
            "choice_head_min_confidence": 0.86,
            "choice_head_min_support_margin": 0.25,
            "choice_override_min_confidence": 0.82,
            "choice_override_min_support_margin": 0.25,
        },
        "evidence_review": {
            "retrieval_retry_min_confidence": 0.92,
            "retrieval_retry_score_margin": 0.06,
            "choice_head_min_confidence": 0.95,
            "choice_head_min_support_margin": 0.0,
            "choice_override_min_confidence": 0.92,
            "choice_override_min_support_margin": 0.0,
        },
    },
    "qtype": {
        "hard_qs_past": {
            "retrieval_retry_min_confidence": 0.76,
            "retrieval_retry_score_margin": 0.03,
            "choice_head_min_confidence": 0.80,
            "choice_head_min_support_margin": 0.25,
            "choice_override_min_confidence": 0.74,
            "choice_override_min_support_margin": 0.25,
        },
        "hard_qs_past_past": {
            "retrieval_retry_min_confidence": 0.72,
            "retrieval_retry_score_margin": 0.02,
            "choice_head_min_confidence": 0.76,
            "choice_head_min_support_margin": 0.8,
            "choice_override_min_confidence": 0.70,
            "choice_override_min_support_margin": 0.8,
        },
    },
}
_MONTH_PHRASE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?:\d{1,2},\s*)?\d{4}\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


_SCRIPT_TO_DATASET = {
    "friends": "jiho283/dialsim-friends",
    "bigbang": "jiho283/dialsim-bigbang",
    "theoffice": "jiho283/dialsim-theoffice",
}


def _strip_openai_v1_suffix(url: str) -> str:
    candidate = str(url or "").strip().rstrip("/")
    if candidate.endswith("/v1"):
        return candidate[: -len("/v1")]
    return candidate


def _is_localhost_url(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(url or "")).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost"}


def _ollama_has_model(ollama_host: str, model_name: str) -> bool:
    if not ollama_host or not model_name:
        return False

    base = str(model_name).strip().lower().split(":", 1)[0]
    try:
        import requests

        response = requests.get(f"{str(ollama_host).rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False

    models = payload.get("models", []) if isinstance(payload, dict) else []
    for item in models:
        name = str((item or {}).get("name") or "").strip().lower()
        if not name:
            continue
        if name == str(model_name).strip().lower():
            return True
        if name.split(":", 1)[0] == base:
            return True
    return False


def _hf_fetch_rows(
    *,
    dataset: str,
    config: str = "default",
    split: str = "train",
    page_size: int = 100,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    import requests

    datasets_server = str(os.environ.get("HF_DATASETS_SERVER_ENDPOINT") or "").strip().rstrip("/")
    if not datasets_server:
        datasets_server = "https://datasets-server.huggingface.co"

    session = requests.Session()
    session.trust_env = False

    rows: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None

    while True:
        url = f"{datasets_server}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={page_size}"
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        page = payload.get("rows", []) or []
        for item in page:
            row = item.get("row")
            if isinstance(row, dict):
                rows.append(row)

        if total is None:
            try:
                total = int(payload.get("num_rows_total"))
            except Exception:
                total = None

        offset += len(page)
        if not page:
            break
        if total is not None and offset >= total:
            break

    return rows


def _load_rows_via_hf_datasets(dataset: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load rows via `datasets.load_dataset`, which respects `HF_ENDPOINT` (Hub mirror).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rows: List[Dict[str, Any]] = []
    for item in ds:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _read_prompt_template() -> str:
    return DIALSIM_PROMPT_PATH.read_text(encoding="utf-8")


def _extract_default_question(question_entry: Any) -> str:
    if isinstance(question_entry, dict):
        value = question_entry.get("default")
        if isinstance(value, str) and value.strip():
            return value.strip()
        # fallback: any string field
        for _, v in question_entry.items():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(question_entry, str):
        return question_entry.strip()
    return str(question_entry or "").strip()


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


def _normalize_retrieval_query(question: str) -> str:
    """
    DialSim questions often start with conversational fillers ("Oh my God, remember..."),
    which can prevent MemoryOS from recognizing them as detail questions.
    Heuristic: strip leading chatter by starting at the first WH-word if present.
    """
    text = " ".join(str(question or "").strip().split())
    if not text:
        return ""
    lowered = text.lower()
    import re

    match = re.search(r"\b(what|which|who|when|where|why|how)\b", lowered)
    if match and match.start() > 0:
        return text[match.start() :].strip()
    return text


@dataclass(frozen=True)
class QuestionInstance:
    qtype: str
    question: str
    options: List[str]  # length 5
    answer_text: str
    idx: Optional[int] = None


def _collect_instances(row: Dict[str, Any]) -> Tuple[List[QuestionInstance], List[QuestionInstance]]:
    hard_prefixes = [
        "hard_qs_past",
        "hard_qs_cur",
        "hard_qs_fu",
        "hard_qs_past_past",
        "hard_qs_cur_past",
    ]
    easy_prefixes = [
        "easy_qs_ans_w_time",
        "easy_qs_ans_wo_time",
        "easy_qs_before_event_unans",
        "easy_qs_dont_know_unans",
        "easy_qs_dont_know_unans_time",
    ]

    hard: List[QuestionInstance] = []
    easy: List[QuestionInstance] = []

    def _add(prefix: str, target: List[QuestionInstance], has_idx: bool) -> None:
        questions = row.get(f"{prefix}_questions")
        options = row.get(f"{prefix}_options")
        answers = row.get(f"{prefix}_answers")
        idxes = row.get(f"{prefix}_idxes") if has_idx else None

        if not isinstance(questions, list) or not isinstance(options, list) or not isinstance(answers, list):
            return
        if len(questions) != len(options) or len(questions) != len(answers):
            return

        for i in range(len(answers)):
            q = _extract_default_question(questions[i])
            if not q:
                continue
            opt = options[i]
            if not isinstance(opt, list) or len(opt) != 5:
                continue
            ans = answers[i]
            if not isinstance(ans, str):
                ans = str(ans or "")
            idx = None
            if has_idx and isinstance(idxes, list) and i < len(idxes):
                try:
                    idx = int(idxes[i])
                except Exception:
                    idx = None
            target.append(
                QuestionInstance(
                    qtype=prefix,
                    question=q,
                    options=[str(x) for x in opt],
                    answer_text=str(ans).strip(),
                    idx=idx,
                )
            )

    for prefix in hard_prefixes:
        _add(prefix, hard, has_idx=False)
    for prefix in easy_prefixes:
        _add(prefix, easy, has_idx=True)

    return hard, easy


def _label_for_answer(answer_text: str, options: Sequence[str]) -> str:
    answer_text_norm = str(answer_text or "").strip().lower()
    if not options:
        return "(E)"
    for idx, opt in enumerate(options[:5]):
        if str(opt).strip().lower() == answer_text_norm:
            return f"({chr(ord('A') + idx)})"
    # If answer isn't an exact match, treat as unknown.
    return "(E)"


def _judge_structured(true_label: str, answer: str) -> Tuple[str, bool]:
    """
    Mirror DialSim's `judge_eq(..., answer_format="multi_choice_structured")` behavior:
      - Correct if the output contains only the true option label and no other option labels.
      - Ambiguous if multiple option labels appear.
      - Wrong otherwise.
    """
    text = str(answer or "")
    if not text:
        return "Wrong", False

    labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(a)", "(b)", "(c)", "(d)", "(e)"]
    present = [lab for lab in labels if lab in text]
    if not present:
        return "Wrong", False

    true_variants = {true_label, true_label.lower()}
    has_true = any(lab in true_variants for lab in present)
    if not has_true:
        return "Wrong", False

    # Any other label besides true -> ambiguous
    for lab in present:
        if lab not in true_variants:
            return "Ambiguous", True
    return "Correct", False


def _format_mc_question(question: str, options: Sequence[str]) -> str:
    parts = [question.strip()]
    labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    for lab, opt in zip(labels, list(options)[:5]):
        parts.append(f"{lab} {str(opt).strip()}")
    # DialSim often uses tab-separated options; but newline is fine as long as model follows (A)-(E).
    return "\n".join(parts)


def _normalize_choice_surface(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").strip().lower())
    tokens = [token for token in cleaned.split() if token and token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _extract_choice_label(text: str) -> str:
    match = re.search(r"\(([A-Ea-e])\)", str(text or ""))
    return match.group(1).upper() if match else ""


def _resolve_candidate_choice_label(candidate: str, options: Sequence[str]) -> Tuple[str, str]:
    candidate_norm = _normalize_choice_surface(candidate)
    if not candidate_norm:
        return "", ""

    candidate_tokens = candidate_norm.split()
    matched: List[Tuple[int, str]] = []
    for index, option in enumerate(list(options)[:5]):
        option_norm = _normalize_choice_surface(option)
        if not option_norm:
            continue
        if candidate_norm == option_norm:
            matched.append((index, "exact"))
            continue
        option_tokens = option_norm.split()
        if len(candidate_tokens) >= 2 and set(candidate_tokens).issubset(set(option_tokens)):
            matched.append((index, "subset"))

    if len({index for index, _ in matched}) != 1:
        return "", ""

    index, match_type = matched[0]
    return f"({chr(ord('A') + index)})", match_type


@dataclass(frozen=True)
class ShapeAdaptiveCalibration:
    enabled: bool
    source: str
    qtype: str
    qtype_prefix: str
    question_shape: str
    reasoning_mode: str
    retrieval_retry_min_confidence: float
    retrieval_retry_score_margin: float
    choice_head_min_confidence: float
    choice_head_min_support_margin: float
    choice_override_min_confidence: float
    choice_override_min_support_margin: float
    applied_layers: Tuple[str, ...]


def _bounded_calibration_value(value: Any, *, lower: float = 0.0, upper: float = 0.99) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = lower
    return round(max(lower, min(upper, numeric)), 4)


def _shape_calibration_base_thresholds(
    *,
    retrieval_retry_min_confidence: float,
    retrieval_retry_score_margin: float,
    choice_head_min_confidence: float,
    choice_override_min_confidence: float,
    choice_head_min_support_margin: float = 0.0,
    choice_override_min_support_margin: float = 0.0,
) -> Dict[str, float]:
    return {
        "retrieval_retry_min_confidence": _bounded_calibration_value(retrieval_retry_min_confidence),
        "retrieval_retry_score_margin": _bounded_calibration_value(retrieval_retry_score_margin, upper=1.5),
        "choice_head_min_confidence": _bounded_calibration_value(choice_head_min_confidence),
        "choice_head_min_support_margin": _bounded_calibration_value(choice_head_min_support_margin, upper=5.0),
        "choice_override_min_confidence": _bounded_calibration_value(choice_override_min_confidence),
        "choice_override_min_support_margin": _bounded_calibration_value(choice_override_min_support_margin, upper=5.0),
    }


def _merge_shape_calibration_profiles(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_shape_calibration_profiles(merged.get(key, {}), value)
        else:
            merged[key] = value
    return merged


def _load_shape_calibration_profile(profile_path: str) -> Dict[str, Any]:
    path = Path(str(profile_path or "").strip()).expanduser().resolve()
    if not str(profile_path or "").strip():
        return {}
    if not path.exists():
        raise FileNotFoundError(f"--shape-calibration-profile not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--shape-calibration-profile must contain a JSON object")
    return data


def _apply_shape_calibration_override(
    thresholds: Dict[str, float],
    override: Dict[str, Any],
) -> Dict[str, float]:
    updated = dict(thresholds)
    for field in _SHAPE_CALIBRATION_FIELDS:
        if field in override:
            upper = 5.0 if "support_margin" in field else (1.5 if field == "retrieval_retry_score_margin" else 0.99)
            updated[field] = _bounded_calibration_value(override.get(field), upper=upper)
    return updated


def _build_shape_adaptive_calibration(
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    enabled: bool,
    base_thresholds: Dict[str, float],
    calibration_profile: Optional[Dict[str, Any]] = None,
    calibration_profile_name: str = "",
) -> ShapeAdaptiveCalibration:
    canonical_query = canonical_evidence.get("canonical_query") or {}
    question_shape = str(canonical_query.get("question_shape") or "").strip() or "open"
    reasoning_mode = str(canonical_evidence.get("reasoning_mode") or "").strip() or "evidence_review"
    qtype_value = str(qtype or "").strip()
    qtype_prefix = qtype_value.split("_", 1)[0] if qtype_value else ""
    thresholds = dict(base_thresholds)
    applied_layers: List[str] = []

    if not enabled:
        return ShapeAdaptiveCalibration(
            enabled=False,
            source="disabled",
            qtype=qtype_value,
            qtype_prefix=qtype_prefix,
            question_shape=question_shape,
            reasoning_mode=reasoning_mode,
            retrieval_retry_min_confidence=thresholds["retrieval_retry_min_confidence"],
            retrieval_retry_score_margin=thresholds["retrieval_retry_score_margin"],
            choice_head_min_confidence=thresholds["choice_head_min_confidence"],
            choice_head_min_support_margin=thresholds["choice_head_min_support_margin"],
            choice_override_min_confidence=thresholds["choice_override_min_confidence"],
            choice_override_min_support_margin=thresholds["choice_override_min_support_margin"],
            applied_layers=tuple(),
        )

    profile = deepcopy(_BUILTIN_SHAPE_CALIBRATION_PROFILE)
    source = "built_in_profile"
    if calibration_profile:
        profile = _merge_shape_calibration_profiles(profile, calibration_profile)
        source = calibration_profile_name or "custom_profile"

    for layer_name, layer_key in (
        ("default", "default"),
        ("qtype_prefix", qtype_prefix),
        ("question_shape", question_shape),
        ("reasoning_mode", reasoning_mode),
        ("qtype", qtype_value),
    ):
        override = {}
        if layer_name == "default":
            override = profile.get("default") or {}
        elif layer_key:
            override = (profile.get(layer_name) or {}).get(layer_key) or {}
        if isinstance(override, dict) and override:
            thresholds = _apply_shape_calibration_override(thresholds, override)
            applied_layers.append(f"{layer_name}:{layer_key or 'default'}")

    relation_coverage = canonical_evidence.get("relation_coverage") or {}
    chain_resolution = canonical_evidence.get("chain_resolution") or {}
    chain_status = str(chain_resolution.get("status") or "").strip().lower()
    support_margin = _bounded_calibration_value(
        chain_resolution.get("score", canonical_evidence.get("confidence", 0.0)),
        upper=5.0,
    )

    if bool(canonical_evidence.get("abstain_recommended")):
        thresholds["choice_head_min_confidence"] = _bounded_calibration_value(
            thresholds["choice_head_min_confidence"] + 0.05
        )
        thresholds["choice_override_min_confidence"] = _bounded_calibration_value(
            thresholds["choice_override_min_confidence"] + 0.05
        )
        thresholds["retrieval_retry_score_margin"] = _bounded_calibration_value(
            max(0.0, thresholds["retrieval_retry_score_margin"] - 0.01),
            upper=1.5,
        )
        applied_layers.append("evidence:abstain_recommended")

    if question_shape == "two_hop_relation_lookup" and chain_status == "resolved":
        thresholds["choice_head_min_confidence"] = _bounded_calibration_value(
            max(0.0, thresholds["choice_head_min_confidence"] - 0.04)
        )
        thresholds["choice_override_min_confidence"] = _bounded_calibration_value(
            max(0.0, thresholds["choice_override_min_confidence"] - 0.04)
        )
        thresholds["choice_head_min_support_margin"] = _bounded_calibration_value(
            max(thresholds["choice_head_min_support_margin"], min(1.2, support_margin * 0.5)),
            upper=5.0,
        )
        thresholds["choice_override_min_support_margin"] = _bounded_calibration_value(
            max(thresholds["choice_override_min_support_margin"], min(1.2, support_margin * 0.5)),
            upper=5.0,
        )
        applied_layers.append("evidence:resolved_two_hop_chain")
    elif question_shape == "two_hop_relation_lookup" and chain_status:
        thresholds["retrieval_retry_min_confidence"] = _bounded_calibration_value(
            max(0.0, thresholds["retrieval_retry_min_confidence"] - 0.04)
        )
        thresholds["retrieval_retry_score_margin"] = _bounded_calibration_value(
            max(0.0, thresholds["retrieval_retry_score_margin"] - 0.01),
            upper=1.5,
        )
        applied_layers.append("evidence:unresolved_two_hop_chain")

    if isinstance(relation_coverage, dict) and relation_coverage and all(bool(value) for value in relation_coverage.values()):
        thresholds["choice_head_min_confidence"] = _bounded_calibration_value(
            max(0.0, thresholds["choice_head_min_confidence"] - 0.03)
        )
        thresholds["choice_override_min_confidence"] = _bounded_calibration_value(
            max(0.0, thresholds["choice_override_min_confidence"] - 0.03)
        )
        applied_layers.append("evidence:full_relation_coverage")

    return ShapeAdaptiveCalibration(
        enabled=True,
        source=source,
        qtype=qtype_value,
        qtype_prefix=qtype_prefix,
        question_shape=question_shape,
        reasoning_mode=reasoning_mode,
        retrieval_retry_min_confidence=thresholds["retrieval_retry_min_confidence"],
        retrieval_retry_score_margin=thresholds["retrieval_retry_score_margin"],
        choice_head_min_confidence=thresholds["choice_head_min_confidence"],
        choice_head_min_support_margin=thresholds["choice_head_min_support_margin"],
        choice_override_min_confidence=thresholds["choice_override_min_confidence"],
        choice_override_min_support_margin=thresholds["choice_override_min_support_margin"],
        applied_layers=tuple(applied_layers),
    )


def _resolve_canonical_choice_candidate(
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    options: Sequence[str],
    min_confidence: float,
    min_support_margin: float = 0.0,
    enabled_qtypes: Sequence[str],
) -> Dict[str, Any]:
    enabled = {str(item).strip() for item in enabled_qtypes if str(item).strip()}
    if enabled and str(qtype or "").strip() not in enabled:
        return {
            "reason": "qtype_not_targeted",
            "support_margin": 0.0,
            "confidence": 0.0,
            "candidate": "",
            "question_shape": "",
            "match_type": "",
            "matched_label": "",
        }
    if not isinstance(canonical_evidence, dict) or not canonical_evidence:
        return {
            "reason": "missing_canonical_evidence",
            "support_margin": 0.0,
            "confidence": 0.0,
            "candidate": "",
            "question_shape": "",
            "match_type": "",
            "matched_label": "",
        }

    if bool(canonical_evidence.get("abstain_recommended")):
        return {
            "reason": "canonical_abstain_recommended",
            "support_margin": 0.0,
            "confidence": 0.0,
            "candidate": "",
            "question_shape": "",
            "match_type": "",
            "matched_label": "",
        }

    try:
        confidence = float(canonical_evidence.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    if confidence < float(min_confidence):
        return {
            "reason": "canonical_confidence_below_threshold",
            "support_margin": 0.0,
            "confidence": round(confidence, 4),
            "candidate": "",
            "question_shape": "",
            "match_type": "",
            "matched_label": "",
        }

    canonical_query = canonical_evidence.get("canonical_query") or {}
    question_shape = str(canonical_query.get("question_shape") or "").strip()
    chain_resolution = canonical_evidence.get("chain_resolution") or {}
    if question_shape == "two_hop_relation_lookup" and str(chain_resolution.get("status", "")).strip().lower() != "resolved":
        return {
            "reason": "canonical_chain_unresolved",
            "support_margin": 0.0,
            "confidence": round(confidence, 4),
            "candidate": "",
            "question_shape": question_shape,
            "match_type": "",
            "matched_label": "",
        }

    support_margin = round(float(chain_resolution.get("score", confidence) or confidence), 4)
    if support_margin < float(min_support_margin):
        return {
            "reason": "canonical_support_margin_below_threshold",
            "support_margin": support_margin,
            "confidence": round(confidence, 4),
            "candidate": "",
            "question_shape": question_shape,
            "match_type": "",
            "matched_label": "",
        }

    candidate = ""
    candidates: List[str] = []
    if question_shape == "two_hop_relation_lookup":
        candidate = str(chain_resolution.get("resolved_candidate") or "").strip()
        if candidate:
            candidates.append(candidate)
    resolved_candidates = canonical_evidence.get("resolved_candidates") or []
    if isinstance(resolved_candidates, list):
        for item in resolved_candidates:
            value = str(item or "").strip()
            if value and value not in candidates:
                candidates.append(value)
    if not candidates:
        return {
            "reason": "canonical_candidate_missing",
            "support_margin": 0.0,
            "confidence": round(confidence, 4),
            "candidate": "",
            "question_shape": question_shape,
            "match_type": "",
            "matched_label": "",
        }

    matched_candidates: List[Tuple[str, str, str]] = []
    for item in candidates:
        matched_label, match_type = _resolve_candidate_choice_label(item, options)
        if matched_label:
            matched_candidates.append((matched_label, item, match_type))
    if not matched_candidates:
        return {
            "reason": "candidate_option_mismatch",
            "support_margin": 0.0,
            "confidence": round(confidence, 4),
            "candidate": candidates[0],
            "question_shape": question_shape,
            "match_type": "",
            "matched_label": "",
        }
    if len({label for label, _, _ in matched_candidates}) != 1:
        return {
            "reason": "canonical_candidate_ambiguous",
            "support_margin": 0.0,
            "confidence": round(confidence, 4),
            "candidate": ",".join(item for _, item, _ in matched_candidates),
            "question_shape": question_shape,
            "match_type": "",
            "matched_label": "",
        }

    matched_label, candidate, match_type = matched_candidates[0]
    return {
        "reason": f"canonical_{question_shape or 'resolved_candidate'}_{match_type}",
        "support_margin": support_margin,
        "confidence": round(confidence, 4),
        "candidate": candidate,
        "question_shape": question_shape,
        "match_type": match_type,
        "matched_label": matched_label,
    }


def _answer_choice_with_canonical_candidate(
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    options: Sequence[str],
    min_confidence: float,
    min_support_margin: float = 0.0,
    enabled_qtypes: Sequence[str],
) -> Dict[str, Any]:
    resolved = _resolve_canonical_choice_candidate(
        canonical_evidence,
        qtype=qtype,
        options=options,
        min_confidence=min_confidence,
        min_support_margin=min_support_margin,
        enabled_qtypes=enabled_qtypes,
    )
    matched_label = str(resolved.get("matched_label") or "").strip()
    if not matched_label:
        return {
            "applied": False,
            "reason": str(resolved.get("reason") or ""),
            "to_label": "",
            "prediction_text": "",
            "support_margin": resolved.get("support_margin", 0.0),
            "confidence": resolved.get("confidence", 0.0),
            "candidate": resolved.get("candidate", ""),
            "question_shape": resolved.get("question_shape", ""),
            "match_type": resolved.get("match_type", ""),
        }
    return {
        "applied": True,
        "reason": f"canonical_head_{resolved.get('reason')}",
        "to_label": matched_label.strip("()"),
        "prediction_text": matched_label,
        "support_margin": resolved.get("support_margin", 0.0),
        "confidence": resolved.get("confidence", 0.0),
        "candidate": resolved.get("candidate", ""),
        "question_shape": resolved.get("question_shape", ""),
        "match_type": resolved.get("match_type", ""),
    }


def _build_canonical_prompt_guidance(
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    options: Sequence[str],
    min_confidence: float,
    min_support_margin: float = 0.0,
    enabled_qtypes: Sequence[str],
) -> List[str]:
    resolved = _resolve_canonical_choice_candidate(
        canonical_evidence,
        qtype=qtype,
        options=options,
        min_confidence=min_confidence,
        min_support_margin=min_support_margin,
        enabled_qtypes=enabled_qtypes,
    )
    matched_label = str(resolved.get("matched_label") or "").strip()
    if not matched_label:
        return []

    fragments = [
        f"Canonical evidence points to option {matched_label}.",
        "If the retrieved history clearly contradicts it, ignore this hint.",
    ]
    if str(resolved.get("question_shape") or "").strip() == "two_hop_relation_lookup":
        fragments.insert(0, "This is a two-hop relation lookup.")
    return fragments


def _extract_time_phrases(text: str) -> List[str]:
    normalized = str(text or "")
    phrases: List[str] = []
    spans: List[Tuple[int, int]] = []
    for match in _MONTH_PHRASE_RE.finditer(normalized):
        value = match.group(0).strip()
        if value and value not in phrases:
            phrases.append(value)
            spans.append(match.span())
    for match in _YEAR_RE.finditer(normalized):
        start, end = match.span()
        if any(existing_start <= start and end <= existing_end for existing_start, existing_end in spans):
            continue
        value = match.group(0).strip()
        if value and value not in phrases:
            phrases.append(value)
    return phrases


def _classify_canonical_retrieval_gap(
    canonical_evidence: Dict[str, Any],
    *,
    question_text: str = "",
) -> Dict[str, Any]:
    canonical_query = {}
    if isinstance(canonical_evidence, dict):
        canonical_query = canonical_evidence.get("canonical_query") or {}
    if not isinstance(canonical_query, dict):
        canonical_query = {}
    if not canonical_query:
        canonical_query = canonicalize_query(question_text, task_hint="dialogue_qa", answer_format="choice")

    question_shape = str(canonical_query.get("question_shape") or "").strip()
    reasoning_mode = str(canonical_evidence.get("reasoning_mode") or "").strip() if isinstance(canonical_evidence, dict) else ""
    relation_coverage = canonical_evidence.get("relation_coverage") or {} if isinstance(canonical_evidence, dict) else {}
    chain_resolution = canonical_evidence.get("chain_resolution") or {} if isinstance(canonical_evidence, dict) else {}
    chain_status = str(chain_resolution.get("status") or "").strip().lower()
    entities = [str(item).strip() for item in canonical_query.get("entities", []) if str(item).strip()]
    relation_terms = [str(item).strip() for item in canonical_query.get("relation_terms", []) if str(item).strip()]
    missing_relations = []
    if isinstance(relation_coverage, dict):
        missing_relations = [str(key).strip() for key, value in relation_coverage.items() if str(key).strip() and not bool(value)]
    time_phrases = _extract_time_phrases(str(canonical_query.get("query_text") or question_text))
    pivot = str(chain_resolution.get("pivot") or "").strip()

    gap_type = "low_confidence"
    if question_shape == "two_hop_relation_lookup" and chain_status == "missing_roommate_anchor":
        gap_type = "two_hop_missing_anchor"
    elif question_shape == "two_hop_relation_lookup" and chain_status == "missing_dating_link":
        gap_type = "two_hop_missing_link"
    elif missing_relations:
        gap_type = "missing_relation_coverage"
    elif isinstance(canonical_evidence, dict) and bool(canonical_evidence.get("abstain_recommended")):
        gap_type = "abstain_recommended"

    return {
        "gap_type": gap_type,
        "question_shape": question_shape,
        "reasoning_mode": reasoning_mode,
        "entities": entities,
        "relation_terms": relation_terms,
        "missing_relations": missing_relations,
        "time_phrases": time_phrases,
        "pivot": pivot,
        "chain_status": chain_status,
    }


def _build_canonical_retrieval_retry_query(
    question_text: str,
    *,
    qtype: str,
    canonical_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    if str(qtype or "").strip() not in _CANONICAL_TARGET_QTYPES:
        return ""

    canonical_query = canonicalize_query(question_text, task_hint="dialogue_qa", answer_format="choice")
    stem = _normalize_retrieval_query(str(canonical_query.get("query_text") or question_text))
    entities = [str(item).strip() for item in canonical_query.get("entities", []) if str(item).strip()]
    relation_terms = [str(item).strip() for item in canonical_query.get("relation_terms", []) if str(item).strip()]
    pronoun_slots = [str(item).strip() for item in canonical_query.get("pronoun_slots", []) if str(item).strip()]
    time_phrases = _extract_time_phrases(stem)
    question_shape = str(canonical_query.get("question_shape") or "").strip()

    detail_lines: List[str] = []
    if question_shape:
        detail_lines.append(f"Question shape: {question_shape}")
    if entities:
        detail_lines.append("Key entities: " + ", ".join(entities))
    if relation_terms:
        detail_lines.append("Required relations: " + ", ".join(relation_terms))
    if pronoun_slots:
        detail_lines.append("Pronoun targets: " + "; ".join(pronoun_slots))
    if time_phrases:
        detail_lines.append("Time anchors: " + ", ".join(time_phrases))

    gap_info = _classify_canonical_retrieval_gap(canonical_evidence or {}, question_text=question_text)
    gap_lines: List[str] = []
    gap_type = str(gap_info.get("gap_type") or "").strip()
    gap_entities = list(gap_info.get("entities") or [])
    gap_times = list(gap_info.get("time_phrases") or [])
    missing_relations = list(gap_info.get("missing_relations") or [])
    pivot = str(gap_info.get("pivot") or "").strip()

    if gap_type == "two_hop_missing_anchor":
        anchor = gap_entities[0] if gap_entities else ""
        gap_lines.append("Repair focus: resolve the intermediate person before answering the second hop.")
        if anchor:
            hop_1 = f"Hop 1: find who is connected to {anchor} through dating"
            if gap_times:
                hop_1 += " around " + ", ".join(gap_times)
            gap_lines.append(hop_1 + ".")
        else:
            gap_lines.append("Hop 1: find the missing person linked by the first relation mention.")
        gap_lines.append("Hop 2: after resolving that person, retrieve roommate-related evidence for that person.")
    elif gap_type == "two_hop_missing_link":
        gap_lines.append("Repair focus: bridge the unresolved second hop after the pivot candidate.")
        if pivot:
            gap_lines.append(f"Known pivot candidate: {pivot}")
            gap_lines.append(f"Find relation evidence that links {pivot} to the final answer target.")
        else:
            gap_lines.append("Find relation evidence that links the intermediate person to the final answer target.")
    elif gap_type == "missing_relation_coverage":
        gap_lines.append("Repair focus: recover explicit evidence for the missing relation.")
        if missing_relations:
            gap_lines.append("Missing relations: " + ", ".join(missing_relations))
        if gap_entities:
            gap_lines.append("Target entities: " + ", ".join(gap_entities))
        if gap_times:
            gap_lines.append("Temporal focus: " + ", ".join(gap_times))
        gap_lines.append("Prefer direct relation statements over broad conversational context.")
    elif gap_type == "abstain_recommended":
        gap_lines.append("Repair focus: retrieve direct answer-bearing evidence.")
        if gap_entities:
            gap_lines.append("Target entities: " + ", ".join(gap_entities))
        if relation_terms:
            gap_lines.append("Priority relations: " + ", ".join(relation_terms))
        if gap_times:
            gap_lines.append("Temporal focus: " + ", ".join(gap_times))
    else:
        gap_lines.append("Repair focus: retrieve more explicit evidence for the answer.")

    if not stem or not detail_lines:
        return ""
    return "\n".join([stem] + gap_lines + detail_lines)


def _canonical_retrieval_score(canonical_evidence: Dict[str, Any]) -> float:
    if not isinstance(canonical_evidence, dict) or not canonical_evidence:
        return 0.0
    try:
        confidence = float(canonical_evidence.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    score = confidence
    if not bool(canonical_evidence.get("abstain_recommended")):
        score += 0.35

    resolved_candidates = canonical_evidence.get("resolved_candidates") or []
    if isinstance(resolved_candidates, list) and resolved_candidates:
        score += 0.12

    relation_coverage = canonical_evidence.get("relation_coverage") or {}
    if isinstance(relation_coverage, dict) and relation_coverage and all(bool(value) for value in relation_coverage.values()):
        score += 0.12

    canonical_query = canonical_evidence.get("canonical_query") or {}
    question_shape = str(canonical_query.get("question_shape") or "").strip()
    chain_resolution = canonical_evidence.get("chain_resolution") or {}
    if question_shape == "two_hop_relation_lookup":
        status = str(chain_resolution.get("status", "") or "").strip().lower()
        if status == "resolved":
            score += 0.45
        elif status:
            score -= 0.1
    return round(score, 4)


def _should_retry_canonical_retrieval(
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    enabled_qtypes: Sequence[str],
    min_confidence: float,
) -> bool:
    enabled = {str(item).strip() for item in enabled_qtypes if str(item).strip()}
    if enabled and str(qtype or "").strip() not in enabled:
        return False
    if not isinstance(canonical_evidence, dict) or not canonical_evidence:
        return True

    canonical_query = canonical_evidence.get("canonical_query") or {}
    question_shape = str(canonical_query.get("question_shape") or "").strip()
    chain_resolution = canonical_evidence.get("chain_resolution") or {}
    try:
        confidence = float(canonical_evidence.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    if question_shape == "two_hop_relation_lookup" and str(chain_resolution.get("status", "")).strip().lower() != "resolved":
        return True
    if bool(canonical_evidence.get("abstain_recommended")):
        return True
    return confidence < float(min_confidence)


def _maybe_override_abstaining_choice_answer_with_canonical(
    prediction_text: str,
    canonical_evidence: Dict[str, Any],
    *,
    qtype: str,
    options: Sequence[str],
    min_confidence: float,
    min_support_margin: float = 0.0,
    enabled_qtypes: Sequence[str],
) -> Dict[str, Any]:
    current_label = _extract_choice_label(prediction_text)
    if current_label != "E":
        return {
            "applied": False,
            "reason": "prediction_not_abstain",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": 0.0,
            "confidence": 0.0,
            "candidate": "",
            "question_shape": "",
            "match_type": "",
        }

    resolved = _resolve_canonical_choice_candidate(
        canonical_evidence,
        qtype=qtype,
        options=options,
        min_confidence=min_confidence,
        min_support_margin=min_support_margin,
        enabled_qtypes=enabled_qtypes,
    )
    matched_label = str(resolved.get("matched_label") or "").strip()
    if not matched_label:
        return {
            "applied": False,
            "reason": str(resolved.get("reason") or ""),
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": resolved.get("support_margin", 0.0),
            "confidence": resolved.get("confidence", 0.0),
            "candidate": resolved.get("candidate", ""),
            "question_shape": resolved.get("question_shape", ""),
            "match_type": resolved.get("match_type", ""),
        }

    return {
        "applied": True,
        "reason": f"canonical_override_{resolved.get('reason')}",
        "from_label": current_label,
        "to_label": matched_label.strip("()"),
        "prediction_text": matched_label,
        "support_margin": resolved.get("support_margin", 0.0),
        "confidence": resolved.get("confidence", 0.0),
        "candidate": resolved.get("candidate", ""),
        "question_shape": resolved.get("question_shape", ""),
        "match_type": resolved.get("match_type", ""),
    }


def _maybe_apply_contrastive_option_verifier(
    prediction_text: str,
    choice_support: Dict[str, Any],
    *,
    canonical_evidence: Optional[Dict[str, Any]] = None,
    qtype: str,
    enabled_qtypes: Sequence[str],
    min_margin: float = 0.6,
    min_time_overlap: float = 0.5,
    min_entity_overlap: float = 0.3,
    min_non_abstain_score: float = 0.0,
    min_non_abstain_overlap: float = 0.5,
) -> Dict[str, Any]:
    current_label = _extract_choice_label(prediction_text)
    if current_label != "E":
        return {
            "applied": False,
            "reason": "prediction_not_abstain",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": 0.0,
            "best_non_abstain_score": 0.0,
            "best_non_abstain_overlap": 0.0,
            "question_time_overlap": 0.0,
            "question_entity_overlap": 0.0,
            "relation_overlap_count": 0,
        }

    enabled = {str(item).strip() for item in enabled_qtypes if str(item).strip()}
    if enabled and str(qtype or "").strip() not in enabled:
        return {
            "applied": False,
            "reason": "qtype_not_targeted",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": 0.0,
            "best_non_abstain_score": 0.0,
            "best_non_abstain_overlap": 0.0,
            "question_time_overlap": 0.0,
            "question_entity_overlap": 0.0,
            "relation_overlap_count": 0,
        }

    best_choice_marker = str(choice_support.get("best_non_abstain_choice") or "").strip()
    best_choice_label = _extract_choice_label(best_choice_marker)
    if not best_choice_label or best_choice_label == "E":
        return {
            "applied": False,
            "reason": "best_non_abstain_missing",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": 0.0,
            "best_non_abstain_score": 0.0,
            "best_non_abstain_overlap": 0.0,
            "question_time_overlap": 0.0,
            "question_entity_overlap": 0.0,
            "relation_overlap_count": 0,
        }

    try:
        support_margin = float(choice_support.get("support_margin", 0.0) or 0.0)
    except Exception:
        support_margin = 0.0
    overall_best_choice_marker = str(choice_support.get("best_choice") or "").strip()
    overall_best_choice_label = _extract_choice_label(overall_best_choice_marker)
    try:
        overall_best_choice_score = float(choice_support.get("best_choice_score", 0.0) or 0.0)
    except Exception:
        overall_best_choice_score = 0.0
    try:
        best_non_abstain_score = float(choice_support.get("best_non_abstain_score", 0.0) or 0.0)
    except Exception:
        best_non_abstain_score = 0.0
    try:
        best_non_abstain_overlap = float(choice_support.get("best_non_abstain_overlap", 0.0) or 0.0)
    except Exception:
        best_non_abstain_overlap = 0.0
    try:
        question_time_overlap = float(choice_support.get("question_time_overlap", 0.0) or 0.0)
    except Exception:
        question_time_overlap = 0.0
    try:
        question_entity_overlap = float(choice_support.get("question_entity_overlap", 0.0) or 0.0)
    except Exception:
        question_entity_overlap = 0.0

    relation_overlap = [str(item).strip() for item in choice_support.get("relation_overlap", []) or [] if str(item).strip()]
    question_relations = [str(item).strip() for item in choice_support.get("question_relations", []) or [] if str(item).strip()]
    abstain_recommended = bool(choice_support.get("abstain_recommended"))
    canonical_evidence = canonical_evidence or {}
    canonical_query = canonical_evidence.get("canonical_query") or {}
    canonical_question_shape = str(canonical_query.get("question_shape") or "").strip()
    canonical_reasoning_mode = str(canonical_evidence.get("reasoning_mode") or "").strip()
    canonical_relation_coverage = canonical_evidence.get("relation_coverage") or {}
    canonical_chain_resolution = canonical_evidence.get("chain_resolution") or {}
    effective_min_non_abstain_overlap = float(min_non_abstain_overlap)
    if str(qtype or "").strip() == "hard_qs_past_past":
        # Two-hop temporal questions often paraphrase the final answer rather than repeating
        # the option text verbatim, so lexical option overlap is a weak negative signal there.
        effective_min_non_abstain_overlap = 0.0

    effective_min_margin = float(min_margin) + (0.15 if abstain_recommended else 0.0)
    if support_margin < effective_min_margin:
        return {
            "applied": False,
            "reason": "support_margin_below_threshold",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if overall_best_choice_label and overall_best_choice_label != "E":
        if support_margin < max(effective_min_margin, 0.75) or overall_best_choice_score < 0.75:
            return {
                "applied": False,
                "reason": "best_choice_non_abstain_not_confident_enough",
                "from_label": current_label,
                "to_label": current_label,
                "support_margin": round(support_margin, 4),
                "best_non_abstain_score": round(best_non_abstain_score, 4),
                "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
                "question_time_overlap": round(question_time_overlap, 4),
                "question_entity_overlap": round(question_entity_overlap, 4),
                "relation_overlap_count": len(relation_overlap),
            }
    if best_non_abstain_score < float(min_non_abstain_score):
        return {
            "applied": False,
            "reason": "best_non_abstain_score_below_threshold",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if (
        str(qtype or "").strip() != "hard_qs_past_past"
        and best_non_abstain_score <= 0.0
        and best_non_abstain_overlap < 1.0
    ):
        # In one-hop questions, zero-scored options with only partial lexical overlap are the
        # least stable overrides. Keep two-hop questions exempt because their final answer often
        # paraphrases the option surface.
        return {
            "applied": False,
            "reason": "zero_score_non_exact_option_match",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if best_non_abstain_overlap < effective_min_non_abstain_overlap:
        return {
            "applied": False,
            "reason": "best_non_abstain_overlap_below_threshold",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if str(qtype or "").strip() == "hard_qs_past_past":
        if canonical_question_shape and canonical_question_shape != "two_hop_relation_lookup":
            return {
                "applied": False,
                "reason": "canonical_shape_mismatch_for_two_hop_qtype",
                "from_label": current_label,
                "to_label": current_label,
                "support_margin": round(support_margin, 4),
                "best_non_abstain_score": round(best_non_abstain_score, 4),
                "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
                "question_time_overlap": round(question_time_overlap, 4),
                "question_entity_overlap": round(question_entity_overlap, 4),
                "relation_overlap_count": len(relation_overlap),
            }
        coverage_values = [bool(value) for value in canonical_relation_coverage.values()]
        chain_status = str(canonical_chain_resolution.get("status") or "").strip()
        chain_resolved_candidate = str(canonical_chain_resolution.get("resolved_candidate") or "").strip()
        if coverage_values and all(coverage_values) and not chain_resolved_candidate and chain_status:
            return {
                "applied": False,
                "reason": "canonical_chain_unresolved_despite_relation_coverage",
                "from_label": current_label,
                "to_label": current_label,
                "support_margin": round(support_margin, 4),
                "best_non_abstain_score": round(best_non_abstain_score, 4),
                "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
                "question_time_overlap": round(question_time_overlap, 4),
                "question_entity_overlap": round(question_entity_overlap, 4),
                "relation_overlap_count": len(relation_overlap),
            }
    if question_time_overlap < float(min_time_overlap):
        return {
            "applied": False,
            "reason": "question_time_overlap_below_threshold",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if question_entity_overlap < float(min_entity_overlap):
        return {
            "applied": False,
            "reason": "question_entity_overlap_below_threshold",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }
    if question_relations and not relation_overlap:
        return {
            "applied": False,
            "reason": "relation_overlap_missing",
            "from_label": current_label,
            "to_label": current_label,
            "support_margin": round(support_margin, 4),
            "best_non_abstain_score": round(best_non_abstain_score, 4),
            "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
            "relation_overlap_count": len(relation_overlap),
        }

    return {
        "applied": True,
        "reason": "contrastive_non_abstain_verified",
        "from_label": current_label,
        "to_label": best_choice_label,
        "prediction_text": best_choice_marker or f"({best_choice_label})",
        "support_margin": round(support_margin, 4),
        "best_non_abstain_score": round(best_non_abstain_score, 4),
        "best_non_abstain_overlap": round(best_non_abstain_overlap, 4),
        "question_time_overlap": round(question_time_overlap, 4),
        "question_entity_overlap": round(question_entity_overlap, 4),
        "relation_overlap_count": len(relation_overlap),
    }


def _maybe_apply_citation_backed_choice_fallback(
    prediction_text: str,
    choice_support: Dict[str, Any],
    *,
    canonical_evidence: Optional[Dict[str, Any]] = None,
    qtype: str,
) -> Dict[str, Any]:
    current_label = _extract_choice_label(prediction_text)
    result = {
        "applied": False,
        "reason": "",
        "from_label": current_label,
        "to_label": current_label,
        "support_margin": 0.0,
        "best_choice_score": 0.0,
        "best_option_overlap": 0.0,
        "question_time_overlap": 0.0,
        "question_entity_overlap": 0.0,
    }
    if current_label != "E":
        result["reason"] = "prediction_not_abstain"
        return result

    canonical_query = (canonical_evidence or {}).get("canonical_query") or {}
    canonical_question_shape = str(canonical_query.get("question_shape") or "").strip()
    best_choice_marker = str(choice_support.get("best_choice") or "").strip()
    best_choice_label = _extract_choice_label(best_choice_marker)
    if not best_choice_label or best_choice_label == "E":
        result["reason"] = "best_choice_abstain_or_missing"
        return result

    try:
        support_margin = float(choice_support.get("support_margin", 0.0) or 0.0)
    except Exception:
        support_margin = 0.0
    try:
        best_choice_score = float(choice_support.get("best_choice_score", 0.0) or 0.0)
    except Exception:
        best_choice_score = 0.0
    try:
        best_option_overlap = float(choice_support.get("best_option_overlap", 0.0) or 0.0)
    except Exception:
        best_option_overlap = 0.0
    try:
        question_time_overlap = float(choice_support.get("question_time_overlap", 0.0) or 0.0)
    except Exception:
        question_time_overlap = 0.0
    try:
        question_entity_overlap = float(choice_support.get("question_entity_overlap", 0.0) or 0.0)
    except Exception:
        question_entity_overlap = 0.0
    try:
        family_count = int(choice_support.get("family_count", 0) or 0)
    except Exception:
        family_count = 0
    try:
        top_family_share = float(choice_support.get("top_family_share", 0.0) or 0.0)
    except Exception:
        top_family_share = 0.0

    citations = [str(item).strip() for item in choice_support.get("citations", []) or [] if str(item).strip()]
    question_relations = [str(item).strip() for item in choice_support.get("question_relations", []) or [] if str(item).strip()]
    relation_overlap = [str(item).strip() for item in choice_support.get("relation_overlap", []) or [] if str(item).strip()]

    result.update(
        {
            "support_margin": round(support_margin, 4),
            "best_choice_score": round(best_choice_score, 4),
            "best_option_overlap": round(best_option_overlap, 4),
            "question_time_overlap": round(question_time_overlap, 4),
            "question_entity_overlap": round(question_entity_overlap, 4),
        }
    )

    if not citations:
        result["reason"] = "citations_missing"
        return result
    if family_count > 1 or top_family_share < 0.8:
        result["reason"] = "support_family_not_concentrated"
        return result
    if question_relations and not relation_overlap:
        result["reason"] = "relation_overlap_missing"
        return result

    qtype_key = str(qtype or "").strip()
    if qtype_key == "hard_qs_past":
        if canonical_question_shape != "entity_time_lookup":
            result["reason"] = "hard_past_shape_not_supported"
            return result
        if support_margin < 0.65 or best_choice_score < 0.75:
            result["reason"] = "support_not_confident_enough"
            return result
        if best_option_overlap < 0.75:
            result["reason"] = "best_option_overlap_below_threshold"
            return result
        if question_time_overlap < 0.5 or question_entity_overlap < 0.3:
            result["reason"] = "question_overlap_below_threshold"
            return result
    elif qtype_key == "easy_qs_ans_w_time":
        if support_margin < 0.6 or best_choice_score < 0.7:
            result["reason"] = "support_not_confident_enough"
            return result
        if best_option_overlap < 0.75:
            result["reason"] = "best_option_overlap_below_threshold"
            return result
        if question_time_overlap < 0.5 or question_entity_overlap < 0.5:
            result["reason"] = "question_overlap_below_threshold"
            return result
    else:
        result["reason"] = "qtype_not_targeted"
        return result

    result.update(
        {
            "applied": True,
            "reason": "citation_backed_choice_fallback",
            "to_label": best_choice_label,
            "prediction_text": best_choice_marker or f"({best_choice_label})",
        }
    )
    return result


@dataclass(frozen=True)
class RememConfig:
    user_id: str
    assistant_id: str
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    data_storage_path: str
    embedding_model_name: str
    embedding_host: str
    short_term_capacity: int
    long_term_knowledge_capacity: int
    keyword_ranker_path: str
    persist_access_updates: bool
    enable_graph_one_hop: bool
    enable_storage_backups: bool


def _build_memoryos(cfg: RememConfig):
    from memoryos.memoryos import Memoryos

    return Memoryos(
        user_id=cfg.user_id,
        assistant_id=cfg.assistant_id,
        openai_api_key=cfg.openai_api_key,
        openai_base_url=cfg.openai_base_url,
        llm_model=cfg.llm_model,
        data_storage_path=cfg.data_storage_path,
        embedding_model_name=cfg.embedding_model_name,
        embedding_model_kwargs={"embedding_host": cfg.embedding_host},
        short_term_capacity=cfg.short_term_capacity,
        long_term_knowledge_capacity=cfg.long_term_knowledge_capacity,
        keyword_ranker_path=cfg.keyword_ranker_path if cfg.keyword_ranker_path else None,
        enable_graph_one_hop=cfg.enable_graph_one_hop,
        enable_storage_backups=cfg.enable_storage_backups,
        enable_async_nli_verify=False,
        enable_async_entity_enrichment=False,
        persist_access_updates=cfg.persist_access_updates,
    )


def _answer_from_pages(
    *,
    client: Any,
    model: str,
    prompt_template: str,
    date: str,
    chatbot: str,
    question_prompt: str,
    pages: List[Dict[str, Any]],
    max_tokens: int,
    answer_format: str = "",
    use_choice_head: bool = False,
    extra_guidance: Sequence[str] = (),
) -> str:
    if use_choice_head and str(answer_format or "").strip().lower() == "choice":
        choice_result = answer_choice_with_context(
            client=client,
            model=model,
            question_text=question_prompt,
            pages=pages,
            max_tokens=max_tokens,
            extra_guidance=extra_guidance,
            task_instruction=prompt_template,
            prefer_supported_choice=any(
                "only use \"i don't know.\"" in str(line).strip().lower()
                or "do not answer with \"i don't know.\"" in str(line).strip().lower()
                for line in extra_guidance
            ),
        )
        if choice_result.get("final_answer"):
            return str(choice_result["final_answer"]).strip()

    blocks: List[str] = []
    for page in pages or []:
        raw = str(page.get("raw_content") or page.get("summary") or "").strip()
        if raw:
            blocks.append(raw)
    dialog_history = "\n".join(blocks).strip() if blocks else "No history.\n"

    prompt = (
        prompt_template.replace("<<<Chatbot>>>", chatbot)
        .replace("<<<Date>>>", date)
        .replace("<<<Dialog_History>>>", dialog_history)
        .replace("<<<Question>>>", question_prompt)
        + (
            "\n\nAdaptive guidance:\n" + "\n".join(f"- {line}" for line in extra_guidance if str(line).strip())
            if any(str(line).strip() for line in extra_guidance)
            else ""
        )
    )
    return str(
        client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
        )
        or ""
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DialSim datasets with Remem (MemoryOS).")
    parser.add_argument("--script", default="friends", choices=sorted(_SCRIPT_TO_DATASET.keys()))
    parser.add_argument("--dataset", default="", help="Override HF dataset name (default: jiho283/dialsim-<script>)")
    parser.add_argument(
        "--rows-jsonl",
        default="",
        help="Offline mode: path to a JSONL file where each line is a DialSim row dict (Episode/Session/Date/Script/... columns).",
    )
    parser.add_argument(
        "--data-source",
        default="auto",
        choices=["auto", "hf_datasets", "viewer_api"],
        help="How to load the DialSim dataset when --rows-jsonl is not provided. `hf_datasets` respects HF_ENDPOINT mirrors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tkg-ratio", type=float, default=0.7, help="Probability to sample from hard_qs_* (else easy_qs_*).")
    parser.add_argument("--questions-per-session", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=0, help="Limit number of sessions (0 = all).")
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit number of episodes (0 = all).")
    parser.add_argument(
        "--sort-policy",
        default="date",
        choices=["date", "episode_session", "none"],
        help="How to order sessions before simulation. `date` matches DialSim's sequential ordering most closely.",
    )
    parser.add_argument(
        "--reset-policy",
        default="none",
        choices=["none", "episode"],
        help="When to reset MemoryOS state during simulation. DialSim sessions are sequential; use `none` for full continuity.",
    )
    parser.add_argument(
        "--ingest-salience-seed",
        type=float,
        default=1.0,
        help="Override salience_seed for each ingested session transcript (>=0). Use 0 to disable override.",
    )
    parser.add_argument(
        "--ingest-important",
        action="store_true",
        default=False,
        help="Mark ingested session transcripts as important to bias durable routing into long-term memory.",
    )
    parser.add_argument(
        "--disable-query-normalization",
        action="store_true",
        default=False,
        help="Disable WH-word based query normalization before retrieval.",
    )
    parser.add_argument(
        "--include-options-in-query",
        action="store_true",
        default=False,
        help="Append answer options (A)-(D) to the retrieval query to improve lexical recall.",
    )
    parser.add_argument(
        "--enable-adaptive-answer-skill",
        action="store_true",
        default=False,
        help="Enable the experimental generic adaptive answer skill (extra guidance + retry). Disabled by default to preserve the original DialSim prompt behavior.",
    )
    parser.add_argument(
        "--enable-choice-answer-head",
        action="store_true",
        default=False,
        help="Enable the experimental generic choice answer head for multi-choice questions.",
    )
    parser.add_argument(
        "--log-choice-support-skill",
        action="store_true",
        default=False,
        help="Log generic choice-support scores without changing the final answer.",
    )
    parser.add_argument(
        "--enable-canonical-choice-head",
        action="store_true",
        default=False,
        help="Use a deterministic canonical-evidence choice head on targeted DialSim hard question types before falling back to the generic answer prompt.",
    )
    parser.add_argument(
        "--canonical-choice-head-min-confidence",
        type=float,
        default=0.9,
        help="Minimum canonical-evidence confidence required for --enable-canonical-choice-head.",
    )
    parser.add_argument(
        "--canonical-choice-head-qtypes",
        default=",".join(sorted(_CANONICAL_TARGET_QTYPES)),
        help="Comma-separated DialSim qtypes allowed for canonical choice head.",
    )
    parser.add_argument(
        "--enable-canonical-retrieval-retry",
        action="store_true",
        default=False,
        help="Retry retrieval with a canonical query rewrite on targeted DialSim hard question types when initial canonical evidence is weak.",
    )
    parser.add_argument(
        "--canonical-retrieval-retry-min-confidence",
        type=float,
        default=0.82,
        help="Minimum canonical-evidence confidence required to skip the canonical retrieval retry.",
    )
    parser.add_argument(
        "--canonical-retrieval-retry-qtypes",
        default=",".join(sorted(_CANONICAL_TARGET_QTYPES)),
        help="Comma-separated DialSim qtypes allowed for canonical retrieval retry.",
    )
    parser.add_argument(
        "--canonical-retrieval-retry-score-margin",
        type=float,
        default=0.05,
        help="Minimum canonical evidence score improvement required to replace the initial retrieval result after retry.",
    )
    parser.add_argument(
        "--enable-choice-support-override",
        action="store_true",
        default=False,
        help="Experimentally replace an abstaining answer with the non-abstain choice from generic choice-support when support is strong.",
    )
    parser.add_argument(
        "--choice-support-override-margin",
        type=float,
        default=0.7,
        help="Minimum support margin required for --enable-choice-support-override.",
    )
    parser.add_argument(
        "--enable-contrastive-option-verifier",
        action="store_true",
        default=False,
        help="Use targeted choice-support signals to contrastively verify a non-abstain option when the model abstains on hard DialSim questions.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-qtypes",
        default=",".join(sorted(_CANONICAL_TARGET_QTYPES)),
        help="Comma-separated DialSim qtypes allowed for the contrastive option verifier.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-min-margin",
        type=float,
        default=0.6,
        help="Minimum choice-support margin required for the contrastive option verifier.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-min-time-overlap",
        type=float,
        default=0.5,
        help="Minimum question-time overlap required for the contrastive option verifier.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-min-entity-overlap",
        type=float,
        default=0.3,
        help="Minimum question-entity overlap required for the contrastive option verifier.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-min-option-score",
        type=float,
        default=0.0,
        help="Minimum best non-abstain option score required for the contrastive option verifier.",
    )
    parser.add_argument(
        "--contrastive-option-verifier-min-option-overlap",
        type=float,
        default=0.5,
        help="Minimum lexical overlap required for the best non-abstain option in the contrastive option verifier.",
    )
    parser.add_argument(
        "--enable-canonical-choice-override",
        action="store_true",
        default=False,
        help="Experimentally replace an abstaining answer with a canonical-evidence candidate on targeted DialSim hard question types.",
    )
    parser.add_argument(
        "--canonical-choice-override-min-confidence",
        type=float,
        default=0.78,
        help="Minimum canonical-evidence confidence required for --enable-canonical-choice-override.",
    )
    parser.add_argument(
        "--canonical-choice-override-qtypes",
        default=",".join(sorted(_CANONICAL_TARGET_QTYPES)),
        help="Comma-separated DialSim qtypes allowed for canonical choice override.",
    )
    parser.add_argument(
        "--enable-shape-adaptive-calibration",
        action="store_true",
        default=False,
        help="Use question-shape and qtype-conditioned threshold calibration for canonical retry/head/override.",
    )
    parser.add_argument(
        "--shape-calibration-profile",
        default="",
        help="Optional JSON profile with per-qtype/per-shape threshold overrides for shape-adaptive calibration.",
    )

    # LLM / embeddings
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--openai-api-key", default="ollama")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:11534/v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-host", default="")

    # Remem retrieval knobs
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=2200)
    parser.add_argument("--answer-max-tokens", type=int, default=16)

    # Storage / report
    parser.add_argument("--model-name", default="remem-qwen3-8b")
    parser.add_argument("--data-storage-root", default=str(REPO_ROOT / "fast_eval_storage" / "dialsim_remem"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--run-tag", default="", help="Optional tag in output filenames.")

    # MemoryOS knobs
    parser.add_argument("--short-term-capacity", type=int, default=5)
    parser.add_argument("--long-term-capacity", type=int, default=6000)
    parser.add_argument("--keyword-ranker-path", default=str(REPO_ROOT / "models" / "keyword_ranker.json"))
    parser.add_argument("--persist-access-updates", action="store_true", default=False)
    parser.add_argument("--disable-graph-one-hop", action="store_true", default=False)
    parser.add_argument("--enable-storage-backups", action="store_true", default=False)

    args = parser.parse_args()

    dataset = str(args.dataset).strip() or _SCRIPT_TO_DATASET[str(args.script)]
    random.seed(int(args.seed))
    shape_calibration_profile_path = str(args.shape_calibration_profile or "").strip()
    shape_calibration_profile = _load_shape_calibration_profile(shape_calibration_profile_path) if shape_calibration_profile_path else {}
    enable_shape_adaptive_calibration = bool(args.enable_shape_adaptive_calibration or shape_calibration_profile_path)

    openai_base_url = str(args.openai_base_url).strip()
    embedding_host = str(args.embedding_host).strip() or _strip_openai_v1_suffix(openai_base_url)

    # Preflight checks for local Ollama setups.
    if _is_localhost_url(embedding_host) and not _ollama_has_model(embedding_host, str(args.embedding_model)):
        raise RuntimeError(
            "Embedding model is not available on the selected Ollama host.\n"
            f"- embedding_host: {embedding_host}\n"
            f"- embedding_model: {args.embedding_model}\n"
            "Fix: pull the embedding model on that Ollama instance, e.g.\n"
            f"  OLLAMA_HOST={embedding_host} ./ollama_bin/bin/ollama pull {args.embedding_model}\n"
        )

    llm_host = _strip_openai_v1_suffix(openai_base_url)
    if _is_localhost_url(llm_host) and not _ollama_has_model(llm_host, str(args.llm_model)):
        raise RuntimeError(
            "LLM model is not available on the selected Ollama host.\n"
            f"- llm_host: {llm_host}\n"
            f"- llm_model: {args.llm_model}\n"
            "Fix: pull the model on that Ollama instance, e.g.\n"
            f"  OLLAMA_HOST={llm_host} ./ollama_bin/bin/ollama pull {args.llm_model}\n"
        )

    prompt_template = _read_prompt_template()

    run_id = f"{int(time.time())}"
    tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"dialsim_{args.script}_{args.model_name}{tag}_{run_id}.csv"
    out_json = out_dir / f"dialsim_{args.script}_{args.model_name}{tag}_{run_id}.json"

    data_storage_path = str(Path(args.data_storage_root).resolve() / args.model_name / f"{args.script}_{run_id}")
    Path(data_storage_path).mkdir(parents=True, exist_ok=True)

    remem_cfg = RememConfig(
        user_id=f"dialsim_{args.model_name}_{run_id}",
        assistant_id="dialsim_runner",
        openai_api_key=str(args.openai_api_key),
        openai_base_url=openai_base_url,
        llm_model=str(args.llm_model),
        data_storage_path=data_storage_path,
        embedding_model_name=str(args.embedding_model),
        embedding_host=embedding_host,
        short_term_capacity=int(args.short_term_capacity),
        long_term_knowledge_capacity=int(args.long_term_capacity),
        keyword_ranker_path=str(args.keyword_ranker_path) if args.keyword_ranker_path else "",
        persist_access_updates=bool(args.persist_access_updates),
        enable_graph_one_hop=not bool(args.disable_graph_one_hop),
        enable_storage_backups=bool(args.enable_storage_backups),
    )

    memoryos = _build_memoryos(remem_cfg)

    rows_jsonl = str(args.rows_jsonl).strip()
    if rows_jsonl:
        rows_path = Path(rows_jsonl).expanduser().resolve()
        if not rows_path.exists():
            raise FileNotFoundError(f"--rows-jsonl not found: {rows_path}")
        loaded: List[Dict[str, Any]] = []
        with rows_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    loaded.append(item)
        rows = loaded
    else:
        source = str(args.data_source or "auto").strip().lower()
        errors: List[str] = []
        rows: List[Dict[str, Any]] = []

        def _try_load(name: str):
            nonlocal rows
            try:
                if name == "hf_datasets":
                    rows = _load_rows_via_hf_datasets(dataset, split="train")
                elif name == "viewer_api":
                    rows = _hf_fetch_rows(dataset=dataset, config="default", split="train")
                else:
                    raise ValueError(f"Unknown loader: {name}")
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if source == "auto":
            _try_load("hf_datasets")
            if not rows:
                _try_load("viewer_api")
        else:
                _try_load(source)

        if not rows:
            raise RuntimeError(
                "Failed to load DialSim dataset.\n"
                f"- dataset: {dataset}\n"
                f"- data_source: {source}\n"
                "- errors:\n  - " + "\n  - ".join(errors) + "\n\n"
                "If you are behind a mirror, try:\n"
                "  export HF_ENDPOINT=https://hf-mirror.com\n"
                "  ./.venv_ruler/bin/python tools/run_dialsim.py --data-source hf_datasets ...\n\n"
                "Or download/export JSONL and rerun with:\n"
                "  --rows-jsonl /path/to/dialsim_<script>.jsonl"
            )

    sort_policy = str(args.sort_policy or "date").strip().lower()
    if sort_policy == "date":
        rows.sort(
            key=lambda r: (
                _parse_dialsim_date(r.get("Date")) or datetime.min,
                str(r.get("Episode") or ""),
                int(r.get("Session") or 0),
            )
        )
    elif sort_policy == "episode_session":
        rows.sort(key=lambda r: (str(r.get("Episode") or ""), int(r.get("Session") or 0)))

    # Determine chatbot per script (as in DialSim).
    chatbot = {"friends": "Ross", "bigbang": "Sheldon", "theoffice": "Michael"}[str(args.script)]

    results: List[Dict[str, Any]] = []
    correct = 0
    ambiguous = 0
    seen_sessions = 0
    seen_episodes: List[str] = []

    reset_policy = str(args.reset_policy or "none").strip().lower()
    current_episode: Optional[str] = None
    for row in rows:
        episode = str(row.get("Episode") or "")
        if not episode:
            continue

        if current_episode != episode:
            current_episode = episode
            if episode not in seen_episodes:
                seen_episodes.append(episode)
            if args.max_episodes and len(seen_episodes) > int(args.max_episodes):
                break
            if reset_policy == "episode":
                memoryos.clear_all()

        if args.max_sessions and seen_sessions >= int(args.max_sessions):
            break

        session_num = int(row.get("Session") or 0)
        date = str(row.get("Date") or "").strip()
        script = str(row.get("Script") or "")
        if not script.strip():
            continue

        # Ingest the session script once.
        doc_id = f"dialsim_{args.script}_{episode}_s{session_num}"
        ingest_started = time.time()
        try:
            meta_data: Dict[str, Any] = {
                "doc_id": doc_id,
                "page_id": doc_id,
                "topic": "DialSim",
                "episode": episode,
                "session": session_num,
                "date": date,
                "script": str(args.script),
            }
            try:
                salience_seed = float(args.ingest_salience_seed)
            except Exception:
                salience_seed = 0.0
            if salience_seed > 0.0:
                meta_data["salience_seed"] = salience_seed
            if bool(args.ingest_important):
                meta_data["flags"] = {**(meta_data.get("flags") or {}), "important": True}
            memoryos.add_memory(
                f"[Episode: {episode}, Session: {session_num}, Date: {date}]\n{script}",
                "",
                meta_data=meta_data,
            )
            ingest_error = ""
        except Exception as exc:
            ingest_error = str(exc)
        ingest_ms = round((time.time() - ingest_started) * 1000.0, 3)

        hard, easy = _collect_instances(row)
        if not hard and not easy:
            continue

        # Sample question instances for this session.
        instances: List[QuestionInstance] = []
        for _ in range(max(1, int(args.questions_per_session))):
            use_hard = (random.random() < float(args.tkg_ratio)) if (hard and easy) else bool(hard)
            pool = hard if use_hard else easy
            if not pool:
                pool = easy if hard else hard
            if not pool:
                break
            instances.append(random.choice(pool))

        for inst in instances:
            query = inst.question
            if not bool(args.disable_query_normalization):
                query = _normalize_retrieval_query(query)
            if bool(args.include_options_in_query) and inst.options:
                query = f"{query}\nOptions: " + " | ".join(str(x) for x in inst.options[:4] if str(x).strip())
            question_prompt = _format_mc_question(inst.question, inst.options)
            true_label = _label_for_answer(inst.answer_text, inst.options)
            routing_plan = memoryos.build_task_routing_plan(
                question_prompt,
                task_hint="dialogue_qa",
                preferred_answer_mode="structured",
            )
            effective_answer_mode = resolve_effective_answer_mode(routing_plan)
            effective_answer_format = resolve_effective_answer_format(routing_plan)

            retrieve_started = time.time()
            try:
                retrieval_results = memoryos.retrieve_context(
                    query,
                    max_results=int(args.max_results),
                    token_budget=int(args.token_budget),
                    task_hint="dialogue_qa",
                    preferred_answer_mode=routing_plan.answer_mode,
                )
                pages = retrieval_results.get("retrieved_pages", []) or []
                source_stats = retrieval_results.get("source_stats", {}) or {}
                hit_ids = retrieval_results.get("hit_ids", []) or []
                retrieve_error = ""
            except Exception as exc:
                pages = []
                source_stats = {}
                hit_ids = []
                retrieve_error = str(exc)
            retrieve_ms = round((time.time() - retrieve_started) * 1000.0, 3)

            canonical_evidence = build_canonical_evidence_trace(
                question_prompt,
                pages,
                task_hint="dialogue_qa",
                answer_format=effective_answer_format,
            )
            shape_calibration = _build_shape_adaptive_calibration(
                canonical_evidence,
                qtype=inst.qtype,
                enabled=enable_shape_adaptive_calibration,
                base_thresholds=_shape_calibration_base_thresholds(
                    retrieval_retry_min_confidence=float(args.canonical_retrieval_retry_min_confidence),
                    retrieval_retry_score_margin=float(args.canonical_retrieval_retry_score_margin),
                    choice_head_min_confidence=float(args.canonical_choice_head_min_confidence),
                    choice_override_min_confidence=float(args.canonical_choice_override_min_confidence),
                ),
                calibration_profile=shape_calibration_profile,
                calibration_profile_name=shape_calibration_profile_path,
            )
            canonical_retrieval_retry = {
                "applied": False,
                "replaced": False,
                "reason": "",
                "gap_type": "",
                "retry_query": "",
                "initial_score": _canonical_retrieval_score(canonical_evidence),
                "retry_score": 0.0,
                "initial_confidence": canonical_evidence.get("confidence", 0.0) if isinstance(canonical_evidence, dict) else 0.0,
                "retry_confidence": 0.0,
                "retry_error": "",
            }
            if bool(args.enable_canonical_retrieval_retry):
                retry_enabled_qtypes = [
                    item.strip()
                    for item in str(args.canonical_retrieval_retry_qtypes).split(",")
                    if item.strip()
                ]
                if _should_retry_canonical_retrieval(
                    canonical_evidence,
                    qtype=inst.qtype,
                    enabled_qtypes=retry_enabled_qtypes,
                    min_confidence=shape_calibration.retrieval_retry_min_confidence,
                ):
                    gap_info = _classify_canonical_retrieval_gap(canonical_evidence, question_text=inst.question)
                    canonical_retrieval_retry["gap_type"] = str(gap_info.get("gap_type") or "")
                    retry_query = _build_canonical_retrieval_retry_query(
                        inst.question,
                        qtype=inst.qtype,
                        canonical_evidence=canonical_evidence,
                    )
                    canonical_retrieval_retry["retry_query"] = retry_query
                    if retry_query and " ".join(retry_query.split()) != " ".join(str(query).split()):
                        canonical_retrieval_retry["applied"] = True
                        retry_started = time.time()
                        try:
                            retry_retrieval_results = memoryos.retrieve_context(
                                retry_query,
                                max_results=int(args.max_results),
                                token_budget=int(args.token_budget),
                                task_hint="dialogue_qa",
                                preferred_answer_mode=routing_plan.answer_mode,
                            )
                            retry_pages = retry_retrieval_results.get("retrieved_pages", []) or []
                            retry_canonical_evidence = build_canonical_evidence_trace(
                                question_prompt,
                                retry_pages,
                                task_hint="dialogue_qa",
                                answer_format=effective_answer_format,
                            )
                            retry_score = _canonical_retrieval_score(retry_canonical_evidence)
                            canonical_retrieval_retry["retry_score"] = retry_score
                            canonical_retrieval_retry["retry_confidence"] = retry_canonical_evidence.get("confidence", 0.0)
                            score_margin = shape_calibration.retrieval_retry_score_margin
                            if retry_score > canonical_retrieval_retry["initial_score"] + score_margin:
                                retrieval_results = retry_retrieval_results
                                pages = retry_pages
                                source_stats = retry_retrieval_results.get("source_stats", {}) or {}
                                hit_ids = retry_retrieval_results.get("hit_ids", []) or []
                                retrieve_error = ""
                                canonical_evidence = retry_canonical_evidence
                                shape_calibration = _build_shape_adaptive_calibration(
                                    canonical_evidence,
                                    qtype=inst.qtype,
                                    enabled=enable_shape_adaptive_calibration,
                                    base_thresholds=_shape_calibration_base_thresholds(
                                        retrieval_retry_min_confidence=float(args.canonical_retrieval_retry_min_confidence),
                                        retrieval_retry_score_margin=float(args.canonical_retrieval_retry_score_margin),
                                        choice_head_min_confidence=float(args.canonical_choice_head_min_confidence),
                                        choice_override_min_confidence=float(args.canonical_choice_override_min_confidence),
                                    ),
                                    calibration_profile=shape_calibration_profile,
                                    calibration_profile_name=shape_calibration_profile_path,
                                )
                                canonical_retrieval_retry["replaced"] = True
                                canonical_retrieval_retry["reason"] = "retry_score_improved"
                            else:
                                canonical_retrieval_retry["reason"] = "retry_score_not_improved"
                        except Exception as exc:
                            canonical_retrieval_retry["retry_error"] = str(exc)
                            canonical_retrieval_retry["reason"] = "retry_error"
                        retrieve_ms = round(retrieve_ms + ((time.time() - retry_started) * 1000.0), 3)
                    elif not canonical_retrieval_retry["reason"]:
                        canonical_retrieval_retry["reason"] = "retry_query_empty_or_same"

            adaptive_state = build_adaptive_answer_state(
                question_prompt,
                retrieval_results if not retrieve_error else {"retrieved_pages": [], "source_stats": {}},
                trace_context={
                    "task_hint": "dialogue_qa",
                    "routing_confidence": routing_plan.confidence,
                    "token_budget": int(args.token_budget),
                    "task_routing_plan": routing_plan.to_payload(),
                },
                retrieval_mode=routing_plan.retrieval_mode,
                answer_mode=effective_answer_mode,
                answer_format=effective_answer_format,
            )
            adaptive_enabled = bool(args.enable_adaptive_answer_skill)
            if adaptive_enabled:
                adaptive_action = select_adaptive_answer_action(adaptive_state)
                adaptive_overrides = apply_adaptive_answer_action(
                    adaptive_state,
                    adaptive_action,
                    routing_plan.to_payload(),
                )
                adaptive_action_payload = adaptive_action.to_payload()
                adaptive_guidance = list(adaptive_overrides.get("instruction_fragments") or [])
            else:
                adaptive_action = None
                adaptive_action_payload = {
                    "answer_strategy": "disabled",
                    "retry_strategy": "none",
                    "abstain_policy": "allow",
                    "format_policy": "default",
                }
                adaptive_overrides = {
                    "effective_answer_mode": effective_answer_mode,
                    "effective_answer_format": effective_answer_format,
                    "retry_strategy": "none",
                    "abstain_policy": "allow",
                    "format_policy": "default",
                    "instruction_fragments": [],
                }
                adaptive_guidance = []

            canonical_choice_head = {
                "applied": False,
                "reason": "",
                "to_label": "",
                "support_margin": 0.0,
                "confidence": 0.0,
                "candidate": "",
                "question_shape": "",
                "match_type": "",
            }
            canonical_prompt_guidance: List[str] = []
            pred = ""
            adaptive_retry_count = 0
            answer_error = ""
            answer_started = time.time()
            if bool(args.enable_canonical_choice_head):
                canonical_choice_head = _answer_choice_with_canonical_candidate(
                    canonical_evidence,
                    qtype=inst.qtype,
                    options=inst.options,
                    min_confidence=shape_calibration.choice_head_min_confidence,
                    min_support_margin=shape_calibration.choice_head_min_support_margin,
                    enabled_qtypes=[item.strip() for item in str(args.canonical_choice_head_qtypes).split(",") if item.strip()],
                )
                if bool(canonical_choice_head.get("applied")):
                    pred = str(canonical_choice_head.get("prediction_text") or "").strip()
            if not pred:
                canonical_prompt_guidance = _build_canonical_prompt_guidance(
                    canonical_evidence,
                    qtype=inst.qtype,
                    options=inst.options,
                    min_confidence=shape_calibration.choice_head_min_confidence,
                    min_support_margin=shape_calibration.choice_head_min_support_margin,
                    enabled_qtypes=[item.strip() for item in str(args.canonical_choice_head_qtypes).split(",") if item.strip()],
                )

            if not pred:
                try:
                    combined_guidance = list(adaptive_guidance)
                    combined_guidance.extend(
                        fragment for fragment in canonical_prompt_guidance if str(fragment).strip()
                    )
                    pred = _answer_from_pages(
                        client=memoryos.client,
                        model=memoryos.llm_model,
                        prompt_template=prompt_template,
                        date=date,
                        chatbot=chatbot,
                        question_prompt=question_prompt,
                        pages=pages,
                        max_tokens=int(args.answer_max_tokens),
                        answer_format=str(adaptive_overrides.get("effective_answer_format") or effective_answer_format),
                        use_choice_head=bool(args.enable_choice_answer_head),
                        extra_guidance=combined_guidance,
                    )
                    if adaptive_enabled and adaptive_action is not None and should_retry_adaptive_answer(
                        pred,
                        state=adaptive_state,
                        action=adaptive_action,
                        answer_mode=str(adaptive_overrides.get("effective_answer_mode") or effective_answer_mode),
                        answer_format=str(adaptive_overrides.get("effective_answer_format") or effective_answer_format),
                    ):
                        retry_pages = pages
                        retry_guidance = list(adaptive_guidance)
                        retry_guidance.extend(
                            fragment for fragment in canonical_prompt_guidance if str(fragment).strip()
                        )
                        retry_strategy = str(adaptive_overrides.get("retry_strategy") or "none")
                        if retry_strategy == "retry_with_shorter_context":
                            retry_pages = list(pages[: max(1, min(3, len(pages)))])
                            retry_guidance.append("Re-answer using only the most relevant retrieved evidence.")
                        elif retry_strategy == "retry_with_stricter_format":
                            retry_guidance.append("Re-answer and follow the required answer format exactly.")
                        retry_pred = _answer_from_pages(
                            client=memoryos.client,
                            model=memoryos.llm_model,
                            prompt_template=prompt_template,
                            date=date,
                            chatbot=chatbot,
                            question_prompt=question_prompt,
                            pages=retry_pages,
                            max_tokens=int(args.answer_max_tokens),
                            answer_format=str(adaptive_overrides.get("effective_answer_format") or effective_answer_format),
                            use_choice_head=bool(args.enable_choice_answer_head),
                            extra_guidance=retry_guidance,
                        )
                        adaptive_retry_count = 1
                        if retry_pred.strip():
                            pred = retry_pred
                    else:
                        adaptive_retry_count = 0
                except Exception as exc:
                    pred = ""
                    adaptive_retry_count = 0
                    answer_error = str(exc)
            answer_ms = round((time.time() - answer_started) * 1000.0, 3)

            canonical_choice_override = {
                "applied": False,
                "reason": "",
                "from_label": _extract_choice_label(pred),
                "to_label": _extract_choice_label(pred),
                "support_margin": 0.0,
                "confidence": 0.0,
                "candidate": "",
                "question_shape": "",
                "match_type": "",
            }
            if bool(args.enable_canonical_choice_override):
                canonical_choice_override = _maybe_override_abstaining_choice_answer_with_canonical(
                    pred,
                    canonical_evidence,
                    qtype=inst.qtype,
                    options=inst.options,
                    min_confidence=shape_calibration.choice_override_min_confidence,
                    min_support_margin=shape_calibration.choice_override_min_support_margin,
                    enabled_qtypes=[item.strip() for item in str(args.canonical_choice_override_qtypes).split(",") if item.strip()],
                )
                if bool(canonical_choice_override.get("applied")):
                    pred = str(canonical_choice_override.get("prediction_text") or pred)

            choice_support = {}
            if (
                bool(args.log_choice_support_skill)
                or bool(args.enable_choice_support_override)
                or bool(args.enable_contrastive_option_verifier)
            ):
                try:
                    choice_support = score_choice_support_with_context(
                        client=memoryos.client,
                        model=memoryos.llm_model,
                        question_text=question_prompt,
                        pages=pages,
                        max_tokens=max(160, int(args.answer_max_tokens)),
                        task_instruction=prompt_template,
                    )
                except Exception:
                    choice_support = {}

            contrastive_option_verifier = {
                "applied": False,
                "reason": "",
                "from_label": _extract_choice_label(pred),
                "to_label": _extract_choice_label(pred),
                "support_margin": 0.0,
                "best_non_abstain_score": 0.0,
                "best_non_abstain_overlap": 0.0,
                "question_time_overlap": 0.0,
                "question_entity_overlap": 0.0,
                "relation_overlap_count": 0,
            }
            if bool(args.enable_contrastive_option_verifier) and choice_support:
                contrastive_option_verifier = _maybe_apply_contrastive_option_verifier(
                    pred,
                    choice_support,
                    canonical_evidence=canonical_evidence,
                    qtype=inst.qtype,
                    enabled_qtypes=[item.strip() for item in str(args.contrastive_option_verifier_qtypes).split(",") if item.strip()],
                    min_margin=float(args.contrastive_option_verifier_min_margin),
                    min_time_overlap=float(args.contrastive_option_verifier_min_time_overlap),
                    min_entity_overlap=float(args.contrastive_option_verifier_min_entity_overlap),
                    min_non_abstain_score=float(args.contrastive_option_verifier_min_option_score),
                    min_non_abstain_overlap=float(args.contrastive_option_verifier_min_option_overlap),
                )
                if bool(contrastive_option_verifier.get("applied")):
                    pred = str(contrastive_option_verifier.get("prediction_text") or pred)

            citation_backed_choice_fallback = {
                "applied": False,
                "reason": "",
                "from_label": _extract_choice_label(pred),
                "to_label": _extract_choice_label(pred),
                "support_margin": 0.0,
                "best_choice_score": 0.0,
                "best_option_overlap": 0.0,
                "question_time_overlap": 0.0,
                "question_entity_overlap": 0.0,
            }
            if choice_support:
                citation_backed_choice_fallback = _maybe_apply_citation_backed_choice_fallback(
                    pred,
                    choice_support,
                    canonical_evidence=canonical_evidence,
                    qtype=inst.qtype,
                )
                if bool(citation_backed_choice_fallback.get("applied")):
                    pred = str(citation_backed_choice_fallback.get("prediction_text") or pred)

            choice_support_override = {
                "applied": False,
                "reason": "",
                "from_label": "",
                "to_label": "",
                "support_margin": 0.0,
            }
            if bool(args.enable_choice_support_override) and choice_support:
                choice_support_override = maybe_override_abstaining_choice_answer(
                    pred,
                    choice_support,
                    min_margin=float(args.choice_support_override_margin),
                )
                if bool(choice_support_override.get("applied")):
                    pred = str(choice_support_override.get("prediction_text") or pred)

            result, is_amb = _judge_structured(true_label, pred)
            if result == "Correct":
                correct += 1
            if is_amb:
                ambiguous += 1

            results.append(
                {
                    "dataset": dataset,
                    "script": str(args.script),
                    "model": str(args.model_name),
                    "llm_model": str(args.llm_model),
                    "embedding_model": str(args.embedding_model),
                    "seed": int(args.seed),
                    "episode": episode,
                    "session": session_num,
                    "date": date,
                    "qtype": inst.qtype,
                    "question": inst.question,
                    "options": inst.options,
                    "answer_text": inst.answer_text,
                    "true_label": true_label,
                    "pred": pred,
                    "result": result,
                    "ambiguous": bool(is_amb),
                    "ingest_ms": ingest_ms,
                    "retrieve_ms": retrieve_ms,
                    "answer_ms": answer_ms,
                    "total_ms": ingest_ms + retrieve_ms + answer_ms,
                    "ingest_error": ingest_error,
                    "retrieve_error": retrieve_error,
                    "answer_error": answer_error,
                    "retrieved_count": len(pages),
                    "source_stats": source_stats,
                    "hit_ids": hit_ids,
                    "choice_support": choice_support,
                    "contrastive_option_verifier": contrastive_option_verifier,
                    "citation_backed_choice_fallback": citation_backed_choice_fallback,
                    "choice_support_override": choice_support_override,
                    "canonical_evidence": canonical_evidence,
                    "canonical_retrieval_retry": canonical_retrieval_retry,
                    "canonical_choice_head": canonical_choice_head,
                    "canonical_choice_override": canonical_choice_override,
                    "shape_calibration": asdict(shape_calibration),
                    "adaptive_answer_action": adaptive_action_payload,
                    "adaptive_answer_overrides": adaptive_overrides,
                    "adaptive_retry_count": adaptive_retry_count,
                }
            )

        seen_sessions += 1
        if seen_sessions % 25 == 0:
            total_answered = len(results)
            acc = (correct / total_answered) if total_answered else 0.0
            print(f"[progress] sessions={seen_sessions} answered={total_answered} acc={acc:.4f} ambiguous={ambiguous}")

    total_answered = len(results)
    accuracy = (correct / total_answered) if total_answered else 0.0

    # Write jsonl-like json array + a flat CSV for analysis.
    out_json.write_text(json.dumps({"accuracy": accuracy, "correct": correct, "total": total_answered, "results": results}, ensure_ascii=False), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "script",
                "model",
                "llm_model",
                "embedding_model",
                "seed",
                "episode",
                "session",
                "date",
                "qtype",
                "true_label",
                "result",
                "ambiguous",
                "retrieved_count",
                "ingest_ms",
                "retrieve_ms",
                "answer_ms",
                "total_ms",
                "ingest_error",
                "retrieve_error",
                "answer_error",
                "question",
                "pred",
                "choice_support_best_choice",
                "choice_support_support_margin",
                "choice_support_abstain_recommended",
                "choice_support_reason_codes",
                "choice_support_best_non_abstain_choice",
                "choice_support_best_choice_score",
                "choice_support_best_non_abstain_score",
                "choice_support_question_time_overlap",
                "choice_support_question_entity_overlap",
                "choice_support_source_family_count",
                "choice_support_source_top_family_share",
                "choice_support_question_relations",
                "choice_support_context_relations",
                "choice_support_relation_overlap",
                "choice_support_relation_missing",
                "canonical_question_shape",
                "canonical_relation_terms",
                "canonical_temporal_forms",
                "canonical_pronoun_slots",
                "canonical_reasoning_mode",
                "canonical_confidence",
                "canonical_abstain_recommended",
                "canonical_resolved_candidates",
                "canonical_top_fact_ids",
                "canonical_fact_count",
                "canonical_retrieval_retry_applied",
                "canonical_retrieval_retry_replaced",
                "canonical_retrieval_retry_reason",
                "canonical_retrieval_retry_gap_type",
                "canonical_retrieval_retry_initial_score",
                "canonical_retrieval_retry_retry_score",
                "canonical_retrieval_retry_initial_confidence",
                "canonical_retrieval_retry_retry_confidence",
                "canonical_choice_head_applied",
                "canonical_choice_head_reason",
                "canonical_choice_head_to_label",
                "canonical_choice_head_support_margin",
                "canonical_choice_head_confidence",
                "canonical_choice_head_candidate",
                "canonical_choice_head_question_shape",
                "canonical_choice_head_match_type",
                "canonical_choice_override_applied",
                "canonical_choice_override_reason",
                "canonical_choice_override_from_label",
                "canonical_choice_override_to_label",
                "canonical_choice_override_support_margin",
                "canonical_choice_override_confidence",
                "canonical_choice_override_candidate",
                "canonical_choice_override_question_shape",
                "canonical_choice_override_match_type",
                "contrastive_option_verifier_applied",
                "contrastive_option_verifier_reason",
                "contrastive_option_verifier_from_label",
                "contrastive_option_verifier_to_label",
                "contrastive_option_verifier_support_margin",
                "contrastive_option_verifier_best_non_abstain_score",
                "contrastive_option_verifier_best_non_abstain_overlap",
                "contrastive_option_verifier_question_time_overlap",
                "contrastive_option_verifier_question_entity_overlap",
                "contrastive_option_verifier_relation_overlap_count",
                "citation_backed_choice_fallback_applied",
                "citation_backed_choice_fallback_reason",
                "citation_backed_choice_fallback_from_label",
                "citation_backed_choice_fallback_to_label",
                "citation_backed_choice_fallback_support_margin",
                "citation_backed_choice_fallback_best_choice_score",
                "citation_backed_choice_fallback_best_option_overlap",
                "citation_backed_choice_fallback_question_time_overlap",
                "citation_backed_choice_fallback_question_entity_overlap",
                "shape_calibration_enabled",
                "shape_calibration_source",
                "shape_calibration_qtype_prefix",
                "shape_calibration_question_shape",
                "shape_calibration_reasoning_mode",
                "shape_calibration_retrieval_retry_min_confidence",
                "shape_calibration_retrieval_retry_score_margin",
                "shape_calibration_choice_head_min_confidence",
                "shape_calibration_choice_head_min_support_margin",
                "shape_calibration_choice_override_min_confidence",
                "shape_calibration_choice_override_min_support_margin",
                "shape_calibration_applied_layers",
                "choice_support_override_applied",
                "choice_support_override_reason",
                "choice_support_override_from_label",
                "choice_support_override_to_label",
                "choice_support_override_support_margin",
                "adaptive_retry_count",
            ],
        )
        writer.writeheader()
        for r in results:
            choice_support = r.get("choice_support") or {}
            row = {k: r.get(k, "") for k in writer.fieldnames}
            row["choice_support_best_choice"] = choice_support.get("best_choice", "")
            row["choice_support_support_margin"] = choice_support.get("support_margin", "")
            row["choice_support_abstain_recommended"] = choice_support.get("abstain_recommended", "")
            row["choice_support_reason_codes"] = ",".join(choice_support.get("reason_codes", []) or [])
            row["choice_support_best_non_abstain_choice"] = choice_support.get("best_non_abstain_choice", "")
            row["choice_support_best_choice_score"] = choice_support.get("best_choice_score", "")
            row["choice_support_best_non_abstain_score"] = choice_support.get("best_non_abstain_score", "")
            row["choice_support_question_time_overlap"] = choice_support.get("question_time_overlap", "")
            row["choice_support_question_entity_overlap"] = choice_support.get("question_entity_overlap", "")
            row["choice_support_source_family_count"] = choice_support.get("family_count", "")
            row["choice_support_source_top_family_share"] = choice_support.get("top_family_share", "")
            row["choice_support_question_relations"] = ",".join(choice_support.get("question_relations", []) or [])
            row["choice_support_context_relations"] = ",".join(choice_support.get("context_relations", []) or [])
            row["choice_support_relation_overlap"] = ",".join(choice_support.get("relation_overlap", []) or [])
            row["choice_support_relation_missing"] = ",".join(choice_support.get("relation_missing", []) or [])
            canonical_evidence = r.get("canonical_evidence") or {}
            canonical_query = canonical_evidence.get("canonical_query") or {}
            row["canonical_question_shape"] = canonical_query.get("question_shape", "")
            row["canonical_relation_terms"] = ",".join(canonical_query.get("relation_terms", []) or [])
            row["canonical_temporal_forms"] = ",".join(canonical_query.get("temporal_forms", []) or [])
            row["canonical_pronoun_slots"] = ",".join(canonical_query.get("pronoun_slots", []) or [])
            row["canonical_reasoning_mode"] = canonical_evidence.get("reasoning_mode", "")
            row["canonical_confidence"] = canonical_evidence.get("confidence", "")
            row["canonical_abstain_recommended"] = canonical_evidence.get("abstain_recommended", "")
            row["canonical_resolved_candidates"] = ",".join(canonical_evidence.get("resolved_candidates", []) or [])
            row["canonical_top_fact_ids"] = ",".join(canonical_evidence.get("top_fact_ids", []) or [])
            row["canonical_fact_count"] = canonical_evidence.get("fact_count", "")
            canonical_retry = r.get("canonical_retrieval_retry") or {}
            row["canonical_retrieval_retry_applied"] = canonical_retry.get("applied", "")
            row["canonical_retrieval_retry_replaced"] = canonical_retry.get("replaced", "")
            row["canonical_retrieval_retry_reason"] = canonical_retry.get("reason", "")
            row["canonical_retrieval_retry_gap_type"] = canonical_retry.get("gap_type", "")
            row["canonical_retrieval_retry_initial_score"] = canonical_retry.get("initial_score", "")
            row["canonical_retrieval_retry_retry_score"] = canonical_retry.get("retry_score", "")
            row["canonical_retrieval_retry_initial_confidence"] = canonical_retry.get("initial_confidence", "")
            row["canonical_retrieval_retry_retry_confidence"] = canonical_retry.get("retry_confidence", "")
            canonical_head = r.get("canonical_choice_head") or {}
            row["canonical_choice_head_applied"] = canonical_head.get("applied", "")
            row["canonical_choice_head_reason"] = canonical_head.get("reason", "")
            row["canonical_choice_head_to_label"] = canonical_head.get("to_label", "")
            row["canonical_choice_head_support_margin"] = canonical_head.get("support_margin", "")
            row["canonical_choice_head_confidence"] = canonical_head.get("confidence", "")
            row["canonical_choice_head_candidate"] = canonical_head.get("candidate", "")
            row["canonical_choice_head_question_shape"] = canonical_head.get("question_shape", "")
            row["canonical_choice_head_match_type"] = canonical_head.get("match_type", "")
            canonical_override = r.get("canonical_choice_override") or {}
            row["canonical_choice_override_applied"] = canonical_override.get("applied", "")
            row["canonical_choice_override_reason"] = canonical_override.get("reason", "")
            row["canonical_choice_override_from_label"] = canonical_override.get("from_label", "")
            row["canonical_choice_override_to_label"] = canonical_override.get("to_label", "")
            row["canonical_choice_override_support_margin"] = canonical_override.get("support_margin", "")
            row["canonical_choice_override_confidence"] = canonical_override.get("confidence", "")
            row["canonical_choice_override_candidate"] = canonical_override.get("candidate", "")
            row["canonical_choice_override_question_shape"] = canonical_override.get("question_shape", "")
            row["canonical_choice_override_match_type"] = canonical_override.get("match_type", "")
            contrastive_verifier = r.get("contrastive_option_verifier") or {}
            row["contrastive_option_verifier_applied"] = contrastive_verifier.get("applied", "")
            row["contrastive_option_verifier_reason"] = contrastive_verifier.get("reason", "")
            row["contrastive_option_verifier_from_label"] = contrastive_verifier.get("from_label", "")
            row["contrastive_option_verifier_to_label"] = contrastive_verifier.get("to_label", "")
            row["contrastive_option_verifier_support_margin"] = contrastive_verifier.get("support_margin", "")
            row["contrastive_option_verifier_best_non_abstain_score"] = contrastive_verifier.get("best_non_abstain_score", "")
            row["contrastive_option_verifier_best_non_abstain_overlap"] = contrastive_verifier.get("best_non_abstain_overlap", "")
            row["contrastive_option_verifier_question_time_overlap"] = contrastive_verifier.get("question_time_overlap", "")
            row["contrastive_option_verifier_question_entity_overlap"] = contrastive_verifier.get("question_entity_overlap", "")
            row["contrastive_option_verifier_relation_overlap_count"] = contrastive_verifier.get("relation_overlap_count", "")
            citation_backed_fallback = r.get("citation_backed_choice_fallback") or {}
            row["citation_backed_choice_fallback_applied"] = citation_backed_fallback.get("applied", "")
            row["citation_backed_choice_fallback_reason"] = citation_backed_fallback.get("reason", "")
            row["citation_backed_choice_fallback_from_label"] = citation_backed_fallback.get("from_label", "")
            row["citation_backed_choice_fallback_to_label"] = citation_backed_fallback.get("to_label", "")
            row["citation_backed_choice_fallback_support_margin"] = citation_backed_fallback.get("support_margin", "")
            row["citation_backed_choice_fallback_best_choice_score"] = citation_backed_fallback.get("best_choice_score", "")
            row["citation_backed_choice_fallback_best_option_overlap"] = citation_backed_fallback.get("best_option_overlap", "")
            row["citation_backed_choice_fallback_question_time_overlap"] = citation_backed_fallback.get("question_time_overlap", "")
            row["citation_backed_choice_fallback_question_entity_overlap"] = citation_backed_fallback.get("question_entity_overlap", "")
            shape_calibration = r.get("shape_calibration") or {}
            row["shape_calibration_enabled"] = shape_calibration.get("enabled", "")
            row["shape_calibration_source"] = shape_calibration.get("source", "")
            row["shape_calibration_qtype_prefix"] = shape_calibration.get("qtype_prefix", "")
            row["shape_calibration_question_shape"] = shape_calibration.get("question_shape", "")
            row["shape_calibration_reasoning_mode"] = shape_calibration.get("reasoning_mode", "")
            row["shape_calibration_retrieval_retry_min_confidence"] = shape_calibration.get("retrieval_retry_min_confidence", "")
            row["shape_calibration_retrieval_retry_score_margin"] = shape_calibration.get("retrieval_retry_score_margin", "")
            row["shape_calibration_choice_head_min_confidence"] = shape_calibration.get("choice_head_min_confidence", "")
            row["shape_calibration_choice_head_min_support_margin"] = shape_calibration.get("choice_head_min_support_margin", "")
            row["shape_calibration_choice_override_min_confidence"] = shape_calibration.get("choice_override_min_confidence", "")
            row["shape_calibration_choice_override_min_support_margin"] = shape_calibration.get("choice_override_min_support_margin", "")
            row["shape_calibration_applied_layers"] = ",".join(shape_calibration.get("applied_layers", []) or [])
            override = r.get("choice_support_override") or {}
            row["choice_support_override_applied"] = override.get("applied", "")
            row["choice_support_override_reason"] = override.get("reason", "")
            row["choice_support_override_from_label"] = override.get("from_label", "")
            row["choice_support_override_to_label"] = override.get("to_label", "")
            row["choice_support_override_support_margin"] = override.get("support_margin", "")
            writer.writerow(row)

    print("Done.")
    print(f"- accuracy: {accuracy:.4f} ({correct}/{total_answered})")
    print(f"- ambiguous: {ambiguous}/{total_answered}")
    print(f"- csv: {out_csv}")
    print(f"- json: {out_json}")

    try:
        memoryos.shutdown(wait_for_background_tasks=False)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
