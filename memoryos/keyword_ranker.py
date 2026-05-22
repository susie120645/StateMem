import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .entity_filters import filter_entity_values
from .utils import fallback_extract_keywords, mask_system_tags, normalize_keyword_values


def prepare_entity_text(text: str, max_chars: int = 1800) -> str:
    compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    head = compact[: int(max_chars * 0.65)]
    tail = compact[-int(max_chars * 0.2) :]
    return f"{head}\n...\n{tail}"


def prepare_keyword_text(text: str, max_chars: int = 1600) -> str:
    compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    head = compact[: int(max_chars * 0.7)]
    tail = compact[-int(max_chars * 0.15) :]
    return f"{head}\n...\n{tail}"


def extract_entities_local(text: str) -> List[str]:
    prepared_text = prepare_entity_text(text)
    if not prepared_text:
        return []

    raw_candidates: List[str] = []
    raw_candidates.extend(
        match.group(0)
        for match in re.finditer(r"\b\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}\b", prepared_text)
    )
    raw_candidates.extend(
        match.group(0)
        for match in re.finditer(r"\b[A-Z][a-z]+\s+\d{4}\b", prepared_text)
    )
    raw_candidates.extend(
        match.group(1)
        for match in re.finditer(r"['\"]([^'\"]{2,80})['\"]", prepared_text)
        if 1 <= len(match.group(1).split()) <= 6
    )
    raw_candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"\b[A-Z][\w'’.-]*(?:\s+(?:[A-Z][\w'’.-]*|of|the|and|de|di|da|la|le|van|von)){0,4}\b",
            prepared_text,
        )
    )
    raw_candidates.extend(
        token
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_:\-./]{2,}\b", prepared_text)
        if any(ch.isdigit() for ch in token) or any(ch in token for ch in "._:/-")
    )
    return filter_entity_values(raw_candidates, limit=24)


def extract_keyword_phrases_local(text: str, max_keywords: int = 6) -> List[str]:
    compact = re.sub(r"\s+", " ", mask_system_tags(text or "")).strip()
    if not compact:
        return []

    phrase_blocked = {
        "a",
        "an",
        "and",
        "answer",
        "answers",
        "assistant",
        "about",
        "after",
        "before",
        "by",
        "during",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "kind",
        "many",
        "memory",
        "name",
        "names",
        "of",
        "on",
        "question",
        "response",
        "session",
        "summary",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "type",
        "unknown",
        "user",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
    }

    raw_phrases: List[str] = []
    raw_phrases.extend(
        candidate.lower()
        for candidate in extract_entities_local(compact)
        if " " in candidate and len(candidate.split()) <= 5
    )

    lowered_tokens = re.findall(r"[A-Za-z0-9]+", compact.lower())
    for n in (2, 3):
        for index in range(len(lowered_tokens) - n + 1):
            gram = lowered_tokens[index : index + n]
            if any(token in phrase_blocked for token in gram):
                continue
            if not any(len(token) >= 5 for token in gram):
                continue
            raw_phrases.append(" ".join(gram))

    return normalize_keyword_values(raw_phrases, limit=max_keywords)


def extract_keyword_candidates(
    text: str,
    *,
    candidate_limit: int = 256,
    phrase_limit: int = 96,
) -> List[str]:
    prepared = prepare_keyword_text(text)
    if not prepared:
        return []
    phrases = extract_keyword_phrases_local(prepared, max_keywords=max(0, int(phrase_limit)))
    tokens = fallback_extract_keywords(prepared, limit=max(0, int(candidate_limit)))
    return normalize_keyword_values(phrases + tokens, limit=max(0, int(candidate_limit)))


def compute_idf(corpus_terms: Iterable[Sequence[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    num_docs = 0
    for terms in corpus_terms:
        num_docs += 1
        for term in set(terms or []):
            df[term] = df.get(term, 0) + 1
    if num_docs <= 0:
        return {}
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((num_docs + 1.0) / (freq + 1.0)) + 1.0
    return idf


FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "idf",
    "log_tf",
    "idf_log_tf",
    "len_norm",
    "token_norm",
    "is_phrase",
    "has_digit",
    "first_pos",
    "in_entities",
    "is_cjk",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def featurize_candidate(
    prepared_text: str,
    candidate: str,
    *,
    idf: Mapping[str, float],
    entities: Optional[Sequence[str]] = None,
) -> np.ndarray:
    compact = prepare_keyword_text(prepared_text).lower()
    term = str(candidate or "").strip().lower()
    if not compact or not term:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    tf = compact.count(term)
    log_tf = math.log1p(float(tf))
    idf_val = float(idf.get(term, 1.0))
    length = len(term)
    len_norm = min(float(length), 64.0) / 64.0
    token_count = len(term.split()) if " " in term else 1
    token_norm = min(float(token_count), 5.0) / 5.0
    is_phrase = 1.0 if " " in term else 0.0
    has_digit = 1.0 if any(ch.isdigit() for ch in term) else 0.0
    idx = compact.find(term)
    first_pos = 1.0 if idx < 0 else float(idx) / max(1.0, float(len(compact)))
    in_entities = 0.0
    if entities:
        lowered_entities = [str(ent or "").lower() for ent in entities]
        if any(term in ent for ent in lowered_entities):
            in_entities = 1.0
    is_cjk = 1.0 if re.search(r"[\u4e00-\u9fff]", term) else 0.0

    values = np.array(
        [
            1.0,
            idf_val,
            log_tf,
            idf_val * log_tf,
            len_norm,
            token_norm,
            is_phrase,
            has_digit,
            first_pos,
            in_entities,
            is_cjk,
        ],
        dtype=np.float32,
    )
    return values


@dataclass
class LinearKeywordRanker:
    weights: np.ndarray
    idf: Dict[str, float]
    candidate_limit: int = 256
    phrase_limit: int = 96
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    version: int = 1

    def score_candidates(self, prepared_text: str, candidates: Sequence[str]) -> List[Tuple[str, float]]:
        entities = extract_entities_local(prepared_text)
        scored: List[Tuple[str, float]] = []
        if not candidates:
            return scored
        for term in candidates:
            feat = featurize_candidate(prepared_text, term, idf=self.idf, entities=entities)
            score = float(_sigmoid(np.array([float(np.dot(self.weights, feat))], dtype=np.float32))[0])
            scored.append((term, score))
        return scored

    def extract(self, text: str, max_keywords: int = 12) -> List[str]:
        prepared_text = prepare_keyword_text(text)
        if not prepared_text:
            return []
        candidates = extract_keyword_candidates(
            prepared_text,
            candidate_limit=self.candidate_limit,
            phrase_limit=self.phrase_limit,
        )
        if not candidates:
            return []
        scored = self.score_candidates(prepared_text, candidates)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked = [term for term, _ in scored]
        return normalize_keyword_values(ranked, limit=max(0, int(max_keywords)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "feature_names": list(self.feature_names),
            "weights": [float(x) for x in self.weights.tolist()],
            "idf": {str(k): float(v) for k, v in (self.idf or {}).items()},
            "candidate_limit": int(self.candidate_limit),
            "phrase_limit": int(self.phrase_limit),
        }

    def save(self, path: str) -> None:
        if not path:
            raise ValueError("model path is required")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearKeywordRanker":
        feature_names = tuple(payload.get("feature_names") or FEATURE_NAMES)
        weights = payload.get("weights") or [0.0] * len(feature_names)
        if len(weights) != len(feature_names):
            raise ValueError(f"weights length {len(weights)} != feature_names length {len(feature_names)}")
        return cls(
            version=int(payload.get("version") or 1),
            feature_names=feature_names,
            weights=np.array([float(x) for x in weights], dtype=np.float32),
            idf={str(k): float(v) for k, v in (payload.get("idf") or {}).items()},
            candidate_limit=int(payload.get("candidate_limit") or 256),
            phrase_limit=int(payload.get("phrase_limit") or 96),
        )

    @classmethod
    def load(cls, path: str) -> "LinearKeywordRanker":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("invalid model payload")
        return cls.from_dict(payload)


def train_linear_ranker(
    training_pairs: Sequence[Tuple[str, Sequence[str]]],
    *,
    seed: int = 42,
    epochs: int = 6,
    lr: float = 0.25,
    l2: float = 0.001,
    candidate_limit: int = 256,
    phrase_limit: int = 96,
    negative_ratio: int = 4,
    min_negatives: int = 12,
    batch_size: int = 256,
) -> LinearKeywordRanker:
    rng = random.Random(int(seed))
    prepared_pairs: List[Tuple[str, List[str]]] = []
    for text, labels in training_pairs:
        prepared = prepare_keyword_text(text)
        clean_labels = normalize_keyword_values(list(labels or []), limit=64)
        if not prepared or not clean_labels:
            continue
        prepared_pairs.append((prepared, clean_labels))

    if not prepared_pairs:
        raise ValueError("no training data after preprocessing")

    idf = compute_idf(
        extract_keyword_candidates(text, candidate_limit=candidate_limit, phrase_limit=phrase_limit)
        for text, _ in prepared_pairs
    )

    xs: List[np.ndarray] = []
    ys: List[float] = []
    for text, labels in prepared_pairs:
        label_set = set(labels)
        candidates = extract_keyword_candidates(text, candidate_limit=candidate_limit, phrase_limit=phrase_limit)
        if not candidates:
            continue
        positives = [c for c in candidates if c in label_set]
        if not positives:
            continue
        negatives = [c for c in candidates if c not in label_set]
        if negatives:
            k = min(len(negatives), max(min_negatives, negative_ratio * len(positives)))
            sampled_negatives = rng.sample(negatives, k=k)
        else:
            sampled_negatives = []

        entities = extract_entities_local(text)
        for term in positives:
            xs.append(featurize_candidate(text, term, idf=idf, entities=entities))
            ys.append(1.0)
        for term in sampled_negatives:
            xs.append(featurize_candidate(text, term, idf=idf, entities=entities))
            ys.append(0.0)

    if not xs:
        raise ValueError("no training samples constructed (check labels/candidates)")

    x = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    weights = np.zeros(x.shape[1], dtype=np.float32)
    indices = list(range(len(y)))

    for _ in range(max(1, int(epochs))):
        rng.shuffle(indices)
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch_idx = indices[start : start + int(batch_size)]
            xb = x[batch_idx]
            yb = y[batch_idx]
            logits = xb @ weights
            probs = _sigmoid(logits)
            grad = (xb.T @ (probs - yb)) / float(len(batch_idx))
            if l2:
                grad = grad + float(l2) * weights
            weights = weights - float(lr) * grad.astype(np.float32)

    return LinearKeywordRanker(
        weights=weights,
        idf=idf,
        candidate_limit=int(candidate_limit),
        phrase_limit=int(phrase_limit),
        feature_names=FEATURE_NAMES,
        version=1,
    )
