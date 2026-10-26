#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets"

QMSUM_SPLITS = ("train", "val", "test")
QMSUM_URL_TEMPLATE = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/{split}.jsonl"

DIALSIM_DATASETS = {
    "dialsim-friends": "jiho283/dialsim-friends",
    "dialsim-bigbang": "jiho283/dialsim-bigbang",
    "dialsim-theoffice": "jiho283/dialsim-theoffice",
}


def _http_get_bytes(url: str, *, timeout: int = 300) -> bytes:
    with requests.get(
        url,
        headers={
            "User-Agent": "RememDatasetPrep/1.0",
            "Accept": "application/json, text/plain, */*",
        },
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()
        chunks: List[bytes] = []
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)


def _http_get_json(url: str, *, timeout: int = 120) -> Dict:
    payload = _http_get_bytes(url, timeout=timeout)
    return json.loads(payload.decode("utf-8"))


def _download_qmsum(output_root: Path, splits: Iterable[str]) -> Dict[str, int]:
    qmsum_root = output_root / "qmsum"
    qmsum_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split in splits:
        url = QMSUM_URL_TEMPLATE.format(split=split)
        target = qmsum_root / f"{split}.jsonl"
        print(f"[download] qmsum:{split} -> {target}")
        payload = _http_get_bytes(url)
        target.write_bytes(payload)
        line_count = sum(1 for line in payload.decode("utf-8").splitlines() if line.strip())
        counts[split] = line_count
    return counts


def _fetch_dialsim_rows(dataset_name: str, *, split: str = "train", page_size: int = 100) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    total = None
    base_url = "https://datasets-server.huggingface.co/rows"

    while True:
        query = urlencode(
            {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _http_get_json(f"{base_url}?{query}")
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


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _download_dialsim(output_root: Path, dataset_keys: Iterable[str]) -> Dict[str, int]:
    dialsim_root = output_root / "dialsim"
    dialsim_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for key in dataset_keys:
        dataset_name = DIALSIM_DATASETS[key]
        print(f"[download] {key}:train -> {dataset_name}")
        rows = _fetch_dialsim_rows(dataset_name, split="train")
        short_name = key.removeprefix("dialsim-")
        target = dialsim_root / short_name / "train.jsonl"
        counts[short_name] = _write_jsonl(target, rows)
    return counts


def _write_manifest(output_root: Path, manifest: Dict) -> Path:
    target = output_root / "benchmark_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets used by statemem runners.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset group to prepare. Repeatable. "
            "Choices: qmsum, dialsim-friends, dialsim-bigbang, dialsim-theoffice, dialsim-all. "
            "Default: qmsum,dialsim-friends."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where local dataset files will be stored.",
    )
    parser.add_argument(
        "--qmsum-split",
        action="append",
        default=[],
        help="QMSum split to download. Repeatable. Choices: train,val,test. Default: test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_selection = [str(item).strip().lower() for item in args.dataset if str(item).strip()]
    selection = raw_selection or ["qmsum", "dialsim-friends"]

    expanded: List[str] = []
    for item in selection:
        if item == "dialsim-all":
            expanded.extend(sorted(DIALSIM_DATASETS))
        else:
            expanded.append(item)

    invalid = [item for item in expanded if item not in {"qmsum", *DIALSIM_DATASETS.keys()}]
    if invalid:
        raise SystemExit(f"Unsupported --dataset values: {', '.join(sorted(invalid))}")

    qmsum_splits = [str(item).strip().lower() for item in args.qmsum_split if str(item).strip()]
    if not qmsum_splits:
        qmsum_splits = ["test"]
    invalid_splits = [item for item in qmsum_splits if item not in QMSUM_SPLITS]
    if invalid_splits:
        raise SystemExit(f"Unsupported --qmsum-split values: {', '.join(sorted(invalid_splits))}")

    output_root = Path(str(args.output_root)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict] = {"output_root": str(output_root), "datasets": {}}

    try:
        if "qmsum" in expanded:
            counts = _download_qmsum(output_root, qmsum_splits)
            manifest["datasets"]["qmsum"] = {
                "path": str(output_root / "qmsum"),
                "splits": counts,
            }

        dialsim_keys = [item for item in expanded if item in DIALSIM_DATASETS]
        if dialsim_keys:
            counts = _download_dialsim(output_root, dialsim_keys)
            manifest["datasets"]["dialsim"] = {
                "path": str(output_root / "dialsim"),
                "rows": counts,
            }
    except requests.RequestException as exc:
        raise SystemExit(f"Dataset download failed: {exc}") from exc

    manifest_path = _write_manifest(output_root, manifest)
    print("Prepared datasets:")
    for name, payload in manifest["datasets"].items():
        print(f"- {name}: {json.dumps(payload, ensure_ascii=False)}")
    print(f"- manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
