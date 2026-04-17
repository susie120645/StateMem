#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List


CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PACKAGE_DIR.parent


DATASET_SPECS: Dict[str, Dict[str, List[str] or str]] = {
    "dev_distractor": {
        "filename": "hotpot_dev_distractor_v1.json",
        "urls": [
            "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        ],
    },
    "dev_fullwiki": {
        "filename": "hotpot_dev_fullwiki_v1.json",
        "urls": [
            "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
        ],
    },
    "train": {
        "filename": "hotpot_train_v1.1.json",
        "urls": [
            "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official HotpotQA JSON files into data/hotpot.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev_distractor"],
        help="One or more of: dev_distractor, dev_fullwiki, train, all",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "hotpot"),
        help="Target directory for downloaded HotpotQA files.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    parser.add_argument("--timeout", type=int, default=60, help="Network timeout in seconds.")
    return parser.parse_args()


def resolve_splits(requested: List[str]) -> List[str]:
    normalized = [item.strip().lower() for item in requested if item and item.strip()]
    if not normalized:
        raise ValueError("At least one split must be provided.")
    if "all" in normalized:
        return list(DATASET_SPECS.keys())

    unknown = [item for item in normalized if item not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown HotpotQA split(s): {', '.join(unknown)}")
    return normalized


def human_size(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def download_file(urls: List[str], destination: Path, timeout: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent)) as tmp_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                os.replace(tmp_file.name, destination)
                return url
        except urllib.error.URLError:
            continue
        except Exception:
            if "tmp_file" in locals():
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
            raise

    raise RuntimeError(f"Failed to download {destination.name} from all configured mirrors.")


def main() -> int:
    args = parse_args()
    target_dir = Path(args.data_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        splits = resolve_splits(args.splits)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"HotpotQA target directory: {target_dir}")
    for split in splits:
        spec = DATASET_SPECS[split]
        path = target_dir / str(spec["filename"])
        if path.exists() and path.stat().st_size > 0 and not args.force:
            print(f"[skip] {split}: {path} ({human_size(path.stat().st_size)})")
            continue

        print(f"[download] {split}: {path.name}")
        used_url = download_file(list(spec["urls"]), path, timeout=args.timeout)
        print(f"[done] {split}: {path} ({human_size(path.stat().st_size)}) <- {used_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
