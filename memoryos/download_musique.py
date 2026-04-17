#!/usr/bin/env python3
import argparse
import os
import zipfile
from pathlib import Path
from typing import Iterable, List


CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PACKAGE_DIR.parent

OFFICIAL_ANSWERABLE_FILE_ID = "1YU8-72Yq6M6iA43KPHj07M1ApxT7aDsA"
OFFICIAL_FULL_FILE_ID = "1fVkVQZ-f5Q4hCrP0r1sJQw6HfMdbf4z2"
HF_REPO_ID = "dgslibisey/MuSiQue"
HF_FILES = [
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_train.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MuSiQue into data/musique.")
    parser.add_argument(
        "--source",
        choices=["auto", "gdrive", "hf"],
        default="auto",
        help="Download source preference. auto tries official Google Drive first, then HF mirror.",
    )
    parser.add_argument(
        "--variant",
        choices=["ans", "full"],
        default="ans",
        help="Dataset variant. HF fallback currently supports ans only.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "musique"),
        help="Destination directory.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if files already exist.")
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Mirror endpoint for Hugging Face fallback.",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def unzip_into(zip_path: Path, data_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
        for member in archive.namelist():
            extracted.append(data_dir / member)
    return extracted


def all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def expected_paths(data_dir: Path, variant: str) -> List[Path]:
    if variant == "ans":
        return [
            data_dir / "musique_ans_v1.0_dev.jsonl",
            data_dir / "musique_ans_v1.0_train.jsonl",
        ]
    return [
        data_dir / "musique_full_v1.0_dev.jsonl",
        data_dir / "musique_full_v1.0_train.jsonl",
    ]


def ensure_gdrive_download(data_dir: Path, variant: str, force: bool) -> List[Path]:
    import gdown

    if variant == "ans":
        file_id = OFFICIAL_ANSWERABLE_FILE_ID
        zip_name = "musique_ans_v1.0.zip"
    else:
        file_id = OFFICIAL_FULL_FILE_ID
        zip_name = "musique_full_v1.0.zip"
    expected = expected_paths(data_dir, variant)

    if all_exist(expected) and not force:
        return expected

    zip_path = data_dir / zip_name
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    if not result:
        raise RuntimeError(f"gdown returned no path for {zip_name}")
    unzip_into(zip_path, data_dir)
    if not all_exist(expected):
        raise RuntimeError(f"Expected files missing after extracting {zip_path}")
    return expected


def ensure_hf_download(data_dir: Path, variant: str, force: bool, hf_endpoint: str) -> List[Path]:
    if variant != "ans":
        raise RuntimeError("HF fallback currently supports only the answerable MuSiQue release.")

    paths = [data_dir / name for name in HF_FILES]
    if all_exist(paths) and not force:
        return paths

    os.environ["HF_ENDPOINT"] = hf_endpoint
    from huggingface_hub import hf_hub_download

    downloaded: List[Path] = []
    for name in HF_FILES:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=name,
            repo_type="dataset",
            local_dir=str(data_dir),
            force_download=force,
        )
        downloaded.append(Path(path))
    if not all_exist(downloaded):
        raise RuntimeError("HF fallback completed but expected MuSiQue files are still missing.")
    return downloaded


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"MuSiQue target directory: {data_dir}")
    local_expected = expected_paths(data_dir, args.variant)
    if all_exist(local_expected) and not args.force:
        for path in local_expected:
            print(f"[done:local] {path.name} ({human_size(path.stat().st_size)})")
        return 0

    if args.source in {"auto", "gdrive"}:
        try:
            downloaded = ensure_gdrive_download(data_dir, args.variant, args.force)
            for path in downloaded:
                print(f"[done:gdrive] {path.name} ({human_size(path.stat().st_size)})")
            return 0
        except Exception as exc:
            if args.source == "gdrive":
                raise
            print(f"[warn] Official Google Drive download failed: {exc}")

    downloaded = ensure_hf_download(data_dir, args.variant, args.force, args.hf_endpoint)
    for path in downloaded:
        print(f"[done:hf] {path.name} ({human_size(path.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
