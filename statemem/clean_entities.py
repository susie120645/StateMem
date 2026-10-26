#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .entity_filters import filter_entity_values, filter_semantic_edge_values
except ImportError:
    from entity_filters import filter_entity_values, filter_semantic_edge_values


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _clean_metadata(metadata: Any) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats = {"removed_entities": 0, "removed_edges": 0, "changed_records": 0}
    if not isinstance(metadata, dict):
        return {}, stats

    cleaned = dict(metadata)
    original_entities = cleaned.get("entities", [])
    original_edges = cleaned.get("semantic_edges", [])
    entities = filter_entity_values(original_entities, limit=24)
    edges = filter_semantic_edge_values(original_edges, limit=48)

    if entities != original_entities:
        cleaned["entities"] = entities
        stats["removed_entities"] += max(0, len(original_entities) - len(entities)) if isinstance(original_entities, list) else 0
    if edges != original_edges:
        cleaned["semantic_edges"] = edges
        stats["removed_edges"] += max(0, len(original_edges) - len(edges)) if isinstance(original_edges, list) else 0
    if cleaned != metadata:
        stats["changed_records"] += 1
    return cleaned, stats


def _clean_short_term_file(path: Path) -> Tuple[bool, Dict[str, int], Any]:
    payload = _load_json(path)
    stats = {"removed_entities": 0, "removed_edges": 0, "changed_records": 0}
    if not isinstance(payload, list):
        return False, stats, payload

    changed = False
    cleaned_pages = []
    for page in payload:
        if not isinstance(page, dict):
            cleaned_pages.append(page)
            continue
        cleaned_page = dict(page)
        metadata = page.get("metadata")
        cleaned_metadata, metadata_stats = _clean_metadata(metadata)
        if metadata_stats["changed_records"]:
            cleaned_page["metadata"] = cleaned_metadata
            changed = True
        for key, value in metadata_stats.items():
            stats[key] += value

        meta_data = page.get("meta_data")
        if isinstance(meta_data, dict):
            cleaned_meta_data, meta_stats = _clean_metadata(meta_data)
            if meta_stats["changed_records"]:
                cleaned_page["meta_data"] = cleaned_meta_data
                changed = True
            for key, value in meta_stats.items():
                stats[key] += value
        cleaned_pages.append(cleaned_page)

    return changed, stats, cleaned_pages


def _clean_long_term_file(path: Path) -> Tuple[bool, Dict[str, int], Any]:
    payload = _load_json(path)
    stats = {"removed_entities": 0, "removed_edges": 0, "changed_records": 0}
    if not isinstance(payload, dict):
        return False, stats, payload

    knowledge_base = payload.get("knowledge_base", [])
    if not isinstance(knowledge_base, list):
        return False, stats, payload

    changed = False
    cleaned_entries = []
    for entry in knowledge_base:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue
        cleaned_entry = dict(entry)

        original_entities = entry.get("entities", [])
        cleaned_entities = filter_entity_values(original_entities, limit=24)
        if cleaned_entities != original_entities:
            cleaned_entry["entities"] = cleaned_entities
            stats["removed_entities"] += max(0, len(original_entities) - len(cleaned_entities)) if isinstance(original_entities, list) else 0
            changed = True

        original_edges = entry.get("semantic_edges", [])
        cleaned_edges = filter_semantic_edge_values(original_edges, limit=48)
        if cleaned_edges != original_edges:
            cleaned_entry["semantic_edges"] = cleaned_edges
            stats["removed_edges"] += max(0, len(original_edges) - len(cleaned_edges)) if isinstance(original_edges, list) else 0
            changed = True

        cleaned_metadata, metadata_stats = _clean_metadata(entry.get("metadata"))
        if metadata_stats["changed_records"]:
            cleaned_entry["metadata"] = cleaned_metadata
            changed = True
        for key, value in metadata_stats.items():
            stats[key] += value

        if cleaned_entry != entry:
            stats["changed_records"] += 1
        cleaned_entries.append(cleaned_entry)

    cleaned_payload = dict(payload)
    cleaned_payload["knowledge_base"] = cleaned_entries
    return changed, stats, cleaned_payload


def _clean_forget_pool_file(path: Path) -> Tuple[bool, Dict[str, int], List[str]]:
    stats = {"removed_entities": 0, "removed_edges": 0, "changed_records": 0}
    changed = False
    cleaned_lines: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                cleaned_lines.append(raw_line)
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                cleaned_lines.append(raw_line)
                continue

            cleaned_record = dict(record)
            original_entities = record.get("entities", record.get("ent", []))
            cleaned_entities = filter_entity_values(original_entities, limit=24)
            if "entities" in record and cleaned_entities != original_entities:
                cleaned_record["entities"] = cleaned_entities
                stats["removed_entities"] += max(0, len(original_entities) - len(cleaned_entities)) if isinstance(original_entities, list) else 0
                changed = True
            if "ent" in record and cleaned_entities != original_entities:
                cleaned_record["ent"] = cleaned_entities
                stats["removed_entities"] += max(0, len(original_entities) - len(cleaned_entities)) if isinstance(original_entities, list) else 0
                changed = True

            original_edges = record.get("semantic_edges", [])
            cleaned_edges = filter_semantic_edge_values(original_edges, limit=48)
            if cleaned_edges != original_edges:
                cleaned_record["semantic_edges"] = cleaned_edges
                stats["removed_edges"] += max(0, len(original_edges) - len(cleaned_edges)) if isinstance(original_edges, list) else 0
                changed = True

            cleaned_metadata, metadata_stats = _clean_metadata(record.get("metadata"))
            if metadata_stats["changed_records"]:
                cleaned_record["metadata"] = cleaned_metadata
                changed = True
            for key, value in metadata_stats.items():
                stats[key] += value

            if cleaned_record != record:
                stats["changed_records"] += 1
            cleaned_lines.append(json.dumps(cleaned_record, ensure_ascii=False) + "\n")

    return changed, stats, cleaned_lines


def _backup_file(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def _resolve_user_dirs(storage_root: Path, user_ids: Iterable[str]) -> List[Path]:
    users_root = storage_root / "users"
    if users_root.is_dir():
        if user_ids:
            return [users_root / user_id for user_id in user_ids if (users_root / user_id).exists()]
        return sorted(path for path in users_root.iterdir() if path.is_dir())

    if user_ids:
        return [storage_root / user_id for user_id in user_ids if (storage_root / user_id).exists()]
    if (storage_root / "long_term").exists():
        return [storage_root]
    return sorted(path for path in storage_root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean structural-label entities from stored statemem data.")
    parser.add_argument("--storage-root", required=True, help="Root directory containing statemem user storage.")
    parser.add_argument("--user-id", action="append", default=[], help="Optional user id to clean. Repeatable.")
    parser.add_argument("--apply", action="store_true", help="Write cleaned data back to disk.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files before rewriting.")
    args = parser.parse_args()

    storage_root = Path(args.storage_root).expanduser().resolve()
    if not storage_root.exists():
        raise SystemExit(f"storage root does not exist: {storage_root}")

    user_dirs = _resolve_user_dirs(storage_root, args.user_id)
    if not user_dirs:
        raise SystemExit(f"no user directories found under: {storage_root}")

    overall = {
        "users_scanned": 0,
        "files_changed": 0,
        "removed_entities": 0,
        "removed_edges": 0,
        "changed_records": 0,
    }

    for user_dir in user_dirs:
        overall["users_scanned"] += 1
        file_targets = [
            ("short_term", user_dir / "short_term.json", _clean_short_term_file),
            ("forget_pool", user_dir / "forget_pool.jsonl", _clean_forget_pool_file),
            ("long_term", user_dir / "long_term" / "memory_graph.json", _clean_long_term_file),
        ]

        for label, path, cleaner in file_targets:
            if not path.exists():
                continue
            changed, stats, cleaned_payload = cleaner(path)
            overall["removed_entities"] += stats["removed_entities"]
            overall["removed_edges"] += stats["removed_edges"]
            overall["changed_records"] += stats["changed_records"]
            if not changed:
                print(f"[skip] {user_dir.name} {label}: no structural entities found")
                continue

            overall["files_changed"] += 1
            print(
                f"[clean] {user_dir.name} {label}: "
                f"records={stats['changed_records']} removed_entities={stats['removed_entities']} removed_edges={stats['removed_edges']}"
            )
            if not args.apply:
                continue

            if not args.no_backup:
                backup_path = _backup_file(path)
                print(f"  backup: {backup_path}")

            if label == "forget_pool":
                with path.open("w", encoding="utf-8") as handle:
                    handle.writelines(cleaned_payload)
            else:
                _write_json(path, cleaned_payload)

    mode = "apply" if args.apply else "dry-run"
    print(
        f"[summary] mode={mode} users={overall['users_scanned']} files_changed={overall['files_changed']} "
        f"removed_entities={overall['removed_entities']} removed_edges={overall['removed_edges']} "
        f"changed_records={overall['changed_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
