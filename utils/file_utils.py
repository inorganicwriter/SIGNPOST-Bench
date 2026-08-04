"""
utils/file_utils.py — Shared file-level operations for evaluation scripts.

Eliminates duplicate code for JSONL rewriting, output ordering,
base ID extraction, and resume-state loading across all evaluation scripts.
"""

import json
import os


def rewrite_results_file(
    output_path: str,
    existing_entries: dict[str, dict],
    entry_order: list[str],
    updated_entries: list[dict],
) -> None:
    """Rewrite JSONL results with one latest row per filename, preserving order."""
    merged_entries = dict(existing_entries)
    ordered_filenames = list(entry_order)

    for entry in updated_entries:
        filename = entry.get("filename")
        if not filename:
            continue
        if filename not in merged_entries:
            ordered_filenames.append(filename)
        merged_entries[filename] = entry

    with open(output_path, "w", encoding="utf-8") as f:
        for filename in ordered_filenames:
            entry = merged_entries.get(filename)
            if entry is None:
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_output_order(
    image_files: list[str],
    previous_order: list[str],
) -> list[str]:
    """Build canonical output order for the current run.

    Current image files come first in their selected evaluation order, while any
    existing filenames outside the current run are appended in their previous
    file order so we do not silently drop older rows.
    """
    ordered = []
    seen: set[str] = set()
    for filename in image_files:
        if filename and filename not in seen:
            ordered.append(filename)
            seen.add(filename)
    for filename in previous_order:
        if filename and filename not in seen:
            ordered.append(filename)
            seen.add(filename)
    return ordered


def get_base_id(filename: str) -> str:
    """Extract base ID from a filename, handling variant suffixes and multi-underscore IDs.

    Compatible with googlesv/baidusv filenames that contain underscores
    in the base ID (e.g. "dVjmdcqKMaYNk3GiDcWKEg_270").
    """
    base = os.path.basename(filename or "")
    base = os.path.splitext(base)[0]

    # Try stripping known variant suffixes
    for suffix in ("_Blank", "_Original", "_similar_", "_random_", "_adversarial_"):
        idx = base.rfind(suffix)
        if idx != -1:
            return base[:idx]

    return base
