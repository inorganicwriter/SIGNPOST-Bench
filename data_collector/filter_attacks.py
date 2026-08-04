"""
filter_attacks.py — Remove watermark / irrelevant text from attacks.jsonl.

Patterns are imported from text_patterns.py (the single source of truth).
This module handles only filtering logic and CLI — no pattern definitions.

Usage:
    python filter_attacks.py --data-dir ./data --dataset googlesv
    python filter_attacks.py --attacks-file /path/to/attacks.jsonl --dry-run
"""

import argparse
import json
import os

from data_collector.text_patterns import (
    INVALID_IMAGE_KEYWORDS,
    URL_KEYWORDS,
    URL_PATTERNS,
    WATERMARK_KEYWORDS,
    WATERMARK_PATTERNS,
    WATERMARK_SERVICE,
)


def is_watermark_or_irrelevant(text: str) -> str:
    """Check if text is a watermark or irrelevant.
    Returns reason string if invalid, empty string if valid."""
    if not text or not text.strip():
        return "empty"

    text_lower = text.strip().lower()
    stripped = text.strip()

    if text_lower in WATERMARK_SERVICE:
        return f"watermark_service: {text_lower}"

    for sub in WATERMARK_KEYWORDS:
        if sub in text_lower:
            return f"watermark_kw: {sub}"

    for pat in WATERMARK_PATTERNS:
        if pat.search(stripped):
            return f"watermark_pattern: {pat.pattern}"

    for pat in URL_PATTERNS:
        if pat.search(stripped):
            return f"url_pattern: {pat.pattern}"

    for kw in URL_KEYWORDS:
        if kw in text_lower:
            return f"url_kw: {kw}"

    return ""


def is_invalid_image_text(texts: list) -> str:
    """Check if texts come from a placeholder/expired image."""
    all_text = " ".join(t.get("original_text", "") for t in texts).lower()
    for kw in INVALID_IMAGE_KEYWORDS:
        if kw in all_text:
            return f"invalid_image: {kw}"
    return ""


def filter_attacks_file(input_path: str, output_path: str = None, dry_run: bool = False) -> dict:
    """Filter a single attacks.jsonl file.

    Returns dict with stats: total_entries, total_texts, filtered_texts,
    removed_entries, kept_entries.
    """
    if output_path is None:
        output_path = input_path

    entries = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    stats: dict = {
        "total_entries": len(entries),
        "total_texts": 0,
        "filtered_texts": 0,
        "removed_entries": 0,
        "kept_entries": 0,
        "invalid_image_entries": 0,
        "filtered_examples": [],
    }

    filtered_entries = []

    for entry in entries:
        texts = entry.get("texts", [])
        stats["total_texts"] += len(texts)

        # Step 1: Whole-image check for expired/placeholder images
        invalid_reason = is_invalid_image_text(texts)
        if invalid_reason:
            stats["invalid_image_entries"] += 1
            stats["removed_entries"] += 1
            stats["filtered_texts"] += len(texts)
            if len(stats["filtered_examples"]) < 20:
                all_orig = ", ".join(t.get("original_text", "") for t in texts)
                stats["filtered_examples"].append((all_orig[:80], invalid_reason))
            continue

        # Step 2: Per-text watermark check
        clean_texts = []
        for t in texts:
            original = t.get("original_text", "")
            reason = is_watermark_or_irrelevant(original)
            if reason:
                stats["filtered_texts"] += 1
                if len(stats["filtered_examples"]) < 20:
                    stats["filtered_examples"].append((original, reason))
            else:
                clean_texts.append(t)

        if clean_texts:
            entry["texts"] = clean_texts
            filtered_entries.append(entry)
            stats["kept_entries"] += 1
        else:
            stats["removed_entries"] += 1

    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in filtered_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


# ===========================================================================
#  CLI
# ===========================================================================


def _print_stats(label: str, stats: dict) -> None:
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total texts:   {stats['total_texts']}")
    print(f"  Filtered out:  {stats['filtered_texts']} texts")
    if stats.get("invalid_image_entries", 0) > 0:
        print(f"  Invalid images: {stats['invalid_image_entries']} (placeholder/expired)")
    print(f"  Entries removed (total): {stats['removed_entries']}")
    print(f"  Entries kept:  {stats['kept_entries']}")


def _print_examples(examples: list) -> None:
    print("\n  Filtered examples:")
    for text, reason in examples[:10]:
        print(f'    X "{text}" -> {reason}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter watermarks and irrelevant text from attacks.jsonl")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Filter specific dataset (e.g. googlesv). Omit to process all."
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to data directory containing <dataset>/attacks.jsonl"
    )
    parser.add_argument("--attacks-file", type=str, default=None, help="Filter a single attacks.jsonl file directly")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not modify files")
    parser.add_argument("--backup", action="store_true", default=True, help="Create .bak backup (default: True)")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip backup creation")
    args = parser.parse_args()

    print("=" * 60)
    print("  SIGNPOST-Bench: Attack Text Filter")
    mode = "DRY RUN (no changes)" if args.dry_run else "LIVE (will modify files)"
    print(f"  Mode: {mode}")
    print("=" * 60)

    # Single-file mode
    if args.attacks_file:
        if not os.path.exists(args.attacks_file):
            print(f"File not found: {args.attacks_file}")
            return
        if not args.dry_run and args.backup:
            import shutil

            backup_path = args.attacks_file + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(args.attacks_file, backup_path)
                print(f"  Backup: {backup_path}")
        stats = filter_attacks_file(args.attacks_file, dry_run=args.dry_run)
        _print_stats("single", stats)
        if stats["filtered_examples"]:
            _print_examples(stats["filtered_examples"])
        return

    # Dataset mode
    datasets = [args.dataset] if args.dataset else ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]
    grand_total = {"texts": 0, "filtered": 0, "removed_entries": 0}

    for ds in datasets:
        attacks_file = os.path.join(args.data_dir, ds, "attacks.jsonl")
        if not os.path.exists(attacks_file):
            print(f"\n[{ds}] attacks.jsonl not found, skipping.")
            continue

        print(f"\n{'─' * 40}")
        print(f"  Dataset: {ds}")
        print(f"{'─' * 40}")

        if not args.dry_run and args.backup:
            import shutil

            backup_path = attacks_file + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(attacks_file, backup_path)
                print(f"  Backup: {backup_path}")

        stats = filter_attacks_file(attacks_file, dry_run=args.dry_run)
        grand_total["texts"] += stats["total_texts"]
        grand_total["filtered"] += stats["filtered_texts"]
        grand_total["removed_entries"] += stats["removed_entries"]

        _print_stats(ds, stats)
        if stats["filtered_examples"]:
            _print_examples(stats["filtered_examples"])

    print(f"\n{'=' * 60}")
    print("  GRAND TOTAL")
    print(f"  Texts filtered: {grand_total['filtered']} / {grand_total['texts']}")
    print(f"  Entries removed: {grand_total['removed_entries']}")
    if not args.dry_run:
        print("  Files updated. Backups saved as .bak")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
