"""
sample_probing_subset.py
========================
Sample a balanced subset from attacks.jsonl for conflict probing experiments.

Produces a JSONL file listing filenames to evaluate, ensuring balanced coverage
across datasets and attack types.

Usage:
    python sample_probing_subset.py \
        --data-root /path/to/SIGNPOST-Bench \
        --datasets im2gps3k yfcc4k googlesv baidusv \
        --per-dataset 250 \
        --attack-types random adversarial \
        --output analysis/subsets/probing_subset.jsonl \
        --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT, get_subset_output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample probing subset from SIGNPOST-Bench")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help="Root data directory (default: SIGNPOST data root from config.py)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["im2gps3k", "yfcc4k", "googlesv", "baidusv"],
        help="Datasets to sample from",
    )
    parser.add_argument(
        "--per-dataset", type=int, default=250, help="Number of entries to sample per dataset (default: 250)"
    )
    parser.add_argument(
        "--attack-types",
        type=str,
        nargs="+",
        default=["random", "adversarial"],
        help="Attack types to include (default: random adversarial)",
    )
    parser.add_argument(
        "--include-similar", action="store_true", help="Also include a small number of Similar samples for sanity check"
    )
    parser.add_argument(
        "--similar-ratio",
        type=float,
        default=0.1,
        help="Ratio of Similar samples relative to per-dataset (default: 0.1)",
    )
    parser.add_argument(
        "--output", type=str, default=str(get_subset_output_path("probing_subset.jsonl")), help="Output JSONL file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_attacks(data_root, dataset_name):
    """Load attacks.jsonl for a dataset."""
    attacks_file = os.path.join(data_root, dataset_name, "attacks.jsonl")
    entries = []
    if not os.path.exists(attacks_file):
        print(f"  Warning: {attacks_file} not found, skipping.")
        return entries

    with open(attacks_file, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def get_image_filenames(data_root, dataset_name, attack_type, entries):
    """Get actual image filenames for a given attack type from the images directory."""
    img_dir = os.path.join(data_root, dataset_name, "images", attack_type.capitalize())
    if not os.path.exists(img_dir):
        print(f"  Warning: {img_dir} not found.")
        return []

    # Build a map from base_id to filenames
    files = os.listdir(img_dir)
    return files


def sample_entries(entries, n, seed):
    """Sample n entries from a list, or return all if fewer than n."""
    if len(entries) <= n:
        return entries
    random.seed(seed)
    return random.sample(entries, n)


def main():
    args = parse_args()
    random.seed(args.seed)

    all_samples = []
    stats = {}

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        entries = load_attacks(args.data_root, dataset)
        print(f"  Total attacks entries: {len(entries)}")

        if not entries:
            continue

        # Sample entries
        sampled = sample_entries(entries, args.per_dataset, args.seed + hash(dataset))
        print(f"  Sampled {len(sampled)} entries")

        dataset_count = 0
        for entry in sampled:
            original_filename = entry.get("original_filename", "")
            base_name = os.path.splitext(original_filename)[0]

            # Get attack texts
            texts_list = entry.get("texts", None)
            if texts_list is None:
                attack_dict = entry.get("attacks", {})
                texts_list = [{"attacks": attack_dict}]

            for attack_type in args.attack_types:
                # Check if this attack type has valid text
                has_text = False
                for t in texts_list:
                    if t.get("attacks", {}).get(attack_type):
                        has_text = True
                        break

                if not has_text:
                    continue

                # Find actual filename in images directory
                img_dir = os.path.join(args.data_root, dataset, "images", attack_type.capitalize())
                if os.path.exists(img_dir):
                    matching = [f for f in os.listdir(img_dir) if f.startswith(base_name + f"_{attack_type}")]
                    for fname in matching:
                        all_samples.append(
                            {
                                "filename": fname,
                                "dataset": dataset,
                                "attack_type": attack_type,
                                "original_filename": original_filename,
                                "base_name": base_name,
                                "img_dir": img_dir,
                            }
                        )
                        dataset_count += 1

            # Also add corresponding Blank image for TBS pairing
            blank_dir = os.path.join(args.data_root, dataset, "images", "Blank")
            if os.path.exists(blank_dir):
                blank_file = f"{base_name}_Blank.png"
                if os.path.exists(os.path.join(blank_dir, blank_file)):
                    all_samples.append(
                        {
                            "filename": blank_file,
                            "dataset": dataset,
                            "attack_type": "blank",
                            "original_filename": original_filename,
                            "base_name": base_name,
                            "img_dir": blank_dir,
                        }
                    )

        # Similar sanity check samples
        if args.include_similar:
            n_similar = max(1, int(args.per_dataset * args.similar_ratio))
            similar_sampled = sample_entries(entries, n_similar, args.seed + hash(dataset) + 1)
            for entry in similar_sampled:
                original_filename = entry.get("original_filename", "")
                base_name = os.path.splitext(original_filename)[0]
                img_dir = os.path.join(args.data_root, dataset, "images", "Similar")
                if os.path.exists(img_dir):
                    matching = [f for f in os.listdir(img_dir) if f.startswith(base_name + "_similar")]
                    for fname in matching:
                        all_samples.append(
                            {
                                "filename": fname,
                                "dataset": dataset,
                                "attack_type": "similar",
                                "original_filename": original_filename,
                                "base_name": base_name,
                                "img_dir": img_dir,
                            }
                        )

        stats[dataset] = dataset_count
        print(f"  Generated {dataset_count} attack samples (+ blanks)")

    # Write output
    output_path = args.output
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print("Probing Subset Summary")
    print(f"{'=' * 50}")
    print(f"Total samples: {len(all_samples)}")
    for ds, count in stats.items():
        print(f"  {ds}: {count} attack samples")

    # Count by attack type
    by_type = {}
    for s in all_samples:
        at = s["attack_type"]
        by_type[at] = by_type.get(at, 0) + 1
    print("\nBy attack type:")
    for at, count in sorted(by_type.items()):
        print(f"  {at}: {count}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
