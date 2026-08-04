"""
Stratified Sampling Script for Google Street View Dataset.

Samples images proportionally by country from the panoids.csv metadata,
then copies/symlinks sampled images to a flat output directory for pipeline processing.

Supports resume (--resume) and multi-process parallelism (--num-workers).

Usage:
    python sample_googlesv.py \
        --metadata /path/to/panoids.csv \
        --images-root /path/to/GoogleSV/images \
        --output-dir /path/to/output/sampled_images \
        --output-csv /path/to/output/googlesv_metadata_address.csv \
        --sample-rate 0.01 \
        --num-workers 4 \
        --resume
"""

import argparse
import csv
import logging
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified Sampling for Google Street View")
    parser.add_argument("--metadata", type=str, required=True, help="Path to panoids.csv")
    parser.add_argument("--images-root", type=str, required=True, help="Root directory of images")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to copy/symlink sampled images (flat structure)"
    )
    parser.add_argument(
        "--output-csv", type=str, required=True, help="Output CSV with sampled metadata (pipeline-compatible)"
    )
    parser.add_argument(
        "--sample-rate", type=float, default=0.01, help="Sampling rate per country (default: 0.01 = 1%%)"
    )
    parser.add_argument("--min-per-country", type=int, default=10, help="Minimum samples per country (default: 10)")
    parser.add_argument("--max-total", type=int, default=0, help="Maximum total samples (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying (saves disk space)")
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of parallel workers for file copy (default: 1)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV")
    return parser.parse_args()


def _resolve_source(images_root: Path, country: str, city: str, panoid: str, angle: str) -> Path:
    """Construct source path: images_root/country/city/panoid_angle.jpg"""
    return images_root / country / city / f"{panoid}_{angle}.jpg"


def _copy_one(args_tuple: tuple) -> dict | None:
    """Worker function for parallel copy/symlink. Returns output row dict or None."""
    row, images_root_str, output_dir_str, use_symlink = args_tuple
    images_root = Path(images_root_str)
    output_dir = Path(output_dir_str)
    panoid = row.get("panoid", "")
    angle = row.get("angle", "")
    country = row.get("country", "")
    city = row.get("city", "")
    lat = row.get("lat", "")
    lon = row.get("lon", "")

    src_filename = f"{panoid}_{angle}.jpg"
    src_path = _resolve_source(images_root, country, city, panoid, angle)

    if not src_path.exists():
        return None

    dst_path = output_dir / src_filename

    try:
        if use_symlink:
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            dst_path.symlink_to(src_path)
        else:
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

        return {
            "photo_id": f"{panoid}_{angle}",
            "panoid": panoid,
            "latitude": lat,
            "longitude": lon,
            "country": country,
            "city": city,
            "angle": angle,
            "source_path": str(src_path),
        }
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume support ----
    already_processed: set[str] = set()
    if args.resume and os.path.exists(args.output_csv):
        with open(args.output_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_processed.add(row.get("photo_id", ""))
        logging.info(f"Resume mode: {len(already_processed)} images already processed")

    # ---- Step 1: Read metadata and group by country ----
    logging.info(f"Reading metadata from {args.metadata}...")
    by_country: defaultdict[str, list] = defaultdict(list)
    total_rows = 0

    with open(args.metadata, encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            country = row.get("country", "Unknown")
            by_country[country].append(row)

    logging.info(f"Total records: {total_rows:,}")
    logging.info(f"Countries: {len(by_country)}")

    # ---- Step 2: Stratified sampling ----
    sampled: list[dict] = []
    for country, rows in sorted(by_country.items()):
        n = max(args.min_per_country, int(len(rows) * args.sample_rate))
        n = min(n, len(rows))
        selected = random.sample(rows, n)
        sampled.extend(selected)
        logging.info(f"  {country}: {len(rows):>8,} -> {n:>5,} sampled")

    if args.max_total > 0 and len(sampled) > args.max_total:
        random.shuffle(sampled)
        sampled = sampled[: args.max_total]
        logging.info(f"Capped total to {args.max_total}")

    # Filter out already processed
    if already_processed:
        before = len(sampled)
        sampled = [r for r in sampled if f"{r.get('panoid', '')}_{r.get('angle', '')}" not in already_processed]
        logging.info(f"After resume filter: {len(sampled)} remaining (skipped {before - len(sampled)})")

    logging.info(f"Candidates to process: {len(sampled):,}")

    # ---- Step 3: Copy/symlink (parallel if num_workers > 1) ----
    logging.info(f"{'Symlinking' if args.symlink else 'Copying'} images to {output_dir}...")

    fieldnames = ["photo_id", "panoid", "latitude", "longitude", "country", "city", "angle", "source_path"]
    resume_mode = args.resume and os.path.exists(args.output_csv)
    mode = "a" if resume_mode else "w"

    csv_file = open(args.output_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not resume_mode:
        writer.writeheader()

    tasks = [(row, str(images_root), str(output_dir), args.symlink) for row in sampled]
    success = 0
    missing = 0

    try:
        if args.num_workers > 1:
            logging.info(f"Starting {args.num_workers} parallel workers...")
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = list(executor.map(_copy_one, tasks))
            for result in tqdm(futures, desc="Collecting"):
                if result is None:
                    missing += 1
                else:
                    writer.writerow(result)
                    success += 1
                    if success % 500 == 0:
                        csv_file.flush()
        else:
            for task in tqdm(tasks, desc="Processing"):
                result = _copy_one(task)
                if result is None:
                    missing += 1
                else:
                    writer.writerow(result)
                    success += 1
                    if success % 500 == 0:
                        csv_file.flush()
    finally:
        csv_file.flush()
        csv_file.close()

    # ---- Summary ----
    logging.info(f"\n{'=' * 50}")
    logging.info("  Sampling Complete!")
    logging.info(f"  Total metadata records:  {total_rows:>10,}")
    logging.info(f"  Candidates sampled:      {len(sampled):>10,}")
    logging.info(f"  Already processed:       {len(already_processed):>10,}")
    logging.info(f"  Successfully saved:      {success:>10,}")
    logging.info(f"  Missing files:           {missing:>10,}")
    logging.info(f"  Output images:           {output_dir}")
    logging.info(f"  Output metadata:         {args.output_csv}")
    logging.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
