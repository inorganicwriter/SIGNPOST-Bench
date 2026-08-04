"""
Baidu Street View Panorama Sampling & Processing Script.

Optimized for MASSIVE datasets (millions of panoramas across all provinces).

Strategy:
  1. Scan metadata CSVs -> filter out invalid records (Unknown province/city)
  2. Stratified sampling by province: 10,000 candidates
  3. Locate images using province/city/county path (NOT global os.walk)
  4. OCR each panorama's 4 crops -> keep best crop with valid text
  5. Stop after collecting 1,000 valid images
  6. Incremental saving + resume support

Usage:
    python sample_baidusv.py \
        --images-root /path/to/BaiduSvs_history/output/images \
        --metadata-root /path/to/BaiduSvs_history/output/metadata/converted \
        --output-dir /path/to/output/sampled_images \
        --output-csv /path/to/output/baidusv_metadata.csv \
        --target-count 1000 \
        --sample-count 10000 \
        --gpu
"""

import argparse
import csv
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from data_collector.text_patterns import (
    is_valid_location,
    is_valid_ocr_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================
#  Panorama cropping: equirectangular -> perspective projection
# ============================================================
#  Panorama cropping: equirectangular → perspective projection
# ============================================================


def equirect_to_perspective(
    pano_np: np.ndarray,
    heading_deg: float,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    out_w: int = 1024,
    out_h: int = 512,
) -> np.ndarray:
    """
    Extract a perspective view from an equirectangular panorama using
    proper spherical-to-planar projection (no fisheye distortion).

    Args:
        pano_np:     Panorama as numpy array (H, W, 3), equirectangular.
        heading_deg: Horizontal viewing angle in degrees (0=front, 90=right, etc.)
        pitch_deg:   Vertical viewing angle in degrees (0=horizon, +up, -down)
        fov_deg:     Field of view in degrees (default 90°)
        out_w:       Output image width  (default 1024)
        out_h:       Output image height (default 512)

    Returns:
        Perspective view as numpy array (out_h, out_w, 3)
    """
    pano_h, pano_w = pano_np.shape[:2]

    # Convert angles to radians
    heading = np.radians(heading_deg)
    pitch = np.radians(pitch_deg)
    half_fov = np.radians(fov_deg / 2.0)

    # Focal length from FOV
    f = out_w / (2.0 * np.tan(half_fov))

    # Create pixel grid for output image
    u = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    v = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    u, v = np.meshgrid(u, v)

    # Direction vectors in camera space (x=right, y=down, z=forward)
    x = u
    y = v
    z = np.full_like(u, f)

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    # Rotation: first pitch (around x-axis), then heading (around y-axis)
    # Pitch rotation (around x-axis)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y2 = cos_p * y - sin_p * z
    z2 = sin_p * y + cos_p * z
    x2 = x

    # Heading rotation (around y-axis)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    x3 = cos_h * x2 + sin_h * z2
    z3 = -sin_h * x2 + cos_h * z2
    y3 = y2

    # Convert to spherical coordinates (longitude, latitude)
    lon = np.arctan2(x3, z3)  # [-pi, pi]
    lat = np.arcsin(np.clip(y3, -1, 1))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    src_x = ((lon / np.pi + 1.0) / 2.0 * pano_w).astype(np.float32)
    src_y = ((lat / (np.pi / 2.0) + 1.0) / 2.0 * pano_h).astype(np.float32)

    # Wrap x coordinate
    src_x = src_x % pano_w

    # Clamp y coordinate
    src_y = np.clip(src_y, 0, pano_h - 1)

    # Bilinear interpolation using cv2.remap if available, else nearest neighbor
    try:
        import cv2

        map_x = src_x.astype(np.float32)
        map_y = src_y.astype(np.float32)
        result = cv2.remap(pano_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    except ImportError:
        # Fallback: nearest neighbor (slightly lower quality)
        ix = np.round(src_x).astype(np.int32) % pano_w
        iy = np.clip(np.round(src_y).astype(np.int32), 0, pano_h - 1)
        result = pano_np[iy, ix]

    return result


def crop_panorama_perspectives(pano_img: Image.Image, angles=(0, 90, 180, 270)):
    """
    Extract perspective views from equirectangular panorama using proper projection.
    Output: 1024×512 perspective images with no fisheye distortion.
    """
    pano_np = np.array(pano_img)
    crops = {}
    for angle in angles:
        persp_np = equirect_to_perspective(
            pano_np, heading_deg=angle, pitch_deg=0.0, fov_deg=90.0, out_w=1024, out_h=512
        )
        crops[angle] = Image.fromarray(persp_np)
    return crops


# ============================================================
#  Metadata scanning (lightweight, with validation)
# ============================================================


def scan_metadata_lightweight(metadata_root: str):
    """
    Scan all CSVs, filter invalid records, return {province -> [records]}.
    Only keeps essential fields to save memory.
    """
    metadata_root = Path(metadata_root)
    csv_files = list(metadata_root.rglob("*.csv"))
    logging.info(f"Found {len(csv_files)} metadata CSV files")

    by_province = defaultdict(list)
    total = 0
    skipped_invalid = 0
    essential_fields = {
        "panoid",
        "province",
        "city",
        "county",
        "wgs_lat",
        "wgs_lon",
        "ret_wgs_lat",
        "ret_wgs_lon",
        "heading",
        "capture_date",
    }

    for csv_path in tqdm(csv_files, desc="Scanning metadata"):
        try:
            with open(csv_path, encoding="utf-8-sig", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    record = {k: row.get(k, "").strip() for k in essential_fields if k in row}
                    record["panoid"] = row.get("panoid", "").strip()
                    record["province"] = row.get("province", "").strip()
                    record["city"] = row.get("city", "").strip()
                    record["county"] = row.get("county", "").strip()

                    if not is_valid_location(record):
                        skipped_invalid += 1
                        continue

                    by_province[record["province"]].append(record)
        except Exception as e:
            logging.warning(f"Error reading {csv_path}: {e}")

    valid_total = total - skipped_invalid
    logging.info(f"Total records: {total}, Valid: {valid_total}, Skipped invalid: {skipped_invalid}")
    logging.info(f"Valid records across {len(by_province)} provinces")
    return by_province, valid_total


# ============================================================
#  Image path resolution (direct path construction, NOT os.walk)
# ============================================================


def find_image_direct(images_root: str, record: dict) -> str:
    """
    Find image by constructing path from province/city/county + recursive search
    within the county directory ONLY (much faster than global os.walk).

    Directory structure: images_root/province/city/county/sub1/sub2/panoid.jpg
    """
    province = record.get("province", "")
    city = record.get("city", "")
    county = record.get("county", "")
    panoid = record.get("panoid", "")

    if not panoid or not province or not city:
        return None

    target = f"{panoid}.jpg"

    # Try direct county path first
    if county:
        county_dir = os.path.join(images_root, province, city, county)
        if os.path.isdir(county_dir):
            # Search within county directory (only 2 levels deep)
            for root, _dirs, files in os.walk(county_dir):
                if target in files:
                    return os.path.join(root, target)

    # Fallback: try city directory
    city_dir = os.path.join(images_root, province, city)
    if os.path.isdir(city_dir):
        for root, _dirs, files in os.walk(city_dir):
            if target in files:
                return os.path.join(root, target)

    return None


# ============================================================
#  Main pipeline
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Baidu Street View Panorama Sampling & Processing")
    parser.add_argument("--images-root", type=str, required=True, help="Root directory of panorama images")
    parser.add_argument("--metadata-root", type=str, required=True, help="Root directory of metadata CSVs")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save cropped perspective images")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV with sampled metadata")
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of FINAL images after OCR filtering (default: 1000)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10000,
        help="Number of candidates to sample BEFORE OCR filtering (default: 10000)",
    )
    parser.add_argument("--min-per-province", type=int, default=10, help="Minimum samples per province (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    parser.add_argument("--ocr-langs", nargs="+", default=["ch_sim", "en"], help="OCR languages (default: ch_sim en)")
    parser.add_argument("--ocr-threshold", type=float, default=0.3, help="OCR confidence threshold (default: 0.3)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs for parallel OCR (default: 4)")
    return parser.parse_args()


# ============================================================
#  Worker function for multi-GPU parallel OCR
# ============================================================


def _ocr_worker(gpu_id, task_queue, result_queue, ocr_langs, ocr_threshold, images_root):
    """
    Worker process: picks tasks from queue, does image loading + crop + OCR on assigned GPU.
    """
    import easyocr

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"  [GPU {gpu_id}] Worker started")

    reader = easyocr.Reader(ocr_langs, gpu=True)

    while True:
        item = task_queue.get()
        if item is None:  # Poison pill
            break

        record = item

        # Find image
        img_path = find_image_direct(images_root, record)
        if img_path is None:
            result_queue.put(("missing", record, None, None, None, None))
            continue

        try:
            pano_img = Image.open(img_path).convert("RGB")
        except Exception:
            result_queue.put(("missing", record, None, None, None, None))
            continue

        # Crop into 4 perspective views
        crops = crop_panorama_perspectives(pano_img)
        del pano_img

        # OCR each crop
        best_crop = None
        best_angle = None
        best_texts = []

        for angle in [0, 90, 180, 270]:
            crop_img = crops[angle]
            crop_np = np.array(crop_img)

            try:
                results = reader.readtext(crop_np, detail=1)
            except Exception:
                continue
            finally:
                del crop_np

            valid_texts = []
            for _bbox, text, conf in results:
                if conf >= ocr_threshold and is_valid_ocr_text(text):
                    valid_texts.append(text)

            if valid_texts and (best_crop is None or len(valid_texts) > len(best_texts)):
                best_crop = crop_img
                best_angle = angle
                best_texts = valid_texts
                if len(valid_texts) >= 3:
                    break

        del crops

        if best_crop is None or not best_texts:
            result_queue.put(("no_text", record, None, None, None, None))
        else:
            result_queue.put(("success", record, best_crop, best_angle, best_texts, img_path))


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume support ----
    already_processed = set()
    if args.resume and os.path.exists(args.output_csv):
        with open(args.output_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_processed.add(row.get("panoid", ""))
        logging.info(f"Resume mode: {len(already_processed)} panoids already processed")

    # ---- Step 1: Scan metadata (with validation) ----
    logging.info("Step 1: Scanning metadata (filtering invalid locations)...")
    by_province, valid_total = scan_metadata_lightweight(args.metadata_root)

    if valid_total == 0:
        logging.error("No valid metadata records found!")
        return

    # ---- Step 2: Stratified sampling → 10,000 candidates ----
    logging.info(f"Step 2: Stratified sampling → {args.sample_count} candidates...")

    sampled = []
    for province, records in sorted(by_province.items()):
        proportion = len(records) / valid_total
        n = max(args.min_per_province, int(args.sample_count * proportion))
        n = min(n, len(records))
        selected = random.sample(records, n)
        sampled.extend(selected)
        logging.info(f"  {province}: {len(records):>8} valid -> {n:>5} sampled")

    random.shuffle(sampled)

    # Filter out already processed
    if already_processed:
        sampled = [r for r in sampled if r["panoid"] not in already_processed]

    logging.info(f"Total candidates for OCR processing: {len(sampled)}")

    # Free metadata memory
    del by_province

    # ---- Step 3 & 4: Multi-GPU parallel OCR processing ----
    import torch

    num_gpus = min(args.num_gpus, torch.cuda.device_count()) if args.gpu else 0

    if num_gpus > 1:
        # ========== Multi-GPU parallel mode ==========
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        logging.info(f"Step 3: Launching {num_gpus} GPU workers for parallel OCR...")

        task_queue = mp.Queue(maxsize=num_gpus * 4)
        result_queue = mp.Queue()

        workers = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_ocr_worker,
                args=(gpu_id, task_queue, result_queue, args.ocr_langs, args.ocr_threshold, args.images_root),
            )
            p.daemon = True
            p.start()
            workers.append(p)

        success_count = len(already_processed)
        missing_count = 0
        no_text_count = 0

        # CSV writing
        csv_exists = os.path.exists(args.output_csv) and args.resume
        fieldnames = [
            "photo_id",
            "panoid",
            "latitude",
            "longitude",
            "province",
            "city",
            "county",
            "heading",
            "capture_date",
            "angle",
            "detected_text",
            "source_path",
        ]
        csv_file = open(args.output_csv, "a" if csv_exists else "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        # Feed tasks and collect results
        sent = 0
        received = 0
        target_reached = False

        try:
            for record in tqdm(sampled, desc="Feeding tasks"):
                if target_reached:
                    break
                task_queue.put(record)
                sent += 1

                # Drain results as they come
                while not result_queue.empty():
                    status, rec, crop, angle, texts, img_path = result_queue.get()
                    received += 1
                    if status == "missing":
                        missing_count += 1
                    elif status == "no_text":
                        no_text_count += 1
                    elif status == "success":
                        panoid = rec.get("panoid", "")
                        out_filename = f"{panoid}_{angle}.jpg"
                        out_path = output_dir / out_filename
                        crop.save(out_path, "JPEG", quality=95)
                        success_count += 1

                        row = {
                            "photo_id": f"{panoid}_{angle}",
                            "panoid": panoid,
                            "latitude": rec.get("wgs_lat", rec.get("ret_wgs_lat", "")),
                            "longitude": rec.get("wgs_lon", rec.get("ret_wgs_lon", "")),
                            "province": rec.get("province", ""),
                            "city": rec.get("city", ""),
                            "county": rec.get("county", ""),
                            "heading": rec.get("heading", ""),
                            "capture_date": rec.get("capture_date", ""),
                            "angle": angle,
                            "detected_text": " | ".join(texts[:5]),
                            "source_path": img_path,
                        }
                        writer.writerow(row)
                        csv_file.flush()

                        if success_count % 50 == 0:
                            logging.info(
                                f"   {success_count}/{args.target_count} | "
                                f"missing: {missing_count} | no_text: {no_text_count}"
                            )

                        if success_count >= args.target_count:
                            target_reached = True
                            break

            # Wait for remaining results
            while received < sent and not target_reached:
                status, rec, crop, angle, texts, img_path = result_queue.get(timeout=120)
                received += 1
                if status == "success":
                    panoid = rec.get("panoid", "")
                    out_filename = f"{panoid}_{angle}.jpg"
                    out_path = output_dir / out_filename
                    crop.save(out_path, "JPEG", quality=95)
                    success_count += 1
                    row = {
                        "photo_id": f"{panoid}_{angle}",
                        "panoid": panoid,
                        "latitude": rec.get("wgs_lat", rec.get("ret_wgs_lat", "")),
                        "longitude": rec.get("wgs_lon", rec.get("ret_wgs_lon", "")),
                        "province": rec.get("province", ""),
                        "city": rec.get("city", ""),
                        "county": rec.get("county", ""),
                        "heading": rec.get("heading", ""),
                        "capture_date": rec.get("capture_date", ""),
                        "angle": angle,
                        "detected_text": " | ".join(texts[:5]),
                        "source_path": img_path,
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    if success_count >= args.target_count:
                        target_reached = True
                elif status == "missing":
                    missing_count += 1
                elif status == "no_text":
                    no_text_count += 1

        finally:
            # Send poison pills
            for _ in workers:
                task_queue.put(None)
            for w in workers:
                w.join(timeout=10)
            csv_file.close()

    else:
        # ========== Single GPU / CPU mode ==========
        logging.info("Step 3: Initializing EasyOCR (single GPU)...")
        import easyocr

        reader = easyocr.Reader(args.ocr_langs, gpu=args.gpu)

        logging.info(f"Step 4: Processing panoramas (target: {args.target_count} valid images)...")

        success_count = len(already_processed)
        missing_count = 0
        no_text_count = 0

        csv_exists = os.path.exists(args.output_csv) and args.resume
        fieldnames = [
            "photo_id",
            "panoid",
            "latitude",
            "longitude",
            "province",
            "city",
            "county",
            "heading",
            "capture_date",
            "angle",
            "detected_text",
            "source_path",
        ]
        csv_file = open(args.output_csv, "a" if csv_exists else "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        try:
            for record in tqdm(sampled, desc=f"OCR filtering ({success_count}/{args.target_count})"):
                if success_count >= args.target_count:
                    logging.info(f" Reached target count: {args.target_count}")
                    break

                panoid = record.get("panoid", "")
                img_path = find_image_direct(args.images_root, record)
                if img_path is None:
                    missing_count += 1
                    continue

                try:
                    pano_img = Image.open(img_path).convert("RGB")
                except Exception:
                    missing_count += 1
                    continue

                crops = crop_panorama_perspectives(pano_img)
                del pano_img

                best_crop = None
                best_angle = None
                best_texts = []

                for angle in [0, 90, 180, 270]:
                    crop_img = crops[angle]
                    crop_np = np.array(crop_img)
                    try:
                        results = reader.readtext(crop_np, detail=1)
                    except Exception:
                        continue
                    finally:
                        del crop_np

                    valid_texts = []
                    for _bbox, text, conf in results:
                        if conf >= args.ocr_threshold and is_valid_ocr_text(text):
                            valid_texts.append(text)

                    if valid_texts and (best_crop is None or len(valid_texts) > len(best_texts)):
                        best_crop = crop_img
                        best_angle = angle
                        best_texts = valid_texts
                        if len(valid_texts) >= 3:
                            break

                del crops

                if best_crop is None or not best_texts:
                    no_text_count += 1
                    continue

                out_filename = f"{panoid}_{best_angle}.jpg"
                out_path = output_dir / out_filename
                best_crop.save(out_path, "JPEG", quality=95)
                del best_crop

                success_count += 1
                row = {
                    "photo_id": f"{panoid}_{best_angle}",
                    "panoid": panoid,
                    "latitude": record.get("wgs_lat", record.get("ret_wgs_lat", "")),
                    "longitude": record.get("wgs_lon", record.get("ret_wgs_lon", "")),
                    "province": record.get("province", ""),
                    "city": record.get("city", ""),
                    "county": record.get("county", ""),
                    "heading": record.get("heading", ""),
                    "capture_date": record.get("capture_date", ""),
                    "angle": best_angle,
                    "detected_text": " | ".join(best_texts[:5]),
                    "source_path": img_path,
                }
                writer.writerow(row)
                csv_file.flush()

                if success_count % 50 == 0:
                    logging.info(
                        f"   {success_count}/{args.target_count} saved | "
                        f"missing: {missing_count} | no_text: {no_text_count}"
                    )
        finally:
            csv_file.close()

    # ---- Summary ----
    logging.info(f"\n{'=' * 60}")
    logging.info("  Baidu Street View Sampling Complete!")
    logging.info(f"{'=' * 60}")
    logging.info(f"  Total valid metadata:    {valid_total:>10}")
    logging.info(f"  Candidates sampled:      {len(sampled):>10}")
    logging.info(f"  Missing image files:     {missing_count:>10}")
    logging.info(f"  No valid text (OCR):     {no_text_count:>10}")
    logging.info(f"  Successfully saved:      {success_count:>10}")
    logging.info(f"  OCR hit rate:            {success_count / (success_count + no_text_count + 0.001) * 100:.1f}%")
    logging.info(f"  Output images:           {output_dir}")
    logging.info(f"  Output metadata:         {args.output_csv}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
