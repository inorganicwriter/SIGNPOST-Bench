"""
utils/data_loader.py — Shared data loading functions.

Used by evaluate.py, evaluate_probing.py, evaluate_generalization.py.
"""

import base64
import json
import logging
import os
from urllib.parse import urlparse

from utils.file_utils import get_base_id

logger = logging.getLogger(__name__)


def load_ground_truth(metadata_path: str) -> dict:
    """Load ground truth GPS from metadata TSV file.

    Returns dict mapping filename -> (lat, lon).
    """
    gt_map = {}
    print(f"Loading ground truth from {metadata_path}...")
    try:
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 15:
                    continue
                try:
                    lon = float(parts[10])
                    lat = float(parts[11])
                    url = parts[14]
                    filename = os.path.basename(urlparse(url).path)
                    gt_map[filename] = (lat, lon)
                    gt_map[parts[1]] = (lat, lon)
                except ValueError:
                    continue
    except Exception as e:
        logger.error("Error reading metadata: %s", e)
    print(f"Loaded {len(gt_map)} ground truth entries.")
    return gt_map


def load_benchmark_meta(meta_path: str) -> dict:
    """Load benchmark metadata JSONL (attack configs, original sources, etc.).

    Returns dict mapping filename -> entry dict.
    """
    if not meta_path or not os.path.exists(meta_path):
        return {}
    meta_map = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                meta_map[entry["filename"]] = entry
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("Skipping malformed benchmark meta line: %s", e)
    return meta_map


def resolve_gt(filename: str, gt_map: dict, bench_meta: dict):
    """Resolve ground truth for a filename using multiple fallback strategies.

    Returns (gt_tuple, meta_info, original_source) or (None, None, None).
    """
    meta_info = bench_meta.get(filename)
    original_source = meta_info.get("original_source") if meta_info else None

    gt = gt_map.get(filename)
    if not gt and original_source:
        gt = gt_map.get(original_source)
    if not gt:
        name_no_ext = os.path.splitext(filename)[0]
        gt = gt_map.get(name_no_ext)
    if not gt:
        base_id = get_base_id(filename)
        gt = gt_map.get(base_id)
        if not gt:
            gt = gt_map.get(os.path.splitext(base_id)[0])
    return gt, meta_info, original_source


def encode_image(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (PermissionError, FileNotFoundError) as e:
        logger.warning("Cannot read image %s: %s", image_path, e)
        return None


def load_results_jsonl(filepath: str) -> list:
    """Load a results JSONL file into a list of dicts."""
    results = []
    if not os.path.exists(filepath):
        return results
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def load_baseline_errors(baseline_path: str) -> dict:
    """Load baseline results and return dict mapping filename -> error_km."""
    clean_map = {}
    if not baseline_path or not os.path.exists(baseline_path):
        return clean_map
    print(f"Loading baseline from {baseline_path}...")
    with open(baseline_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("error_km") is not None:
                    clean_map[entry["filename"]] = entry["error_km"]
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Loaded {len(clean_map)} baseline entries.")
    return clean_map


def scan_images(img_dir: str, valid_exts=(".png", ".jpg", ".jpeg", ".webp")) -> list:
    """Scan a directory for image files."""
    if not os.path.exists(img_dir):
        print(f"[WARNING] Image directory not found: {img_dir}")
        return []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(files)} images in {img_dir}")
    return files
