"""
evaluate.py — Standard evaluation for SIGNPOST-Bench (Phase 1).

Evaluates MLLMs on geo-localization with standard prompt.
Computes WLA / TBS / TFR metrics.

Usage:
    # Using explicit paths (override config):
    python evaluate.py --img-dir data/im2gps3k/images/Blank --metadata-file data/im2gps3k/metadata/im2gps3k_gt.tsv --output results/test.jsonl --model gemma3-27b-free
"""

import argparse
import asyncio
import copy
import json
import logging
import os

from config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    get_dataset_config,
    get_images_dir,
    get_output_path,
)
from evaluation.api_client import GeoLocalizationClient, build_client
from evaluation.metric_calculator import MetricCalculator
from utils.data_loader import (
    load_baseline_errors,
    load_benchmark_meta,
    load_ground_truth,
    resolve_gt,
    scan_images,
)
from utils.file_utils import rewrite_results_file

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNPOST-Bench Standard Evaluation")

    # Dataset-based arguments (preferred)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["im2gps3k", "yfcc4k", "googlesv", "baidusv", "all"],
        help="Dataset name or 'all' for all datasets",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="Blank",
        choices=["Original", "Blank", "Similar", "Random", "Adversarial", "all"],
        help="Image variant or 'all' for all variants",
    )

    # Explicit path arguments (override dataset-based)
    parser.add_argument("--img-dir", type=str, help="Override: directory containing images")
    parser.add_argument("--metadata-file", type=str, help="Override: path to metadata TSV")
    parser.add_argument("--bench-meta", type=str, help="Override: path to benchmark_meta.jsonl")
    parser.add_argument("--baseline", type=str, help="Path to Blank results JSONL (for TBS)")
    parser.add_argument("--output", type=str, help="Override: output JSONL file")

    # Model and API
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model short name")
    parser.add_argument("--api-base", type=str, default=None, help="Override API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--provider", type=str, default=None, help="Override provider")

    # Execution
    parser.add_argument("--limit", type=int, default=0, help="Limit images (0 = all)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent API requests (default: {DEFAULT_CONCURRENCY})",
    )

    return parser.parse_args()


def resolve_paths(args):
    """Resolve all paths from dataset config or explicit arguments."""
    if args.dataset:
        ds = get_dataset_config(args.dataset)
        img_dir = args.img_dir or str(get_images_dir(args.dataset, args.variant))
        metadata_file = args.metadata_file or str(ds["metadata_file"])
        bench_meta = args.bench_meta or str(ds["bench_meta"])
        output = args.output or str(get_output_path(args.dataset, args.variant, args.model))
        # Auto-resolve baseline for non-Blank variants
        if not args.baseline and args.variant != "Blank":
            baseline_path = get_output_path(args.dataset, "Blank", args.model)
            if baseline_path.exists():
                args.baseline = str(baseline_path)
    else:
        if not args.img_dir or not args.metadata_file or not args.output:
            raise ValueError("Either --dataset or (--img-dir, --metadata-file, --output) required")
        img_dir = args.img_dir
        metadata_file = args.metadata_file
        bench_meta = args.bench_meta
        output = args.output

    return img_dir, metadata_file, bench_meta, output


# Retry policy for failed predictions.
# Only empty / null responses are treated as transient failures and retried.
MAX_EMPTY_RESPONSE_RETRIES = 2


def _is_empty_prediction_text(value):
    return value is None or (isinstance(value, str) and not value.strip())


def should_retry_entry(entry):
    """Decide whether an existing result row should be retried on resume."""
    if not entry.get("parse_failed"):
        return False
    return _is_empty_prediction_text(entry.get("prediction_text"))


def load_resume_state(output_path):
    """
    Load existing results and compact retryable transient failures.

    Rows with parse_failed=true and empty/null prediction_text are treated as
    transient API failures. They are removed from the persisted output and
    re-queued on the next run instead of being permanently skipped.
    """
    if not os.path.exists(output_path):
        return {}, [], set(), 0, False

    latest_entries = {}
    entry_order = []
    valid_line_count = 0
    normalized_entries = False

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = entry.get("filename")
            if not filename:
                continue
            valid_line_count += 1
            if (not entry.get("parse_failed")) or (not _is_empty_prediction_text(entry.get("prediction_text"))):
                if "failure_reason" in entry:
                    entry.pop("failure_reason", None)
                    normalized_entries = True
            if filename not in latest_entries:
                entry_order.append(filename)
            latest_entries[filename] = entry

    already_done = set()
    retryable = set()
    for filename, entry in latest_entries.items():
        if should_retry_entry(entry):
            retryable.add(filename)
        else:
            already_done.add(filename)

    preserved_entries = {filename: entry for filename, entry in latest_entries.items() if filename not in retryable}
    needs_compaction = len(latest_entries) != valid_line_count or bool(retryable) or normalized_entries
    return preserved_entries, entry_order, already_done, len(retryable), needs_compaction


def resolve_clean_error(original_source, clean_results_map):
    """Resolve paired Blank/clean error for one sample using exact and base-id matching."""
    orig_src = original_source or ""
    clean_err = clean_results_map.get(orig_src)
    if clean_err is not None:
        return clean_err

    base_id = orig_src.split(".")[0] if orig_src else ""
    if not base_id:
        return None

    for key, value in clean_results_map.items():
        if key.startswith(base_id):
            return value
    return None


def compute_entry_tbs(entry, clean_results_map):
    """Compute TBS for one entry using the current Blank baseline map."""
    attack_type = str(entry.get("attack_type", "")).lower()
    if attack_type in ("blank", "original", "clean", "unknown"):
        return None

    error_km = entry.get("error_km")
    if error_km is None:
        return None

    clean_err = resolve_clean_error(entry.get("original_source", ""), clean_results_map)
    if clean_err is None:
        return entry.get("tbs")

    return MetricCalculator.calculate_tbs(clean_err, error_km)


def refresh_tbs_entries(entries, clean_results_map, only_zero=False):
    """Recompute TBS for selected entries when baseline data is available."""
    updated = 0
    for entry in entries:
        if only_zero and entry.get("tbs") != 0:
            continue
        new_tbs = compute_entry_tbs(entry, clean_results_map)
        old_tbs = entry.get("tbs")
        if old_tbs != new_tbs:
            entry["tbs"] = new_tbs
            updated += 1
    return updated


async def process_single_async(
    client, filename, img_dir, gt, meta_info, original_source, clean_results_map, clean_results_lock
):
    """Process a single image: predict location, compute metrics.

    Fully async — uses asyncio.to_thread for the synchronous API call,
    and asyncio.sleep for retry waits (non-blocking).

    Only empty / null API responses are retried here. Non-empty responses are
    recorded as-is, even if they do not contain valid coordinates.
    """
    image_path = os.path.join(img_dir, filename)

    pred_text = None
    pred_lat, pred_lon = None, None
    empty_attempts = 0

    while empty_attempts < MAX_EMPTY_RESPONSE_RETRIES:
        # Use asyncio.to_thread to avoid blocking the event loop
        response = await asyncio.to_thread(client.predict_location_from_path, image_path)
        if response and response.strip():
            lat, lon = GeoLocalizationClient.parse_coordinates(response)
            pred_text = response
            if lat is not None:
                pred_text, pred_lat, pred_lon = response, lat, lon
            break

        # Empty/null response: treat as transient transport/provider failure.
        pred_text = None
        empty_attempts += 1
        if empty_attempts < MAX_EMPTY_RESPONSE_RETRIES:
            print(
                f"  [retry] {filename}: empty API response, retrying "
                f"({empty_attempts + 1}/{MAX_EMPTY_RESPONSE_RETRIES})..."
            )
            await asyncio.sleep(min(1.5 * empty_attempts, 5.0))

    error_km = None
    if pred_lat is not None:
        error_km = MetricCalculator.haversine_distance(gt[0], gt[1], pred_lat, pred_lon)
    wla_score = MetricCalculator.calculate_wla(error_km)
    failure_reason = "empty_response" if error_km is None and _is_empty_prediction_text(pred_text) else None

    attack_type = meta_info.get("attack_type", "unknown") if meta_info else "original"
    if attack_type == "original" and error_km is not None:
        async with clean_results_lock:
            clean_results_map[filename] = error_km

    result = {
        "filename": filename,
        "original_source": original_source,
        "attack_type": attack_type,
        "injected_text": (meta_info.get("injected_text") or ", ".join(meta_info.get("injected_texts", [])))
        if meta_info
        else None,
        "prediction_text": pred_text,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "gt_lat": gt[0],
        "gt_lon": gt[1],
        "error_km": error_km,
        "wla_score": wla_score,
        "parse_failed": error_km is None,
    }
    if failure_reason is not None:
        result["failure_reason"] = failure_reason
    return result


async def evaluate_async(
    args, client, image_files, gt_map, bench_meta, clean_results_map, already_done, output_path, img_dir
):
    """Run evaluation with true concurrent API calls.

    Rate limiting is handled by the API client's built-in 429 retry logic,
    so we use a pure semaphore for concurrency control without artificial delays.
    """
    sem = asyncio.Semaphore(args.concurrency)
    _counter_lock = asyncio.Lock()
    _processed_count = [0]
    file_lock = asyncio.Lock()
    clean_lock = asyncio.Lock()

    async def eval_one(filename):
        async with sem:
            gt, meta_info, original_source = resolve_gt(filename, gt_map, bench_meta)
            if not gt:
                return None

            res = await process_single_async(
                client, filename, img_dir, gt, meta_info, original_source, clean_results_map, clean_lock
            )
            if res is None:
                return None

            async with _counter_lock:
                _processed_count[0] += 1
                count = _processed_count[0]
            async with file_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")

            if res.get("error_km") is not None:
                print(f"  [{count}] {filename} -> {res['error_km']:.1f} km | WLA: {res.get('wla_score', 0):.2f}")
            else:
                print(f"  [{count}] {filename} -> Failed to parse: {(res.get('prediction_text') or '')[:80]}")
            return res

    # Filter pending files
    pending = []
    for filename in image_files:
        if args.limit > 0 and len(pending) >= args.limit:
            break
        if filename in already_done:
            continue
        pending.append(filename)

    print(f"Evaluating {len(pending)} images (concurrency={args.concurrency})...")
    tasks = [eval_one(f) for f in pending]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def print_summary(output_path, clean_results_map):
    """Print evaluation summary from output file."""
    total_wla, valid_count, tbs_sum, tbs_count = 0, 0, 0, 0

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("error_km") is not None:
                total_wla += entry.get("wla_score", 0)
                valid_count += 1

            if entry.get("tbs") is not None:
                tbs_sum += entry["tbs"]
                tbs_count += 1

    print(f"\n{'=' * 40}")
    print("  SIGNPOST-Bench Evaluation Report")
    print(f"{'=' * 40}")
    print(f"  Valid predictions: {valid_count}")
    if valid_count > 0:
        print(f"  Mean WLA: {total_wla / valid_count * 100:.2f}%")
    if tbs_count > 0:
        print(f"  Mean TBS: {tbs_sum / tbs_count:.1f} km (n={tbs_count})")
    else:
        print("  Mean TBS: N/A (no baseline paired)")
    print(f"  Results: {output_path}")


def run_single(args):
    """Run evaluation for a single dataset + variant combination."""
    # Resolve paths
    img_dir, metadata_file, bench_meta_path, output_path = resolve_paths(args)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load data
    gt_map = load_ground_truth(metadata_file)
    bench_meta = load_benchmark_meta(bench_meta_path)
    clean_results_map = load_baseline_errors(args.baseline) if args.baseline else {}

    if not gt_map:
        print(f"[ERROR] Ground truth empty. Check: {metadata_file}")
        return

    # Build client
    try:
        client = build_client(
            model_short_name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            api_base=args.api_base,
        )
        print(f"[Client] {client.model_name} via {client.provider}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # Scan images
    image_files = scan_images(img_dir)

    # Resume support
    existing_entries, entry_order, already_done, retryable_count, needs_compaction = load_resume_state(output_path)
    if already_done:
        print(f"Resuming: {len(already_done)} already done, skipping.")
    if retryable_count:
        print(f"Re-queueing {retryable_count} transient failures with empty responses.")

    # Run
    results = asyncio.run(
        evaluate_async(
            args,
            client,
            image_files,
            gt_map,
            bench_meta,
            clean_results_map,
            already_done,
            output_path,
            img_dir,
        )
    )

    refreshed_existing_tbs = refresh_tbs_entries(existing_entries.values(), clean_results_map, only_zero=True)
    refreshed_new_tbs = refresh_tbs_entries(results, clean_results_map) if results else 0

    if results or needs_compaction or refreshed_existing_tbs or refreshed_new_tbs:
        rewrite_results_file(output_path, existing_entries, entry_order, results)

    print_summary(output_path, clean_results_map)


def main():
    args = parse_args()

    ALL_DATASETS = ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]
    ALL_VARIANTS = ["Blank", "Original", "Adversarial", "Random", "Similar"]

    if args.dataset is None:
        run_single(args)
        return

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    variants = ALL_VARIANTS if args.variant == "all" else [args.variant]

    # Strategy: if "all" variants, run Blank first (needed as baseline for TBS)
    if args.variant == "all":
        variants = ["Blank"] + [v for v in ALL_VARIANTS if v != "Blank"]

    total_runs = len(datasets) * len(variants)
    run_idx = 0

    for dataset in datasets:
        for variant in variants:
            run_idx += 1
            print(f"\n{'#' * 60}")
            print(f"  [{run_idx}/{total_runs}] {dataset} / {variant} / {args.model}")
            print(f"{'#' * 60}\n")

            # Create a copy of args with specific dataset/variant
            run_args = copy.copy(args)
            run_args.dataset = dataset
            run_args.variant = variant
            # Reset path overrides so resolve_paths uses config
            run_args.img_dir = None
            run_args.metadata_file = None
            run_args.bench_meta = None
            run_args.output = None
            run_args.baseline = None

            try:
                run_single(run_args)
            except Exception as e:
                print(f"  [ERROR] {dataset}/{variant}: {e}")
                continue

    print(f"\n{'=' * 60}")
    print(f"  All done! {total_runs} runs completed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
