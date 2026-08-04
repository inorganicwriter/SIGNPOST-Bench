"""
evaluate_probing.py — Conflict Probing & Defense evaluation (Phase 2 & 3).

Probing mode: structured JSON output to diagnose HOW models handle conflict.
Defense mode: conflict-aware prompt with lightweight structured output.

Metrics: CDA / MPR / RCS / CAA / CSG + WLA / TBS

Usage:
    # Probing (Phase 2):
    python evaluate_probing.py --dataset im2gps3k --variant Adversarial --model gpt-5.4 --mode probing

    # Defense (Phase 3):
    python evaluate_probing.py --dataset im2gps3k --variant Adversarial --model gpt-5.4 --mode defense
"""

import argparse
import asyncio
import json
import logging
import os
import random

from config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    get_dataset_config,
    get_experiment_output_path,
    get_images_dir,
    get_named_output_path,
)
from evaluation.api_client import GeoLocalizationClient, build_client
from evaluation.metric_calculator import MetricCalculator
from evaluation.prompts import DEFENSE_PROMPT, PROBING_PROMPT
from utils.data_loader import (
    load_baseline_errors,
    load_benchmark_meta,
    load_ground_truth,
    resolve_gt,
    scan_images,
)
from utils.file_utils import build_output_order, rewrite_results_file
from utils.parsers import (
    normalize_consistent,
    normalize_trusted_source,
    parse_defense_response,
    parse_json_response,
)

logger = logging.getLogger(__name__)

# Retry policy for malformed probing JSON.
MAX_PROBING_JSON_RETRIES = 2
PROBING_JSON_FAILURE_REASON = "probing_json_parse_failure"

# ===========================================================================
#  Prompts — imported from evaluation.prompts (single source of truth)
# ===========================================================================
# PROBING_PROMPT and DEFENSE_PROMPT are imported at the top of this file
# from evaluation.prompts. See that module for the canonical text.


# ===========================================================================
#  Args
# ===========================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNPOST-Bench Conflict Probing / Defense")

    parser.add_argument("--dataset", type=str, choices=["im2gps3k", "yfcc4k", "googlesv", "baidusv"])
    parser.add_argument(
        "--variant", type=str, default="Adversarial", choices=["Original", "Blank", "Similar", "Random", "Adversarial"]
    )

    parser.add_argument("--img-dir", type=str, help="Override image directory")
    parser.add_argument("--metadata-file", type=str, help="Override metadata TSV")
    parser.add_argument("--bench-meta", type=str, help="Override benchmark_meta.jsonl")
    parser.add_argument("--baseline", type=str, help="Blank results JSONL for TBS")
    parser.add_argument("--output", type=str, help="Override output JSONL")
    parser.add_argument("--subset", type=str, help="Subset JSONL (from sample_probing_subset.py)")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)

    parser.add_argument("--mode", type=str, choices=["probing", "defense"], default="probing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(args):
    """Resolve paths from dataset config or explicit args."""
    mode_prefix = "probing" if args.mode == "probing" else "defense"
    if args.dataset:
        ds = get_dataset_config(args.dataset)
        img_dir = args.img_dir or str(get_images_dir(args.dataset, args.variant))
        metadata_file = args.metadata_file or str(ds["metadata_file"])
        bench_meta = args.bench_meta or str(ds["bench_meta"])
        output = args.output or str(get_named_output_path(args.dataset, mode_prefix, args.variant, args.model))
    else:
        if not args.img_dir or not args.metadata_file:
            raise ValueError("Either --dataset or (--img-dir, --metadata-file) required")
        img_dir = args.img_dir
        metadata_file = args.metadata_file
        bench_meta = args.bench_meta
        output = args.output or str(
            get_experiment_output_path(
                mode_prefix,
                f"{mode_prefix}_output.jsonl",
                model_name=args.model,
                ensure_parent=True,
            )
        )
    return img_dir, metadata_file, bench_meta, output


def should_retry_entry(entry):
    """Decide whether an existing probing row should be retried on resume.

    Policy:
      - If a row already has an explicit failure_reason, treat it as final and
        do NOT retry it on future runs.
      - Only legacy probing failures that predate failure_reason support are
        re-queued automatically.
    """
    if entry.get("mode") != "probing":
        return False
    if entry.get("failure_reason"):
        return False

    # Backward compatibility: older malformed probing rows did not carry
    # failure_reason. Detect them from the characteristic "all structured
    # fields missing" pattern plus an unparseable raw_response payload.
    if entry.get("pred_lat") is not None or entry.get("pred_lon") is not None:
        return False
    structured_fields = ("consistent", "trusted_source", "cda_hit", "mpr_choice", "rcs_consistent")
    if any(entry.get(field) is not None for field in structured_fields):
        return False
    raw_response = entry.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return False
    return parse_json_response(raw_response) is None


def load_resume_state(output_path):
    """Load existing results and compact retryable legacy probing JSON failures."""
    if not os.path.exists(output_path):
        return {}, [], set(), 0, False

    latest_entries = {}
    entry_order = []
    valid_line_count = 0

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
    needs_compaction = len(latest_entries) != valid_line_count or bool(retryable)
    return preserved_entries, entry_order, already_done, len(retryable), needs_compaction


def load_subset_filenames(subset_path):
    """Load subset filenames while preserving file order and removing duplicates."""
    ordered = []
    seen = set()
    with open(subset_path, encoding="utf-8") as f:
        for line in f:
            try:
                filename = json.loads(line).get("filename", "")
            except json.JSONDecodeError:
                continue
            if filename and filename not in seen:
                ordered.append(filename)
                seen.add(filename)
    return ordered


def is_success_entry(entry):
    """A probing/defense sample counts as complete only if coordinates were parsed."""
    return entry.get("error_km") is not None


def build_candidate_schedule(primary_files, available_images, existing_entries, seed):
    """Build a deterministic candidate schedule with supplementation candidates appended.

    primary_files define the target subset. Supplemental candidates are drawn
    from the remaining images of the same dataset/variant and are only used if
    failures prevent us from reaching the target number of successful rows.
    """
    existing_filenames = set(existing_entries)
    success_filenames = {fn for fn, entry in existing_entries.items() if is_success_entry(entry)}
    blocked_filenames = existing_filenames - success_filenames

    schedule = []
    seen = set()
    for filename in primary_files:
        if filename in success_filenames or filename in blocked_filenames or filename in seen:
            continue
        schedule.append(filename)
        seen.add(filename)

    supplemental = [
        filename
        for filename in sorted(available_images)
        if filename not in set(primary_files) and filename not in existing_filenames and filename not in seen
    ]
    rng = random.Random(seed)
    rng.shuffle(supplemental)
    schedule.extend(supplemental)
    return schedule, success_filenames


# ===========================================================================
#  Probing Metrics
# ===========================================================================


def extract_probing_metrics(probing_data, pred_lat, pred_lon, gt_lat, gt_lon, trap_lat=None, trap_lon=None):
    """Extract mechanism metrics from parsed probing JSON."""
    if probing_data is None:
        return {"consistent": None, "trusted_source": None, "cda_hit": None, "mpr_choice": None, "rcs_consistent": None}

    consistent = normalize_consistent(probing_data.get("consistent"))
    trusted_source = normalize_trusted_source(probing_data.get("trusted_source", ""))

    cda_hit = consistent is False

    rcs_consistent = None
    if pred_lat is not None and gt_lat is not None and trap_lat is not None and trap_lon is not None:
        dist_to_gt = MetricCalculator.haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
        dist_to_trap = MetricCalculator.haversine_distance(pred_lat, pred_lon, trap_lat, trap_lon)

        if trusted_source == "Visual" and dist_to_gt is not None and dist_to_trap is not None:
            rcs_consistent = dist_to_gt < dist_to_trap
        elif trusted_source == "Textual" and dist_to_gt is not None and dist_to_trap is not None:
            rcs_consistent = dist_to_trap < dist_to_gt

    return {
        "consistent": consistent,
        "trusted_source": trusted_source,
        "visual_evidence": probing_data.get("visual_evidence", []),
        "textual_evidence": probing_data.get("textual_evidence", []),
        "visual_region_estimate": probing_data.get("visual_region_estimate", ""),
        "textual_region_estimate": probing_data.get("textual_region_estimate", ""),
        "consistency_explanation": probing_data.get("consistency_explanation", ""),
        "trust_explanation": probing_data.get("trust_explanation", ""),
        "cda_hit": cda_hit,
        "mpr_choice": trusted_source,
        "rcs_consistent": rcs_consistent,
    }


# ===========================================================================
#  Client Wrapper
# ===========================================================================


class ProbingClient:
    """Wraps GeoLocalizationClient to use probing or defense prompts.

    Supports all providers (Gemini native API, OpenAI-compatible, etc.)
    via the unified predict_custom_from_path / predict_custom_from_base64 methods.
    """

    def __init__(self, base_client, mode="probing"):
        self.client = base_client
        self.mode = mode
        self.prompt = PROBING_PROMPT if mode == "probing" else DEFENSE_PROMPT
        self.max_tokens = 4096 if mode == "probing" else base_client.max_tokens

    def predict_from_path(self, image_path: str) -> str:
        """Predict using file path (preferred for Gemini — avoids base64 overhead)."""
        return self.client.predict_custom_from_path(
            image_path,
            self.prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
            json_mode=(self.mode == "probing"),
        )

    def predict(self, base64_image: str) -> str:
        """Predict using base64 image (fallback for legacy callers)."""
        return self.client.predict_custom_from_base64(
            base64_image,
            self.prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )


# ===========================================================================
#  Processing
# ===========================================================================


def process_single(probing_client, filename, img_dir, gt, meta_info, original_source, clean_results_map, mode):
    """Process a single image with probing or defense prompt."""
    image_path = os.path.join(img_dir, filename)

    attack_type = meta_info.get("attack_type", "unknown") if meta_info else "original"
    trap_lat = meta_info.get("trap_lat") if meta_info else None
    trap_lon = meta_info.get("trap_lon") if meta_info else None
    if trap_lat is None and meta_info:
        trap_lat = meta_info.get("target_lat")
    if trap_lon is None and meta_info:
        trap_lon = meta_info.get("target_lon")
    try:
        trap_lat = float(trap_lat) if trap_lat is not None else None
        trap_lon = float(trap_lon) if trap_lon is not None else None
    except (TypeError, ValueError):
        trap_lat, trap_lon = None, None

    raw_response = None
    failure_reason = None

    if mode == "probing":
        probing_data = None
        for attempt in range(MAX_PROBING_JSON_RETRIES):
            raw_response = probing_client.predict_from_path(image_path)
            if not raw_response:
                return None
            probing_data = parse_json_response(raw_response)
            if probing_data is not None:
                break
            if attempt + 1 < MAX_PROBING_JSON_RETRIES:
                print(
                    f"  [retry] {filename}: malformed probing JSON, retrying "
                    f"({attempt + 2}/{MAX_PROBING_JSON_RETRIES})..."
                )
        if probing_data is None:
            failure_reason = PROBING_JSON_FAILURE_REASON
        pred_lat, pred_lon = None, None
        if probing_data and probing_data.get("final_prediction"):
            pred_lat, pred_lon = GeoLocalizationClient.parse_coordinates(str(probing_data["final_prediction"]))
        if pred_lat is None:
            pred_lat, pred_lon = GeoLocalizationClient.parse_coordinates(raw_response)
        probing_metrics = extract_probing_metrics(
            probing_data,
            pred_lat,
            pred_lon,
            gt[0],
            gt[1],
            trap_lat=trap_lat,
            trap_lon=trap_lon,
        )
    else:
        raw_response = probing_client.predict_from_path(image_path)
        if not raw_response:
            return None
        defense_fields = parse_defense_response(raw_response)
        pred_lat, pred_lon = GeoLocalizationClient.parse_coordinates(defense_fields.get("final", ""))
        if pred_lat is None:
            pred_lat, pred_lon = GeoLocalizationClient.parse_coordinates(raw_response)

        conflict_str = defense_fields.get("conflict", "").strip().lower()
        consistent = not conflict_str.startswith("yes")
        trusted_source = normalize_trusted_source(defense_fields.get("trusted", ""))

        probing_metrics = {
            "consistent": consistent,
            "trusted_source": trusted_source,
            "cda_hit": not consistent,
            "mpr_choice": trusted_source,
            "rcs_consistent": None,
            "defense_visual": defense_fields.get("visual", ""),
            "defense_text": defense_fields.get("text", ""),
        }

    error_km = (
        MetricCalculator.haversine_distance(gt[0], gt[1], pred_lat, pred_lon)
        if pred_lat is not None and pred_lon is not None
        else None
    )
    wla_score = MetricCalculator.calculate_wla(error_km)

    tbs = None
    if original_source and clean_results_map:
        clean_err = clean_results_map.get(original_source)
        if clean_err is None:
            base_id = original_source.split(".")[0]
            for key in clean_results_map:
                if key.startswith(base_id):
                    clean_err = clean_results_map[key]
                    break
        if clean_err is not None and error_km is not None:
            tbs = MetricCalculator.calculate_tbs(clean_err, error_km)

    res = {
        "filename": filename,
        "original_source": original_source,
        "attack_type": attack_type,
        "mode": mode,
        "raw_response": raw_response[:2000],
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "gt_lat": gt[0],
        "gt_lon": gt[1],
        "error_km": error_km,
        "wla_score": wla_score,
        "tbs": tbs,
        "trap_lat": trap_lat,
        "trap_lon": trap_lon,
    }
    if failure_reason is not None:
        res["failure_reason"] = failure_reason
    res.update(probing_metrics)
    return res


# ===========================================================================
#  Async Runner
# ===========================================================================


async def evaluate_async(
    args,
    probing_client,
    image_files,
    gt_map,
    bench_meta,
    clean_results_map,
    already_done,
    output_path,
    img_dir,
):
    sem = asyncio.Semaphore(args.concurrency)
    _counter_lock = asyncio.Lock()
    _processed_count = [0]
    file_lock = asyncio.Lock()

    async def eval_one(filename):
        async with sem:
            gt, meta_info, original_source = resolve_gt(filename, gt_map, bench_meta)
            if not gt:
                return None

            res = await asyncio.to_thread(
                process_single,
                probing_client,
                filename,
                img_dir,
                gt,
                meta_info,
                original_source,
                clean_results_map,
                args.mode,
            )
            if res is None:
                return None

            async with _counter_lock:
                _processed_count[0] += 1
                count = _processed_count[0]
            async with file_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

            status = f"{res['error_km']:.1f}km" if res["error_km"] is not None else "FAIL"
            extra = f" | trust={res.get('trusted_source', '?')}" if res.get("trusted_source") else ""
            print(f"  [{count}] {filename} -> {status}{extra}")
            return res

    pending = [f for f in image_files if f not in already_done]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Evaluating {len(pending)} images (mode={args.mode}, concurrency={args.concurrency})...")
    results = await asyncio.gather(*[eval_one(f) for f in pending])
    return [r for r in results if r is not None]


def compute_summary(results, mode):
    """Print summary statistics."""
    valid = [r for r in results if r.get("error_km") is not None]
    structured_failures = [r for r in results if r.get("failure_reason") == PROBING_JSON_FAILURE_REASON]
    print(f"\n{'=' * 50}")
    print(f"  {mode.upper()} Evaluation Report")
    print(f"{'=' * 50}")
    print(f"Total: {len(results)} | Valid: {len(valid)}")
    if structured_failures:
        print(f"Structured parse failures: {len(structured_failures)} (excluded from CDA/MPR/RCS)")

    if valid:
        mean_wla = sum(r["wla_score"] for r in valid) / len(valid) * 100
        errors = sorted([r["error_km"] for r in valid])
        print(f"Mean WLA: {mean_wla:.2f}% | Median Error: {errors[len(errors) // 2]:.1f}km")

    tbs_vals = [r["tbs"] for r in results if r.get("tbs") is not None]
    if tbs_vals:
        print(f"Mean TBS: {sum(tbs_vals) / len(tbs_vals):.1f}km (n={len(tbs_vals)})")

    # CDA
    conflict_samples = [
        r for r in results if r.get("failure_reason") != PROBING_JSON_FAILURE_REASON and r.get("consistent") is not None
    ]
    if conflict_samples:
        cda = sum(1 for r in conflict_samples if r.get("cda_hit")) / len(conflict_samples) * 100
        print(f"CDA: {cda:.1f}% (n={len(conflict_samples)})")

    # MPR
    mpr = {"Visual": 0, "Textual": 0, "Both": 0, "Unknown": 0}
    mpr_results = [r for r in results if r.get("failure_reason") != PROBING_JSON_FAILURE_REASON]
    for r in mpr_results:
        choice = r.get("mpr_choice", "Unknown")
        mpr[choice] = mpr.get(choice, 0) + 1
    total_mpr = sum(mpr.values())
    if total_mpr:
        print("MPR: " + " | ".join(f"{k}:{v / total_mpr * 100:.0f}%" for k, v in mpr.items() if v))

    # RCS
    rcs_samples = [
        r
        for r in results
        if r.get("failure_reason") != PROBING_JSON_FAILURE_REASON and r.get("rcs_consistent") is not None
    ]
    if rcs_samples:
        rcs = sum(1 for r in rcs_samples if r["rcs_consistent"]) / len(rcs_samples) * 100
        print(f"RCS: {rcs:.1f}% (n={len(rcs_samples)})")


def main():
    args = parse_args()
    img_dir, metadata_file, bench_meta_path, output_path = resolve_paths(args)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    gt_map = load_ground_truth(metadata_file)
    bench_meta = load_benchmark_meta(bench_meta_path)
    clean_results_map = load_baseline_errors(args.baseline) if args.baseline else {}

    if not gt_map:
        print(f"[ERROR] Ground truth empty. Check: {metadata_file}")
        return

    try:
        base_client = build_client(
            model_short_name=args.model, provider=args.provider, api_key=args.api_key, api_base=args.api_base
        )
        print(f"[Client] {base_client.model_name} via {base_client.provider} | mode={args.mode}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    probing_client = ProbingClient(base_client, mode=args.mode)

    # Image files (support subset)
    available_images = set(scan_images(img_dir))
    if args.subset:
        subset_files = load_subset_filenames(args.subset)
        image_files = [f for f in subset_files if f in available_images]
        print(f"Using subset: {len(image_files)} images")
    else:
        image_files = sorted(available_images)
        if args.limit > 0:
            image_files = image_files[: args.limit]

    target_success_count = len(image_files)
    allow_supplement = bool(args.subset or args.limit > 0)

    existing_entries, entry_order, already_done, retryable_count, needs_compaction = load_resume_state(output_path)
    output_order = build_output_order(image_files, entry_order)
    if needs_compaction or entry_order != output_order:
        rewrite_results_file(output_path, existing_entries, output_order, [])
    success_done = {fn for fn, entry in existing_entries.items() if is_success_entry(entry)}
    if success_done:
        print(f"Resuming: {len(success_done)} successful rows already complete.")
    if retryable_count:
        print(f"Re-queueing {retryable_count} legacy malformed probing JSON failures.")

    final_results = []
    if allow_supplement:
        candidate_schedule, success_done = build_candidate_schedule(
            image_files, available_images, existing_entries, args.seed
        )
        cursor = 0
        while len(success_done) < target_success_count and cursor < len(candidate_schedule):
            deficit = target_success_count - len(success_done)
            batch_size = max(deficit, args.concurrency)
            batch = candidate_schedule[cursor : cursor + batch_size]
            cursor += len(batch)
            for filename in batch:
                if filename not in output_order:
                    output_order.append(filename)
            results = asyncio.run(
                evaluate_async(
                    args,
                    probing_client,
                    batch,
                    gt_map,
                    bench_meta,
                    clean_results_map,
                    set(),
                    output_path,
                    img_dir,
                )
            )
            rewrite_results_file(output_path, existing_entries, output_order, results)
            for entry in results:
                filename = entry.get("filename")
                if not filename:
                    continue
                existing_entries[filename] = entry
                if is_success_entry(entry):
                    success_done.add(filename)
            final_results.extend(results)
        if len(success_done) < target_success_count:
            print(
                f"[WARN] Filled {len(success_done)}/{target_success_count} successful rows; candidate pool exhausted."
            )
        else:
            print(f"Filled target with {len(success_done)}/{target_success_count} successful rows.")
    else:
        results = asyncio.run(
            evaluate_async(
                args,
                probing_client,
                image_files,
                gt_map,
                bench_meta,
                clean_results_map,
                success_done,
                output_path,
                img_dir,
            )
        )
        rewrite_results_file(output_path, existing_entries, output_order, results)
        final_results = results

    # Load all for summary (including resumed)
    from utils.data_loader import load_results_jsonl

    all_results = load_results_jsonl(output_path)
    compute_summary(all_results, args.mode)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
