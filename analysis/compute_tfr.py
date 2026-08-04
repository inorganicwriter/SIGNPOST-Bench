"""
Compute Trap-Fit Rate (TFR) and paired Trap Distance Reduction (TDR).

Expects dataset-scoped result files under model subdirectories:
  data/<dataset>/results/<model>/results_Adversarial_<model>.jsonl
  data/<dataset>/results/<model>/results_Blank_<model>.jsonl
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.compute_results import (
    discover_result_files,
    normalize_dataset_display_name,
)
from config import GEOCODE_CACHE_FILE
from evaluation.metric_calculator import MetricCalculator

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
TRAP_RADIUS_KM = 50
USER_AGENT = "SIGNPOST-Bench/1.0 (research benchmark)"
RATE_LIMIT_SECONDS = 1.1


def haversine_distance(lat1, lon1, lat2, lon2):
    return MetricCalculator.haversine_distance(lat1, lon1, lat2, lon2)


def load_geocode_cache():
    if GEOCODE_CACHE_FILE.exists():
        with GEOCODE_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache):
    with GEOCODE_CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_text(text, cache):
    text_key = text.strip().lower()
    if text_key in cache:
        cached = cache[text_key]
        if cached is None:
            return None, None
        return cached["lat"], cached["lon"]

    clean_text = re.sub(r"[\"'\(\)]", "", text.strip())
    if len(clean_text) < 2:
        cache[text_key] = None
        return None, None

    try:
        url = f"{NOMINATIM_URL}?q={quote(clean_text)}&format=json&limit=1&addressdetails=0"
        req = Request(url, headers={"User-Agent": USER_AGENT})

        time.sleep(RATE_LIMIT_SECONDS)
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            cache[text_key] = {"lat": lat, "lon": lon, "display": data[0].get("display_name", "")}
            return lat, lon

        cache[text_key] = None
        return None, None
    except Exception as exc:
        print(f"    Geocoding error for '{clean_text}': {exc}")
        # Do not freeze transient network/API failures as permanent misses.
        return None, None


def resolve_pred_entry_id(entry):
    """Resolve an entry ID from TFR results, preferring original_source (preserving its full format)."""
    src = entry.get("original_source")
    if src:
        return str(src).strip()
    fn = entry.get("filename", "")
    base = os.path.basename(fn or "")
    base = os.path.splitext(base)[0]
    for suffix in ("_Blank", "_Original", "_similar_", "_random_", "_adversarial_"):
        idx = base.rfind(suffix)
        if idx != -1:
            return base[:idx]
    return base


def discover_models(dataset_name, dataset_dir, base_dir):
    results_dir = Path(dataset_dir) / "results"
    discovered = discover_result_files(results_dir, dataset_name=dataset_name)
    return sorted({model for attack, model in discovered if attack == "Adversarial"})


def find_results_file(dataset_name, dataset_dir, base_dir, model_short, variant):
    results_dir = Path(dataset_dir) / "results"
    discovered = discover_result_files(results_dir, dataset_name=dataset_name)
    return discovered.get((variant, model_short))


def load_prediction_map(results_file, target_ids):
    """Load the latest valid coordinate prediction for each target base ID."""
    predictions = {}
    if not results_file:
        return predictions
    with Path(results_file).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            base_id = resolve_pred_entry_id(entry)
            pred_lat = entry.get("pred_lat")
            pred_lon = entry.get("pred_lon")
            if base_id not in target_ids or pred_lat is None or pred_lon is None:
                continue
            predictions[base_id] = (float(pred_lat), float(pred_lon))
    return predictions


def summarize_trap_distance_reductions(reductions):
    """Summarize paired Blank-to-Adversarial changes in trap distance."""
    values = [float(value) for value in reductions]
    if not values:
        return {
            "paired_count": 0,
            "mean_km": None,
            "median_km": None,
            "attraction_rate_percent": None,
        }
    return {
        "paired_count": len(values),
        "mean_km": statistics.fmean(values),
        "median_km": statistics.median(values),
        "attraction_rate_percent": sum(value > 0 for value in values) / len(values) * 100,
    }


def equal_dataset_macro(items, key):
    """Average a dataset-level metric equally, never weighting by sample count."""
    values = [item.get(key) for item in items]
    if not values or any(value is None for value in values):
        return None
    return statistics.fmean(float(value) for value in values)


def prepare_geocode_cache(texts, cache, geocode_missing=False):
    """Apply the explicit frozen/refresh cache policy and return missing queries."""
    uncached = sorted(text for text in texts if text not in cache)
    if not geocode_missing:
        return uncached
    for index, text in enumerate(uncached):
        if index and index % 10 == 0:
            print(f"    Geocoded {index}/{len(uncached)}...")
            save_geocode_cache(cache)
        geocode_text(text, cache)
    save_geocode_cache(cache)
    return uncached


def compute_tfr(
    dataset_name,
    dataset_dir,
    model_short,
    base_dir,
    tier_filter=None,
    geocode_missing=False,
):
    taxonomy_file = Path(dataset_dir) / "taxonomy_labels.jsonl"
    if not taxonomy_file.exists():
        print(f"  Error: {taxonomy_file} not found. Run classify_taxonomy.py first.")
        return None

    taxonomy = {}
    with taxonomy_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            taxonomy[entry["base_id"]] = entry

    if tier_filter:
        target_ids = {base_id for base_id, info in taxonomy.items() if info.get("tier") == tier_filter}
        print(f"  Filtering to {tier_filter}: {len(target_ids)} entries")
    else:
        target_ids = set(taxonomy.keys())
        print(f"  Using all tiers: {len(target_ids)} entries")

    cache = load_geocode_cache()

    texts_to_geocode = set()
    for base_id in target_ids:
        adv_text = taxonomy[base_id].get("adversarial_text", "")
        if adv_text:
            texts_to_geocode.add(adv_text.strip().lower())

    uncached = sorted(text for text in texts_to_geocode if text not in cache)
    print(f"  Adversarial texts to geocode: {len(texts_to_geocode)} total, {len(uncached)} uncached")

    if uncached and not geocode_missing:
        print(
            f"  Frozen-cache mode: excluding {len(uncached)} queries that were not "
            "part of the analysis-time cache; no network requests or cache writes."
        )
    prepare_geocode_cache(texts_to_geocode, cache, geocode_missing=geocode_missing)

    adversarial_file = find_results_file(dataset_name, dataset_dir, Path(base_dir), model_short, "Adversarial")
    blank_file = find_results_file(dataset_name, dataset_dir, Path(base_dir), model_short, "Blank")
    if not adversarial_file:
        print(f"  Error: Adversarial results for {model_short} not found.")
        return None
    if not blank_file:
        print(f"  Error: Blank results for {model_short} not found.")
        return None

    blank_predictions = load_prediction_map(blank_file, target_ids)

    total_geocodable = 0
    total_trapped = 0
    tfr_details = []
    trap_distance_reductions = []

    adversarial_predictions = load_prediction_map(adversarial_file, target_ids)
    for base_id, (pred_lat, pred_lon) in sorted(adversarial_predictions.items()):
        adv_text = taxonomy.get(base_id, {}).get("adversarial_text", "")
        if not adv_text:
            continue

        cached = cache.get(adv_text.strip().lower())
        if cached is None or not isinstance(cached, dict):
            continue

        trap_lat = cached["lat"]
        trap_lon = cached["lon"]
        total_geocodable += 1

        dist_to_trap = haversine_distance(pred_lat, pred_lon, trap_lat, trap_lon)
        if dist_to_trap is not None and dist_to_trap < TRAP_RADIUS_KM:
            total_trapped += 1
            tfr_details.append(
                {
                    "base_id": base_id,
                    "adv_text": adv_text,
                    "trap_lat": trap_lat,
                    "trap_lon": trap_lon,
                    "pred_lat": pred_lat,
                    "pred_lon": pred_lon,
                    "dist_to_trap_km": round(dist_to_trap, 2),
                }
            )

        blank_prediction = blank_predictions.get(base_id)
        if blank_prediction is not None and dist_to_trap is not None:
            blank_dist_to_trap = haversine_distance(blank_prediction[0], blank_prediction[1], trap_lat, trap_lon)
            if blank_dist_to_trap is not None:
                # Positive values mean that the Adversarial prediction moved
                # closer to the injected target than the paired Blank output.
                trap_distance_reductions.append(blank_dist_to_trap - dist_to_trap)

    if total_geocodable == 0:
        print("  No geocodable adversarial texts found.")
        return None

    tfr = total_trapped / total_geocodable * 100
    paired_summary = summarize_trap_distance_reductions(trap_distance_reductions)
    paired_count = paired_summary["paired_count"]
    mean_reduction = paired_summary["mean_km"]
    median_reduction = paired_summary["median_km"]
    attraction_rate = paired_summary["attraction_rate_percent"]
    print(f"\n  TFR Results ({model_short} on {dataset_name}):")
    print(f"    Geocodable samples: {total_geocodable}")
    print(f"    Trapped (within {TRAP_RADIUS_KM}km): {total_trapped}")
    print(f"    TFR: {tfr:.1f}%")
    if paired_count:
        print(f"    Paired Blank/Adversarial samples: {paired_count}")
        print(f"    Mean trap-distance reduction: {mean_reduction:.1f} km")
        print(f"    Median trap-distance reduction: {median_reduction:.1f} km")
        print(f"    Attraction rate (reduction > 0): {attraction_rate:.1f}%")

    if tfr_details:
        print("    Examples of successful traps:")
        for detail in tfr_details[:5]:
            adv_safe = detail["adv_text"].encode("ascii", errors="replace").decode("ascii")
            print(
                f'      "{adv_safe}" -> pred ({detail["pred_lat"]:.2f}, {detail["pred_lon"]:.2f}), '
                f"trap ({detail['trap_lat']:.2f}, {detail['trap_lon']:.2f}), "
                f"dist {detail['dist_to_trap_km']}km"
            )

    return {
        "model": model_short,
        "dataset": dataset_name,
        "tier_filter": tier_filter,
        "geocodable": total_geocodable,
        "trapped": total_trapped,
        "tfr_percent": round(tfr, 2),
        "paired_count": paired_count,
        "mean_trap_distance_reduction_km": (round(mean_reduction, 2) if mean_reduction is not None else None),
        "median_trap_distance_reduction_km": (round(median_reduction, 2) if median_reduction is not None else None),
        "trap_attraction_rate_percent": (round(attraction_rate, 2) if attraction_rate is not None else None),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Trap-Fit Rate (TFR)")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="One or more datasets to process (recommended for batch analysis)",
    )
    parser.add_argument("--model", type=str, default=None, help="Single model short name")
    parser.add_argument("--models", nargs="+", default=None, help="One or more model short names")
    try:
        from config import DATA_ROOT

        default_base_dir = str(DATA_ROOT)
    except ImportError:
        default_base_dir = os.getcwd()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=default_base_dir,
        help="Base directory containing dataset folders",
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Deprecated compatibility flag; aggregate all-tier TFR is now the default",
    )
    parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3"],
        default=None,
        help="Restrict TFR to one taxonomy tier (default: aggregate all tiers)",
    )
    parser.add_argument(
        "--per-tier",
        action="store_true",
        help="Report T1, T2, and T3 separately instead of the aggregate all-tier result",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for per-dataset results and equal-dataset model averages",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional MCRS-compatible JSON path for aggregate model TFR proportions",
    )
    parser.add_argument(
        "--geocode-missing",
        action="store_true",
        help=(
            "Query live Nominatim for missing texts. This can change denominators and "
            "must not be used to reproduce the paper without validating the frozen cohort."
        ),
    )
    args = parser.parse_args()

    if args.tier and args.per_tier:
        parser.error("--tier and --per-tier cannot be used together")

    datasets = args.datasets or ([args.dataset] if args.dataset else ["im2gps3k"])
    explicit_models = args.models or ([args.model] if args.model else None)
    if args.per_tier:
        tier_filters = ["T1", "T2", "T3"]
    elif args.tier:
        tier_filters = [args.tier]
    else:
        # The paper's headline TFR is the aggregate over all taxonomy tiers.
        # --all-tiers is retained as a no-op compatibility alias for this default.
        tier_filters = [None]
    base_dir = Path(args.base_dir)

    print("=" * 50)
    print("  SIGNPOST-Bench Trap-Fit Rate (TFR) Computation")
    print("=" * 50)

    summaries = []
    for dataset_name in datasets:
        dataset_dir = base_dir / dataset_name
        models = explicit_models or discover_models(dataset_name, dataset_dir, base_dir)
        print(f"\n[{normalize_dataset_display_name(dataset_name)}]")
        if not models:
            print("  No adversarial result files found.")
            continue

        for model_short in models:
            print(f"\n=== Model: {model_short} ===")
            for tier in tier_filters:
                tier_label = tier or "ALL"
                print(f"\n--- Tier: {tier_label} ---")
                summary = compute_tfr(
                    dataset_name,
                    dataset_dir,
                    model_short,
                    base_dir,
                    tier_filter=tier,
                    geocode_missing=args.geocode_missing,
                )
                if summary:
                    summaries.append(summary)

    if summaries:
        print("\n" + "=" * 50)
        print("  TFR Summary")
        print("=" * 50)
        for item in summaries:
            tier_label = item["tier_filter"] or "ALL"
            tdr = item["mean_trap_distance_reduction_km"]
            attraction = item["trap_attraction_rate_percent"]
            paired_text = (
                f"TDR={tdr:>8.2f} km AR={attraction:>6.2f}%"
                if tdr is not None and attraction is not None
                else "TDR=      NA AR=    NA"
            )
            print(
                f"{item['dataset']:<10} {item['model']:<25} {tier_label:<3} "
                f"TFR={item['tfr_percent']:>6.2f}% "
                f"({item['trapped']}/{item['geocodable']}) "
                f"{paired_text}"
            )

        if args.output:
            by_model = {}
            for item in summaries:
                if item["tier_filter"] is not None:
                    continue
                by_model.setdefault(item["model"], []).append(item)

            macro = {}
            for model, items in sorted(by_model.items()):
                if len(items) != len(datasets):
                    continue
                macro[model] = {
                    "datasets": len(items),
                    "tfr_percent": round(equal_dataset_macro(items, "tfr_percent"), 2),
                    "mean_trap_distance_reduction_km": round(
                        equal_dataset_macro(items, "mean_trap_distance_reduction_km"), 2
                    ),
                    "median_trap_distance_reduction_km": round(
                        equal_dataset_macro(items, "median_trap_distance_reduction_km"), 2
                    ),
                    "trap_attraction_rate_percent": round(
                        equal_dataset_macro(items, "trap_attraction_rate_percent"), 2
                    ),
                }

            payload = {
                "aggregation": "Compute within each dataset, then average the four dataset-level values equally.",
                "trap_distance_reduction_definition": "distance(Blank prediction, trap) - distance(Adversarial prediction, trap); positive means movement toward the trap.",
                "taxonomy_files": {dataset: str(base_dir / dataset / "taxonomy_labels.jsonl") for dataset in datasets},
                "geocode_cache": "geocode_cache.json",
                "per_dataset": summaries,
                "model_macro_average": macro,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\n  Saved paired TFR/TDR analysis to {args.output}")

            if args.summary_output:
                summary_payload = {
                    "description": "Adversarial Trap-Fit Rate averaged equally across the four benchmark datasets.",
                    "unit": "proportion",
                    "source": str(args.output),
                    "tfr_adv": {model: round(values["tfr_percent"] / 100.0, 4) for model, values in macro.items()},
                }
                args.summary_output.parent.mkdir(parents=True, exist_ok=True)
                with args.summary_output.open("w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, ensure_ascii=False, indent=2)
                print(f"  Saved MCRS TFR summary to {args.summary_output}")


if __name__ == "__main__":
    main()
