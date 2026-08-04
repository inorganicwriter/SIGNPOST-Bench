"""Recompute metrics that depend on the T1/T2/T3 taxonomy labels.

This script deliberately treats the geocode cache as frozen input.  It never
queries a live service and never writes to the cache.  The headline all-tier
metrics are outside its scope; it produces only the tier-stratified WLA/TBS
summary and the T3 trap-radius sensitivity analysis used in the supplement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analysis.compute_results import resolve_entry_id
from config import DATA_ROOT, GEOCODE_CACHE_FILE, get_analysis_output_path
from evaluation.metric_calculator import MetricCalculator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("im2gps3k", "yfcc4k", "googlesv", "baidusv")
TIERS = ("T1", "T2", "T3")
ATTACKS = ("Original", "Blank", "Adversarial")
SENSITIVITY_MODELS = ("gemini-2.5-flash", "gpt-4o-mini")
TRAP_RADII_KM = (10, 25, 50, 100, 250, 500)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_taxonomy(path: Path) -> dict[str, dict]:
    taxonomy = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                taxonomy[str(row["base_id"])] = row
    return taxonomy


def load_latest(path: Path) -> dict[str, dict]:
    """Match the evaluation pipeline's last-record-wins JSONL behavior."""
    latest = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[row.get("filename", "")] = row
    return latest


def result_path(data_root: Path, dataset: str, model: str, attack: str) -> Path:
    return data_root / dataset / "results" / model / f"results_{attack}_{model}.jsonl"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return MetricCalculator.haversine_distance(lat1, lon1, lat2, lon2)


def compute_stratified_metrics(data_root: Path, models: list[str]) -> dict:
    """Pool same-tier samples across datasets per model, then average models."""
    shares = dict.fromkeys(TIERS, 0)
    per_model = {
        model: {tier: {attack: [] for attack in ATTACKS} | {"adversarial_tbs": []} for tier in TIERS}
        for model in models
    }

    for dataset in DATASETS:
        taxonomy = load_taxonomy(data_root / dataset / "taxonomy_labels.jsonl")
        for row in taxonomy.values():
            shares[row["tier"]] += 1

        for model in models:
            rows = {attack: load_latest(result_path(data_root, dataset, model, attack)) for attack in ATTACKS}
            blank_errors = {
                resolve_entry_id(row): float(row["error_km"])
                for row in rows["Blank"].values()
                if row.get("error_km") is not None
            }

            for attack in ATTACKS:
                for row in rows[attack].values():
                    base_id = resolve_entry_id(row)
                    error = row.get("error_km")
                    tier = taxonomy.get(base_id, {}).get("tier")
                    if tier not in TIERS or error is None:
                        continue

                    error = float(error)
                    per_model[model][tier][attack].append(MetricCalculator.calculate_wla(error) * 100.0)
                    if attack == "Adversarial" and base_id in blank_errors:
                        per_model[model][tier]["adversarial_tbs"].append(error - blank_errors[base_id])

    total_groups = sum(shares.values())
    aggregate = {}
    model_detail = {}
    for model in models:
        model_detail[model] = {}
        for tier in TIERS:
            detail = per_model[model][tier]
            model_detail[model][tier] = {
                attack.lower(): {
                    "value": statistics.fmean(detail[attack]),
                    "count": len(detail[attack]),
                }
                for attack in ATTACKS
            }
            model_detail[model][tier]["adversarial_tbs"] = {
                "value_km": statistics.fmean(detail["adversarial_tbs"]),
                "paired_count": len(detail["adversarial_tbs"]),
            }

    for tier in TIERS:
        original = statistics.fmean(model_detail[model][tier]["original"]["value"] for model in models)
        blank = statistics.fmean(model_detail[model][tier]["blank"]["value"] for model in models)
        adversarial = statistics.fmean(model_detail[model][tier]["adversarial"]["value"] for model in models)
        tbs = statistics.fmean(model_detail[model][tier]["adversarial_tbs"]["value_km"] for model in models)
        drop = original - adversarial
        aggregate[tier] = {
            "groups": shares[tier],
            "share_percent": shares[tier] / total_groups * 100.0,
            "original_wla": original,
            "blank_wla": blank,
            "adversarial_wla": adversarial,
            "adversarial_drop_wla": drop,
            "relative_drop_percent": drop / original * 100.0,
            "adversarial_tbs_km": tbs,
        }

    return {
        "aggregation": (
            "For each model, pool same-tier samples across all four datasets; "
            "compute that model's tier metric; then average equally over 20 models."
        ),
        "taxonomy_groups": total_groups,
        "aggregate": aggregate,
        "per_model": model_detail,
    }


def compute_t3_radius_sensitivity(data_root: Path, cache: dict, models: tuple[str, ...] = SENSITIVITY_MODELS) -> dict:
    """Compute the paper's T3 radius sweep using the standard prediction cohort.

    This cohort is defined by T3 membership, an available Adversarial
    coordinate prediction, and a target coordinate in the frozen cache,
    consistent with the main-table TFR/TDR cohort.
    """
    output = {}
    for dataset in DATASETS:
        taxonomy = load_taxonomy(data_root / dataset / "taxonomy_labels.jsonl")
        output[dataset] = {}
        for model in models:
            rows = load_latest(result_path(data_root, dataset, model, "Adversarial"))
            distances = []
            for row in rows.values():
                base_id = resolve_entry_id(row)
                info = taxonomy.get(base_id, {})
                if info.get("tier") != "T3" or row.get("pred_lat") is None or row.get("pred_lon") is None:
                    continue
                query = (info.get("adversarial_text") or "").strip().lower()
                target = cache.get(query)
                if not isinstance(target, dict):
                    continue
                distances.append(
                    haversine_km(
                        float(row["pred_lat"]),
                        float(row["pred_lon"]),
                        float(target["lat"]),
                        float(target["lon"]),
                    )
                )

            output[dataset][model] = {
                "denominator": len(distances),
                "tfr_percent": {
                    str(radius): sum(distance < radius for distance in distances) / len(distances) * 100.0
                    for radius in TRAP_RADII_KM
                },
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--cache",
        type=Path,
        default=GEOCODE_CACHE_FILE,
        help="Frozen geocode cache; this script never modifies it.",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=None,
        help="MCRS leaderboard JSON (required in this supplement; not bundled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_analysis_output_path("tier_metrics.json"),
    )
    args = parser.parse_args()

    if args.leaderboard is None:
        parser.error("--leaderboard is required in this supplement (the MCRS leaderboard JSON is not bundled)")

    leaderboard = load_json(args.leaderboard)
    models = [row["model"] for row in leaderboard["ranking"]]
    if len(models) != 20:
        raise ValueError(f"Expected the paper's 20-model suite, found {len(models)}")

    cache = load_json(args.cache)
    try:
        cache_label = str(args.cache.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        cache_label = str(args.cache.resolve())

    payload = {
        "inputs": {
            "taxonomy": "data/<dataset>/taxonomy_labels.jsonl",
            "geocode_cache": cache_label,
            "t3_radius_sensitivity_selection": (
                "T3 taxonomy membership, available Adversarial coordinate prediction, "
                "and target coordinate in the frozen geocode cache, consistent with the "
                "main-table TFR/TDR cohort."
            ),
            "models": models,
        },
        "tier_stratified": compute_stratified_metrics(args.data_root, models),
        "t3_radius_sensitivity": compute_t3_radius_sensitivity(args.data_root, cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved taxonomy-dependent metrics to {args.output}")


if __name__ == "__main__":
    main()
