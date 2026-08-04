import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evaluation.metric_calculator import MetricCalculator
from utils.file_utils import get_base_id

logger = logging.getLogger(__name__)

ATTACK_ORDER = ["Blank", "Original", "Similar", "Random", "Adversarial"]
DATASET_KEY_MAP = {
    "im2gps3k": "IM2GPS3K",
    "im2gps": "IM2GPS3K",
    "yfcc4k": "YFCC4K",
    "yfcc": "YFCC4K",
    "googlesv": "GoogleSV",
    "baidusv": "BaiduSV",
}
STANDARD_RESULT_RE = re.compile(r"^results_(?P<attack>[^_]+)_(?P<model>.+)\.jsonl$")


def calculate_wla(error_km):
    return MetricCalculator.calculate_wla(error_km)


def resolve_entry_id(entry):
    """Resolve an entry ID, preferring original_source (preserving its full format)."""
    src = entry.get("original_source")
    if src:
        return str(src).strip()
    fn = entry.get("filename", "")
    return get_base_id(fn)


def normalize_dataset_display_name(dataset_name):
    lower = dataset_name.lower()
    if "im2gps" in lower:
        return "IM2GPS3K"
    if "yfcc" in lower:
        return "YFCC4K"
    if "google" in lower:
        return "GoogleSV"
    if "baidu" in lower:
        return "BaiduSV"
    return dataset_name.upper()


def _parse_result_filename(filename):
    match = STANDARD_RESULT_RE.match(filename)
    if not match:
        return None
    return match.group("attack"), match.group("model")


def discover_result_files(res_dir, dataset_name=None):
    """Discover standard dataset-scoped results under model subdirectories."""
    discovered = {}

    res_path = Path(res_dir)
    if res_path.exists():
        for path in sorted(res_path.rglob("*.jsonl")):
            parsed = _parse_result_filename(path.name)
            if parsed and parsed not in discovered:
                discovered[parsed] = path

    return discovered


def _iter_valid_entries(result_path):
    """Iterate result entries."""
    with Path(result_path).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_id = resolve_entry_id(entry)
            yield entry, entry_id


def _load_baseline_errors(result_path):
    errors = {}
    for entry, base_id in _iter_valid_entries(result_path):
        err = entry.get("error_km")
        if err is not None:
            errors[base_id] = err
    return errors


def _get_attack_order(discovered):
    attacks = []
    seen = set()
    for attack in ATTACK_ORDER:
        if any(found_attack == attack for found_attack, _ in discovered):
            attacks.append(attack)
            seen.add(attack)
    for attack, _ in sorted(discovered):
        if attack not in seen:
            attacks.append(attack)
    return attacks


def analyze_results(res_dir, dataset_name=None):
    out_data = {}
    discovered = discover_result_files(res_dir, dataset_name=dataset_name)
    if not discovered:
        return out_data

    models = sorted({model for _, model in discovered})
    attacks = _get_attack_order(discovered)

    baseline_errors_by_model = {}
    for model in models:
        baseline_path = discovered.get(("Blank", model)) or discovered.get(("Original", model))
        baseline_errors_by_model[model] = _load_baseline_errors(baseline_path) if baseline_path else {}

    for model in models:
        model_results = {}
        baseline_errors = baseline_errors_by_model.get(model, {})

        for attack in attacks:
            result_path = discovered.get((attack, model))
            if not result_path:
                continue

            errors = []
            tbs_values = []

            for entry, base_id in _iter_valid_entries(result_path):
                err = entry.get("error_km")
                if err is None:
                    continue

                errors.append(err)

                baseline_err = baseline_errors.get(base_id)
                if baseline_err is not None and attack not in ("Blank", "Original"):
                    tbs_values.append(err - baseline_err)

            if not errors:
                continue

            mean_tbs = float(np.mean(tbs_values)) if tbs_values else None
            model_results[attack] = {
                "WLA": round(sum(calculate_wla(e) for e in errors) / len(errors) * 100, 2),
                "MedErr": round(float(np.median(errors)), 2),
                "TBS": round(mean_tbs, 2) if mean_tbs is not None else None,
                "Count": len(errors),
                "TBSPairs": len(tbs_values),
            }

        if model_results:
            out_data[model] = model_results

    return out_data


def compute_alpha_sensitivity(res_dir, dataset_name=None, alphas=(0.002, 0.005, 0.01)):
    out = {}
    discovered = discover_result_files(res_dir, dataset_name=dataset_name)
    if not discovered:
        return out

    models = sorted({model for _, model in discovered})
    attacks = _get_attack_order(discovered)

    for model in models:
        out[model] = {}
        for attack in attacks:
            result_path = discovered.get((attack, model))
            if not result_path:
                continue

            errors = [
                entry.get("error_km")
                for entry, _ in _iter_valid_entries(result_path)
                if entry.get("error_km") is not None
            ]
            if not errors:
                continue

            out[model][attack] = {}
            for alpha in alphas:
                wla_scores = [math.exp(-alpha * e) for e in errors]
                out[model][attack][f"alpha={alpha}"] = round(sum(wla_scores) / len(wla_scores) * 100, 2)

    return out


def compute_threshold_distribution(
    res_dir,
    dataset_name=None,
    thresholds=(1, 25, 200, 750, 2500),
):
    bin_labels = [
        "Street (<1km)",
        "City (1-25km)",
        "Region (25-200km)",
        "Country (200-750km)",
        "Continental (750-2500km)",
        "Failed (>2500km)",
    ]
    bin_edges = [0] + list(thresholds) + [float("inf")]
    out = {}

    discovered = discover_result_files(res_dir, dataset_name=dataset_name)
    if not discovered:
        return out

    models = sorted({model for _, model in discovered})
    attacks = _get_attack_order(discovered)

    for model in models:
        out[model] = {}
        for attack in attacks:
            result_path = discovered.get((attack, model))
            if not result_path:
                continue

            errors = [
                entry.get("error_km")
                for entry, _ in _iter_valid_entries(result_path)
                if entry.get("error_km") is not None
            ]
            if not errors:
                continue

            counts = [0] * len(bin_labels)
            for error in errors:
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i] <= error < bin_edges[i + 1]:
                        counts[i] += 1
                        break

            total = len(errors)
            bins = {
                label: {"count": count, "pct": round(count / total * 100, 1)}
                for label, count in zip(bin_labels, counts, strict=False)
            }
            out[model][attack] = {"bins": bins, "total": total}

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute SIGNPOST-Bench Evaluation Results Summary")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing dataset folders (default: from config.py)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["im2gps3k", "yfcc4k", "googlesv", "baidusv"],
        help="Datasets to compute results for (default: all 4 datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: parsed_results.json in script directory)",
    )
    parser.add_argument(
        "--alpha-sensitivity",
        action="store_true",
        help="Compute WLA under multiple alpha values (0.002, 0.005, 0.01)",
    )
    parser.add_argument(
        "--threshold-dist",
        action="store_true",
        help="Compute error distribution across 5-tier threshold bins",
    )
    args = parser.parse_args()

    default_out_path = None
    if args.base_dir is None:
        try:
            from config import DATA_ROOT, get_analysis_output_path

            args.base_dir = str(DATA_ROOT)
            default_out_path = get_analysis_output_path("parsed_results.json")
        except ImportError:
            args.base_dir = "."
            logger.warning("config.py not found, using current directory as base_dir")
    elif args.output is None:
        try:
            from config import get_analysis_output_path

            default_out_path = get_analysis_output_path("parsed_results.json")
        except ImportError:
            default_out_path = None

    base_dir = Path(args.base_dir)
    final_out = {}
    for dataset_name in args.datasets:
        ds_display = normalize_dataset_display_name(dataset_name)
        dataset_results_dir = base_dir / dataset_name / "results"
        final_out[ds_display] = analyze_results(
            dataset_results_dir,
            dataset_name=dataset_name,
        )

    out_path = (
        Path(args.output)
        if args.output
        else (default_out_path or Path(__file__).resolve().parent / "parsed_results.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2)

    print("=" * 60)
    print("  SIGNPOST-Bench Evaluation Results Summary")
    print("=" * 60)
    for ds, ds_data in final_out.items():
        if not ds_data:
            continue
        print(f"\n[{ds}]")
        print(f"{'Model':<25} {'Attack':<15} {'WLA (%)':<10} {'MedErr (km)':<15} {'TBS (km)':<15} {'Count'}")
        print("-" * 95)
        for model, model_data in ds_data.items():
            for attack, metrics in model_data.items():
                tbs_display = f"{metrics['TBS']:.2f}" if metrics["TBS"] is not None else "N/A"
                print(
                    f"{model:<25} {attack:<15} {metrics['WLA']:<10.2f} "
                    f"{metrics['MedErr']:<15.2f} {tbs_display:<15} {metrics['Count']}"
                )
        print("-" * 95)
    print(f"\nMetrics computation complete. JSON saved to {out_path}")

    if args.alpha_sensitivity:
        print("\n" + "=" * 60)
        print("  WLA Alpha Sensitivity Analysis")
        print("  (α = 0.002 / 0.005 / 0.01)")
        print("=" * 60)
        alpha_out = {}
        for dataset_name in args.datasets:
            ds_display = normalize_dataset_display_name(dataset_name)
            dataset_results_dir = base_dir / dataset_name / "results"
            sensitivity = compute_alpha_sensitivity(
                dataset_results_dir,
                dataset_name=dataset_name,
            )
            if sensitivity:
                alpha_out[ds_display] = sensitivity
                print(f"\n[{ds_display}]")
                print(f"{'Model':<25} {'Attack':<15} {'α=0.002':<10} {'α=0.005':<10} {'α=0.01':<10}")
                print("-" * 75)
                for model, attacks in sensitivity.items():
                    for attack, alphas in attacks.items():
                        vals = [str(alphas.get(f"alpha={a}", "--")) for a in (0.002, 0.005, 0.01)]
                        print(f"{model:<25} {attack:<15} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10}")

        alpha_path = out_path.with_name(out_path.stem + "_alpha_sensitivity.json")
        with alpha_path.open("w", encoding="utf-8") as f:
            json.dump(alpha_out, f, indent=2)
        print(f"\nAlpha sensitivity saved to {alpha_path}")

    if args.threshold_dist:
        print("\n" + "=" * 60)
        print("  Error Distribution (5-Tier Threshold Bins)")
        print("=" * 60)
        dist_out = {}
        for dataset_name in args.datasets:
            ds_display = normalize_dataset_display_name(dataset_name)
            dataset_results_dir = base_dir / dataset_name / "results"
            distribution = compute_threshold_distribution(
                dataset_results_dir,
                dataset_name=dataset_name,
            )
            if distribution:
                dist_out[ds_display] = distribution
                print(f"\n[{ds_display}]")
                for model, attacks in distribution.items():
                    for attack, data in attacks.items():
                        print(f"\n  {model} / {attack} (n={data['total']}):")
                        for bin_label, bin_data in data["bins"].items():
                            bar = "█" * int(bin_data["pct"] / 2)
                            print(f"    {bin_label:<25} {bin_data['count']:>5} ({bin_data['pct']:>5.1f}%) {bar}")

        dist_path = out_path.with_name(out_path.stem + "_threshold_dist.json")
        with dist_path.open("w", encoding="utf-8") as f:
            json.dump(dist_out, f, indent=2)
        print(f"\nThreshold distribution saved to {dist_path}")


if __name__ == "__main__":
    main()
