"""Paired significance tests for Original versus Adversarial predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon

from analysis.compute_results import (
    calculate_wla,
    discover_result_files,
    resolve_entry_id,
)
from config import DATA_ROOT, get_analysis_output_path


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def rank_biserial_from_differences(differences: Sequence[float]) -> float:
    """Compute matched-pairs rank-biserial correlation."""
    nonzero = np.asarray([value for value in differences if value != 0], dtype=float)
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_wilcoxon(
    original_scores: Sequence[float],
    adversarial_scores: Sequence[float],
) -> dict:
    if len(original_scores) != len(adversarial_scores):
        raise ValueError("Paired score sequences must have equal length.")
    if not original_scores:
        raise ValueError("At least one paired observation is required.")

    differences = np.asarray(original_scores, dtype=float) - np.asarray(adversarial_scores, dtype=float)
    statistic, p_value = wilcoxon(
        differences,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "n": int(differences.size),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "rank_biserial": float(rank_biserial_from_differences(differences)),
        "median_wla_change": float(np.median(differences) * 100.0),
    }


def _load_scores(path: Path) -> dict[str, float]:
    scores = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            base_id = resolve_entry_id(row)
            error_km = row.get("error_km")
            if error_km is None:
                continue
            scores[base_id] = calculate_wla(float(error_km))
    return scores


def collect_model_pairs(
    base_dir: Path,
    datasets: Iterable[str],
) -> dict[str, tuple[list[float], list[float]]]:
    paired_by_model: dict[str, tuple[list[float], list[float]]] = {}
    for dataset in datasets:
        results_dir = base_dir / dataset / "results"
        discovered = discover_result_files(results_dir, dataset_name=dataset)
        models = {
            model for attack, model in discovered if attack == "Original" and ("Adversarial", model) in discovered
        }
        for model in models:
            original = _load_scores(discovered[("Original", model)])
            adversarial = _load_scores(discovered[("Adversarial", model)])
            common_ids = sorted(original.keys() & adversarial.keys())
            original_values, adversarial_values = paired_by_model.setdefault(model, ([], []))
            original_values.extend(original[base_id] for base_id in common_ids)
            adversarial_values.extend(adversarial[base_id] for base_id in common_ids)
    return paired_by_model


def validate_models(base_dir: Path, datasets: Iterable[str]) -> dict:
    paired = collect_model_pairs(base_dir, datasets)
    rows = []
    for model in sorted(paired):
        original, adversarial = paired[model]
        result = paired_wilcoxon(original, adversarial)
        rows.append({"model": model, **result})

    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted, strict=False):
        row["holm_p_value"] = adjusted_p
    return {
        "test": "one-sided paired Wilcoxon signed-rank on per-image WLA",
        "alternative": "Original WLA > Adversarial WLA",
        "multiple_testing": "Holm correction across models",
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["im2gps3k", "yfcc4k", "googlesv", "baidusv"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_analysis_output_path("statistical_validation.json"),
    )
    args = parser.parse_args()

    payload = validate_models(args.base_dir, args.datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    max_adjusted = max(row["holm_p_value"] for row in payload["models"])
    min_effect = min(row["rank_biserial"] for row in payload["models"])
    min_n = min(row["n"] for row in payload["models"])
    print(f"Models: {len(payload['models'])}")
    print(f"Minimum paired N: {min_n}")
    print(f"Maximum Holm-adjusted p-value: {max_adjusted:.3e}")
    print(f"Minimum rank-biserial effect: {min_effect:.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
