"""Validate MCRS through weight sensitivity and component ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from scipy.stats import kendalltau, spearmanr

from config import DATA_ROOT, get_analysis_output_path
from evaluation import mcrs as mcrs_core

BASE_C_WEIGHTS = dict(mcrs_core.CAPABILITY_WEIGHTS)
BASE_R_WEIGHTS = dict(mcrs_core.ROBUSTNESS_WEIGHTS)
ABLATION_COMPONENTS = (
    "random_retention",
    "adversarial_retention",
    "tbs_quality",
    "tfr_quality",
)


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("At least one positive component weight is required.")
    return {key: float(value) / total for key, value in weights.items()}


def compute_scores(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    c_weights: Mapping[str, float] = BASE_C_WEIGHTS,
    r_weights: Mapping[str, float] = BASE_R_WEIGHTS,
    capability_exponent: float = 0.40,
) -> dict[str, dict]:
    return mcrs_core.compute_mcrs(
        dict(metrics),
        capability_weights=dict(c_weights),
        robustness_weights=dict(r_weights),
        capability_exponent=capability_exponent,
    )


def _ranking(scores: Mapping[str, Mapping[str, float]]) -> list[str]:
    return [model for model, _ in sorted(scores.items(), key=lambda item: (-item[1]["mcrs"], item[0]))]


def rank_stability(
    baseline: Mapping[str, Mapping[str, float]],
    variant: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    models = sorted(set(baseline) & set(variant))
    baseline_order = _ranking({model: baseline[model] for model in models})
    variant_order = _ranking({model: variant[model] for model in models})
    baseline_ranks = {model: rank for rank, model in enumerate(baseline_order)}
    variant_ranks = {model: rank for rank, model in enumerate(variant_order)}
    first = [baseline_ranks[model] for model in models]
    second = [variant_ranks[model] for model in models]
    return {
        "kendall_tau": float(kendalltau(first, second).statistic),
        "spearman_rho": float(spearmanr(first, second).statistic),
    }


def leave_one_component_out(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, dict]:
    variants = {}
    for component in ABLATION_COMPONENTS:
        weights = dict(BASE_R_WEIGHTS)
        weights.pop(component)
        weights = _normalize_weights(weights)
        variants[component] = {
            "weights": weights,
            "scores": compute_scores(
                metrics,
                r_weights=weights,
            ),
        }
    return variants


def build_validation_metrics(base_dir: Path) -> dict[str, dict]:
    return mcrs_core.build_metrics(base_dir, mcrs_core.DATASETS)


def validate_mcrs(metrics: Mapping[str, Mapping[str, float]]) -> dict:
    baseline = compute_scores(metrics)
    weight_variants = {}

    for adversarial_weight in (0.30, 0.44, 0.50):
        remainder = 1.0 - adversarial_weight
        other_total = 1.0 - BASE_R_WEIGHTS["adversarial_retention"]
        weights = {
            key: (adversarial_weight if key == "adversarial_retention" else value / other_total * remainder)
            for key, value in BASE_R_WEIGHTS.items()
        }
        name = f"adversarial_weight_{adversarial_weight:.2f}"
        scores = compute_scores(metrics, r_weights=weights)
        weight_variants[name] = {
            "weights": weights,
            **rank_stability(baseline, scores),
        }

    for exponent in (0.30, 0.40, 0.50):
        name = f"capability_exponent_{exponent:.2f}"
        scores = compute_scores(metrics, capability_exponent=exponent)
        weight_variants[name] = {
            "capability_exponent": exponent,
            **rank_stability(baseline, scores),
        }

    ablations = {}
    for component, variant in leave_one_component_out(metrics).items():
        ablations[component] = {
            **rank_stability(baseline, variant["scores"]),
            "weights": variant["weights"],
        }

    baseline_values = [baseline[model]["mcrs"] for model in sorted(baseline)]
    robustness_values = [baseline[model]["R"] for model in sorted(baseline)]
    original_values = [metrics[model]["wla_original"] for model in sorted(baseline)]
    adversarial_values = [metrics[model]["wla_adversarial"] for model in sorted(baseline)]
    return {
        "baseline": {
            model: {key: round(value, 6) for key, value in scores.items()} for model, scores in baseline.items()
        },
        "weight_sensitivity": weight_variants,
        "component_ablation": ablations,
        "correlations": {
            "mcrs_vs_original_wla_spearman": float(spearmanr(baseline_values, original_values).statistic),
            "mcrs_vs_adversarial_wla_spearman": float(spearmanr(baseline_values, adversarial_values).statistic),
            "robustness_vs_adversarial_wla_spearman": float(spearmanr(robustness_values, adversarial_values).statistic),
            "robustness_vs_tbs_spearman": float(
                spearmanr(
                    robustness_values,
                    [metrics[model]["tbs"] for model in sorted(baseline)],
                ).statistic
            ),
            "robustness_vs_tfr_spearman": float(
                spearmanr(
                    robustness_values,
                    [metrics[model]["tfr_adv"] for model in sorted(baseline)],
                ).statistic
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--metrics-snapshot",
        type=Path,
        default=None,
        help="Frozen primitive-metrics JSON used for the published paper tables",
    )
    parser.add_argument("--output", type=Path, default=get_analysis_output_path("mcrs_validation.json"))
    args = parser.parse_args()

    metrics = (
        mcrs_core.load_metrics_snapshot(args.metrics_snapshot)
        if args.metrics_snapshot
        else build_validation_metrics(args.base_dir)
    )
    payload = validate_mcrs(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    weight_tau = [row["kendall_tau"] for row in payload["weight_sensitivity"].values()]
    ablation_tau = [row["kendall_tau"] for row in payload["component_ablation"].values()]
    print(f"Models: {len(metrics)}")
    print(f"Minimum weight-sensitivity Kendall tau: {min(weight_tau):.3f}")
    print(f"Minimum ablation Kendall tau: {min(ablation_tau):.3f}")
    print(
        "Spearman(MCRS, Original/Adversarial WLA): "
        f"{payload['correlations']['mcrs_vs_original_wla_spearman']:.3f} / "
        f"{payload['correlations']['mcrs_vs_adversarial_wla_spearman']:.3f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
