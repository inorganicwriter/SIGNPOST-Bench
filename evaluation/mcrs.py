"""Compute the Multimodal Conflict Robustness Score (MCRS).

MCRS = 100 * C^0.40 * R^0.60

Capability:
  C = 0.50 * WLA_original + 0.50 * WLA_blank

Conflict robustness:
  R = 0.22 * retention_random
    + 0.44 * retention_adversarial
    + 0.17 * TBS_quality
    + 0.17 * TFR_quality

The retention terms measure performance relative to the shared Blank
baseline. C and R are reported separately so that the integrated leaderboard
does not obscure the distinction between task capability and conflict
handling.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compute_results import analyze_results, resolve_entry_id
from config import DATA_ROOT, get_analysis_output_path
from evaluation.metric_calculator import MetricCalculator

_MATPLOTLIB_AVAILABLE = None
plt = None


ANCHORS = {
    "wla": (0.0, 1.0),  # WLA ∈ [0,1] by definition
    "tbs": (0.0, 3000.0),  # km; scoring cap
    "tfr": (0.0, 0.40),  # 40% trap rate is catastrophic; higher values
    # carry no additional discriminative information
}

DATASETS = ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]
ATTACKS = ["Original", "Blank", "Similar", "Random", "Adversarial"]
GEOCODE_CACHE_FILE = DATA_ROOT / "geocode_cache.json"
TFR_SUMMARY_FILE = get_analysis_output_path("tfr_model_summary.json")
CAPABILITY_WEIGHTS = {"wla_original": 0.50, "wla_blank": 0.50}
ROBUSTNESS_WEIGHTS = {
    "random_retention": 0.22,
    "adversarial_retention": 0.44,
    "tbs_quality": 0.17,
    "tfr_quality": 0.17,
}
CAPABILITY_EXPONENT = 0.40
RETENTION_FLOOR = 0.10


def norm_high(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def norm_low(x: float, lo: float, hi: float) -> float:
    return float(np.clip((hi - x) / (hi - lo + 1e-9), 0.0, 1.0))


def retention_score(variant_wla: float, blank_wla: float, floor: float = RETENTION_FLOOR) -> float:
    denominator = max(float(blank_wla), floor)
    degradation = max(0.0, float(blank_wla) - float(variant_wla))
    return float(np.clip(1.0 - degradation / denominator, 0.0, 1.0))


def load_jsonl_latest(path: Path) -> dict[str, dict]:
    latest = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = row.get("filename")
            if filename:
                latest[filename] = row
    return latest


def load_geocode_cache() -> dict:
    if GEOCODE_CACHE_FILE.exists():
        with GEOCODE_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_taxonomy_targets(base_dir: Path, dataset: str) -> dict[str, str]:
    """Load the authoritative sample-to-trap mapping for one dataset."""
    path = base_dir / dataset / "taxonomy_labels.jsonl"
    targets = {}
    if not path.exists():
        return targets
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            base_id = str(row.get("base_id", "")).strip()
            target = str(row.get("adversarial_text", "")).strip().lower()
            if base_id and target:
                targets[base_id] = target
    return targets


def load_tfr_summary() -> dict:
    if not TFR_SUMMARY_FILE.exists():
        return {}
    with TFR_SUMMARY_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {model: float(value) for model, value in payload.get("tfr_adv", {}).items()}


def load_metrics_snapshot(path: Path) -> dict[str, dict]:
    """Load primitive MCRS inputs from a frozen paper leaderboard snapshot.

    Derived fields such as C, R, and MCRS are intentionally ignored and
    recomputed by this module, so the snapshot remains auditable.
    """

    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    rows = payload.get("ranking", payload)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a ranking list in metrics snapshot: {path}")

    required = (
        "wla_original",
        "wla_blank",
        "wla_similar",
        "wla_random",
        "wla_adversarial",
        "tbs",
        "tfr_adv",
    )
    optional_components = (
        "random_retention",
        "adversarial_retention",
        "tbs_quality",
        "tfr_quality",
    )
    metrics = {}
    for row in rows:
        model = str(row.get("model", "")).strip().lower()
        if not model:
            continue
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise ValueError(f"Snapshot row for {model} is missing: {', '.join(missing)}")
        metrics[model] = {key: float(row[key]) for key in required}
        for key in optional_components:
            if row.get(key) is not None:
                metrics[model][key] = float(row[key])
    return metrics


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return MetricCalculator.haversine_distance(lat1, lon1, lat2, lon2)


def discover_models(base_dir: Path, datasets: Iterable[str]) -> list[str]:
    per_dataset = []
    for ds in datasets:
        ds_dir = base_dir / ds / "results"
        models = set()
        if not ds_dir.exists():
            continue
        for child in ds_dir.iterdir():
            if child.is_dir() and any(
                f.is_file() and f.name.startswith("results_") and f.suffix == ".jsonl" for f in child.iterdir()
            ):
                models.add(child.name)
        per_dataset.append(models)
    return sorted(set.intersection(*per_dataset)) if per_dataset else []


def compute_dataset_level_metrics(
    dataset: str,
    model: str,
    base_dir: Path,
    geocode_cache: dict,
    taxonomy_targets: dict[str, str] | None = None,
    analyzed_model: dict | None = None,
) -> dict | None:
    model_dir = base_dir / dataset / "results" / model
    attack_files = {attack: model_dir / f"results_{attack}_{model}.jsonl" for attack in ATTACKS}
    if not all(path.exists() for path in attack_files.values()):
        return None

    rows = {attack: load_jsonl_latest(path) for attack, path in attack_files.items()}

    # Dataset-level summary stats from existing analysis
    analyzed = analyzed_model
    if analyzed is None:
        analyzed = analyze_results(base_dir / dataset / "results", dataset_name=dataset).get(model)
    if not analyzed:
        return None

    # Build per-base-id maps for Original / Adversarial
    original = {}
    adversarial = {}
    for row in rows["Original"].values():
        bid = resolve_entry_id(row)
        if row.get("error_km") is not None:
            original[bid] = row
    for row in rows["Adversarial"].values():
        bid = resolve_entry_id(row)
        if row.get("error_km") is not None:
            adversarial[bid] = row

    def safe_rate(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    # TFR for Adversarial only (Similar/Random excluded — geocode_cache lacks those keys)
    taxonomy_targets = taxonomy_targets or {}
    geocodable = 0
    trapped = 0
    for row in rows["Adversarial"].values():
        pred_lat = row.get("pred_lat")
        pred_lon = row.get("pred_lon")
        bid = resolve_entry_id(row)
        text = taxonomy_targets.get(bid, "")
        if pred_lat is None or pred_lon is None or not text:
            continue
        cached = geocode_cache.get(text)
        if not isinstance(cached, dict):
            continue
        geocodable += 1
        dist = haversine(float(pred_lat), float(pred_lon), float(cached["lat"]), float(cached["lon"]))
        if dist is not None and dist < 50.0:
            trapped += 1
    tfr_adv = safe_rate(trapped, geocodable)

    return {
        "wla_original": analyzed["Original"]["WLA"] / 100.0,
        "wla_blank": analyzed["Blank"]["WLA"] / 100.0,
        "wla_similar": analyzed["Similar"]["WLA"] / 100.0,
        "wla_random": analyzed["Random"]["WLA"] / 100.0,
        "wla_adversarial": analyzed["Adversarial"]["WLA"] / 100.0,
        "tbs": float(analyzed["Adversarial"]["TBS"]),
        "tfr_adv": tfr_adv,
    }


def average_metrics_over_datasets(per_dataset_metrics: dict[str, dict]) -> dict:
    keys = next(iter(per_dataset_metrics.values())).keys()
    out = {}
    for key in keys:
        out[key] = float(np.mean([metrics[key] for metrics in per_dataset_metrics.values()]))
    return out


def _normalize_component_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Component weights must sum to a positive value.")
    return {key: float(value) / total for key, value in weights.items()}


def compute_mcrs(
    metrics: dict[str, dict],
    capability_weights: dict[str, float] | None = None,
    robustness_weights: dict[str, float] | None = None,
    capability_exponent: float = CAPABILITY_EXPONENT,
) -> dict[str, dict]:
    nh = norm_high
    nl = norm_low
    A = ANCHORS
    results = {}
    capability_weights = _normalize_component_weights(capability_weights or CAPABILITY_WEIGHTS)
    robustness_weights = _normalize_component_weights(robustness_weights or ROBUSTNESS_WEIGHTS)
    robustness_exponent = 1.0 - capability_exponent

    for model, d in metrics.items():
        n_orig = nh(d["wla_original"], *A["wla"])
        n_blank = nh(d["wla_blank"], *A["wla"])
        components = {
            "similar_retention": retention_score(d["wla_similar"], d["wla_blank"]),
            "random_retention": d.get(
                "random_retention",
                retention_score(d["wla_random"], d["wla_blank"]),
            ),
            "adversarial_retention": d.get(
                "adversarial_retention",
                retention_score(d["wla_adversarial"], d["wla_blank"]),
            ),
            "tbs_quality": d.get(
                "tbs_quality",
                nl(max(0.0, d["tbs"]), *A["tbs"]),
            ),
            "tfr_quality": d.get(
                "tfr_quality",
                nl(d["tfr_adv"], *A["tfr"]),
            ),
        }

        capability_values = {
            "wla_original": n_orig,
            "wla_blank": n_blank,
        }
        c_score = sum(capability_weights[key] * capability_values[key] for key in capability_weights)
        r_score = sum(robustness_weights[key] * components[key] for key in robustness_weights)
        mcrs = 100.0 * max(c_score, 1e-6) ** capability_exponent * max(r_score, 1e-6) ** robustness_exponent
        results[model] = {
            "mcrs": round(mcrs, 2),
            "C": round(c_score, 4),
            "R": round(r_score, 4),
            **{key: round(value, 4) for key, value in components.items()},
            **{k: round(v, 4) for k, v in d.items()},
        }
    return results


def save_rankings(results: dict[str, dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mcrs_leaderboard.json"
    csv_path = out_dir / "mcrs_leaderboard.csv"

    ranked = sorted(results.items(), key=lambda x: (-x[1]["mcrs"], x[0]))
    payload = {"ranking": [{"rank": i + 1, "model": model, **metrics} for i, (model, metrics) in enumerate(ranked)]}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "rank",
        "model",
        "mcrs",
        "C",
        "R",
        "similar_retention",
        "random_retention",
        "adversarial_retention",
        "tbs_quality",
        "tfr_quality",
        "wla_original",
        "wla_blank",
        "wla_similar",
        "wla_random",
        "wla_adversarial",
        "tbs",
        "tfr_adv",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (model, metrics) in enumerate(ranked, start=1):
            row = {"rank": i, "model": model}
            row.update(metrics)
            writer.writerow(row)

    return json_path, csv_path


def plot_leaderboard(results: dict[str, dict], out_dir: Path) -> Path | None:
    global _MATPLOTLIB_AVAILABLE, plt
    if _MATPLOTLIB_AVAILABLE is None:
        try:
            import matplotlib.pyplot as _plt

            plt = _plt
            _MATPLOTLIB_AVAILABLE = True
        except Exception:
            _MATPLOTLIB_AVAILABLE = False
    if not _MATPLOTLIB_AVAILABLE:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results.items(), key=lambda x: (-x[1]["mcrs"], x[0]))
    labels = [m for m, _ in ranked]
    scores = [v["mcrs"] for _, v in ranked]

    fig_h = max(6, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    bars = ax.barh(range(len(labels)), scores, color="#2c7fb8")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MCRS")
    ax.set_title("MCRS Leaderboard")
    ax.set_xlim(0, max(scores) * 1.12 if scores else 100)
    for bar, score in zip(bars, scores, strict=False):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "mcrs_leaderboard.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def build_metrics(base_dir: Path, datasets: Iterable[str]) -> dict[str, dict]:
    geocode_cache = load_geocode_cache()
    tfr_summary = load_tfr_summary()
    datasets = list(datasets)
    models = discover_models(base_dir, datasets)
    analyzed_by_dataset = {ds: analyze_results(base_dir / ds / "results", dataset_name=ds) for ds in datasets}
    taxonomy_by_dataset = {ds: load_taxonomy_targets(base_dir, ds) for ds in datasets}
    all_metrics = {}
    for model in models:
        per_dataset = {}
        for ds in datasets:
            metrics = compute_dataset_level_metrics(
                ds,
                model,
                base_dir,
                geocode_cache,
                taxonomy_targets=taxonomy_by_dataset.get(ds),
                analyzed_model=analyzed_by_dataset.get(ds, {}).get(model),
            )
            if metrics is None:
                per_dataset = {}
                break
            per_dataset[ds] = metrics
        if per_dataset and len(per_dataset) == len(datasets):
            all_metrics[model] = average_metrics_over_datasets(per_dataset)
            if model in tfr_summary:
                all_metrics[model]["tfr_adv"] = tfr_summary[model]
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MCRS leaderboard from SIGNPOST results.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DATA_ROOT),
        help="Dataset root containing <dataset>/results/ (default: SIGNPOST data root)",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to include")
    parser.add_argument(
        "--metrics-snapshot",
        type=Path,
        default=None,
        help="Frozen primitive-metrics JSON used for the published paper tables",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for leaderboard files")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    metrics = (
        load_metrics_snapshot(args.metrics_snapshot)
        if args.metrics_snapshot
        else build_metrics(base_dir, args.datasets)
    )
    if not metrics:
        raise SystemExit(
            f"No model metrics found under {base_dir} for datasets {args.datasets}. "
            "Run the evaluation first (see README) or pass --metrics-snapshot."
        )
    results = compute_mcrs(metrics)
    json_path, csv_path = save_rankings(results, out_dir)
    fig_path = plot_leaderboard(results, out_dir)

    ranked = sorted(results.items(), key=lambda x: (-x[1]["mcrs"], x[0]))
    print(f"{'Rank':<5} {'Model':<24} {'MCRS':>7} {'C':>7} {'R':>7}")
    print("-" * 56)
    for rank, (model, vals) in enumerate(ranked, start=1):
        print(f"{rank:<5} {model:<24} {vals['mcrs']:>7.2f} {vals['C']:>7.4f} {vals['R']:>7.4f}")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if fig_path:
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
