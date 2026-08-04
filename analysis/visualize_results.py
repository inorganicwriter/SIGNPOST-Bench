"""
analysis/visualize_results.py
==============================
Comprehensive visualization for SIGNPOST-Bench evaluation results.

Supports two modes:
  1. --results Model=path.jsonl  (raw JSONL files)
  2. --from-json parsed_results.json  (aggregated JSON from compute_results.py)

Generates:
  - Leaderboard (mean error bar chart)
  - Robustness comparison (attack type grouped bars)
  - CDF comparison (adversarial error distribution)
  - WLA heatmap (model × attack type)
  - TBS heatmap (model × attack type)
  - Alpha sensitivity comparison
  - Threshold distribution stacked bars
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd  # noqa: F401

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


# ── Color palette ──────────────────────────────────────────
ATTACK_COLORS = {
    "blank": "#95a5a6",
    "original": "#3498db",
    "similar": "#2ecc71",
    "random": "#f39c12",
    "adversarial": "#e74c3c",
}
ATTACK_ORDER = ["blank", "original", "similar", "random", "adversarial"]
ALPHA_COLORS = {0.002: "#3498db", 0.005: "#e74c3c", 0.01: "#2ecc71"}


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════


def load_jsonl_data(results_args):
    """Load raw JSONL files specified as ModelName=path."""
    rows = []
    for item in results_args:
        if "=" not in item:
            print(f"[WARN] Skipping '{item}' — expected format ModelName=path.jsonl")
            continue
        model_name, file_path = item.split("=", 1)
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue
        print(f"  Loading {model_name} from {file_path}...")
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    attack = entry.get("attack_type", "")
                    if not attack or attack == "unknown":
                        fn = entry.get("filename", "")
                        if "_similar_" in fn:
                            attack = "similar"
                        elif "_random_" in fn:
                            attack = "random"
                        elif "_adversarial_" in fn:
                            attack = "adversarial"
                        elif "_blank" in fn.lower():
                            attack = "blank"
                        else:
                            attack = "original"
                    entry["model"] = model_name
                    entry["attack_type"] = attack.lower()
                    rows.append(entry)
                except (json.JSONDecodeError, KeyError):
                    pass
    return rows


def load_aggregated_json(path):
    """Load pre-computed aggregated JSON from compute_results.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  Plot 1: Leaderboard
# ══════════════════════════════════════════════════════════════


def plot_leaderboard(rows, output_dir):
    """Bar chart of mean geodesic error per model (lower is better)."""
    model_errors = {}
    for r in rows:
        err = r.get("error_km")
        if err is not None:
            model_errors.setdefault(r["model"], []).append(err)

    models = sorted(model_errors, key=lambda m: np.mean(model_errors[m]))
    means = [np.mean(model_errors[m]) for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    bars = ax.bar(models, means, color="#3498db", edgecolor="white")
    for bar, val in zip(bars, means, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}", ha="center", va="bottom", fontsize=9
        )
    ax.set_ylabel("Mean Error (km)")
    ax.set_title("Leaderboard: Mean Geodesic Distance (↓ better)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = os.path.join(output_dir, "leaderboard.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 2: Robustness Grouped Bars
# ══════════════════════════════════════════════════════════════


def plot_robustness(rows, output_dir):
    """Grouped bar chart: mean error by model × attack type."""
    data = {}
    for r in rows:
        err = r.get("error_km")
        if err is None:
            continue
        key = (r["model"], r["attack_type"])
        data.setdefault(key, []).append(err)

    models = sorted({k[0] for k in data})
    attacks = [a for a in ATTACK_ORDER if any(k[1] == a for k in data)]
    n_models, n_attacks = len(models), len(attacks)
    if n_models == 0 or n_attacks == 0:
        return

    x = np.arange(n_models)
    width = 0.8 / n_attacks

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.8), 6))
    for i, atk in enumerate(attacks):
        means = [np.mean(data.get((m, atk), [0])) for m in models]
        color = ATTACK_COLORS.get(atk, "#777")
        ax.bar(x + i * width - 0.4 + width / 2, means, width, label=atk.capitalize(), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Mean Error (km)")
    ax.set_title("Robustness Under Different Attack Types")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "robustness_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 3: CDF of Adversarial Errors
# ══════════════════════════════════════════════════════════════


def plot_cdf(rows, output_dir):
    """CDF of adversarial error per model (log-x)."""
    model_errs = {}
    for r in rows:
        if r.get("attack_type") == "adversarial" and r.get("error_km") is not None:
            model_errs.setdefault(r["model"], []).append(r["error_km"])
    if not model_errs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model, errs in sorted(model_errs.items()):
        errs_sorted = np.sort(errs)
        cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
        ax.plot(errs_sorted, cdf, linewidth=2, label=model)

    ax.set_xscale("log")
    ax.set_xlabel("Error (km) — Log Scale")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF: Adversarial Sample Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "cdf_adversarial.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 4: WLA Heatmap
# ══════════════════════════════════════════════════════════════


def plot_wla_heatmap(rows, output_dir, alpha=0.005):
    """Heatmap of mean WLA (model × attack type)."""
    data = {}
    for r in rows:
        err = r.get("error_km")
        if err is None:
            continue
        key = (r["model"], r["attack_type"])
        data.setdefault(key, []).append(math.exp(-alpha * err))

    models = sorted({k[0] for k in data})
    attacks = [a for a in ATTACK_ORDER if any(k[1] == a for k in data)]
    if not models or not attacks:
        return

    matrix = np.zeros((len(models), len(attacks)))
    for i, m in enumerate(models):
        for j, a in enumerate(attacks):
            vals = data.get((m, a), [])
            matrix[i, j] = np.mean(vals) * 100 if vals else 0

    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 2), max(4, len(models) * 0.6)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([a.capitalize() for a in attacks])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(attacks)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] < 40 else "black",
            )
    fig.colorbar(im, ax=ax, label="WLA (%)")
    ax.set_title(f"WLA Heatmap (α={alpha})")
    fig.tight_layout()
    path = os.path.join(output_dir, "wla_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 5: TBS Heatmap
# ══════════════════════════════════════════════════════════════


def plot_tbs_heatmap(rows, output_dir):
    """Heatmap of mean TBS (model × attack type), excluding blank/original."""
    data = {}
    for r in rows:
        tbs = r.get("tbs")
        if tbs is None:
            continue
        atk = r.get("attack_type", "")
        if atk in ("blank", "original", "clean"):
            continue
        key = (r["model"], atk)
        data.setdefault(key, []).append(tbs)

    models = sorted({k[0] for k in data})
    attacks = [a for a in ["similar", "random", "adversarial"] if any(k[1] == a for k in data)]
    if not models or not attacks:
        return

    matrix = np.zeros((len(models), len(attacks)))
    for i, m in enumerate(models):
        for j, a in enumerate(attacks):
            vals = data.get((m, a), [])
            matrix[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(6, len(attacks) * 2.5), max(4, len(models) * 0.6)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([a.capitalize() for a in attacks])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(attacks)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="TBS (km)")
    ax.set_title("Text Bias Score Heatmap (positive = text hurts)")
    fig.tight_layout()
    path = os.path.join(output_dir, "tbs_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 6: Alpha Sensitivity (from JSON)
# ══════════════════════════════════════════════════════════════


def plot_alpha_sensitivity(json_path, output_dir):
    """Bar chart comparing WLA under α=0.002/0.005/0.01 from alpha_sensitivity JSON."""
    if not os.path.exists(json_path):
        print(f"  [SKIP] Alpha sensitivity JSON not found: {json_path}")
        return

    data = load_aggregated_json(json_path)
    alphas = [0.002, 0.005, 0.01]

    for dataset, models in data.items():
        for model, attacks in models.items():
            atk_names = [a for a in ATTACK_ORDER if a.capitalize() in attacks or a in attacks]
            if not atk_names:
                continue

            n_atk = len(atk_names)
            x = np.arange(n_atk)
            width = 0.25

            fig, ax = plt.subplots(figsize=(max(6, n_atk * 2), 5))
            for i, alpha in enumerate(alphas):
                key = f"alpha={alpha}"
                vals = []
                for atk in atk_names:
                    atk_key = atk.capitalize() if atk.capitalize() in attacks else atk
                    vals.append(attacks.get(atk_key, {}).get(key, 0))
                ax.bar(x + (i - 1) * width, vals, width, label=f"α={alpha}", color=ALPHA_COLORS[alpha])

            ax.set_xticks(x)
            ax.set_xticklabels([a.capitalize() for a in atk_names])
            ax.set_ylabel("WLA (%)")
            ax.set_title(f"Alpha Sensitivity — {dataset} / {model}")
            ax.legend()
            fig.tight_layout()
            safe_name = f"alpha_sensitivity_{dataset}_{model}".replace(" ", "_")
            path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Plot 7: Threshold Distribution (from JSON)
# ══════════════════════════════════════════════════════════════


def plot_threshold_distribution(json_path, output_dir):
    """Stacked bar chart showing error distribution across 5-tier bins."""
    if not os.path.exists(json_path):
        print(f"  [SKIP] Threshold dist JSON not found: {json_path}")
        return

    data = load_aggregated_json(json_path)
    bin_colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

    for dataset, models in data.items():
        for model, attacks in models.items():
            atk_names = [a for a in ["Blank", "Original", "Similar", "Random", "Adversarial"] if a in attacks]
            if not atk_names:
                continue

            # Collect bin labels from first available attack
            first_atk = attacks[atk_names[0]]
            bin_labels = list(first_atk["bins"].keys())
            n_atk = len(atk_names)

            fig, ax = plt.subplots(figsize=(max(8, n_atk * 1.8), 5))
            x = np.arange(n_atk)
            bottoms = np.zeros(n_atk)

            for bi, bl in enumerate(bin_labels):
                pcts = [attacks[a]["bins"][bl]["pct"] for a in atk_names]
                color = bin_colors[bi % len(bin_colors)]
                ax.bar(x, pcts, bottom=bottoms, label=bl, color=color, edgecolor="white", linewidth=0.5)
                bottoms += pcts

            ax.set_xticks(x)
            ax.set_xticklabels(atk_names, rotation=15)
            ax.set_ylabel("Percentage (%)")
            ax.set_title(f"Error Distribution — {dataset} / {model}")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            safe_name = f"threshold_dist_{dataset}_{model}".replace(" ", "_")
            path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"   {path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SIGNPOST-Bench Results")
    parser.add_argument("--results", nargs="+", default=None, help="Raw JSONL files: ModelName=path.jsonl")
    parser.add_argument(
        "--from-json", type=str, default=None, help="Aggregated parsed_results.json from compute_results.py"
    )
    parser.add_argument(
        "--alpha-json", type=str, default=None, help="Alpha sensitivity JSON (parsed_results_alpha_sensitivity.json)"
    )
    parser.add_argument(
        "--dist-json", type=str, default=None, help="Threshold distribution JSON (parsed_results_threshold_dist.json)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        try:
            from config import get_analysis_plot_dir

            args.output_dir = str(get_analysis_plot_dir("visualizations"))
        except ImportError:
            args.output_dir = "./visualizations"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # ── Raw JSONL mode ──
    if args.results:
        rows = load_jsonl_data(args.results)
        if not rows:
            print("[ERROR] No data loaded from JSONL files.")
        else:
            n_models = len({r["model"] for r in rows})
            print(f"\nLoaded {len(rows)} records from {n_models} models.\n")

            print("Generating plots from JSONL data:")
            plot_leaderboard(rows, args.output_dir)
            plot_robustness(rows, args.output_dir)
            plot_cdf(rows, args.output_dir)
            plot_wla_heatmap(rows, args.output_dir)
            plot_tbs_heatmap(rows, args.output_dir)

    # ── Alpha sensitivity ──
    alpha_path = args.alpha_json
    if alpha_path is None:
        # Auto-detect
        candidates = []
        try:
            from config import get_analysis_output_path

            candidates.append(str(get_analysis_output_path("parsed_results_alpha_sensitivity.json")))
        except ImportError:
            pass
        candidates.extend(
            [
                "analysis/parsed_results_alpha_sensitivity.json",
                "parsed_results_alpha_sensitivity.json",
            ]
        )
        for c in candidates:
            if os.path.exists(c):
                alpha_path = c
                break
    if alpha_path:
        print("\nGenerating alpha sensitivity plots:")
        plot_alpha_sensitivity(alpha_path, args.output_dir)

    # ── Threshold distribution ──
    dist_path = args.dist_json
    if dist_path is None:
        candidates = []
        try:
            from config import get_analysis_output_path

            candidates.append(str(get_analysis_output_path("parsed_results_threshold_dist.json")))
        except ImportError:
            pass
        candidates.extend(
            [
                "analysis/parsed_results_threshold_dist.json",
                "parsed_results_threshold_dist.json",
            ]
        )
        for c in candidates:
            if os.path.exists(c):
                dist_path = c
                break
    if dist_path:
        print("\nGenerating threshold distribution plots:")
        plot_threshold_distribution(dist_path, args.output_dir)

    if not args.results and not alpha_path and not dist_path:
        print("[ERROR] No input data provided.")
        print("Usage:")
        print("  python -m analysis.visualize_results --results Model1=file1.jsonl Model2=file2.jsonl")
        print("  python -m analysis.visualize_results --alpha-json parsed_results_alpha_sensitivity.json")
        print("  python -m analysis.visualize_results --dist-json parsed_results_threshold_dist.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
