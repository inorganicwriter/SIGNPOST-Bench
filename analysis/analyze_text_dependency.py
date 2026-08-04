"""
Analyze text dependency patterns in SIGNPOST-Bench result files.

This script focuses on cases that are not well captured by WLA/TBS alone:
  1. Text-dependent recovery:
     Blank parse fails, but attack variant succeeds.
  2. Inverse bias:
     Attack variant produces lower error than Blank.
  3. Parse-failure breakdown:
     Failure rate and failure reason by attack type.

Default output:
  analysis/text_dependency_analysis.json

Optional visualization:
  --plot --plot-dir analysis/text_dependency_plots
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.compute_results import (
    discover_result_files,
    normalize_dataset_display_name,
)

TARGET_ATTACKS = ["Similar", "Random", "Adversarial"]
PARSE_ATTACKS = ["Blank", "Similar", "Random", "Adversarial"]
FAILURE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi\s*(?:am|'m)?\s*sorry\b",
        r"\bsorry\b",
        r"\bcan(?:not|'t)\b",
        r"\bunable to\b",
        r"\bnot able to\b",
        r"\bcannot determine\b",
        r"\bcan't determine\b",
        r"\bdo not have enough\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\bi do not have access\b",
        r"\bi can't identify\b",
        r"\bi cannot identify\b",
        r"\bi can't provide\b",
        r"\bi cannot provide\b",
        r"\bpolicy\b",
        r"\brefuse\b",
        r"\bdecline\b",
    ]
]


def strip_ext_name(value):
    return os.path.splitext(os.path.basename(value or ""))[0]


def derive_pair_key(entry):
    original_source = entry.get("original_source")
    if original_source:
        return strip_ext_name(original_source)

    stem = strip_ext_name(entry.get("filename", ""))
    for suffix in ("_Blank", "_Original"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    lower_stem = stem.lower()
    for marker in ("_similar_", "_random_", "_adversarial_"):
        idx = lower_stem.find(marker)
        if idx != -1:
            return stem[:idx]

    return stem


def is_parse_failed(entry):
    parse_failed = entry.get("parse_failed")
    if isinstance(parse_failed, bool):
        return parse_failed
    return entry.get("error_km") is None


def classify_failure_reason(entry):
    if not is_parse_failed(entry):
        return None

    text = (entry.get("prediction_text") or entry.get("raw_response") or "").strip()
    if not text:
        return "empty_response"

    for pattern in FAILURE_PATTERNS:
        if pattern.search(text):
            return "refusal"

    return "format_parse_failure"


def shorten_text(text, limit=200):
    if not text:
        return ""
    text = str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_entries(result_path):
    entries = []
    result_file = Path(result_path)
    if not result_file.exists():
        return entries

    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry["_pair_key"] = derive_pair_key(entry)
            entry["_parse_failed"] = is_parse_failed(entry)
            entry["_failure_reason"] = classify_failure_reason(entry)
            entries.append(entry)

    return entries


def build_blank_index(blank_entries):
    blank_map = {}
    for entry in blank_entries:
        blank_map[entry["_pair_key"]] = entry
    return blank_map


def summarize_parse_failures(entries_by_attack):
    out = {}
    for attack in PARSE_ATTACKS:
        entries = entries_by_attack.get(attack, [])
        total = len(entries)
        failed = [entry for entry in entries if entry["_parse_failed"]]
        refusal = sum(1 for entry in failed if entry["_failure_reason"] == "refusal")
        empty = sum(1 for entry in failed if entry["_failure_reason"] == "empty_response")
        format_fail = sum(1 for entry in failed if entry["_failure_reason"] == "format_parse_failure")

        out[attack.lower()] = {
            "rate": round(len(failed) / total, 4) if total else None,
            "failed": len(failed),
            "total": total,
            "refusal": refusal,
            "empty_response": empty,
            "format_parse_failure": format_fail,
        }
    return out


def make_text_dependent_case(blank_entry, attack_entry):
    return {
        "attack_type": (attack_entry.get("attack_type") or "").lower(),
        "pair_key": attack_entry.get("_pair_key"),
        "blank_filename": blank_entry.get("filename"),
        "attack_filename": attack_entry.get("filename"),
        "original_source": attack_entry.get("original_source"),
        "blank_failure_reason": blank_entry.get("_failure_reason"),
        "blank_prediction_text": shorten_text(blank_entry.get("prediction_text")),
        "attack_prediction_text": shorten_text(attack_entry.get("prediction_text")),
        "attack_error_km": attack_entry.get("error_km"),
        "attack_tbs": attack_entry.get("tbs"),
        "injected_text": attack_entry.get("injected_text"),
    }


def make_inverse_bias_case(blank_entry, attack_entry):
    blank_error = blank_entry.get("error_km")
    attack_error = attack_entry.get("error_km")
    delta = None
    if isinstance(blank_error, (int, float)) and isinstance(attack_error, (int, float)):
        delta = blank_error - attack_error

    return {
        "attack_type": (attack_entry.get("attack_type") or "").lower(),
        "pair_key": attack_entry.get("_pair_key"),
        "blank_filename": blank_entry.get("filename"),
        "attack_filename": attack_entry.get("filename"),
        "original_source": attack_entry.get("original_source"),
        "blank_error_km": blank_error,
        "attack_error_km": attack_error,
        "improvement_km": delta,
        "blank_prediction_text": shorten_text(blank_entry.get("prediction_text")),
        "attack_prediction_text": shorten_text(attack_entry.get("prediction_text")),
        "injected_text": attack_entry.get("injected_text"),
    }


def summarize_attack_pairs(blank_map, attack_entries, case_limit):
    blank_failed_count = 0
    attack_success_count = 0
    paired_with_recovery = 0
    inverse_count = 0
    valid_pairs = 0
    pairable_count = 0

    text_cases = []
    inverse_cases = []

    for attack_entry in attack_entries:
        pair_key = attack_entry["_pair_key"]
        blank_entry = blank_map.get(pair_key)
        if blank_entry is None:
            continue

        pairable_count += 1
        blank_failed = blank_entry["_parse_failed"]
        attack_success = not attack_entry["_parse_failed"]

        if blank_failed:
            blank_failed_count += 1
        if attack_success:
            attack_success_count += 1
        if blank_failed and attack_success:
            paired_with_recovery += 1
            text_cases.append(make_text_dependent_case(blank_entry, attack_entry))

        blank_error = blank_entry.get("error_km")
        attack_error = attack_entry.get("error_km")
        if isinstance(blank_error, (int, float)) and isinstance(attack_error, (int, float)):
            valid_pairs += 1
            if attack_error < blank_error:
                inverse_count += 1
                inverse_cases.append(make_inverse_bias_case(blank_entry, attack_entry))

    text_cases = text_cases[:case_limit]
    inverse_cases.sort(
        key=lambda item: (item.get("improvement_km") is None, -(item.get("improvement_km") or -math.inf))
    )
    inverse_cases = inverse_cases[:case_limit]

    tdr = paired_with_recovery / attack_success_count if attack_success_count else None
    ibs = inverse_count / valid_pairs if valid_pairs else None

    return {
        "text_dependency": {
            "tdr": round(tdr, 4) if tdr is not None else None,
            "blank_failed_count": blank_failed_count,
            "attack_success_count": attack_success_count,
            "paired_with_recovery": paired_with_recovery,
            "pairable_count": pairable_count,
        },
        "inverse_bias": {
            "ibs": round(ibs, 4) if ibs is not None else None,
            "inverse_count": inverse_count,
            "valid_pairs": valid_pairs,
        },
        "cases": {
            "text_dependent_samples": text_cases,
            "inverse_bias_samples": inverse_cases,
        },
    }


def aggregate_overall(by_attack):
    blank_failed_count = 0
    attack_success_count = 0
    paired_with_recovery = 0
    inverse_count = 0
    valid_pairs = 0
    pairable_count = 0

    for attack_stats in by_attack.values():
        td = attack_stats["text_dependency"]
        ib = attack_stats["inverse_bias"]
        blank_failed_count += td["blank_failed_count"]
        attack_success_count += td["attack_success_count"]
        paired_with_recovery += td["paired_with_recovery"]
        pairable_count += td["pairable_count"]
        inverse_count += ib["inverse_count"]
        valid_pairs += ib["valid_pairs"]

    tdr = paired_with_recovery / attack_success_count if attack_success_count else None
    ibs = inverse_count / valid_pairs if valid_pairs else None

    return {
        "text_dependency": {
            "tdr": round(tdr, 4) if tdr is not None else None,
            "blank_failed_count": blank_failed_count,
            "attack_success_count": attack_success_count,
            "paired_with_recovery": paired_with_recovery,
            "pairable_count": pairable_count,
        },
        "inverse_bias": {
            "ibs": round(ibs, 4) if ibs is not None else None,
            "inverse_count": inverse_count,
            "valid_pairs": valid_pairs,
        },
    }


def merge_cases(by_attack, case_limit):
    text_cases = []
    inverse_cases = []
    for attack_stats in by_attack.values():
        text_cases.extend(attack_stats["cases"]["text_dependent_samples"])
        inverse_cases.extend(attack_stats["cases"]["inverse_bias_samples"])

    inverse_cases.sort(
        key=lambda item: (item.get("improvement_km") is None, -(item.get("improvement_km") or -math.inf))
    )

    return {
        "text_dependent_samples": text_cases[:case_limit],
        "inverse_bias_samples": inverse_cases[:case_limit],
    }


def analyze_model_dataset(dataset_name, model_name, result_files, case_limit):
    entries_by_attack = {}
    for attack in PARSE_ATTACKS:
        result_path = result_files.get((attack, model_name))
        entries_by_attack[attack] = load_entries(result_path) if result_path else []

    blank_entries = entries_by_attack.get("Blank", [])
    blank_map = build_blank_index(blank_entries)

    by_attack = {}
    for attack in TARGET_ATTACKS:
        attack_entries = entries_by_attack.get(attack, [])
        by_attack[attack.lower()] = summarize_attack_pairs(blank_map, attack_entries, case_limit)

    overall = aggregate_overall(by_attack)

    blank_failed = [entry for entry in blank_entries if entry["_parse_failed"]]
    blank_summary = {
        "parse_failed_rate": round(len(blank_failed) / len(blank_entries), 4) if blank_entries else None,
        "parse_failed_count": len(blank_failed),
        "total": len(blank_entries),
    }

    return {
        "model": model_name,
        "dataset": normalize_dataset_display_name(dataset_name),
        "blank_summary": blank_summary,
        "text_dependency": {
            **overall["text_dependency"],
            "by_attack": {attack: stats["text_dependency"] for attack, stats in by_attack.items()},
        },
        "inverse_bias": {
            **overall["inverse_bias"],
            "by_attack": {attack: stats["inverse_bias"] for attack, stats in by_attack.items()},
        },
        "parse_failure": summarize_parse_failures(entries_by_attack),
        "cases": merge_cases(by_attack, case_limit),
    }


def discover_models_for_dataset(dataset_name, base_dir):
    dataset_results_dir = base_dir / dataset_name / "results"
    discovered = discover_result_files(dataset_results_dir, dataset_name=dataset_name)
    models = {
        model_name for attack, model_name in discovered if attack in {"Blank", "Similar", "Random", "Adversarial"}
    }
    return sorted(models), discovered


def plot_heatmap(matrix, row_labels, col_labels, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 1.7), max(3, len(row_labels) * 0.5)))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            label = "N/A" if np.isnan(value) else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_plots(records, output_dir):
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping text dependency heatmaps. ({exc})")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attacks = ["similar", "random", "adversarial"]
    by_dataset = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)

    for dataset_name, dataset_records in by_dataset.items():
        models = sorted(record["model"] for record in dataset_records)
        tdr_matrix = np.full((len(models), len(attacks)), np.nan)
        ibs_matrix = np.full((len(models), len(attacks)), np.nan)

        record_by_model = {record["model"]: record for record in dataset_records}
        for i, model_name in enumerate(models):
            record = record_by_model[model_name]
            for j, attack in enumerate(attacks):
                tdr = record["text_dependency"]["by_attack"].get(attack, {}).get("tdr")
                ibs = record["inverse_bias"]["by_attack"].get(attack, {}).get("ibs")
                if tdr is not None:
                    tdr_matrix[i, j] = tdr * 100
                if ibs is not None:
                    ibs_matrix[i, j] = ibs * 100

        safe_dataset = dataset_name.lower()
        tdr_path = output_dir / f"text_dependency_heatmap_{safe_dataset}.png"
        ibs_path = output_dir / f"inverse_bias_heatmap_{safe_dataset}.png"

        plot_heatmap(
            tdr_matrix,
            models,
            [attack.capitalize() for attack in attacks],
            f"{dataset_name} — Text Dependency Rate",
            tdr_path,
        )
        plot_heatmap(
            ibs_matrix,
            models,
            [attack.capitalize() for attack in attacks],
            f"{dataset_name} — Inverse Bias Score",
            ibs_path,
        )
        print(f"   {tdr_path}")
        print(f"   {ibs_path}")


def build_output(records):
    if len(records) == 1:
        return records[0]
    return {"results": records}


def main():
    parser = argparse.ArgumentParser(description="Analyze text dependency in SIGNPOST-Bench results")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base data directory containing dataset folders (default: from config.py)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["im2gps3k", "yfcc4k", "googlesv", "baidusv"],
        help="Datasets to analyze",
    )
    parser.add_argument("--model", type=str, default=None, help="Single model short name")
    parser.add_argument("--models", nargs="+", default=None, help="One or more model short names")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON output",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=20,
        help="Maximum number of example cases to keep per category",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate text dependency and inverse bias heatmaps",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save heatmaps when --plot is enabled",
    )
    args = parser.parse_args()

    if args.base_dir is None:
        try:
            from config import DATA_ROOT

            args.base_dir = str(DATA_ROOT)
        except ImportError:
            args.base_dir = "data"

    if args.output is None:
        try:
            from config import get_analysis_output_path

            args.output = str(get_analysis_output_path("text_dependency_analysis.json"))
        except ImportError:
            args.output = "analysis/text_dependency_analysis.json"

    if args.plot_dir is None:
        try:
            from config import get_analysis_plot_dir

            args.plot_dir = str(get_analysis_plot_dir("text_dependency_plots"))
        except ImportError:
            args.plot_dir = "analysis/text_dependency_plots"

    base_dir = Path(args.base_dir)
    explicit_models = args.models or ([args.model] if args.model else None)

    records = []
    for dataset_name in args.datasets:
        discovered_models, discovered_files = discover_models_for_dataset(dataset_name, base_dir)
        model_names = explicit_models or discovered_models
        if not model_names:
            print(f"[WARN] No result files found for dataset: {dataset_name}")
            continue

        for model_name in model_names:
            record = analyze_model_dataset(
                dataset_name,
                model_name,
                discovered_files,
                args.case_limit,
            )
            records.append(record)

    output = build_output(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved text dependency analysis to {output_path}")

    if args.plot and records:
        generate_plots(records, args.plot_dir)


if __name__ == "__main__":
    main()
