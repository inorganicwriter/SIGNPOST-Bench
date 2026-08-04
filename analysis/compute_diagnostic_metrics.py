"""Recompute the probing, defense, and cross-task diagnostic tables.

The paper reports equal-dataset macro averages for probing/defense and pooled
rates for cross-task generalization.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from config import DATA_ROOT, get_analysis_output_path

DATASETS = ("im2gps3k", "yfcc4k", "googlesv", "baidusv")
MODELS = ("gemini-2.5-flash", "gpt-4o-mini")


def load_latest_rows(path: Path) -> list[dict]:
    """Load the last valid row for each filename from a resumable JSONL file."""
    latest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = row.get("filename")
            if filename:
                latest[filename] = row
    return list(latest.values())


def load_mode_rows(base_dir: Path, dataset: str, model: str, mode: str) -> list[dict]:
    result_dir = base_dir / dataset / "results" / model / mode
    paths = sorted(result_dir.glob("*.jsonl"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one {mode} JSONL for {dataset}/{model}, found {len(paths)}")
    return load_latest_rows(paths[0])


def probing_summary(rows: list[dict]) -> dict:
    valid_predictions = [row for row in rows if row.get("error_km") is not None and row.get("wla_score") is not None]
    valid_judgments = [
        row
        for row in rows
        if row.get("failure_reason") != "probing_json_parse_failure" and row.get("consistent") is not None
    ]
    return {
        "n_valid_predictions": len(valid_predictions),
        "n_valid_judgments": len(valid_judgments),
        "wla": 100.0 * sum(row["wla_score"] for row in valid_predictions) / len(valid_predictions),
        "median_error_km": statistics.median(row["error_km"] for row in valid_predictions),
        "cda_percent": 100.0 * sum(bool(row.get("cda_hit")) for row in valid_judgments) / len(valid_judgments),
    }


def generalization_summary(rows: list[dict]) -> dict:
    conflict_hits: list[bool] = []
    text_choices: list[bool] = []
    for row in rows:
        consistency = row.get("tasks", {}).get("consistency", {}).get("parsed")
        if consistency:
            value = str(consistency.get("consistency") or "").strip().lower()
            conflict_hits.append(value.startswith("conflict"))

        country = row.get("tasks", {}).get("country", {}).get("parsed")
        if country and str(country.get("trusted_source") or "").strip():
            value = str(country["trusted_source"]).strip().lower()
            text_choices.append(value.startswith("text"))

    return {
        "n_rows": len(rows),
        "n_consistency": len(conflict_hits),
        "conflict_recall_percent": 100.0 * sum(conflict_hits) / len(conflict_hits),
        "n_country_trust": len(text_choices),
        "text_dominance_percent": 100.0 * sum(text_choices) / len(text_choices),
    }


def compute_diagnostic_metrics(base_dir: Path) -> dict:
    output: dict[str, dict] = {}
    for model in MODELS:
        model_output: dict[str, dict] = {}
        for mode in ("probing", "defense"):
            per_dataset = {
                dataset: probing_summary(load_mode_rows(base_dir, dataset, model, mode)) for dataset in DATASETS
            }
            model_output[mode] = {
                "per_dataset": per_dataset,
                "macro_average": {
                    "n_valid_predictions": sum(item["n_valid_predictions"] for item in per_dataset.values()),
                    "wla": sum(item["wla"] for item in per_dataset.values()) / 4.0,
                    "median_error_km": sum(item["median_error_km"] for item in per_dataset.values()) / 4.0,
                    "cda_percent": sum(item["cda_percent"] for item in per_dataset.values()) / 4.0,
                },
            }

        generalization_rows = {
            dataset: load_mode_rows(base_dir, dataset, model, "generalization") for dataset in DATASETS
        }
        per_dataset_generalization = {
            dataset: generalization_summary(rows) for dataset, rows in generalization_rows.items()
        }
        model_output["generalization"] = {
            "per_dataset": per_dataset_generalization,
            "pooled": generalization_summary([row for rows in generalization_rows.values() for row in rows]),
        }
        output[model] = model_output
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=get_analysis_output_path("diagnostic_metrics.json"))
    args = parser.parse_args()

    payload = compute_diagnostic_metrics(args.base_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
