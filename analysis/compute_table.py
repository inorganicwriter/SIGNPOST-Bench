"""
Compute Table 1 data from dataset result folders.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.compute_results import analyze_results, normalize_dataset_display_name


def compute_table(datasets=None, base_dir=None, output_json=None, print_results=True):
    if datasets is None:
        datasets = ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]

    if base_dir is None:
        try:
            from config import DATA_ROOT

            base_dir = DATA_ROOT
        except ImportError:
            base_dir = Path.cwd() / "data"

    base_dir = Path(base_dir)

    all_results = {}
    for dataset_name in datasets:
        display_name = normalize_dataset_display_name(dataset_name)
        dataset_results_dir = base_dir / dataset_name / "results"
        dataset_results = analyze_results(
            dataset_results_dir,
            dataset_name=dataset_name,
        )
        if dataset_results:
            all_results[display_name] = dataset_results

    if print_results:
        print("=" * 108)
        print(
            f"{'Dataset':<12} {'Model':<25} {'Attack':<14} {'WLA(%)':<10} "
            f"{'MedErr(km)':<14} {'TBS(km)':<12} {'N':<6} {'TBS_pairs'}"
        )
        print("-" * 108)
        for dataset_name, dataset_results in all_results.items():
            for model_name, model_results in dataset_results.items():
                for attack_name, metrics in model_results.items():
                    tbs_str = f"{metrics['TBS']:.2f}" if metrics["TBS"] is not None else "N/A"
                    print(
                        f"{dataset_name:<12} {model_name:<25} {attack_name:<14} "
                        f"{metrics['WLA']:<10.2f} {metrics['MedErr']:<14.2f} "
                        f"{tbs_str:<12} {metrics['Count']:<6} {metrics.get('TBSPairs', 0)}"
                    )
        print("=" * 108)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        if print_results:
            print(f"\nSaved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compute SIGNPOST-Bench Table 1 data")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing dataset folders (default: from config.py)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include (default: all 4 datasets)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON output",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print the table to stdout",
    )
    args = parser.parse_args()

    output_json = args.output_json
    if output_json is None:
        try:
            from config import get_paper_output_path

            output_json = str(get_paper_output_path("table_data.json"))
        except ImportError:
            output_json = "paper/table_data.json"

    compute_table(
        datasets=args.datasets,
        base_dir=args.base_dir,
        output_json=output_json,
        print_results=not args.no_print,
    )


if __name__ == "__main__":
    main()
