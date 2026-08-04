"""Recompute taxonomy-label agreement from data/taxonomy_annotations.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.compute_human_agreement import cohens_kappa
from config import TAXONOMY_ANNOTATIONS_FILE, get_analysis_output_path


def agreement(rows: list[dict], first: str, second: str) -> dict:
    pairs = [
        (row.get(first, "").strip(), row.get(second, "").strip())
        for row in rows
        if row.get(first, "").strip() and row.get(second, "").strip()
    ]
    labels_first = [pair[0] for pair in pairs]
    labels_second = [pair[1] for pair in pairs]
    return {
        "n": len(pairs),
        "agreement_percent": 100.0 * sum(a == b for a, b in pairs) / len(pairs),
        "cohens_kappa": cohens_kappa(labels_first, labels_second),
    }


def compute(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "source": str(input_path),
        "rows": len(rows),
        "annotator1_vs_annotator2": agreement(rows, "annotator1_tier", "annotator2_tier"),
        "annotator1_vs_auto": agreement(rows, "annotator1_tier", "tier_auto"),
        "annotator2_vs_auto": agreement(rows, "annotator2_tier", "tier_auto"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TAXONOMY_ANNOTATIONS_FILE)
    parser.add_argument("--output", type=Path, default=get_analysis_output_path("taxonomy_agreement.json"))
    args = parser.parse_args()
    payload = compute(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
