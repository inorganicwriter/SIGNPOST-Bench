"""
compute_human_agreement.py
============================
Compute agreement between human annotations and auto-extracted probing metrics.

Reports:
  - CDA validation: human conflict_detected vs auto consistent
  - MPR validation: human text_dominant vs auto trusted_source
  - Cohen's kappa for inter-annotator agreement (if two annotator columns exist)

Usage:
    python compute_human_agreement.py \
        --annotations human_validation/human_validation_sheet.csv \
        --probing-data human_validation/human_validation_full.jsonl
"""

import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Compute human-auto agreement for probing metrics")
    parser.add_argument("--annotations", type=str, required=True, help="Path to completed annotation CSV")
    parser.add_argument(
        "--probing-data", type=str, default=None, help="Path to full probing JSONL (for additional context)"
    )
    parser.add_argument(
        "--annotator2", type=str, default=None, help="Path to second annotator CSV (for inter-annotator agreement)"
    )
    return parser.parse_args()


def load_annotations(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def cohens_kappa(labels1, labels2):
    """Compute Cohen's kappa for two lists of categorical labels."""
    assert len(labels1) == len(labels2), "Label lists must be same length"
    n = len(labels1)
    if n == 0:
        return 0.0

    categories = sorted(set(labels1) | set(labels2))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for l1, l2 in zip(labels1, labels2, strict=False):
        matrix[cat_idx[l1]][cat_idx[l2]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    pe = 0.0
    for i in range(k):
        row_sum = sum(matrix[i][j] for j in range(k)) / n
        col_sum = sum(matrix[j][i] for j in range(k)) / n
        pe += row_sum * col_sum

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def validate_cda(rows):
    """Compare human conflict_detected with auto consistent field."""
    print("\n--- CDA Validation ---")
    print("Comparing: human_conflict_detected vs auto_consistent")

    valid = []
    for row in rows:
        human = row.get("human_conflict_detected", "").strip()
        auto_consistent = row.get("auto_consistent", "").strip()

        if not human or human.lower() == "unclear":
            continue
        if auto_consistent == "":
            continue

        # Map: human "Detected" = auto consistent=False (model detected conflict)
        # Map: human "Not Detected" = auto consistent=True (model missed conflict)
        human_detected = human.lower().startswith("detected")

        auto_val = auto_consistent.lower()
        if auto_val in ("false", "no", "0"):
            auto_detected = True
        elif auto_val in ("true", "yes", "1"):
            auto_detected = False
        else:
            continue

        valid.append(
            {
                "filename": row.get("filename", ""),
                "human": human_detected,
                "auto": auto_detected,
                "agree": human_detected == auto_detected,
            }
        )

    if not valid:
        print("  No valid samples for CDA validation.")
        return

    agree = sum(1 for v in valid if v["agree"])
    total = len(valid)
    print(f"  Samples: {total}")
    print(f"  Agreement: {agree}/{total} = {agree / total * 100:.1f}%")

    # Kappa
    human_labels = ["detected" if v["human"] else "not_detected" for v in valid]
    auto_labels = ["detected" if v["auto"] else "not_detected" for v in valid]
    kappa = cohens_kappa(human_labels, auto_labels)
    print(f"  Cohen's κ (human vs auto): {kappa:.3f}")

    # Breakdown
    tp = sum(1 for v in valid if v["human"] and v["auto"])
    fp = sum(1 for v in valid if not v["human"] and v["auto"])
    fn = sum(1 for v in valid if v["human"] and not v["auto"])
    tn = sum(1 for v in valid if not v["human"] and not v["auto"])
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


def validate_mpr(rows):
    """Compare human text_dominant with auto trusted_source."""
    print("\n--- MPR Validation ---")
    print("Comparing: human_text_dominant vs auto_trusted_source")

    valid = []
    for row in rows:
        human = row.get("human_text_dominant", "").strip()
        auto_trusted = row.get("auto_trusted_source", "").strip()

        if not human or human.lower() == "unclear":
            continue
        if not auto_trusted:
            continue

        human_text_dom = human.lower().startswith("text")
        auto_text_dom = auto_trusted.lower().startswith("text")

        valid.append(
            {
                "filename": row.get("filename", ""),
                "human": human_text_dom,
                "auto": auto_text_dom,
                "agree": human_text_dom == auto_text_dom,
            }
        )

    if not valid:
        print("  No valid samples for MPR validation.")
        return

    agree = sum(1 for v in valid if v["agree"])
    total = len(valid)
    print(f"  Samples: {total}")
    print(f"  Agreement: {agree}/{total} = {agree / total * 100:.1f}%")

    human_labels = ["text_dom" if v["human"] else "not_text_dom" for v in valid]
    auto_labels = ["text_dom" if v["auto"] else "not_text_dom" for v in valid]
    kappa = cohens_kappa(human_labels, auto_labels)
    print(f"  Cohen's κ (human vs auto): {kappa:.3f}")


def validate_conflict_presence(rows):
    """Validate whether ground-truth conflict labels align with human judgment."""
    print("\n--- Conflict Presence Validation ---")
    print("Comparing: human_conflict_presence vs attack_type-based ground truth")

    valid = []
    for row in rows:
        human = row.get("human_conflict_presence", "").strip()
        attack_type = row.get("attack_type", "").lower()

        if not human or human.lower() == "unclear":
            continue

        # Ground truth: random/adversarial = Conflict, blank/original/similar = No Conflict
        gt_conflict = attack_type in ("random", "adversarial")
        human_conflict = human.lower().startswith("conflict")

        valid.append(
            {
                "filename": row.get("filename", ""),
                "human": human_conflict,
                "gt": gt_conflict,
                "agree": human_conflict == gt_conflict,
            }
        )

    if not valid:
        print("  No valid samples.")
        return

    agree = sum(1 for v in valid if v["agree"])
    total = len(valid)
    print(f"  Samples: {total}")
    print(f"  Agreement (human vs GT): {agree}/{total} = {agree / total * 100:.1f}%")

    # This tells us if our attack_type-based GT is reasonable
    human_labels = ["conflict" if v["human"] else "no_conflict" for v in valid]
    gt_labels = ["conflict" if v["gt"] else "no_conflict" for v in valid]
    kappa = cohens_kappa(human_labels, gt_labels)
    print(f"  Cohen's κ (human vs GT): {kappa:.3f}")


def inter_annotator_agreement(csv1_path, csv2_path):
    """Compute inter-annotator agreement between two annotators."""
    print("\n--- Inter-Annotator Agreement ---")

    rows1 = {r["filename"]: r for r in load_annotations(csv1_path)}
    rows2 = {r["filename"]: r for r in load_annotations(csv2_path)}

    common = set(rows1.keys()) & set(rows2.keys())
    print(f"  Common samples: {len(common)}")

    if not common:
        print("  No overlapping samples found.")
        return

    for field in ["human_conflict_presence", "human_conflict_detected", "human_text_dominant"]:
        labels1, labels2 = [], []
        for fname in common:
            v1 = rows1[fname].get(field, "").strip().lower()
            v2 = rows2[fname].get(field, "").strip().lower()
            if v1 and v2 and v1 != "unclear" and v2 != "unclear":
                labels1.append(v1)
                labels2.append(v2)

        if labels1:
            agree = sum(1 for a, b in zip(labels1, labels2, strict=False) if a == b)
            kappa = cohens_kappa(labels1, labels2)
            print(f"\n  {field}:")
            print(f"    Samples: {len(labels1)}")
            print(f"    Raw agreement: {agree}/{len(labels1)} = {agree / len(labels1) * 100:.1f}%")
            print(f"    Cohen's κ: {kappa:.3f}")


def main():
    args = parse_args()

    rows = load_annotations(args.annotations)
    print(f"Loaded {len(rows)} annotations from {args.annotations}")

    # Check if any human annotations exist
    has_annotations = any(
        row.get("human_conflict_presence", "").strip()
        or row.get("human_conflict_detected", "").strip()
        or row.get("human_text_dominant", "").strip()
        for row in rows
    )

    if not has_annotations:
        print("\n  No human annotations found in the CSV.")
        print("   Please fill in human_conflict_presence, human_conflict_detected,")
        print("   and human_text_dominant columns first.")
        return

    print(f"\n{'=' * 60}")
    print("  Human Validation Agreement Report")
    print(f"{'=' * 60}")

    validate_conflict_presence(rows)
    validate_cda(rows)
    validate_mpr(rows)

    if args.annotator2:
        inter_annotator_agreement(args.annotations, args.annotator2)

    print(f"\n{'=' * 60}")
    print("  Summary for Paper")
    print(f"{'=' * 60}")
    print("  Report the CDA and MPR agreement rates + Cohen's κ in §7 or Appendix.")
    print("  If inter-annotator κ > 0.7, state this as evidence of annotation quality.")


if __name__ == "__main__":
    main()
