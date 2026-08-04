"""
export_human_validation.py
============================
Export probing results for human validation of CDA/MPR/RCS metrics.

Samples ~120 probing results stratified by attack_type, and outputs:
  1. A CSV for human annotation
  2. Image file paths for reference

Usage:
    python export_human_validation.py \
        --probing-results data/im2gps3k/results/gpt-5.4/probing_Adversarial_gpt-5.4.jsonl data/im2gps3k/results/gpt-5.4/probing_Random_gpt-5.4.jsonl \
        --img-dir /data/SIGNPOST-Bench/im2gps3k/images \
        --output human_validation/ \
        --total 120 \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Export probing results for human validation")
    parser.add_argument("--probing-results", nargs="+", required=True, help="One or more probing result JSONL files")
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="Base image directory (containing subdirs like Adversarial/, Random/, etc.)",
    )
    parser.add_argument("--output", type=str, default="human_validation/", help="Output directory")
    parser.add_argument("--total", type=int, default=120, help="Total samples to export")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to output directory for easy viewing")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_probing_results(paths):
    """Load all probing result JSONL files."""
    results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Infer attack type from filename or field
                    if "attack_type" not in entry:
                        if "Random" in path:
                            entry["attack_type"] = "random"
                        elif "Adversarial" in path:
                            entry["attack_type"] = "adversarial"
                        elif "Similar" in path:
                            entry["attack_type"] = "similar"
                        elif "Blank" in path:
                            entry["attack_type"] = "blank"
                        else:
                            entry["attack_type"] = "unknown"
                    entry["_source_file"] = os.path.basename(path)
                    results.append(entry)
                except json.JSONDecodeError:
                    continue
    return results


def stratified_sample(results, total, seed):
    """Stratified sampling: 40 Random, 40 Adversarial, 20 Similar, 20 Blank/Original."""
    random.seed(seed)

    by_type = {}
    for r in results:
        t = r.get("attack_type", "unknown").lower()
        by_type.setdefault(t, []).append(r)

    quotas = {
        "random": min(40, total * 40 // 120),
        "adversarial": min(40, total * 40 // 120),
        "similar": min(20, total * 20 // 120),
        "blank": min(10, total * 10 // 120),
        "original": min(10, total * 10 // 120),
    }

    sampled = []
    for attack_type, quota in quotas.items():
        pool = by_type.get(attack_type, [])
        n = min(quota, len(pool))
        if n > 0:
            sampled.extend(random.sample(pool, n))

    # Fill remaining from any type
    remaining = total - len(sampled)
    if remaining > 0:
        already = {r.get("filename") for r in sampled}
        extras = [r for r in results if r.get("filename") not in already]
        random.shuffle(extras)
        sampled.extend(extras[:remaining])

    random.shuffle(sampled)
    return sampled[:total]


def export_csv(samples, output_dir):
    """Export annotation CSV with pre-filled auto metrics and empty human columns."""
    csv_path = os.path.join(output_dir, "human_validation_sheet.csv")

    fieldnames = [
        "sample_id",
        "filename",
        "attack_type",
        "dataset",
        "model_name",
        # Auto-extracted fields (for annotator reference)
        "auto_consistent",
        "auto_trusted_source",
        "auto_visual_evidence",
        "auto_textual_evidence",
        "auto_consistency_explanation",
        "auto_final_prediction",
        "auto_error_km",
        # Human annotation columns (to fill)
        "human_conflict_presence",  # Conflict / No Conflict / Unclear
        "human_conflict_detected",  # Detected / Not Detected / Unclear
        "human_text_dominant",  # Text-Dominant / Not Text-Dominant / Unclear
        "human_notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(samples):
            # Determine dataset from source file or filename
            dataset = "unknown"
            src = sample.get("_source_file", "")
            for ds in ["im2gps3k", "yfcc4k", "googlesv", "baidusv"]:
                if ds in src.lower() or ds in sample.get("filename", "").lower():
                    dataset = ds
                    break

            # Truncate evidence lists for readability
            visual_ev = sample.get("visual_evidence", [])
            if isinstance(visual_ev, list):
                visual_ev = "; ".join(str(v) for v in visual_ev[:5])

            textual_ev = sample.get("textual_evidence", [])
            if isinstance(textual_ev, list):
                textual_ev = "; ".join(str(v) for v in textual_ev[:5])

            row = {
                "sample_id": i + 1,
                "filename": sample.get("filename", ""),
                "attack_type": sample.get("attack_type", ""),
                "dataset": dataset,
                "model_name": sample.get("_source_file", "").replace("probing_", "").replace(".jsonl", ""),
                "auto_consistent": sample.get("consistent"),
                "auto_trusted_source": sample.get("trusted_source", ""),
                "auto_visual_evidence": visual_ev,
                "auto_textual_evidence": textual_ev,
                "auto_consistency_explanation": sample.get("consistency_explanation", "")[:200],
                "auto_final_prediction": sample.get(
                    "final_prediction", f"({sample.get('pred_lat', '?')}, {sample.get('pred_lon', '?')})"
                ),
                "auto_error_km": f"{sample.get('error_km', 0):.1f}" if sample.get("error_km") else "",
                "human_conflict_presence": "",
                "human_conflict_detected": "",
                "human_text_dominant": "",
                "human_notes": "",
            }
            writer.writerow(row)

    print(f"Annotation CSV saved to: {csv_path}")
    return csv_path


def export_reference_json(samples, output_dir):
    """Export full probing data as JSON for detailed review."""
    json_path = os.path.join(output_dir, "human_validation_full.jsonl")
    with open(json_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Remove very long raw_response to keep file manageable
            export = {k: v for k, v in sample.items() if k != "_source_file"}
            if "raw_response" in export:
                export["raw_response"] = export["raw_response"][:500]
            f.write(json.dumps(export, ensure_ascii=False) + "\n")
    print(f"Full reference data saved to: {json_path}")


def copy_images(samples, img_dir, output_dir):
    """Copy referenced images to output directory for easy viewing."""
    img_out = os.path.join(output_dir, "images")
    os.makedirs(img_out, exist_ok=True)
    copied = 0
    for sample in samples:
        filename = sample.get("filename", "")
        attack_type = sample.get("attack_type", "").capitalize()
        # Try common subdirectory patterns
        candidates = [
            os.path.join(img_dir, attack_type, filename),
            os.path.join(img_dir, filename),
        ]
        for src in candidates:
            if os.path.exists(src):
                dst = os.path.join(img_out, f"{sample.get('attack_type', '')}_{filename}")
                shutil.copy2(src, dst)
                copied += 1
                break
    print(f"Copied {copied} images to {img_out}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load
    results = load_probing_results(args.probing_results)
    print(f"Loaded {len(results)} probing results from {len(args.probing_results)} files.")

    if not results:
        print("No results to process. Check --probing-results paths.")
        return

    # Sample
    samples = stratified_sample(results, args.total, args.seed)
    print(f"Sampled {len(samples)} for human validation.")

    by_type = {}
    for s in samples:
        t = s.get("attack_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")

    # Export
    export_csv(samples, args.output)
    export_reference_json(samples, args.output)

    if args.copy_images:
        copy_images(samples, args.img_dir, args.output)

    # Print instructions
    print(f"\n{'=' * 60}")
    print("  Human Validation Export Complete")
    print(f"{'=' * 60}")
    print(f"\nOutput directory: {args.output}")
    print("\nAnnotation instructions:")
    print("  1. Open human_validation_sheet.csv in Excel/Google Sheets")
    print("  2. For each row, view the image and auto-extracted fields")
    print("  3. Fill in three human judgment columns:")
    print("     - human_conflict_presence: Conflict / No Conflict / Unclear")
    print("     - human_conflict_detected: Detected / Not Detected / Unclear")
    print("     - human_text_dominant: Text-Dominant / Not Text-Dominant / Unclear")
    print("  4. Save the completed CSV")
    print("\nTo compute agreement metrics, run:")
    print(
        f"  python compute_human_agreement.py --annotations {os.path.join(args.output, 'human_validation_sheet.csv')}"
    )


if __name__ == "__main__":
    main()
