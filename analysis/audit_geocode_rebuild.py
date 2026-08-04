"""Audit a rebuilt geocode cache against taxonomy and attack sources."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analysis.rebuild_geocode_cache import DATASETS, normalize_target
from config import DATA_ROOT, get_analysis_output_path

POI_CATEGORIES = {"amenity", "shop", "office", "building"}
WEB_RE = re.compile(r"(?:https?://|www\.|\.(?:com|net|org|co|uk)\b)", re.I)
NUMERIC_RE = re.compile(r"^[\W_]*[\d\W_]+$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def audit_flags(target: str, value: dict[str, Any] | None) -> list[str]:
    if value is None:
        return ["no_result"]
    flags = []
    importance = value.get("importance")
    place_rank = value.get("place_rank")
    category = value.get("category")
    if place_rank == 30 and category in POI_CATEGORIES:
        flags.append("rank30_poi")
    if isinstance(importance, (int, float)) and importance < 0.001:
        flags.append("very_low_importance")
    if len(target) < 3:
        flags.append("very_short_query")
    if NUMERIC_RE.match(target):
        flags.append("numeric_query")
    if WEB_RE.search(target):
        flags.append("web_or_domain_query")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache", type=Path, default=get_analysis_output_path("geocode_cache_rebuild.json"))
    parser.add_argument("--output", type=Path, default=get_analysis_output_path("geocode_rebuild_audit.json"))
    args = parser.parse_args()

    cache = load_json(args.cache)
    target_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    taxonomy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attack_targets: dict[str, dict[str, set[str]]] = defaultdict(dict)

    for dataset in DATASETS:
        attacks_path = args.data_root / dataset / "attacks.jsonl"
        with attacks_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                base_id = str(row.get("original_filename") or "").strip()
                targets = {
                    normalize_target((item.get("attacks") or {}).get("adversarial")) for item in row.get("texts") or []
                }
                attack_targets[dataset][base_id] = {target for target in targets if target}

        taxonomy_path = args.data_root / dataset / "taxonomy_labels.jsonl"
        with taxonomy_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                base_id = str(row.get("base_id") or "").strip()
                target = normalize_target(row.get("adversarial_text"))
                if not base_id or not target:
                    continue
                taxonomy_rows[dataset].append({"base_id": base_id, "target": target})
                target_sources[target][dataset].add(base_id)

    mismatches = []
    for dataset, rows in taxonomy_rows.items():
        for row in rows:
            if row["target"] not in attack_targets[dataset].get(row["base_id"], set()):
                mismatches.append({"dataset": dataset, **row})

    records = {}
    flag_counts = Counter()
    for target in sorted(target_sources):
        value = cache.get(target)
        flags = audit_flags(target, value)
        flag_counts.update(flags)
        if flags:
            records[target] = {
                "flags": flags,
                "datasets": sorted(target_sources[target]),
                "sample_count": sum(len(base_ids) for base_ids in target_sources[target].values()),
                "result": value,
            }

    per_dataset = {}
    for dataset, rows in taxonomy_rows.items():
        resolved = sum(isinstance(cache.get(row["target"]), dict) for row in rows)
        no_result = sum(cache.get(row["target"]) is None for row in rows)
        review = sum(bool(audit_flags(row["target"], cache.get(row["target"]))) for row in rows)
        per_dataset[dataset] = {
            "groups": len(rows),
            "resolved_top1": resolved,
            "no_result": no_result,
            "groups_with_any_review_flag": review,
        }

    payload = {
        "schema_version": 1,
        "cache": str(args.cache),
        "canonical_unique_targets": len(target_sources),
        "taxonomy_groups": sum(len(rows) for rows in taxonomy_rows.values()),
        "taxonomy_attack_mismatch_count": len(mismatches),
        "taxonomy_attack_mismatches": mismatches,
        "cache_resolved": sum(isinstance(value, dict) for value in cache.values()),
        "cache_no_result": sum(value is None for value in cache.values()),
        "flag_counts_unique_targets": dict(sorted(flag_counts.items())),
        "per_dataset": per_dataset,
        "flagged_targets": records,
        "note": (
            "Flags identify records requiring review; they are not automatic exclusion "
            "rules and do not modify the cache."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "flagged_targets"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
