"""Rebuild the SIGNPOST adversarial-target geocode cache without overwriting it.

The target universe is the representative ``adversarial_text`` selected for
each benchmark group in ``data/<dataset>/taxonomy_labels.jsonl``.  Results are
written to a separate, resumable cache.  Genuine empty search results are
stored as ``null`` for compatibility; transport/API failures are recorded only
in the audit file and remain eligible for retry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DATASETS = ("im2gps3k", "yfcc4k", "googlesv", "baidusv")
DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "SIGNPOST-Bench-Geocode-Rebuild/1.0 (research benchmark)"
PUBLIC_NOMINATIM_HOST = "nominatim.openstreetmap.org"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_ROOT, get_analysis_output_path


def normalize_target(text: Any) -> str:
    return str(text or "").strip().lower()


def clean_query(text: str) -> str:
    return re.sub(r"[\"'()]", "", text.strip())


def load_target_manifest(data_root: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for dataset in DATASETS:
        taxonomy_path = data_root / dataset / "taxonomy_labels.jsonl"
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Missing taxonomy file: {taxonomy_path}")
        with taxonomy_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {taxonomy_path}:{line_number}: {exc}") from exc
                target = normalize_target(row.get("adversarial_text"))
                base_id = str(row.get("base_id") or "").strip()
                if not target or not base_id:
                    continue
                sources[target][dataset].append(base_id)

    for target, by_dataset in sorted(sources.items()):
        manifest[target] = {
            "query": clean_query(target),
            "datasets": sorted(by_dataset),
            "base_ids": {dataset: sorted(set(base_ids)) for dataset, base_ids in sorted(by_dataset.items())},
            "sample_count": sum(len(set(ids)) for ids in by_dataset.values()),
        }
    return manifest


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    # On Windows, virus scanners or a concurrent read can briefly hold the
    # destination open.  Retry the atomic replacement without rewriting or
    # discarding the completed temporary file.
    for attempt in range(1, 11):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            if attempt == 10:
                raise
            time.sleep(0.25 * attempt)


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.last_request_started: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_request_started is not None:
            remaining = self.min_interval_seconds - (now - self.last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started = time.monotonic()


def query_nominatim(
    text: str,
    endpoint: str,
    user_agent: str,
    limiter: RateLimiter,
    max_retries: int,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    query = clean_query(text)
    attempts: list[dict[str, Any]] = []
    if len(query) < 2:
        return "no_result", None, [{"reason": "query_too_short"}]

    url = f"{endpoint}?q={quote(query)}&format=jsonv2&limit=1&addressdetails=1&namedetails=0"
    for attempt in range(1, max_retries + 1):
        limiter.wait()
        started = time.time()
        request = Request(url, headers={"User-Agent": user_agent})
        try:
            with urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            elapsed = round(time.time() - started, 3)
            attempts.append({"attempt": attempt, "outcome": "ok", "seconds": elapsed})
            if not data:
                return "no_result", None, attempts
            top = data[0]
            result = {
                "lat": float(top["lat"]),
                "lon": float(top["lon"]),
                "display": top.get("display_name", ""),
                "category": top.get("category", top.get("class", "")),
                "type": top.get("type", ""),
                "importance": top.get("importance"),
                "place_rank": top.get("place_rank"),
                "osm_type": top.get("osm_type", ""),
                "osm_id": top.get("osm_id"),
                "address": top.get("address", {}),
                "boundingbox": top.get("boundingbox", []),
            }
            return "resolved", result, attempts
        except HTTPError as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "outcome": "http_error",
                    "code": exc.code,
                    "reason": str(exc.reason),
                }
            )
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable:
                break
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "outcome": "transport_error",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        if attempt < max_retries:
            time.sleep(min(5 * attempt, 15))
    return "error", None, attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=get_analysis_output_path("geocode_cache_rebuild.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=get_analysis_output_path("geocode_cache_rebuild_status.json"),
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("NOMINATIM_USER_AGENT", DEFAULT_USER_AGENT),
    )
    parser.add_argument("--min-interval", type=float, default=1.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if PUBLIC_NOMINATIM_HOST in args.endpoint and args.min_interval < 1.0:
        raise ValueError("Public Nominatim requires an interval of at least 1 second")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")

    manifest = load_target_manifest(args.data_root)
    cache = load_json_object(args.output)
    audit = load_json_object(args.status_output)
    statuses = audit.get("targets", {}) if isinstance(audit.get("targets"), dict) else {}

    target_keys = sorted(manifest)
    pending = [key for key in target_keys if key not in cache]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Canonical unique targets: {len(target_keys)}", flush=True)
    print(f"Already cached: {sum(key in cache for key in target_keys)}", flush=True)
    print(f"Pending this run: {len(pending)}", flush=True)
    print(f"Output: {args.output.resolve()}", flush=True)
    print(f"Audit: {args.status_output.resolve()}", flush=True)
    if args.dry_run:
        return

    limiter = RateLimiter(args.min_interval)
    run_started = time.time()
    for index, target in enumerate(pending, start=1):
        status, result, attempts = query_nominatim(
            target,
            endpoint=args.endpoint,
            user_agent=args.user_agent,
            limiter=limiter,
            max_retries=args.max_retries,
        )
        record = {
            **manifest[target],
            "status": status,
            "attempts": attempts,
            "queried_at_unix": time.time(),
        }
        if result is not None:
            record["result"] = result
            cache[target] = result
        elif status == "no_result":
            cache[target] = None
        statuses[target] = record

        if index % args.save_every == 0 or index == len(pending):
            audit_payload = {
                "schema_version": 1,
                "target_source": "data/<dataset>/taxonomy_labels.jsonl::adversarial_text",
                "normalization": "strip + lowercase; remove quotes/apostrophes/parentheses only from query",
                "endpoint": args.endpoint,
                "min_interval_seconds": args.min_interval,
                "canonical_target_count": len(target_keys),
                "cache_entry_count": len(cache),
                "targets": statuses,
            }
            atomic_write_json(args.output, cache)
            atomic_write_json(args.status_output, audit_payload)
            resolved = sum(isinstance(value, dict) for value in cache.values())
            no_result = sum(value is None for value in cache.values())
            errors = sum(item.get("status") == "error" for item in statuses.values() if isinstance(item, dict))
            elapsed = time.time() - run_started
            print(
                f"[{index}/{len(pending)}] cache={len(cache)} "
                f"resolved={resolved} no_result={no_result} errors={errors} "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )


if __name__ == "__main__":
    main()
