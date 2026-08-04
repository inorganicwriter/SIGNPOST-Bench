"""
utils/parsers.py — Shared response parsing functions.

Used by evaluate_probing.py and evaluate_generalization.py.
"""

import json
import re


def parse_json_response(text: str) -> dict | None:
    """Parse JSON from model response, handling thinking tags and code blocks.

    Returns parsed dict or None.
    """
    if not text:
        return None

    # Strip thinking tags if present
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Try 1: JSON in code block
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 2: Find JSON object directly
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try 3: Full text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_defense_response(text: str) -> dict:
    """Parse the 5-line structured defense prompt output.

    Expected format:
        Visual: Japan
        Text: Paris, France
        Conflict: Yes
        Trusted: Visual
        Final: (35.68, 139.76)

    Returns dict with keys: visual, text, conflict, trusted, final.
    """
    fields = {"visual": "", "text": "", "conflict": "", "trusted": "", "final": ""}
    if not text:
        return fields

    # Strip thinking tags
    if "</think>" in text:
        text = text.split("</think>")[-1]

    for line in text.strip().split("\n"):
        line = line.strip()
        lower = line.lower()
        if lower.startswith("visual:"):
            fields["visual"] = line.split(":", 1)[1].strip()
        elif lower.startswith("text:"):
            fields["text"] = line.split(":", 1)[1].strip()
        elif lower.startswith("conflict:"):
            fields["conflict"] = line.split(":", 1)[1].strip()
        elif lower.startswith("trusted:"):
            fields["trusted"] = line.split(":", 1)[1].strip()
        elif lower.startswith("final:"):
            fields["final"] = line.split(":", 1)[1].strip()

    # Fallback: if Final not found, try to find coordinates anywhere
    if not fields["final"]:
        coord_match = re.search(r"\(?\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)?", text)
        if coord_match:
            fields["final"] = coord_match.group(0)

    return fields


def normalize_trusted_source(raw: str) -> str:
    """Normalize trusted_source string to Visual/Textual/Both/Unknown."""
    if not raw:
        return "Unknown"
    lower = raw.strip().lower()
    if lower.startswith("visual"):
        return "Visual"
    elif lower.startswith("text"):
        return "Textual"
    elif lower.startswith("both"):
        return "Both"
    return "Unknown"


def normalize_consistent(value) -> bool | None:
    """Normalize consistent field to bool. Returns None if unparseable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return None
