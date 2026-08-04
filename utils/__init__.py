"""
utils — Shared utilities for SIGNPOST-Bench evaluation pipeline.

Eliminates duplicate code across evaluate.py, evaluate_probing.py,
and evaluate_generalization.py.
"""

from utils.data_loader import (
    encode_image,
    load_benchmark_meta,
    load_ground_truth,
    resolve_gt,
)
from utils.file_utils import get_base_id
from utils.parsers import (
    parse_defense_response,
    parse_json_response,
)

__all__ = [
    "load_ground_truth",
    "load_benchmark_meta",
    "resolve_gt",
    "encode_image",
    "get_base_id",
    "parse_json_response",
    "parse_defense_response",
]
