"""Pytest configuration shared across all tests.

This file:
  1. Inserts the repo root at the front of ``sys.path`` so tests can import
     top-level packages (``config``, ``evaluation``, ``utils``, ``analysis``,
     ``data_collector``) without each test module having to do its own
     ``sys.path`` manipulation.
  2. Keeps ``unittest.TestCase``-style tests working unchanged (pytest
     discovers them natively).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
