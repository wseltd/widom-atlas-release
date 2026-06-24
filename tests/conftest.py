"""Top-level conftest — exposes ``tests/fixtures/real_structures`` as an importable namespace.

Pytest already adds the test root to ``sys.path``, so ``from
fixtures.real_structures import load_real_material`` is what real-material
integration tests use. This conftest makes that import work without
relying on package imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
