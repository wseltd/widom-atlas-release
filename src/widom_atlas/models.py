"""Re-export shim for benchmark models (public alias used by external callers)."""

from widom_atlas.core.benchmark_models import (
    BenchmarkComparison,
    BenchmarkMaterial,
    BenchmarkRun,
)

__all__ = ["BenchmarkComparison", "BenchmarkMaterial", "BenchmarkRun"]
