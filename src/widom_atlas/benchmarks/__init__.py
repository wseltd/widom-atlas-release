"""Public-dataset benchmark + launch-validation layer.

Default unit tests (Layer 1) and real-material integration tests (Layer 2)
do not depend on this module. The benchmark runner here is the opt-in
Layer 3 — it downloads / loads real porous-material CIFs, runs the full
widom-atlas pipeline on each, and produces a launch-readiness report.
"""

from widom_atlas.benchmarks.download import (
    BenchmarkDataUnavailable,
    fetch_benchmark_material,
)
from widom_atlas.benchmarks.hashing import (
    ProvenanceMismatch,
    ProvenanceRecord,
    record_provenance,
    sha256_file,
)
from widom_atlas.benchmarks.launch_report import (
    LaunchReadinessReport,
    write_launch_report,
)
from widom_atlas.benchmarks.mofxdb import MOFXDBRecord, load_mofxdb_scalars
from widom_atlas.benchmarks.registry import (
    SMALL_BENCHMARK_SET,
    get_benchmark_set,
)
from widom_atlas.benchmarks.runner import run_benchmark_set
from widom_atlas.benchmarks.scalar_compare import (
    ScalarComparisonRow,
    ScalarComparisonTable,
    compare_scalars,
)

__all__ = [
    "SMALL_BENCHMARK_SET",
    "BenchmarkDataUnavailable",
    "LaunchReadinessReport",
    "MOFXDBRecord",
    "ProvenanceMismatch",
    "ProvenanceRecord",
    "ScalarComparisonRow",
    "ScalarComparisonTable",
    "compare_scalars",
    "fetch_benchmark_material",
    "get_benchmark_set",
    "load_mofxdb_scalars",
    "record_provenance",
    "run_benchmark_set",
    "sha256_file",
    "write_launch_report",
]
