"""I/O adapters for widom-atlas: AtlasInput model + array / npz / Widom-result constructors."""

from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.from_widom_result import from_widom_result
from widom_atlas.io.models import AtlasInput
from widom_atlas.io.npz import from_npz, save_samples_npz

__all__ = ["AtlasInput", "from_arrays", "from_npz", "from_widom_result", "save_samples_npz"]
