"""Shared vs. private population structure and connectivity-constrained dynamics.

Two experiments on Allen Institute open Neuropixels data:

1. ``latent`` — how much of the activity of two simultaneously recorded
   populations is shared between them vs. private to each, and which part
   encodes a continuously measured behavioral variable.
2. ``coupling`` — whether anatomical projection structure constrains the
   fitted inter-area coupling of a linear dynamical model.

Everything runs end-to-end on synthetic data (``acd demo``) so the pipeline is
testable without the multi-gigabyte NWB files.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("allen-circuit-dynamics")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
