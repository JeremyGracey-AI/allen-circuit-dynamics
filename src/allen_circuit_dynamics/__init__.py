"""Latent-variable models of simultaneously recorded neural populations.

One model underneath — a few shared latents, a few private latents, per-neuron
noise, optionally linear dynamics on the latents — scored on three output
contracts:

* **observations** (:func:`run_latent_experiment`): shared vs. private
  structure of two populations and which part carries a behavioral signal;
* **state** (:func:`run_coupling_experiment`): whether anatomical projection
  structure constrains the fitted inter-area coupling of a linear dynamical
  model;
* **action** (:func:`run_force_experiment`): whether the shared or the private
  subspace of two cell-type-defined populations predicts a motor output.

Everything runs end-to-end on synthetic data (``acd demo``) so the pipeline is
testable without the multi-gigabyte recordings. Not affiliated with the Allen
Institute; the data are theirs and their collaborators', the code is not.
"""

from importlib.metadata import PackageNotFoundError, version

from .dataset import PopulationDataset
from .experiments import (
    CouplingConfig,
    LatentConfig,
    run_coupling_experiment,
    run_latent_experiment,
)
from .experiments_force import ForceConfig, run_force_experiment
from .models.coupling import CoupledLDS
from .models.shared_private import SharedPrivateFA

try:
    __version__ = version("allen-circuit-dynamics")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = [
    "CoupledLDS",
    "CouplingConfig",
    "ForceConfig",
    "LatentConfig",
    "PopulationDataset",
    "SharedPrivateFA",
    "__version__",
    "run_coupling_experiment",
    "run_force_experiment",
    "run_latent_experiment",
]
