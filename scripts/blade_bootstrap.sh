#!/usr/bin/env bash
# Run inside WSL2 Ubuntu-22.04 on the Blade 18 (or any Linux box).
#   bash scripts/blade_bootstrap.sh            # main env + package + demo + tests
#   bash scripts/blade_bootstrap.sh --fetch    # also pull one Neuropixels session (~2 GB) and run both experiments
#
# allensdk pins old scipy/pandas/numpy, so it lives in its OWN venv (.venv-allen, Python 3.10) and is used
# only for `acd fetch` and the connectivity matrix. The main env (.venv) runs the experiments on the .npz files.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# A sourced ROS 2 setup puts /opt/ros/*/site-packages on PYTHONPATH, which leaks ROS pytest plugins
# (launch_testing -> lark) into our venvs. Our venvs are self-contained, so drop it.
unset PYTHONPATH

echo "== main env (.venv) =="
uv venv --clear --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

echo "== smoke: both experiments on synthetic data =="
.venv/bin/acd demo --out outputs/demo
.venv/bin/pytest -q

if [[ "${1:-}" == "--fetch" ]]; then
  echo "== allensdk env (.venv-allen, Python 3.10) =="
  uv venv --clear --python 3.10 .venv-allen
  uv pip install --python .venv-allen/bin/python -e ".[allen]"

  # Visual Coding Neuropixels, Brain Observatory 1.1 stimulus set.
  # 715093703 has probes in VISp, VISl, VISal, VISpm, VISam, VISrl.
  SESSION="${SESSION:-715093703}"
  AREAS="${AREAS:-VISp VISl VISal VISpm VISam VISrl}"
  echo "== fetching session $SESSION (areas: $AREAS) into data/ — this is the slow step =="
  # shellcheck disable=SC2086
  .venv-allen/bin/acd fetch --session "$SESSION" --areas $AREAS --bin 0.05 --stimulus natural_movie_one \
      --cache data/ecephys_cache --out data/ecephys

  echo "== connectivity matrix from the Mouse Connectivity Atlas =="
  .venv-allen/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from allen_circuit_dynamics.allen_data import projection_matrix
areas = tuple("VISp VISl VISal VISpm VISam VISrl".split())
conn = projection_matrix(areas, Path("data/ecephys_cache"))
np.save("data/conn_VIS6.npy", conn)
print("connectivity matrix saved -> data/conn_VIS6.npy")
print(np.round(conn, 2))
PY

  echo "== experiment 1 (main env) =="
  .venv/bin/acd latent --data "data/ecephys/${SESSION}_VISp_VISl.npz" --out "outputs/latent_${SESSION}"
  echo "== experiment 2 (main env) =="
  .venv/bin/acd coupling --data "data/ecephys/${SESSION}_areas.npz" --conn data/conn_VIS6.npy \
      --out "outputs/coupling_${SESSION}"
fi
echo "done."
