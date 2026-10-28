#!/usr/bin/env bash
# Run inside WSL2 Ubuntu-22.04 on the Blade 18.
#   bash scripts/blade_bootstrap.sh            # env + package + demo
#   bash scripts/blade_bootstrap.sh --fetch    # also pull one Neuropixels session (~2 GB)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev,allen]"

echo "== smoke: both experiments on synthetic data =="
acd demo --out outputs/demo
pytest -q

if [[ "${1:-}" == "--fetch" ]]; then
  # Visual Coding Neuropixels, Brain Observatory 1.1 stimulus set.
  # 715093703 is a commonly used session with VISp + VISl + VISal + VISam + VISpm + VISrl probes.
  SESSION="${SESSION:-715093703}"
  AREAS="${AREAS:-VISp VISl VISal VISpm VISam VISrl}"
  echo "== fetching session $SESSION (areas: $AREAS) into data/ =="
  # shellcheck disable=SC2086
  acd fetch --session "$SESSION" --areas $AREAS --bin 0.05 --stimulus natural_movie_one \
      --cache data/ecephys_cache --out data/ecephys
  python - <<'PY'
from pathlib import Path
import numpy as np
from allen_circuit_dynamics.allen_data import projection_matrix
areas = tuple("VISp VISl VISal VISpm VISam VISrl".split())
conn = projection_matrix(areas, Path("data/ecephys_cache"))
np.save("data/conn_VIS6.npy", conn)
print("connectivity matrix saved -> data/conn_VIS6.npy")
print(np.round(conn, 2))
PY
  echo "== experiment 1 =="
  acd latent --data data/ecephys/${SESSION}_VISp_VISl.npz --out outputs/latent_${SESSION}
  echo "== experiment 2 =="
  acd coupling --data data/ecephys/${SESSION}_areas.npz --conn data/conn_VIS6.npy \
      --out outputs/coupling_${SESSION}
fi
echo "done."
