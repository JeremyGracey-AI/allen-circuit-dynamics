#!/usr/bin/env bash
# Post-fetch steps for one session: connectivity matrix (allensdk env) + both experiments (main env).
# Resumable on its own once `acd fetch` has written data/ecephys/<SESSION>_*.npz:
#   SESSION=715093703 bash scripts/run_experiments.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
unset PYTHONPATH
SESSION="${SESSION:-715093703}"

PAIR_NPZ=$(ls -t data/ecephys/${SESSION}_VIS*_VIS*.npz | head -1)
AREAS_NPZ="data/ecephys/${SESSION}_areas.npz"
CONN="data/conn_${SESSION}.npy"
echo "experiment-1 pair file: $PAIR_NPZ"

if [[ ! -s "$CONN" ]]; then
  echo "== connectivity matrix from the Mouse Connectivity Atlas =="
  .venv-allen/bin/python - "$SESSION" "$AREAS_NPZ" "$CONN" <<'PY'
import sys
from pathlib import Path
import numpy as np
from allen_circuit_dynamics.allen_data import projection_matrix
sess, areas_npz, out = sys.argv[1:4]
with np.load(areas_npz) as z:
    areas = tuple(str(a) for a in z["area_names"])
print("areas for connectivity:", areas)
conn = projection_matrix(areas, Path("data/ecephys_cache"))
np.save(out, conn)
print(f"connectivity matrix saved -> {out}")
print(np.round(conn, 2))
PY
else
  echo "== connectivity matrix already at $CONN =="
fi

echo "== experiment 1 (main env) =="
.venv/bin/acd latent --data "$PAIR_NPZ" --out "outputs/latent_${SESSION}"
echo "== experiment 2 (main env) =="
.venv/bin/acd coupling --data "$AREAS_NPZ" --conn "$CONN" --out "outputs/coupling_${SESSION}"
echo "done."
