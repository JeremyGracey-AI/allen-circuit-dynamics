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
.venv/bin/acd demo --out outputs/demo_local   # outputs/demo/ is the committed reference run
.venv/bin/pytest -q

if [[ "${1:-}" == "--fetch" ]]; then
  echo "== allensdk env (.venv-allen, Python 3.10) =="
  uv venv --clear --python 3.10 .venv-allen
  uv pip install --python .venv-allen/bin/python -e ".[allen]"

  # Visual Coding Neuropixels, Brain Observatory 1.1 stimulus set.
  # 715093703 has probes in VISp, VISl, VISal, VISpm, VISam, VISrl.
  SESSION="${SESSION:-715093703}"
  # Areas are chosen from the session itself (top visual areas by QC-passing unit count) unless
  # AREAS is set. The two best-populated become Experiment 1's pair; all become Experiment 2's areas.
  AREAS="${AREAS:-}"
  # allensdk streams the 2.9 GB NWB through api.brain-map.org with a 1200 s timeout and no resume,
  # and on a failed fetch its cache DELETES whatever is at the cache path. So the resumable curl
  # download lives in data/nwb/ (never touched by allensdk) and is copied into the cache path
  # right before the cache runs.
  DL_DIR="data/nwb"
  DL="${DL_DIR}/session_${SESSION}.nwb"
  NWB_DIR="data/ecephys_cache/session_${SESSION}"
  NWB="${NWB_DIR}/session_${SESSION}.nwb"
  S3_URL="https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache/session_${SESSION}/session_${SESSION}.nwb"
  EXPECTED_BYTES=$(curl -sSI "$S3_URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  mkdir -p "$DL_DIR" "$NWB_DIR"
  # Salvage a partial download allensdk's path may hold from an earlier run (it never completes there).
  # Leftover single-stream partials can't seed the chunked download; clear them.
  for f in "$NWB" "$DL"; do
    if [[ -s "$f" && "$(stat -c %s "$f")" != "$EXPECTED_BYTES" ]]; then rm -f "$f"; fi
  done
  if [[ ! -s "$DL" || "$(stat -c %s "$DL")" != "$EXPECTED_BYTES" ]]; then
    echo "== downloading $DL from S3 (expected $EXPECTED_BYTES bytes; parallel ranged, resumable — rerun to resume) =="
    # Single-stream throughput to S3 is capped on some links (~200 kB/s per connection); 16 ranged
    # streams in parallel get the aggregate. Set PDL_PARTS to change the count.
    bash scripts/pdl.sh "$S3_URL" "$DL" "${PDL_PARTS:-16}"
  fi
  if [[ "$(stat -c %s "$DL")" != "$EXPECTED_BYTES" ]]; then
    echo "download incomplete ($(stat -c %s "$DL") of $EXPECTED_BYTES bytes); rerun to resume"; exit 1
  fi
  cp -f "$DL" "$NWB"
  ls -lh "$DL" "$NWB"
  # Fail fast if the NWB can't be opened with the pinned pynwb/hdmf, instead of letting the
  # allensdk cache misread a reader error as a missing file and start a multi-hour re-download.
  .venv-allen/bin/python - "$NWB" <<'PY'
import sys, pynwb, hdmf
with pynwb.NWBHDF5IO(sys.argv[1], "r", load_namespaces=True) as io:
    f = io.read()
print(f"NWB opens: pynwb {pynwb.__version__}, hdmf {hdmf.__version__}, session {f.identifier}")
PY

  echo "== building session tables / binning (areas: ${AREAS:-auto}) =="
  # shellcheck disable=SC2086
  .venv-allen/bin/acd fetch --session "$SESSION" --areas $AREAS --bin 0.05 --stimulus natural_movie_one \
      --cache data/ecephys_cache --out data/ecephys
  SESSION="$SESSION" bash scripts/run_experiments.sh
fi
echo "done."
