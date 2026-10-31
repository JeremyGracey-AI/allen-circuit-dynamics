"""Regenerate Experiment 3 figures (and the summary block) from saved metrics.json files.
usage: .venv/bin/python scripts/replot_force.py outputs/force_klee
"""

import json
import sys
from pathlib import Path

from allen_circuit_dynamics.experiments_force import summarize_context
from allen_circuit_dynamics.report import plot_force

root = Path(sys.argv[1])
for mp in sorted(root.glob("*/metrics.json")):
    m = json.loads(mp.read_text())
    m["summary"] = {c: summarize_context(v) for c, v in m["per_context"].items() if v}
    mp.write_text(json.dumps(m, indent=2, default=float))
    figs = plot_force(m, mp.parent)
    print(mp.parent.name, "->", len(figs), "figures")
