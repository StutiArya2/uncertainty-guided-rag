# Notebooks

Exploratory analysis and result plots.

`scripts/run_eval.py --json results.json` writes a machine-readable record of a run —
per-arm totals plus a per-question row carrying branch, token counts, uncertainty, and
support scores. That file is the intended input for anything in here.

```python
import json
results = json.load(open("results.json"))
for arm in results["arms"]:
    print(arm["mode"], arm["reduction"], arm["false_abstain_rate"])
    rows = arm["rows"]   # per-question detail
```

Plotting libraries (matplotlib, seaborn, pandas) are deliberately **not** in
`requirements.txt`. Per CLAUDE.md the runtime dependency set is kept minimal and additions
are discussed first — analysis-only tools do not belong in the dependencies needed to run
the pipeline. Install them into your environment directly:

```bash
pip install matplotlib pandas
```
