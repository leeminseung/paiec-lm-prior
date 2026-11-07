"""Write a run directory whose budget-0 judgments are the logit average of several runs' judgments.

An item missing from some runs (refusals) is averaged over the runs that have it. No fitted weights.

Usage: python -I mix_runs.py <out_run_dir> <run_dir> <run_dir> ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import load_p, logit


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    runs = [load_p(d) for d in dirs]
    keys = sorted(set().union(*runs))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "predictions.jsonl"), "w") as f:
        for pair, key in keys:
            z = [logit(r[(pair, key)]) for r in runs if (pair, key) in r]
            p = float(1 / (1 + np.exp(-np.mean(z))))
            f.write(json.dumps({"budget": 0, "pair": pair, "key": key, "probability": p, "n_sources": len(z)}) + "\n")
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump({"logit_average_of": dirs}, f, indent=1)
    print(f"{out}: {len(keys)} items from {len(dirs)} runs")


if __name__ == "__main__":
    main()
