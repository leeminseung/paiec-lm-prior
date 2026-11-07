"""Score the round-1 predictor with its prior replaced by budget-0 LLM judgments, on a whole practice exam.

Round-1 predictor (submissions/r1-shrunk-peers/model.py):
    center = (S_others + A0 * prior) / (N_others + A0);  p = (x + ALPHA * center) / (k + ALPHA)
with prior = M for every item. Here prior = the item's budget-0 LLM probability p0 (M where a judgment is missing).
x, k: the pair's own revealed successes and results; S_others, N_others: other subjects' on the same benchmark.
Every evaluation item at every budget is scored (response-weighted within pair, pairs equal, competition weights).
Intervals: 4,000 draws resampling pairs, then items within each drawn pair.

Usage: python -I eval_prior_combo.py <exam_dir> <name>=<budget0_run_dir> [<name>=<run_dir> ...]
"""
import json
import os
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, brier_alc, load_exam
from compare_runs import Sample

M, ALPHA, A0 = 0.575, 4.0, 4.0


def counts(exam):
    """{(budget, pair): (x, k, S_others, N_others)} under the default random acquisition order."""
    out = {}
    for b in BUDGETS:
        own = {pi: (sum(a["label"] for a in p["acquisitions"][:b]), len(p["acquisitions"][:b]))
               for pi, p in enumerate(exam["pairs"])}
        for pi, p in enumerate(exam["pairs"]):
            others = [q for qi, q in enumerate(exam["pairs"]) if qi != pi and q["anon_benchmark_id"] == p["anon_benchmark_id"]
                      and q["subject"] != p["subject"]]
            S = sum(own[exam["pairs"].index(q)][0] for q in others)
            N = sum(own[exam["pairs"].index(q)][1] for q in others)
            out[(b, pi)] = (*own[pi], S, N)
    return out


def predictions(exam, cnt, prior):
    preds = {}
    for b in BUDGETS:
        for pi, p in enumerate(exam["pairs"]):
            x, k, S, N = cnt[(b, pi)]
            for t in p["targets"]:
                center = (S + A0 * prior(pi, t["key"])) / (N + A0)
                preds[(b, pi, t["key"])] = (x + ALPHA * center) / (k + ALPHA)
    return preds


def load_p0(run_dir):
    p0, info = {}, {"n": 0, "errors": 0, "cost": 0.0}
    for line in open(os.path.join(run_dir, "predictions.jsonl")):
        r = json.loads(line)
        if r["budget"] != 0:
            continue
        info["n"] += 1
        info["cost"] += r.get("cost_usd_equiv") or 0
        if r.get("probability") is None:
            info["errors"] += 1
        else:
            p0[(r["pair"], r["key"])] = r["probability"]
    return p0, info


def main():
    exam = load_exam(sys.argv[1])
    cnt = counts(exam)
    items = [(pi, t["key"]) for pi, p in enumerate(exam["pairs"]) for t in p["targets"]]
    sources = {"round 1 (prior 0.575 for every item)": (lambda pi, k: M, None)}
    for arg in sys.argv[2:]:
        name, run_dir = arg.split("=", 1)
        p0, info = load_p0(run_dir)
        missing = sum(1 for u in items if u not in p0)
        sources[name] = ((lambda d: (lambda pi, k: d.get((pi, k), M)))(p0), {**info, "missing": missing})
    calls = [(b, pi, k) for b in BUDGETS for pi, k in items]
    S = Sample(exam, calls)
    arrays = {}
    print(f"whole exam: {len(items)} evaluation items x 6 budgets, competition ALC (lower is better)")
    for name, (prior, info) in sources.items():
        P = predictions(exam, cnt, prior)
        df, alc = brier_alc(exam, P)
        arrays[name] = np.array([P[c] for c in calls])
        curve = [round(float(df[f"B{b}"].mean()), 3) for b in BUDGETS]
        extra = "" if info is None else (f" | judgments {info['n']}, failed {info['errors']}, missing {info['missing']}"
                                         f" (scored with 0.575), cost ${info['cost']:.2f}")
        print(f"  {name:48} ALC {alc:.4f} | by budget {curve}{extra}")
    draws = S.draws(4000)
    base = "round 1 (prior 0.575 for every item)"
    print("\ndifference from round 1 (negative = better) [95%]:")
    for name in arrays:
        if name == base:
            continue
        d = S.alc(arrays[name])[0] - S.alc(arrays[base])[0]
        bs = np.array([S.alc(arrays[name], dr)[0] - S.alc(arrays[base], dr)[0] for dr in draws])
        print(f"  {name:48} {d:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")


if __name__ == "__main__":
    main()
