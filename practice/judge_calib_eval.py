"""Is the judge's level better than the base rate in general, and what blend of the two is best?

On the measurement-db judge set (make_judge_set.py, judgments from subscription_judge.py), with 5 benchmark folds:
  - budget-0 Brier on the judged items (responses weighted within a pair, pairs equal within a benchmark, benchmarks equal)
  - pair level: mean |pair mean prediction over the judged items - pair success rate over all the pair's responses|
for the base rate (training folds' mean pair success rate), the raw judge, and a calibrated judge
q = s(a + b * logit p) fitted on the training folds' items (each benchmark the same total weight).
Then the calibration fitted on all of the set is applied to both proxy evaluations' target and exemplar judgments, and
written as run folders for restore_eval.py (ALC with the 고정 갱신 공식 and item 차이 되살리기).

Usage: python -I judge_calib_eval.py <judge_set_dir> <judge_run_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

ROOT = "./"
PROXY_RUNS = {"r02": (ROOT + "runs/r02-opus-low-fb", ROOT + "runs/r02-exemplar-opus-low-fb"),
              "pc": (ROOT + "runs/pc-sub-opus-low", ROOT + "runs/pc-sub-exemplar-opus-low")}
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))


def fit_calibration(z, ns, nf, w):
    """a, b minimizing the weighted Brier of s(a + b z) over responses (ns successes, nf failures per item)."""
    def loss(th):
        q = sig(th[0] + th[1] * z)
        return np.sum(w * (ns * (q - 1) ** 2 + nf * q ** 2))
    return minimize(loss, np.array([0.0, 1.0]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9}).x


def main():
    set_dir, run_dir, out_dir = sys.argv[1:4]
    exam = load_exam(set_dir)
    pj = load(run_dir)
    rows = []
    for pi, pair in enumerate(exam["pairs"]):
        for t in pair["targets"]:
            if (pi, t["key"]) not in pj:
                continue
            rows.append({"pair": pi, "bench": pair["benchmark_name"], "p": pj[(pi, t["key"])], "ns": sum(t["labels"]),
                         "nf": len(t["labels"]) - sum(t["labels"]), "rate": pair["pair_success_rate"]})
    D = pd.DataFrame(rows)
    D["z"] = logit(D.p.values)
    # weights: responses within a pair, pairs equal within a benchmark, benchmarks equal
    D["n"] = D.ns + D.nf
    D["w"] = 1.0 / D.groupby("pair").n.transform("sum") / D.groupby("bench").pair.transform("nunique")
    D["w"] /= D.w.mean()
    print(f"judged items {len(D)} of {sum(len(p['targets']) for p in exam['pairs'])}, pairs {D.pair.nunique()}, benchmarks {D.bench.nunique()}")
    benches = np.array(sorted(D.bench.unique())); rng = np.random.default_rng(0); rng.shuffle(benches)
    folds = np.array_split(benches, 5)
    pred = {"base rate": np.empty(len(D)), "judge (raw)": D.p.values.copy(), "judge calibrated": np.empty(len(D))}
    coefs = []
    for f in folds:
        te = D.bench.isin(f).values; tr = ~te
        pr = D[tr].groupby("pair").rate.first(); pb = D[tr].groupby("pair").bench.first()
        base = pr.groupby(pb).mean().mean()  # benchmarks equal
        pred["base rate"][te] = base
        a, b = fit_calibration(D.z.values[tr], D.ns.values[tr], D.nf.values[tr], D.w.values[tr])
        coefs.append((round(a, 3), round(b, 3)))
        pred["judge calibrated"][te] = sig(a + b * D.z.values[te])
    print("calibration a, b per fold:", coefs)
    print("\nmeasurement-db, benchmarks held out:")
    for name, q in pred.items():
        br = (D.ns * (q - 1) ** 2 + D.nf * q ** 2)
        per_pair = br.groupby(D.pair).sum() / D.groupby("pair").n.sum()
        b0 = per_pair.groupby(D.groupby("pair").bench.first()).mean().mean()
        lvl = pd.Series(q).groupby(D.pair.values).mean()
        err = (lvl - D.groupby("pair").rate.first()).abs()
        e = err.groupby(D.groupby("pair").bench.first()).mean().mean()
        print(f"  {name:18} budget-0 Brier {b0:.4f} | pair level error {e:.3f}")

    a, b = fit_calibration(D.z.values, D.ns.values, D.nf.values, D.w.values)
    print(f"\ncalibration on the whole set: q = s({a:.3f} + {b:.3f} * logit p)  (p = 0.5 -> {sig(a):.3f}; p = 0.2 -> {sig(a + b * logit(0.2)):.3f}; p = 0.8 -> {sig(a + b * logit(0.8)):.3f})")
    for name, dirs in PROXY_RUNS.items():
        for role, d in zip(("t", "e"), dirs):
            os.makedirs(os.path.join(out_dir, f"{name}-{role}"), exist_ok=True)
            with open(os.path.join(out_dir, f"{name}-{role}", "predictions.jsonl"), "w") as f:
                for (pi, key), p in load(d).items():
                    f.write(json.dumps({"budget": 0, "pair": pi, "key": key, "probability": float(sig(a + b * logit(p)))}) + "\n")
    print(f"calibrated proxy judgments written under {out_dir}")


if __name__ == "__main__":
    main()
