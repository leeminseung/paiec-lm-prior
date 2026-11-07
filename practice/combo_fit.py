"""Combine a budget-0 item probability p with the revealed results: the fixed round-2 formula versus a fitted
combination, scored on the whole practice exam with one benchmark held out at a time.

Fixed (round 2):  center = (S + 4 p) / (N + 4);  f = (x + 4 center) / (k + 4)
Fitted:           budget 0:      logit q = a0 + a1 logit p              (calibration only; f = p there)
                  budget b > 0:  logit q = c0 + c1 logit f + c2 logit p   (the formula's answer, plus how much of the
                                                                          item's own judgment to keep)
x, k: the pair's revealed successes and results; S, N: other subjects' on the same benchmark.
The weights for each budget are fitted on the other benchmarks (Brier, responses weighted so each pair counts once)
and applied to the held-out benchmark, so no benchmark is scored with weights that saw it.
A source "mix:A+B" averages the logits of sources A and B (both judgments would be asked in the submission).

Usage: python -I combo_fit.py <exam_dir> <name>=<run_dir> [<name>=<run_dir> ...] [mix:<name>+<name> ...]
Missing judgments use 0.575. Intervals: 4,000 draws resampling pairs, then items within pairs.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from eval_prior_combo import ALPHA, A0, M, counts


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def load_p(run_dir):
    d = {}
    for line in open(os.path.join(run_dir, "predictions.jsonl")):
        r = json.loads(line)
        if r["budget"] == 0 and r.get("probability") is not None:
            d[(r["pair"], r["key"])] = r["probability"]
    return d


def fit(X, ns, nf, w):
    """Weights minimizing the weighted Brier of sigmoid(X @ theta), with a small pull toward theta0."""
    theta0 = np.zeros(X.shape[1])
    theta0[1] = 1.0

    def loss(th):
        q = 1 / (1 + np.exp(-(X @ th)))
        return float(np.sum(w * (ns * (q - 1) ** 2 + nf * q ** 2)) / np.sum(w * (ns + nf)) + 1e-4 * np.sum((th - theta0) ** 2))

    return minimize(loss, theta0, method="L-BFGS-B").x


def main():
    exam = load_exam(sys.argv[1])
    cnt = counts(exam)
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    S = Sample(exam, calls)
    bench = np.array([exam["pairs"][pi]["anon_benchmark_id"] for _, pi, _ in calls])
    bud = np.array([b for b, _, _ in calls])
    # each pair counts once per budget when fitting, as in the competition score
    resp = S.ns + S.nf
    pair_total = {(b, pi): 0.0 for b in BUDGETS for pi in range(len(exam["pairs"]))}
    for (b, pi, _), n in zip(calls, resp):
        pair_total[(b, pi)] += n
    w = np.array([1.0 / pair_total[(b, pi)] for b, pi, _ in calls])

    sources = {}
    for arg in sys.argv[2:]:
        if arg.startswith("mix:"):
            a, c = arg[4:].split("+")
            keys = set(sources[a]) | set(sources[c])
            sources[arg] = {u: 1 / (1 + np.exp(-(logit(sources[a].get(u, M)) + logit(sources[c].get(u, M))) / 2)) for u in keys}
        else:
            name, run_dir = arg.split("=", 1)
            sources[name] = load_p(run_dir)

    results = {}
    print(f"whole exam: {len(calls) // len(BUDGETS)} evaluation items x {len(BUDGETS)} budgets, competition ALC (lower is better)")
    for name, d in sources.items():
        p = np.array([d.get((pi, k), M) for _, pi, k in calls])
        f = np.empty(len(calls))
        for i, (b, pi, k) in enumerate(calls):
            x, kk, s_oth, n_oth = cnt[(b, pi)]
            center = (s_oth + A0 * p[i]) / (n_oth + A0)
            f[i] = (x + ALPHA * center) / (kk + ALPHA)
        q = np.empty(len(calls))
        weights = {}
        for held in sorted(set(bench)):
            for b in BUDGETS:
                train = (bench != held) & (bud == b)
                test = (bench == held) & (bud == b)
                X = np.column_stack([np.ones(len(calls)), logit(p)] if b == 0 else [np.ones(len(calls)), logit(f), logit(p)])
                th = fit(X[train], S.ns[train], S.nf[train], w[train])
                q[test] = 1 / (1 + np.exp(-(X[test] @ th)))
                weights.setdefault(b, []).append(th)
        results[(name, "fixed formula")] = f
        results[(name, "fitted, benchmark held out")] = q
        for kind, arr in (("fixed formula", f), ("fitted, benchmark held out", q)):
            alc, curve = S.alc(arr)
            print(f"  {name:22} {kind:27} ALC {alc:.4f} | by budget {[round(float(c), 3) for c in curve]}")
        print("    fitted weights (mean over held-out folds): " + "; ".join(
            f"b{b}: " + ", ".join(f"{v:+.2f}" for v in np.mean(weights[b], axis=0)) for b in BUDGETS))

    base_key = (next(iter(sources)), "fixed formula")
    print(f"\ndifference from {base_key[0]} with the fixed formula (negative = better) [95%]:")
    draws = S.draws(4000)
    base = results[base_key]
    for key, arr in results.items():
        if key == base_key:
            continue
        d = S.alc(arr)[0] - S.alc(base)[0]
        bs = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
        print(f"  {key[0]:22} {key[1]:27} {d:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")


if __name__ == "__main__":
    main()
