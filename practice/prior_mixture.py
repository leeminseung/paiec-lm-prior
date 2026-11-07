"""Pair-adaptive trust in the judge: a mixture over prior strengths instead of the fixed strength 4 (report flow 5).

The current method reads as a nested Beta-binomial: the pair's success rate has a Beta prior with mean c (the judgment,
pulled toward other subjects' outcomes on the same benchmark) and strength A = 4, updated by the pair's own x successes
in k revealed exemplars, then item differences are restored with weight k/(k + A). Here A is uncertain: a prior over
A in a small grid (equal weights, fixed before looking at proxy results), and each pair's revealed outcomes reweight
the grid by the Beta-binomial marginal likelihood of x successes in k under prior mean c_pair (the mean exemplar
judgment, pulled the same way) and strength A. Outcomes that agree with the judge move weight to strong priors (trust
the judge), outcomes that disagree move it to weak ones. The prediction mixes the full current-method prediction
computed at each A (formula and restore) with those weights. Budget 0 is unchanged.

Rows: frozen judge (no update), fixed A = 1, 4 (current), 16, and mixtures. Scored as ALC on all five proxy
evaluations, pooled with a benchmark bootstrap against the current method; the three untouched proxies separately.

Usage: python -I prior_mixture.py
"""
import os
import sys

import numpy as np
from scipy.special import betaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, load_exam
from eval_prior_combo import A0, M, counts
from exemplar_shift import load

W = np.array(ALC_WEIGHTS)
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))
SPECS = [("(10/20)", "rehearsal-02", "r02-opus-low-fb", "r02-exemplar-opus-low-fb"),
         ("(10/26)", "proxy-confirm-1026", "pc-sub-opus-low", "pc-sub-exemplar-opus-low")] + \
        [(f"(10/28-{k})", f"proxy-1028-{k}", f"p1028-{k}-sub-opus-low", f"p1028-{k}-sub-exemplar-opus-low") for k in (1, 2, 3)]
ROOT = "."
UNTOUCHED = {"(10/28-1)", "(10/28-2)", "(10/28-3)"}


def at_strength(a, x, k, s, n, p0, zbar):
    c = (s + A0 * p0) / (n + A0)
    f = (x + a * c) / (k + a)
    return sig(logit(f) + k / (k + a) * (logit(p0) - zbar)) if k > 0 else f


def log_marginal(a, x, k, c):
    """log P(x successes in k | Beta(a c, a (1 - c)) prior), without the binomial coefficient (shared across a)."""
    c = min(max(c, 1e-3), 1 - 1e-3)
    return betaln(x + a * c, k - x + a * (1 - c)) - betaln(a * c, a * (1 - c))


def predict(grid, x, k, s, n, p0, zbar, pe_mean):
    if len(grid) == 1 or k == 0:
        return at_strength(grid[0] if len(grid) == 1 else 4.0, x, k, s, n, p0, zbar)
    c_pair = (s + A0 * pe_mean) / (n + A0)
    lw = np.array([log_marginal(a, x, k, c_pair) for a in grid])
    w = np.exp(lw - lw.max()); w /= w.sum()
    return sum(wi * at_strength(a, x, k, s, n, p0, zbar) for wi, a in zip(w, grid))


ROWS = {"frozen judge": None, "fixed 1": [1.0], "fixed 4 (current)": [4.0], "fixed 16": [16.0],
        "mixture 1, 4, 16": [1.0, 4.0, 16.0], "mixture 1, 4, 16, 64": [1.0, 4.0, 16.0, 64.0],
        "mixture 2, 4, 8": [2.0, 4.0, 8.0]}


def main():
    alc = {r: [] for r in ROWS}
    clusters, labels_of = [], []
    for label, ex, t, e in SPECS:
        exam = load_exam(f"{ROOT}/exams/{ex}"); pt = load(f"{ROOT}/runs/{t}"); pe = load(f"{ROOT}/runs/{e}")
        cnt = counts(exam)
        per = {r: [] for r in ROWS}
        for pi, p in enumerate(exam["pairs"]):
            p0 = np.array([pt.get((pi, tt["key"]), M) for tt in p["targets"]])

            def brier(v):
                return sum(sum((v[j] - y) ** 2 for y in tt["labels"]) for j, tt in enumerate(p["targets"])) / \
                    sum(len(tt["labels"]) for tt in p["targets"])
            curves = {r: [] for r in ROWS}
            for b in BUDGETS:
                x, k, s, n = cnt[(b, pi)]
                acq = p["acquisitions"][:b]
                pes = np.array([pe.get((pi, a["key"]), M) for a in acq])
                zbar = float(np.mean(logit(pes))) if b else 0.0
                pe_mean = float(pes.mean()) if b else 0.0
                for r, grid in ROWS.items():
                    v = p0 if grid is None else predict(grid, x, k, s, n, p0, zbar, pe_mean)
                    curves[r].append(brier(v))
            for r in ROWS:
                per[r].append(float(np.array(curves[r]) @ W))
            clusters.append((label, p["benchmark_name"])); labels_of.append(label)
        for r in ROWS:
            alc[r] += per[r]
        print(f"{label:10} " + " | ".join(f"{r} {np.mean(per[r]):.4f}" for r in ROWS), flush=True)
    keys = sorted(set(clusters))
    idx = {kk: [i for i, c in enumerate(clusters) if c == kk] for kk in keys}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(4000)]
    ref = np.array(alc["fixed 4 (current)"])
    unt = np.array([lb in UNTOUCHED for lb in labels_of])
    print(f"pooled: {len(ref)} pairs, {len(keys)} benchmark clusters; untouched three: {unt.sum()} pairs")
    for r in ROWS:
        v = np.array(alc[r]); d = v - ref
        bs = [np.concatenate([d[idx[keys[j]]] for j in dr]).mean() for dr in draws]
        print(f"  {r:22} ALC five {v.mean():.4f} minus current {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, "
              f"{np.quantile(bs, .975):+.4f}] | untouched three {v[unt].mean():.4f} ({d[unt].mean():+.4f})")


if __name__ == "__main__":
    main()
