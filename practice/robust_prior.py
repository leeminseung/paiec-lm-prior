"""Pair-specific prior strength through a robust mixture prior on the pair's success rate.

The 고정 갱신 공식 is a Beta-binomial posterior mean: the judgment (through the benchmark-level center) is a Beta prior
worth A = 4 exemplars. Its weakness (FINDINGS 9-74, 9-75): judge errors are heavy-tailed, so in the few pairs where
the judge is far off, 4 virtual exemplars hold the prediction back for many budgets. A robust prior (as in robust
meta-analytic-predictive priors for clinical trials, Schmidli et al. 2014) mixes the judge's Beta with a vague one:

    prior = w * Beta(A c, A (1 - c)) + (1 - w) * Beta(n2 m2, n2 (1 - m2))

c is the judge-based center; m2 = 0.627 (measurement-db mean pair success rate), n2 a small strength. After x successes
among k exemplars the weight on the judge becomes w' = w L1 / (w L1 + (1 - w) L2), with L the beta-binomial marginal
likelihoods of x, so the judge loses weight exactly when the revealed outcomes contradict it. The prediction is the
mixture's posterior mean; item 차이 되살리기 then adds back (1 - w' A / (k + A)) of the target's judgment logit's distance
from the exemplars' mean judgment logit (with w = 1 this is today's k / (k + 4)). w' uses the pair-level center (mean
judgment), the posterior mean uses the target's own judgment, as today.

Modes
  mdb <judge_set_dir> <judge_run_dir> <draws_per_pair>: grid over w and n2 on the measurement-db judge set, replaying
      random exemplar orders as alpha_mdb.py does (the exemplars' mean judgment is stood in for by the mean judgment of
      the pair's judged items; other subjects are left out). Intervals resample benchmarks.
  proxy <exam_dir> <target_run> <exemplar_run> <w> <n2>: whole-exam ALC against today's method with chosen values.
"""
import os
import random
import sys

import numpy as np
from scipy.special import betaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, MAX_LABELS, load_exam
from exemplar_shift import load

A, M2, M = 4.0, 0.627, 0.575
W = np.array(ALC_WEIGHTS)
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))


def judge_weight(x, k, c, w, n2):
    """Posterior weight on the judge component after x successes in k (arrays broadcast)."""
    a1, b1 = A * c, A * (1 - c)
    a2, b2 = n2 * M2, n2 * (1 - M2)
    l1 = betaln(a1 + x, b1 + k - x) - betaln(a1, b1)
    l2 = betaln(a2 + x, b2 + k - x) - betaln(a2, b2)
    return w / (w + (1 - w) * np.exp(np.clip(l2 - l1, -50, 50)))


def predict(x, k, c_target, c_pair, dev, w, n2):
    wp = judge_weight(x, k, c_pair, w, n2)
    m = wp * (x + A * c_target) / (k + A) + (1 - wp) * (x + n2 * M2) / (k + n2)
    return sig(logit(m) + (1 - wp * A / (k + A)) * dev)


def mdb(set_dir, run_dir, n_draws):
    from alpha_mdb import exemplar_pools
    exam = load_exam(set_dir); pj = load(run_dir); pairs = exam["pairs"]
    pools = exemplar_pools(exam)
    grid = [(1.0, 1.0)] + [(w, n2) for w in (0.95, 0.9, 0.8, 0.7, 0.5) for n2 in (1.0, 2.0)]
    per = {g: np.zeros(len(pairs)) for g in grid}
    curves = {g: np.zeros((len(pairs), len(BUDGETS))) for g in grid}
    for pi, p in enumerate(pairs):
        ids = sorted(pools[pi])
        xk = []
        for d in range(n_draws):
            rng = random.Random(f"alpha:{pi}:{d}")
            order = ids[:]; rng.shuffle(order)
            labels = [rng.choice(pools[pi][iid]) for iid in order[:MAX_LABELS]]
            xk.append([(sum(labels[:b]), len(labels[:b])) for b in BUDGETS])
        xk = np.array(xk, dtype=float)
        x, k = xk[:, :, 0][:, :, None], xk[:, :, 1][:, :, None]
        ts = p["targets"]
        p0 = np.array([pj.get((pi, t["key"]), M) for t in ts])
        z = logit(p0)
        dev = np.array([z[i] - np.delete(z, i).mean() for i in range(len(z))]) if len(z) > 1 else np.zeros(len(z))
        ns = np.array([sum(t["labels"]) for t in ts], dtype=float); nf = np.array([len(t["labels"]) for t in ts]) - ns
        cbar = float(sig(z.mean()))
        for g in grid:
            f = predict(x, k, p0, cbar, k / np.maximum(k, 1) * dev, *g)  # no restore at k = 0, as today
            brier = (ns * (f - 1) ** 2 + nf * f ** 2).sum(axis=2) / (ns + nf).sum()
            curves[g][pi] = brier.mean(axis=0)
            per[g][pi] = float(curves[g][pi] @ W)
    bench_of = np.array([p["benchmark_name"] for p in pairs]); benches = np.array(sorted(set(bench_of)))
    members = {b: np.where(bench_of == b)[0] for b in benches}
    rng = np.random.default_rng(0); boot = [rng.choice(len(benches), len(benches)) for _ in range(4000)]
    print(f"{len(pairs)} pairs, {len(benches)} benchmarks, {n_draws} exemplar orders per pair")
    for g in grid:
        diff = per[g] - per[(1.0, 1.0)]
        bs = [np.concatenate([diff[members[benches[j]]] for j in draw]).mean() for draw in boot]
        print(f"  w {g[0]:.2f} n2 {g[1]:.0f}: ALC {per[g].mean():.4f} | vs today {diff.mean():+.4f} "
              f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] | {[round(float(c), 4) for c in curves[g].mean(axis=0)]}")


def proxy(exam_dir, t_run, e_run, w, n2):
    from compare_runs import Sample
    from eval_prior_combo import counts
    exam = load_exam(exam_dir); pairs = exam["pairs"]; pt, pe = load(t_run), load(e_run)
    cnt = counts(exam)
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    S = Sample(exam, calls); draws = S.draws(4000)
    out = {"today": np.empty(len(calls)), f"robust w {w} n2 {n2}": np.empty(len(calls))}
    for i, (b, pi, key) in enumerate(calls):
        x, k, s, n = cnt[(b, pi)]
        p0 = pt.get((pi, key), M)
        ex = [logit(pe.get((pi, a["key"]), M)) for a in pairs[pi]["acquisitions"][:b]]
        dev = logit(p0) - np.mean(ex) if b > 0 else 0.0
        pbar = float(sig(np.mean(ex))) if b > 0 else p0
        c_t = (s + A * p0) / (n + A)
        c_p = (s + A * pbar) / (n + A)
        out["today"][i] = predict(x, k, c_t, c_p, dev, 1.0, 1.0)
        out[f"robust w {w} n2 {n2}"][i] = predict(x, k, c_t, c_p, dev, w, n2)
    base = out["today"]
    for name, arr in out.items():
        alc, curve = S.alc(arr)
        iv = ""
        if name != "today":
            d = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
            iv = f" | vs today {alc - S.alc(base)[0]:+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]"
        print(f"  {name:22} ALC {alc:.4f} | {[round(float(c), 3) for c in curve]}{iv}")


if __name__ == "__main__":
    if sys.argv[1] == "mdb":
        mdb(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        proxy(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]))
