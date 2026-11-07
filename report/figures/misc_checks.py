"""Two numbers of the report that had no saved script. No model calls.

1. Tails of the benchmark component of the forecast error (Section 5.2). On the calibration set (exams/mdb-judge-1027,
   Opus low forecasts), each pair's logit error is logit(mean forecast) - logit(success rate, all of the pair's
   responses, with a half-count correction); a benchmark's error is the mean over its pairs. Maximum-likelihood fits of
   a normal (mean and scale) and of a Student-t with three degrees of freedom (location and scale) to the benchmark
   errors, with their log-likelihoods.
2. The analyse-only instruction (Appendix E): correlation between Opus low forecasts with and without "do not solve",
   and the Brier of each, on the items of the 847-item subset that both answered.

Usage: python -I misc_checks.py
"""
import os
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, "./practice")
import memory_generalization as mg  # noqa: E402
from common import load_exam  # noqa: E402
from exemplar_shift import load  # noqa: E402

ROOT = "."
logit = lambda p: np.log(p / (1 - p))


def tails():
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
    pj = load(f"{ROOT}/runs/mdb-judge-opus-low")
    by = {}
    for pi, p in enumerate(exam["pairs"]):
        ps = [pj[(pi, t["key"])] for t in p["targets"] if (pi, t["key"]) in pj]
        if len(ps) < 3:
            continue
        r, n = p["pair_success_rate"], p["n_graded"]
        by.setdefault(p["benchmark_name"], []).append(logit(np.clip(np.mean(ps), .005, .995)) - logit((r * n + .5) / (n + 1)))
    e = np.array([np.mean(v) for v in by.values()])
    ll_norm = stats.norm.logpdf(e, e.mean(), e.std()).sum()
    nll = lambda th: -stats.t.logpdf(e, 3, th[0], np.exp(th[1])).sum()
    th = optimize.minimize(nll, [np.median(e), np.log(e.std())], method="Nelder-Mead").x
    print(f"benchmark mean logit errors: {len(e)} benchmarks, mean {e.mean():+.2f}, SD {e.std():.2f}")
    print(f"  normal log-likelihood {ll_norm:.1f}; Student-t (3 df) log-likelihood {-nll(th):.1f}, "
          f"location {th[0]:+.2f}, scale {np.exp(th[1]):.2f}")


def analyse_only():
    exam = load_exam(mg.SUBSET)
    a = mg.judgments("mdb-judge-opus-low", "full", exam)
    b = mg.judgments("onepair-sub-opus-5-5-low-analyse", "subset", exam)
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    lab = {(pi, t["key"]): t["labels"] for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}
    brier = lambda v: np.mean([np.mean([(q - l) ** 2 for l in lab[k]]) for k, q in zip(keys, v)])
    print(f"analyse-only: {len(keys)} items both answered, correlation {np.corrcoef(x, y)[0, 1]:.3f}, "
          f"mean |difference| {np.abs(x - y).mean():.3f}, item-mean Brier {brier(x):.3f} (normal) and {brier(y):.3f} (analyse-only)")


if __name__ == "__main__":
    tails()
    analyse_only()
