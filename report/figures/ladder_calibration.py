"""Does the hierarchical model need forecaster-specific constants? For a forecaster other than Opus, fit on its own
budget-0 forecasts of the 847-item calibration subset (one pair per benchmark, 121 benchmarks):
  - a logistic recalibration q' = sigmoid(a + b logit q), minimizing Brier with every benchmark weighted equally
    (judge_calib_eval.fit_calibration);
  - the total spread of its logit-scale pair error, tau^2 = var(logit mean forecast - logit success rate) minus the
    sampling noise of seven-item means. One pair per benchmark cannot split tau^2 into sigma_B and sigma_P, so both are
    scaled by tau_forecaster / tau_Opus on the same subset, keeping Opus's split (sigma_B 1.09, sigma_P 0.46).
Then ALC on the five proxies for: raw forecasts with Opus's sigmas (the ladder table), recalibrated forecasts with
Opus's sigmas, recalibrated forecasts with scaled sigmas, and the conditional form, in which the fitted intercept and
slope are divided by the model's own budget-0 shrinkage s = 1 / sqrt(1 + pi (sigma_B^2 + sigma_P^2) / 8) = 0.80, so that
the marginal recalibration is not applied on top of the shrinkage the model already makes. Differences from the Beta-binomial baseline with 95%
intervals from 4,000 bootstrap draws over benchmarks.

Usage: python -I ladder_calibration.py <forecaster> [<forecaster> ...]   (keys of SUBSET_RUNS below)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_ablations import TB, TS, offsets, pair_alc  # noqa: E402
from ladder_table import FORECASTERS, PROXIES, baseline, ci  # noqa: E402
from make_figures import ROOT, logit, sig  # noqa: E402
sys.path.insert(0, "./practice")
import memory_generalization as mg  # noqa: E402
from common import load_exam  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import predict_all  # noqa: E402
from judge_calib_eval import fit_calibration  # noqa: E402

SUBSET_RUNS = {"Opus 5.5, low": ("mdb-judge-opus-low", "full"),
               "DeepSeek V4.1 Flash, low": ("or-mdb-dsv41flash-low", "subset"),
               "GLM-5.3 Flash, low": ("or-mdb-glm53flash-low", "subset")}


def subset_fit(name):
    exam = load_exam(mg.SUBSET)
    run, kind = SUBSET_RUNS[name]
    pred = mg.judgments(run, kind, exam)
    z, ns, nf, w, per_bench = [], [], [], [], {}
    for pi, p in enumerate(exam["pairs"]):
        ts = [t for t in p["targets"] if (pi, t["key"]) in pred]
        for t in ts:
            z.append(logit(pred[(pi, t["key"])])); ns.append(sum(t["labels"])); nf.append(len(t["labels"]) - sum(t["labels"]))
            w.append(1 / (len(ts) * len(t["labels"])))
        if len(ts) >= 3:
            per_bench[pi] = ([pred[(pi, t["key"])] for t in ts], p["pair_success_rate"], p["n_graded"])
    a, b = fit_calibration(np.array(z), np.array(ns), np.array(nf), np.array(w))

    def tau(f):
        e, noise = [], []
        for ps, rate, n in per_bench.values():
            ps = np.array([f(q) for q in ps]); mp = ps.mean()
            e.append(logit(mp) - logit((rate * n + .5) / (n + 1)))
            noise.append((ps.var(ddof=1) / len(ps) + rate * (1 - rate) / n) / max(mp * (1 - mp), 1e-3) ** 2)
        return np.sqrt(max(np.var(e, ddof=1) - np.mean(noise), 1e-6)), float(np.mean(e))
    return (a, b), tau(lambda q: q), tau(lambda q: float(sig(a + b * logit(q))))


def main():
    opus_tau = subset_fit("Opus 5.5, low")[1][0]
    print(f"Opus 5.5, low on the subset: tau {opus_tau:.2f} (sigma_B 1.09, sigma_P 0.46 on the full calibration set)")
    for name in sys.argv[1:]:
        (a, b), (t_raw, m_raw), (t_cal, m_cal) = subset_fit(name)
        scale = t_cal / opus_tau
        print(f"\n{name}: recalibration a {a:+.2f}, b {b:.2f}; tau raw {t_raw:.2f} (mean error {m_raw:+.2f}), "
              f"recalibrated {t_cal:.2f} (mean error {m_cal:+.2f}); sigma scale {scale:.2f}")
        rows, clusters, held = {}, [], []
        for label, ex, short in PROXIES:
            exam = load_exam(f"{ROOT}/exams/{ex}")
            rows.setdefault("baseline", []).extend(pair_alc(exam, baseline(exam)))
            t, e = FORECASTERS[name](short)
            pt, pe = load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}")
            zt, ze = offsets(exam, pt, pe)
            ct = {k: float(a + b * v) for k, v in zt.items()}; ce = {k: float(a + b * v) for k, v in ze.items()}
            att = 1 / np.sqrt(1 + np.pi * (TB ** 2 + TS ** 2) / 8)  # the model's own shrinkage at budget 0
            kt = {k: float((a + b * v) / att) for k, v in zt.items()}; ke = {k: float((a + b * v) / att) for k, v in ze.items()}
            for row, args in (("raw, Opus sigmas", (zt, ze, TB, TS)), ("recalibrated, Opus sigmas", (ct, ce, TB, TS)),
                              ("recalibrated, scaled sigmas", (ct, ce, TB * scale, TS * scale)),
                              ("conditional slope a/s + (b/s) logit q", (kt, ke, TB, TS))):
                rows.setdefault(row, []).extend(pair_alc(exam, predict_all(exam, *args)))
            clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
            held += [label.startswith("held-out")] * len(exam["pairs"])
        held = np.array(held); base = np.array(rows["baseline"])
        for row, v in rows.items():
            if row == "baseline":
                continue
            v = np.array(v); d = v - base; line = f"  {row:28}"
            for nm, sel in (("held-out", held), ("five", np.ones_like(held))):
                lo, hi = ci(d[sel], [c for c, s in zip(clusters, sel) if s])
                line += f" | {nm} {v[sel].mean():.4f} ({d[sel].mean():+.4f} [{lo:+.4f}, {hi:+.4f}])"
            print(line)


if __name__ == "__main__":
    main()
