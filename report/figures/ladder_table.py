"""Forecaster ladder for Section 4.3: the same hierarchical model (sigma_B 1.09, sigma_P 0.46, fitted on Opus forecast
errors and not refitted) with three forecasters, on the five proxy evaluations. No model calls.

Forecasters: Claude Opus 5.5 at low effort (the main one), DeepSeek V4.1 Flash at low effort (deepinfra fp8) and
GLM-5.3 Flash at low effort (gmicloud fp8), all with the same prompt. A forecast that is missing (refusal, no answer
within the token limit) falls back to 0.575, as in the submission.
Rows per forecaster: the forecast alone and the hierarchical model, ALC on the three held-out proxies and on all five,
with the difference from the Beta-binomial baseline and its 95% interval from 4,000 bootstrap draws over benchmarks.
Also the update's share (ours minus the forecast alone) with the same intervals, and each forecaster's budget-0 Brier skill score on the 847-item calibration subset where a run exists.

Usage: python -I ladder_table.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_ablations import TB, TS, offsets, pair_alc  # noqa: E402
from make_figures import ROOT  # noqa: E402
sys.path.insert(0, "./practice")
import memory_generalization as mg  # noqa: E402
from common import BUDGETS, load_exam  # noqa: E402
from eval_prior_combo import A0, ALPHA, M, counts  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import predict_all  # noqa: E402

PROXIES = [("validation 1", "rehearsal-02", "r02"), ("validation 2", "proxy-confirm-1026", "pc"),
           ("held-out 1", "proxy-1028-1", "p1028-1"), ("held-out 2", "proxy-1028-2", "p1028-2"),
           ("held-out 3", "proxy-1028-3", "p1028-3")]
OPUS = {"r02": ("r02-opus-low-fb", "r02-exemplar-opus-low-fb"), "pc": ("pc-sub-opus-low", "pc-sub-exemplar-opus-low")}
OPUS.update({f"p1028-{k}": (f"p1028-{k}-sub-opus-low", f"p1028-{k}-sub-exemplar-opus-low") for k in (1, 2, 3)})
FORECASTERS = {"Opus 5.5, low": lambda s: OPUS[s],
               "DeepSeek V4.1 Flash, low": lambda s: (f"or-{s}-dsv41flash-low-t", f"or-{s}-dsv41flash-low-e"),
               "GLM-5.3 Flash, low": lambda s: (f"or-{s}-glm53flash-low-t", f"or-{s}-glm53flash-low-e")}
CALIB = {"Opus 5.5, low": ("mdb-judge-opus-low", "full"), "DeepSeek V4.1 Flash, low": ("or-mdb-dsv41flash-low", "subset"),
         "GLM-5.3 Flash, low": ("or-mdb-glm53flash-low", "subset")}


def baseline(exam):
    cnt = counts(exam)
    return {(b, pi, t["key"]): (cnt[(b, pi)][0] + ALPHA * (cnt[(b, pi)][2] + A0 * M) / (cnt[(b, pi)][3] + A0))
            / (cnt[(b, pi)][1] + ALPHA) for b in BUDGETS for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}


def ci(d, clusters):
    keys = sorted(set(clusters)); idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rng = np.random.default_rng(0)
    bs = [np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(4000)]
    return np.quantile([d[s].mean() for s in bs], [.025, .975])


def main():
    rows, clusters, held = {}, [], []
    for label, ex, short in PROXIES:
        exam = load_exam(f"{ROOT}/exams/{ex}")
        rows.setdefault("baseline", []).extend(pair_alc(exam, baseline(exam)))
        for fc, runs in FORECASTERS.items():
            t, e = runs(short)
            pt, pe = load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}")
            n_t = sum(len(p["targets"]) for p in exam["pairs"]); n_e = sum(len(p["acquisitions"]) for p in exam["pairs"])
            miss = (n_t - len(pt), n_e - len(pe))
            zt, ze = offsets(exam, pt, pe)
            alone = {(b, pi, k): float(1 / (1 + np.exp(-z))) for b in BUDGETS for (pi, k), z in zt.items()}
            rows.setdefault((fc, "forecast alone"), []).extend(pair_alc(exam, alone))
            rows.setdefault((fc, "ours"), []).extend(pair_alc(exam, predict_all(exam, zt, ze, TB, TS)))
            print(f"{label:13} {fc:26} missing forecasts: targets {miss[0]}/{n_t}, revealed items {miss[1]}/{n_e}")
        clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
        held += [label.startswith("held-out")] * len(exam["pairs"])
    held = np.array(held)
    base = np.array(rows["baseline"])
    print(f"\nBeta-binomial baseline: held-out {base[held].mean():.4f}, five {base.mean():.4f}")
    for key, v in rows.items():
        if key == "baseline":
            continue
        v = np.array(v); d = v - base
        line = f"{key[0]:26} {key[1]:15}"
        for name, sel in (("held-out", held), ("five", np.ones_like(held))):
            cl = [c for c, s in zip(clusters, sel) if s]
            lo, hi = ci(d[sel], cl)
            line += f" | {name} ALC {v[sel].mean():.4f} minus baseline {d[sel].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(line)
    print("\nthe update's share: ours minus the forecast alone")
    for fc in FORECASTERS:
        d = np.array(rows[(fc, "ours")]) - np.array(rows[(fc, "forecast alone")])
        line = f"{fc:26}"
        for name, sel in (("held-out", held), ("five", np.ones_like(held))):
            lo, hi = ci(d[sel], [c for c, s in zip(clusters, sel) if s])
            line += f" | {name} {d[sel].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(line)
    print("\nbudget-0 Brier skill score on the 847-item calibration subset (forecaster at the same setting)")
    exam = load_exam(mg.SUBSET)
    for fc, (run, kind) in CALIB.items():
        s = mg.summarize(mg.pair_stats(exam, mg.judgments(run, kind, exam), lambda pi, t: True))
        print(f"  {fc:26} BSS {s['skill']:.3f}, Brier {s['brier']:.4f}, AUC {s['auc']:.3f}")


if __name__ == "__main__":
    main()
