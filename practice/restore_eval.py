"""Whole-exam ALC for budget-0 judgment sources with the 고정 갱신 공식, and with item 차이 되살리기:
    logit(final) = logit(formula) + k/(k+4) * (logit(target judgment) - mean logit of the pair's first k exemplar judgments)
No fitted values. Intervals: 4,000 draws resampling pairs, then items within pairs; differences from the first source's formula.
Two more run dirs (<dev_target_run_dir>:<dev_exemplar_run_dir>) take the deviation term from other judgments, so the
formula's level and the item differences can come from different judges.

Usage: python -I restore_eval.py <exam_dir> <name>=<target_run_dir>:<exemplar_run_dir>[:<dev_target_run_dir>:<dev_exemplar_run_dir>] ...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import logit
from eval_prior_combo import ALPHA, A0, M, counts
from exemplar_shift import load


def main():
    exam = load_exam(sys.argv[1])
    pairs = exam["pairs"]
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    bvec = np.array([c[0] for c in calls])
    draws = S.draws(4000)
    base, rows = None, []
    for arg in sys.argv[2:]:
        name, dirs = arg.split("=", 1)
        parts = dirs.split(":")
        p_t, p_e = load(parts[0]), load(parts[1])
        d_t, d_e = (load(parts[2]), load(parts[3])) if len(parts) == 4 else (p_t, p_e)
        f, dev = np.empty(len(calls)), np.zeros(len(calls))
        for i, (b, pi, k) in enumerate(calls):
            x, kk, s, n = cnt[(b, pi)]
            f[i] = (x + ALPHA * (s + A0 * p_t.get((pi, k), M)) / (n + A0)) / (kk + ALPHA)
            if b > 0:
                zbar = np.mean([logit(d_e.get((pi, a["key"]), M)) for a in pairs[pi]["acquisitions"][:b]])
                dev[i] = logit(d_t.get((pi, k), M)) - zbar
        q = 1 / (1 + np.exp(-(logit(f) + bvec / (bvec + 4.0) * dev)))
        if base is None:
            base = f
        for label, arr in ((f"{name} + 고정 갱신 공식", f), (f"{name} + 공식 + item 차이 되살리기", q)):
            alc, curve = S.alc(arr)
            bs = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
            print(f"{label:38} ALC {alc:.4f} | {[round(float(c), 3) for c in curve]} | vs first {alc - S.alc(base)[0]:+.4f} "
                  f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")


if __name__ == "__main__":
    main()
