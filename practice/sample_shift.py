"""Correct the judge's overall lean with every pair's exemplars, on a practice exam (proxy evaluation) with known answers.

At budget b >= 1 the predict input holds the first b few-shot exemplars of every pair in the sample (9 pairs -> 9b items).
Their budget-0 judgments z_e and revealed outcomes y_e give one shift for the whole sample:
    delta_b = argmax over d of  sum_e [ y_e log s(z_e + d) + (1 - y_e) log(1 - s(z_e + d)) ]  -  d^2 / (2 tau^2)
Every target judgment is moved by delta_b before the 고정 갱신 공식 (p' = s(z + delta_b)). item 차이 되살리기 is unchanged,
since a common shift cancels in (target logit - exemplar mean logit). Budget 0 has no exemplars, so delta_0 = 0.

Usage: python -I sample_shift.py <exam_dir> <target_run_dir> <exemplar_run_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import logit
from eval_prior_combo import ALPHA, A0, M, counts
from exemplar_shift import fit_shift, load

TAUS = [0.25, 0.5, 1.0, 2.0]


def main():
    exam = load_exam(sys.argv[1])
    p_t, p_e = load(sys.argv[2]), load(sys.argv[3])
    pairs = exam["pairs"]
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    bvec = np.array([c[0] for c in calls], float)
    z_t = np.array([logit(p_t.get((pi, k), M)) for _, pi, k in calls])
    dev = np.zeros(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        if b > 0:
            dev[i] = z_t[i] - np.mean([logit(p_e.get((pi, a["key"]), M)) for a in pairs[pi]["acquisitions"][:b]])

    def predict(delta):
        f = np.empty(len(calls))
        for i, (b, pi, k) in enumerate(calls):
            x, kk, s, n = cnt[(b, pi)]
            p = 1 / (1 + np.exp(-(z_t[i] + delta.get(b, 0.0))))
            f[i] = (x + ALPHA * (s + A0 * p) / (n + A0)) / (kk + ALPHA)
        return f, 1 / (1 + np.exp(-(logit(f) + bvec / (bvec + ALPHA) * dev)))

    draws = S.draws(4000)
    base_f, base_q = predict({})
    rows = [("no shift", {}, base_f, base_q)]
    for tau in TAUS:
        delta = {}
        for b in BUDGETS[1:]:
            ex = [(pi, a) for pi, p in enumerate(pairs) for a in p["acquisitions"][:b]]
            z = np.array([logit(p_e.get((pi, a["key"]), M)) for pi, a in ex])
            y = np.array([a["label"] for _, a in ex], float)
            delta[b] = fit_shift(z, y, tau)
        rows.append((f"tau {tau:g}", delta, *predict(delta)))
    for name, delta, f, q in rows:
        shifts = " ".join(f"{delta.get(b, 0.0):+.2f}" for b in BUDGETS[1:])
        for label, arr, base in (("formula", f, base_f), ("restore", q, base_q)):
            alc, curve = S.alc(arr)
            bs = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
            print(f"{name:9} {label:8} ALC {alc:.4f} | {[round(float(c), 3) for c in curve]} | vs no shift {alc - S.alc(base)[0]:+.4f} "
                  f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] | shift by budget {shifts}")


if __name__ == "__main__":
    main()
