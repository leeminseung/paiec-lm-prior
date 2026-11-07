"""Ceilings for each way of raising the score, on the whole practice exam (oracles use the answers, so they are
upper bounds, not methods).

1. Budget-0 judgment as is (Sonnet alone, critic) with the round-2 formula.
2. Level fixed by an oracle: every judgment of a pair moved by one logit constant so the pair's budget-0 Brier is
   lowest (item differences kept). Shows what perfect "how good is this system on this benchmark" would give.
3. Level only: every item of a pair gets the pair's true success rate (no item differences).
4. Formula strength: the round-2 formula with other prior strengths (in-sample, so optimistic).

Usage: python -I headroom.py <exam_dir> <name>=<run_dir> ...
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import load_p, logit
from eval_prior_combo import M, counts


def formula(exam, cnt, calls, p, alpha=4.0, a0=4.0):
    out = np.empty(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s, n = cnt[(b, pi)]
        center = (s + a0 * p[i]) / (n + a0)
        out[i] = (x + alpha * center) / (kk + alpha)
    return out


def main():
    exam = load_exam(sys.argv[1])
    cnt = counts(exam)
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    S = Sample(exam, calls)
    labels = {(pi, t["key"]): t["labels"] for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]}
    rate = {pi: np.mean([y for t in pair["targets"] for y in t["labels"]]) for pi, pair in enumerate(exam["pairs"])}
    show = lambda name, arr: print(f"  {name:58} ALC {S.alc(arr)[0]:.4f} | by budget {[round(float(c), 3) for c in S.alc(arr)[1]]}")
    print("whole exam, round-2 formula unless noted")
    show("level only: every item gets its pair's true rate (oracle)", formula(exam, cnt, calls, np.array([rate[pi] for _, pi, _ in calls])))
    for arg in sys.argv[2:]:
        name, run_dir = arg.split("=", 1)
        d = load_p(run_dir)
        p = np.array([d.get((pi, k), M) for _, pi, k in calls])
        show(f"{name}: as is", formula(exam, cnt, calls, p))
        shifted = {}
        for pi in range(len(exam["pairs"])):
            keys = [t["key"] for t in exam["pairs"][pi]["targets"]]
            z = np.array([logit(d.get((pi, k), M)) for k in keys])
            ys = [np.array(labels[(pi, k)], float) for k in keys]

            def b0(delta):
                q = 1 / (1 + np.exp(-(z + delta)))
                return sum(float(np.sum((qi - y) ** 2)) for qi, y in zip(q, ys)) / sum(len(y) for y in ys)

            delta = minimize_scalar(b0, bounds=(-8, 8), method="bounded").x
            for k, zi in zip(keys, z):
                shifted[(pi, k)] = 1 / (1 + np.exp(-(zi + delta)))
        ps = np.array([shifted[(pi, k)] for _, pi, k in calls])
        show(f"{name}: level fixed by oracle, item differences kept", formula(exam, cnt, calls, ps))
        if name.startswith("sonnet"):
            for alpha in (2.0, 8.0, 16.0):
                show(f"{name}: formula strength {alpha:g} instead of 4 (in-sample)", formula(exam, cnt, calls, p, alpha=alpha))


if __name__ == "__main__":
    main()
