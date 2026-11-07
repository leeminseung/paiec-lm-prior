"""Compare budget-0 judgment sources on a practice exam (proxy evaluation) with known answers.

Per source, on the items every source answered (responses weighted within a pair, pairs weighted equally):
  level error     mean over pairs of |pair mean prediction - pair success rate|
  budget-0 Brier  Brier of the judgments themselves
  AUC             how well the within-pair ranking matches the outcomes (0.5 = no better than random)
  mean-matched    Brier after moving each pair by the oracle shift that fits its answers (ranking kept)
Then the whole-exam ALC with the 고정 갱신 공식, on all items (missing judgments scored with 0.575).
A source "mix:A+B" averages the logits of A and B.

Usage: python -I compare_judges.py <exam_dir> <name>=<run_dir> ... [mix:<name>+<name> ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import load_p, logit
from eval_prior_combo import ALPHA, A0, M, counts
from eval_solver_critic import auc, shifted_brier


def main():
    exam = load_exam(sys.argv[1])
    pairs = exam["pairs"]
    sources = {}
    for arg in sys.argv[2:]:
        if arg.startswith("mix:"):
            a, b = arg[4:].split("+")
            keys = set(sources[a]) & set(sources[b])
            sources[arg] = {u: float(1 / (1 + np.exp(-(logit(sources[a][u]) + logit(sources[b][u])) / 2))) for u in keys}
        else:
            name, run_dir = arg.split("=", 1)
            sources[name] = load_p(run_dir)
    items = [(pi, t["key"]) for pi, p in enumerate(pairs) for t in p["targets"]]
    common = [u for u in items if all(u in d for d in sources.values())]
    labels = {(pi, t["key"]): t["labels"] for pi, p in enumerate(pairs) for t in p["targets"]}
    print(f"items {len(items)}, answered by every source {len(common)}")
    print(f"{'source':34} {'level err':>9} {'B0':>7} {'AUC':>6} {'mean-matched':>12} | pair mean prediction (rate)")
    for name, d in sources.items():
        errs, b0s, aucs, mms, means = [], [], [], [], []
        for pi in range(len(pairs)):
            ps, ys = [], []
            for u in common:
                if u[0] == pi:
                    for y in labels[u]:
                        ps.append(d[u]); ys.append(y)
            ps, ys = np.array(ps), np.array(ys, float)
            errs.append(abs(ps.mean() - ys.mean()))
            b0s.append(float(np.mean((ps - ys) ** 2)))
            a = auc(ps, ys)
            if a is not None:
                aucs.append(a)
            mms.append(shifted_brier(ps, ys))
            means.append(f"{ps.mean():.2f}({ys.mean():.2f})")
        print(f"{name:34} {np.mean(errs):9.3f} {np.mean(b0s):7.4f} {np.mean(aucs):6.3f} {np.mean(mms):12.4f} | {' '.join(means)}")

    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    preds = {}
    for name, d in sources.items():
        arr = np.empty(len(calls))
        for i, (b, pi, k) in enumerate(calls):
            x, kk, s, n = cnt[(b, pi)]
            center = (s + A0 * d.get((pi, k), M)) / (n + A0)
            arr[i] = (x + ALPHA * center) / (kk + ALPHA)
        preds[name] = arr
    base = next(iter(preds))
    draws = S.draws(4000)
    print(f"\nwhole exam with the 고정 갱신 공식 (all {len(items)} items), difference from {base} [95%]")
    for name, arr in preds.items():
        alc, curve = S.alc(arr)
        line = f"  {name:34} ALC {alc:.4f} | by budget {[round(float(c), 3) for c in curve]}"
        if name != base:
            bs = np.array([S.alc(arr, dr)[0] - S.alc(preds[base], dr)[0] for dr in draws])
            line += f" | {alc - S.alc(preds[base])[0]:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]"
        print(line)


if __name__ == "__main__":
    main()
