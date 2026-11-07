"""How much does a budget-0 LLM judgment know about individual items, beyond the pair's overall success rate?

On a practice exam with known answers, for every pair:
  r        response-weighted success rate of the evaluation items
  floor    r(1-r): Brier of the oracle that knows r and predicts it for every item (no item-level knowledge)
  B0       Brier of the LLM's budget-0 probabilities p0
  shifted  Brier after moving every logit(p0) of the pair by one oracle constant (best possible level fix,
           item-to-item differences kept); below floor = the LLM ranks items within the pair usefully
  combo    Brier of the round-2 submission formula (prior p0, then counting) at budgets 0 and 31

Usage: python -I item_signal.py <exam_dir> <budget0_run_dir>
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, load_exam
from eval_prior_combo import ALPHA, A0, M, counts


def logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def main():
    exam = load_exam(sys.argv[1])
    p0 = {}
    for line in open(os.path.join(sys.argv[2], "predictions.jsonl")):
        r = json.loads(line)
        if r["budget"] == 0 and r.get("probability") is not None:
            p0[(r["pair"], r["key"])] = r["probability"]
    cnt = counts(exam)
    rows = []
    for pi, pair in enumerate(exam["pairs"]):
        ps, ys = [], []
        for t in pair["targets"]:
            for y in t["labels"]:
                ps.append(p0.get((pi, t["key"]), M))
                ys.append(y)
        ps, ys = np.array(ps), np.array(ys, float)
        r = ys.mean()
        z = np.array([logit(p) for p in ps])
        best = minimize_scalar(lambda d: np.mean((1 / (1 + np.exp(-(z + d))) - ys) ** 2), bounds=(-8, 8), method="bounded")
        combo = []
        for b in BUDGETS:
            x, k, S, N = cnt[(b, pi)]
            center = (S + A0 * ps) / (N + A0)
            combo.append(float(np.mean(((x + ALPHA * center) / (k + ALPHA) - ys) ** 2)))
        rows.append(dict(pair=pi, bench=pair["anon_benchmark_id"], n=len(ys), r=r, mean_p0=ps.mean(), floor=r * (1 - r),
                         B0=float(np.mean((ps - ys) ** 2)), shifted=float(best.fun), combo=combo))
    print(f"{'pair':>4} {'bench':>16} {'resp':>5} {'r':>5} {'mean p0':>7} | {'floor':>6} {'B0':>6} {'shifted':>7} |"
          f" {'combo B0':>8} {'combo B31':>9} {'combo ALC':>9}")
    for d in rows:
        alc = sum(w * v for w, v in zip(ALC_WEIGHTS, d["combo"]))
        print(f"{d['pair']:>4} {d['bench']:>16} {d['n']:>5} {d['r']:5.2f} {d['mean_p0']:7.2f} | {d['floor']:6.3f} {d['B0']:6.3f}"
              f" {d['shifted']:7.3f} | {d['combo'][0]:8.3f} {d['combo'][-1]:9.3f} {alc:9.3f}")
    mean = lambda key: np.mean([d[key] for d in rows])
    print(f"\nmean over pairs: floor {mean('floor'):.4f}  B0 {mean('B0'):.4f}  shifted {mean('shifted'):.4f}"
          f"  | combo by budget {[round(float(np.mean([d['combo'][i] for d in rows])), 4) for i in range(6)]}")
    print(f"pairs where B0 < combo B31: {sum(d['B0'] < d['combo'][-1] for d in rows)} of {len(rows)};"
          f" pairs where shifted < floor: {sum(d['shifted'] < d['floor'] for d in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
