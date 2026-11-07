"""Score "an LLM's budget-0 judgment + the shrinkage update" on a whole practice exam, without sampling.

For every evaluation item the run holds one budget-0 probability p0 (nothing revealed). At budget k with x successes
among the pair's k revealed results, the combination predicts (x + alpha * p0) / (k + alpha), alpha from the exam's
shrinkage.json (fitted on training data). Also scored: p0 at every budget with no update, and the baselines.
Every evaluation item at every budget is scored, response-weighted within a pair, pairs equal, competition weights.
Intervals (4,000 draws) resample pairs, then items within each drawn pair.

Usage: python -I combine_eval.py <exam_dir> <budget0_run_dir> [<earlier_run_dir_for_repeat_check>]
"""
import json
import os
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, brier_alc, load_exam
from compare_runs import Sample, load_run
from evaluate import baselines_for, run_python_predictor


def main():
    exam_dir, run_dir = sys.argv[1], sys.argv[2]
    exam = load_exam(exam_dir)
    fit = json.load(open(os.path.join(exam_dir, "shrinkage.json")))
    alpha, m = fit["alpha"], fit["m"]
    recs = load_run(run_dir)
    p0 = {(p, k): r["probability"] for (b, p, k), r in recs.items() if b == 0 and r.get("probability") is not None}
    all_items = [(pi, t["key"]) for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    missing = [u for u in all_items if u not in p0]
    print(f"budget-0 answers: {len(p0)} of {len(all_items)} evaluation items; missing {len(missing)} "
          f"(scored with the shrinkage prior m = {m})")
    for u in missing:
        p0[u] = m

    def revealed(pi, b):
        ys = [a["label"] for a in exam["pairs"][pi]["acquisitions"][:b]]
        return sum(ys), len(ys)

    calls = [(b, p, k) for b in BUDGETS for p, k in all_items]
    preds = {}
    preds["LLM budget-0 + shrinkage update"] = {c: (revealed(c[1], c[0])[0] + alpha * p0[(c[1], c[2])])
                                                  / (revealed(c[1], c[0])[1] + alpha) for c in calls}
    preds["LLM budget-0, no update"] = {c: p0[(c[1], c[2])] for c in calls}
    for name, fn in baselines_for(exam_dir).items():
        preds[f"baseline {name}"] = run_python_predictor(exam, fn)

    S = Sample(exam, calls)
    arrays = {name: np.array([P[c] for c in calls]) for name, P in preds.items()}
    print("\nwhole exam (every evaluation item, every budget), lower is better:")
    per_pair = {}
    for name, P in preds.items():
        df, alc = brier_alc(exam, P)
        per_pair[name] = df["ALC"].values
        curve = [round(float(df[f"B{b}"].mean()), 3) for b in BUDGETS]
        print(f"  {name:34} ALC {alc:.4f} | by budget {dict(zip(BUDGETS, curve))}")

    draws = S.draws(4000)
    main_name = "LLM budget-0 + shrinkage update"
    print("\ndifference (row minus column; negative = row better) [95%]:")
    for other in arrays:
        if other == main_name:
            continue
        d = S.alc(arrays[main_name])[0] - S.alc(arrays[other])[0]
        bs = np.array([S.alc(arrays[main_name], dr)[0] - S.alc(arrays[other], dr)[0] for dr in draws])
        print(f"  {main_name} - {other}: {d:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")

    print("\nper pair ALC:")
    names = [main_name, "baseline shrunk mean", "baseline empirical mean"]
    print("  " + " | ".join(f"{n[:30]:>30}" for n in names) + " | pair")
    for pi, pair in enumerate(exam["pairs"]):
        print("  " + " | ".join(f"{per_pair[n][pi]:30.4f}" for n in names)
              + f" | {pair['benchmark_name'][:24]} x {pair['subject']['normalized_name'][:26]}")

    if len(sys.argv) > 3:
        old = load_run(sys.argv[3])
        both = [(p, k) for (b, p, k), r in old.items() if b == 0 and (p, k) in p0 and (p, k) not in missing
                and r.get("probability") is not None]
        a = np.array([old[(0, p, k)]["probability"] for p, k in both])
        c = np.array([p0[u] for u in both])
        print(f"\nsame item asked twice at budget 0 ({len(both)} items): mean |difference| {np.mean(np.abs(a - c)):.3f}, "
              f"largest {np.max(np.abs(a - c)):.2f}, correlation {np.corrcoef(a, c)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
