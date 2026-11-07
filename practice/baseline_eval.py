"""Score predictors that need no LLM call on several proxy evaluations, through the competition interface.

Each predictor is called as predict(input, labeled) with input = [subject, item] and labeled = the first b acquisitions
of every pair in the exam (nested sets, as the platform supplies them), at budgets 0, 1, 3, 7, 15, 31, for every
evaluation item. Brier per pair and budget (responses weighted within a pair), ALC per pair with the competition
weights, pairs equal. Intervals for differences from the round-1 predictor resample benchmarks (all pairs of a drawn benchmark
together), across the chosen exams.

Predictors
- constant 0.5
- empirical mean: the organizers' baseline, the mean of the pair's revealed results, 0.5 when there are none
- J2: a public rule (IvanYamasaki): center = (other subjects' successes on the benchmark + 2) / (their results + 4),
  then (pair successes + 2 * center) / (pair results + 2)
- round 1: our first submission, center = (other successes + 4 * 0.575) / (other results + 4), then
  (pair successes + 4 * center) / (pair results + 4)
- any predictor module passed as name=module:factory (the factory returns predict(input, labeled))

Usage: python -I baseline_eval.py <exam_dir>[,<exam_dir>...] [name=module:factory ...]
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, labeled_at, load_exam

W = np.array(ALC_WEIGHTS)


def same_pair(a, b):
    return a == b


def split(labeled, subject, bench):
    own = [y for (s, it), y in labeled if it["benchmark_id"] == bench and same_pair(s, subject)]
    other = [y for (s, it), y in labeled if it["benchmark_id"] == bench and not same_pair(s, subject)]
    return own, other


def constant(inp, labeled):
    return 0.5


def empirical_mean(inp, labeled):
    own, _ = split(labeled, inp[0], inp[1]["benchmark_id"])
    return sum(own) / len(own) if own else 0.5


def shrink(m, a0, alpha):
    def predict(inp, labeled):
        own, other = split(labeled, inp[0], inp[1]["benchmark_id"])
        center = (sum(other) + a0 * m) / (len(other) + a0)
        return (sum(own) + alpha * center) / (len(own) + alpha)
    return predict


BUILTIN = {"constant 0.5": constant, "empirical mean": empirical_mean,
           "J2": shrink(0.5, 4, 2), "round 1": shrink(0.575, 4, 4)}


def pair_alcs(exam, predict):
    """ALC per pair (list in pair order) and Brier per pair and budget."""
    curves = np.zeros((len(exam["pairs"]), len(BUDGETS)))
    for bi, b in enumerate(BUDGETS):
        labeled = labeled_at(exam, b)
        for pi, pair in enumerate(exam["pairs"]):
            se, n = 0.0, 0
            for t in pair["targets"]:
                p = float(predict([pair["subject"], t["item"]], labeled))
                se += sum((p - y) ** 2 for y in t["labels"])
                n += len(t["labels"])
            curves[pi, bi] = se / n
    return curves @ W, curves


def main():
    exams = sys.argv[1].split(",")
    predictors = dict(BUILTIN)
    for arg in sys.argv[2:]:
        name, spec = arg.split("=", 1)
        mod, fac = spec.split(":")
        predictors[name] = getattr(importlib.import_module(mod), fac)()
    rows = {}  # name -> list of (exam, benchmark, pair ALC)
    for ed in exams:
        exam = load_exam(ed)
        print(f"\n{os.path.basename(ed.rstrip('/'))}: {len(exam['pairs'])} pairs, "
              f"{sum(len(p['targets']) for p in exam['pairs'])} evaluation items")
        for name, pred in predictors.items():
            alc, curves = pair_alcs(exam, pred)
            rows.setdefault(name, []).extend((ed, p["benchmark_name"], a) for p, a in zip(exam["pairs"], alc))
            print(f"  {name:22} ALC {alc.mean():.4f} | per budget {[round(float(c), 4) for c in curves.mean(axis=0)]}")
    if len(exams) > 1:
        names = list(predictors)
        clusters = sorted({(e, b) for e, b, _ in rows[names[0]]})
        print(f"\nall {len(exams)} exams pooled: {len(rows[names[0]])} pairs, {len(clusters)} benchmarks (pairs equal)")
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(clusters), len(clusters)) for _ in range(4000)]
        ref = "round 1"
        for name in names:
            per = {}
            for (e, b, a), (_, _, r) in zip(rows[name], rows[ref]):
                per.setdefault((e, b), []).append((a, r))
            keys = clusters
            sums = np.array([sum(a for a, _ in per[k]) for k in keys]); refs = np.array([sum(r for _, r in per[k]) for k in keys])
            cnt = np.array([len(per[k]) for k in keys])
            mean = sums.sum() / cnt.sum()
            diffs = [((sums[d] - refs[d]).sum() / cnt[d].sum()) for d in draws]
            print(f"  {name:22} ALC {mean:.4f} | minus {ref}: {mean - refs.sum() / cnt.sum():+.4f} "
                  f"[{np.quantile(diffs, .025):+.4f}, {np.quantile(diffs, .975):+.4f}]")


if __name__ == "__main__":
    main()
