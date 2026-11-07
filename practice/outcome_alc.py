"""Whole-exam ALC when the outcome-aware judgments (outcome_judge.py, all targets) replace or join the current method at
some budgets, the current method (Opus low + 고정 갱신 공식 + item 차이 되살리기, as in restore_eval.py) everywhere else.

Rows: the current method; the outcome-aware judgment alone at the given budgets; the plain logit average of the two at
those budgets. A missing outcome-aware judgment falls back to the current method. Intervals: 4,000 draws resampling
pairs, then items within pairs, for each row minus the current method; the two proxy evaluations are also averaged.

Usage: python -I outcome_alc.py <budgets e.g. 1,3,7>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import logit
from common import BUDGETS, load_exam
from compare_runs import Sample
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load

ROOT = "./"
EXAMS = {"(10/20)": ("exams/rehearsal-02", "runs/r02-opus-low-fb", "runs/r02-exemplar-opus-low-fb", "runs/outcome-opus-low-r02"),
         "(10/26)": ("exams/proxy-confirm-1026", "runs/pc-sub-opus-low", "runs/pc-sub-exemplar-opus-low", "runs/outcome-opus-low-pc")}
sig = lambda z: 1 / (1 + np.exp(-z))


def arrays(exam_dir, t_dir, e_dir, o_dir, swap):
    exam = load_exam(ROOT + exam_dir)
    pairs = exam["pairs"]
    pt, pe = load(ROOT + t_dir), load(ROOT + e_dir)
    out = {(r["budget"], r["pair"], r["key"]): r["probability"]
           for r in map(json.loads, open(ROOT + o_dir + "/predictions.jsonl")) if r.get("probability") is not None}
    cnt = counts(exam)
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    cur, alone, avg, missing = np.empty(len(calls)), np.empty(len(calls)), np.empty(len(calls)), 0
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s, n = cnt[(b, pi)]
        p0 = pt.get((pi, k), M)
        f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (kk + ALPHA)
        if b > 0:
            zbar = np.mean([logit(pe.get((pi, a["key"]), M)) for a in pairs[pi]["acquisitions"][:b]])
            f = float(sig(logit(f) + b / (b + 4.0) * (logit(p0) - zbar)))
        cur[i] = alone[i] = avg[i] = f
        if b in swap:
            o = out.get((b, pi, k))
            if o is None:
                missing += 1
                continue
            alone[i] = o
            avg[i] = float(sig((logit(f) + logit(o)) / 2))
    return Sample(exam, calls), {"current method": cur, "outcome judge alone": alone, "average of the two": avg}, missing


def main():
    swap = {int(b) for b in sys.argv[1].split(",")}
    print(f"outcome-aware judgments used at budgets {sorted(swap)}; current method elsewhere")
    totals = {}
    for name, dirs in EXAMS.items():
        S, A, missing = arrays(*dirs, swap)
        draws = S.draws(4000)
        base = A["current method"]
        print(f"\n{name}: missing outcome-aware judgments {missing} (current method used)")
        for label, arr in A.items():
            alc, curve = S.alc(arr)
            d = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
            totals.setdefault(label, []).append((alc, alc - S.alc(base)[0]))
            iv = f" vs current {alc - S.alc(base)[0]:+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]" if label != "current method" else ""
            print(f"  {label:22} ALC {alc:.4f} | {[round(float(c), 4) for c in curve]}{iv}")
    print("\nmean of the two proxy evaluations")
    for label, v in totals.items():
        print(f"  {label:22} ALC {np.mean([a for a, _ in v]):.4f} vs current {np.mean([d for _, d in v]):+.4f}")


if __name__ == "__main__":
    main()
