"""Score judgments that saw the exemplar outcomes (outcome_judge.py) against the current method on the same targets.

Current method: Opus low budget-0 judgments + 고정 갱신 공식 + item 차이 되살리기, exactly as in restore_eval.py.
Rows per budget: the current method, the outcome-aware judgment as it is, the plain logit average of the two, the
formula without item 차이 되살리기, and the outcome-aware judgment averaged with that formula alone.
Brier on the chosen targets (responses weighted within a pair, pairs equal), the pair level error (mean prediction over the
chosen targets vs the pair success rate over all the pair's targets), and within-pair AUC. Intervals: 4,000 draws
resampling pairs, for each row minus the current method. Both proxy evaluations are also pooled (pairs equal).

Usage: python -I outcome_eval.py <budgets e.g. 1,3,7>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import logit
from common import load_exam
from embed_eval import auc
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load
from outcome_judge import chosen_targets
import json

ROOT = "./"
EXAMS = {"(10/20)": ("exams/rehearsal-02", "runs/r02-opus-low-fb", "runs/r02-exemplar-opus-low-fb", "runs/outcome-opus-low-r02"),
         "(10/26)": ("exams/proxy-confirm-1026", "runs/pc-sub-opus-low", "runs/pc-sub-exemplar-opus-low", "runs/outcome-opus-low-pc")}
sig = lambda z: 1 / (1 + np.exp(-z))
COLS = ("current", "outcome judge", "average", "formula only", "average with formula only")


def rows_for(exam_dir, t_dir, e_dir, o_dir, budgets):
    exam = load_exam(ROOT + exam_dir)
    pt, pe = load(ROOT + t_dir), load(ROOT + e_dir)
    cnt = counts(exam)
    chosen = chosen_targets(exam, 10)
    out = {}
    for r in map(json.loads, open(ROOT + o_dir + "/predictions.jsonl")):
        if r.get("probability") is not None:
            out[(r["budget"], r["pair"], r["key"])] = r["probability"]
    rows = []
    for b in budgets:
        for pi, pair in enumerate(exam["pairs"]):
            x, kk, s, n = cnt[(b, pi)]
            zbar = np.mean([logit(pe.get((pi, a["key"]), M)) for a in pair["acquisitions"][:b]]) if b > 0 else 0.0
            rate = np.mean([y for t in pair["targets"] for y in t["labels"]])
            for t in pair["targets"]:
                if t["key"] not in chosen[pi]:
                    continue
                p0 = pt.get((pi, t["key"]), M)
                f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (kk + ALPHA)
                cur = float(sig(logit(f) + (b / (b + 4.0)) * (logit(p0) - zbar))) if b > 0 else f
                new = out.get((b, pi, t["key"]))
                rows.append({"b": b, "pair": pi, "key": t["key"], "ns": sum(t["labels"]), "n": len(t["labels"]), "rate": rate,
                             "current": cur, "outcome judge": new if new is not None else cur, "missing": new is None,
                             "average": float(sig((logit(cur) + logit(new)) / 2)) if new is not None else cur,
                             "formula only": f, "average with formula only": float(sig((logit(f) + logit(new)) / 2)) if new is not None else f})
    return rows


def report(name, rows, budgets):
    print(f"\n{name}: {len({r['pair'] for r in rows})} pairs, missing outcome judgments {sum(r['missing'] for r in rows)} (current used)")
    rng = np.random.default_rng(0)
    for b in budgets:
        R = [r for r in rows if r["b"] == b]
        pairs = sorted({r["pair"] for r in R})
        per = {}
        for col in COLS:
            br, lvl, au = [], [], []
            for pi in pairs:
                P = [r for r in R if r["pair"] == pi]
                q = np.array([r[col] for r in P]); ns = np.array([r["ns"] for r in P], float); n = np.array([r["n"] for r in P], float)
                br.append(((ns * (q - 1) ** 2 + (n - ns) * q ** 2).sum() / n.sum()))
                lvl.append(abs(q.mean() - P[0]["rate"]))
                ys = np.concatenate([[1] * int(a) + [0] * int(c - a) for a, c in zip(ns, n)])
                v = auc(np.repeat(q, n.astype(int)), ys)
                if v is not None:
                    au.append(v)
            per[col] = (np.array(br), np.array(lvl), np.mean(au))
        line = []
        for col in COLS:
            br, lvl, au = per[col]
            d = br - per["current"][0]
            ds = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            iv = f" {d.mean():+.4f} [{np.quantile(ds, .025):+.4f}, {np.quantile(ds, .975):+.4f}]" if col != "current" else ""
            line.append(f"{col} {br.mean():.4f}{iv} lvl {lvl.mean():.3f} AUC {au:.3f}")
        print(f"  budget {b:>2}:\n    " + "\n    ".join(line))


def main():
    budgets = [int(b) for b in sys.argv[1].split(",")]
    allrows = []
    for name, dirs in EXAMS.items():
        rows = rows_for(*dirs, budgets)
        report(name, rows, budgets)
        offset = 100 if name == "(10/26)" else 0
        allrows += [{**r, "pair": r["pair"] + offset} for r in rows]
    report("both proxy evaluations", allrows, budgets)


if __name__ == "__main__":
    main()
