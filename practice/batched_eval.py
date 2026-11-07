"""Compare item-difference restoring with batched vs independent judgments on the sampled (target, budget) cases.
Usage: python -I batched_eval.py <exam_dir> <batched.jsonl> <indep_target_dir> <indep_exemplar_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_exam
from combo_fit import logit
from eval_prior_combo import ALPHA, A0, M, counts
from exemplar_shift import load

exam = load_exam(sys.argv[1]); cnt = counts(exam)
recs = [r for r in map(json.loads, open(sys.argv[2])) if r.get("probs")]
it, ie = load(sys.argv[3]), load(sys.argv[4])
lab = {(pi, t["key"]): t["labels"] for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}
sg = lambda z: 1 / (1 + np.exp(-z))
acc = {}
pt_b, pt_i = [], []
for r in recs:
    pi, key, b = r["pair"], r["key"], r["budget"]
    x, k, s, n = cnt[(b, pi)]
    for name, p, zbar in (("indep", it.get((pi, key), M), np.mean([logit(ie.get((pi, a["key"]), M)) for a in exam["pairs"][pi]["acquisitions"][:b]])),
                          ("batched", r["probs"][-1], np.mean([logit(v) for v in r["probs"][:-1]]))):
        f = (x + ALPHA * (s + A0 * p) / (n + A0)) / (k + ALPHA)
        q = sg(logit(f) + b / (b + 4) * (logit(p) - zbar))
        for y in lab[(pi, key)]:
            for kind, v in (("formula", f), ("restore", q)):
                acc.setdefault((name, kind, b, pi), []).append((v - y) ** 2)
    pt_b.append(r["probs"][-1]); pt_i.append(it.get((pi, key), M))
print(f"cases {len(recs)} | target judgment corr batched vs independent {np.corrcoef(pt_b, pt_i)[0,1]:.3f}, mean {np.mean(pt_b):.3f} vs {np.mean(pt_i):.3f}")
for b in sorted({r['budget'] for r in recs}):
    line = f"  budget {b:2}:"
    for name in ("indep", "batched"):
        for kind in ("formula", "restore"):
            vals = [np.mean(acc[(name, kind, b, pi)]) for pi in range(len(exam["pairs"])) if (name, kind, b, pi) in acc]
            line += f" {name}-{kind} {np.mean(vals):.4f} |"
    print(line)
