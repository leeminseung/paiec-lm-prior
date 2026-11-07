"""Main results table: Brier ALC of every predictor on the three untouched proxy evaluations and on all five, with 95%
benchmark-bootstrap intervals for the difference from the non-LLM rule. No model calls.

Rows: constant 0.5, the organizers' Empirical Mean, the public J2 rule, the non-LLM rule (our count update with the
constant 0.575 as prior mean), the Opus judgment without update, Opus + count update, Opus + count update + item
offsets (ours), and the hierarchical logit model with the judgment as offset (hier_llm.py, SDs from measurement-db).

Usage: python -I report_tables.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import ROOT, logit, sig  # noqa: E402
sys.path.insert(0, "./practice")
from common import BUDGETS, load_exam  # noqa: E402
from eval_prior_combo import A0, ALPHA, M, counts  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import predict_all  # noqa: E402

W = np.array([.1, .2, .2, .2, .2, .1])
SPECS = [("(10/20)", "rehearsal-02", "r02-opus-low-fb", "r02-exemplar-opus-low-fb"),
         ("(10/26)", "proxy-confirm-1026", "pc-sub-opus-low", "pc-sub-exemplar-opus-low")] + \
        [(f"(10/28-{k})", f"proxy-1028-{k}", f"p1028-{k}-sub-opus-low", f"p1028-{k}-sub-exemplar-opus-low") for k in (1, 2, 3)]
ROWS = ["Constant 0.5", "Empirical Mean (organizers)", "J2 public rule", "Non-LLM rule", "Opus judgment, no update",
        "Opus + count update", "Opus + count update + item offsets (ours)", "Hierarchical logit model with Opus offsets"]
REF = "Non-LLM rule"


def pair_alc(label, ex, t, e):
    exam = load_exam(f"{ROOT}/exams/{ex}"); pt, pe = load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}")
    cnt = counts(exam)
    zt = {(pi, tt["key"]): logit(pt.get((pi, tt["key"]), M)) for pi, p in enumerate(exam["pairs"]) for tt in p["targets"]}
    ze = {(pi, a["key"]): logit(pe.get((pi, a["key"]), M)) for pi, p in enumerate(exam["pairs"]) for a in p["acquisitions"]}
    hier = predict_all(exam, zt, ze, 1.09, 0.46)
    out = {r: [] for r in ROWS}
    for pi, p in enumerate(exam["pairs"]):
        curve = {r: [] for r in ROWS}
        for b in BUDGETS:
            x, k, s, n = cnt[(b, pi)]
            zb = np.mean([logit(pe.get((pi, a["key"]), M)) for a in p["acquisitions"][:b]]) if b else 0.0
            se = {r: 0.0 for r in ROWS}; nn = 0
            for tt in p["targets"]:
                p0 = pt.get((pi, tt["key"]), M)
                f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (k + ALPHA)
                v = {"Constant 0.5": 0.5, "Empirical Mean (organizers)": x / k if k else 0.5,
                     "J2 public rule": (x + 2 * (s + 2) / (n + 4)) / (k + 2),
                     "Non-LLM rule": (x + ALPHA * (s + A0 * M) / (n + A0)) / (k + ALPHA),
                     "Opus judgment, no update": p0, "Opus + count update": f,
                     "Opus + count update + item offsets (ours)": float(sig(logit(f) + k / (k + 4.0) * (logit(p0) - zb))) if b else f,
                     "Hierarchical logit model with Opus offsets": hier[(b, pi, tt["key"])]}
                for r in ROWS:
                    se[r] += sum((v[r] - y) ** 2 for y in tt["labels"])
                nn += len(tt["labels"])
            for r in ROWS:
                curve[r].append(se[r] / nn)
        for r in ROWS:
            out[r].append(float(np.array(curve[r]) @ W))
    return out, [(label, p["benchmark_name"]) for p in exam["pairs"]]


def table(sel_labels, allp, clusters, title):
    idx_all = [i for i, c in enumerate(clusters) if c[0] in sel_labels]
    cl = [clusters[i] for i in idx_all]
    keys = sorted(set(cl)); members = {kk: [j for j, c in enumerate(cl) if c == kk] for kk in keys}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([members[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(4000)]
    ref = np.array([allp[REF][i] for i in idx_all])
    print(f"\n{title}: {len(idx_all)} pairs, {len(keys)} benchmarks")
    for r in ROWS:
        v = np.array([allp[r][i] for i in idx_all]); d = v - ref
        lo, hi = np.quantile([d[s].mean() for s in draws], [.025, .975])
        print(f"  {r:44} ALC {v.mean():.4f}  minus non-LLM {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main():
    allp, clusters = {r: [] for r in ROWS}, []
    for spec in SPECS:
        out, cl = pair_alc(*spec)
        print(spec[0], " | ".join(f"{r[:18]} {np.mean(out[r]):.4f}" for r in ROWS))
        for r in ROWS:
            allp[r] += out[r]
        clusters += cl
    table({"(10/28-1)", "(10/28-2)", "(10/28-3)"}, allp, clusters, "three untouched proxies")
    table({"(10/26)", "(10/28-1)", "(10/28-2)", "(10/28-3)"}, allp, clusters, "four non-development proxies")
    table({s[0] for s in SPECS}, allp, clusters, "all five proxies")


if __name__ == "__main__":
    main()
