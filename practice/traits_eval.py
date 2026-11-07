"""Does an LLM's reading of item properties predict benchmark difficulty on benchmarks never seen?

Uses item_traits.py output. Item target z = logit((s + 0.5) / (n + 1)). Weighted ridge (every benchmark the same total
weight) on item features; a held-out benchmark's pair mean prediction is the mean predicted probability over its items,
scored against each of its pairs' success rates (benchmarks weighted equally), with 5 benchmark folds as in embed_eval.py.
Feature sets: traits without the success guess, the success guess alone (its logit), and both.
Then both proxy evaluations with the model fit on all measurement-db items, next to the Opus low judge and the base rate,
and a plain logit average of the judge's and the traits model's pair mean prediction (no fitted weight).

Usage: python -I traits_eval.py <run_dir> [ridge_lambda]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from embed_eval import CATALOG, auc, ridge, sig
from exemplar_shift import load
from item_traits import DOMAINS, FORMATS, GRADING

ROOT = "./"
EXAMS = {"proxy-10/20": (ROOT + "exams/rehearsal-02", ROOT + "runs/r02-opus-low-fb"),
         "proxy-10/26": (ROOT + "exams/proxy-confirm-1026", ROOT + "runs/pc-sub-opus-low")}
logit = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))


def features(t, which):
    cols = {}
    if which in ("traits", "both"):
        for f in FORMATS:
            cols[f"format={f}"] = float(t["answer_format"] == f)
        for g in GRADING:
            cols[f"grading={g}"] = float(t["grading"] == g)
        for d in DOMAINS:
            cols[f"domain={d}"] = float(t["domain"] == d)
        n_opt = t["n_options"] if t["answer_format"] in ("multiple_choice", "yes_no") and t["n_options"] > 0 else 0
        cols["chance"] = 1.0 / n_opt if n_opt else 0.0
        cols["all_parts_required"] = float(t["all_parts_required"])
        cols["english"] = float(t["english"])
        cols["needs_unseen_media"] = float(t["needs_unseen_media"])
        cols["steps"] = float(min(max(t["steps"], 1), 5))
    if which in ("guess", "both"):
        cols["logit_p_strong_model"] = float(logit(t["p_strong_model"]))
    return cols


def matrix(rows, which):
    df = pd.DataFrame([features(r["traits"], which) for r in rows]).fillna(0.0)
    return df


def main():
    run_dir = sys.argv[1]
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "traits.jsonl"))]
    rows = [r for r in rows if r.get("traits")]
    md = [r for r in rows if r["source"] == "measurement-db"]
    M = pd.DataFrame([{k: r[k] for k in ("benchmark", "item_id", "n", "s")} for r in md])
    M["z"] = np.log((M.s + 0.5) / (M.n - M.s + 0.5))
    M["w"] = 1.0 / M.groupby("benchmark").z.transform("count"); M["w"] *= len(M) / M.w.sum()
    ps = pd.read_parquet(CATALOG + "pair_stats.parquet")
    pairs = ps[ps.folder.isin(set(M.benchmark)) & (ps.response_type == "binary") & (ps.n_graded >= 20)]
    base = float(pairs.mean_response.mean())
    benches = np.array(sorted(M.benchmark.unique()))
    rng = np.random.default_rng(0); rng.shuffle(benches)
    folds = np.array_split(benches, 5)
    print(f"measurement-db: {len(M)} items with traits, {len(benches)} benchmarks, {len(pairs)} pairs, ridge lambda {lam:g}")

    results, fits = {}, {}
    for which in ("traits", "guess", "both"):
        X = matrix(md, which); cols = list(X.columns); X = X.values
        level = {}
        for f in folds:
            te = M.benchmark.isin(f).values; tr = ~te
            p = sig(ridge(X[tr], M.z.values[tr], M.w.values[tr], lam)(X[te]))
            level.update(pd.Series(p).groupby(M.benchmark.values[te]).mean().to_dict())
        results[which] = level
        fits[which] = (cols, ridge(X, M.z.values, M.w.values, lam))
    pairs = pairs.assign(bmean=pairs.groupby("folder").mean_response.transform("mean"), bcnt=pairs.groupby("folder").mean_response.transform("count"))
    rows_out = []
    for b, g in pairs.groupby("folder"):
        r = g.mean_response.values; n = len(r)
        loo = (g.bmean.values * n - r) / (n - 1) if n > 1 else np.full(n, base)
        rows_out.append({"base rate": np.mean(abs(base - r)), "oracle benchmark level": np.mean(abs(loo - r)),
                         **{f"traits model ({w})": np.mean(abs(results[w][b] - r)) for w in results}})
    t = pd.DataFrame(rows_out)
    print("\n1. pair level on held-out measurement-db benchmarks: mean |pair mean prediction - pair success rate|, benchmarks equal")
    for c in t.columns:
        print(f"  {c:32} {t[c].mean():.3f}")

    # what the traits model learned (fit on all items, traits only)
    cols, _ = fits["traits"]
    X = matrix(md, "traits")[cols].values
    mu_x = np.average(X, axis=0, weights=M.w.values); Xc = (X - mu_x) * np.sqrt(M.w.values)[:, None]
    coef = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ ((M.z.values - np.average(M.z.values, weights=M.w.values)) * np.sqrt(M.w.values)))
    share = (X > 0).mean(axis=0)
    order = np.argsort(coef)
    print("\n  traits model, logit effect (share of items with the trait): lowest", [(cols[i], round(coef[i], 2), round(share[i], 2)) for i in order[:6]])
    print("  highest", [(cols[i], round(coef[i], 2), round(share[i], 2)) for i in order[::-1][:6]])

    print("\n2. proxy evaluations (traits models fit on all measurement-db items)")
    for name, (exam_dir, judge_dir) in EXAMS.items():
        exam = load_exam(exam_dir); pj = load(judge_dir)
        P = [r for r in rows if r["source"] == name]
        out = []
        for pi, pair in enumerate(exam["pairs"]):
            pr = [r for r in P if r["pair"] == pi]
            lab = {t["key"]: t["labels"] for t in pair["targets"]}
            rate = np.mean([y for t in pair["targets"] for y in t["labels"]])
            judge = np.mean([pj.get((pi, t["key"]), 0.575) for t in pair["targets"]])
            rec = {"pair": f"{pair['benchmark_name'][:16]} x {pair['subject']['normalized_name'][:18]}", "rate": rate, "judge": judge}
            for which, (cols, f) in fits.items():
                Xp = matrix(pr, which).reindex(columns=cols, fill_value=0.0).values
                rec[which] = float(sig(f(Xp)).mean())
            rec["judge+traits"] = float(sig((logit(judge) + logit(rec["traits"])) / 2))
            rec["judge+both"] = float(sig((logit(judge) + logit(rec["both"])) / 2))
            ys = np.concatenate([lab[r["item_id"]] for r in pr]); rep = [len(lab[r["item_id"]]) for r in pr]
            rec["guess AUC"] = auc(np.repeat([r["traits"]["p_strong_model"] for r in pr], rep), ys)
            rec["judge AUC (same items)"] = auc(np.repeat([pj.get((pi, r["item_id"]), 0.575) for r in pr], rep), ys)
            out.append(rec)
        o = pd.DataFrame(out)
        errs = {c: np.mean(abs(o[c] - o.rate)) for c in ["judge", "traits", "guess", "both", "judge+traits", "judge+both"]}
        print(f"  {name}: pair level error  base rate {np.mean(abs(base - o.rate)):.3f}, " + ", ".join(f"{k} {v:.3f}" for k, v in errs.items())
              + f" | AUC guess {o['guess AUC'].mean():.3f}, judge {o['judge AUC (same items)'].mean():.3f}")
        print(o.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
