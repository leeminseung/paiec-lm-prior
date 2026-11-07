"""Where does the budget-0 judge do well or badly? Breakdowns on the measurement-db judge set (Opus low by default).

Per pair: judge Brier over its judged items (responses weighted within the pair), base-rate Brier (0.627) on the same
items, and the signed pair error (pair mean prediction - pair success rate). Groups (pairs equal within a group):
  - subject provider; whether the subject is an Anthropic model (self-preference check: a judge that favours its own
    family would show a more positive signed error there); subject release year; harness recorded or not;
    reasoning effort recorded or not.
A second judge run on the same set (or a subset) can be given to repeat the provider check for that judge's family.

Usage: python -I judge_breakdowns.py <judge_set_dir> <judge_run_dir> [<other_run_dir> <provider_substring>]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

BASE = 0.627


def pair_table(exam, pj):
    rows = []
    for pi, p in enumerate(exam["pairs"]):
        items = [(pj[(pi, t["key"])], sum(t["labels"]), len(t["labels"])) for t in p["targets"] if (pi, t["key"]) in pj]
        if not items:
            continue
        pr = np.array([i[0] for i in items]); ns = np.array([i[1] for i in items], float); n = np.array([i[2] for i in items], float)
        s = p["subject"]
        rows.append({"pair": pi, "bench": p["benchmark_name"], "provider": s.get("provider", "") or "(none)",
                     "name": s.get("normalized_name", ""), "year": (s.get("release_date", "") or "")[:4] or "(none)",
                     "harness": "recorded" if s.get("harness") else "empty",
                     "effort": "recorded" if s.get("reasoning_effort") else "empty",
                     "judge": float(((ns * (pr - 1) ** 2 + (n - ns) * pr ** 2).sum()) / n.sum()),
                     "base": float(((ns * (BASE - 1) ** 2 + (n - ns) * BASE ** 2).sum()) / n.sum()),
                     "err": float(pr.mean() - p["pair_success_rate"]), "rate": p["pair_success_rate"]})
    return pd.DataFrame(rows)


def show(D, col, min_pairs=8):
    g = D.groupby(col).agg(pairs=("pair", "size"), benches=("bench", "nunique"), judge=("judge", "mean"),
                           base=("base", "mean"), signed_err=("err", "mean"), abs_err=("err", lambda e: np.abs(e).mean()),
                           rate=("rate", "mean"))
    g = g[g.pairs >= min_pairs].sort_values("pairs", ascending=False)
    g["gain"] = g.base - g.judge
    print(f"\nby {col} (groups with >= {min_pairs} pairs)")
    print(g[["pairs", "benches", "rate", "judge", "base", "gain", "signed_err", "abs_err"]].round(3).to_string())


def main():
    exam = load_exam(sys.argv[1]); pj = load(sys.argv[2])
    D = pair_table(exam, pj)
    D["anthropic"] = np.where(D.provider.str.contains("anthropic", case=False), "Anthropic", "other")
    print(f"{len(D)} pairs, {D.bench.nunique()} benchmarks; overall judge Brier {D.judge.mean():.4f}, base {D.base.mean():.4f}, "
          f"signed error {D.err.mean():+.3f}")
    for col in ("provider", "anthropic", "year", "harness", "effort"):
        show(D, col)
    # self-preference with benchmark held fixed: Anthropic pair's signed error minus other pairs' on the same benchmark
    diffs = []
    for b, g in D.groupby("bench"):
        a, o = g[g.anthropic == "Anthropic"], g[g.anthropic == "other"]
        if len(a) and len(o):
            diffs.append(a.err.mean() - o.err.mean())
    diffs = np.array(diffs)
    rng = np.random.default_rng(0)
    bs = [rng.choice(diffs, len(diffs)).mean() for _ in range(4000)]
    print(f"\nsame benchmark, Anthropic pair minus other pairs, signed error: {diffs.mean():+.3f} "
          f"[{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}] over {len(diffs)} benchmarks")
    if len(sys.argv) > 4:
        other, fam = load(sys.argv[3]), sys.argv[4]
        E = pair_table(exam, other)
        E["family"] = np.where(E.provider.str.contains(fam, case=False) | E.name.str.contains(fam, case=False), fam, "other")
        print(f"\nsecond judge {os.path.basename(sys.argv[3].rstrip('/'))}: {len(E)} pairs")
        show(E, "family", min_pairs=3)


if __name__ == "__main__":
    main()
