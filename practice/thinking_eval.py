"""Summarize thinking_judge.py judgments with their tag_summaries.py tags.

Per sample cell (benchmark older or newer, Opus low close or far on pair level): how often each tag appears, the share
of items with no thinking, and Brier on the items for the effort-high judgment and the low judgment of the same items.
Per tag: Brier of the high judgment on items with and without the tag, and the distance of the item's high judgment from
its pair success rate (how far the judge's level is off when it reasons that way). Also the agreement of high and low.

Usage: python -I thinking_eval.py <run_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

TAGS = ["names_benchmark", "recalls_results", "subject_capability", "item_difficulty", "grading", "falls_back_to_default"]


def main():
    d = sys.argv[1]
    J = [json.loads(line) for line in open(os.path.join(d, "judgments.jsonl"))]
    J = {(r["pair"], r["key"]): r for r in J if r.get("probability") is not None}
    T = {(r["pair"], r["key"]): r["tags"] for r in map(json.loads, open(os.path.join(d, "tags.jsonl"))) if r.get("tags")}
    rows = []
    for k, r in J.items():
        ns, n = sum(r["labels"]), len(r["labels"])
        row = {"cell": f"{r['age']}/{r['fit']}", "age": r["age"], "fit": r["fit"], "thinking": bool(r["summary"]),
               "high": r["probability"], "low": r["low_probability"], "rate": r["pair_success_rate"],
               "brier_high": (ns * (r["probability"] - 1) ** 2 + (n - ns) * r["probability"] ** 2) / n,
               "brier_low": (ns * (r["low_probability"] - 1) ** 2 + (n - ns) * r["low_probability"] ** 2) / n,
               "level_off_high": abs(r["probability"] - r["pair_success_rate"]), "tagged": k in T}
        tags = T.get(k) or {}
        for t in TAGS:
            row[t] = bool(tags.get(t)) if row["thinking"] else False
        row["main_basis"] = tags.get("main_basis", "none") if row["thinking"] else "none"
        rows.append(row)
    D = pd.DataFrame(rows)
    print(f"{len(D)} judgments, {int(D.thinking.sum())} with thinking, {int(D.tagged.sum())} tagged")
    print(f"high vs low: correlation {np.corrcoef(D.high, D.low)[0, 1]:.3f}, mean |difference| {np.mean(abs(D.high - D.low)):.3f}")
    g = D.groupby("cell").agg(items=("high", "size"), thinking=("thinking", "mean"),
                              **{t: (t, "mean") for t in TAGS}, brier_high=("brier_high", "mean"), brier_low=("brier_low", "mean"))
    print("\nshare of items per cell (tags counted only where there is thinking)")
    print(g.round(2).to_string())
    g2 = D.groupby("age").agg(items=("high", "size"), thinking=("thinking", "mean"), **{t: (t, "mean") for t in TAGS})
    print("\nby benchmark age")
    print(g2.round(2).to_string())
    print("\nmain basis counts by age")
    print(pd.crosstab(D.main_basis, D.age).to_string())
    print("\nper tag, items with thinking: Brier of the high judgment and |high judgment - pair success rate|, with vs without")
    W = D[D.thinking]
    for t in TAGS:
        a, b = W[W[t]], W[~W[t]]
        print(f"  {t:22} with n={len(a):3} Brier {a.brier_high.mean():.3f} level off {a.level_off_high.mean():.3f} | "
              f"without n={len(b):3} Brier {b.brier_high.mean():.3f} level off {b.level_off_high.mean():.3f}")
    print(f"  no thinking            n={int((~D.thinking).sum()):3} Brier {D[~D.thinking].brier_high.mean():.3f} "
          f"level off {D[~D.thinking].level_off_high.mean():.3f} (mean judgment {D[~D.thinking].high.mean():.2f})")


if __name__ == "__main__":
    main()
