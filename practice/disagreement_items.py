"""Items where the budget-0 judge and the recorded outcomes strongly disagree, as benchmark bug candidates.

For every judged item of the measurement-db judge set, the item's success rate over all subjects in measurement-db
(one mean per subject, then averaged over subjects) is read from the local responses tables. Two lists:
  - bug candidates: the judge is confident the item is easy (>= 0.85), and it is hard for almost everyone
    (item success rate <= 0.15 over >= 5 subjects). In "Fantastic Bugs" terms (Truong et al., NeurIPS 2025), an item
    that capable systems fail and a careful reader finds easy often has a wrong reference answer or grader;
  - judge misses: the judge is confident the item is hard (<= 0.15) and almost everyone solves it (>= 0.85).
Each list is ranked by the gap and written to a CSV with the item text's first 300 characters for human review.

Usage: python -I disagreement_items.py <judge_set_dir> <judge_run_dir> <out_csv>
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

DATA = "./data/hf/"


def main():
    exam = load_exam(sys.argv[1]); pj = load(sys.argv[2]); out = sys.argv[3]
    con = duckdb.connect()
    rows = []
    by_bench = {}
    for pi, p in enumerate(exam["pairs"]):
        for t in p["targets"]:
            if (pi, t["key"]) in pj:
                by_bench.setdefault(p["benchmark_name"], []).append((pi, t, pj[(pi, t["key"])]))
    for bench, entries in sorted(by_bench.items()):
        ids = sorted({t["key"] for _, t, _ in entries})
        r = con.execute(f"select subject_id, item_id, avg(response) as s from read_parquet('{DATA}{bench}/formatted_tables/responses.parquet') "
                        f"where response is not null and item_id in (select unnest(?)) group by subject_id, item_id", [ids]).df()
        agg = r.groupby("item_id").agg(item_rate=("s", "mean"), n_subjects=("s", "size"))
        for pi, t, prob in entries:
            a = agg.loc[t["key"]] if t["key"] in agg.index else None
            rows.append({"benchmark": bench, "item_id": t["key"], "subject": exam["pairs"][pi]["subject"].get("normalized_name", ""),
                         "judge": prob, "subject_outcome": float(np.mean(t["labels"])),
                         "item_rate": float(a.item_rate) if a is not None else np.nan,
                         "n_subjects": int(a.n_subjects) if a is not None else 0,
                         "text": (t["item"].get("item_content", "") or "")[:300].replace("\n", " ")})
    D = pd.DataFrame(rows)
    D["gap"] = D.judge - D.item_rate
    bug = D[(D.judge >= .85) & (D.item_rate <= .15) & (D.n_subjects >= 5)].sort_values("gap", ascending=False)
    miss = D[(D.judge <= .15) & (D.item_rate >= .85) & (D.n_subjects >= 5)].sort_values("gap")
    print(f"judged items {len(D)}, with >= 5 subjects {int((D.n_subjects >= 5).sum())}")
    print(f"bug candidates (judge >= 0.85, item rate <= 0.15): {len(bug)} items in {bug.benchmark.nunique()} benchmarks")
    print(bug.benchmark.value_counts().head(10).to_string())
    print(f"judge misses (judge <= 0.15, item rate >= 0.85): {len(miss)} items in {miss.benchmark.nunique()} benchmarks")
    print(miss.benchmark.value_counts().head(10).to_string())
    pd.concat([bug.assign(kind="bug candidate"), miss.assign(kind="judge miss")]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
