"""Build an exam-format file of measurement-db pairs for measuring the judge's pair level against pair success rates.

Benchmarks: 0/1 (higher is better) with item tables local, both proxy evaluations' benchmarks removed. Per benchmark up to
<pairs> pairs with >= 20 graded responses (seed), and per pair <items> items that subject answered (seed). Each target
carries its recorded labels; each pair carries its pair success rate over all its graded responses (pair_stats).
No acquisitions: only budget-0 judgments are asked. Items and subjects are built exactly as make_exam.py builds them, so
subscription_judge.py sends the judge the same message shape as in both proxy evaluations.

Usage: python -I make_judge_set.py <out_dir> <seed> <pairs_per_benchmark> <items_per_pair>
"""
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import anon_id, as_str, item_dict, subject_dict, table

ROOT = "./"
DATA = ROOT + "data/hf"
CATALOG = ROOT + "agent_box_02/data/catalog/"
EXAMS = [ROOT + "exams/rehearsal-02", ROOT + "exams/proxy-confirm-1026"]


def main():
    out_dir, seed, n_pairs, n_items = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ps = pd.read_parquet(CATALOG + "pair_stats.parquet")
    bm = pd.read_parquet(CATALOG + "benchmarks_all.parquet")
    bm["direction"] = bm["response_scale"].map(lambda v: (json.loads(v) or {}).get("direction") if v else None)
    ok = ps.merge(bm[["folder", "direction"]], on="folder")
    ok = ok[(ok["response_type"] == "binary") & (ok["direction"] == "higher_is_better") & (ok["n_graded"] >= 20)]
    held = set()
    for d in EXAMS:
        held |= set(json.load(open(os.path.join(d, "exam.json")))["held_out_benchmarks"])
    ok = ok[~ok.folder.isin(held)]
    pairs = []
    for bench in sorted(ok.folder.unique()):
        if not os.path.exists(f"{DATA}/{bench}/formatted_tables/items.parquet"):
            continue
        cand = ok[ok.folder == bench].sort_values("subject_id")
        chosen = cand.sample(min(n_pairs, len(cand)), random_state=random.Random(f"{seed}:{bench}").randrange(2 ** 31))
        subjects = table(DATA, bench, "subjects.parquet").set_index("subject_id")
        items = table(DATA, bench, "items.parquet").set_index("item_id")
        resp = table(DATA, bench, "responses.parquet")
        anon = anon_id("benchmark", bench, seed)
        for row in chosen.itertuples():
            r = resp[(resp.subject_id == row.subject_id) & resp.response.notna() & resp.item_id.isin(items.index)]
            by_item = {iid: g for iid, g in r.groupby("item_id")}
            ids = sorted(by_item)
            if not ids:
                continue
            pick = random.Random(f"{seed}:{bench}:{row.subject_id}").sample(ids, min(n_items, len(ids)))
            targets = []
            for iid in pick:
                g = by_item[iid].sort_values("trial") if "trial" in by_item[iid] else by_item[iid]
                targets.append({"key": iid, "item": item_dict(items.loc[iid], as_str(g["interactors"].iloc[0]), anon),
                                "labels": [int(round(v)) for v in g["response"]]})
            pairs.append({"benchmark_name": bench, "anon_benchmark_id": anon, "subject_id": row.subject_id,
                          "subject": subject_dict(subjects.loc[row.subject_id]), "pair_success_rate": float(row.mean_response),
                          "n_graded": int(row.n_graded), "acquisitions": [], "targets": targets})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exam.json"), "w") as f:
        json.dump({"seed": seed, "held_out_benchmarks": [], "purpose": "judge pair level on measurement-db", "pairs": pairs},
                  f, ensure_ascii=False)
    print(f"{len(pairs)} pairs over {len({p['benchmark_name'] for p in pairs})} benchmarks, "
          f"{sum(len(p['targets']) for p in pairs)} targets")


if __name__ == "__main__":
    main()
