"""Build a practice exam from held-out benchmarks, following the competition protocol.

For each chosen (benchmark, subject) pair with at least 80 distinct graded items:
  - optionally keep only a random n of its items first (the competition scores each submission on a sample of
    at most 1,000 (system, item) combinations over the whole round)
  - split the items 50/50 into an acquisition pool and an evaluation pool (fixed seed)
  - fix a random order over the acquisition pool and keep the first 31 as the reveal sequence;
    each acquired item reveals one recorded response (default random acquisition policy)
  - every graded response of an evaluation item is scored
Benchmark ids are anonymized. The exam file holds the answers, so keep it outside the agent's folder.

Usage: python -I make_exam.py <data_dir> <out_dir> <seed> <bench:subject_id,...>
       python -I make_exam.py <data_dir> <out_dir> <seed> <bench|subject_id|n_items,...>   (pick_round.py prints this)
"""
import json
import os
import random
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only (no downloaded data here)
from common import MAX_LABELS, anon_id, as_str, item_dict, subject_dict, table


def build_pair(data_dir, bench, subject_id, seed, anon_bench, n_keep=None):
    subj = table(data_dir, bench, "subjects.parquet").set_index("subject_id").loc[subject_id]
    items = table(data_dir, bench, "items.parquet").set_index("item_id")
    resp = table(data_dir, bench, "responses.parquet")
    resp = resp[(resp["subject_id"] == subject_id) & resp["response"].notna()]
    by_item = {}
    for item_id, g in resp.groupby("item_id"):
        by_item[item_id] = {"labels": [int(round(v)) for v in g.sort_values("trial")["response"]],
                            "interactors": as_str(g["interactors"].iloc[0])}
    item_ids = sorted(by_item)
    if len(item_ids) < 80:
        raise SystemExit(f"{bench}/{subject_id}: only {len(item_ids)} graded items")
    rng = random.Random(f"{seed}:{bench}:{subject_id}")
    rng.shuffle(item_ids)
    n_graded = len(item_ids)
    if n_keep is not None:
        item_ids = item_ids[:n_keep]
        if len(item_ids) < 80:
            raise SystemExit(f"{bench}/{subject_id}: keeping {len(item_ids)} items, below the 80-item rule")
    half = len(item_ids) // 2
    acq_pool, eval_pool = item_ids[:half], item_ids[half:]
    order = acq_pool[:]
    rng.shuffle(order)
    subject = subject_dict(subj)

    def as_item(iid):
        return item_dict(items.loc[iid], by_item[iid]["interactors"], anon_bench)

    acquisitions = [{"key": iid, "item": as_item(iid), "label": rng.choice(by_item[iid]["labels"])}
                    for iid in order[:MAX_LABELS]]
    targets = [{"key": iid, "item": as_item(iid), "labels": by_item[iid]["labels"]} for iid in sorted(eval_pool)]
    return {"benchmark_name": bench, "anon_benchmark_id": anon_bench, "subject_id": subject_id,
            "subject": subject, "n_items": len(item_ids), "n_graded_items": n_graded,
            "n_acquisition_pool": len(acq_pool), "acquisitions": acquisitions, "targets": targets}


def main():
    data_dir, out_dir, seed, spec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    pairs = []
    for part in spec.split(","):
        if "|" in part:
            bench, subject_id, n_keep = part.split("|")
            n_keep = int(n_keep)
        else:
            (bench, subject_id), n_keep = part.split(":"), None
        pairs.append(build_pair(data_dir, bench, subject_id, seed, anon_id("benchmark", bench, seed), n_keep))
    os.makedirs(out_dir, exist_ok=True)
    exam = {"seed": seed, "held_out_benchmarks": sorted({p["benchmark_name"] for p in pairs}), "pairs": pairs}
    with open(os.path.join(out_dir, "exam.json"), "w") as f:
        json.dump(exam, f, ensure_ascii=False)
    for i, p in enumerate(pairs):
        n_resp = sum(len(t["labels"]) for t in p["targets"])
        rate = sum(sum(t["labels"]) for t in p["targets"]) / n_resp
        print(f"pair {i}: {p['benchmark_name']} / {p['subject']['normalized_name']} | items {p['n_items']} "
              f"(of {p['n_graded_items']} graded, acquisition pool {p['n_acquisition_pool']}) | "
              f"eval targets {len(p['targets'])} (responses {n_resp}, success {rate:.3f}) | anon id {p['anon_benchmark_id']}")


if __name__ == "__main__":
    main()
