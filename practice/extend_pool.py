"""Rebuild an exam with its whole acquisition pool in reveal order, for acquisition-function experiments.

make_exam.py keeps only the first 31 acquisitions of each pair. This script replays make_exam.build_pair with the
exam's own seed and spec (benchmark, subject, kept item count) but keeps every pool item, drawing each revealed trial
from the same random stream, so the first 31 come out identical. It checks that the rebuilt pairs match the exam on
targets (keys, labels, item text) and on the first 31 acquisitions (keys, labels, item text), and stops on any
mismatch. Output: <exam_dir>/full_pool/exam.json, the same exam with full acquisition lists.

Usage: python -I extend_pool.py <data_dir> <exam_dir> [<exam_dir> ...]
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import MAX_LABELS, as_str, item_dict, load_exam, subject_dict, table


def full_pair(data_dir, bench, subject_id, seed, anon_bench, n_keep):
    """make_exam.build_pair without the 31-item cut on the reveal sequence."""
    subj = table(data_dir, bench, "subjects.parquet").set_index("subject_id").loc[subject_id]
    items = table(data_dir, bench, "items.parquet").set_index("item_id")
    resp = table(data_dir, bench, "responses.parquet")
    resp = resp[(resp["subject_id"] == subject_id) & resp["response"].notna()]
    by_item = {}
    for item_id, g in resp.groupby("item_id"):
        by_item[item_id] = {"labels": [int(round(v)) for v in g.sort_values("trial")["response"]],
                            "interactors": as_str(g["interactors"].iloc[0])}
    item_ids = sorted(by_item)
    rng = random.Random(f"{seed}:{bench}:{subject_id}")
    rng.shuffle(item_ids)
    if n_keep is not None:
        item_ids = item_ids[:n_keep]
    half = len(item_ids) // 2
    acq_pool, eval_pool = item_ids[:half], item_ids[half:]
    order = acq_pool[:]
    rng.shuffle(order)
    as_item = lambda iid: item_dict(items.loc[iid], by_item[iid]["interactors"], anon_bench)
    acquisitions = [{"key": iid, "item": as_item(iid), "label": rng.choice(by_item[iid]["labels"])} for iid in order]
    targets = [{"key": iid, "item": as_item(iid), "labels": by_item[iid]["labels"]} for iid in sorted(eval_pool)]
    return acquisitions, targets, subject_dict(subj)


def main():
    data_dir = sys.argv[1]
    for exam_dir in sys.argv[2:]:
        exam = load_exam(exam_dir)
        seed = exam["seed"]
        out = json.loads(json.dumps(exam))
        for pi, p in enumerate(exam["pairs"]):
            n_keep = p["n_items"] if p["n_items"] < p["n_graded_items"] else None
            acq, tg, subj = full_pair(data_dir, p["benchmark_name"], p["subject_id"], seed, p["anon_benchmark_id"], n_keep)
            ok = (len(acq) == p["n_acquisition_pool"]
                  and [(a["key"], a["label"], a["item"]) for a in acq[:MAX_LABELS]]
                  == [(a["key"], a["label"], a["item"]) for a in p["acquisitions"]]
                  and [(t["key"], t["labels"], t["item"]) for t in tg] == [(t["key"], t["labels"], t["item"]) for t in p["targets"]]
                  and subj == p["subject"])
            if not ok:
                raise SystemExit(f"{exam_dir} pair {pi} ({p['benchmark_name']}): rebuilt pair does not match the exam")
            out["pairs"][pi]["acquisitions"] = acq
        os.makedirs(os.path.join(exam_dir, "full_pool"), exist_ok=True)
        with open(os.path.join(exam_dir, "full_pool", "exam.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False)
        extra = sum(len(p["acquisitions"]) - MAX_LABELS for p in out["pairs"])
        print(f"{exam_dir}: all {len(out['pairs'])} pairs match; pool items beyond the first 31: {extra}")


if __name__ == "__main__":
    main()
