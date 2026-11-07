"""A baseline without any language model: shrink the pair's revealed success rate toward a prior mean.

For a target of system s on benchmark b, with k revealed results for (s, b) of which x were successes:
    p = (x + alpha * m) / (k + alpha)
At budget 0 this is m; as results accumulate it moves toward x / k, more slowly for a larger alpha.
It reads only (input, labeled), as the competition's predict() must.

m and alpha are fitted on training data only. The competition protocol is replayed on competition-like training
pairs (0/1 scores where 1 = success, a named system released 2025 or later by one of the nine listed companies,
at least 80 items, benchmark released 2024 or later), leaving out the exam's held-out benchmarks: keep a random
88-152 items, split them 50/50, reveal up to 31 acquisitions in random order, score every response of the
evaluation items at budgets 0, 1, 3, 7, 15, 31, and combine with the competition weights. The grid point with
the lowest mean ALC over pairs wins.

Usage: python -I shrinkage.py <data_dir> <pairs.parquet> <exam_dir> [replays]   -> writes <exam_dir>/shrinkage.json
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, MAX_LABELS

COMPANIES = {"Anthropic", "Google", "Moonshot", "OpenAI", "Zhipu", "Alibaba", "ByteDance", "DeepSeek", "xAI"}
M_GRID = np.round(np.arange(0.20, 0.801, 0.025), 3)
A_GRID = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128])


def make_predictor(m, alpha):
    def predict(inp, labeled):
        subject, item = inp
        ys = [y for (s, it), y in labeled if s == subject and it["benchmark_id"] == item["benchmark_id"]]
        return (sum(ys) + alpha * m) / (len(ys) + alpha)
    return predict


def training_pairs(pairs_path, held_out):
    p = pd.read_parquet(pairs_path)
    c = p[p.graded & p.one_is_success & p.named & p.big & p.recent_subject & p.recent_bench
          & p.provider.isin(COMPANIES) & ~p.folder.isin(held_out)]
    return c[["folder", "subject_id"]].drop_duplicates()


def load_labels(data_dir, pairs):
    """{(folder, subject_id): {item_id: [labels...]}} for graded responses."""
    import duckdb  # fitting only; predicting needs no extra package
    out = {}
    con = duckdb.connect()
    for folder, g in pairs.groupby("folder"):
        path = os.path.join(data_dir, folder, "formatted_tables", "responses.parquet")
        df = con.execute("SELECT subject_id, item_id, response FROM read_parquet(?) "
                         "WHERE response IS NOT NULL AND subject_id IN (SELECT unnest(?))",
                         [path, list(g.subject_id)]).df()
        for (sid, iid), r in df.groupby(["subject_id", "item_id"]):
            out.setdefault((folder, sid), {})[iid] = [int(round(v)) for v in r.response]
    return out


def replay(labels, rng):
    """One competition-style draw for one pair: cumulative revealed successes per budget and the eval success rate."""
    items = sorted(labels)
    rng.shuffle(items)
    items = items[:min(len(items), rng.randint(88, 152))]
    half = len(items) // 2
    acq, ev = items[:half], items[half:]
    rng.shuffle(acq)
    revealed = [rng.choice(labels[i]) for i in acq[:MAX_LABELS]]
    x = np.array([sum(revealed[:b]) for b in BUDGETS], dtype=float)
    k = np.array([min(b, len(revealed)) for b in BUDGETS], dtype=float)
    ys = [y for i in ev for y in labels[i]]
    return x, k, float(np.mean(ys))


def main():
    data_dir, pairs_path, exam_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    replays = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    held_out = json.load(open(os.path.join(exam_dir, "exam.json")))["held_out_benchmarks"]
    pairs = training_pairs(pairs_path, held_out)
    labels = load_labels(data_dir, pairs)
    labels = {k: v for k, v in labels.items() if len(v) >= 80}
    rows = []  # one row per (pair, replay)
    for key in sorted(labels):
        for r in range(replays):
            x, k, ybar = replay(labels[key], random.Random(f"{key[0]}:{key[1]}:{r}"))
            rows.append((key, x, k, ybar))
    X = np.stack([r[1] for r in rows])        # rows x budgets
    K = np.stack([r[2] for r in rows])
    Y = np.array([r[3] for r in rows])[:, None]
    W = np.array(ALC_WEIGHTS)
    pair_of = np.array([hash(r[0]) for r in rows])

    def mean_alc(m, a):
        p = (X + a * m) / (K + a)
        brier = p * p - 2 * p * Y + Y          # mean squared error against 0/1 outcomes with success rate Y
        alc = brier @ W
        return pd.Series(alc).groupby(pair_of).mean().mean()  # replays averaged within a pair, pairs equally

    grid = [(m, a, mean_alc(m, a)) for m in M_GRID for a in A_GRID]
    best = min(grid, key=lambda g: g[2])
    ref = {"always 0.5": mean_alc(0.5, 1e9), "revealed mean (0.5 at budget 0)": None}
    p_emp = np.where(K > 0, X / np.maximum(K, 1), 0.5)
    ref["revealed mean (0.5 at budget 0)"] = pd.Series(((p_emp * p_emp - 2 * p_emp * Y + Y) @ W)).groupby(pair_of).mean().mean()
    out = {"m": float(best[0]), "alpha": float(best[1]), "train_mean_alc": float(best[2]),
           "train_pairs": len(labels), "train_benchmarks": len({k[0] for k in labels}), "replays": replays,
           "held_out_excluded": held_out, "train_reference": {k: float(v) for k, v in ref.items()},
           "best_alpha_for_each_m": {f"{m:.3f}": float(min((g for g in grid if g[0] == m), key=lambda g: g[2])[1])
                                     for m in M_GRID[::4]}}
    json.dump(out, open(os.path.join(exam_dir, "shrinkage.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["m", "alpha", "train_mean_alc", "train_pairs", "train_benchmarks", "train_reference"]}, indent=1))


if __name__ == "__main__":
    main()
