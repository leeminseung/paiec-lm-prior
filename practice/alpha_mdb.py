"""How many exemplars should the budget-0 judgment be worth in the 고정 갱신 공식, measured on the measurement-db judge set?

The formula at budget k with x successes among the pair's first k exemplars: p = (x + A * judge) / (k + A); A is 4 today,
chosen when the starting point was the constant 0.575. With item 차이 되살리기 the logit gets k / (k + A) of the target
judgment's distance from the exemplars' mean judgment back. The judge set has no exemplar judgments, so that mean is
stood in for by the mean judgment logit of the pair's other judged items (6 of 7), which is what it tends to at large k.
Other subjects on the same benchmark are left out (the competition's formative samples rarely have them).

Per pair (exams/mdb-judge-1027, judgments runs/mdb-judge-opus-low): the 7 judged items are the targets, scored on all
their recorded responses; the exemplar pool is every other item that subject has a graded response on, in a random
order per draw, each revealed as one of its trials at random (as make_exam.py does). ALC per draw with the competition's
budget weights, pairs equal, responses weighted within a pair, averaged over draws. Intervals: 4,000 draws resampling
benchmarks. Rows for the constant base rate as the starting point show whether 4 still comes out there.

Usage: python -I alpha_mdb.py <judge_set_dir> <judge_run_dir> <draws_per_pair>
"""
import os
import random
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, MAX_LABELS, load_exam
from exemplar_shift import load

DATA = "./data/hf/"
M, BASE_RATE = 0.575, 0.627
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
W = np.array(ALC_WEIGHTS)
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))


def exemplar_pools(exam):
    """{pair index: {item_id: [labels]}} for the subject's graded items outside the pair's targets."""
    con = duckdb.connect()
    out = {}
    for bench in sorted({p["benchmark_name"] for p in exam["pairs"]}):
        idx = [pi for pi, p in enumerate(exam["pairs"]) if p["benchmark_name"] == bench]
        subs = [exam["pairs"][pi]["subject_id"] for pi in idx]
        r = con.execute(f"select subject_id, item_id, trial, response from read_parquet('{DATA}{bench}/formatted_tables/responses.parquet') "
                        f"where response is not null and subject_id in (select unnest(?)) and item_id in "
                        f"(select item_id from read_parquet('{DATA}{bench}/formatted_tables/items.parquet'))", [subs]).df()
        for pi in idx:
            p = exam["pairs"][pi]
            targets = {t["key"] for t in p["targets"]}
            g = r[(r.subject_id == p["subject_id"]) & ~r.item_id.isin(targets)]
            g = g.sort_values(["item_id", "trial"]) if "trial" in g else g.sort_values("item_id")
            out[pi] = {iid: [int(round(v)) for v in h.response] for iid, h in g.groupby("item_id")}
    return out


def main():
    exam = load_exam(sys.argv[1]); pj = load(sys.argv[2]); n_draws = int(sys.argv[3])
    pools = exemplar_pools(exam)
    pairs = exam["pairs"]
    # per pair, per draw: x successes among the first k exemplars at each budget, and k
    xk = {}
    for pi, p in enumerate(pairs):
        ids = sorted(pools[pi])
        rows = []
        for d in range(n_draws):
            rng = random.Random(f"alpha:{pi}:{d}")
            order = ids[:]; rng.shuffle(order)
            labels = [rng.choice(pools[pi][iid]) for iid in order[:MAX_LABELS]]
            rows.append([(sum(labels[:b]), len(labels[:b])) for b in BUDGETS])
        xk[pi] = np.array(rows, dtype=float)  # draws x budgets x 2
    print(f"{len(pairs)} pairs, {len({p['benchmark_name'] for p in pairs})} benchmarks, {n_draws} exemplar orders per pair; "
          f"pairs with fewer than 31 exemplars available: {sum(len(pools[pi]) < MAX_LABELS for pi in range(len(pairs)))}; "
          f"missing judgments filled with {M}: {sum((pi, t['key']) not in pj for pi, p in enumerate(pairs) for t in p['targets'])}")

    def pair_alc(pi, A, start, restore):
        """ALC of one pair averaged over its exemplar orders."""
        ts = pairs[pi]["targets"]
        p0 = np.array([pj.get((pi, t["key"]), M) for t in ts]) if start == "judge" else np.full(len(ts), BASE_RATE)
        ns = np.array([sum(t["labels"]) for t in ts], dtype=float); nf = np.array([len(t["labels"]) for t in ts]) - ns
        z = logit(p0)
        dev = np.array([z[i] - np.delete(z, i).mean() for i in range(len(z))]) if len(z) > 1 else np.zeros(len(z))
        x, k = xk[pi][:, :, 0][:, :, None], xk[pi][:, :, 1][:, :, None]  # draws x budgets x 1
        f = (x + A * p0) / (k + A)  # draws x budgets x items
        if restore:
            f = sig(logit(f) + k / (k + A) * dev)
        brier = (ns * (f - 1) ** 2 + nf * f ** 2).sum(axis=2) / (ns + nf).sum()  # draws x budgets
        return float((brier.mean(axis=0) * W).sum()), brier.mean(axis=0)

    bench_of = np.array([p["benchmark_name"] for p in pairs])
    benches = np.array(sorted(set(bench_of)))
    rng = np.random.default_rng(0)
    boot = [rng.choice(len(benches), len(benches)) for _ in range(4000)]
    members = {b: np.where(bench_of == b)[0] for b in benches}
    big = np.array([len(pools[pi]) + len(pairs[pi]["targets"]) >= 80 for pi in range(len(pairs))])

    for start, restore, label in (("judge", False, "judge + 고정 갱신 공식"), ("judge", True, "judge + 공식 + item 차이 되살리기 (stand-in)"),
                                  ("base", False, "base rate + 고정 갱신 공식")):
        per = {A: np.array([pair_alc(pi, A, start, restore)[0] for pi in range(len(pairs))]) for A in GRID}
        best = min(GRID, key=lambda A: per[A].mean())
        print(f"\n{label}: ALC by A (all {len(pairs)} pairs | pairs with >= 80 graded items, {int(big.sum())})")
        for A in GRID:
            diff = per[A] - per[4]
            bs = [np.concatenate([diff[members[benches[j]]] for j in draw]).mean() for draw in boot] if A != 4 else [0.0]
            print(f"  A {A:>2}: {per[A].mean():.4f} | {per[A][big].mean():.4f} | vs 4 {diff.mean():+.4f} "
                  f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")
        for A in sorted({4, best}):
            curve = np.mean([pair_alc(pi, A, start, restore)[1] for pi in range(len(pairs))], axis=0)
            print(f"  per budget, A {A}: {[round(float(c), 4) for c in curve]}")


if __name__ == "__main__":
    main()
