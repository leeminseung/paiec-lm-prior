"""Compare budget-0 judges on a subset of the measurement-db judge set (subset_exam.py output) against the Opus low
judgments already made on the full set (matched through each pair's "source_pair").

On the items every judge answered: budget-0 Brier (responses weighted within a pair, pairs equal), mean |pair mean
prediction - pair success rate|, mean prediction minus mean outcome, within-pair AUC (pairs with both outcomes, mean).
Per judge: answered share, cost per call, seconds p50 / p95 / max and the share over 120 s (the competition's per-attempt
cap), output and thinking tokens. Intervals: 4,000 draws resampling pairs, for each judge minus Opus low.

Usage: python -I compare_open_judges.py <subset_exam_dir> <opus_full_run_dir> <name>=<run_dir> ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from embed_eval import auc
from exemplar_shift import load


def records(run_dir):
    return [json.loads(line) for line in open(os.path.join(run_dir, "predictions.jsonl"))]


def main():
    exam = load_exam(sys.argv[1])
    opus_full = load(sys.argv[2])
    pairs = exam["pairs"]
    judges = {"Opus low": {(pi, t["key"]): opus_full[(p["source_pair"], t["key"])] for pi, p in enumerate(pairs)
                           for t in p["targets"] if (p["source_pair"], t["key"]) in opus_full}}
    stats = {}
    for arg in sys.argv[3:]:
        name, d = arg.split("=", 1)
        R = records(d)
        judges[name] = {(r["pair"], r["key"]): r["probability"] for r in R if r.get("probability") is not None}
        sec = np.array([r["seconds"] for r in R if r.get("probability") is not None])
        stats[name] = {"answered": len(judges[name]), "asked": len({(r["pair"], r["key"]) for r in R}),
                       "cost": np.mean([r.get("cost_usd") or 0 for r in R]),
                       "p50": np.median(sec), "p95": np.percentile(sec, 95), "max": sec.max(), "over120": (sec > 120).mean(),
                       "out": np.mean([r.get("output_tokens") or 0 for r in R if r.get("probability") is not None]),
                       "think": np.mean([r.get("reasoning_tokens") or 0 for r in R if r.get("probability") is not None])}
    common = set.intersection(*(set(j) for j in judges.values()))
    items = sorted(common)
    by_pair = {}
    for pi, k in items:
        by_pair.setdefault(pi, []).append(k)
    label = {(pi, t["key"]): t["labels"] for pi, p in enumerate(pairs) for t in p["targets"]}
    print(f"items every judge answered: {len(items)} over {len(by_pair)} pairs ({len(pairs)} pairs, "
          f"{sum(len(p['targets']) for p in pairs)} items in the subset)")

    def per_pair(j):
        out = {}
        for pi, ks in by_pair.items():
            q = np.array([j[(pi, k)] for k in ks]); ys = [label[(pi, k)] for k in ks]
            ns = np.array([sum(y) for y in ys], float); n = np.array([len(y) for y in ys], float)
            brier = float((ns * (q - 1) ** 2 + (n - ns) * q ** 2).sum() / n.sum())
            lvl = abs(q.mean() - pairs[pi]["pair_success_rate"])
            bias = q.mean() - ns.sum() / n.sum()
            a = auc(np.repeat(q, n.astype(int)), np.concatenate(ys))
            out[pi] = (brier, lvl, bias, a)
        return out

    res = {name: per_pair(j) for name, j in judges.items()}
    pis = sorted(by_pair)
    rng = np.random.default_rng(0)
    draws = [rng.choice(len(pis), len(pis)) for _ in range(4000)]
    base = np.array([res["Opus low"][pi][0] for pi in pis]); base_l = np.array([res["Opus low"][pi][1] for pi in pis])
    print(f"\n{'judge':28} {'Brier':>7} {'vs Opus [95%]':>26} {'level err':>9} {'vs Opus':>8} {'bias':>7} {'AUC':>6}")
    for name in judges:
        b = np.array([res[name][pi][0] for pi in pis]); l = np.array([res[name][pi][1] for pi in pis])
        bias = np.mean([res[name][pi][2] for pi in pis])
        a = np.nanmean([res[name][pi][3] if res[name][pi][3] is not None else np.nan for pi in pis])
        db = [(b - base)[d].mean() for d in draws]
        print(f"{name:28} {b.mean():7.4f} {(b - base).mean():+8.4f} [{np.quantile(db, .025):+.4f}, {np.quantile(db, .975):+.4f}] "
              f"{l.mean():9.3f} {(l - base_l).mean():+8.3f} {bias:+7.3f} {a:6.3f}")
    print(f"\n{'judge':28} {'answered':>12} {'$/call':>8} {'p50 s':>6} {'p95 s':>6} {'max s':>6} {'>120s':>6} {'out tok':>8} {'think tok':>9}")
    for name, s in stats.items():
        print(f"{name:28} {s['answered']:>5}/{s['asked']:<6} {s['cost']:8.4f} {s['p50']:6.0f} {s['p95']:6.0f} {s['max']:6.0f} "
              f"{s['over120']:6.1%} {s['out']:8.0f} {s['think']:9.0f}")


if __name__ == "__main__":
    main()
