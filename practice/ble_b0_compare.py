"""Budget 0: the organizers' BLE (an LLM agent that searches measurement-db) against our judge, same LLM, same items.

For every target BLE answered (run_ble.py at budget 0), compares on those items: Brier over all the item's recorded
responses (responses weighted within a pair, pairs equal), the pair-level gap |mean prediction - mean outcome| over
the pair's sampled items, and within-pair AUC. Rows: constant 0.575 (the no-LLM start), our Opus low judgment,
BLE with Opus, and their logit average. Intervals: 4,000 draws resampling benchmarks, then the pairs inside.

Usage: python -I ble_b0_compare.py <exam_dir>:<judge_run>:<ble_run> [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))


def auc(p, y):
    pos, neg = p[y >= .5], p[y < .5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(a > b) + .5 * (a == b) for a in pos for b in neg]))


def main():
    pairs = []  # (benchmark, {row: probs}, labels list, item mean outcomes)
    for arg in sys.argv[1:]:
        exam_dir, j_run, b_run = arg.split(":")
        exam = load_exam(exam_dir); pj = load(j_run)
        ble = {(r["pair"], r["key"]): r["probability"] for r in map(json.loads, open(os.path.join(b_run, "predictions.jsonl")))
               if r["budget"] == 0 and r["probability"] is not None}
        for pi, p in enumerate(exam["pairs"]):
            ts = [t for t in p["targets"] if (pi, t["key"]) in ble]
            if not ts:
                continue
            judge = np.array([pj.get((pi, t["key"]), .575) for t in ts]); b = np.array([ble[(pi, t["key"])] for t in ts])
            rows = {"constant 0.575": np.full(len(ts), .575), "our judge (Opus low)": judge, "BLE (Opus)": b,
                    "average of the two": sig((logit(judge) + logit(b)) / 2)}
            pairs.append((exam_dir + p["benchmark_name"], rows, [t["labels"] for t in ts]))
    names = list(pairs[0][1])

    def stats(sel):
        out = {n: [] for n in names}
        for i in sel:
            _, rows, labels = pairs[i]
            ymean = np.array([np.mean(l) for l in labels])
            for n in names:
                p = rows[n]
                se = sum(sum((p[j] - y) ** 2 for y in labels[j]) for j in range(len(p))) / sum(len(l) for l in labels)
                out[n].append((se, abs(p.mean() - ymean.mean()), auc(p, ymean)))
        return {n: np.array(v) for n, v in out.items()}

    full = stats(range(len(pairs)))
    benches = sorted({b for b, _, _ in pairs})
    members = {b: [i for i, (bb, _, _) in enumerate(pairs) if bb == b] for b in benches}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(4000):
        sel = []
        for j in rng.integers(0, len(benches), len(benches)):
            m = members[benches[j]]
            sel += list(rng.choice(m, len(m)))
        draws.append(sel)
    ref = "our judge (Opus low)"
    print(f"{len(pairs)} pairs, {len(benches)} benchmarks, {sum(len(l) for _, _, l in pairs)} items answered by BLE")
    for n in names:
        v = full[n]
        d = v[:, 0] - full[ref][:, 0]
        bs = [np.mean([d[i] for i in sel]) for sel in draws]
        print(f"  {n:22} Brier {v[:, 0].mean():.4f} | minus judge {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]"
              f" | pair gap {v[:, 1].mean():.3f} | within-pair AUC {np.nanmean(v[:, 2]):.3f}")


if __name__ == "__main__":
    main()
