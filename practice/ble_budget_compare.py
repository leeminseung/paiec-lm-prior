"""At a label budget b > 0: the organizers' BLE (its LLM reads the revealed outcomes and revises in text) against our
fixed update formula, same Opus judge, same items.

For every target BLE answered at budget 0 and at budget b (run_ble.py), compares on those items: Brier over all the
item's recorded responses (responses weighted within a pair, pairs equal), the pair-level gap |mean prediction - mean
outcome| over the pair's sampled items, and within-pair AUC. Rows:
  our judge alone          Opus low budget-0 judgment, no update
  BLE budget 0             BLE with Opus, no revealed outcomes
  formula only             고정 갱신 공식: own pair's b outcomes + other subjects' outcomes on the same benchmark
  current method           formula + item 차이 되살리기 (the submitted design)
  current, own pair only   the same with other subjects' outcomes left out (BLE also ignores other subjects)
  BLE budget b             BLE with Opus, reading the same subject's revealed outcomes
  average                  logit average of the current method and BLE budget b
Intervals: 4,000 draws resampling benchmarks, then the pairs inside; differences are against the current method.

Usage: python -I ble_budget_compare.py <b> <exam_dir>:<target_run>:<exemplar_run>:<ble_run_b0>:<ble_run_b> [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from ble_b0_compare import auc
from common import load_exam
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load
from hier_llm import predict_all

logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))


def ble_preds(run, b):
    return {(r["pair"], r["key"]): r["probability"] for r in map(json.loads, open(os.path.join(run, "predictions.jsonl")))
            if r["budget"] == b and r["probability"] is not None}


def main():
    b = int(sys.argv[1])
    pairs = []
    for arg in sys.argv[2:]:
        exam_dir, t_run, e_run, b0_run, bb_run = arg.split(":")
        exam = load_exam(exam_dir); pt = load(t_run); pe = load(e_run)
        q0, qb = ble_preds(b0_run, 0), ble_preds(bb_run, b)
        cnt = counts(exam)
        zt = {(pi, t["key"]): float(logit(pt.get((pi, t["key"]), M))) for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}
        ze = {(pi, a["key"]): float(logit(pe.get((pi, a["key"]), M))) for pi, p in enumerate(exam["pairs"]) for a in p["acquisitions"]}
        hier = predict_all(exam, zt, ze, 1.09, 0.46)
        for pi, p in enumerate(exam["pairs"]):
            ts = [t for t in p["targets"] if (pi, t["key"]) in q0 and (pi, t["key"]) in qb]
            if not ts:
                continue
            x, k, s, n = cnt[(b, pi)]
            zbar = np.mean([logit(pe.get((pi, a["key"]), M)) for a in p["acquisitions"][:b]]) if b else 0.0
            p0 = np.array([pt.get((pi, t["key"]), M) for t in ts])
            formula = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (k + ALPHA)
            own = (x + ALPHA * p0) / (k + ALPHA)
            shift = k / (k + 4.0) * (logit(p0) - zbar) if b else 0.0
            current = sig(logit(formula) + shift)
            rows = {"our judge alone": p0, "BLE budget 0": np.array([q0[(pi, t["key"])] for t in ts]),
                    "formula only": formula, "current method": current, "current, own pair only": sig(logit(own) + shift),
                    "hierarchical model": np.array([hier[(b, pi, t["key"])] for t in ts]),
                    f"BLE budget {b}": np.array([qb[(pi, t["key"])] for t in ts])}
            rows["average"] = sig((logit(current) + logit(rows[f"BLE budget {b}"])) / 2)
            rows["average with hierarchical"] = sig((logit(rows["hierarchical model"]) + logit(rows[f"BLE budget {b}"])) / 2)
            pairs.append((exam_dir + p["benchmark_name"], rows, [t["labels"] for t in ts]))
    names = list(pairs[0][1])
    ref = os.environ.get("REF_ROW", "current method")
    full = {n: [] for n in names}
    for _, rows, labels in pairs:
        ymean = np.array([np.mean(l) for l in labels])
        for n in names:
            v = rows[n]
            se = sum(sum((v[j] - y) ** 2 for y in labels[j]) for j in range(len(v))) / sum(len(l) for l in labels)
            full[n].append((se, abs(v.mean() - ymean.mean()), auc(v, ymean)))
    full = {n: np.array(v) for n, v in full.items()}
    benches = sorted({bb for bb, _, _ in pairs})
    members = {bb: [i for i, (c, _, _) in enumerate(pairs) if c == bb] for bb in benches}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(4000):
        sel = []
        for j in rng.integers(0, len(benches), len(benches)):
            m = members[benches[j]]
            sel += list(rng.choice(m, len(m)))
        draws.append(sel)
    print(f"budget {b}: {len(pairs)} pairs, {len(benches)} benchmarks, {sum(len(l) for _, _, l in pairs)} items answered by BLE at budgets 0 and {b}")
    for n in names:
        v = full[n]
        d = v[:, 0] - full[ref][:, 0]
        bs = [np.mean([d[i] for i in sel]) for sel in draws]
        print(f"  {n:24} Brier {v[:, 0].mean():.4f} | minus current {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]"
              f" | pair gap {v[:, 1].mean():.3f} | within-pair AUC {np.nanmean(v[:, 2]):.3f}")


if __name__ == "__main__":
    main()
