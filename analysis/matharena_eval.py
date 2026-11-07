"""Post-cutoff check on MathArena (Aug/Sep 2026 arXiv sets, 52 pairs, 2,872 targets): does Opus's lead over the open
forecasters survive where no forecaster can have seen the items or results?

Per forecaster, budget-0 forecasts q (runs ma-*-low), recalibrated with its own calibration-set a, b (9-98):
- within-pair AUC (item success rate over runs > 0.5 as positive), pairs with both outcomes;
- Brier against each run's 0/1, pairs weighted equally;
- pair level: |mean forecast - pair success rate|; set level: |mean over the set's pairs - set mean rate|.
Calibration-set values for the same measures are printed alongside (gap_levels.py / 11/1 logs).
Intervals: 4,000 bootstrap draws over pairs (pairs in a set share items, so these are optimistic).
"""
import json
import os
import sys
import numpy as np
sys.path.insert(0, "./practice")
from exemplar_shift import load

R = "."
exam = json.load(open(f"{R}/exams/matharena-1101/exam.json"))
RUNS = {"Opus": "ma-opus-low", "GLM": "ma-glm53flash-low", "DeepSeek": "ma-dsv41flash-low"}
RUNS = {k: v for k, v in RUNS.items() if k in os.environ.get("MA_FORECASTERS", "Opus,GLM,DeepSeek").split(",")}
CONST = {"Opus": (-0.11, 1.01), "GLM": (0.00, 0.63), "DeepSeek": (-0.26, 0.62)}
lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))
preds = {n: load(f"{R}/runs/{r}") for n, r in RUNS.items()}
rng = np.random.default_rng(0)

rows = {n: [] for n in RUNS}
oracle = []   # within-pair AUC of item difficulty from the other systems in the same set (an upper reference)   # per pair: (set, subject, auc, brier_raw, brier_rec, pair_mean_rec, rate)
for pi, p in enumerate(exam["pairs"]):
    ts = [t for t in p["targets"] if all((pi, t["key"]) in preds[n] for n in RUNS)]
    if len(ts) < 5:
        continue
    y = np.array([np.mean(t["labels"]) for t in ts])
    others = [q for q in exam["pairs"] if q["benchmark_name"] == p["benchmark_name"] and q is not p]
    od = {t["key"]: [] for t in ts}
    for q in others:
        for t in q["targets"]:
            if t["key"] in od:
                od[t["key"]] += t["labels"]
    orc = np.array([np.mean(od[t["key"]]) if od[t["key"]] else .5 for t in ts])
    pos, neg = orc[y > .5], orc[y <= .5]
    oracle.append(np.mean([(u > v) + .5 * (u == v) for u in pos for v in neg]) if len(pos) and len(neg) else np.nan)
    for n in RUNS:
        a, b = CONST[n]
        q = np.array([preds[n][(pi, t["key"])] for t in ts]); qr = sig(a + b * lg(q))
        pos, neg = qr[y > .5], qr[y <= .5]
        auc = np.mean([(u > v) + .5 * (u == v) for u in pos for v in neg]) if len(pos) and len(neg) else np.nan
        br = lambda f: sum(((f[j] - np.asarray(t["labels"])) ** 2).sum() for j, t in enumerate(ts)) / sum(len(t["labels"]) for t in ts)
        rows[n].append((p["benchmark_name"], p["subject"]["normalized_name"] + " " + p["subject"]["reasoning_effort"], auc, br(q), br(qr), qr.mean(), p["pair_success_rate"]))

def summary(n, idx=None):
    r = rows[n] if idx is None else [rows[n][i] for i in idx]
    auc = np.nanmean([x[2] for x in r]); b0 = np.mean([x[3] for x in r]); b1 = np.mean([x[4] for x in r])
    pl = np.mean([abs(x[5] - x[6]) for x in r])
    sets = sorted({x[0] for x in r})
    sl = np.mean([abs(np.mean([x[5] for x in r if x[0] == s]) - np.mean([x[6] for x in r if x[0] == s])) for s in sets])
    return auc, b0, b1, pl, sl

npairs = len(rows["Opus"])
print(f"{npairs} pairs in {len({x[0] for x in rows['Opus']})} sets; pairs with both outcomes for AUC: {sum(1 for x in rows['Opus'] if not np.isnan(x[2]))}")
print(f"{'':9} within-pair AUC | Brier raw | Brier recal | pair-level |error| | set-level |error|")
for n in RUNS:
    a = summary(n); print(f"{n:9} {a[0]:.3f} | {a[1]:.4f} | {a[2]:.4f} | {a[3]:.3f} | {a[4]:.3f}")
print(f"reference: item difficulty from the other systems in the set, within-pair AUC {np.nanmean(oracle):.3f}")
print("calibration set, for comparison: within-pair AUC Opus 0.693, GLM 0.654, DeepSeek 0.657 (109 benchmarks, 2+ pairs); pair-level |error| 0.139 / 0.165 / 0.194 raw")
draws = [rng.integers(0, npairs, npairs) for _ in range(4000)]
for n in [x for x in ("GLM", "DeepSeek") if x in RUNS]:
    est = np.array(summary("Opus")) - np.array(summary(n))
    bs = np.array([np.array(summary("Opus", d)) - np.array(summary(n, d)) for d in draws])
    lo, hi = np.nanquantile(bs, [.025, .975], axis=0)
    print(f"Opus minus {n}: AUC {est[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}] | Brier recal {est[2]:+.4f} [{lo[2]:+.4f}, {hi[2]:+.4f}] | pair-level |error| {est[3]:+.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}]")
print("\nper set, pair-level mean forecast (recalibrated) vs mean rate:")
for s in sorted({x[0] for x in rows['Opus']}):
    line = f"  {s:28} rate {np.mean([x[6] for x in rows['Opus'] if x[0] == s]):.2f}"
    for n in RUNS:
        line += f" | {n} {np.mean([x[5] for x in rows[n] if x[0] == s]):.2f} AUC {np.nanmean([x[2] for x in rows[n] if x[0] == s]):.2f}"
    print(line)
print("\nself-forecasts (subject is the forecaster's own model):")
for n, key in (("Opus", "Claude Opus 5.5"), ("DeepSeek", "DeepSeek V4.1 Flash")):
    for x in rows.get(n, []):
        if key in x[1]:
            print(f"  {n} on {x[1]:30} {x[0]:28} forecast {x[5]:.2f} vs rate {x[6]:.2f} | AUC {x[2]:.2f}")
