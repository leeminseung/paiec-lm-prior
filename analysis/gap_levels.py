"""Where Opus's budget-0 lead sits: benchmark level, subject-within-benchmark level, or item ranking.

Calibration set (exams/mdb-judge-1027), items every forecaster answered, each forecaster recalibrated with its own a, b
(forecaster_constants.py). Brier is scored per response with pairs weighted equally. Sequential oracle split:
  calibration: raw -> sigmoid(a + b logit q)
  benchmark:   one logit shift per benchmark so the benchmark's mean prediction (pairs equal) equals its observed rate
  subject:     one further shift per pair so the pair mean prediction equals the pair's observed rate on these items
  ranking:     what is left (within-pair ranking and spread)
Only benchmarks with at least two pairs, so the benchmark and subject steps are separable. 95% intervals for the
difference to Opus from 2,000 bootstrap draws over benchmarks.
Orderings, with the pair success rate over all of a pair's responses as truth: across benchmarks (Spearman of the
benchmark mean prediction), subjects within a benchmark (share of subject pairs ordered correctly), items within a pair
(AUC on these items).
"""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr
sys.path.insert(0, "./report/figures")
sys.path.insert(0, "./practice")
from forecaster_constants import CALIB_RUNS, ROOT, logit
from common import load_exam
from exemplar_shift import load

CONST = {"Opus 5.5, low": (-0.11, 1.01), "GLM-5.3 Flash, low": (0.00, 0.63), "DeepSeek V4.1 Flash, low": (-0.26, 0.62)}
sig = lambda z: 1 / (1 + np.exp(-z))
OPUS = "Opus 5.5, low"


def brier(p, labels):
    return sum(((p[j] - np.asarray(l)) ** 2).sum() for j, l in enumerate(labels)) / sum(len(l) for l in labels)


def wmean(p, labels):
    n = np.array([len(l) for l in labels], float)
    return (p * n).sum() / n.sum()


def auc(p, y):
    pos, neg = p[y > .5], p[y <= .5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return np.mean([(a > b) + .5 * (a == b) for a in pos for b in neg])


exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
preds = {n: load(f"{ROOT}/runs/{r}") for n, r in CALIB_RUNS.items()}
bench = {}
for pi, p in enumerate(exam["pairs"]):
    ts = [t for t in p["targets"] if all((pi, t["key"]) in pj for pj in preds.values())]
    if len(ts) >= 2:
        bench.setdefault(p["benchmark_name"], []).append((pi, ts))
bench = {b: v for b, v in bench.items() if len(v) >= 2}
names = list(CONST)

# per pair stage Brier, per forecaster
stages = {n: {} for n in names}   # n -> bench -> list of (raw, recal, bench-matched, pair-matched)
order = {n: {"bmean": [], "conc": [0, 0], "auc": []} for n in names}
brate = []
for b, prs in bench.items():
    rates = [exam["pairs"][pi]["pair_success_rate"] for pi, _ in prs]
    brate.append(np.mean(rates))
    for n in names:
        a0, b0 = CONST[n]
        zs = [a0 + b0 * np.array([float(logit(preds[n][(pi, t["key"])])) for t in ts]) for pi, ts in prs]
        raws = [sig(np.array([float(logit(preds[n][(pi, t["key"])])) for t in ts])) for pi, ts in prs]
        labs = [[t["labels"] for t in ts] for _, ts in prs]
        obs = [wmean(np.array([np.mean(l) for l in L]), L) for L in labs]
        tgt = np.clip(np.mean(obs), 1e-4, 1 - 1e-4)
        cB = brentq(lambda c: np.mean([wmean(sig(z + c), L) for z, L in zip(zs, labs)]) - tgt, -30, 30)
        rows = []
        for z, r, L, o in zip(zs, raws, labs, obs):
            cP = brentq(lambda c: wmean(sig(z + c), L) - np.clip(o, 1e-4, 1 - 1e-4), -30, 30)
            rows.append((brier(r, L), brier(sig(z), L), brier(sig(z + cB), L), brier(sig(z + cP), L)))
        stages[n][b] = np.array(rows)
        pm = [wmean(sig(z), L) for z, L in zip(zs, labs)]
        order[n]["bmean"].append(np.mean(pm))
        for i in range(len(pm)):
            for j in range(i + 1, len(pm)):
                if rates[i] != rates[j]:
                    order[n]["conc"][0] += (pm[i] - pm[j]) * (rates[i] - rates[j]) > 0
                    order[n]["conc"][1] += 1
        for z, L in zip(zs, labs):
            order[n]["auc"].append(auc(z, np.array([np.mean(l) for l in L])))


def comps(n, keys):
    v = np.concatenate([stages[n][k] for k in keys])
    m = v.mean(0)
    return np.array([m[0] - m[1], m[1] - m[2], m[2] - m[3], m[3], m[0]])  # calibration, benchmark, subject, ranking, total


keys = list(bench)
print(f"benchmarks {len(keys)}, pairs {sum(len(v) for v in bench.values())}, items {sum(len(ts) for v in bench.values() for _, ts in v)}")
print(f"\n{'':26} raw Brier | parts of own Brier removed: calibration  benchmark  subject | left (ranking)")
for n in names:
    c = comps(n, keys)
    print(f"{n:26} {c[4]:.4f}    | {c[0]:+.4f}  {c[1]:+.4f}  {c[2]:+.4f} | {c[3]:.4f}")
rng = np.random.default_rng(0)
draws = [[keys[i] for i in rng.integers(0, len(keys), len(keys))] for _ in range(2000)]
print("\ngap to Opus, split [95% CI over benchmarks]: total = calibration + benchmark + subject + ranking")
lab = ["total", "calibration", "benchmark", "subject", "ranking"]
for n in names[1:]:
    d = lambda ks: (lambda x, o: np.array([x[4] - o[4], x[0] - o[0], x[1] - o[1], x[2] - o[2], x[3] - o[3]]))(comps(n, ks), comps(OPUS, ks))
    est = d(keys); bs = np.array([d(ks) for ks in draws]); lo, hi = np.quantile(bs, [.025, .975], axis=0)
    print(f"{n:26} " + "  ".join(f"{l} {e:+.4f} [{a:+.4f}, {b:+.4f}]" for l, e, a, b in zip(lab, est, lo, hi)))
print("\norderings (truth: pair success rate over all responses)")
for n in names:
    o = order[n]
    print(f"{n:26} benchmarks Spearman {spearmanr(o['bmean'], brate)[0]:.3f} | subjects within benchmark {o['conc'][0] / o['conc'][1]:.3f} of {o['conc'][1]} subject pairs | items within pair AUC {np.nanmean(o['auc']):.3f}")

# Split by whether Opus recognized the benchmark (runs/know-recognized-20261030.csv, match True) and by release date.
import pandas as pd
import memory_generalization as mg
rec = pd.read_csv(f"{ROOT}/runs/know-recognized-20261030.csv")
opus_rec = set(rec[(rec.run == "know-opus-low") & (rec.match == True)].benchmark)
cat = pd.read_parquet(mg.CATALOG)
date = dict(zip(cat.folder, pd.to_datetime(cat.release_date, errors="coerce")))
groups = {"Opus recognizes": [k for k in keys if k in opus_rec], "Opus does not": [k for k in keys if k not in opus_rec],
          "released before 2025": [k for k in keys if pd.notna(date.get(k)) and date[k] < pd.Timestamp("2025-01-01")],
          "released 2025 on": [k for k in keys if pd.notna(date.get(k)) and date[k] >= pd.Timestamp("2025-01-01")]}
print("\ngap to Opus by group [95% CI over benchmarks in the group]: benchmark part | ranking part | total")
for g, ks in groups.items():
    for n in names[1:]:
        d = lambda kk: comps(n, kk) - comps(OPUS, kk)
        est = d(ks); bs = np.array([d([ks[i] for i in rng.integers(0, len(ks), len(ks))]) for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975], axis=0)
        print(f"{g:22} ({len(ks):3} benchmarks) {n:26} benchmark {est[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}] | ranking {est[3]:+.4f} [{lo[3]:+.4f}, {hi[3]:+.4f}] | total {est[4]:+.4f} [{lo[4]:+.4f}, {hi[4]:+.4f}]")
