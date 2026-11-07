"""MathArena with the hierarchical update (computation only, 11/2): how fast do revealed outcomes repair the
forecaster's level error on post-cutoff benchmarks?

Each MathArena pair's problems are split at random (seeded per pair and repeat) into an acquisition pool of 31
(revealed in order; the first run's 0/1 is the revealed outcome) and an evaluation pool (the rest, 22-26 problems,
scored on every run). Offsets are Opus 5.5 low budget-0 forecasts (`runs/ma-opus-low`) with the submitted a = 0,
b = 1; the model pools pairs of the same set as one benchmark, as on the proxies. Compared: the Beta-binomial
baseline, the forecast alone, and the hierarchical model with the calibration-set sigma (1.09, 0.46). A larger
benchmark sigma is shown as a sensitivity only. Five random splits; 4,000 bootstrap draws over pairs.
"""
import json
import random
import sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from hier_ablations import W, pair_alc
from ladder_table import baseline
from common import BUDGETS
from exemplar_shift import load
from hier_llm import predict_all

R = "."
base = json.load(open(f"{R}/exams/matharena-1101/exam.json"))
pq = load(f"{R}/runs/ma-opus-low")
lg = lambda p: float(np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995))))


def split(seed):
    pairs = []
    for pi, p in enumerate(base["pairs"]):
        ts = [t for t in p["targets"] if (pi, t["key"]) in pq]
        rnd = random.Random(f"{seed}|{pi}"); rnd.shuffle(ts)
        acq = [{"key": t["key"], "item": t["item"], "label": int(t["labels"][0]), "orig": pi} for t in ts[:31]]
        tg = [dict(t, orig=pi) for t in ts[31:]]
        pairs.append(dict(p, acquisitions=acq, targets=tg))
    return dict(base, pairs=pairs)


def curve(exam, pred):
    out = []
    for b in BUDGETS:
        v = []
        for pi, p in enumerate(exam["pairs"]):
            se = sum((pred[(b, pi, t["key"])] - y) ** 2 for t in p["targets"] for y in t["labels"])
            v.append(se / sum(len(t["labels"]) for t in p["targets"]))
        out.append(np.mean(v))
    return np.array(out)


res = {k: [] for k in ("baseline", "forecast alone", "hierarchical (1.09, 0.46)", "hierarchical (2.0, 0.46), sensitivity")}
curves = {k: [] for k in res}
for seed in range(5):
    ex = split(seed)
    zt = {(pi, t["key"]): lg(pq[(pi, t["key"])]) for pi, p in enumerate(ex["pairs"]) for t in p["targets"]}
    ze = {(pi, a["key"]): lg(pq[(pi, a["key"])]) for pi, p in enumerate(ex["pairs"]) for a in p["acquisitions"]}
    preds = {"baseline": baseline(ex),
             "forecast alone": {(b, pi, t["key"]): pq[(pi, t["key"])] for b in BUDGETS for pi, p in enumerate(ex["pairs"]) for t in p["targets"]},
             "hierarchical (1.09, 0.46)": predict_all(ex, zt, ze, 1.09, 0.46),
             "hierarchical (2.0, 0.46), sensitivity": predict_all(ex, zt, ze, 2.0, 0.46)}
    for k, pr in preds.items():
        res[k].append(pair_alc(ex, pr)); curves[k].append(curve(ex, pr))
print(f"{len(base['pairs'])} pairs, 5 random splits (31 revealed, 22-26 evaluated per pair)")
print(f"{'':40} ALC     | Brier at budget " + "  ".join(f"{b:>5}" for b in BUDGETS))
for k in res:
    a = np.mean(res[k], axis=0); c = np.mean(curves[k], axis=0)
    print(f"{k:40} {a.mean():.4f} | " + "  ".join(f"{x:.3f}" for x in c))
rng = np.random.default_rng(0)
b0 = np.mean(res["baseline"], axis=0)
for k in ("forecast alone", "hierarchical (1.09, 0.46)", "hierarchical (2.0, 0.46), sensitivity"):
    d = np.mean(res[k], axis=0) - b0
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print(f"{k} minus baseline: {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")
