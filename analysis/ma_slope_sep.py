"""Final check (once) of the benchmark-specific slope on the MathArena September sets (11/2), same procedure as ma_slope_aug.py.

Same splits as ma_update.py (31 revealed, the rest evaluated, 5 seeds), restricted to arxivmath-0826 and
brokenarxiv-0826 (33 pairs). Compared: Beta-binomial baseline, current hierarchical model (hier_llm.predict_all,
sigma 1.09/0.46), the slope model with tau 0.3 (the calibration-set estimate), and tau 0 in the slope code as a check
that it reproduces the current model. tau 0.6 is shown as a sensitivity only.
"""
import json, random, sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from hier_ablations import pair_alc
from ladder_table import baseline
from common import BUDGETS
from exemplar_shift import load
from hier_llm import predict_all
from hier_slope import predict_all_slope

R = "."
full = json.load(open(f"{R}/exams/matharena-1101/exam.json"))
AUG = ("matharena_arxivmath_0926", "matharena_brokenarxiv_0926", "matharena_arxivlean_0926")  # the September sets, final check
pq = load(f"{R}/runs/ma-opus-low")
lg = lambda p: float(np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995))))
keep = [pi for pi, p in enumerate(full["pairs"]) if p["benchmark_name"] in AUG]


def split(seed):
    pairs = []
    for pi in keep:
        p = full["pairs"][pi]
        ts = [t for t in p["targets"] if (pi, t["key"]) in pq]
        rnd = random.Random(f"{seed}|{pi}"); rnd.shuffle(ts)   # same seeds as ma_update.py
        pairs.append(dict(p, acquisitions=[{"key": t["key"], "item": t["item"], "label": int(t["labels"][0]), "src": pi} for t in ts[:31]],
                          targets=[dict(t, src=pi) for t in ts[31:]]))
    return dict(full, pairs=pairs)


def curve(ex, pred):
    return np.array([np.mean([sum((pred[(b, i, t["key"])] - y) ** 2 for t in p["targets"] for y in t["labels"]) /
                              sum(len(t["labels"]) for t in p["targets"]) for i, p in enumerate(ex["pairs"])]) for b in BUDGETS])


names = ["baseline", "current model", "slope model, tau 0 (check)", "slope model, tau 0.3", "slope model, tau 0.6 (sensitivity)"]
alc = {n: [] for n in names}; cur = {n: [] for n in names}
for seed in range(5):
    ex = split(seed)
    zt = {(i, t["key"]): lg(pq[(t["src"], t["key"])]) for i, p in enumerate(ex["pairs"]) for t in p["targets"]}
    ze = {(i, a["key"]): lg(pq[(a["src"], a["key"])]) for i, p in enumerate(ex["pairs"]) for a in p["acquisitions"]}
    preds = {"baseline": baseline(ex), "current model": predict_all(ex, zt, ze, 1.09, 0.46),
             "slope model, tau 0 (check)": predict_all_slope(ex, zt, ze, 1.09, 0.46, 0.0),
             "slope model, tau 0.3": predict_all_slope(ex, zt, ze, 1.09, 0.46, 0.3),
             "slope model, tau 0.6 (sensitivity)": predict_all_slope(ex, zt, ze, 1.09, 0.46, 0.6)}
    for n, pr in preds.items():
        alc[n].append(pair_alc(ex, pr)); cur[n].append(curve(ex, pr))
print(f"MathArena September sets: {len(keep)} pairs, 5 splits")
print(f"{'':36} ALC    | Brier at budget " + "  ".join(f"{b:>5}" for b in BUDGETS))
for n in names:
    print(f"{n:36} {np.mean(alc[n]):.4f} | " + "  ".join(f"{x:.3f}" for x in np.mean(cur[n], axis=0)))
rng = np.random.default_rng(0)
ref = np.mean(alc["current model"], axis=0); b0 = np.mean(alc["baseline"], axis=0)
for n in names[2:]:
    d = np.mean(alc[n], axis=0) - ref; e = np.mean(alc[n], axis=0) - b0
    bs = lambda v: np.quantile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)], [.025, .975])
    lo, hi = bs(d); lo2, hi2 = bs(e)
    print(f"{n}: minus current {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | minus baseline {e.mean():+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
for s in AUG:
    idx = [i for i, pi in enumerate(keep) if full["pairs"][pi]["benchmark_name"] == s]
    print(f"  {s}: ALC baseline {np.mean(alc['baseline'], axis=0)[idx].mean():.4f} | current {ref[idx].mean():.4f} | tau 0.3 {np.mean(alc['slope model, tau 0.3'], axis=0)[idx].mean():.4f}")
