"""Adoption check of the benchmark-specific slope on the two validation proxies (11/2, computation only).

Same Opus 5.5 low forecasts and settings as the paper's main result (ladder_table OPUS runs, sigma 1.09/0.46,
offsets = logit q). Compared: current hierarchical model, slope model tau 0 (must match), slope model tau 0.3.
Differences per pair with 95% intervals from 4,000 bootstrap draws over benchmarks (ladder_table.ci).
"""
import sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from hier_ablations import offsets, pair_alc
from ladder_table import PROXIES, OPUS, baseline, ci
from make_figures import ROOT
from common import load_exam, BUDGETS
from exemplar_shift import load
from hier_llm import predict_all
from hier_slope import predict_all_slope

which = sys.argv[1] if len(sys.argv) > 1 else "validation"
rows = {k: [] for k in ("baseline", "current", "slope tau 0", "slope tau 0.3")}
curves = {k: [] for k in rows}; clusters = []
for label, ex, short in PROXIES:
    if not label.startswith(which):
        continue
    exam = load_exam(f"{ROOT}/exams/{ex}")
    t, e = OPUS[short]
    zt, ze = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
    preds = {"baseline": baseline(exam), "current": predict_all(exam, zt, ze, 1.09, 0.46),
             "slope tau 0": predict_all_slope(exam, zt, ze, 1.09, 0.46, 0.0),
             "slope tau 0.3": predict_all_slope(exam, zt, ze, 1.09, 0.46, 0.3)}
    for k, pr in preds.items():
        rows[k].extend(pair_alc(exam, pr))
        curves[k].append([np.mean([sum((pr[(b, pi, tt["key"])] - y) ** 2 for tt in p["targets"] for y in tt["labels"]) /
                                   sum(len(tt["labels"]) for tt in p["targets"]) for pi, p in enumerate(exam["pairs"])]) for b in BUDGETS])
    clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
print(f"{which} proxies: {len(clusters)} pairs in {len(set(clusters))} benchmarks")
for k in rows:
    print(f"  {k:14} ALC {np.mean(rows[k]):.4f} | Brier by budget " + " ".join(f"{x:.3f}" for x in np.mean(curves[k], axis=0)))
cur = np.array(rows["current"])
for k in ("slope tau 0", "slope tau 0.3", "baseline"):
    d = np.array(rows[k]) - cur; lo, hi = ci(d, clusters)
    print(f"  {k} minus current: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
