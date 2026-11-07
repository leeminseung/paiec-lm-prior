"""Effort high minus low for GLM-5.3 and GLM-5.3 Flash, paired by pair (11/3, computation only).

Each run gets its own constants (a, b, sigma_B, sigma_P, tau) fitted on the calibration set exactly as in
report/figures/forecaster_constants.py, then held-out (three proxies) and all five proxies are scored with our model
and with the forecast alone. Differences per pair, 95% intervals from 4,000 bootstrap draws over benchmarks
(ladder_table.ci).
"""
import sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from forecaster_constants import benchmarks, pair_rows, fit, fit_tau  # noqa: E402
from hier_ablations import offsets, pair_alc  # noqa: E402
from ladder_table import PROXIES, ci  # noqa: E402
from make_figures import ROOT  # noqa: E402
from common import load_exam  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import predict_all  # noqa: E402

MODELS = {"GLM-5.3": "glm53", "GLM-5.3 Flash": "glm53flash"}
rows, clusters, held = {}, [], []
consts = {}
for name, tag in MODELS.items():
    for eff in ("low", "high"):
        run = f"or-mdbfull-{tag}-{eff}"
        data, pr = benchmarks(run), pair_rows(run)
        est = fit(data, pr); consts[(name, eff)] = (*est, fit_tau(data, *est))
        print(name, eff, "a %+.2f b %.2f sB %.2f sP %.2f tau %.2f" % consts[(name, eff)], flush=True)
for label, ex, short in PROXIES:
    exam = load_exam(f"{ROOT}/exams/{ex}")
    for (name, eff), (a, b, sb, sp, tau) in consts.items():
        tag = MODELS[name]
        zt, ze = offsets(exam, load(f"{ROOT}/runs/or-{short}-{tag}-{eff}-t"), load(f"{ROOT}/runs/or-{short}-{tag}-{eff}-e"))
        alone = {(bb, pi, k): float(1 / (1 + np.exp(-v))) for bb in (0, 1, 3, 7, 15, 31) for (pi, k), v in zt.items()}
        rows.setdefault((name, eff, "alone"), []).extend(pair_alc(exam, alone))
        zt = {k: a + b * v for k, v in zt.items()}; ze = {k: a + b * v for k, v in ze.items()}
        rows.setdefault((name, eff, "ours"), []).extend(pair_alc(exam, predict_all(exam, zt, ze, sb, sp, tau=tau)))
    clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
    held += [label.startswith("held-out")] * len(exam["pairs"])
held = np.array(held)
for name in MODELS:
    for kind in ("ours", "alone"):
        hi, lo = np.array(rows[(name, "high", kind)]), np.array(rows[(name, "low", kind)])
        line = f"{name:14} {kind:5} high minus low"
        for nm, sel in (("held-out", held), ("five", np.ones_like(held))):
            l, h = ci((hi - lo)[sel], [c for c, s in zip(clusters, sel) if s])
            line += f" | {nm} {lo[sel].mean():.4f} -> {hi[sel].mean():.4f}, {(hi - lo)[sel].mean():+.4f} [{l:+.4f}, {h:+.4f}]"
        print(line, flush=True)
