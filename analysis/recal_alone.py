"""Held-out ALC of each forecaster's recalibrated forecast without any update, sigmoid(a + b logit q) at every budget,
next to the raw forecast and our method (11/5, computation only; review finding: the constants vs the update).
Constants from fc_all_families.txt (forecaster_constants.py with FC_EXTRA, full calibration set); Opus with a = 0, b = 1.
"""
import sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from forecaster_constants import PROXY_RUNS  # noqa: E402
from hier_ablations import offsets, pair_alc  # noqa: E402
from ladder_table import PROXIES  # noqa: E402
from make_figures import ROOT  # noqa: E402
from common import load_exam  # noqa: E402
from exemplar_shift import load  # noqa: E402

AB = {"Opus 5.5, low": (0.0, 1.0), "Sonnet 5.5, low": (0.07, 0.85), "Haiku 5.5, low": (0.15, 0.73),
      "GPT-6.1 Sol, low": (-0.89, 0.61), "GPT-6.1 Terra, low": (-0.51, 0.57), "GPT-6 Luna, low": (-0.57, 0.46),
      "GLM-5.3, low": (-0.02, 0.65), "GLM-5.3 Flash, low": (0.00, 0.63), "DeepSeek V4.1 Flash, low": (-0.29, 0.63)}
sig = lambda z: 1 / (1 + np.exp(-z))
rows = {n: [] for n in AB}
for label, ex, short in PROXIES:
    if not label.startswith("held-out"):
        continue
    exam = load_exam(f"{ROOT}/exams/{ex}")
    for n, (a, b) in AB.items():
        t, e = PROXY_RUNS[n](short)
        zt, _ = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
        rows[n].extend(pair_alc(exam, {(bb, pi, k): float(sig(a + b * v)) for bb in (0, 1, 3, 7, 15, 31) for (pi, k), v in zt.items()}))
for n, v in rows.items():
    print(f"{n:26} held-out ALC, recalibrated forecast alone {np.mean(v):.4f}")
