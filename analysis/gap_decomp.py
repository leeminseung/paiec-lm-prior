"""Split each forecaster's budget-0 Brier gap to Opus into calibration, pair level and within-pair ranking."""
import sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, "./report/figures")
sys.path.insert(0, "./practice")
from forecaster_constants import CALIB_RUNS, PROXY_RUNS, ROOT, logit
from hier_ablations import offsets
from ladder_table import PROXIES
from common import load_exam
from exemplar_shift import load

CONST = {"Opus 5.5, low": (-0.11, 1.01), "GLM-5.3 Flash, low": (0.00, 0.63), "DeepSeek V4.1 Flash, low": (-0.26, 0.62)}
sig = lambda z: 1 / (1 + np.exp(-z))

def pair_parts(z, labels, a, b):
    """z: raw logits of the pair's items; labels: list of response arrays. Brier raw, recalibrated, level-matched."""
    n = np.array([len(l) for l in labels], float); ym = np.array([np.mean(l) for l in labels])
    br = lambda p: sum(((p[j] - np.asarray(l)) ** 2).sum() for j, l in enumerate(labels)) / n.sum()
    zr = a + b * z
    target = (ym * n).sum() / n.sum()
    t = np.clip(target, 1e-4, 1 - 1e-4)
    c = brentq(lambda c: (sig(zr + c) * n).sum() / n.sum() - t, -30, 30)
    return br(sig(z)), br(sig(zr)), br(sig(zr + c))

def collect(pairs_iter):
    rows = {}
    for name, z, labels in pairs_iter:
        rows.setdefault(name, []).append(pair_parts(z, labels, *CONST[name]))
    return {k: np.array(v) for k, v in rows.items()}

def calib():
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
    common = None
    preds = {n: load(f"{ROOT}/runs/{r}") for n, r in CALIB_RUNS.items()}
    for pi, p in enumerate(exam["pairs"]):
        ts = [t for t in p["targets"] if all((pi, t["key"]) in pj for pj in preds.values())]
        if len(ts) < 2:
            continue
        for n, pj in preds.items():
            yield n, np.array([float(logit(pj[(pi, t["key"])])) for t in ts]), [t["labels"] for t in ts]

def heldout():
    for label, ex, short in PROXIES:
        if not label.startswith("held-out"):
            continue
        exam = load_exam(f"{ROOT}/exams/{ex}")
        zs = {}
        for n in CONST:
            t, e = PROXY_RUNS[n](short)
            zs[n], _ = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
        for pi, p in enumerate(exam["pairs"]):
            ts = [t for t in p["targets"] if all((pi, t["key"]) in zs[n] for n in CONST)]
            if len(ts) < 2:
                continue
            for n in CONST:
                yield n, np.array([zs[n][(pi, t["key"])] for t in ts]), [t["labels"] for t in ts]

for title, it in (("calibration set (a, b fitted here)", calib()), ("held-out proxies (a, b from calibration set)", heldout())):
    r = collect(it)
    print(f"\n{title}: pairs {len(r['Opus 5.5, low'])}")
    print(f"{'':26} raw    recal  level-matched | gap to Opus: total = calibration + pair level + ranking")
    o = r["Opus 5.5, low"].mean(0)
    for n, v in r.items():
        m = v.mean(0)
        tot = m[0] - o[0]; cal = (m[0] - m[1]) - (o[0] - o[1]); lvl = (m[1] - m[2]) - (o[1] - o[2]); rk = m[2] - o[2]
        print(f"{n:26} {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}       | {tot:+.4f} = {cal:+.4f} + {lvl:+.4f} + {rk:+.4f}")
