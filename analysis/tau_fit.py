"""How much does the forecaster's trustworthiness (slope on its logit) vary across benchmarks? (11/2, computation only)

Model per benchmark B: logit P(y) = a + (b + beta_B) z + u_B + v_sB, with z = logit of the Opus 5.5 low forecast,
u_B ~ N(0, sB^2), v_sB ~ N(0, sP^2), beta_B ~ N(0, tau^2). a, b, sB, sP fixed at the Opus calibration-set values
(9-98: a -0.11, b 1.01, sB 1.09, sP 0.46). tau by the Laplace-approximated marginal likelihood over a grid; each item
counts once with its success rate as a fractional outcome (as in forecaster_constants.py). Also per-benchmark
slopes fitted alone (intercepts per pair, slope per benchmark, light ridge) on the calibration set and on the
MathArena August sets only (the September sets stay untouched).
"""
import json, sys
import numpy as np
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from forecaster_constants import benchmarks
from exemplar_shift import load

A, B, SB, SP = -0.11, 1.01, 1.09, 0.46
sig = lambda e: 0.5 * (1 + np.tanh(0.5 * e))   # overflow-free logistic


def laplace(z, y, member, k, tau, a=A, b=B, sb=SB, sp=SP):
    """Latent theta = [u, beta, v_1..v_k]; returns log marginal (Laplace) and the mode."""
    n = len(z)
    X = np.zeros((n, 2 + k)); X[:, 0] = 1; X[:, 1] = z; X[np.arange(n), 2 + member] = 1
    prec = np.array([1 / sb ** 2, 1 / max(tau, 1e-6) ** 2] + [1 / sp ** 2] * k)
    off = a + b * z
    th = np.zeros(2 + k)
    f = lambda t: np.sum(y * (off + X @ t) - np.logaddexp(0, off + X @ t)) - 0.5 * np.sum(prec * t ** 2)
    for _ in range(200):
        eta = off + X @ th; p = sig(eta)
        g = X.T @ (y - p) - prec * th
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
        step = np.linalg.solve(H, g); f0 = f(th); lam = 1.0
        while f(th + lam * step) < f0 and lam > 1e-6:   # damped Newton: never move downhill
            lam /= 2
        th = th + lam * step
        if np.abs(lam * step).max() < 1e-9:
            break
    eta = off + X @ th
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    lp = -0.5 * np.sum(prec * th ** 2) + 0.5 * np.sum(np.log(prec)) - 0.5 * len(th) * np.log(2 * np.pi)
    p = sig(eta); H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
    return ll + lp + 0.5 * len(th) * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(H)[1], th


def total(data, tau):
    tot = 0.0
    for members, obs in data:
        z = np.array([o[1] for o in obs]); y = np.array([o[2] for o in obs]); m = np.array([o[0] for o in obs])
        tot += laplace(z, y, m, len(members), tau)[0]
    return tot


data = benchmarks("mdb-judge-opus-low")
print(f"calibration set: {len(data)} benchmarks, {sum(len(o) for _, o in data)} items")
grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
prof = [(t, total(data, t)) for t in grid]
best = max(prof, key=lambda x: x[1])
for t, v in prof:
    print(f"  tau {t:4.2f}  log marginal {v:10.2f}  {'<- max' if t == best[0] else ''}  (vs tau 0: {v - prof[0][1]:+.2f})")
lr = 2 * (best[1] - prof[0][1])
print(f"likelihood ratio vs tau = 0: {lr:.2f} (boundary test: p about {0.5 * __import__('math').erfc((lr / 2) ** .5):.3f} for lr > 0)")

# per-benchmark slope alone, for a picture of the spread
def slope_alone(members, obs, ridge=0.5):
    z = np.array([o[1] for o in obs]); y = np.array([o[2] for o in obs]); m = np.array([o[0] for o in obs]); k = len(members)
    X = np.zeros((len(z), 1 + k)); X[:, 0] = z; X[np.arange(len(z)), 1 + m] = 1
    th = np.zeros(1 + k); th[0] = 1.0
    prec = np.array([ridge] + [0.25] * k); prior = np.zeros(1 + k); prior[0] = 1.0
    f = lambda t: np.sum(y * (X @ t) - np.logaddexp(0, X @ t)) - 0.5 * np.sum(prec * (t - prior) ** 2)
    for _ in range(300):
        p = sig(X @ th)
        g = X.T @ (y - p) - prec * (th - prior); H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
        st = np.linalg.solve(H, g); f0 = f(th); lam = 1.0
        while f(th + lam * st) < f0 and lam > 1e-6:
            lam /= 2
        th = th + lam * st
        if np.abs(lam * st).max() < 1e-9: break
    return th[0]

cal_sl = np.array([slope_alone(m, o) for m, o in data])
print(f"\nper-benchmark slope alone, calibration set: median {np.median(cal_sl):.2f}, 10th-90th percentile {np.quantile(cal_sl, .1):.2f} to {np.quantile(cal_sl, .9):.2f}, share below 0.5: {np.mean(cal_sl < .5):.2f}")

ma = json.load(open("./exams/matharena-1101/exam.json")); pq = load("./runs/ma-opus-low")
lg = lambda p: float(np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995))))
for s in ("matharena_arxivmath_0826", "matharena_brokenarxiv_0826"):
    pis = [pi for pi, p in enumerate(ma["pairs"]) if p["benchmark_name"] == s]
    obs = [(j, lg(pq[(pi, t["key"])]), float(np.mean(t["labels"]))) for j, pi in enumerate(pis) for t in ma["pairs"][pi]["targets"] if (pi, t["key"]) in pq]
    print(f"MathArena {s}: {len(pis)} pairs, slope alone {slope_alone(pis, obs):.2f}; log marginal tau 0 vs 0.5 vs 1.0: " +
          ", ".join(f"{laplace(np.array([o[1] for o in obs]), np.array([o[2] for o in obs]), np.array([o[0] for o in obs]), len(pis), t)[0]:.1f}" for t in (0.0, 0.5, 1.0)))
