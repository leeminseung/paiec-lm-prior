"""Hierarchical model with a benchmark-specific slope on the forecast (the report's and submission's model from 11/2, D26).

    logit P(y_si = 1) = (1 + beta_B) z_si + u_B + v_sB,
    u_B ~ N(0, tb^2), v_sB ~ N(0, ts^2), beta_B ~ N(0, tau^2),

with z the forecast's logit (offset). tau = 0 gives the hierarchical model of hier_llm.py. For each benchmark and
budget the posterior of (u_B, beta_B, v_1..v_k) given the revealed outcomes is approximated at its mode (damped
Newton, Laplace), and a target is predicted with the probit approximation over the linear predictor's variance,
which now includes z^2 Var(beta) and the covariances. At budget 0 the mean is the forecast itself, but the slope's
prior variance widens the predictive variance, so extreme forecasts are pulled a little more toward 0.5.
"""
import numpy as np

from common import BUDGETS

sig = lambda e: 0.5 * (1 + np.tanh(0.5 * e))


def posterior_slope(obs, k, tb, ts, tau, nu=None):
    """obs: list of (member index, offset z, outcome 0/1). Returns mode and covariance of [u, beta, v_1..v_k].
    nu: if set, u has a Student-t prior with nu degrees of freedom and scale tb instead of a normal one."""
    prec = np.array([1 / tb ** 2, 1 / max(tau, 1e-6) ** 2] + [1 / ts ** 2] * k)

    def prior(t):  # log prior, its gradient and its negative curvature (diagonal)
        lp, g, c = -0.5 * np.sum(prec[1:] * t[1:] ** 2), -prec * t, prec.copy()
        if nu is None:
            lp -= 0.5 * prec[0] * t[0] ** 2
        else:
            d = nu * tb ** 2 + t[0] ** 2
            lp -= 0.5 * (nu + 1) * np.log1p(t[0] ** 2 / (nu * tb ** 2))
            g[0], c[0] = -(nu + 1) * t[0] / d, (nu + 1) / d
        return lp, g, c

    th = np.zeros(2 + k)
    if not obs:
        return th, np.diag(1 / prior(th)[2])
    m = np.array([o[0] for o in obs]); z = np.array([o[1] for o in obs], float); y = np.array([o[2] for o in obs], float)
    X = np.zeros((len(z), 2 + k)); X[:, 0] = 1; X[:, 1] = z; X[np.arange(len(z)), 2 + m] = 1
    f = lambda t: np.sum(y * (z + X @ t) - np.logaddexp(0, z + X @ t)) + prior(t)[0]
    for _ in range(200):
        p = sig(z + X @ th)
        _, gp, cp = prior(th)
        g = X.T @ (y - p) + gp
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(cp)
        step = np.linalg.solve(H, g); f0 = f(th); lam = 1.0
        while f(th + lam * step) < f0 and lam > 1e-6:
            lam /= 2
        th = th + lam * step
        if np.abs(lam * step).max() < 1e-9:
            break
    p = sig(z + X @ th)
    H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prior(th)[2])
    return th, np.linalg.inv(H)


def predict_all_slope(exam, zt, ze, tb, ts, tau, nu=None):
    """Same interface as hier_llm.predict_all plus tau. Returns {(b, pi, key): probability}."""
    pairs = exam["pairs"]
    bench = {}
    for pi, p in enumerate(pairs):
        bench.setdefault(p["anon_benchmark_id"], []).append(pi)
    out = {}
    for b in BUDGETS:
        for members in bench.values():
            obs = [(j, ze[(pi, a["key"])], a["label"]) for j, pi in enumerate(members) for a in pairs[pi]["acquisitions"][:b]]
            th, C = posterior_slope(obs, len(members), tb, ts, tau, nu)
            for j, pi in enumerate(members):
                for t in pairs[pi]["targets"]:
                    z = zt[(pi, t["key"])]
                    x = np.zeros(2 + len(members)); x[0] = 1; x[1] = z; x[2 + j] = 1
                    mean, var = float(x @ th), float(x @ C @ x)
                    out[(b, pi, t["key"])] = float(sig((z + mean) / np.sqrt(1 + np.pi * var / 8)))
    return out
