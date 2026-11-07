"""The 고정 갱신 공식 + item 차이 되살리기 rewritten as one hierarchical success model with LLM priors, compared with it.

    logit P(y = 1 | subject s, item i) = z_si + u_B + v_s
      z_si: logit of the judge's budget-0 judgment for (s, i) (targets from the target run, exemplars from the exemplar run)
      u_B ~ N(0, TB^2): a shift shared by every subject of the benchmark
      v_s ~ N(0, TS^2): the pair's own shift
TB and TS come from the measurement-db judge set (error_sharing.py: between-benchmark SD 1.09, within 0.46 on the
logit scale), not from any proxy evaluation. A second row puts a Student-t prior on u_B (3 degrees of freedom, scale
0.58: the best fit to the benchmark-mean judge errors there; a normal fits far worse), so the shared shift can follow
revealed outcomes quickly when they contradict the judge strongly. At budget 0 the shifts are zero and the probit approximation shrinks
the judgment's logit by 1 / sqrt(1 + pi (TB^2 + TS^2) / 8), about 0.80. At budget b the shifts
get a Laplace posterior from the first b exemplars of every pair on the benchmark, each exemplar entering with its own
judgment as its offset, so a success on an item the judge already thought easy moves the shift less. The target's
probability integrates the shift's uncertainty (probit approximation).

Ablations, for the analysis only (they use the pair's mean judgment over all its judged items, which a predict call
cannot see):
  - level only: z_si replaced by the pair's mean judgment logit (no item differences);
  - item differences only: z_si - pair mean + the measurement-db mean pair logit 0.07, with the spread of pair success
    rates as the prior (between 1.49, within 0.73).
Intervals: 4,000 draws resampling pairs, then items within pairs, for each row minus the current method.

Usage: python -I hier_llm.py <exam_dir> <target_run> <exemplar_run> [TB TS]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import logit
from common import BUDGETS, load_exam
from compare_runs import Sample
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load

sig = lambda z: 1 / (1 + np.exp(-z))
RATE_MEAN, RATE_TB, RATE_TS = 0.07, 1.49, 0.73
# SD of the benchmark-specific slope on the forecast (D26, 11/2): 0.3 is Opus's calibration-set estimate
# (report/figures/forecaster_constants.py). HIER_TAU=0 reproduces the model without the slope.
TAU = float(os.environ.get("HIER_TAU", "0.3"))


def posterior(obs, members, tb, ts, nu=None):
    """obs: list of (member index, offset, label). Returns mean and covariance of [u, v_0..v_{m-1}].
    nu: if set, u has a Student-t prior with nu degrees of freedom and scale tb instead of a normal one."""
    m = len(members)
    th = np.zeros(m + 1)
    prec = np.array([1 / tb ** 2] + [1 / ts ** 2] * m)
    idx = np.array([o[0] for o in obs], dtype=int)
    off = np.array([o[1] for o in obs], dtype=float)
    y = np.array([o[2] for o in obs], dtype=float)
    for _ in range(100):
        H = -np.diag(prec)
        g = -prec * th
        if nu is not None:  # heavy-tailed prior on the shared shift: curvature (nu+1)/(nu tb^2 + u^2), always positive
            c = nu * tb ** 2 + th[0] ** 2
            g[0] = -(nu + 1) * th[0] / c
            H[0, 0] = -(nu + 1) / c
        if len(obs):
            q = sig(off + th[0] + th[1 + idx])
            r, w = y - q, q * (1 - q)
            g[0] += r.sum()
            np.add.at(g, 1 + idx, r)
            H[0, 0] -= w.sum()
            for j in range(m):
                wj = w[idx == j].sum()
                H[0, 1 + j] -= wj
                H[1 + j, 0] -= wj
                H[1 + j, 1 + j] -= wj
        step = np.linalg.solve(H, -g)
        # Damped Newton: halve the step until the log posterior does not fall. The plain step diverged on a
        # MathArena set at budgets 15 and 31 (u ran to -92 and -203) when forecasts were far below the outcomes
        # (11/2); where the plain step converges the result is unchanged.
        f0, lam = _log_post(th, idx, off, y, prec, nu, tb), 1.0
        while _log_post(th + lam * step, idx, off, y, prec, nu, tb) < f0 and lam > 1e-6:
            lam /= 2
        th = th + lam * step
        if np.abs(lam * step).max() < 1e-8:
            break
    return th, np.linalg.inv(-H)


def _log_post(th, idx, off, y, prec, nu, tb):
    """Log posterior of [u, v_0..] up to a constant, for the step control in posterior()."""
    eta = off + th[0] + th[1 + idx] if len(off) else np.zeros(0)
    lp = -0.5 * np.sum(prec[1:] * th[1:] ** 2)
    lp += -0.5 * (nu + 1) * np.log1p(th[0] ** 2 / (nu * tb ** 2)) if nu is not None else -0.5 * prec[0] * th[0] ** 2
    return float(np.sum(y * eta - np.logaddexp(0, eta)) + lp)


def predict_all(exam, zt, ze, tb, ts, nu=None, tau=None):
    """zt[(pi, key)], ze[(pi, key)]: offsets for targets and exemplars. Returns {(b, pi, key): probability}.
    tau: SD of the benchmark-specific slope on the offset (hier_slope.py); None takes TAU, 0 gives the model without
    the slope (the report's model before 11/2)."""
    tau = TAU if tau is None else tau
    if tau > 0:
        from hier_slope import predict_all_slope
        return predict_all_slope(exam, zt, ze, tb, ts, tau, nu)
    pairs = exam["pairs"]
    bench = {}
    for pi, p in enumerate(pairs):
        bench.setdefault(p["anon_benchmark_id"], []).append(pi)
    out = {}
    for b in BUDGETS:
        for members in bench.values():
            obs = [(j, ze[(pi, a["key"])], a["label"]) for j, pi in enumerate(members)
                   for a in pairs[pi]["acquisitions"][:b]]
            th, C = posterior(obs, members, tb, ts, nu)
            for j, pi in enumerate(members):
                mean = th[0] + th[1 + j]
                var = C[0, 0] + C[1 + j, 1 + j] + 2 * C[0, 1 + j]
                for t in pairs[pi]["targets"]:
                    out[(b, pi, t["key"])] = float(sig((zt[(pi, t["key"])] + mean) / np.sqrt(1 + np.pi * var / 8)))
    return out


def main():
    exam = load_exam(sys.argv[1])
    pt, pe = load(sys.argv[2]), load(sys.argv[3])
    tb, ts = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (1.09, 0.46)
    pairs = exam["pairs"]
    zt = {(pi, t["key"]): logit(pt.get((pi, t["key"]), M)) for pi, p in enumerate(pairs) for t in p["targets"]}
    ze = {(pi, a["key"]): logit(pe.get((pi, a["key"]), M)) for pi, p in enumerate(pairs) for a in p["acquisitions"]}
    zbar = {pi: np.mean([zt[(pi, t["key"])] for t in p["targets"]] + [ze[(pi, a["key"])] for a in p["acquisitions"]])
            for pi, p in enumerate(pairs)}
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, p in enumerate(pairs) for t in p["targets"]]
    S = Sample(exam, calls)
    draws = S.draws(4000)
    cnt = counts(exam)
    cur = np.empty(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s, n = cnt[(b, pi)]
        p0 = pt.get((pi, k), M)
        f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (kk + ALPHA)
        if b > 0:
            zb = np.mean([ze[(pi, a["key"])] for a in pairs[pi]["acquisitions"][:b]])
            f = float(sig(logit(f) + b / (b + 4.0) * (zt[(pi, k)] - zb)))
        cur[i] = f
    rows = {"current (formula + restore)": cur}
    variants = {
        f"hierarchical, TB {tb} TS {ts}": (zt, ze, tb, ts),
        "hierarchical, heavy-tailed shared shift (t, 3 df, scale 0.58)": (zt, ze, 0.58, ts, 3.0),
        "level only (analysis)": ({k: zbar[k[0]] for k in zt}, {k: zbar[k[0]] for k in ze}, tb, ts),
        "item differences only (analysis)": ({k: v - zbar[k[0]] + RATE_MEAN for k, v in zt.items()},
                                             {k: v - zbar[k[0]] + RATE_MEAN for k, v in ze.items()}, RATE_TB, RATE_TS),
    }
    for name, (a, e, b1, b2, *nu) in variants.items():
        pr = predict_all(exam, a, e, b1, b2, *nu)
        rows[name] = np.array([pr[c] for c in calls])
    base = rows["current (formula + restore)"]
    for name, arr in rows.items():
        alc, curve = S.alc(arr)
        iv = ""
        if name != "current (formula + restore)":
            d = np.array([S.alc(arr, dr)[0] - S.alc(base, dr)[0] for dr in draws])
            iv = f" | vs current {alc - S.alc(base)[0]:+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]"
        print(f"  {name:36} ALC {alc:.4f} | {[round(float(c), 3) for c in curve]}{iv}")


if __name__ == "__main__":
    main()
