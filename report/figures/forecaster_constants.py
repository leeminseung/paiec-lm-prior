"""Per-forecaster constants of the hierarchical model, fitted on the calibration set, and held-out ALC with them.

For each forecaster, the same logistic mixed model is fitted to its budget-0 forecasts q on the calibration set
(exams/mdb-judge-1027: 121 benchmarks, 332 pairs, 2,324 items; each item counts once, with its success rate over its
recorded responses as a fractional outcome, because one benchmark repeats items up to 10,000 times and would otherwise
hold 97% of the responses):
    logit P(y = 1) = a + b logit q + u_B + v_sB,   u_B ~ N(0, sigma_B^2),  v_sB ~ N(0, sigma_P^2),
in two parts that are iterated until they agree:
  - sigma_B and sigma_P by the moment method of the report (error_sharing.py): each pair's logit error
    logit(mean recalibrated forecast) - logit(pair success rate over all its responses), split into between- and
    within-benchmark variance after removing the sampling noise of seven-item means;
  - a and b by maximum likelihood with sigma fixed, using the Laplace approximation of each benchmark's marginal
    likelihood (mode and curvature from hier_llm.posterior);
  - then tau, the SD of a benchmark-specific slope on the offset (D26), by the same Laplace marginal likelihood with
    a, b and sigma fixed (hier_slope.posterior_slope).
A joint maximum-likelihood fit of all four drives sigma_P to 0, because seven single-response items per pair barely
identify pair-level spread, while the pair success rate over all responses does. Items without a forecast are left out.
Intervals for a and b: bootstrap draws over benchmarks with sigma held at the full-data values.
Then held-out ALC (three proxies) with offsets a + b logit q and the fitted sigmas, with the difference from the
Beta-binomial baseline and from the forecast alone (4,000 bootstrap draws over benchmarks, seed 0), and each
forecaster's budget-0 Brier skill score on the whole calibration set.

Usage: python -I forecaster_constants.py [draws]
"""
import os
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_ablations import offsets, pair_alc  # noqa: E402
from ladder_table import PROXIES, baseline, ci  # noqa: E402
from make_figures import ROOT  # noqa: E402
sys.path.insert(0, "./practice")
from common import load_exam  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import posterior, predict_all  # noqa: E402
from hier_slope import posterior_slope  # noqa: E402

CALIB_RUNS = {"Opus 5.5, low": "mdb-judge-opus-low", "GLM-5.3 Flash, low": "or-mdbfull-glm53flash-low",
              "DeepSeek V4.1 Flash, low": "or-mdbfull-dsv41flash-low"}
EXTRA_RUNS = {"GLM-5.3, low": "or-mdbfull-glm53-low", "GLM-5.3, high": "or-mdbfull-glm53-high",
              "GLM-5.3 Flash, high": "or-mdbfull-glm53flash-high", "Sonnet 5.5, low": "mdb-judge-sonnet-low",
              "Haiku 5.5, low": "mdb-judge-haiku-low", "GPT-6.1 Sol, low": "or-mdbfull-gpt61sol-low",
              "GPT-6.1 Terra, low": "or-mdbfull-gpt61terra-low", "GPT-6 Luna, low": "or-mdbfull-gpt6luna-low"}
if os.environ.get("FC_EXTRA"):  # stronger open forecasters and effort high (11/2), added only when asked so the paper's
    # table stays reproducible; FC_EXTRA=1 adds all, or a ';'-separated list of names adds those
    want = os.environ["FC_EXTRA"]
    CALIB_RUNS.update({k: v for k, v in EXTRA_RUNS.items() if want == "1" or k in want.split(";")})
PROXY_RUNS = {"Opus 5.5, low": lambda s: {"r02": ("r02-opus-low-fb", "r02-exemplar-opus-low-fb"),
                                         "pc": ("pc-sub-opus-low", "pc-sub-exemplar-opus-low")}.get(
                                             s, (f"{s}-sub-opus-low", f"{s}-sub-exemplar-opus-low")),
              "GLM-5.3 Flash, low": lambda s: (f"or-{s}-glm53flash-low-t", f"or-{s}-glm53flash-low-e"),
              "DeepSeek V4.1 Flash, low": lambda s: (f"or-{s}-dsv41flash-low-t", f"or-{s}-dsv41flash-low-e"),
              "GLM-5.3, low": lambda s: (f"or-{s}-glm53-low-t", f"or-{s}-glm53-low-e"),
              "GLM-5.3, high": lambda s: (f"or-{s}-glm53-high-t", f"or-{s}-glm53-high-e"),
              "GLM-5.3 Flash, high": lambda s: (f"or-{s}-glm53flash-high-t", f"or-{s}-glm53flash-high-e"),
              "Sonnet 5.5, low": lambda s: (f"{s}-sub-sonnet-low", f"{s}-sub-exemplar-sonnet-low"),
              "Haiku 5.5, low": lambda s: (f"{s}-sub-haiku-low", f"{s}-sub-exemplar-haiku-low"),
              "GPT-6.1 Sol, low": lambda s: (f"or-{s}-gpt61sol-low-t", f"or-{s}-gpt61sol-low-e"),
              "GPT-6.1 Terra, low": lambda s: (f"or-{s}-gpt61terra-low-t", f"or-{s}-gpt61terra-low-e"),
              "GPT-6 Luna, low": lambda s: (f"or-{s}-gpt6luna-low-t", f"or-{s}-gpt6luna-low-e")}
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
# FC_EXCLUDE_HELDOUT=1 fits the constants without the calibration-set benchmarks that also appear in the held-out proxies
# (a leakage check for the held-out ALC of Table 2, 11/5).
EXCLUDE = ({p["benchmark_name"] for ex in ("proxy-1028-1", "proxy-1028-2", "proxy-1028-3")
            for p in load_exam(f"{ROOT}/exams/{ex}")["pairs"]} if os.environ.get("FC_EXCLUDE_HELDOUT") else set())


def pair_rows(run):
    """benchmark -> list of (logit forecasts of the pair's items, pair success rate, graded responses)."""
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
    pj = load(f"{ROOT}/runs/{run}")
    out = {}
    for pi, p in enumerate(exam["pairs"]):
        if p["benchmark_name"] in EXCLUDE:
            continue
        zs = [float(logit(pj[(pi, t["key"])])) for t in p["targets"] if (pi, t["key"]) in pj]
        if len(zs) >= 3:
            out.setdefault(p["benchmark_name"], []).append((np.array(zs), p["pair_success_rate"], p["n_graded"]))
    return out


def sigma_moments(rows, a, b):
    """The report's moment estimates of sigma_B and sigma_P for forecasts recalibrated as sigmoid(a + b logit q)."""
    groups, noise = [], []
    for prs in rows.values():
        if len(prs) < 2:
            continue
        e = []
        for zs, r, n in prs:
            ps = 1 / (1 + np.exp(-(a + b * zs))); mp = ps.mean()
            e.append(logit(mp) - logit((r * n + .5) / (n + 1)))
            noise.append((ps.var(ddof=1) / len(ps) + r * (1 - r) / n) / max(mp * (1 - mp), 1e-3) ** 2)
        groups.append(np.array(e))
    nz = np.mean(noise)
    n = np.array([len(x) for x in groups], float); N, G = n.sum(), len(groups)
    sw = sum(((x - x.mean()) ** 2).sum() for x in groups) / (N - G)
    msb = (n * (np.array([x.mean() for x in groups]) - np.concatenate(groups).mean()) ** 2).sum() / (G - 1)
    sb = (msb - sw) / ((N - (n ** 2).sum() / N) / (G - 1))
    return float(np.sqrt(max(sb, 0))), float(np.sqrt(max(sw - nz, 0)))


def benchmarks(run):
    """benchmark -> (members, list of (member index, logit q, item success rate))."""
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
    pj = load(f"{ROOT}/runs/{run}")
    out = {}
    for pi, p in enumerate(exam["pairs"]):
        if p["benchmark_name"] in EXCLUDE:
            continue
        b = out.setdefault(p["benchmark_name"], ([], []))
        j = len(b[0]); b[0].append(pi)
        for t in p["targets"]:
            if (pi, t["key"]) in pj:
                z = float(logit(pj[(pi, t["key"])]))
                b[1].append((j, z, float(np.mean(t["labels"]))))
    return [v for v in out.values() if v[1]]


def neg_marginal(params, data):
    a, b, lsb, lsp = params
    sb, sp = np.exp(lsb), np.exp(lsp)
    total = 0.0
    for members, obs in data:
        o = [(j, a + b * z, y) for j, z, y in obs]
        th, C = posterior(o, members, sb, sp)
        idx = np.array([x[0] for x in o]); off = np.array([x[1] for x in o]); y = np.array([x[2] for x in o])
        eta = off + th[0] + th[1 + idx]
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        var = np.array([sb ** 2] + [sp ** 2] * len(members))
        lp = -0.5 * np.sum(th ** 2 / var) - 0.5 * np.sum(np.log(2 * np.pi * var))
        sign, logdet = np.linalg.slogdet(C)
        total += ll + lp + 0.5 * len(th) * np.log(2 * np.pi) + 0.5 * logdet
    return -total


def fit_ab(data, sb, sp, start=(0.0, 1.0)):
    f = lambda ab: neg_marginal((ab[0], ab[1], np.log(sb), np.log(sp)), data)
    return optimize.minimize(f, np.array(start), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4}).x


def neg_marginal_slope(tau, data, a, b, sb, sp):
    """Minus the Laplace log marginal likelihood of the slope model, logit P = (1 + beta_B)(a + b z) + u_B + v_sB,
    beta_B ~ N(0, tau^2), with a, b, sigma fixed (hier_slope.posterior_slope gives the mode and covariance)."""
    total = 0.0
    for members, obs in data:
        o = [(j, a + b * z, y) for j, z, y in obs]
        th, C = posterior_slope(o, len(members), sb, sp, tau)
        idx = np.array([x[0] for x in o]); off = np.array([x[1] for x in o]); y = np.array([x[2] for x in o])
        eta = off * (1 + th[1]) + th[0] + th[2 + idx]
        prec = np.array([1 / sb ** 2, 1 / max(tau, 1e-6) ** 2] + [1 / sp ** 2] * len(members))
        total += (np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * np.sum(prec * th ** 2) + 0.5 * np.sum(np.log(prec))
                  + 0.5 * np.linalg.slogdet(C)[1])
    return -total


def fit_tau(data, a, b, sb, sp):
    """tau by maximum Laplace marginal likelihood: a grid from 0 to 1, then a golden-section refinement."""
    grid = np.linspace(0, 1, 21)
    vals = [neg_marginal_slope(t, data, a, b, sb, sp) for t in grid]
    i = int(np.argmin(vals))
    if i == 0:
        return 0.0
    r = optimize.minimize_scalar(lambda t: neg_marginal_slope(t, data, a, b, sb, sp),
                                 bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded")
    return float(r.x)


def fit(data, rows, rounds=4):
    a, b = 0.0, 1.0
    for _ in range(rounds):
        sb, sp = sigma_moments(rows, a, b)
        # floor: the moment estimate of sigma_P can hit 0 at the start values (a = 0, b = 1) for a badly calibrated
        # forecaster on a subset (GPT-6.1 Sol without the held-out benchmarks); the fitted values are far above it
        a, b = fit_ab(data, max(sb, 0.05), max(sp, 0.05), (a, b))
    return float(a), float(b), *sigma_moments(rows, a, b)


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rng = np.random.default_rng(0)
    consts = {}
    for name, run in CALIB_RUNS.items():
        data, rows = benchmarks(run), pair_rows(run)
        est = fit(data, rows)
        bs = [fit_ab([data[i] for i in rng.integers(0, len(data), len(data))], est[2], est[3], est[:2]) for _ in range(draws)]
        lo, hi = np.quantile(np.array(bs), [.025, .975], axis=0) if draws else ((np.nan,) * 2, (np.nan,) * 2)
        tau = fit_tau(data, *est)
        consts[name] = (*est, tau)
        print(f"{name:26} {len(data)} benchmarks, {sum(len(o) for _, o in data)} items | a {est[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]"
              f" | b {est[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}] | sigma_B {est[2]:.2f} | sigma_P {est[3]:.2f} | tau {tau:.2f}", flush=True)
        if name.startswith("Opus"):
            print(f"{'':26} tau with the submitted a = 0, b = 1: {fit_tau(data, 0.0, 1.0, est[2], est[3]):.2f}", flush=True)
    rows, clusters, held = {}, [], []
    for label, ex, short in PROXIES:
        exam = load_exam(f"{ROOT}/exams/{ex}")
        rows.setdefault("baseline", []).extend(pair_alc(exam, baseline(exam)))
        for name, (a, b, sb, sp, tau) in consts.items():
            t, e = PROXY_RUNS[name](short)
            zt, ze = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
            zt = {k: a + b * v for k, v in zt.items()}; ze = {k: a + b * v for k, v in ze.items()}
            rows.setdefault(name, []).extend(pair_alc(exam, predict_all(exam, zt, ze, sb, sp, tau=tau)))
            raw, _ = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
            alone = {(bb, pi, k): float(1 / (1 + np.exp(-v))) for bb in (0, 1, 3, 7, 15, 31) for (pi, k), v in raw.items()}
            rows.setdefault((name, "alone"), []).extend(pair_alc(exam, alone))
        clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
        held += [label.startswith("held-out")] * len(exam["pairs"])
    held = np.array(held); base = np.array(rows["baseline"])
    print(f"\nBeta-binomial baseline: held-out {base[held].mean():.4f}, five {base.mean():.4f}")
    for name in consts:
        v = np.array(rows[name]); d = v - base; line = f"{name:26} own constants"
        for nm, sel in (("held-out", held), ("five", np.ones_like(held))):
            lo, hi = ci(d[sel], [c for c, s in zip(clusters, sel) if s])
            line += f" | {nm} ALC {v[sel].mean():.4f} minus baseline {d[sel].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(line)
        d = v - np.array(rows[(name, "alone")]); line = f"{name:26} ours minus forecast alone"
        for nm, sel in (("held-out", held), ("five", np.ones_like(held))):
            lo, hi = ci(d[sel], [c for c, s in zip(clusters, sel) if s])
            line += f" | {nm} {d[sel].mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (alone {np.array(rows[(name, 'alone')])[sel].mean():.4f})"
        print(line)
    import memory_generalization as mg
    full = load_exam(f"{ROOT}/exams/mdb-judge-1027")
    print("\nbudget-0 Brier skill score on the whole calibration set (pairs equal)")
    for name, run in CALIB_RUNS.items():
        st = mg.summarize(mg.pair_stats(full, load(f"{ROOT}/runs/{run}"), lambda pi, t: True))
        print(f"  {name:26} BSS {st['skill']:.3f}, Brier {st['brier']:.4f}, AUC {st['auc']:.3f}")


if __name__ == "__main__":
    main()
