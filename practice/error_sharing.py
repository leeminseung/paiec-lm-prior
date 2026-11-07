"""Is the judge's pair-level error shared by the subjects of a benchmark (a benchmark-side cause, such as how the
benchmark grades) or specific to each subject?

On the measurement-db judge set (make_judge_set.py, up to 3 pairs per benchmark, 7 judged items per pair):
  error e = pair mean prediction (mean judgment over the judged items) - pair success rate (all the pair's responses),
  on the probability scale and on the logit scale.
Reported, over benchmarks with at least 2 pairs:
  - intraclass correlation: share of the error variance that lies between benchmarks (one-way random effects, method
    of moments), raw and with the within-pair sampling noise of the 7-item mean removed;
  - leave-one-out: how much of the squared error goes away when each pair's error is predicted by the mean error of
    the other pairs on the same benchmark (what other subjects' revealed outcomes could fix at test time);
  - sign agreement: share of benchmarks where every pair errs in the same direction (|e| > 0.05 for all);
  - the same intraclass correlation for the pair success rates themselves, for reference.
Intervals: 4,000 draws resampling benchmarks.

Usage: python -I error_sharing.py <judge_set_dir> <judge_run_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))


def icc(groups, noise=None):
    """groups: list of arrays (one per benchmark). noise: list of arrays of within-pair noise variances, or None."""
    n = np.array([len(g) for g in groups], dtype=float)
    N, G = n.sum(), len(groups)
    allv = np.concatenate(groups)
    means = np.array([g.mean() for g in groups])
    sw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - G)
    msb = (n * (means - allv.mean()) ** 2).sum() / (G - 1)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    if noise is not None:
        sw = max(sw - np.concatenate(noise).mean(), 1e-9)
    sb = max((msb - sw) / n0, 0.0) if noise is None else max((msb - (sw + np.concatenate(noise).mean())) / n0, 0.0)
    return sb / (sb + sw)


def loo_removed(groups):
    num = sum(sum((g[i] - np.delete(g, i).mean()) ** 2 for i in range(len(g))) for g in groups)
    den = sum((g ** 2).sum() for g in groups)
    return 1 - num / den


def sign_agree(groups, thr=0.05):
    ok = [g for g in groups if np.all(np.abs(g) > thr)]
    return np.mean([np.all(g > 0) or np.all(g < 0) for g in ok]) if ok else np.nan, len(ok)


def main():
    exam = load_exam(sys.argv[1])
    pj = load(sys.argv[2])
    by_bench = {}
    for pi, pair in enumerate(exam["pairs"]):
        ps = np.array([pj[(pi, t["key"])] for t in pair["targets"] if (pi, t["key"]) in pj])
        if len(ps) < 3:
            continue
        rate, n = pair["pair_success_rate"], pair["n_graded"]
        mp = ps.mean()
        noise_p = ps.var(ddof=1) / len(ps) + rate * (1 - rate) / n
        dz = 1 / max(mp * (1 - mp), 1e-3)
        by_bench.setdefault(pair["benchmark_name"], []).append(
            (mp - rate, logit(mp) - logit((rate * n + .5) / (n + 1)), noise_p, noise_p * dz ** 2, rate))
    multi = {b: np.array(v) for b, v in by_bench.items() if len(v) >= 2}
    print(f"benchmarks with >= 2 judged pairs: {len(multi)}, pairs {sum(len(v) for v in multi.values())}")
    keys = sorted(multi)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(4000)]

    def report(label, col, noise_col=None):
        groups = [multi[k][:, col] for k in keys]
        noise = [multi[k][:, noise_col] for k in keys] if noise_col is not None else None
        stats = {"ICC raw": lambda gs, ns: icc(gs),
                 "ICC noise removed": (lambda gs, ns: icc(gs, ns)) if noise is not None else None,
                 "LOO removed": lambda gs, ns: loo_removed(gs)}
        parts = []
        for name, f in stats.items():
            if f is None:
                continue
            v = f(groups, noise)
            bs = [f([groups[i] for i in d], [noise[i] for i in d] if noise is not None else None) for d in draws]
            parts.append(f"{name} {v:.2f} [{np.quantile(bs, .025):.2f}, {np.quantile(bs, .975):.2f}]")
        print(f"  {label:26} " + " | ".join(parts))

    print("judge error, pair mean prediction - pair success rate")
    report("probability scale", 0, 2)
    report("logit scale", 1, 3)
    sa, m = sign_agree([multi[k][:, 0] for k in keys])
    print(f"  same sign on every pair of the benchmark (all |e| > 0.05): {sa:.2f} of {m} benchmarks "
          f"(independent errors would give about 0.25 for 3 pairs, 0.5 for 2)")
    print("pair success rate itself (benchmark share of where pairs sit)")
    report("probability scale", 4)
    e = np.concatenate([multi[k][:, 0] for k in keys])
    print(f"mean |error| {np.abs(e).mean():.3f}, mean error {e.mean():+.3f}")
    gz = [multi[k][:, 1] for k in keys]
    nz = np.concatenate([multi[k][:, 3] for k in keys]).mean()
    n = np.array([len(g) for g in gz], dtype=float)
    N, G = n.sum(), len(gz)
    sw = sum(((g - g.mean()) ** 2).sum() for g in gz) / (N - G)
    msb = (n * (np.array([g.mean() for g in gz]) - np.concatenate(gz).mean()) ** 2).sum() / (G - 1)
    sb = (msb - sw) / ((N - (n ** 2).sum() / N) / (G - 1))
    print(f"logit-scale components: between-benchmark SD {np.sqrt(max(sb, 0)):.2f}, within-benchmark SD "
          f"{np.sqrt(sw):.2f} raw, {np.sqrt(max(sw - nz, 0)):.2f} with sampling noise removed, mean {np.concatenate(gz).mean():+.2f}")


if __name__ == "__main__":
    main()
