"""Checks behind Section 5.1 (recognition) of the report, on the 847-item subset (121 benchmarks, one pair each).

1. Difference in differences. Benchmarks Opus recognizes may simply be harder for every forecaster, so a recognition
   advantage for Opus shows as Opus losing less skill on them than a forecaster that does not recognize them:
       DiD(X) = [skill_Opus(recognized) - skill_Opus(not)] - [skill_X(recognized) - skill_X(not)],
   recognition by Opus low, X each other base model at its own setting. Positive favors a recognition advantage.
   Also pooled: Opus's gap minus the mean gap of all the others (the average DiD), with its own interval (added 11/6).
   Also recalibrated (added 11/6): the forecasters with calibration-set constants (a, b) in the report's Table 2 are
   recalibrated as sigmoid(a + b logit q) and compared again with Opus low (a = 0, b = 1), to see how much of a DiD
   comes from miscalibration rather than recognition. Fresh draws (seed 0) per forecaster.
2. The same split with recognition by name only (grade "benchmark"), leaving out answers that named only the source
   dataset (grade "source").
3. Release date, on the whole calibration set (exams/mdb-judge-1027, 332 pairs). Share of the base rate's mean
   absolute pair-level error that the Opus low forecast removes,
       1 - mean |mean forecast - pair success rate| / mean |0.627 - pair success rate|  over a window's pairs,
   the success rate taken over all of the pair's responses, for benchmarks released before 2025, in three later
   windows and from 2025 on, with how many benchmarks of each window Opus recognizes.
Skill = 1 - Brier / Brier of the base rate 0.627 on the same responses. Intervals: 4,000 bootstrap draws over
benchmarks (seed 0), resampled within each group and shared by the forecasters compared.

Usage: python -I recognition_checks.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "./practice")
import memory_generalization as mg  # noqa: E402
from common import load_exam  # noqa: E402

OTHERS = ["Sonnet low", "Haiku low", "GLM-5.3 Flash", "Muse Spark Open", "DeepSeek V4.1 Flash low",
          "MiniMax M3.1 Flash", "MiniMax M3", "GPT-6.1 Sol low", "GPT-6.1 Terra low", "GPT-6 Luna low"]
# calibration-set constants (a, b) at effort low, as in the report's Table 2
AB = {"Sonnet low": (0.07, 0.85), "Haiku low": (0.15, 0.73), "DeepSeek V4.1 Flash low": (-0.29, 0.63),
      "GPT-6.1 Sol low": (-0.89, 0.61), "GPT-6.1 Terra low": (-0.51, 0.57), "GPT-6 Luna low": (-0.57, 0.46)}
WINDOWS = [("before 2025", None, "2025-01-01"), ("2025 Jan-Jun", "2025-01-01", "2025-07-01"),
           ("2025 Jul-Dec", "2025-07-01", "2026-01-01"), ("2026", "2026-01-01", None)]
DRAWS = 4000


def by_bench(rows):
    """benchmark -> (Brier, base Brier, gap, base gap); one pair per benchmark in this subset."""
    return {r[0]: r for r in rows}


def skill(stats, benches):
    b = np.mean([stats[x][1] for x in benches]); base = np.mean([stats[x][2] for x in benches])
    return 1 - b / base


def main():
    exam = load_exam(mg.SUBSET)
    cat = pd.read_parquet(mg.CATALOG)
    names = {f: [n, bid] for f, n, bid in zip(cat.folder, cat.name, cat.benchmark_id)}
    date = dict(zip(cat.folder, pd.to_datetime(cat.release_date, errors="coerce")))
    grades_all = {(r["run"], r["benchmark"]): r["match"] for r in map(json.loads, open(mg.GRADES)) if r.get("match")}
    rec, _ = mg.recognition("know-opus-low", names, grades_all)
    name_only = {b: ok and grades_all.get(("know-opus-low", b)) != "source" for b, ok in rec.items()}
    stats = {}
    for jn in ["Opus low"] + OTHERS:
        run, kind, _ = mg.JUDGES[jn]
        stats[jn] = by_bench(mg.pair_stats(exam, mg.judgments(run, kind, exam), lambda pi, t: True))
    rng = np.random.default_rng(0)

    for label, flag in (("recognized by Opus (name or source dataset)", rec), ("recognized by Opus, name only", name_only)):
        print(f"\n{label}: {sum(flag.get(b, False) for b in stats['Opus low'])} of {len(stats['Opus low'])} benchmarks")
        for jn in OTHERS:
            common = [b for b in stats["Opus low"] if b in stats[jn]]
            yes = [b for b in common if flag.get(b)]; no = [b for b in common if not flag.get(b)]
            gap = lambda s, y, n: skill(s, y) - skill(s, n)
            did = gap(stats["Opus low"], yes, no) - gap(stats[jn], yes, no)
            ds = []
            for _ in range(DRAWS):
                y = [yes[i] for i in rng.integers(0, len(yes), len(yes))]
                n = [no[i] for i in rng.integers(0, len(no), len(no))]
                ds.append(gap(stats["Opus low"], y, n) - gap(stats[jn], y, n))
            lo, hi = np.quantile(ds, [.025, .975])
            print(f"  Opus minus {jn:24} ({len(yes)} / {len(no)}): Opus gap {gap(stats['Opus low'], yes, no):+.3f}, "
                  f"{jn} gap {gap(stats[jn], yes, no):+.3f}, DiD {did:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        common = [b for b in stats["Opus low"] if all(b in stats[j] for j in OTHERS)]
        yes = [b for b in common if flag.get(b)]; no = [b for b in common if not flag.get(b)]
        pooled = lambda y, n: gap(stats["Opus low"], y, n) - np.mean([gap(stats[j], y, n) for j in OTHERS])
        prng = np.random.default_rng(0); ds = []
        for _ in range(DRAWS):
            ds.append(pooled([yes[i] for i in prng.integers(0, len(yes), len(yes))],
                             [no[i] for i in prng.integers(0, len(no), len(no))]))
        lo, hi = np.quantile(ds, [.025, .975])
        print(f"  Opus minus the mean of all {len(OTHERS)} others ({len(yes)} / {len(no)}): "
              f"DiD {pooled(yes, no):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nrecalibrated forecasts, recognized by Opus (name or source dataset)")
    sig = lambda z: 1 / (1 + np.exp(-z))
    lg = lambda p: np.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    for jn, (a, b) in AB.items():
        run, kind, _ = mg.JUDGES[jn]
        pred = {k: float(sig(a + b * lg(v))) for k, v in mg.judgments(run, kind, exam).items()}
        st = by_bench(mg.pair_stats(exam, pred, lambda pi, t: True))
        common = [x for x in stats["Opus low"] if x in st]
        yes = [x for x in common if rec.get(x)]; no = [x for x in common if not rec.get(x)]
        gap = lambda s, y, n: skill(s, y) - skill(s, n)
        r2 = np.random.default_rng(0); ds = []
        for _ in range(DRAWS):
            y = [yes[i] for i in r2.integers(0, len(yes), len(yes))]; n = [no[i] for i in r2.integers(0, len(no), len(no))]
            ds.append(gap(stats["Opus low"], y, n) - gap(st, y, n))
        lo, hi = np.quantile(ds, [.025, .975])
        print(f"  Opus minus {jn:24} recalibrated (a {a:+.2f}, b {b:.2f}): "
              f"DiD {gap(stats['Opus low'], yes, no) - gap(st, yes, no):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nrelease windows, Opus low, calibration set (all pairs, pair success rate over all its responses): "
          "share of the base rate's mean absolute pair-level error removed")
    full = load_exam(f"{mg.ROOT}/exams/mdb-judge-1027")
    low = mg.load(f"{mg.ROOT}/runs/mdb-judge-opus-low")
    pairs = {}
    for pi, p in enumerate(full["pairs"]):
        v = [low[(pi, t["key"])] for t in p["targets"] if (pi, t["key"]) in low]
        if v:
            r = p["pair_success_rate"]
            pairs.setdefault(p["benchmark_name"], []).append((abs(np.mean(v) - r), abs(mg.BASE - r)))
    for label, a, b in WINDOWS + [("2025 on", "2025-01-01", None)]:
        sel = [x for x in pairs if not pd.isna(date.get(x)) and (a is None or date[x] >= pd.Timestamp(a))
               and (b is None or date[x] < pd.Timestamp(b))]
        if not sel:
            continue
        red = lambda s: 1 - np.mean([e for x in s for e, _ in pairs[x]]) / np.mean([g for x in s for _, g in pairs[x]])
        ds = [red([sel[i] for i in rng.integers(0, len(sel), len(sel))]) for _ in range(DRAWS)]
        lo, hi = np.quantile(ds, [.025, .975])
        print(f"  {label:13} {len(sel):3d} benchmarks, {sum(len(pairs[x]) for x in sel):3d} pairs, Opus recognizes "
              f"{sum(rec.get(x, False) for x in sel):3d}: {red(sel):.0%} [{lo:.0%}, {hi:.0%}]")
    print(f"  without a release date: {sum(pd.isna(date.get(x)) for x in pairs)} benchmarks")


if __name__ == "__main__":
    main()
