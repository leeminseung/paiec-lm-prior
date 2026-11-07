"""Effort medium minus low for Sonnet 5.5 and Haiku 5.5, high minus low for GPT-6.1 Sol (11/6) and Opus (reference) on the 847-item
one-pair-per-benchmark set (11/4, computation only).

Same scoring as practice/memory_generalization.py (Brier, skill against the base rate 0.627, pair-level gap, within-pair
AUC), restricted for each comparison to the items both runs answered. Paired differences with 95% intervals from 4,000
draws resampling the 121 pairs (one per benchmark).
"""
import sys
import numpy as np
sys.path.insert(0, "./practice")
import memory_generalization as mg  # noqa: E402
from common import load_exam  # noqa: E402

exam = load_exam(mg.SUBSET)
RUNS = {"Opus low": ("mdb-judge-opus-low", "full"), "Opus high": ("thinking-opus-high-onepair", "thinking"),
        "Sonnet low": ("onepair-sub-sonnet-5-5-low", "subset"), "Sonnet medium": ("onepair-sub-sonnet-5-5-medium", "subset"),
        "Haiku low": ("onepair-sub-haiku-5-5-low", "subset"), "Haiku medium": ("onepair-sub-haiku-5-5-medium", "subset"),
        "Sol low": ("or-mdbfull-gpt61sol-low", "full"), "Sol high": ("or-onepair-gpt61sol-high", "subset")}
J = {k: mg.judgments(r, kind, exam) for k, (r, kind) in RUNS.items()}
rng = np.random.default_rng(0)
for lo, hi in (("Opus low", "Opus high"), ("Sonnet low", "Sonnet medium"), ("Haiku low", "Haiku medium"), ("Sol low", "Sol high")):
    common = set(J[lo]) & set(J[hi])
    keep = lambda pi, t: (pi, t["key"]) in common
    a, b = mg.pair_stats(exam, J[lo], keep), mg.pair_stats(exam, J[hi], keep)
    sa, sb = mg.summarize(a), mg.summarize(b)
    ds = []
    for _ in range(4000):
        idx = rng.integers(0, len(a), len(a))
        x, y = mg.summarize([a[i] for i in idx]), mg.summarize([b[i] for i in idx])
        ds.append([y[k] - x[k] for k in ("skill", "auc", "gap")])
    q = np.quantile(np.array(ds), [.025, .975], axis=0)
    print(f"{hi} minus {lo} ({len(common)} items, {len(a)} pairs): "
          + " | ".join(f"{k} {sa[k]:.3f} -> {sb[k]:.3f}, {sb[k] - sa[k]:+.3f} [{q[0][i]:+.3f}, {q[1][i]:+.3f}]"
                       for i, k in enumerate(("skill", "auc", "gap"))))
