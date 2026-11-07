"""Main comparison table across proxy evaluations: the non-LLM round-1 rule against the LLM method.

Rows per exam and pooled: round 1 (고정 갱신 공식 with the constant 0.575, no LLM), J2 (the public counting rule),
Opus low + 공식, Opus low + 공식 + item 차이 되살리기. Each pair's ALC uses the competition budget weights over all
its evaluation responses; pairs are equal. Pooled intervals for differences resample benchmarks (all pairs of a drawn
benchmark together, across the given exams), 4,000 draws.

Usage: python -I main_table.py <label>=<exam_dir>:<target_run>:<exemplar_run> [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import logit
from common import ALC_WEIGHTS, BUDGETS, load_exam
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load

W = np.array(ALC_WEIGHTS)
sig = lambda z: 1 / (1 + np.exp(-z))
ROWS = ["round 1 (no LLM)", "J2 (no LLM)", "Opus low + formula", "Opus low + formula + restore"]


def pair_alcs(exam_dir, t_run, e_run):
    exam = load_exam(exam_dir); pairs = exam["pairs"]; pt, pe = load(t_run), load(e_run)
    cnt = counts(exam)
    out = {r: np.zeros(len(pairs)) for r in ROWS}
    for pi, p in enumerate(pairs):
        curves = {r: [] for r in ROWS}
        for b in BUDGETS:
            x, k, s, n = cnt[(b, pi)]
            se = {r: 0.0 for r in ROWS}; nn = 0
            zb = np.mean([logit(pe.get((pi, a["key"]), M)) for a in p["acquisitions"][:b]]) if b > 0 else 0.0
            for t in p["targets"]:
                p0 = pt.get((pi, t["key"]), M)
                r1 = (x + ALPHA * (s + A0 * M) / (n + A0)) / (k + ALPHA)
                j2 = (x + 2 * (s + 2) / (n + 4)) / (k + 2)
                f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (k + ALPHA)
                q = float(sig(logit(f) + k / (k + 4.0) * (logit(p0) - zb))) if b > 0 else f
                for r, v in zip(ROWS, (r1, j2, f, q)):
                    se[r] += sum((v - y) ** 2 for y in t["labels"])
                nn += len(t["labels"])
            for r in ROWS:
                curves[r].append(se[r] / nn)
        for r in ROWS:
            out[r][pi] = float(np.array(curves[r]) @ W)
    return out, [p["benchmark_name"] for p in pairs]


def main():
    allrows, clusters = {r: [] for r in ROWS}, []
    for arg in sys.argv[1:]:
        label, spec = arg.split("=", 1)
        exam_dir, t_run, e_run = spec.split(":")
        out, benches = pair_alcs(exam_dir, t_run, e_run)
        print(f"{label:10} " + " | ".join(f"{r} {out[r].mean():.4f}" for r in ROWS))
        for r in ROWS:
            allrows[r].extend(out[r])
        clusters.extend((label, b) for b in benches)
    keys = sorted(set(clusters))
    idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(4000)]
    ref = np.array(allrows[ROWS[0]])
    print(f"pooled over {len(sys.argv) - 1} exams: {len(ref)} pairs, {len(keys)} benchmarks")
    for r in ROWS:
        v = np.array(allrows[r]); d = v - ref
        bs = [np.concatenate([d[idx[keys[j]]] for j in dr]).mean() for dr in draws]
        print(f"  {r:30} ALC {v.mean():.4f} | minus round 1 {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")


if __name__ == "__main__":
    main()
