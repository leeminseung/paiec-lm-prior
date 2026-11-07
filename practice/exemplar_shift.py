"""Exemplars로 평균 맞추기, simulated on a practice exam (proxy evaluation) with known answers.

At budget b, for each pair: the pair's own first b few-shot exemplars have budget-0 judgments z_e (logits) and
revealed outcomes y_e. The shift
    delta = argmax over d of  sum_e [ y_e log s(z_e + d) + (1 - y_e) log(1 - s(z_e + d)) ]  -  d^2 / (2 tau^2)
is applied to every target judgment of that pair: q = s(z_target + delta). Budget 0 has no exemplars, so delta = 0.
tau (how far a few exemplars may move the pair) is chosen with one benchmark held out at a time.

Compared with the 고정 갱신 공식 (round-2 predictor) and with the mean-matched score (an oracle ceiling that moves each
pair by the shift that fits the evaluation answers). A "blend" variant averages the logits of the shift prediction
and the 고정 갱신 공식 prediction.

Usage: python -I exemplar_shift.py <exam_dir> <target_judgments_run_dir> <exemplar_judgments_run_dir>
Missing judgments (refusals) use 0.575, as the submission does.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import logit
from eval_prior_combo import ALPHA, A0, M, counts

TAUS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def load(run_dir):
    d = {}
    for line in open(os.path.join(run_dir, "predictions.jsonl")):
        r = json.loads(line)
        if r.get("budget", 0) == 0 and r.get("probability") is not None:
            d[(r["pair"], r["key"])] = r["probability"]
    return d


def fit_shift(z, y, tau):
    if len(z) == 0:
        return 0.0
    d = 0.0
    for _ in range(60):
        q = 1 / (1 + np.exp(-(z + d)))
        g = np.sum(y - q) - d / tau ** 2
        h = -np.sum(q * (1 - q)) - 1 / tau ** 2
        step = g / h
        d -= step
        if abs(step) < 1e-9:
            break
    return float(d)


def main():
    exam = load_exam(sys.argv[1])
    p_t, p_e = load(sys.argv[2]), load(sys.argv[3])
    pairs = exam["pairs"]
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, pair in enumerate(pairs) for t in pair["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    z_t = np.array([logit(p_t.get((pi, k), M)) for _, pi, k in calls])
    bench = [pair["anon_benchmark_id"] for pair in pairs]
    print(f"targets {len(calls) // len(BUDGETS)}, target judgments {sum(1 for _, pi, k in calls[:len(calls) // 6] if (pi, k) in p_t)},"
          f" exemplar judgments {len(p_e)} of {sum(len(p['acquisitions']) for p in pairs)}")

    # the 고정 갱신 공식 (round-2 predictor)
    formula = np.empty(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s, n = cnt[(b, pi)]
        center = (s + A0 * p_t.get((pi, k), M)) / (n + A0)
        formula[i] = (x + ALPHA * center) / (kk + ALPHA)

    # shift predictions for every tau
    shift = {}
    for tau in TAUS:
        delta = {}
        for pi, pair in enumerate(pairs):
            for b in BUDGETS:
                ex = pair["acquisitions"][:b]
                z = np.array([logit(p_e.get((pi, a["key"]), M)) for a in ex])
                y = np.array([a["label"] for a in ex], float)
                delta[(pi, b)] = fit_shift(z, y, tau)
        shift[tau] = 1 / (1 + np.exp(-(z_t + np.array([delta[(pi, b)] for b, pi, _ in calls]))))
    blend = {tau: 1 / (1 + np.exp(-(logit(shift[tau]) + logit(formula)) / 2)) for tau in TAUS}

    # oracle ceiling: each pair moved by the shift that best fits its evaluation answers (same at every budget)
    ceiling = np.empty(len(calls))
    for pi, pair in enumerate(pairs):
        idx = [i for i, (b, p, k) in enumerate(calls) if p == pi and b == 0]
        zz = z_t[idx]
        ns, nf = S.ns[idx], S.nf[idx]
        best = min(np.linspace(-6, 6, 1201), key=lambda d: np.sum(ns * (1 / (1 + np.exp(-(zz + d))) - 1) ** 2
                                                                 + nf * (1 / (1 + np.exp(-(zz + d)))) ** 2))
        for i, (b, p, k) in enumerate(calls):
            if p == pi:
                ceiling[i] = 1 / (1 + np.exp(-(z_t[i] + best)))

    def pair_alc(pred):
        """{pair: ALC} with the competition weights (responses weighted within the pair)."""
        out = {}
        for pi in range(len(pairs)):
            tot = 0.0
            for w, b in zip(ALC_WEIGHTS, BUDGETS):
                m = (S.pair == pi) & (np.array([c[0] for c in calls]) == b)
                e = np.sum(S.ns[m] * (pred[m] - 1) ** 2 + S.nf[m] * pred[m] ** 2) / np.sum(S.ns[m] + S.nf[m])
                tot += w * e
            out[pi] = tot
        return out

    def held_out(family):
        """Pick tau on the other benchmarks for each benchmark, return the composed prediction and the picks."""
        per = {tau: pair_alc(family[tau]) for tau in TAUS}
        pred, picks = np.empty(len(calls)), {}
        for bm in sorted(set(bench)):
            others = [pi for pi in range(len(pairs)) if bench[pi] != bm]
            tau = min(TAUS, key=lambda t: np.mean([per[t][pi] for pi in others]))
            picks[bm] = tau
            for pi in range(len(pairs)):
                if bench[pi] == bm:
                    m = S.pair == pi
                    pred[m] = family[tau][m]
        return pred, picks

    show = lambda name, arr: print(f"  {name:56} ALC {S.alc(arr)[0]:.4f} | by budget {[round(float(c), 3) for c in S.alc(arr)[1]]}")
    print("\nwhole exam, competition ALC (lower is better)")
    show("고정 갱신 공식 (round-2 predictor)", formula)
    for tau in TAUS:
        show(f"exemplar shift, tau {tau:g} (fixed for all, in-sample)", shift[tau])
    shift_ho, picks = held_out(shift)
    show("exemplar shift, tau picked with benchmark held out", shift_ho)
    print(f"    tau picked per held-out benchmark: {picks}")
    blend_ho, picks_b = held_out(blend)
    show("blend of shift and formula, tau picked held out", blend_ho)
    print(f"    tau picked per held-out benchmark: {picks_b}")
    show("ceiling: mean-matched with evaluation answers (oracle)", ceiling)

    draws = S.draws(4000)
    print("\ndifference from the 고정 갱신 공식 (negative = better) [95%]:")
    for name, arr in (("exemplar shift, held out", shift_ho), ("blend, held out", blend_ho)):
        d = S.alc(arr)[0] - S.alc(formula)[0]
        bs = np.array([S.alc(arr, dr)[0] - S.alc(formula, dr)[0] for dr in draws])
        print(f"  {name:28} {d:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")
    pa_f, pa_s = pair_alc(formula), pair_alc(shift_ho)
    print("\nper pair ALC, formula -> shift (held out):")
    for pi, pair in enumerate(pairs):
        print(f"  p{pi} {pair['benchmark_name'][:26]:26} x {pair['subject']['normalized_name'][:24]:24} {pa_f[pi]:.3f} -> {pa_s[pi]:.3f}")


if __name__ == "__main__":
    main()


def deviation_variant(exam_dir, target_dir, exemplar_dir):
    """고정 갱신 공식 keeps the level; the target's judgment adds only its deviation from the exemplars' average
    judgment:  logit q = logit(formula) + a_b * (z_target - mean_e z_e)  for b >= 1 (budget 0 unchanged).
    a_b is chosen per budget with one benchmark held out."""
    exam = load_exam(exam_dir)
    p_t, p_e = load(target_dir), load(exemplar_dir)
    pairs = exam["pairs"]
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, pair in enumerate(pairs) for t in pair["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    bvec = np.array([c[0] for c in calls])
    z_t = np.array([logit(p_t.get((pi, k), M)) for _, pi, k in calls])
    formula = np.empty(len(calls))
    zbar = np.zeros(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s, n = cnt[(b, pi)]
        center = (s + A0 * p_t.get((pi, k), M)) / (n + A0)
        formula[i] = (x + ALPHA * center) / (kk + ALPHA)
        if b > 0:
            zbar[i] = np.mean([logit(p_e.get((pi, a["key"]), M)) for a in pairs[pi]["acquisitions"][:b]])
    dev = np.where(bvec > 0, z_t - zbar, 0.0)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    bench = [pair["anon_benchmark_id"] for pair in pairs]

    def brier_pair_budget(pred):
        out = {}
        for pi in range(len(pairs)):
            for b in BUDGETS:
                m = (S.pair == pi) & (bvec == b)
                out[(pi, b)] = np.sum(S.ns[m] * (pred[m] - 1) ** 2 + S.nf[m] * pred[m] ** 2) / np.sum(S.ns[m] + S.nf[m])
        return out

    cand = {a: 1 / (1 + np.exp(-(logit(formula) + a * dev))) for a in grid}
    tables = {a: brier_pair_budget(cand[a]) for a in grid}
    pred, picks = formula.copy(), {}
    for bm in sorted(set(bench)):
        others = [pi for pi in range(len(pairs)) if bench[pi] != bm]
        for b in BUDGETS[1:]:
            a = min(grid, key=lambda g: np.mean([tables[g][(pi, b)] for pi in others]))
            picks.setdefault(b, []).append(a)
            m = np.isin(S.pair, [pi for pi in range(len(pairs)) if bench[pi] == bm]) & (bvec == b)
            pred[m] = cand[a][m]
    show = lambda name, arr: print(f"  {name:56} ALC {S.alc(arr)[0]:.4f} | by budget {[round(float(c), 3) for c in S.alc(arr)[1]]}")
    print("\nformula level + judgment deviation from the exemplars' average")
    show("고정 갱신 공식", formula)
    for a in grid[1:]:
        show(f"deviation weight {a:g} at every budget (in-sample)", cand[a])
    show("deviation weight picked per budget, benchmark held out", pred)
    print("    picks per budget (one per held-out benchmark):", {b: v for b, v in picks.items()})
    draws = S.draws(4000)
    d = S.alc(pred)[0] - S.alc(formula)[0]
    bs = np.array([S.alc(pred, dr)[0] - S.alc(formula, dr)[0] for dr in draws])
    print(f"  difference from the formula {d:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")


if __name__ == "__main__" and os.environ.get("DEVIATION"):
    deviation_variant(*sys.argv[1:4])
