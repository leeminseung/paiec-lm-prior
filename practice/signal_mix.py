"""Does a cheap item signal from the weak solvers improve Sonnet's budget-0 judgment? Whole practice exam, one
benchmark held out at a time.

Budget-0 prior q0: logit q0 = a0 + a1 logit(p_sonnet) + sum_j a_j feature_j, fitted on the other benchmarks
(Brier, each pair counted once); then the fixed round-2 formula turns q0 into predictions at every budget.
Features per item from the weak-solver run: log median output tokens of the attempts (Haiku 5.5, thinking on),
share of attempts that refused or hit the output limit, share of the most common short answer (0 when not
comparable, with an indicator), and, from the critic, majority_looks_right and its probability.

Usage: python -I signal_mix.py <exam_dir> <sonnet_run_dir> <solver_critic_run_dir>
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, load_exam
from compare_runs import Sample
from combo_fit import load_p, logit
from eval_prior_combo import ALPHA, A0, M, counts


def main():
    exam = load_exam(sys.argv[1])
    alone = load_p(sys.argv[2])
    run = sys.argv[3]
    att, cri = {}, {}
    for line in open(os.path.join(run, "attempts.jsonl")):
        r = json.loads(line)
        if r["setting"] == "on" and r["usage"]:
            att.setdefault((r["pair"], r["key"]), []).append(r)
    for line in open(os.path.join(run, "critic.jsonl")):
        r = json.loads(line)
        if r["variant"] == "ae":
            cri[(r["pair"], r["key"])] = r
    items = [(pi, t["key"]) for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    feats = {}
    for u in items:
        a = att.get(u, [])
        toks = [x["usage"].get("output_tokens", 0) for x in a] or [0]
        stops = np.mean([x["stop_reason"] in ("refusal", "max_tokens") for x in a]) if a else 0.0
        c = cri.get(u, {})
        ag = c.get("agreement") or {}
        top = ag.get("top_share")
        feats[u] = {"log_tokens": np.log1p(np.median(toks)), "stopped": stops,
                    "top_share": top if top is not None else 0.0, "has_short": float(top is not None),
                    "looks_right": c.get("majority_looks_right") if c.get("majority_looks_right") is not None else 0.5,
                    "critic_logit": logit(c["probability"]) if c.get("probability") is not None else logit(M)}
    variants = {"sonnet only (calibrated)": [],
                "+ solver tokens and stops": ["log_tokens", "stopped"],
                "+ tokens, stops, short-answer agreement": ["log_tokens", "stopped", "top_share", "has_short"],
                "+ critic majority_looks_right": ["looks_right"],
                "+ critic probability": ["critic_logit"]}
    calls = [(b, pi, t["key"]) for b in BUDGETS for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    S = Sample(exam, calls)
    cnt = counts(exam)
    bench_of = {u: exam["pairs"][u[0]]["anon_benchmark_id"] for u in items}
    resp = {u: (sum(t["labels"]), len(t["labels"]) - sum(t["labels"])) for u, t in
            (((pi, t["key"]), t) for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"])}
    pair_n = {}
    for u in items:
        pair_n[u[0]] = pair_n.get(u[0], 0) + sum(resp[u])
    base_p = np.array([alone.get((pi, k), M) for _, pi, k in calls])
    base_arr = None
    results = {}
    for name, cols in variants.items():
        X = np.array([[1.0, logit(alone.get(u, M))] + [feats[u][c] for c in cols] for u in items])
        mu, sd = X[:, 2:].mean(axis=0), X[:, 2:].std(axis=0) + 1e-9
        X[:, 2:] = (X[:, 2:] - mu) / sd
        ns = np.array([resp[u][0] for u in items], float)
        nf = np.array([resp[u][1] for u in items], float)
        w = np.array([1.0 / pair_n[u[0]] for u in items])
        q0 = {}
        for held in sorted(set(bench_of.values())):
            tr = np.array([bench_of[u] != held for u in items])
            th0 = np.zeros(X.shape[1]); th0[1] = 1.0

            def loss(th, tr=tr):
                q = 1 / (1 + np.exp(-(X[tr] @ th)))
                return float(np.sum(w[tr] * (ns[tr] * (q - 1) ** 2 + nf[tr] * q ** 2)) / np.sum(w[tr] * (ns[tr] + nf[tr]))
                             + 1e-4 * np.sum((th - th0) ** 2))

            th = minimize(loss, th0, method="L-BFGS-B").x
            for i, u in enumerate(items):
                if bench_of[u] == held:
                    q0[u] = float(1 / (1 + np.exp(-(X[i] @ th))))
        pred = np.empty(len(calls))
        for i, (b, pi, k) in enumerate(calls):
            x, kk, s_oth, n_oth = cnt[(b, pi)]
            center = (s_oth + A0 * q0[(pi, k)]) / (n_oth + A0)
            pred[i] = (x + ALPHA * center) / (kk + ALPHA)
        results[name] = pred
    fixed = np.empty(len(calls))
    for i, (b, pi, k) in enumerate(calls):
        x, kk, s_oth, n_oth = cnt[(b, pi)]
        center = (s_oth + A0 * base_p[i]) / (n_oth + A0)
        fixed[i] = (x + ALPHA * center) / (kk + ALPHA)
    draws = S.draws(4000)
    print(f"round-2 model (Sonnet, fixed formula): ALC {S.alc(fixed)[0]:.4f} | by budget {[round(float(c), 3) for c in S.alc(fixed)[1]]}")
    for name, pred in results.items():
        alc, curve = S.alc(pred)
        bs = np.array([S.alc(pred, dr)[0] - S.alc(fixed, dr)[0] for dr in draws])
        print(f"  {name:42} ALC {alc:.4f} | diff {alc - S.alc(fixed)[0]:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]"
              f" | B0 {curve[0]:.3f}")


if __name__ == "__main__":
    main()
