"""Which exemplars to reveal: acquisition policies compared on full-pool proxy evaluations, with no new LLM calls.

Every pool item already has a budget-0 judgment (targets, first 31 and the rest of the pool), so a policy only changes
which items' outcomes are revealed. Candidates arrive in each pair's fixed random order; a policy accepts or skips
each one for good, up to 31, and must accept when the candidates left are no more than the labels left (as the
competition's acquisition_function and the organizers' BLE labeling.py do). The prediction a policy sees for a
candidate is the predictor's own prediction from the pair's labels accepted so far (other pairs ignored at decision
time, a simplification).

Policies
  random       accept in arrival order (today's default; identical to the exam's first 31)
  uncertainty  the organizers' BLE rule: accept when |prediction - 0.5| <= 0.15
Predictors, each scored under each policy on the same evaluation items
  current      고정 갱신 공식 + item 차이 되살리기 (counts revealed outcomes; selection can bias the count)
  hierarchical hier_llm.py: logit P = judgment logit + shared and pair shifts, exemplars entering with their own
               judgments, which corrects for which items were chosen
Pooled intervals resample benchmarks across exams, for each row minus (current, random).

Usage: python -I acquisition_eval.py <label>=<full_pool_exam_dir>:<target_run>:<exemplar_run>:<pool_run> [...]
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from combo_fit import logit
from common import ALC_WEIGHTS, BUDGETS, MAX_LABELS, load_exam
from eval_prior_combo import A0, ALPHA, M, counts
from exemplar_shift import load
from hier_llm import predict_all

W = np.array(ALC_WEIGHTS)
sig = lambda z: 1 / (1 + np.exp(-z))


def current_pred(x, k, s, n, p0, zbar):
    f = (x + ALPHA * (s + A0 * p0) / (n + A0)) / (k + ALPHA)
    return float(sig(logit(f) + k / (k + 4.0) * (logit(p0) - zbar))) if k > 0 else f


def choose(pair, pj, policy):
    """Accepted acquisitions for one pair under a policy (decisions use the current predictor on own-pair labels)."""
    pool = pair["acquisitions"]
    if policy == "random":
        return pool[:MAX_LABELS]
    taken = []
    for i, cand in enumerate(pool):
        left_labels, left_items = MAX_LABELS - len(taken), len(pool) - i
        if left_labels <= 0:
            break
        if left_items <= left_labels:
            taken.append(cand)
            continue
        x, k = sum(a["label"] for a in taken), len(taken)
        zbar = np.mean([logit(pj.get(a["key"], M)) for a in taken]) if taken else 0.0
        pred = current_pred(x, k, 0, 0, pj.get(cand["key"], M), zbar)
        if abs(pred - 0.5) <= 0.15:
            taken.append(cand)
    return taken


def score(exam, pt, pe, predictor):
    pairs = exam["pairs"]
    out = np.zeros(len(pairs))
    if predictor == "hierarchical":
        zt = {(pi, t["key"]): logit(pt.get((pi, t["key"]), M)) for pi, p in enumerate(pairs) for t in p["targets"]}
        ze = {(pi, a["key"]): logit(pe.get((pi, a["key"]), M)) for pi, p in enumerate(pairs) for a in p["acquisitions"]}
        pr = predict_all(exam, zt, ze, 1.09, 0.46)
    cnt = counts(exam)
    for pi, p in enumerate(pairs):
        curve = []
        for b in BUDGETS:
            x, k, s, n = cnt[(b, pi)]
            acq = p["acquisitions"][:b]
            zbar = np.mean([logit(pe.get((pi, a["key"]), M)) for a in acq]) if acq else 0.0
            se = nn = 0.0
            for t in p["targets"]:
                v = (pr[(b, pi, t["key"])] if predictor == "hierarchical"
                     else current_pred(x, k, s, n, pt.get((pi, t["key"]), M), zbar))
                se += sum((v - y) ** 2 for y in t["labels"]); nn += len(t["labels"])
            curve.append(se / nn)
        out[pi] = float(np.array(curve) @ W)
    return out


def main():
    rows, clusters, base_key = {}, [], ("current", "random")
    for arg in sys.argv[1:]:
        label, spec = arg.split("=", 1)
        exam_dir, t_run, e_run, p_run = spec.split(":")
        full = load_exam(exam_dir)
        pt = load(t_run)
        pe = {**load(e_run), **load(p_run)}
        line = []
        for policy in ("random", "uncertainty"):
            exam = copy.deepcopy(full)
            for pi, p in enumerate(exam["pairs"]):
                pj = {a["key"]: pe.get((pi, a["key"]), M) for a in p["acquisitions"]}
                p["acquisitions"] = choose(p, pj, policy)
            # re-key exemplar judgments by (pair, key) for the chosen items (same keys, so pe is reused)
            for predictor in ("current", "hierarchical"):
                v = score(exam, pt, pe, predictor)
                rows.setdefault((predictor, policy), []).extend(v)
                line.append(f"{predictor}/{policy} {v.mean():.4f}")
        clusters.extend((label, p["benchmark_name"]) for p in full["pairs"])
        print(f"{label:8} " + " | ".join(line), flush=True)
    keys = sorted(set(clusters))
    idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(4000)]
    ref = np.array(rows[base_key])
    print(f"pooled: {len(ref)} pairs, {len(keys)} benchmarks")
    for key, v in rows.items():
        v = np.array(v); d = v - ref
        bs = [np.concatenate([d[idx[keys[j]]] for j in dr]).mean() for dr in draws]
        print(f"  {key[0]:12} {key[1]:11} ALC {v.mean():.4f} | minus current/random {d.mean():+.4f} "
              f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")


if __name__ == "__main__":
    main()
