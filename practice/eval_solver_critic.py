"""Score the critic's budget-0 probabilities against the answers and compare them with Sonnet alone on the same items.

Per predictor (solver setting x critic variant, and "sonnet alone" = the round-2 judgment) and per pair:
  Brier   response-weighted, as in the competition (an item graded several times counts once per grade)
  AUC     chance that a graded success gets a higher probability than a graded failure in the same pair (0.5 = no
          ranking signal); it ignores the pair's overall level, so it isolates telling items apart
  shifted Brier after moving every logit(p) of the pair by one oracle constant (item differences kept), with floor
          r(1 - r) = Brier of knowing only the pair's success rate r
Also: what the solvers produced, how often their short answers agree, tokens they used, the same AUC for the
code-side and critic-side agreement signals, and measured costs with projections.
Writes pred-<setting>-<variant>/predictions.jsonl (budget 0) for eval_prior_combo.py and item_signal.py.

Usage: python -I eval_solver_critic.py <exam_dir> <run_dir> <sonnet_alone_run_dir>
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam

SUBMISSION_CALLS, HIDDEN_LENGTH_FACTOR, N_ATTEMPTS = 2982, 1.3, 4


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def auc(ps, ys):
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    pos, neg = ps[ys == 1], ps[ys == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def shifted_brier(ps, ys):
    z = np.log(np.clip(ps, 0.01, 0.99) / (1 - np.clip(ps, 0.01, 0.99)))
    return float(minimize_scalar(lambda d: np.mean((1 / (1 + np.exp(-(z + d))) - ys) ** 2), bounds=(-8, 8),
                                 method="bounded").fun)


def responses(exam, items, value):
    """{pair: (probabilities, labels)} at response level for items with a value."""
    out = defaultdict(lambda: ([], []))
    for pi, key in items:
        v = value.get((pi, key))
        if v is None:
            continue
        for y in exam_targets[(pi, key)]["labels"]:
            out[pi][0].append(v)
            out[pi][1].append(y)
    return out


def score(exam, items, value):
    rows = {}
    for pi, (ps, ys) in responses(exam, items, value).items():
        ps, ys = np.array(ps), np.array(ys, float)
        r = ys.mean()
        rows[pi] = {"n": len(ys), "r": r, "brier": float(np.mean((ps - ys) ** 2)), "auc": auc(ps, ys),
                    "shifted": shifted_brier(ps, ys), "floor": r * (1 - r), "mean_p": float(ps.mean())}
    return rows


def mean_of(rows, field):
    vals = [row[field] for row in rows.values() if row[field] is not None]
    return float(np.mean(vals)) if vals else float("nan")


exam_targets = {}


def main():
    exam_dir, run_dir, alone_dir = sys.argv[1:4]
    exam = load_exam(exam_dir)
    exam_targets.update({(pi, t["key"]): t for pi, p in enumerate(exam["pairs"]) for t in p["targets"]})
    items = [tuple(x) for x in json.load(open(os.path.join(run_dir, "selection.json")))["items"]]
    attempts = load_jsonl(os.path.join(run_dir, "attempts.jsonl"))
    critics = load_jsonl(os.path.join(run_dir, "critic.jsonl"))
    alone = {(r["pair"], r["key"]): r["probability"] for r in load_jsonl(os.path.join(alone_dir, "predictions.jsonl"))
             if r["budget"] == 0 and r.get("probability") is not None}

    preds, extra = {}, {}
    for r in critics:
        if r["probability"] is not None:
            preds.setdefault((r["setting"], r["variant"]), {})[(r["pair"], r["key"])] = r["probability"]
            extra.setdefault((r["setting"], r["variant"]), {})[(r["pair"], r["key"])] = r
    common = [u for u in items if alone.get(u) is not None and all(u in d for d in preds.values())]
    print(f"items selected {len(items)}, scored with every predictor {len(common)}")
    names = {"sonnet alone": alone, **{f"{s} / {v}": d for (s, v), d in sorted(preds.items())}}
    pairs = sorted({pi for pi, _ in common})

    print("\n== critic probabilities vs Sonnet alone (same items; pairs weighted equally) ==")
    print(f"{'predictor':16} {'Brier':>7} {'AUC':>6} {'shifted':>8} {'floor':>6} | per pair Brier / AUC")
    for name, d in names.items():
        rows = score(exam, common, d)
        per = "  ".join(f"p{pi}:{rows[pi]['brier']:.3f}/{rows[pi]['auc'] if rows[pi]['auc'] is not None else float('nan'):.2f}"
                        for pi in pairs if pi in rows)
        print(f"{name:16} {mean_of(rows, 'brier'):7.4f} {mean_of(rows, 'auc'):6.3f} {mean_of(rows, 'shifted'):8.4f}"
              f" {mean_of(rows, 'floor'):6.4f} | {per}")
    print("pairs:", {pi: f"{exam['pairs'][pi]['benchmark_name']} x {exam['pairs'][pi]['subject']['normalized_name']}" for pi in pairs})

    print("\n== what the solvers produced ==")
    for s in sorted({r["setting"] for r in attempts}):
        rs = [r for r in attempts if r["setting"] == s and (r["pair"], r["key"]) in set(common)]
        by_pair = defaultdict(list)
        for r in rs:
            by_pair[r["pair"]].append(r)
        for pi in pairs:
            g = by_pair.get(pi, [])
            if not g:
                continue
            kinds = defaultdict(int)
            for r in g:
                kinds[r["kind"] or f"none:{r['stop_reason'] or 'error'}"] += 1
            toks = [r["usage"].get("output_tokens", 0) for r in g if r["usage"]]
            print(f"  {s:3} p{pi} attempts {len(g):3} | {dict(kinds)} | output tokens mean {np.mean(toks):.0f}"
                  f" median {np.median(toks):.0f} max {max(toks)}")

    print("\n== agreement and other signals: AUC within pair (0.5 = no signal) ==")
    for (s, v), ex in sorted(extra.items()):
        sig = {"code: share of most common short answer": {u: ex[u]["agreement"]["top_share"] for u in common
                                                             if u in ex and ex[u]["agreement"]["top_share"] is not None},
               "critic: same_conclusion / attempts": {u: ex[u]["same_conclusion"] / max(1, ex[u]["n_attempts"])
                                                      for u in common if u in ex and ex[u]["same_conclusion"] is not None},
               "critic: majority_looks_right": {u: ex[u]["majority_looks_right"] for u in common if u in ex}}
        tok = defaultdict(list)
        for r in attempts:
            if r["setting"] == s and r["usage"]:
                tok[(r["pair"], r["key"])].append(r["usage"].get("output_tokens", 0))
        sig["code: minus median output tokens"] = {u: -float(np.median(tok[u])) for u in common if tok.get(u)}
        for name, d in sig.items():
            rows = score(exam, list(d), d)
            print(f"  {s:3} / {v:3} {name:44} items {len(d):3} | AUC {mean_of(rows, 'auc'):.3f} | "
                  + "  ".join(f"p{pi}:{rows[pi]['auc']:.2f}" for pi in pairs if pi in rows and rows[pi]['auc'] is not None))

    print("\n== cost ==")
    groups = defaultdict(list)
    for r in attempts:
        groups[f"solver {r['setting']}"].append(r)
    for r in critics:
        groups[f"critic {r['setting']} / {r['variant']}"].append(r)
    per_call = {}
    for name, rs in sorted(groups.items()):
        c = [r["cost_usd"] for r in rs]
        inp = [r["usage"].get("input_tokens", 0) for r in rs if r["usage"]]
        per_call[name] = float(np.mean(c))
        print(f"  {name:22} calls {len(rs):4} | total ${sum(c):.3f} | per call ${np.mean(c):.5f} | input tokens mean {np.mean(inp):.0f}")
    print(f"  all calls so far ${sum(r['cost_usd'] for r in attempts + critics):.3f}")
    for s in sorted({r["setting"] for r in attempts}):
        for v in [v for (s2, v) in extra if s2 == s]:
            unit = N_ATTEMPTS * per_call[f"solver {s}"] + per_call[f"critic {s} / {v}"]
            print(f"  projection {s} / {v}: per item ${unit:.4f} | all 490 practice items ${490 * unit:.2f}"
                  f" | one submission (x{HIDDEN_LENGTH_FACTOR} length, {SUBMISSION_CALLS} calls) about"
                  f" ${SUBMISSION_CALLS * unit * HIDDEN_LENGTH_FACTOR:.0f}")

    for (s, v), d in preds.items():
        out = os.path.join(run_dir, f"pred-{s}-{v}")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "predictions.jsonl"), "w") as f:
            for (pi, key), p in d.items():
                f.write(json.dumps({"budget": 0, "pair": pi, "key": key, "probability": p,
                                    "cost_usd_equiv": extra[(s, v)][(pi, key)]["cost_usd"]}) + "\n")


if __name__ == "__main__":
    main()
