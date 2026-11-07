"""Appendix ablations of the hierarchical model (hier_llm.py), all without model calls.

Validation proxies (rehearsal-02 and proxy-confirm-1026, 18 pairs), each row against the submitted configuration
(Opus 5.5 low forecasts, normal intercepts, sigma_B 1.09, sigma_P 0.46):
  - forecasts from Opus 5.5 at medium effort;
  - forecasts recalibrated on the calibration set (judge_calib_eval.py output, runs/calib-opus-low);
  - a Student-t prior on u_B (3 df, scale 0.58);
  - the outcome-aware forecast (outcome_judge.py), which reads the revealed items and their outcomes, replacing our
    prediction, or logit-averaged with it, at budgets 1, 3 and 7.
All five proxies (45 pairs) and the three held-out ones (27 pairs):
  - the forecast alone, and the model fitted on each pair's own outcomes only (other subjects ignored);
Sampled cases of the validation proxies (ten targets per pair at budgets 7 and 31, batched_judge.py):
  - the submission's batched forecasts (one request returns the pair's revealed items and the target) in place of the
    separate forecasts for the pair's own items; other subjects keep their separate forecasts, since the sample has no
    batched forecasts for them. Also the token use and cost of those batched requests.
All five proxies (45 pairs):
  - the BLE acquisition rule (accept a candidate when its prediction lies within 0.15 of 0.5) against random order.
    Decisions use the hierarchical model's prediction from the pair's own accepted outcomes, other pairs ignored at
    decision time; scoring uses the hierarchical model with every pair's accepted outcomes.
ALC is computed per pair and averaged over pairs. Intervals: 4,000 bootstrap draws over benchmarks (seed 0) of the
paired difference.

Usage: python -I hier_ablations.py
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import ROOT, logit, sig  # noqa: E402
sys.path.insert(0, "./practice")
from common import BUDGETS, load_exam  # noqa: E402
from eval_prior_combo import M  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import posterior, predict_all  # noqa: E402

W = np.array([.1, .2, .2, .2, .2, .1])
TB, TS = 1.09, 0.46
VALID = {"validation 1": ("rehearsal-02", "r02-opus-low-fb", "r02-exemplar-opus-low-fb",
                          "r02-api-opus-medium-sonnet-fallback", "r02-exemplar-opus-medium-sonnet-fallback",
                          "calib-opus-low/r02-t", "calib-opus-low/r02-e", "outcome-opus-low-r02"),
         "validation 2": ("proxy-confirm-1026", "pc-sub-opus-low", "pc-sub-exemplar-opus-low",
                          "pc-opus-medium-sonnet-fallback", "pc-exemplar-opus-medium-sonnet-fallback",
                          "calib-opus-low/pc-t", "calib-opus-low/pc-e", "outcome-opus-low-pc")}
POOLS = {"validation 1": ("rehearsal-02", "r02-opus-low-fb", "r02-exemplar-opus-low-fb", "r02-sub-pool-opus-low"),
         "validation 2": ("proxy-confirm-1026", "pc-sub-opus-low", "pc-sub-exemplar-opus-low", "pc-sub-pool-opus-low")}
POOLS.update({f"held-out {k}": (f"proxy-1028-{k}", f"p1028-{k}-sub-opus-low", f"p1028-{k}-sub-exemplar-opus-low",
                                f"p1028-{k}-sub-pool-opus-low") for k in (1, 2, 3)})
OUTCOME_BUDGETS = (1, 3, 7)


def offsets(exam, pt, pe):
    pairs = exam["pairs"]
    zt = {(pi, t["key"]): logit(pt.get((pi, t["key"]), M)) for pi, p in enumerate(pairs) for t in p["targets"]}
    ze = {(pi, a["key"]): logit(pe.get((pi, a["key"]), M)) for pi, p in enumerate(pairs) for a in p["acquisitions"]}
    return zt, ze


def pair_alc(exam, pred):
    """pred[(b, pi, key)] -> per-pair ALC (Brier over each target's responses, pooled within the pair)."""
    out = []
    for pi, p in enumerate(exam["pairs"]):
        curve = []
        for b in BUDGETS:
            se = nn = 0.0
            for t in p["targets"]:
                v = pred[(b, pi, t["key"])]
                se += sum((v - y) ** 2 for y in t["labels"]); nn += len(t["labels"])
            curve.append(se / nn)
        out.append(float(np.array(curve) @ W))
    return np.array(out)


def report(title, rows, ref, clusters):
    keys = sorted(set(clusters)); idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(4000)]
    base = np.array(rows[ref])
    print(f"\n{title}: {len(base)} pairs, {len(keys)} benchmarks; differences are row minus '{ref}'")
    for name, v in rows.items():
        v = np.array(v); d = v - base
        iv = ""
        if name != ref:
            lo, hi = np.quantile([d[s].mean() for s in draws], [.025, .975])
            iv = f"  diff {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"  {name:58} ALC {v.mean():.4f}{iv}")


def validation_rows():
    rows, clusters = {}, []
    for label, (ex, t, e, tm, em, tc, ec, oc) in VALID.items():
        exam = load_exam(f"{ROOT}/exams/{ex}")
        pairs = exam["pairs"]
        zt, ze = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
        ours = predict_all(exam, zt, ze, TB, TS)
        found = {
            "ours (Opus low, normal intercepts)": ours,
            "Opus at medium effort": predict_all(exam, *offsets(exam, load(f"{ROOT}/runs/{tm}"), load(f"{ROOT}/runs/{em}")), TB, TS),
            "forecast recalibrated on the calibration set": predict_all(exam, *offsets(exam, load(f"{ROOT}/runs/{tc}"), load(f"{ROOT}/runs/{ec}")), TB, TS),
            "Student-t prior on u_B (3 df, scale 0.58)": predict_all(exam, zt, ze, 0.58, TS, 3.0),
        }
        oa = {(r["budget"], r["pair"], r["key"]): r["probability"]
              for r in map(json.loads, open(f"{ROOT}/runs/{oc}/predictions.jsonl")) if r.get("probability") is not None}
        alone, avg, missing = dict(ours), dict(ours), 0
        for (b, pi, k), v in ours.items():
            if b in OUTCOME_BUDGETS:
                o = oa.get((b, pi, k))
                if o is None:
                    missing += 1
                    continue
                alone[(b, pi, k)] = o
                avg[(b, pi, k)] = float(sig((logit(v) + logit(o)) / 2))
        print(f"{label}: outcome-aware forecasts missing for {missing} predictions (ours used)")
        found["outcome-aware forecast at budgets 1, 3, 7"] = alone
        found["logit average of ours and the outcome-aware forecast"] = avg
        for name, pred in found.items():
            rows.setdefault(name, []).extend(pair_alc(exam, pred))
        clusters += [(label, p["benchmark_name"]) for p in pairs]
    report("validation proxies", rows, "ours (Opus low, normal intercepts)", clusters)


def choose(pair, pj, policy):
    """Accepted candidates for one pair; the BLE rule decides from the hierarchical prediction on own-pair outcomes."""
    pool = pair["acquisitions"]
    if policy == "random":
        return pool[:31]
    taken = []
    for i, cand in enumerate(pool):
        left = 31 - len(taken)
        if left <= 0:
            break
        if len(pool) - i <= left:
            taken.append(cand)
            continue
        th, C = posterior([(0, logit(pj.get(a["key"], M)), a["label"]) for a in taken], [0], TB, TS)
        var = C[0, 0] + C[1, 1] + 2 * C[0, 1]
        pred = float(sig((logit(pj.get(cand["key"], M)) + th[0] + th[1]) / np.sqrt(1 + np.pi * var / 8)))
        if abs(pred - 0.5) <= 0.15:
            taken.append(cand)
    return taken


def acquisition_rows():
    rows, clusters = {}, []
    for label, (ex, t, e, pool) in POOLS.items():
        full = load_exam(f"{ROOT}/exams/{ex}/full_pool")
        pt = load(f"{ROOT}/runs/{t}")
        pe = {**load(f"{ROOT}/runs/{e}"), **load(f"{ROOT}/runs/{pool}")}
        line = []
        for policy in ("random", "BLE rule"):
            exam = copy.deepcopy(full)
            for pi, p in enumerate(exam["pairs"]):
                pj = {a["key"]: pe.get((pi, a["key"]), M) for a in p["acquisitions"]}
                p["acquisitions"] = choose(p, pj, policy)
            v = pair_alc(exam, predict_all(exam, *offsets(exam, pt, pe), TB, TS))
            rows.setdefault(policy, []).extend(v)
            line.append(f"{policy} {v.mean():.4f}")
        clusters += [(label, p["benchmark_name"]) for p in full["pairs"]]
        print(f"{label:13} " + " | ".join(line), flush=True)
    report("acquisition, all five proxies", rows, "random", clusters)


SPECS = [("validation 1", "rehearsal-02", "r02-opus-low-fb", "r02-exemplar-opus-low-fb"),
         ("validation 2", "proxy-confirm-1026", "pc-sub-opus-low", "pc-sub-exemplar-opus-low")] + \
        [(f"held-out {k}", f"proxy-1028-{k}", f"p1028-{k}-sub-opus-low", f"p1028-{k}-sub-exemplar-opus-low") for k in (1, 2, 3)]


def own_pair_rows():
    rows, clusters = {}, []
    for label, ex, t, e in SPECS:
        exam = load_exam(f"{ROOT}/exams/{ex}")
        zt, ze = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
        alone = copy.deepcopy(exam)
        for pi, p in enumerate(alone["pairs"]):
            p["anon_benchmark_id"] = f"{p['anon_benchmark_id']}/{pi}"
        found = {"ours": predict_all(exam, zt, ze, TB, TS),
                 "each pair on its own outcomes": predict_all(alone, zt, ze, TB, TS),
                 "forecast alone": {(b, pi, k): float(sig(z)) for b in BUDGETS for (pi, k), z in zt.items()}}
        for name, pred in found.items():
            rows.setdefault(name, []).extend(pair_alc(exam, pred))
        clusters += [(label, p["benchmark_name"]) for p in exam["pairs"]]
    report("all five proxies", rows, "ours", clusters)
    keep = [i for i, c in enumerate(clusters) if c[0].startswith("held-out")]
    report("three held-out proxies", {k: [v[i] for i in keep] for k, v in rows.items()}, "ours", [clusters[i] for i in keep])


def batched_rows():
    """Brier on the sampled (target, budget) cases with batched versus separate forecasts for the pair's own items."""
    for label, ex, t, e in SPECS[:2]:
        exam = load_exam(f"{ROOT}/exams/{ex}")
        pairs = exam["pairs"]
        zt, ze = offsets(exam, load(f"{ROOT}/runs/{t}"), load(f"{ROOT}/runs/{e}"))
        sep = predict_all(exam, zt, ze, TB, TS)
        bench = {}
        for pi, p in enumerate(pairs):
            bench.setdefault(p["anon_benchmark_id"], []).append(pi)
        tag = {"r": "r02", "p": "pc"}[ex[0]]
        recs = [r for r in map(json.loads, open(f"{ROOT}/runs/{tag}-batched-opus-low/batched.jsonl")) if r.get("probs")]
        se, qb, qs, cost, tok = {}, [], [], {}, {}
        for r in recs:
            pi, key, b = r["pair"], r["key"], r["budget"]
            assert r["exemplar_keys"] == [a["key"] for a in pairs[pi]["acquisitions"][:b]]
            members = bench[pairs[pi]["anon_benchmark_id"]]
            own = dict(zip(r["exemplar_keys"], r["probs"][:-1]))
            obs = [(j, logit(own[a["key"]]) if m == pi else ze[(m, a["key"])], a["label"])
                   for j, m in enumerate(members) for a in pairs[m]["acquisitions"][:b]]
            th, C = posterior(obs, members, TB, TS)
            j = members.index(pi)
            var = C[0, 0] + C[1 + j, 1 + j] + 2 * C[0, 1 + j]
            v = float(sig((logit(r["probs"][-1]) + th[0] + th[1 + j]) / np.sqrt(1 + np.pi * var / 8)))
            labels = next(tt["labels"] for tt in pairs[pi]["targets"] if tt["key"] == key)
            for name, p in (("batched", v), ("separate", sep[(b, pi, key)])):
                se.setdefault((name, b, pi), []).extend((p - y) ** 2 for y in labels)
            qb.append(r["probs"][-1]); qs.append(float(sig(zt[(pi, key)])))
            u = r["usage"]
            cost.setdefault(b, []).append(r["cost_usd_equiv"])
            tok.setdefault(b, []).append((u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
                                          u["output_tokens"]))
        print(f"\n{label}: {len(recs)} batched cases; target forecast correlation batched vs separate "
              f"{np.corrcoef(qb, qs)[0, 1]:.3f}")
        for b in sorted(cost):
            br = {n: np.mean([np.mean(se[(n, b, pi)]) for pi in sorted({k[2] for k in se if k[1] == b})]) for n in ("batched", "separate")}
            ti, to = np.array(tok[b]).T
            print(f"  budget {b:2d}: Brier batched {br['batched']:.4f} separate {br['separate']:.4f} "
                  f"(diff {br['batched'] - br['separate']:+.4f}); input tokens {np.median(ti):.0f} median "
                  f"[{ti.min():.0f}, {ti.max():.0f}], output {np.median(to):.0f} [{to.min():.0f}, {to.max():.0f}], "
                  f"cost as billed with a 1-hour cache write ${np.mean(cost[b]):.3f} mean")


if __name__ == "__main__":
    validation_rows()
    own_pair_rows()
    batched_rows()
    acquisition_rows()
