"""Follow-up experiments of 11/2 (subscription runs): MathArena score probe, problem-only forecasts, pair-share
question on the calibration set, reason-first forecasts, and Sonnet/Haiku on MathArena."""
import json, sys
import numpy as np
sys.path.insert(0, "./practice")
from exemplar_shift import load
R = "."
sig = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
ex = lambda name: json.load(open(f"{R}/exams/{name}/exam.json"))
part = sys.argv[1] if len(sys.argv) > 1 else "all"


def auc(q, y):
    pos, neg = q[y > .5], q[y <= .5]
    return np.mean([(a > b) + .5 * (a == b) for a in pos for b in neg]) if len(pos) and len(neg) else np.nan


def pair_stats(exam, pr, keyset=None):
    out = []
    for pi, p in enumerate(exam["pairs"]):
        ts = [t for t in p["targets"] if (pi, t["key"]) in pr and (keyset is None or (pi, t["key"]) in keyset)]
        if len(ts) < 2:
            continue
        q = np.array([pr[(pi, t["key"])] for t in ts]); y = np.array([np.mean(t["labels"]) for t in ts])
        rate = p.get("pair_success_rate", sum(sum(t["labels"]) for t in p["targets"]) / sum(len(t["labels"]) for t in p["targets"]))
        br = sum(((q[j] - np.asarray(t["labels"])) ** 2).sum() for j, t in enumerate(ts)) / sum(len(t["labels"]) for t in ts)
        out.append({"pi": pi, "b": p["benchmark_name"], "subj": p["subject"]["normalized_name"], "mean": q.mean(), "rate": rate,
                    "auc": auc(q, y), "brier": br, "n": len(ts)})
    return out


def show(label, rows):
    print(f"  {label:44} pairs {len(rows):3} | pair-level |err| {np.mean([abs(r['mean'] - r['rate']) for r in rows]):.3f} | "
          f"corr {np.corrcoef([r['mean'] for r in rows], [r['rate'] for r in rows])[0, 1]:.3f} | AUC {np.nanmean([r['auc'] for r in rows]):.3f} | Brier {np.mean([r['brier'] for r in rows]):.4f}")


if part in ("all", "ma-probe"):
    print("== MathArena score probe (problem text only, anon, 52 pairs)")
    mp = ex("matharena-1101-problems")
    for m in ("opus", "sonnet"):
        rows = [json.loads(l) for l in open(f"{R}/runs/memprobe-score-ma-1102/score_{m}.jsonl")]
        v = [(r["parsed"]["rate"], mp["pairs"][r["pair"]]["pair_success_rate"], r["parsed"].get("source"), r["parsed"].get("benchmark")) for r in rows if (r.get("parsed") or {}).get("rate") is not None]
        e = np.array([abs(a - b) for a, b, _, _ in v])
        src = {s: sum(1 for x in v if x[2] == s) for s in ("remembered", "partly remembered", "not remembered")}
        print(f"  {m:7} answered {len(v)} | mean |rate error| {e.mean():.3f} | mean stated {np.mean([x[0] for x in v]):.2f} vs true {np.mean([x[1] for x in v]):.2f} | corr {np.corrcoef([x[0] for x in v], [x[1] for x in v])[0, 1]:.3f} | sources {src}")
    print("  (calibration set, same probe: Opus 0.104)")

if part in ("all", "problemonly"):
    print("== MathArena framing check: same 10 pairs, prompt shown vs problem text only (Opus low)")
    full = ex("matharena-1101"); sub = ex("matharena-1101-problems-sub")
    po = load(f"{R}/runs/ma-opus-low-problemonly"); pf = load(f"{R}/runs/ma-opus-low")
    idx = {(p["benchmark_name"], p["subject_id"]): i for i, p in enumerate(full["pairs"])}
    a = pair_stats(sub, po)
    bmap = {(pi_sub): idx[(p["benchmark_name"], p["subject_id"])] for pi_sub, p in enumerate(sub["pairs"])}
    pf_sub = {(ps, k): pf[(bmap[ps], k)] for (ps, k) in po if (bmap[ps], k) in pf}
    b = pair_stats(sub, pf_sub)
    show("full prompt (as in the main run)", b); show("problem text only", a)
    for x, y in zip(sorted(b, key=lambda r: r["pi"]), sorted(a, key=lambda r: r["pi"])):
        print(f"    {x['b'][-16:]:16} {x['subj'][-20:]:20} rate {x['rate']:.2f} | full {x['mean']:.2f} | problem only {y['mean']:.2f}")

if part in ("all", "pairshare"):
    print("== Pair-share question (one item shown, no recall wording), calibration set, first 3 targets per pair (Opus low)")
    cal = ex("mdb-judge-1027"); f3 = ex("mdb-judge-1027-first3")
    ps = load(f"{R}/runs/mdb-opus-low-pairshare"); pf = load(f"{R}/runs/mdb-judge-opus-low")
    keys = {(pi, t["key"]) for pi, p in enumerate(f3["pairs"]) for t in p["targets"]}
    share = pair_stats(f3, ps)
    item3 = pair_stats(cal, pf, keyset=keys)
    item7 = pair_stats(cal, pf)
    common = {r["pi"] for r in share} & {r["pi"] for r in item3}
    sel = lambda rows: [r for r in rows if r["pi"] in common]
    show("item forecasts, mean of the same 3 items", sel(item3)); show("item forecasts, mean of all items", sel(item7))
    show("pair-share answers, mean of 3", sel(share))
    probe = {json.loads(l)["pair"]: json.loads(l) for l in open(f"{R}/runs/memprobe-score-1101/score_opus.jsonl") if json.loads(l)["cond"] == "anon"}
    pr = [(probe[pi]["parsed"]["rate"], cal["pairs"][pi]["pair_success_rate"]) for pi in common if (probe.get(pi, {}).get("parsed") or {}).get("rate") is not None]
    print(f"  {'score probe (3 items, recall wording), for reference':44} pairs {len(pr):3} | pair-level |err| {np.mean([abs(a - b) for a, b in pr]):.3f} | corr {np.corrcoef(*zip(*pr))[0, 1]:.3f}")

if part in ("all", "reasonfirst"):
    print("== Reason-first forecasts vs the usual forecasts, calibration set (Opus low)")
    cal = ex("mdb-judge-1027")
    rf = load(f"{R}/runs/mdb-opus-low-reasonfirst"); pf = load(f"{R}/runs/mdb-judge-opus-low")
    keys = set(rf) & set(pf)
    show("usual (JSON only)", pair_stats(cal, pf, keys)); show("reason first", pair_stats(cal, rf, keys))

if part in ("all", "claude3"):
    print("== MathArena, three Claude forecasters (all low, subscription)")
    ma = ex("matharena-1101")
    P = {m: load(f"{R}/runs/ma-{m}-low") for m in ("opus", "sonnet", "haiku")}
    keys = set(P["opus"]) & set(P["sonnet"]) & set(P["haiku"])
    for m in P:
        show(m, pair_stats(ma, P[m], keys))
    print("  reference, calibration subset of 847 items (9-89): AUC Opus 0.686, Sonnet 0.653, Haiku 0.634")
