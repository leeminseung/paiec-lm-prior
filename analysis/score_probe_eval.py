"""Score-memory probe, full run (332 pairs x anon/named x 3 forecasters, 11/1): accuracy of the stated success rate.

Error = logit(stated rate) - logit(pair success rate over all responses), rates clipped to [0.01, 0.99]. Pairs where
any forecaster gave no parsed answer in a condition are left out of that condition's comparison. Split of the anon
error into a benchmark part (mean over the benchmark's pairs) and a within-benchmark part, as in gap_levels.py.
Intervals: 4,000 bootstrap draws over benchmarks.
"""
import json
import numpy as np
F = "./runs/memprobe-score-1101"
exam = json.load(open("./exams/mdb-judge-1027/exam.json"))
M = {"Opus": "opus", "GLM": "glm", "DeepSeek": "ds"}
lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
ans = {}
for name, f in M.items():
    for r in map(json.loads, open(f"{F}/score_{f}.jsonl")):
        p = r.get("parsed") or {}
        if p.get("rate") is not None:
            ans[(name, r["pair"], r["cond"])] = p
truth = {pi: p["pair_success_rate"] for pi, p in enumerate(exam["pairs"])}
bench = {pi: p["benchmark_name"] for pi, p in enumerate(exam["pairs"])}
rng = np.random.default_rng(0)


def common(cond):
    return [pi for pi in truth if all((n, pi, cond) in ans for n in M)]


def err(n, pi, cond):
    return float(lg(ans[(n, pi, cond)]["rate"]) - lg(truth[pi]))


def boot(pis, fn, draws=4000):
    bs = sorted({bench[p] for p in pis}); by = {b: [p for p in pis if bench[p] == b] for b in bs}
    out = []
    for _ in range(draws):
        pick = [p for b in rng.choice(bs, len(bs)) for p in by[b]]
        out.append(fn(pick))
    return np.quantile(out, [.025, .975])


for cond in ("anon", "named"):
    pis = common(cond)
    print(f"\n{cond}: {len(pis)} pairs in {len({bench[p] for p in pis})} benchmarks")
    for n in M:
        e = np.array([err(n, p, cond) for p in pis]); a = np.array([abs(ans[(n, p, cond)]["rate"] - truth[p]) for p in pis])
        print(f"  {n:9} mean |logit error| {np.abs(e).mean():.3f} | mean |rate error| {a.mean():.3f} | bias {e.mean():+.2f}")
    for n in ("GLM", "DeepSeek"):
        fn = lambda pk: np.mean([abs(err(n, p, cond)) - abs(err("Opus", p, cond)) for p in pk])
        lo, hi = boot(pis, fn)
        print(f"  {n} minus Opus, |logit error|: {fn(pis):+.3f} [{lo:+.3f}, {hi:+.3f}]")

pis = [p for p in common("anon") if p in set(common("named"))]
print(f"\nnamed minus anon, |logit error| (negative = the name helped), {len(pis)} pairs")
for n in M:
    fn = lambda pk: np.mean([abs(err(n, p, "named")) - abs(err(n, p, "anon")) for p in pk])
    lo, hi = boot(pis, fn)
    print(f"  {n:9} {fn(pis):+.3f} [{lo:+.3f}, {hi:+.3f}]")

pis = common("anon")
multi = {b for b in {bench[p] for p in pis} if sum(1 for q in pis if bench[q] == b) >= 2}
print(f"\nanon error split, {len(multi)} benchmarks with 2+ pairs: RMS of benchmark part | RMS of within-benchmark part")
for n in M:
    bp, wp = [], []
    for b in multi:
        e = np.array([err(n, p, "anon") for p in pis if bench[p] == b])
        bp += [e.mean()] * len(e); wp += list(e - e.mean())
    print(f"  {n:9} {np.sqrt(np.mean(np.square(bp))):.3f} | {np.sqrt(np.mean(np.square(wp))):.3f}")

print("\nanon answers by stated source: n, mean |logit error|")
for n in M:
    line = f"  {n:9}"
    for s in ("remembered", "partly remembered", "not remembered"):
        e = [abs(err(n, p, "anon")) for p in pis if ans[(n, p, "anon")].get("source") == s]
        line += f" | {s} {len(e)}, {np.mean(e) if e else float('nan'):.3f}"
    print(line)
nr = [p for p in pis if ans[("Opus", p, "anon")].get("source") == "not remembered"]
for n in ("GLM", "DeepSeek"):
    fn = lambda pk: np.mean([abs(err(n, p, "anon")) - abs(err("Opus", p, "anon")) for p in pk]) if pk else np.nan
    lo, hi = boot(nr, fn)
    print(f"  on the {len(nr)} pairs Opus marks not remembered: {n} minus Opus {fn(nr):+.3f} [{lo:+.3f}, {hi:+.3f}]")
