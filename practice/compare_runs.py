"""Compare rehearsal runs and baselines on the same calls: score, time, turns, cost, and which files sessions touched.

A call is identified by (budget, pair, item key), read from each run's own records, never by call id alone.
Several run folders can be passed; a run folder may hold more than one batch (for example extra targets added later).

Scores on the sample of calls, three ways:
- per-call mean: every call weighs the same (not the competition mix; kept for the first exam's reports);
- ALC: per budget, the pair's Brier over every graded response of its sampled items (an item tried 8 times counts
  8 times), pairs averaged equally, budgets weighted 0.1, 0.2, 0.2, 0.2, 0.2, 0.1 as in the competition;
- ALC post-stratified: as ALC, but within each pair the success responses and the failure responses of the sample
  are reweighted to the pair's true success share over its whole evaluation pool. This removes the luck of drawing
  mostly failures or mostly successes, which otherwise favors predictors that lean the same way. A pair-budget with
  no success (or no failure) response in the sample falls back to the plain ALC.
A call with no answer (timeout, error) is scored as 0.5 and counted; the competition treats a call timeout as a failure.
Intervals (4,000 draws, shared by all comparisons) resample pairs, then items within each drawn pair, keeping an
item's calls at all budgets together. With few pairs they are wide; that width is real uncertainty.

"Touched" means the file or folder name appears in a tool input somewhere in that session's transcript.

Usage: python -I compare_runs.py <exam_dir> <run_dir> [<run_dir> ...]
"""
import json
import os
import re
import statistics
import sys

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import ALC_WEIGHTS, BUDGETS, brier_alc, load_exam
from evaluate import baselines_for, run_python_predictor

TOUCH = {
    "README": r"README",
    "labeled.json": r"labeled\.json",
    "labeled_summary.csv": r"labeled_summary",
    "items/<n>.txt": r"items/",
    "pair_stats": r"pair_stats",
    "model_index": r"model_index",
    "benchmark_index": r"benchmark_index",
    "benchmarks_all / subjects_all": r"benchmarks_all|subjects_all",
    "benchmark folders": r"data/(?!catalog\b)[\w.\-*]+/|(?<![\w])(responses|items|subjects|benchmarks)\.parquet",
}
REFUSED = re.compile(r"requires? approval|cannot be statically analyzed|Contains simple_expansion")
FAILED = re.compile(r"Traceback|No module named|Error|No such file")
W = np.array(ALC_WEIGHTS)
B_INDEX = {b: i for i, b in enumerate(BUDGETS)}


def load_run(run_dir):
    """Records keyed by (budget, pair, key), each with transcript-derived fields."""
    recs = {}
    with open(os.path.join(run_dir, "predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            k = (r["budget"], r["pair"], r["key"])
            if k in recs:
                print(f"warning: {run_dir} answers {k} twice; keeping the later record")
            recs[k] = r
    for r in recs.values():
        uses, results = [], {}
        path = os.path.join(run_dir, "calls", f"{r['call_id']}.jsonl")
        if os.path.exists(path):
            for line in open(path):
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                            uses.append((block["id"], block["name"], block.get("input", {})))
                if ev.get("type") == "user" and isinstance(ev.get("message", {}).get("content"), list):
                    for block in ev["message"]["content"]:
                        if block.get("type") == "tool_result":
                            t = block.get("content")
                            if isinstance(t, list):
                                t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                            results[block["tool_use_id"]] = str(t)
        text = "\n".join(json.dumps(inp, ensure_ascii=False) for _, _, inp in uses)
        r["touched"] = {name: bool(re.search(pat, text)) for name, pat in TOUCH.items()}
        bash = [(inp.get("command", ""), results.get(uid, "")) for uid, name, inp in uses if name == "Bash"]
        r["bash_calls"] = len(bash)
        r["refused"] = sum(bool(REFUSED.search(res)) for _, res in bash)
        r["read_parquet"] = any("parquet" in cmd and not REFUSED.search(res) and not FAILED.search(res)
                                for cmd, res in bash)
    return recs


class Sample:
    """The sampled calls with their outcome counts, and the estimators over any resampled set of them."""

    def __init__(self, exam, calls):
        target = {(pi, t["key"]): t for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]}
        self.calls = calls
        self.b = np.array([B_INDEX[b] for b, _, _ in calls])
        self.pair = np.array([p for _, p, _ in calls])
        self.ns = np.array([sum(target[(p, k)]["labels"]) for _, p, k in calls], dtype=float)
        self.nf = np.array([len(target[(p, k)]["labels"]) for _, p, k in calls], dtype=float) - self.ns
        self.share = {pi: sum(sum(t["labels"]) for t in pair["targets"]) / sum(len(t["labels"]) for t in pair["targets"])
                      for pi, pair in enumerate(exam["pairs"])}
        units = {}
        for i, (b, p, k) in enumerate(calls):
            units.setdefault(p, {}).setdefault(k, []).append(i)
        self.units = {p: [np.array(ix) for ix in d.values()] for p, d in units.items()}
        self.incomplete = [(p, k) for p, d in units.items() for k, ix in d.items() if len(ix) != len(BUDGETS)]
        self.identity = (np.arange(len(calls)), self.pair.copy())

    def draws(self, n, seed=0):
        rng = np.random.default_rng(seed)
        pairs = sorted(self.units)
        out = []
        for _ in range(n):
            idx, lab = [], []
            for new_p, p in enumerate(rng.choice(pairs, size=len(pairs))):
                items = self.units[p]
                for j in rng.integers(0, len(items), size=len(items)):
                    idx.append(items[j])
                    lab.append(np.full(len(items[j]), new_p))
            idx = np.concatenate(idx)
            out.append((idx, np.concatenate(lab), self.pair[idx]))
        return out

    def alc(self, pred, draw=None, post_stratify=False):
        """pred: array aligned with self.calls. draw: (indices, new pair labels, original pairs) or None."""
        idx, lab, orig = draw if draw is not None else (self.identity[0], self.identity[1], self.pair)
        p, ns, nf, b = pred[idx], self.ns[idx], self.nf[idx], self.b[idx]
        n_pairs = int(lab.max()) + 1
        g = lab * len(BUDGETS) + b
        size = n_pairs * len(BUDGETS)
        es = np.bincount(g, ns * (p - 1) ** 2, size)
        ef = np.bincount(g, nf * p ** 2, size)
        cs = np.bincount(g, ns, size)
        cf = np.bincount(g, nf, size)
        plain = (es + ef) / np.maximum(cs + cf, 1e-12)
        if post_stratify:
            share_of = np.zeros(n_pairs)
            share_of[lab] = [self.share[o] for o in orig]
            sh = np.repeat(share_of, len(BUDGETS))
            ps = sh * es / np.maximum(cs, 1e-12) + (1 - sh) * ef / np.maximum(cf, 1e-12)
            plain = np.where((cs > 0) & (cf > 0), ps, plain)
        per_budget = plain.reshape(n_pairs, len(BUDGETS)).mean(axis=0)
        return float(per_budget @ W), per_budget

    def per_call_mean(self, pred):
        return float(np.mean((self.ns * (pred - 1) ** 2 + self.nf * pred ** 2) / (self.ns + self.nf)))


def main():
    exam_dir = sys.argv[1]
    exam = load_exam(exam_dir)
    runs = {os.path.basename(os.path.normpath(p)): load_run(p) for p in sys.argv[2:]}
    calls = sorted(set().union(*[set(recs) for recs in runs.values()]))
    S = Sample(exam, calls)
    mix = {b: sum(1 for c in calls if c[0] == b) for b in BUDGETS}
    print(f"calls: {len(calls)} over {len({(p, k) for _, p, k in calls})} distinct items in {len(S.units)} pairs | "
          f"calls per budget {mix}")
    if S.incomplete:
        print(f"WARNING: {len(S.incomplete)} items are not asked at all six budgets, e.g. {S.incomplete[:3]}")
    present_b = {b for b, _, _ in calls}
    if present_b != set(BUDGETS):
        print(f"WARNING: budgets missing from the sample: {sorted(set(BUDGETS) - present_b)}; ALC is not comparable")

    preds, whole_alc = {}, {}
    for name, fn in baselines_for(exam_dir).items():
        pp = run_python_predictor(exam, fn, only=set(calls))
        preds[f"baseline {name}"] = np.array([pp[c] for c in calls])
        whole_alc[name] = brier_alc(exam, run_python_predictor(exam, fn))[1]
    for name, recs in runs.items():
        preds[name] = np.array([recs[c]["probability"] if c in recs and recs[c].get("probability") is not None else 0.5
                                for c in calls])

    print("\nscores on the sampled calls (lower is better):")
    for name, pred in preds.items():
        a, curve = S.alc(pred)
        a_ps, curve_ps = S.alc(pred, post_stratify=True)
        extra = f" | whole exam ALC {whole_alc[name[len('baseline '):]]:.4f}" if name.startswith("baseline ") else ""
        print(f"  {name:34} per-call {S.per_call_mean(pred):.4f} | ALC {a:.4f} | ALC post-stratified {a_ps:.4f}{extra}")
        print(f"  {'':34} by budget, post-stratified {dict(zip(BUDGETS, [round(float(v), 3) for v in curve_ps]))}")

    for name, recs in runs.items():
        present = list(recs.values())
        missing = sum(1 for c in calls if c not in recs or recs[c].get("probability") is None)
        walls = [r["wall_s"] for r in present]
        turns = [r["num_turns"] for r in present if r["num_turns"] is not None]
        cost = sum(r["cost_usd_equiv"] or 0 for r in present)
        print(f"\n== {name}: {len(present)} records, calls without an answer (scored 0.5) {missing}")
        print(f"  wall median {statistics.median(walls):.1f}s max {max(walls):.1f}s | turns median {statistics.median(turns)} "
              f"max {max(turns)} | API-equivalent cost ${cost:.2f}")
        print("  sessions touching: " + ", ".join(f"{t} {sum(r['touched'][t] for r in present)}" for t in TOUCH))
        print(f"  refused Bash calls {sum(r['refused'] for r in present)}/{sum(r['bash_calls'] for r in present)} | "
              f"sessions that read a parquet file without error {sum(r['read_parquet'] for r in present)}")

    draws = S.draws(4000)
    names = list(preds)
    print("\ndifference (row minus column; negative = row better): ALC [95%] | ALC post-stratified [95%]")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith("baseline") and b.startswith("baseline"):
                continue
            d = S.alc(preds[a])[0] - S.alc(preds[b])[0]
            d_ps = S.alc(preds[a], post_stratify=True)[0] - S.alc(preds[b], post_stratify=True)[0]
            bs = np.array([[S.alc(preds[a], dr)[0] - S.alc(preds[b], dr)[0],
                            S.alc(preds[a], dr, True)[0] - S.alc(preds[b], dr, True)[0]] for dr in draws])
            lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
            print(f"  {a} - {b}: {d:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | {d_ps:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")


if __name__ == "__main__":
    main()
