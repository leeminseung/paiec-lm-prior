"""Judge-proposed axes: show Opus where the direct judge erred on measurement-db and ask for scoring axes (report flow 4).

Data: the measurement-db judge set (exams/mdb-judge-1027) with its Opus low budget-0 judgments, leaving out every
benchmark used by a proxy evaluation, so axes proposed here can later be tested on the proxies. Benchmarks are shown
under anonymous ids (B001, ...) and grouped, since judge errors are mostly shared within a benchmark; for each pair the
subject description, the judge's pair mean prediction, the pair success rate, and two items (the judge's largest
over- and under-prediction, 350 characters each) with their judgment and outcome rate. Axes must be scorable at
prediction time from the subject description and item text alone.

Each run shuffles the benchmark order with its own seed, so three runs show which axes are stable.
Backend: claude -p through the subscription (clean environment as in subscription_judge.py), free-text answer.

Usage: python -I propose_axes.py <full_judge_set_dir> <low_run_dir> <out_dir> <model> <effort> <empty_dir> <seed> [...]
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load
from subscription_judge import clean_env

PROXIES = ["rehearsal-02", "proxy-confirm-1026", "proxy-1028-1", "proxy-1028-2", "proxy-1028-3"]
EXCERPT = 350

SYSTEM = """You are helping design a scoring rubric for predicting whether an AI system (the subject) answers a benchmark item correctly.

A direct LLM judge estimated each success probability from the subject description and the item text alone (no outcomes, benchmark anonymous). The data below shows, on public benchmarks, where that judge was right and where it was wrong. Judge errors are mostly shared by all subjects within a benchmark, so look for causes at the benchmark level as well as the item and subject levels.

Propose 6 to 10 scoring axes that
(a) can be scored at prediction time from the subject description and the item text alone (no benchmark name, no outcomes, no knowledge of published results);
(b) would help explain the judge's systematic errors, so that a model combining the judge with these scores predicts better.

For each axis give: a short name; its level (item, benchmark as inferred from its items, subject, or subject-by-item); a definition; a 0 to 4 scoring guide with an anchor for 0, 2 and 4; the error pattern it explains, citing benchmark and pair ids from the data; and the expected direction (higher score means the judge over- or under-predicts).

Then list error patterns you saw but could not turn into a scorable axis, and why."""


def subject_line(s):
    parts = [s.get("normalized_name") or "unknown", s.get("provider") or "-", f"released {s.get('release_date') or '-'}",
             f"harness {s.get('harness') or '-'}", f"effort {s.get('reasoning_effort') or '-'}"]
    return " | ".join(parts)


def build(exam, low, seed):
    proxy = set()
    for d in PROXIES:
        proxy |= {p["benchmark_name"] for p in load_exam(os.path.join(os.path.dirname(exam["_dir"]), d))["pairs"]}
    by_bench = {}
    for pi, p in enumerate(exam["pairs"]):
        if p["benchmark_name"] in proxy:
            continue
        ts = [t for t in p["targets"] if (pi, t["key"]) in low]
        if not ts:
            continue
        by_bench.setdefault(p["benchmark_name"], []).append((pi, p, ts))
    names = sorted(by_bench)
    random.Random(seed).shuffle(names)
    lines, n_pairs = [], 0
    for bi, name in enumerate(names):
        rows = by_bench[name]
        errs = [np.mean([low[(pi, t["key"])] for t in ts]) - p["pair_success_rate"] for pi, p, ts in rows]
        lines.append(f"\n## Benchmark B{bi + 1:03d} ({len(rows)} pairs, mean judge error {np.mean(errs):+.2f})")
        for (pi, p, ts), e in zip(rows, errs):
            n_pairs += 1
            pm = np.mean([low[(pi, t["key"])] for t in ts])
            lines.append(f"Pair P{pi:03d}: {subject_line(p['subject'])}\n"
                         f"  judge mean {pm:.2f}, actual success rate {p['pair_success_rate']:.2f}, error {e:+.2f}")
            gaps = sorted(ts, key=lambda t: low[(pi, t["key"])] - np.mean(t["labels"]))
            for tag, t in (("most under-predicted", gaps[0]), ("most over-predicted", gaps[-1])):
                text = " ".join(t["item"]["item_content"].split())[:EXCERPT]
                lines.append(f"  - {tag}: judge {low[(pi, t['key'])]:.2f}, outcome rate {np.mean(t['labels']):.2f}: {text}")
    head = (f"Data: {len(names)} public benchmarks, {n_pairs} subject-benchmark pairs, Opus judge at low effort. "
            "Error = judge mean minus actual success rate (positive = over-predicted).\n")
    return head + "\n".join(lines), len(names), n_pairs


def main():
    set_dir, low_dir, out_dir, model, effort, cwd = sys.argv[1:7]
    seeds = [int(s) for s in sys.argv[7:]]
    exam = load_exam(set_dir)
    exam["_dir"] = set_dir.rstrip("/")
    low = load(low_dir)
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        path = os.path.join(out_dir, f"axes-seed{seed}.json")
        if os.path.exists(path):
            continue
        prompt, nb, npairs = build(exam, low, seed)
        cmd = ["claude", "-p", prompt, "--model", model, "--effort", effort, "--tools", "", "--setting-sources", "",
               "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}), "--system-prompt", SYSTEM,
               "--output-format", "json", "--no-session-persistence"]
        start = time.time()
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=1800,
                             stdin=subprocess.DEVNULL)
        res = None
        for line in out.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("type") == "result":
                res = d
        rec = {"seed": seed, "model": model, "effort": effort, "benchmarks": nb, "pairs": npairs,
               "prompt_chars": len(prompt), "seconds": round(time.time() - start, 1),
               "answer": (res or {}).get("result"), "error": None if res else (out.stderr or out.stdout)[:500],
               "model_usage": (res or {}).get("modelUsage")}
        json.dump(rec, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"seed {seed}: {nb} benchmarks, {npairs} pairs, prompt {len(prompt)} chars, {rec['seconds']}s, "
              f"{'ok' if res else 'error'}", flush=True)


if __name__ == "__main__":
    main()
