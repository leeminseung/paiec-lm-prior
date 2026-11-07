"""Batched judging test: one `claude -p` request per (target, budget) holding the subject, the pair's first b exemplar
items (text only, no outcomes) and the target item last; the model returns one probability per item in order.

Exemplar texts are cut to 4,000 characters (middle cut), the target to 12,000 as in llm_prior.py.
Usage: python -I batched_judge.py <exam_dir> <run_dir> <model> <effort> <workers> <empty_dir> <budgets e.g. 7,31> <per_pair> [limit]
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from subscription_judge import clean_env

SYSTEM = """You estimate, for each of several benchmark items, the probability that a given AI system (the subject) answers that item correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. Each item gives the task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. All items come from the same benchmark. Empty values mean unknown.

Think about how capable this system is under these settings, how hard each particular item is, and how strictly tasks like this are usually graded. Each probability is scored by Brier score, the squared error between the probability and the 0/1 outcome, so give calibrated probabilities. Judge every item on its own merits; the items are listed in no particular order of difficulty.

Answer with JSON: {"probabilities": [<one number from 0 to 1 per item, in the order the items are listed>]}."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probabilities"],
          "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}}}


def cut(text, limit):
    text = text or ""
    if len(text) <= limit:
        return text
    h = limit // 2
    return text[:h] + f"\n[... {len(text) - limit} characters omitted ...]\n" + text[-h:]


def message(subject, items, limits):
    rows = [{"index": i + 1, "item_content": cut(it.get("item_content", ""), lim), "item_features": it.get("item_features", ""),
             "interactors": it.get("interactors", ""), "benchmark_id": it.get("benchmark_id", "")}
            for i, (it, lim) in enumerate(zip(items, limits))]
    return json.dumps({"subject": subject, "items": rows}, ensure_ascii=False, indent=1)


def call(subject, items, limits, model, effort, cwd, timeout=600):
    cmd = ["claude", "-p", message(subject, items, limits), "--model", model, "--effort", effort, "--tools", "",
           "--setting-sources", "", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
           "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--no-session-persistence"]
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"probs": None, "error": "timeout", "seconds": round(time.time() - start, 1)}
    res = None
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            res = d
    if res is None:
        return {"probs": None, "error": (out.stderr or out.stdout)[:300], "seconds": round(time.time() - start, 1)}
    probs = None
    try:
        so = res.get("structured_output") or json.loads(res.get("result", ""))
        probs = [float(x) for x in so["probabilities"]]
        if len(probs) != len(items) or not all(0 <= x <= 1 for x in probs):
            probs = None
    except (TypeError, ValueError, KeyError):
        probs = None
    return {"probs": probs, "error": None if probs else f"bad answer: {str(res.get('result'))[:200]}",
            "seconds": round(time.time() - start, 1), "usage": res.get("usage"), "cost_usd_equiv": res.get("total_cost_usd")}


def main():
    exam_dir, run_dir, model, effort, workers, cwd, budgets, per_pair = sys.argv[1:9]
    limit = int(sys.argv[9]) if len(sys.argv) > 9 else None
    budgets = [int(b) for b in budgets.split(",")]
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "batched.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["pair"], r["key"], r["budget"]) for r in map(json.loads, open(out_path)) if r.get("probs")}
    rng = random.Random(0)
    jobs = []
    for pi, pair in enumerate(exam["pairs"]):
        keys = sorted(t["key"] for t in pair["targets"])
        chosen = set(rng.sample(keys, min(int(per_pair), len(keys))))
        for t in pair["targets"]:
            if t["key"] in chosen:
                for b in budgets:
                    if (pi, t["key"], b) not in done:
                        jobs.append((pi, t, b))
    jobs.sort(key=lambda j: (j[2], j[0]))
    if limit is not None:
        jobs = jobs[:limit]
    lock = threading.Lock()

    def run(job):
        pi, t, b = job
        ex = exam["pairs"][pi]["acquisitions"][:b]
        items = [a["item"] for a in ex] + [t["item"]]
        r = call(exam["pairs"][pi]["subject"], items, [4000] * len(ex) + [12000], model, effort, cwd)
        rec = {"pair": pi, "key": t["key"], "budget": b, "exemplar_keys": [a["key"] for a in ex], **r}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"{len(jobs)} batched calls", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex_:
        list(ex_.map(run, jobs))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
