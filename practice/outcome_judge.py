"""Judgments that see the pair's revealed results: one `claude -p` request per (target, budget) whose system prompt holds
the instructions, the subject and the pair's first b exemplars with their outcomes, and whose user message is the target
item alone. The 10 targets of one (pair, budget) share that system prompt, so after the first request writes it to the
prompt cache the others read it; each group's first request runs alone, the rest follow.

Exemplar texts are cut to 4,000 characters (middle cut) as in batched_judge.py, the target to 12,000 as in llm_prior.py.
Targets: 10 per pair, drawn as in batched_judge.py (random.Random(0) over each pair's sorted target keys, pairs in order).
For local research through the subscription; a submission would need the same prompt through the API.

Usage: python -I outcome_judge.py <exam_dir> <run_dir> <model> <effort> <groups_in_parallel> <calls_per_group> <empty_dir>
                                  <budgets e.g. 1,3,7> <per_pair>
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
from batched_judge import cut
from common import load_exam
from llm_prior import SCHEMA, user_message
from subscription_judge import clean_env

SYSTEM = """You estimate the probability that a given AI system (the subject) answers a given benchmark item (the target) correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. Each item gives the task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. Empty values mean unknown.

You also see results this same subject got on other items of the same benchmark (exemplars): each exemplar gives the item and its outcome, 1 if the subject succeeded and 0 if it failed. The exemplars were drawn at random from this subject's items on the benchmark. Use them as evidence about how capable the subject is on this benchmark, how strictly the benchmark grades, and which kinds of items it gets right or wrong. A few results are a small sample, so weigh them against what you know about the subject and the benchmark.

Think about how capable this system is under these settings, how hard the target item is, and how strictly tasks like this are graded. Your answer is scored by Brier score, the squared error between your probability and the 0/1 outcome, so give a calibrated probability.

Answer with JSON: {"probability": <number from 0 to 1>}."""


def system_prompt(subject, exemplars):
    rows = [{"index": i + 1, "item_content": cut(a["item"].get("item_content", ""), 4000),
             "item_features": a["item"].get("item_features", ""), "interactors": a["item"].get("interactors", ""),
             "outcome": int(a["label"])} for i, a in enumerate(exemplars)]
    block = json.dumps({"subject": subject, "exemplars": rows}, ensure_ascii=False, indent=1)
    return f"{SYSTEM}\n\nThe subject and its {len(rows)} exemplar result(s):\n{block}"


def call(system, target_item, model, effort, cwd, timeout=600):
    msg = json.loads(user_message({}, target_item))["item"]
    cmd = ["claude", "-p", json.dumps({"target": msg}, ensure_ascii=False, indent=1), "--model", model, "--effort", effort,
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
           "--system-prompt", system, "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--no-session-persistence"]
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"probability": None, "error": "timeout", "seconds": round(time.time() - start, 1)}
    res = None
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            res = d
    if res is None:
        return {"probability": None, "error": (out.stderr or out.stdout)[:300], "seconds": round(time.time() - start, 1)}
    p = None
    try:
        p = float((res.get("structured_output") or json.loads(res.get("result", "")))["probability"])
        p = p if 0 <= p <= 1 else None
    except (TypeError, ValueError, KeyError):
        p = None
    return {"probability": p, "error": None if p is not None else f"unparsed: {str(res.get('result'))[:200]}",
            "seconds": round(time.time() - start, 1), "usage": res.get("usage"), "cost_usd_equiv": res.get("total_cost_usd")}


def chosen_targets(exam, per_pair):
    rng = random.Random(0)
    out = {}
    for pi, pair in enumerate(exam["pairs"]):
        keys = sorted(t["key"] for t in pair["targets"])
        out[pi] = set(rng.sample(keys, min(per_pair, len(keys))))
    return out


def main():
    exam_dir, run_dir, model, effort, n_groups, per_group, cwd, budgets, per_pair = sys.argv[1:10]
    budgets = [int(b) for b in budgets.split(",")]
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["budget"], r["pair"], r["key"]) for r in map(json.loads, open(out_path)) if r.get("probability") is not None}
    chosen = chosen_targets(exam, int(per_pair))
    groups = []
    for b in budgets:
        for pi, pair in enumerate(exam["pairs"]):
            ts = [t for t in pair["targets"] if t["key"] in chosen[pi] and (b, pi, t["key"]) not in done]
            if ts:
                groups.append((b, pi, ts))
    lock = threading.Lock()

    def one(b, pi, t, system):
        r = call(system, t["item"], model, effort, cwd)
        rec = {"budget": b, "pair": pi, "key": t["key"], "model": model, "effort": effort, "via": "claude -p",
               "exemplar_keys": [a["key"] for a in exam["pairs"][pi]["acquisitions"][:b]], **r}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def run_group(g):
        b, pi, ts = g
        system = system_prompt(exam["pairs"][pi]["subject"], exam["pairs"][pi]["acquisitions"][:b])
        one(b, pi, ts[0], system)  # writes the shared prefix to the cache
        with ThreadPoolExecutor(int(per_group)) as inner:
            list(inner.map(lambda t: one(b, pi, t, system), ts[1:]))

    print(f"{len(groups)} groups, {sum(len(g[2]) for g in groups)} calls", flush=True)
    with ThreadPoolExecutor(int(n_groups)) as ex:
        list(ex.map(run_group, groups))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
