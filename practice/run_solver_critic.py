"""Run weak solvers and the critic over practice-exam evaluation items through the Claude API (resumable, spend cap).

Phase 1: for each selected item and each solver setting, N_ATTEMPTS attempts by the solver model (sees the item only).
Phase 2: for each item and each critic variant, one critic call that sees the subject, the item and that setting's
attempts. Variants: setting "off" -> answers + explanations ("ae"); setting "on" -> "ae" and "aes" (also the thinking
summaries and output-token counts), so the only difference between the two "on" critics is the summaries.
Attempts that never got an API response are left out of the critic's input; refusals and output-limit stops stay in.

Selection "slice": three pairs (Robust Reasoning, MedCalc, SWE-rebench), 15 items each, with up to 7 from the pair's
minority side (items the subject mostly failed, or mostly solved), seed 0. Selection "all": every evaluation item.

Usage: python -I run_solver_critic.py <exam_dir> <run_dir> <key: keychain:<service> or a file> <slice|all> <settings e.g. off,on> <budget_usd> [workers] [variants]
variants (optional, comma-separated) replaces the default critic variants; a trailing 2 (ae2, aes2) uses critic prompt version 2
"""
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import api_key, load_exam
from solver_critic import agreement, critic, same_model, solve

SOLVER_MODEL, CRITIC_MODEL = "claude-haiku-5-5", "claude-sonnet-5-5"
N_ATTEMPTS = 4
SLICE_BENCHMARKS = ["robust_reasoning_benchmark", "medhelm_medcalc_bench", "swe_rebench"]
SLICE_PER_PAIR, SLICE_MINORITY_MAX = 15, 7
VARIANTS = {"off": ["ae"], "on": ["ae", "aes"]}
# USD per million tokens: input, output (shared/models.md of the claude-api reference, read 2026-10-20)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def cost(model, usage):
    p_in, p_out = PRICES[model]
    return (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out
            + usage.get("cache_read_input_tokens", 0) * p_in * 0.1
            + usage.get("cache_creation_input_tokens", 0) * p_in * 1.25) / 1e6


def select(exam, mode):
    if mode == "all":
        return [(pi, t["key"]) for pi, p in enumerate(exam["pairs"]) for t in p["targets"]]
    rng, chosen = random.Random(0), []
    for pi, p in enumerate(exam["pairs"]):
        if p["benchmark_name"] not in SLICE_BENCHMARKS:
            continue
        solved = [t["key"] for t in p["targets"] if sum(t["labels"]) * 2 > len(t["labels"])]
        other = [t["key"] for t in p["targets"] if sum(t["labels"]) * 2 <= len(t["labels"])]
        minority, majority = (solved, other) if len(solved) < len(other) else (other, solved)
        k = min(SLICE_MINORITY_MAX, len(minority))
        chosen += [(pi, key) for key in rng.sample(minority, k) + rng.sample(majority, SLICE_PER_PAIR - k)]
    return chosen


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main():
    exam_dir, run_dir, key_file, mode, settings, budget = sys.argv[1:7]
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    settings, budget = settings.split(","), float(budget)
    variants = {s: sys.argv[8].split(",") for s in settings} if len(sys.argv) > 8 else VARIANTS
    api_key = api_key(key_file)
    exam = load_exam(exam_dir)
    targets = {(p_i, t["key"]): t for p_i, p in enumerate(exam["pairs"]) for t in p["targets"]}
    os.makedirs(run_dir, exist_ok=True)
    chosen = select(exam, mode)
    with open(os.path.join(run_dir, "selection.json"), "w") as f:
        json.dump({"mode": mode, "items": chosen}, f)
    att_path, cri_path = os.path.join(run_dir, "attempts.jsonl"), os.path.join(run_dir, "critic.jsonl")
    attempts, critics = load_jsonl(att_path), load_jsonl(cri_path)
    lock = threading.Lock()
    spent = {"usd": sum(r["cost_usd"] for r in attempts + critics), "new": 0, "skipped_budget": 0}
    print(f"{len(chosen)} items, settings {settings}, budget ${budget:.2f}, already spent ${spent['usd']:.3f}", flush=True)

    def over_budget():
        with lock:
            if spent["usd"] >= budget:
                spent["skipped_budget"] += 1
                return True
            return False

    def record(path, rec, store):
        with lock:
            store.append(rec)
            spent["usd"] += rec["cost_usd"]
            spent["new"] += 1
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if spent["new"] % 25 == 0:
                print(f"  {spent['new']} new calls, total spent ${spent['usd']:.3f}", flush=True)

    # Phase 1: solver attempts (an attempt is done once the API answered, whatever the answer was)
    done = {(r["setting"], r["pair"], r["key"], r["idx"]) for r in attempts if r["usage"]}
    jobs = [(s, pi, key, i) for s in settings for pi, key in chosen for i in range(N_ATTEMPTS)
            if (s, pi, key, i) not in done and not same_model(exam["pairs"][pi]["subject"], SOLVER_MODEL)]

    def run_solver(job):
        s, pi, key, i = job
        if over_budget():
            return
        res = solve(targets[(pi, key)]["item"], api_key, SOLVER_MODEL, thinking=s)
        record(att_path, {"setting": s, "pair": pi, "key": key, "idx": i, "model": SOLVER_MODEL, **res,
                          "cost_usd": round(cost(SOLVER_MODEL, res["usage"]), 6)}, attempts)

    print(f"phase 1: {len(jobs)} solver calls", flush=True)
    with ThreadPoolExecutor(workers * N_ATTEMPTS) as ex:
        list(ex.map(run_solver, jobs))

    # Phase 2: critic calls
    latest = {}
    for r in attempts:
        if r["usage"]:
            latest[(r["setting"], r["pair"], r["key"], r["idx"])] = r
    # a critic call is done once the API answered (a refusal is an outcome, not something to retry)
    done = {(r["setting"], r["variant"], r["pair"], r["key"]) for r in critics if r["usage"]}
    jobs = [(s, v, pi, key) for s in settings for v in variants[s] for pi, key in chosen if (s, v, pi, key) not in done]

    def run_critic(job):
        s, v, pi, key = job
        subject = exam["pairs"][pi]["subject"]
        att = [latest[(s, pi, key, i)] for i in range(N_ATTEMPTS) if (s, pi, key, i) in latest]
        if len(att) < 2 or same_model(subject, CRITIC_MODEL) or over_budget():
            return
        res = critic(subject, targets[(pi, key)]["item"], att, api_key, CRITIC_MODEL, with_summaries=v.startswith("aes"),
                     version="v2" if v.endswith("2") else "v1")
        record(cri_path, {"setting": s, "variant": v, "pair": pi, "key": key, "model": CRITIC_MODEL,
                          "n_attempts": len(att), "agreement": agreement(att), **res,
                          "cost_usd": round(cost(CRITIC_MODEL, res["usage"]), 6)}, critics)

    print(f"phase 2: {len(jobs)} critic calls", flush=True)
    with ThreadPoolExecutor(workers * 2) as ex:
        list(ex.map(run_critic, jobs))
    print(f"finished: {spent['new']} new calls, total spent ${spent['usd']:.3f},"
          f" calls skipped by the budget cap {spent['skipped_budget']}", flush=True)


if __name__ == "__main__":
    main()
