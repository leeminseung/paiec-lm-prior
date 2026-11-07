"""Get one budget-0 LLM judgment for every evaluation item of a practice exam through the Claude API.

Writes <run_dir>/predictions.jsonl in the format compare_runs.py and combine_eval.py read (budget 0 only).
Already answered items are skipped, so an interrupted run can be resumed. The API key is read from a file and
never printed. Cost is computed from the token counts each response reports.

Usage: python -I run_llm_prior.py <exam_dir> <run_dir> <model> <effort> <adaptive|off> <workers> <key: keychain:<service> or a file> [limit]
"""
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import api_key, load_exam
from llm_prior import judge

# USD per million tokens: input, output, cache read (shared/models.md of the claude-api reference, 2026-10-20)
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-haiku-5-5": (0.10, 0.50, 0.01), "claude-opus-5-5": (4.0, 20.0, 0.40)}


def cost(model, usage):
    p_in, p_out, p_cache = PRICES[model]
    return (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out
            + usage.get("cache_read_input_tokens", 0) * p_cache + usage.get("cache_creation_input_tokens", 0) * p_in * 1.25) / 1e6


def main():
    exam_dir, run_dir, model, effort, thinking, workers, key_file = sys.argv[1:8]
    limit = int(sys.argv[8]) if len(sys.argv) > 8 else None
    api_key = api_key(key_file)
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            if r.get("probability") is not None:
                done.add((r["pair"], r["key"]))
    jobs = [(pi, t) for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"] if (pi, t["key"]) not in done]
    if limit is not None:
        jobs = jobs[:limit]
    lock, totals = threading.Lock(), {"n": 0, "cost": 0.0, "errors": 0}

    def run(job):
        pi, t = job
        res = judge(exam["pairs"][pi]["subject"], t["item"], api_key, model, effort, thinking)
        u = res["usage"]
        rec = {"call_id": f"b00_p{pi}_{hashlib.sha1(str(t['key']).encode()).hexdigest()[:6]}", "budget": 0, "pair": pi,
               "key": t["key"], "status": "ok" if res["probability"] is not None else "error",
               "probability": res["probability"], "rationale": None, "wall_s": res["seconds"], "num_turns": 1,
               "cost_usd_equiv": round(cost(model, u), 6), "input_tokens": u.get("input_tokens"),
               "cache_read_tokens": u.get("cache_read_input_tokens"), "cache_write_tokens": u.get("cache_creation_input_tokens"),
               "output_tokens": u.get("output_tokens"), "tools": {}, "bash": [], "labeled_n": 0,
               "model": model, "effort": effort, "thinking": thinking, "stop_reason": res["stop_reason"], "error": res["error"],
               "attempts": res["attempts"]}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            totals["n"] += 1
            totals["cost"] += rec["cost_usd_equiv"]
            totals["errors"] += rec["status"] != "ok"
            if totals["n"] % 50 == 0 or totals["n"] == len(jobs):
                print(f"{totals['n']}/{len(jobs)} done | errors {totals['errors']} | cost so far ${totals['cost']:.3f}", flush=True)

    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, jobs))
    print(f"finished {totals['n']} calls, {totals['errors']} errors, ${totals['cost']:.3f}")


if __name__ == "__main__":
    main()
