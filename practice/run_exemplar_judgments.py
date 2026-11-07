"""Budget-0 judgments for the few-shot exemplars of a practice exam (the acquisition items each pair reveals).

Same call as the round-2 submission (llm_prior.judge: the subject and the item, no labels, no tools). Writes
<run_dir>/predictions.jsonl with one record per (pair, exemplar key); resumable.

Usage: python -I run_exemplar_judgments.py <exam_dir> <run_dir> <model> <effort> <adaptive|off> <workers> <key: keychain:<service> or a file> [max_per_pair]
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import api_key, load_exam
from llm_prior import judge
from run_llm_prior import cost


def main():
    exam_dir, run_dir, model, effort, thinking, workers, key_file = sys.argv[1:8]
    max_per_pair = int(sys.argv[8]) if len(sys.argv) > 8 else 31
    api_key = api_key(key_file)
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            if r.get("probability") is not None or r.get("stop_reason") == "refusal":
                done.add((r["pair"], r["key"]))
    jobs = [(pi, a) for pi, pair in enumerate(exam["pairs"]) for a in pair["acquisitions"][:max_per_pair]
            if (pi, a["key"]) not in done]
    lock, totals = threading.Lock(), {"n": 0, "cost": 0.0, "errors": 0}

    def run(job):
        pi, a = job
        res = judge(exam["pairs"][pi]["subject"], a["item"], api_key, model, effort, thinking)
        rec = {"budget": 0, "pair": pi, "key": a["key"], "role": "exemplar", "probability": res["probability"],
               "cost_usd_equiv": round(cost(model, res["usage"]), 6), "input_tokens": res["usage"].get("input_tokens"),
               "output_tokens": res["usage"].get("output_tokens"), "stop_reason": res["stop_reason"], "error": res["error"],
               "model": model, "effort": effort, "thinking": thinking, "attempts": res["attempts"]}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            totals["n"] += 1
            totals["cost"] += rec["cost_usd_equiv"]
            totals["errors"] += rec["probability"] is None

    print(f"{len(jobs)} exemplar judgments to get", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, jobs))
    print(f"finished {totals['n']} calls, {totals['errors']} without a probability, ${totals['cost']:.3f}", flush=True)


if __name__ == "__main__":
    main()
