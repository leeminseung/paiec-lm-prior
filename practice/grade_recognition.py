"""Check each "recognized" answer from know_benchmark.py against the benchmark's catalog record (report flow 3).

Many measurement-db benchmarks repackage an older dataset (a HELM run of BBQ, a method paper's runs on BIRD), so for
memory what matters is whether the model knows the items' source. A grader model sees the benchmark's catalog name,
id, one-line description and description (cut to 800 characters) and the guessed name, and answers "benchmark" (the
guess names this benchmark), "source" (it names the dataset the items come from) or "no". Only answers with
recognized = true are graded; the rest stay "not recognized". Resumable, keyed by (run, benchmark).

Usage: python -I grade_recognition.py <out.jsonl> <grader_model> <effort> <workers> <empty_dir> <know_run_dir> [...]
"""
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from subscription_judge import clean_env

CATALOG = "./agent_box_02/data/catalog/benchmarks_all.parquet"
SYSTEM = """You check whether a model correctly identified a benchmark. You get the benchmark's catalog record and the name the model guessed.

Answer "benchmark" if the guess names this benchmark (allowing abbreviations, versions and wording differences), "source" if it names the dataset or benchmark this benchmark's items come from (as stated or clearly implied by the record), and "no" otherwise, including a guess that names only the general task type.

Answer with JSON: {"match": "benchmark" | "source" | "no"}."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["match"],
          "properties": {"match": {"type": "string", "enum": ["benchmark", "source", "no"]}}}


def ask(record, guess, model, effort, cwd):
    msg = json.dumps({"catalog_record": record, "guessed_name": guess}, ensure_ascii=False, indent=1)
    cmd = ["claude", "-p", msg, "--model", model, "--effort", effort, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}), "--system-prompt", SYSTEM,
           "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--no-session-persistence"]
    out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=300,
                         stdin=subprocess.DEVNULL)
    for line in reversed(out.stdout.splitlines()):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            so = d.get("structured_output") or {}
            if so.get("match") in ("benchmark", "source", "no"):
                return so["match"], None
            return None, f"unparsed: {str(d.get('result'))[:200]}"
    return None, (out.stderr or out.stdout)[:200]


def main():
    out_path, model, effort, workers, cwd = sys.argv[1:6]
    runs = sys.argv[6:]
    cat = pd.read_parquet(CATALOG).set_index("folder")
    done = set()
    if os.path.exists(out_path):
        done = {(r["run"], r["benchmark"]) for r in map(json.loads, open(out_path)) if r.get("match")}
    jobs = []
    for run in runs:
        for r in map(json.loads, open(os.path.join(run, "answers.jsonl"))):
            a = r.get("answer")
            key = (os.path.basename(run.rstrip("/")), r["benchmark"])
            if a and a["recognized"] and key not in done:
                c = cat.loc[r["benchmark"]] if r["benchmark"] in cat.index else None
                record = {"name": r["benchmark"]} if c is None else {
                    "name": c["name"], "id": c["benchmark_id"], "one_line_description": c["one_line_description"],
                    "description": str(c["description"])[:800]}
                jobs.append((key, record, a["benchmark_name"]))
    jobs = list({k: (k, rec, g) for k, rec, g in jobs}.values())
    lock = threading.Lock()

    def run_one(job):
        key, record, guess = job
        m, err = ask(record, guess, model, effort, cwd)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps({"run": key[0], "benchmark": key[1], "guess": guess, "match": m, "error": err,
                                    "grader": model}, ensure_ascii=False) + "\n")

    print(f"{len(jobs)} recognized answers to grade", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run_one, jobs))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
