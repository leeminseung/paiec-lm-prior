"""Budget-0 judgments through the Claude subscription (`claude -p`) with the same system prompt, message and answer
schema as the API call in llm_prior.py, and nothing else from Claude Code (no tools, no settings, no MCP, no memory,
no session file). For local research only; submissions call the API.

Usage: python -I subscription_judge.py <exam_dir> <run_dir> <model> <effort> <workers> <empty_dir> <targets|exemplars|pool> [limit]
JUDGE_SYSTEM_SUFFIX_FILE=<path> appends that file's text to the system prompt as a separate paragraph (an instruction
variant, e.g. prompts/analyse_do_not_solve.txt); records then carry "system_suffix" with the file name.
JUDGE_SYSTEM_FILE=<path> replaces the system prompt (records carry "system_file"); JUDGE_ANSWER_KEY=<name> renames the
answer field (default "probability"); JUDGE_FREE_TEXT=1 drops the answer schema so the reply can reason in text before
its final JSON line, which is parsed and kept with the reply text (records carry "free_text": true).
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from llm_prior import SCHEMA, SYSTEM, user_message


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC") or k == "CLAUDECODE")}
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


SUFFIX_FILE = os.environ.get("JUDGE_SYSTEM_SUFFIX_FILE")
SYSTEM_FILE = os.environ.get("JUDGE_SYSTEM_FILE")
ANSWER_KEY = os.environ.get("JUDGE_ANSWER_KEY", "probability")
FREE_TEXT = os.environ.get("JUDGE_FREE_TEXT") == "1"
SYSTEM_USED = (open(SYSTEM_FILE).read().strip() if SYSTEM_FILE else SYSTEM) + \
    ("\n\n" + open(SUFFIX_FILE).read().strip() if SUFFIX_FILE else "")
SCHEMA_USED = SCHEMA if ANSWER_KEY == "probability" else {
    "type": "object", "additionalProperties": False, "required": [ANSWER_KEY],
    "properties": {ANSWER_KEY: {"type": "number", "minimum": 0, "maximum": 1}}}


def last_json(text):
    """The last {...} object in a free-text reply that parses as JSON."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            j = text.find("}", i)
            try:
                return json.loads(text[i:j + 1])
            except ValueError:
                continue
    raise ValueError("no JSON object")


def judge(subject, item, model, effort, cwd, timeout=300):
    cmd = ["claude", "-p", user_message(subject, item), "--model", model, "--effort", effort, "--tools", "",
           "--setting-sources", "", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
           "--system-prompt", SYSTEM_USED] + ([] if FREE_TEXT else ["--json-schema", json.dumps(SCHEMA_USED)]) + \
          ["--output-format", "json", "--no-session-persistence"]
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
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
    so = res.get("structured_output")
    try:
        ans = so if (so and not FREE_TEXT) else (last_json(res.get("result", "")) if FREE_TEXT else json.loads(res.get("result", "")))
        p = float(ans[ANSWER_KEY])
        p = p if 0 <= p <= 1 else None
    except (TypeError, ValueError, KeyError):
        p = None
    return {"probability": p, "error": None if p is not None else f"unparsed: {str(res.get('result'))[:200]}",
            "seconds": round(time.time() - start, 1), "usage": res.get("usage"), "model_usage": res.get("modelUsage"),
            "cost_usd_equiv": res.get("total_cost_usd"), "is_error": res.get("is_error"),
            **({"text": res.get("result")} if FREE_TEXT else {})}


def main():
    exam_dir, run_dir, model, effort, workers, cwd, role = sys.argv[1:8]
    limit = int(sys.argv[8]) if len(sys.argv) > 8 else None
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["pair"], r["key"]) for r in map(json.loads, open(out_path)) if r.get("probability") is not None}
    # targets; exemplars = the first 31 acquisitions; pool = acquisitions after the first 31 (needs a full_pool exam)
    pick = {"targets": lambda pair: pair["targets"], "exemplars": lambda pair: pair["acquisitions"][:31],
            "pool": lambda pair: pair["acquisitions"][31:]}[role]
    jobs = [(pi, it) for pi, pair in enumerate(exam["pairs"]) for it in pick(pair) if (pi, it["key"]) not in done]
    if limit is not None:
        jobs = jobs[:limit]
    lock = threading.Lock()

    def run(job):
        pi, it = job
        r = judge(exam["pairs"][pi]["subject"], it["item"], model, effort, cwd)
        rec = {"budget": 0, "pair": pi, "key": it["key"], "role": role, "model": model, "effort": effort, "via": "claude -p",
               **({"system_suffix": os.path.basename(SUFFIX_FILE)} if SUFFIX_FILE else {}),
               **({"system_file": os.path.basename(SYSTEM_FILE)} if SYSTEM_FILE else {}),
               **({"answer_key": ANSWER_KEY} if ANSWER_KEY != "probability" else {}),
               **({"free_text": True} if FREE_TEXT else {}), **r}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"{len(jobs)} judgments", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, jobs))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
