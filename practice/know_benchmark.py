"""Does a judge model recognize a benchmark from its items? Offline analysis for the report (memory vs generalization).

For every benchmark in the given exams (the 847-item measurement-db subset and the proxy evaluations), the model sees
three of its items (the first three targets of its first pair by key, each cut to 1,200 characters), with no subject and
no benchmark name, and answers whether it recognizes the benchmark, its name, a confidence, and what it recalls about
how AI systems score on it. Recognition is later checked against the true name, so a model's "yes" is not taken on
trust. This is analysis on public training benchmarks only; the submitted predictor never asks the judge to name a
benchmark (competition rule on anonymous benchmark ids).

Backends: claude -p through the subscription (clean environment as in subscription_judge.py), or OpenRouter with a
pinned endpoint (key file as in openrouter_judge.py, never printed).

Usage: python -I know_benchmark.py <out_dir> claude <model> <effort> <workers> <empty_dir> <exam_dir> [...]
       python -I know_benchmark.py <out_dir> openrouter <model> <endpoint_tag> <workers> <effort|default> <cap_usd> <exam_dir> [...]
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from openrouter_judge import KEY_FILE, URL
from subscription_judge import clean_env

EXCERPT, N_ITEMS = 1200, 3

SYSTEM = """You will see three items from one public AI evaluation benchmark. The benchmark's name has been removed.

Say whether you recognize which benchmark these items come from. Recognize means you know this benchmark from your training (its name, its source or its published results), not merely the general task type. If you recognize it, give its name and say briefly what you recall about how AI systems score on it. If you do not, say so; a confident wrong name is worse than "not recognized".

Answer with JSON only: {"recognized": true or false, "benchmark_name": "<name or empty>", "confidence": <number from 0 to 1>, "known_results": "<one or two sentences, or empty>"}."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["recognized", "benchmark_name", "confidence", "known_results"],
          "properties": {"recognized": {"type": "boolean"}, "benchmark_name": {"type": "string"},
                         "confidence": {"type": "number"}, "known_results": {"type": "string"}}}


def benchmarks(exam_dirs):
    seen = {}
    for d in exam_dirs:
        for p in load_exam(d)["pairs"]:
            if p["benchmark_name"] in seen:
                continue
            items = sorted(p["targets"], key=lambda t: t["key"])[:N_ITEMS]
            seen[p["benchmark_name"]] = {"benchmark": p["benchmark_name"], "source_exam": d,
                                         "keys": [t["key"] for t in items],
                                         "items": [" ".join(t["item"]["item_content"].split())[:EXCERPT] for t in items]}
    return list(seen.values())


def message(b):
    return "\n\n".join(f"Item {i + 1}:\n{text}" for i, text in enumerate(b["items"]))


def parse(text):
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {"recognized": bool(d["recognized"]), "benchmark_name": str(d.get("benchmark_name", "")),
                "confidence": float(d.get("confidence", 0)), "known_results": str(d.get("known_results", ""))}
    except (ValueError, KeyError, TypeError):
        return None


def ask_claude(b, model, effort, cwd, timeout=300):
    cmd = ["claude", "-p", message(b), "--model", model, "--effort", effort, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}), "--system-prompt", SYSTEM,
           "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--no-session-persistence"]
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"answer": None, "error": "timeout", "seconds": round(time.time() - start, 1)}
    res = None
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            res = d
    if res is None:
        return {"answer": None, "error": (out.stderr or out.stdout)[:300], "seconds": round(time.time() - start, 1)}
    so = res.get("structured_output")
    ans = parse(json.dumps(so)) if so else parse(res.get("result", ""))
    return {"answer": ans, "error": None if ans else f"unparsed: {str(res.get('result'))[:200]}",
            "seconds": round(time.time() - start, 1), "model_usage": res.get("modelUsage")}


def ask_openrouter(b, model, tag, key, effort, attempts=4, timeout=300):
    body = {"model": model, "max_tokens": int(os.environ.get("KNOW_MAX_TOKENS", "24000")),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message(b)}],
            "response_format": {"type": "json_object"},
            "provider": {"order": [tag], "allow_fallbacks": False, "data_collection": "deny", "require_parameters": True}}
    if effort:
        body["reasoning"] = {"effort": effort}
    start, err = time.time(), None
    for attempt in range(attempts):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            ans = parse(msg.get("content") or "")
            u = data.get("usage", {})
            return {"answer": ans, "error": None if ans else f"unparsed: {str(msg.get('content'))[:200]}",
                    "seconds": round(time.time() - start, 1), "attempts": attempt + 1,
                    "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                    "cost_usd": u.get("cost", 0) or 0}
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, IndexError, ValueError) as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt)
    return {"answer": None, "error": err, "seconds": round(time.time() - start, 1), "attempts": attempts, "cost_usd": 0}


def main():
    out_dir, backend, model = sys.argv[1:4]
    if backend == "claude":
        effort, workers, cwd = sys.argv[4:7]
        exam_dirs, cap = sys.argv[7:], float("inf")
    else:
        tag, workers, effort, cap = sys.argv[4:8]
        exam_dirs, cap = sys.argv[8:], float(cap)
        effort = None if effort == "default" else effort
        key = open(os.path.expanduser(KEY_FILE)).read().strip()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "answers.jsonl")
    old = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {r["benchmark"] for r in old if r.get("answer")}
    spent = [sum(r.get("cost_usd", 0) or 0 for r in old)]
    jobs = [b for b in benchmarks(exam_dirs) if b["benchmark"] not in done]
    if os.environ.get("KNOW_LIMIT"):  # a small probe before the full run
        jobs = jobs[:int(os.environ["KNOW_LIMIT"])]
    lock = threading.Lock()

    def run(b):
        with lock:
            if spent[0] >= cap:
                return
        r = ask_claude(b, model, effort, cwd) if backend == "claude" else ask_openrouter(b, model, tag, key, effort)
        rec = {"benchmark": b["benchmark"], "source_exam": b["source_exam"], "keys": b["keys"], "backend": backend,
               "model": model, "effort": effort, **({"endpoint": tag} if backend != "claude" else {}), **r}
        with lock:
            spent[0] += rec.get("cost_usd", 0) or 0
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{len(jobs)} benchmarks to ask {model}", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, jobs))
    print(f"finished, spent ${spent[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
