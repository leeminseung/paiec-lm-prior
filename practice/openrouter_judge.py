"""Budget-0 judgments from open-weight models through OpenRouter, with the same system prompt, message and answer format
as the Claude judge in llm_prior.py. One pinned endpoint per run (no fallbacks, provider data collection denied),
a dollar cap on the run, and per-call tokens and cost recorded. For local research.

Usage: python -I openrouter_judge.py <exam_dir> <run_dir> <model> <endpoint_tag> <workers> <targets|exemplars> <cap_usd>
                                     [per_pair] [reasoning_effort] [schema|json]
per_pair: judge only the first n items of each pair (a probe), or all; reasoning_effort: e.g. low, or default for the
model's own; json: ask for a JSON object without the schema, for endpoints without structured outputs.
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from llm_prior import SCHEMA, SYSTEM, user_message

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = "~/.config/openrouter/api_key"
MAX_TOKENS = 24000  # thinking plus answer; at least 4,096 above any thinking the endpoint allows (compute skill)


def judge(subject, item, model, tag, key, effort=None, fmt="schema", attempts=4, timeout=300):
    body = {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(subject, item)}],
            "response_format": ({"type": "json_schema", "json_schema": {"name": "judgment", "strict": True, "schema": SCHEMA}}
                                if fmt == "schema" else {"type": "json_object"}),
            "provider": {"order": [tag], "allow_fallbacks": False, "data_collection": "deny", "require_parameters": True}}
    if effort:
        body["reasoning"] = {"effort": effort}
    start, err = time.time(), None
    for attempt in range(attempts):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.load(resp)
        except urllib.error.HTTPError as e:
            err = f"http {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (408, 429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            break
        except Exception as e:  # network errors and timeouts
            err = f"{type(e).__name__}: {e}"[:300]
            time.sleep(5 * (attempt + 1))
            continue
        if "error" in d:
            err = json.dumps(d["error"])[:300]
            time.sleep(5 * (attempt + 1))
            continue
        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        p = None
        try:
            p = float(json.loads(text)["probability"])
        except (ValueError, KeyError, TypeError):
            m = re.search(r'"probability"\s*:\s*([0-9.]+)', text)
            p = float(m.group(1)) if m else None
        u = d.get("usage") or {}
        return {"probability": p if p is not None and 0 <= p <= 1 else None,
                "error": None if p is not None else f"unparsed: {text[:200]}",
                "finish_reason": d["choices"][0].get("finish_reason"), "provider": d.get("provider"),
                "seconds": round(time.time() - start, 1), "attempts": attempt + 1,
                "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "cost_usd": u.get("cost")}
    return {"probability": None, "error": err, "seconds": round(time.time() - start, 1), "attempts": attempts, "cost_usd": 0}


def main():
    exam_dir, run_dir, model, tag, workers, role, cap = sys.argv[1:8]
    per_pair = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != "all" else None
    effort = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9] != "default" else None
    fmt = sys.argv[10] if len(sys.argv) > 10 else "schema"
    key = open(os.path.expanduser(KEY_FILE)).read().strip()
    exam = load_exam(exam_dir)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "predictions.jsonl")
    old = [json.loads(line) for line in open(out_path)] if os.path.exists(out_path) else []
    done = {(r["pair"], r["key"]) for r in old if r.get("probability") is not None}
    spent = {"usd": sum(r.get("cost_usd") or 0 for r in old), "skipped": 0}
    pool = "targets" if role == "targets" else "acquisitions"
    jobs = []
    for pi, pair in enumerate(exam["pairs"]):
        items = pair[pool] if pool == "targets" else pair[pool][:31]
        jobs += [(pi, it) for it in items[:per_pair] if (pi, it["key"]) not in done]
    lock = threading.Lock()

    def run(job):
        pi, it = job
        with lock:
            if spent["usd"] >= float(cap):
                spent["skipped"] += 1
                return
        r = judge(exam["pairs"][pi]["subject"], it["item"], model, tag, key, effort, fmt)
        rec = {"budget": 0, "pair": pi, "key": it["key"], "role": role, "model": model, "endpoint": tag,
               "effort": effort, "format": fmt, "via": "openrouter", **r}
        with lock:
            spent["usd"] += r.get("cost_usd") or 0
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"{len(jobs)} judgments, already spent ${spent['usd']:.4f}, cap ${float(cap):.2f}", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, jobs))
    print(f"finished, total spent ${spent['usd']:.4f}, skipped by the cap {spent['skipped']}", flush=True)


if __name__ == "__main__":
    main()
