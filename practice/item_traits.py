"""Ask an LLM to describe properties of one benchmark item that set how often AI systems succeed on items like it
(answer format, grading, strictness, domain, language, unseen media, steps, a subject-free success guess).
Claude API, Sonnet 5.5 with thinking off, structured output. Resumable, with a dollar cap.

Items: measurement-db, 20 per benchmark from the embedding sample (runs/embed-20261027/meta.parquet, seed 1), with their
graded count and successes; both proxy evaluations, 20 targets per pair (seed 1). The model sees what predict() receives
as the item: item_content (cut in the middle like llm_prior.py), item_features and interactors.

Usage: python -I item_traits.py <run_dir> <key: keychain:<service> or a file> <cap_usd> <workers> [limit]
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import api_key, as_str, load_exam
from llm_prior import API_URL, MAX_ITEM_CHARS, RETRY_STATUS, thinking_off

ROOT = "./"
EMBED_META = ROOT + "runs/embed-20261027/meta.parquet"
EXAMS = {"proxy-10/20": ROOT + "exams/rehearsal-02", "proxy-10/26": ROOT + "exams/proxy-confirm-1026"}
MODEL, EFFORT, PER_BENCH, PER_PAIR = "claude-sonnet-5-5", "medium", 20, 20
PRICE_IN, PRICE_OUT = 2.0, 10.0  # USD per million tokens (claude-api reference, read 2026-10-20)

FORMATS = ["multiple_choice", "yes_no", "number", "short_text", "long_text", "code", "agent_task", "label"]
GRADING = ["exact_match", "numeric_tolerance", "tests", "rubric_or_llm_judge", "environment_check", "unknown"]
DOMAINS = ["math", "code", "medical", "legal", "science", "general_knowledge", "language", "safety", "web_or_app", "other"]
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["answer_format", "n_options", "grading", "all_parts_required", "domain", "english",
                       "needs_unseen_media", "steps", "p_strong_model"],
          "properties": {"answer_format": {"type": "string", "enum": FORMATS},
                         "n_options": {"type": "integer"},
                         "grading": {"type": "string", "enum": GRADING},
                         "all_parts_required": {"type": "boolean"},
                         "domain": {"type": "string", "enum": DOMAINS},
                         "english": {"type": "boolean"},
                         "needs_unseen_media": {"type": "boolean"},
                         "steps": {"type": "integer"},
                         "p_strong_model": {"type": "number"}}}

SYSTEM = """You read one benchmark item exactly as an AI system received it, and describe the properties that decide how often AI systems succeed on items like it. You do not see any outcome.

Fields:
- answer_format: multiple_choice (pick from listed options), yes_no, number, short_text (a word, name or short phrase), long_text (paragraphs, an essay, a free-form explanation), code (a function, script or patch), agent_task (several actions in an environment, website, app or tool), label (choose a category or grade that is not listed as options).
- n_options: the number of listed options for multiple_choice or yes_no; 0 otherwise.
- grading: how success is most likely decided: exact_match, numeric_tolerance, tests (code is run against tests), rubric_or_llm_judge, environment_check (the final state of an environment is checked), unknown.
- all_parts_required: true if success needs several separate parts or conditions to all be right.
- domain: the main subject area.
- english: true if the task is in English.
- needs_unseen_media: true if solving it needs an image, audio, video or file that is referred to but not included in the text.
- steps: reasoning or action steps a capable solver needs, from 1 (direct recall or reading) to 5 (long multi-step work).
- p_strong_model: your estimate of the probability that a strong current AI system succeeds on this item.

Answer with JSON matching the schema."""


def cut(text):
    if len(text) <= MAX_ITEM_CHARS:
        return text
    half = MAX_ITEM_CHARS // 2
    return text[:half] + f"\n[... {len(text) - MAX_ITEM_CHARS} characters omitted ...]\n" + text[-half:]


def message(item):
    return json.dumps({"item_content": cut(item["item_content"]), "item_features": item.get("item_features", ""),
                       "interactors": item.get("interactors", "")}, ensure_ascii=False, indent=1)


def ask(item, api_key, attempts=5, timeout=120):
    body = json.dumps({"model": MODEL, "max_tokens": 1000, "system": SYSTEM, "thinking": thinking_off(MODEL),
                       "messages": [{"role": "user", "content": message(item)}],
                       "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}}).encode()
    headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
    start, error = time.time(), None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(API_URL, data=body, headers=headers, method="POST"),
                                        timeout=timeout) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in RETRY_STATUS or attempt == attempts:
                return {"traits": None, "usage": {}, "error": error, "seconds": round(time.time() - start, 1)}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
            if attempt == attempts:
                return {"traits": None, "usage": {}, "error": error, "seconds": round(time.time() - start, 1)}
        time.sleep(min(30.0, 2 ** attempt) * (0.5 + random.random()))
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    try:
        traits = json.loads(texts[-1])
    except (IndexError, ValueError):
        traits, error = None, f"unparsed, stop_reason {data.get('stop_reason')}"
    return {"traits": traits, "usage": data.get("usage", {}), "stop_reason": data.get("stop_reason"),
            "error": None if traits else error, "seconds": round(time.time() - start, 1)}


def jobs():
    out = []
    meta = pd.read_parquet(EMBED_META)
    md = meta[meta.source == "measurement-db"]
    for b, g in md.groupby("benchmark"):
        g = g.sample(min(PER_BENCH, len(g)), random_state=1)
        items = pd.read_parquet(f"{ROOT}data/hf/{b}/formatted_tables/items.parquet",
                                columns=["item_id", "content", "item_features"]).set_index("item_id")
        for r in g.itertuples():
            row = items.loc[r.item_id]
            out.append({"source": "measurement-db", "benchmark": b, "item_id": r.item_id, "n": int(r.n), "s": int(r.s),
                        "item": {"item_content": as_str(row["content"]), "item_features": as_str(row["item_features"]),
                                 "interactors": ""}})
    for name, d in EXAMS.items():
        exam = load_exam(d)
        for pi, pair in enumerate(exam["pairs"]):
            for t in random.Random(f"1:{name}:{pi}").sample(pair["targets"], min(PER_PAIR, len(pair["targets"]))):
                out.append({"source": name, "benchmark": pair["benchmark_name"], "item_id": t["key"], "pair": pi,
                            "item": t["item"]})
    return out


def main():
    run_dir, key_file, cap, workers = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    api_key = api_key(key_file)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "traits.jsonl")
    old = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {(r["source"], r["benchmark"], r["item_id"], r.get("pair")) for r in old if r.get("usage")}
    todo = [j for j in jobs() if (j["source"], j["benchmark"], j["item_id"], j.get("pair")) not in done]
    if limit is not None:
        rng = random.Random(0); todo = rng.sample(todo, min(limit, len(todo)))
    spent = {"usd": sum(r.get("cost_usd", 0) for r in old), "skipped": 0}
    lock = threading.Lock()
    print(f"{len(todo)} calls to make, already spent ${spent['usd']:.3f}, cap ${cap:.2f}", flush=True)

    def run(j):
        with lock:
            if spent["usd"] >= cap:
                spent["skipped"] += 1
                return
        r = ask(j["item"], api_key)
        u = r["usage"]
        cost = (u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT
                + u.get("cache_read_input_tokens", 0) * PRICE_IN * 0.1 + u.get("cache_creation_input_tokens", 0) * PRICE_IN * 1.25) / 1e6
        rec = {k: v for k, v in j.items() if k != "item"} | {"model": MODEL, **r, "cost_usd": round(cost, 6)}
        with lock:
            spent["usd"] += cost
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))
    print(f"finished, total spent ${spent['usd']:.3f}, skipped by the cap {spent['skipped']}", flush=True)


if __name__ == "__main__":
    main()
