"""What does a frontier judge reason about? Budget-0 judgments with the API's thinking summary, on a stratified sample of
the measurement-db judge set.

Sample: pairs whose benchmark has a release date, split into older (before 2025) and newer (2025 on); within each, pairs
where the Opus low judgment's pair mean prediction was close to the pair success rate (below that group's median gap)
and pairs where it was far; 10 pairs per cell with distinct benchmarks, 3 of each pair's judged items (seed 20261028).
Call: the same system prompt, message and answer schema as llm_prior.py, with adaptive thinking shown as a summary
(display "summarized") at the given effort. Each record keeps the sample cell, the low judgment, the outcome labels and
the summary text. Resumable, with a dollar cap (list prices, no caching).

Usage: python -I thinking_judge.py <judge_set_dir> <low_run_dir> <out_dir> <key: keychain:<service> or a file>
                                   <model> <effort> <workers> <cap_usd>
THINKING_ALL=1 asks every target of <judge_set_dir> instead of the stratified sample (for the 847-item subset
exams/mdb-judge-1027-onepair, whose pairs point to the full set through "source_pair"; <low_run_dir> is then the low
run on the full set).
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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import api_key, load_exam
from exemplar_shift import load
from llm_prior import API_URL, RETRY_STATUS, SCHEMA, SYSTEM, user_message

CATALOG = "./agent_box_02/data/catalog/benchmarks_all.parquet"
PRICES = {"claude-opus-5-5": (4.0, 20.0)}  # USD per million tokens, input and output (claude-api reference, 2026-10-20)
SEED, PER_CELL, PER_PAIR = 20261028, 10, 3


def sample(exam, low):
    dates = pd.read_parquet(CATALOG)[["folder", "release_date"]]
    date = dict(zip(dates.folder, pd.to_datetime(dates.release_date, errors="coerce")))
    rows = []
    for pi, p in enumerate(exam["pairs"]):
        d = date.get(p["benchmark_name"])
        judged = [t for t in p["targets"] if (pi, t["key"]) in low]
        if d is None or pd.isna(d) or len(judged) < PER_PAIR:
            continue
        gap = abs(np.mean([low[(pi, t["key"])] for t in judged]) - p["pair_success_rate"])
        rows.append({"pair": pi, "bench": p["benchmark_name"], "date": d, "gap": gap,
                     "age": "older" if d < pd.Timestamp("2025-01-01") else "newer"})
    df = pd.DataFrame(rows)
    rng = random.Random(SEED)
    chosen = []
    for age, g in df.groupby("age"):
        med = g.gap.median()
        for fit, cell in (("close", g[g.gap < med]), ("far", g[g.gap >= med])):
            idx = list(cell.index); rng.shuffle(idx)
            seen = set()
            for i in idx:
                r = cell.loc[i]
                if r.bench in seen:
                    continue
                seen.add(r.bench)
                p = exam["pairs"][r.pair]
                judged = sorted([t for t in p["targets"] if (r.pair, t["key"]) in low], key=lambda t: t["key"])
                for t in rng.sample(judged, PER_PAIR):
                    chosen.append({"pair": int(r.pair), "key": t["key"], "benchmark": r.bench, "release_date": str(r.date.date()),
                                   "age": age, "fit": fit, "low_gap": round(float(r.gap), 4), "low_probability": low[(r.pair, t["key"])],
                                   "pair_success_rate": p["pair_success_rate"], "labels": t["labels"]})
                if len(seen) == PER_CELL:
                    break
    return chosen


def every_target(exam, low):
    dates = pd.read_parquet(CATALOG)[["folder", "release_date"]]
    date = dict(zip(dates.folder, pd.to_datetime(dates.release_date, errors="coerce")))
    out = []
    for pi, p in enumerate(exam["pairs"]):
        src = p.get("source_pair", pi)
        d = date.get(p["benchmark_name"])
        has_date = d is not None and not pd.isna(d)
        for t in p["targets"]:
            out.append({"pair": pi, "key": t["key"], "benchmark": p["benchmark_name"],
                        "release_date": str(d.date()) if has_date else "",
                        "age": ("older" if d < pd.Timestamp("2025-01-01") else "newer") if has_date else "",
                        "fit": "", "low_probability": low.get((src, t["key"])), "pair_success_rate": p["pair_success_rate"],
                        "labels": t["labels"]})
    return out


def ask(subject, item, key, model, effort, attempts=5, timeout=240):
    body = json.dumps({"model": model, "max_tokens": 16000, "system": SYSTEM,
                       "messages": [{"role": "user", "content": user_message(subject, item)}],
                       "thinking": {"type": "adaptive", "display": "summarized"},
                       "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}).encode()
    headers = {"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}
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
                return {"probability": None, "summary": "", "usage": {}, "error": error, "seconds": round(time.time() - start, 1)}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
            if attempt == attempts:
                return {"probability": None, "summary": "", "usage": {}, "error": error, "seconds": round(time.time() - start, 1)}
        time.sleep(min(30.0, 2 ** attempt) * (0.5 + random.random()))
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    summary = "\n".join(b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking").strip()
    p = None
    try:
        p = float(json.loads(texts[-1])["probability"])
        p = p if 0 <= p <= 1 else None
    except (IndexError, KeyError, TypeError, ValueError):
        error = f"unparsed, stop_reason {data.get('stop_reason')}"
    return {"probability": p, "summary": summary, "usage": data.get("usage", {}), "stop_reason": data.get("stop_reason"),
            "error": None if p is not None else error, "seconds": round(time.time() - start, 1)}


def main():
    set_dir, low_dir, out_dir, key_src, model, effort, workers, cap = sys.argv[1:9]
    exam = load_exam(set_dir)
    jobs = every_target(exam, load(low_dir)) if os.environ.get("THINKING_ALL") == "1" else sample(exam, load(low_dir))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "judgments.jsonl")
    old = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done = {(r["pair"], r["key"]) for r in old if r.get("probability") is not None}
    spent = {"usd": sum(r.get("cost_usd", 0) for r in old)}
    todo = [j for j in jobs if (j["pair"], j["key"]) not in done]
    key = api_key(key_src)
    p_in, p_out = PRICES[model]
    lock = threading.Lock()
    print(f"sample {len(jobs)} items over {len({(j['age'], j['fit'], j['pair']) for j in jobs})} pairs; {len(todo)} to ask, "
          f"spent ${spent['usd']:.3f}, cap ${float(cap):.2f}", flush=True)

    def run(j):
        with lock:
            if spent["usd"] >= float(cap):
                return
        pair = exam["pairs"][j["pair"]]
        item = next(t["item"] for t in pair["targets"] if t["key"] == j["key"])
        r = ask(pair["subject"], item, key, model, effort)
        u = r["usage"]
        cost = (u.get("input_tokens", 0) * p_in + u.get("output_tokens", 0) * p_out) / 1e6
        rec = {**j, "subject": pair["subject"], "model": model, "effort": effort, **r, "cost_usd": round(cost, 6)}
        with lock:
            spent["usd"] += cost
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, todo))
    print(f"finished, total spent ${spent['usd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
