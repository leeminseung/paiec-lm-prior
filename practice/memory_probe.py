"""Memory probes for the report's question "where does the forecaster's lead come from" (offline analysis, public
training benchmarks only; the submitted predictor never names a benchmark).

score: the score-memory probe. For a pair of the calibration exam, a user-style request for the subject's success
  rate on the benchmark, with the subject's details and three of the pair's items (first three targets by key, each
  cut to 1,200 characters with whitespace collapsed, as in know_benchmark.py). Condition "anon" gives no benchmark
  name and also asks which benchmark it is; condition "named" adds the benchmark's name. The reply ends with a line
  giving the rate, its source (remembered, partly remembered, not remembered) and, for "anon", the benchmark's name.
rewrite: three same-meaning variants of an item excerpt, for the quiz (a model not under test writes them):
  "sentence" rewords whole sentences; "word" swaps a few words for near-synonyms and keeps everything else,
  because whole-sentence rewordings let a model spot the original by its rougher style (pilot 11/1); "swap" picks a
  few spans and gives each version a different word at every span, because with "word" the original was the one
  version the other three each differed from (pilot 11/1).
spans: a model not under test picks the cloze span of each item (checked to occur exactly once in the excerpt).
cloze: one span of an item excerpt that the context does not give away (a wrong option, a specific number or name) is
  blanked out; the reply ends with the missing text and whether it was remembered or guessed. Scored by exact match
  after lowercasing and dropping punctuation.
quiz: the original-wording quiz (failed in the 11/1 pilot: model-made variants are told apart by style or structure). The excerpt and its three rewordings in shuffled order; the reply ends with the
  letter of the original and whether it was remembered or guessed.

Backends: claude -p through the subscription (clean environment, one-line system prompt), or OpenRouter with a pinned
endpoint (key file never printed). Every call is one line in the output JSONL; reruns skip finished jobs.

Usage:
  python -I memory_probe.py score   <out.jsonl> <jobs.txt> claude <model> <effort> <workers> <empty_dir>
  python -I memory_probe.py score   <out.jsonl> <jobs.txt> openrouter <model> <endpoint_tag> <effort> <workers>
  python -I memory_probe.py rewrite <out.jsonl> <items.txt> <model> <effort> <workers> <empty_dir> [sentence|word|swap]
  python -I memory_probe.py spans   <out.jsonl> <items.txt> <model> <effort> <workers> <empty_dir>
  python -I memory_probe.py cloze   <out.jsonl> <cloze_items.jsonl> claude <model> <effort> <workers> <empty_dir>
  python -I memory_probe.py cloze   <out.jsonl> <cloze_items.jsonl> openrouter <model> <endpoint_tag> <effort> <workers>
  python -I memory_probe.py quiz    <out.jsonl> <rewrites.jsonl> claude <model> <effort> <workers> <empty_dir>
  python -I memory_probe.py quiz    <out.jsonl> <rewrites.jsonl> openrouter <model> <endpoint_tag> <effort> <workers>
jobs.txt: one "<pair index> <anon|named>" per line. items.txt: one "<pair index> <item key>" per line.
"""
import json
import os
import random
import re
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

ROOT = "."
EXAM = os.environ.get("MP_EXAM", f"{ROOT}/exams/mdb-judge-1027")  # MP_EXAM=<exam dir> for another exam
CATALOG = f"{ROOT}/agent_box_02/data/catalog/benchmarks_all.parquet"
EXCERPT, N_ITEMS = 1200, 3
SYSTEM = "You are a helpful assistant."
# Catalog names that carry a description of the data import rather than the benchmark's own name.
NAME_FIX = {
    "EgoSchema (released LLoVi configurations)": "EgoSchema",
    "EngineMT-QA (unambiguous single-choice release subset)": "EngineMT-QA",
    "FELM (released ChatGPT segment annotations)": "FELM",
    "FaithCoT-Bench (released FINE-CoT generator measurements)": "FaithCoT-Bench",
    "Interchangeable Token Embeddings: released LTL alpha-renaming runs": "Interchangeable Token Embeddings",
    "LLMs-Lean historical evaluations of miniF2F and miniCTX": "miniF2F and miniCTX",
    "MJ-Bench released judge observations": "MJ-Bench",
    "MMBench V1.1 recorded answers": "MMBench V1.1",
    "MasakhaNER 1.0 and 2.0: released NER predictions": "MasakhaNER 1.0 and 2.0",
}

SCORE_ANON_INTRO = ("I'm trying to find out how well a particular AI system did on a public AI benchmark, and it matters "
                    "to me to get the actual number right. I only have three of the benchmark's questions here, plus the "
                    "system's details.")
SCORE_NAMED_INTRO = ("I'm trying to find out how well a particular AI system did on a public AI benchmark, and it matters "
                     "to me to get the actual number right. The benchmark is {name}. Here are three of its questions, "
                     "plus the system's details.")
SCORE_ASK = ("Could you try hard to recall what share of this benchmark's questions this system got right, under the "
             "benchmark's own grading? A result you remember from a paper, a leaderboard or a model report is exactly "
             "what I'm hoping for, so please search your memory before working anything out.")
SCORE_ASK_NAME = " If you recognize which benchmark these questions come from, please tell me its name as well."
SCORE_TAIL = ("If you really can't recall a result, please still give me your best estimate, mark it as not remembered, "
              "and tell me briefly what you based it on, so I know how far to rely on it. If you do remember it, please "
              "tell me where it's from.\n\n"
              "Please end with one line in this format so I can put it in my table:\n{line}\n\n"
              "remembered: you recall a result for this system on this benchmark.\n"
              "partly remembered: you recall results on this benchmark for other systems, or for this system on a "
              "closely related benchmark, and adjusted from them.\n"
              "not remembered: you worked it out from the questions and the system.")
LINE_ANON = "success rate: <0 to 1> | source: remembered, partly remembered, or not remembered | benchmark: <name, or unknown>"
LINE_NAMED = "success rate: <0 to 1> | source: remembered, partly remembered, or not remembered"

REWRITE_SYSTEM = ("You reword texts for a study. Keep the meaning exactly the same: every fact, number, name, symbol, "
                  "option, formula, code line and data value stays as it is. Change only the wording and sentence "
                  "structure of the natural-language parts, keep about the same length, and keep the same language as "
                  "the text. Write three rewordings that differ from the text and from each other. If the text has no "
                  "natural-language part that can be reworded without changing its meaning, return three empty strings.")
REWRITE_WORD_SYSTEM = ("You make word-level variants of texts for a study. Write three variants of the text. In each, "
                       "replace a few words (about one in every fifteen, at least two and at most eight) with near-synonyms "
                       "that fit equally well, and choose different words to replace in each variant. Keep everything else "
                       "exactly as it is, character for character: sentence structure, word order, punctuation, spacing, "
                       "tone, typos and rough or unidiomatic phrasing. Do not polish, correct or smooth anything. Never change "
                       "numbers, names, symbols, option letters, formulas, code or data values, and keep the same language. "
                       "If the text has no words that can be swapped without changing its meaning, return three empty strings.")
REWRITE_SWAP_SYSTEM = ("You prepare word swaps for a study. Pick three to six words or short phrases in the text (about one "
                       "in every twenty-five words) that can be replaced without changing the meaning. Each must appear exactly "
                       "once in the text, copied character for character, and they must not overlap. For each, give three "
                       "different near-synonyms that fit the sentence equally well and keep its tone, including any rough or "
                       "unidiomatic phrasing around it. Never pick numbers, names, symbols, option letters, formulas, code or "
                       "data values, and keep the same language. If nothing can be swapped without changing the meaning, "
                       "return an empty list.")
SWAP_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["swaps"],
               "properties": {"swaps": {"type": "array", "maxItems": 6, "items": {
                   "type": "object", "additionalProperties": False, "required": ["original", "alternatives"],
                   "properties": {"original": {"type": "string"},
                                  "alternatives": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"}}}}}}}


def apply_swaps(text, swaps):
    """Three variants that each replace every chosen span, with a different alternative per variant, so every version
    (the original included) differs from every other at every chosen span and none is the common source of the rest.
    Spans that are missing, repeated, overlapping or without three distinct new alternatives are dropped."""
    spans = []
    for sw in swaps or []:
        o, alts = sw.get("original", ""), [a.strip() for a in sw.get("alternatives", [])]
        if not o or text.count(o) != 1 or len(alts) != 3 or len({o, *alts}) != 4 or any(not a for a in alts):
            continue
        i = text.index(o)
        if any(i < j + len(p) and j < i + len(o) for j, p, _ in spans):
            continue
        spans.append((i, o, alts))
    if len(spans) < 2:
        return None, len(spans)
    spans.sort()
    out = []
    for v in range(3):
        t, last = [], 0
        for i, o, alts in spans:
            t.append(text[last:i]); t.append(alts[v]); last = i + len(o)
        out.append("".join(t) + text[last:])
    return out, len(spans)


REWRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rewordings"],
                  "properties": {"rewordings": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"}}}}
QUIZ = ("I have four versions of one question. One of them is the original wording, and the other three are rewordings "
        "of it with the same meaning. I'd really like to know which one is the original. If you've seen this question "
        "before, for example in a public AI benchmark, your memory of its exact wording is the best guide, so please "
        "think back carefully. If you can't tell, your best guess is fine, but please say it's a guess.\n\n"
        "{versions}\n\nPlease end with one line in this format:\noriginal: <A, B, C or D> | source: remembered or guessed")


CLOZE = ("I'm looking at a question from a public AI benchmark, and one part of it is blanked out below, marked [MISSING]. "
         "Do you know what the missing part says? If you've seen this question before, your memory of the exact original "
         "wording is what I need, so please think back carefully. If you can't remember it, your best guess is fine, but "
         "please say it's a guess.\n\n{text}\n\nPlease end with one line in this format:\n"
         "missing: <the exact missing text> | source: remembered or guessed")


SPAN_SYSTEM = ("You help prepare a fill-in-the-blank memory check on public benchmark items. Given a text, choose one short "
               "span, one to twelve words, that a reader who has never seen this exact text could not work out from the "
               "rest of it. Good choices: an answer option that is clearly not the correct answer, a specific name, an "
               "identifier, a distinctive phrase, or one question in a list of examples. Avoid spans the rest of the text "
               "gives away: the correct answer, numbers that follow from other numbers, words repeated elsewhere, and "
               "common or generic values. Prefer a span without digits. Copy the span exactly, character for character; it "
               "must appear exactly once in the text. If no span qualifies, return an empty span.")
SPAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["span", "kind"],
               "properties": {"span": {"type": "string"},
                              "kind": {"type": "string", "enum": ["option", "name", "identifier", "phrase", "question",
                                                                  "number", "other", "none"]}}}


def norm(s):
    return " ".join(re.sub(r"[\s\"'.,;:!?()\[\]]+", " ", (s or "").lower()).split())


def parse_cloze(text):
    ln = last_line(text, "missing:")
    if not ln:
        return None
    m = re.search(r"missing:\s*(.*?)\s*(\|\s*source:\s*(.*))?$", ln, re.I)
    if not m:
        return None
    src = (m.group(3) or "").lower()
    return {"fill": m.group(1).strip().strip("*`").strip(), "source": "remembered" if "remember" in src else
            "guessed" if "guess" in src else None}


def excerpt(text):
    return " ".join((text or "").split())[:EXCERPT]


def subject_lines(s):
    rows = [("Model", s.get("normalized_name")), ("Provider", s.get("provider")), ("Released", s.get("release_date")),
            ("Run with", ", ".join(x for x in (s.get("harness"), s.get("harness_version") and f"version {s['harness_version']}",
                                                s.get("reasoning_effort") and f"reasoning effort {s['reasoning_effort']}") if x)),
            ("Other details", s.get("subject_features_extra"))]
    return "\n".join(f"- {k}: {v}" for k, v in rows if v)


def benchmark_name(folder):
    import pandas as pd
    name = str(pd.read_parquet(CATALOG).set_index("folder").loc[folder, "name"])
    return NAME_FIX.get(name, name)


def score_message(pair, cond, name=None):
    items = sorted(pair["targets"], key=lambda t: t["key"])[:N_ITEMS]
    qs = "\n".join(f"[{i + 1}] {excerpt(t['item']['item_content'])}" for i, t in enumerate(items))
    intro = SCORE_ANON_INTRO if cond == "anon" else SCORE_NAMED_INTRO.format(name=name)
    ask = SCORE_ASK + (SCORE_ASK_NAME if cond == "anon" else "")
    tail = SCORE_TAIL.format(line=LINE_ANON if cond == "anon" else LINE_NAMED)
    return (f"{intro}\n\nThe system:\n{subject_lines(pair['subject'])}\n\nThree questions from the benchmark:\n{qs}\n\n"
            f"{ask}\n\n{tail}"), [t["key"] for t in items]


def last_line(text, word):
    lines = [ln for ln in (text or "").splitlines() if word in ln.lower()]
    return lines[-1] if lines else None


def parse_score(text):
    ln = last_line(text, "success rate")
    if not ln:
        return None
    m = re.search(r"success rate\W*\s*([0-9]*\.?[0-9]+)\s*(%?)", ln, re.I)
    if not m:
        return None
    v = float(m.group(1)) / (100 if m.group(2) or float(m.group(1)) > 1 else 1)
    src = ln.lower().split("source", 1)[1] if "source" in ln.lower() else ""
    src = src.split("|")[0]
    source = ("partly remembered" if "partly" in src else "not remembered" if "not" in src
              else "remembered" if "remembered" in src else None)
    bm = re.search(r"benchmark\W*\s*(.+)$", ln, re.I)
    return {"rate": v, "source": source, "benchmark": bm.group(1).strip().strip("*") if bm else None}


def parse_quiz(text):
    ln = last_line(text, "original")
    if not ln:
        return None
    m = re.search(r"original\W*\s*\(?([ABCD])\b", ln, re.I)
    if not m:
        return None
    src = ln.lower().split("source", 1)[1] if "source" in ln.lower() else ""
    return {"choice": m.group(1).upper(), "source": "remembered" if "remember" in src else "guessed" if "guess" in src else None}


def ask_claude(text, model, effort, cwd, system=SYSTEM, schema=None, timeout=600):
    cmd = ["claude", "-p", text, "--model", model, "--effort", effort, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}), "--system-prompt", system,
           "--output-format", "json", "--no-session-persistence"] + (["--json-schema", json.dumps(schema)] if schema else [])
    start = time.time()
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"text": None, "error": "timeout", "seconds": round(time.time() - start, 1)}
    res = None
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            res = d
    if res is None:
        return {"text": None, "error": (out.stderr or out.stdout)[:300], "seconds": round(time.time() - start, 1)}
    return {"text": res.get("result"), "structured": res.get("structured_output"), "error": None,
            "seconds": round(time.time() - start, 1), "model_usage": res.get("modelUsage")}


def ask_openrouter(text, model, tag, key, effort, attempts=4, timeout=600):
    body = {"model": model, "max_tokens": 24000,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
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
            u = data.get("usage", {})
            return {"text": msg.get("content") or "", "error": None, "seconds": round(time.time() - start, 1),
                    "attempts": attempt + 1, "finish_reason": data["choices"][0].get("finish_reason"),
                    "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "cost_usd": u.get("cost", 0) or 0}
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, IndexError, ValueError) as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 ** attempt)
    return {"text": None, "error": err, "seconds": round(time.time() - start, 1), "attempts": attempts, "cost_usd": 0}


def run_jobs(out, jobs, call, workers):
    done = set()
    if os.path.exists(out):
        done = {r["job"] for r in map(json.loads, open(out)) if r.get("text") or r.get("structured")}
    todo = [j for j in jobs if j["job"] not in done]
    lock = threading.Lock()

    def one(j):
        r = call(j)
        with lock, open(out, "a") as f:
            f.write(json.dumps({**j, **r}, ensure_ascii=False) + "\n")

    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(one, todo))
    rows = [json.loads(x) for x in open(out)]
    print(f"done; cost so far ${sum(r.get('cost_usd', 0) or 0 for r in rows):.4f}", flush=True)


def backend_call(args):
    if args[0] == "claude":
        model, effort, workers, cwd = args[1:5]
        return model, workers, lambda text, extra: {"backend": "claude", "model": model, "effort": effort,
                                                    **ask_claude(text, model, effort, cwd)}
    model, tag, effort, workers = args[1:5]
    key = open(os.path.expanduser(KEY_FILE)).read().strip()
    eff = None if effort == "default" else effort
    return model, workers, lambda text, extra: {"backend": "openrouter", "model": model, "endpoint": tag, "effort": eff,
                                                **ask_openrouter(text, model, tag, key, eff)}


def main():
    mode, out, src = sys.argv[1:4]
    exam = load_exam(EXAM)
    if mode == "score":
        model, workers, call = backend_call(sys.argv[4:])
        jobs = []
        for ln in open(src):
            if ln.strip():
                pi, cond = ln.split()
                pair = exam["pairs"][int(pi)]
                name = benchmark_name(pair["benchmark_name"]) if cond == "named" else None
                text, keys = score_message(pair, cond, name)
                jobs.append({"job": f"{model}|{pi}|{cond}", "pair": int(pi), "cond": cond, "benchmark": pair["benchmark_name"],
                             "name_given": name, "keys": keys, "prompt": text})

        def go(j):
            r = call(j["prompt"], j)
            return {**r, "parsed": parse_score(r.get("text"))}
        run_jobs(out, jobs, go, workers)
    elif mode == "rewrite":
        model, effort, workers, cwd = sys.argv[4:8]
        style = (sys.argv[8:9] or ["sentence"])[0]
        system = {"word": REWRITE_WORD_SYSTEM, "swap": REWRITE_SWAP_SYSTEM}.get(style, REWRITE_SYSTEM)
        jobs = []
        for ln in open(src):
            if ln.strip():
                pi, key = ln.split()
                t = next(t for t in exam["pairs"][int(pi)]["targets"] if t["key"] == key)
                jobs.append({"job": f"{pi}|{key}", "pair": int(pi), "key": key,
                             "benchmark": exam["pairs"][int(pi)]["benchmark_name"], "original": excerpt(t["item"]["item_content"])})

        def go(j):
            if style == "swap":
                r = ask_claude(f"Text:\n\n{j['original']}", model, effort, cwd, system=system, schema=SWAP_SCHEMA)
                swaps = (r.get("structured") or {}).get("swaps")
                rw, n = apply_swaps(j["original"], swaps)
                return {"backend": "claude", "model": model, "effort": effort, "style": style, **r, "swaps": swaps,
                        "valid_swaps": n, "rewordings": rw}
            r = ask_claude(f"Text to reword:\n\n{j['original']}", model, effort, cwd, system=system,
                           schema=REWRITE_SCHEMA)
            rw = (r.get("structured") or {}).get("rewordings")
            return {"backend": "claude", "model": model, "effort": effort, "style": style, **r,
                    "rewordings": [" ".join(x.split()) for x in rw] if rw else None}
        run_jobs(out, jobs, go, workers)
    elif mode == "spans":
        model, effort, workers, cwd = sys.argv[4:8]
        jobs = []
        for ln in open(src):
            if ln.strip():
                pi, key = ln.split()
                t = next(t for t in exam["pairs"][int(pi)]["targets"] if t["key"] == key)
                jobs.append({"job": f"{pi}|{key}", "pair": int(pi), "key": key,
                             "benchmark": exam["pairs"][int(pi)]["benchmark_name"], "text": excerpt(t["item"]["item_content"])})

        def go(j):
            r = ask_claude(f"Text:\n\n{j['text']}", model, effort, cwd, system=SPAN_SYSTEM, schema=SPAN_SCHEMA)
            so = r.get("structured") or {}
            sp = (so.get("span") or "").strip()
            ok = len(sp) >= 3 and j["text"].count(sp) == 1
            return {"backend": "claude", "model": model, "effort": effort, **r, "span": sp if ok else None,
                    "proposed": so.get("span"), "kind": so.get("kind")}
        run_jobs(out, jobs, go, workers)
    elif mode == "cloze":
        model, workers, call = backend_call(sys.argv[4:])
        jobs = []
        for r in map(json.loads, open(src)):
            if not r.get("span"):
                continue
            t = next(t for t in exam["pairs"][r["pair"]]["targets"] if t["key"] == r["key"])
            text = excerpt(t["item"]["item_content"])
            if text.count(r["span"]) != 1:
                continue
            jobs.append({"job": f"{model}|{r['pair']}|{r['key']}", "pair": r["pair"], "key": r["key"],
                         "benchmark": exam["pairs"][r["pair"]]["benchmark_name"], "span": r["span"],
                         "prompt": CLOZE.format(text=text.replace(r["span"], "[MISSING]"))})

        def go(j):
            r = call(j["prompt"], j)
            p = parse_cloze(r.get("text"))
            if p:
                p["exact"] = norm(p["fill"]) == norm(j["span"])
            return {**r, "parsed": p}
        run_jobs(out, jobs, go, workers)
    elif mode == "quiz":
        model, workers, call = backend_call(sys.argv[4:])
        jobs = []
        for r in map(json.loads, open(src)):
            rw = r.get("rewordings")
            if not rw or any(not x or x == r["original"] for x in rw):
                continue
            # The rewriter writes \uXXXX escapes as the characters and drops zero-width characters, so every version is
            # shown with escapes decoded and zero-width characters removed.
            unesc = lambda s: re.sub(r"[\u200b\u200c\u200d\ufeff]", "",
                                     re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s))
            opts = [unesc(x) for x in [r["original"]] + rw]
            order = list(range(4))
            random.Random(f"{r['pair']}|{r['key']}").shuffle(order)
            letters = "ABCD"
            versions = "\n\n".join(f"{letters[i]}: {opts[o]}" for i, o in enumerate(order))
            jobs.append({"job": f"{model}|{r['pair']}|{r['key']}", "pair": r["pair"], "key": r["key"], "benchmark": r["benchmark"],
                         "answer": letters[order.index(0)], "prompt": QUIZ.format(versions=versions)})

        def go(j):
            r = call(j["prompt"], j)
            return {**r, "parsed": parse_quiz(r.get("text"))}
        run_jobs(out, jobs, go, workers)


if __name__ == "__main__":
    main()
