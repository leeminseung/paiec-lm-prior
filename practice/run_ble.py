"""Run the organizers' BLE predictor (paiec_baseline, ble/) on proxy-evaluation targets, logging cost per prediction.

BLE's own code is not modified. This runner imports it and wraps litellm.completion to (1) mark the system message
and the latest user message with Anthropic prompt-caching breakpoints, so each turn re-reads the earlier
conversation from cache instead of paying for it again (behavior unchanged), and (2) record every turn's token usage.
The prediction follows ble/model.py: build_question(subject, item, labeled) and run_agent with the competition
settings (5 turns, 240 s); labeled is the first b acquisitions of every pair, as the platform supplies it.

Environment: BLF_DATA_ROOT (leakage-safe corpus from build_ble_corpus.py), ANTHROPIC_API_KEY (never printed).
Optional: BLE_DROP_PARAMS=1 sets litellm.drop_params, which turns BLE's forced final submit (tool_choice naming
submit, unsupported by Opus 5.5) into tool_choice auto; the final turn offers only the submit tool, so behavior is
nearly the same. BLE_COST_CAP=<USD> stops starting new predictions once this run's spend reaches the cap (skipped
predictions are not recorded, so a later run resumes them).
Prices for the cost column (USD per million tokens): Opus 5.5 4 / 20, Haiku 5.5 0.10 / 0.50, Sonnet 5.5 2 / 10;
cache reads 10% and cache writes 125% of the input price.

Usage: python -I run_ble.py <exam_dir> <llm e.g. anthropic/claude-opus-5-5> <out_dir> <budgets e.g. 0,7,31>
                           <targets_per_pair> [pair indices e.g. 0,3,5] [workers]
"""
import copy
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

BLE = "./third_party/paiec_baseline/ble"
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), BLE, os.path.join(BLE, "src")]
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no cost-map download at import

import litellm  # noqa: E402

from common import labeled_at, load_exam  # noqa: E402

PRICES = {"opus": (4.0, 20.0), "haiku": (0.10, 0.50), "sonnet": (2.0, 10.0)}
_usage = threading.local()
_orig = litellm.completion


def _with_cache(messages):
    msgs = copy.deepcopy(messages)
    marks = [0] + [i for i, m in enumerate(msgs) if m["role"] == "user"][-1:]
    for i in sorted(set(marks)):
        m = msgs[i]
        if isinstance(m.get("content"), str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return msgs


def _wrapped(**kwargs):
    if str(kwargs.get("model", "")).startswith("anthropic/"):
        kwargs["messages"] = _with_cache(kwargs["messages"])
    resp = _orig(**kwargs)
    u = getattr(resp, "usage", None)
    rec = u.model_dump() if hasattr(u, "model_dump") else dict(u or {})
    getattr(_usage, "turns", []).append(rec)
    return resp


litellm.completion = _wrapped
if os.environ.get("BLE_DROP_PARAMS") == "1":
    litellm.drop_params = True

from agent.agent import run_agent  # noqa: E402
from agent.belief_state import MAX_PROBABILITY, MIN_PROBABILITY  # noqa: E402
from config.config import AgentConfig  # noqa: E402
from model import build_question  # noqa: E402  (ble/model.py; reads no config file here)


def cost(llm, turns):
    p_in, p_out = next(v for k, v in PRICES.items() if k in llm)
    total = 0.0
    for t in turns:
        read = t.get("cache_read_input_tokens") or (t.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        write = t.get("cache_creation_input_tokens") or 0
        plain = max((t.get("prompt_tokens") or 0) - read - write, 0)
        total += (plain * p_in + read * p_in * 0.1 + write * p_in * 1.25 + (t.get("completion_tokens") or 0) * p_out) / 1e6
    return total


def main():
    exam_dir, llm, out_dir, budgets, per_pair = sys.argv[1:6]
    pairs_sel = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 and sys.argv[6] else None
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    budgets = [int(b) for b in budgets.split(",")]
    exam = load_exam(exam_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.environ["BLF_OUT"] = os.path.join(out_dir, "ble_logs")
    out_path = os.path.join(out_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["budget"], r["pair"], r["key"]) for r in map(json.loads, open(out_path)) if r.get("probability") is not None}
    jobs = []
    for pi, p in enumerate(exam["pairs"]):
        if pairs_sel is not None and pi not in pairs_sel:
            continue
        keys = sorted(t["key"] for t in p["targets"])
        random.Random(f"ble:{exam_dir}:{pi}").shuffle(keys)
        chosen = set(keys[:int(per_pair)])
        for t in p["targets"]:
            if t["key"] in chosen:
                jobs += [(b, pi, t) for b in budgets if (b, pi, t["key"]) not in done]
    labeled = {b: labeled_at(exam, b) for b in budgets}
    config = AgentConfig(llm=llm, max_steps=5, question_timeout=240, reasoning_effort="medium")
    lock = threading.Lock()
    cap = float(os.environ.get("BLE_COST_CAP", "inf"))
    spent = [0.0]

    def run(job):
        b, pi, t = job
        with lock:
            if spent[0] >= cap:
                return
        _usage.turns = []
        start = time.time()
        rec = {"budget": b, "pair": pi, "key": t["key"], "llm": llm}
        try:
            res = run_agent(build_question(exam["pairs"][pi]["subject"], t["item"], labeled=labeled[b]), config)
            rec.update(probability=min(max(float(res["forecast"]), MIN_PROBABILITY), MAX_PROBABILITY),
                       n_steps=res["n_steps"], tools=[s["tool"] for s in res["tool_log"]], error=None)
        except Exception as e:  # BLE raises on API errors, malformed calls and deadline expiry
            rec.update(probability=None, error=f"{type(e).__name__}: {str(e)[:300]}")
        rec.update(seconds=round(time.time() - start, 1), turns=_usage.turns, cost_usd=round(cost(llm, _usage.turns), 5))
        with lock:
            spent[0] += rec["cost_usd"]
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"b{b} pair {pi} p={rec['probability']} steps={rec.get('n_steps')} {rec['seconds']}s ${rec['cost_usd']:.4f} "
                  f"{rec['error'] or ''}", flush=True)

    print(f"{len(jobs)} predictions with {llm}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    print(f"spent ${spent[0]:.2f}" + (f" (cap ${cap:.2f} reached, rest skipped)" if spent[0] >= cap else ""), flush=True)


if __name__ == "__main__":
    main()
