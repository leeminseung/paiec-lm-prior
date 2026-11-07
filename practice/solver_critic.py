"""Weak solvers try an item; a stronger critic then estimates how likely the subject is to succeed on it.

solve():  one attempt by a small model (Claude Haiku 5.5 by default) at the item itself. It sees only the item, never
          the subject. It returns a short final answer, or a plan when the task needs tools, files, a repository, a
          website or a long document, plus a short explanation; with thinking on it also returns the API's summary
          of its thinking and the tokens it used.
critic(): one judgment by a stronger model (Claude Sonnet 5.5, thinking off) that sees the subject, the item and the
          attempts, and returns the probability that the subject succeeds, how many attempts share the most common
          conclusion, and its probability that that conclusion is correct.
agreement(): counts in code how many short answers coincide (no judgment of meaning).

The model being predicted is never used as solver or critic (same_model). Revealed results are not sent; they enter
afterwards through the predictor's counting formula. Standard library only, so the file can go into a submission.
"""
import json
import random
import re
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ITEM_CHARS = 12000
MAX_ANSWER_CHARS = 1500
MAX_EXPLANATION_CHARS = 600
MAX_SUMMARY_CHARS = 2000
SHORT_ANSWER_CHARS = 60

SOLVER_SYSTEM = """You are solving one task from an AI benchmark. Work on it as well as you can.

Answer with JSON:
- "kind": "answer" if you produced the task's final answer, or "plan" if the task needs tools, files, a code repository, a website, or a long document that you cannot produce here.
- "final_answer": for "answer", the final answer exactly in the form the task asks for (a number, a label, a short phrase, or a concise final response). For "plan", the plan you would follow, in at most eight short steps.
- "explanation": at most three sentences on why this is the answer or plan and which part you are least sure about."""

SOLVER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["kind", "final_answer", "explanation"],
    "properties": {"kind": {"type": "string", "enum": ["answer", "plan"]},
                   "final_answer": {"type": "string"}, "explanation": {"type": "string"}}}

CRITIC_SYSTEM = """You estimate the probability that a given AI system (the subject) answers a given benchmark item correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. The item gives the full task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. Empty values mean unknown.

You also get several independent attempts at the same item by a smaller, faster AI model that is NOT the subject. Each attempt gives a final answer (or, for a task that needs tools or a long output, a plan) and a short explanation{summary_note}. Treat the attempts as evidence about the item, not as the subject's work: whether they agree, whether the most common answer looks correct, where they get stuck, and how much work the item seems to need. A capable subject may solve items the smaller model fails, and graders can be strict.

Your probability is scored by Brier score, the squared error against the 0/1 outcome, so give a calibrated probability.

Answer with JSON: {{"probability": <number from 0 to 1>, "same_conclusion": <how many attempts reach the same conclusion as the most common one>, "majority_looks_right": <your probability from 0 to 1 that the most common conclusion is correct>}}."""

SUMMARY_NOTE = ", plus a summary of the smaller model's private reasoning and the number of output tokens it used"

# Version 2 (10/22): the round-2 judgment prompt kept word for word (including "Think about how capable this system
# is ..."), the attempts added as evidence about the item only, and the attempts judged before the probability is written.
CRITIC_SYSTEM_V2 = """You estimate the probability that a given AI system (the subject) answers a given benchmark item correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. The item gives the full task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. Empty values mean unknown.

You also get several independent attempts at the same item by a smaller, faster AI model that is NOT the subject. Each attempt gives a final answer (or, for a task that needs tools or a long output, a plan) and a short explanation{summary_note}. Use them only as evidence about this item (whether they agree, whether the most common answer looks correct, where they get stuck), not as evidence about the subject, which may be much stronger or weaker than the smaller model.

First judge the attempts. Then think about how capable this system is under these settings, how hard this particular item is, and how strictly tasks like this are usually graded. Your probability is scored by Brier score, the squared error between your probability and the 0/1 outcome, so give a calibrated probability.

Answer with JSON in this order: {{"same_conclusion": <how many attempts reach the same conclusion as the most common one>, "majority_looks_right": <your probability from 0 to 1 that the most common conclusion is correct>, "probability": <number from 0 to 1>}}."""

CRITIC_SCHEMA_V2 = {
    "type": "object", "additionalProperties": False,
    "required": ["same_conclusion", "majority_looks_right", "probability"],
    "properties": {"same_conclusion": {"type": "integer"}, "majority_looks_right": {"type": "number"},
                   "probability": {"type": "number"}}}  # judgment fields first, probability last

CRITIC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["probability", "same_conclusion", "majority_looks_right"],
    "properties": {"probability": {"type": "number"}, "same_conclusion": {"type": "integer"},
                   "majority_looks_right": {"type": "number"}}}  # the API rejects minimum/maximum; ranges are checked below


def same_model(subject, model_id):
    """True when the subject's model is the model we would call (for example Claude Sonnet 5.5 and claude-sonnet-5-5)."""
    name = re.sub(r"[^a-z0-9.]+", " ", (subject.get("normalized_name") or "").lower())
    m = re.match(r"claude-(haiku|sonnet|opus)-(\d+)-(\d+)", model_id)
    if not m:
        return False
    family, version = m.group(1), f"{m.group(2)}.{m.group(3)}"
    return family in name.split() and version in name.split()


def thinking_off(model):
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}  # Sonnet 5.5 rejects "disabled"; effort must be high or below
    if model.startswith("claude-haiku-5-5"):
        return {"type": "disabled"}  # accepted at effort high or below
    raise ValueError(f"no thinking-off setting known for {model}")


def truncate(text, limit):
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit} characters omitted ...]\n" + text[-half:]


def post(body, api_key, timeout, attempts):
    """POST to the Messages API with retries. Returns (response dict or None, error or None, attempts used, seconds)."""
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
    start, error = time.time(), None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read()), None, attempt, round(time.time() - start, 1)
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in RETRY_STATUS or attempt == attempts:
                break
            wait = float(e.headers.get("retry-after") or 0) or min(30.0, 2 ** attempt) * (0.5 + random.random())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
            if attempt == attempts:
                break
            wait = min(30.0, 2 ** attempt) * (0.5 + random.random())
        time.sleep(wait)
    return None, error, attempt, round(time.time() - start, 1)


def parse_json_text(data):
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    try:
        return json.loads(texts[-1])
    except (IndexError, TypeError, ValueError):
        return None


def solver_message(item):
    fields = {"task": truncate(item.get("item_content", ""), MAX_ITEM_CHARS)}
    if item.get("item_features"):
        fields["task_features"] = item["item_features"]
    if item.get("interactors"):
        fields["interactors"] = item["interactors"]
    return json.dumps(fields, ensure_ascii=False, indent=1)


def solve(item, api_key, model="claude-haiku-5-5", thinking="off", effort="medium", max_tokens=None, timeout=None,
          attempts=4):
    """One attempt. thinking: "off" or "on" (adaptive thinking with summarized display)."""
    if max_tokens is None:
        max_tokens = 2000 if thinking == "off" else 16000
    if timeout is None:
        timeout = 90 if thinking == "off" else 240
    body = {"model": model, "max_tokens": max_tokens, "system": SOLVER_SYSTEM,
            "messages": [{"role": "user", "content": solver_message(item)}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SOLVER_SCHEMA}}}
    body["thinking"] = thinking_off(model) if thinking == "off" else {"type": "adaptive", "display": "summarized"}
    data, error, used, seconds = post(body, api_key, timeout, attempts)
    out = {"kind": None, "final_answer": None, "explanation": None, "summary": "", "usage": {}, "stop_reason": None,
           "error": error, "seconds": seconds, "attempts": used}
    if data is None:
        return out
    out["usage"] = data.get("usage", {})
    out["stop_reason"] = data.get("stop_reason")
    out["summary"] = "\n".join(b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking").strip()
    parsed = parse_json_text(data)
    if isinstance(parsed, dict) and isinstance(parsed.get("final_answer"), str):
        out.update(kind=parsed.get("kind"), final_answer=parsed["final_answer"], explanation=parsed.get("explanation", ""))
    else:
        out["error"] = f"unparsed answer, stop_reason {out['stop_reason']}"
    return out


def critic_message(subject, item, attempts, with_summaries):
    shown = []
    for a in attempts:
        if a.get("final_answer") is None:
            entry = {"outcome": f"no answer ({a.get('stop_reason') or a.get('error') or 'failed'})"}
        else:
            entry = {"kind": a.get("kind"), "final_answer": truncate(a["final_answer"], MAX_ANSWER_CHARS),
                     "explanation": truncate(a.get("explanation") or "", MAX_EXPLANATION_CHARS)}
        if with_summaries:
            entry["reasoning_summary"] = truncate(a.get("summary") or "", MAX_SUMMARY_CHARS)
            entry["output_tokens_used"] = (a.get("usage") or {}).get("output_tokens")
        shown.append(entry)
    fields = {"subject": subject,
              "item": {"item_content": truncate(item.get("item_content", ""), MAX_ITEM_CHARS),
                       "item_features": item.get("item_features", ""), "interactors": item.get("interactors", ""),
                       "benchmark_id": item.get("benchmark_id", "")},
              "attempts_by_smaller_model": shown}
    return json.dumps(fields, ensure_ascii=False, indent=1)


def critic(subject, item, attempts, api_key, model="claude-sonnet-5-5", with_summaries=False, effort="medium",
           max_tokens=512, timeout=90, tries=4, version="v1"):
    template, schema = (CRITIC_SYSTEM, CRITIC_SCHEMA) if version == "v1" else (CRITIC_SYSTEM_V2, CRITIC_SCHEMA_V2)
    system = template.format(summary_note=SUMMARY_NOTE if with_summaries else "")
    body = {"model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": critic_message(subject, item, attempts, with_summaries)}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            "thinking": thinking_off(model)}
    data, error, used, seconds = post(body, api_key, timeout, tries)
    out = {"probability": None, "same_conclusion": None, "majority_looks_right": None, "usage": {}, "stop_reason": None,
           "error": error, "seconds": seconds, "attempts": used}
    if data is None:
        return out
    out["usage"] = data.get("usage", {})
    out["stop_reason"] = data.get("stop_reason")
    parsed = parse_json_text(data)
    try:
        p, q = float(parsed["probability"]), float(parsed["majority_looks_right"])
        if 0.0 <= p <= 1.0 and 0.0 <= q <= 1.0:
            out.update(probability=p, majority_looks_right=q, same_conclusion=int(parsed["same_conclusion"]))
        else:
            out["error"] = "value out of range"
    except (TypeError, KeyError, ValueError):
        out["error"] = f"unparsed answer, stop_reason {out['stop_reason']}"
    return out


def normalize_answer(text):
    """A comparable form of a short answer, or None when the answer is long or not an answer."""
    if text is None or len(text) > SHORT_ANSWER_CHARS:
        return None
    s = text.strip().lower()
    boxed = re.search(r"\\boxed\{([^{}]*)\}", s)
    if boxed:
        s = boxed.group(1)
    s = re.sub(r"\s+", " ", s).strip(" .$")
    try:
        return f"{float(s.replace(',', '')):.4g}"
    except ValueError:
        return s


def agreement(attempts):
    """Share of comparable short answers equal to the most common one, and the number of distinct ones."""
    keys = [normalize_answer(a.get("final_answer")) for a in attempts if a.get("kind") == "answer"]
    keys = [k for k in keys if k]
    if len(keys) < 2:
        return {"n_comparable": len(keys), "top_share": None, "n_distinct": None}
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return {"n_comparable": len(keys), "top_share": max(counts.values()) / len(keys), "n_distinct": len(counts)}
