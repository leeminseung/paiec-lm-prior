"""Ask a Claude model once, without tools, how likely an AI system is to succeed on one item (a budget-0 judgment).

Standard library only (urllib): the competition's scoring server installs no extra packages by default, and the
organizers' own LLM baseline calls its API the same way. Revealed results are deliberately not sent; they enter
afterwards through the counting formula of the predictor, so this judgment is the same at every budget.

The same file is used for local measurement and inside the submission.
"""
import json
import random
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MAX_ITEM_CHARS = 12000
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probability"],
          "properties": {"probability": {"type": "number"}}}  # the API rejects minimum/maximum; the range is checked below

SYSTEM = """You estimate the probability that a given AI system (the subject) answers a given benchmark item correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. The item gives the full task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. Empty values mean unknown.

Think about how capable this system is under these settings, how hard this particular item is, and how strictly tasks like this are usually graded. Your answer is scored by Brier score, the squared error between your probability and the 0/1 outcome, so give a calibrated probability.

Answer with JSON: {"probability": <number from 0 to 1>}."""


def user_message(subject, item):
    content = item.get("item_content", "")
    if len(content) > MAX_ITEM_CHARS:
        half = MAX_ITEM_CHARS // 2
        content = content[:half] + f"\n[... {len(content) - MAX_ITEM_CHARS} characters omitted ...]\n" + content[-half:]
    fields = {"subject": subject, "item": {"item_content": content, "item_features": item.get("item_features", ""),
                                           "interactors": item.get("interactors", ""),
                                           "benchmark_id": item.get("benchmark_id", "")}}
    return json.dumps(fields, ensure_ascii=False, indent=1)


def thinking_off(model):
    """The documented way to turn thinking off differs by model (claude-api reference, 2026-10-20)."""
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}  # Sonnet 5.5 rejects "disabled"; effort must be high or below
    if model.startswith("claude-haiku-5-5"):
        return {"type": "disabled"}  # accepted at effort high or below
    raise ValueError(f"no thinking-off setting known for {model}")


def judge(subject, item, api_key, model, effort="medium", thinking="adaptive", max_tokens=8000, timeout=120, attempts=5):
    """Return {"probability": float or None, "usage": {...}, "stop_reason", "error", "seconds", "attempts"}.
    thinking: "adaptive" (the model's default, thinking on) or "off"."""
    request_body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_message(subject, item)}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    }
    if thinking == "off":
        request_body["thinking"] = thinking_off(model)
    body = json.dumps(request_body).encode()
    headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
    start, error = time.time(), None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            break
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in RETRY_STATUS or attempt == attempts:
                return {"probability": None, "usage": {}, "stop_reason": None, "error": error,
                        "seconds": round(time.time() - start, 1), "attempts": attempt}
            wait = float(e.headers.get("retry-after") or 0) or min(30.0, 2 ** attempt) * (0.5 + random.random())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
            if attempt == attempts:
                return {"probability": None, "usage": {}, "stop_reason": None, "error": error,
                        "seconds": round(time.time() - start, 1), "attempts": attempt}
            wait = min(30.0, 2 ** attempt) * (0.5 + random.random())
        time.sleep(wait)
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    probability = None
    try:
        value = float(json.loads(texts[-1])["probability"])
        if 0.0 <= value <= 1.0:
            probability = value
    except (IndexError, KeyError, TypeError, ValueError):
        error = f"unparsed answer, stop_reason {data.get('stop_reason')}"
    return {"probability": probability, "usage": data.get("usage", {}), "stop_reason": data.get("stop_reason"),
            "error": None if probability is not None else error, "seconds": round(time.time() - start, 1),
            "attempts": attempt}
