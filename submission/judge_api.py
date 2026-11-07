"""Batched budget-0 judgments through the Claude API: one request returns a success probability for each listed item.

The request holds the subject's fields and the items (each item's text middle-cut: revealed items to 4,000 characters,
the target to 12,000), and never any outcome. The system prompt and the block of revealed items carry prompt-cache
breakpoints, so requests for the same pair and budget, which differ only in the target, reuse the cached prefix.
Standard library only. The API key is passed in by the caller and never logged.
"""
import json
import random
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
REVEALED_CHARS, TARGET_CHARS = 4000, 12000

SYSTEM = """You estimate, for each of several benchmark items, the probability that a given AI system (the subject) answers that item correctly, without running the system.

The subject is described by its model name, provider, release date, access date, harness (the agent scaffold or evaluation runner that drove the model), harness version, reasoning effort and extra settings. Each item gives the task text the subject received, item features, interactors (other tools or agents involved) and an anonymous benchmark id. All items come from the same benchmark. Empty values mean unknown.

Think about how capable this system is under these settings, how hard each particular item is, and how strictly tasks like this are usually graded. Each probability is scored by Brier score, the squared error between the probability and the 0/1 outcome, so give calibrated probabilities. Judge every item on its own merits; the items are listed in no particular order of difficulty.

Answer with JSON: {"probabilities": [<one number from 0 to 1 per item, in the order the items are listed>]}."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probabilities"],
          "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}}}


def _cut(text, limit):
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit} characters omitted ...]\n" + text[-half:]


def _item(item, limit):
    return {"item_content": _cut(item.get("item_content", ""), limit), "item_features": item.get("item_features", ""),
            "interactors": item.get("interactors", ""), "benchmark_id": item.get("benchmark_id", "")}


def judge_items(subject, revealed, target, api_key, model, effort, deadline=None, timeout=60, attempts=2):
    """revealed: list of item dicts (outcomes are never passed); target: item dict or None.
    deadline: time.monotonic() value after which no request is started or waited for.
    Returns a list of probabilities (or None where unavailable), revealed items first and the target last. A reply that
    was cut at the output cap or refused counts as unavailable for every item of the request."""
    n = len(revealed) + (target is not None)
    if n == 0:
        return []
    head = {"subject": subject, "items": [{"index": i + 1, **_item(it, REVEALED_CHARS)} for i, it in enumerate(revealed)]}
    content = [{"type": "text", "text": json.dumps(head, ensure_ascii=False, indent=1)}]
    if revealed:
        content[0]["cache_control"] = {"type": "ephemeral"}
    if target is not None:
        content.append({"type": "text", "text": json.dumps({"items": [{"index": n, **_item(target, TARGET_CHARS)}]},
                                                           ensure_ascii=False, indent=1)})
    body = json.dumps({
        "model": model, "max_tokens": 4096 + 64 * n,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    }).encode()
    headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
    for attempt in range(1, attempts + 1):
        left = timeout if deadline is None else min(timeout, deadline - time.monotonic())
        if left < 5:
            return [None] * n
        try:
            req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=left) as resp:
                data = json.loads(resp.read())
            if data.get("stop_reason") in ("max_tokens", "refusal"):
                return [None] * n
            texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            probs = json.loads(texts[-1])["probabilities"]
            if len(probs) != n:
                return [None] * n
            return [float(p) if isinstance(p, (int, float)) and 0.0 <= p <= 1.0 else None for p in probs]
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == attempts:
                return [None] * n
            wait = float(e.headers.get("retry-after") or 0) or 2 ** attempt * (0.5 + random.random())
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == attempts:
                return [None] * n
            wait = 2 ** attempt * (0.5 + random.random())
        except (ValueError, KeyError, IndexError, TypeError):
            return [None] * n
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic() - 5))
        time.sleep(min(wait, 10.0))
    return [None] * n
