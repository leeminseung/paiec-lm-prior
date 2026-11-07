"""Tag what each judge thinking summary (thinking_judge.py output) relies on, with a Claude model through the subscription
(`claude -p`, nothing else from Claude Code). One request per summary; the tagger sees the summary text only, not the
item, the outcome or the sample cell. Resumable.

Usage: python -I tag_summaries.py <judgments.jsonl> <out.jsonl> <model> <effort> <workers> <empty_dir>
"""
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from subscription_judge import clean_env

TAGS = ["names_benchmark", "recalls_results", "subject_capability", "item_difficulty", "grading", "falls_back_to_default"]
BASES = TAGS + ["none"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": TAGS + ["main_basis"],
          "properties": {**{t: {"type": "boolean"} for t in TAGS}, "main_basis": {"type": "string", "enum": BASES}}}
SYSTEM = """You read a short summary of a model's reasoning. The model was estimating the probability that a given AI system (the subject) answers one benchmark item correctly. Mark what the reasoning relies on:

- names_benchmark: it names or identifies the specific benchmark, dataset or paper the item comes from.
- recalls_results: it cites known results, such as scores, leaderboard positions, typical accuracies or findings reported for that benchmark, paper or subject.
- subject_capability: it reasons about how strong the subject is (model family, generation, size, reasoning mode, harness).
- item_difficulty: it reasons about what makes this particular item easy or hard (steps, domain knowledge, length, trickiness).
- grading: it reasons about how success is decided (exact match, tests, strictness, answer format, partial credit).
- falls_back_to_default: it says it is unsure and settles on a middle or default value.
- main_basis: the one of these that most drives the final number, or none if the summary is empty or gives no basis.

Answer with JSON matching the schema."""


def tag(summary, model, effort, cwd, timeout=180):
    cmd = ["claude", "-p", json.dumps({"summary": summary or ""}, ensure_ascii=False), "--model", model, "--effort", effort,
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
           "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--no-session-persistence"]
    try:
        out = subprocess.run(cmd, cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "result":
            try:
                return d.get("structured_output") or json.loads(d.get("result", "")), None
            except (TypeError, ValueError):
                return None, f"unparsed: {str(d.get('result'))[:200]}"
    return None, (out.stderr or out.stdout)[:300]


def main():
    src, out_path, model, effort, workers, cwd = sys.argv[1:7]
    rows = [json.loads(line) for line in open(src)]
    rows = [r for r in rows if r.get("probability") is not None]
    done = set()
    if os.path.exists(out_path):
        done = {(r["pair"], r["key"]) for r in map(json.loads, open(out_path)) if r.get("tags")}
    todo = [r for r in rows if (r["pair"], r["key"]) not in done]
    lock = threading.Lock()

    def run(r):
        tags, err = tag(r.get("summary", ""), model, effort, cwd)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps({"pair": r["pair"], "key": r["key"], "tags": tags, "error": err, "tagger": model}) + "\n")

    print(f"{len(todo)} summaries to tag", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        list(ex.map(run, todo))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
