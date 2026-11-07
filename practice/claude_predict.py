"""Run selected practice-exam calls with Claude Code in print mode, one fresh session per prediction.

Each call mirrors one competition predict(input, labeled) call. Its inputs go to a fresh call folder inside
the run folder, outside the agent folder. The session runs in the agent folder with Read/Grep/Glob/Bash and
must answer with {"probability", "rationale"}. Every Bash command is allowed without a prompt but runs inside
the OS sandbox: no network; the home folder, /tmp and the macOS per-user temp folders are unreadable except the
agent folder, the Python install and this call's folder; writes go only to this call's folder, which also holds
the session's temp folder. Other places outside the home folder (system files) stay readable. So one call
cannot see another call's revealed results, the answers, or the orchestrator's logs.
Transcripts and metrics are written to the run folder.

Usage: python -I claude_predict.py <exam_dir> <box_dir> <run_dir> <n_calls|kN> <concurrency> <model> <effort> <timeout_s> [start_index]
       n_calls = a number of calls spread over pairs and budgets; kN = N fixed targets per pair asked at all six budgets;
       keys=<file.json> = the listed targets [{"pair": i, "key": k}, ...] asked at all six budgets;
       b0all = every evaluation item once at budget 0
"""
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only
from common import BUDGETS, SUBJECT_KEYS, input_for, labeled_at, load_exam

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.dumps({"type": "object", "additionalProperties": False, "required": ["probability", "rationale"],
                     "properties": {"probability": {"type": "number", "minimum": 0, "maximum": 1},
                                    "rationale": {"type": "string"}}})
UV_PYTHON = os.path.expanduser("~/.local/share/uv/python")
TMP_ROOT = "/private/tmp/paiec-t"  # short on purpose, see run_one


def select_calls_listed(exam, path):
    """Targets named in a JSON file [{"pair": i, "key": k}, ...], each asked at all six budgets."""
    wanted = json.load(open(path))
    by_key = {(pi, t["key"]): t for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]}
    chosen = [(w["pair"], by_key[(w["pair"], w["key"])]) for w in wanted]
    return [(b, pi, t) for b in BUDGETS for pi, t in chosen]


def select_calls_fixed(exam, per_pair):
    """As the competition scores: the same evaluation items at every budget. Draw per_pair targets per pair
    (seeded by the exam) and ask each one at all six budgets, budget by budget."""
    import random
    rng = random.Random(f"{exam['seed']}:calls:{per_pair}")
    chosen = {pi: rng.sample(pair["targets"], min(per_pair, len(pair["targets"])))
              for pi, pair in enumerate(exam["pairs"])}
    return [(b, pi, t) for b in BUDGETS for pi in range(len(exam["pairs"])) for t in chosen[pi]]


def select_calls(exam, n):
    """Spread calls over pairs and budgets: one target per (pair, budget) in turn, distinct targets per pair.
    Used for the first practice exam; it mixes different items across budgets."""
    calls, used = [], {i: 0 for i in range(len(exam["pairs"]))}
    rounds = 0
    while len(calls) < n:
        for b in BUDGETS:
            for pi, pair in enumerate(exam["pairs"]):
                if len(calls) >= n:
                    break
                t = pair["targets"][(used[pi] * 37 + 11) % len(pair["targets"])]
                used[pi] += 1
                calls.append((b, pi, t))
        rounds += 1
    return calls


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC") or k == "CLAUDECODE")}
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


def command(box, cdir, tdir, model, effort, user_prompt):
    # Without the allow rule, print mode silently denies any command the static checker cannot approve,
    # which included most multi-line pandas analysis (2026-10-19 finding).
    settings = {"permissions": {"allow": ["Bash"]},
                "sandbox": {"enabled": True, "allowUnsandboxedCommands": False, "failIfUnavailable": True,
                            "filesystem": {"denyRead": ["~/", "/tmp", "/private/tmp", "/private/var/folders"],
                                           "allowRead": [box, UV_PYTHON, cdir, tdir],
                                           "denyWrite": [box], "allowWrite": [cdir, tdir]},
                            "network": {"allowedDomains": []}}}
    return ["claude", "-p", user_prompt, "--model", model, "--effort", effort,
            "--restricted", "--tools", "Read,Grep,Glob,Bash", "--disallowedTools", "WebSearch,WebFetch",
            "--add-dir", cdir,
            "--setting-sources", "", "--settings", json.dumps(settings),
            "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": {}}),
            "--append-system-prompt-file", os.path.join(HERE, "prompts", "agent_append.md"),
            "--json-schema", SCHEMA, "--output-format", "stream-json", "--verbose", "--no-session-persistence"]


def summarize(lines):
    tools, bash, result = {}, [], None
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tools[block["name"]] = tools.get(block["name"], 0) + 1
                    if block["name"] == "Bash":
                        bash.append(str(block.get("input", {}).get("command", ""))[:300])
        if ev.get("type") == "result":
            result = ev
    return tools, bash, result


def write_summary(cdir, labeled):
    """Same revealed results as labeled.json, as a small table plus one text file per distinct item."""
    import csv
    import hashlib
    os.makedirs(os.path.join(cdir, "items"), exist_ok=True)
    numbers = {}
    with open(os.path.join(cdir, "labeled_summary.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["result_no", *SUBJECT_KEYS, "benchmark_id", "item_no", "outcome"])
        for i, ((s, it), y) in enumerate(labeled):
            h = hashlib.sha256(it["item_content"].encode()).hexdigest()
            if h not in numbers:
                numbers[h] = len(numbers) + 1
                with open(os.path.join(cdir, "items", f"{numbers[h]}.txt"), "w") as g:
                    g.write(it["item_content"])
            w.writerow([i + 1, *[s[k] for k in SUBJECT_KEYS], it["benchmark_id"], numbers[h], y])


def call_dir(run_dir, idx, b, pi, key):
    """The id names the target item too, so a different draw can never reuse an id for another item."""
    import hashlib
    call_id = f"c{idx:03d}_b{b:02d}_p{pi}_{hashlib.sha1(str(key).encode()).hexdigest()[:6]}"
    return call_id, os.path.join(os.path.abspath(run_dir), "inputs", call_id)


def run_one(args):
    idx, (b, pi, t), exam, box, run_dir, model, effort, timeout_s = args
    call_id, cdir = call_dir(run_dir, idx, b, pi, t["key"])
    os.makedirs(cdir)  # fresh folder only: main() refuses to reuse one
    # Private temp folder per session; otherwise sessions share /tmp/claude-<uid> with every other session.
    # Claude Code ignores CLAUDE_CODE_TMPDIR for commands when "<it>/claude-<uid>" is longer than 44 bytes,
    # so the path must stay short; it is moved into the call folder afterwards.
    os.makedirs(TMP_ROOT, mode=0o700, exist_ok=True)
    tdir = os.path.join(TMP_ROOT, secrets.token_hex(4))
    os.makedirs(tdir, mode=0o700)
    env = clean_env()
    env["CLAUDE_CODE_TMPDIR"] = tdir
    pair = exam["pairs"][pi]
    json.dump(input_for(pair, t), open(os.path.join(cdir, "input.json"), "w"), ensure_ascii=False)
    json.dump(labeled_at(exam, b), open(os.path.join(cdir, "labeled.json"), "w"), ensure_ascii=False)
    write_summary(cdir, labeled_at(exam, b))
    prompt = (f"Predict for the call folder {cdir} (files input.json and labeled.json there). "
              f"The data folder is {box}, your starting working directory; its README.md describes the data.")
    start = time.time()
    status, lines, err = "ok", [], ""
    try:
        proc = subprocess.run(command(box, cdir, tdir, model, effort, prompt), cwd=box, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
        lines, err = proc.stdout.splitlines(), proc.stderr
        if proc.returncode != 0:
            status = f"exit {proc.returncode}"
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        lines = (e.stdout or b"").decode(errors="replace").splitlines() if isinstance(e.stdout, bytes) else (e.stdout or "").splitlines()
    wall = time.time() - start
    shutil.move(tdir, os.path.join(cdir, "tmp"))
    with open(os.path.join(run_dir, "calls", f"{call_id}.jsonl"), "w") as f:
        f.write("\n".join(lines))
    if err:
        open(os.path.join(run_dir, "calls", f"{call_id}.stderr.txt"), "w").write(err[-5000:])
    tools, bash, result = summarize(lines)
    prob, rationale = None, None
    if result:
        out = result.get("structured_output")
        if out is None:
            try:
                out = json.loads(result.get("result", ""))
            except (json.JSONDecodeError, TypeError):
                out = None
        if isinstance(out, dict) and isinstance(out.get("probability"), (int, float)):
            prob, rationale = float(out["probability"]), out.get("rationale")
        if result.get("is_error"):
            status = f"error {result.get('subtype')}"
    usage = (result or {}).get("usage", {}) or {}
    rec = {"call_id": call_id, "budget": b, "pair": pi, "key": t["key"], "status": status, "probability": prob,
           "rationale": rationale, "wall_s": round(wall, 1), "num_turns": (result or {}).get("num_turns"),
           "cost_usd_equiv": (result or {}).get("total_cost_usd"), "input_tokens": usage.get("input_tokens"),
           "cache_read_tokens": usage.get("cache_read_input_tokens"), "cache_write_tokens": usage.get("cache_creation_input_tokens"),
           "output_tokens": usage.get("output_tokens"), "tools": tools, "bash": bash, "labeled_n": len(labeled_at(exam, b))}
    return rec


def main():
    exam_dir, box, run_dir = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
    design, conc, model, effort, timeout_s = sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7], int(sys.argv[8])
    exam = load_exam(exam_dir)
    os.makedirs(os.path.join(run_dir, "calls"), exist_ok=True)
    start = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    if design == "b0all":  # every evaluation item once, at budget 0 (nothing revealed)
        calls = [(0, pi, t) for pi, pair in enumerate(exam["pairs"]) for t in pair["targets"]]
    elif design.startswith("keys="):
        calls = select_calls_listed(exam, design[len("keys="):])
    elif design.startswith("k"):
        calls = select_calls_fixed(exam, int(design[1:]))
    else:
        calls = select_calls(exam, int(design))
    jobs = [(i, c, exam, box, run_dir, model, effort, timeout_s) for i, c in enumerate(calls) if i >= start]
    reused = [call_dir(run_dir, i, b, pi, t["key"])[0] for i, (b, pi, t), *_ in jobs
              if os.path.exists(call_dir(run_dir, i, b, pi, t["key"])[1])]
    if reused:
        sys.exit(f"call folders already exist in {run_dir}/inputs: {reused[:5]}... use a new run folder")
    with open(os.path.join(run_dir, "predictions.jsonl"), "a") as out, ThreadPoolExecutor(conc) as ex:
        for rec in ex.map(run_one, jobs):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{rec['call_id']} {rec['status']} p={rec['probability']} {rec['wall_s']}s turns={rec['num_turns']} "
                  f"tools={rec['tools']} cost≈{rec['cost_usd_equiv']}", flush=True)


if __name__ == "__main__":
    main()
