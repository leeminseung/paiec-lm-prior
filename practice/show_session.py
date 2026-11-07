"""Print what rehearsal sessions did: each tool call, a slice of its result, and the final answer.

Usage: python -I show_session.py <run_dir> [call_id ...]      (all calls when none given)
       Set WIDTH=<n> to change how much of each command and result is shown (default 300).
"""
import json
import os
import re
import sys

REFUSED = re.compile(r"requires? approval|cannot be statically analyzed|Contains simple_expansion")
WIDTH = int(os.environ.get("WIDTH", "300"))


def show(path):
    uses, results = [], {}
    for line in open(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    uses.append((block["id"], block["name"], block.get("input", {})))
        if ev.get("type") == "user" and isinstance(ev.get("message", {}).get("content"), list):
            for block in ev["message"]["content"]:
                if block.get("type") == "tool_result":
                    t = block.get("content")
                    if isinstance(t, list):
                        t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                    results[block["tool_use_id"]] = str(t)
    for uid, name, inp in uses:
        if name == "StructuredOutput":
            print(f"  ANSWER p={inp.get('probability')} | {inp.get('rationale', '')}")
            continue
        shown = inp.get("command", inp.get("file_path", inp.get("pattern", json.dumps(inp, ensure_ascii=False))))
        res = results.get(uid, "")
        flag = " [REFUSED]" if REFUSED.search(res) else ""
        print(f"  [{name}]{flag} {shown[:WIDTH]!r}\n      -> {res[:WIDTH]!r}")


def main():
    run_dir = sys.argv[1]
    calls = sys.argv[2:] or sorted(f[:-6] for f in os.listdir(os.path.join(run_dir, "calls")) if f.endswith(".jsonl"))
    for cid in calls:
        print(f"===== {cid}")
        show(os.path.join(run_dir, "calls", f"{cid}.jsonl"))


if __name__ == "__main__":
    main()
