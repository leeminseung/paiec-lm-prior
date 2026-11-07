"""Exam file for the post-cutoff check: MathArena sets built from arXiv papers of August and September 2026.

Each (set, system) is one pair; each problem is one target item whose labels are the system's graded runs (0/1).
The item content is the prompt the system received (the same for every system in a set); item features are left
empty so the arXiv id of the source paper is not shown. Budget 0 only (no acquisitions). Read-only on the data folder.

Usage: python -I make_matharena_exam.py <data_dir> <exam_dir> [prompt|problem]
"problem" puts only the problem statement in the item content, for the cloze probe: the prompt's first 1,200
characters are the shared instructions, so cloze spans would test the template, not the problem.
"""
import hashlib
import json
import os
import re
import sys

import pandas as pd

SETS = ["arxivmath-0826", "brokenarxiv-0826", "arxivmath-0926", "brokenarxiv-0926", "arxivlean-0926"]
PROVIDERS = [("Claude", "Anthropic"), ("GPT", "OpenAI"), ("Gemini", "Google"), ("DeepSeek", "DeepSeek"), ("Grok", "xAI"),
             ("Qwen", "Alibaba"), ("Kimi", "Moonshot"), ("Mistral", "Mistral"), ("Muse", "Meta")]
h = lambda *x: hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()[:16]


def subject(model_name):
    m = re.match(r"^(.*?)\s*(?:\((.*)\))?$", model_name.strip())
    base, paren = m.group(1), (m.group(2) or "")
    provider = next((p for k, p in PROVIDERS if base.startswith(k)), "")
    name = base.replace("-", " ").replace("Claude Opus", "Claude Opus").strip()
    harness, effort = "MathArena", ""
    if paren in ("OpenCode", "Kimi Code", "Qwen Code"):
        harness = f"MathArena via {paren}"
    elif paren:
        effort = paren.lower()
    return {"normalized_name": f"{provider} {name}".strip(), "provider": provider, "release_date": "", "access_date": "",
            "harness": harness, "harness_version": "", "reasoning_effort": effort, "subject_features_extra": ""}


def main():
    data_dir, exam_dir = sys.argv[1:3]
    content = (sys.argv[3:4] or ["prompt"])[0]
    pairs = []
    for s in SETS:
        d = pd.read_parquet(f"{data_dir}/{s}_outputs.parquet")
        bid = f"benchmark_{h('matharena', s)[:6]}"
        prompts = d.groupby("problem_idx")["user_message" if content == "prompt" else "problem"].first()
        for mname, g in d.groupby("model_name"):
            targets = []
            for pidx, gg in g.groupby("problem_idx"):
                targets.append({"key": h(s, pidx), "labels": [int(x) for x in gg.correct],
                                "item": {"item_content": prompts[pidx], "item_features": "", "interactors": "", "benchmark_id": bid}})
            pairs.append({"benchmark_name": f"matharena_{s.replace('-', '_')}", "anon_benchmark_id": bid,
                          "subject_id": h("subject", mname), "subject": subject(mname),
                          "pair_success_rate": float(g.correct.mean()), "n_graded": int(len(g)), "acquisitions": [],
                          "targets": sorted(targets, key=lambda t: t["key"])})
    exam = {"seed": 0, "held_out_benchmarks": [], "purpose": "post-cutoff check: MathArena sets from August and September 2026 arXiv papers (budget 0)",
            "pairs": pairs}
    os.makedirs(exam_dir, exist_ok=True)
    json.dump(exam, open(f"{exam_dir}/exam.json", "w"), ensure_ascii=False)
    print(len(pairs), "pairs,", sum(len(p["targets"]) for p in pairs), "subject-item targets")
    for p in pairs[:3] + pairs[-2:]:
        print("  ", p["benchmark_name"], p["subject"]["normalized_name"], "|", p["subject"]["reasoning_effort"], p["subject"]["harness"], "| rate", round(p["pair_success_rate"], 3))


if __name__ == "__main__":
    main()
