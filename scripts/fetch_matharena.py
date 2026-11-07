"""Download the MathArena output sets used in Section 4.4 (problems from arXiv papers of August and September 2026),
pinned to the dataset revisions we read on 1 November 2026, keeping only the columns the exam builder needs.
DuckDB reads the needed columns straight from Hugging Face (a few MB instead of the full transcripts). Data: CC BY-SA 4.0.

Usage: uv run --no-project --with duckdb==2.0.0 python -I scripts/fetch_matharena.py data/matharena-1101
"""
import os
import sys
import time

import duckdb

SETS = {"arxivmath-0826": "abee8f419908493ffff061bebfe8e9c34c151440",
        "brokenarxiv-0826": "107d15fa81cf14a0537907b7f711617e94e0be67",
        "arxivmath-0926": "17dcbe7b355b7fcb56231a430f684d534f96959d",
        "brokenarxiv-0926": "46f411a3b1b220fd66120b6d2ad32592c31fe26b",
        "arxivlean-0926": "1f0871bd4e193ccdc98bb5206be33f35ce84849e"}
WANTED = ["problem_idx", "problem", "model_name", "model_config", "idx_answer", "source", "gold_answer",
          "formal_statement", "user_message", "correct", "points", "max_points", "points_judge_1",
          "max_points_judge_1"]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 1")  # parallel range reads failed on Hugging Face connection errors
    for s, rev in SETS.items():
        url = f"hf://datasets/MathArena/{s}_outputs@{rev}/**/*.parquet"
        cols = [c for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchall()]
        have = [c[0] for c in cols if c[0] in WANTED]
        sel = ", ".join(f'"{c}"' for c in have)
        for attempt in range(4):
            try:
                con.execute(f"COPY (SELECT {sel} FROM read_parquet('{url}')) TO '{out}/{s}_outputs.parquet' (FORMAT parquet)")
                break
            except duckdb.Error as e:
                print(s, "retry after", type(e).__name__); time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"{s}: download failed")
        print(s, rev[:8], len(have), "columns")


if __name__ == "__main__":
    main()
