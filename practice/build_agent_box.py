"""Build the agent's sandbox folder: training data minus the held-out benchmarks, a catalog, and a README.

Files are hard-linked (no extra disk space) so they live inside the agent's folder, which is the only
place the agent may read. Held-out benchmarks are left out entirely, so their answers are unreachable.

Usage: python -I build_agent_box.py <data_dir> <exam_dir> <box_dir>
"""
import json
import os
import sys

import duckdb
import pandas as pd

TABLES = ["benchmarks.parquet", "subjects.parquet", "responses.parquet", "items.parquet"]


def main():
    data_dir, exam_dir, box = sys.argv[1:4]
    held_out = set(json.load(open(os.path.join(exam_dir, "exam.json")))["held_out_benchmarks"])
    out = os.path.join(box, "data")
    os.makedirs(os.path.join(out, "catalog"), exist_ok=True)
    benches, linked = [], 0
    for b in sorted(os.listdir(data_dir)):
        src = os.path.join(data_dir, b, "formatted_tables")
        if b in held_out or not os.path.isdir(src):
            continue
        benches.append(b)
        os.makedirs(os.path.join(out, b), exist_ok=True)
        for t in TABLES:
            s, d = os.path.join(src, t), os.path.join(out, b, t)
            if os.path.exists(s) and not os.path.exists(d):
                os.link(s, d)
                linked += 1
    bm = pd.concat([pd.read_parquet(os.path.join(out, b, "benchmarks.parquet")).assign(folder=b) for b in benches], ignore_index=True)
    su = pd.concat([pd.read_parquet(os.path.join(out, b, "subjects.parquet")).assign(folder=b) for b in benches], ignore_index=True)
    bm.to_parquet(os.path.join(out, "catalog", "benchmarks_all.parquet"))
    su.to_parquet(os.path.join(out, "catalog", "subjects_all.parquet"))
    resp_files = [os.path.join(out, b, "responses.parquet") for b in benches if os.path.exists(os.path.join(out, b, "responses.parquet"))]
    stats = duckdb.connect().execute("""
        SELECT regexp_extract(filename, '([^/]+)/responses\\.parquet$', 1) AS folder, subject_id,
               count(DISTINCT item_id) AS n_items, count(response) AS n_graded, avg(response) AS mean_response
        FROM read_parquet(?, filename = true) GROUP BY ALL
    """, [resp_files]).df()
    stats = stats.merge(su[["folder", "subject_id", "normalized_name", "provider", "release_date", "harness",
                            "harness_version", "reasoning_effort"]], on=["folder", "subject_id"], how="left")
    stats = stats.merge(bm[["folder", "name", "response_type", "release_date", "domain"]].rename(
        columns={"name": "benchmark_name", "release_date": "benchmark_release_date"}), on="folder", how="left")
    stats.to_parquet(os.path.join(out, "catalog", "pair_stats.parquet"))
    print(f"benchmarks: {len(benches)} (held out: {sorted(held_out)}) | files linked: {linked} | "
          f"responses tables: {len(resp_files)} | pair_stats rows: {len(stats)}")


if __name__ == "__main__":
    main()
