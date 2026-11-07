"""Add two plain index tables to the agent folder's catalog (built only from data already in that folder).

model_index.parquet: one row per canonical model name.
benchmark_index.parquet: one row per benchmark.
Success statistics use only 0/1 benchmarks where 1 means success, and pairs with at least 20 graded responses.

Usage: python -I build_indexes.py <box_dir>
"""
import json
import os
import sys

import pandas as pd


def direction(v):
    try:
        return json.loads(v).get("direction")
    except Exception:
        return None


def main():
    cat = os.path.join(sys.argv[1], "data", "catalog")
    ps = pd.read_parquet(os.path.join(cat, "pair_stats.parquet"))
    bm = pd.read_parquet(os.path.join(cat, "benchmarks_all.parquet"))
    bm["direction"] = bm["response_scale"].map(direction)
    ok = ps.merge(bm[["folder", "direction"]], on="folder")
    ok = ok[(ok["response_type"] == "binary") & (ok["direction"] == "higher_is_better") & (ok["n_graded"] >= 20)].copy()
    ok["percentile_in_benchmark"] = ok.groupby("folder")["mean_response"].rank(pct=True)

    named = ok[ok["normalized_name"].notna() & (ok["normalized_name"] != "")]
    mi = named.groupby("normalized_name").agg(
        provider=("provider", lambda x: x.mode().iat[0] if len(x.mode()) else ""),
        release_date=("release_date", "min"), n_benchmarks=("folder", "nunique"), n_pairs=("folder", "size"),
        mean_success=("mean_response", "mean"), mean_percentile_in_benchmark=("percentile_in_benchmark", "mean"),
        benchmarks=("folder", lambda x: ",".join(sorted(set(x))))).reset_index()
    mi.to_parquet(os.path.join(cat, "model_index.parquet"))

    stats = ok.groupby("folder")["mean_response"].agg(
        median_success="median", p10_success=lambda x: x.quantile(0.1), p90_success=lambda x: x.quantile(0.9),
        n_pairs_with_success_stats="size").reset_index()
    bi = bm[["folder", "name", "one_line_description", "description", "domain", "modality", "release_date",
             "response_type", "direction", "n_items", "n_subjects"]].copy()
    bi["description"] = bi["description"].fillna("").str.slice(0, 400)
    bi["domain"] = bi["domain"].map(lambda d: ",".join(d) if d is not None else "")
    bi["modality"] = bi["modality"].map(lambda d: ",".join(d) if d is not None else "")
    bi = bi.merge(stats, on="folder", how="left")
    bi.to_parquet(os.path.join(cat, "benchmark_index.parquet"))
    print(f"model_index rows {len(mi)} | benchmark_index rows {len(bi)} (with success stats {bi['median_success'].notna().sum()})")


if __name__ == "__main__":
    main()
