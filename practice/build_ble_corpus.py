"""Build a leakage-safe data bank for the organizers' BLE baseline from our local measurement-db copy.

BLE reads <root>/<benchmark_dir>/{benchmarks,subjects,items,response[,traces]}.parquet plus corpus_manifest.json
(tools/prepare_data.py in paiec_baseline). Instead of downloading the release again, this links the local copy's
tables (data/hf, revision in data/hf/REVISION.txt), keeps the benchmarks on the 10/10 eligible list that have all four
tables, and drops every benchmark used in any proxy evaluation so BLE cannot look up the held-out outcomes.

Usage: python -I build_ble_corpus.py <out_dir> <exam_dir> [<exam_dir> ...]
"""
import json
import os
import sys
import time

HF = "./data/hf"
NEED = {"benchmarks": "benchmarks.parquet", "subjects": "subjects.parquet", "items": "items.parquet",
        "response": "responses.parquet"}


def main():
    out = os.path.abspath(sys.argv[1])
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("out_dir must be new or empty")
    held = set()
    for exam_dir in sys.argv[2:]:
        held |= {p["benchmark_name"] for p in json.load(open(os.path.join(exam_dir, "exam.json")))["pairs"]}
    revision = open(os.path.join(HF, "REVISION.txt")).read().strip()
    eligible = [l.split()[0] for l in open(os.path.join(HF, "eligible_public_20261010.txt")) if l.strip()]
    keep, files = [], []
    for d in sorted(eligible):
        src = os.path.join(HF, d, "formatted_tables")
        if d in held or not all(os.path.isfile(os.path.join(src, f)) for f in NEED.values()):
            continue
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for table, f in NEED.items():
            os.symlink(os.path.join(src, f), os.path.join(out, d, f"{table}.parquet"))
            files.append(f"{d}/{table}.parquet")
        keep.append(d)
    with open(os.path.join(out, "corpus_manifest.json"), "w") as fh:
        json.dump({"db_repo": "aims-foundations/measurement-db", "corpus_scope": "paec-public-training",
                   "revision": revision, "generated_unix": int(time.time()), "eligible_dirs": keep,
                   "files": sorted(files), "note": "local build: proxy-evaluation benchmarks removed"}, fh, indent=2)
    print(f"held-out benchmarks {len(held)}; kept {len(keep)} of {len(eligible)} eligible "
          f"(dropped {len(set(eligible) & held)} held-out, {len(eligible) - len(keep) - len(set(eligible) & held)} without full local tables)")


if __name__ == "__main__":
    main()
