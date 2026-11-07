"""Write a smaller exam file: the first <pairs_per_benchmark> pairs of every benchmark, in the source order, optionally
only the first <first_n_pairs> of those. Pairs keep their content; each records its index in the source exam
("source_pair") so judgments can be matched back to runs on the full exam.

Usage: python -I subset_exam.py <src_exam_dir> <out_dir> <pairs_per_benchmark> [first_n_pairs]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam


def main():
    src, out_dir, per_bench = sys.argv[1], sys.argv[2], int(sys.argv[3])
    first_n = int(sys.argv[4]) if len(sys.argv) > 4 else None
    exam = load_exam(src)
    seen, pairs = {}, []
    for pi, p in enumerate(exam["pairs"]):
        b = p["benchmark_name"]
        if seen.get(b, 0) < per_bench:
            seen[b] = seen.get(b, 0) + 1
            pairs.append({**p, "source_pair": pi})
    pairs = pairs[:first_n] if first_n else pairs
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exam.json"), "w") as f:
        json.dump({**exam, "purpose": f"{exam.get('purpose', '')}; subset of {src}", "pairs": pairs}, f, ensure_ascii=False)
    print(f"{len(pairs)} pairs over {len({p['benchmark_name'] for p in pairs})} benchmarks, "
          f"{sum(len(p['targets']) for p in pairs)} targets")


if __name__ == "__main__":
    main()
