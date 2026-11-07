"""Pick one competition-shaped round for a practice exam: 7 benchmarks, 9 (system, benchmark) pairs, about 1,000 items.

Shape follows the competition and one participant's published feedback (three rounds of 8-9 pairs over the same
7 hidden benchmarks, nearly all systems distinct, 44-76 scoring items per pair, at most 1,000 items per round):
  - candidate pairs: 0/1 scores where 1 = success, a named system released 2025 or later by one of the nine
    companies the organizers list, at least 80 items, benchmark released 2024 or later, item texts available;
  - 7 benchmarks drawn at random; one system per benchmark, plus a second system on 2 of them (9 pairs);
    no system name repeats within the round;
  - each pair keeps a random 88-152 items (half acquisition, half evaluation), at most 1,000 in total.

Usage: python -I pick_round.py <pairs.parquet> <table_sizes.json> <seed> <exclude1,exclude2,...>
Prints a spec for make_exam.py: bench|subject_id|n_items,...
"""
import json
import random
import sys

import pandas as pd

COMPANIES = {"Anthropic", "Google", "Moonshot", "OpenAI", "Zhipu", "Alibaba", "ByteDance", "DeepSeek", "xAI"}
N_BENCH, N_DOUBLE, CAP = 7, 2, 1000


def main():
    pairs_path, sizes_path, seed, exclude = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4].split(","))
    sizes = json.load(open(sizes_path))
    p = pd.read_parquet(pairs_path)
    c = p[p.graded & p.one_is_success & p.named & p.big & p.recent_subject & p.recent_bench
          & p.provider.isin(COMPANIES) & ~p.folder.isin(exclude)
          & p.folder.map(lambda f: "items.parquet" in sizes.get(f, {}))].copy()
    print(f"candidates: {c.folder.nunique()} benchmarks, {len(c)} pairs, {c.normalized_name.nunique()} systems")

    rng = random.Random(seed)
    benches = sorted(c.folder.unique())
    rng.shuffle(benches)
    used, picked, chosen = set(), [], []

    def pick_from(b):
        opts = c[(c.folder == b) & ~c.normalized_name.isin(used)].sort_values("subject_id")
        if opts.empty:
            return None
        row = opts.iloc[rng.randrange(len(opts))]
        used.add(row.normalized_name)
        return row

    for b in benches:  # skip a benchmark whose only systems are already in the round
        if len(chosen) == N_BENCH:
            break
        row = pick_from(b)
        if row is not None:
            chosen.append(b)
            picked.append(row)
    doubles = chosen[:]
    rng.shuffle(doubles)
    added = 0
    for b in doubles:
        if added == N_DOUBLE:
            break
        row = pick_from(b)
        if row is not None:
            picked.append(row)
            added += 1

    n = [min(int(r.n_items), rng.randint(88, 152)) for r in picked]
    while sum(n) > CAP:  # trim the largest pair until the round fits the cap
        i = max(range(len(n)), key=lambda k: n[k])
        n[i] -= 1
    for r, k in zip(picked, n):
        items_mb = sizes[r.folder]["items.parquet"] / 1e6
        print(f"  {r.folder:28} {str(r['name'])[:44]:44} | {r.normalized_name[:30]:30} {r.provider:9} | "
              f"items in pair {int(r.n_items):5d} -> keep {k:3d} | success {r.succ:.3f} | items.parquet {items_mb:7.1f} MB")
    print(f"total items kept {sum(n)} (scoring about {sum(k - k // 2 for k in n)})")
    print("SPEC " + ",".join(f"{r.folder}|{r.subject_id}|{k}" for r, k in zip(picked, n)))


if __name__ == "__main__":
    main()
