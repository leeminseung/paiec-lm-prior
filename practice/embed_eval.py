"""Can item-text embeddings predict benchmark difficulty and within-benchmark difficulty on benchmarks never seen?

Uses embed_items.py output. measurement-db benchmarks are split into 5 folds; each fold is predicted from the other four.
Item target: z_i = logit((s + 0.5) / (n + 1)) over the graded responses of the kept pairs.

1. Benchmark difficulty -> pair level. Each item of a held-out benchmark gets a predicted success probability from
   (a) ridge regression on z_i, or (b) the mean z of its k nearest training items (cosine). The pair mean prediction for
   every pair of that benchmark is the mean predicted probability over its items. Scored as mean |pair mean prediction -
   pair success rate|, benchmarks weighted equally, next to the base rate and the oracle benchmark level.
2. Within-benchmark difficulty -> ranking inside a pair. Target z_i minus its benchmark's mean z. Ridge or kNN give a
   predicted ease per item; per held-out subject, AUC against that subject's outcomes on the benchmark's sampled items,
   next to the ceiling (ease = other subjects' success on the same item).
3. Both proxy evaluations, trained on all measurement-db items: per pair, the same AUC for the embedding ease and for the
   Opus low judge, and the pair mean prediction error for the embedding level, the Opus judge and the base rate.

Usage: python -I embed_eval.py <embed_dir> [ridge_lambda]
Sample weights give every benchmark the same total weight and average 1 per item, so lambda is on the item-count scale.
"""
import json
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from exemplar_shift import load

ROOT = "./"
DATA = ROOT + "data/hf/"
CATALOG = ROOT + "agent_box_02/data/catalog/"
EXAMS = {"proxy-10/20": (ROOT + "exams/rehearsal-02", ROOT + "runs/r02-opus-low-fb"),
         "proxy-10/26": (ROOT + "exams/proxy-confirm-1026", ROOT + "runs/pc-sub-opus-low")}
LAMBDA, K = 30.0, 20
sig = lambda x: 1 / (1 + np.exp(-x))


def auc(score, y):
    y = np.asarray(y)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = np.empty(len(allv)); ranks[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    ranks = (np.bincount(inv, ranks) / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ridge(X, y, w, lam=None):
    """Weighted ridge with an unpenalized intercept."""
    lam = LAMBDA if lam is None else lam
    mu_x = np.average(X, axis=0, weights=w); mu_y = np.average(y, weights=w)
    Xc = (X - mu_x) * np.sqrt(w)[:, None]
    coef = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ ((y - mu_y) * np.sqrt(w)))
    return lambda Z: mu_y + (Z - mu_x) @ coef


def knn(Xtr, ytr, Xte, k=K):
    out = np.empty(len(Xte))
    for i in range(0, len(Xte), 1000):
        sim = Xte[i:i + 1000] @ Xtr.T
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        out[i:i + 1000] = ytr[top].mean(axis=1)
    return out


def main():
    d = sys.argv[1]
    global LAMBDA
    if len(sys.argv) > 2:
        LAMBDA = float(sys.argv[2])
    meta = pd.read_parquet(os.path.join(d, "meta.parquet"))
    emb = np.load(os.path.join(d, "emb.npy")).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    keep = meta.embedded.values
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]
    md = meta.source.values == "measurement-db"
    M, E = meta[md].reset_index(drop=True), emb[md]
    M["z"] = np.log((M.s + 0.5) / (M.n - M.s + 0.5))
    M["dz"] = M.z - M.groupby("benchmark").z.transform("mean")
    M["w"] = 1.0 / M.groupby("benchmark").z.transform("count")  # benchmarks weighted equally in fitting
    M["w"] *= len(M) / M.w.sum()  # mean weight 1 per item

    ps = pd.read_parquet(CATALOG + "pair_stats.parquet")
    pairs = ps[ps.folder.isin(set(M.benchmark)) & (ps.response_type == "binary") & (ps.n_graded >= 20)]
    benches = np.array(sorted(M.benchmark.unique()))
    rng = np.random.default_rng(0); rng.shuffle(benches)
    folds = np.array_split(benches, 5)
    print(f"measurement-db: {len(M)} items, {len(benches)} benchmarks, {len(pairs)} pairs, ridge lambda {LAMBDA:g}")

    # 1. benchmark difficulty -> pair level
    level = {"ridge": {}, "knn": {}}
    for f in folds:
        te = M.benchmark.isin(f).values; tr = ~te
        pr = sig(ridge(E[tr], M.z.values[tr], M.w.values[tr])(E[te]))
        pk = sig(knn(E[tr], M.z.values[tr], E[te]))
        for name, p in (("ridge", pr), ("knn", pk)):
            for b, v in pd.Series(p).groupby(M.benchmark.values[te]).mean().items():
                level[name][b] = v
    base = float(pairs.mean_response.mean())
    pairs = pairs.assign(bmean=pairs.groupby("folder").mean_response.transform("mean"))
    rows = []
    for b, g in pairs.groupby("folder"):
        r = g.mean_response.values
        n = len(r)
        loo = (g.bmean.values * n - r) / (n - 1) if n > 1 else np.full(n, base)  # oracle: the benchmark's other pairs
        rows.append({"b": b, "base rate": np.mean(abs(base - r)), "embedding ridge": np.mean(abs(level["ridge"][b] - r)),
                     "embedding kNN": np.mean(abs(level["knn"][b] - r)), "oracle benchmark level": np.mean(abs(loo - r))})
    t = pd.DataFrame(rows)
    print("\n1. pair level: mean |pair mean prediction - pair success rate|, benchmarks weighted equally"
          f" (oracle uses benchmarks with >= 2 pairs: {int((pairs.groupby('folder').size() > 1).sum())})")
    for c in ["base rate", "embedding ridge", "embedding kNN", "oracle benchmark level"]:
        print(f"  {c:24} {t[c].mean():.3f}")

    # 2. within-benchmark difficulty -> ranking inside a pair
    ease = {"ridge": np.empty(len(M)), "knn": np.empty(len(M))}
    for f in folds:
        te = M.benchmark.isin(f).values; tr = ~te
        ease["ridge"][te] = ridge(E[tr], M.dz.values[tr], M.w.values[tr])(E[te])
        ease["knn"][te] = knn(E[tr], M.dz.values[tr], E[te])
    con = duckdb.connect()
    res = []
    for b in benches:
        sub = M[M.benchmark == b]
        subs = pairs[pairs.folder == b].subject_id.tolist()
        r = con.execute(f"select subject_id, item_id, avg(response) y from read_parquet('{DATA}{b}/formatted_tables/responses.parquet') "
                        f"where response is not null and item_id in (select unnest(?)) and subject_id in (select unnest(?)) group by 1, 2",
                        [sub.item_id.tolist(), subs]).df()
        r["y"] = (r.y >= 0.5).astype(int)
        m = r.pivot(index="item_id", columns="subject_id", values="y").reindex(sub.item_id)
        tot, cnt = m.sum(axis=1).values, m.notna().sum(axis=1).values
        a = {"ridge": [], "knn": [], "ceiling": []}
        for s in m.columns:
            y = m[s].values; has = ~np.isnan(y)
            if has.sum() < 20:
                continue
            others = cnt[has] - 1
            for name in ("ridge", "knn"):
                v = auc(ease[name][sub.index.values][has], y[has])
                if v is not None:
                    a[name].append(v)
            if (others > 0).all():
                v = auc((tot[has] - y[has]) / others, y[has])
                if v is not None:
                    a["ceiling"].append(v)
        if a["ridge"]:
            res.append({"b": b, **{k: (np.mean(v) if v else np.nan) for k, v in a.items()}})
    t2 = pd.DataFrame(res)
    print(f"\n2. ranking inside a pair: AUC per held-out subject, averaged per benchmark, then over {len(t2)} benchmarks")
    for c in ["ridge", "knn", "ceiling"]:
        print(f"  {c:24} {t2[c].mean():.3f} (median {t2[c].median():.3f}, benchmarks with a value {int(t2[c].notna().sum())})")

    # 3. proxy evaluations, trained on all measurement-db items
    f_level = ridge(E, M.z.values, M.w.values); f_ease = ridge(E, M.dz.values, M.w.values)
    print("\n3. proxy evaluations (embedding model fit on all measurement-db items above)")
    for name, (exam_dir, judge_dir) in EXAMS.items():
        exam = load_exam(exam_dir); pj = load(judge_dir)
        P = meta[(meta.source == name) & (meta.role == "target")]
        Ep = emb[P.index.values]
        lvl, es = sig(f_level(Ep)), f_ease(Ep)
        rows = []
        for pi, pair in enumerate(exam["pairs"]):
            sel = (P.pair.values == pi)
            keys = P.item_id.values[sel]
            lab = {t["key"]: t["labels"] for t in pair["targets"]}
            ys = np.concatenate([lab[k] for k in keys]); rep = [len(lab[k]) for k in keys]
            e_rep = np.repeat(es[sel], rep); j_rep = np.repeat([pj.get((pi, k), 0.575) for k in keys], rep)
            rate = ys.mean()
            rows.append({"pair": f"{pair['benchmark_name'][:18]} x {pair['subject']['normalized_name'][:20]}", "rate": rate,
                         "emb level": lvl[sel].mean(), "judge level": np.mean([pj.get((pi, k), 0.575) for k in keys]),
                         "emb AUC": auc(e_rep, ys), "judge AUC": auc(j_rep, ys)})
        t3 = pd.DataFrame(rows)
        print(f"  {name}: pair level error  base rate {np.mean(abs(base - t3.rate)):.3f}, embedding {np.mean(abs(t3['emb level'] - t3.rate)):.3f}, "
              f"Opus low judge {np.mean(abs(t3['judge level'] - t3.rate)):.3f} | AUC embedding {t3['emb AUC'].mean():.3f}, Opus low judge {t3['judge AUC'].mean():.3f}")
        print(t3.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
