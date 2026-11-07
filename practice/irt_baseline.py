"""A strong predictor without any LLM: a hierarchical logistic (IRT-style) model with priors learned on measurement-db
and a Bayesian update from the revealed labels at prediction time.

For subject s, benchmark B and item i:  logit P(y = 1) = mu + b_B + c_sB + d_i
  b_B   benchmark offset, prior N(0, sB^2)                         (cold start: the benchmark is new)
  c_sB  subject's offset on this benchmark, prior N(g(s), sC^2)    g = ridge on subject attributes (provider, release
                                                                   date, reasoning effort, harness given) plus a shrunk
                                                                   effect of the model name when it occurs in training
  d_i   item offset, prior N(beta . x_i, sD^2)                     x_i = simple text features of the item
At prediction time the revealed labels of every pair on the target's benchmark (labeled holds all sampled pairs) give a
joint posterior over b_B, every c_sB seen, and d_i of every revealed item, by Laplace approximation (Newton to the
mode). The target's logit mean m and variance v give P = s(m / sqrt(1 + pi v / 8)). Other subjects' labels on the
benchmark inform b_B; a revealed item shared with another subject informs its d_i.

Fitting (training benchmarks only: binary, 1 = success, all proxy-evaluation benchmarks excluded, benchmarks equal):
pair logits on the eta scale (divided by kappa = 1 / sqrt(1 + pi sD^2 / 8), since a pair's success rate averages over
items), mu and sB from benchmark means, sC from the within-benchmark spread left after g, sD from the within-benchmark
spread of item logits (minus binomial noise), beta by ridge on item features within benchmarks.

Usage: python -I irt_baseline.py fit <out.json> <exclude_exam_dir>[,...]   (the factory reads IRT_PARAMS or the default path)
"""
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam

ROOT = "./"
PAIRS = ROOT + "eda/out/pairs_20261010.parquet"
ITEMS_META = ROOT + "runs/embed-20261027/meta.parquet"
DEFAULT_PARAMS = ROOT + "runs/irt-baseline/params.json"
PROVIDERS = ["OpenAI", "Anthropic", "Google", "Meta", "Alibaba", "DeepSeek", "Mistral", "Microsoft", "xAI", "Moonshot",
             "Zhipu", "ByteDance", "Qwen", "Amazon", "Cohere", "NVIDIA"]
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
logit = lambda p: math.log(p / (1 - p))


def subject_features(provider, release_date, effort, harness):
    """Vector of subject attributes (strings as the competition supplies them; empty = unknown)."""
    v = [1.0 if provider == p else 0.0 for p in PROVIDERS]
    v.append(1.0 if provider and provider not in PROVIDERS else 0.0)
    d = pd.to_datetime(release_date, errors="coerce") if release_date else pd.NaT
    v += [((d.year + d.month / 12) - 2025.0) if not pd.isna(d) else 0.0, 0.0 if not pd.isna(d) else 1.0]
    e = (effort or "").strip().lower()
    v += [1.0 if e == x else 0.0 for x in EFFORTS] + [1.0 if e and e not in EFFORTS else 0.0]
    v.append(1.0 if harness else 0.0)
    return np.array(v)


MC = re.compile(r"(^|\n)\s*(\(?[A-Ja-j][\).:]|[A-J]\s*[:.)])\s+\S", re.M)


def item_features(content):
    text = content or ""
    return np.array([math.log1p(len(text)) - 7.0, 1.0 if len(MC.findall(text)) >= 3 else 0.0,
                     1.0 if re.search(r"```|def |class |import |#include", text) else 0.0])


def s(x):
    return 1 / (1 + np.exp(-x))


def fit(exclude):
    p = pd.read_parquet(PAIRS)
    p = p[p.graded & p.one_is_success & ~p.folder.isin(exclude) & (p.n_graded >= 20)].copy()
    # item spread sD from item-level success rates (embedding sample: up to 300 items per benchmark)
    meta = pd.read_parquet(ITEMS_META)
    meta = meta[(meta.source == "measurement-db") & ~meta.benchmark.isin(exclude) & (meta.n >= 3)].copy()
    meta["z"] = np.log((meta.s + 0.5) / (meta.n - meta.s + 0.5))
    meta["noise"] = 1 / (meta.s + 0.5) + 1 / (meta.n - meta.s + 0.5)
    g = meta.groupby("benchmark")
    var_d = (g.z.var() - g.noise.mean()).clip(lower=0)
    sD = float(np.sqrt(var_d.median()))
    kappa = 1 / math.sqrt(1 + math.pi * sD ** 2 / 8)
    # pair logits on the eta scale
    rate = (p.succ * p.n_graded + 0.5) / (p.n_graded + 1)
    p["eta"] = np.log(rate / (1 - rate)) / kappa
    bm = p.groupby("folder").eta.mean()
    mu = float(bm.mean())
    p["dev"] = p.eta - p.folder.map(bm)
    X = np.stack([subject_features(*(None if pd.isna(v) else str(v) for v in r))
                  for r in p[["provider", "release_date", "reasoning_effort", "harness"]].itertuples(index=False)])
    w = 1 / p.groupby("folder").eta.transform("count").values  # benchmarks equal
    lam = 5.0
    Xc = X * np.sqrt(w)[:, None]
    coef = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]) * w.sum() / len(w), Xc.T @ (p.dev.values * np.sqrt(w)))
    p["resid"] = p.dev - X @ coef
    multi = p.groupby("folder").filter(lambda d: len(d) >= 2)
    sC = float(np.sqrt(multi.groupby("folder").resid.var().median()))
    sB = float(np.sqrt(max(bm.var() - (p.groupby("folder").eta.var() / p.groupby("folder").eta.count()).median(), 0.01)))
    # name effects: shrunk mean residual over the name's training pairs (benchmarks other than the target's, by design)
    named = p[p.normalized_name.notna()]
    ne = named.groupby("normalized_name").resid.agg(["sum", "count"])
    name_effect = {k: float(r["sum"] / (r["count"] + 2.0)) for k, r in ne.iterrows()}
    # item-feature slope beta within benchmarks
    items = []
    for b, gg in meta.groupby("benchmark"):
        path = f"{ROOT}data/hf/{b}/formatted_tables/items.parquet"
        if not os.path.exists(path):
            continue
        t = pd.read_parquet(path, columns=["item_id", "content"]).set_index("item_id")
        gg = gg[gg.item_id.isin(t.index)]
        if len(gg) < 10:
            continue
        F = np.stack([item_features(str(t.loc[i, "content"]) if t.loc[i, "content"] is not None else "") for i in gg.item_id])
        items.append((F, (gg.z - gg.z.mean()).values / kappa, np.full(len(gg), 1 / len(gg))))
    F = np.concatenate([a for a, _, _ in items]); y = np.concatenate([b for _, b, _ in items]); wi = np.concatenate([c for _, _, c in items])
    F = F - np.average(F, axis=0, weights=wi)
    Fc = F * np.sqrt(wi)[:, None]
    beta = np.linalg.solve(Fc.T @ Fc + 1.0 * np.eye(F.shape[1]) * wi.sum() / len(wi), Fc.T @ (y * np.sqrt(wi)))
    return {"mu": mu, "sB": sB, "sC": sC, "sD": sD, "kappa": kappa, "coef": coef.tolist(), "beta": beta.tolist(),
            "name_effect": name_effect, "n_train_pairs": int(len(p)), "n_train_benchmarks": int(p.folder.nunique()),
            "excluded": sorted(exclude)}


class IRT:
    def __init__(self, params, use_names=True, use_subject=True, use_items=True, scale=(1.0, 1.0, 1.0)):
        self.P = params
        self.coef = np.array(params["coef"]); self.beta = np.array(params["beta"])
        self.use_names, self.use_subject, self.use_items = use_names, use_subject, use_items
        self.sB, self.sC, self.sD = params["sB"] * scale[0], params["sC"] * scale[1], params["sD"] * scale[2]
        self.cache = {}

    def g(self, subj):
        v = 0.0
        if self.use_subject:
            v += float(subject_features(subj.get("provider"), subj.get("release_date"), subj.get("reasoning_effort"),
                                        subj.get("harness")) @ self.coef)
        if self.use_names:
            v += self.P["name_effect"].get(subj.get("normalized_name") or "", 0.0)
        return v

    def dprior(self, item):
        return float(item_features(item.get("item_content", "")) @ self.beta) if self.use_items else 0.0

    def posterior(self, labeled, bench):
        """Mode and covariance over [b, c_s..., d_i...] from the revealed labels on this benchmark."""
        obs = [(tuple(sorted(sj.items())), it.get("item_content", ""), it, y) for (sj, it), y in labeled
               if it["benchmark_id"] == bench]
        subs = sorted({o[0] for o in obs}); items = sorted({o[1] for o in obs})
        si = {k: 1 + j for j, k in enumerate(subs)}; ii = {k: 1 + len(subs) + j for j, k in enumerate(items)}
        n = 1 + len(subs) + len(items)
        prior_m = np.zeros(n); prior_v = np.zeros(n)
        prior_v[0] = self.sB ** 2
        for k, j in si.items():
            prior_m[j] = self.g(dict(k)); prior_v[j] = self.sC ** 2
        item_obj = {o[1]: o[2] for o in obs}
        for k, j in ii.items():
            prior_m[j] = self.dprior(item_obj[k]); prior_v[j] = self.sD ** 2
        A = np.zeros((len(obs), n)); y = np.array([o[3] for o in obs], dtype=float)
        for r, o in enumerate(obs):
            A[r, 0] = 1; A[r, si[o[0]]] = 1; A[r, ii[o[1]]] = 1
        th = prior_m.copy()
        P = np.diag(1 / prior_v)
        for _ in range(30):
            q = s(self.P["mu"] + A @ th)
            grad = A.T @ (y - q) - P @ (th - prior_m)
            H = A.T @ (A * (q * (1 - q))[:, None]) + P
            step = np.linalg.solve(H, grad)
            th += step
            if np.max(np.abs(step)) < 1e-6:
                break
        q = s(self.P["mu"] + A @ th)
        H = A.T @ (A * (q * (1 - q))[:, None]) + P
        return th, np.linalg.inv(H), si, ii

    def __call__(self, inp, labeled):
        subj, item = inp
        bench = item["benchmark_id"]
        key = (id(labeled), len(labeled), bench)
        if key not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[key] = self.posterior(labeled, bench)
        th, cov, si, ii = self.cache[key]
        sk = tuple(sorted(subj.items())); ik = item.get("item_content", "")
        m, a = self.P["mu"] + th[0], np.zeros(len(th)); a[0] = 1
        if sk in si:
            m += th[si[sk]]; a[si[sk]] = 1
            v = 0.0
        else:
            m += self.g(subj); v = self.sC ** 2
        if ik in ii:
            m += th[ii[ik]]; a[ii[ik]] = 1
        else:
            m += self.dprior(item); v += self.sD ** 2
        v += float(a @ cov @ a)
        return float(s(m / math.sqrt(1 + math.pi * v / 8)))


def params_path():
    return os.environ.get("IRT_PARAMS", DEFAULT_PARAMS)


def full():
    return IRT(json.load(open(params_path())))


def no_names():
    return IRT(json.load(open(params_path())), use_names=False)


def no_subject():
    return IRT(json.load(open(params_path())), use_names=False, use_subject=False)


def no_items():
    return IRT(json.load(open(params_path())), use_items=False)


def main():
    out, exams = sys.argv[2], sys.argv[3].split(",")
    exclude = set()
    for d in exams:
        exclude |= {p["benchmark_name"] for p in load_exam(d)["pairs"]}
    params = fit(exclude)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(params, open(out, "w"), indent=1)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items() if k not in ("name_effect", "excluded")},
          "| names with an effect:", len(params["name_effect"]))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        main()
