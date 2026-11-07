"""Why Opus judges tasks better: (1) which benchmarks carry the benchmark-level lead, (2) whether its item ranking is
explained by surface cues.

Calibration set, items every forecaster answered, own a, b per forecaster (forecaster_constants.py).
(1) Benchmarks with at least two pairs. Benchmark observed rate = mean over its pairs of the pair success rate over all
responses; benchmark prediction = mean over its pairs of the mean recalibrated forecast. Each benchmark's share of the
benchmark part of the Brier gap (gap_levels.py) and catalog fields (domain, turns, response type, release date).
(2) All pairs. Within-pair AUC (item success rate > 0.5 as positive) of each forecast; of surface cues alone (log length,
log line count, digit share, answer-option marker, difficulty tag in item_features), fitted on pair-demeaned values
across benchmarks with 5 benchmark folds; of each forecast after removing its pooled within-pair linear dependence on
the cues (residual); and of each open forecast combined with the cues (5 benchmark folds). 2,000 bootstrap draws over
benchmarks for Opus minus each open forecaster.
"""
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, "./report/figures")
sys.path.insert(0, "./practice")
from forecaster_constants import CALIB_RUNS, ROOT, logit
from common import load_exam
from exemplar_shift import load
import memory_generalization as mg

CONST = {"Opus 5.5, low": (-0.11, 1.01), "GLM-5.3 Flash, low": (0.00, 0.63), "DeepSeek V4.1 Flash, low": (-0.26, 0.62)}
NAMES = list(CONST); OPUS, OPEN = NAMES[0], NAMES[1:]
sig = lambda z: 1 / (1 + np.exp(-z))
exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
preds = {n: load(f"{ROOT}/runs/{r}") for n, r in CALIB_RUNS.items()}
cat = pd.read_parquet(mg.CATALOG).set_index("folder")
rec = pd.read_csv(f"{ROOT}/runs/know-recognized-20261030.csv")
known = {n: set(rec[(rec.run == n) & (rec.match == True)].benchmark) for n in rec.run.unique()}

pairs = []  # (benchmark, pair index, targets)
for pi, p in enumerate(exam["pairs"]):
    ts = [t for t in p["targets"] if all((pi, t["key"]) in pj for pj in preds.values())]
    if len(ts) >= 2:
        pairs.append((p["benchmark_name"], pi, ts))


def wmean(p, L):
    n = np.array([len(l) for l in L], float)
    return (p * n).sum() / n.sum()


def brier(p, L):
    return sum(((p[j] - np.asarray(l)) ** 2).sum() for j, l in enumerate(L)) / sum(len(l) for l in L)


# ---------- (1) benchmark level ----------
bys = {}
for b, pi, ts in pairs:
    bys.setdefault(b, []).append((pi, ts))
multi = {b: v for b, v in bys.items() if len(v) >= 2}
npairs = sum(len(v) for v in multi.values())
rows = []
for b, prs in multi.items():
    obs = np.mean([exam["pairs"][pi]["pair_success_rate"] for pi, _ in prs])
    r = {"benchmark": b, "pairs": len(prs), "observed": obs}
    for n in NAMES:
        a0, b0 = CONST[n]
        zs = [a0 + b0 * np.array([float(logit(preds[n][(pi, t["key"])])) for t in ts]) for pi, ts in prs]
        labs = [[t["labels"] for t in ts] for _, ts in prs]
        o = [wmean(np.array([np.mean(l) for l in L]), L) for L in labs]
        cB = brentq(lambda c: np.mean([wmean(sig(z + c), L) for z, L in zip(zs, labs)]) - np.clip(np.mean(o), 1e-4, 1 - 1e-4), -30, 30)
        gain = np.mean([brier(sig(z), L) - brier(sig(z + cB), L) for z, L in zip(zs, labs)])
        r[n] = np.mean([sig(z).mean() for z in zs])
        r[f"gain {n}"] = gain * len(prs) / npairs   # share of the mean over pairs
    for col in ("domain", "multi_single_turn", "response_type", "release_date", "name"):
        r[col] = cat[col].get(b) if b in cat.index else None
    r["Opus recognizes"] = b in known.get("know-opus-low", set())
    rows.append(r)
B = pd.DataFrame(rows)
B["gap"] = np.mean([B[f"gain {n}"] - B[f"gain {OPUS}"] for n in OPEN], axis=0)
for n in NAMES:
    B[f"err {n}"] = logit(B[n].values) - logit(B["observed"].values)
B = B.sort_values("gap", ascending=False)
tot = B.gap.sum()
print(f"(1) {len(B)} benchmarks, {npairs} pairs. Benchmark part of the gap (mean of the two open models): {tot:+.4f}")
cs = B.gap.cumsum().values
print(f"    share from the top 5 / 10 / 20 benchmarks: {cs[4] / tot:.0%} / {cs[9] / tot:.0%} / {cs[19] / tot:.0%}; benchmarks where Opus is worse: {(B.gap < 0).sum()}")
print("    mean |logit error| of the benchmark prediction: " + ", ".join(f"{n.split(',')[0]} {B[f'err {n}'].abs().mean():.2f}" for n in NAMES))
lo = B.observed < B.observed.quantile(1 / 3); hi = B.observed > B.observed.quantile(2 / 3)
print("    mean signed logit error (prediction minus observed) on the hardest third / easiest third of benchmarks:")
for n in NAMES:
    print(f"      {n:26} {B.loc[lo, f'err {n}'].mean():+.2f} / {B.loc[hi, f'err {n}'].mean():+.2f}")
print("\n    top 15 benchmarks by the open models' extra benchmark-level loss")
print(f"    {'benchmark':34} {'domain':28} {'turns':6} {'response':10} {'released':8} {'rec':3} {'obs':>5} {'Opus':>5} {'GLM':>5} {'DS':>5} {'share':>6}")
for _, r in B.head(15).iterrows():
    dom = ",".join(r.domain) if r.domain is not None and len(r.domain) else ""
    print(f"    {str(r['name'])[:34]:34} {dom[:28]:28} {str(r.multi_single_turn)[:6]:6} {str(r.response_type)[:10]:10} {str(r.release_date)[:7]:8} {'yes' if r['Opus recognizes'] else 'no':3} "
          f"{r.observed:5.2f} {r[OPUS]:5.2f} {r[OPEN[0]]:5.2f} {r[OPEN[1]]:5.2f} {r.gap / tot:6.0%}")
print("\n    by catalog field (groups of at least 8 benchmarks): benchmarks, share of the gap, mean |logit error| Opus / GLM / DeepSeek")
E = B.explode("domain")
for field, frame in (("domain", E), ("multi_single_turn", B), ("response_type", B)):
    for v, g in frame.groupby(field):
        if len(g) >= 8:
            print(f"      {field}={v:24} {len(g):3}  {g.gap.sum() / tot:+5.0%}  " + " / ".join(f"{g[f'err {n}'].abs().mean():.2f}" for n in NAMES))

# ---------- (2) item ranking: surface cues ----------
DIFF = {"easy": -1, "simple": -1, "low": -1, "medium": 0, "moderate": 0, "mid": 0, "hard": 1, "challenging": 1, "difficult": 1,
        "high": 1, "expert": 2, "extra": 2, "very_hard": 2}


def cues(item):
    s = item.get("item_content", "") or ""
    f = str(item.get("item_features", "") or "")
    m = re.search(r"(?:difficulty|level|hardness)\s*[=:]\s*([A-Za-z_]+|\d+)", f, re.I)
    d = 0.0
    if m:
        v = m.group(1).lower()
        d = float(v) if v.isdigit() else float(DIFF.get(v, 0))
    opts = len(re.findall(r"(?:^|\n)\s*\(?[A-Ea-e][\).:]\s", s))
    return [np.log1p(len(s)), np.log1p(s.count("\n")), sum(ch.isdigit() for ch in s) / max(len(s), 1), float(opts >= 2), d, float(bool(m))]


rowsI = []
for b, pi, ts in pairs:
    for t in ts:
        rowsI.append([b, pi, np.mean(t["labels"])] + cues(t["item"]) + [float(logit(preds[n][(pi, t["key"])])) for n in NAMES])
cols = ["b", "pi", "y", "len", "lines", "digits", "options", "difftag", "has_tag"] + NAMES
I = pd.DataFrame(rowsI, columns=cols)
X = ["len", "lines", "digits", "options", "difftag"]
dm = lambda c: I[c] - I.groupby("pi")[c].transform("mean")
D = pd.DataFrame({c: dm(c) for c in X + NAMES})
D["y"] = I.y; D["pi"] = I.pi; D["b"] = I.b
print(f"\n(2) {I.pi.nunique()} pairs, {len(I)} items; items with a difficulty tag in item_features: {int(I.has_tag.sum())} in {I[I.has_tag > 0].b.nunique()} benchmarks")


def pair_auc(score, frame):
    out = {}
    for pi, g in frame.assign(s=score).groupby("pi"):
        pos, neg = g.s[g.y > .5].values, g.s[g.y <= .5].values
        if len(pos) and len(neg):
            out[pi] = np.mean([(a > c) + .5 * (a == c) for a in pos for c in neg])
    return pd.Series(out)


def ols(Xm, y):
    return np.linalg.lstsq(np.c_[Xm], y, rcond=None)[0]


benches = np.array(sorted(D.b.unique())); rng = np.random.default_rng(0)
fold = {b: i % 5 for i, b in enumerate(rng.permutation(benches))}
D["fold"] = D.b.map(fold)


def cv_score(feats):
    s = pd.Series(0.0, index=D.index)
    for k in range(5):
        tr, te = D.fold != k, D.fold == k
        w = ols(D.loc[tr, feats].values, D.loc[tr, "y"].values - D.loc[tr, "y"].groupby(D.loc[tr, "pi"]).transform("mean").values)
        s[te] = D.loc[te, feats].values @ w
    return s


scores = {f"{n.split(',')[0]} forecast": D[n] for n in NAMES}
scores["surface cues alone (cross-benchmark fit)"] = cv_score(X)
for n in NAMES:
    w = ols(D[X].values, D[n].values)
    scores[f"{n.split(',')[0]} after removing cues"] = D[n] - D[X].values @ w
for n in OPEN:
    scores[f"{n.split(',')[0]} + cues (cross-benchmark fit)"] = cv_score([n] + X)
A = {k: pair_auc(v, D) for k, v in scores.items()}
pb = D.groupby("pi").b.first()
print("    within-pair AUC (mean over pairs with both outcomes)")
for k, a in A.items():
    print(f"      {k:44} {a.mean():.3f}  ({len(a)} pairs)")
print("    within-pair correlation of each forecast with the cues (pooled, pair-demeaned): " + "; ".join(
    f"{n.split(',')[0]}: " + ", ".join(f"{x} {np.corrcoef(D[n], D[x])[0, 1]:+.2f}" for x in X) for n in NAMES))
print("    correlation of the outcome with the cues: " + ", ".join(f"{x} {np.corrcoef(D.y, D[x])[0, 1]:+.2f}" for x in X))


def diff_ci(ka, kb):
    a, c = A[ka], A[kb]; common = a.index.intersection(c.index)
    d = (a[common] - c[common]); g = pb[common]
    bl = g.unique(); est = d.mean(); bs = []
    idx = {x: d[g == x].values for x in bl}
    for _ in range(2000):
        pick = rng.choice(bl, len(bl)); bs.append(np.concatenate([idx[x] for x in pick]).mean())
    lo, hi = np.quantile(bs, [.025, .975])
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"


print("    Opus minus open, within-pair AUC [95% CI over benchmarks]")
for n in OPEN:
    s = n.split(",")[0]
    print(f"      {s:22} forecast {diff_ci('Opus 5.5 forecast', f'{s} forecast')} | after removing cues {diff_ci('Opus 5.5 after removing cues', f'{s} after removing cues')} | open + cues {diff_ci('Opus 5.5 forecast', f'{s} + cues (cross-benchmark fit)')}")
