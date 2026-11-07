"""Does Opus's item-ranking lead sit on items it can quote? (cloze runs 11/1; CLOZE_DIR picks the run: the 28-pair trial or the full calibration set)

Per item: each forecaster's forecast recalibrated with its own a, b, then shifted within the pair so the pair mean
prediction equals the observed rate on these items (the ranking part of gap_levels.py); squared error per response.
Items split by whether Opus filled the cloze span exactly. Also pairwise ordering: item pairs in the same pair with
different outcomes, ordered correctly or not, split by whether the item pair includes an Opus-quoted item.
Intervals: 4,000 bootstrap draws over pairs.
"""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, "./report/figures"); sys.path.insert(0, "./practice")
from forecaster_constants import CALIB_RUNS, ROOT, logit
from common import load_exam
from exemplar_shift import load

C = os.environ.get("CLOZE_DIR", f"{ROOT}/runs/memprobe-cloze-1101")
CONST = {"Opus 5.5, low": (-0.11, 1.01), "GLM-5.3 Flash, low": (0.00, 0.63), "DeepSeek V4.1 Flash, low": (-0.26, 0.62)}
OPUS = "Opus 5.5, low"; OPEN = [n for n in CONST if n != OPUS]
sig = lambda z: 1 / (1 + np.exp(-z))
exam = load_exam(f"{ROOT}/exams/mdb-judge-1027")
preds = {n: load(f"{ROOT}/runs/{r}") for n, r in CALIB_RUNS.items()}
quoted = {}
for m, n in (("opus", OPUS), ("glm", OPEN[0]), ("ds", OPEN[1])):
    for r in map(json.loads, open(f"{C}/cloze_{m}.jsonl")):
        quoted.setdefault(n, {})[(r["pair"], r["key"])] = bool((r.get("parsed") or {}).get("exact"))
items = [tuple(l.split()) for l in open(f"{C}/items.txt") if l.strip()]
pairs = {}
for pi, k in items:
    pairs.setdefault(int(pi), []).append(k)

rows = []   # (pair, key, opus quoted, {forecaster: ranking sq err}, y, {forecaster: shifted logit})
for pi, keys in pairs.items():
    ts = [t for t in exam["pairs"][pi]["targets"] if t["key"] in keys and all((pi, t["key"]) in preds[f] for f in CONST)]
    if len(ts) < 2:
        continue
    labs = [np.asarray(t["labels"], float) for t in ts]
    n = np.array([len(l) for l in labs], float); obs = sum(l.sum() for l in labs) / n.sum()
    per = {}
    for f, (a, b) in CONST.items():
        z = a + b * np.array([float(logit(preds[f][(pi, t["key"])])) for t in ts])
        c = brentq(lambda c: (sig(z + c) * n).sum() / n.sum() - np.clip(obs, 1e-4, 1 - 1e-4), -30, 30)
        per[f] = z + c
    for j, t in enumerate(ts):
        q = quoted[OPUS].get((pi, t["key"]))
        rows.append((pi, t["key"], q, {f: float(((sig(per[f][j]) - labs[j]) ** 2).mean()) for f in CONST}, labs[j].mean(),
                     {f: per[f][j] for f in CONST}))

pairs_list = sorted({r[0] for r in rows})
rng = np.random.default_rng(0)
draws = [rng.choice(pairs_list, len(pairs_list)) for _ in range(4000)]


def stat(sel_pairs, fn):
    idx = {p: [r for r in rows if r[0] == p] for p in pairs_list}
    pick = [r for p in sel_pairs for r in idx[p]]
    return fn(pick)


def gap(pick, f, flag):
    v = [r[3][f] - r[3][OPUS] for r in pick if r[2] is flag]
    return np.mean(v) if v else np.nan


print(f"{len(pairs_list)} pairs, {len(rows)} items; Opus quoted {sum(1 for r in rows if r[2])}, not quoted {sum(1 for r in rows if r[2] is False)}, no span {sum(1 for r in rows if r[2] is None)}")
for f in OPEN:
    est_q, est_n = gap(rows, f, True), gap(rows, f, False)
    bs = np.array([[stat(d, lambda p: gap(p, f, True)), stat(d, lambda p: gap(p, f, False))] for d in draws])
    did = bs[:, 0] - bs[:, 1]
    lo, hi = np.nanquantile(did, [.025, .975])
    print(f"{f.split(',')[0]:22} ranking error, open minus Opus: quoted items {est_q:+.4f} | not quoted {est_n:+.4f} | difference {est_q - est_n:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def concord(pick, f, want):
    hit = tot = 0
    byp = {}
    for r in pick:
        byp.setdefault(r[0], []).append(r)
    for p, rs in byp.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if (a[4] > .5) == (b[4] > .5) or a[2] is None or b[2] is None:
                    continue
                if (a[2] or b[2]) != want:
                    continue
                s = (a[5][f] - b[5][f]) * (a[4] - b[4])
                hit += (s > 0) + .5 * (s == 0); tot += 1
    return hit / tot if tot else np.nan, tot


for want in (True, False):
    line = f"item pairs {'with' if want else 'without'} an Opus-quoted item: "
    for f in CONST:
        c, t = concord(rows, f, want); line += f"{f.split(',')[0]} {c:.3f}  "
    print(line + f"(n={concord(rows, OPUS, want)[1]})")
