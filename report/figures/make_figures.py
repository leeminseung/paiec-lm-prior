"""Key figures for the technical report, drawn from the saved judgments and proxy evaluations (no model calls).

fig_learning_curve   Brier by label budget on the three held-out proxies (27 pairs): the organizers'
                     Empirical Mean, the Beta-binomial baseline, the Opus forecast without update, and the hierarchical
                     model with the forecast as offset; 95% band for the hierarchical model (benchmark bootstrap).
fig_memory           Budget-0 skill (1 - Brier / base-rate Brier) of ten judges on the 847-item set, on the 75
                     benchmarks Opus recognizes by name versus the 46 it does not.
fig_shared_errors    Each measurement-db pair's judge error against the mean error of the other pairs on the same
                     benchmark; colored groups are benchmarks the judge itself cited when proposing scoring axes.

Usage: python -I make_figures.py <out_dir>
"""
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRACTICE = "./practice"
sys.path.insert(0, PRACTICE)
from common import BUDGETS, load_exam  # noqa: E402
from eval_prior_combo import A0, ALPHA, M, counts  # noqa: E402
from exemplar_shift import load  # noqa: E402
import memory_generalization as mg  # noqa: E402
from hier_llm import predict_all  # noqa: E402
import propose_axes as pa  # noqa: E402

ROOT = "."
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
logit = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
sig = lambda z: 1 / (1 + np.exp(-z))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------- Figure: learning curves on the untouched proxies ----------
METHODS = ["Empirical Mean (organizers)", "Beta-binomial baseline", "Opus forecast, no update", "Hierarchical model (ours)"]


def curves(exam_dir, t_run, e_run):
    exam = load_exam(exam_dir); pt, pe = load(t_run), load(e_run); cnt = counts(exam)
    zt = {(pi, t["key"]): float(logit(pt.get((pi, t["key"]), M))) for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}
    ze = {(pi, a["key"]): float(logit(pe.get((pi, a["key"]), M))) for pi, p in enumerate(exam["pairs"]) for a in p["acquisitions"]}
    hier = predict_all(exam, zt, ze, 1.09, 0.46)
    out, benches = {m: [] for m in METHODS}, []
    for pi, p in enumerate(exam["pairs"]):
        per = {m: [] for m in METHODS}
        for b in BUDGETS:
            x, k, s, n = cnt[(b, pi)]
            se = {m: 0.0 for m in METHODS}; nn = 0
            for t in p["targets"]:
                p0 = pt.get((pi, t["key"]), M)
                em = x / k if k else 0.5
                r1 = (x + ALPHA * (s + A0 * M) / (n + A0)) / (k + ALPHA)
                h = hier[(b, pi, t["key"])]
                for m, v in zip(METHODS, (em, r1, p0, h)):
                    se[m] += sum((v - y) ** 2 for y in t["labels"])
                nn += len(t["labels"])
            for m in METHODS:
                per[m].append(se[m] / nn)
        for m in METHODS:
            out[m].append(per[m])
        benches.append(exam_dir + p["benchmark_name"])
    return {m: np.array(v) for m, v in out.items()}, benches


def fig_learning_curve(out_dir):
    allc, benches = {m: [] for m in METHODS}, []
    for k in (1, 2, 3):
        c, b = curves(f"{ROOT}/exams/proxy-1028-{k}", f"{ROOT}/runs/p1028-{k}-sub-opus-low",
                      f"{ROOT}/runs/p1028-{k}-sub-exemplar-opus-low")
        for m in METHODS:
            allc[m].append(c[m])
        benches += b
    allc = {m: np.vstack(v) for m, v in allc.items()}
    keys = sorted(set(benches)); idx = {kk: [i for i, c in enumerate(benches) if c == kk] for kk in keys}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(2000)]
    x = np.arange(len(BUDGETS))
    style = {"Empirical Mean (organizers)": dict(color=MUTED, ls=(0, (1, 1.5))),
             "Beta-binomial baseline": dict(color=INK2, ls="-"),
             "Opus forecast, no update": dict(color=ORANGE, ls="-"),
             "Hierarchical model (ours)": dict(color=BLUE, ls="-")}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.5, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    for a in (ax, bx):
        grid(a)
    top = 0.272
    ends = {}
    for m in METHODS:
        y = allc[m].mean(0)
        lw = 2.2 if m.endswith("(ours)") else 1.4
        ax.plot(x, np.minimum(y, top), lw=lw, marker="o", ms=3.2, mec="white", mew=0.6, clip_on=False, **style[m])
        ends[m] = y[-1]
        if (y > top).any():
            i = int(np.argmax(y))
            ax.annotate(f"{y[i]:.2f}", (x[i], top), xytext=(7, -9), textcoords="offset points", fontsize=6.5, color=MUTED)
    placed = []
    for m in sorted(METHODS, key=lambda m: ends[m]):
        yy = ends[m]
        while any(abs(yy - p) < 0.0085 for p in placed):
            yy += 0.0085
        placed.append(yy)
        ax.annotate(m.replace(" (organizers)", "\n(organizers)") if False else m, (x[-1], ends[m]), xytext=(x[-1] + 0.15, yy),
                    fontsize=6.8, color=INK, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5) if abs(yy - ends[m]) > 0.003 else None)
    ax.set_ylim(0.15, top)
    ax.set_ylabel("Brier score\n(lower is better)")
    for xi, w in zip(x, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)):
        ax.text(xi, 1.0, f"ALC weight {w}" if xi == 0 else f"{w}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6, color=MUTED)
    ax.set_title("(a) Brier score by label budget, three held-out proxies (27 pairs)", fontsize=7.5,
                 color=INK, loc="left", pad=12)
    # (b) differences from the Beta-binomial baseline, paired by pair, benchmark bootstrap
    ref = allc["Beta-binomial baseline"]
    bx.axhline(0, color=INK2, lw=0.8)
    for m in ("Opus forecast, no update", "Hierarchical model (ours)"):
        d = allc[m] - ref
        bs = np.array([d[sel].mean(0) for sel in draws])
        lo, hi = np.quantile(bs, [.025, .975], axis=0)
        if m.endswith("(ours)"):
            bx.fill_between(x, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
        lw = 2.2 if m.endswith("(ours)") else 1.4
        bx.plot(x, d.mean(0), lw=lw, marker="o", ms=3.2, mec="white", mew=0.6, **style[m])
    bx.text(x[-1] + 0.15, 0, "Beta-binomial baseline", fontsize=6.8, color=INK, va="center")
    bx.set_ylabel("Difference from the\nBeta-binomial baseline")
    bx.set_title("(b) Paired difference from the Beta-binomial baseline, with 95% benchmark-bootstrap bands", fontsize=7.5,
                 color=INK, loc="left")
    bx.set_xticks(x, [str(b) for b in BUDGETS])
    bx.set_xlabel("Revealed outcomes per pair (label budget, log scale)")
    bx.set_xlim(-0.2, len(BUDGETS) - 1 + 0.15)
    fig.subplots_adjust(right=0.70, hspace=0.35)
    save(fig, out_dir, "fig_learning_curve")
    alc = {m: float((allc[m] @ np.array([.1, .2, .2, .2, .2, .1])).mean()) for m in METHODS}
    return {m: [round(float(v), 4) for v in allc[m].mean(0)] for m in METHODS}, alc


# ---------- Figure: recognition vs accuracy ----------
JUDGE_LABEL = {"Opus high, thinking": "Opus 5.5 (high, thinking)", "Opus low": "Opus 5.5",
               "Opus low, analyse only": "Opus 5.5 (analyse only)", "Sonnet low": "Sonnet 5.5",
               "Haiku low": "Haiku 5.5", "GLM-5.3 Flash": "GLM-5.3 Flash (default effort)", "Muse Spark Open": "Muse Spark Open",
               "DeepSeek V4.1 Flash low": "DeepSeek V4.1 Flash", "MiniMax M3.1 Flash": "MiniMax M3.1 Flash",
               "MiniMax M3": "MiniMax M3", "GPT-6.1 Sol low": "GPT-6.1 Sol", "GPT-6.1 Terra low": "GPT-6.1 Terra",
               "GPT-6 Luna low": "GPT-6 Luna"}


def fig_memory(out_dir):
    exam = load_exam(mg.SUBSET)
    import pandas as pd
    cat = pd.read_parquet(mg.CATALOG)
    names = {f: [n, bid] for f, n, bid in zip(cat.folder, cat.name, cat.benchmark_id)}
    grades = {(r["run"], r["benchmark"]): r["match"] for r in map(json.loads, open(mg.GRADES)) if r.get("match")}
    opus, _ = mg.recognition("know-opus-low", names, grades)
    rows = []
    for jn, (run, kind, _) in mg.JUDGES.items():
        pred = mg.judgments(run, kind, exam)
        st = mg.pair_stats(exam, pred, lambda pi, t: True)
        d = mg.split_diff(st, opus)
        rows.append((JUDGE_LABEL[jn], mg.summarize(st)["skill"], d[0]["skill"], d[1]["skill"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    grid(ax, "x")
    y = np.arange(len(rows))
    for yi, (lab, overall, yes, no) in zip(y, rows):
        ax.plot([yes, no], [yi, yi], color=GRID, lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter([no], [yi], s=26, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([yes], [yi], s=26, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Brier skill score at budget 0 (reference: base rate 0.627; higher is better)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, label="46 benchmarks Opus does not recognize"),
                       plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="75 benchmarks Opus recognizes by name"),],
              loc="lower right", frameon=False, fontsize=7)
    save(fig, out_dir, "fig_memory")
    return rows


# ---------- Figure: benchmark-shared judge errors ----------
GROUPS = {"strict answer checking\n(SQL, formal proofs, exact output)": ["B077", "B065", "B031", "B027", "B056"],
          "lenient grading of long deliverables": ["B066", "B024", "B054", "B042", "B067"],
          "knowledge the subject lacks\n(post-cutoff events, local facts)": ["B048", "B006", "B032", "B085", "B059"]}


def fig_shared_errors(out_dir):
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027"); low = load(f"{ROOT}/runs/mdb-judge-opus-low")
    # anonymous ids exactly as shown to the judge in propose_axes.py, seed 1
    exam["_dir"] = f"{ROOT}/exams/mdb-judge-1027"
    proxy = set()
    for d in pa.PROXIES:
        proxy |= {p["benchmark_name"] for p in load_exam(f"{ROOT}/exams/{d}")["pairs"]}
    by = {}
    for pi, p in enumerate(exam["pairs"]):
        ts = [t for t in p["targets"] if (pi, t["key"]) in low]
        if ts:
            by.setdefault(p["benchmark_name"], []).append(np.mean([low[(pi, t["key"])] for t in ts]) - p["pair_success_rate"])
    shown = sorted(b for b in by if b not in proxy)
    random.Random(1).shuffle(shown)
    anon = {f"B{i + 1:03d}": b for i, b in enumerate(shown)}
    group_of = {anon[a]: g for g, ids in GROUPS.items() for a in ids if a in anon}
    xs, ys, gs = [], [], []
    for b, errs in by.items():
        if len(errs) < 2:
            continue
        for i, e in enumerate(errs):
            xs.append(np.mean(errs[:i] + errs[i + 1:])); ys.append(e); gs.append(group_of.get(b))
    xs, ys = np.array(xs), np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    grid(ax, "both")
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    lim = 0.85
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    other = np.array([g is None for g in gs])
    ax.scatter(xs[other], ys[other], s=10, color="#c9c8c3", edgecolor="white", linewidth=0.4, zorder=2)
    for g, c in zip(GROUPS, (BLUE, ORANGE, AQUA)):
        m = np.array([gg == g for gg in gs])
        ax.scatter(xs[m], ys[m], s=22, color=c, edgecolor="white", linewidth=0.8, zorder=3, label=g)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Forecast error on the other pairs of the same benchmark")
    ax.set_ylabel("Forecast error on this pair\n(mean forecast − success rate)")
    ax.text(0.02, 0.97, f"r = {r:.2f}, {len(xs)} pairs on {sum(len(e) >= 2 for e in by.values())} benchmarks",
            transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=0.9, fontsize=6.3,
              title="benchmarks the forecaster cited for", title_fontsize=6.5, alignment="left")
    save(fig, out_dir, "fig_shared_errors")
    return r, len(xs)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    lc, alc = fig_learning_curve(out_dir)
    print("learning curve means:", json.dumps(lc, indent=0)); print("ALC:", alc)
    for row in fig_memory(out_dir):
        print("memory:", row)
    print("shared errors r, n:", fig_shared_errors(out_dir))


if __name__ == "__main__":
    main()
