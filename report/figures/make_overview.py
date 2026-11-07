"""One full-width results figure (2 x 2), and the same panels split into the report's Figures 2 and 3, built from the same data as make_figures.py.

(a) Brier by label budget on the three held-out proxies, legend inside the panel; (b) paired difference from the
Beta-binomial baseline with a 95% benchmark-bootstrap band for the hierarchical model; (c) budget-0 skill of thirteen
forecasters on benchmarks Opus recognizes or not; (d) benchmark-shared forecast errors on the calibration set, colored by
the groups the forecaster cited when naming candidate error factors.

Usage: python -I make_overview.py <out_dir>
"""
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402
from make_figures import (AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, METHODS, ROOT, curves, grid, load, load_exam,  # noqa: E402
                          mg, pa, save, JUDGE_LABEL, GROUPS)
from common import BUDGETS  # noqa: E402

SHORT = {"Empirical Mean (organizers)": "Empirical Mean", "Beta-binomial baseline": "Beta-binomial baseline",
         "Opus forecast, no update": "Forecast, no update", "Hierarchical model (ours)": "Ours"}
STYLE = {"Empirical Mean (organizers)": dict(color=MUTED, ls=(0, (1, 1.5))), "Beta-binomial baseline": dict(color=INK2, ls="-"),
         "Opus forecast, no update": dict(color=ORANGE, ls="-"),
         "Hierarchical model (ours)": dict(color=BLUE, ls="-")}


def panel_curves(ax, bx=None, methods=METHODS, top=0.30):
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
    W = np.array([.1, .2, .2, .2, .2, .1])
    for m in methods:
        y = allc[m].mean(0)
        ax.plot(x, np.minimum(y, top), lw=2.0 if m.endswith("(ours)") else 1.2, marker="o", ms=2.6, mec="white",
                mew=0.5, clip_on=False, label=SHORT[m], **STYLE[m])
        if (y > top).any():
            i = int(np.argmax(y))
            ax.annotate(f"{y[i]:.2f}", (x[i], top), xytext=(-16, 1), textcoords="offset points", fontsize=6, color=MUTED)
    ax.set_ylim(0.15, top); grid(ax)
    ax.set_xticks(x, [str(b) for b in BUDGETS]); ax.set_xlim(-0.2, 5.2)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_xlabel("Label budget (revealed outcomes per pair)")
    ax.set_title("(a) Three held-out proxies, 27 pairs", fontsize=7.2, loc="left", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=5.6, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    if bx is None:
        return
    ref = allc["Beta-binomial baseline"]
    bx.axhline(0, color=INK2, lw=0.7); grid(bx)
    for m in ("Opus forecast, no update", "Hierarchical model (ours)"):
        d = allc[m] - ref
        if m.endswith("(ours)"):
            lo, hi = np.quantile(np.array([d[s].mean(0) for s in draws]), [.025, .975], axis=0)
            bx.fill_between(x, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
        bx.plot(x, d.mean(0), lw=2.0 if m.endswith("(ours)") else 1.2, marker="o", ms=2.6, mec="white", mew=0.5,
                label=SHORT[m], **STYLE[m])
    bx.set_xticks(x, [str(b) for b in BUDGETS]); bx.set_xlim(-0.2, 5.2)
    bx.set_ylabel("Brier minus Beta-binomial baseline")
    bx.set_xlabel("Label budget (revealed outcomes per pair)")
    bx.set_title("(b) Paired difference from the Beta-binomial baseline (95% band)", fontsize=7.2, loc="left", color=INK)


def level_errors(exam_dir, t_run, e_run):
    """Per pair and budget: mean prediction over the pair's scored responses minus its success rate, for the
    Beta-binomial baseline and our method."""
    exam = load_exam(exam_dir); pt, pe = load(t_run), load(e_run); cnt = mf.counts(exam)
    zt = {(pi, t["key"]): float(mf.logit(pt.get((pi, t["key"]), mf.M))) for pi, p in enumerate(exam["pairs"]) for t in p["targets"]}
    ze = {(pi, a["key"]): float(mf.logit(pe.get((pi, a["key"]), mf.M))) for pi, p in enumerate(exam["pairs"]) for a in p["acquisitions"]}
    hier = mf.predict_all(exam, zt, ze, 1.09, 0.46)
    out = {"base": [], "ours": []}
    for pi, p in enumerate(exam["pairs"]):
        ys = np.concatenate([t["labels"] for t in p["targets"]]); rate = ys.mean()
        rb, ro = [], []
        for b in BUDGETS:
            x, k, s, n = cnt[(b, pi)]
            rb.append((x + mf.ALPHA * (s + mf.A0 * mf.M) / (n + mf.A0)) / (k + mf.ALPHA) - rate)
            ro.append(np.concatenate([[hier[(b, pi, t["key"])]] * len(t["labels"]) for t in p["targets"]]).mean() - rate)
        out["base"].append(rb); out["ours"].append(ro)
    return {k: np.array(v) for k, v in out.items()}


def panel_level(bx):
    errs = {"base": [], "ours": []}
    for k in (1, 2, 3):
        e = level_errors(f"{ROOT}/exams/proxy-1028-{k}", f"{ROOT}/runs/p1028-{k}-sub-opus-low",
                         f"{ROOT}/runs/p1028-{k}-sub-exemplar-opus-low")
        for m in errs:
            errs[m].append(e[m])
    errs = {m: np.vstack(v) for m, v in errs.items()}
    rng = np.random.default_rng(0)
    x = np.arange(len(BUDGETS))
    for m, dx, col, lab in (("base", -0.17, "#9a9893", "Beta-binomial baseline"), ("ours", 0.17, BLUE, "Ours")):
        for i in range(len(BUDGETS)):
            v = errs[m][:, i]
            q1, q3 = np.percentile(v, [25, 75])
            bx.plot([i + dx] * 2, [q1, q3], color=col, lw=2.4, alpha=0.30, solid_capstyle="round", zorder=2)
            bx.scatter(i + dx + rng.uniform(-0.06, 0.06, len(v)), v, s=4.5, color=col, alpha=0.8, edgecolor="none",
                       label=lab if i == 0 else None, zorder=3)
        print(f"level error {m}: mean |error| by budget " + " ".join(f"{np.mean(np.abs(errs[m][:, i])):.3f}" for i in range(len(BUDGETS))))
    bx.axhline(0, color=INK2, lw=0.6); grid(bx)
    bx.set_xticks(x, [str(b) for b in BUDGETS]); bx.set_xlim(-0.5, 5.5)
    bx.set_ylabel("Mean prediction minus success rate")
    bx.set_xlabel("Label budget (revealed outcomes per pair)")
    bx.legend(loc="upper right", frameon=False, fontsize=5.6, markerscale=1.6, handletextpad=0.2, borderaxespad=0.2)
    bx.set_title("(b) Level error of each held-out pair", fontsize=7.2, loc="left", color=INK)


def panel_memory(ax):
    import pandas as pd
    exam = load_exam(mg.SUBSET)
    cat = pd.read_parquet(mg.CATALOG)
    names = {f: [n, bid] for f, n, bid in zip(cat.folder, cat.name, cat.benchmark_id)}
    grades = {(r["run"], r["benchmark"]): r["match"] for r in map(json.loads, open(mg.GRADES)) if r.get("match")}
    opus, _ = mg.recognition("know-opus-low", names, grades)
    rows = []
    for jn, (run, kind, _) in mg.JUDGES.items():
        st = mg.pair_stats(exam, mg.judgments(run, kind, exam), lambda pi, t: True)
        d = mg.split_diff(st, opus)
        rows.append((JUDGE_LABEL[jn], mg.summarize(st)["skill"], d[0]["skill"], d[1]["skill"]))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows)); grid(ax, "x")
    for yi, (_, _, yes, no) in zip(y, rows):
        ax.plot([yes, no], [yi, yi], color=GRID, lw=2.0, solid_capstyle="round", zorder=1)
        ax.scatter([no], [yi], s=16, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter([yes], [yi], s=16, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6)
    ax.set_xlabel("Brier skill score at budget 0")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3.5, color=BLUE, label="Opus does not recognize (46)"),
                       plt.Line2D([], [], marker="o", ls="", ms=3.5, color=ORANGE, label="Opus recognizes (75)")],
              loc="upper left", frameon=False, fontsize=5.6, handletextpad=0.3)
    ax.set_title("(c) Skill on benchmarks Opus recognizes or not", fontsize=7.2, loc="left", color=INK)


def panel_errors(ax):
    exam = load_exam(f"{ROOT}/exams/mdb-judge-1027"); low = load(f"{ROOT}/runs/mdb-judge-opus-low")
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
        if len(errs) >= 2:
            for i, e in enumerate(errs):
                xs.append(np.mean(errs[:i] + errs[i + 1:])); ys.append(e); gs.append(group_of.get(b))
    xs, ys = np.array(xs), np.array(ys); r = np.corrcoef(xs, ys)[0, 1]
    grid(ax, "both"); lim = 0.85
    ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    other = np.array([g is None for g in gs])
    ax.scatter(xs[other], ys[other], s=6, color="#c9c8c3", edgecolor="white", linewidth=0.3, zorder=2)
    short = ["strict answer checking", "lenient long-output grading", "knowledge gap of subject"]
    for g, c, lab in zip(GROUPS, (BLUE, ORANGE, AQUA), short):
        m = np.array([gg == g for gg in gs])
        ax.scatter(xs[m], ys[m], s=14, color=c, edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Error on the other pairs of the benchmark")
    ax.set_ylabel("Error on this pair")
    ax.text(0.03, 0.97, f"r = {r:.2f}, {len(xs)} pairs, {sum(len(e) >= 2 for e in by.values())} benchmarks",
            transform=ax.transAxes, fontsize=5.8, color=INK2, va="top")
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.01), frameon=False, fontsize=5.2, title="cited by the forecaster for",
              title_fontsize=5.4, handletextpad=0.1, borderpad=0.2, labelspacing=0.25, alignment="left")
    ax.set_title("(d) Forecast errors are shared within a benchmark", fontsize=7.2, loc="left", color=INK)


def main():
    out_dir = sys.argv[1]
    plt.rcParams.update({"font.size": 6.5})
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.7))
    panel_curves(axs[0, 0], axs[0, 1])
    panel_memory(axs[1, 0])
    panel_errors(axs[1, 1])
    fig.subplots_adjust(wspace=0.42, hspace=0.48)
    save(fig, out_dir, "fig_overview")
    # the report shows the same panels as two figures: the main result (learning curves) and the analysis panels
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.95))
    panel_curves(axs[0], methods=[m for m in METHODS if not m.startswith("Empirical")], top=0.27)
    panel_level(axs[1])
    fig.subplots_adjust(wspace=0.42, bottom=0.22, top=0.89)
    save(fig, out_dir, "fig_main")
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.1))
    panel_memory(axs[0])
    axs[0].set_title("(a) Skill on benchmarks Opus recognizes or not", fontsize=7.2, loc="left", color=INK)
    panel_errors(axs[1])
    axs[1].set_title("(b) Forecast errors are shared within a benchmark", fontsize=7.2, loc="left", color=INK)
    fig.subplots_adjust(wspace=0.42, bottom=0.2, top=0.9)
    save(fig, out_dir, "fig_analysis")


if __name__ == "__main__":
    main()
