"""Worked example of the predictor on one held-out benchmark with two subjects: tau2-bench (held-out proxy 1), with
OpenAI o4-mini and GPT-4.1 mini. Chosen to show each part of the model: the forecast ranks items well but sets their
level far too low, the benchmark intercept corrects the level, outcomes of one subject move the other's predictions, and
the subject intercepts and the slope take smaller parts.

Writes fig_worked_table.tex (five GPT-4.1 mini items evenly spaced in forecast rank, shown by domain and the first words of
the customer's request, with the forecast, our predictions at budgets 1, 7 and 31 and the outcome; then the revealed
successes of both subjects, the posterior modes and the Beta-binomial baseline) and fig_worked.pdf (mean prediction over
each subject's scored items by budget, against its success rate, and for GPT-4.1 mini also from its own outcomes only;
(c) the Laplace posterior of each random effect at budgets 0, 1, 7 and 31).
Prints the numbers behind the caption. No model calls.

Usage: python -I worked_example.py <out_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = "./report/figures"
sys.path.insert(0, FIG)
from make_figures import AQUA, BLUE, GRID, INK, INK2, MUTED, ROOT, logit  # noqa: E402
sys.path.insert(0, f"{ROOT}/practice")
from common import BUDGETS, load_exam  # noqa: E402
from eval_prior_combo import A0, ALPHA, M, counts  # noqa: E402
from exemplar_shift import load  # noqa: E402
from hier_llm import TAU  # noqa: E402
from hier_slope import posterior_slope  # noqa: E402

EXAM, T_RUN, E_RUN = "proxy-1028-1", "p1028-1-sub-opus-low", "p1028-1-sub-exemplar-opus-low"
BENCH, MAIN = "tau2_bench", "OpenAI GPT-4.1 mini"
SB, SP = 1.09, 0.46
SHOW = (1, 7, 31)
sig = lambda e: 1 / (1 + np.exp(-e))


def request(content):
    d = json.loads(content)
    task = d.get("task", {})
    ins = task.get("user_scenario", {}).get("instructions", {})
    text = (ins.get("reason_for_call") if isinstance(ins, dict) else None) or task.get("ticket") or ""
    return d["environment_info"]["domain_name"], " ".join(text.split())


def first_words(text, limit):
    out = ""
    for w in text.split(" "):
        if len(out) + len(w) + 1 > limit:
            return out + "\\ldots"
        out = f"{out} {w}".strip()
    return out


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")


def predict(th, C, z, j, k):
    x = np.zeros(2 + k); x[0] = 1; x[1] = z; x[2 + j] = 1
    return float(sig((z + x @ th) / np.sqrt(1 + np.pi * (x @ C @ x) / 8)))


def main():
    out_dir = sys.argv[1]
    exam = load_exam(f"{ROOT}/exams/{EXAM}")
    pt, pe = load(f"{ROOT}/runs/{T_RUN}"), load(f"{ROOT}/runs/{E_RUN}")
    pairs = exam["pairs"]
    members = [pi for pi, p in enumerate(pairs) if p["benchmark_name"] == BENCH]
    names = [pairs[pi]["subject"]["normalized_name"].replace("OpenAI ", "") for pi in members]
    jm = [pairs[pi]["subject"]["normalized_name"] for pi in members].index(MAIN)
    k = len(members)
    zt = lambda pi, t: logit(pt.get((pi, t["key"]), M))
    ze = lambda pi, a: logit(pe.get((pi, a["key"]), M))
    cnt = counts(exam)

    post, mean, own, base = {}, {}, {}, {}
    for b in BUDGETS:
        obs = [(j, ze(pi, a), a["label"]) for j, pi in enumerate(members) for a in pairs[pi]["acquisitions"][:b]]
        th, C = posterior_slope(obs, k, SB, SP, TAU)
        post[b] = (th, C)
        for j, pi in enumerate(members):
            tg = pairs[pi]["targets"]
            mean[(b, j)] = np.mean([predict(th, C, zt(pi, t), j, k) for t in tg])
            th1, C1 = posterior_slope([(0, o[1], o[2]) for o in obs if o[0] == j], 1, SB, SP, TAU)
            own[(b, j)] = np.mean([predict(th1, C1, zt(pi, t), 0, 1) for t in tg])
            x, kk, s, n = cnt[(b, pi)]
            base[(b, j)] = (x + ALPHA * (s + A0 * M) / (n + A0)) / (kk + ALPHA)
    rate = [np.mean([np.mean(t["labels"]) for t in pairs[pi]["targets"]]) for pi in members]
    fmean = [np.mean([pt.get((pi, t["key"]), M) for t in pairs[pi]["targets"]]) for pi in members]

    # (a) table
    pm = members[jm]
    items = sorted(pairs[pm]["targets"], key=lambda t: -pt.get((pm, t["key"]), M))
    picks = [items[int(round(r))] for r in np.linspace(0, len(items) - 1, 5)]
    sh = lambda v: f"\\shs{{better}}{{{v:.2f}}}{{0}}{{1}} {v:.2f}"
    rows = ["\\begin{tabular}{@{}lccccc@{}}", "    \\toprule",
            "    & & \\multicolumn{3}{c}{Prediction at budget} & \\\\", "    \\cmidrule(lr){3-5}",
            "    Item: domain, request & Forecast & 1 & 7 & 31 & Outcome \\\\", "    \\midrule",
            f"    \\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{names[jm]}, five of its {len(items)} scored items}}}} \\\\"]
    for t in picks:
        dom, req = request(t["item"]["item_content"])
        dom = dom.replace("-workflow", "")
        rows.append(f"    {esc(dom)}: ``{esc(first_words(req, 31 - len(dom)))}'' & {sh(pt.get((pm, t['key']), M))} & "
                    + " & ".join(sh(predict(*post[b], zt(pm, t), jm, k)) for b in SHOW)
                    + f" & {sum(t['labels'])} of {len(t['labels'])} \\\\")
    rows.append("    Beta-binomial baseline, every item & & " + " & ".join(sh(base[(b, jm)]) for b in SHOW) + " & \\\\")
    rows.append("    \\midrule")
    rows.append("    \\multicolumn{6}{@{}l}{\\emph{Update from both subjects' revealed outcomes}} \\\\")
    for j, pi in enumerate(members):
        acq = pairs[pi]["acquisitions"]
        rows.append(f"    Successes revealed, {names[j]} & & " + " & ".join(f"{sum(a['label'] for a in acq[:b])} of {b}" for b in SHOW) + " & \\\\")
    rows.append("    $u_B$ (benchmark) & $0$ & " + " & ".join(f"${post[b][0][0]:+.2f}$" for b in SHOW) + " & \\\\")
    rows.append("    $\\beta_B$ (slope) & $0$ & " + " & ".join(f"${post[b][0][1]:+.2f}$" for b in SHOW) + " & \\\\")
    for j in range(k):
        rows.append(f"    $v_{{sB}}$, {names[j]} & $0$ & " + " & ".join(f"${post[b][0][2 + j]:+.2f}$" for b in SHOW) + " & \\\\")
    rows += ["    \\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_dir, "fig_worked_table.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # (b) mean prediction by budget, (c) posterior of each random effect, stacked to fill the column beside the table
    plt.rcParams.update({"font.size": 6.5})
    fig, (ax, cx) = plt.subplots(2, 1, figsize=(2.25, 2.95), gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.62})
    xs = np.log2(np.array(BUDGETS) + 1)
    cols = {jm: BLUE, 1 - jm: AQUA}
    for j in range(k):
        ax.axhline(rate[j], color=cols[j], lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.plot(xs, [mean[(b, j)] for b in BUDGETS], color=cols[j], lw=1.2, marker="o", ms=2.2, zorder=3,
                label=f"{names[j]}")
        if j == jm:
            ax.plot(xs, [own[(b, j)] for b in BUDGETS], color=cols[j], lw=0.9, ls=(0, (3, 2)), zorder=2,
                    label=f"{names[j]}, without {names[1 - j]}'s outcomes")
    ax.set_xticks(xs, [str(b) for b in BUDGETS]); ax.set_xlim(-0.2, xs[-1] + 0.2)
    ax.set_ylim(0.2, 1.18); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Budget", fontsize=6, labelpad=1); ax.set_ylabel("Mean prediction", fontsize=6)
    ax.legend(loc="upper left", frameon=False, fontsize=5, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("(b) Mean prediction over scored items", fontsize=7.2, loc="left", color=INK)

    rows = [("$u_B$", 0), ("$\\beta_B$", 1)] + [(f"$v_{{sB}}$, {names[j]}", 2 + j) for j in range(k)]
    shades = {0: MUTED, 1: "#9cc0ea", 7: "#3f84d8", 31: "#1d4f91"}
    grid_x = np.linspace(-2.6, 3.6, 500)
    for r, (lab, i) in enumerate(rows):
        base_y = (len(rows) - 1 - r) * 1.0
        dens = {}
        for bb in (0, 1, 7, 31):
            th, C = post[bb]
            m, sd = th[i], np.sqrt(C[i, i])
            dens[bb] = np.exp(-0.5 * ((grid_x - m) / sd) ** 2) / sd
        top = max(d.max() for d in dens.values())
        for bb in (0, 1, 7, 31):
            yv = base_y + 0.85 * dens[bb] / top
            cx.plot(grid_x, yv, color=shades[bb], lw=0.8 if bb == 0 else 1.1, ls=(0, (2, 1.5)) if bb == 0 else "-",
                    label=("budget 0 (prior)" if bb == 0 else f"{bb}") if r == 0 else None, zorder=3)
            if bb == 31:
                cx.fill_between(grid_x, base_y, yv, color=shades[bb], alpha=0.10, lw=0)
        cx.axhline(base_y, color=GRID, lw=0.5, zorder=1)
        cx.text(-2.55, base_y + 0.55, lab, fontsize=5.6, color=INK, ha="left", va="center")
    cx.axvline(0, color=MUTED, lw=0.5, zorder=1)
    cx.set_yticks([]); cx.set_ylim(-0.05, len(rows) + 0.15); cx.set_xlim(grid_x[0], grid_x[-1])
    cx.set_xlabel("Value on the logit scale", fontsize=6, labelpad=1)
    cx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, frameon=False, fontsize=5, handlelength=1.3, columnspacing=0.7, handletextpad=0.3, borderaxespad=0)
    cx.set_title("(c) Posterior of each random effect", fontsize=7.2, loc="left", color=INK)
    for a_ in (ax, cx):
        a_.yaxis.grid(a_ is ax, color=GRID, lw=0.5); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(MUTED)
        a_.tick_params(colors=INK2, labelsize=5.6, pad=1.5)
    cx.spines["left"].set_visible(False)
    ax.spines["left"].set_bounds(0.2, 1.0)
    # (b) narrower so that its tick labels and axis label sit inside the column, flush with (c)'s left edge
    pc, pb = cx.get_position(), ax.get_position()
    ax.set_position([pc.x0 + 0.16, pb.y0, pc.x1 - (pc.x0 + 0.16), pb.height])
    ax.set_title("(b) Mean prediction over scored items", fontsize=7.2, loc="left", color=INK, x=-0.16 / (pc.x1 - pc.x0 - 0.16))
    fig.savefig(os.path.join(out_dir, "fig_worked.pdf"), bbox_inches="tight", pad_inches=0.02)

    for j, pi in enumerate(members):
        print(f"{names[j]}: {len(pairs[pi]['targets'])} scored items, success rate {rate[j]:.3f}, mean forecast {fmean[j]:.3f}")
        print("   mean prediction " + " ".join(f"b{b}={mean[(b, j)]:.2f}" for b in BUDGETS))
        print("   own-only        " + " ".join(f"b{b}={own[(b, j)]:.2f}" for b in BUDGETS))
    for b in BUDGETS:
        th = post[b][0]
        print(f"b={b:2} u={th[0]:+.2f} beta={th[1]:+.2f} " + " ".join(f"v[{names[j]}]={th[2 + j]:+.2f}" for j in range(k)))
    for j, pi in enumerate(members):  # within-pair AUC over responses, ties count half
        sc = [(pt.get((pi, t["key"]), M), y) for t in pairs[pi]["targets"] for y in t["labels"]]
        pos, neg = [q for q, y in sc if y == 1], [q for q, y in sc if y == 0]
        print(f"{names[j]}: within-pair AUC {np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]):.3f}")

    # where this benchmark sits among the 27 held-out pairs: ALC gain over the baseline and forecast level error
    import report_tables as rt
    gains, level = [], []
    for spec in rt.SPECS[2:]:
        out, cl = rt.pair_alc(*spec)
        ex = load_exam(f"{ROOT}/exams/{spec[1]}"); tq = load(f"{ROOT}/runs/{spec[2]}")
        for pi, p in enumerate(ex["pairs"]):
            g = out[rt.REF][pi] - out["Hierarchical logit model with Opus offsets"][pi]
            err = abs(np.mean([tq.get((pi, t["key"]), M) for t in p["targets"]]) - np.mean([np.mean(t["labels"]) for t in p["targets"]]))
            gains.append((g, spec[1], p["benchmark_name"], p["subject"]["normalized_name"]))
            level.append((err, spec[1], p["benchmark_name"], p["subject"]["normalized_name"]))
    print(f"held-out pairs {len(gains)}: median ALC gain over the baseline {np.median([g[0] for g in gains]):.4f}")
    for g in gains:
        if g[1] == EXAM and g[2] == BENCH:
            print(f"  {g[3]}: ALC gain {g[0]:.4f}")
    print("largest forecast level errors (|mean forecast - success rate|):", [(round(e[0], 3), e[2], e[3]) for e in sorted(level, reverse=True)[:3]])


if __name__ == "__main__":
    main()
