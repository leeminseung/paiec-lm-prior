"""Add value shading to the report's tables (11/6). Usage: python shade_tables.py <main.tex> (rewrites it in place).

Blue marks better values and red worse ones, darker for larger differences; gray marks magnitudes with no better side.
\\shmax (preamble) caps the colour share in a cell. ALC-type columns share one scale: a difference of 0.045 from the
reference (Opus's held-out gain) is full strength. The cell's first number is the value; the cell text is kept.
Rules: ("div", center, span, sign) diverging, sign +1 if lower is better and -1 if higher is better;
("seq", colour, lo, hi) one hue, light to dark; ("sqrt", colour, hi) one hue on a square-root scale from 0 to hi.
Run once per table; rows already shaded are left alone."""
import re, sys
PRE = r"""\usepackage{colortbl}
\usepackage{pgf}
\definecolor{better}{HTML}{2A78D6}
\definecolor{worse}{HTML}{E34948}
\definecolor{neutral}{HTML}{8C8B86}
% value shading in tables (tools/shade_tables.py): \shd{value}{center}{span}{sign}, \shs{colour}{value}{lo}{hi},
% \shq{colour}{value}{hi}; \shmax is the strongest colour share in percent
\newcommand{\shmax}{35}
\newcommand{\shd}[4]{\pgfmathsetmacro{\shx}{(#4)*((#2)-(#1))/(#3)}\pgfmathsetmacro{\shp}{min(1,abs(\shx))*\shmax}%
  \pgfmathparse{\shx>0 ? 1 : 0}\ifnum\pgfmathresult=1\relax\edef\shc{\noexpand\cellcolor{better!\shp}}%
  \else\edef\shc{\noexpand\cellcolor{worse!\shp}}\fi\shc}
\newcommand{\shs}[4]{\pgfmathsetmacro{\shp}{max(0,min(1,((#2)-(#3))/((#4)-(#3))))*\shmax}%
  \edef\shc{\noexpand\cellcolor{#1!\shp}}\shc}
\newcommand{\shq}[3]{\pgfmathsetmacro{\shp}{sqrt(max(0,min(1,(#2)/(#3))))*\shmax}%
  \edef\shc{\noexpand\cellcolor{#1!\shp}}\shc}
"""
SPAN = 0.045


def ablation_rule(cells, state):
    head = cells[0]
    if "Validation proxies" in head:
        state["block"] = "val"
    elif "Acquisition" in head:
        state["block"] = "acq"
    if state.get("block") == "val":
        return {1: ("div", 0.2078, SPAN, 1)}
    if state.get("block") == "acq":
        return {1: ("div", 0.1846, SPAN, 1)}
    return {1: ("div", 0.1846, SPAN, 1), 2: ("div", 0.1692, SPAN, 1)}


def effort_rule(cells, state):
    if cells[1].strip():
        state["measure"] = cells[1].strip()
    return {4: ("div", 0.0, 0.05, 1 if state.get("measure") == "ALC" else -1)}


def proxies_rule(cells, state):
    m = NUM.search(cells[2])
    if not m:
        return {}
    c = float(m.group(0))
    return {3: ("div", c, SPAN, 1), 4: ("div", c, SPAN, 1)}


RULES = {
    "tab:main": lambda c, s: {1: ("div", 0.214, SPAN, 1), 3: ("div", 0.219, SPAN, 1)},
    "tab:ladder": lambda c, s: {1: ("seq", "neutral", 25, 60), 2: ("div", 0.0, 0.262, -1), 3: ("seq", "better", 0.62, 0.70),
                                7: ("div", 0.214, SPAN, 1), 8: ("div", 0.214, SPAN, 1)},
    "tab:lead": lambda c, s: {k: ("sqrt", "worse", 0.101) for k in range(1, 6)},
    "tab:proxies": proxies_rule,
    "tab:ble": lambda c, s: {2: ("div", 0.169, SPAN, 1), 3: ("div", 0.717, 0.1, -1),
                             5: ("div", 0.161, SPAN, 1), 6: ("div", 0.723, 0.1, -1)},
    "tab:ablations": ablation_rule,
    "tab:effort": effort_rule,
    "tab:judges": lambda c, s: {2: ("div", 0.0, 0.234, -1), 3: ("seq", "better", 0.58, 0.72), 4: ("seq", "neutral", 0, 75),
                                5: ("div", 0.0, 0.312, -1), 6: ("div", 0.0, 0.312, -1)},
}
NUM = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")


def shade_cell(cell, rule):
    m = NUM.search(cell.replace("{,}", ""))
    if not m:
        return cell
    v = float(m.group(0))
    if rule[0] == "div":
        return f" \\shd{{{v}}}{{{rule[1]}}}{{{rule[2]}}}{{{rule[3]}}}" + cell
    if rule[0] == "seq":
        return f" \\shs{{{rule[1]}}}{{{v}}}{{{rule[2]}}}{{{rule[3]}}}" + cell
    return f" \\shq{{{rule[1]}}}{{{v}}}{{{rule[2]}}}" + cell


def main(path):
    t = open(path).read()
    if "\\newcommand{\\shd}" not in t:
        t = t.replace("\\newcommand{\\todo}[1]", PRE + "\n\\newcommand{\\todo}[1]", 1)
    for label, rule in RULES.items():
        i = t.index(f"\\label{{{label}}}"); a = t.index("\\midrule", i); b = t.index("\\bottomrule", i)
        state, out = {}, []
        for line in t[a:b].split("\n"):
            cells0 = line.split("&")
            if "&" not in line and "multicolumn" in line:
                rule([line], state)
            if "&" not in line or "\\shd" in line or "\\shs" in line or "\\shq" in line or line.strip().startswith("Variant"):
                out.append(line); continue
            end = line.rstrip(); tail = " \\\\" if end.endswith("\\\\") else ""
            cells = (end[:-2] if tail else end).split("&")
            for k, r in rule(cells, state).items():
                if k < len(cells):
                    cells[k] = shade_cell(cells[k], r)
            out.append("&".join(cells) + tail)
        t = t[:a] + "\n".join(out) + t[b:]
    open(path, "w").write(t)


if __name__ == "__main__":
    main(sys.argv[1])
