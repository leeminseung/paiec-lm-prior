"""Budget 0, memory or generalization (report flow 3): many judges on the same 847 items, split by whether each judge
recognizes the benchmark.

Items: the 847-item subset of the measurement-db judge set (exams/mdb-judge-1027-onepair, 121 benchmarks, one pair each,
7 items per pair). Judges: Claude (Opus, Sonnet, Haiku at effort low; Opus with the "analyse, do not solve" instruction;
Opus high with thinking summaries), five open-weight models and three OpenAI models (added 11/6, from their full
calibration-set runs at effort low; recognition runs know-or-gpt61sol, -gpt61terra, -gpt6luna on the 121 benchmarks), all with
the same judge prompt.
Recognition: know_benchmark.py answers; a benchmark counts as recognized by a model only when the model said so and
grade_recognition.py judged the name it gave to be this benchmark or the dataset its items come from (GRADES file);
without a grade, a normalized-name rule decides. A confident wrong name is "not".
Skill = 1 - Brier / Brier of the base rate 0.627 on the same items (removes how hard the items are to predict at all).

Prints: per judge, overall Brier, pair-level gap |pair mean prediction - pair success rate|, within-pair AUC, skill;
the same split by the judge's own recognition and by Opus's recognition (common strata for every judge) and by
benchmark release date (before 2025 or not), with recognized-minus-not differences resampled over benchmarks. Writes a
CSV of every "recognized" answer with the match decision for checking by eye.

Usage: python -I memory_generalization.py <out_csv>
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam
from ble_b0_compare import auc
from exemplar_shift import load

ROOT = "."
SUBSET = f"{ROOT}/exams/mdb-judge-1027-onepair"
CATALOG = f"{ROOT}/agent_box_02/data/catalog/benchmarks_all.parquet"
BASE = 0.627
GRADES = f"{ROOT}/runs/know-grades.jsonl"
# judge name: (run, kind, recognition run)
JUDGES = {
    "Opus low": ("mdb-judge-opus-low", "full", "know-opus-low"),
    "Sonnet low": ("onepair-sub-sonnet-5-5-low", "subset", "know-sonnet-low"),
    "Haiku low": ("onepair-sub-haiku-5-5-low", "subset", "know-haiku-low"),
    "Opus low, analyse only": ("onepair-sub-opus-5-5-low-analyse", "subset", "know-opus-low"),
    "Opus high, thinking": ("thinking-opus-high-onepair", "thinking", "know-opus-low"),
    "GLM-5.3 Flash": ("or-mdb-glm53flash", "subset", "know-or-glm53flash"),
    "Muse Spark Open": ("or-mdb-muse-spark-open", "subset", "know-or-muse-spark-open"),
    "DeepSeek V4.1 Flash low": ("or-mdb-dsv41flash-low", "subset", "know-or-dsv41flash"),
    "MiniMax M3.1 Flash": ("or-mdb-minimax-m31flash", "subset", "know-or-minimax-m31flash"),
    "MiniMax M3": ("or-mdb-minimax-m3", "subset", "know-or-minimax-m3"),
    "GPT-6.1 Sol low": ("or-mdbfull-gpt61sol-low", "full", "know-or-gpt61sol"),
    "GPT-6.1 Terra low": ("or-mdbfull-gpt61terra-low", "full", "know-or-gpt61terra"),
    "GPT-6 Luna low": ("or-mdbfull-gpt6luna-low", "full", "know-or-gpt6luna"),
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def matches(guess, names):
    g = norm(guess)
    if len(g) < 3:
        return False
    gt = {w for w in g.split() if len(w) >= 3}
    for n in names:
        n = norm(n)
        if not n:
            continue
        if (len(n) >= 4 and (n in g or g in n)) or n.replace(" ", "") == g.replace(" ", ""):
            return True
        nt = {w for w in n.split() if len(w) >= 3}
        if nt and gt and len(nt & gt) / len(nt | gt) >= 0.5:
            return True
    return False


def judgments(run, kind, exam):
    path = f"{ROOT}/runs/{run}"
    if kind == "thinking":
        return {(r["pair"], r["key"]): r["probability"] for r in map(json.loads, open(f"{path}/judgments.jsonl"))
                if r.get("probability") is not None}
    if not os.path.exists(f"{path}/predictions.jsonl"):
        return {}
    j = load(path)
    if kind == "full":
        return {(pi, t["key"]): j[(p["source_pair"], t["key"])] for pi, p in enumerate(exam["pairs"]) for t in p["targets"]
                if (p["source_pair"], t["key"]) in j}
    return j


def recognition(run, names, grades):
    path = f"{ROOT}/runs/{run}/answers.jsonl"
    out, rows = {}, []
    if not os.path.exists(path):
        return out, rows
    for r in map(json.loads, open(path)):
        a = r.get("answer")
        if not a:
            continue
        g = grades.get((run, r["benchmark"]))
        ok = bool(a["recognized"]) and (g in ("benchmark", "source") if g else
                                         matches(a["benchmark_name"], names.get(r["benchmark"], [r["benchmark"]])))
        out[r["benchmark"]] = ok
        if a["recognized"]:
            rows.append({"run": run, "benchmark": r["benchmark"], "true_names": " | ".join(names.get(r["benchmark"], [])),
                         "guess": a["benchmark_name"], "confidence": a["confidence"], "grade": g, "match": ok,
                         "known_results": a["known_results"]})
    return out, rows


def pair_stats(exam, pred, keep):
    """Per pair: (benchmark, squared-error sum, response count, base squared-error sum, pair gap, AUC)."""
    out = []
    for pi, p in enumerate(exam["pairs"]):
        ts = [t for t in p["targets"] if (pi, t["key"]) in pred and keep(pi, t)]
        if not ts:
            continue
        v = np.array([pred[(pi, t["key"])] for t in ts]); ym = np.array([np.mean(t["labels"]) for t in ts])
        se = sum(sum((v[j] - y) ** 2 for y in t["labels"]) for j, t in enumerate(ts))
        sb = sum(sum((BASE - y) ** 2 for y in t["labels"]) for t in ts)
        n = sum(len(t["labels"]) for t in ts)
        out.append((p["benchmark_name"], se / n, sb / n, abs(v.mean() - ym.mean()), auc(v, ym)))
    return out


def summarize(rows):
    b = np.array([r[1] for r in rows]); base = np.array([r[2] for r in rows])
    return {"pairs": len(rows), "brier": b.mean(), "skill": 1 - b.mean() / base.mean(),
            "gap": np.mean([r[3] for r in rows]), "auc": np.nanmean([r[4] for r in rows])}


def split_diff(rows, flag, draws=2000, seed=0):
    """Skill(recognized) - skill(not), resampling benchmarks within each group."""
    yes = [r for r in rows if flag.get(r[0])]; no = [r for r in rows if not flag.get(r[0])]
    if len(yes) < 3 or len(no) < 3:
        return None
    sy, sn = summarize(yes), summarize(no)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(draws):
        a = [yes[i] for i in rng.integers(0, len(yes), len(yes))]; c = [no[i] for i in rng.integers(0, len(no), len(no))]
        ds.append(summarize(a)["skill"] - summarize(c)["skill"])
    return sy, sn, sy["skill"] - sn["skill"], np.quantile(ds, [.025, .975])


def main():
    out_csv = sys.argv[1]
    exam = load_exam(SUBSET)
    cat = pd.read_parquet(CATALOG)
    names = {f: [n, bid] for f, n, bid in zip(cat.folder, cat.name, cat.benchmark_id)}
    date = dict(zip(cat.folder, pd.to_datetime(cat.release_date, errors="coerce")))
    older = {b: (d is not None and not pd.isna(d) and d < pd.Timestamp("2025-01-01")) for b, d in date.items()}
    grades = {}
    if os.path.exists(GRADES):
        grades = {(r["run"], r["benchmark"]): r["match"] for r in map(json.loads, open(GRADES)) if r.get("match")}
    rec, all_rows = {}, []
    for jn, (_, _, krun) in JUDGES.items():
        if krun not in rec:
            rec[krun], rows = recognition(krun, names, grades)
            all_rows += rows
    pd.DataFrame(all_rows).drop_duplicates(["run", "benchmark"]).to_csv(out_csv, index=False)
    opus_flag = rec["know-opus-low"]
    benches = {p["benchmark_name"] for p in exam["pairs"]}
    print(f"{len(benches)} benchmarks; recognized (verified name) by: " +
          ", ".join(f"{k} {sum(v.get(b, False) for b in benches)}/{sum(b in v for b in benches)}" for k, v in rec.items()))
    print(f"\n{'judge':26} {'items':>5} {'Brier':>7} {'skill':>6} {'gap':>6} {'AUC':>6} | own-recognition split: skill yes / no, "
          f"diff [95%] | Opus-recognition split | release before 2025 / 2025 on")
    for jn, (run, kind, krun) in JUDGES.items():
        pred = judgments(run, kind, exam)
        if not pred:
            print(f"{jn:26} (no run yet)")
            continue
        rows = pair_stats(exam, pred, lambda pi, t: True)
        s = summarize(rows)
        line = f"{jn:26} {len(pred):5d} {s['brier']:.4f} {s['skill']:.3f} {s['gap']:.3f} {s['auc']:.3f}"
        for flag in (rec.get(krun, {}), opus_flag, older):
            d = split_diff(rows, flag)
            line += " | " + ("n/a" if d is None else
                             f"{d[0]['skill']:.3f} ({d[0]['pairs']}) / {d[1]['skill']:.3f} ({d[1]['pairs']}), "
                             f"{d[2]:+.3f} [{d[3][0]:+.3f}, {d[3][1]:+.3f}]")
        print(line)


if __name__ == "__main__":
    main()
