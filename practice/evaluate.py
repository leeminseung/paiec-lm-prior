"""Score predictors on a practice exam. Baselines run here; claude -p predictions are read from a run folder.

Usage:
  python -I evaluate.py <exam_dir> baselines
  python -I evaluate.py <exam_dir> run <run_dir>     # score predictions.jsonl, plus baselines on the same calls
"""
import json
import os
import sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # our own code folder only (no downloaded data here)
from common import BUDGETS, brier_alc, input_for, labeled_at, load_exam


def predict_half(inp, labeled):
    return 0.5


def predict_empirical_mean(inp, labeled):
    """Organizers' empirical-mean baseline: mean of revealed labels with the same subject and benchmark id."""
    subject, item = inp
    ys = [y for (s, it), y in labeled if s == subject and it["benchmark_id"] == item["benchmark_id"]]
    return sum(ys) / len(ys) if ys else 0.5


BASELINES = {"always 0.5": predict_half, "empirical mean": predict_empirical_mean}


def baselines_for(exam_dir):
    """The fixed baselines, plus the shrinkage baseline when shrinkage.py has fitted it for this exam."""
    out = dict(BASELINES)
    path = os.path.join(exam_dir, "shrinkage.json")
    if os.path.exists(path):
        from shrinkage import make_predictor
        fit = json.load(open(path))
        out["shrunk mean"] = make_predictor(fit["m"], fit["alpha"])
    return out


def run_python_predictor(exam, fn, only=None):
    preds = {}
    for b in BUDGETS:
        labeled = labeled_at(exam, b)
        for pi, pair in enumerate(exam["pairs"]):
            for t in pair["targets"]:
                if only is not None and (b, pi, t["key"]) not in only:
                    continue
                preds[(b, pi, t["key"])] = float(fn(input_for(pair, t), labeled))
    return preds


def per_call_brier(exam, preds):
    out = {}
    for (b, pi, key), p in preds.items():
        t = next(t for t in exam["pairs"][pi]["targets"] if t["key"] == key)
        out[(b, pi, key)] = sum((p - y) ** 2 for y in t["labels"]) / len(t["labels"])
    return out


def main():
    exam_dir, mode = sys.argv[1], sys.argv[2]
    exam = load_exam(exam_dir)
    if mode == "baselines":
        for name, fn in baselines_for(exam_dir).items():
            df, mean_alc = brier_alc(exam, run_python_predictor(exam, fn))
            print(f"== {name}: mean ALC {mean_alc:.4f}")
            print(df.round(4).to_string(index=False))
        return
    run_dir = sys.argv[3]
    preds = {}
    with open(os.path.join(run_dir, "predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("probability") is not None:
                preds[(r["budget"], r["pair"], r["key"])] = r["probability"]
    keys = set(preds)
    rows = {"claude -p": per_call_brier(exam, preds)}
    for name, fn in baselines_for(exam_dir).items():
        rows[name] = per_call_brier(exam, run_python_predictor(exam, fn, only=keys))
    print(f"calls scored: {len(keys)} (mean Brier over the same calls, each call = one item at one budget)")
    for name, d in rows.items():
        print(f"  {name:16} mean Brier {sum(d.values()) / len(d):.4f}")
    by_budget = {}
    for (b, pi, key), v in rows["claude -p"].items():
        by_budget.setdefault(b, []).append(v)
    print("  claude -p by budget:", {b: round(sum(v) / len(v), 4) for b, v in sorted(by_budget.items())})


if __name__ == "__main__":
    main()
