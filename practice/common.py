"""Shared helpers for the local practice exam that mimics the competition protocol.

The competition calls predict(input, labeled) with:
  input   = [subject, item]
  subject = 8 string fields, missing values as ""
  item    = item_content, item_features, interactors, benchmark_id (anonymous)
  labeled = [[[subject, item], label], ...] for every result revealed so far, across all pairs
"""
import getpass
import hashlib
import json
import os
import subprocess

import pandas as pd

SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra"]
BUDGETS = [0, 1, 3, 7, 15, 31]
ALC_WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
MAX_LABELS = 31


def as_str(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def table(data_dir, bench, name):
    return pd.read_parquet(os.path.join(data_dir, bench, "formatted_tables", name))


def subject_dict(row):
    return {k: as_str(row.get(k)) for k in SUBJECT_KEYS}


def item_dict(item_row, interactors, anon_benchmark_id):
    return {"item_content": as_str(item_row.get("content")),
            "item_features": as_str(item_row.get("item_features")),
            "interactors": as_str(interactors),
            "benchmark_id": anon_benchmark_id}


def anon_id(prefix, name, seed):
    digest = hashlib.sha256(f"{seed}:{prefix}:{name}".encode()).hexdigest()
    return f"{prefix}_{int(digest[:8], 16) % 900000 + 100000}"


def api_key(source):
    """An API key from "keychain:<service>" (macOS Keychain, account = this user) or from a file path."""
    if source.startswith("keychain:"):
        return subprocess.run(["/usr/bin/security", "find-generic-password", "-a", getpass.getuser(), "-s",
                               source.split(":", 1)[1], "-w"], capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.expanduser(source)) as f:
        return f.read().strip()


def load_exam(exam_dir):
    with open(os.path.join(exam_dir, "exam.json")) as f:
        return json.load(f)


def labeled_at(exam, budget):
    """Results revealed at a budget: the first `budget` acquisitions of every pair (nested sets)."""
    out = []
    for pair in exam["pairs"]:
        for acq in pair["acquisitions"][:budget]:
            out.append([[pair["subject"], acq["item"]], acq["label"]])
    return out


def input_for(pair, target):
    return [pair["subject"], target["item"]]


def brier_alc(exam, predictions):
    """predictions: {(budget, pair_index, item_key): probability}. Returns per-pair and mean ALC.

    Brier at a budget averages over every graded response of the pair's evaluation items,
    so items attempted several times count once per attempt, as in the competition.
    """
    rows = []
    for pi, pair in enumerate(exam["pairs"]):
        per_budget = []
        for b in BUDGETS:
            errs, n = 0.0, 0
            missing = 0
            for t in pair["targets"]:
                p = predictions.get((b, pi, t["key"]))
                if p is None:
                    missing += 1
                    continue
                for y in t["labels"]:
                    errs += (p - y) ** 2
                    n += 1
            per_budget.append(errs / n if n else None)
            if missing:
                per_budget[-1] = None if missing == len(pair["targets"]) else per_budget[-1]
        alc = None
        if all(v is not None for v in per_budget):
            alc = sum(w * v for w, v in zip(ALC_WEIGHTS, per_budget))
        rows.append({"pair": pi, "benchmark": pair["benchmark_name"], "model": pair["subject"]["normalized_name"],
                     **{f"B{b}": v for b, v in zip(BUDGETS, per_budget)}, "ALC": alc})
    df = pd.DataFrame(rows)
    mean_alc = df["ALC"].mean() if df["ALC"].notna().all() else None
    return df, mean_alc
