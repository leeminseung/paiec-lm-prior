"""Round 5: the Claude Opus 5.5 forecast as an offset in a hierarchical logistic model of success, with a
benchmark-specific slope on the forecast.

For a target item t of subject s on benchmark B:
    logit P(y = 1) = (1 + beta_B) z_st + u_B + v_sB,    z = logit of the forecast (budget 0, outcomes never shown)
    u_B    ~ N(0, 1.09^2)   a shift shared by every subject on the benchmark
    v_sB   ~ N(0, 0.46^2)   this subject's own shift on the benchmark
    beta_B ~ N(0, 0.3^2)    how much more or less the forecast deserves trust on this benchmark
The three standard deviations were estimated once on public measurement-db data (forecasts on 121 benchmarks), not on
any test or proxy data. Every revealed outcome on the same anonymous benchmark id, from this subject or another one,
enters with its own forecast as offset, and (u_B, beta_B, v) get a Laplace posterior (damped Newton steps: a step that
lowers the log posterior is halved, which keeps the solver from diverging when forecasts sit far from the outcomes).
The target's probability integrates the posterior uncertainty with the probit approximation. With nothing revealed
this is the forecast shrunk toward 0.5 by the prior spread.

Judgments come from batched API requests inside each call (judge_api.py): one for this subject's revealed items plus the
target, and one per other subject with revealed items on the benchmark, sent in parallel within a time budget. If the
request holding the target fails (API error, refusal, reply cut at the output cap), the target alone is requested once
more. When the target has a judgment, revealed items without one are left out of the update, so every offset comes from
the same judge. When the target has none, every item uses the constant 0.575, the round-1 prior mean, and the model
reduces to a judgment-free hierarchical model. Every prediction depends only on (input, labeled); nothing is kept
between calls. The API key is read from submission_config.json next to this file and never printed or logged.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # our own submission folder, for judge_api.py
from judge_api import judge_items  # noqa: E402

MODEL, EFFORT = "claude-opus-5-5", "low"
TB, TS, TAU, M = 1.09, 0.46, 0.3, 0.575
TIME_BUDGET = 100  # seconds for all API requests of one call, below the platform's per-call limit


def _load_key():
    try:
        with open(os.path.join(HERE, "submission_config.json")) as f:
            return json.load(f).get("anthropic_api_key") or None
    except (OSError, ValueError):
        return None


_KEY = _load_key()


def _logit(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))


def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(z, dtype=float)))  # the logistic function, without overflow


def posterior(obs, m):
    """obs: list of (subject index, offset, label); m subjects. Mode and covariance of [u, beta, v_0..v_{m-1}]."""
    prec = np.array([1 / TB ** 2, 1 / TAU ** 2] + [1 / TS ** 2] * m)
    th = np.zeros(m + 2)
    if not obs:
        return th, np.diag(1 / prec)
    idx = np.array([o[0] for o in obs], dtype=int)
    z = np.array([o[1] for o in obs], dtype=float)
    y = np.array([o[2] for o in obs], dtype=float)
    X = np.zeros((len(z), m + 2))
    X[:, 0], X[:, 1] = 1.0, z
    X[np.arange(len(z)), 2 + idx] = 1.0

    def log_post(t):
        eta = z + X @ t
        return float(np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * np.sum(prec * t ** 2))

    for _ in range(200):
        q = _sig(z + X @ th)
        g = X.T @ (y - q) - prec * th
        H = (X * (q * (1 - q))[:, None]).T @ X + np.diag(prec)
        step = np.linalg.solve(H, g)
        f0, lam = log_post(th), 1.0
        while log_post(th + lam * step) < f0 and lam > 1e-6:
            lam /= 2
        th = th + lam * step
        if np.abs(lam * step).max() < 1e-9:
            break
    q = _sig(z + X @ th)
    H = (X * (q * (1 - q))[:, None]).T @ X + np.diag(prec)
    return th, np.linalg.inv(H)


def _judge(subject, revealed, target, deadline):
    if not _KEY:
        return [None] * (len(revealed) + (target is not None))
    try:
        return judge_items(subject, revealed, target, _KEY, MODEL, EFFORT, deadline=deadline)
    except Exception:  # predict() must not raise; a failed request falls back to the constant
        return [None] * (len(revealed) + (target is not None))


def predict(input, labeled=None):
    subject, item = input[0], input[1]
    bench = item.get("benchmark_id", "")
    me = json.dumps(subject, sort_keys=True)
    groups = {me: (subject, [], [])}  # subject key -> (subject dict, revealed items, labels); this subject first
    for entry in labeled or []:
        try:
            (other_subject, other_item), label = entry
            y = int(label)
        except (TypeError, ValueError):
            continue
        if y not in (0, 1) or other_item.get("benchmark_id", "") != bench:
            continue
        key = json.dumps(other_subject, sort_keys=True)
        groups.setdefault(key, (other_subject, [], []))
        groups[key][1].append(other_item)
        groups[key][2].append(y)
    keys = [me] + [k for k in groups if k != me and groups[k][1]]
    deadline = time.monotonic() + TIME_BUDGET
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = {k: pool.submit(_judge, groups[k][0], groups[k][1], item if k == me else None, deadline) for k in keys}
        judged = {k: f.result() for k, f in futures.items()}
    p_target = judged[me][-1]
    if p_target is None and groups[me][1]:
        p_target = _judge(subject, [], item, deadline)[-1]
    obs = []
    for j, k in enumerate(keys):
        for p, y in zip(judged[k], groups[k][2]):
            if p_target is None:
                obs.append((j, _logit(M), y))
            elif p is not None:
                obs.append((j, _logit(p), y))
    z = _logit(M if p_target is None else p_target)
    th, C = posterior(obs, len(keys))
    x = np.zeros(len(keys) + 2)
    x[0], x[1], x[2] = 1.0, z, 1.0  # u, beta times the target's offset, this subject's v (index 0)
    mean, var = float(x @ th), float(x @ C @ x)
    return float(_sig((z + mean) / math.sqrt(1 + math.pi * var / 8)))
