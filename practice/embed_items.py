"""Embed measurement-db items and both proxy evaluations' items with OpenAI text-embedding-3-small (through OpenRouter).

measurement-db side: 0/1 benchmarks (higher is better) with at least one pair of >= 20 graded responses, minus both proxy
evaluations' benchmarks, whose item table is local. Per benchmark, up to <per_bench> items (seed 0) among items with >= 2
graded responses from those pairs. Each item keeps its graded count and successes.
Proxy side: every target and the first 31 exemplars of every pair in both proxy evaluations.
Only item_content is embedded (what predict() receives as the item text), cut in the middle to MAX_CHARS.

Outputs in <out_dir>: meta.parquet (one row per embedded text) and emb.npy (float16, same row order).

Usage: python -I embed_items.py <out_dir> <cap_usd> [per_bench]
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # our own code folder only
from common import load_exam

ROOT = "./"
DATA = ROOT + "data/hf/"
CATALOG = ROOT + "agent_box_02/data/catalog/"
EXAMS = {"proxy-10/20": ROOT + "exams/rehearsal-02", "proxy-10/26": ROOT + "exams/proxy-confirm-1026"}
URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "openai/text-embedding-3-small"
MAX_CHARS, FALLBACK_CHARS, BATCH = 6000, 2000, 64


def cut(text, n):
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n // 2] + "\n[...]\n" + text[-n // 2:]


def measurement_db_items(per_bench):
    ps = pd.read_parquet(CATALOG + "pair_stats.parquet")
    bm = pd.read_parquet(CATALOG + "benchmarks_all.parquet")
    bm["direction"] = bm["response_scale"].map(lambda v: (json.loads(v) or {}).get("direction") if v else None)
    ok = ps.merge(bm[["folder", "direction"]], on="folder")
    ok = ok[(ok["response_type"] == "binary") & (ok["direction"] == "higher_is_better") & (ok["n_graded"] >= 20)]
    held_out = set()
    for d in EXAMS.values():
        held_out |= set(json.load(open(os.path.join(d, "exam.json")))["held_out_benchmarks"])
    ok = ok[~ok.folder.isin(held_out)]
    con, rows = duckdb.connect(), []
    for b in sorted(ok.folder.unique()):
        items_path = f"{DATA}{b}/formatted_tables/items.parquet"
        if not os.path.exists(items_path):
            continue
        subs = ok[ok.folder == b].subject_id.tolist()
        st = con.execute(f"select item_id, count(*) n, sum(case when response >= 0.5 then 1 else 0 end) s, "
                         f"count(distinct subject_id) n_sub from read_parquet('{DATA}{b}/formatted_tables/responses.parquet') "
                         f"where response is not null and subject_id in (select unnest(?)) group by 1 having count(*) >= 2",
                         [subs]).df()
        if len(st) > per_bench:
            st = st.sample(per_bench, random_state=0)
        text = pd.read_parquet(items_path, columns=["item_id", "content"]).set_index("item_id")["content"]
        st = st[st.item_id.isin(text.index)]
        for r in st.itertuples():
            rows.append({"source": "measurement-db", "benchmark": b, "item_id": r.item_id, "n": int(r.n), "s": int(r.s),
                         "n_sub": int(r.n_sub), "text": text.loc[r.item_id]})
    return rows


def proxy_items():
    rows = []
    for name, d in EXAMS.items():
        exam = load_exam(d)
        for pi, pair in enumerate(exam["pairs"]):
            for role, items in (("target", pair["targets"]), ("exemplar", pair["acquisitions"][:31])):
                for it in items:
                    rows.append({"source": name, "benchmark": pair["benchmark_name"], "item_id": it["key"], "pair": pi,
                                 "role": role, "text": it["item"]["item_content"]})
    return rows


def main():
    out, cap = sys.argv[1], float(sys.argv[2])
    per_bench = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    os.makedirs(out, exist_ok=True)
    rows = measurement_db_items(per_bench) + proxy_items()
    meta = pd.DataFrame(rows)
    print(f"{len(meta)} texts: measurement-db {int((meta.source == 'measurement-db').sum())} "
          f"from {meta[meta.source == 'measurement-db'].benchmark.nunique()} benchmarks, proxy {int((meta.source != 'measurement-db').sum())}",
          flush=True)
    key = open(os.path.expanduser("~/.config/openrouter/api_key")).read().strip()
    emb = np.zeros((len(meta), 1536), dtype=np.float32)
    done = np.zeros(len(meta), dtype=bool)
    spent, lock = {"usd": 0.0, "tokens": 0, "cut_short": 0}, threading.Lock()

    def call(texts):
        body = json.dumps({"model": MODEL, "input": texts}).encode()
        for attempt in range(5):
            req = urllib.request.Request(URL, data=body, method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                if "data" in d:
                    return d
                err = json.dumps(d)[:200]
            except urllib.error.HTTPError as e:
                err = f"http {e.code}: {e.read()[:200].decode(errors='replace')}"
                if e.code == 400:
                    return {"error": err}
            except Exception as e:  # network errors and timeouts
                err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(3 * (attempt + 1))
        return {"error": err}

    def run(start):
        idx = list(range(start, min(start + BATCH, len(meta))))
        with lock:
            if spent["usd"] >= cap:
                return
        d = call([cut(meta.text.iat[i], MAX_CHARS) for i in idx])
        if "error" in d:  # one text over the token limit fails the batch; retry one by one, shorter
            for i in idx:
                d1 = call([cut(meta.text.iat[i], MAX_CHARS)])
                if "error" in d1:
                    d1 = call([cut(meta.text.iat[i], FALLBACK_CHARS)])
                    with lock:
                        spent["cut_short"] += 1
                if "data" in d1:
                    with lock:
                        emb[i] = d1["data"][0]["embedding"]; done[i] = True
                        spent["usd"] += (d1.get("usage") or {}).get("cost", 0) or 0
            return
        with lock:
            for j, e in zip(idx, d["data"]):
                emb[j] = e["embedding"]; done[j] = True
            u = d.get("usage") or {}
            spent["usd"] += u.get("cost", 0) or 0
            spent["tokens"] += u.get("total_tokens", 0) or 0

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, range(0, len(meta), BATCH)))
    meta["embedded"] = done
    meta.drop(columns=["text"]).assign(chars=meta.text.str.len()).to_parquet(os.path.join(out, "meta.parquet"))
    np.save(os.path.join(out, "emb.npy"), emb.astype(np.float16))
    print(f"embedded {int(done.sum())} of {len(meta)}, tokens {spent['tokens']}, cost ${spent['usd']:.4f}, "
          f"retried shorter {spent['cut_short']}", flush=True)


if __name__ == "__main__":
    main()
