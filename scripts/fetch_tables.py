"""Download chosen tables for chosen benchmarks from measurement-db at a pinned revision.

Usage: python -I fetch_tables.py <dest_dir> <bench1,bench2,...|@list.txt> <table1.parquet,...> [--dry-run]
"""
import sys, json, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"

def main():
    dest, tables = sys.argv[1], sys.argv[3].split(",")
    arg = sys.argv[2]
    benches = [l.strip() for l in open(arg[1:]) if l.strip()] if arg.startswith("@") else arg.split(",")
    dry = "--dry-run" in sys.argv
    api = HfApi()
    rev = open(f"{dest}/REVISION.txt").read().strip()
    paths = [f"{b}/formatted_tables/{t}" for b in benches for t in tables]
    infos = []
    for i in range(0, len(paths), 100):
        infos += api.get_paths_info(REPO, paths[i:i + 100], repo_type="dataset", revision=rev)
    found = {p.path: p.size for p in infos}
    print(json.dumps({"revision": rev, "files": len(found), "missing": [p for p in paths if p not in found],
                      "total_mb": round(sum(found.values()) / 1e6, 2)}))
    if dry:
        return
    def get(p):
        for attempt in range(6):
            try:
                return hf_hub_download(REPO, p, repo_type="dataset", revision=rev, local_dir=dest)
            except Exception:  # network resets happen on long runs; back off and retry
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(4) as ex:
        done = list(ex.map(get, sorted(found)))
    print("downloaded:", len(done))

if __name__ == "__main__":
    main()
