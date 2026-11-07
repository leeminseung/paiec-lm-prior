"""Download only benchmarks.parquet and subjects.parquet for every benchmark folder.

Usage: python -I fetch_small_tables.py <dest_dir> [--dry-run]
Pins the dataset revision so every table comes from the same snapshot.
"""
import sys, json, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
TABLES = ("benchmarks.parquet", "subjects.parquet")

def main():
    dest = sys.argv[1]
    dry = "--dry-run" in sys.argv
    api = HfApi()
    rev = api.dataset_info(REPO).sha
    top = api.list_repo_tree(REPO, repo_type="dataset", revision=rev)
    benches = sorted(e.path for e in top if not hasattr(e, "size") or e.__class__.__name__ == "RepoFolder")
    paths = [f"{b}/formatted_tables/{t}" for b in benches for t in TABLES]
    infos = []
    for i in range(0, len(paths), 100):
        infos += api.get_paths_info(REPO, paths[i:i + 100], repo_type="dataset", revision=rev)
    found = {p.path: p.size for p in infos}
    missing = [p for p in paths if p not in found]
    print(json.dumps({"revision": rev, "benchmark_folders": len(benches), "files_found": len(found),
                      "files_missing": len(missing), "total_mb": round(sum(found.values()) / 1e6, 2)}))
    if missing:
        print("missing (first 10):", missing[:10])
    if dry:
        return
    def get(p):
        for attempt in range(6):
            try:
                return hf_hub_download(REPO, p, repo_type="dataset", revision=rev, local_dir=dest)
            except Exception as e:  # network resets happen on long runs; back off and retry
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(4) as ex:
        done = list(ex.map(get, sorted(found)))
    print("downloaded:", len(done))
    with open(f"{dest}/REVISION.txt", "w") as f:
        f.write(rev + "\n")

if __name__ == "__main__":
    main()
