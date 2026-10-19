import os
import sys
import time

# Ensure backend app is importable
curr = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr, "..")))

from app.data.loader import list_egos
from app.pipeline import run_pipeline, get_results_dir


def main():
    print("=" * 60)
    print(" CircleSense: Batch Execution Across SNAP Facebook Networks")
    print("=" * 60)

    egos = list_egos()
    print(f"Found {len(egos)} ego-networks: {egos}\n")

    results_dir = get_results_dir()
    print(f"Output directory: {results_dir}\n")

    start_all = time.time()
    summaries = []

    for ego_id in egos:
        print(f"--> Processing Ego Network: {ego_id} ...")
        t0 = time.time()
        try:
            res = run_pipeline(ego_id, config=None, save_results=True)
            elapsed = round(time.time() - t0, 2)
            hybrid_m = res["metrics"]["hybrid"]
            print(f"    Done in {elapsed}s | Nodes: {res['graph_summary']['num_nodes']} | "
                  f"Circles: {res['graph_summary']['num_detected_circles']} | "
                  f"Hybrid ONMI: {hybrid_m['onmi']} | F1: {hybrid_m['avg_f1']}")
            summaries.append({
                "ego_id": ego_id,
                "nodes": res["graph_summary"]["num_nodes"],
                "edges": res["graph_summary"]["num_edges"],
                "onmi": hybrid_m["onmi"],
                "f1": hybrid_m["avg_f1"],
                "omega": hybrid_m["omega"],
                "runtime": elapsed
            })
        except Exception as e:
            print(f"    ERROR on ego {ego_id}: {e}")

    total_time = round(time.time() - start_all, 2)
    print("\n" + "=" * 60)
    print(f" Batch run complete in {total_time}s across {len(summaries)} networks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
