"""
Ablation and tuning sweep for CircleSense (single run per setting, fixed seed).

Part 1: one-factor-at-a-time ablation around the default configuration.
Part 2: small grid search to pick a better configuration.

Only the hybrid model is retrained per setting; seeds and graph tensors are
cached per ego-network. Results are averaged over all ego-networks and written
to backend/experiments/ablation_results.md and ablation_results.csv.

Usage: python backend/scripts/ablation.py [--quick]
"""
import csv
import itertools
import os
import sys
import time

curr = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr, "..")))

import numpy as np
import torch

from app.data.loader import list_egos, load_ego, to_graph_tensors
from app.seeding.cpm import k_clique_seeds
from app.seeding.weak_clique import weak_clique_seeds
from app.seeding.merge import merge_seeds, build_seed_matrix
from app.model.train import train_model, assign_communities
from app.baselines.run import run_leiden_louvain, run_gnn_no_seeds
from app.evaluation.metrics import evaluate_all

SEED = 42
DEFAULT = {
    "k": 4, "seeding": "cpm", "encoder_type": "sage", "structure_loss": "bp",
    "lambda_seed": 1.0, "tau": 0.5, "epochs": 150, "hidden_dim": 64, "lr": 0.01,
}
GAT_MAX_NODES = 800  # GAT here is O(N^2) memory; skip the largest networks
OUT_DIR = os.path.abspath(os.path.join(curr, "..", "experiments"))

_data_cache = {}
_seed_cache = {}


def get_data(ego):
    if ego not in _data_cache:
        G, node_list, node_to_idx, X, truth, feat_names = load_ego(ego)
        _data_cache[ego] = (G, node_list, node_to_idx, X, truth)
    return _data_cache[ego]


def get_seeds(ego, seeding, k):
    key = (ego, seeding, k)
    if key not in _seed_cache:
        G, node_list, node_to_idx, X, truth = get_data(ego)
        raw = weak_clique_seeds(G) if seeding == "weak_clique" else k_clique_seeds(G, k=k)
        merged = merge_seeds(raw, jaccard_threshold=0.6)
        K = max(3, min(len(truth) if truth else 10, len(merged), 20))
        S, _ = build_seed_matrix(merged, node_to_idx, K=K)
        _seed_cache[key] = (S, K)
    return _seed_cache[key]


def mean_metrics(rows):
    keys = ["onmi", "avg_f1", "omega", "bridge_f1"]
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    out["circles"] = round(float(np.mean([r["circles"] for r in rows])), 1)
    out["time_s"] = round(float(np.mean([r["time_s"] for r in rows])), 2)
    return out


def run_hybrid(cfg, taus):
    """Train once per ego, evaluate at each tau. Returns {tau: mean metrics}."""
    per_tau = {t: [] for t in taus}
    for ego in list_egos():
        G, node_list, node_to_idx, X, truth = get_data(ego)
        if cfg["encoder_type"] == "gat" and len(node_list) > GAT_MAX_NODES:
            continue
        S, K = get_seeds(ego, cfg["seeding"], cfg["k"])
        _, adj_dense, adj_norm, X_t, S_t = to_graph_tensors(G, X, S)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        F_np, _ = train_model(X_t, S_t, adj_norm, adj_dense, cfg)
        elapsed = time.time() - t0
        for t in taus:
            comms = assign_communities(F_np, G, node_list, tau=t)
            m = evaluate_all(comms, truth, node_list)
            m["circles"] = len(comms)
            m["time_s"] = elapsed
            per_tau[t].append(m)
    return {t: mean_metrics(r) for t, r in per_tau.items() if r}


def run_no_seeds(cfg, tau):
    rows = []
    for ego in list_egos():
        G, node_list, node_to_idx, X, truth = get_data(ego)
        _, K = get_seeds(ego, "cpm", cfg["k"])
        _, adj_dense, adj_norm, X_t, _ = to_graph_tensors(G, X, None)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t0 = time.time()
        comms, _ = run_gnn_no_seeds(X_t, adj_norm, adj_dense, G, node_list, K=K, tau=tau)
        m = evaluate_all(comms, truth, node_list)
        m["circles"] = len(comms)
        m["time_s"] = time.time() - t0
        rows.append(m)
    return mean_metrics(rows)


def run_louvain():
    rows = []
    for ego in list_egos():
        G, node_list, _, _, truth = get_data(ego)
        t0 = time.time()
        comms, _ = run_leiden_louvain(G)
        m = evaluate_all(comms, truth, node_list)
        m["circles"] = len(comms)
        m["time_s"] = time.time() - t0
        rows.append(m)
    return mean_metrics(rows)


def main():
    quick = "--quick" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []  # (section, label, metrics)

    def record(section, label, m):
        results.append((section, label, m))
        print(f"  {section:<10} {label:<34} ONMI {m['onmi']:.3f}  F1 {m['avg_f1']:.3f}  "
              f"Omega {m['omega']:.3f}  bridgeF1 {m['bridge_f1']:.3f}  circles {m['circles']}", flush=True)

    t_all = time.time()
    print("Reference baselines")
    record("reference", "Louvain (disjoint)", run_louvain())
    record("reference", "GNN without seeds", run_no_seeds(DEFAULT, DEFAULT["tau"]))

    print("Part 1: ablation (one factor changed from default)")
    record("default", "default", run_hybrid(DEFAULT, [DEFAULT["tau"]])[DEFAULT["tau"]])

    variants = [
        ("seeding", "weak-clique seeds", {"seeding": "weak_clique"}),
        ("k", "k = 3", {"k": 3}),
        ("k", "k = 5", {"k": 5}),
        ("loss", "soft modularity loss", {"structure_loss": "modularity"}),
        ("seed loss", "lambda_seed = 0 (no seed loss)", {"lambda_seed": 0.0}),
        ("seed loss", "lambda_seed = 0.5", {"lambda_seed": 0.5}),
        ("seed loss", "lambda_seed = 2", {"lambda_seed": 2.0}),
        ("seed loss", "lambda_seed = 5", {"lambda_seed": 5.0}),
        ("epochs", "epochs = 50", {"epochs": 50}),
        ("epochs", "epochs = 300", {"epochs": 300}),
        ("encoder", "GAT encoder (N <= 800)", {"encoder_type": "gat"}),
    ]
    if quick:
        variants = variants[:3]
    for section, label, change in variants:
        cfg = {**DEFAULT, **change}
        record(section, label, run_hybrid(cfg, [cfg["tau"]])[cfg["tau"]])

    print("Part 1b: threshold tau (default model, no retraining)")
    taus = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for t, m in run_hybrid(DEFAULT, taus).items():
        record("tau", f"tau = {t}", m)

    print("Part 2: grid search")
    grid = {
        "lambda_seed": [0.5, 1.0, 2.0],
        "epochs": [100, 150, 300],
        "k": [3, 4],
        "seeding": ["cpm", "weak_clique"],
    }
    if quick:
        grid = {"lambda_seed": [1.0], "epochs": [150], "k": [4], "seeding": ["cpm"]}
    grid_taus = [0.2, 0.3, 0.5, 0.7]
    best = None
    for vals in itertools.product(*grid.values()):
        change = dict(zip(grid.keys(), vals))
        if change["seeding"] == "weak_clique" and change["k"] != 4:
            continue  # k is unused for weak-clique seeds
        cfg = {**DEFAULT, **change}
        for t, m in run_hybrid(cfg, grid_taus).items():
            label = (f"{change['seeding']} k={change['k']} lam={change['lambda_seed']} "
                     f"ep={change['epochs']} tau={t}")
            results.append(("grid", label, m))
            score = m["onmi"] + m["avg_f1"] + m["omega"]
            if best is None or score > best[0]:
                best = (score, label, m)
    print(f"  best grid setting (sum of ONMI+F1+Omega): {best[1]}")
    print(f"    {best[2]}")

    # Write outputs
    cols = ["section", "setting", "onmi", "avg_f1", "omega", "bridge_f1", "circles", "time_s"]
    with open(os.path.join(OUT_DIR, "ablation_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sec, label, m in results:
            w.writerow([sec, label, m["onmi"], m["avg_f1"], m["omega"], m["bridge_f1"], m["circles"], m["time_s"]])

    def table(rows):
        lines = ["| Setting | ONMI | Avg F1 | Omega | Bridge F1 | Circles | Train s |",
                 "|---|---|---|---|---|---|---|"]
        for _, label, m in rows:
            lines.append(f"| {label} | {m['onmi']} | {m['avg_f1']} | {m['omega']} | "
                         f"{m['bridge_f1']} | {m['circles']} | {m['time_s']} |")
        return "\n".join(lines)

    grid_rows = sorted([r for r in results if r[0] == "grid"],
                       key=lambda r: -(r[2]["onmi"] + r[2]["avg_f1"] + r[2]["omega"]))[:10]
    md = [
        "# Ablation and tuning results",
        "",
        "Mean over all ego-networks, one run per setting (seed 42). Differences of about 0.01-0.02 "
        "are within run-to-run noise and should not be over-interpreted.",
        "",
        "## Reference and ablation (one factor changed from default)",
        "",
        table([r for r in results if r[0] in ("reference", "default", "seeding", "k", "loss",
                                              "seed loss", "epochs", "encoder")]),
        "",
        "## Threshold tau (default model)",
        "",
        table([r for r in results if r[0] == "tau"]),
        "",
        "## Top 10 grid-search settings (ranked by ONMI + F1 + Omega)",
        "",
        table(grid_rows),
        "",
        f"Total runtime: {round(time.time() - t_all)} s",
    ]
    with open(os.path.join(OUT_DIR, "ablation_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote results to {OUT_DIR} in {round(time.time() - t_all)} s")


if __name__ == "__main__":
    main()
