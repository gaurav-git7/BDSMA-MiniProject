import os
import json
import time
from typing import Dict, Any, Optional
import numpy as np

from .data.loader import load_ego, to_graph_tensors, get_data_dir
from .seeding.cpm import k_clique_seeds
from .seeding.weak_clique import weak_clique_seeds
from .seeding.merge import merge_seeds, build_seed_matrix
from .model.train import train_model, assign_communities
from .baselines.run import run_cpm, run_leiden_louvain, run_bigclam, run_gnn_no_seeds
from .evaluation.metrics import evaluate_all, community_quality
from .analysis.bridges import find_bridge_friends
from .analysis.recommend import suggest_friends, evaluate_suggestions
from .analysis.explain import explain_circle


def get_results_dir() -> str:
    curr = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(curr, "..", "results"),
        os.path.join(curr, "..", "..", "results"),
        os.path.join(os.getcwd(), "backend", "results"),
        os.path.join(os.getcwd(), "results"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    target = os.path.abspath(candidates[0])
    os.makedirs(target, exist_ok=True)
    return target


def run_pipeline(
    ego_id: int,
    config: Optional[Dict[str, Any]] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run full CircleSense pipeline for an ego-network:
    1. Load data
    2. Seeding (CPM or Weak-clique)
    3. GNN training & soft-membership output
    4. Community assignment & bridge detection
    5. Baselines execution
    6. Full metric evaluation
    7. Recommendation & explanation generation
    8. Serialize results
    """
    t_start = time.time()
    cfg = {
        "k": 4,
        "seeding": "cpm",  # 'cpm' or 'weak_clique'
        "encoder_type": "sage",  # 'sage' or 'gat'
        "hidden_dim": 64,
        "lr": 0.01,
        "epochs": 150,
        "structure_loss": "bp",  # 'bp' or 'modularity'
        "lambda_seed": 1.0,
        "tau": 0.5,
        "jaccard_merge": 0.6,
        "max_seeds": 20
    }
    if config:
        cfg.update(config)

    # 1. Load data
    G, node_list, node_to_idx, X, ground_truth, feat_names = load_ego(ego_id)
    N = len(node_list)

    # 2. Seeding
    t_seed_start = time.time()
    if cfg["seeding"] == "weak_clique":
        raw_seeds = weak_clique_seeds(G)
    else:
        raw_seeds = k_clique_seeds(G, k=cfg["k"])

    merged_seeds = merge_seeds(raw_seeds, jaccard_threshold=cfg["jaccard_merge"])

    # Determine K: use number of ground-truth circles or max_seeds
    K = max(3, min(len(ground_truth) if ground_truth else 10, len(merged_seeds), cfg["max_seeds"]))
    S_np, top_seeds = build_seed_matrix(merged_seeds, node_to_idx, K=K)
    seed_time = round(time.time() - t_seed_start, 4)

    # 3. Graph tensors
    edge_index, adj_dense, adj_norm, X_t, S_t = to_graph_tensors(G, X, S_np)

    # 4. GNN Training
    F_np, train_meta = train_model(X_t, S_t, adj_norm, adj_dense, cfg)

    # 5. Community Assignment
    hybrid_comms = assign_communities(F_np, G, node_list, tau=cfg["tau"])

    # 6. Baselines
    cpm_comms, cpm_time = run_cpm(G, k=cfg["k"])
    leiden_comms, leiden_time = run_leiden_louvain(G)
    bigclam_comms, bigclam_time = run_bigclam(G, node_list, K=K)
    gnn_noseed_comms, gnn_noseed_time = run_gnn_no_seeds(
        X_t, adj_norm, adj_dense, G, node_list, K=K, tau=cfg["tau"]
    )

    # 7. Metrics evaluation on ground truth
    metrics_hybrid = evaluate_all(hybrid_comms, ground_truth, node_list)
    metrics_cpm = evaluate_all(cpm_comms, ground_truth, node_list)
    metrics_leiden = evaluate_all(leiden_comms, ground_truth, node_list)
    metrics_bigclam = evaluate_all(bigclam_comms, ground_truth, node_list)
    metrics_gnn_noseed = evaluate_all(gnn_noseed_comms, ground_truth, node_list)

    metrics_comparison = {
        "hybrid": {**metrics_hybrid, "runtime": train_meta["train_time_sec"] + seed_time},
        "cpm": {**metrics_cpm, "runtime": cpm_time},
        "leiden": {**metrics_leiden, "runtime": leiden_time},
        "bigclam": {**metrics_bigclam, "runtime": bigclam_time},
        "gnn_no_seeds": {**metrics_gnn_noseed, "runtime": gnn_noseed_time}
    }

    # 8. Analysis: Bridges & Explanations
    bridges = find_bridge_friends(F_np, node_list, tau=cfg["tau"], top_n=30)

    explanations = {}
    for c_idx in range(K):
        c_key = f"circle_{c_idx}"
        if c_key in hybrid_comms:
            explanations[c_key] = explain_circle(
                c_idx, top_seeds, G, hybrid_comms, X, node_to_idx, feat_names
            )

    # Recommendation evaluations
    rec_eval = evaluate_suggestions(G, F_np, node_list, node_to_idx, hide_ratio=0.1)

    # Graph visualization format (Cytoscape.js)
    # Map each node to its circles across methods
    node_circles_map = {n: [] for n in node_list}
    for c_name, members in hybrid_comms.items():
        for m in members:
            node_circles_map[m].append(c_name)

    bridge_set = {b["node_id"] for b in bridges}

    cytoscape_elements = []
    degrees = dict(G.degree())
    for nid in node_list:
        assigned_c = node_circles_map.get(nid, [])
        cytoscape_elements.append({
            "data": {
                "id": str(nid),
                "label": f"Friend {nid}",
                "degree": degrees.get(nid, 0),
                "circles": assigned_c,
                "is_bridge": nid in bridge_set,
                "circle_count": len(assigned_c)
            }
        })

    for u, v in G.edges():
        cytoscape_elements.append({
            "data": {
                "id": f"e_{u}_{v}",
                "source": str(u),
                "target": str(v)
            }
        })

    # Summary payload
    results_payload = {
        "ego_id": ego_id,
        "config": cfg,
        "graph_summary": {
            "num_nodes": N,
            "num_edges": G.number_of_edges(),
            "num_ground_truth_circles": len(ground_truth),
            "num_detected_circles": len(hybrid_comms),
            "num_bridges": len(bridges),
            "features_dim": X.shape[1]
        },
        "metrics": metrics_comparison,
        "communities": {
            "hybrid": hybrid_comms,
            "cpm": cpm_comms,
            "leiden": leiden_comms,
            "bigclam": bigclam_comms,
            "ground_truth": ground_truth
        },
        "bridges": bridges,
        "explanations": explanations,
        "recommendation_eval": rec_eval,
        "cytoscape": cytoscape_elements,
        "node_list": node_list,
        "total_pipeline_time_sec": round(time.time() - t_start, 4)
    }

    # Save to disk
    if save_results:
        out_dir = get_results_dir()
        json_path = os.path.join(out_dir, f"ego_{ego_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)

        # Save numpy matrix F
        npz_path = os.path.join(out_dir, f"ego_{ego_id}_membership.npz")
        np.savez_compressed(npz_path, F=F_np, node_list=node_list)

    return results_payload
