import time
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np
import torch

from ..seeding.cpm import k_clique_seeds
from ..model.train import train_model, assign_communities


def run_cpm(G: nx.Graph, k: int = 4) -> Tuple[Dict[str, List[str]], float]:
    """
    Run baseline Clique Percolation Method (CPM).
    Returns (communities_dict, runtime_sec).
    """
    t0 = time.time()
    raw_seeds = k_clique_seeds(G, k=k)
    communities = {}
    for i, s in enumerate(raw_seeds):
        communities[f"circle_{i}"] = sorted(list(s["members"]))
    elapsed = time.time() - t0
    return communities, round(elapsed, 4)


def run_leiden_louvain(G: nx.Graph) -> Tuple[Dict[str, List[str]], float]:
    """
    Run baseline disjoint community detection (Louvain/Leiden modularity).
    Returns (communities_dict, runtime_sec).
    """
    t0 = time.time()
    try:
        raw_comms = nx.community.louvain_communities(G, seed=42)
    except Exception:
        raw_comms = list(nx.connected_components(G))

    communities = {}
    for i, c in enumerate(raw_comms):
        communities[f"circle_{i}"] = sorted(list(c))
    elapsed = time.time() - t0
    return communities, round(elapsed, 4)


def run_bigclam(
    G: nx.Graph,
    node_list: List[str],
    K: int = 10,
    iterations: int = 50,
    lr: float = 0.05
) -> Tuple[Dict[str, List[str]], float]:
    """
    Run baseline BigCLAM (Overlapping Community Detection via Non-negative Factorization).
    Models P(u,v) = 1 - exp(- F_u . F_v).
    """
    t0 = time.time()
    N = len(node_list)
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    # Initialize F randomly >= 0
    np.random.seed(42)
    F_mat = np.random.uniform(0.01, 0.1, size=(N, K)).astype(np.float32)

    # Pre-extract adjacency list for fast updates
    adj_set = {i: set() for i in range(N)}
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            ui, vi = node_to_idx[u], node_to_idx[v]
            adj_set[ui].add(vi)
            adj_set[vi].add(ui)

    # Simplified gradient descent update for BigCLAM
    # grad_u = sum_{v in N(u)} (F_v * exp(-F_u . F_v) / (1 - exp(-F_u . F_v))) - sum_{v not in N(u)} F_v
    for _ in range(iterations):
        sum_F = np.sum(F_mat, axis=0)  # (K,)
        for u in range(N):
            neighs = list(adj_set[u])
            if not neighs:
                continue

            F_u = F_mat[u]
            F_neigh = F_mat[neighs]  # (|N|, K)
            dots = np.dot(F_neigh, F_u)
            dots = np.clip(dots, 1e-4, 15.0)

            # Positive gradient from neighbors
            factors = np.exp(-dots) / (1.0 - np.exp(-dots) + 1e-6)
            factors = np.clip(factors, 0.0, 1e4)
            pos_grad = np.sum(F_neigh * factors[:, None], axis=0)

            # Negative gradient: sum_{v not in N(u)} F_v = sum_F - F_u - sum_{v in N(u)} F_v
            neg_grad = sum_F - F_u - np.sum(F_neigh, axis=0)

            grad = pos_grad - neg_grad
            # Gradient ascent step with safe clipping
            F_mat[u] = np.clip(F_u + lr * (grad / (len(neighs) + 1.0)), 0.0, 5.0)

    # Threshold membership
    tau = np.sqrt(-np.log(1.0 - 0.5)) if False else 0.2
    communities = {}
    for c in range(K):
        members = [node_list[i] for i in range(N) if F_mat[i, c] >= tau]
        if members:
            communities[f"circle_{c}"] = members

    # If any node unassigned, give to highest column
    assigned_nodes = set().union(*communities.values()) if communities else set()
    unassigned = [node_list[i] for i in range(N) if node_list[i] not in assigned_nodes]
    for un in unassigned:
        best_c = int(np.argmax(F_mat[node_to_idx[un]]))
        c_name = f"circle_{best_c}"
        if c_name not in communities:
            communities[c_name] = []
        communities[c_name].append(un)

    elapsed = time.time() - t0
    return communities, round(elapsed, 4)


def run_gnn_no_seeds(
    X_t: torch.Tensor,
    adj_norm: torch.Tensor,
    adj_dense: torch.Tensor,
    G: nx.Graph,
    node_list: List[str],
    K: int = 10,
    tau: float = 0.5
) -> Tuple[Dict[str, List[str]], float]:
    """
    Run ablation baseline: GNN without clique seeds (pure NOCD-style).
    """
    t0 = time.time()
    cfg = {
        "encoder_type": "sage",
        "hidden_dim": 64,
        "lr": 0.01,
        "epochs": 150,
        "structure_loss": "bp",
        "lambda_seed": 0.0,
    }
    # Pass dummy zero seed or None
    S_dummy = torch.zeros((X_t.size(0), K), dtype=torch.float32)
    F_np, _ = train_model(X_t, S_dummy, adj_norm, adj_dense, cfg)
    comms = assign_communities(F_np, G, node_list, tau=tau)
    elapsed = time.time() - t0
    return comms, round(elapsed, 4)
