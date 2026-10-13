from typing import List, Dict, Any, Optional
import networkx as nx
import numpy as np


def suggest_friends(
    F_np: np.ndarray,
    G: nx.Graph,
    user_id: str,
    node_list: List[str],
    node_to_idx: Dict[str, int],
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Suggest new friends to user_id based on shared-circle affinity (F_u . F_v).
    Excludes existing friends and the user themself.
    """
    if user_id not in node_to_idx:
        return []

    u_idx = node_to_idx[user_id]
    u_vec = F_np[u_idx]

    # Current neighbors
    existing_friends = set(G.neighbors(user_id)) if user_id in G else set()
    existing_friends.add(user_id)

    suggestions = []
    N, K = F_np.shape

    for v_nid in node_list:
        if v_nid in existing_friends:
            continue
        v_idx = node_to_idx[v_nid]
        v_vec = F_np[v_idx]

        # Dot product affinity
        dot_score = float(np.dot(u_vec, v_vec))
        if dot_score <= 1e-4:
            continue

        # Detail shared circles
        shared_circles = []
        for c in range(K):
            contrib = float(u_vec[c] * v_vec[c])
            if contrib > 0.01:
                shared_circles.append({
                    "circle_id": c,
                    "circle_name": f"circle_{c}",
                    "affinity_contribution": round(contrib, 4),
                    "user_score": round(float(u_vec[c]), 4),
                    "friend_score": round(float(v_vec[c]), 4)
                })

        shared_circles.sort(key=lambda x: x["affinity_contribution"], reverse=True)

        suggestions.append({
            "target_node": v_nid,
            "affinity_score": round(dot_score, 4),
            "shared_circles": shared_circles,
            "common_neighbors_count": len(existing_friends.intersection(set(G.neighbors(v_nid)))) if v_nid in G else 0
        })

    suggestions.sort(key=lambda x: x["affinity_score"], reverse=True)
    return suggestions[:top_n]


def evaluate_suggestions(
    G: nx.Graph,
    F_np: np.ndarray,
    node_list: List[str],
    node_to_idx: Dict[str, int],
    hide_ratio: float = 0.1,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Evaluate recommendation quality by hiding 10% of edges and measuring
    Hit-Rate@K against a standard Common-Neighbors baseline.
    """
    edges = list(G.edges())
    num_hide = int(len(edges) * hide_ratio)
    if num_hide == 0:
        return {"hybrid_hit_rate": 0.0, "cn_hit_rate": 0.0}

    np.random.seed(42)
    hide_idx = np.random.choice(len(edges), size=num_hide, replace=False)
    hidden_edges = set(edges[i] for i in hide_idx)

    # Subgraph without hidden edges
    G_train = G.copy()
    G_train.remove_edges_from(hidden_edges)

    hits_hybrid = 0
    hits_cn = 0
    test_nodes = set()
    for u, v in hidden_edges:
        test_nodes.add(u)

    eval_count = 0
    for u in list(test_nodes)[:30]:  # evaluate on sample for speed
        if u not in node_to_idx:
            continue
        eval_count += 1
        true_hidden_neighbors = {v for v in G.neighbors(u) if (u, v) in hidden_edges or (v, u) in hidden_edges}

        # Hybrid predictions
        recs_hybrid = suggest_friends(F_np, G_train, u, node_list, node_to_idx, top_n=top_k)
        pred_nodes_hybrid = {r["target_node"] for r in recs_hybrid}
        if true_hidden_neighbors.intersection(pred_nodes_hybrid):
            hits_hybrid += 1

        # Common neighbors baseline
        cn_scores = []
        u_neighs = set(G_train.neighbors(u)) if u in G_train else set()
        for v in node_list:
            if v != u and not G_train.has_edge(u, v):
                v_neighs = set(G_train.neighbors(v)) if v in G_train else set()
                cn_scores.append((v, len(u_neighs.intersection(v_neighs))))
        cn_scores.sort(key=lambda x: x[1], reverse=True)
        top_cn_nodes = {x[0] for x in cn_scores[:top_k]}
        if true_hidden_neighbors.intersection(top_cn_nodes):
            hits_cn += 1

    return {
        "hybrid_hit_rate": round(hits_hybrid / max(1, eval_count), 4),
        "cn_hit_rate": round(hits_cn / max(1, eval_count), 4),
        "evaluated_nodes": eval_count
    }
