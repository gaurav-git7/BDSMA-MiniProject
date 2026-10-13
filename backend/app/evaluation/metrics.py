import math
from typing import Dict, List, Set, Tuple, Any
from itertools import combinations
import numpy as np
import networkx as nx


def _f1_pair(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    common = len(set_a.intersection(set_b))
    precision = common / len(set_a)
    recall = common / len(set_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_f1(pred_dict: Dict[str, List[str]], truth_dict: Dict[str, List[str]]) -> float:
    """
    Compute average F1 score between predicted and ground-truth communities
    with bidirectional best-match pairing (standard in SNAP ego-Facebook benchmarks).
    """
    pred_sets = [set(v) for v in pred_dict.values() if len(v) > 0]
    truth_sets = [set(v) for v in truth_dict.values() if len(v) > 0]

    if not pred_sets or not truth_sets:
        return 0.0

    # Match each truth community to best pred community
    f1_truth_to_pred = []
    for t in truth_sets:
        best_f1 = max((_f1_pair(t, p) for p in pred_sets), default=0.0)
        f1_truth_to_pred.append(best_f1)

    # Match each pred community to best truth community
    f1_pred_to_truth = []
    for p in pred_sets:
        best_f1 = max((_f1_pair(p, t) for t in truth_sets), default=0.0)
        f1_pred_to_truth.append(best_f1)

    score = 0.5 * (np.mean(f1_truth_to_pred) + np.mean(f1_pred_to_truth))
    return float(score)


def overlapping_nmi(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
    all_nodes: List[str]
) -> float:
    """
    Overlapping Normalized Mutual Information (ONMI) based on Lancichinetti et al.
    For each community pair, computes Shannon mutual information over binary membership.
    Normalized by average entropy.
    """
    pred_sets = [set(v) for v in pred_dict.values() if len(v) > 0]
    truth_sets = [set(v) for v in truth_dict.values() if len(v) > 0]

    if not pred_sets or not truth_sets or not all_nodes:
        return 0.0

    N = len(all_nodes)

    def binary_entropy(s: Set[str]) -> float:
        p1 = len(s) / N
        p0 = 1.0 - p1
        h = 0.0
        if p1 > 0:
            h -= p1 * math.log2(p1)
        if p0 > 0:
            h -= p0 * math.log2(p0)
        return h

    def mi_pair(s1: Set[str], s2: Set[str]) -> float:
        # 2x2 contingency table
        n11 = len(s1.intersection(s2))
        n10 = len(s1) - n11
        n01 = len(s2) - n11
        n00 = N - (n11 + n10 + n01)

        counts = [n11, n10, n01, n00]
        p_row1 = len(s1) / N
        p_row0 = 1.0 - p_row1
        p_col1 = len(s2) / N
        p_col0 = 1.0 - p_col1

        marginals = [
            p_row1 * p_col1,
            p_row1 * p_col0,
            p_row0 * p_col1,
            p_row0 * p_col0
        ]

        mi = 0.0
        for c, m in zip(counts, marginals):
            if c > 0 and m > 0:
                p_joint = c / N
                mi += p_joint * math.log2(p_joint / m)
        return max(0.0, mi)

    h_truth = sum(binary_entropy(t) for t in truth_sets)
    h_pred = sum(binary_entropy(p) for p in pred_sets)

    if h_truth == 0.0 or h_pred == 0.0:
        return 0.0

    # Best-match mutual information sum
    mi_truth = sum(max(mi_pair(t, p) for p in pred_sets) for t in truth_sets)
    mi_pred = sum(max(mi_pair(p, t) for t in truth_sets) for p in pred_sets)

    total_mi = 0.5 * (mi_truth + mi_pred)
    norm = 0.5 * (h_truth + h_pred)

    onmi_val = total_mi / norm if norm > 0 else 0.0
    return float(np.clip(onmi_val, 0.0, 1.0))


def omega_index(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
    all_nodes: List[str],
    sample_limit: int = 1000
) -> float:
    """
    Omega Index measures agreement between two overlapping clusterings
    based on the number of shared communities for each pair of nodes.
    Omega = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    """
    pred_sets = [set(v) for v in pred_dict.values()]
    truth_sets = [set(v) for v in truth_dict.values()]

    nodes = list(all_nodes)
    if len(nodes) > sample_limit:
        # Sample to keep computation fast on very large networks
        np.random.seed(42)
        nodes = list(np.random.choice(nodes, size=sample_limit, replace=False))

    N = len(nodes)
    if N < 2:
        return 1.0

    pairs_total = N * (N - 1) // 2

    # Map each node to its community indices
    pred_map = {n: set() for n in nodes}
    for c_idx, p in enumerate(pred_sets):
        for n in p:
            if n in pred_map:
                pred_map[n].add(c_idx)

    truth_map = {n: set() for n in nodes}
    for c_idx, t in enumerate(truth_sets):
        for n in t:
            if n in truth_map:
                truth_map[n].add(c_idx)

    matches = 0
    t_counts = {}
    p_counts = {}

    for i in range(N):
        u = nodes[i]
        u_p = pred_map[u]
        u_t = truth_map[u]
        for j in range(i + 1, N):
            v = nodes[j]
            k_pred = len(u_p.intersection(pred_map[v]))
            k_truth = len(u_t.intersection(truth_map[v]))

            p_counts[k_pred] = p_counts.get(k_pred, 0) + 1
            t_counts[k_truth] = t_counts.get(k_truth, 0) + 1

            if k_pred == k_truth:
                matches += 1

    obs = matches / pairs_total

    # Expected agreement by chance
    exp = 0.0
    all_k = set(p_counts.keys()).union(t_counts.keys())
    for k in all_k:
        cp = p_counts.get(k, 0) / pairs_total
        ct = t_counts.get(k, 0) / pairs_total
        exp += cp * ct

    if exp >= 1.0:
        return 1.0
    omega = (obs - exp) / (1.0 - exp)
    return float(np.clip(omega, -1.0, 1.0))


def bridge_node_scores(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluate detection of bridge (overlapping) nodes.
    A node is a bridge if it belongs to >= 2 communities.
    """
    pred_node_counts = {}
    for members in pred_dict.values():
        for n in members:
            pred_node_counts[n] = pred_node_counts.get(n, 0) + 1

    truth_node_counts = {}
    for members in truth_dict.values():
        for n in members:
            truth_node_counts[n] = truth_node_counts.get(n, 0) + 1

    pred_bridges = {n for n, cnt in pred_node_counts.items() if cnt >= 2}
    truth_bridges = {n for n, cnt in truth_node_counts.items() if cnt >= 2}

    if not truth_bridges and not pred_bridges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    common = len(pred_bridges.intersection(truth_bridges))
    prec = common / len(pred_bridges) if pred_bridges else 0.0
    rec = common / len(truth_bridges) if truth_bridges else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_bridges_count": len(truth_bridges),
        "pred_bridges_count": len(pred_bridges)
    }


def community_quality(G: nx.Graph, members: List[str]) -> Dict[str, float]:
    """
    Compute internal density and conductance of a community.
    """
    k = len(members)
    if k <= 1:
        return {"density": 0.0, "conductance": 0.0, "size": k}

    member_set = set(members)
    subG = G.subgraph(member_set)
    in_edges = subG.number_of_edges()
    max_possible = k * (k - 1) / 2
    density = (in_edges / max_possible) if max_possible > 0 else 0.0

    cut_edges = 0
    for u in member_set:
        if u in G:
            for v in G.neighbors(u):
                if v not in member_set:
                    cut_edges += 1

    vol = (2 * in_edges) + cut_edges
    conductance = (cut_edges / vol) if vol > 0 else 0.0

    return {
        "density": round(float(density), 4),
        "conductance": round(float(conductance), 4),
        "size": k
    }


def evaluate_all(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
    all_nodes: List[str]
) -> Dict[str, Any]:
    """
    Run full metric suite: ONMI, Average F1, Omega Index, Bridge scores.
    """
    f1 = average_f1(pred_dict, truth_dict)
    nmi = overlapping_nmi(pred_dict, truth_dict, all_nodes)
    omega = omega_index(pred_dict, truth_dict, all_nodes)
    bridge_metrics = bridge_node_scores(pred_dict, truth_dict)

    return {
        "avg_f1": round(f1, 4),
        "onmi": round(nmi, 4),
        "omega": round(omega, 4),
        "bridge_precision": bridge_metrics["precision"],
        "bridge_recall": bridge_metrics["recall"],
        "bridge_f1": bridge_metrics["f1"]
    }
