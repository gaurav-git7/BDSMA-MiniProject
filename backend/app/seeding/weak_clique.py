import math
from typing import List, Set, Dict, Any, Tuple
import networkx as nx


def salton_similarity(G: nx.Graph, u: str, v: str) -> float:
    """Compute Salton cosine similarity between nodes u and v."""
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    common = len(nu.intersection(nv))
    denom = math.sqrt(len(nu) * len(nv))
    if denom == 0:
        return 0.0
    return common / denom


def node_priority(G: nx.Graph) -> List[Tuple[str, float]]:
    """
    Compute density-based priority for nodes:
    priority(u) = degree(u) * (clustering_coeff(u) + 0.1)
    """
    clustering = nx.clustering(G)
    degrees = dict(G.degree())
    scores = []
    for node, deg in degrees.items():
        score = deg * (clustering.get(node, 0.0) + 0.1)
        scores.append((node, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def weak_clique_seeds(G: nx.Graph, min_size: int = 3) -> List[Dict[str, Any]]:
    """
    Extract weak-clique seeds (Paper #26 / WOCD).
    For a linked pair u, v, weak clique is {u, v} U (N(u) cap N(v)).
    """
    priorities = node_priority(G)
    covered: Set[str] = set()
    seeds: List[Dict[str, Any]] = []

    for u, _ in priorities:
        if u in covered:
            continue
        neighbors = list(G.neighbors(u))
        if not neighbors:
            continue

        # Find best neighbor v with highest Salton similarity
        best_v = None
        best_sim = -1.0
        for v in neighbors:
            sim = salton_similarity(G, u, v)
            if sim > best_sim:
                best_sim = sim
                best_v = v

        if best_v is None:
            continue

        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        common = nu.intersection(nv)
        weak_clique = {u, best_v}.union(common)

        if len(weak_clique) >= min_size:
            seeds.append({
                "members": weak_clique,
                "cliques": [[u, best_v] + list(common)[:3]],
                "size": len(weak_clique),
                "type": "weak_clique"
            })
            covered.update(weak_clique)

    seeds.sort(key=lambda x: x["size"], reverse=True)
    return seeds
