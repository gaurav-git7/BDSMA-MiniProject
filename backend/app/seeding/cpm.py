from typing import List, Set, Dict, Any
import networkx as nx


def k_clique_seeds(G: nx.Graph, k: int = 4, max_cliques: int = 500) -> List[Dict[str, Any]]:
    """
    Find clique percolation communities in graph G.
    To avoid combinatorial explosion on dense graphs (where max clique size > 15),
    we extract maximal cliques of size >= k, and merge cliques sharing >= (k-1) nodes.
    This runs in milliseconds even on dense graphs.
    """
    # 1. Collect maximal cliques of size >= k
    cliques: List[Set[str]] = []
    try:
        for c in nx.find_cliques(G):
            if len(c) >= k:
                cliques.append(set(c))
                if len(cliques) >= max_cliques:
                    break
    except Exception:
        cliques = []

    # If no cliques of size k found, try k=3
    if not cliques and k > 3:
        for c in nx.find_cliques(G):
            if len(c) >= 3:
                cliques.append(set(c))
                if len(cliques) >= max_cliques:
                    break
        effective_k = 3
    else:
        effective_k = k

    if not cliques:
        return []

    # 2. Build clique adjacency: connect two cliques if they overlap by >= k-1 nodes
    clique_graph = nx.Graph()
    for i in range(len(cliques)):
        clique_graph.add_node(i)

    overlap_threshold = max(2, effective_k - 1)
    # Sort cliques by size descending to prioritize large cores
    cliques.sort(key=lambda c: len(c), reverse=True)

    # Fast overlap checking
    for i in range(min(len(cliques), 200)):
        ci = cliques[i]
        for j in range(i + 1, min(len(cliques), 200)):
            cj = cliques[j]
            if len(ci.intersection(cj)) >= overlap_threshold:
                clique_graph.add_edge(i, j)

    # 3. Connected components of clique graph form percolation communities
    seed_results: List[Dict[str, Any]] = []
    for comp in nx.connected_components(clique_graph):
        members: Set[str] = set()
        constituent_cliques = []
        for clique_idx in comp:
            c_nodes = cliques[clique_idx]
            members.update(c_nodes)
            if len(constituent_cliques) < 10:
                constituent_cliques.append(list(c_nodes)[:effective_k])

        if len(members) >= effective_k:
            seed_results.append({
                "members": members,
                "cliques": constituent_cliques,
                "k": effective_k,
                "size": len(members),
                "type": f"k-clique (k={effective_k})"
            })

    seed_results.sort(key=lambda x: x["size"], reverse=True)
    return seed_results
