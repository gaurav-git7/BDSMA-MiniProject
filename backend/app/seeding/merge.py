from typing import List, Dict, Any, Tuple
import numpy as np


def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def merge_seeds(seeds: List[Dict[str, Any]], jaccard_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Merge seeds with Jaccard overlap >= jaccard_threshold.
    """
    if not seeds:
        return []

    merged = [dict(s) for s in seeds]
    changed = True

    while changed:
        changed = False
        new_merged = []
        skip_indices = set()

        for i in range(len(merged)):
            if i in skip_indices:
                continue

            current = merged[i]
            cur_members = set(current["members"])
            cur_cliques = list(current.get("cliques", []))

            for j in range(i + 1, len(merged)):
                if j in skip_indices:
                    continue
                other = merged[j]
                other_members = set(other["members"])

                sim = jaccard_similarity(cur_members, other_members)
                if sim >= jaccard_threshold:
                    # Merge j into current
                    cur_members.update(other_members)
                    cur_cliques.extend(other.get("cliques", []))
                    skip_indices.add(j)
                    changed = True

            current["members"] = cur_members
            current["cliques"] = cur_cliques[:10]  # cap stored cliques
            current["size"] = len(cur_members)
            new_merged.append(current)

        merged = new_merged

    merged.sort(key=lambda x: x["size"], reverse=True)
    return merged


def build_seed_matrix(
    seeds: List[Dict[str, Any]],
    node_to_idx: Dict[str, int],
    K: int = 10
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Keep the top K largest seeds and build binary seed matrix S (N x K).
    Returns (S, top_seeds).
    """
    N = len(node_to_idx)
    top_seeds = seeds[:K]
    actual_K = max(1, len(top_seeds))
    S = np.zeros((N, actual_K), dtype=np.float32)

    for c_idx, s in enumerate(top_seeds):
        for node in s["members"]:
            if node in node_to_idx:
                S[node_to_idx[node], c_idx] = 1.0

    return S, top_seeds
