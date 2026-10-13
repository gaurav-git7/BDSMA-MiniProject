from typing import List, Dict, Any
import numpy as np


def find_bridge_friends(
    F_np: np.ndarray,
    node_list: List[str],
    tau: float = 0.5,
    top_n: int = 50
) -> List[Dict[str, Any]]:
    """
    Identify and rank bridge friends who belong to two or more circles.
    Ranked by number of circles, then by aggregate membership strength.
    """
    N, K = F_np.shape
    bridges = []

    for i in range(N):
        node_id = node_list[i]
        scores = F_np[i]

        active_circles = []
        for c in range(K):
            if scores[c] >= tau:
                active_circles.append({
                    "circle_id": c,
                    "circle_name": f"circle_{c}",
                    "score": round(float(scores[c]), 4)
                })

        # Multi-circle members
        if len(active_circles) >= 2:
            total_strength = float(np.sum([c["score"] for c in active_circles]))
            bridges.append({
                "node_id": node_id,
                "node_index": i,
                "circle_count": len(active_circles),
                "total_strength": round(total_strength, 4),
                "circles": active_circles
            })

    # Sort descending by circle_count, then total_strength
    bridges.sort(key=lambda x: (x["circle_count"], x["total_strength"]), reverse=True)
    return bridges[:top_n]
