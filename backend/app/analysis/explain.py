from typing import Dict, List, Any, Optional
import networkx as nx
import numpy as np

from ..evaluation.metrics import community_quality


def explain_circle(
    circle_idx: int,
    top_seeds: List[Dict[str, Any]],
    G: nx.Graph,
    communities: Dict[str, List[str]],
    X: Optional[np.ndarray] = None,
    node_to_idx: Optional[Dict[str, int]] = None,
    feat_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Explain a learned circle:
    - Which clique seed it originated from
    - The constituent cliques
    - Density, conductance, and size
    - Most shared/distinctive profile features
    """
    circle_key = f"circle_{circle_idx}"
    members = communities.get(circle_key, [])

    # Retrieve corresponding seed if within range
    seed_info = {}
    if circle_idx < len(top_seeds):
        s = top_seeds[circle_idx]
        seed_info = {
            "seed_id": circle_idx,
            "seed_members": sorted(list(s.get("members", []))),
            "seed_size": s.get("size", len(s.get("members", []))),
            "constituent_cliques": s.get("cliques", []),
            "seed_type": s.get("type", f"k-clique (k={s.get('k', 4)})")
        }

    # Graph quality metrics
    quality = community_quality(G, members)

    # Feature prominence
    top_features = []
    if X is not None and node_to_idx is not None and len(members) > 0:
        member_indices = [node_to_idx[m] for m in members if m in node_to_idx]
        if member_indices:
            sub_X = X[member_indices]
            # Frequency in this circle vs overall
            circle_freq = np.mean(sub_X, axis=0)
            overall_freq = np.mean(X, axis=0)
            # Distinctiveness ratio: circle_freq - overall_freq
            distinct = circle_freq - overall_freq
            best_feat_idx = np.argsort(distinct)[::-1][:5]

            for fid in best_feat_idx:
                if circle_freq[fid] > 0.05:  # present in at least 5% of circle
                    fname = feat_names[fid] if (feat_names and fid < len(feat_names)) else f"Feature #{fid}"
                    top_features.append({
                        "feature_id": int(fid),
                        "name": fname,
                        "circle_frequency": round(float(circle_freq[fid]), 3),
                        "overall_frequency": round(float(overall_freq[fid]), 3),
                        "distinctiveness": round(float(distinct[fid]), 3)
                    })

    return {
        "circle_id": circle_idx,
        "circle_name": circle_key,
        "size": len(members),
        "members": members,
        "quality": quality,
        "seed_origin": seed_info,
        "top_features": top_features,
        "summary": (
            f"Circle {circle_idx} contains {len(members)} friends. "
            f"It originated from a {seed_info.get('seed_type', 'clique')} seed of {seed_info.get('seed_size', 'N/A')} nodes "
            f"with internal density of {quality['density']} and conductance of {quality['conductance']}."
        )
    }
