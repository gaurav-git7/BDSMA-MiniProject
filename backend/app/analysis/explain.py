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


_PRETTY = {
    "education;school": "School",
    "education;year": "Graduation-year",
    "education;type": "Education-level",
    "education;concentration": "Study-field",
    "education;degree": "Degree",
    "work;employer": "Workplace",
    "work;position": "Job-role",
    "work;location": "Work-location",
    "work;start_date": "Work-period",
    "work;end_date": "Work-period",
    "location": "Location",
    "hometown": "Hometown",
    "languages": "Language",
    "last_name": "Family (shared surname)",
    "first_name": "First-name",
    "birthday": "Birthday",
    "gender": "Gender",
    "locale": "Locale",
    "political": "Political-view",
    "religion": "Religion",
    "quotes": "Quotes",
}


_UNINFORMATIVE = {"gender", "locale", "first_name", "birthday", "quotes"}


def _feature_category(name: str) -> str:
    """'166 work;end_date;anonymized feature 163' -> 'work;end_date'."""
    parts = name.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1]
    tokens = [t for t in name.split(";") if t and t != "id" and not t.startswith("anonymized")]
    return ";".join(tokens)


def label_circles(
    communities: Dict[str, List[str]],
    X: np.ndarray,
    node_to_idx: Dict[str, int],
    feat_names: List[str],
    min_freq: float = 0.25,
    min_lift: float = 0.15,
) -> Dict[str, str]:
    """
    Give each circle a readable name from the profile-feature category that is most
    over-represented among its members (e.g. 'Workplace group').
    Feature values are anonymised in the dataset, so we can name the kind of tie
    (school, workplace, family surname) but not the actual school or employer.
    """
    if X is None or not feat_names:
        return {c: c.replace("_", " ").title() for c in communities}

    categories = [_feature_category(n) for n in feat_names[: X.shape[1]]]
    overall = X.mean(axis=0)
    labels: Dict[str, str] = {}
    used: Dict[str, int] = {}

    for cname, members in communities.items():
        idx = [node_to_idx[m] for m in members if m in node_to_idx]
        base = None
        if len(idx) >= 3:
            freq = X[idx].mean(axis=0)
            lift = freq - overall
            best_score, best_cat = 0.0, None
            for f, cat in enumerate(categories):
                if cat in _UNINFORMATIVE:
                    continue
                if freq[f] >= min_freq and lift[f] >= min_lift and lift[f] > best_score:
                    best_score, best_cat = float(lift[f]), cat
            if best_cat:
                base = f"{_PRETTY.get(best_cat, best_cat.replace(';', ' ').replace('_', ' ').title())} group"
        if base is None:
            base = "Mixed circle"
        used[base] = used.get(base, 0) + 1
        labels[cname] = base if used[base] == 1 else f"{base} #{used[base]}"
    # Number the first occurrence too when a name repeats, so labels are unambiguous
    for base, count in used.items():
        if count > 1:
            first = next(c for c, l in labels.items() if l == base)
            labels[first] = f"{base} #1"
    return labels
