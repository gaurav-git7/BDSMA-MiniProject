import os
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from .data.loader import list_egos, load_ego
from .pipeline import run_pipeline, get_results_dir
from .analysis.recommend import suggest_friends


app = FastAPI(
    title="CircleSense API",
    description="Overlapping Friend-Circle Detection with Clique Percolation and Graph Neural Networks",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_or_run_result(ego_id: int) -> Dict[str, Any]:
    """Retrieve saved results or run pipeline on demand."""
    res_dir = get_results_dir()
    json_path = os.path.join(res_dir, f"ego_{ego_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # If not yet generated, run with default settings
    return run_pipeline(ego_id, save_results=True)


class RunConfig(BaseModel):
    k: int = 4
    seeding: str = "cpm"  # 'cpm' or 'weak_clique'
    encoder_type: str = "sage"  # 'sage' or 'gat'
    hidden_dim: int = 64
    lr: float = 0.01
    epochs: int = 120
    structure_loss: str = "bp"  # 'bp' or 'modularity'
    lambda_seed: float = 1.0
    tau: float = 0.5


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "CircleSense API"}


@app.get("/api/egos")
def get_all_egos():
    """List all ego networks with node, edge, and circle counts."""
    egos = list_egos()
    results = []
    res_dir = get_results_dir()

    for ego_id in egos:
        json_path = os.path.join(res_dir, f"ego_{ego_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "ego_id": ego_id,
                    "nodes": data["graph_summary"]["num_nodes"],
                    "edges": data["graph_summary"]["num_edges"],
                    "ground_truth_circles": data["graph_summary"]["num_ground_truth_circles"],
                    "detected_circles": data["graph_summary"]["num_detected_circles"],
                    "bridges": data["graph_summary"]["num_bridges"],
                    "processed": True
                })
        else:
            try:
                G, node_list, _, _, circles, _ = load_ego(ego_id)
                results.append({
                    "ego_id": ego_id,
                    "nodes": len(node_list),
                    "edges": G.number_of_edges(),
                    "ground_truth_circles": len(circles),
                    "detected_circles": 0,
                    "bridges": 0,
                    "processed": False
                })
            except Exception:
                pass

    return results


@app.get("/api/egos/{ego_id}/graph")
def get_ego_graph(ego_id: int):
    """Return nodes and edges formatted for Cytoscape.js."""
    data = load_or_run_result(ego_id)
    return {
        "ego_id": ego_id,
        "elements": data.get("cytoscape", []),
        "summary": data.get("graph_summary", {})
    }


@app.get("/api/egos/{ego_id}/communities")
def get_ego_communities(
    ego_id: int,
    method: str = Query("hybrid", description="Method: hybrid, cpm, leiden, bigclam, ground_truth")
):
    """Return detected friend circles for the specified method."""
    data = load_or_run_result(ego_id)
    comms = data.get("communities", {})
    method_key = method.lower()
    if method_key not in comms:
        # Fallback to hybrid
        method_key = "hybrid"

    return {
        "ego_id": ego_id,
        "method": method_key,
        "circles": comms.get(method_key, {})
    }


@app.get("/api/egos/{ego_id}/nodes/{node_id}")
def get_node_details(ego_id: int, node_id: str):
    """Return circles and membership vector for one specific friend."""
    data = load_or_run_result(ego_id)
    res_dir = get_results_dir()
    npz_path = os.path.join(res_dir, f"ego_{ego_id}_membership.npz")

    F_row = []
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        F_mat = npz["F"]
        n_list = list(npz["node_list"])
        if node_id in n_list:
            idx = n_list.index(node_id)
            F_row = [round(float(x), 4) for x in F_mat[idx]]

    # Membership circles
    active_circles = []
    hybrid_comms = data.get("communities", {}).get("hybrid", {})
    for c_name, members in hybrid_comms.items():
        if node_id in members:
            active_circles.append(c_name)

    return {
        "ego_id": ego_id,
        "node_id": node_id,
        "active_circles": active_circles,
        "circle_count": len(active_circles),
        "membership_vector": F_row
    }


@app.get("/api/egos/{ego_id}/bridges")
def get_bridge_friends(ego_id: int, top: int = Query(25, description="Number of bridge friends to return")):
    """Return ranked bridge friends belonging to two or more circles."""
    data = load_or_run_result(ego_id)
    bridges = data.get("bridges", [])
    return {
        "ego_id": ego_id,
        "total_bridges": len(bridges),
        "bridges": bridges[:top]
    }


@app.get("/api/egos/{ego_id}/recommend/{node_id}")
def get_friend_recommendations(ego_id: int, node_id: str, top: int = Query(10)):
    """Suggest friends based on shared-circle affinity with breakdown."""
    data = load_or_run_result(ego_id)
    res_dir = get_results_dir()
    npz_path = os.path.join(res_dir, f"ego_{ego_id}_membership.npz")

    if not os.path.exists(npz_path):
        raise HTTPException(status_code=404, detail="Membership data not available")

    npz = np.load(npz_path)
    F_mat = npz["F"]
    node_list = list(npz["node_list"])
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    G, _, _, _, _, _ = load_ego(ego_id)
    recs = suggest_friends(F_mat, G, node_id, node_list, node_to_idx, top_n=top)

    return {
        "ego_id": ego_id,
        "node_id": node_id,
        "recommendations": recs
    }


@app.get("/api/egos/{ego_id}/circles/{circle_name}/explain")
def get_circle_explanation(ego_id: int, circle_name: str):
    """Retrieve explanation, constituent seed cliques, and quality metrics for a circle."""
    data = load_or_run_result(ego_id)
    explanations = data.get("explanations", {})
    if circle_name in explanations:
        return explanations[circle_name]

    # Try formatted name
    if not circle_name.startswith("circle_") and f"circle_{circle_name}" in explanations:
        return explanations[f"circle_{circle_name}"]

    raise HTTPException(status_code=404, detail=f"Circle '{circle_name}' not found")


@app.get("/api/egos/{ego_id}/metrics")
def get_metrics_comparison(ego_id: int):
    """Return metric evaluation across all algorithms."""
    data = load_or_run_result(ego_id)
    return {
        "ego_id": ego_id,
        "metrics": data.get("metrics", {}),
        "recommendation_eval": data.get("recommendation_eval", {})
    }


@app.post("/api/egos/{ego_id}/run")
def trigger_rerun(ego_id: int, config: RunConfig):
    """Re-run pipeline with custom hyperparameters."""
    new_res = run_pipeline(ego_id, config=config.dict(), save_results=True)
    return {
        "status": "success",
        "ego_id": ego_id,
        "summary": new_res.get("graph_summary", {}),
        "metrics": new_res.get("metrics", {})
    }
