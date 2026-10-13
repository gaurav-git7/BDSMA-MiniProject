import os
import glob
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import torch


def get_data_dir() -> str:
    """Find the path to data/raw/facebook robustly."""
    curr = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(curr, "..", "..", "..", "data", "raw", "facebook"),
        os.path.join(curr, "..", "..", "data", "raw", "facebook"),
        os.path.join(os.getcwd(), "data", "raw", "facebook"),
        os.path.join(os.getcwd(), "..", "data", "raw", "facebook"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    # Default fallback
    return os.path.abspath(candidates[0])


def list_egos(data_dir: Optional[str] = None) -> List[int]:
    """Return sorted list of available ego IDs."""
    if data_dir is None:
        data_dir = get_data_dir()
    edge_files = glob.glob(os.path.join(data_dir, "*.edges"))
    ego_ids = []
    for f in edge_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base.isdigit():
            ego_ids.append(int(base))
    return sorted(ego_ids)


def load_ego(ego_id: int, data_dir: Optional[str] = None):
    """
    Load network, profile features, and circles for an ego-network.

    Returns:
        G (nx.Graph): NetworkX graph with node IDs as strings, with node attribute 'index'
        node_list (List[str]): List of all friend IDs in order 0..N-1
        node_to_idx (Dict[str, int]): Mapping from node ID string to index 0..N-1
        X (np.ndarray): Shape (N, D), binary feature matrix
        circles (Dict[str, List[str]]): Ground-truth circles mapping name -> list of node IDs
        feat_names (List[str]): List of feature description strings
    """
    if data_dir is None:
        data_dir = get_data_dir()

    base_path = os.path.join(data_dir, str(ego_id))
    feat_path = f"{base_path}.feat"
    edges_path = f"{base_path}.edges"
    circles_path = f"{base_path}.circles"
    featnames_path = f"{base_path}.featnames"

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing feature file: {feat_path}")

    # 1. Read features and collect all friend nodes
    node_features: Dict[str, np.ndarray] = {}
    with open(feat_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            node_id = parts[0]
            feats = np.array([int(x) for x in parts[1:]], dtype=np.float32)
            node_features[node_id] = feats

    node_list = sorted(list(node_features.keys()), key=lambda x: int(x) if x.isdigit() else x)
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}
    N = len(node_list)
    D = len(next(iter(node_features.values()))) if N > 0 else 0

    X = np.zeros((N, D), dtype=np.float32)
    for i, nid in enumerate(node_list):
        X[i] = node_features[nid]

    # 2. Read graph edges
    G = nx.Graph()
    for nid in node_list:
        G.add_node(nid, index=node_to_idx[nid])

    if os.path.exists(edges_path):
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    if u in node_to_idx and v in node_to_idx:
                        G.add_edge(u, v)

    # 3. Read ground truth circles
    circles: Dict[str, List[str]] = {}
    if os.path.exists(circles_path):
        with open(circles_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    cname = parts[0]
                    members = [p for p in parts[1:] if p in node_to_idx]
                    circles[cname] = members

    # 4. Read feature names if available
    feat_names: List[str] = []
    if os.path.exists(featnames_path):
        with open(featnames_path, "r", encoding="utf-8") as f:
            for line in f:
                feat_names.append(line.strip())

    return G, node_list, node_to_idx, X, circles, feat_names


def to_graph_tensors(G: nx.Graph, X: np.ndarray, S: Optional[np.ndarray] = None):
    """
    Convert graph and features to PyTorch tensors.

    Returns:
        edge_index (torch.LongTensor): Shape (2, 2*E)
        adj_dense (torch.FloatTensor): Shape (N, N)
        adj_norm (torch.FloatTensor): Normalized adjacency (N, N) for SAGE/GCN
        X_tensor (torch.FloatTensor): Shape (N, D)
        S_tensor (torch.FloatTensor or None): Shape (N, K)
    """
    N = len(G.nodes)
    adj_dense = np.zeros((N, N), dtype=np.float32)
    edges = []

    for u, v in G.edges():
        u_idx = G.nodes[u]["index"]
        v_idx = G.nodes[v]["index"]
        adj_dense[u_idx, v_idx] = 1.0
        adj_dense[v_idx, u_idx] = 1.0
        edges.append((u_idx, v_idx))
        edges.append((v_idx, u_idx))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Normalized adjacency with self-loops: D^{-1/2} (A + I) D^{-1/2}
    A_self = adj_dense + np.eye(N, dtype=np.float32)
    deg = np.sum(A_self, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg <= 0] = 0.0
    deg_mat = np.diag(deg_inv_sqrt)
    adj_norm = deg_mat @ A_self @ deg_mat

    adj_dense_t = torch.from_numpy(adj_dense)
    adj_norm_t = torch.from_numpy(adj_norm.astype(np.float32))
    X_t = torch.from_numpy(X)
    S_t = torch.from_numpy(S.astype(np.float32)) if S is not None else None

    return edge_index, adj_dense_t, adj_norm_t, X_t, S_t
