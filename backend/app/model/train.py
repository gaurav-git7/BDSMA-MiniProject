import time
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from .encoder import MembershipGNN
from .losses import bernoulli_poisson_loss, soft_modularity_loss, seed_consistency_loss


def train_model(
    X_t: torch.Tensor,
    S_t: Optional[torch.Tensor],
    adj_norm: torch.Tensor,
    adj_dense: torch.Tensor,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Train the MembershipGNN encoder.

    Returns:
        F_np (np.ndarray): Shape (N, K), learned non-negative membership scores.
        metadata (Dict[str, Any]): Training history and timing.
    """
    cfg = {
        "encoder_type": "sage",
        "hidden_dim": 64,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "epochs": 150,
        "structure_loss": "bp",  # 'bp' or 'modularity'
        "lambda_seed": 1.0,
        "dropout": 0.1,
    }
    if config:
        cfg.update(config)

    N = X_t.size(0)
    K = S_t.size(1) if S_t is not None else 10

    # Node input is [X | S] so the GNN starts from clique structure
    if S_t is not None:
        node_input = torch.cat([X_t, S_t], dim=1)
    else:
        node_input = X_t

    in_dim = node_input.size(1)
    model = MembershipGNN(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        out_dim=K,
        encoder_type=cfg["encoder_type"],
        dropout=cfg["dropout"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    start_time = time.time()
    losses = []

    model.train()
    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        F_mat = model(node_input, adj_norm, adj_dense)

        # Structure loss
        if cfg["structure_loss"] == "modularity":
            l_struct = soft_modularity_loss(F_mat, adj_dense)
        else:
            l_struct = bernoulli_poisson_loss(F_mat, adj_dense)

        # Seed consistency loss
        if S_t is not None and cfg["lambda_seed"] > 0:
            l_seed = seed_consistency_loss(F_mat, S_t)
        else:
            l_seed = torch.tensor(0.0)

        total_loss = l_struct + cfg["lambda_seed"] * l_seed

        if torch.isnan(total_loss):
            break

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    total_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        F_final = model(node_input, adj_norm, adj_dense)
        F_np = F_final.cpu().numpy()

    metadata = {
        "losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "train_time_sec": round(total_time, 4),
        "config": cfg,
        "K": K
    }
    return F_np, metadata


def assign_communities(
    F_np: np.ndarray,
    G: nx.Graph,
    node_list: List[str],
    tau: float = 0.5,
    fill_unassigned: bool = True
) -> Dict[str, List[str]]:
    """
    Threshold membership matrix F to assign nodes to communities.
    Unassigned nodes take the most common circle among their neighbors (Paper #20).

    Returns:
        communities: Dict mapping 'circle_c' -> list of node IDs.
    """
    N, K = F_np.shape
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    # Binary assignments
    assignments: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        for c in range(K):
            if F_np[i, c] >= tau:
                assignments[i].append(c)

    # Fill unassigned nodes using neighbor votes
    if fill_unassigned:
        for i, nid in enumerate(node_list):
            if not assignments[i] and nid in G:
                neighbor_circles = []
                for neighbor in G.neighbors(nid):
                    if neighbor in node_to_idx:
                        n_idx = node_to_idx[neighbor]
                        neighbor_circles.extend(assignments[n_idx])
                if neighbor_circles:
                    most_common_c = Counter(neighbor_circles).most_common(1)[0][0]
                    assignments[i].append(most_common_c)
                else:
                    # Fallback to node's highest scoring column in F
                    best_c = int(np.argmax(F_np[i]))
                    assignments[i].append(best_c)

    # Invert into dict of communities
    communities: Dict[str, List[str]] = {}
    for c in range(K):
        c_name = f"circle_{c}"
        members = [node_list[i] for i in range(N) if c in assignments[i]]
        if members:  # only include non-empty circles
            communities[c_name] = members

    return communities
