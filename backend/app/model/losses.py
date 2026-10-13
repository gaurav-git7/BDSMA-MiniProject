import torch
import torch.nn.functional as F


def bernoulli_poisson_loss(
    F_mat: torch.Tensor,
    adj_dense: torch.Tensor,
    neg_ratio: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Bernoulli-Poisson link probability loss (standard for overlapping community detection).
    P(edge u,v) = 1 - exp(- F_u . F_v)

    Positive loss on edges: - log(1 - exp(- F_u . F_v))
    Negative loss on non-edges: F_u . F_v
    """
    N = F_mat.size(0)

    # Compute dot products F * F^T
    affinity = torch.matmul(F_mat, F_mat.t())

    # Diagonal elements should not be considered self-edges
    mask_diag = torch.eye(N, dtype=torch.bool, device=F_mat.device)

    # Positive edges mask (upper triangle without diagonal)
    pos_mask = (adj_dense > 0) & (~mask_diag)
    # Negative edges mask
    neg_mask = (adj_dense == 0) & (~mask_diag)

    pos_affinity = affinity[pos_mask]
    if len(pos_affinity) == 0:
        return torch.tensor(0.0, device=F_mat.device, requires_grad=True)

    # Stable -log(1 - exp(-x)):
    # 1 - exp(-x) with clamp
    pos_affinity = torch.clamp(pos_affinity, min=1e-5, max=20.0)
    pos_loss = -torch.log(torch.clamp(1.0 - torch.exp(-pos_affinity), min=eps))
    pos_loss_val = torch.mean(pos_loss)

    # Subsample negative edges to prevent non-edge dominance
    num_neg = int(min(len(pos_affinity) * neg_ratio, neg_mask.sum().item()))
    if num_neg > 0:
        neg_indices = torch.nonzero(neg_mask, as_tuple=False)
        perm = torch.randperm(neg_indices.size(0))[:num_neg]
        selected_neg = neg_indices[perm]
        neg_affinity = affinity[selected_neg[:, 0], selected_neg[:, 1]]
        neg_loss_val = torch.mean(neg_affinity)
    else:
        neg_loss_val = torch.tensor(0.0, device=F_mat.device)

    return pos_loss_val + neg_loss_val


def soft_modularity_loss(
    F_mat: torch.Tensor,
    adj_dense: torch.Tensor
) -> torch.Tensor:
    """
    Differentiable modularity loss:
    Maximizes Tr(S^T B S) / (2m), so loss is -Tr(S^T B S) / (2m).
    """
    degrees = torch.sum(adj_dense, dim=1, keepdim=True)
    m2 = torch.sum(degrees)
    if m2 <= 0:
        return torch.tensor(0.0, device=F_mat.device, requires_grad=True)

    # B = A - (d d^T) / 2m
    B = adj_dense - torch.matmul(degrees, degrees.t()) / m2

    # Softmax over communities for partition representation
    S = F.softmax(F_mat, dim=1)

    # Tr(S^T B S) = sum((S^T B) * S^T)
    # or (S^T @ B @ S).trace()
    SB = torch.matmul(S.t(), B)
    Q = torch.trace(torch.matmul(SB, S)) / m2
    return -Q


def seed_consistency_loss(
    F_mat: torch.Tensor,
    S_mat: torch.Tensor
) -> torch.Tensor:
    """
    Binary Cross-Entropy between squashed F and seed matrix S,
    computed only on nodes that belong to at least one seed.
    """
    if S_mat is None:
        return torch.tensor(0.0, device=F_mat.device, requires_grad=True)

    # Identify nodes in at least one seed
    seeded_mask = torch.sum(S_mat, dim=1) > 0
    if seeded_mask.sum() == 0:
        return torch.tensor(0.0, device=F_mat.device, requires_grad=True)

    # Squash F with sigmoid
    pred_prob = torch.sigmoid(F_mat[seeded_mask])
    target = S_mat[seeded_mask]

    return F.binary_cross_entropy(pred_prob, target)
