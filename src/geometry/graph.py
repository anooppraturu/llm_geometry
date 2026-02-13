from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from scipy.sparse import csr_matrix

def _symmetrize_csr(A: csr_matrix, mode: str = "max") -> csr_matrix:
    AT = A.T.tocsr()
    if mode == "avg":
        return 0.5 * (A + AT)
    elif mode == "sum":
        return A + AT
    elif mode == "max":
        # elementwise maximum
        return A.maximum(AT)
    else:
        raise ValueError(f"Unknown symmetrize mode: {mode}")

@dataclass
class Graph:
    """
    Lightweight kNN graph representation.

    Attributes:
        indices: (N, k) neighbor indices
        distances: (N, k) neighbor distances
        weights: (N, k) edge weights
        adjacency: optional sparse CSR matrix
    """
    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    # adjacency: np.ndarray

    @property
    def N(self) -> int:
        return self.indices.shape[0]
    
    @property
    def k(self) -> int:
        return self.indices.shape[1]
    
    def to_csr(
        self,
        symmetrize: bool = True,
        sym_mode: str = "max",
        include_self_loops: bool = False,
    ) -> csr_matrix:
        """
        Build sparse adjacency matrix A (N x N) from knn indices/weights.
        """
        N, k = self.indices.shape

        idx = self.indices.reshape(-1)
        w = self.weights.reshape(-1)

        # mask out invalid edges and zero weights
        mask = (idx >= 0) & (w > 0)
        idx = idx[mask]
        w = w[mask]

        rows = np.repeat(np.arange(N), k)[mask]
        cols = idx.astype(np.int64)

        A = csr_matrix((w, (rows, cols)), shape=(N, N))

        if not include_self_loops:
            A.setdiag(0)
            A.eliminate_zeros()

        if symmetrize:
            A = _symmetrize_csr(A, mode=sym_mode)

        return A
    


def build_knn_graph(
    Y: torch.Tensor,
    k: int = 20,
    metric: str = "euclidean",
    mutual: bool = True,
    self_tuning: bool = True,
    beta: float = 1.0,
) -> Graph:
    """
    Y: (N, D) whitened points (torch tensor, CPU)
    Returns Graph object.
    TODO: sparse adjacency matrix for spectral diffusion methods
    """
    if metric != "euclidean":
        raise NotImplementedError
    
    if Y.device.type != "cpu":
        Y = Y.cpu()

    dmat = torch.cdist(Y, Y)
    KNN_dists, KNN_indices = torch.topk(dmat, k=k+1, largest=False)
    KNN_dists, KNN_indices = KNN_dists[:,1:], KNN_indices[:,1:]

    if mutual:
        KNN_indices = mutual_knn(KNN_indices)

    if self_tuning:
        weights = self_tuning_weights(KNN_dists, KNN_indices, beta)
    else:
        weights = torch.exp(-beta * KNN_dists**2) * (KNN_indices >= 0)

    return Graph(KNN_indices.numpy(), KNN_dists.numpy(), weights.numpy())


def mutual_knn(knn_idx: torch.Tensor) -> torch.Tensor:
    """
    knn_idx: (N, k) long tensor of neighbor indices
    returns: (N, k) long tensor where non-mutual neighbors are set to -1
    """
    N, k = knn_idx.shape
    i = torch.arange(N, device=knn_idx.device).view(N, 1)

    # neighbors_of_j: (N, k, k) == knn neighbors of each neighbor j
    neighbors_of_j = knn_idx[knn_idx]

    # mutual if i appears in neighbor list of j
    mutual_mask = (neighbors_of_j == i[:, :, None]).any(dim=2)  # (N, k)

    return torch.where(mutual_mask, knn_idx, torch.full_like(knn_idx, -1))



def self_tuning_weights(
    dists: torch.Tensor,
    idx: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    w_ij = exp( -beta * d_ij^2 / (sigma_i * sigma_j) )
    sigma_i = distance to k-th neighbor of i (last column of dists)
    """
    # sigma_i: (N,)
    sigma = dists[:, -1].clamp_min(eps)
    # -1 -> 0 for indexing, will mask them out
    idx_clamped = idx.clamp_min(0)
    sigma_j = sigma[idx_clamped]    # (N, k)

    denom  = (sigma[:, None] * sigma_j).clamp_min(eps)      # (N, k)
    w = torch.exp(-beta * (dists ** 2) / denom)

    # zero out invalid edges
    w = torch.where(idx >= 0, w, torch.zeros_like(w))
    return w