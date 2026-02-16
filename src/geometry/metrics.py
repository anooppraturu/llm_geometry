from __future__ import annotations
from typing import Literal, Optional, Tuple
import numpy as np


def _build_adjacency_lists(indices: np.ndarray, weights: np.ndarray):
    """
    Build adjacency list representation from knn arrays.

    Returns:
        nbrs: list[list[int]]
        wts : list[list[float]]
    """
    N, k = indices.shape
    nbrs = [[] for _ in range(N)]
    wts = [[] for _ in range(N)]

    for i in range(N):
        js = indices[i]
        ws = weights[i]
        mask = (js >= 0) & (ws > 0)
        for j, w in zip(js[mask], ws[mask]):
            nbrs[i].append(int(j))
            wts[i].append(float(w))
    return nbrs, wts

# ============================================================
# Forman Curvature
# ============================================================

def forman_edge_curvature(
    indices: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Forman-Ricci curvature per directed edge represented by (i -> j).
        F(e=uv) = 2
                 - sum_{u' in N(u)\{v}} sqrt(w_uv / w_uu')
                 - sum_{v' in N(v)\{u}} sqrt(w_uv / w_vv')

    This captures “negative curvature” when endpoints have many strong alternative
    connections (graph expands), and positive when edge is “bridge-like”.

    Inputs:
      indices, weights: (N,k) kNN representation.

    Returns:
      src: (E,) source node indices
      dst: (E,) destination node indices
      F  : (E,) curvature values for each listed edge
    """
    nbrs, wts = _build_adjacency_lists(indices, weights)
    N = len(nbrs)

    # Build a quick lookup for edge weight w(i,j)
    # Use dict per node for O(1) access
    w_lookup = [dict(zip(nbrs[i], wts[i])) for i in range(N)]

    src_list = []
    dst_list = []
    F_list = []

    for i in range(N):
        for j in nbrs[i]:
            w_ij = w_lookup[i].get(j, 0.0)
            if w_ij <= 0:
                continue

            # neighbors of i excluding j
            sum_i = 0.0
            for ip in nbrs[i]:
                if ip == j:
                    continue
                w_iip = w_lookup[i].get(ip, 0.0)
                if w_iip > 0:
                    sum_i += np.sqrt(w_ij / w_iip)

            # neighbors of j excluding i
            sum_j = 0.0
            for jp in nbrs[j]:
                if jp == i:
                    continue
                w_jjp = w_lookup[j].get(jp, 0.0)
                if w_jjp > 0:
                    sum_j += np.sqrt(w_ij / w_jjp)

            F = 2.0 - sum_i - sum_j

            src_list.append(i)
            dst_list.append(j)
            F_list.append(F)

    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    F = np.asarray(F_list, dtype=np.float64)

    return src, dst, F


def forman_node_curvature(
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Aggregate Forman edge curvature into per-node curvature.

    Returns:
      K: (N,) node curvature values (NaN if node has no edges)
    """
    src, dst, F = forman_edge_curvature(indices, weights)

    N = indices.shape[0]
    K = np.zeros(N, dtype=np.float64)
    deg = np.zeros(N, dtype=np.int64)

    # treat edges as incident to both endpoints
    for s, d, f in zip(src, dst, F):
        K[s] += f
        deg[s] += 1
        K[d] += f
        deg[d] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        K = K / np.maximum(deg, 1)
    K[deg == 0] = np.nan

    return K


# ============================================================
# Levina–Bickel intrinsic dimension (MLE)
# ============================================================

def levina_bickel_dimension(
    distances: np.ndarray,
    indices: np.ndarray,
    k_eff: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-point intrinsic dimension estimator (Levina & Bickel MLE).

    For each i, using its neighbor distances r_{i,1..k} (sorted ascending):
        d_i = [ (1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j) ]^{-1}

    Inputs:
      distances: (N,k) neighbor distances (should be >0 and sorted)
      indices:   (N,k) neighbor indices with -1 for invalid
      k_eff:     if provided, use only first k_eff neighbors (must be <= k).
                 If None, use all valid neighbors per row.
      eps:       small number to avoid log(0)

    Returns:
      d_hat: (N,) estimated intrinsic dimension (NaN if insufficient neighbors)
    """
    N, k = distances.shape
    d_hat = np.full(N, np.nan, dtype=np.float64)

    for i in range(N):
        valid = indices[i] >= 0
        r = distances[i][valid].astype(np.float64)

        if k_eff is not None:
            r = r[:k_eff]

        # Need at least 2 neighbors to form estimator (k >= 2)
        if r.size < 2:
            continue

        # Ensure positive and avoid zeros (can happen if duplicates / self included)
        r = np.maximum(r, eps)

        r_k = r[-1]
        logs = np.log(r_k / r[:-1])

        denom = logs.mean()
        if denom <= 0 or not np.isfinite(denom):
            d_hat[i] = np.nan
        else:
            d_hat[i] = 1.0 / denom

    return d_hat
