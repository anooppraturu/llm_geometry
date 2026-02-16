from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from typing import Tuple
import numpy as np

import torch


@dataclass
class FunctionalProjector:
    """
    Projects residual stream vectors into functional subspace.

    U_f: (d_model, Df)
    """
    U_f: torch.Tensor

    def __post_init__(self):
        assert self.U_f.ndim == 2
        self.Df = self.U_f.shape[1]
    

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (..., d_model)
        Returns: (..., Df)
        """
        return X @ self.U_f
    

@dataclass
class WhiteningTransform:
    """
    Represents affine whitening transform:

        y -> W (y - mu)

    where:
        mu: (Df,)
        W : (Df, Df)
    
    This is applied per layer
    """
    mu: torch.Tensor
    W: torch.Tensor

    def whiten(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Y: (..., Df)
        Returns: (..., Df)
        """
        return (Y - self.mu) @ self.W.T
    
    def whiten_updates(self, dy: torch.Tensor) -> torch.Tensor:
        """
        For residual updates (no mean subtraction).
        """
        return dy @ self.W.T
    

class LayerCoordinates:
    """
    Manages projection + whitening for all layers.

    Typically built from saved whitening stats.
    """

    def __init__(
        self,
        projector: FunctionalProjector,
        whiteners: Dict[int, WhiteningTransform],
    ):
        """
        whiteners: dict[layer_index -> WhiteningTransform]
        """
        self.projector = projector
        self.whiteners = whiteners

    def state(self, layer: int, X: torch.Tensor) -> torch.Tensor:
        """
        X: (seq, d_model) or (batch, seq, d_model)
        Returns whitened functional coords.
        """
        Y = self.projector.project(X)      # (batch, seq, Df)
        return self.whiteners[layer].whiten(Y)
    
    def update(self, layer: int, dX: torch.Tensor) -> torch.Tensor:
        """
        For attention/MLP outputs.
        """
        dY = self.projector.project(dX)
        return self.whiteners[layer].whiten_updates(dY)
    
    @classmethod
    def from_stats(cls, stats_dict, epsilon: float = 1e-5):
        """
        Construct LayerCoordinates from saved whitening stats.
        Only load hidden state whiteners.
        """
        projector = FunctionalProjector(stats_dict['U_f'])

        whiteners = {}
        for l, stats in stats_dict['stats']['state'].items():
            l = int(l)
            n = int(stats['n'])
            if n <= 1:
                raise ValueError(f"Layer {l} has n={n}; cannot form covariance.")
            
            mu = stats['mean'].detach().to("cpu")
            cov = (stats['M2']/(n - 1)).detach().to("cpu")
            
            evals, evecs = torch.linalg.eigh(cov)
            inv_sqrt = torch.rsqrt(evals + epsilon)
            W = (evecs * inv_sqrt) @ evecs.T

            whiteners[l] = WhiteningTransform(mu=mu, W=W)

        return cls(projector = projector, whiteners = whiteners)
    

@torch.no_grad()
def orthogonal_procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align X to Y with an orthogonal transform + translation:
        X_aligned = X @ R + t

    Args:
        X, Y: (N, D) tensors, corresponding points in same order
        allow_reflection: if False, enforce det(R)=+1

    Returns:
        X_aligned: (N, D)
        R: (D, D) orthogonal matrix
        t: (D,) translation vector
    """
    assert X.shape == Y.shape and X.ndim == 2
    N, D = X.shape

    # center
    muX = X.mean(axis=0, keepdims=True)   # (1, D)
    muY = Y.mean(axis=0, keepdims=True)   # (1, D)
    Xc = X - muX
    Yc = Y - muY

    # cross-covariance
    M = Xc.T @ Yc  # (D, D)

    # SVD
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    R = U @ Vh  # (D, D)

    # enforce det(R)=+1 (no reflection)
    if np.linalg.det(R) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vh

    # translation
    t = (muY - muX @ R).squeeze(0)  # (D,)

    X_aligned = X @ R + t
    return X_aligned, R, t