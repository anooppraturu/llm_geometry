from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

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
        state_at: str = "out",
    ):
        """
        whiteners: dict[layer_index -> WhiteningTransform]
        state_at: "in" or "out"
        """
        self.projector = projector
        self.whiteners = whiteners
        self.state_at = state_at

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
        You'll fill in:
            - compute covariance from M2 / n
            - compute whitening matrix (e.g. via eigh)
        """
        projector = FunctionalProjector(stats_dict['U_f'])
        state_at = stats_dict['state_at']

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

        return cls(projector = projector, whiteners = whiteners, state_at = state_at)