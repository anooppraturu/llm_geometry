# geometry/diffusion.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from .graph import Graph

#utilities for safe matrix inversion
def _safe_inv(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 1.0 / np.maximum(x, eps)

def _safe_invsqrt(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(x, eps))


@dataclass
class DiffusionResult:
    evals: np.ndarray          # (m+1,)
    evecs_sym: np.ndarray      # (N, m+1) eigenvectors of S (symmetric)
    psi: np.ndarray            # (N, m+1) right eigenvectors of P (Markov)
    embedding: Optional[np.ndarray] = None  # (N, m)


class DiffusionMap:
    """
    Diffusion maps on a kNN graph.

    Parameters:
        n_components: number of diffusion coords (excluding trivial first)
        alpha: density normalization exponent (0 = none, 1 = full)
        t: diffusion time (integer >= 1) or float
        symmetrize: whether to symmetrize adjacency
        sym_mode: how to symmetrize ('max', 'avg', 'sum')
    """


    def __init__(
        self,
        n_components: int = 3,
        alpha: float = 0.5,
        t: float = 1.0,
        symmetrize: bool = True,
        sym_mode: str = "max",
        eps: float = 1e-12,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.t = t
        self.symmetrize = symmetrize
        self.sym_mode = sym_mode
        self.eps = eps

        self.result: Optional[DiffusionResult] = None

    def fit(self, graph: Graph) -> "DiffusionMap":
        # 1) affinity / adjacency
        W = graph.to_csr(symmetrize=self.symmetrize, sym_mode=self.sym_mode)

        # 2) D = degree(W)
        d = np.asarray(W.sum(axis=1)).reshape(-1)  # (N,)

        # 3) density normalization: K = D^{-alpha} W D^{-alpha}
        if self.alpha != 0.0:
            d_alpha = d ** (-self.alpha)
            D_alpha = diags(d_alpha, 0, shape=W.shape)
            K = D_alpha @ W @ D_alpha
        else:
            K = W

        # 4) symmetric conjugate: S = D_K^{1/2} P D_K^{-1/2} = D_K^{-1/2} K D_K^{-1/2}
        dK = np.asarray(K.sum(axis=1)).reshape(-1)
        Dk_invsqrt = diags(_safe_invsqrt(dK, self.eps), 0, shape=K.shape)
        S = Dk_invsqrt @ K @ Dk_invsqrt  # symmetric

        # 5) eigensolve: get top m+1 eigenpairs (including trivial)
        m = self.n_components
        # 'LA' = largest algebraic eigenvalues
        evals, evecs = eigsh(S, k=m + 1, which="LA")

        # sort descending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        # recover right eigenvectors of P:
        # if S u = λ u, then psi = D_K^{-1/2} u are right eigenvectors of P
        psi = (Dk_invsqrt @ evecs).astype(np.float64)

        self.result = DiffusionResult(evals=evals, evecs_sym=evecs, psi=psi)
        return self
    
    def transform(self, t: Optional[float] = None) -> np.ndarray:
        """
        Returns diffusion embedding (N, n_components), skipping the first trivial component.
        """
        assert self.result is not None, "Call fit(graph) first."
        if t is None:
            t = self.t

        evals = self.result.evals
        psi = self.result.psi

        # skip trivial first eigenpair
        lamb = evals[1 : self.n_components + 1]
        vecs = psi[:, 1 : self.n_components + 1]

        emb = vecs * (lamb[None, :] ** t)
        self.result.embedding = emb
        return emb

    def fit_transform(self, graph: Graph, t: Optional[float] = None) -> np.ndarray:
        return self.fit(graph).transform(t=t)
