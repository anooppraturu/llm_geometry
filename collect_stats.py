import torch
from dataclasses import dataclass
from typing import Dict, Optional, Literal

Kind = Literal["state", "attn", "mlp"]

@dataclass
class RunningStats:
    Df: int
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.float64  # accumulate in float64 for stability

    def __post_init__(self):
        self.n = 0
        self.mean = torch.zeros(self.Df, device=self.device, dtype=self.dtype)
        self.M2 = torch.zeros(self.Df, self.Df, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(self, Y: torch.Tensor):
        """
        Y: (m, Df) samples
        """
        if Y.numel() == 0:
            return

        Y.to(self.device, dtype=self.dtype)
        m = Y.shape[0]

        mu_b = Y.mean(dim=0) # (Df,)
        Yc = Y - mu_b        # (m, Df)
        S_b = Yc.T @ Yc      # (Df, Df)

        if self.n == 0:
            self.n = m
            self.mean = mu_b
            self.M2 = S_b
            return
        
        n = self.n
        n_new = n + m
        delta = (mu_b - self.mean)

        self.mean = self.mean + delta*(m / n_new)
        self.M2 = self.M2 + S_b + torch.outer(delta, delta)*(n*m/n_new)
        self.n = n_new

    def cov(self) -> torch.Tensor:
        if self.n <= 1:
            return torch.full((self.Df, self.Df), float("nan"), device=self.device, dtype=self.dtype)
        return self.M2 / (self.n - 1)
    
class LayerStatsCollector:
    """
    Collects running mean/cov per layer in functional coords, for different kinds:
      - state: hidden_states[l] or hidden_states[l+1] (configurable)
      - attn: hooks.attn_out[l]
      - mlp : hooks.mlp_out[l]
    """
    def __init__(self, n_layers: int, U_f: torch.Tensor, state_at: Literal["in", "out"] = "out", proj_dtype=torch.float32):
        """
        U_f: (d_model, Df) orthonormal basis (CPU is fine)
        state_at:
          - "in"  uses hidden_states[l]   (entering block l)
          - "out" uses hidden_states[l+1] (leaving block l)
        """
        assert U_f.ndim == 2
        self.proj_dtype = proj_dtype
        self.U_f = U_f.to('cpu', dtype=self.proj_dtype)
        self.Df = U_f.shape[1]
        self.n_layers = n_layers
        self.state_at = state_at

        self.stats: Dict[Kind, Dict[int, RunningStats]] = {
            "state": {l: RunningStats(self.Df) for l in range(n_layers)},
            "attn": {l: RunningStats(self.Df) for l in range(n_layers)},
            "mlp": {l: RunningStats(self.Df) for l in range(n_layers)}
        }

    @torch.no_grad()
    def _proj(self, X: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        X: (batch, seq, d_model) on CPU
        returns Y: (batch*seq, Df) on CPU
        """
        X = X.to('cpu', dtype = self.proj_dtype)
        b, s, d = X.shape
        Y = (X.reshape(b * s, d) @ self.U_f)    # (m, Df)

        if attention_mask is not None:
            mask = attention_mask.reshape(b*s).to('cpu')
            Y = Y[mask.bool()]
            
        return Y
    
    @torch.no_grad()
    def update_from_batch(
        self,
        hidden_states,   # tuple of tensors, each (b, s, d_model), on CPU or device
        hooks,           # hook object with attn_out/mlp_out dicts (CPU tensors)
        include: Optional[Dict[Kind, bool]] = None
    ):
        if include is None:
            include = {"state": True, "attn": True, "mlp": True}

        for l in range(self.n_layers):
            if include.get("state", False):
                # There are n_layers + 1 hidden statea
                X = hidden_states[l+1] if self.state_at == "out" else hidden_states[l]
                self.stats["state"][l].update(self._proj(X))

            if include.get("attn", False):
                self.stats["attn"][l].update(self._proj(hooks.attn_out[l]))

            if include.get("mlp", False):
                self.stats["mlp"][l].update(self._proj(hooks.mlp_out[l]))
        
