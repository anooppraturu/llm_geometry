import torch

# --- Hook manager ---
class ResidualHooks:
    """
    Captures:
      - attn_out[l]: output of layer l attention module (the "update" added to residual)
      - mlp_out[l] : output of layer l MLP module (the "update" added to residual)
    """
    def __init__(self, model, store_on_cpu=True):
        self.model = model
        self.store_on_cpu = store_on_cpu
        self.handles = []
        self.attn_out = {}
        self.mlp_out = {}

    def _to_store(self, x):
        if not torch.is_tensor(x):
            return x
        x = x.detach()
        if self.store_on_cpu:
            x = x.to("cpu")
        return x

    def _make_hook(self, store_dict, layer_idx):
        def hook(module, inp, out):
            # Attention sometimes returns (output, weights)
            if isinstance(out, tuple):
                out = out[0]
            store_dict[layer_idx] = self._to_store(out)
        return hook

    def add(self):
        # Pythia uses GPTNeoX under model.gpt_neox
        layers = self.model.gpt_neox.layers
        for l, block in enumerate(layers):
            self.handles.append(block.attention.register_forward_hook(self._make_hook(self.attn_out, l)))
            self.handles.append(block.mlp.register_forward_hook(self._make_hook(self.mlp_out, l)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []