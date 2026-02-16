from __future__ import annotations

import torch
import numpy as np
from geometry.coords import LayerCoordinates
from transformers import AutoTokenizer, AutoModelForCausalLM
from geometry.diffusion import DiffusionMap
from geometry.graph import build_knn_graph

class diffusion_pipeline:
    """
    Given model, tokenizer, and whitener/projector, implements a thin wrapper
    to produce per-layer diffusion embeddings from a given input string.
    For now, alpha=0.5 density normalization hardcoded.
    """
    def __init__(
            self, 
            model: AutoModelForCausalLM, 
            tokenizer: AutoTokenizer, 
            LC: LayerCoordinates
        ):
        #model objects
        self.model = model
        self.tokenizer = tokenizer
        self.embed_dim = self.model.embed_out.in_features

        #geometry and diffusion
        self.LC = LC
        #TODO: are my layer whiteners lining up correctly with layer outputs? If I have state at out is layer 0 of hidden states the 0th whitener?
        self.n_layers = len(self.LC.whiteners)
        self.diffusion = DiffusionMap()


    def get_hidden_states(self, text):
        """
        in: string of text
        out: list of length n_layers each of which is a (N_tokens, D_embed) tensor
        """
        enc = self.tokenizer(text, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        N_tok = enc['input_ids'].shape[1]

        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True, use_cache=False)

        return [dat.view(N_tok, self.embed_dim) for dat in out.hidden_states]
    
    
    def get_functional_coordinates(self, states):
        """
        in: list of length n_layers each of which is a (N_tokens, D_embed) tensor
        out: list of length n_layers each of which is a (N_tokens, Df) tensor
        """
        return [self.LC.state(l, hidden) for l, hidden in enumerate(states)]
    
    
    def build_graphs(self, func_coords, k, mutual=False):
        """
        build n_layer graphs from functional basis coordinates for diffusion embeddings
        """
        return [build_knn_graph(func, k=k, mutual=mutual) for func in func_coords]
    

    def process(self, text, k, t=1.0, mutual=False):
        """
        in: text string and graph/diffusion parameters
        out: n_layer list of (N_tok, D_diff=3) diffusion embeddings
        """
        hidden_states = self.get_hidden_states(text)
        functional_coords = self.get_functional_coordinates(hidden_states)
        graphs = self.build_graphs(functional_coords, k=k, mutual=mutual)

        diffusion_embeds = [self.diffusion.fit_transform(graph = g, t=t) for g in graphs]
        return hidden_states, functional_coords, graphs, diffusion_embeds