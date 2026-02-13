from __future__ import annotations

import argparse
from pathlib import Path

from geometry.collect_stats import (
    LayerStatsCollector
)
from geometry.functional_space import (
    get_functional_basis_from_unembedding
)
from geometry.data.wikitext import (
    make_block_dataloader
)
from geometry.hooks import (
    ResidualHooks
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch

MODEL_NAME = "EleutherAI/pythia-70m-deduped"


def main():
    #set device and load model
    print("Loading model: {}".format(MODEL_NAME))
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "mps" else torch.float32,
    )
    model.to(device)
    model.eval()
    n_layers = len(model.gpt_neox.layers)
    #add hooks
    hooks = ResidualHooks(model, store_on_cpu=True)
    hooks.add()

    #load dataset and put into loader
    print("Loading data from data_cache/wikitext103_blocks/train_bs256")
    ds = load_from_disk('data_cache/wikitext103_blocks/train_bs256')
    loader = make_block_dataloader(ds)

    #get functional basis from top right eigenvectors of un-embedding matrix
    print("solving for functional basis vectors with Df={}".format(128))
    U_f, svals2 = get_functional_basis_from_unembedding(model, Df=128, device="cpu")
    print("U_f shape:", U_f.shape)
    print("top eigenvalues (WU^T WU):", svals2[:5])

    #initialize layer collector
    collector = LayerStatsCollector(n_layers=n_layers, U_f=U_f)

    print("looping over data, building stats")
    max_batches = 1000
    for i, batch in enumerate(loader):
        if i >= max_batches:
            continue
        if i % 100 == 0:
            print("processing batch {}".format(i))
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        collector.update_from_batch(hidden_states, hooks)

    #remove hooks
    hooks.remove()

    #save collected stats data
    state = collector.state_dict()
    torch.save(state, 'data_cache/stats/pythia70m_wikitext103_train_bs256_Df128_stateAtOut_withUf.pt')


if __name__ == "__main__":
    main()