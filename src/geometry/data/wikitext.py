from __future__ import annotations

from itertools import chain
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

import torch
from torch.utils.data import DataLoader


def load_wikitext(split: str = "train", streaming: bool = False):
    """
    Load raw Wikitext-103 and drop empty strings.
    Returns a Dataset (streaming=False) or IterableDataset (streaming=True).
    """
    ds = load_dataset(
        "wikitext", 
        "wikitext-103-raw-v1", 
        split=split, 
        streaming=streaming
    )
    ds = ds.filter(lambda x: x['text'] is not None and len(x['text'].strip()) > 0)
    return ds


def tokenize_wikitext(ds, tokenizer: PreTrainedTokenizerBase, batch_size: int = 1000):
    """
    Tokenize dataset of {'text': ...} into {'input_ids': ...} (list[int]).
    """
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], 
            return_attention_mask=False, 
            truncation=False,
            return_token_type_ids=False
            )
    
    return ds.map(tokenize_fn, batched=True, batch_size=batch_size, remove_columns=["text"])


def group_texts(examples, block_size: int = 256):
    """
    Concatenate token lists and split into fixed-size blocks.
    Works on dict-of-lists where each value is list[list[int]].
    """
    concatenated = {}
    for k, v in examples.items():
        concatenated[k] = list(chain.from_iterable(v))

    total_len = len(concatenated["input_ids"])
    total_len = (total_len // block_size) * block_size  # drop remainder

    result = {}
    for k, tokens in concatenated.items():
        result[k] = [tokens[i : i + block_size] for i in range(0, total_len, block_size)]

    return result


def build_wikitext_blocks(
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    block_size: int = 256,
    streaming: bool = False,
    tokenization_batch_size: int = 1000,
):
    """
    End-to-end: load -> filter -> tokenize -> pack into fixed-length blocks.
    Returns Dataset (if streaming=False). If streaming=True, returns IterableDataset
    but note: packed block construction is harder to do correctly in pure streaming.
    """

    if streaming:
        # For a first pass on laptop, prefer non-streaming + caching to disk.
        raise ValueError("For packed blocks, start with streaming=False and cache to disk.")
    
    ds = load_wikitext(split=split, streaming=False)
    tok = tokenize_wikitext(ds, tokenizer, batch_size=tokenization_batch_size)

    # Pack into blocks
    lm_ds = tok.map(lambda ex: group_texts(ex, block_size=block_size), batched=True)

    # Keep only input_ids (simplify)
    keep_cols = ["input_ids"]
    remove_cols = [c for c in lm_ds.column_names if c not in keep_cols]
    lm_ds = lm_ds.remove_columns(remove_cols)

    return lm_ds


def load_or_build_cached_blocks(
    cache_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    block_size: int = 256,
    overwrite: bool = False,
    tokenization_batch_size: int = 1000,
) -> Dataset:
    """
    Loads prebuilt blocks from disk if present; otherwise builds and saves them.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and not overwrite:
        return load_from_disk(str(cache_dir))

    lm_ds = build_wikitext_blocks(
        tokenizer=tokenizer,
        split=split,
        block_size=block_size,
        streaming=False,
        tokenization_batch_size=tokenization_batch_size,
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lm_ds.save_to_disk(str(cache_dir))
    return lm_ds

def make_block_dataloader(ds, batch_size=8, shuffle=True, num_workers=0):
    ds = ds.with_format("torch", columns=["input_ids"])

    def collate(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)