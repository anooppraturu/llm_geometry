# scripts/build_wikitext_blocks.py
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from geometry.data.wikitext import load_or_build_cached_blocks


def parse_args():
    p = argparse.ArgumentParser(description="Build and cache Wikitext-103 packed token blocks.")
    p.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped",
                   help="Tokenizer/model name on Hugging Face.")
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--tokenization-batch-size", type=int, default=1000)
    p.add_argument("--cache-dir", type=str, default="data_cache/wikitext103_blocks")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Make cache dir encode block size + split
    cache_dir = Path(args.cache_dir) / f"{args.split}_bs{args.block_size}"

    ds = load_or_build_cached_blocks(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        split=args.split,
        block_size=args.block_size,
        overwrite=args.overwrite,
        tokenization_batch_size=args.tokenization_batch_size,
    )

    print(f"Saved dataset to: {cache_dir}")
    print(ds)
    print("Example:", ds[0]["input_ids"][:10], "...", "len =", len(ds[0]["input_ids"]))


if __name__ == "__main__":
    main()
