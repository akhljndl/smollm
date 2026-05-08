"""Regenerate the augmented SMILES cache deterministically.

The augmenter uses atom-renumber with a Python random.Random(seed) — the cache
is byte-reproducible across machines for a given (n_aug, seed, source CSV,
RDKit version, Python version).

Run this once before training to (re)build `data/zinc250k_aug_x{n}.pt`.

Usage:
    python scripts/regenerate_aug_cache.py --n-aug 10 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from smollm.data import (
    download_zinc,
    load_smiles,
    prepare_augmented_smiles,
)
from smollm.tokenizer import SmilesTokenizer

BLOCK_SIZE = 128


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-aug", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for both 90/10 split shuffle AND augmenter RNG")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing cache file")
    args = p.parse_args()

    data_path = download_zinc(Path(args.data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)
    print(f"Loaded {len(smiles):,} canonical SMILES from {data_path}")

    tokenizer = SmilesTokenizer().fit(smiles)
    print(f"Vocab size: {tokenizer.vocab_size}")

    rng = random.Random(args.seed)
    shuffled = list(smiles)
    rng.shuffle(shuffled)
    split = int(0.9 * len(shuffled))
    train_smiles = shuffled[:split]
    print(f"Train split (seed={args.seed}): {len(train_smiles):,} SMILES")

    aug_path = Path(args.data_dir) / f"zinc250k_aug_x{args.n_aug}.pt"
    if aug_path.exists() and not args.force:
        raise SystemExit(
            f"Cache already exists at {aug_path}. Pass --force to overwrite."
        )

    print(f"Generating x{args.n_aug} augmentation (this can take 5-15 min)...")
    train_data = prepare_augmented_smiles(
        train_smiles, tokenizer, BLOCK_SIZE, args.n_aug, args.seed,
    )
    aug_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_data, aug_path)
    size_mb = aug_path.stat().st_size / 1e6
    print(f"Saved {tuple(train_data.shape)} -> {aug_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
