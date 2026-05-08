"""Evaluate a trained checkpoint: generate N SMILES + report validity / FCD / IntDiv.

Re-derives the 90% train split from --seed (matching train.py) so novelty and the
train-split FCD reference reproduce exactly.

Examples:
    python eval.py --checkpoint checkpoints/ws-53k-s42.pt
    python eval.py --checkpoint checkpoints/ws-206k-s42.pt --n 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from smollm.checkpoint import load_checkpoint
from smollm.data import download_zinc, load_smiles, train_val_split
from smollm.sample import evaluate_smiles, generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained SMILES checkpoint.")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to .pt produced by train.py / distill.py / dpo.py")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the training seed — used to re-derive the train split for novelty.")
    p.add_argument("--n", type=int, default=1000, help="Number of molecules to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write metrics as JSON (defaults to <ckpt>.eval.json)")
    return p.parse_args()


def derive_train_set(seed: int, data_dir: str) -> set[str]:
    data_path = download_zinc(Path(data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)
    train_smiles, _ = train_val_split(smiles, seed)
    return set(train_smiles)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"==> {ckpt_path}  (device={device}, n={args.n}, seed={args.seed})")
    model, tokenizer = load_checkpoint(str(ckpt_path), device=device)

    train_set = derive_train_set(args.seed, args.data_dir)
    print(f"    train split: {len(train_set)} canonical SMILES")

    print(f"  Generating {args.n} molecules...")
    molecules = generate(
        model, tokenizer,
        n=args.n, temperature=args.temperature, top_k=args.top_k, device=device,
    )
    metrics = evaluate_smiles(molecules, train_set)
    print(
        f"  Validity: {metrics['validity'] * 100:.2f}%  "
        f"Unique: {metrics['uniqueness'] * 100:.2f}%  "
        f"Novel: {metrics['novelty'] * 100:.2f}%  "
        f"IntDiv: {metrics['intdiv']:.4f}"
    )
    if metrics.get("fcd") is not None:
        print(f"  FCD (train-split ref): {metrics['fcd']:.4f}")

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".eval.json")
    payload = {
        "checkpoint": str(ckpt_path),
        "seed": args.seed,
        "n": args.n,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "validity": metrics.get("validity"),
        "uniqueness": metrics.get("uniqueness"),
        "novelty": metrics.get("novelty"),
        "intdiv": metrics.get("intdiv"),
        "fcd_train_split": metrics.get("fcd"),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
