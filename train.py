"""Train a SMILES generator from scratch.

Examples:
    python train.py --config ws-e64-x8 --seed 42         # WS-53K headline (~95% validity)
    python train.py --config ws-e128-x8 --seed 42        # WS-206K headline (~98% validity)
    python train.py --config L2H2E144 --seed 42          # GPT-527K vanilla baseline
    python train.py --config GRU-53K --seed 42 --no-amp  # GRU baseline (AMP off for stability)

The script:
    1. Downloads ZINC-250K (~22 MB) on first run.
    2. Builds a tokenizer + augmented training set (cached as data/zinc250k_aug_x{N}.pt).
    3. Trains for --epochs (default 100) with cosine LR + AdamW.
    4. Generates 1000 SMILES and reports validity / uniqueness / novelty / IntDiv / FCD.
    5. Saves checkpoint to checkpoints/{config}-s{seed}.pt.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from smollm._utils import cosine_lr
from smollm.checkpoint import make_smiles_gpt, make_smiles_gru, save_checkpoint
from smollm.configs import ARCHITECTURES
from smollm.data import (
    AugmentedSmilesDataset,
    SmilesDataset,
    download_zinc,
    load_smiles,
    prepare_augmented_smiles,
    train_val_split,
)
from smollm.sample import evaluate_smiles, generate, log_samples
from smollm.tokenizer import SmilesTokenizer

BLOCK_SIZE = 128
N_GENERATE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a SMILES generator.")
    p.add_argument("--config", type=str, required=True,
                   help=f"Architecture name. One of: {sorted(ARCHITECTURES)}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--n-aug", type=int, default=10,
                   help="SMILES augmentation factor (n random orderings per molecule)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP autocast (required for GRU on ROCm/MIOpen — fp16 grads unstable)")
    p.add_argument("--wandb", action="store_true",
                   help="Log to wandb (requires WANDB_PROJECT env var or default `smollm`)")
    return p.parse_args()


def make_model(cfg: dict, tokenizer: SmilesTokenizer) -> torch.nn.Module:
    if cfg.get("model_type") == "gru":
        return make_smiles_gru(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            block_size=BLOCK_SIZE,
            n_layers=cfg["n_layers"],
            hidden_size=cfg["hidden_size"],
        )
    return make_smiles_gpt(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        block_size=BLOCK_SIZE,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        swiglu=cfg["swiglu"],
        rope=cfg["rope"],
        weight_sharing=cfg["weight_sharing"],
    )


def prepare_data(args: argparse.Namespace, tokenizer: SmilesTokenizer, train_smiles: list[str]):
    if args.n_aug > 0:
        aug_path = Path(args.data_dir) / f"zinc250k_aug_x{args.n_aug}.pt"
        if aug_path.exists():
            print(f"Loading cached augmented data from {aug_path}")
            return torch.load(aug_path, weights_only=True)
        print(f"Generating {args.n_aug}x augmented SMILES (one-time cost)...")
        train_data = prepare_augmented_smiles(
            train_smiles, tokenizer, BLOCK_SIZE, args.n_aug, args.seed,
        )
        aug_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_data, aug_path)
        print(f"Saved augmented data to {aug_path} ({train_data.shape})")
        return train_data
    return SmilesDataset(train_smiles, tokenizer, BLOCK_SIZE).data


def main() -> None:
    args = parse_args()

    if args.config not in ARCHITECTURES:
        raise ValueError(f"Unknown config '{args.config}'. Available: {sorted(ARCHITECTURES)}")
    cfg = {"name": args.config, **ARCHITECTURES[args.config]}

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = download_zinc(Path(args.data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)
    print(f"{len(smiles)} canonical SMILES")

    tokenizer = SmilesTokenizer().fit(smiles)
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_smiles, val_smiles = train_val_split(smiles, args.seed)

    train_data = prepare_data(args, tokenizer, train_smiles)
    val_ds = SmilesDataset(val_smiles, tokenizer, BLOCK_SIZE)
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_ds)}")

    loader_kwargs = dict(
        batch_size=args.batch_size, drop_last=True,
        pin_memory=True, num_workers=2, persistent_workers=True,
    )
    dl_generator = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(AugmentedSmilesDataset(train_data),
                          shuffle=True, generator=dl_generator, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = make_model(cfg, tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {cfg['name']}: {n_params:,} params")

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "smollm"),
            group="train",
            name=f"{cfg['name']}-s{args.seed}",
            job_type="train",
            config={**{k: v for k, v in cfg.items() if k != "name"},
                    "config": cfg["name"], "seed": args.seed,
                    "block_size": BLOCK_SIZE, "batch_size": args.batch_size,
                    "epochs": args.epochs, "lr": args.lr,
                    "vocab_size": tokenizer.vocab_size,
                    "label_smoothing": args.label_smoothing, "n_aug": args.n_aug,
                    "warmup": args.warmup, "params": n_params},
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    use_amp = device == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    total_steps = args.epochs * len(train_dl)
    step = 0
    lr_now = args.lr

    for epoch in range(args.epochs):
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        n_batches = 0
        for xb, yb in tqdm(train_dl, desc=f"  Epoch {epoch + 1}/{args.epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            lr_now = cosine_lr(step, args.warmup, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(xb)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), yb.view(-1),
                        ignore_index=tokenizer.pad_id,
                        label_smoothing=args.label_smoothing,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(xb)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), yb.view(-1),
                    ignore_index=tokenizer.pad_id,
                    label_smoothing=args.label_smoothing,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.detach()
            n_batches += 1
            step += 1

        train_loss = (running_loss / n_batches).item()

        model.eval()
        val_running = torch.tensor(0.0, device=device)
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        _, vloss = model(xb, yb)
                else:
                    _, vloss = model(xb, yb)
                val_running += vloss.detach()
                val_count += 1
        val_loss = (val_running / val_count).item() if val_count > 0 else float("nan")

        log_dict = {
            "train/loss": train_loss, "val/loss": val_loss,
            "train/lr": lr_now, "epoch": epoch + 1,
        }
        sample_every = max(args.epochs // 6, 1)
        if (epoch + 1) % sample_every == 0 or epoch + 1 == args.epochs:
            sample_info = log_samples(model, tokenizer, epoch + 1, device=device)
            log_dict["samples/validity"] = sample_info["validity"]
        if use_wandb:
            import wandb
            wandb.log(log_dict)
        print(f"  Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f}")

    print(f"  Generating {N_GENERATE} molecules...")
    train_set = set(train_smiles)
    molecules = generate(model, tokenizer, n=N_GENERATE, temperature=0.8, top_k=40, device=device)
    metrics = evaluate_smiles(molecules, train_set)
    print(
        f"  Validity: {metrics['validity'] * 100:.1f}%  "
        f"Unique: {metrics['uniqueness'] * 100:.1f}%  "
        f"Novel: {metrics['novelty'] * 100:.1f}%  "
        f"IntDiv: {metrics['intdiv']:.3f}"
    )
    if metrics.get("fcd") is not None:
        print(f"  FCD (train-split ref): {metrics['fcd']:.3f}")
    if use_wandb:
        from smollm.sample import log_metrics_to_wandb
        log_metrics_to_wandb(metrics, n_params=n_params, fcd_key="fcd_train_split")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['name']}-s{args.seed}.pt"
    save_checkpoint(model, tokenizer, str(ckpt_path), extra={"epoch": args.epochs})
    print(f"  Saved: {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
