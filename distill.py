"""Curriculum + knowledge-distillation training.

Examples:
    python distill.py --config ws-53k --teacher checkpoints/ws-206k-s42.pt
    python distill.py --config ws-53k --teacher <ckpt> --alpha 0.3 --temperature 3
    python distill.py --config ws-53k --teacher <ckpt> --no-curriculum
    python distill.py --config ws-53k --teacher <ckpt> --no-distill
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
from smollm.checkpoint import load_checkpoint, make_smiles_gpt, save_checkpoint
from smollm.configs import ARCHITECTURES
from smollm.data import (
    AugmentedSmilesDataset,
    SmilesDataset,
    download_zinc,
    load_smiles,
    train_val_split,
)
from smollm.sample import evaluate_smiles, generate, log_metrics_to_wandb, log_samples
from smollm.tokenizer import SmilesTokenizer

BLOCK_SIZE = 128
N_GENERATE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curriculum + distillation training")
    p.add_argument("--config", type=str, required=True,
                   help=f"Student arch. One of: {sorted(ARCHITECTURES)}")
    p.add_argument("--teacher", type=str, required=True,
                   help="Path to teacher checkpoint (.pt produced by train.py)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Hard-label loss weight (1-alpha for soft KD loss)")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Skip curriculum (length-sorted stages)")
    p.add_argument("--no-distill", action="store_true",
                   help="Skip distillation (curriculum-only training)")
    p.add_argument("--curriculum-stages", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-aug", type=int, default=1)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def sort_by_length(smiles: list[str]) -> list[str]:
    return sorted(smiles, key=len)


def build_curriculum_stages(
    smiles: list[str], tokenizer: SmilesTokenizer, n_stages: int,
) -> list[torch.Tensor]:
    sorted_smi = sort_by_length(smiles)
    stage_size = len(sorted_smi) // n_stages
    stages = []
    for i in range(n_stages):
        end = len(sorted_smi) if i == n_stages - 1 else (i + 1) * stage_size
        stage_smiles = sorted_smi[:end]
        ds = SmilesDataset(stage_smiles, tokenizer, BLOCK_SIZE)
        stages.append(ds.data)
    return stages


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    pad_id: int,
    alpha: float,
    temperature: float,
    label_smoothing: float,
) -> torch.Tensor:
    V = student_logits.size(-1)
    student_flat = student_logits.view(-1, V)
    teacher_flat = teacher_logits.view(-1, V)
    hard_flat = hard_targets.view(-1)

    hard_loss = F.cross_entropy(
        student_flat, hard_flat,
        ignore_index=pad_id, label_smoothing=label_smoothing,
    )

    mask = (hard_flat != pad_id).float()
    student_log_probs = F.log_softmax(student_flat / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_flat / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    soft_loss = (kl * mask).sum() / mask.sum().clamp(min=1)
    soft_loss = soft_loss * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def main() -> None:
    args = parse_args()

    if args.config not in ARCHITECTURES:
        raise ValueError(f"Unknown config '{args.config}'. Available: {sorted(ARCHITECTURES)}")
    cfg = ARCHITECTURES[args.config]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_generator = torch.Generator().manual_seed(args.seed)

    data_path = download_zinc(Path(args.data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)
    tokenizer = SmilesTokenizer().fit(smiles)
    print(f"{len(smiles)} SMILES, vocab={tokenizer.vocab_size}")

    train_smiles, val_smiles = train_val_split(smiles, args.seed)
    val_ds = SmilesDataset(val_smiles, tokenizer, BLOCK_SIZE)

    use_curriculum = not args.no_curriculum
    use_distill = not args.no_distill

    if use_curriculum:
        stages = build_curriculum_stages(train_smiles, tokenizer, args.curriculum_stages)
        epochs_per_stage = args.epochs // args.curriculum_stages
        print(f"Curriculum: {args.curriculum_stages} stages, {epochs_per_stage} epochs each")
        for i, s in enumerate(stages):
            print(f"  Stage {i+1}: {len(s)} samples")
    else:
        ds = SmilesDataset(train_smiles, tokenizer, BLOCK_SIZE)
        stages = [ds.data]
        epochs_per_stage = args.epochs

    student = make_smiles_gpt(
        vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
        block_size=BLOCK_SIZE,
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        swiglu=cfg["swiglu"], rope=cfg["rope"], weight_sharing=cfg["weight_sharing"],
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student {args.config}: {n_params:,} params")

    teacher = None
    if use_distill:
        print(f"Loading teacher from {args.teacher}...")
        teacher, _ = load_checkpoint(args.teacher, device=device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_params = sum(p.numel() for p in teacher.parameters())
        print(f"Teacher: {t_params:,} params (frozen)")

    mode = []
    if use_curriculum:
        mode.append("curriculum")
    if use_distill:
        mode.append("distill")
    mode_str = "+".join(mode) if mode else "baseline"
    config_name = f"{args.config}-{mode_str}"

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "smollm"),
            group="distill",
            name=f"{config_name}-s{args.seed}",
            job_type="train",
            config={**{k: v for k, v in cfg.items() if k != "name"},
                    "config": args.config, "seed": args.seed,
                    "block_size": BLOCK_SIZE, "batch_size": args.batch_size,
                    "epochs": args.epochs, "lr": args.lr, "params": n_params,
                    "label_smoothing": args.label_smoothing, "warmup": args.warmup,
                    "alpha": args.alpha, "temperature": args.temperature,
                    "curriculum": use_curriculum, "distillation": use_distill,
                    "curriculum_stages": args.curriculum_stages if use_curriculum else 0,
                    "teacher": args.teacher if use_distill else None},
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    loader_kwargs = dict(
        batch_size=args.batch_size, drop_last=True,
        pin_memory=True, num_workers=2, persistent_workers=True,
    )
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    total_steps = sum(
        epochs_per_stage * (len(s) // args.batch_size)
        for s in stages
    )
    step = 0
    global_epoch = 0

    for stage_idx, stage_data in enumerate(stages):
        train_ds = AugmentedSmilesDataset(stage_data)
        train_dl = DataLoader(train_ds, shuffle=True, generator=dl_generator, **loader_kwargs)
        if use_curriculum:
            print(f"\n--- Curriculum stage {stage_idx + 1}/{len(stages)}: "
                  f"{len(stage_data)} samples ---")

        for epoch_in_stage in range(epochs_per_stage):
            global_epoch += 1
            student.train()
            running_loss = torch.tensor(0.0, device=device)
            n_batches = 0
            lr_now = args.lr
            for xb, yb in tqdm(train_dl, desc=f"  Epoch {global_epoch}/{args.epochs}", leave=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                lr_now = cosine_lr(step, args.warmup, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        student_logits, _ = student(xb)
                        if use_distill and teacher is not None:
                            with torch.no_grad():
                                teacher_logits, _ = teacher(xb)
                            loss = distillation_loss(
                                student_logits, teacher_logits, yb,
                                tokenizer.pad_id, args.alpha,
                                args.temperature, args.label_smoothing,
                            )
                        else:
                            loss = F.cross_entropy(
                                student_logits.view(-1, student_logits.size(-1)),
                                yb.view(-1), ignore_index=tokenizer.pad_id,
                                label_smoothing=args.label_smoothing,
                            )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    student_logits, _ = student(xb)
                    if use_distill and teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(xb)
                        loss = distillation_loss(
                            student_logits, teacher_logits, yb,
                            tokenizer.pad_id, args.alpha,
                            args.temperature, args.label_smoothing,
                        )
                    else:
                        loss = F.cross_entropy(
                            student_logits.view(-1, student_logits.size(-1)),
                            yb.view(-1), ignore_index=tokenizer.pad_id,
                            label_smoothing=args.label_smoothing,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                running_loss += loss.detach()
                n_batches += 1
                step += 1
            train_loss = (running_loss / n_batches).item()

            student.eval()
            val_running = torch.tensor(0.0, device=device)
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    if scaler is not None:
                        with torch.amp.autocast("cuda"):
                            _, vloss = student(xb, yb)
                    else:
                        _, vloss = student(xb, yb)
                    val_running += vloss.detach()
                    val_count += 1
            val_loss = (val_running / val_count).item() if val_count > 0 else float("nan")

            log_dict = {
                "train/loss": train_loss, "val/loss": val_loss,
                "train/lr": lr_now, "epoch": global_epoch,
                "curriculum_stage": stage_idx + 1,
            }
            sample_every = max(args.epochs // 6, 1)
            if global_epoch % sample_every == 0 or global_epoch == args.epochs:
                sample_info = log_samples(student, tokenizer, global_epoch, device=device)
                log_dict["samples/validity"] = sample_info["validity"]
            if use_wandb:
                import wandb
                wandb.log(log_dict)
            print(f"  Epoch {global_epoch}: train={train_loss:.4f} val={val_loss:.4f}")

    print(f"  Generating {N_GENERATE} molecules...")
    train_set = set(train_smiles)
    molecules = generate(student, tokenizer, n=N_GENERATE, temperature=0.8, top_k=40, device=device)
    metrics = evaluate_smiles(molecules, train_set)
    print(
        f"  Validity: {metrics['validity'] * 100:.1f}%  "
        f"Unique: {metrics['uniqueness'] * 100:.1f}%  "
        f"Novel: {metrics['novelty'] * 100:.1f}%  "
        f"IntDiv: {metrics['intdiv']:.3f}"
    )
    if metrics.get("fcd") is not None:
        print(f"  FCD: {metrics['fcd']:.3f}")
    if use_wandb:
        log_metrics_to_wandb(metrics, n_params=n_params)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config_name}-s{args.seed}.pt"
    save_checkpoint(student, tokenizer, str(ckpt_path))
    print(f"  Saved: {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
