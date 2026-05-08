"""DPO (Direct Preference Optimization) for SMILES generation.

Uses validity as the preference signal: valid molecules are preferred over
invalid ones. After from-scratch or distilled training, DPO fine-tunes the
model to prefer generating valid SMILES.

Note: in our experiments DPO consistently *hurt* both validity and FCD —
this script is included as a robustness check, not a recommended pipeline.

Examples:
    python dpo.py --checkpoint checkpoints/ws-206k-s42.pt
    python dpo.py --checkpoint checkpoints/ws-53k-s42.pt --beta 0.1 --epochs 5
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from smollm._utils import cosine_lr
from smollm.checkpoint import load_checkpoint, save_checkpoint
from smollm.data import download_zinc, load_smiles, train_val_split
from smollm.sample import (
    evaluate_smiles,
    generate,
    log_samples,
    validate_smiles,
)
from smollm.tokenizer import SmilesTokenizer

RDLogger.DisableLog("rdApp.*")

N_GENERATE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO fine-tuning for SMILES generation")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained model checkpoint (from train.py or distill.py)")
    p.add_argument("--n-samples", type=int, default=10_000,
                   help="Number of molecules to sample for preference pairs")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO KL penalty coefficient")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    return p.parse_args()


def _log_probs_for_sequence(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Compute per-sequence log probability summed over non-pad target tokens."""
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != pad_id).float()
    return (token_log_probs * mask).sum(dim=-1)


def build_preference_pairs(
    molecules: list[str],
    tokenizer: SmilesTokenizer,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (chosen, rejected) pairs from generated molecules.

    Chosen = valid SMILES, rejected = invalid SMILES.
    Pairs are matched by approximate length (within 20% or 10 chars).
    """
    valid: list[str] = []
    invalid: list[str] = []

    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(smi)
        else:
            invalid.append(smi)

    if not valid or not invalid:
        return None

    valid.sort(key=len)
    invalid.sort(key=len)

    chosen_ids: list[list[int]] = []
    rejected_ids: list[list[int]] = []

    inv_idx = 0
    for v in valid:
        if inv_idx >= len(invalid):
            break
        v_len = len(v)
        best_idx = inv_idx
        best_diff = abs(len(invalid[inv_idx]) - v_len)
        for j in range(inv_idx, min(inv_idx + 20, len(invalid))):
            diff = abs(len(invalid[j]) - v_len)
            if diff < best_diff:
                best_diff = diff
                best_idx = j
        threshold = max(v_len * 0.2, 10)
        if best_diff > threshold:
            continue

        chosen_ids.append(tokenizer.encode(v))
        rejected_ids.append(tokenizer.encode(invalid[best_idx]))
        inv_idx = best_idx + 1

    if not chosen_ids:
        return None

    def _pad(sequences: list[list[int]]) -> torch.Tensor:
        max_len = min(max(len(s) for s in sequences), block_size)
        padded = []
        for s in sequences:
            s = s[:max_len]
            s = s + [tokenizer.pad_id] * (max_len - len(s))
            padded.append(s)
        return torch.tensor(padded, dtype=torch.long)

    return _pad(chosen_ids), _pad(rejected_ids)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss (Rafailov et al., 2023)."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    block_size = model.config.block_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    data_path = download_zinc(Path(args.data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)
    train_smiles, _ = train_val_split(smiles, args.seed)
    train_set = set(train_smiles)

    print(f"Sampling {args.n_samples} molecules...")
    molecules = generate(
        model, tokenizer, n=args.n_samples,
        temperature=0.8, top_k=40, device=device,
    )

    pre_metrics = validate_smiles(molecules, train_set)
    print(f"Pre-DPO validity: {pre_metrics['validity'] * 100:.1f}%")

    print("Building preference pairs...")
    result = build_preference_pairs(molecules, tokenizer, block_size)
    if result is None:
        print("Cannot build pairs (all valid or all invalid). Exiting.")
        return
    chosen_data, rejected_data = result
    print(f"Built {len(chosen_data)} preference pairs")

    print("Computing reference log probs...")
    ref_model, _ = load_checkpoint(args.checkpoint, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    pad_id = tokenizer.pad_id

    with torch.no_grad():
        ref_chosen_logps_list = []
        ref_rejected_logps_list = []
        bs = args.batch_size
        for i in range(0, len(chosen_data), bs):
            c = chosen_data[i:i + bs].to(device)
            r = rejected_data[i:i + bs].to(device)
            ref_chosen_logps_list.append(
                _log_probs_for_sequence(ref_model, c[:, :-1], c[:, 1:], pad_id).cpu()
            )
            ref_rejected_logps_list.append(
                _log_probs_for_sequence(ref_model, r[:, :-1], r[:, 1:], pad_id).cpu()
            )
        ref_chosen_logps = torch.cat(ref_chosen_logps_list)
        ref_rejected_logps = torch.cat(ref_rejected_logps_list)

    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ckpt_stem = Path(args.checkpoint).stem
    config_name = f"{ckpt_stem}-dpo"
    run_name = f"{config_name}-s{args.seed}"

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "smollm"),
            group="dpo", name=run_name, job_type="dpo",
            config={
                "config": ckpt_stem, "seed": args.seed,
                "checkpoint": args.checkpoint,
                "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
                "n_samples": args.n_samples, "n_pairs": len(chosen_data),
                "params": n_params,
                "pre_dpo_validity": pre_metrics["validity"],
            },
        )

    dataset = TensorDataset(chosen_data, rejected_data, ref_chosen_logps, ref_rejected_logps)
    drop_last = len(dataset) >= args.batch_size
    dl_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=drop_last, generator=dl_generator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        lr_now = args.lr

        for c_batch, r_batch, rc_logps, rr_logps in tqdm(loader, desc=f"DPO epoch {epoch + 1}"):
            c_batch = c_batch.to(device)
            r_batch = r_batch.to(device)
            rc_logps = rc_logps.to(device)
            rr_logps = rr_logps.to(device)

            lr_now = cosine_lr(step, args.warmup, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            policy_chosen = _log_probs_for_sequence(model, c_batch[:, :-1], c_batch[:, 1:], pad_id)
            policy_rejected = _log_probs_for_sequence(model, r_batch[:, :-1], r_batch[:, 1:], pad_id)

            loss = dpo_loss(policy_chosen, policy_rejected, rc_logps, rr_logps, args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            step += 1

        avg_loss = running_loss / n_batches if n_batches else 0
        print(f"  Epoch {epoch + 1}: dpo_loss={avg_loss:.4f}")

        sample_info = log_samples(model, tokenizer, epoch + 1, device=device)

        if use_wandb:
            import wandb
            wandb.log({"dpo/loss": avg_loss, "dpo/lr": lr_now, "epoch": epoch + 1,
                       "samples/validity": sample_info["validity"]})

    print("Evaluating post-DPO...")
    molecules = generate(model, tokenizer, n=N_GENERATE, temperature=0.8, top_k=40, device=device)
    post_metrics = evaluate_smiles(molecules, train_set)
    print(
        f"Post-DPO: validity={post_metrics['validity'] * 100:.1f}%  "
        f"unique={post_metrics['uniqueness'] * 100:.1f}%  "
        f"novel={post_metrics['novelty'] * 100:.1f}%  "
        f"IntDiv: {post_metrics['intdiv']:.3f}"
    )
    if post_metrics.get("fcd") is not None:
        print(f"  FCD (train-split ref): {post_metrics['fcd']:.3f}")

    if use_wandb:
        from smollm.sample import log_metrics_to_wandb
        import wandb
        log_metrics_to_wandb(post_metrics, n_params=n_params, fcd_key="fcd_train_split")
        wandb.run.summary["pre_validity"] = pre_metrics["validity"]
        wandb.run.summary["validity_delta"] = post_metrics["validity"] - pre_metrics["validity"]

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config_name}-s{args.seed}.pt"
    save_checkpoint(model, tokenizer, str(ckpt_path), extra={
        "epoch": args.epochs,
        "dpo_beta": args.beta,
        "source_checkpoint": args.checkpoint,
        "pre_dpo_validity": pre_metrics.get("validity"),
    })
    print(f"Saved: {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
