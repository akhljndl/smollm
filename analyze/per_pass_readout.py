"""Per-pass linear readout heads: behavioral error classification from x8 internals.

Trains one independent linear head per pass of a frozen weight-shared x8 model,
then generates molecules through each per-pass head and classifies errors.
Closes the training-depth confound in Table 2 by showing the same
brackets-then-rings-then-valence ordering inside the working x8 model.

Usage:
    python analyze/per_pass_readout.py \
        --checkpoint checkpoints/ws-e128-x8.pt \
        --device cuda

    python analyze/per_pass_readout.py \
        --checkpoint checkpoints/ws-e64-x8.pt \
        --n-generate 500 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from smollm.checkpoint import load_checkpoint
from smollm.data import download_zinc, load_smiles, train_val_split
from smollm.model import GPT
from smollm.tokenizer import SmilesTokenizer

from smollm._meta import build_eval_meta
from smollm.classify import classify_error


def extract_single_pass(
    model: GPT,
    data: torch.Tensor,
    target_pass: int,
    device: str,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (activations, targets) for a single pass from frozen model.

    Runs the model through all passes up to target_pass, collects the
    ln_f-normalized residual stream at that pass, and returns (X, Y) on CPU.
    """
    model.eval()
    pad_id = model.config.pad_id

    all_X: list[torch.Tensor] = []
    all_Y: list[torch.Tensor] = []

    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc=f"Extracting pass {target_pass + 1}", leave=False):
            inp = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)
            _, T = inp.size()

            x = model.tok_emb(inp)
            if model.config.rope:
                x = model.drop(x)
                rope_cos = model.rope_cos[:, :, :T, :]
                rope_sin = model.rope_sin[:, :, :T, :]
            else:
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                x = model.drop(x + model.pos_emb(pos))
                rope_cos = rope_sin = None

            for _ in range(target_pass + 1):
                x = model.shared_block(x, rope_cos, rope_sin)

            normed = model.ln_f(x)
            mask = tgt != pad_id
            all_X.append(normed[mask].cpu())
            all_Y.append(tgt[mask].cpu())

    return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)


def train_readout_head(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_embd: int,
    vocab_size: int,
    device: str,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 4096,
) -> nn.Linear:
    """Train a single linear readout head on extracted representations."""
    head = nn.Linear(n_embd, vocab_size, bias=False).to(device)
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = head(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        print(f"    epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    head.eval()
    return head


class PerPassReadoutGPT(nn.Module):
    """Wrapper: runs frozen model for n_passes, applies a custom readout head."""

    def __init__(self, model: GPT, readout_head: nn.Linear, n_passes: int) -> None:
        super().__init__()
        self.model = model
        self.readout_head = readout_head
        self.n_passes = n_passes
        self.config = model.config

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        m = self.model
        x = m.tok_emb(idx)

        if m.config.rope:
            x = m.drop(x)
            rope_cos = m.rope_cos[:, :, :T, :]
            rope_sin = m.rope_sin[:, :, :T, :]
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = m.drop(x + m.pos_emb(pos))
            rope_cos = rope_sin = None

        for _ in range(self.n_passes):
            x = m.shared_block(x, rope_cos, rope_sin)

        x = m.ln_f(x)
        logits = self.readout_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=m.config.pad_id,
            )
        return logits, loss


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: SmilesTokenizer,
    n: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    max_len: int = 120,
    device: str = "cpu",
) -> list[str]:
    model.eval()
    molecules: list[str] = []
    block_size = model.config.block_size

    for _ in tqdm(range(n), desc="Generating", leave=False):
        idx = torch.tensor([[tokenizer.bos_id]], device=device)
        for _ in range(max_len):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break
        molecules.append(tokenizer.decode(idx[0].tolist()))

    return molecules


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-pass readout head experiment")
    parser.add_argument("--checkpoint", required=True, help="Path to ws-eN-x8 checkpoint")
    parser.add_argument("--n-generate", type=int, default=500, help="Molecules per pass")
    parser.add_argument("--epochs", type=int, default=5, help="Readout head training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Readout head learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Extraction batch size")
    parser.add_argument("--head-batch-size", type=int, default=4096, help="Head training batch size")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Output JSON path (auto-named if omitted)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    loaded, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    from typing import cast as _cast
    model = _cast(GPT, loaded)
    config = model.config
    assert config.weight_sharing, "Requires a weight-shared model"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_passes = config.n_layer
    n_embd = config.n_embd
    vocab_size = config.vocab_size

    print(f"Model: n_embd={n_embd}, n_layer={n_passes}, params={sum(p.numel() for p in model.parameters()):,}")

    # Load training data (same split as train.py)
    data_path = download_zinc(Path(args.data_dir) / "zinc250k.csv")
    smiles = load_smiles(data_path)

    train_smiles, _ = train_val_split(smiles, args.seed)

    # Encode training data (no augmentation -- training tiny linear heads)
    print(f"Encoding {len(train_smiles)} training SMILES...")
    block_size = config.block_size
    encoded = []
    for smi in train_smiles:
        ids = tokenizer.encode(smi)
        if len(ids) <= block_size:
            ids = ids + [tokenizer.pad_id] * (block_size - len(ids))
            encoded.append(ids)
    train_data = torch.tensor(encoded, dtype=torch.long)
    print(f"Training sequences: {len(train_data)}")

    # Process one pass at a time to stay memory-safe
    results: dict[int, dict] = {}
    n_tokens = 0

    for p in range(n_passes):
        pass_num = p + 1
        print(f"\n{'=' * 60}")
        print(f"Pass {pass_num}/{n_passes}")
        print(f"{'=' * 60}")

        # Extract representations for this pass
        X, Y = extract_single_pass(model, train_data, p, args.device, args.batch_size)
        n_tokens = X.shape[0]
        if p == 0:
            print(f"  Non-pad tokens: {n_tokens:,}")

        # Train readout head
        print(f"  Training readout head ({n_embd} -> {vocab_size})...")
        head = train_readout_head(
            X, Y, n_embd, vocab_size, args.device,
            epochs=args.epochs, lr=args.lr, batch_size=args.head_batch_size,
        )
        del X, Y

        # Build wrapper and generate
        print(f"  Generating {args.n_generate} molecules...")
        wrapper = PerPassReadoutGPT(model, head, pass_num)
        torch.manual_seed(args.seed)
        molecules = generate(wrapper, tokenizer, n=args.n_generate, device=args.device)

        # Classify errors
        errors: dict[str, int] = {
            "valid": 0,
            "unmatched_brackets": 0,
            "unclosed_rings": 0,
            "valence_error": 0,
            "parse_error": 0,
        }
        for smi in molecules:
            errors[classify_error(smi)] += 1

        total = len(molecules)
        validity = errors["valid"] / total

        results[pass_num] = {
            "n": total,
            "validity": validity,
            "errors": errors,
            "error_rates": {k: v / total for k, v in errors.items()},
        }

        print(f"  Validity: {validity:.1%}")
        for err_type, count in errors.items():
            if err_type != "valid" and count > 0:
                print(f"    {err_type}: {count} ({count / total:.1%})")

        del head, wrapper

    # Summary table (Table 2b format)
    print(f"\n{'=' * 80}")
    print(f"  Table 2b: Per-pass readout error classification (n_embd={n_embd})")
    print(f"  {'Passes':<8} {'Valid':>7} {'Brackets':>10} {'Rings':>10} {'Valence':>10} {'Parse':>10}")
    print(f"{'-' * 80}")
    for pass_num in range(1, n_passes + 1):
        r = results[pass_num]["error_rates"]
        print(
            f"  {pass_num:<8} {r['valid']:>6.1%} "
            f"{r['unmatched_brackets']:>9.1%} "
            f"{r['unclosed_rings']:>9.1%} "
            f"{r['valence_error']:>9.1%} "
            f"{r['parse_error']:>9.1%}"
        )
    print(f"{'=' * 80}")

    # Save
    if args.output is None:
        output_path = Path(f"results/per_pass_readout_e{n_embd}.json")
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {str(k): v for k, v in results.items()}
    serializable["_meta"] = {
        "checkpoint": args.checkpoint,
        "n_embd": n_embd,
        "n_passes": n_passes,
        "n_generate": args.n_generate,
        "n_train_tokens": n_tokens,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        **build_eval_meta(Path(args.checkpoint)),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
