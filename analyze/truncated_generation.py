"""Generate molecules at different inference-time pass counts and classify errors.

Takes a trained weight-shared model and truncates to 1, 2, 3, 4, 5, 6, 7, 8
passes at inference time. For each truncation level, generates molecules and
categorizes WHY invalid ones fail (unmatched brackets, unclosed rings, valence
violations, other parse errors).

This directly shows what chemistry the shared block learns at each pass.

Usage:
    python analyze/truncated_generation.py \
        --checkpoint checkpoints/ws-e128-x8.pt

    python analyze/truncated_generation.py \
        --checkpoint checkpoints/ws-e64-x8.pt \
        --n 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import RDLogger

from smollm._meta import build_eval_meta
from smollm.checkpoint import load_checkpoint
from smollm.classify import classify_error
from smollm.model import GPT
from smollm.sample import generate

RDLogger.DisableLog("rdApp.*")


# ── Truncated model wrapper ─────────────────────────────────────────────


class TruncatedGPT(torch.nn.Module):
    """Wraps a weight-shared GPT and runs only `n_passes` through the shared block."""

    def __init__(self, model: GPT, n_passes: int) -> None:
        super().__init__()
        self.model = model
        self.n_passes = n_passes
        self.config = model.config

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
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
        logits = m.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=m.config.pad_id,
            )
        return logits, loss


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncated inference-time generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=1000, help="Molecules per truncation level")
    parser.add_argument("--passes", default="1,2,3,4,5,6,7,8", help="Comma-separated pass counts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/truncated_generation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pass_counts = [int(p) for p in args.passes.split(",")]

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    from typing import cast
    loaded, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    model = cast(GPT, loaded)
    config = model.config
    model.eval()

    assert config.weight_sharing, "This analysis requires a weight-shared model"
    print(f"Model: n_embd={config.n_embd}, n_layer={config.n_layer}, weight_sharing={config.weight_sharing}")
    print(f"Pass counts: {pass_counts}")
    print(f"Generating {args.n} molecules per truncation level")
    print()

    results = {}

    for n_passes in pass_counts:
        print(f"--- {n_passes} pass{'es' if n_passes != 1 else ''} ---")
        truncated = TruncatedGPT(model, n_passes)

        torch.manual_seed(args.seed)  # same seed each level for comparability
        molecules = generate(truncated, tokenizer, n=args.n, device=args.device)

        # Classify each molecule
        errors: dict[str, int] = {
            "valid": 0,
            "unmatched_brackets": 0,
            "unclosed_rings": 0,
            "valence_error": 0,
            "parse_error": 0,
        }
        for smi in molecules:
            cat = classify_error(smi)
            errors[cat] += 1

        total = len(molecules)
        validity = errors["valid"] / total

        results[n_passes] = {
            "n": total,
            "validity": validity,
            "errors": errors,
            "error_rates": {k: v / total for k, v in errors.items()},
        }

        # Print summary
        print(f"  Validity: {validity:.1%}")
        for err_type, count in errors.items():
            if err_type != "valid" and count > 0:
                print(f"  {err_type}: {count} ({count/total:.1%})")
        print()

    # Summary table
    print("=" * 80)
    print(f"  {'Passes':<8} {'Valid':>7} {'Brackets':>10} {'Rings':>10} {'Valence':>10} {'Parse':>10}")
    print("-" * 80)
    for n_passes in pass_counts:
        r = results[n_passes]["error_rates"]
        print(
            f"  {n_passes:<8} {r['valid']:>6.1%} "
            f"{r['unmatched_brackets']:>9.1%} "
            f"{r['unclosed_rings']:>9.1%} "
            f"{r['valence_error']:>9.1%} "
            f"{r['parse_error']:>9.1%}"
        )
    print("=" * 80)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    serializable["_meta"] = {
        "checkpoint": args.checkpoint,
        "n_per_level": args.n,
        "n_embd": config.n_embd,
        "pass_counts": pass_counts,
        **build_eval_meta(Path(args.checkpoint)),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
