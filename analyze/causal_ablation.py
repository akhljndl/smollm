"""Causal ablation: zero out individual attention heads and measure which chemistry rules break.

For each head (0-3) at each pass (1-8), we ablate (zero) that head's
contribution and generate molecules. By comparing the error profile to
the unablated baseline, we identify which heads are causally responsible
for bracket matching, ring closure, and valence checking.

Usage:
    python analyze/causal_ablation.py \
        --checkpoint checkpoints/ws-e128-x8.pt

    python analyze/causal_ablation.py \
        --checkpoint checkpoints/ws-e128-x8.pt \
        --n 200 --passes 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from smollm._meta import build_eval_meta
from smollm.checkpoint import load_checkpoint
from smollm.classify import classify_error
from smollm.model import GPT
from smollm.tokenizer import SmilesTokenizer


# ── Ablated generation ──────────────────────────────────────────────────


class HeadAblatedGPT(torch.nn.Module):
    """Wrapper that zeros out a specific attention head at a specific pass."""

    def __init__(self, model: GPT, ablate_head: int, ablate_pass: int) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.ablate_head = ablate_head
        self.ablate_pass = ablate_pass
        self._hook_handle = None
        self._current_pass = 0

    def _install_hook(self) -> None:
        """Hook into c_proj to zero out the ablated head's contribution."""
        n_head = self.model.config.n_head
        head_dim = self.model.config.n_embd // n_head

        def pre_hook_fn(module: torch.nn.Module, input: tuple) -> tuple:
            if self._current_pass == self.ablate_pass:
                # input[0] is the pre-projection tensor: (B, T, n_embd)
                # It's the concatenated head outputs. Zero out the ablated head's slice.
                x = input[0].clone()
                start = self.ablate_head * head_dim
                end = start + head_dim
                x[:, :, start:end] = 0.0
                return (x,)
            return input

        # Pre-hook on c_proj (modifies input before forward, no recursion)
        self._hook_handle = self.model.shared_block.attn.c_proj.register_forward_pre_hook(pre_hook_fn)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

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

        for p in range(m.config.n_layer):
            self._current_pass = p
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


@torch.no_grad()
def generate_ablated(
    model: HeadAblatedGPT,
    tokenizer: SmilesTokenizer,
    n: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    max_len: int = 120,
    device: str = "cpu",
) -> list[str]:
    model.eval()
    model._install_hook()
    try:
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
    finally:
        model._remove_hook()


def run_error_analysis(molecules: list[str]) -> dict[str, float]:
    """Classify errors and return rates."""
    errors: dict[str, int] = {
        "valid": 0, "unmatched_brackets": 0, "unclosed_rings": 0,
        "valence_error": 0, "parse_error": 0,
    }
    for smi in molecules:
        errors[classify_error(smi)] += 1
    n = len(molecules)
    return {k: v / n for k, v in errors.items()}


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal head ablation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=300, help="Molecules per ablation")
    parser.add_argument("--passes", default="0,1,2,3,4,5,6,7", help="Which passes to ablate at")
    parser.add_argument("--heads", default=None, help="Which heads to ablate (default: all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/causal_ablation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pass_list = [int(p) for p in args.passes.split(",")]

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    loaded, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    model = cast(GPT, loaded)
    config = model.config
    assert config.weight_sharing, "causal_ablation requires a weight-shared model (uses model.shared_block)"
    model.eval()

    n_heads = config.n_head
    head_list = [int(h) for h in args.heads.split(",")] if args.heads else list(range(n_heads))

    print(f"Model: n_embd={config.n_embd}, n_head={n_heads}, n_layer={config.n_layer}")
    print(f"Ablating heads {head_list} at passes {pass_list}")
    print(f"Generating {args.n} molecules per ablation ({len(head_list) * len(pass_list) + 1} runs total)")
    print()

    # Baseline (no ablation)
    print("=== Baseline (no ablation) ===")
    torch.manual_seed(args.seed)
    from smollm.sample import generate
    baseline_mols = generate(model, tokenizer, n=args.n, temperature=0.8, top_k=40, device=args.device)
    baseline = run_error_analysis(baseline_mols)
    print(f"  Valid: {baseline['valid']:.1%}  Brackets: {baseline['unmatched_brackets']:.1%}  "
          f"Rings: {baseline['unclosed_rings']:.1%}  Valence: {baseline['valence_error']:.1%}")
    print()

    results = {"baseline": baseline}

    # Ablation sweep
    for pass_idx in pass_list:
        for head_idx in head_list:
            label = f"pass{pass_idx}_head{head_idx}"
            print(f"--- Ablating head {head_idx} at pass {pass_idx} ---")

            ablated = HeadAblatedGPT(model, ablate_head=head_idx, ablate_pass=pass_idx)
            torch.manual_seed(args.seed)
            mols = generate_ablated(ablated, tokenizer, n=args.n, device=args.device)
            errs = run_error_analysis(mols)

            # Compute delta from baseline
            delta_valid = errs["valid"] - baseline["valid"]
            delta_brackets = errs["unmatched_brackets"] - baseline["unmatched_brackets"]
            delta_rings = errs["unclosed_rings"] - baseline["unclosed_rings"]
            delta_valence = errs["valence_error"] - baseline["valence_error"]

            results[label] = {
                "errors": errs,
                "delta_valid": delta_valid,
                "delta_brackets": delta_brackets,
                "delta_rings": delta_rings,
                "delta_valence": delta_valence,
            }

            print(f"  Valid: {errs['valid']:.1%} ({delta_valid:+.1%})  "
                  f"Brackets: {errs['unmatched_brackets']:.1%} ({delta_brackets:+.1%})  "
                  f"Rings: {errs['unclosed_rings']:.1%} ({delta_rings:+.1%})  "
                  f"Valence: {errs['valence_error']:.1%} ({delta_valence:+.1%})")

    # Summary: find the most impactful ablations
    print()
    print("=" * 90)
    print(f"  {'Ablation':<20} {'Valid':>8} {'dValid':>8} {'dBracket':>10} {'dRing':>8} {'dValence':>10}")
    print("-" * 90)
    print(f"  {'baseline':<20} {baseline['valid']:>7.1%} {'--':>8} {'--':>10} {'--':>8} {'--':>10}")
    for key in sorted(results.keys()):
        if key == "baseline":
            continue
        r = results[key]
        print(f"  {key:<20} {r['errors']['valid']:>7.1%} {r['delta_valid']:>+7.1%} "
              f"{r['delta_brackets']:>+9.1%} {r['delta_rings']:>+7.1%} {r['delta_valence']:>+9.1%}")
    print("=" * 90)

    # Identify heads with largest causal effect on each error type
    print()
    print("Most impactful ablations:")
    for err_type in ["delta_brackets", "delta_rings", "delta_valence"]:
        worst = max(
            ((k, v[err_type]) for k, v in results.items() if k != "baseline"),
            key=lambda x: x[1],
        )
        if worst[1] > 0.01:
            print(f"  {err_type.replace('delta_', '').upper()}: {worst[0]} -> +{worst[1]:.1%} increase")
        else:
            print(f"  {err_type.replace('delta_', '').upper()}: no single head dominates")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **results,
        "_meta": {
            "checkpoint": args.checkpoint,
            **build_eval_meta(Path(args.checkpoint)),
        },
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
