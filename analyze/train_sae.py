"""Train sparse autoencoders on weight-shared transformer activations.

Collects residual-stream activations after each pass of the shared block,
trains one SAE per pass, and analyzes which learned features correspond to
specific chemical concepts (brackets, rings, atoms, valence-related depth).

Usage:
    python analyze/train_sae.py \
        --checkpoint checkpoints/ws-206k.pt \
        --n-smiles 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast

from smollm._meta import build_eval_meta
from smollm.checkpoint import load_checkpoint
from smollm.model import GPT, TransformerConfig
from smollm.tokenizer import SmilesTokenizer

SMILES_PATTERN = r"Cl|Br|@@|\[.*?\]|[CNOSPFIcnosp()=\#\-\+\.\/\\@0-9]"
ATOMS = set("CNOSPFIcnosp")


# ── Part 1: Collect activations ───────────────────────────────────────────


def load_model(
    checkpoint: str, device: str = "cpu"
) -> tuple[GPT, SmilesTokenizer, TransformerConfig]:
    loaded, tokenizer = load_checkpoint(checkpoint, device=device)
    model = cast(GPT, loaded)
    model.eval()
    return model, tokenizer, model.config


def load_smiles(data_path: Path, n: int, seed: int = 42) -> list[str]:
    if not data_path.exists():
        from smollm.data import download_zinc

        download_zinc(data_path)
    with open(data_path) as f:
        all_smiles = [row["smiles"] for row in csv.DictReader(f)]
    rng = np.random.RandomState(seed)
    return list(rng.choice(all_smiles, size=min(n, len(all_smiles)), replace=False))


def collect_activations(
    model: GPT,
    tokenizer: SmilesTokenizer,
    smiles_list: list[str],
    device: str = "cpu",
) -> tuple[dict[int, torch.Tensor], list[str], list[list[str]]]:
    """Collect residual stream activations after each pass.

    Returns:
        activations: {pass_idx: tensor(total_tokens, n_embd)}
        valid_smiles: list of SMILES that were successfully processed
        tokens_per_smiles: list of token lists for each valid SMILES
    """
    n_passes = model.config.n_layer
    per_pass: dict[int, list[torch.Tensor]] = {p: [] for p in range(n_passes)}
    valid_smiles: list[str] = []
    tokens_per_smiles: list[list[str]] = []

    with torch.no_grad():
        for smi in smiles_list:
            toks = re.findall(SMILES_PATTERN, smi)
            if not toks or len(toks) >= 120:
                continue
            try:
                ids = tokenizer.encode(smi)
            except (KeyError, IndexError):
                continue

            idx = torch.tensor([ids], device=device)
            _, T = idx.size()
            x = model.tok_emb(idx)

            if model.config.rope:
                x = model.drop(x)
                rope_cos = model.rope_cos[:, :, :T, :]
                rope_sin = model.rope_sin[:, :, :T, :]
            else:
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                x = model.drop(x + model.pos_emb(pos))
                rope_cos = rope_sin = None

            n_tok = len(toks)
            for p in range(n_passes):
                block = (
                    model.shared_block
                    if model.config.weight_sharing
                    else model.blocks[p]
                )
                x = block(x, rope_cos, rope_sin)
                per_pass[p].append(x[0, 1 : 1 + n_tok].cpu())

            valid_smiles.append(smi)
            tokens_per_smiles.append(toks)

    activations = {p: torch.cat(tensors, dim=0) for p, tensors in per_pass.items()}
    return activations, valid_smiles, tokens_per_smiles


# ── Part 2: Sparse autoencoder ────────────────────────────────────────────


class SparseAutoencoder(nn.Module):
    def __init__(self, n_input: int, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(n_input, n_features)
        self.decoder = nn.Linear(n_features, n_input, bias=False)
        # Linear weights are (out, in), so dictionary atoms are columns -> dim=0
        with torch.no_grad():
            self.decoder.weight.copy_(
                F.normalize(self.encoder.weight.data.T.clone(), dim=0)
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = F.relu(self.encoder(x))
        recon = self.decoder(latent)
        return recon, latent, x


def train_sae(
    activations: torch.Tensor,
    n_features: int = 512,
    lr: float = 3e-4,
    l1_coeff: float = 1e-3,
    batch_size: int = 256,
    n_steps: int = 5000,
) -> tuple[SparseAutoencoder, dict[str, list[float]]]:
    n_input = activations.shape[1]
    sae = SparseAutoencoder(n_input, n_features)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    logs: dict[str, list[float]] = {"mse": [], "l1": [], "total": [], "l0": []}
    n_samples = activations.shape[0]

    for step in range(n_steps):
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = activations[indices]

        recon, latent, _ = sae(batch)
        mse_loss = F.mse_loss(recon, batch)
        l1_loss = latent.abs().mean()
        loss = mse_loss + l1_coeff * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sae.decoder.weight.copy_(F.normalize(sae.decoder.weight, dim=0))

        if step % 1000 == 0 or step == n_steps - 1:
            l0 = (latent > 0).float().sum(dim=1).mean().item()
            logs["mse"].append(mse_loss.item())
            logs["l1"].append(l1_loss.item())
            logs["total"].append(loss.item())
            logs["l0"].append(l0)
            print(
                f"    step {step:5d}  mse={mse_loss.item():.4f}  "
                f"l1={l1_loss.item():.4f}  L0={l0:.1f}"
            )

    return sae, logs


# ── Part 3: Analyze features ─────────────────────────────────────────────


def compute_token_properties(
    tokens_per_smiles: list[list[str]],
) -> dict[str, np.ndarray]:
    """Compute per-token chemical properties aligned to the flat activation tensor."""
    is_bracket: list[int] = []
    is_ring_digit: list[int] = []
    is_atom: list[int] = []
    bracket_depth: list[int] = []
    ring_state: list[int] = []

    for toks in tokens_per_smiles:
        depth = 0
        open_rings: set[str] = set()
        for tok in toks:
            is_bracket.append(1 if tok in ("(", ")") else 0)
            is_ring_digit.append(1 if tok in "0123456789" else 0)
            is_atom.append(
                1
                if tok in ATOMS or tok in ("Cl", "Br") or tok.startswith("[")
                else 0
            )
            bracket_depth.append(depth)
            ring_state.append(len(open_rings))
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth = max(0, depth - 1)
            elif tok in "0123456789":
                if tok in open_rings:
                    open_rings.remove(tok)
                else:
                    open_rings.add(tok)

    return {
        "is_bracket": np.array(is_bracket),
        "is_ring_digit": np.array(is_ring_digit),
        "is_atom": np.array(is_atom),
        "bracket_depth": np.array(bracket_depth),
        "ring_state": np.array(ring_state),
    }


def analyze_features(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    properties: dict[str, np.ndarray],
    tokens_per_smiles: list[list[str]],
    top_k: int = 5,
) -> list[dict]:
    """Find the most interpretable SAE features via correlation with chemical properties."""
    with torch.no_grad():
        _, latent, _ = sae(activations)
    latent_np = latent.numpy()  # (n_tokens, n_features)
    n_features = latent_np.shape[1]

    # Flatten all tokens for top-activating example lookup
    all_tokens: list[str] = []
    for toks in tokens_per_smiles:
        all_tokens.extend(toks)

    # Compute correlation of each feature with each property
    feature_scores: list[dict] = []
    for feat_idx in range(n_features):
        feat_acts = latent_np[:, feat_idx]
        if feat_acts.std() < 1e-8:
            continue
        best_corr = 0.0
        best_prop = ""
        correlations: dict[str, float] = {}
        for prop_name, prop_vals in properties.items():
            pv = prop_vals.astype(float)
            if pv.std() < 1e-8:
                continue
            corr = float(np.corrcoef(feat_acts, pv)[0, 1])
            correlations[prop_name] = corr
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_prop = prop_name

        # Top activating tokens
        top_indices = np.argsort(feat_acts)[-10:][::-1]
        top_tokens = [all_tokens[i] for i in top_indices]
        top_acts = [float(feat_acts[i]) for i in top_indices]

        # Fraction of activation mass on each token type
        total_act = feat_acts.sum()
        if total_act > 0:
            bracket_frac = float(feat_acts[properties["is_bracket"] == 1].sum() / total_act)
            ring_frac = float(feat_acts[properties["is_ring_digit"] == 1].sum() / total_act)
            atom_frac = float(feat_acts[properties["is_atom"] == 1].sum() / total_act)
        else:
            bracket_frac = ring_frac = atom_frac = 0.0

        feature_scores.append(
            {
                "feature_idx": feat_idx,
                "best_property": best_prop,
                "best_correlation": best_corr,
                "correlations": correlations,
                "top_tokens": top_tokens,
                "top_activations": top_acts,
                "bracket_frac": bracket_frac,
                "ring_frac": ring_frac,
                "atom_frac": atom_frac,
            }
        )

    # Sort by absolute best correlation
    feature_scores.sort(key=lambda x: abs(x["best_correlation"]), reverse=True)
    return feature_scores[:top_k]


# ── Part 4: Cross-pass comparison ─────────────────────────────────────────


def cross_pass_analysis(
    saes: dict[int, SparseAutoencoder],
    activations: dict[int, torch.Tensor],
    properties: dict[str, np.ndarray],
) -> dict[str, dict[int, float]]:
    """For each property, compute the max feature correlation at each pass."""
    prop_by_pass: dict[str, dict[int, float]] = defaultdict(dict)

    for pass_idx, sae in saes.items():
        with torch.no_grad():
            _, latent, _ = sae(activations[pass_idx])
        latent_np = latent.numpy()

        for prop_name, prop_vals in properties.items():
            pv = prop_vals.astype(float)
            if pv.std() < 1e-8:
                prop_by_pass[prop_name][pass_idx] = 0.0
                continue
            best = 0.0
            for feat_idx in range(latent_np.shape[1]):
                feat_acts = latent_np[:, feat_idx]
                if feat_acts.std() < 1e-8:
                    continue
                corr = abs(float(np.corrcoef(feat_acts, pv)[0, 1]))
                if corr > best:
                    best = corr
            prop_by_pass[prop_name][pass_idx] = best

    return dict(prop_by_pass)


def print_results(
    top_features: dict[int, list[dict]],
    cross_pass: dict[str, dict[int, float]],
    n_passes: int,
) -> None:
    print("\n" + "=" * 70)
    print("TOP INTERPRETABLE SAE FEATURES PER PASS")
    print("=" * 70)

    for pass_idx in sorted(top_features.keys()):
        print(f"\n--- Pass {pass_idx} ---")
        for rank, feat in enumerate(top_features[pass_idx]):
            print(
                f"  #{rank+1} Feature {feat['feature_idx']:3d}  "
                f"best_prop={feat['best_property']:<15s}  "
                f"r={feat['best_correlation']:+.3f}  "
                f"bracket={feat['bracket_frac']:.0%}  "
                f"ring={feat['ring_frac']:.0%}  "
                f"atom={feat['atom_frac']:.0%}"
            )
            print(f"       top tokens: {feat['top_tokens'][:5]}")

    print("\n" + "=" * 70)
    print("MAX FEATURE-PROPERTY CORRELATION ACROSS PASSES")
    print("=" * 70)
    header = f"  {'Property':<15s}"
    for p in range(n_passes):
        header += f"  Pass {p:>2d}"
    print(header)
    print("-" * 70)
    for prop_name, pass_corrs in cross_pass.items():
        row = f"  {prop_name:<15s}"
        values = []
        for p in range(n_passes):
            v = pass_corrs.get(p, 0.0)
            row += f"  {v:>6.3f}"
            values.append(v)
        # Peak pass
        peak = int(np.argmax(values))
        row += f"   peak@{peak}"
        print(row)
    print("=" * 70)


def save_results(
    top_features: dict[int, list[dict]],
    cross_pass: dict[str, dict[int, float]],
    output_path: Path,
    meta: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "top_features": {
            str(p): [
                {
                    "feature_idx": f["feature_idx"],
                    "best_property": f["best_property"],
                    "best_correlation": f["best_correlation"],
                    "correlations": f["correlations"],
                    "top_tokens": f["top_tokens"],
                    "bracket_frac": f["bracket_frac"],
                    "ring_frac": f["ring_frac"],
                    "atom_frac": f["atom_frac"],
                }
                for f in feats
            ]
            for p, feats in top_features.items()
        },
        "cross_pass": {
            prop: {str(k): v for k, v in vals.items()}
            for prop, vals in cross_pass.items()
        },
        "_meta": meta,
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


def save_figure(
    cross_pass: dict[str, dict[int, float]], n_passes: int, output_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure")
        return

    props = list(cross_pass.keys())
    data = np.zeros((len(props), n_passes))
    for i, prop in enumerate(props):
        for p in range(n_passes):
            data[i, p] = cross_pass[prop].get(p, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(props)))
    ax.set_yticklabels(props)
    ax.set_xticks(range(n_passes))
    ax.set_xticklabels([f"Pass {p}" for p in range(n_passes)])
    ax.set_xlabel("Pass")
    ax.set_title("Max SAE Feature-Property Correlation by Pass")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    fig_path = output_path.with_suffix(".png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on shared-block activations")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--n-smiles", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/sae_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed both numpy (for SMILES sampling) and torch (for SAE init/training)
    torch.manual_seed(args.seed)

    # Load model and data
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    print(
        f"Model: n_embd={config.n_embd}, n_layer={config.n_layer}, "
        f"weight_sharing={config.weight_sharing}"
    )

    smiles_list = load_smiles(Path("data/zinc250k.csv"), args.n_smiles, args.seed)

    # Part 1: Collect activations
    print(f"\nCollecting activations from {len(smiles_list)} SMILES...")
    activations, valid_smiles, tokens_per_smiles = collect_activations(
        model, tokenizer, smiles_list, args.device
    )
    n_tokens = activations[0].shape[0]
    n_passes = config.n_layer
    print(f"  {len(valid_smiles)} valid SMILES, {n_tokens} total tokens")

    # Compute chemical properties
    properties = compute_token_properties(tokens_per_smiles)

    # Part 2 & 3: Train SAE per pass and analyze
    saes: dict[int, SparseAutoencoder] = {}
    top_features: dict[int, list[dict]] = {}

    for p in range(n_passes):
        print(f"\n{'='*40}")
        print(f"Training SAE for pass {p}...")
        sae, logs = train_sae(
            activations[p],
            n_features=args.n_features,
            lr=args.lr,
            l1_coeff=args.l1_coeff,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
        )
        saes[p] = sae

        print(f"  Analyzing features for pass {p}...")
        top_features[p] = analyze_features(
            sae, activations[p], properties, tokens_per_smiles, top_k=5
        )

    # Part 4: Cross-pass comparison
    print("\nComputing cross-pass analysis...")
    cross_pass = cross_pass_analysis(saes, activations, properties)

    # Report
    print_results(top_features, cross_pass, n_passes)

    # Save
    output_path = Path(args.output)
    meta = {
        "checkpoint": args.checkpoint,
        "n_smiles": len(valid_smiles),
        "n_tokens": n_tokens,
        "n_passes": n_passes,
        "n_embd": config.n_embd,
        "n_features": args.n_features,
        "n_steps": args.n_steps,
        "l1_coeff": args.l1_coeff,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        **build_eval_meta(Path(args.checkpoint)),
    }
    save_results(top_features, cross_pass, output_path, meta)
    save_figure(cross_pass, n_passes, output_path)


if __name__ == "__main__":
    main()
