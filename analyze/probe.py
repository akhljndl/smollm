"""Probe what chemical properties the shared block encodes at each pass.

Trains linear classifiers on hidden states after each pass to predict:
- Bracket depth (how many open brackets at this position)
- Ring state (number of currently unpaired ring digits)
- Token type (atom, bond, bracket, digit, special)

If accuracy rises across passes, the shared weights progressively encode
chemical grammar — direct evidence for iterative refinement.

Usage:
    python analyze/probe.py \
        --checkpoint checkpoints/ws-206k.pt

    python analyze/probe.py \
        --checkpoint checkpoints/ws-53k.pt \
        --n-smiles 200
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from typing import cast

from smollm._meta import build_eval_meta
from smollm.checkpoint import load_checkpoint
from smollm.model import GPT
from smollm.tokenizer import SmilesTokenizer


# ── SMILES structural label extraction ──────────────────────────────────

SMILES_PATTERN = r"Cl|Br|@@|\[.*?\]|[CNOSPFIcnosp()=\#\-\+\.\/\\@0-9]"

ATOMS = set("CNOSPFIcnosp")
BONDS = set("=-#/\\.")
BRACKETS = set("()")
DIGITS = set("0123456789")


def token_type(tok: str) -> int:
    """Classify a SMILES token: 0=atom, 1=bond, 2=open_bracket, 3=close_bracket, 4=ring_digit, 5=other."""
    if tok in ATOMS or tok in ("Cl", "Br") or tok.startswith("["):
        return 0
    if tok in BONDS:
        return 1
    if tok == "(":
        return 2
    if tok == ")":
        return 3
    if tok in DIGITS:
        return 4
    return 5


def compute_labels(smiles: str) -> dict[str, list[int]]:
    """Compute per-token structural labels for a SMILES string.

    Returns dict with lists aligned to the tokenized SMILES (excluding BOS/EOS).
    """
    tokens = re.findall(SMILES_PATTERN, smiles)
    n = len(tokens)

    bracket_depth = []
    ring_state = []
    tok_types = []
    next_is_close = []

    depth = 0
    open_rings: set[str] = set()

    for i, tok in enumerate(tokens):
        # Record state BEFORE this token modifies it
        bracket_depth.append(depth)
        ring_state.append(len(open_rings))
        tok_types.append(token_type(tok))

        # Next-is-close: look ahead
        if i + 1 < n:
            next_is_close.append(1 if tokens[i + 1] == ")" else 0)
        else:
            next_is_close.append(0)

        # Update state
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif tok in DIGITS:
            if tok in open_rings:
                open_rings.remove(tok)
            else:
                open_rings.add(tok)

    return {
        "bracket_depth": bracket_depth,
        "ring_state": ring_state,
        "token_type": tok_types,
        "next_is_close": next_is_close,
    }


# ── Hidden state extraction ─────────────────────────────────────────────


def extract_hidden_states(
    model: GPT, tokenizer: SmilesTokenizer, smiles_list: list[str], device: str = "cpu"
) -> dict[int, list[torch.Tensor]]:
    """Extract hidden states after each pass through the shared block.

    Returns: {pass_idx: [tensor(T_i, D) for each SMILES]} where T_i is the
    sequence length (excluding BOS but including all SMILES tokens).
    """
    model.eval()
    n_passes = model.config.n_layer
    states_per_pass: dict[int, list[torch.Tensor]] = {p: [] for p in range(n_passes)}

    with torch.no_grad():
        for smi in smiles_list:
            ids = tokenizer.encode(smi)
            idx = torch.tensor([ids], device=device)
            B, T = idx.size()

            x = model.tok_emb(idx)
            if model.config.rope:
                x = model.drop(x)
                rope_cos = model.rope_cos[:, :, :T, :]
                rope_sin = model.rope_sin[:, :, :T, :]
            else:
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                x = model.drop(x + model.pos_emb(pos))
                rope_cos = rope_sin = None

            for p in range(n_passes):
                block = model.shared_block if model.config.weight_sharing else model.blocks[p]
                x = block(x, rope_cos, rope_sin)
                # Store hidden state for SMILES tokens only (skip BOS at position 0, skip EOS at end)
                smiles_tokens = re.findall(SMILES_PATTERN, smi)
                n_tok = len(smiles_tokens)
                # Positions 1..n_tok correspond to SMILES tokens (position 0 is BOS)
                states_per_pass[p].append(x[0, 1 : 1 + n_tok].cpu())

    return states_per_pass


# ── Probing ─────────────────────────────────────────────────────────────


def train_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Train a linear probe and return test accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Cap classes at what's in train
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train_s, y_train)
    return float(clf.score(X_test_s, y_test))


def run_probing(
    states_per_pass: dict[int, list[torch.Tensor]],
    labels_per_smiles: list[dict[str, list[int]]],
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Train linear probes for each property at each pass.

    Returns: {property_name: {pass_idx: accuracy}}
    """
    rng = np.random.RandomState(seed)

    # Flatten all tokens across SMILES
    all_labels: dict[str, list[int]] = {k: [] for k in labels_per_smiles[0].keys()}
    smiles_boundaries: list[int] = []  # cumulative token counts
    total = 0
    for labels in labels_per_smiles:
        n = len(labels["bracket_depth"])
        for k in all_labels:
            all_labels[k].extend(labels[k])
        total += n
        smiles_boundaries.append(total)

    # Train/test split by SMILES (not by token)
    n_smiles = len(labels_per_smiles)
    perm = rng.permutation(n_smiles)
    n_train = int(train_frac * n_smiles)
    train_idx = set(perm[:n_train].tolist())

    train_mask = np.zeros(total, dtype=bool)
    prev = 0
    for i, boundary in enumerate(smiles_boundaries):
        if i in train_idx:
            train_mask[prev:boundary] = True
        prev = boundary

    results: dict[str, dict[int, float]] = {}
    properties = ["bracket_depth", "ring_state", "token_type", "next_is_close"]

    for prop in properties:
        y = np.array(all_labels[prop])
        y_train = y[train_mask]
        y_test = y[~train_mask]

        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            continue

        results[prop] = {}
        for p, hidden_list in states_per_pass.items():
            X = torch.cat(hidden_list, dim=0).numpy()
            X_train = X[train_mask]
            X_test = X[~train_mask]

            acc = train_probe(X_train, y_train, X_test, y_test)
            results[prop][p] = acc

    return results


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe chemical properties across passes")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--n-smiles", type=int, default=500, help="Number of SMILES to use")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/probing_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    loaded, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    model = cast(GPT, loaded)
    config = model.config
    model.eval()

    print(f"Model: n_embd={config.n_embd}, n_layer={config.n_layer}, weight_sharing={config.weight_sharing}")

    # Load SMILES from ZINC-250K
    data_path = Path("data/zinc250k.csv")
    if not data_path.exists():
        from smollm.data import download_zinc
        download_zinc(data_path)

    import csv
    with open(data_path) as f:
        reader = csv.DictReader(f)
        all_smiles = [row["smiles"] for row in reader]

    rng = np.random.RandomState(args.seed)
    selected = rng.choice(all_smiles, size=min(args.n_smiles, len(all_smiles)), replace=False)

    # Filter to valid SMILES that tokenize correctly
    valid_smiles = []
    for smi in selected:
        tokens = re.findall(SMILES_PATTERN, smi)
        if len(tokens) > 0 and len(tokens) < 120:
            try:
                tokenizer.encode(smi)
                valid_smiles.append(smi)
            except (KeyError, IndexError):
                continue
    print(f"Using {len(valid_smiles)} SMILES")

    # Compute labels
    print("Computing structural labels...")
    labels_per_smiles = [compute_labels(smi) for smi in valid_smiles]
    total_tokens = sum(len(lab["bracket_depth"]) for lab in labels_per_smiles)
    print(f"  Total tokens: {total_tokens}")

    # Show label distributions
    all_depths = [d for lab in labels_per_smiles for d in lab["bracket_depth"]]
    all_rings = [r for lab in labels_per_smiles for r in lab["ring_state"]]
    print(f"  Bracket depth range: {min(all_depths)}-{max(all_depths)}")
    print(f"  Ring state range: {min(all_rings)}-{max(all_rings)}")

    # Extract hidden states
    print(f"Extracting hidden states across {config.n_layer} passes...")
    states = extract_hidden_states(model, tokenizer, valid_smiles, device=args.device)

    # Train probes
    print("Training linear probes...")
    results = run_probing(states, labels_per_smiles, seed=args.seed)

    # Print results
    n_passes = config.n_layer
    print()
    print("=" * 70)
    header = f"  {'Property':<20}"
    for p in range(n_passes):
        header += f"  Pass {p+1:>2}"
    header += "   Trend"
    print(header)
    print("-" * 70)

    for prop, accs in results.items():
        row = f"  {prop:<20}"
        values = []
        for p in range(n_passes):
            acc = accs.get(p, 0.0)
            row += f"  {acc:>6.1%}"
            values.append(acc)
        # Trend
        if len(values) >= 2:
            delta = values[-1] - values[0]
            if delta > 0.02:
                trend = f"  +{delta:.1%} RISING"
            elif delta < -0.02:
                trend = f"  {delta:.1%} FALLING"
            else:
                trend = f"  {delta:+.1%} FLAT"
        else:
            trend = ""
        row += trend
        print(row)
    print("=" * 70)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        prop: {str(k): v for k, v in accs.items()}
        for prop, accs in results.items()
    }
    serializable["_meta"] = {
        "checkpoint": args.checkpoint,
        "n_smiles": len(valid_smiles),
        "n_tokens": total_tokens,
        "n_passes": n_passes,
        "n_embd": config.n_embd,
        **build_eval_meta(Path(args.checkpoint)),
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
