"""Attention pattern analysis: does Head 1's function shift across passes?

Hypothesis: the same attention weights compute different things depending
on the input representation. Head 1 is a "structural anchor" whose function
shifts across passes as the residual stream evolves.

Methodologist fixes (v2):
  - Multiple seeds with bootstrap CIs
  - Distance-stratified bracket/ring matching
  - Per-token-type normalization (controls for atom count bias)
  - Null baseline support (--null-baseline flag)
  - Median alongside mean

Usage:
    # Single model, 3 seeds:
    python analyze/attention.py \
        --checkpoint checkpoints/sub100k/ws-e128-x8.pt \
        --n-smiles 500 --seeds 42,123,456

    # With null baseline (random init):
    python analyze/attention.py \
        --checkpoint checkpoints/sub100k/ws-e128-x8.pt \
        --n-smiles 500 --seeds 42,123,456 --null-baseline
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
import torch.nn.functional as F

from typing import cast

from smollm.checkpoint import load_checkpoint
from smollm.model import GPT, _apply_rope
from smollm.tokenizer import SmilesTokenizer

SMILES_PATTERN = r"Cl|Br|@@|\[.*?\]|[CNOSPFIcnosp()=\#\-\+\.\/\\@0-9]"
ATOMS = set("CNOSPFIcnosp")

DIST_BINS = {"short": (1, 5), "medium": (6, 15), "long": (16, 999)}


def classify_token(tok: str) -> str:
    if tok in ("(", ")"):
        return "bracket"
    if tok in "0123456789":
        return "ring_digit"
    if tok in ATOMS or tok in ("Cl", "Br") or tok.startswith("["):
        return "atom"
    if tok in ("=", "#", "-", "/", "\\"):
        return "bond"
    return "other"


def dist_bin(distance: int) -> str:
    for name, (lo, hi) in DIST_BINS.items():
        if lo <= distance <= hi:
            return name
    return "long"


def load_model(checkpoint: str, device: str = "cpu"):
    loaded, tokenizer = load_checkpoint(checkpoint, device=device)
    model = cast(GPT, loaded)
    model.eval()
    return model, tokenizer, model.config


def load_model_random_init(checkpoint: str, device: str = "cpu"):
    """Load config and tokenizer from checkpoint, but keep random weights."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.eval()
    tokenizer = SmilesTokenizer.from_state_dict(ckpt["tokenizer"])
    return model, tokenizer, config


def load_smiles(data_path: Path, n: int, seed: int = 42) -> list[str]:
    if not data_path.exists():
        from smollm.data import download_zinc
        download_zinc(data_path)
    with open(data_path) as f:
        all_smiles = [row["smiles"] for row in csv.DictReader(f)]
    rng = np.random.RandomState(seed)
    return list(rng.choice(all_smiles, size=min(n, len(all_smiles)), replace=False))


def collect_attention_patterns(
    model: GPT,
    tokenizer: SmilesTokenizer,
    smiles_list: list[str],
    device: str = "cpu",
) -> dict:
    """Run SMILES through the model, capture attention weights at each pass."""
    n_passes = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    # bracket_match[pass][head] = list of (attn_value, distance)
    bracket_match_attn = {
        p: {h: [] for h in range(n_heads)} for p in range(n_passes)
    }
    # ring_match[pass][head] = list of (attn_value, distance)
    ring_match_attn = {
        p: {h: [] for h in range(n_heads)} for p in range(n_passes)
    }
    # Per-token-type normalized attention: attn_to_type[pass][head][target_type] = list of per-sequence means
    attn_to_type_norm = {
        p: {h: defaultdict(list) for h in range(n_heads)} for p in range(n_passes)
    }

    n_processed = 0

    with torch.no_grad():
        for smi in smiles_list:
            toks = re.findall(SMILES_PATTERN, smi)
            if not toks or len(toks) >= 100:
                continue
            try:
                ids = tokenizer.encode(smi)
            except (KeyError, IndexError):
                continue

            tok_types = [classify_token(t) for t in toks]

            # Count tokens per type for normalization
            type_counts = defaultdict(int)
            for tt in tok_types:
                type_counts[tt] += 1

            # Bracket pairs (stack-based)
            bracket_pairs = {}
            stack = []
            for i, t in enumerate(toks):
                if t == "(":
                    stack.append(i)
                elif t == ")" and stack:
                    bracket_pairs[i] = stack.pop()

            # Ring digit pairs
            ring_pairs = {}
            ring_first = {}
            for i, t in enumerate(toks):
                if t in "0123456789":
                    if t in ring_first:
                        ring_pairs[i] = ring_first[t]
                        del ring_first[t]
                    else:
                        ring_first[t] = i

            # Forward pass
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
                block = model.shared_block if model.config.weight_sharing else model.blocks[p]
                x_norm = block.ln_1(x)
                B_sz, T_sz, C = x_norm.size()
                qkv = block.attn.c_attn(x_norm)
                q, k, v = qkv.split(model.config.n_embd, dim=2)
                q = q.view(B_sz, T_sz, n_heads, head_dim).transpose(1, 2)
                k = k.view(B_sz, T_sz, n_heads, head_dim).transpose(1, 2)
                v = v.view(B_sz, T_sz, n_heads, head_dim).transpose(1, 2)

                if model.config.rope and rope_cos is not None:
                    q = _apply_rope(q, rope_cos[:, :, :T_sz, :], rope_sin[:, :, :T_sz, :])
                    k = _apply_rope(k, rope_cos[:, :, :T_sz, :], rope_sin[:, :, :T_sz, :])

                # Compute attention weights for analysis
                scale = head_dim ** -0.5
                attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale
                causal_mask = torch.triu(torch.ones(T_sz, T_sz, device=device), diagonal=1).bool()
                attn_w.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_w = F.softmax(attn_w, dim=-1)

                # Actual forward pass for x update
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                y = y.transpose(1, 2).contiguous().view(B_sz, T_sz, C)
                attn_out = block.attn.resid_dropout(block.attn.c_proj(y))
                x_after_attn = x + attn_out
                x = x_after_attn + block.mlp(block.ln_2(x_after_attn))

                attn_np = attn_w[0].cpu().numpy()  # (n_heads, T, T)

                for h in range(n_heads):
                    # Bracket matching with distance
                    for close_idx, open_idx in bracket_pairs.items():
                        dist = close_idx - open_idx
                        bracket_match_attn[p][h].append(
                            (float(attn_np[h, close_idx + 1, open_idx + 1]), dist)
                        )

                    # Ring matching with distance
                    for second_idx, first_idx in ring_pairs.items():
                        dist = second_idx - first_idx
                        ring_match_attn[p][h].append(
                            (float(attn_np[h, second_idx + 1, first_idx + 1]), dist)
                        )

                    # Per-token-type normalized attention for this sequence
                    # For each target type: sum of attention to that type / count of tokens of that type
                    for tgt_type in ["bracket", "ring_digit", "atom", "bond"]:
                        if type_counts[tgt_type] == 0:
                            continue
                        # Average over all query positions: mean attention per target token of this type
                        total_attn = 0.0
                        n_queries = 0
                        for qi in range(n_tok):
                            seq_qi = qi + 1
                            attn_sum = 0.0
                            for ki in range(n_tok):
                                if ki >= qi:
                                    break
                                if tok_types[ki] == tgt_type:
                                    attn_sum += attn_np[h, seq_qi, ki + 1]
                            total_attn += attn_sum / type_counts[tgt_type]
                            n_queries += 1
                        if n_queries > 0:
                            attn_to_type_norm[p][h][tgt_type].append(total_attn / n_queries)

            n_processed += 1
            if n_processed % 100 == 0:
                print(f"  Processed {n_processed} SMILES...")

    return {
        "bracket_match_attn": bracket_match_attn,
        "ring_match_attn": ring_match_attn,
        "attn_to_type_norm": attn_to_type_norm,
        "n_processed": n_processed,
        "n_passes": n_passes,
        "n_heads": n_heads,
    }


def summarize_match_metric(data: dict, n_passes: int, n_heads: int) -> dict:
    """Summarize bracket/ring match data with distance bins, mean, median, CI."""
    summary = {}
    for p in range(n_passes):
        for h in range(n_heads):
            pairs = data[p][h]
            if not pairs:
                summary[f"pass{p}_head{h}"] = {"all": _empty_stats(), **{b: _empty_stats() for b in DIST_BINS}}
                continue

            vals = np.array([v for v, d in pairs])
            dists = np.array([d for v, d in pairs])

            entry = {"all": _compute_stats(vals)}
            for bin_name, (lo, hi) in DIST_BINS.items():
                mask = (dists >= lo) & (dists <= hi)
                entry[bin_name] = _compute_stats(vals[mask]) if mask.any() else _empty_stats()

            summary[f"pass{p}_head{h}"] = entry
    return summary


def _compute_stats(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return _empty_stats()
    # Bootstrap 95% CI for the mean
    n_boot = 1000
    rng = np.random.RandomState(0)
    boot_means = np.array([
        np.mean(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(len(vals)),
        "frac_gt_0.1": float(np.mean(vals > 0.1)),
    }


def _empty_stats() -> dict:
    return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0, "frac_gt_0.1": 0.0}


def summarize_results(results: dict) -> dict:
    n_passes = results["n_passes"]
    n_heads = results["n_heads"]

    summary = {
        "n_processed": results["n_processed"],
        "n_passes": n_passes,
        "n_heads": n_heads,
    }

    summary["bracket_match"] = summarize_match_metric(
        results["bracket_match_attn"], n_passes, n_heads
    )
    summary["ring_match"] = summarize_match_metric(
        results["ring_match_attn"], n_passes, n_heads
    )

    # Normalized attention to type
    attn_norm = {}
    for p in range(n_passes):
        for h in range(n_heads):
            for tgt in ["bracket", "ring_digit", "atom", "bond"]:
                vals = results["attn_to_type_norm"][p][h][tgt]
                key = f"pass{p}_head{h}_to_{tgt}"
                attn_norm[key] = _compute_stats(np.array(vals)) if vals else _empty_stats()
    summary["attn_to_type_norm"] = attn_norm

    return summary


def print_report(summary: dict, label: str = "") -> None:
    n_passes = summary["n_passes"]
    n_heads = summary["n_heads"]

    if label:
        print(f"\n{'#' * 80}")
        print(f"# {label}")
        print(f"{'#' * 80}")

    print(f"\nProcessed {summary['n_processed']} SMILES")

    # === Bracket matching ===
    print("\n" + "=" * 90)
    print("BRACKET MATCHING: ')' -> matching '('  [mean (median) CI95]")
    print("=" * 90)
    header = f"  {'':>8}"
    for h in range(n_heads):
        header += f"  {'Head ' + str(h):>18}"
    print(header)
    print("-" * 90)
    for p in range(n_passes):
        row = f"  Pass {p:>2}"
        for h in range(n_heads):
            s = summary["bracket_match"][f"pass{p}_head{h}"]["all"]
            row += f"  {s['mean']:.3f} ({s['median']:.3f})"
        print(row)

    # Distance-stratified for Head 0 and Head 1
    for h in [0, 1]:
        print(f"\n  Head {h} by distance:")
        print(f"  {'':>8}  {'short':>12}  {'medium':>12}  {'long':>12}")
        for p in range(n_passes):
            row = f"  Pass {p:>2}"
            for b in ["short", "medium", "long"]:
                s = summary["bracket_match"][f"pass{p}_head{h}"][b]
                row += f"  {s['mean']:>5.3f} n={s['n']:<4d}" if s["n"] > 0 else f"  {'--':>12}"
            print(row)

    # === Ring matching ===
    print("\n" + "=" * 90)
    print("RING MATCHING: 2nd ring digit -> 1st  [mean (median)]")
    print("=" * 90)
    header = f"  {'':>8}"
    for h in range(n_heads):
        header += f"  {'Head ' + str(h):>18}"
    print(header)
    print("-" * 90)
    for p in range(n_passes):
        row = f"  Pass {p:>2}"
        for h in range(n_heads):
            s = summary["ring_match"][f"pass{p}_head{h}"]["all"]
            row += f"  {s['mean']:.3f} ({s['median']:.3f})"
        print(row)

    for h in [0, 1]:
        print(f"\n  Head {h} by distance:")
        print(f"  {'':>8}  {'short':>12}  {'medium':>12}  {'long':>12}")
        for p in range(n_passes):
            row = f"  Pass {p:>2}"
            for b in ["short", "medium", "long"]:
                s = summary["ring_match"][f"pass{p}_head{h}"][b]
                row += f"  {s['mean']:>5.3f} n={s['n']:<4d}" if s["n"] > 0 else f"  {'--':>12}"
            print(row)

    # === Normalized attention to type (Head 1 only) ===
    print("\n" + "=" * 90)
    print("HEAD 1: normalized attention per token of each type")
    print("(Controls for atom count bias -- attention per target token, not total)")
    print("=" * 90)
    target_types = ["bracket", "ring_digit", "atom", "bond"]
    header2 = f"  {'':>8}"
    for tgt in target_types:
        header2 += f"  {tgt:>12}"
    print(header2)
    print("-" * 90)
    for p in range(n_passes):
        row = f"  Pass {p:>2}"
        for tgt in target_types:
            s = summary["attn_to_type_norm"].get(f"pass{p}_head1_to_{tgt}", _empty_stats())
            row += f"  {s['mean']:>12.4f}"
        print(row)


def merge_summaries(summaries: list[dict]) -> dict:
    """Merge summaries across seeds by pooling raw statistics."""
    if len(summaries) == 1:
        return summaries[0]

    merged = {
        "n_processed": sum(s["n_processed"] for s in summaries),
        "n_passes": summaries[0]["n_passes"],
        "n_heads": summaries[0]["n_heads"],
        "n_seeds": len(summaries),
    }

    # For bracket/ring match: collect per-seed means and compute cross-seed stats
    for metric in ["bracket_match", "ring_match"]:
        merged[metric] = {}
        all_keys = summaries[0][metric].keys()
        for key in all_keys:
            merged[metric][key] = {}
            for dist_key in summaries[0][metric][key].keys():
                seed_means = [
                    s[metric][key][dist_key]["mean"]
                    for s in summaries
                    if s[metric][key][dist_key]["n"] > 0
                ]
                seed_medians = [
                    s[metric][key][dist_key]["median"]
                    for s in summaries
                    if s[metric][key][dist_key]["n"] > 0
                ]
                total_n = sum(s[metric][key][dist_key]["n"] for s in summaries)
                if seed_means:
                    merged[metric][key][dist_key] = {
                        "mean": float(np.mean(seed_means)),
                        "median": float(np.mean(seed_medians)),
                        "std": float(np.std(seed_means)) if len(seed_means) > 1 else 0.0,
                        "sem": float(np.std(seed_means) / np.sqrt(len(seed_means))) if len(seed_means) > 1 else 0.0,
                        "ci_lo": float(np.mean(seed_means) - 1.96 * np.std(seed_means) / np.sqrt(len(seed_means))) if len(seed_means) > 1 else float(np.mean(seed_means)),
                        "ci_hi": float(np.mean(seed_means) + 1.96 * np.std(seed_means) / np.sqrt(len(seed_means))) if len(seed_means) > 1 else float(np.mean(seed_means)),
                        "n": total_n,
                        "n_seeds": len(seed_means),
                        "frac_gt_0.1": float(np.mean([s[metric][key][dist_key]["frac_gt_0.1"] for s in summaries if s[metric][key][dist_key]["n"] > 0])),
                    }
                else:
                    merged[metric][key][dist_key] = _empty_stats()

    # attn_to_type_norm: average across seeds
    merged["attn_to_type_norm"] = {}
    for key in summaries[0]["attn_to_type_norm"]:
        seed_means = [s["attn_to_type_norm"][key]["mean"] for s in summaries if s["attn_to_type_norm"][key]["n"] > 0]
        if seed_means:
            merged["attn_to_type_norm"][key] = {
                "mean": float(np.mean(seed_means)),
                "std": float(np.std(seed_means)) if len(seed_means) > 1 else 0.0,
                "n": sum(s["attn_to_type_norm"][key]["n"] for s in summaries),
            }
        else:
            merged["attn_to_type_norm"][key] = _empty_stats()

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention pattern analysis across passes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-smiles", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/attention_patterns_v2.json")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--null-baseline", action="store_true", help="Also run with random init weights")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    print(f"Model: n_embd={config.n_embd}, n_head={config.n_head}, "
          f"n_layer={config.n_layer}, weight_sharing={config.weight_sharing}")

    # Run across seeds
    all_summaries = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        smiles_list = load_smiles(Path("data/zinc250k.csv"), args.n_smiles, seed)
        print(f"Loaded {len(smiles_list)} SMILES")
        results = collect_attention_patterns(model, tokenizer, smiles_list, args.device)
        summary = summarize_results(results)
        all_summaries.append(summary)

    # Merge across seeds
    merged = merge_summaries(all_summaries)
    print_report(merged, label=f"TRAINED MODEL ({len(seeds)} seeds)")

    output = {"trained": merged}

    # Null baseline
    if args.null_baseline:
        print(f"\n\n{'=' * 80}")
        print("RUNNING NULL BASELINE (random init weights)")
        print(f"{'=' * 80}")
        null_model, _, _ = load_model_random_init(args.checkpoint, args.device)
        torch.manual_seed(0)
        smiles_list = load_smiles(Path("data/zinc250k.csv"), args.n_smiles, 42)
        results = collect_attention_patterns(null_model, tokenizer, smiles_list, args.device)
        null_summary = summarize_results(results)
        print_report(null_summary, label="NULL BASELINE (random init)")
        output["null_baseline"] = null_summary

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
