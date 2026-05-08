"""Tanimoto nearest-neighbor memorization check for generated molecules.

Measures how similar generated SMILES are to the training set using Morgan
fingerprint Tanimoto similarity. Reports nearest-neighbor statistics and
compares against a held-out baseline (training molecules vs. other training
molecules) to contextualize the results.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from smollm._meta import build_eval_meta
from smollm.data import download_zinc, load_smiles

RDLogger.DisableLog("rdApp.*")

ZINC_DEFAULT = Path("data/zinc250k.csv")


def smiles_to_fp(smi: str) -> DataStructs.ExplicitBitVect | None:
    """Convert SMILES to Morgan fingerprint (radius 2, 1024 bits)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def nearest_neighbor_tanimoto(
    query_fps: list[DataStructs.ExplicitBitVect],
    ref_fps: list[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    """For each query fingerprint, find max Tanimoto similarity to any ref."""
    nn_sims = np.empty(len(query_fps))
    for i, qfp in enumerate(tqdm(query_fps, desc="NN search")):
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
        nn_sims[i] = max(sims)
    return nn_sims


def summarize(nn_sims: np.ndarray) -> dict:
    """Compute summary statistics from nearest-neighbor similarities."""
    hist_counts, hist_edges = np.histogram(nn_sims, bins=20, range=(0, 1))
    return {
        "n": len(nn_sims),
        "mean": float(np.mean(nn_sims)),
        "median": float(np.median(nn_sims)),
        "std": float(np.std(nn_sims)),
        "min": float(np.min(nn_sims)),
        "max": float(np.max(nn_sims)),
        "frac_exact": float(np.mean(nn_sims == 1.0)),
        "frac_gt_0.9": float(np.mean(nn_sims > 0.9)),
        "frac_gt_0.8": float(np.mean(nn_sims > 0.8)),
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        },
    }


def load_generated_smiles(path: Path) -> list[str]:
    """Load SMILES from a text file (one per line)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tanimoto nearest-neighbor memorization check"
    )
    parser.add_argument(
        "--generated-file",
        type=Path,
        required=True,
        help="File with generated SMILES (one per line)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing zinc250k.csv (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memorization_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Subsample training set to 10K for fast testing",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=1000,
        help="Number of held-out molecules for baseline comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional source checkpoint (records provenance in _meta)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Load training data ---
    zinc_path = args.data_dir / "zinc250k.csv"
    download_zinc(zinc_path)
    train_smiles = load_smiles(zinc_path)
    print(f"Loaded {len(train_smiles)} training SMILES")

    # --- Load generated SMILES ---
    gen_smiles = load_generated_smiles(args.generated_file)
    print(f"Loaded {len(gen_smiles)} generated SMILES")

    # --- Compute fingerprints for generated molecules (valid only) ---
    gen_fps = []
    gen_valid = []
    for smi in gen_smiles:
        fp = smiles_to_fp(smi)
        if fp is not None:
            gen_fps.append(fp)
            gen_valid.append(smi)
    print(f"Valid generated: {len(gen_fps)}/{len(gen_smiles)}")

    if not gen_fps:
        print("No valid generated molecules. Exiting.")
        return

    # --- Split training set: hold out baseline subset ---
    indices = list(range(len(train_smiles)))
    random.shuffle(indices)
    baseline_size = min(args.baseline_size, len(train_smiles) // 10)
    baseline_idx = set(indices[:baseline_size])
    ref_smiles = [s for i, s in enumerate(train_smiles) if i not in baseline_idx]
    baseline_smiles = [train_smiles[i] for i in sorted(baseline_idx)]

    if args.quick:
        ref_smiles = random.sample(ref_smiles, min(10_000, len(ref_smiles)))
        print(f"Quick mode: subsampled reference set to {len(ref_smiles)}")

    # --- Compute reference fingerprints ---
    print("Computing reference fingerprints...")
    ref_fps = []
    for smi in tqdm(ref_smiles, desc="Ref FPs"):
        fp = smiles_to_fp(smi)
        if fp is not None:
            ref_fps.append(fp)
    print(f"Reference fingerprints: {len(ref_fps)}")

    # --- NN search: generated vs training ---
    print("\n=== Generated vs Training ===")
    gen_nn = nearest_neighbor_tanimoto(gen_fps, ref_fps)
    gen_stats = summarize(gen_nn)
    print(f"  Mean NN Tanimoto:  {gen_stats['mean']:.4f}")
    print(f"  Median:            {gen_stats['median']:.4f}")
    print(f"  Exact copies:      {gen_stats['frac_exact']:.4f}")
    print(f"  Tanimoto > 0.9:    {gen_stats['frac_gt_0.9']:.4f}")
    print(f"  Tanimoto > 0.8:    {gen_stats['frac_gt_0.8']:.4f}")

    # --- Baseline: held-out training vs rest of training ---
    print("\n=== Baseline (held-out training vs rest) ===")
    baseline_fps = [fp for smi in baseline_smiles if (fp := smiles_to_fp(smi)) is not None]
    baseline_nn = nearest_neighbor_tanimoto(baseline_fps, ref_fps)
    baseline_stats = summarize(baseline_nn)
    print(f"  Mean NN Tanimoto:  {baseline_stats['mean']:.4f}")
    print(f"  Median:            {baseline_stats['median']:.4f}")
    print(f"  Exact copies:      {baseline_stats['frac_exact']:.4f}")
    print(f"  Tanimoto > 0.9:    {baseline_stats['frac_gt_0.9']:.4f}")
    print(f"  Tanimoto > 0.8:    {baseline_stats['frac_gt_0.8']:.4f}")

    # --- Save results ---
    results = {
        "generated": gen_stats,
        "baseline": baseline_stats,
        "config": {
            "generated_file": str(args.generated_file),
            "n_generated": len(gen_smiles),
            "n_valid_generated": len(gen_fps),
            "n_reference": len(ref_fps),
            "n_baseline": len(baseline_fps),
            "quick": args.quick,
            "seed": args.seed,
        },
        "_meta": build_eval_meta(args.checkpoint) if args.checkpoint else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
