from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
from tqdm.auto import tqdm

from smollm.tokenizer import SmilesTokenizer

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: SmilesTokenizer,
    n: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    max_len: int = 120,
    device: str = "cuda",
) -> list[str]:
    model.eval()
    molecules: list[str] = []

    for _ in tqdm(range(n), desc="Generating"):
        idx = torch.tensor([[tokenizer.bos_id]], device=device)

        for _ in range(max_len):
            idx_cond = idx[:, -model.config.block_size :]
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


def log_samples(
    model: nn.Module,
    tokenizer: SmilesTokenizer,
    epoch: int,
    n: int = 8,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda",
) -> dict:
    """Sample a few molecules and print them raw — Karpathy-style training monitor."""
    molecules = generate(model, tokenizer, n=n, temperature=temperature, top_k=top_k, device=device)
    n_valid = 0
    print(f"  --- samples (epoch {epoch}) ---")
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        tag = "ok" if mol is not None else "INVALID"
        if mol is not None:
            n_valid += 1
        print(f"    [{tag:>7s}] {smi}")
    validity = n_valid / len(molecules) if molecules else 0
    print(f"  --- {n_valid}/{len(molecules)} valid ({validity * 100:.0f}%) ---")
    return {"molecules": molecules, "validity": validity, "n_valid": n_valid}


def validate_smiles(molecules: list[str], train_set: set[str]) -> dict:
    valid_smiles: list[str] = []
    valid_mols: list = []

    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))

    unique = set(valid_smiles)
    novel = unique - train_set

    return {
        "total": len(molecules),
        "valid": len(valid_mols),
        "validity": len(valid_mols) / len(molecules) if molecules else 0,
        "uniqueness": len(unique) / len(valid_mols) if valid_mols else 0,
        "novelty": len(novel) / len(unique) if unique else 0,
        "valid_smiles": valid_smiles,
        "valid_mols": valid_mols,
        "novel_smiles": list(novel),
    }


def _internal_diversity(mols: list) -> float:
    if len(mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    sims = [
        DataStructs.TanimotoSimilarity(a, b) for a, b in combinations(fps, 2)
    ]
    return 1.0 - sum(sims) / len(sims)


def _property_distributions(mols: list) -> dict[str, list[float]]:
    mw, logp, qed_vals, tpsa = [], [], [], []
    for m in mols:
        mw.append(Descriptors.MolWt(m))
        logp.append(Descriptors.MolLogP(m))
        tpsa.append(Descriptors.TPSA(m))
        try:
            qed_vals.append(QED.qed(m))
        except Exception:
            pass  # skip molecules that fail kekulization in QED
    return {"mw": mw, "logp": logp, "qed": qed_vals, "tpsa": tpsa}


def evaluate_smiles(
    molecules: list[str],
    train_smiles: list[str] | set[str],
) -> dict:
    train_set = set(train_smiles) if not isinstance(train_smiles, set) else train_smiles
    results = validate_smiles(molecules, train_set)

    valid_mols = results["valid_mols"]
    valid_smiles = results["valid_smiles"]

    # Internal diversity over UNIQUE molecules only.
    # Computing IntDiv across duplicates inflates the metric for low-uniqueness
    # samples; canonical IntDiv is defined on the unique set.
    seen: set[str] = set()
    unique_mols: list = []
    for smi, mol in zip(valid_smiles, valid_mols):
        if smi in seen:
            continue
        seen.add(smi)
        unique_mols.append(mol)
    results["intdiv"] = _internal_diversity(unique_mols)

    # Property distributions
    results["properties"] = _property_distributions(valid_mols) if valid_mols else {
        "mw": [], "logp": [], "qed": [], "tpsa": [],
    }

    # FCD (optional dependency)
    # Force CPU — MIOpen Conv1d crashes on ROCm 6.2 (miopenStatusInternalError)
    # Set SKIP_FCD=1 to skip entirely (useful for smoke tests)
    import os
    if os.environ.get("SKIP_FCD") == "1":
        results["fcd"] = None
    else:
        try:
            import fcd as fcd_pkg

            valid_smiles = results["valid_smiles"]
            train_list = list(train_smiles) if isinstance(train_smiles, set) else train_smiles
            results["fcd"] = fcd_pkg.get_fcd(train_list, valid_smiles, device="cpu") if valid_smiles else float("nan")
        except ImportError:
            results["fcd"] = None

    return results


def log_metrics_to_wandb(
    metrics: dict,
    n_params: int | None = None,
    fcd_key: str = "fcd",
) -> None:
    """Log evaluate_smiles metrics to wandb run summary.

    `fcd_key` should be "fcd" only when the FCD reference is the full ZINC-250K
    set (the paper's headline reference). Train-time FCD computed against a
    90% split should use "fcd_train_split" to avoid being confused with headline
    numbers.
    """
    import wandb

    if n_params is not None:
        wandb.run.summary["params"] = n_params
    for key in ("validity", "uniqueness", "novelty", "intdiv"):
        if key in metrics:
            wandb.run.summary[key] = metrics[key]
    if metrics.get("fcd") is not None:
        wandb.run.summary[fcd_key] = metrics["fcd"]
    props = metrics.get("properties", {})
    for key in ("mw", "logp", "qed", "tpsa"):
        vals = props.get(key, [])
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
            wandb.run.summary[f"prop/{key}_mean"] = mean
            wandb.run.summary[f"prop/{key}_std"] = std
