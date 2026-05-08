import random
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from smollm.tokenizer import SmilesTokenizer

RDLogger.DisableLog("rdApp.*")

ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)


def download_zinc(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ZINC-250K to {path}...")
        urllib.request.urlretrieve(ZINC_URL, path)
    return path


def canonicalize(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    raw = df[smiles_col].astype(str).tolist()
    canon = [canonicalize(s) for s in tqdm(raw, desc="Canonicalizing")]
    return [s for s in canon if s is not None]


def train_val_split(
    smiles: list[str], seed: int, val_frac: float = 0.1
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(smiles)
    rng.shuffle(shuffled)
    split = int((1 - val_frac) * len(shuffled))
    return shuffled[:split], shuffled[split:]


class SmilesDataset(Dataset):
    def __init__(
        self, smiles_list: list[str], tokenizer: SmilesTokenizer, block_size: int = 128
    ) -> None:
        encoded = []
        for smi in smiles_list:
            ids = tokenizer.encode(smi)
            if len(ids) <= block_size:
                ids = ids + [tokenizer.pad_id] * (block_size - len(ids))
                encoded.append(ids)
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx, :-1], self.data[idx, 1:]


def _randomize_smiles(mol: Chem.Mol, rng: random.Random) -> str:
    """Seeded SMILES randomization via atom renumbering.

    `MolToSmiles(doRandom=True)` uses RDKit's internal RNG which is not
    Python-seedable. We instead permute atom indices with a Python RNG and
    canonicalize from the new ordering — equivalent randomization, but
    deterministic given (mol, seed).
    """
    atom_indices = list(range(mol.GetNumAtoms()))
    rng.shuffle(atom_indices)
    renumbered = Chem.RenumberAtoms(mol, atom_indices)
    return Chem.MolToSmiles(renumbered, canonical=False)


def prepare_augmented_smiles(
    smiles_list: list[str],
    tokenizer: SmilesTokenizer,
    block_size: int = 128,
    n_aug: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    rng = random.Random(seed)
    encoded = []
    for smi in tqdm(smiles_list, desc=f"Augmenting (x{n_aug})"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for _ in range(n_aug):
            rand_smi = _randomize_smiles(mol, rng)
            try:
                ids = tokenizer.encode(rand_smi)
            except KeyError:
                ids = tokenizer.encode(smi)
            if len(ids) > block_size:
                ids = ids[:block_size]
            else:
                ids = ids + [tokenizer.pad_id] * (block_size - len(ids))
            encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


class AugmentedSmilesDataset(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx, :-1], self.data[idx, 1:]
