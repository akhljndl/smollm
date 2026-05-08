"""Classify SMILES validity errors by type (brackets / rings / valence / parse)."""

from __future__ import annotations

import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def classify_error(smiles: str) -> str:
    """Classify why a SMILES string is invalid.

    Returns one of: 'valid', 'unmatched_brackets', 'unclosed_rings',
    'valence_error', 'parse_error'.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return "valid"

    depth = 0
    for ch in smiles:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return "unmatched_brackets"
    if depth != 0:
        return "unmatched_brackets"

    ring_digits: dict[str, int] = {}
    tokens = re.findall(r"Cl|Br|\[.*?\]|[CNOSPFIcnosp()=\#\-\+\.\/\\@0-9]", smiles)
    for tok in tokens:
        if tok in "0123456789":
            ring_digits[tok] = ring_digits.get(tok, 0) + 1
    unpaired = sum(1 for count in ring_digits.values() if count % 2 != 0)
    if unpaired > 0:
        return "unclosed_rings"

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return "parse_error"
        try:
            Chem.SanitizeMol(mol)
            return "valid"
        except Exception:
            return "valence_error"
    except Exception:
        return "parse_error"
