from __future__ import annotations

import torch
import torch.nn as nn

from smollm.gru import GRUConfig, SmilesGRU
from smollm.model import GPT, TransformerConfig
from smollm.tokenizer import SmilesTokenizer


def make_smiles_gpt(
    vocab_size: int,
    pad_id: int,
    block_size: int = 128,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    dropout: float = 0.1,
    ffn_mult: float = 4.0,
    swiglu: bool = False,
    rope: bool = False,
    weight_sharing: bool = False,
) -> GPT:
    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        pad_id=pad_id,
        ffn_mult=ffn_mult,
        swiglu=swiglu,
        rope=rope,
        weight_sharing=weight_sharing,
    )
    return GPT(config)


def make_smiles_gru(
    vocab_size: int,
    pad_id: int,
    block_size: int = 128,
    n_layers: int = 2,
    hidden_size: int = 64,
    dropout: float = 0.1,
) -> SmilesGRU:
    config = GRUConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        pad_id=pad_id,
    )
    return SmilesGRU(config)


def save_checkpoint(
    model: nn.Module,
    tokenizer: SmilesTokenizer,
    path: str,
    extra: dict | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "config": model.config,
        "tokenizer": tokenizer.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str, device: str = "cuda"
) -> tuple[nn.Module, SmilesTokenizer]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, GRUConfig):
        model = SmilesGRU(config).to(device)
    else:
        model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = SmilesTokenizer.from_state_dict(checkpoint["tokenizer"])

    cfg_vocab = getattr(config, "vocab_size", None)
    if cfg_vocab is not None and cfg_vocab != tokenizer.vocab_size:
        raise ValueError(
            f"Vocab size mismatch in checkpoint {path}: "
            f"config.vocab_size={cfg_vocab} but tokenizer.vocab_size={tokenizer.vocab_size}"
        )
    return model, tokenizer
