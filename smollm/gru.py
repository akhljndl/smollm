"""GRU model for SMILES generation.

Interface-compatible with GPT (transformer.py) so both can be used
interchangeably in train.py and sample.py.

Implementation note (2026-04-30): we unroll an explicit nn.GRUCell loop
instead of nn.GRU. On ROCm/MIOpen the fused recurrent kernel produces
no-op gradients on this cluster (jids 302053 fp16, 302100 fp32 — both
froze at the unigram-CE floor). The GRUCell path goes through stock
ATen ops and trains correctly. Param count is identical to the prior
nn.GRU(num_layers=N).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRUConfig:
    vocab_size: int = 50
    block_size: int = 128
    n_layers: int = 2
    hidden_size: int = 64
    dropout: float = 0.1
    pad_id: int = 0


class SmilesGRU(nn.Module):
    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.cells = nn.ModuleList(
            [nn.GRUCell(config.hidden_size, config.hidden_size)
             for _ in range(config.n_layers)]
        )
        self.inter_layer_drop = (
            nn.Dropout(config.dropout) if config.n_layers > 1 else nn.Identity()
        )
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.drop(self.tok_emb(idx))
        b, t, h = x.shape
        states = [torch.zeros(b, h, device=x.device, dtype=x.dtype)
                  for _ in range(len(self.cells))]
        outputs = []
        last = len(self.cells) - 1
        for step in range(t):
            inp = x[:, step]
            for li, cell in enumerate(self.cells):
                states[li] = cell(inp, states[li])
                inp = states[li] if li == last else self.inter_layer_drop(states[li])
            outputs.append(inp)
        x = torch.stack(outputs, dim=1)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_id,
            )
        return logits, loss
