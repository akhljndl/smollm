# SMolLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/akhljndl/smollm)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-5a5fd1)](https://akhljndl.github.io/smollm/)

**A 53K-parameter weight-shared transformer for SMILES generation, trained on ZINC-250K.**

By [Akhil Jindal](https://github.com/akhljndl) and [Harang Ju](https://github.com/harangju).

## Install

```bash
git clone https://github.com/akhljndl/smollm
cd smollm
uv sync
uv sync --extra fcd --extra analyze --extra wandb  # optional: FCD metric, probes, wandb logging
```

ZINC-250K (~22 MB) downloads on first training run.

> macOS: `conda install -c conda-forge rdkit` before `uv sync` (pip wheels for rdkit are unreliable on macOS).

## Pipeline

```bash
uv run python train.py --config ws-53k --seed 42                        # ~8h on a single MI210 / 3090
uv run python eval.py  --checkpoint checkpoints/ws-53k-s42.pt --n 10000 # validity, uniqueness, novelty, IntDiv, FCD

# Additional experiments
uv run python distill.py --teacher checkpoints/ws-206k-s42.pt --config ws-53k  # teacher→student distillation
uv run python dpo.py     --checkpoint checkpoints/ws-53k-s42.pt                # negative result — DPO hurt validity
```

`--config <name>` accepts any architecture in `smollm/configs.py:ARCHITECTURES` (10 configs: vanilla GPT scaling curve, weight-shared headlines, and GRU baselines).

See `analyze/` for mechanistic analysis scripts.

## Demo

The interactive trace demo is hosted at [akhljndl.github.io/smollm](https://akhljndl.github.io/smollm/). It is a static GitHub Pages artifact built from `demo/`, with curated trace JSON and vendored browser RDKit assets for deterministic molecule rendering.

## Checkpoints

```bash
huggingface-cli download akhljndl/smollm --include "checkpoints/0504_paper/*" --local-dir .
uv run python eval.py --checkpoint checkpoints/0504_paper/ws-53k-s42.pt --n 10000
```

See `MANIFEST.md` on the HF repo for the full per-config / per-seed table.

## Citation

```bibtex
@misc{jindal2026smollm,
      title={SMolLM: Small Language Models Learn Small Molecular Grammar}, 
      author={Akhil Jindal and Harang Ju},
      year={2026},
      eprint={2605.06322},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06322}, 
}
```
