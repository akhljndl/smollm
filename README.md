# smollm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/akhljndl/smollm)

**A 53,000-parameter weight-shared transformer for SMILES generation, trained on ZINC-250K.**

By [Akhil Jindal](https://github.com/akhljndl) and [Harang Ju](https://github.com/harangju).

> Interactive demo coming soon!

## Install

```bash
git clone https://github.com/akhljndl/smollm
cd smollm
uv sync                              # or: pip install -e .
uv sync --extra fcd --extra analyze  # optional: FCD metric + probes
```

ZINC-250K (~22 MB) downloads on first training run.

> macOS: `conda install -c conda-forge rdkit` before `uv sync` (pip wheels for rdkit are unreliable on macOS).

## Pipeline

```bash
python train.py   --config ws-e64-x8 --seed 42                              # ~8h on a single MI210 / 3090
python distill.py --teacher checkpoints/ws-e128-x8-s42.pt --config ws-e64-x8
python dpo.py     --checkpoint checkpoints/ws-e64-x8-s42.pt                 # negative result — DPO hurt validity
python eval.py    --checkpoint checkpoints/ws-e64-x8-s42.pt --n 10000       # validity, uniqueness, novelty, IntDiv, FCD
```

`--config <name>` accepts any architecture in `smollm/configs.py:ARCHITECTURES` (10 configs: vanilla GPT scaling curve, weight-shared headlines, and GRU baselines).

See `analyze/` for mechanistic analysis scripts and `python figures/make_figures.py` to regenerate figures.

## Demo

The public demo in `demo/` is a zero-build static site for GitHub Pages. It
ships curated JSON traces and renders them entirely in the browser; there is no
backend, live inference server, queue, or cloud runtime.

```bash
cd demo
python3 -m http.server 8712
open http://127.0.0.1:8712/
```

## Checkpoints

```bash
huggingface-cli download akhljndl/smollm --include "checkpoints/0504_paper/*" --local-dir .
python eval.py --checkpoint checkpoints/0504_paper/ws-53k-s42.pt --n 10000
```

See `MANIFEST.md` on the HF repo for the full per-config / per-seed table.

## Citation

```bibtex
@misc{jindal2026smollm,
  author = {Jindal, Akhil and Ju, Harang},
  title  = {SmolLM: Weight-Shared Transformers for SMILES Generation},
  year   = {2026},
  url    = {https://github.com/akhljndl/smollm},
  note   = {HuggingFace model: akhljndl/smollm}
}
```

## License

MIT — see `LICENSE`.
