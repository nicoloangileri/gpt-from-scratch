# gpt-from-scratch

A decoder-only Transformer (GPT) **built and trained from scratch in PyTorch**, then
made **fast** — with reproducible configs, benchmarks, and a serving path.

The point isn't to beat GPT-4. It's to show, end-to-end, that I understand the core
object of modern ML — the Transformer, the training loop, the tokenizer, sampling —
and can take it from training to production **with numbers**.

Every block (attention, MLP, LayerNorm, positional embeddings, the AdamW param
grouping) is hand-written in [`src/model.py`](src/model.py) — no `from transformers
import` of the model.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # model tests (CPU, seconds)

# prepare a small corpus (any UTF-8 .txt; start with TinyStories)
python -c "from src.data import prepare; prepare('data/tinystories.txt', 'data/tinystories')"

# train (edit configs/config.yaml or override on the CLI)
python -m src.train --config configs/config.yaml

# sample
python -m src.sample --ckpt checkpoints/best.pt --prompt "Once upon a time" --top_k 200
```

Runs are defined by `configs/config.yaml` + one command; seeds are fixed → reproducible.

## Hardware

This repo is developed on a Mac (MPS is fine only for tiny debug runs). Real training
runs on free cloud GPU (Kaggle ~30h/wk, or Colab) — a 10–30M-param model trains there
comfortably. The Level-3 CUDA/Triton kernel needs an NVIDIA GPU (a few hours on
Vast.ai / RunPod is enough).

## Roadmap (each level ships something)

### Level 1 — Base (this scaffold) ✅ *runnable*
- [x] Hand-written decoder-only Transformer (`src/model.py`)
- [x] BPE tokenizer (tiktoken) + memory-mapped token stream (`src/data.py`)
- [x] Config-driven training loop, cosine LR, grad clip, checkpointing (`src/train.py`)
- [x] Sampling with temperature/top-k (`src/sample.py`), tests (`tests/`)
- [ ] Train to coherent text on TinyStories; plot loss curves
- [ ] 2–3 ablations: learning rate, model size, context length

### Level 2 — Engineering (where it stops looking like homework)
- [ ] Mixed precision (AMP autocast + GradScaler) — *hooks wired in `train.py`*
- [ ] Gradient accumulation — *wired* — and `torch.compile`
- [ ] Throughput logging: **tokens/sec and MFU**
- [ ] Multi-GPU **DDP** run (even 2 GPUs, just to show it works)
- [ ] Inference with a **KV-cache** + a minimal **FastAPI** endpoint with batching
- [ ] A benchmark table (tokens/sec, memory, latency)

### Level 3 — Stretch (the differentiator)
- [ ] A custom **Triton or CUDA** kernel (fused attention or fused LayerNorm)
- [ ] Benchmarked against the PyTorch baseline, **speedup reported with numbers**

## What makes this read as "ML engineer," not a homework dump
Reproducible commands + one config; fixed seeds; a benchmark table (tokens/sec,
memory, kernel speedup); a serving endpoint; and a short writeup of the design
choices and trade-offs. The generated text is the first "wow"; the systems numbers
are what convince whoever is hiring.

## Layout
```
src/model.py    hand-written GPT (attention, MLP, LayerNorm, GPT, generate)
src/data.py     tiktoken BPE + .bin token stream + batching
src/train.py    config-driven loop (AMP + grad-accum hooks, cosine LR, ckpt)
src/sample.py   load checkpoint, generate
configs/        run configs (single source of truth)
benchmarks/     throughput.py (Level 2), kernel bench (Level 3)
tests/          shape / causality / generation tests
```
