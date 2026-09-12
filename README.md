# gpt-from-scratch

A decoder-only Transformer **built from scratch in PyTorch**, pointed at chess, then
made **fast** — with reproducible configs, benchmarks, and every number backed by a
run log in [`results/`](results/).

Every block (attention, MLP, LayerNorm, positional embeddings, the AdamW param
grouping) is hand-written in [`src/model.py`](src/model.py) — no `from transformers
import` of the model. The fused LayerNorm in [`src/kernels/`](src/kernels/) is
hand-written too, in Triton.

## The result

The model reads a chess position and a candidate move and predicts how good the move
is, learning from Stockfish 16 annotations (DeepMind's ChessBench). No search, no
rollouts — one forward pass per legal move, best expected value wins.

**20.5% of Lichess puzzles solved**, after 50M training examples on a free Kaggle T4:

| rating | solved |
|---|---|
| < 500 | 80.0% |
| 500–999 | 38.2% |
| 1000–1499 | 25.8% |
| 1500–1999 | 8.7% |
| 2000–2499 | 9.1% |

The point of the project is the curve, not the single number — how much accuracy an
hour of free GPU buys:

| model | examples | T4 hours | puzzles solved | val loss |
|---|---|---|---|---|
| 15M | 1M | 0.2 | 5.4% | 3.234 |
| 15M | 6.1M | 0.9 | 9.7% | 3.088 |
| 15M | 10M | 1.5 | 10.0% | 3.036 |
| 15M | 20.5M | 3.9 | 10.9% | 2.911 |
| 15M | 30.7M | 5.3 | 14.7% | 2.751 |
| 15M | **50M** | 10.0 | **20.5%** | **2.543** |
| 60M | 15.4M | 9.5 | 11.9% | 3.006 |

The curve is not flattening: the last point gained 5.8 points from 63% more data,
having gained 5.2 from tripling it before that. Validation loss was still falling at
the final step and train and val sit on top of each other, so this model is nowhere
near saturated — and 2145 of the dataset's 2148 shards are untouched.

**Capacity was the wrong lever, and the experiment is in the table.** Chinchilla
scaling puts the optimum for a 9-hour T4 budget at ~60M parameters; that model scored
11.9%, worse than the 15M one on a shorter run. At comparable data it is slightly
better, but it sees half as many examples per hour, so ten million more examples beat
forty-five million more parameters. That is a statement about a budget where the
larger model is 4x slower, not about architecture — and it would flip on better
hardware, which is exactly the kind of thing papers running on clusters never have to
report.

For scale: the ChessBench paper reaches 88.9% at a comparable model size with ~40
billion examples, about 800x this budget.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # CPU, seconds

# chess: pack one ChessBench shard (1.34 GB, ~17M annotated positions)
python -c "from src.chess_data import prepare_chessbench; \
           prepare_chessbench(['action_value-00000.bag'], 'data/chess', val_tokens=None)"
python -m src.train_chess --config configs/chess.yaml
python -m src.puzzles_eval --ckpt checkpoints/chess_best.pt --num 1000

# language model: any UTF-8 corpus
python -c "from src.data import prepare; prepare('data/tinystories.txt', 'data/tinystories')"
python -m src.train --config configs/config.yaml
python -m src.sample --ckpt checkpoints/best.pt --prompt "Once upon a time"
```

Runs are one config plus one command, seeds fixed. GPU runs are assembled into Kaggle
notebooks by [`kaggle_chess/build_chess_notebook.py`](kaggle_chess/build_chess_notebook.py).

## Systems

Throughput on a T4, measured with [`src/bench_chess.py`](src/bench_chess.py):

| change | samples/s | MFU |
|---|---|---|
| baseline | 1537 | 17.1% |
| + fused AdamW | 1750 | 19.5% |
| + torch.compile, batch 1024 | **1849** | **20.6%** |

The fused LayerNorm ([`src/kernels/layernorm.py`](src/kernels/layernorm.py)) against
PyTorch, at the model's shapes:

| rows | eager | torch.compile | Triton | bandwidth |
|---|---|---|---|---|
| 9,984 | 1.00x | 0.93x | 0.86x | 147 GB/s |
| 39,936 | 1.00x | 1.71x | **1.86x** | 298 GB/s |
| 79,872 | 1.00x | 1.40x | **1.48x** | 295 GB/s |

295 GB/s is ~92% of the T4's 320 GB/s ceiling, so for a memory-bound op there is
little left. Below ~10k rows the kernel *loses* to eager — launch overhead outweighs
the win — which is why the PyTorch fallback stays in.

End to end that kernel is worth 3%: LayerNorm is a small share of the step. Amdahl.
The remaining bottleneck is GEMMs and launch overhead, which wants CUDA graphs next.

## What was learned the hard way

Documented because the mistakes were more instructive than the successes:

- `torch.cuda.is_bf16_supported()` counts emulation and answers `True` on a T4, where
  bf16 is emulated and slow. Detect the card by compute capability instead: worth 4.1x.
- ChessBench's `puzzles.csv` is pre-advanced past the opponent's move, so `Moves[0]`
  is already the move to find. Reading it as raw Lichess scores the model on the
  opponent's forced replies and quietly divides accuracy by ~4.5 — from 14.7% to 2.4%.
  The tell was accuracy that stayed *flat* across puzzle difficulty.
- Timing a training step without `torch.cuda.synchronize()` measures how long the CPU
  took to *queue* the work. It reported an MFU of 882%.
- Benchmarking configurations in one process lets the first one leave the allocator
  fragmented; a larger batch then looks slower, and OOMs on an allocation the card
  has room for.

## Layout

```
src/model.py         hand-written GPT (attention, MLP, LayerNorm, generate)
src/chess_data.py    ChessBench .bag reader, FEN tokenizer, 1968-move vocabulary
src/train_chess.py   action-value training loop
src/puzzles_eval.py  Lichess puzzle accuracy
src/kernels/         fused LayerNorm in Triton + its bench
src/bench_chess.py   throughput bench (isolated subprocess per config)
src/data.py          BPE tokenizer + streaming corpus packing
src/train.py         language-model training loop
kaggle_chess/        builds and pushes the GPU notebooks
results/             the log behind every number above
```

## Still open

- Alpha-beta search on top of the network, benchmarked separately (the numbers above
  are search-free, and stay that way to remain comparable with the paper)
- A bot account on Lichess for a public Elo
- CUDA graphs; DDP; KV-cache serving for the language-model path
