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

**49.1% of Lichess puzzles solved** on all 10,000 test puzzles, after 100M training
examples on a free Kaggle T4. The number only means something next to a baseline, so
here is one: a ten-line rule that grabs the most valuable capture, or mates when it
can, solves **27.4%** — and picking legal moves at random solves 0.6%.

| rating | random | grab-the-piece | 50M examples | **100M examples** |
|---|---|---|---|---|
| < 500 | — | 74.3% | 74.3% | **93.2%** |
| 500–999 | — | 58.3% | 61.8% | **84.4%** |
| 1000–1499 | — | 32.0% | 34.6% | **61.6%** |
| 1500–1999 | — | 12.0% | 14.5% | **30.3%** |
| 2000–2499 | — | 2.8% | 6.2% | **10.8%** |
| 2500+ | — | 1.1% | 2.5% | **3.2%** |
| **all 10,000** | **0.6%** | **27.4%** | **30.3%** | **49.1%** |

The gap widens with difficulty — nearly 4x the heuristic above 2000 — which is the
part that cannot be explained by capturing hanging pieces. Note also what the baseline
exposes: **every model below 50M examples scores worse than the ten-line rule.** One
puzzle in five needs a single move, and it is usually a capture.

The point of the project is the curve, not the single number — how much accuracy an
hour of free GPU buys:

| model | examples | T4 hours | puzzles solved | val loss |
|---|---|---|---|---|
| 15M | 1M | 0.2 | 20.5% | 3.234 |
| 15M | 6.1M | 0.9 | 21.5% | 3.088 |
| 15M | 10M | 1.5 | 21.8% | 3.036 |
| 15M | 20.5M | 3.9 | 21.9% | 2.911 |
| 15M | 30.7M | 5.3 | 24.6% | 2.751 |
| 15M | 50M | 10.0 | 30.3% | 2.543 |
| 15M | **100M** | 21.4 | **49.1%** | **2.242** |
| 60M | 15.4M | 9.5 | 21.8% | 3.006 |

(Rows up to 30.7M are measured on the first 2,000 puzzles, the rest on all 10,000.)

The curve is not flattening: the last point nearly doubled the score by doubling the
data. Train and validation loss sit on top of each other and were still falling at the
final step, so the model is nowhere near saturated — and 2,142 of the dataset's 2,148
shards are untouched.

Per move, on 400 held-out positions: the model picks Stockfish's best move **38.0%** of
the time (29.5% at 50M, 20.0% at 30.7M), gives up **3.6 points** of win probability per
move on average (8.7 at 30.7M), and blunders more than 15 points on **5.5%** of moves,
down from 17.5% — one every eighteen moves instead of one every six.

**Capacity was the wrong lever, and the experiment is in the table.** Chinchilla
scaling puts the optimum for a 9-hour T4 budget at ~60M parameters; that model scored
21.8%, level with a 15M model trained on 10M examples and below the same model at
30.7M. It is not that scaling laws are wrong: the larger model is 4x slower, so it
sees far fewer examples per hour, and on this budget data returns more than parameters.
That would flip on better hardware — exactly the kind of thing papers running on
clusters never have to report.

For scale: the ChessBench paper reaches 88.9% at a comparable model size with ~40
billion examples, about 400x this budget.

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
- The same metric was got wrong twice, in opposite directions. Lichess puzzle files
  give the position *before* the opponent's move, so the solver plays the odd plies.
  A first version read it right; a "fix" then flipped it, on the evidence that
  `Moves[0]` matched the first move of the `Solution` column — which is that same
  line written in another notation, so the check could only agree. For ten days every
  reported accuracy was the model guessing the *opponent's* moves. What settled it:
  ChessBench's file is byte-identical to `lichess_db_puzzle` for the same PuzzleId,
  and DeepMind's own evaluator asks the engine on odd plies. The lesson is not about
  chess: **a test that compares a field with a restatement of itself will certify
  anything.** The replacement asks who ends up checkmated.
- A metric without a baseline is not a measurement. Until a ten-line heuristic was run
  on the same puzzles, there was no way to know that most models here were below it.
- Timing a training step without `torch.cuda.synchronize()` measures how long the CPU
  took to *queue* the work. It reported an MFU of 882%.
- Benchmarking configurations in one process lets the first one leave the allocator
  fragmented; a larger batch then looks slower, and OOMs on an allocation the card
  has room for.
- A CPU-only test cannot catch a device-placement bug: resuming loaded the whole
  checkpoint onto the GPU, RNG state included, and `torch.set_rng_state` refused it.
  Running the same test on MPS reproduced it locally.
- A mutation check can be fooled by a stale `.pyc`: the mutated and restored files had
  the same size and the same mtime second, so Python reused the old bytecode and the
  "restored" run still failed.

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
