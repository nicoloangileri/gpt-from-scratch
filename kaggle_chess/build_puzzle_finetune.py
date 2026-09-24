"""Assemble the Kaggle notebook that fine-tunes on Lichess puzzles.

The action-value model knows how good a move is and still misses tactics: at 225M
examples it picks Stockfish's first choice on 45% of the test positions. Lichess publishes 6.1M puzzles (CC0)
whose label is exactly "this move, not the others", so this run adds a listwise
ranking loss over the legal moves on top of the action-value loss it was trained on.

Both losses run: fine-tuning on the ranking alone would teach the model which move is
best and let it forget how good moves ARE, which is what the search and the bot read
out of it.

    python kaggle_chess/build_puzzle_finetune.py
    kaggle kernels push -p kaggle_chess_puzzle/

The starting weights come from another kernel's output, mounted read-only, so the
225M checkpoint cannot be damaged by this run whatever it does.

Learning rate, deliberately: the 225M run warm-restarted at 5e-4 and pushed its own
validation loss from 2.2412 up to 2.4834 before spending 52,000 steps getting back.
A fine-tune that cannot afford that has to start low and decay inside its budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import (ROOT, SESSION_HOURS, _notebook, data_cell,  # noqa: E402
                            init_cell, setup_cell)

PUZZLE_DB = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def puzzle_data_cell(max_puzzles: int | None, val_positions: int) -> str:
    """Download the puzzle database and pack it, minus the held-out test ids."""
    return f'''\
import os, time, urllib.request
os.system("pip install -q zstandard")
t = time.time()
ZST = DATA + "/lichess_db_puzzle.csv.zst"
urllib.request.urlretrieve("{PUZZLE_DB}", ZST)
print(f"[puzzles] database downloaded in {{time.time() - t:.0f}}s "
      f"({{os.path.getsize(ZST) / 1e6:.0f}} MB)", flush=True)

from src.puzzle_data import held_out_puzzle_ids, pack_puzzles
# DATA + "/puzzles.csv" is the 10,000-puzzle benchmark this repo reports on. Excluding
# those ids is what keeps the number honest: the two files are byte-identical for the
# same PuzzleId, so without this the fine-tune would be training on the test set.
held_out = held_out_puzzle_ids(DATA + "/puzzles.csv")
print(f"[puzzles] excluding {{len(held_out):,}} held-out ids", flush=True)
t = time.time()
stats = pack_puzzles(ZST, DATA + "/puzzles", held_out,
                     max_puzzles={max_puzzles}, val_positions={val_positions})
os.remove(ZST)
assert stats["excluded"] > 0, "no held-out puzzle was skipped: the exclusion is not working"
print(f"[puzzles] packed in {{time.time() - t:.0f}}s", flush=True)
'''


def finetune_cell(max_steps: int, batch: int, rank_batch: int, lr: float,
                  rank_weight: float = 1.0) -> str:
    return f'''\
import sys, time
budget = {SESSION_HOURS} * 3600 - (time.time() - T0)
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--data_dir", DATA + "/chess", "--out_dir", "checkpoints",
            "--puzzle_dir", DATA + "/puzzles", "--rank_batch", "{rank_batch}",
            "--rank_weight", "{rank_weight}",
            "--max_steps", "{max_steps}", "--batch_size", "{batch}", "--compile", "true",
            "--lr", "{lr}", "--min_lr", "{lr / 10}",
            "--warmup_steps", "{max(100, max_steps // 50)}", "--max_ram_gb", str(MAX_RAM_GB),
            "--eval_interval", "1000", "--eval_iters", "25", "--log_interval", "250",
            "--save_every", "2000", "--deadline_s", str(budget), "--init_from", INIT]
from src.train_chess import main
main()
'''


def compare_cell(puzzles: int, search_puzzles: int) -> str:
    """Both checkpoints on the same benchmark, in the same session.

    Comparing against a number measured in another run on another day is how a
    difference in the harness gets read as a difference in the model.
    """
    return f'''\
from src.puzzles_eval import run
print("===== before: the checkpoint this run started from =====", flush=True)
before = run(INIT, DATA + "/puzzles.csv", {puzzles}, "cuda")
print("===== after: fine-tuned on Lichess puzzles =====", flush=True)
after = run("checkpoints/chess_best.pt", DATA + "/puzzles.csv", {puzzles}, "cuda")

# And the other open question: does alpha-beta search help on PUZZLES? The old answer
# was measured with the parity bug and is void. Same checkpoint, same puzzles, depth 1
# against depth 2, so the only difference is the search.
print("===== search, on the first {search_puzzles} puzzles =====", flush=True)
d1 = run("checkpoints/chess_best.pt", DATA + "/puzzles.csv", {search_puzzles}, "cuda", 1)
d2 = run("checkpoints/chess_best.pt", DATA + "/puzzles.csv", {search_puzzles}, "cuda", 2)
print("[summary]", {{"before": before, "after": after, "depth1": d1, "depth2": d2}}, flush=True)
'''


def named_init_cell(name: str, dataset: str) -> str:
    """Start from a checkpoint in the private dataset, chosen by name -- the dataset
    holds ten of them, so "the first .pt found" would be a coin toss."""
    return f'''\
import glob, os
found = {{os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True)}}
assert "{name}" in found, "{name} not under /kaggle/input: is {dataset} attached?"
INIT = found["{name}"]
print("[init] warm-starting from", INIT, flush=True)
'''


def new_model_eval_cell(puzzles: int, search_puzzles: int) -> str:
    """Only the new checkpoint, with the same code and precision that measured the
    others (results/17_eval_on_kaggle.json): that evaluation reproduced its numbers
    exactly across sessions, so re-measuring the references would only spend quota.
    The move-quality pass is the slow one -- ~27 min per model in fp32 on a T4."""
    return f'''\
import json, urllib.request
from src.puzzles_eval import run as puzzles
from src.agreement_eval import run as agreement
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
NEW = "checkpoints/chess_best.pt"
results = {{"puzzles": puzzles(NEW, DATA + "/puzzles.csv", {puzzles}, "cuda")}}
json.dump(results, open("results.json", "w"), indent=1, default=float)
results["agreement"] = agreement(NEW, DATA + "/test.bag", None, "cuda")
json.dump(results, open("results.json", "w"), indent=1, default=float)
print("===== search, on the first {search_puzzles} puzzles =====", flush=True)
results["search"] = {{d: puzzles(NEW, DATA + "/puzzles.csv", {search_puzzles}, "cuda", d) for d in (1, 2)}}
json.dump(results, open("results.json", "w"), indent=1, default=float)
print("[done]", flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--rank_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--shards", type=int, default=3,
                    help="action-value shards: enough to keep the old loss honest")
    ap.add_argument("--max_puzzles", type=int, default=None, help="None = all 6.1M")
    ap.add_argument("--val_positions", type=int, default=20_000)
    ap.add_argument("--puzzles", type=int, default=10_000, help="benchmark size")
    ap.add_argument("--search_puzzles", type=int, default=1000)
    ap.add_argument("--dir", default="kaggle_chess_puzzle")
    ap.add_argument("--id", default="nicolangileri/gpt-from-scratch-chess-puzzle")
    ap.add_argument("--source", default="nicolangileri/gpt-from-scratch-chess-225b",
                    help="kernel whose output holds the chess_best.pt to start from")
    ap.add_argument("--rank_weight", type=float, default=1.0,
                    help="weight of the ranking loss; 1.0 is the first fine-tune")
    ap.add_argument("--init-dataset", dest="init_dataset", default=None,
                    help="start from a named checkpoint in this dataset instead of --source")
    ap.add_argument("--init-name", dest="init_name", default="chess_225M.pt")
    a = ap.parse_args()

    out = ROOT / a.dir
    out.mkdir(exist_ok=True)
    from_dataset = a.init_dataset is not None
    cells = [
        setup_cell(),
        # fail before any download if the starting weights are missing
        named_init_cell(a.init_name, a.init_dataset) if from_dataset else init_cell(a.source),
        data_cell(a.shards),
        puzzle_data_cell(a.max_puzzles, a.val_positions),
        finetune_cell(a.steps, a.batch_size, a.rank_batch, a.lr, a.rank_weight),
        new_model_eval_cell(a.puzzles, a.search_puzzles) if from_dataset
        else compare_cell(a.puzzles, a.search_puzzles),
    ]
    (out / "puzzle_kernel.ipynb").write_text(json.dumps(_notebook(cells), indent=1))

    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": a.id, "title": a.id.split("/")[-1], "code_file": "puzzle_kernel.ipynb",
                 "kernel_sources": [] if from_dataset else [a.source],
                 "dataset_sources": [a.init_dataset] if from_dataset else []})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/puzzle_kernel.ipynb and kernel-metadata.json\n"
          f"  start from : {a.init_dataset + '/' + a.init_name if from_dataset else a.source}\n"
          f"  {a.steps} steps, batch {a.batch_size} + {a.rank_batch} puzzle positions, "
          f"lr {a.lr}, rank weight {a.rank_weight}\n"
          f"  push with  : kaggle kernels push -p {a.dir}/")


if __name__ == "__main__":
    main()
