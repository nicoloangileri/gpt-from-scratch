"""Run every evaluation on a Kaggle GPU instead of on the laptop.

    python kaggle_chess/build_eval_notebook.py
    kaggle datasets create -p kaggle_ckpt_dataset          # once: the checkpoints, private
    kaggle kernels push -p kaggle_chess_eval/              # then as often as needed

The evaluations are cheap for a T4 and expensive for a laptop CPU: ten checkpoints on
10,000 puzzles and on the 61,833 test positions kept a MacBook hot for an hour and
still had not finished. The checkpoints of the small runs exist only on local disk
(the kernels that trained them were pushed again, which replaces their output), so
they go up once as a private dataset. The staging folder holds hard links, not
copies, and is git-ignored.

Everything measured in one session with one evaluator, so the numbers are
comparable with each other by construction:
- puzzle accuracy on all 10,000 puzzles, for every point of the curve
- move quality (action accuracy, Kendall's tau-b) and calibration on the test set
- depth 1 against depth 2 on the checkpoint BEFORE the puzzle fine-tune: the open
  question of whether the fine-tune is why search hurts on puzzles
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

# name in the dataset -> local file. Order = the curve's order.
CHECKPOINTS = {
    "chess_1M.pt": "kaggle_chess_out/checkpoints/chess_best.pt",
    "chess_6.1M.pt": "kaggle_ab_out/checkpoints/causal/chess_best.pt",
    "chess_10M.pt": "kaggle_abbatch_out/checkpoints/batch128/chess_best.pt",
    "chess_20.5M.pt": "kaggle_chess_out2/checkpoints/chess_best.pt",
    "chess_30.7M.pt": "checkpoints/chess_30M.pt",
    "chess_50M.pt": "checkpoints/chess_50M.pt",
    "chess_100M.pt": "checkpoints/chess_100M.pt",
    "chess_225M.pt": "checkpoints/chess_225M.pt",
    "chess_225M_puzzles.pt": "checkpoints/chess_225M_puzzles.pt",
    "chess_60M_params.pt": "kaggle_60m_out/checkpoints/chess_best.pt",
}
BASE = "https://storage.googleapis.com/searchless_chess/data/"


def stage_dataset(folder: Path, dataset_id: str) -> None:
    folder.mkdir(exist_ok=True)
    for name, src in CHECKPOINTS.items():
        dst = folder / name
        if dst.exists():
            dst.unlink()
        os.link(ROOT / src, dst)              # a hard link: no second copy on disk
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": "gpt-from-scratch chess checkpoints",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "description": "Checkpoints of github.com/nicoloangileri/gpt-from-scratch "
                       "(Apache-2.0), for evaluation on a Kaggle GPU.",
    }, indent=2) + "\n")


def data_cell() -> str:
    return f'''\
import os, time, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
t = time.time()
urllib.request.urlretrieve("{BASE}puzzles.csv", DATA + "/puzzles.csv")
urllib.request.urlretrieve("{BASE}test/action_value_data.bag", DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard we measured on"
print(f"[data] puzzles + test shard in {{time.time() - t:.0f}}s", flush=True)
'''


def eval_cell() -> str:
    order = json.dumps(list(CHECKPOINTS))
    return f'''\
import glob, json, os
from src.puzzles_eval import run as puzzles
from src.agreement_eval import run as agreement

found = {{os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True)}}
missing = [n for n in {order} if n not in found]
assert not missing, f"not in the attached dataset: {{missing}}"

results = {{}}
for name in {order}:
    print(f"===== {{name}} =====", flush=True)
    results[name] = {{"puzzles": puzzles(found[name], DATA + "/puzzles.csv", 10000, "cuda"),
                      "agreement": agreement(found[name], DATA + "/test.bag", None, "cuda")}}
    # written after every checkpoint, so a session that dies keeps what it finished
    json.dump(results, open("results.json", "w"), indent=1, default=float)

# The open question from the fine-tune: search cost 3.4 points on puzzles AFTER it.
# Same comparison BEFORE it -- if search hurts there too, the fine-tune is not the cause.
print("===== search on chess_225M.pt, first 1000 puzzles =====", flush=True)
results["search_225M"] = {{d: puzzles(found["chess_225M.pt"], DATA + "/puzzles.csv", 1000, "cuda", d)
                           for d in (1, 2)}}
json.dump(results, open("results.json", "w"), indent=1, default=float)
print("[done]", flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nicolangileri/gpt-from-scratch-checkpoints")
    ap.add_argument("--dataset-dir", dest="dataset_dir", default="kaggle_ckpt_dataset")
    ap.add_argument("--dir", default="kaggle_chess_eval")
    ap.add_argument("--id", default="nicolangileri/gpt-from-scratch-chess-eval")
    a = ap.parse_args()

    stage_dataset(ROOT / a.dataset_dir, a.dataset)
    out = ROOT / a.dir
    out.mkdir(exist_ok=True)
    (out / "eval_kernel.ipynb").write_text(json.dumps(
        _notebook([setup_cell(), data_cell(), eval_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": a.id, "title": a.id.split("/")[-1], "code_file": "eval_kernel.ipynb",
                 "dataset_sources": [a.dataset], "kernel_sources": []})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"staged {len(CHECKPOINTS)} checkpoints in {a.dataset_dir}/ (hard links)\n"
          f"wrote {a.dir}/eval_kernel.ipynb\n"
          f"  1. kaggle datasets create -p {a.dataset_dir}      (once, private by default)\n"
          f"  2. kaggle kernels push -p {a.dir}/")


if __name__ == "__main__":
    main()
