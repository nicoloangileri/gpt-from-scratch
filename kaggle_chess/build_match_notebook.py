"""Assemble the Kaggle notebook for a long head-to-head match.

    python kaggle_chess/build_match_notebook.py                 # depth 2 on a T4
    kaggle kernels push -p kaggle_chess_match_d2/

    python kaggle_chess/build_match_notebook.py --depth 1 --device cpu \
        --b chess_best.pt --kernel-source nicolangileri/gpt-from-scratch-chess-puzzle-w03 \
        --dir kaggle_chess_match_w03_d1 --id nicolangileri/gpt-from-scratch-chess-match-w03-d1

At depth 1 the 225M model beat its puzzle-fine-tuned self by +103 Elo over 400
games (results/19). At depth 2 a first-to-7 went 7.5-6.5 the other way, over 14
games -- too few to say anything, and 400 games at depth 2 are about 30 hours of
laptop CPU. On a GPU the network calls inside the search are much cheaper.

Same 200 starting positions as the depth-1 match (same shard, same seed), so the
two results compare on identical openings. When the depth-2 match was built, its
speed on a T4 had not been measured, so the match carries a deadline inside the
session limit and rewrites results.json after every position: if it is slower than
hoped, it stops cleanly with what it has, never between the two colours of a
position. The deadline turned out not to matter: 400 games took 8.0 hours on the T4
(28,795 s, results/21).

A depth-1 match can run in a CPU session, which does not count against the weekly GPU
quota -- slowly: 400 games took 5.6 hours on Kaggle's four CPU threads (results/22),
against 41 minutes on two laptop threads. A checkpoint that
is on Kaggle only as another kernel's output, not in the checkpoints dataset, is
attached with --kernel-source instead of uploading a new version of the dataset.
Checkpoints are found by file name, and the notebook refuses to start if the name
given to --a or --b matches more than one attached file: every training kernel writes
checkpoints/chess_best.pt, so two of them attached would otherwise pick one silently.

The folders in the repository are the notebooks that ran; rebuilding one with later
sources can differ from it (kaggle_chess_match_d2 embeds a match.py from before the
empty-match guard in summarise). So the default --dir/--id, which belong to that
depth-2 match, are refused for any other match: a push to that kernel id would
replace the output behind results/21.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, SESSION_HOURS, _notebook, setup_cell  # noqa: E402

D2_DIR = "kaggle_chess_match_d2"
D2_ID = "nicolangileri/gpt-from-scratch-chess-match-d2"


def data_cell() -> str:
    return '''\
import os, time, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard the positions come from"
'''


def match_cell(a: str, b: str, depth: int, positions: int, device: str) -> str:
    return f'''\
import glob, os, sys, time
paths = {{}}
for p in glob.glob("/kaggle/input/**/*.pt", recursive=True):
    paths.setdefault(os.path.basename(p), []).append(p)
for name in ("{a}", "{b}"):
    assert len(paths.get(name, [])) == 1, f"{{name}}: {{paths.get(name)}} (attached: {{sorted(paths)}})"
found = {{name: p[0] for name, p in paths.items()}}
hours = {SESSION_HOURS} - 0.5 - (time.time() - T0) / 3600      # leave time to save outputs
print(f"[session] match budget {{hours:.2f}}h", flush=True)
sys.argv = ["match", "--a", found["{a}"], "--b", found["{b}"], "--depth", "{depth}",
            "--positions", "{positions}", "--bag", DATA + "/test.bag", "--device", "{device}",
            "--threads", "4", "--seed", "0", "--hours", str(hours), "--out", "results.json"]
from src.match import main
main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="chess_225M.pt")
    ap.add_argument("--b", default="chess_225M_puzzles.pt")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cpu: no GPU attached, so the weekly GPU quota is not touched")
    ap.add_argument("--dataset", default="nicolangileri/gpt-from-scratch-checkpoints")
    ap.add_argument("--kernel-source", dest="kernel_sources", action="append", default=[],
                    help="a kernel whose output holds a checkpoint (repeatable)")
    ap.add_argument("--dir", default=D2_DIR)
    ap.add_argument("--id", default=D2_ID)
    a = ap.parse_args()

    is_d2 = (a.a, a.b, a.depth, a.positions, a.device, a.kernel_sources) == \
        ("chess_225M.pt", "chess_225M_puzzles.pt", 2, 200, "cuda", [])
    if not is_d2 and (a.dir == D2_DIR or a.id == D2_ID):
        ap.error(f"a different match needs its own --dir and --id: {D2_DIR} and {D2_ID} "
                 "hold the depth-2 match behind results/21")
    if a.device == "cpu" and a.depth > 1:
        print(f"warning: depth {a.depth} on a CPU is ~4.4 min a game (400 games ~29 h); "
              "the session deadline will cut the match short", file=sys.stderr)

    out = ROOT / a.dir
    out.mkdir(exist_ok=True)
    (out / "match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(), match_cell(a.a, a.b, a.depth, a.positions, a.device)]),
        indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": a.id, "title": a.id.split("/")[-1], "code_file": "match_kernel.ipynb",
                 "dataset_sources": [a.dataset], "kernel_sources": a.kernel_sources})
    if a.device == "cpu":
        meta["enable_gpu"] = False
        meta.pop("machine_shape", None)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.dir}/: {a.a} vs {a.b}, depth {a.depth}, {a.positions} positions x 2, "
          f"on {a.device}")


if __name__ == "__main__":
    main()
