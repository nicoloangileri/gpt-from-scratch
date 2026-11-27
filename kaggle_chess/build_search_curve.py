"""One GPU session: puzzle accuracy against search compute, for four amounts of training.

    python kaggle_chess/build_search_curve.py --quota_h 10.5
    kaggle kernels push -p kaggle_chess_search_curve/

The question. DeepMind's ChessBench models play without search and reach their numbers on
about 41 times the training examples of this repository's largest run. How much of the
missing data can search at play time buy back? The paper does not ask; its models never
search. Jones (2021, "Scaling Scaling Laws with Board Games") measured a trade between
training and search compute for AlphaZero on Hex; this is the same question for a
supervised, searchless-style chess transformer, on one small model trained for four
lengths.

The grid. The base models trained on 100M, 225M, 500M and 1B examples (no puzzle
fine-tune, the same network at the root and at the leaves, as the paper's would be), each
with the tree search of src/mcts.py at 0 (the bare network), 4, 8, 16, 32, 64, 128 and
256 simulations, on the first 2,000 of the 10,000 test puzzles (those of results/39 and
45). c 0.3 and prior temperature 0.05 for every model: the bot's setting, which results/45
found no better alternative to at 64 simulations. Not tuned per model or per budget; that
is a limitation, stated before the results.

Each card runs two models, every budget of both from the cheapest up, so if the deadline
comes the 256-simulation jobs are the ones cut; a cut job stops between puzzles and keeps
the first ones, which pair with the same puzzles of every other job.

Decided before the results, the analysis:
1. accuracy against log2(simulations) for each model, with a binomial interval;
2. the exchange rate: for each pair of neighbouring models (about 2.2 times the data
   each), the simulations at which the smaller model reaches the larger one's bare
   accuracy, interpolated linearly in log2(simulations) between the two budgets that
   bracket it; reported as not reached if 256 does not;
3. whether the curves converge (search closes the gap) or run parallel (it does not):
   the accuracy gap between 100M and 1B at 0 and at the largest budget both finished.
Puzzles are not games: results/30-32 found search can cost a fine-tuned model puzzles
and still help it in games. So the games decide any claim about play: after this
session, a match fixed in advance from these curves (a smaller model with more search
against a larger one with less) tests whether the puzzle exchange rate holds over the
board. No prediction is made here about the size of the effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook  # noqa: E402
from build_mcts_match import (ONE_B_KERNEL, TUNE_PUZZLES, W1B_KERNEL,  # noqa: E402
                              setup_cell, start_cell)

DIR = "kaggle_chess_search_curve"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-search-curve"
BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
CHECKPOINTS_DATASET = "nicolangileri/gpt-from-scratch-checkpoints"      # 100M, 225M
SIMS = [0, 4, 8, 16, 32, 64, 128, 256]
CARDS = [["1B", "225M"], ["500M", "100M"]]          # about the same work on each card
C_PUCT, TAU = 0.3, 0.05
END_BEFORE_QUOTA_H = 0.08
JOB_MARGIN_H = 0.05                                  # a job's deadline, before the session's


def curve_cell() -> str:
    return f'''\
CKPT = {{"1B": ONE_B,
        "500M": one("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt"),
        "225M": one("/kaggle/input/**/{CHECKPOINTS_DATASET.split("/")[1]}/**/chess_225M.pt"),
        "100M": one("/kaggle/input/**/{CHECKPOINTS_DATASET.split("/")[1]}/**/chess_100M.pt")}}
print("[session] checkpoints", CKPT, flush=True)
SIMS = {json.dumps(SIMS)}
CARDS = {json.dumps(CARDS)}

def curve_job(model, sims):
    out = f"curve_{{model}}_sims{{sims}}.json"
    return ["src.mcts", "--ckpt", CKPT[model], "--sims", str(sims),
            "--c_puct", "{C_PUCT}", "--tau", "{TAU}", "--puzzles", DATA + "/puzzles.csv",
            "--num", "{TUNE_PUZZLES}", "--device", "cuda", "--out", out,
            "--hours", f"{{max(0.01, hours_left() - {JOB_MARGIN_H}):.3f}}"], out

def run_queue(card, models, codes):
    for sims in SIMS:
        for model in models:
            if hours_left() < {JOB_MARGIN_H} + 0.02:
                codes[f"{{model}}_{{sims}}"] = "not started: deadline"
                continue
            args, out = curve_job(model, sims)
            codes[out] = finish(*launch(args, card, out[:-5]))

print(f"[session] curve from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
codes = {{}}
threads = [threading.Thread(target=run_queue, args=(card, m, codes)) for card, m in enumerate(CARDS)]
for t in threads: t.start()
for t in threads: t.join()
curve = {{}}
for models in CARDS:
    for model in models:
        for sims in SIMS:
            path = f"curve_{{model}}_sims{{sims}}.json"
            if os.path.exists(path):
                r = json.load(open(path))
                curve.setdefault(model, {{}})[str(sims)] = {{k: r.get(k) for k in (
                    "solved", "attempted", "accuracy", "seconds_per_move", "stopped_early")}}
json.dump({{"puzzles": {TUNE_PUZZLES}, "c_puct": {C_PUCT}, "tau": {TAU}, "sims": SIMS,
           "curve": curve, "codes": codes}}, open("search_curve_summary.json", "w"), indent=1)
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
for model, row in curve.items():
    print(f"[curve] {{model}}: " + ", ".join(
        f"{{s}}: {{v['solved']}}/{{v['attempted']}}" for s, v in row.items()), flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 9 <= end_h <= 11.5:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; the grid needs "
                         "about 8 hours per card")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "search_curve_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), curve_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "search_curve_kernel.ipynb",
                 "dataset_sources": [CHECKPOINTS_DATASET],
                 "kernel_sources": [W1B_KERNEL, ONE_B_KERNEL, BASE_KERNEL]})
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota_h", type=float, required=True,
                    help="GPU hours this session may use; check `kaggle quota` and keep a margin")
    a = ap.parse_args()
    out = build(a.quota_h, ROOT / DIR)
    print(f"wrote {out.name}/: jobs end {a.quota_h - END_BEFORE_QUOTA_H:.2f}h after the "
          "session starts")


if __name__ == "__main__":
    main()
