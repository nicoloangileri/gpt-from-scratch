"""Build the Kaggle notebooks for the 1B run: 500M examples seen -> 1B.

    python kaggle_chess/build_1b.py pack                 # four CPU kernels, no GPU quota
    python kaggle_chess/build_1b.py session 1 --hours 11.25 --pack c
    nohup caffeinate -i .venv/bin/python kaggle_chess/queue_1b.py > /dev/null 2>&1 &

The 500M run again with one thing changed: how much it sees. Same model, batch, both T4s,
and the same schedule shape, a warm restart with a fresh cosine over all the new examples,
peak 3e-4, floor 3e-5, 2.5% warmup. Along the 500M run the val loss went from 2.05 up to
about 2.24 near the peak and came down to 1.9371 only in the annealing, so expect the same
here: the val loss sits above 1.9371 until the last sessions, and chess_best.pt stays the
starting weights until the annealing beats them. That is the plan working, not a failure.

- New data. Shards 42-121, 504.9M positions never used. Record counts come from each
  shard's index, read over the network without downloading the shard (bag_records'
  arithmetic); the same read reproduces build_500m's counts for shards 10, 30 and 41.
- Four packs instead of two, at most 11 GB each, split between two ranks that can hold 12.5 GB
  each, so that a session mostly reads positions no session has read.
- The validation split is the one every run has used, the first 200,000 records of shard 0,
  so the val loss stays comparable with 1.9371.
- 488,281 steps at about 3,300 samples/s is about 42 hours of training, more than one
  week's 30 hours of quota. Five sessions: three in the first week, two after the reset.

Session 1 warm-starts from the 500M run's last session, whose output has never been
replaced; nothing here writes to its kernel or to checkpoints/chess_500M.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_500m import BASE, USER, _meta, _write, evaluate_cell, gpu_cell, run_cell  # noqa: E402
from build_long_run import SESSION_HOURS, setup_cell  # noqa: E402

INIT_KERNEL = f"{USER}/gpt-from-scratch-chess-500m-s3"      # the 500M model
EXAMPLES = 500_000_000                     # 500M + 500M = 1B
BATCH = 1024
MAX_STEPS = EXAMPLES // BATCH
LR, MIN_LR = 3e-4, 3e-5
WARMUP = MAX_STEPS // 40
# Records per shard, from each shard's index. The pack kernels assert these exactly.
SHARD_RECORDS = {
    42: 1661626, 43: 16784209, 44: 16756423, 45: 8481349, 46: 1891801, 47: 286134,
    48: 706173, 49: 1677719, 50: 1321735, 51: 1298550, 52: 347275, 53: 759154,
    54: 16778184, 55: 16782773, 56: 16765174, 57: 15480887, 58: 1284467, 59: 16778126,
    60: 16772312, 61: 16768605, 62: 16765441, 63: 8437797, 64: 7047582, 65: 1289325,
    66: 16783192, 67: 16579455, 68: 196592, 69: 16781321, 70: 12193783, 71: 4582012,
    72: 16764654, 73: 16773737, 74: 8464079, 75: 2609904, 76: 573488, 77: 1063983,
    78: 187916, 79: 1650964, 80: 862595, 81: 1375187, 82: 1145938, 83: 458549,
    84: 588961, 85: 65146, 86: 219658, 87: 1072972, 88: 198452, 89: 654795,
    90: 12373309, 91: 6691670, 92: 1748792, 93: 8332852, 94: 6099989, 95: 1676325,
    96: 2247731, 97: 354845, 98: 1162912, 99: 448998, 100: 436147, 101: 342223,
    102: 697600, 103: 11839676, 104: 6694539, 105: 1548771, 106: 16076700, 107: 696189,
    108: 8455675, 109: 4234444, 110: 392324, 111: 770569, 112: 1123246, 113: 341179,
    114: 365973, 115: 901374, 116: 196514, 117: 16780733, 118: 16771987, 119: 16775110,
    120: 16765707, 121: 16783418,
}
PACKS = {"c": list(range(42, 60)), "d": list(range(60, 71)), "e": list(range(71, 107)), "f": list(range(107, 122))}
VAL_RECORDS = 200_000
EVAL_RESERVE_H = 0.9                       # puzzles + move quality on the best checkpoint


def pack_id(name: str) -> str:
    return f"{USER}/gpt-from-scratch-chess-1b-pack-{name}"


def session_id(k: int) -> str:
    return f"{USER}/gpt-from-scratch-chess-1b-s{k}"


def pack_cells(name: str) -> list[str]:
    shards = PACKS[name]
    expected = sum(SHARD_RECORDS[s] for s in shards)
    order = sorted(shards, key=lambda s: -SHARD_RECORDS[s])
    first = f'''\
import json, os, time, urllib.request
from src.chess_data import pack_val

OUT = "/kaggle/working/packed"            # the kernel's output, mounted by the GPU sessions
TMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
os.makedirs(OUT, exist_ok=True)
T_PACK = time.time()
urllib.request.urlretrieve("{BASE}train/action_value-00000-of-02148_data.bag", TMP + "/shard0.bag")
val = pack_val(TMP + "/shard0.bag", OUT, {VAL_RECORDS})   # the val split of every run so far
os.remove(TMP + "/shard0.bag")
print(f"[pack] val: {{val['val']:,}} records in {{time.time() - T_PACK:.0f}}s", flush=True)
'''
    second = f'''\
import functools
from multiprocessing import Pool
from src.chess_data import download_and_pack

# One process per core, largest shards first so the pool does not end on one big shard.
work = functools.partial(download_and_pack, out_root=OUT, tmp_dir=TMP, base_url="{BASE}")
results = []
with Pool(os.cpu_count()) as pool:
    for r in pool.imap_unordered(work, {order}):
        results.append(r)
        print(f"[pack] shard {{r['shard']}}: {{r['train']:,}} records, {{r['total_s']}}s "
              f"({{len(results)}}/{len(order)})", flush=True)
packed = sum(r["train"] for r in results)
skipped = sum(r["skipped"] for r in results)
assert packed + skipped == {expected}, f"packed {{packed:,}} + skipped {{skipped:,}}, expected {expected:,}"
manifest = {{"pack": "{name}", "shards": {shards}, "train_records": packed, "skipped": skipped,
            "val_records": val["val"], "val_source": "first {VAL_RECORDS:,} records of shard 0",
            "per_shard": sorted(results, key=lambda r: r["shard"]),
            "seconds": round(time.time() - T_PACK)}}
json.dump(manifest, open(OUT + "/manifest.json", "w"), indent=1)
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"[pack] {{packed:,}} records, {{size / 1e9:.1f}} GB, {{time.time() - T_PACK:.0f}}s", flush=True)
'''
    return [setup_cell(), first, second]


def build_pack(name: str) -> Path:
    return _write(f"kaggle_chess_1b_pack_{name}", "pack_kernel.ipynb", pack_cells(name),
                  _meta(pack_id(name), "pack_kernel.ipynb", gpu=False, datasets=[], kernels=[]))


def locate_cell(k: int, pack: str) -> str:
    if k == 1:
        start = f'''\
found = glob.glob("/kaggle/input/**/{INIT_KERNEL.split("/")[1]}/**/chess_best.pt", recursive=True)
assert len(found) == 1, f"expected the 500M model's chess_best.pt, found {{found}}"
START = ["--init_from", found[0]]
print("[init] warm-starting from", found[0], flush=True)
'''
    else:
        start = f'''\
found = glob.glob("/kaggle/input/**/last.pt", recursive=True)
assert len(found) == 1, f"expected the last.pt of session {k - 1}, found {{found}}"
START = ["--resume", found[0]]
best = os.path.join(os.path.dirname(found[0]), "chess_best.pt")
if os.path.exists(best):                  # the best so far survives this session
    shutil.copy(best, "checkpoints/chess_best.pt")
print("[resume] from", found[0], flush=True)
'''
    return f'''\
import glob, json, os, shutil
os.makedirs("checkpoints", exist_ok=True)
manifests = glob.glob("/kaggle/input/**/packed/manifest.json", recursive=True)
assert len(manifests) == 1, f"expected pack {pack}'s output only, found {{manifests}}"
DATA = os.path.dirname(manifests[0])
MANIFEST = json.load(open(manifests[0]))
assert MANIFEST["pack"] == "{pack}", MANIFEST["pack"]
print(f"[data] pack {{MANIFEST['pack']}}: {{MANIFEST['train_records']:,}} records from shards "
      f"{{MANIFEST['shards'][0]}}-{{MANIFEST['shards'][-1]}}", flush=True)
{start}'''


def train_cell(hours: float) -> str:
    return f'''\
import time
budget = {hours} * 3600 - (time.time() - T0)
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
torchrun(["--config", "configs/chess.yaml", "--device", "cuda", "--data_dir", DATA,
          "--out_dir", "checkpoints", "--max_steps", "{MAX_STEPS}", "--batch_size", "{BATCH}",
          "--compile", "true", "--lr", "{LR}", "--min_lr", "{MIN_LR}",
          "--warmup_steps", "{WARMUP}", "--max_ram_gb", str(MAX_RAM_GB),
          "--eval_interval", "4000", "--eval_iters", "50", "--log_interval", "1000",
          "--save_every", "2000", "--deadline_s", str(budget), *START])
'''


def finish_cell(k: int, hours: float) -> str:
    return f'''\
import json, time, torch
state = torch.load("checkpoints/last.pt", map_location="cpu", weights_only=False)
done = state["next_step"] > {MAX_STEPS}
json.dump({{"session": {k}, "next_step": state["next_step"], "max_steps": {MAX_STEPS},
           "best_val": state["best_val"], "done": done}}, open("progress.json", "w"))
print(f"[session] next step {{state['next_step']:,}} of {MAX_STEPS:,}, best val {{state['best_val']:.4f}}",
      flush=True)
left_h = {hours} - (time.time() - T0) / 3600      # this session's hours, not the maximum
if done and left_h >= {EVAL_RESERVE_H}:
    evaluate("checkpoints/chess_best.pt")
elif done:
    print(f"[session] training done, {{left_h:.2f}}h left: evaluate chess_best.pt in its own session")
'''


def build_session(k: int, hours: float, pack: str) -> Path:
    if not 0.5 <= hours <= SESSION_HOURS:
        raise SystemExit(f"--hours {hours}: between 0.5 and {SESSION_HOURS}")
    if pack not in PACKS:
        raise SystemExit(f"--pack {pack}: one of {sorted(PACKS)}")
    kernels = [pack_id(pack), INIT_KERNEL if k == 1 else session_id(k - 1)]
    cells = [setup_cell(), locate_cell(k, pack), gpu_cell(), run_cell(), train_cell(hours),
             evaluate_cell(), finish_cell(k, hours)]
    return _write(f"kaggle_chess_1b_s{k}", "session_kernel.ipynb", cells,
                  _meta(session_id(k), "session_kernel.ipynb", gpu=True, datasets=[], kernels=kernels))


def build_eval(k: int) -> Path:
    """Evaluate session k's best checkpoint, when that session had no time left for it."""
    locate = f'''\
import glob
found = glob.glob("/kaggle/input/**/chess_best.pt", recursive=True)
assert len(found) == 1, f"expected session {k}'s chess_best.pt, found {{found}}"
'''
    cells = [setup_cell(), locate, evaluate_cell(), "evaluate(found[0])\n"]
    return _write("kaggle_chess_1b_eval", "eval_kernel.ipynb", cells,
                  _meta(f"{USER}/gpt-from-scratch-chess-1b-eval", "eval_kernel.ipynb", gpu=True,
                        datasets=[], kernels=[session_id(k)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("pack")
    s = sub.add_parser("session")
    s.add_argument("k", type=int)
    s.add_argument("--hours", type=float, default=SESSION_HOURS,
                   help="session length, at most the GPU quota left")
    s.add_argument("--pack", choices=sorted(PACKS), required=True)
    e = sub.add_parser("eval")
    e.add_argument("k", type=int, help="the session whose chess_best.pt to evaluate")
    a = ap.parse_args()
    if a.what == "pack":
        for name in PACKS:
            print("wrote", build_pack(name))
    elif a.what == "eval":
        print("wrote", build_eval(a.k))
    else:
        print("wrote", build_session(a.k, a.hours, a.pack))
    print(f"{EXAMPLES:,} examples = {MAX_STEPS:,} steps x {BATCH}, lr {LR} -> {MIN_LR}, "
          f"warmup {WARMUP:,}")


if __name__ == "__main__":
    main()
