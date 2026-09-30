"""Build the Kaggle notebooks for the 500M run: 225M examples seen -> 500M.

    python kaggle_chess/build_500m.py pack               # two CPU kernels: pack A and B
    python kaggle_chess/build_500m.py probe              # a short GPU check, ~0.4 h
    python kaggle_chess/build_500m.py session 1 --hours 11.25
    kaggle kernels push -p kaggle_chess_500m_s1/         # kaggle_chess/queue_500m.sh does this

What makes it cheaper than the 225M run, in the order of what it saves:

- Both GPUs. Kaggle's "GPU" accelerator is two T4s (kaggle-cli PR 1198, merged
  17 Sep 2026: "GPU T4 x2 (default GPU; single T4 is not supported)"), and the quota
  counts the session, not the cards (`kaggle quota` shows 14.15 h used for two sessions
  that lasted 6.13 h and 8.0 h). Every run so far trained on one of the two. The
  sessions now launch src.train_chess under torchrun, one rank per GPU, global batch
  still 1024.
- No packing on GPU time. The 225M sessions spent 24-25 minutes each re-packing their
  shards. Here two CPU-only kernels -- which do not touch the GPU quota -- pack the
  data once, in parallel across the CPU's cores, and the GPU sessions mount their
  output and only read it.
- New data. Shards 10-41 (285M positions) were never used; pack A is 10-30, pack B
  31-41, so the 275M new examples are about one pass over positions the model has
  never seen, where the 225M run went over shards 0-9 about 1.5 times. Session 1
  reads pack A, every later session pack B.
- The validation split is the one every run so far has used, the first 200,000
  records of shard 0 (none of which is in 10-41), so the val loss stays comparable
  with the 225M run's 2.0557.

Schedule: a warm restart from chess_225M.pt with a fresh cosine over all 275M
examples (268,554 steps), peak 3e-4, floor 3e-5, 2.5% warmup. The 225M run restarted at
5e-4 and its val loss went from 2.2412 to 2.4834 before coming down; along that run the
val loss tracked the learning rate closely (2.22-2.25 while it was near 3e-4, 2.0557 at
5e-5), so a lower peak means a smaller detour, and the full decay is kept -- the lesson
of that run is that the gain arrives in the annealing.

Each session writes checkpoints/last.pt (to resume), checkpoints/chess_best.pt (the best
by val loss so far, carried over from the previous session) and progress.json, which
the queue reads to decide whether another session is needed. The session that
finishes training evaluates the best checkpoint as results/17 did, if time is left.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, SESSION_HOURS, _notebook, setup_cell  # noqa: E402

BASE = "https://storage.googleapis.com/searchless_chess/data/"
USER = "nicolangileri"
CHECKPOINTS_DATASET = f"{USER}/gpt-from-scratch-checkpoints"
INIT = "chess_225M.pt"                     # in that dataset
EXAMPLES = 275_000_000                     # 225M + 275M = 500M
BATCH = 1024
MAX_STEPS = EXAMPLES // BATCH
LR, MIN_LR = 3e-4, 3e-5
WARMUP = MAX_STEPS // 40
# Records per shard, from each shard's index (see chess_data.bag_records). The pack
# kernels assert that they packed exactly this many.
SHARD_RECORDS = {
    10: 22677302, 11: 12346148, 12: 6939595, 13: 1526246, 14: 8308296, 15: 8447128,
    16: 4380589, 17: 604414, 18: 8888524, 19: 598174, 20: 2273145, 21: 3012883,
    22: 1225038, 23: 461772, 24: 2488725, 25: 1160127, 26: 16765692, 27: 16758119,
    28: 8487930, 29: 8282072, 30: 8508259, 31: 5362274, 32: 2907007, 33: 16778286,
    34: 16769844, 35: 4334472, 36: 20946914, 37: 16688538, 38: 16845665, 39: 16688996,
    40: 7170227, 41: 16314361,
}
PACKS = {"a": list(range(10, 31)), "b": list(range(31, 42))}
VAL_RECORDS = 200_000
EVAL_RESERVE_H = 0.9                       # puzzles + move quality on the best checkpoint


def pack_id(name: str) -> str:
    return f"{USER}/gpt-from-scratch-chess-500m-pack-{name}"


def session_id(k: int) -> str:
    return f"{USER}/gpt-from-scratch-chess-500m-s{k}"


def pack_for_session(k: int) -> str:
    return "a" if k == 1 else "b"


def _meta(kernel_id: str, code_file: str, gpu: bool, datasets: list[str],
          kernels: list[str]) -> dict:
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": kernel_id, "title": kernel_id.split("/")[-1], "code_file": code_file,
                 "dataset_sources": datasets, "kernel_sources": kernels})
    if not gpu:
        meta["enable_gpu"] = False
        meta.pop("machine_shape", None)
    return meta


def _write(folder: str, code_file: str, cells: list[str], meta: dict) -> Path:
    out = ROOT / folder
    out.mkdir(exist_ok=True)
    (out / code_file).write_text(json.dumps(_notebook(cells), indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


# --- the packing kernels (CPU) ------------------------------------------------------

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

# One process per core, one shard per task, largest shards first so the pool does not
# end on a single big shard with the other workers idle.
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
    folder = f"kaggle_chess_500m_pack_{name}"
    return _write(folder, "pack_kernel.ipynb", pack_cells(name),
                  _meta(pack_id(name), "pack_kernel.ipynb", gpu=False, datasets=[], kernels=[]))


# --- the training sessions (GPU) ----------------------------------------------------

def locate_cell(k: int, pack: str) -> str:
    if k == 1:
        start = f'''\
found = glob.glob("/kaggle/input/**/{INIT}", recursive=True)
assert found, "{INIT} not under /kaggle/input: is {CHECKPOINTS_DATASET} attached?"
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


def gpu_cell() -> str:
    return '''\
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout, flush=True)
NPROC = torch.cuda.device_count()
avail_gb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1e6
MAX_RAM_GB = round(0.8 * avail_gb / NPROC, 1)      # each rank holds 1/NPROC of the pack
print(f"[session] {NPROC} GPU(s), {avail_gb:.1f} GB RAM available, {MAX_RAM_GB} GB per rank", flush=True)
'''


SILENCE_MIN = 45     # no output for this long = stuck; a normal run prints every ~5 min


def run_cell() -> str:
    """torchrun from a notebook cell. The ranks' output goes through a pipe and is
    printed by this cell: written straight to the process's stdout, it would miss the
    notebook's log, which is the only place a Kaggle batch run can be read from.

    And a watchdog. A rank stuck without an error (a collective that never returns, a
    read that never ends) would otherwise hold the session until Kaggle kills it at
    12 hours -- a whole session of quota for nothing. Training prints every 1,000
    steps (about five minutes on two T4s); after SILENCE_MIN minutes of silence the
    ranks are killed, the cell fails, the session ends, and the queue stops on it."""
    return f'''\
import os, queue, signal, subprocess, sys, threading, time

def torchrun(args, silence_min={SILENCE_MIN}):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", str(NPROC), "-m", "src.train_chess", *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)       # its own group: the ranks die with it
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in p.stdout] + [lines.put(None)],
                     daemon=True).start()
    last = time.time()
    while True:
        try:
            line = lines.get(timeout=min(60, silence_min * 30))
        except queue.Empty:
            if time.time() - last > silence_min * 60:
                print(f"[watchdog] no output for {{silence_min}} min: killing the ranks", flush=True)
                os.killpg(p.pid, signal.SIGKILL)
                break
            continue
        if line is None:
            break
        last = time.time()
        print(line, end="", flush=True)
    assert p.wait() == 0, f"training exited with {{p.returncode}}"
'''


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


def evaluate_cell() -> str:
    """The measurements of results/17, on all 10,000 puzzles and the whole test shard."""
    return f'''\
import json, urllib.request

def evaluate(ckpt):
    urllib.request.urlretrieve("{BASE}puzzles.csv", "/tmp/puzzles.csv")
    urllib.request.urlretrieve("{BASE}test/action_value_data.bag", "/tmp/test.bag")
    from src.puzzles_eval import run as puzzles
    from src.agreement_eval import run as agreement
    results = {{"checkpoint": ckpt, "puzzles": puzzles(ckpt, "/tmp/puzzles.csv", 10000, "cuda")}}
    json.dump(results, open("results.json", "w"), indent=1, default=float)
    results["agreement"] = agreement(ckpt, "/tmp/test.bag", None, "cuda")
    json.dump(results, open("results.json", "w"), indent=1, default=float)
    print("[eval] results.json written", flush=True)
'''


def build_session(k: int, hours: float, pack: str | None = None) -> Path:
    if not 0.5 <= hours <= SESSION_HOURS:
        raise SystemExit(f"--hours {hours}: between 0.5 and {SESSION_HOURS}")
    pack = pack or pack_for_session(k)
    if pack not in PACKS:
        raise SystemExit(f"--pack {pack}: one of {sorted(PACKS)}")
    kernels = [pack_id(pack)] + ([session_id(k - 1)] if k > 1 else [])
    datasets = [CHECKPOINTS_DATASET] if k == 1 else []
    cells = [setup_cell(), locate_cell(k, pack), gpu_cell(), run_cell(), train_cell(hours),
             evaluate_cell(), finish_cell(k, hours)]
    return _write(f"kaggle_chess_500m_s{k}", "session_kernel.ipynb", cells,
                  _meta(session_id(k), "session_kernel.ipynb", gpu=True, datasets=datasets,
                        kernels=kernels))


# --- the probe (GPU, short) ---------------------------------------------------------

def probe_cell() -> str:
    """Two GPUs or one, throughput of each, and a stop + resume under torchrun -- all
    on the real pack, from the real starting checkpoint, before 25 hours depend on it."""
    common = (f'"--config", "configs/chess.yaml", "--device", "cuda", "--data_dir", DATA, '
              f'"--max_steps", "{MAX_STEPS}", "--batch_size", "{BATCH}", "--compile", "true", '
              f'"--lr", "{LR}", "--min_lr", "{MIN_LR}", "--warmup_steps", "{WARMUP}", '
              f'"--max_ram_gb", str(round(MAX_RAM_GB * NPROC_ALL / nproc, 1)), '
              f'"--eval_interval", "100000", '
              f'"--eval_iters", "5", "--log_interval", "100", "--save_every", "0"')
    return f'''\
import time
NPROC_ALL = NPROC          # one GPU alone holds the whole pack: give it the whole budget
for label, nproc, extra in [("two GPUs", NPROC_ALL, ["--out_dir", "probe2", "--stop_at_step", "800", *START]),
                            ("resume on two GPUs", NPROC_ALL, ["--out_dir", "probe2r", "--stop_at_step", "900",
                                                               "--resume", "probe2/last.pt"]),
                            ("one GPU", 1, ["--out_dir", "probe1", "--stop_at_step", "500", *START])]:
    NPROC = nproc
    t = time.time()
    print(f"===== {{label}} =====", flush=True)
    torchrun([{common}, *extra])
    print(f"[probe] {{label}}: {{time.time() - t:.0f}}s wall", flush=True)
NPROC = NPROC_ALL
'''


def build_probe() -> Path:
    cells = [setup_cell(), locate_cell(1, "a"), gpu_cell(), run_cell(), probe_cell()]
    return _write("kaggle_chess_500m_probe", "probe_kernel.ipynb", cells,
                  _meta(f"{USER}/gpt-from-scratch-chess-500m-probe", "probe_kernel.ipynb",
                        gpu=True, datasets=[CHECKPOINTS_DATASET], kernels=[pack_id("a")]))


def build_eval(k: int) -> Path:
    """Evaluate session k's best checkpoint, when that session had no time left for it."""
    locate = f'''\
import glob
found = glob.glob("/kaggle/input/**/chess_best.pt", recursive=True)
assert len(found) == 1, f"expected session {k}'s chess_best.pt, found {{found}}"
'''
    cells = [setup_cell(), locate, evaluate_cell(), "evaluate(found[0])\n"]
    return _write("kaggle_chess_500m_eval", "eval_kernel.ipynb", cells,
                  _meta(f"{USER}/gpt-from-scratch-chess-500m-eval", "eval_kernel.ipynb", gpu=True,
                        datasets=[], kernels=[session_id(k)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("pack")
    sub.add_parser("probe")
    s = sub.add_parser("session")
    s.add_argument("k", type=int)
    s.add_argument("--hours", type=float, default=SESSION_HOURS,
                   help="session length, at most the GPU quota left")
    s.add_argument("--pack", choices=sorted(PACKS), default=None,
                   help="default: a for session 1, b after; the queue picks the less used")
    e = sub.add_parser("eval")
    e.add_argument("k", type=int, help="the session whose chess_best.pt to evaluate")
    a = ap.parse_args()
    if a.what == "pack":
        for name in PACKS:
            print("wrote", build_pack(name))
    elif a.what == "probe":
        print("wrote", build_probe())
    elif a.what == "eval":
        print("wrote", build_eval(a.k))
    else:
        print("wrote", build_session(a.k, a.hours, a.pack))
    print(f"{EXAMPLES:,} examples = {MAX_STEPS:,} steps x {BATCH}, lr {LR} -> {MIN_LR}, "
          f"warmup {WARMUP:,}")


if __name__ == "__main__":
    main()
