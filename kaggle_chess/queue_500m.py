"""Run the 500M sessions one after the other, sized to the GPU quota that is left.

    nohup caffeinate -i .venv/bin/python kaggle_chess/queue_500m.py \
        --start "2026-10-01 00:05" > /dev/null 2>&1 &            # progress in kaggle_500m.log

The two pack kernels (CPU) must have finished first; the queue checks their status and
refuses to start without them. Then, until training is done:

1. read `kaggle quota`; with less than MIN_SESSION_H left, sleep until the weekly reset
   (the quota command gives its time) -- a session is never started longer than the
   quota it can use, since what a session killed at zero quota saves is not
   documented anywhere;
2. build session k for min(11.25 h, quota left - margin), on the pack consumed least
   so far relative to its size, and push it;
3. wait for it, fetch progress.json, and check that it arrived (the fetch that sat for
   18 hours and came back empty is why every fetch is checked);
4. stop at the first session that is not COMPLETE: nothing is retried blindly, since
   every attempt costs quota.

When training is done the best checkpoint is fetched and copied to
checkpoints/chess_500M.pt (never over an existing file); if the last session had no
time left to evaluate it, an eval-only kernel does. The laptop only has to be awake
at the hand-overs between sessions; the sessions themselves run on Kaggle.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_500m as b  # noqa: E402

K = str(ROOT / ".venv" / "bin" / "kaggle")
PY = str(ROOT / ".venv" / "bin" / "python")
LOG = ROOT / "kaggle_500m.log"
STATE = ROOT / "kaggle_500m_state.json"
OUT = ROOT / "kaggle_500m_out"
MIN_SESSION_H = 1.5          # below this a session is mostly start-up and data loading
MARGIN_H = 0.4               # start-up, output saving, and the quota's own rounding
POLL_S = 600


def say(msg: str) -> None:
    with open(LOG, "a") as fh:
        fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def kaggle(*args: str, timeout_s: int = 1800) -> str:
    """One CLI call, never longer than `timeout_s`. A download once sat for 9 hours on
    a 0-byte file (the laptop slept that night); the file check caught it, but
    every retry was behind that one hung call."""
    try:
        r = subprocess.run([K, *args], capture_output=True, text=True, cwd=ROOT,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout_s}s"
    return (r.stdout + r.stderr).strip()


def quota() -> tuple[float, dt.datetime]:
    """GPU hours left and when they reset (UTC), from `kaggle quota`."""
    for line in kaggle("quota").splitlines():
        cols = line.split()
        if cols and cols[0] == "GPU":
            left = float(cols[2].rstrip("h"))
            reset = dt.datetime.fromisoformat(cols[4]).replace(tzinfo=dt.timezone.utc)
            return left, reset
    raise RuntimeError("no GPU line in `kaggle quota`")


def status(kernel: str) -> str:
    return kaggle("kernels", "status", kernel).splitlines()[-1]


def session_state(k: int) -> str:
    """absent, running, complete or failed. A kernel that was never pushed answers
    "Cannot access kernel ...", with no status line."""
    s = status(b.session_id(k))
    if "has status" not in s:
        return "absent"
    if "COMPLETE" in s:
        return "complete"
    if "ERROR" in s or "CANCEL" in s:
        return "failed"
    return "running"


def wait_done(kernel: str) -> str:
    while True:
        s = status(kernel)
        if any(w in s for w in ("COMPLETE", "ERROR", "CANCEL")):
            say(f"{kernel}: {s}")
            return s
        time.sleep(POLL_S)


def fetch(kernel: str, folder: Path, pattern: str, must: str) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 7):
        kaggle("kernels", "output", kernel, "-p", str(folder), "--file-pattern", pattern, "-q")
        if (folder / must).is_file() and (folder / must).stat().st_size > 0:
            return True
        say(f"{kernel}: {folder.name}/{must} not there after try {attempt}, retrying in 10 min")
        time.sleep(600)
    say(f"{kernel}: {must} NOT fetched -- fetch it by hand")
    return False


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"sessions": [], "consumed": {name: 0 for name in b.PACKS}}


def pick_pack(state: dict) -> str:
    """The pack whose records have been drawn the least, relative to its size."""
    size = {name: sum(b.SHARD_RECORDS[s] for s in shards) for name, shards in b.PACKS.items()}
    return min(b.PACKS, key=lambda name: (state["consumed"][name] / size[name], name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="local time to begin, e.g. '2026-10-01 00:05'")
    a = ap.parse_args()
    say("queue started")
    for name in b.PACKS:
        s = status(b.pack_id(name))
        if "COMPLETE" not in s:
            say(f"pack {name} is not complete ({s}): run the pack kernels first")
            sys.exit(1)
    if a.start:
        start = dt.datetime.strptime(a.start, "%Y-%m-%d %H:%M")
        while dt.datetime.now() < start:
            time.sleep(min(POLL_S, max(1.0, (start - dt.datetime.now()).total_seconds())))

    state = load_state()
    while True:
        k = len(state["sessions"]) + 1
        left, reset = quota()
        if left < MIN_SESSION_H and session_state(k) == "absent":
            wake = reset + dt.timedelta(minutes=10)
            say(f"{left:.2f} h of GPU left: waiting for the reset at {wake:%Y-%m-%d %H:%M} UTC")
            while dt.datetime.now(dt.timezone.utc) < wake:
                time.sleep(POLL_S)
            continue
        hours = round(min(b.SESSION_HOURS, left - MARGIN_H), 2)
        pack = pick_pack(state)
        # A kernel id already in use is never pushed again: a push REPLACES that
        # session's output, and the next session resumes from it. That covers a queue
        # restarted after a crash (the session it launched is running or done) and a
        # session that failed (its last.pt may hold hours of progress: look first).
        found = session_state(k)
        if found == "failed":
            say(f"session {k} exists and failed: inspect it; it is never replaced automatically")
            sys.exit(1)
        if found in ("running", "complete"):
            say(f"session {k} is already {found} (queue restarted?): waiting for it, not pushing")
            pack = json.loads((ROOT / f"kaggle_chess_500m_s{k}" / "kernel-metadata.json")
                              .read_text())["kernel_sources"][0].rsplit("-", 1)[-1]
        else:
            built = subprocess.run([PY, "kaggle_chess/build_500m.py", "session", str(k),
                                    "--hours", str(hours), "--pack", pack],
                                   capture_output=True, text=True, cwd=ROOT)
            if built.returncode:
                say(f"session {k}: build failed: {built.stderr.strip()[-300:]}")
                sys.exit(1)
            pushed = kaggle("kernels", "push", "-p", f"kaggle_chess_500m_s{k}")
            if "successfully pushed" not in pushed:
                say(f"session {k}: push refused: {pushed.splitlines()[-1] if pushed else '?'}")
                sys.exit(1)
            say(f"session {k}: pushed, {hours} h on pack {pack} ({left:.2f} h of quota left)")
        result = wait_done(b.session_id(k))
        if "COMPLETE" not in result:
            say(f"session {k} did not complete: stopping here, nothing retried")
            sys.exit(1)
        folder = OUT / f"s{k}"
        if not fetch(b.session_id(k), folder, r"(progress|results)\.json$|\.log$", "progress.json"):
            sys.exit(1)
        progress = json.loads((folder / "progress.json").read_text())
        before = state["sessions"][-1]["next_step"] if state["sessions"] else 0
        state["consumed"][pack] += (min(progress["next_step"], b.MAX_STEPS + 1) - before) * b.BATCH
        state["sessions"].append({"k": k, "pack": pack, "hours": hours, **progress})
        STATE.write_text(json.dumps(state, indent=1) + "\n")
        say(f"session {k}: step {progress['next_step']:,} of {b.MAX_STEPS:,}, "
            f"best val {progress['best_val']:.4f}")
        if progress["done"]:
            break

    final = OUT / "final"
    if not fetch(b.session_id(k), final, r"chess_best\.pt$", "checkpoints/chess_best.pt"):
        sys.exit(1)
    dest = ROOT / "checkpoints" / "chess_500M.pt"
    if dest.exists():
        say(f"{dest} exists already: left alone, the new one is in {final}")
    else:
        shutil.copy(final / "checkpoints" / "chess_best.pt", dest)
        say(f"saved {dest}")
    if not (folder / "results.json").exists():
        subprocess.run([PY, "kaggle_chess/build_500m.py", "eval", str(k)], cwd=ROOT, check=True)
        while quota()[0] < 1.2:
            time.sleep(POLL_S)
        say("eval: " + kaggle("kernels", "push", "-p", "kaggle_chess_500m_eval").splitlines()[-1])
        if "COMPLETE" in wait_done(f"{b.USER}/gpt-from-scratch-chess-500m-eval"):
            fetch(f"{b.USER}/gpt-from-scratch-chess-500m-eval", OUT / "eval",
                  r"results\.json$|\.log$", "results.json")
    say("queue done")


if __name__ == "__main__":
    main()
