"""Throughput bench for the chess model: measure before optimising.

    python -m src.bench_chess --data_dir data/chess --steps 300

Runs the real training step (forward, backward, optimizer) under a handful of
configurations and reports samples/s and MFU for each, so the choice of batch
size and whether torch.compile pays for itself are measured rather than guessed.

Each configuration runs in a fresh subprocess: sharing one process lets the first
config leave the allocator fragmented, which made a bigger batch look slower than
a smaller one and OOM'd on an allocation the card had room for. The bench also
splits data time from compute time, since a synchronous loader stalls the GPU and
that stall grows with the batch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import torch

from .chess_data import FULL_VOCAB, SAMPLE_LENGTH
from .model import GPT, GPTConfig
from .train_chess import ActionValueStream


def run_config(stream, device, batch_size, compile_model, steps, peak_flops, warmup=20):
    """Time one configuration; returns (samples/s, MFU, share of time in the loader)."""
    cfg = GPTConfig(vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH, num_classes=64,
                    n_layer=8, n_head=8, n_embd=384, dropout=0.0, bias=False)
    model = GPT(cfg).to(device)
    flops_per_sample = model.flops_per_sample(SAMPLE_LENGTH)
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95))
    if compile_model:
        model = torch.compile(model)
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    scaler = torch.amp.GradScaler("cuda")

    # Loader cost is timed on its own: measuring it inside the training loop needs a
    # synchronize, which then charges the previous step's async GPU work to the
    # loader (that is how an earlier version got an absurd "97% in loader").
    torch.cuda.synchronize()
    t_data = time.time()
    for _ in range(20):
        stream.batch(batch_size, device)
    torch.cuda.synchronize()
    data_per_step = (time.time() - t_data) / 20

    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        x, y = stream.batch(batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    dt = time.time() - t0
    sps = steps * batch_size / dt
    return sps, sps * flops_per_sample / peak_flops, data_per_step / (dt / steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/chess")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--peak_tflops", type=float, default=65.0)
    ap.add_argument("--single", default=None, help="internal: '<batch>,<compile>'")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    peak = args.peak_tflops * 1e12

    if args.single is not None:                       # one config, fresh process
        batch_size, compiled = args.single.split(",")
        stream = ActionValueStream(args.data_dir, "train")
        try:
            sps, mfu, data_share = run_config(
                stream, "cuda", int(batch_size), compiled == "True", args.steps, peak)
            print(f"RESULT {batch_size} {compiled} {sps:.0f} {mfu*100:.2f} {data_share*100:.1f}")
        except Exception as exc:
            print(f"RESULT {batch_size} {compiled} failed 0 0  {type(exc).__name__}: {str(exc)[:80]}")
        return

    configs = [(256, False), (512, False), (1024, False), (2048, False),
               (512, True), (1024, True)]
    print(f"{'batch':>6} {'compile':>8} {'samples/s':>12} {'MFU':>7} {'in loader':>10}")
    results = []
    for batch_size, compiled in configs:
        proc = subprocess.run(
            [sys.executable, "-m", "src.bench_chess", "--data_dir", args.data_dir,
             "--steps", str(args.steps), "--single", f"{batch_size},{compiled}"],
            capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        if line is None:
            print(f"{batch_size:>6} {str(compiled):>8} {'no result':>12}", flush=True)
            continue
        _, b, c, sps, mfu, *rest = line.split()
        if sps == "failed":
            print(f"{batch_size:>6} {str(compiled):>8} {'failed':>12}   {' '.join(rest[1:])}", flush=True)
            continue
        results.append((int(b), c == "True", float(sps), float(mfu)))
        print(f"{batch_size:>6} {str(compiled):>8} {float(sps):>12,.0f} {float(mfu):>6.1f}% "
              f"{float(rest[0]):>9.1f}%", flush=True)

    if results:
        best = max(results, key=lambda r: r[2])
        print(f"\nbest: batch {best[0]}, compile={best[1]} -> {best[2]:,.0f} samples/s, "
              f"MFU {best[3]:.1f}%")


if __name__ == "__main__":
    main()
