"""Fused LayerNorm vs PyTorch eager vs torch.compile, at the model's real shapes.

    python -m src.kernels.bench_layernorm

Reports GB/s, since LayerNorm is memory-bound: the ceiling is the card's
bandwidth (320 GB/s on a T4), not its FLOPs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .layernorm import layer_norm


def _timed(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep          # ms per call


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA: nothing to benchmark")
        return
    N = 384                                        # the model's n_embd
    compiled = torch.compile(lambda x, w: F.layer_norm(x, (N,), w, None, 1e-5))

    print(f"{'rows':>8} {'mode':>10} {'fwd+bwd ms':>12} {'GB/s':>8} {'vs eager':>9}")
    for M in (1024 * 78 // 8, 1024 * 78 // 2, 1024 * 78):   # batch 128 / 512 / 1024
        x = torch.randn(M, N, device="cuda", requires_grad=True)
        w = torch.rand(N, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        # one read of x, dy and one write of dx, plus the forward's read+write
        gb = 6 * M * N * x.element_size() / 1e9

        def run(fn):
            x.grad = w.grad = None
            fn(x, w).backward(dy, retain_graph=False)

        modes = {
            "eager": lambda a, b: F.layer_norm(a, (N,), b, None, 1e-5),
            "compile": compiled,
            "triton": lambda a, b: layer_norm(a, b, None),
        }
        base = None
        for name, fn in modes.items():
            try:
                ms = _timed(lambda: run(fn))
            except Exception as exc:
                print(f"{M:>8} {name:>10} {'failed':>12}   {type(exc).__name__}: {str(exc)[:60]}")
                continue
            if name == "eager":
                base = ms
            speedup = f"{base / ms:.2f}x" if base else "-"
            print(f"{M:>8} {name:>10} {ms:>12.3f} {gb / (ms * 1e-3):>8.0f} {speedup:>9}", flush=True)


if __name__ == "__main__":
    main()
