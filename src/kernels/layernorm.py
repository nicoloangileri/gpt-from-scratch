"""Fused LayerNorm in Triton.

Why this op: profiling put the model in the launch-overhead regime (15M params,
78-token sequences, ~20% MFU on a T4), and LayerNorm is where that shows most --
eager PyTorch walks the row several times (mean, variance, normalise, affine),
each pass a separate kernel and a separate trip through memory. Fusing the whole
row into one kernel turns those passes into one load and one store.

The model has 17 LayerNorms (two per block, plus the final one), each with a
forward and a backward, so the launches add up.

Correctness is checked against F.layer_norm in tests/test_layernorm_triton.py,
including gradients; speed against both eager and torch.compile in
src/kernels/bench_layernorm.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:                                    # CPU-only machines (dev laptop)
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _layer_norm_fwd(
        X, Y, W, B, Mean, Rstd,
        stride, N, eps,
        HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row: mean, variance, normalise and affine in one pass."""
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride

        # Welford would be the numerically nicer choice, but a row is small enough
        # that the two-pass sum is both accurate in fp32 and faster.
        mean = 0.0
        _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            _sum += a
        mean = tl.sum(_sum, axis=0) / N

        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            a = tl.where(cols < N, a - mean, 0.0)
            _var += a * a
        rstd = 1 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)

        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w
            if HAS_BIAS:
                y += tl.load(B + cols, mask=mask)
            tl.store(Y + cols, y, mask=mask)

    @triton.jit
    def _layer_norm_bwd_dx(
        DX, DY, DW, DB, X, W, Mean, Rstd, Lock, Count,
        stride, N, GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """dx per row, plus a partial dw/db accumulated into one of GROUP_SIZE slots.

        Rows are far more numerous than features, so dw/db are reduced in two
        stages: each row adds into a slot it locks (cheap, GROUP_SIZE-way
        contention), and a second kernel sums the slots.
        """
        row = tl.program_id(0)
        X += row * stride
        DY += row * stride
        DX += row * stride
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        xhat = tl.where(mask, (x - mean) * rstd, 0.0)
        wdy = tl.where(mask, w * dy, 0.0)
        # d/dx of a normalised row: remove the components along the row mean and
        # along xhat, which is what keeps the row zero-mean and unit-variance.
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        tl.store(DX + cols, (wdy - (xhat * c1 + c2)) * rstd, mask=mask)

        slot = row % GROUP_SIZE
        lock = Lock + slot
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
        count_ptr = Count + slot
        dw_ptrs = DW + slot * N + cols
        db_ptrs = DB + slot * N + cols
        if tl.load(count_ptr) == 0:                    # first writer initialises
            tl.atomic_xchg(count_ptr, 1)
            partial_dw = xhat * dy
            partial_db = dy
        else:
            partial_dw = xhat * dy + tl.load(dw_ptrs, mask=mask, other=0.0)
            partial_db = dy + tl.load(db_ptrs, mask=mask, other=0.0)
        tl.store(dw_ptrs, partial_dw, mask=mask)
        if HAS_BIAS:
            tl.store(db_ptrs, partial_db, mask=mask)
        tl.debug_barrier()
        tl.atomic_xchg(lock, 0)

    @triton.jit
    def _layer_norm_bwd_dwdb(
        PartialDW, PartialDB, FinalDW, FinalDB, M, N,
        HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Stage two: sum the GROUP_SIZE partial rows into the real gradients."""
        pid = tl.program_id(0)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        dw = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(0, M, BLOCK_M):
            rows = i + tl.arange(0, BLOCK_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(PartialDW + offs, mask=mask, other=0.0)
            if HAS_BIAS:
                db += tl.load(PartialDB + offs, mask=mask, other=0.0)
        tl.store(FinalDW + cols, tl.sum(dw, axis=0), mask=cols < N)
        if HAS_BIAS:
            tl.store(FinalDB + cols, tl.sum(db, axis=0), mask=cols < N)


class _FusedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        M, N = x.shape
        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)
        num_warps = max(min(BLOCK_SIZE // 256, 8), 4)
        _layer_norm_fwd[(M,)](
            x, y, weight, bias if bias is not None else weight, mean, rstd,
            x.stride(0), N, eps,
            HAS_BIAS=bias is not None, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE, ctx.num_warps, ctx.eps = BLOCK_SIZE, num_warps, eps
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        shape = dy.shape
        dy = dy.reshape(-1, shape[-1]).contiguous()
        M, N = dy.shape
        GROUP_SIZE = 256 if N <= 8192 else 96

        locks = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)
        counts = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)
        partial_dw = torch.zeros((GROUP_SIZE, N), dtype=torch.float32, device=x.device)
        partial_db = torch.zeros((GROUP_SIZE, N), dtype=torch.float32, device=x.device)
        dx = torch.empty_like(dy)

        _layer_norm_bwd_dx[(M,)](
            dx, dy, partial_dw, partial_db, x, weight, mean, rstd, locks, counts,
            x.stride(0), N, GROUP_SIZE=GROUP_SIZE, HAS_BIAS=bias is not None,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
        )
        dw = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        db = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        _layer_norm_bwd_dwdb[grid](
            partial_dw, partial_db, dw, db, GROUP_SIZE, N,
            HAS_BIAS=bias is not None, BLOCK_M=32, BLOCK_N=128,
        )
        return dx.reshape(shape), dw, db if bias is not None else None, None


def layer_norm(x, weight, bias, eps: float = 1e-5):
    """Fused LayerNorm; falls back to PyTorch off-CUDA or without Triton."""
    if not HAS_TRITON or not x.is_cuda:
        return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
    return _FusedLayerNorm.apply(x, weight, bias, eps)


class TritonLayerNorm(nn.Module):
    """Drop-in replacement for the model's LayerNorm."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias)
