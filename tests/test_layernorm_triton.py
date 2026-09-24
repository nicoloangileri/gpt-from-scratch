"""Correctness of the fused LayerNorm against PyTorch, forward and backward.

Skipped without a GPU: Triton needs CUDA, so on the dev laptop these run only as
the fallback path (covered by test_fallback_matches_pytorch).

Nothing runs the CUDA tests automatically -- not the Mac, not the CPU-only CI. They
last ran in full on a Kaggle T4 on 10 Sept 2026, 13 passed
(results/07_triton_layernorm.log), and the kernel's code has not changed since:
the only later edit to src/kernels/layernorm.py is its docstring. Any change to the
kernel itself must be re-run on a GPU before it is trusted.
"""

import pytest
import torch
import torch.nn.functional as F

from src.kernels.layernorm import HAS_TRITON, TritonLayerNorm, layer_norm

needs_cuda = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="needs CUDA + Triton"
)


def _reference(x, w, b):
    return F.layer_norm(x, w.shape, w, b, 1e-5)


def test_fallback_matches_pytorch():
    """Without CUDA the wrapper must return exactly F.layer_norm."""
    x = torch.randn(8, 32)
    w, b = torch.ones(32), torch.zeros(32)
    assert torch.equal(layer_norm(x, w, b), _reference(x, w, b))


def test_module_shapes_and_params():
    m = TritonLayerNorm(384, bias=False)
    assert m.bias is None and m.weight.shape == (384,)
    assert m(torch.randn(2, 78, 384)).shape == (2, 78, 384)


@needs_cuda
@pytest.mark.parametrize("M,N", [(1151, 384), (4096, 384), (128, 768), (77, 1024)])
@pytest.mark.parametrize("bias", [True, False])
def test_forward_matches_pytorch(M, N, bias):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w = torch.rand(N, device="cuda")
    b = torch.rand(N, device="cuda") if bias else None
    torch.testing.assert_close(layer_norm(x, w, b), _reference(x, w, b), atol=1e-5, rtol=1e-5)


@needs_cuda
@pytest.mark.parametrize("bias", [True, False])
def test_backward_matches_pytorch(bias):
    M, N = 2048, 384
    x = torch.randn(M, N, device="cuda", requires_grad=True)
    w = torch.rand(N, device="cuda", requires_grad=True)
    b = torch.rand(N, device="cuda", requires_grad=True) if bias else None
    dy = torch.randn_like(x)

    ref_x = x.detach().clone().requires_grad_()
    ref_w = w.detach().clone().requires_grad_()
    ref_b = b.detach().clone().requires_grad_() if bias else None

    layer_norm(x, w, b).backward(dy)
    _reference(ref_x, ref_w, ref_b).backward(dy)

    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-4, rtol=1e-4)
    # dw/db go through a two-stage reduction, so they accumulate a little more error
    torch.testing.assert_close(w.grad, ref_w.grad, atol=1e-2, rtol=1e-2)
    if bias:
        torch.testing.assert_close(b.grad, ref_b.grad, atol=1e-2, rtol=1e-2)


@needs_cuda
def test_survives_a_non_contiguous_input():
    x = torch.randn(64, 128, 384, device="cuda").transpose(0, 1)
    w = torch.rand(384, device="cuda")
    torch.testing.assert_close(layer_norm(x, w, None), _reference(x, w, None),
                               atol=1e-5, rtol=1e-5)
