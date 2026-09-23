"""A decoder-only Transformer (GPT), written from scratch in PyTorch.

Every block is hand-written on purpose — no `from transformers import ...` — so the
attention, MLP, normalization, and positional handling are all explicit and
readable. This is the Level-1 artifact: it demonstrates understanding of the core
object of modern ML, and it runs.

Level-2/3 extension points are marked with `# LEVEL 2:` / `# LEVEL 3:` comments
(mixed precision, KV-cache, a fused Triton/CUDA kernel) — see README.md.

Credit where it is due: the structure follows Andrej Karpathy's nanoGPT (MIT) --
the config dataclass, the fused `c_attn` projection with the `flash` flag, the
`c_fc`/`c_proj` MLP, a LayerNorm class that exists so bias can be turned off, and
the weight-decay split in `configure_optimizers`. The code here is written out
rather than copied, and the chess head, the classification path and the Triton
norm are new, but the skeleton is that one and saying otherwise would be a lie.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 BPE (tiktoken "gpt2") vocab
    block_size: int = 256       # max context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384           # must be divisible by n_head
    dropout: float = 0.0
    bias: bool = False          # bias in Linear/LayerNorm — False is a touch faster/better
    triton_ln: bool = False     # swap LayerNorm for the fused Triton kernel
    causal: bool = True         # False = full attention. A chess position has no
                                # temporal order, so masking is only meaningful for
                                # the language-model task.
    num_classes: int | None = None  # set for classification: one label per sequence,
                                    # read off the last position (chess value buckets).
                                    # None keeps the autoregressive language-model head.


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Q, K, V are produced by one fused linear projection, split into heads, then a
    scaled dot-product with a causal mask so position t only attends to <= t.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        # one projection for all of q, k, v (3 * n_embd out), then the output proj.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # PyTorch's fused SDPA (flash attention when available) — the Level-3 Triton
        # kernel replaces THIS call and is benchmarked against it.
        self.causal = cfg.causal
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # explicit causal mask fallback for older torch
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            if self.causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nn.LayerNorm always has bias)."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def _norm(cfg: GPTConfig) -> nn.Module:
    """LayerNorm, optionally the hand-written fused kernel."""
    if cfg.triton_ln:
        from .kernels import TritonLayerNorm
        return TritonLayerNorm(cfg.n_embd, cfg.bias)
    return LayerNorm(cfg.n_embd, cfg.bias)


class Block(nn.Module):
    """Pre-norm Transformer block: x = x + attn(ln(x)); x = x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = _norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = _norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # learned positional emb
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=_norm(cfg),
        ))
        if cfg.num_classes is None:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            # weight tying: input embedding and output projection share weights.
            self.transformer.wte.weight = self.lm_head.weight
        else:
            # Sequence classifier: no tying (input and output spaces differ).
            self.lm_head = nn.Linear(cfg.n_embd, cfg.num_classes, bias=False)
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            if self.cfg.num_classes is not None:
                n -= self.transformer.wte.weight.numel()  # untied here, so it counts
        return n

    def flops_per_sample(self, seq_len: int | None = None) -> float:
        """Forward+backward FLOPs for one sequence (PaLM appendix B estimate)."""
        T = seq_len or self.cfg.block_size
        n = self.num_params(non_embedding=False)
        # 6ND for the dense layers, plus the attention score/value matmuls
        return 6 * n * T + 12 * self.cfg.n_layer * self.cfg.n_embd * T * T

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.cfg.num_classes is not None:
            # one prediction per sequence, taken from the final position
            logits = self.lm_head(x[:, -1, :])
            if targets is None:
                return logits, None
            return logits, F.cross_entropy(logits, targets.view(-1))

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        # inference: only need logits for the last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. LEVEL 2: add a KV-cache so each step is O(1) in
        context length instead of re-encoding the whole prefix every step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay: float, lr: float, betas: tuple[float, float]):
        """AdamW with decay on 2D+ params (matmuls/embeddings) only — not on
        biases/LayerNorm gains. Standard, and a detail interviewers look for."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # The fused kernel folds the whole optimizer step into one launch, which
        # matters here: the model is small, so per-kernel overhead is a real share
        # of step time.
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and any(
            p.is_cuda for p in decay
        )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))
