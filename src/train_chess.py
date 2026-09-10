"""Training loop for the chess action-value model.

    python -m src.train_chess --config configs/chess.yaml

Same skeleton as src/train.py (config + CLI overrides, cosine LR, AMP with the
bf16->fp16 fallback, checkpointing), but the task is classification: given a
position and a candidate move, predict which bucket the Stockfish win
probability falls into. One label per sequence instead of next-token prediction.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .chess_data import ACTION_OFFSET, FULL_VOCAB, SAMPLE_LENGTH, SEQUENCE_LENGTH
from .model import GPT, GPTConfig
from .train import get_lr, _nullctx


class ActionValueStream:
    """Memory-mapped (state, action, bucket) triples -> batched tensors."""

    def __init__(self, data_dir: str | Path, split: str, max_ram_gb: float = 4.0) -> None:
        d = Path(data_dir)
        self.states = np.memmap(d / f"{split}_states.bin", dtype=np.uint8, mode="r")
        self.states = self.states.reshape(-1, SEQUENCE_LENGTH)
        self.actions = np.memmap(d / f"{split}_actions.bin", dtype=np.uint16, mode="r")
        self.buckets = np.memmap(d / f"{split}_buckets.bin", dtype=np.uint8, mode="r")
        n = min(len(self.states), len(self.actions), len(self.buckets))
        self.n = n
        # Random access into a memmap costs one page fault per sample; holding the
        # arrays in RAM is ~13x faster and a packed shard is under a GB.
        if self.states.nbytes / 1e9 <= max_ram_gb:
            self.states = np.asarray(self.states)
            self.actions = np.asarray(self.actions)
            self.buckets = np.asarray(self.buckets)
            self.in_ram = True
        else:
            self.in_ram = False

    def batch(self, batch_size: int, device: str, generator=None):
        ix = torch.randint(self.n, (batch_size,), generator=generator).numpy()
        states = torch.from_numpy(self.states[ix].astype(np.int64))
        actions = torch.from_numpy(self.actions[ix].astype(np.int64)) + ACTION_OFFSET
        x = torch.cat([states, actions[:, None]], dim=1)
        y = torch.from_numpy(self.buckets[ix].astype(np.int64))
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


@torch.no_grad()
def evaluate(model, streams, cfg, ctx) -> dict:
    out = {}
    model.eval()
    for split, stream in streams.items():
        losses = torch.zeros(cfg["eval_iters"])
        correct = total = 0
        for k in range(cfg["eval_iters"]):
            x, y = stream.batch(cfg["batch_size"], cfg["device"])
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
        out[split] = losses.mean().item()
        out[f"{split}_acc"] = correct / total
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chess.yaml")
    ap.add_argument("--data_dir", default=None)
    args, overrides = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    for i in range(0, len(overrides) - 1, 2):
        key = overrides[i].lstrip("-")
        if key in cfg:
            cur, raw = cfg[key], overrides[i + 1]
            cfg[key] = (raw.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(raw)

    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = cfg["device"]
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

    use_amp = cfg.get("amp", True) and device_type == "cuda"
    if use_amp and cfg.get("dtype", "bf16") == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        print(f"[amp] bf16 unsupported on {torch.cuda.get_device_name()} -> using fp16")
        cfg["dtype"] = "fp16"
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("dtype", "bf16"))
    ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp else _nullctx()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.get("dtype") == "fp16")

    streams = {s: ActionValueStream(cfg["data_dir"], s, cfg.get("max_ram_gb", 4.0))
               for s in ("train", "val")}
    print(f"[data] train {streams['train'].n:,} | val {streams['val'].n:,} samples "
          f"| in RAM: {streams['train'].in_ram}")

    model_cfg = GPTConfig(
        vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH, num_classes=cfg["num_buckets"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg["dropout"], bias=cfg["bias"], causal=cfg.get("causal", True),
        triton_ln=cfg.get("triton_ln", False),
    )
    model = GPT(model_cfg).to(device)
    print(f"[model] {model.num_params()/1e6:.2f}M non-embedding "
          f"({sum(p.numel() for p in model.parameters())/1e6:.2f}M total) | {asdict(model_cfg)}")
    optimizer = model.configure_optimizers(cfg["weight_decay"], cfg["lr"], (cfg["beta1"], cfg["beta2"]))
    if cfg.get("compile", False):
        model = torch.compile(model)

    accum = cfg.get("grad_accum_steps", 1)
    # MFU reference: T4 peak is 65 TFLOP/s fp16, A100 312 TFLOP/s bf16.
    peak_flops = cfg.get("peak_tflops", 65.0) * 1e12
    flops_per_sample = model.flops_per_sample(SAMPLE_LENGTH)
    ckpt_dir = Path(cfg.get("out_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = t_log = time.time()
    s_log = 0

    for step in range(cfg["max_steps"] + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, cfg)

        if step % cfg["eval_interval"] == 0:
            m = evaluate(model, streams, cfg, ctx)
            print(f"step {step:6d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| val acc {m['val_acc']*100:.1f}% | {time.time()-t0:.0f}s", flush=True)
            if m["val"] < best_val and step > 0:
                best_val = m["val"]
                torch.save({"model": model.state_dict(), "cfg": asdict(model_cfg),
                            "step": step, "val": best_val, "val_acc": m["val_acc"],
                            "num_buckets": cfg["num_buckets"]}, ckpt_dir / "chess_best.pt")
            t_log, s_log = time.time(), step

        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = streams["train"].batch(cfg["batch_size"], device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            scaler.scale(loss).backward()
        if cfg.get("grad_clip", 0):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["log_interval"] == 0 and step > 0:
            if device_type == "cuda":
                # CUDA is async: without this the clock measures how long the CPU
                # took to QUEUE the step, not to run it (one A/B run reported 79k
                # samples/s and an MFU of 882%).
                torch.cuda.synchronize()
            now = time.time()
            samples = cfg["batch_size"] * accum * (step + 1 - s_log)
            sps = samples / max(1e-6, now - t_log)
            mfu = sps * flops_per_sample / peak_flops
            print(f"  step {step} | loss {loss.item()*accum:.4f} | lr {get_lr(step, cfg):.2e} "
                  f"| {sps:,.0f} samples/s | MFU {mfu*100:.1f}%", flush=True)
            t_log, s_log = now, step + 1

    print(f"[done] best val loss {best_val:.4f} -> {ckpt_dir/'chess_best.pt'}")


if __name__ == "__main__":
    main()
