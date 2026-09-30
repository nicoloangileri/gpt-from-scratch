"""Training loop for the chess action-value model.

    python -m src.train_chess --config configs/chess.yaml

Same skeleton as src/train.py (config + CLI overrides, cosine LR, AMP with the
bf16->fp16 fallback, checkpointing), but the task is classification: given a
position and a candidate move, predict which bucket the Stockfish win
probability falls into. One label per sequence instead of next-token prediction.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .chess_data import ACTION_OFFSET, FULL_VOCAB, SAMPLE_LENGTH, SEQUENCE_LENGTH
from .model import GPT, GPTConfig
from .train import get_lr, _nullctx


def _parts(data_dir: Path, split: str) -> list[Path]:
    """Folders holding a packed `split`: the data dir itself, or any folder below it.

    One pack per shard (the 500M run packs each shard in its own folder, in parallel)
    keeps packing embarrassingly parallel; empty files, which a packer leaves for a
    split it had nothing for, are not parts.
    """
    found = sorted(p.parent for p in data_dir.glob(f"**/{split}_states.bin")
                   if p.stat().st_size > 0)
    if not found:
        raise FileNotFoundError(f"no {split}_states.bin under {data_dir}")
    return found


class ActionValueStream:
    """Memory-mapped (state, action, bucket) triples -> batched tensors.

    With `world` > 1 (data-parallel training), rank r keeps only its slice of every
    part: rows [n*r/world, n*(r+1)/world). The slices are disjoint and cover the data,
    so each GPU holds 1/world of it in RAM instead of a full copy, and a global batch
    made of one local batch per rank is still a uniform sample of the whole.
    """

    def __init__(self, data_dir: str | Path, split: str, max_ram_gb: float = 4.0,
                 rank: int = 0, world: int = 1) -> None:
        parts = _parts(Path(data_dir), split)
        if len(parts) == 1 and world == 1:
            d = parts[0]
            self.states = np.memmap(d / f"{split}_states.bin", dtype=np.uint8, mode="r")
            self.states = self.states.reshape(-1, SEQUENCE_LENGTH)
            self.actions = np.memmap(d / f"{split}_actions.bin", dtype=np.uint16, mode="r")
            self.buckets = np.memmap(d / f"{split}_buckets.bin", dtype=np.uint8, mode="r")
            self.n = min(len(self.states), len(self.actions), len(self.buckets))
            # Random access into a memmap costs one page fault per sample; holding the
            # arrays in RAM is ~13x faster and a packed shard is under a GB.
            if self.states.nbytes / 1e9 <= max_ram_gb:
                self.states = np.asarray(self.states)
                self.actions = np.asarray(self.actions)
                self.buckets = np.asarray(self.buckets)
                self.in_ram = True
            else:
                self.in_ram = False
            return
        # Several parts, or a slice per rank: always in RAM. Sampling across many
        # memmaps would need an index of indices, and the slices are sized to fit.
        slices = []
        for d in parts:
            n = min((d / f"{split}_states.bin").stat().st_size // SEQUENCE_LENGTH,
                    (d / f"{split}_actions.bin").stat().st_size // 2,
                    (d / f"{split}_buckets.bin").stat().st_size)
            lo, hi = n * rank // world, n * (rank + 1) // world
            slices.append((d, lo, hi))
        need_gb = sum(hi - lo for _, lo, hi in slices) * (SEQUENCE_LENGTH + 3) / 1e9
        if need_gb > max_ram_gb:
            raise MemoryError(f"{split}: {need_gb:.1f} GB for this rank, max_ram_gb is "
                              f"{max_ram_gb}: pack fewer shards or use more ranks")

        total = sum(hi - lo for _, lo, hi in slices)

        # Read each slice straight into one preallocated array: concatenating a list
        # of arrays would hold everything twice at the peak, per rank.
        def gather(name, dtype, width=1):
            out = np.empty(total * width, dtype=dtype)
            size, pos = np.dtype(dtype).itemsize, 0
            for d, lo, hi in slices:
                count = (hi - lo) * width
                view = memoryview(out[pos:pos + count]).cast("B")
                with open(d / f"{split}_{name}.bin", "rb") as fh:
                    fh.seek(lo * width * size)
                    done = 0
                    while done < len(view):          # a single read() stops near 2 GB
                        got = fh.readinto(view[done:done + (1 << 28)])
                        if not got:
                            raise IOError(f"short read in {d / f'{split}_{name}.bin'}")
                        done += got
                pos += count
            return out
        self.states = gather("states", np.uint8, SEQUENCE_LENGTH).reshape(-1, SEQUENCE_LENGTH)
        self.actions = gather("actions", np.uint16)
        self.buckets = gather("buckets", np.uint8)
        self.n = total
        self.in_ram = True

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


class PuzzleRankStream:
    """Puzzle positions with their correct move and the legal moves it must beat.

    Same packing idea as ActionValueStream, but a sample is a whole SET of moves for
    one position, so a batch of B positions becomes B * num_candidates sequences.
    """

    def __init__(self, data_dir: str | Path, split: str, max_ram_gb: float = 2.0) -> None:
        d = Path(data_dir)
        self.states = np.memmap(d / f"{split}_states.bin", dtype=np.uint8, mode="r")
        self.states = self.states.reshape(-1, SEQUENCE_LENGTH)
        self.ncand = np.memmap(d / f"{split}_ncand.bin", dtype=np.uint8, mode="r")
        self.cands = np.memmap(d / f"{split}_cands.bin", dtype=np.uint16, mode="r")
        self.k = self.cands.size // len(self.ncand)
        self.cands = self.cands.reshape(-1, self.k)
        self.n = min(len(self.states), len(self.ncand), len(self.cands))
        if self.states.nbytes / 1e9 <= max_ram_gb:
            self.states, self.cands = np.asarray(self.states), np.asarray(self.cands)
            self.ncand = np.asarray(self.ncand)
            self.in_ram = True
        else:
            self.in_ram = False

    def batch(self, batch_size: int, device: str, generator=None):
        """-> x (B*k, 78) and mask (B, k), the correct move always at column 0."""
        ix = torch.randint(self.n, (batch_size,), generator=generator).numpy()
        states = torch.from_numpy(self.states[ix].astype(np.int64))
        cands = torch.from_numpy(self.cands[ix].astype(np.int64)) + ACTION_OFFSET
        ncand = torch.from_numpy(self.ncand[ix].astype(np.int64))
        # one sequence per candidate move: the position repeated, each with its action
        x = torch.cat([states.repeat_interleave(self.k, 0),
                       cands.reshape(-1, 1)], dim=1)
        mask = torch.arange(self.k)[None, :] < ncand[:, None]
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)
        else:
            x, mask = x.to(device), mask.to(device)
        return x, mask


def listwise_loss(values: torch.Tensor, mask: torch.Tensor, temp: float) -> torch.Tensor:
    """Cross-entropy over the legal moves of a position, correct move at index 0.

    `values` are expected win probabilities in [0, 1], so the spread between a good
    and a bad move is a few hundredths; fed to a softmax as they are, every move looks
    equally likely and there is almost no gradient. `temp` sets how much of a win
    probability difference counts as a preference -- 0.05 makes a 5-point difference
    one logit.
    """
    logits = (values / temp).masked_fill(~mask, float("-inf"))
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def rank_step(model, centres, stream, cfg, ctx, device):
    """One puzzle batch -> (loss, top-1 accuracy over the candidate moves)."""
    x, mask = stream.batch(cfg["rank_batch"], device)
    with ctx:
        logits, _ = model(x)
        values = (F.softmax(logits.float(), dim=-1) @ centres).view(len(mask), -1)
        loss = listwise_loss(values, mask, cfg["rank_temp"])
    picked = values.masked_fill(~mask, float("-inf")).argmax(-1)
    return loss, (picked == 0).float().mean().item()


@torch.no_grad()
def evaluate(model, streams, cfg, ctx, rank_streams=None, centres=None) -> dict:
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
    if rank_streams:
        # "Picks the puzzle move out of its legal alternatives" -- the quantity the
        # puzzle data is here to improve, measured on puzzles never trained on.
        for split, stream in rank_streams.items():
            hits = 0.0
            for _ in range(cfg["eval_iters"]):
                _, acc = rank_step(model, centres, stream, cfg, ctx, cfg["device"])
                hits += acc
            out[f"rank_{split}_acc"] = hits / cfg["eval_iters"]
    model.train()
    return out


DEADLINE_CHECK = 50                           # steps between deadline votes across ranks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chess.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--resume", default=None, help="last.pt from an earlier session")
    ap.add_argument("--init_from", default=None,
                    help="start from these weights with a fresh schedule (warm restart)")
    ap.add_argument("--deadline_s", type=float, default=None,
                    help="stop cleanly and save last.pt after this many seconds")
    ap.add_argument("--stop_at_step", type=int, default=None,
                    help="stop cleanly and save last.pt before running this step")
    ap.add_argument("--save_every", type=int, default=2000)
    args, overrides = ap.parse_known_args()
    t_start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    for i in range(0, len(overrides) - 1, 2):
        key = overrides[i].lstrip("-")
        if key not in cfg:
            # Ignoring it silently is how --max_ram_gb once did nothing at all: the key
            # was read with cfg.get() but missing from the yaml, so no override landed.
            print(f"[config] WARNING: unknown option --{key} ignored (not in {args.config})",
                  flush=True)
            continue
        cur, raw = cfg[key], overrides[i + 1]
        cfg[key] = (raw.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(raw)

    # Data-parallel when launched by torchrun (the 500M run: Kaggle's "GPU" is two T4s).
    # Every rank seeds the same RNG, so all of them draw the same batch indices -- each
    # from its own slice of the data -- and the RNG state saved by rank 0 is the state
    # of every rank: resuming stays exact without per-rank bookkeeping.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp, is_main = world > 1, rank == 0
    say = print if is_main else (lambda *a, **k: None)
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ["LOCAL_RANK"])
        if cfg["device"].startswith("cuda"):
            torch.cuda.set_device(local_rank)
            cfg["device"] = f"cuda:{local_rank}"
        dist.init_process_group("nccl" if cfg["device"].startswith("cuda") else "gloo")
        if cfg["batch_size"] % world:
            sys.exit(f"--batch_size {cfg['batch_size']} is the global batch: it must "
                     f"divide by the {world} ranks")
        if cfg.get("puzzle_dir") and cfg["rank_weight"]:
            # Two backward passes per step would make DDP all-reduce twice; the ranking
            # fine-tunes are short and single-GPU, so this is not worth supporting.
            sys.exit("the puzzle ranking loss runs on one GPU only")
    local_batch = cfg["batch_size"] // world

    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = cfg["device"]
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

    use_amp = cfg.get("amp", True) and device_type == "cuda"
    if use_amp and cfg.get("dtype", "bf16") == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        say(f"[amp] bf16 unsupported on {torch.cuda.get_device_name()} -> using fp16")
        cfg["dtype"] = "fp16"
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("dtype", "bf16"))
    ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp else _nullctx()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.get("dtype") == "fp16")

    # Only train is sliced across ranks; every rank keeps the whole (small) val split,
    # so all ranks measure the same val loss on the same batches.
    streams = {"train": ActionValueStream(cfg["data_dir"], "train", cfg.get("max_ram_gb", 4.0),
                                          rank, world),
               "val": ActionValueStream(cfg["data_dir"], "val", cfg.get("max_ram_gb", 4.0))}
    say(f"[data] train {streams['train'].n:,} | val {streams['val'].n:,} samples "
        f"| in RAM: {streams['train'].in_ram}"
        + (f" | {world} ranks, each with its own slice of train" if ddp else ""), flush=True)

    # The puzzle ranking loss is optional and additive: the action-value loss keeps
    # running underneath it. Training on the ranking alone would teach the model which
    # move is best and let it forget how good moves ARE, which is what the search and
    # the bot read out of it.
    rank_streams, centres = None, None
    if cfg.get("puzzle_dir"):
        rank_streams = {s: PuzzleRankStream(cfg["puzzle_dir"], s) for s in ("train", "val")}
        centres = torch.tensor((np.arange(cfg["num_buckets"]) + 0.5) / cfg["num_buckets"],
                               dtype=torch.float32, device=device)
        say(f"[puzzles] train {rank_streams['train'].n:,} | val {rank_streams['val'].n:,} "
              f"positions | {rank_streams['train'].k} candidate moves each "
              f"| weight {cfg['rank_weight']} | in RAM: {rank_streams['train'].in_ram}")

    model_cfg = GPTConfig(
        vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH, num_classes=cfg["num_buckets"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg["dropout"], bias=cfg["bias"], causal=cfg.get("causal", True),
        triton_ln=cfg.get("triton_ln", False),
    )
    model = GPT(model_cfg).to(device)
    say(f"[model] {model.num_params()/1e6:.2f}M non-embedding "
          f"({sum(p.numel() for p in model.parameters())/1e6:.2f}M total) | {asdict(model_cfg)}")
    optimizer = model.configure_optimizers(cfg["weight_decay"], cfg["lr"], (cfg["beta1"], cfg["beta2"]))
    raw_model = model                      # compile wraps it; checkpoints store the raw one

    ckpt_dir = Path(cfg.get("out_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_step, best_val = 0, float("inf")
    if args.init_from:
        # Warm restart: take the weights, leave everything else new. Unlike --resume
        # this starts a fresh cosine schedule (the finished run annealed to min_lr, so
        # continuing on the old one would train at the floor) and fresh AdamW moments.
        # The source checkpoint is only read, never written to.
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if init["cfg"] != asdict(model_cfg):
            sys.exit(f"model config differs from {args.init_from}")
        raw_model.load_state_dict(init["model"])
        say(f"[init] weights from {args.init_from} "
              f"(its step {init.get('step', init.get('next_step', '?'))}), fresh schedule",
              flush=True)
    if args.resume:
        if args.init_from:
            sys.exit("--init_from and --resume are mutually exclusive")
        # A run longer than one Kaggle session (12h) is split in two. Resuming only
        # the weights would reset AdamW's moment estimates and restart the cosine
        # schedule, so everything that shapes the next step is restored -- including
        # the RNG, which picks the batches, so a split run matches an unsplit one
        # exactly (tests/test_resume.py).
        # Load on the CPU: with map_location=device the RNG states would land on the
        # GPU too, and torch.set_rng_state only accepts a CPU ByteTensor (this killed
        # the first attempt at session B). load_state_dict moves the weights and the
        # optimizer moments onto the parameters' device by itself.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["max_steps"] != cfg["max_steps"]:
            sys.exit(f"--max_steps {cfg['max_steps']} differs from the checkpoint's "
                     f"{state['max_steps']}: the LR schedule would silently change")
        if state["cfg"] != asdict(model_cfg):
            sys.exit("model config differs from the checkpoint's")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step, best_val = state["next_step"], state["best_val"]
        torch.set_rng_state(state["rng"].cpu())
        if device_type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        say(f"[resume] from {args.resume}: step {start_step}, best val {best_val:.4f}", flush=True)

    def save_state(next_step: int) -> None:
        if not is_main:                       # all ranks hold the same state
            return
        tmp = ckpt_dir / "last.pt.tmp"
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "next_step": next_step, "best_val": best_val,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device_type == "cuda" else None,
                    "cfg": asdict(model_cfg), "max_steps": cfg["max_steps"],
                    "num_buckets": cfg["num_buckets"]}, tmp)
        os.replace(tmp, ckpt_dir / "last.pt")     # atomic: a kill mid-save cannot corrupt it

    if ddp:
        # No buffer broadcast: the only buffer is the constant attention mask, and a
        # broadcast is a collective in every forward, evaluation included.
        model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None,
                    broadcast_buffers=False)
    if cfg.get("compile", False):
        model = torch.compile(model)

    def any_rank(flag: bool) -> bool:
        """True on every rank if it is true on any: a stop must be taken together, or
        the ranks still training wait forever for the one that left."""
        if not ddp:
            return flag
        t = torch.tensor([int(flag)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    accum = cfg.get("grad_accum_steps", 1)
    # MFU reference: T4 peak is 65 TFLOP/s fp16, A100 312 TFLOP/s bf16.
    peak_flops = cfg.get("peak_tflops", 65.0) * 1e12 * world
    flops_per_sample = raw_model.flops_per_sample(SAMPLE_LENGTH)
    t0 = t_log = time.time()
    s_log = start_step
    stopped = False

    for step in range(start_step, cfg["max_steps"] + 1):
        # Saving happens at the top, before this step's eval and update, so the saved
        # RNG state is exactly the one the next session needs to replay from here.
        # Across ranks the clock is only compared every DEADLINE_CHECK steps (a
        # collective each step would stall the pipeline); stop_at_step needs no vote.
        out_of_time = False
        if args.deadline_s is not None and (not ddp or step % DEADLINE_CHECK == 0):
            out_of_time = any_rank(time.time() - t_start > args.deadline_s)
        if out_of_time or (args.stop_at_step is not None and step >= args.stop_at_step):
            save_state(step)
            stopped = True
            say(f"[stopped] before step {step} ({'deadline' if out_of_time else 'stop_at_step'}); "
                f"resume with --resume {ckpt_dir/'last.pt'}", flush=True)
            break
        if args.save_every and step > start_step and step % args.save_every == 0:
            save_state(step)

        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, cfg)

        if step % cfg["eval_interval"] == 0:
            m = evaluate(model, streams, cfg, ctx, rank_streams, centres)
            note = (f" | puzzle move {m['rank_val_acc']*100:.1f}%"
                    if "rank_val_acc" in m else "")
            say(f"step {step:6d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| val acc {m['val_acc']*100:.1f}%{note} | {time.time()-t0:.0f}s",
                  flush=True)
            if m["val"] < best_val and step > 0:
                best_val = m["val"]
                if is_main:
                        torch.save({"model": raw_model.state_dict(), "cfg": asdict(model_cfg),
                                "step": step, "val": best_val, "val_acc": m["val_acc"],
                                "num_buckets": cfg["num_buckets"]}, ckpt_dir / "chess_best.pt")
            t_log, s_log = time.time(), step

        optimizer.zero_grad(set_to_none=True)
        rank_loss = rank_acc = None
        for _ in range(accum):
            x, y = streams["train"].batch(local_batch, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            scaler.scale(loss).backward()
        if rank_streams is not None and cfg["rank_weight"]:
            # A separate batch, backward on its own: the two losses never share a graph,
            # so adding the ranking term cannot change what the action-value term does to
            # the gradients -- it only adds to them.
            rank_loss, rank_acc = rank_step(model, centres, rank_streams["train"],
                                            cfg, ctx, device)
            scaler.scale(rank_loss * cfg["rank_weight"]).backward()
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
            # Every sequence through the model counts, including the puzzle batch
            # (rank_batch positions x k candidates each). Leaving it out made the
            # fine-tune report 822 samples/s at 9% MFU while it was really moving
            # twice that -- a halved number that looks exactly like a slowdown.
            per_step = cfg["batch_size"] * accum
            if rank_streams is not None and cfg["rank_weight"]:
                per_step += cfg["rank_batch"] * rank_streams["train"].k
            samples = per_step * (step + 1 - s_log)
            sps = samples / max(1e-6, now - t_log)
            mfu = sps * flops_per_sample / peak_flops
            note = (f" | rank {rank_loss.item():.4f} ({rank_acc*100:.0f}%)"
                    if rank_loss is not None else "")
            say(f"  step {step} | loss {loss.item()*accum:.4f}{note} "
                  f"| lr {get_lr(step, cfg):.2e} "
                  f"| {sps:,.0f} samples/s | MFU {mfu*100:.1f}%", flush=True)
            t_log, s_log = now, step + 1

    if not stopped:
        save_state(cfg["max_steps"] + 1)
        say(f"[done] best val loss {best_val:.4f} -> {ckpt_dir/'chess_best.pt'}")
    if ddp:
        dist.barrier()                        # rank 0 has finished writing
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
