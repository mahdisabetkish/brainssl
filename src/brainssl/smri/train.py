"""Pretraining and supervised training of the 3D U-Net.

Both loops are iteration-based, write a checkpoint every ``save_every`` steps
and resume from it if one exists, so an interrupted multi-hour run loses at most
a few minutes. Each run writes to its own directory:

    config.json      the resolved configuration
    log.jsonl        one line per logging step (loss, learning rate, validation)
    last.pt          full training state for resuming
    best.pt / final.pt  model weights (supervised: best validation Dice / last step)
    summary.json     written last; its presence marks the run as finished
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..utils import seed_everything, write_json
from .augment import AugmentConfig, GPUAugment
from .dataset import SegmentationPatches, UnlabelledPatches
from .infer import enforce_hierarchy, sliding_window
from .losses import deep_supervision_loss
from .masking import apply_masks, channel_dropout, contralateral_mask, masked_mse, random_mask
from .metrics import region_metrics
from .model import UNet3D, UNetConfig, count_parameters
from .preprocess import load_case, regions_from_labels

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    out_dir: str = "runs/smri/debug"
    raw: str = "data/raw"
    index: str = "data/processed/msd/index.csv"
    splits: str = "data/processed/msd/splits.json"
    bilateral: bool = False
    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    iterations: int = 10000
    batch: int = 2
    patch: tuple[int, int] = (128, 128)
    lr_width: int = 160
    workers: int = 6
    amp: bool = True
    seed: int = 0
    log_every: int = 50
    save_every: int = 250


@dataclass
class PretrainConfig(RunConfig):
    masking: str = "contralateral"     # or "random"
    mask_ratio: float = 0.6            # random masking only
    both_ratio: float = 0.2            # contralateral masking: pairs hidden on both sides
    channel_drop: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 500


@dataclass
class FinetuneConfig(RunConfig):
    budget: str = "100pct"
    init: str = ""                     # pretrained checkpoint, empty for random init
    lr: float = 1e-2
    momentum: float = 0.99
    weight_decay: float = 3e-5
    warmup: int = 0                    # linear lr warm-up steps before the poly decay
    foreground: float = 0.5
    val_every: int = 500
    val_cases: int = 20
    augment: AugmentConfig = field(default_factory=AugmentConfig)


def _loader(dataset, cfg: RunConfig) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.batch, num_workers=cfg.workers, pin_memory=True,
                      persistent_workers=cfg.workers > 0, drop_last=True,
                      prefetch_factor=2 if cfg.workers > 0 else None)


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(cfg: RunConfig, extra_heads: dict[str, int] | None = None) -> UNet3D:
    return UNet3D(UNetConfig(channels=tuple(cfg.channels), bilateral=cfg.bilateral,
                             extra_heads=extra_heads or {}))


def _resume(path: Path, model, opt, scaler) -> int:
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    log.info("resumed from step %d", state["step"])
    return int(state["step"])


def _save(path: Path, model, opt, scaler, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "step": step}, tmp)
    tmp.replace(path)


def pretrain(cfg: PretrainConfig) -> dict:
    out = Path(cfg.out_dir)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dataclasses.asdict(cfg))
    seed_everything(cfg.seed)
    device = _device()
    cfg.amp = cfg.amp and device.type == "cuda"
    index = pd.read_csv(cfg.index)
    splits = json.loads(Path(cfg.splits).read_text())
    data = UnlabelledPatches(index, splits["pretrain"], samples=10**9, patch=cfg.patch, seed=cfg.seed,
                             lr_width=cfg.lr_width, raw_root=cfg.raw)
    batches = _infinite(_loader(data, cfg))

    model = _build_model(cfg, {"recon": 4}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    step = _resume(out / "last.pt", model, opt, scaler)
    log.info("pretraining %s masking, %d parameters", cfg.masking, count_parameters(model))

    shape = (cfg.lr_width,) + tuple(cfg.patch)
    t0, running = time.time(), []
    with open(out / "log.jsonl", "a") as fh:
        while step < cfg.iterations:
            image = next(batches).to(device, non_blocking=True).float()
            b = image.shape[0]
            if cfg.masking == "contralateral":
                spatial = contralateral_mask(b, shape, both=cfg.both_ratio, device=device)
            elif cfg.masking == "random":
                spatial = random_mask(b, shape, ratio=cfg.mask_ratio, device=device)
            else:
                raise ValueError(cfg.masking)
            dropped = channel_dropout(b, image.shape[1], cfg.channel_drop, device=device)
            corrupted, weight = apply_masks(image, spatial, dropped)
            lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * step / cfg.iterations))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp):
                recon = model.forward_extra(corrupted, "recon")
            loss = masked_mse(recon, image, weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            running.append(loss.item())
            if step % cfg.log_every == 0:
                rec = {"step": step, "loss": float(np.mean(running)), "lr": lr,
                       "sec_per_step": (time.time() - t0) / cfg.log_every}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log.info("pretrain %d/%d loss %.4f (%.2fs/step)", step, cfg.iterations, rec["loss"],
                         rec["sec_per_step"])
                running, t0 = [], time.time()
            if step % cfg.save_every == 0 or step == cfg.iterations:
                _save(out / "last.pt", model, opt, scaler, step)
    torch.save(model.state_dict(), out / "final.pt")
    summary = {"iterations": cfg.iterations, "masking": cfg.masking, "bilateral": cfg.bilateral}
    write_json(out / "summary.json", summary)
    return summary


@torch.no_grad()
def evaluate_cases(model: UNet3D, index: pd.DataFrame, case_ids: list[str], amp: bool = True,
                   mirror: bool = True, keep: Collection[str] = (), distances: bool = True,
                   window: tuple[int, int, int] = (160, 128, 128),
                   raw_root: str = "data/raw") -> tuple[list[dict], dict]:
    model.eval()
    device = next(model.parameters()).device
    amp = amp and device.type == "cuda"
    rows = index.set_index("case_id")
    results, predictions = [], {}
    for case_id in case_ids:
        image, label, box = load_case(rows.loc[case_id], lr_width=window[0], raw_root=raw_root)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            prob = sliding_window(model, torch.from_numpy(image), window=window, mirror=mirror)
        pred = enforce_hierarchy(prob)
        metrics = region_metrics(pred, regions_from_labels(label), distances=distances)
        metrics["case_id"] = case_id
        results.append(metrics)
        if case_id in keep:
            predictions[case_id] = (prob.cpu().numpy().astype(np.float16), box)
    return results, predictions


def finetune(cfg: FinetuneConfig) -> dict:
    out = Path(cfg.out_dir)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dataclasses.asdict(cfg))
    seed_everything(cfg.seed)
    device = _device()
    cfg.amp = cfg.amp and device.type == "cuda"
    index = pd.read_csv(cfg.index)
    splits = json.loads(Path(cfg.splits).read_text())
    train_ids = splits["budgets"][cfg.budget]
    val_ids = splits["val"][: cfg.val_cases]
    data = SegmentationPatches(index, train_ids, samples=10**9, patch=cfg.patch, seed=cfg.seed,
                               foreground=cfg.foreground, lr_width=cfg.lr_width, raw_root=cfg.raw)
    batches = _infinite(_loader(data, cfg))

    model = _build_model(cfg).to(device)
    if cfg.init:
        state = torch.load(cfg.init, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("extra.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("initialised from %s (missing %s, unexpected %s)", cfg.init, missing, unexpected)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=True,
                          weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    step = _resume(out / "last.pt", model, opt, scaler)
    augment = GPUAugment(cfg.augment)
    best = -1.0
    if (out / "best.json").exists():
        best = json.loads((out / "best.json").read_text())["dice_mean"]
    log.info("training on %d cases (%s), %d parameters, bilateral=%s, init=%s",
             len(train_ids), cfg.budget, count_parameters(model), cfg.bilateral, cfg.init or "random")

    t0, running = time.time(), []
    with open(out / "log.jsonl", "a") as fh:
        while step < cfg.iterations:
            model.train()
            image, target = next(batches)
            image = image.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True)
            image, target = augment(image, target)
            lr = cfg.lr * (1 - step / cfg.iterations) ** 0.9 * min(1.0, (step + 1) / max(1, cfg.warmup))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp):
                outputs = model(image)
            loss = deep_supervision_loss(outputs, target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            running.append(loss.item())
            if step % cfg.log_every == 0:
                rec = {"step": step, "loss": float(np.mean(running)), "lr": lr,
                       "sec_per_step": (time.time() - t0) / cfg.log_every}
                if step % cfg.val_every == 0 or step == cfg.iterations:
                    results, _ = evaluate_cases(model, index, val_ids, amp=cfg.amp, mirror=False,
                                                distances=False, window=(cfg.lr_width,) + tuple(cfg.patch),
                                                raw_root=cfg.raw)
                    val = {k: float(np.mean([r[k] for r in results])) for k in results[0] if k != "case_id"}
                    rec["val"] = val
                    if val["dice_mean"] > best:
                        best = val["dice_mean"]
                        torch.save(model.state_dict(), out / "best.pt")
                        write_json(out / "best.json", {"step": step, **val})
                    log.info("step %d val dice WT %.3f TC %.3f ET %.3f (best %.3f)", step,
                             val["dice_WT"], val["dice_TC"], val["dice_ET"], best)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                log.info("train %d/%d loss %.4f lr %.4f (%.2fs/step)", step, cfg.iterations,
                         rec["loss"], lr, rec["sec_per_step"])
                running, t0 = [], time.time()
            if step % cfg.save_every == 0 or step == cfg.iterations:
                _save(out / "last.pt", model, opt, scaler, step)
    torch.save(model.state_dict(), out / "final.pt")
    summary = {"budget": cfg.budget, "cases": len(train_ids), "best_val": best,
               "bilateral": cfg.bilateral, "init": cfg.init}
    write_json(out / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the tumour segmentation network")
    parser.add_argument("stage", choices=["pretrain", "finetune"])
    parser.add_argument("--config", type=Path, help="JSON file with field overrides")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override single fields, values parsed as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cls = PretrainConfig if args.stage == "pretrain" else FinetuneConfig
    overrides = json.loads(args.config.read_text()) if args.config else {}
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    if "augment" in overrides and isinstance(overrides["augment"], dict):
        overrides["augment"] = AugmentConfig(**overrides["augment"])
    for key in ("channels", "patch"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    cfg = cls(**overrides)
    (pretrain if args.stage == "pretrain" else finetune)(cfg)


if __name__ == "__main__":
    main()
