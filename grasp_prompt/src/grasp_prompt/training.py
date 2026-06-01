from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from grasp_prompt.data import GraspPromptDataset, collate_prompt_batch
from grasp_prompt.models import PromptDiffusionModel
from grasp_prompt.utils import ensure_dir, to_device


def build_dataloader(
    data_root: str | Path,
    cfg: dict,
    split: str,
    shuffle: bool,
) -> DataLoader:
    dataset = GraspPromptDataset(
        data_root,
        split=split,
        max_prompts=int(cfg["data"]["max_prompts"]),
        max_regions=int(cfg["data"]["max_regions"]),
        seed=int(cfg["seed"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=collate_prompt_batch,
    )


def train_one_epoch(
    model: PromptDiffusionModel,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "loss_diff": 0.0, "loss_conf": 0.0, "loss_part": 0.0}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = model(batch)
        losses["loss"].backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        bsz = batch["object_points"].shape[0]
        count += bsz
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * bsz
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def loss_epoch(model: PromptDiffusionModel, loader: Iterable, device: str) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_diff": 0.0, "loss_conf": 0.0, "loss_part": 0.0}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        losses = model(batch)
        bsz = batch["object_points"].shape[0]
        count += bsz
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * bsz
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: PromptDiffusionModel,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict,
    epoch: int,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model": model.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: PromptDiffusionModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

