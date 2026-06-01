from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from grasp_prompt.models.condition import ConditionEncoder
from grasp_prompt.models.diffusion import Denoiser, GaussianDiffusion
from grasp_prompt.models.selector import CandidateSelector


def default_part_to_joint(num_parts: int, num_joints: int) -> torch.Tensor:
    """Small contiguous mapping used when no study-specific matrix is supplied."""
    matrix = torch.zeros(num_parts, num_joints, dtype=torch.float32)
    for part in range(num_parts):
        start = int(round(part * num_joints / num_parts))
        end = int(round((part + 1) * num_joints / num_parts))
        end = max(end, start + 1)
        matrix[part, start:min(end, num_joints)] = 1.0
    return matrix


def read_part_to_joint(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as item:
            key = "part_to_joint" if "part_to_joint" in item.files else item.files[0]
            array = item[key]
    elif path.suffix == ".csv":
        rows = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                try:
                    rows.append([float(x) for x in row])
                    continue
                except ValueError:
                    pass
                try:
                    rows.append([float(x) for x in row[1:]])
                except ValueError:
                    continue
        array = np.asarray(rows, dtype=np.float32)
    else:
        array = np.loadtxt(path)
    if array.ndim != 2:
        raise ValueError(f"part-to-joint matrix must be 2-D, got {array.shape}")
    return torch.as_tensor(array, dtype=torch.float32)


class PromptDiffusionModel(nn.Module):
    def __init__(self, cfg: dict, part_to_joint: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ConditionEncoder(cfg)
        self.denoiser = Denoiser(cfg)
        self.diffusion = GaussianDiffusion(cfg)
        self.selector = CandidateSelector(cfg)
        if part_to_joint is None:
            mapping_path = cfg["data"].get("part_to_joint_path")
            if mapping_path:
                part_to_joint = read_part_to_joint(mapping_path)
            else:
                part_to_joint = default_part_to_joint(
                    int(cfg["data"]["num_contact_parts"]),
                    int(cfg["data"]["num_joints"]),
                )
        expected_shape = (
            int(cfg["data"]["num_contact_parts"]),
            int(cfg["data"]["num_joints"]),
        )
        if tuple(part_to_joint.shape) != expected_shape:
            raise ValueError(f"part_to_joint shape {tuple(part_to_joint.shape)} != {expected_shape}")
        self.register_buffer("part_to_joint", part_to_joint.float())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        diff_loss, _ = self.diffusion.training_loss(
            self.denoiser,
            batch["prompt_points"],
            encoded["condition_tokens"],
        )
        selector_candidates = self._make_selector_training_candidates(batch)
        selector_out = self.selector(
            batch,
            selector_candidates,
            encoded["point_features"],
            encoded["global_feature"],
            self.part_to_joint,
        )
        selector_losses = self.selector.selection_losses(selector_out, batch)
        eta_conf = float(self.cfg["training"]["eta_conf"])
        eta_part = float(self.cfg["training"]["eta_part"])
        total = diff_loss + eta_conf * selector_losses["loss_conf"] + eta_part * selector_losses["loss_part"]
        return {
            "loss": total,
            "loss_diff": diff_loss.detach(),
            "loss_conf": selector_losses["loss_conf"].detach(),
            "loss_part": selector_losses["loss_part"].detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        bsz = batch["object_points"].shape[0]
        max_prompts = int(self.cfg["data"]["max_prompts"])
        candidates = self.diffusion.sample(
            self.denoiser,
            encoded["condition_tokens"],
            shape=(bsz, max_prompts, 3),
            steps=steps,
        )
        selector_out = self.selector(
            batch,
            candidates,
            encoded["point_features"],
            encoded["global_feature"],
            self.part_to_joint,
        )
        return selector_out

    def _make_selector_training_candidates(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prompts = batch["prompt_points"]
        bsz, max_prompts, _ = prompts.shape
        pos_count = max(1, max_prompts // 2)
        neg_count = max_prompts - pos_count
        if neg_count == 0:
            return prompts
        object_points = batch["object_points"]
        rand_idx = torch.randint(
            low=0,
            high=object_points.shape[1],
            size=(bsz, neg_count),
            device=object_points.device,
        )
        gather_idx = rand_idx.unsqueeze(-1).expand(-1, -1, 3)
        negatives = torch.gather(object_points, dim=1, index=gather_idx)
        return torch.cat([prompts[:, :pos_count], negatives], dim=1)
