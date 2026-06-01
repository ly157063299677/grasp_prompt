from __future__ import annotations

import torch
import torch.nn as nn

from grasp_prompt.utils import mlp


class ConditionEncoder(nn.Module):
    """Encode object, hand, task, and target constraints into condition tokens."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        ablation_cfg = cfg.get("ablation", {})
        self.use_valid_mask = bool(ablation_cfg.get("use_valid_mask", True))
        self.use_target_condition = bool(ablation_cfg.get("use_target_condition", True))
        dim = int(model_cfg["hidden_dim"])
        self.num_joints = int(data_cfg["num_joints"])
        self.affordance_embedding = nn.Embedding(int(data_cfg["num_affordance_classes"]), dim // 4)
        self.task_embedding = nn.Embedding(int(data_cfg["num_tasks"]), dim)
        point_in = 3 + 3 + dim // 4 + 1
        self.point_encoder = mlp(point_in, dim, dim, layers=3)
        hand_in = (
            int(data_cfg["hand_pose_dim"])
            + int(data_cfg["num_joints"]) * 3
            + int(data_cfg["hand_global_dim"])
            + int(data_cfg["hand_object_dim"])
        )
        self.hand_encoder = mlp(hand_in, dim, dim, layers=3)
        target_in = int(data_cfg["num_contact_parts"]) + int(data_cfg["num_joints"])
        self.target_encoder = mlp(target_in, dim, dim, layers=2)
        self.object_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.hand_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.task_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.target_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        points = batch["object_points"]
        normals = batch["object_normals"]
        afford = batch["affordance_labels"].clamp_min(0)
        valid = batch["valid_mask"].unsqueeze(-1)
        valid_input = valid if self.use_valid_mask else torch.zeros_like(valid)
        afford_feat = self.affordance_embedding(afford)
        point_input = torch.cat([points, normals, afford_feat, valid_input], dim=-1)
        point_features = self.point_encoder(point_input)

        mean_pool = point_features.mean(dim=1)
        if self.use_valid_mask:
            valid_weight = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            valid_pool = (point_features * valid_weight).sum(dim=1)
            object_summary = torch.where(
                (batch["valid_mask"].sum(dim=1, keepdim=True) > 0),
                valid_pool,
                mean_pool,
            )
        else:
            object_summary = mean_pool

        hand_flat = torch.cat(
            [
                batch["hand_pose"],
                batch["hand_joints"].flatten(start_dim=1),
                batch["hand_global"],
                batch["hand_object_state"],
            ],
            dim=-1,
        )
        hand_summary = self.hand_encoder(hand_flat)
        task_summary = self.task_embedding(batch["task_id"])
        target_input = torch.cat([batch["target_part"], batch["target_joint_mask"]], dim=-1)
        if not self.use_target_condition:
            target_input = torch.zeros_like(target_input)
        target_summary = self.target_encoder(target_input)

        bsz = points.shape[0]
        summary_tokens = torch.cat(
            [
                self.object_token.expand(bsz, -1, -1) + object_summary.unsqueeze(1),
                self.hand_token.expand(bsz, -1, -1) + hand_summary.unsqueeze(1),
                self.task_token.expand(bsz, -1, -1) + task_summary.unsqueeze(1),
                self.target_token.expand(bsz, -1, -1) + target_summary.unsqueeze(1),
            ],
            dim=1,
        )
        condition_tokens = self.norm(torch.cat([summary_tokens, point_features], dim=1))
        global_feature = self.norm(summary_tokens.mean(dim=1))
        return {
            "condition_tokens": condition_tokens,
            "point_features": point_features,
            "global_feature": global_feature,
        }
