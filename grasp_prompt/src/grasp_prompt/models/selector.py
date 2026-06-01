from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_prompt.utils import mlp


class CandidateSelector(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim = int(cfg["model"]["hidden_dim"])
        parts = int(cfg["data"]["num_contact_parts"])
        self.knn = int(cfg["selector"]["knn"])
        self.kernel_sigma = float(cfg["selector"]["kernel_sigma"])
        self.alpha = float(cfg["selector"]["alpha"])
        self.beta = float(cfg["selector"]["beta"])
        self.gamma = float(cfg["selector"]["gamma"])
        self.lambda_penalty = float(cfg["selector"]["lambda_penalty"])
        self.nms_radius = float(cfg["selector"]["nms_radius"])
        ablation_cfg = cfg.get("ablation", {})
        self.use_confidence_score = bool(ablation_cfg.get("use_confidence_score", True))
        self.use_contact_score = bool(ablation_cfg.get("use_contact_score", True))
        self.use_joint_score = bool(ablation_cfg.get("use_joint_score", True))
        self.use_compensation_penalty = bool(ablation_cfg.get("use_compensation_penalty", True))
        self.use_spatial_nms = bool(ablation_cfg.get("use_spatial_nms", True))
        self.encoder = mlp(3 + dim + dim, dim, dim, layers=3)
        self.confidence_head = nn.Linear(dim, 1)
        self.part_head = nn.Linear(dim, parts)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        point_features: torch.Tensor,
        global_feature: torch.Tensor,
        part_to_joint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_feature = self.interpolate_object_feature(
            candidates,
            batch["object_points"],
            point_features,
            self.knn,
            self.kernel_sigma,
        )
        global_rep = global_feature.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        h = self.encoder(torch.cat([candidates, local_feature, global_rep], dim=-1))
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        part_logits = self.part_head(h)
        part_probs = F.softmax(part_logits, dim=-1)
        scores, score_terms = self.score_candidates(
            confidence,
            part_probs,
            batch["target_part"],
            batch["target_joint_mask"],
            part_to_joint,
        )
        if self.use_spatial_nms:
            selected_idx = self.spatial_nms_first(candidates, scores, self.nms_radius)
        else:
            selected_idx = scores.argmax(dim=1)
        selected = candidates[torch.arange(candidates.shape[0], device=candidates.device), selected_idx]
        selected_part = part_probs[
            torch.arange(candidates.shape[0], device=candidates.device), selected_idx
        ].argmax(dim=-1)
        selected_confidence = confidence[
            torch.arange(candidates.shape[0], device=candidates.device), selected_idx
        ]
        return {
            "candidates": candidates,
            "confidence": confidence,
            "part_logits": part_logits,
            "part_probs": part_probs,
            "scores": scores,
            "selected_idx": selected_idx,
            "selected_prompt": selected,
            "selected_part": selected_part,
            "selected_confidence": selected_confidence,
            **score_terms,
        }

    @staticmethod
    def interpolate_object_feature(
        candidates: torch.Tensor,
        object_points: torch.Tensor,
        point_features: torch.Tensor,
        knn: int,
        sigma: float,
    ) -> torch.Tensor:
        bsz, num_candidates, _ = candidates.shape
        _, num_points, dim = point_features.shape
        k = min(knn, num_points)
        dist = torch.cdist(candidates, object_points)
        values, idx = dist.topk(k=k, dim=-1, largest=False)
        features_expanded = point_features.unsqueeze(1).expand(bsz, num_candidates, num_points, dim)
        idx_expanded = idx.unsqueeze(-1).expand(bsz, num_candidates, k, dim)
        neigh = torch.gather(features_expanded, dim=2, index=idx_expanded)
        weights = torch.exp(-(values**2) / max(sigma**2, 1e-8))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * neigh).sum(dim=2)

    def score_candidates(
        self,
        confidence: torch.Tensor,
        part_probs: torch.Tensor,
        target_part: torch.Tensor,
        target_joint_mask: torch.Tensor,
        part_to_joint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        contact_consistency = (part_probs * target_part.unsqueeze(1)).sum(dim=-1)
        joint_response = torch.matmul(part_probs, part_to_joint)
        numerator = 2.0 * (joint_response * target_joint_mask.unsqueeze(1)).sum(dim=-1)
        denominator = joint_response.abs().sum(dim=-1) + target_joint_mask.unsqueeze(1).abs().sum(dim=-1)
        joint_consistency = numerator / denominator.clamp_min(1e-8)
        compensation_penalty = (part_probs * (1.0 - target_part).unsqueeze(1)).sum(dim=-1)
        score = torch.zeros_like(confidence)
        if self.use_confidence_score:
            score = score + self.alpha * confidence
        if self.use_contact_score:
            score = score + self.beta * contact_consistency
        if self.use_joint_score:
            score = score + self.gamma * joint_consistency
        if self.use_compensation_penalty:
            score = score - self.lambda_penalty * compensation_penalty
        return score, {
            "contact_consistency": contact_consistency,
            "joint_consistency": joint_consistency,
            "compensation_penalty": compensation_penalty,
        }

    @staticmethod
    def spatial_nms_first(points: torch.Tensor, scores: torch.Tensor, radius: float) -> torch.Tensor:
        selected = []
        for b in range(points.shape[0]):
            order = torch.argsort(scores[b], descending=True)
            keep: list[torch.Tensor] = []
            for idx in order:
                p = points[b, idx]
                if not keep:
                    keep.append(idx)
                    continue
                kept_points = points[b, torch.stack(keep)]
                if torch.norm(kept_points - p.unsqueeze(0), dim=-1).min() > radius:
                    keep.append(idx)
            selected.append(keep[0] if keep else order[0])
        return torch.stack(selected)

    @staticmethod
    def candidate_targets(
        candidates: torch.Tensor,
        label_centers: torch.Tensor,
        label_scales: torch.Tensor,
        label_parts: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.cdist(candidates, label_centers)
        norm_dist = dist / label_scales.clamp_min(1e-6).unsqueeze(1)
        inf = torch.full_like(norm_dist, 1e6)
        norm_dist = torch.where(label_mask.bool().unsqueeze(1), norm_dist, inf)
        min_dist, matched_idx = norm_dist.min(dim=-1)
        has_label = label_mask.sum(dim=-1, keepdim=True) > 0
        positive = (min_dist <= 1.0) & has_label
        matched_parts = torch.gather(
            label_parts.clamp_min(0),
            dim=1,
            index=matched_idx.clamp_max(label_parts.shape[1] - 1),
        )
        return positive.float(), matched_parts.long(), min_dist

    def selection_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        positive, matched_parts, _ = self.candidate_targets(
            outputs["candidates"],
            batch["label_centers"],
            batch["label_scales"],
            batch["label_parts"],
            batch["label_mask"],
        )
        conf_loss = F.binary_cross_entropy(outputs["confidence"], positive)
        pos_mask = positive.bool()
        if pos_mask.any():
            part_loss = F.cross_entropy(outputs["part_logits"][pos_mask], matched_parts[pos_mask])
        else:
            part_loss = outputs["part_logits"].sum() * 0.0
        return {
            "loss_conf": conf_loss,
            "loss_part": part_loss,
            "candidate_positive": positive,
            "candidate_part_target": matched_parts,
        }
