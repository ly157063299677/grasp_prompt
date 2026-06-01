from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grasp_prompt.utils import pad_array


class GraspPromptDataset:
    """Dataset for processed `.npz` samples described in `configs/affordpose_schema.md`."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_prompts: int = 16,
        max_regions: int = 12,
        seed: int = 7,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_prompts = int(max_prompts)
        self.max_regions = int(max_regions)
        self.seed = int(seed)
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"split directory not found: {split_dir}")
        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as item:
            sample = {key: item[key] for key in item.files}

        centers = sample["prompt_centers"].astype(np.float32)
        scales = sample["prompt_scales"].astype(np.float32)
        part_labels = sample["prompt_part_labels"].astype(np.int64)
        normals = sample.get("prompt_normals", np.zeros_like(centers)).astype(np.float32)

        label_centers, label_mask = pad_array(centers, self.max_regions, 0.0)
        label_scales, _ = pad_array(scales[:, None], self.max_regions, 1.0)
        label_parts, _ = pad_array(part_labels[:, None], self.max_regions, -1)
        label_normals, _ = pad_array(normals, self.max_regions, 0.0)

        prompt_points, prompt_parts = self._sample_fixed_prompts(idx, centers, scales, part_labels)

        out = {
            "path": str(path),
            "object_points": sample["object_points"].astype(np.float32),
            "object_normals": sample["object_normals"].astype(np.float32),
            "affordance_labels": sample["affordance_labels"].astype(np.int64),
            "valid_mask": sample["valid_mask"].astype(np.float32),
            "hand_pose": sample["hand_pose"].astype(np.float32),
            "hand_joints": sample["hand_joints"].astype(np.float32),
            "hand_global": sample["hand_global"].astype(np.float32),
            "hand_object_state": sample["hand_object_state"].astype(np.float32),
            "task_id": np.asarray(sample["task_id"], dtype=np.int64),
            "target_part": sample["target_part"].astype(np.float32),
            "target_joint_mask": sample["target_joint_mask"].astype(np.float32),
            "prompt_points": prompt_points.astype(np.float32),
            "prompt_point_parts": prompt_parts.astype(np.int64),
            "label_centers": label_centers.astype(np.float32),
            "label_normals": label_normals.astype(np.float32),
            "label_scales": label_scales.squeeze(-1).astype(np.float32),
            "label_parts": label_parts.squeeze(-1).astype(np.int64),
            "label_mask": label_mask.astype(np.float32),
        }
        return out

    def _sample_fixed_prompts(
        self,
        idx: int,
        centers: np.ndarray,
        scales: np.ndarray,
        part_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + idx)
        if centers.shape[0] == 0:
            return (
                np.zeros((self.max_prompts, 3), dtype=np.float32),
                np.full((self.max_prompts,), -1, dtype=np.int64),
            )
        choice = rng.integers(0, centers.shape[0], size=self.max_prompts)
        noise = rng.normal(size=(self.max_prompts, 3)).astype(np.float32)
        scale = np.maximum(scales[choice], 1e-4).reshape(-1, 1)
        points = centers[choice] + 0.25 * scale * noise
        return points.astype(np.float32), part_labels[choice].astype(np.int64)


def collate_prompt_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    tensor_keys = [
        "object_points",
        "object_normals",
        "affordance_labels",
        "valid_mask",
        "hand_pose",
        "hand_joints",
        "hand_global",
        "hand_object_state",
        "task_id",
        "target_part",
        "target_joint_mask",
        "prompt_points",
        "prompt_point_parts",
        "label_centers",
        "label_normals",
        "label_scales",
        "label_parts",
        "label_mask",
    ]
    out: dict[str, Any] = {"path": [item["path"] for item in batch]}
    for key in tensor_keys:
        values = [item[key] for item in batch]
        arr = np.stack(values, axis=0)
        if arr.dtype.kind in {"i", "u"}:
            out[key] = torch.from_numpy(arr).long()
        else:
            out[key] = torch.from_numpy(arr).float()
    out["task_id"] = out["task_id"].view(-1).long()
    return out

