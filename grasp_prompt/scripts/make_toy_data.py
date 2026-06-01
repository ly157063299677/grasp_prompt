from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.utils import ensure_dir, normalize_vectors_np


def default_part_to_joint_np(num_parts: int, num_joints: int) -> np.ndarray:
    matrix = np.zeros((num_parts, num_joints), dtype=np.float32)
    for part in range(num_parts):
        start = int(round(part * num_joints / num_parts))
        end = int(round((part + 1) * num_joints / num_parts))
        end = max(end, start + 1)
        matrix[part, start:min(end, num_joints)] = 1.0
    return matrix


def make_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    points = rng.normal(size=(n, 3)).astype(np.float32)
    points = normalize_vectors_np(points)
    radius = rng.uniform(0.35, 1.0, size=(n, 1)).astype(np.float32)
    points = points * radius
    normals = normalize_vectors_np(points)
    return points.astype(np.float32), normals.astype(np.float32)


def make_sample(rng: np.random.Generator, idx: int, cfg: dict) -> dict[str, np.ndarray]:
    n = cfg["num_points"]
    parts = cfg["num_contact_parts"]
    joints = cfg["num_joints"]
    affordances = cfg["num_affordance_classes"]
    points, normals = make_points(rng, n)

    task_id = int(idx % min(affordances, cfg["num_tasks"]))
    target_center = rng.normal(size=(3,)).astype(np.float32)
    target_center = target_center / max(np.linalg.norm(target_center), 1e-6) * 0.55
    dist = np.linalg.norm(points - target_center[None, :], axis=1)
    valid_mask = dist < np.quantile(dist, 0.25)
    affordance_labels = rng.integers(0, affordances, size=(n,), dtype=np.int64)
    affordance_labels[valid_mask] = task_id % affordances

    part_to_joint = default_part_to_joint_np(parts, joints)
    part_label = int(idx % parts)
    target_part = np.zeros((parts,), dtype=np.float32)
    target_part[part_label] = 1.0
    target_joint_mask = (part_to_joint[part_label] > 0).astype(np.float32)

    valid_points = points[valid_mask]
    if valid_points.shape[0] == 0:
        valid_points = points
    num_regions = int(rng.integers(2, min(5, cfg["max_regions"]) + 1))
    region_idx = rng.choice(valid_points.shape[0], size=num_regions, replace=True)
    centers = valid_points[region_idx] + rng.normal(scale=0.015, size=(num_regions, 3)).astype(np.float32)
    prompt_normals = normalize_vectors_np(centers)
    prompt_scales = rng.uniform(0.035, 0.07, size=(num_regions,)).astype(np.float32)
    prompt_parts = np.full((num_regions,), part_label, dtype=np.int64)
    if num_regions > 1:
        prompt_parts[1:] = rng.integers(0, parts, size=(num_regions - 1,), dtype=np.int64)

    hand_pose = rng.normal(scale=0.1, size=(cfg["hand_pose_dim"],)).astype(np.float32)
    hand_joints = centers[0][None, :] + rng.normal(scale=0.05, size=(joints, 3)).astype(np.float32)
    hand_global = rng.normal(scale=0.05, size=(cfg["hand_global_dim"],)).astype(np.float32)
    hand_object_state = rng.normal(scale=0.05, size=(cfg["hand_object_dim"],)).astype(np.float32)
    hand_object_state[: min(3, cfg["hand_object_dim"])] = centers[0][: min(3, cfg["hand_object_dim"])]

    return {
        "object_points": points,
        "object_normals": normals,
        "affordance_labels": affordance_labels,
        "valid_mask": valid_mask.astype(np.float32),
        "hand_pose": hand_pose,
        "hand_joints": hand_joints,
        "hand_global": hand_global,
        "hand_object_state": hand_object_state,
        "task_id": np.asarray(task_id, dtype=np.int64),
        "target_part": target_part,
        "target_joint_mask": target_joint_mask,
        "prompt_centers": centers.astype(np.float32),
        "prompt_normals": prompt_normals.astype(np.float32),
        "prompt_part_labels": prompt_parts,
        "prompt_scales": prompt_scales,
        "object_id": np.asarray(f"toy_object_{idx:04d}"),
        "category": np.asarray("toy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-points", type=int, default=512)
    args = parser.parse_args()

    cfg = {
        "num_points": args.num_points,
        "max_regions": 12,
        "num_affordance_classes": 32,
        "num_contact_parts": 8,
        "num_joints": 21,
        "hand_pose_dim": 48,
        "hand_global_dim": 6,
        "hand_object_dim": 12,
        "num_tasks": 32,
    }
    out = Path(args.out)
    for split in ["train", "val", "test"]:
        ensure_dir(out / split)
    rng = np.random.default_rng(args.seed)
    train_end = int(args.num_samples * 0.7)
    val_end = int(args.num_samples * 0.85)
    counters = {"train": 0, "val": 0, "test": 0}
    for idx in range(args.num_samples):
        split = "train" if idx < train_end else "val" if idx < val_end else "test"
        sample = make_sample(rng, idx, cfg)
        sample_idx = counters[split]
        counters[split] += 1
        np.savez_compressed(out / split / f"sample_{sample_idx:06d}.npz", **sample)
    print(f"wrote toy dataset to {out}")


if __name__ == "__main__":
    main()
