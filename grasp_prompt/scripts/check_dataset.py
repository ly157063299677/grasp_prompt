from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
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
    "prompt_centers",
    "prompt_normals",
    "prompt_part_labels",
    "prompt_scales",
}


def scalar_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0]) if arr.size else ""


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    with np.load(path, allow_pickle=True) as item:
        missing = sorted(REQUIRED_FIELDS.difference(item.files))
        if missing:
            errors.append(f"missing fields: {missing}")
            return errors
        points = item["object_points"]
        normals = item["object_normals"]
        labels = item["affordance_labels"]
        valid = item["valid_mask"]
        centers = item["prompt_centers"]
        prompt_normals = item["prompt_normals"]
        part_labels = item["prompt_part_labels"]
        scales = item["prompt_scales"]
        joints = item["hand_joints"]

        if points.ndim != 2 or points.shape[1] != 3:
            errors.append(f"object_points shape should be [N, 3], got {points.shape}")
        if normals.shape != points.shape:
            errors.append(f"object_normals shape {normals.shape} does not match points {points.shape}")
        if labels.shape[0] != points.shape[0]:
            errors.append("affordance_labels length does not match object_points")
        if valid.shape[0] != points.shape[0]:
            errors.append("valid_mask length does not match object_points")
        if centers.ndim != 2 or centers.shape[1] != 3:
            errors.append(f"prompt_centers shape should be [L, 3], got {centers.shape}")
        if prompt_normals.shape != centers.shape:
            errors.append("prompt_normals shape does not match prompt_centers")
        if part_labels.shape[0] != centers.shape[0]:
            errors.append("prompt_part_labels length does not match prompt_centers")
        if scales.shape[0] != centers.shape[0]:
            errors.append("prompt_scales length does not match prompt_centers")
        if joints.ndim != 2 or joints.shape[1] != 3:
            errors.append(f"hand_joints shape should be [J, 3], got {joints.shape}")
        if np.asarray(valid).sum() <= 0:
            errors.append("valid_mask has no positive point")
        if centers.shape[0] == 0:
            errors.append("no prompt region is available")
        if scales.size and np.any(scales <= 0):
            errors.append("prompt_scales must be positive")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.data)
    total = 0
    failed = 0
    object_to_splits: dict[str, set[str]] = defaultdict(set)
    per_split: dict[str, int] = {}
    errors_printed = 0

    for split in args.splits:
        split_dir = root / split
        files = sorted(split_dir.glob("*.npz"))
        per_split[split] = len(files)
        for path in files:
            total += 1
            with np.load(path, allow_pickle=True) as item:
                if "object_id" in item.files:
                    object_to_splits[scalar_text(item["object_id"])].add(split)
            errors = check_file(path)
            if errors:
                failed += 1
                if errors_printed < args.max_errors:
                    print(f"[bad] {path}")
                    for err in errors:
                        print(f"  - {err}")
                    errors_printed += 1

    leakage = {obj: splits for obj, splits in object_to_splits.items() if len(splits) > 1 and obj}
    print("samples:", total)
    print("by_split:", per_split)
    print("failed_files:", failed)
    print("object_leakage:", len(leakage))
    for obj, splits in list(sorted(leakage.items()))[: args.max_errors]:
        print(f"  - {obj}: {sorted(splits)}")
    if failed or leakage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

