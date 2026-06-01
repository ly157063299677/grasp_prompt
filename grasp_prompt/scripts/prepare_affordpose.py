from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.data.label_builder import build_prompt_regions, make_valid_region_mask, target_from_contact_part
from grasp_prompt.utils import ensure_dir


def load_array(path: str | Path, field: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        if field is not None:
            raise ValueError(f"field selection is not supported for .npy files: {path}")
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if field is not None:
                if field not in data.files:
                    raise KeyError(f"{field!r} is not present in {path}")
                return data[field]
            if len(data.files) != 1:
                raise ValueError(f"{path} has multiple arrays; set a manifest field name")
            return data[data.files[0]]
    if path.suffix == ".csv":
        if field is not None:
            raise ValueError(f"field selection is not supported for CSV files: {path}")
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
        return np.asarray(rows, dtype=np.float32)
    raise ValueError(f"unsupported array file: {path}")


def load_array_spec(base: Path, spec) -> np.ndarray:
    if isinstance(spec, str):
        return load_array(base / spec)
    if isinstance(spec, dict):
        return load_array(base / spec["path"], field=spec.get("field"))
    raise TypeError(f"array spec must be a path string or an object, got {type(spec).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON manifest describing raw or processed samples.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    out_root = ensure_dir(args.out)
    base = manifest_path.parent

    task_to_affordances = {
        int(k): [int(x) for x in v] for k, v in manifest.get("task_to_affordances", {}).items()
    }
    part_to_joint = None
    if "part_to_joint" in manifest:
        part_to_joint = load_array_spec(base, manifest["part_to_joint"]).astype(np.float32)

    counters: dict[str, int] = {}
    for record in manifest["samples"]:
        split = record.get("split", "train")
        split_dir = ensure_dir(out_root / split)
        counters.setdefault(split, 0)
        out_path = split_dir / f"sample_{counters[split]:06d}.npz"
        counters[split] += 1

        if "processed_npz" in record:
            shutil.copy2(base / record["processed_npz"], out_path)
            continue

        source = {}
        if "source_npz" in record:
            with np.load(base / record["source_npz"], allow_pickle=True) as item:
                source.update({key: item[key] for key in item.files})
        for key, value in record.get("arrays", {}).items():
            source[key] = load_array_spec(base, value)

        task_id = int(record.get("task_id", np.asarray(source["task_id"]).item()))
        valid_mask = source.get("valid_mask")
        if valid_mask is None:
            valid_mask = make_valid_region_mask(source["affordance_labels"], task_id, task_to_affordances)

        if {"prompt_centers", "prompt_part_labels", "prompt_scales"}.issubset(source):
            centers = source["prompt_centers"]
            normals = source.get("prompt_normals", np.zeros_like(centers))
            part_labels = source["prompt_part_labels"]
            scales = source["prompt_scales"]
        else:
            hand_parts = {int(k): load_array_spec(base, v) for k, v in record["hand_part_vertices"].items()}
            regions = build_prompt_regions(
                source["object_points"],
                source["object_normals"],
                valid_mask,
                source["hand_vertices"],
                hand_parts,
                contact_threshold=float(record.get("contact_threshold", 0.025)),
                cluster_radius=float(record.get("cluster_radius", 0.04)),
                min_cluster_size=int(record.get("min_cluster_size", 3)),
            )
            centers = regions.centers
            normals = regions.normals
            part_labels = regions.part_labels
            scales = regions.scales

        if "target_part" in source and "target_joint_mask" in source:
            target_part = source["target_part"]
            target_joint = source["target_joint_mask"]
        else:
            if part_to_joint is None:
                raise ValueError("part_to_joint is required to derive target masks")
            if len(part_labels) == 0:
                raise ValueError("cannot derive target from an empty prompt label set")
            target_part, target_joint = target_from_contact_part(int(part_labels[0]), part_to_joint)

        np.savez_compressed(
            out_path,
            object_points=source["object_points"].astype(np.float32),
            object_normals=source["object_normals"].astype(np.float32),
            affordance_labels=source["affordance_labels"].astype(np.int64),
            valid_mask=np.asarray(valid_mask).astype(np.float32),
            hand_pose=source["hand_pose"].astype(np.float32),
            hand_joints=source["hand_joints"].astype(np.float32),
            hand_global=source["hand_global"].astype(np.float32),
            hand_object_state=source["hand_object_state"].astype(np.float32),
            task_id=np.asarray(task_id, dtype=np.int64),
            target_part=target_part.astype(np.float32),
            target_joint_mask=target_joint.astype(np.float32),
            prompt_centers=centers.astype(np.float32),
            prompt_normals=normals.astype(np.float32),
            prompt_part_labels=part_labels.astype(np.int64),
            prompt_scales=scales.astype(np.float32),
            object_id=np.asarray(record.get("object_id", "")),
            category=np.asarray(record.get("category", "")),
        )
    print(f"wrote processed samples to {out_root}")


if __name__ == "__main__":
    main()
