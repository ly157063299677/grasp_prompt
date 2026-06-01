from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.config import load_config
from grasp_prompt.metrics import (
    candidate_binary_labels,
    compensation_penalty_score,
    contact_part_accuracy,
    cover_at_k,
    hit_at_1,
    joint_consistency_score,
    mean_prompt_distance,
    nearest_region_parts,
    pr_auc,
    roc_auc,
)
from grasp_prompt.models import PromptDiffusionModel
from grasp_prompt.training import build_dataloader, load_checkpoint
from grasp_prompt.utils import to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-steps", type=int, default=0, help="0 uses all diffusion steps.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader = build_dataloader(args.data, cfg, split=args.split, shuffle=False)
    model = PromptDiffusionModel(cfg).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    selected = []
    candidates = []
    centers = []
    scales = []
    masks = []
    parts = []
    conf = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, args.device)
            steps = None if args.sample_steps <= 0 else args.sample_steps
            out = model.generate(batch, steps=steps)
            selected.append(out["selected_prompt"].cpu().numpy())
            candidates.append(out["candidates"].cpu().numpy())
            centers.append(batch["label_centers"].cpu().numpy())
            scales.append(batch["label_scales"].cpu().numpy())
            masks.append(batch["label_mask"].cpu().numpy())
            parts.append(batch["label_parts"].cpu().numpy())
            conf.append(out["confidence"].cpu().numpy())
            probs.append(out["part_probs"].cpu().numpy())

    selected_np = np.concatenate(selected, axis=0)
    candidates_np = np.concatenate(candidates, axis=0)
    centers_np = np.concatenate(centers, axis=0)
    scales_np = np.concatenate(scales, axis=0)
    masks_np = np.concatenate(masks, axis=0)
    parts_np = np.concatenate(parts, axis=0)
    conf_np = np.concatenate(conf, axis=0)
    probs_np = np.concatenate(probs, axis=0)

    binary, matched_candidate_parts = candidate_binary_labels(
        candidates_np, centers_np, scales_np, parts_np, masks_np
    )
    matched_selected_parts = nearest_region_parts(selected_np, centers_np, parts_np, masks_np)
    part_to_joint = model.part_to_joint.detach().cpu().numpy()
    target_joint = []
    target_part = []
    for batch in loader:
        target_joint.append(batch["target_joint_mask"].numpy())
        target_part.append(batch["target_part"].numpy())
    target_joint_np = np.concatenate(target_joint, axis=0)
    target_part_np = np.concatenate(target_part, axis=0)

    metrics = {
        "Hit@1": hit_at_1(selected_np, centers_np, scales_np, masks_np),
        "Cover@K": cover_at_k(candidates_np, centers_np, scales_np, masks_np),
        "MPD": mean_prompt_distance(selected_np, centers_np, masks_np),
        "CPA": contact_part_accuracy(probs_np, matched_candidate_parts, binary),
        "ROC-AUC": roc_auc(binary, conf_np),
        "PR-AUC": pr_auc(binary, conf_np),
        "JCS": joint_consistency_score(matched_selected_parts, target_joint_np, part_to_joint),
        "CPS": compensation_penalty_score(matched_selected_parts, target_part_np),
    }
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
