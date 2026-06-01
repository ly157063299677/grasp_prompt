from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.config import load_config
from grasp_prompt.data import GraspPromptDataset, collate_prompt_batch
from grasp_prompt.models import PromptDiffusionModel
from grasp_prompt.training import load_checkpoint
from grasp_prompt.utils import to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--sample-steps", type=int, default=0, help="0 uses all diffusion steps.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sample_path = Path(args.sample).resolve()
    split = sample_path.parent.name
    data_root = sample_path.parent.parent
    cfg = load_config(args.config)
    dataset = GraspPromptDataset(
        data_root,
        split=split,
        max_prompts=int(cfg["data"]["max_prompts"]),
        max_regions=int(cfg["data"]["max_regions"]),
        seed=int(cfg["seed"]),
    )
    try:
        idx = dataset.files.index(sample_path)
    except ValueError as exc:
        raise FileNotFoundError(f"{sample_path} is not in {data_root / split}") from exc
    batch = collate_prompt_batch([dataset[idx]])
    batch = to_device(batch, args.device)

    model = PromptDiffusionModel(cfg).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()
    with torch.no_grad():
        steps = None if args.sample_steps <= 0 else args.sample_steps
        out = model.generate(batch, steps=steps)
    result = {
        "sample": str(sample_path),
        "selected_prompt": out["selected_prompt"][0].detach().cpu().tolist(),
        "selected_part": int(out["selected_part"][0].detach().cpu()),
        "confidence": float(out["selected_confidence"][0].detach().cpu()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
