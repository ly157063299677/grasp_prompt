from __future__ import annotations

import argparse
from pathlib import Path


ABLATION_CONFIGS = [
    "configs/affordpose_main.yaml",
    "configs/ablation_no_valid_mask.yaml",
    "configs/ablation_no_target_condition.yaml",
    "configs/ablation_no_tda.yaml",
    "configs/ablation_no_cdci.yaml",
    "configs/ablation_no_confidence.yaml",
    "configs/ablation_no_joint_guidance.yaml",
    "configs/ablation_no_snms.yaml",
]


def run_name(config: str) -> str:
    return Path(config).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=0)
    args = parser.parse_args()

    for config in ABLATION_CONFIGS:
        name = run_name(config)
        run_dir = f"{args.runs}/{name}"
        print(f"python scripts/train.py --config {config} --data {args.data} --run-dir {run_dir} --device {args.device}")
        print(
            "python scripts/evaluate.py "
            f"--config {config} --data {args.data} "
            f"--checkpoint {run_dir}/checkpoint_best.pt --split test "
            f"--sample-steps {args.sample_steps} --device {args.device} "
            f"--out {run_dir}/metrics_test.json"
        )


if __name__ == "__main__":
    main()

