from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.config import load_config
from grasp_prompt.models import PromptDiffusionModel
from grasp_prompt.training import (
    build_dataloader,
    load_checkpoint,
    loss_epoch,
    save_checkpoint,
    train_one_epoch,
)
from grasp_prompt.utils import ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    set_seed(int(cfg["seed"]))
    run_dir = ensure_dir(args.run_dir)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    train_loader = build_dataloader(args.data, cfg, split="train", shuffle=True)
    val_loader = None
    if (Path(args.data) / "val").exists():
        val_loader = build_dataloader(args.data, cfg, split="val", shuffle=False)

    model = PromptDiffusionModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    start_epoch = 1
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=optimizer, map_location=args.device)
        start_epoch = int(payload.get("epoch", 0)) + 1

    best_val = float("inf")
    history_path = run_dir / "history.jsonl"
    for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            grad_clip_norm=float(cfg["training"].get("grad_clip_norm", 0.0)),
        )
        record = {"epoch": epoch, "train": train_stats}
        msg = f"epoch={epoch} train_loss={train_stats['loss']:.5f}"
        if val_loader is not None:
            val_stats = loss_epoch(model, val_loader, args.device)
            record["val"] = val_stats
            msg += f" val_loss={val_stats['loss']:.5f}"
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, cfg, epoch)
        print(msg)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        save_checkpoint(run_dir / "checkpoint_last.pt", model, optimizer, cfg, epoch)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch:04d}.pt", model, optimizer, cfg, epoch)


if __name__ == "__main__":
    main()
