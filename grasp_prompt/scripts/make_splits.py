from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_key(path: Path, field: str) -> str:
    with np.load(path, allow_pickle=True) as item:
        if field not in item.files:
            return path.stem
        value = np.asarray(item[field])
        if value.shape == ():
            return str(value.item())
        return str(value.reshape(-1)[0]) if value.size else path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing processed .npz samples.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-key", default="object_id")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating hard links.")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    files = sorted(in_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files found in {in_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[read_key(path, args.group_key)].append(path)

    keys = np.array(sorted(groups))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(keys)
    n_train = int(round(len(keys) * args.train_ratio))
    n_val = int(round(len(keys) * args.val_ratio))
    split_keys = {
        "train": set(keys[:n_train]),
        "val": set(keys[n_train : n_train + n_val]),
        "test": set(keys[n_train + n_val :]),
    }

    for split in split_keys:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    counters = {"train": 0, "val": 0, "test": 0}
    for split, key_set in split_keys.items():
        for key in sorted(key_set):
            for src in groups[key]:
                dst = out_dir / split / f"sample_{counters[split]:06d}.npz"
                counters[split] += 1
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)

    print("groups:", len(keys))
    print("samples:", counters)


if __name__ == "__main__":
    main()

