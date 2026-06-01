from __future__ import annotations

import argparse
import csv
from pathlib import Path


def contiguous_rows(num_parts: int, num_joints: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for part in range(num_parts):
        start = int(round(part * num_joints / num_parts))
        end = int(round((part + 1) * num_joints / num_parts))
        end = max(end, start + 1)
        rows.append([1 if start <= joint < min(end, num_joints) else 0 for joint in range(num_joints)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-parts", type=int, default=8)
    parser.add_argument("--num-joints", type=int, default=21)
    parser.add_argument("--with-header", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = contiguous_rows(args.num_parts, args.num_joints)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if args.with_header:
            writer.writerow(["part"] + [f"joint_{idx:02d}" for idx in range(args.num_joints)])
            for idx, row in enumerate(rows):
                writer.writerow([f"part_{idx:02d}", *row])
        else:
            writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()

