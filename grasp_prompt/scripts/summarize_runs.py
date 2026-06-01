from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_ORDER = ["Hit@1", "Cover@K", "MPD", "CPA", "ROC-AUC", "PR-AUC", "JCS", "CPS"]


def read_metrics(path: Path) -> dict[str, float | str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, float | str] = {"run": path.parent.name, "file": str(path)}
    for key in METRIC_ORDER:
        row[key] = data.get(key, "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Metric JSON files or run directories.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metric_files: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            metric_files.extend(sorted(path.glob("**/metrics*.json")))
        else:
            metric_files.append(path)
    rows = [read_metrics(path) for path in metric_files]
    fields = ["run", *METRIC_ORDER, "file"]

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))


if __name__ == "__main__":
    main()

