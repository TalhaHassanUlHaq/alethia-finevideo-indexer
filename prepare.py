from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.althia.constants import TARGET_CLASS_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate the YOLO11l weights before an offline run."
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11l.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model))
    names = model.names
    mismatches = {
        class_id: (expected, names.get(class_id))
        for class_id, expected in TARGET_CLASS_IDS.items()
        if names.get(class_id) != expected
    }
    if mismatches:
        raise RuntimeError(f"unexpected model class map: {mismatches}")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Ultralytics loaded the model but {args.model} was not created"
        )
    print(f"Prepared {args.model.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

