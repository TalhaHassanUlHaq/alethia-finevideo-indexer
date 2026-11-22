from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alethia.validation import validate_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate assignment JSONL output.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--no-errors", action="store_true")
    args = parser.parse_args()
    report = validate_jsonl(
        args.path,
        expected_videos=args.expected_videos,
        allow_errors=not args.no_errors,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

