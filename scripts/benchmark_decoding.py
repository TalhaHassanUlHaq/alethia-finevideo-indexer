from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import av
import numpy as np
import pyarrow.parquet as pq

from src.althia.video import DecodedFrame, decode_video, decode_video_prefetched, frame_timestamp, rotate_bgr


def extract_videos_from_parquet(parquet_path: Path) -> list[tuple[str, bytes]]:
    table = pq.read_table(parquet_path, columns=["mp4"])
    videos: list[tuple[str, bytes]] = []
    for i in range(len(table)):
        vid_id = f"{parquet_path.name}:{i}"
        payload = table.column(0)[i].as_py()
        videos.append((vid_id, payload))
    return videos


def benchmark_stream(
    videos: list[tuple[str, bytes]],
    decoder_func: Callable[[bytes], Iterator[DecodedFrame]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    total_frames = 0
    video_summaries: list[dict[str, Any]] = []

    for vid_id, vid_bytes in videos:
        frame_count = 0
        h = hashlib.sha256()
        first_ts: Fraction | None = None
        last_ts: Fraction | None = None
        shapes: set[tuple[int, ...]] = set()
        dtypes: set[str] = set()

        for frame in decoder_func(vid_bytes):
            if first_ts is None:
                first_ts = frame.timestamp
            last_ts = frame.timestamp
            shapes.add(frame.image.shape)
            dtypes.add(str(frame.image.dtype))
            h.update(str(frame.timestamp).encode())
            h.update(frame.image.tobytes())
            frame_count += 1

        total_frames += frame_count
        video_summaries.append(
            {
                "video_id": vid_id,
                "frame_count": frame_count,
                "first_ts": str(first_ts),
                "last_ts": str(last_ts),
                "shapes": [list(s) for s in shapes],
                "dtypes": list(dtypes),
                "pixel_digest": h.hexdigest(),
            }
        )

    elapsed = time.perf_counter() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    return {
        "wall_seconds": round(elapsed, 4),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "video_summaries": video_summaries,
    }


def run_decode_benchmark(fixture_path: Path, output_json: Path | None = None) -> dict[str, Any]:
    print(f"Loading videos from fixture: {fixture_path}")
    videos = extract_videos_from_parquet(fixture_path)
    print(f"Loaded {len(videos)} videos.")

    # 1. Baseline D0
    print("\n--- Running D0 (Serial PyAV Baseline) ---")
    d0_res = benchmark_stream(videos, decode_video)
    print(f"D0: {d0_res['total_frames']} frames in {d0_res['wall_seconds']:.2f}s ({d0_res['fps']:.1f} FPS)")

    d0_hashes = [v["pixel_digest"] for v in d0_res["video_summaries"]]
    d0_counts = [v["frame_count"] for v in d0_res["video_summaries"]]

    # 2. D1 Candidates
    d1_candidates = [
        ("AUTO", 2),
        ("AUTO", 4),
        ("FRAME", 2),
        ("FRAME", 4),
    ]

    candidate_results: dict[str, Any] = {}
    equivalent_candidates: list[tuple[str, str, int, float, float]] = []

    for thread_type, thread_count in d1_candidates:
        name = f"D1_{thread_type}_t{thread_count}"
        print(f"\n--- Running {name} ---")
        try:
            res = benchmark_stream(
                videos,
                lambda b, tt=thread_type, tc=thread_count: decode_video(b, thread_type=tt, thread_count=tc),
            )
            cand_hashes = [v["pixel_digest"] for v in res["video_summaries"]]
            cand_counts = [v["frame_count"] for v in res["video_summaries"]]
            exact_match = (cand_hashes == d0_hashes and cand_counts == d0_counts)
            speedup = res["fps"] / d0_res["fps"] if d0_res["fps"] > 0 else 1.0

            res["exact_match_d0"] = exact_match
            res["speedup_vs_d0"] = round(speedup, 3)
            candidate_results[name] = res

            print(f"{name}: {res['total_frames']} frames in {res['wall_seconds']:.2f}s ({res['fps']:.1f} FPS, {speedup:.2f}x vs D0) - Exact match: {exact_match}")
            if exact_match:
                equivalent_candidates.append((name, thread_type, thread_count, res["fps"], res["wall_seconds"]))
        except Exception as exc:
            print(f"{name} failed or unsupported: {exc}")
            candidate_results[name] = {"supported": False, "error": str(exc)}

    # Selection logic:
    # Pick fastest equivalent candidate. If within 5%, pick simpler/lower-thread setting.
    if equivalent_candidates:
        # Sort descending by FPS
        equivalent_candidates.sort(key=lambda x: x[3], reverse=True)
        fastest = equivalent_candidates[0]
        # Check if another equivalent candidate with lower thread count is within 5% of fastest
        selected = fastest
        for cand in equivalent_candidates[1:]:
            # If cand has lower thread count and is within 5% of fastest FPS
            if cand[2] < selected[2] and (fastest[3] - cand[3]) / fastest[3] <= 0.05:
                selected = cand
        selected_setting = {
            "name": selected[0],
            "thread_type": selected[1],
            "thread_count": selected[2],
            "fps": selected[3],
            "speedup_vs_d0": round(selected[3] / d0_res["fps"], 3),
        }
    else:
        selected_setting = {"name": "D0_fallback", "thread_type": "AUTO", "thread_count": 0}

    print(f"\nSelected D1 deterministic setting: {selected_setting['name']} (thread_type={selected_setting.get('thread_type')}, thread_count={selected_setting.get('thread_count')})")

    # 3. D2 (D0 + Prefetch capacity 64)
    print("\n--- Running D2 (D0 + Bounded Prefetch) ---")
    d2_res = benchmark_stream(videos, lambda b: decode_video_prefetched(b, capacity=64))
    d2_hashes = [v["pixel_digest"] for v in d2_res["video_summaries"]]
    d2_exact = (d2_hashes == d0_hashes)
    d2_res["exact_match_d0"] = d2_exact
    d2_res["speedup_vs_d0"] = round(d2_res["fps"] / d0_res["fps"], 3)
    print(f"D2: {d2_res['total_frames']} frames in {d2_res['wall_seconds']:.2f}s ({d2_res['fps']:.1f} FPS, {d2_res['speedup_vs_d0']:.2f}x) - Match D0: {d2_exact}")

    # 4. D3 (Selected D1 + Prefetch capacity 64)
    sel_tt = selected_setting.get("thread_type", "AUTO")
    sel_tc = selected_setting.get("thread_count", 2)
    print(f"\n--- Running D3 (D1[{sel_tt}, t{sel_tc}] + Bounded Prefetch) ---")
    d3_res = benchmark_stream(
        videos,
        lambda b: decode_video_prefetched(b, thread_type=sel_tt, thread_count=sel_tc, capacity=64),
    )
    d3_hashes = [v["pixel_digest"] for v in d3_res["video_summaries"]]
    d3_exact = (d3_hashes == d0_hashes)
    d3_res["exact_match_d0"] = d3_exact
    d3_res["speedup_vs_d0"] = round(d3_res["fps"] / d0_res["fps"], 3)
    print(f"D3: {d3_res['total_frames']} frames in {d3_res['wall_seconds']:.2f}s ({d3_res['fps']:.1f} FPS, {d3_res['speedup_vs_d0']:.2f}x) - Match D0: {d3_exact}")

    full_report = {
        "fixture": str(fixture_path),
        "total_videos": len(videos),
        "d0_baseline": d0_res,
        "d1_candidates": candidate_results,
        "selected_d1_setting": selected_setting,
        "d2_prefetch": d2_res,
        "d3_threaded_prefetch": d3_res,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote decode benchmark report: {output_json}")

    return full_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and verify PyAV decode strategies")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/reference/input/reference.parquet"),
        help="Path to reference parquet fixture",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    run_decode_benchmark(args.fixture, args.output)
    return 0


if __name__ == "__main__":
    main()
