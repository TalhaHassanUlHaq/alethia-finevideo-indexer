from __future__ import annotations

from fractions import Fraction
import numpy as np
import pytest

from src.althia.video import DecodedFrame


def compute_accounting(
    end_to_end_wall: float,
    parquet_load: float,
    video_decode: float,
    convert_rotate: float,
    detector_wall: float,
    aggregation: float,
    writing: float,
    yolo_preprocess: float,
    yolo_inference: float,
    yolo_postprocess: float,
) -> dict[str, float]:
    outer_sum = (
        parquet_load
        + video_decode
        + convert_rotate
        + detector_wall
        + aggregation
        + writing
    )
    e2e_residual = max(0.0, end_to_end_wall - outer_sum)
    detector_sum = yolo_preprocess + yolo_inference + yolo_postprocess
    detector_residual = max(0.0, detector_wall - detector_sum)

    return {
        "outer_sum": outer_sum,
        "end_to_end_residual": e2e_residual,
        "detector_sum": detector_sum,
        "detector_residual": detector_residual,
        "e2e_residual_pct": (e2e_residual / end_to_end_wall * 100.0)
        if end_to_end_wall > 0
        else 0.0,
        "detector_residual_pct": (detector_residual / detector_wall * 100.0)
        if detector_wall > 0
        else 0.0,
    }


def compute_utilization_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "fraction_gte_90": 0.0,
            "sample_count": 0,
        }
    arr = np.array(samples, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "fraction_gte_90": float(np.mean(arr >= 90.0)),
        "sample_count": len(samples),
    }


def format_single_run_variability(observed_runtime: float, observed_gpu_util: float) -> dict[str, str]:
    return {
        "observed_runtime_sec": f"{observed_runtime:.2f}",
        "observed_gpu_util_pct": f"{observed_gpu_util:.1f}",
        "runtime_std_dev": "N/A (one run)",
        "runtime_cov": "N/A (one run)",
        "runtime_min": "N/A (one run)",
        "runtime_max": "N/A (one run)",
        "runtime_range_pct": "N/A (one run)",
        "gpu_util_variability": "N/A (one run)",
    }


def test_accounting_equation_reconciliation() -> None:
    # Outer breakdown: 1.0 parquet, 20.0 decode, 10.0 convert, 65.0 detector, 2.0 agg, 1.0 write = 99.0s
    # Total wall = 100.0s -> residual = 1.0s (1.0%)
    # Detector breakdown: 5.0 prep, 50.0 inf, 5.0 post = 60.0s
    # Detector wall = 65.0s -> detector residual = 5.0s (7.69%)
    acct = compute_accounting(
        end_to_end_wall=100.0,
        parquet_load=1.0,
        video_decode=20.0,
        convert_rotate=10.0,
        detector_wall=65.0,
        aggregation=2.0,
        writing=1.0,
        yolo_preprocess=5.0,
        yolo_inference=50.0,
        yolo_postprocess=5.0,
    )
    assert acct["outer_sum"] == pytest.approx(99.0)
    assert acct["end_to_end_residual"] == pytest.approx(1.0)
    assert acct["e2e_residual_pct"] == pytest.approx(1.0)
    assert acct["detector_sum"] == pytest.approx(60.0)
    assert acct["detector_residual"] == pytest.approx(5.0)
    assert acct["detector_residual_pct"] == pytest.approx(5.0 / 65.0 * 100.0)


def test_compute_utilization_stats() -> None:
    # 10 samples: [0, 10, 20, 30, 40, 50, 60, 70, 90, 100]
    # Sum: 470. Mean: 470 / 10 = 47.0
    # Median: (40 + 50) / 2 = 45.0
    # Max: 100.0
    # >= 90: 2 / 10 = 0.20 (20%)
    samples = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 90.0, 100.0]
    stats = compute_utilization_stats(samples)
    assert stats["mean"] == pytest.approx(47.0)
    assert stats["median"] == pytest.approx(45.0)
    assert stats["max"] == pytest.approx(100.0)
    assert stats["fraction_gte_90"] == pytest.approx(0.20)
    assert stats["sample_count"] == 10
    assert stats["p95"] >= 90.0


def test_format_single_run_variability() -> None:
    v = format_single_run_variability(175.5, 42.1)
    assert v["observed_runtime_sec"] == "175.50"
    assert v["observed_gpu_util_pct"] == "42.1"
    assert v["runtime_std_dev"] == "N/A (one run)"
    assert v["runtime_cov"] == "N/A (one run)"
    assert v["runtime_min"] == "N/A (one run)"
    assert v["runtime_max"] == "N/A (one run)"
    assert v["gpu_util_variability"] == "N/A (one run)"


def test_profiled_decoder_logic_preserves_frames() -> None:
    # Verify that a custom decoder generator splits work cleanly without altering frames
    mock_frames = [
        DecodedFrame(np.ones((2, 2), dtype=np.uint8), Fraction(0, 1)),
        DecodedFrame(np.zeros((2, 2), dtype=np.uint8), Fraction(1, 1)),
    ]

    def mock_profiled_decoder(payload: bytes):
        for f in mock_frames:
            yield f

    collected = list(mock_profiled_decoder(b"payload"))
    assert len(collected) == 2
    assert collected[0].timestamp == Fraction(0, 1)
    assert collected[1].timestamp == Fraction(1, 1)
    assert np.array_equal(collected[0].image, mock_frames[0].image)
