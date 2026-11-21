from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.althia.config import PipelineConfig
from src.althia.constants import TARGET_OBJECTS
from src.althia.pipeline import Detector, VideoAggregation, short_error, write_json_line
from src.althia.video import DecodedFrame, decode_video, decode_video_prefetched


# -----------------------------------------------------------------------------
# GPU Monitor via NVML
# -----------------------------------------------------------------------------
class NVMLGpuMonitor:
    def __init__(self, device_index: int = 0, interval_seconds: float = 0.5) -> None:
        self.device_index = device_index
        self.interval = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._running = False
        self._thread = None
        self._pynvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
        except Exception as exc:
            print(f"[NVMLGpuMonitor] NVML init failed or unavailable: {exc}", file=sys.stderr)

    def start(self) -> None:
        if self._handle is None:
            return
        import threading

        self.samples.clear()
        self._running = True

        def _monitor() -> None:
            while self._running:
                try:
                    rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    power = 0.0
                    try:
                        power = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    except Exception:
                        pass
                    sample = {
                        "timestamp": time.time(),
                        "gpu_util": float(rates.gpu),
                        "mem_util": float(rates.memory),
                        "used_mib": float(mem.used) / (1024 * 1024),
                        "free_mib": float(mem.free) / (1024 * 1024),
                        "total_mib": float(mem.total) / (1024 * 1024),
                        "power_w": power,
                    }
                    self.samples.append(sample)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_monitor, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self.samples)


def compute_gpu_summary(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {
            "gpu_util_mean": 0.0,
            "gpu_util_median": 0.0,
            "gpu_util_p95": 0.0,
            "gpu_util_max": 0.0,
            "fraction_gte_90": 0.0,
            "peak_device_used_mib": 0.0,
            "headroom_pct": 100.0,
            "power_w_mean": 0.0,
            "sample_count": 0,
        }
    utils = [s["gpu_util"] for s in samples]
    used = [s["used_mib"] for s in samples]
    totals = [s["total_mib"] for s in samples]
    powers = [s["power_w"] for s in samples if s.get("power_w", 0) > 0]

    peak_used = float(max(used)) if used else 0.0
    total_mem = float(totals[0]) if totals else 15360.0
    headroom = max(0.0, (total_mem - peak_used) / total_mem * 100.0) if total_mem > 0 else 100.0

    return {
        "gpu_util_mean": round(float(np.mean(utils)), 2),
        "gpu_util_median": round(float(np.median(utils)), 2),
        "gpu_util_p95": round(float(np.percentile(utils, 95)), 2),
        "gpu_util_max": round(float(np.max(utils)), 2),
        "fraction_gte_90": round(float(sum(1 for u in utils if u >= 90.0) / len(utils)), 4),
        "peak_device_used_mib": round(peak_used, 1),
        "headroom_pct": round(headroom, 2),
        "power_w_mean": round(float(np.mean(powers)) if powers else 0.0, 2),
        "sample_count": len(samples),
    }


# -----------------------------------------------------------------------------
# Instrumented Prefetch Decoder with Queue Telemetry
# -----------------------------------------------------------------------------
@dataclass
class QueueTelemetry:
    capacity: int = 64
    samples: list[int] = None  # type: ignore[assignment]
    enqueue_wait_sec: float = 0.0
    dequeue_wait_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.samples is None:
            self.samples = []


def make_profiled_prefetched_decoder(
    thread_type: str | None = None,
    thread_count: int | None = None,
    capacity: int = 64,
    telemetry: QueueTelemetry | None = None,
) -> Callable[[bytes], Iterator[DecodedFrame]]:
    def _decoder(video_bytes: bytes) -> Iterator[DecodedFrame]:
        import queue
        import threading

        sentinel = object()
        q: queue.Queue[object] = queue.Queue(maxsize=capacity)
        exc_box: list[BaseException] = []
        stop_event = threading.Event()

        def producer() -> None:
            try:
                for frame in decode_video(
                    video_bytes, thread_type=thread_type, thread_count=thread_count
                ):
                    if stop_event.is_set():
                        break
                    while not stop_event.is_set():
                        t_wait0 = time.perf_counter()
                        try:
                            q.put(frame, timeout=0.05)
                            if telemetry is not None:
                                telemetry.enqueue_wait_sec += max(0.0, time.perf_counter() - t_wait0)
                                telemetry.samples.append(q.qsize())
                            break
                        except queue.Full:
                            if telemetry is not None:
                                telemetry.enqueue_wait_sec += max(0.0, time.perf_counter() - t_wait0)
                            continue
            except BaseException as exc:
                exc_box.append(exc)
            finally:
                while True:
                    try:
                        q.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            break

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            while True:
                t_deq0 = time.perf_counter()
                item = q.get()
                if telemetry is not None:
                    telemetry.dequeue_wait_sec += max(0.0, time.perf_counter() - t_deq0)
                if item is sentinel:
                    break
                yield item  # type: ignore[misc]
            if exc_box:
                raise exc_box[0]
        finally:
            stop_event.set()
            while t.is_alive():
                try:
                    drain = q.get_nowait()
                    if drain is sentinel:
                        break
                except queue.Empty:
                    time.sleep(0.005)
            t.join(timeout=2.0)

    return _decoder


# -----------------------------------------------------------------------------
# Instrumented Benchmark Detector (Wraps UltralyticsDetector)
# -----------------------------------------------------------------------------
class ProfiledDetector:
    def __init__(self, base_detector: Any) -> None:
        self.detector = base_detector
        self.total_frames = 0
        self.inference_calls = 0
        self.summed_preprocess_ms = 0.0
        self.summed_inference_ms = 0.0
        self.summed_postprocess_ms = 0.0
        self.synchronized_detector_wall_sec = 0.0

    def detect_many(self, images: Sequence[np.ndarray]) -> list[list[int]]:
        import torch

        if not images:
            return []
        self.total_frames += len(images)
        self.inference_calls += 1

        t0 = time.perf_counter()
        results = self.detector.model.predict(
            source=images[0] if len(images) == 1 else list(images),
            imgsz=640,
            conf=0.4,
            iou=0.7,
            device=self.detector.device,
            half=self.detector.half,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.synchronized_detector_wall_sec += time.perf_counter() - t0

        for res in results:
            speed = getattr(res, "speed", {})
            self.summed_preprocess_ms += speed.get("preprocess", 0.0)
            self.summed_inference_ms += speed.get("inference", 0.0)
            self.summed_postprocess_ms += speed.get("postprocess", 0.0)

        detections: list[list[int]] = []
        for result in results:
            if result.boxes is None or result.boxes.cls is None:
                detections.append([])
            else:
                detections.append(
                    [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
                )
        return detections


# -----------------------------------------------------------------------------
# Worker Implementation
# -----------------------------------------------------------------------------
def run_worker(args: argparse.Namespace) -> int:
    import torch
    from scripts.compare_predictions import compare_jsonl_predictions
    from src.althia.inference import UltralyticsDetector

    fixture = Path(args.fixture).resolve()
    expected = Path(args.expected).resolve()
    model = Path(args.model).resolve()
    output_json = Path(args.output_json).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        decoder_strategy=args.decoder,
        decode_thread_type=args.decode_thread_type,
        decode_threads=args.decode_threads,
        prefetch_capacity=args.prefetch_capacity,
        precision=args.precision,
        batch_size=args.batch_size,
    )

    t0_cold = time.perf_counter()

    # Initialize Model
    t0_init = time.perf_counter()
    cuda_device = "0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    base_detector = UltralyticsDetector(
        model_path=model,
        device=cuda_device,
        precision=config.precision,
    )
    init_time_sec = time.perf_counter() - t0_init

    # Model Warmup
    t0_warmup = time.perf_counter()
    warmup_shape = (640, 640, 3)
    dummy_frames = [np.zeros(warmup_shape, dtype=np.uint8) for _ in range(min(config.batch_size, 4))]
    for _ in range(2):
        base_detector.detect_many(dummy_frames)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    warmup_time_sec = time.perf_counter() - t0_warmup

    detector = ProfiledDetector(base_detector)

    # Prepare Decoder
    queue_telemetry = QueueTelemetry(capacity=config.prefetch_capacity)
    if config.decoder_strategy == "D0":
        decoder_func = decode_video
    elif config.decoder_strategy == "D1":
        decoder_func = lambda b: decode_video(
            b,
            thread_type=config.decode_thread_type,
            thread_count=config.decode_threads,
        )
    elif config.decoder_strategy == "D2":
        decoder_func = make_profiled_prefetched_decoder(
            capacity=config.prefetch_capacity,
            telemetry=queue_telemetry,
        )
    elif config.decoder_strategy == "D3":
        decoder_func = make_profiled_prefetched_decoder(
            thread_type=config.decode_thread_type,
            thread_count=config.decode_threads,
            capacity=config.prefetch_capacity,
            telemetry=queue_telemetry,
        )
    else:
        raise ValueError(f"Unknown decoder strategy: {config.decoder_strategy}")

    # Load Parquet
    t0_parquet = time.perf_counter()
    table = pq.read_table(fixture, columns=["mp4"])
    parquet_load_sec = time.perf_counter() - t0_parquet

    max_videos = args.max_videos if args.max_videos and args.max_videos > 0 else len(table)
    video_payloads = [
        (f"{fixture.name}:{i}", table.column(0)[i].as_py())
        for i in range(min(len(table), max_videos))
    ]

    # Start GPU monitor
    gpu_monitor = NVMLGpuMonitor(interval_seconds=0.5)
    gpu_monitor.start()

    # Steady State Execution
    t0_steady = time.perf_counter()
    t0_cpu = os.times()

    aggregation_sec = 0.0
    writing_sec = 0.0
    decode_wall_sec = 0.0
    total_records = 0
    errors = 0

    with output_jsonl.open("w", encoding="utf-8", newline="\n") as out_f:
        for vid_id, payload in video_payloads:
            aggregation = VideoAggregation()
            pending: list[DecodedFrame] = []
            err: str | None = None

            def flush_pending() -> None:
                nonlocal aggregation_sec
                if not pending:
                    return
                t_inf = time.perf_counter()
                dets = detector.detect_many([f.image for f in pending])
                t_agg = time.perf_counter()
                for f, class_ids in zip(pending, dets, strict=True):
                    aggregation.add_frame(f.timestamp, class_ids)
                aggregation_sec += time.perf_counter() - t_agg
                pending.clear()

            t_dec0 = time.perf_counter()
            try:
                for frame in decoder_func(payload):
                    pending.append(frame)
                    if len(pending) >= config.batch_size:
                        flush_pending()
            except Exception as exc:
                err = short_error(exc)
            finally:
                decode_wall_sec += time.perf_counter() - t_dec0

            try:
                flush_pending()
            except Exception as exc:
                err = short_error(exc)

            records = aggregation.records(vid_id)
            if not records and err is None:
                err = "could not decode video"

            t_w0 = time.perf_counter()
            for rec in records:
                write_json_line(out_f, rec)
            if err is not None:
                write_json_line(out_f, {"video": vid_id, "error": err})
                errors += 1
            out_f.flush()
            writing_sec += time.perf_counter() - t_w0
            total_records += len(records)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1_steady = time.perf_counter()
    t1_cpu = os.times()

    steady_wall_sec = t1_steady - t0_steady
    cold_wall_sec = t1_steady - t0_cold

    raw_samples = gpu_monitor.stop()
    gpu_stats = compute_gpu_summary(raw_samples)

    user_cpu = t1_cpu.user - t0_cpu.user
    system_cpu = t1_cpu.system - t0_cpu.system
    total_cpu = user_cpu + system_cpu
    cpu_wall_ratio = total_cpu / steady_wall_sec if steady_wall_sec > 0 else 0.0

    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0
    if torch.cuda.is_available():
        peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # Queue stats
    q_mean = float(np.mean(queue_telemetry.samples)) if queue_telemetry.samples else 0.0
    q_max = int(max(queue_telemetry.samples)) if queue_telemetry.samples else 0

    # JSONL Comparison against Expected
    comp = compare_jsonl_predictions(output_jsonl, expected)

    result_payload: dict[str, Any] = {
        "config_id": args.config_id,
        "decoder": config.decoder_strategy,
        "precision": config.precision,
        "batch_size": config.batch_size,
        "decode_thread_type": config.decode_thread_type,
        "decode_threads": config.decode_threads,
        "prefetch_capacity": config.prefetch_capacity,
        "total_videos": len(video_payloads),
        "total_frames": detector.total_frames,
        "total_records": total_records,
        "errors": errors,
        "init_time_sec": round(init_time_sec, 4),
        "warmup_time_sec": round(warmup_time_sec, 4),
        "steady_wall_sec": round(steady_wall_sec, 4),
        "cold_wall_sec": round(cold_wall_sec, 4),
        "steady_fps": round(detector.total_frames / steady_wall_sec, 2) if steady_wall_sec > 0 else 0.0,
        "inference_fps": round(detector.total_frames / detector.synchronized_detector_wall_sec, 2) if detector.synchronized_detector_wall_sec > 0 else 0.0,
        "parquet_load_sec": round(parquet_load_sec, 4),
        "decode_wall_sec": round(decode_wall_sec, 4),
        "detector_wall_sec": round(detector.synchronized_detector_wall_sec, 4),
        "yolo_preprocess_sec": round(detector.summed_preprocess_ms / 1000.0, 4),
        "yolo_inference_sec": round(detector.summed_inference_ms / 1000.0, 4),
        "yolo_postprocess_sec": round(detector.summed_postprocess_ms / 1000.0, 4),
        "aggregation_sec": round(aggregation_sec, 4),
        "writing_sec": round(writing_sec, 4),
        "q_mean_occupancy": round(q_mean, 1),
        "q_max_occupancy": q_max,
        "q_enqueue_wait_sec": round(queue_telemetry.enqueue_wait_sec, 4),
        "q_dequeue_wait_sec": round(queue_telemetry.dequeue_wait_sec, 4),
        "user_cpu_sec": round(user_cpu, 2),
        "system_cpu_sec": round(system_cpu, 2),
        "total_cpu_sec": round(total_cpu, 2),
        "cpu_wall_ratio": round(cpu_wall_ratio, 2),
        "peak_allocated_mib": round(peak_allocated_mib, 1),
        "peak_reserved_mib": round(peak_reserved_mib, 1),
        "exact_match": comp["exact_match"],
        "differing_records_count": comp["differing_records_count"],
        "exact_record_pct": comp["exact_record_pct"],
        "object_micro_f1": comp["object_micro_f1"],
        "people_exact_pct": comp["people_exact_pct"],
        "people_within_one_pct": comp["people_within_one_pct"],
        "meets_brief_thresholds_vs_ref": comp["meets_brief_thresholds_vs_ref"],
        "actual_assignment_accuracy": "unverified (no ground truth available)",
        "gpu_stats": gpu_stats,
    }

    output_json.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Worker] Config {args.config_id} complete: {detector.total_frames} frames in {steady_wall_sec:.2f}s "
        f"({result_payload['steady_fps']} FPS) | Exact: {comp['exact_match']} | F1: {comp['object_micro_f1']}",
        flush=True,
    )
    return 0


# -----------------------------------------------------------------------------
# Configuration Generation & Driver
# -----------------------------------------------------------------------------
def generate_all_configurations() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    decoders = ["D0", "D1", "D2", "D3"]
    precisions = ["fp32", "fp16"]
    batches = [1, 2, 4, 8, 16, 32]

    idx = 1
    for dec in decoders:
        for prec in precisions:
            for b in batches:
                configs.append(
                    {
                        "config_id": f"cfg_{idx:02d}_{dec}_{prec}_b{b}",
                        "decoder": dec,
                        "precision": prec,
                        "batch_size": b,
                        "decode_thread_type": "AUTO",
                        "decode_threads": 2,
                        "prefetch_capacity": 64,
                    }
                )
                idx += 1
    return configs


def write_configurations_csv(configs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "decoder",
        "precision",
        "batch_size",
        "decode_thread_type",
        "decode_threads",
        "prefetch_capacity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in configs:
            writer.writerow({k: c[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="T4 Comprehensive Optimization Experiment Runner")
    parser.add_argument("--worker", action="store_true", help="Run isolated worker mode")
    parser.add_argument("--config-id", type=str, default="cfg_01")
    parser.add_argument("--decoder", choices=["D0", "D1", "D2", "D3"], default="D0")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-thread-type", choices=["AUTO", "FRAME"], default="AUTO")
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--prefetch-capacity", type=int, default=64)
    parser.add_argument("--fixture", type=Path, default=Path("artifacts/reference/input/reference.parquet"))
    parser.add_argument("--expected", type=Path, default=Path("reference/expected.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("yolo11l.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/optimization/worker_result.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("artifacts/optimization/worker_output.jsonl"))
    parser.add_argument("--max-videos", type=int, default=0)

    # Driver arguments
    parser.add_argument("--driver", action="store_true", help="Run full orchestrating driver")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/optimization"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/optimization"))
    parser.add_argument("--screen-only", action="store_true", help="Stop after screening runs")

    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    if args.driver:
        from scripts.benchmark_configurations_driver import run_driver
        return run_driver(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
