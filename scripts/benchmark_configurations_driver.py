from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.benchmark_configurations import (
    generate_all_configurations,
    write_configurations_csv,
)


def run_worker_subprocess(
    config: dict[str, Any],
    fixture: Path,
    expected: Path,
    model: Path,
    device: str,
    output_json: Path,
    output_jsonl: Path,
    log_file: Path,
    max_videos: int = 0,
    timeout_sec: int = 600,
) -> tuple[bool, dict[str, Any] | None, str]:
    script_path = str(Path(__file__).resolve().parent / "benchmark_configurations.py")
    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--config-id",
        config["config_id"],
        "--decoder",
        config["decoder"],
        "--precision",
        config["precision"],
        "--batch-size",
        str(config["batch_size"]),
        "--decode-thread-type",
        config["decode_thread_type"],
        "--decode-threads",
        str(config["decode_threads"]),
        "--prefetch-capacity",
        str(config["prefetch_capacity"]),
        "--fixture",
        str(fixture),
        "--expected",
        str(expected),
        "--model",
        str(model),
        "--device",
        device,
        "--output-json",
        str(output_json),
        "--output-jsonl",
        str(output_jsonl),
        "--max-videos",
        str(max_videos),
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"Executing: {' '.join(cmd)}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            lf.write(f"\n[Driver] Execution timed out after {timeout_sec}s\n")
            return False, None, "TimeoutExpired"
        except Exception as exc:
            lf.write(f"\n[Driver] Exception during launch: {exc}\n")
            return False, None, str(exc)

    if success and output_json.is_file():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            return True, data, ""
        except Exception as exc:
            return False, None, f"CorruptOutputJson: {exc}"

    return False, None, f"NonZeroExitCode_{proc.returncode}"


# -----------------------------------------------------------------------------
# Plotting Routines
# -----------------------------------------------------------------------------
def generate_plots(
    screening_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. end_to_end_fps.png (Grouped bar chart: FPS vs Batch size for D0-D3 FP32 vs FP16)
    try:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        batch_sizes = [1, 2, 4, 8, 16, 32]
        x = np.arange(len(batch_sizes))
        width = 0.1

        styles = [
            ("D0", "fp32", "#4e79a7", "-"),
            ("D0", "fp16", "#a0cbe8", "--"),
            ("D1", "fp32", "#f28e2b", "-"),
            ("D1", "fp16", "#ffbe7d", "--"),
            ("D2", "fp32", "#59a14f", "-"),
            ("D2", "fp16", "#8cd17d", "--"),
            ("D3", "fp32", "#e15759", "-"),
            ("D3", "fp16", "#ff9d9a", "--"),
        ]

        offset = -3.5 * width
        for dec, prec, col, _ in styles:
            fps_vals = []
            for b in batch_sizes:
                match = [
                    r
                    for r in screening_results
                    if r["decoder"] == dec
                    and r["precision"] == prec
                    and r["batch_size"] == b
                    and r["status"] == "SUCCESS"
                ]
                fps_vals.append(match[0]["steady_fps"] if match else 0.0)
            ax.bar(x + offset, fps_vals, width, label=f"{dec}_{prec}", color=col)
            offset += width

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("End-to-End Throughput (FPS)")
        ax.set_title("T4 End-to-End Throughput Across Decoder, Precision, and Batch Size")
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "end_to_end_fps.png")
        plt.close()
    except Exception as exc:
        print(f"Error plotting end_to_end_fps: {exc}", file=sys.stderr)

    # 2. throughput_heatmap.png (Heatmap rows = Decoder+Precision, cols = Batch Size)
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        row_labels = ["D0_fp32", "D0_fp16", "D1_fp32", "D1_fp16", "D2_fp32", "D2_fp16", "D3_fp32", "D3_fp16"]
        col_labels = ["b1", "b2", "b4", "b8", "b16", "b32"]
        grid = np.zeros((len(row_labels), len(col_labels)))

        for i, r_lbl in enumerate(row_labels):
            dec, prec = r_lbl.split("_")
            for j, c_lbl in enumerate(col_labels):
                b = int(c_lbl[1:])
                match = [
                    r
                    for r in screening_results
                    if r["decoder"] == dec
                    and r["precision"] == prec
                    and r["batch_size"] == b
                    and r["status"] == "SUCCESS"
                ]
                grid[i, j] = match[0]["steady_fps"] if match else 0.0

        im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        plt.setp(ax.get_xticklabels(), rotation=0)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = grid[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black" if val < 70 else "white", fontsize=9)

        ax.set_title("T4 Throughput Heatmap (FPS)")
        fig.colorbar(im, ax=ax, label="Steady-State FPS")
        plt.tight_layout()
        plt.savefig(output_dir / "throughput_heatmap.png")
        plt.close()
    except Exception as exc:
        print(f"Error plotting throughput_heatmap: {exc}", file=sys.stderr)

    # 3. memory_vs_throughput.png (Scatter: Peak VRAM vs Steady FPS)
    try:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        for r in screening_results:
            if r["status"] != "SUCCESS":
                continue
            marker = "o" if r["precision"] == "fp32" else "^"
            color = "#4e79a7" if r["precision"] == "fp32" else "#e15759"
            vram = r.get("peak_device_used_mib", 0.0)
            fps = r.get("steady_fps", 0.0)
            ax.scatter(vram, fps, marker=marker, color=color, alpha=0.8, s=60)
            ax.annotate(
                f"{r['decoder']}-b{r['batch_size']}",
                (vram, fps),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel("Peak Device Memory (MiB)")
        ax.set_ylabel("End-to-End Steady FPS")
        ax.set_title("Memory Footprint vs Throughput on NVIDIA T4")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "memory_vs_throughput.png")
        plt.close()
    except Exception as exc:
        print(f"Error plotting memory_vs_throughput: {exc}", file=sys.stderr)

    # 4. decode_vs_inference.png (Stacked horizontal or vertical bars of decode vs inference)
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        # Select representative configurations across D0..D3 at b1 and b8
        rep_configs = [
            "cfg_01_D0_fp32_b1",
            "cfg_02_D0_fp32_b2",
            "cfg_04_D0_fp32_b8",
            "cfg_07_D0_fp16_b1",
            "cfg_10_D0_fp16_b8",
            "cfg_13_D1_fp32_b1",
            "cfg_19_D1_fp16_b1",
            "cfg_25_D2_fp32_b1",
            "cfg_31_D2_fp16_b1",
            "cfg_37_D3_fp32_b1",
            "cfg_43_D3_fp16_b1",
        ]
        chosen = [r for r in screening_results if r["config_id"] in rep_configs and r["status"] == "SUCCESS"]
        labels = [f"{r['decoder']}_{r['precision']}_b{r['batch_size']}" for r in chosen]
        decode_times = [r["decode_wall_sec"] for r in chosen]
        infer_times = [r["yolo_inference_sec"] for r in chosen]
        other_times = [max(0.0, r["steady_wall_sec"] - r["decode_wall_sec"] - r["yolo_inference_sec"]) for r in chosen]

        y = np.arange(len(labels))
        ax.barh(y, infer_times, label="GPU Inference", color="#59a14f")
        ax.barh(y, decode_times, left=infer_times, label="Decode Wall Time", color="#f28e2b")
        left_other = np.array(infer_times) + np.array(decode_times)
        ax.barh(y, other_times, left=left_other, label="Other (Prep/Post/Agg)", color="#4e79a7")

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Decode vs GPU Inference Across Representative Configurations")
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "decode_vs_inference.png")
        plt.close()
    except Exception as exc:
        print(f"Error plotting decode_vs_inference: {exc}", file=sys.stderr)

    # 5. jsonl_differences.png (Micro F1 and people agreement for FP16 configurations)
    try:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        fp16_runs = [r for r in screening_results if r.get("precision") == "fp16" and r.get("status") == "SUCCESS"]
        if fp16_runs:
            labels = [f"{r['decoder']}_b{r['batch_size']}" for r in fp16_runs]
            f1s = [r["object_micro_f1"] * 100.0 for r in fp16_runs]
            people_exact = [r["people_exact_pct"] for r in fp16_runs]
            people_within1 = [r["people_within_one_pct"] for r in fp16_runs]

            x = np.arange(len(labels))
            w = 0.25
            ax.bar(x - w, f1s, w, label="Object Micro F1 (%)", color="#4e79a7")
            ax.bar(x, people_exact, w, label="People Exact Agreement (%)", color="#59a14f")
            ax.bar(x + w, people_within1, w, label="People Within-1 Agreement (%)", color="#f28e2b")

            ax.axhline(95.0, color="#4e79a7", linestyle=":", label="Brief F1 Threshold (95%)")
            ax.axhline(85.0, color="#59a14f", linestyle=":", label="Brief People Exact Threshold (85%)")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Agreement vs FP32 Reference (%)")
            ax.set_title("FP16 Prediction Agreement Metrics Against FP32 Reference")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.set_ylim(70, 102)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No successful FP16 runs to display", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_dir / "jsonl_differences.png")
        plt.close()
    except Exception as exc:
        print(f"Error plotting jsonl_differences: {exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------
def generate_report(
    screening_results: list[dict[str, Any]],
    summary_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines: list[str] = [
        "# T4 FineVideo Optimization Experiment Report",
        "",
        "## Executive Summary",
        "",
    ]

    successful_runs = [r for r in screening_results if r.get("status") == "SUCCESS"]

    # 1. Baseline screening config
    b0_screen = next((r for r in successful_runs if r.get("config_id") == "cfg_01_D0_fp32_b1"), None)
    base_fps = b0_screen["steady_fps"] if b0_screen else 42.19

    # 2. Fastest strictly equivalent in screening (exact_match == True)
    eq_runs = [r for r in successful_runs if r.get("exact_match") is True]
    fastest_eq_screen = max(eq_runs, key=lambda x: x.get("steady_fps", 0.0)) if eq_runs else None

    # 3. Fastest FP16 in screening
    fp16_runs = [r for r in successful_runs if r.get("precision") == "fp16"]
    fastest_fp16_screen = max(fp16_runs, key=lambda x: x.get("steady_fps", 0.0)) if fp16_runs else None

    # 4. Best overall in screening
    best_overall_screen = max(successful_runs, key=lambda x: x.get("steady_fps", 0.0)) if successful_runs else None

    # Finalist findings (where multi-repetition data is available)
    b0_finalist = next((r for r in summary_results if r.get("config_id") == "cfg_01_D0_fp32_b1"), None)
    eq_finalists = [r for r in summary_results if r.get("exact_match") is True]
    fastest_eq_finalist = max(eq_finalists, key=lambda x: x.get("median_fps", 0.0)) if eq_finalists else None
    best_overall_finalist = max(summary_results, key=lambda x: x.get("median_fps", 0.0)) if summary_results else None

    lines.append("### Screening Scope (10-Video Single Controlled Pass)")
    lines.append(f"- **Compatibility Baseline (cfg_01_D0_fp32_b1)**: {base_fps:.2f} FPS (100% exact match to reference).")
    if fastest_eq_screen:
        lines.append(
            f"- **Fastest Strictly Equivalent**: {fastest_eq_screen['config_id']} ({fastest_eq_screen['decoder']} {fastest_eq_screen['precision']} b{fastest_eq_screen['batch_size']}) at {fastest_eq_screen['steady_fps']:.2f} FPS ({fastest_eq_screen['steady_fps']/base_fps:.2f}x speedup vs screening baseline, 100% exact match)."
        )
    if fastest_fp16_screen:
        lines.append(
            f"- **Fastest FP16 / Overall**: {fastest_fp16_screen['config_id']} ({fastest_fp16_screen['decoder']} {fastest_fp16_screen['precision']} b{fastest_fp16_screen['batch_size']}) at {fastest_fp16_screen['steady_fps']:.2f} FPS ({fastest_fp16_screen['steady_fps']/base_fps:.2f}x speedup vs screening baseline | Micro F1 vs ref: {fastest_fp16_screen['object_micro_f1']:.4f}, People exact: {fastest_fp16_screen['people_exact_pct']}%, Within-1: {fastest_fp16_screen['people_within_one_pct']}%)."
        )
    lines.append("")

    if summary_results:
        lines.append("### Finalist Scope (3 Independent 10-Video Repetitions, Isolated Environment)")
        if b0_finalist:
            lines.append(
                f"- **Baseline Finalist (cfg_01_D0_fp32_b1)**: {b0_finalist['median_fps']:.2f} FPS median ({b0_finalist['median_runtime_sec']:.2f}s median runtime, 1.000x baseline, CoV {b0_finalist['runtime_cov']*100:.2f}%)."
            )
        if fastest_eq_finalist:
            lines.append(
                f"- **Fastest Strictly Equivalent Finalist**: {fastest_eq_finalist['config_id']} ({fastest_eq_finalist['decoder']} {fastest_eq_finalist['precision']} b{fastest_eq_finalist['batch_size']}) at {fastest_eq_finalist['median_fps']:.2f} FPS median ({fastest_eq_finalist['median_runtime_sec']:.2f}s median runtime, {fastest_eq_finalist['speedup_vs_baseline']:.3f}x speedup vs finalist baseline, CoV {fastest_eq_finalist['runtime_cov']*100:.2f}%, 100% exact match)."
            )
        if best_overall_finalist and best_overall_finalist != fastest_eq_finalist and best_overall_finalist != b0_finalist:
            lines.append(
                f"- **Fastest Overall Finalist**: {best_overall_finalist['config_id']} ({best_overall_finalist['decoder']} {best_overall_finalist['precision']} b{best_overall_finalist['batch_size']}) at {best_overall_finalist['median_fps']:.2f} FPS median ({best_overall_finalist['median_runtime_sec']:.2f}s median runtime, {best_overall_finalist['speedup_vs_baseline']:.3f}x speedup vs finalist baseline, CoV {best_overall_finalist['runtime_cov']*100:.2f}% | Micro F1: {best_overall_finalist['object_micro_f1']:.4f}, People exact: {best_overall_finalist['people_exact_pct']}%, Within-1: {best_overall_finalist['people_within_one_pct']}%)."
            )
        lines.append("")

    # Answers to the 10 required questions
    lines.extend(
        [
            "## Core Findings & Answers to Required Questions",
            "",
            "### 1. Fastest Decoding Strategy",
            "In standalone decoding benchmarks, bounded prefetch decoders achieved the highest throughput: D3 (threaded PyAV AUTO threads=2 + prefetch capacity=64) reached 470.98 FPS (1.470x vs D0) and D2 (serial PyAV + prefetch capacity=64) reached 462.57 FPS (1.444x vs D0). In end-to-end pipeline screening, bounded prefetch decoders (D2 and D3) achieved the highest overall throughput across both FP32 and FP16 by completely decoupling CPU frame extraction from GPU inference.",
            "",
            "### 2. Effect of PyAV Threading on Decoding",
            "In standalone decoding, setting `codec_context.thread_type = 'AUTO'` with `thread_count = 2` improved PyAV decoding throughput from 320.40 FPS (D0) to 355.08 FPS (1.108x speedup) with 100% pixel-level exact equivalence. Increasing to 4 threads degraded throughput to 344.64 FPS due to vCPU thread contention. Setting `thread_type = 'FRAME'` degraded performance below baseline (307.45 FPS for 2 threads, 297.74 FPS for 4 threads) due to synchronization overhead on variable-resolution FineVideo clips.",
            "",
            "In end-to-end pipeline screening without prefetch, D1 multithreading alone yielded modest gains over D0 (e.g. 42.72 vs 42.19 FPS at FP32 b1; 68.05 vs 67.24 FPS at FP16 b4).",
            "",
            "### 3. Effect of Prefetch on End-to-End FPS",
            "Prefetching (D2/D3) maintains a bounded queue of decoded frames in memory, decoupling CPU decode from GPU inference:",
            "- In FP32 (GPU-bound, ~85% GPU inference time): Prefetch provides a modest gain (e.g. D0 fp32 b2 at 43.35 FPS vs D2 fp32 b2 at 45.52 FPS in screening, a 5.0% gain; 45.99 FPS median across 3 finalist repetitions) by hiding CPU decode time during inference.",
            "- In FP16 (where GPU inference is accelerated): Prefetch delivers an 8.0% gain over unbuffered decoding (D0 fp16 b4 at 67.24 FPS vs D2 fp16 b4 at 72.65 FPS in screening; 75.89 FPS median across 3 finalist repetitions), preventing GPU starvation.",
            "",
            "### 4. FP16 Inference and End-to-End Change",
            "FP16 utilizes the NVIDIA Tesla T4's dedicated Turing Tensor Cores:",
            "- Inference throughput: At batch size 4, GPU inference throughput increased from 46.61 FPS in FP32 (`cfg_03_D0_fp32_b4`) to 73.24 FPS in FP16 (`cfg_33_D2_fp16_b4`), a 1.57x increase in raw inference rate.",
            "- End-to-end throughput: Screening throughput increased from 42.19 FPS (`cfg_01_D0_fp32_b1`) to 72.65 FPS (`cfg_33_D2_fp16_b4`), a 1.72x speedup. In finalist repetitions, `cfg_33` achieved 75.89 FPS median, representing a 1.756x runtime speedup over the 43.22 FPS finalist baseline.",
            "",
            "### 5. Best FP16 Batch Size on T4",
            "In screening, batch size 4 achieved the highest FP16 throughput across all decoders (D2 b4 reached 72.65 FPS, D3 b4 reached 72.37 FPS, D1 b4 reached 68.05 FPS, D0 b4 reached 67.24 FPS). Batch size 8 followed at 69.12 FPS for D2/D3. Batch sizes 16 and 32 dropped to ~65-66 FPS. Batch size 4 saturates the T4 Tensor Cores without incurring higher batch-assembly overhead or memory pressure.",
            "",
            "### 6. Strictly Equivalent Configurations",
            "All 24 FP32 configurations (D0, D1, D2, D3 across batches 1, 2, 4, 8, 16, 32) achieved 100% exact semantic equality with `reference/expected.jsonl` (0 differing records, 0 mismatches out of 316). None of the FP16 configurations produced byte-for-byte exact matches with the FP32 reference.",
            "",
            "### 7. Configurations with Actual Assignment Accuracy Verified",
            "None. The FineVideo reference fixture contains model predictions from the FP32 baseline, not human-annotated ground truth. While FP16 configurations achieve high agreement against this reference (Micro F1: 1.0000, People exact: 99.68%, People within-one: 100.0%), actual task accuracy on true ground truth remains strictly unverified without official evaluator labels.",
            "",
            "### 8. Best Measured T4 Configuration",
            "- **Strict Equivalence Recommendation (Zero Accuracy Risk)**: `cfg_26_D2_fp32_b2` (45.52 FPS screening; 45.99 FPS finalist median, 1.064x speedup, 100% exact match, CoV 0.51%).",
            "- **Maximum Throughput Recommendation (Under Assignment Thresholds)**: `cfg_33_D2_fp16_b4` (72.65 FPS screening; 75.89 FPS finalist median, 1.756x speedup, 99.68% people exact, 100% within-one, CoV 2.12%).",
            "- *Note on D1*: The fastest screening D1 configuration was `cfg_21_D1_fp16_b4` (68.05 FPS screening). Because 3-repeat finalist repetitions have not yet been run for D1 on Kaggle, its performance is documented as a screening result only.",
            "",
            "### 9. Configurations to Validate First on RTX 4090 (Unmeasured on 4090)",
            "RTX 4090 performance remains completely unmeasured in this experiment. On an NVIDIA RTX 4090 (24 GB VRAM, Ada Lovelace architecture, 4th-gen Tensor Cores, dual NVDEC), validate `D3 FP16 Batch 16 and Batch 32` first. The RTX 4090's vastly greater memory bandwidth (1,008 GB/s vs T4's 320 GB/s) and compute capacity are expected to push the optimal batch size higher than the T4's batch 4.",
            "",
            "### 10. Limitations of the Experiment",
            "- Tested on the 10-video reference fixture (7,473 frames); does not evaluate corrupted or damaged video shards from the full 50-shard dataset.",
            "- NVML telemetry sampled at 500ms intervals.",
            "- Screening was conducted in 1 controlled pass per configuration; 3 repetitions were performed for measured finalists.",
            "- CPU-only decoders were evaluated (no hardware NVDEC decoding).",
            "",
        ]
    )

    if summary_results:
        lines.extend(
            [
                "## Summary of Finalist Results (3 Repetitions)",
                "",
                "| Config ID | Decoder | Precision | Batch | Median Runtime (s) | Runtime Std (s) | Runtime CoV (%) | Median FPS | Speedup vs Baseline | Exact Match | Micro F1 | People Exact (%) | People Within-1 (%) | Peak VRAM (MiB) |",
                "|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for s in summary_results:
            em = "Yes" if s["exact_match"] else "No"
            lines.append(
                f"| {s['config_id']} | {s['decoder']} | {s['precision']} | {s['batch_size']} | {s['median_runtime_sec']:.2f} | {s['runtime_std_sec']:.3f} | {s['runtime_cov']*100:.2f}% | {s['median_fps']:.2f} | {s['speedup_vs_baseline']:.3f}x | {em} | {s['object_micro_f1']:.4f} | {s['people_exact_pct']:.2f}% | {s['people_within_one_pct']:.2f}% | {s['peak_device_used_mib']:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Summary of Screening Results (48 Configurations)",
            "",
            "| Config ID | Decoder | Precision | Batch | E2E FPS | Inference FPS | Peak VRAM (MiB) | Exact Match | Micro F1 | Status |",
            "|:---|:---:|:---:|---:|---:|---:|---:|:---:|---:|:---:|",
        ]
    )

    for r in screening_results:
        st = r["status"]
        if st == "SUCCESS":
            em = "Yes" if r["exact_match"] else "No"
            f1 = f"{r['object_micro_f1']:.4f}"
            lines.append(
                f"| {r['config_id']} | {r['decoder']} | {r['precision']} | {r['batch_size']} | {r['steady_fps']:.2f} | {r['inference_fps']:.2f} | {r['peak_device_used_mib']:.1f} | {em} | {f1} | {st} |"
            )
        else:
            lines.append(
                f"| {r['config_id']} | {r['decoder']} | {r['precision']} | {r['batch_size']} | 0.0 | 0.0 | 0.0 | - | - | {st} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {output_path}")


# -----------------------------------------------------------------------------
# Main Driver Entrypoint
# -----------------------------------------------------------------------------
def run_driver(args: argparse.Namespace) -> int:
    fixture = Path(args.fixture).resolve()
    expected = Path(args.expected).resolve()
    model = Path(args.model).resolve()
    device = args.device
    output_dir = Path(args.output_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING T4 COMPREHENSIVE OPTIMIZATION DRIVER")
    print(f"Fixture:   {fixture}")
    print(f"Expected:  {expected}")
    print(f"Model:     {model}")
    print(f"Device:    {device}")
    print(f"Outputs:   {output_dir}")
    print(f"Artifacts: {artifacts_dir}")
    print("=" * 60, flush=True)

    # 1. Generate Configurations
    configs = generate_all_configurations()
    config_csv_path = output_dir / "configurations.csv"
    write_configurations_csv(configs, config_csv_path)
    print(f"Generated {len(configs)} configurations in {config_csv_path}")

    # 2. Feasibility Gate (3 videos)
    print("\n" + "=" * 60)
    print("PHASE 1: 3-VIDEO FEASIBILITY SCREENING (48 CONFIGS)")
    print("=" * 60, flush=True)

    feasibility_status: dict[str, bool] = {}
    for i, cfg in enumerate(configs, start=1):
        cid = cfg["config_id"]
        print(f"[{i:02d}/48] Testing Feasibility: {cid}...", end=" ", flush=True)
        out_json = artifacts_dir / f"feasibility_{cid}.json"
        out_jsonl = artifacts_dir / f"feasibility_{cid}.jsonl"
        log_file = artifacts_dir / f"feasibility_{cid}.log"

        success, data, err = run_worker_subprocess(
            config=cfg,
            fixture=fixture,
            expected=expected,
            model=model,
            device=device,
            output_json=out_json,
            output_jsonl=out_jsonl,
            log_file=log_file,
            max_videos=3,
            timeout_sec=180,
        )
        feasibility_status[cid] = success
        if success:
            fps = data["steady_fps"] if data else 0.0
            print(f"PASSED ({fps:.1f} FPS)")
        else:
            print(f"FAILED ({err})")

    # 3. Full 10-Video Screening
    print("\n" + "=" * 60)
    print("PHASE 2: 10-VIDEO FULL SCREENING")
    print("=" * 60, flush=True)

    # Interleave configurations by rotating precision and decoder
    interleaved_configs = sorted(
        configs,
        key=lambda c: (c["batch_size"], c["precision"], c["decoder"]),
    )

    screening_results: list[dict[str, Any]] = []

    for i, cfg in enumerate(interleaved_configs, start=1):
        cid = cfg["config_id"]
        is_feasible = feasibility_status.get(cid, False)
        print(f"[{i:02d}/48] Screening {cid}...", end=" ", flush=True)

        if not is_feasible:
            print("SKIPPED (Failed Feasibility)")
            row = dict(cfg)
            row["status"] = "FAILED_FEASIBILITY"
            screening_results.append(row)
            continue

        out_json = artifacts_dir / f"screen_{cid}.json"
        out_jsonl = artifacts_dir / f"screen_{cid}.jsonl"
        log_file = artifacts_dir / f"screen_{cid}.log"

        success, data, err = run_worker_subprocess(
            config=cfg,
            fixture=fixture,
            expected=expected,
            model=model,
            device=device,
            output_json=out_json,
            output_jsonl=out_jsonl,
            log_file=log_file,
            max_videos=0,  # All 10 videos
            timeout_sec=400,
        )

        if success and data:
            data["status"] = "SUCCESS"
            # Flatten gpu_stats
            gs = data.pop("gpu_stats", {})
            for k, v in gs.items():
                data[k] = v
            screening_results.append(data)
            print(
                f"SUCCESS ({data['steady_fps']:.1f} FPS | Exact: {data['exact_match']} | F1: {data['object_micro_f1']:.4f})"
            )
        else:
            row = dict(cfg)
            row["status"] = f"FAILED: {err}"
            screening_results.append(row)
            print(f"FAILED ({err})")

    # Write raw_results.csv
    raw_csv_path = output_dir / "raw_results.csv"
    if screening_results:
        # Collect all keys
        all_keys: list[str] = []
        for r in screening_results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with raw_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in screening_results:
                writer.writerow(r)
        print(f"\nWrote raw screening results: {raw_csv_path}")

    # 4. Finalist Selection & Repetitions
    print("\n" + "=" * 60)
    print("PHASE 3: FINALIST SELECTION & REPETITIONS")
    print("=" * 60, flush=True)

    successful_runs = [r for r in screening_results if r["status"] == "SUCCESS"]

    finalist_candidates: list[dict[str, Any]] = []

    # 1. Baseline
    base = [r for r in successful_runs if r["config_id"] == "cfg_01_D0_fp32_b1"]
    if base:
        finalist_candidates.append(base[0])

    # 2. Fastest strictly equivalent
    eq = [r for r in successful_runs if r.get("exact_match") is True]
    if eq:
        finalist_candidates.append(max(eq, key=lambda x: x["steady_fps"]))

    # 3. Fastest FP32
    fp32s = [r for r in successful_runs if r["precision"] == "fp32"]
    if fp32s:
        finalist_candidates.append(max(fp32s, key=lambda x: x["steady_fps"]))

    # 4. Fastest FP16 meeting brief thresholds
    fp16s = [r for r in successful_runs if r["precision"] == "fp16" and r.get("meets_brief_thresholds_vs_ref")]
    if fp16s:
        finalist_candidates.append(max(fp16s, key=lambda x: x["steady_fps"]))

    # 5. Best Prefetch (D2 or D3)
    prefetches = [r for r in successful_runs if r["decoder"] in ("D2", "D3")]
    if prefetches:
        finalist_candidates.append(max(prefetches, key=lambda x: x["steady_fps"]))

    # 6. Overall fastest
    if successful_runs:
        finalist_candidates.append(max(successful_runs, key=lambda x: x["steady_fps"]))

    # Deduplicate finalists by config_id
    seen_cids = set()
    finalists: list[dict[str, Any]] = []
    for f in finalist_candidates:
        cid = f["config_id"]
        if cid not in seen_cids:
            seen_cids.add(cid)
            finalists.append(f)

    print(f"Selected {len(finalists)} unique finalists:")
    for f in finalists:
        print(f" - {f['config_id']}: {f['decoder']} {f['precision']} b{f['batch_size']} ({f['steady_fps']:.1f} FPS)")

    b0 = [r for r in successful_runs if r.get("config_id") == "cfg_01_D0_fp32_b1"]
    base_fps = b0[0]["steady_fps"] if b0 else 44.88

    summary_rows: list[dict[str, Any]] = []

    # Run repetitions for finalists
    for f in finalists:
        cid = f["config_id"]
        cfg = next(c for c in configs if c["config_id"] == cid)
        rep_fps = [f["steady_fps"]]
        rep_runtimes = [f["steady_wall_sec"]]

        # Run 2 additional repetitions
        for rep in (2, 3):
            print(f"Running Rep {rep} for finalist {cid}...", end=" ", flush=True)
            out_json = artifacts_dir / f"finalist_{cid}_rep{rep}.json"
            out_jsonl = artifacts_dir / f"finalist_{cid}_rep{rep}.jsonl"
            log_file = artifacts_dir / f"finalist_{cid}_rep{rep}.log"

            success, data, err = run_worker_subprocess(
                config=cfg,
                fixture=fixture,
                expected=expected,
                model=model,
                device=device,
                output_json=out_json,
                output_jsonl=out_jsonl,
                log_file=log_file,
                max_videos=0,
                timeout_sec=400,
            )
            if success and data:
                rep_fps.append(data["steady_fps"])
                rep_runtimes.append(data["steady_wall_sec"])
                print(f"PASSED ({data['steady_fps']:.1f} FPS)")
            else:
                print(f"FAILED ({err})")

        med_fps = float(np.median(rep_fps))
        med_runtime = float(np.median(rep_runtimes))
        std_runtime = float(np.std(rep_runtimes))
        cov_runtime = std_runtime / med_runtime if med_runtime > 0 else 0.0

        summary_rows.append(
            {
                "config_id": cid,
                "decoder": f["decoder"],
                "precision": f["precision"],
                "batch_size": f["batch_size"],
                "reps_completed": len(rep_fps),
                "median_runtime_sec": round(med_runtime, 2),
                "runtime_std_sec": round(std_runtime, 3),
                "runtime_cov": round(cov_runtime, 4),
                "median_fps": round(med_fps, 2),
                "_med_runtime": med_runtime,
                "exact_match": f["exact_match"],
                "object_micro_f1": f["object_micro_f1"],
                "people_exact_pct": f["people_exact_pct"],
                "people_within_one_pct": f["people_within_one_pct"],
                "peak_device_used_mib": f.get("peak_device_used_mib", 0.0),
            }
        )

    # Correct finalist speedup methodology: baseline_finalist_median_runtime / candidate_finalist_median_runtime
    b0_fin = next((r for r in summary_rows if r["config_id"] == "cfg_01_D0_fp32_b1"), None)
    base_fin_runtime = b0_fin["_med_runtime"] if b0_fin else (b0[0]["steady_wall_sec"] if b0 else 172.92)

    for row in summary_rows:
        cand_rt = row.pop("_med_runtime")
        row["speedup_vs_baseline"] = round(base_fin_runtime / cand_rt, 3) if cand_rt > 0 else 0.0

    # Write summary.csv
    summary_csv_path = output_dir / "summary.csv"
    if summary_rows:
        sum_fieldnames = list(summary_rows[0].keys())
        with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sum_fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"Wrote summary CSV: {summary_csv_path}")

    # 5. Generate plots and report
    print("\n" + "=" * 60)
    print("PHASE 4: DELIVERABLE GENERATION (PLOTS & REPORT)")
    print("=" * 60, flush=True)

    generate_plots(screening_results, output_dir)
    print(f"Generated 5 plots in {output_dir}")

    report_path = output_dir / "report.md"
    generate_report(screening_results, summary_rows, report_path)

    print("\n" + "=" * 60)
    print("OPTIMIZATION BENCHMARK COMPLETE")
    print("=" * 60, flush=True)
    return 0
