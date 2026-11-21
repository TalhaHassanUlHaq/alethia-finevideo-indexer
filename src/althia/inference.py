from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from ultralytics import YOLO

from .constants import TARGET_CLASS_IDS


class UltralyticsDetector:
    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        precision: str = "fp32",
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"model weights not found: {model_path}; run python prepare.py first"
            )
        self.device = "0" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"invalid precision: {precision!r}")
        if precision == "fp16" and self.device == "cpu":
            raise RuntimeError("FP16 inference requires a CUDA device, but device is cpu")
        self.precision = precision
        self.half = precision == "fp16"
        self.model = YOLO(str(model_path))
        mismatches = {
            class_id: (expected, self.model.names.get(class_id))
            for class_id, expected in TARGET_CLASS_IDS.items()
            if self.model.names.get(class_id) != expected
        }
        if mismatches:
            raise RuntimeError(f"unexpected model class map: {mismatches}")

    def detect_many(self, images: Sequence[np.ndarray]) -> list[list[int]]:
        if not images:
            return []
        source: object = images[0] if len(images) == 1 else list(images)
        results = self.model.predict(
            source=source,
            imgsz=640,
            conf=0.4,
            iou=0.7,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        detections: list[list[int]] = []
        for result in results:
            if result.boxes is None or result.boxes.cls is None:
                detections.append([])
            else:
                detections.append(
                    [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
                )
        if len(detections) != len(images):
            raise RuntimeError(
                f"model returned {len(detections)} results for {len(images)} frames"
            )
        return detections
