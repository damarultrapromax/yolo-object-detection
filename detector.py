from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo26n.pt"


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class ObjectDetector:
    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        conf: float = 0.35,
        imgsz: int = 640,
        device: str = "auto",
    ) -> None:
        self.model = YOLO(str(model))
        self.conf = conf
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.names = self.model.names

    def predict(self, source, stream: bool = True, classes: list[int] | None = None):
        return self.model.predict(
            source=source,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=classes,
            stream=stream,
            verbose=False,
        )

    def counts(self, result) -> Counter:
        counter: Counter = Counter()
        if result.boxes is None:
            return counter
        for box in result.boxes:
            counter[result.names[int(box.cls[0])]] += 1
        return counter

    def details(self, result) -> list[str]:
        if result.boxes is None:
            return []
        return [
            f"{result.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            for box in result.boxes
        ]


def annotate_frame(frame, counts: Counter, fps: float | None = None):
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    lines.extend(f"{name}: {count}" for name, count in counts.most_common(6))
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    pad = 8
    line_height = 24
    text_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    box_width = text_width + pad * 2
    box_height = line_height * len(lines) + pad

    frame = frame.copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, line in enumerate(lines):
        baseline = 10 + pad + line_height * index + 16
        cv2.putText(
            frame,
            line,
            (10 + pad, baseline),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame
