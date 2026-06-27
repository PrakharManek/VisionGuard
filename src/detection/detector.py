"""
VisionGuard – Object Detection Core
Runs YOLOv8 inference on frames captured from multiple camera feeds.
Optimised for low-latency CPU inference via model quantization.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detection result for a single object in a frame
@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameResult:
    camera_id: str
    frame_id: int
    detections: List[Detection]
    inference_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def has_alerts(self) -> bool:
        """True if any detection exceeds alert confidence threshold."""
        return any(d.confidence >= 0.6 for d in self.detections)


class YOLOv8Detector:
    """
    Wraps YOLOv8 for real-time object detection.
    Falls back to a stub (random detections) when ultralytics is not installed,
    so the rest of the pipeline can be tested without GPU/model weights.
    """

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    ]

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.45, use_stub: bool = False):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.use_stub = use_stub
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.use_stub:
            logger.warning("[Detector] Running in STUB mode — no real model loaded")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"[Detector] YOLOv8 loaded from {self.model_path}")
        except ImportError:
            logger.warning("[Detector] ultralytics not installed — falling back to stub mode")
            self.use_stub = True

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if self.use_stub:
            return self._stub_infer(frame)
        return self._real_infer(frame)

    def _real_infer(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.COCO_CLASSES[cls] if cls < len(self.COCO_CLASSES) else f"class_{cls}"
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections

    def _stub_infer(self, frame: np.ndarray) -> List[Detection]:
        """Returns 0-2 random detections for pipeline testing without real weights."""
        h, w = frame.shape[:2]
        detections = []
        n = np.random.randint(0, 3)
        for _ in range(n):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = x1 + np.random.randint(50, w // 4)
            y2 = y1 + np.random.randint(50, h // 4)
            conf = round(np.random.uniform(0.45, 0.95), 3)
            label = np.random.choice(["person", "car", "dog"])
            detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections


def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (0, 255, 0) if det.confidence >= 0.7 else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, label_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated
