"""
VisionGuard – Multi-threaded Camera Capture Pipeline
Producer-consumer architecture: each camera runs a dedicated capture thread
that feeds frames into a shared queue consumed by detector workers.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.detector import FrameResult, YOLOv8Detector

logger = logging.getLogger(__name__)

FRAME_QUEUE_MAXSIZE = 64  # prevent unbounded memory usage


@dataclass
class CameraConfig:
    camera_id: str
    source: str | int  # file path, RTSP URL, or device index
    fps_target: int = 15
    resolution: Tuple[int, int] = (640, 480)


class CameraCapture(threading.Thread):
    """
    Producer thread: continuously reads frames from a camera source
    and pushes them onto the shared frame queue.
    Drops frames if the queue is full (non-blocking put).
    """

    def __init__(self, config: CameraConfig, frame_queue: queue.Queue):
        super().__init__(daemon=True, name=f"capture-{config.camera_id}")
        self.config = config
        self.frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._dropped_frames = 0

    def run(self):
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            logger.warning(f"[Capture {self.config.camera_id}] Could not open source — using blank frames")
            cap = None

        interval = 1.0 / self.config.fps_target
        logger.info(f"[Capture {self.config.camera_id}] Started | source={self.config.source}")

        while not self._stop_event.is_set():
            t0 = time.time()

            if cap:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[Capture {self.config.camera_id}] Frame read failed — reconnecting...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.config.source)
                    continue
                frame = cv2.resize(frame, self.config.resolution)
            else:
                # Stub: blank grey frame with timestamp text
                frame = np.full((*self.config.resolution[::-1], 3), 128, dtype=np.uint8)
                cv2.putText(frame, f"CAM {self.config.camera_id} | stub | {time.strftime('%H:%M:%S')}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self._frame_count += 1
            payload = (self.config.camera_id, self._frame_count, frame)
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                self._dropped_frames += 1

            elapsed = time.time() - t0
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

        if cap:
            cap.release()
        logger.info(f"[Capture {self.config.camera_id}] Stopped | frames={self._frame_count} dropped={self._dropped_frames}")

    def stop(self):
        self._stop_event.set()


class DetectorWorker(threading.Thread):
    """
    Consumer thread: pulls (camera_id, frame_id, frame) tuples from the queue,
    runs YOLOv8 inference, and dispatches results via a callback.
    """

    def __init__(self, worker_id: str, frame_queue: queue.Queue,
                 detector: YOLOv8Detector, on_result: Callable[[FrameResult], None]):
        super().__init__(daemon=True, name=f"detector-{worker_id}")
        self.worker_id = worker_id
        self.frame_queue = frame_queue
        self.detector = detector
        self.on_result = on_result
        self._stop_event = threading.Event()
        self._processed = 0

    def run(self):
        logger.info(f"[Detector Worker {self.worker_id}] Started")
        while not self._stop_event.is_set():
            try:
                camera_id, frame_id, frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            detections = self.detector.infer(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            result = FrameResult(
                camera_id=camera_id,
                frame_id=frame_id,
                detections=detections,
                inference_ms=round(inference_ms, 2),
            )
            self._processed += 1
            self.frame_queue.task_done()
            self.on_result(result)

            if self._processed % 100 == 0:
                logger.debug(f"[Detector Worker {self.worker_id}] {self._processed} frames processed")

    def stop(self):
        self._stop_event.set()


class CameraPipeline:
    """
    Orchestrates N camera capture threads + M detector worker threads.
    Shared frame queue with producer-consumer pattern.
    """

    def __init__(self, camera_configs: List[CameraConfig],
                 on_result: Callable[[FrameResult], None],
                 num_detector_workers: int = 2,
                 use_stub_detector: bool = True):
        self.camera_configs = camera_configs
        self.on_result = on_result
        self.num_detector_workers = num_detector_workers

        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._detector = YOLOv8Detector(use_stub=use_stub_detector)
        self._capture_threads: List[CameraCapture] = []
        self._detector_threads: List[DetectorWorker] = []

    def start(self):
        # Start detector workers first (consumers ready before producers)
        for i in range(self.num_detector_workers):
            t = DetectorWorker(
                worker_id=str(i),
                frame_queue=self._frame_queue,
                detector=self._detector,
                on_result=self.on_result,
            )
            t.start()
            self._detector_threads.append(t)

        # Start camera capture threads
        for config in self.camera_configs:
            t = CameraCapture(config=config, frame_queue=self._frame_queue)
            t.start()
            self._capture_threads.append(t)

        logger.info(
            f"[Pipeline] Started | cameras={len(self.camera_configs)} "
            f"detector_workers={self.num_detector_workers}"
        )

    def stop(self):
        for t in self._capture_threads:
            t.stop()
        for t in self._capture_threads:
            t.join(timeout=3.0)
        for t in self._detector_threads:
            t.stop()
        for t in self._detector_threads:
            t.join(timeout=3.0)
        logger.info("[Pipeline] All threads stopped")

    def queue_stats(self) -> dict:
        return {
            "queue_size": self._frame_queue.qsize(),
            "queue_maxsize": FRAME_QUEUE_MAXSIZE,
            "cameras": len(self._capture_threads),
            "detector_workers": len(self._detector_threads),
        }
