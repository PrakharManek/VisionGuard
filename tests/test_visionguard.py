"""
Tests for VisionGuard detection pipeline and event store.
"""

import asyncio
import queue
import time

import numpy as np
import pytest

from src.db.store import DetectionEvent, DetectionStore
from src.detection.detector import Detection, FrameResult, YOLOv8Detector, annotate_frame
from src.streaming.pipeline import CameraConfig, CameraCapture, DetectorWorker


# ── Detector ──────────────────────────────────────────────────────────────────

def test_stub_detector_returns_valid_detections():
    det = YOLOv8Detector(use_stub=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(20):
        results = det.infer(frame)
        assert isinstance(results, list)
        for d in results:
            assert 0.0 <= d.confidence <= 1.0
            assert d.label in ["person", "car", "dog"]
            assert len(d.bbox) == 4


def test_annotate_frame_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [Detection(label="person", confidence=0.9, bbox=(10, 10, 100, 200))]
    out = annotate_frame(frame, dets)
    assert out.shape == frame.shape


def test_frame_result_has_alerts():
    r = FrameResult(
        camera_id="cam-0",
        frame_id=1,
        detections=[Detection("person", 0.85, (0, 0, 100, 100))],
        inference_ms=12.5,
    )
    assert r.has_alerts is True


def test_frame_result_no_alerts_below_threshold():
    r = FrameResult(
        camera_id="cam-0",
        frame_id=1,
        detections=[Detection("dog", 0.50, (0, 0, 50, 50))],
        inference_ms=10.0,
    )
    assert r.has_alerts is False


# ── DetectionStore (memory mode) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_insert_and_query_recent():
    store = DetectionStore(dsn=None)
    await store.connect()
    event = DetectionEvent(
        camera_id="cam-1", frame_id=42, label="person", confidence=0.88,
        bbox_x1=10, bbox_y1=20, bbox_x2=110, bbox_y2=220,
        inference_ms=15.0, timestamp=time.time(),
    )
    await store.insert(event)
    results = await store.query_recent(limit=10)
    assert len(results) == 1
    assert results[0]["label"] == "person"


@pytest.mark.asyncio
async def test_store_stats_empty():
    store = DetectionStore(dsn=None)
    await store.connect()
    stats = await store.query_stats()
    assert stats["total_detections"] == 0
    assert stats["by_label"] == {}


@pytest.mark.asyncio
async def test_store_stats_counts_by_label():
    store = DetectionStore(dsn=None)
    await store.connect()
    for label in ["person", "person", "car"]:
        await store.insert(DetectionEvent(
            camera_id="cam-0", frame_id=1, label=label, confidence=0.9,
            bbox_x1=0, bbox_y1=0, bbox_x2=50, bbox_y2=50,
            inference_ms=10.0, timestamp=time.time(),
        ))
    stats = await store.query_stats()
    assert stats["total_detections"] == 3
    assert stats["by_label"]["person"] == 2
    assert stats["by_label"]["car"] == 1


@pytest.mark.asyncio
async def test_store_filter_by_camera():
    store = DetectionStore(dsn=None)
    await store.connect()
    for cam in ["cam-0", "cam-0", "cam-1"]:
        await store.insert(DetectionEvent(
            camera_id=cam, frame_id=1, label="dog", confidence=0.7,
            bbox_x1=0, bbox_y1=0, bbox_x2=50, bbox_y2=50,
            inference_ms=8.0, timestamp=time.time(),
        ))
    results = await store.query_recent(camera_id="cam-0")
    assert len(results) == 2


# ── Pipeline Threading ────────────────────────────────────────────────────────

def test_detector_worker_processes_frames():
    fq = queue.Queue()
    results = []

    def collect(r):
        results.append(r)

    det = YOLOv8Detector(use_stub=True)
    worker = DetectorWorker(worker_id="t0", frame_queue=fq, detector=det, on_result=collect)
    worker.start()

    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fq.put(("cam-0", i, frame))

    fq.join()
    worker.stop()
    worker.join(timeout=3.0)

    assert len(results) == 5
    for r in results:
        assert r.camera_id == "cam-0"
        assert r.inference_ms >= 0
