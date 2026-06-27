"""
VisionGuard – API Server
FastAPI REST endpoints + WebSocket live alert stream.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.db.store import DetectionEvent, DetectionStore
from src.detection.detector import FrameResult
from src.streaming.pipeline import CameraConfig, CameraPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
store = DetectionStore(dsn=None)          # set DSN via env for real Postgres
pipeline: Optional[CameraPipeline] = None
websocket_clients: Set[WebSocket] = set()


async def broadcast_alert(payload: dict):
    """Push detection alert to all connected WebSocket clients."""
    dead = set()
    for ws in websocket_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    websocket_clients.difference_update(dead)


def on_frame_result(result: FrameResult):
    """Callback invoked by pipeline for every processed frame."""
    if not result.detections:
        return
    for det in result.detections:
        event = DetectionEvent(
            camera_id=result.camera_id,
            frame_id=result.frame_id,
            label=det.label,
            confidence=det.confidence,
            bbox_x1=det.bbox[0], bbox_y1=det.bbox[1],
            bbox_x2=det.bbox[2], bbox_y2=det.bbox[3],
            inference_ms=result.inference_ms,
            timestamp=result.timestamp,
        )
        asyncio.run_coroutine_threadsafe(store.insert(event), asyncio.get_event_loop())
        if det.confidence >= 0.6:
            alert = {
                "type": "alert",
                "camera_id": result.camera_id,
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "inference_ms": result.inference_ms,
                "timestamp": result.timestamp,
            }
            asyncio.run_coroutine_threadsafe(broadcast_alert(alert), asyncio.get_event_loop())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    await store.connect()

    # Demo: 4 stub cameras (replace source with RTSP URLs or device indices in production)
    cameras = [
        CameraConfig(camera_id=f"cam-{i}", source=0, fps_target=10)
        for i in range(4)
    ]
    pipeline = CameraPipeline(
        camera_configs=cameras,
        on_result=on_frame_result,
        num_detector_workers=2,
        use_stub_detector=True,
    )
    pipeline.start()
    logger.info("[API] VisionGuard started — 4 cameras, 2 detector workers")
    yield
    pipeline.stop()
    await store.close()


app = FastAPI(
    title="VisionGuard – Real-Time Object Detection",
    description="Multi-camera surveillance with YOLOv8, WebSocket alerts, and PostgreSQL event history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/pipeline/stats")
async def pipeline_stats():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not running")
    return pipeline.queue_stats()


@app.get("/detections/recent")
async def recent_detections(limit: int = 50, camera_id: Optional[str] = None):
    return await store.query_recent(limit=limit, camera_id=camera_id)


@app.get("/detections/stats")
async def detection_stats():
    return await store.query_stats()


# ── WebSocket Live Alerts ─────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    logger.info(f"[WS] Client connected | total={len(websocket_clients)}")
    try:
        while True:
            # Keep connection alive; server pushes alerts via broadcast_alert()
            await asyncio.sleep(10)
            await websocket.send_json({"type": "ping", "timestamp": time.time()})
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
        logger.info(f"[WS] Client disconnected | total={len(websocket_clients)}")


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8001, reload=False)
