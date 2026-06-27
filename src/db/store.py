"""
VisionGuard – Detection Event Store
Persists detection events to PostgreSQL for historical querying and dashboard data.
Uses asyncpg for non-blocking DB operations.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    camera_id: str
    frame_id: int
    label: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    inference_ms: float
    timestamp: float


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detection_events (
    id              BIGSERIAL PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    frame_id        INTEGER NOT NULL,
    label           TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox_x1         INTEGER,
    bbox_y1         INTEGER,
    bbox_x2         INTEGER,
    bbox_y2         INTEGER,
    inference_ms    REAL,
    timestamp       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_det_camera ON detection_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_det_label ON detection_events (label);
CREATE INDEX IF NOT EXISTS idx_det_timestamp ON detection_events (timestamp DESC);
"""


class DetectionStore:
    """
    Async PostgreSQL store for detection events.
    Falls back to in-memory storage when DB is unavailable (for local dev/testing).
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn
        self._pool = None
        self._memory_store: List[DetectionEvent] = []
        self._use_memory = dsn is None

    async def connect(self):
        if self._use_memory:
            logger.warning("[Store] No DSN provided — using in-memory store")
            return
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(CREATE_TABLE_SQL)
            logger.info("[Store] Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"[Store] DB connection failed: {e} — falling back to memory")
            self._use_memory = True

    async def insert(self, event: DetectionEvent):
        if self._use_memory:
            self._memory_store.append(event)
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO detection_events
                  (camera_id, frame_id, label, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, inference_ms, timestamp)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                event.camera_id, event.frame_id, event.label, event.confidence,
                event.bbox_x1, event.bbox_y1, event.bbox_x2, event.bbox_y2,
                event.inference_ms, event.timestamp,
            )

    async def query_recent(self, limit: int = 50, camera_id: Optional[str] = None) -> List[dict]:
        if self._use_memory:
            results = self._memory_store
            if camera_id:
                results = [e for e in results if e.camera_id == camera_id]
            return [e.__dict__ for e in sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]]

        filters = "WHERE camera_id = $2" if camera_id else ""
        params = [limit, camera_id] if camera_id else [limit]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM detection_events {filters} ORDER BY timestamp DESC LIMIT $1",
                *params,
            )
        return [dict(r) for r in rows]

    async def query_stats(self) -> dict:
        if self._use_memory:
            total = len(self._memory_store)
            labels = {}
            for e in self._memory_store:
                labels[e.label] = labels.get(e.label, 0) + 1
            return {"total_detections": total, "by_label": labels}

        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM detection_events")
            rows = await conn.fetch(
                "SELECT label, COUNT(*) as cnt FROM detection_events GROUP BY label ORDER BY cnt DESC"
            )
        return {
            "total_detections": total,
            "by_label": {r["label"]: r["cnt"] for r in rows},
        }

    async def close(self):
        if self._pool:
            await self._pool.close()
