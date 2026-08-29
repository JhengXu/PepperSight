import json
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QJ_CAMERA_DISABLED", "1")

from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.database.db import SessionLocal
from app.main import app
from app.models.entities import Batch, Detection


def test_detection_database_stats_and_websocket_flow():
    batch_id = f"BATCH-TEST-{uuid4().hex[:8]}"
    payload = {
        "batch_id": batch_id,
        "color_score": 92,
        "integrity_score": 85,
        "shape_score": 88,
        "size_score": 90,
        "defect_score": 86,
        "confidence": 0.94,
        "defects": [
            {
                "type": "black_spot",
                "name": "表面黑斑",
                "severity": "mild",
                "confidence": 0.87,
            }
        ],
    }
    uploaded_path: Path | None = None
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/inspection") as websocket:
                assert websocket.receive_json()["type"] == "connected"
                response = client.post(
                    "/api/detections",
                    data={"payload": json.dumps(payload, ensure_ascii=False)},
                    files={"image": ("pepper.jpg", b"\xff\xd8test-image\xff\xd9", "image/jpeg")},
                )
                assert response.status_code == 201
                created = response.json()
                assert isinstance(created["id"], int)
                assert created["sample_code"].startswith("P-")
                assert created["image_url"].startswith("/uploads/")
                assert created["annotated_image_url"] == created["image_url"]
                uploaded_path = Path(__file__).resolve().parents[2] / created["image_url"].lstrip("/")
                assert uploaded_path.exists()
                event = websocket.receive_json()
                assert event["type"] == "new_detection"
                assert event["data"]["id"] == created["id"]
                assert event["stats"]["total"] == 1

            details = client.get(f"/api/detections/{created['id']}")
            assert details.status_code == 200
            assert details.json()["sample_code"] == created["sample_code"]
            stats = client.get(f"/api/batches/{batch_id}/stats").json()
            assert stats["grades"][created["grade"]] == 1
            assert "online" in client.get("/api/camera/status").json()
    finally:
        with SessionLocal() as db:
            db.execute(delete(Detection).where(Detection.batch_id == batch_id))
            db.execute(delete(Batch).where(Batch.id == batch_id))
            db.commit()
        if uploaded_path and uploaded_path.exists():
            uploaded_path.unlink()


def test_multipart_severe_defect_forces_grade_c():
    batch_id = f"BATCH-SEVERE-{uuid4().hex[:8]}"
    payload = {
        "batch_id": batch_id,
        "color_score": 98,
        "integrity_score": 98,
        "shape_score": 98,
        "size_score": 98,
        "defect_score": 98,
        "confidence": 0.99,
        "defects": [
            {
                "type": "shape_abnormal",
                "name": "形态异常",
                "severity": "severe",
                "confidence": 0.97,
            }
        ],
    }
    uploaded_path: Path | None = None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/detections",
                data={"payload": json.dumps(payload, ensure_ascii=False)},
                files={"image": ("severe.jpg", b"\xff\xd8severe\xff\xd9", "image/jpeg")},
            )
            assert response.status_code == 201
            created = response.json()
            assert created["quality_score"] == 98.0
            assert created["grade"] == "C"
            assert created["grade_reason"] == "检测到严重缺陷（shape_abnormal），触发强制降级"
            uploaded_path = Path(__file__).resolve().parents[2] / created["image_url"].lstrip("/")
    finally:
        with SessionLocal() as db:
            db.execute(delete(Detection).where(Detection.batch_id == batch_id))
            db.execute(delete(Batch).where(Batch.id == batch_id))
            db.commit()
        if uploaded_path and uploaded_path.exists():
            uploaded_path.unlink()
