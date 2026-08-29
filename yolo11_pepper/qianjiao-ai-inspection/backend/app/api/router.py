import asyncio
import math
from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from app.database.db import SessionLocal, get_db
from app.models.entities import Batch, Detection, GradingRule
from app.schemas.domain import (
    BatchOut,
    DetectionCreate,
    DetectionList,
    DetectionOut,
    GradingRuleOut,
    GradingRuleUpdate,
)
from app.services.analysis_service import generate_batch_analysis
from app.services.camera_pipeline import process_camera_frame
from app.services.camera_service import camera_service
from app.services.detection_service import (
    create_detection_record,
    get_or_create_batch,
    get_or_create_rule,
)
from app.services.mock_service import mock_service
from app.services.model_service import NoPepperDetectedError, model_service
from app.services.stats_service import get_batch_stats
from app.services.storage_service import save_camera_frame, save_upload_image
from app.websocket.manager import manager


router = APIRouter()


async def create_mock_detection(payload: DetectionCreate) -> None:
    with SessionLocal() as db:
        await create_detection_record(payload, db, source="demo")


@router.get("/api/health")
def health() -> dict:
    camera = camera_service.status()
    model = model_service.status()
    return {
        "status": "ok",
        "camera": "online" if camera["online"] else "offline",
        "model": "online" if model["online"] else "offline",
        "model_detail": model,
        "service": "厉辣",
        "camera_detail": camera,
    }


@router.get("/api/stream")
def camera_stream():
    return StreamingResponse(
        camera_service.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@router.get("/api/camera/status")
def camera_status():
    return camera_service.status()


@router.post("/api/camera/trigger", response_model=list[DetectionOut], status_code=201)
async def trigger_camera(batch_id: str | None = None):
    frame = camera_service.get_frame()
    if frame is None or not camera_service.status()["online"]:
        raise HTTPException(503, "摄像头当前没有可用画面")
    result = await process_camera_frame(frame, batch_id)
    if not result:
        raise HTTPException(422, "触发区域内未检测到辣椒")
    return result


@router.post("/api/camera/snapshot-infer")
async def snapshot_and_infer():
    """Save one raw external-camera frame and run the original model once."""
    frame = camera_service.get_frame()
    if frame is None or not camera_service.status()["online"]:
        raise HTTPException(503, "外接摄像头当前没有可用画面")

    image_url = save_camera_frame(frame, "snapshot")
    try:
        predictions = await asyncio.to_thread(
            model_service.predict_all,
            frame,
            None,
        )
    except NoPepperDetectedError as exc:
        raise HTTPException(
            422,
            {"message": "保存成功，但原模型在整张照片中未检测到辣椒", "image_url": image_url},
        ) from exc

    annotated_image_url = save_camera_frame(
        predictions[0].annotated_frame,
        "snapshot-result",
    )
    return {
        "camera": {
            "name": camera_service.camera_name,
            "serial": camera_service.camera_serial,
        },
        "image_url": image_url,
        "annotated_image_url": annotated_image_url,
        "count": len(predictions),
        "results": [
            {
                "index": index,
                "bbox": list(prediction.bbox),
                "species": prediction.species,
                "grade": prediction.grade,
                "species_confidence": round(prediction.species_confidence, 4),
                "grade_confidence": round(prediction.grade_confidence, 4),
                "joint_confidence": round(prediction.joint_confidence, 4),
                "detector_confidence": round(prediction.detector_confidence, 4),
                "processing_time_ms": prediction.processing_time_ms,
            }
            for index, prediction in enumerate(predictions, start=1)
        ],
    }


@router.post("/api/detections", response_model=DetectionOut, status_code=201)
async def create_detection(
    payload: str = Form(..., description="DetectionCreate JSON 字符串"),
    image: UploadFile = File(..., description="原始检测图片"),
    annotated_image: UploadFile | None = File(None, description="可选 AI 标注图片"),
    db: Session = Depends(get_db),
):
    try:
        parsed_payload = DetectionCreate.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(422, f"payload 数据校验失败：{exc}") from exc
    image_url = await save_upload_image(image, "detection")
    annotated_url = (
        await save_upload_image(annotated_image, "annotated")
        if annotated_image is not None
        else parsed_payload.annotated_image_url or image_url
    )
    detection, _ = await create_detection_record(
        parsed_payload,
        db,
        image_url=image_url,
        annotated_image_url=annotated_url,
        source="vision",
    )
    return detection


@router.get("/api/detections", response_model=DetectionList)
def list_detections(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    grade: str | None = Query(None, pattern="^[ABC]$"),
    batch_id: str | None = None,
    q: str | None = None,
    model_only: bool = False,
    source: str | None = Query(None, pattern="^(camera|vision|demo|legacy)$"),
    db: Session = Depends(get_db),
):
    filters = []
    if grade:
        filters.append(Detection.grade == grade)
    if batch_id:
        filters.append(Detection.batch_id == batch_id)
    if q:
        filters.append(or_(Detection.sample_code.contains(q), Detection.variety.contains(q)))
    if model_only:
        filters.append(Detection.grade_reason.startswith("YOLO11识别："))
    if source:
        filters.append(Detection.source == source)
    count_stmt = select(func.count()).select_from(Detection).where(*filters)
    total = db.scalar(count_stmt) or 0
    stmt = (
        select(Detection)
        .where(*filters)
        .order_by(Detection.timestamp.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    items = list(db.scalars(stmt).all())
    return DetectionList(
        items=items,
        total=total,
        page=page,
        limit=limit,
        pages=math.ceil(total / limit) if total else 0,
    )


@router.get("/api/detections/{detection_id}", response_model=DetectionOut)
def get_detection(detection_id: int, db: Session = Depends(get_db)):
    detection = db.get(Detection, detection_id)
    if detection is None:
        raise HTTPException(404, "质检记录不存在")
    return detection


@router.get("/api/batches", response_model=list[BatchOut])
def list_batches(db: Session = Depends(get_db)):
    batches = list(db.scalars(select(Batch).order_by(Batch.start_time.desc())).all())
    result = []
    for batch in batches:
        stats = get_batch_stats(db, batch.id)
        result.append(
            BatchOut(
                id=batch.id,
                start_time=batch.start_time,
                source=batch.source,
                status=batch.status,
                note=batch.note,
                total=stats["total"],
                average_score=stats["average_score"],
                grade_percentages=stats["grade_percentages"],
            )
        )
    return result


@router.get("/api/batches/{batch_id}", response_model=BatchOut)
def get_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(404, "批次不存在")
    stats = get_batch_stats(db, batch_id)
    return BatchOut(
        id=batch.id,
        start_time=batch.start_time,
        source=batch.source,
        status=batch.status,
        note=batch.note,
        total=stats["total"],
        average_score=stats["average_score"],
        grade_percentages=stats["grade_percentages"],
    )


@router.get("/api/batches/{batch_id}/stats")
def batch_stats(batch_id: str, db: Session = Depends(get_db)):
    if db.get(Batch, batch_id) is None:
        raise HTTPException(404, "批次不存在")
    return get_batch_stats(db, batch_id)


@router.post("/api/batches/{batch_id}/analysis")
def batch_analysis(batch_id: str, db: Session = Depends(get_db)):
    if db.get(Batch, batch_id) is None:
        raise HTTPException(404, "批次不存在")
    return generate_batch_analysis(get_batch_stats(db, batch_id))


@router.get("/api/settings/grading", response_model=GradingRuleOut)
def get_grading_rule(db: Session = Depends(get_db)):
    return get_or_create_rule(db)


@router.put("/api/settings/grading", response_model=GradingRuleOut)
def update_grading_rule(payload: GradingRuleUpdate, db: Session = Depends(get_db)):
    rule = get_or_create_rule(db)
    for key, value in payload.model_dump().items():
        setattr(rule, key, value)
    rule.updated_at = datetime.now()
    db.commit()
    db.refresh(rule)
    return rule


@router.post("/api/demo/start")
async def start_demo(batch_id: str | None = None):
    await mock_service.start(create_mock_detection, batch_id)
    await manager.broadcast({"type": "demo_status", "data": {"running": True, "batch_id": mock_service.batch_id}})
    return {"running": True, "batch_id": mock_service.batch_id}


@router.post("/api/demo/pause")
async def pause_demo():
    await mock_service.pause()
    await manager.broadcast({"type": "demo_status", "data": {"running": False, "batch_id": mock_service.batch_id}})
    return {"running": False, "batch_id": mock_service.batch_id}


@router.post("/api/demo/single", response_model=DetectionOut, status_code=201)
async def demo_single(batch_id: str | None = None, db: Session = Depends(get_db)):
    if batch_id:
        mock_service.batch_id = batch_id
    detection, _ = await create_detection_record(
        mock_service.generate(mock_service.batch_id), db, source="demo"
    )
    return detection


@router.post("/api/demo/clear")
async def clear_demo(batch_id: str | None = None, db: Session = Depends(get_db)):
    target_batch = batch_id or mock_service.batch_id
    result = db.execute(delete(Detection).where(Detection.batch_id == target_batch))
    get_or_create_batch(db, target_batch, "demo")
    db.commit()
    stats = get_batch_stats(db, target_batch)
    await manager.broadcast({"type": "batch_cleared", "data": {"batch_id": target_batch}, "stats": stats})
    return {"cleared": result.rowcount or 0, "batch_id": target_batch}


@router.get("/api/demo/status")
def demo_status():
    return {"running": mock_service.running, "batch_id": mock_service.batch_id}


@router.websocket("/ws/inspection")
async def inspection_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "data": {"message": "实时检测通道已连接"}})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
