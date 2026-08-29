import asyncio

from app.database.db import SessionLocal
from app.schemas.domain import DefectIn, DetectionCreate, DetectionOut
from app.services.camera_service import camera_service
from app.services.detection_service import create_detection_record
from app.services.mock_service import mock_service
from app.services.model_service import NoPepperDetectedError, PepperPrediction, model_service
from app.services.storage_service import save_camera_frame
from app.websocket.manager import manager


_target_gate = asyncio.Lock()
_target_present = False
_missing_scans = 0
CLEAR_AFTER_MISSES = 3


async def _predict_group(frame) -> list[PepperPrediction]:
    """Return every pepper visible in the full stable camera frame."""
    await asyncio.sleep(0.08)
    stable_frame = camera_service.get_frame()
    try:
        return await asyncio.to_thread(
            model_service.predict_all,
            stable_frame if stable_frame is not None else frame,
            None,
        )
    except NoPepperDetectedError:
        return []


def _payload_for_prediction(
    prediction: PepperPrediction, batch_id: str
) -> DetectionCreate:
    defects = []
    if prediction.grade == "二级":
        defects.append(
            DefectIn(
                type="model_bad_grade",
                name="模型判定为二级",
                severity="moderate",
                confidence=prediction.grade_confidence,
            )
        )
    return DetectionCreate(
        batch_id=batch_id,
        variety=prediction.species,
        color_score=prediction.species_confidence * 100,
        integrity_score=prediction.grade_confidence * 100,
        shape_score=prediction.good_probability * 100,
        size_score=prediction.detector_confidence * 100,
        defect_score=(1 - prediction.bad_probability) * 100,
        confidence=prediction.joint_confidence,
        processing_time=prediction.processing_time_ms,
        defects=defects,
        model_grade="A" if prediction.grade == "一级" else "C",
        model_quality_score=prediction.good_probability * 100,
        model_grade_reason=(
            f"YOLO11识别：{prediction.label}；"
            f"品种置信度{prediction.species_confidence:.1%}，"
            f"条件品级置信度{prediction.grade_confidence:.1%}，"
            f"联合概率{prediction.joint_confidence:.1%}"
        ),
    )


async def process_camera_frame(
    frame, batch_id: str | None = None
) -> list[DetectionOut]:
    """Continuously update live boxes; persist only once per target passage."""
    global _target_present, _missing_scans
    async with _target_gate:
        predictions = await _predict_group(frame)
        if not predictions:
            if _target_present:
                _missing_scans += 1
            if _target_present and _missing_scans >= CLEAR_AFTER_MISSES:
                _target_present = False
                _missing_scans = 0
                camera_service.clear_live_annotations()
                await manager.broadcast({"type": "target_cleared", "data": {}})
            return []

        _missing_scans = 0
        camera_service.update_live_annotations(predictions)
        summary = camera_service.live_summary()
        await manager.broadcast(
            {
                "type": "live_detection",
                "data": {
                    **summary,
                    "processing_time": predictions[0].processing_time_ms,
                },
            }
        )
        if _target_present:
            return []
        _target_present = True

    target_batch = batch_id or mock_service.batch_id
    image_url = save_camera_frame(frame)
    annotated_frame = predictions[0].annotated_frame
    annotated_image_url = save_camera_frame(annotated_frame, "annotated")
    outputs: list[DetectionOut] = []
    stats = None
    with SessionLocal() as db:
        for prediction in predictions:
            detection, stats = await create_detection_record(
                _payload_for_prediction(prediction, target_batch),
                db,
                image_url=image_url,
                annotated_image_url=annotated_image_url,
                source="camera",
                broadcast_event=False,
            )
            outputs.append(DetectionOut.model_validate(detection))

    await manager.broadcast(
        {
            "type": "detection_group",
            "data": [item.model_dump(mode="json") for item in outputs],
            "stats": stats,
        }
    )
    return outputs
