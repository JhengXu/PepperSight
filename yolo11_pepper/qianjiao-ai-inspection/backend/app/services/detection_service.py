import random
from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models.entities import Batch, Detection, GradingRule
from app.schemas.domain import DetectionCreate, DetectionOut
from app.services.quality_service import QualityResult, calculate_quality
from app.services.stats_service import get_batch_stats
from app.websocket.manager import manager


def get_or_create_rule(db: Session) -> GradingRule:
    rule = db.get(GradingRule, 1)
    if rule is None:
        rule = GradingRule(id=1)
        db.add(rule)
        db.commit()
        db.refresh(rule)
    return rule


def get_or_create_batch(db: Session, batch_id: str, source: str = "vision") -> Batch:
    batch = db.get(Batch, batch_id)
    if batch is None:
        batch = Batch(id=batch_id, source=source)
        db.add(batch)
        db.flush()
    return batch


async def create_detection_record(
    payload: DetectionCreate,
    db: Session,
    *,
    image_url: str | None = None,
    annotated_image_url: str | None = None,
    source: str = "vision",
    broadcast_event: bool = True,
) -> tuple[Detection, dict]:
    get_or_create_batch(db, payload.batch_id, source)
    rule = get_or_create_rule(db)
    defects = [item.model_dump() for item in payload.defects]
    scores = {
        "color": payload.color_score,
        "integrity": payload.integrity_score,
        "shape": payload.shape_score,
        "size": payload.size_score,
        "defect": payload.defect_score,
    }
    result = calculate_quality(scores, defects, rule)
    if payload.model_grade is not None:
        result = QualityResult(
            quality_score=round(payload.model_quality_score or result.quality_score, 1),
            grade=payload.model_grade,
            grade_reason=payload.model_grade_reason or f"视觉模型直接判定为{payload.model_grade}级",
        )
    grade_asset = result.grade.lower()
    resolved_image = image_url or payload.image_url or f"/images/pepper-{grade_asset}.svg"
    resolved_annotated = (
        annotated_image_url
        or payload.annotated_image_url
        or resolved_image
    )
    detection = Detection(
        sample_code=f"P-{datetime.now():%Y%m%d}-{uuid4().hex[:6].upper()}",
        batch_id=payload.batch_id,
        source=source,
        image_url=resolved_image,
        annotated_image_url=resolved_annotated,
        variety=payload.variety,
        length=payload.length,
        width=payload.width,
        color_score=payload.color_score,
        integrity_score=payload.integrity_score,
        shape_score=payload.shape_score,
        size_score=payload.size_score,
        defect_score=payload.defect_score,
        quality_score=result.quality_score,
        grade=result.grade,
        confidence=payload.confidence,
        defects=defects,
        processing_time=payload.processing_time or round(random.uniform(60, 105), 1),
        grade_reason=result.grade_reason,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    stats = get_batch_stats(db, payload.batch_id)
    if broadcast_event:
        await manager.broadcast(
            {
                "type": "new_detection",
                "data": DetectionOut.model_validate(detection).model_dump(mode="json"),
                "stats": stats,
            }
        )
    return detection, stats
