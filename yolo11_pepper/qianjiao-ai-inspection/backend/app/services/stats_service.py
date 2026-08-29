from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import Detection


SCORE_BUCKETS = ["0-60", "60-70", "70-80", "80-90", "90-100"]


def empty_stats(batch_id: str) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "total": 0,
        "average_score": 0,
        "average_processing_time": 0,
        "grades": {"A": 0, "B": 0, "C": 0},
        "grade_percentages": {"A": 0, "B": 0, "C": 0},
        "average_metrics": {"color": 0, "integrity": 0, "shape": 0, "size": 0, "defect": 0},
        "score_distribution": {bucket: 0 for bucket in SCORE_BUCKETS},
        "defect_counts": {},
    }


def get_batch_stats(db: Session, batch_id: str) -> dict[str, Any]:
    detections = list(
        db.scalars(select(Detection).where(Detection.batch_id == batch_id)).all()
    )
    if not detections:
        return empty_stats(batch_id)

    total = len(detections)
    grades = Counter(item.grade for item in detections)
    defect_counts: Counter[str] = Counter()
    distribution = Counter({bucket: 0 for bucket in SCORE_BUCKETS})
    for item in detections:
        for defect in item.defects:
            defect_counts[defect.get("name", defect.get("type", "未知缺陷"))] += 1
        score = item.quality_score
        if score < 60:
            distribution["0-60"] += 1
        elif score < 70:
            distribution["60-70"] += 1
        elif score < 80:
            distribution["70-80"] += 1
        elif score < 90:
            distribution["80-90"] += 1
        else:
            distribution["90-100"] += 1

    avg = lambda attr: round(sum(getattr(item, attr) for item in detections) / total, 1)
    return {
        "batch_id": batch_id,
        "total": total,
        "average_score": avg("quality_score"),
        "average_processing_time": avg("processing_time"),
        "grades": {grade: grades.get(grade, 0) for grade in "ABC"},
        "grade_percentages": {
            grade: round(grades.get(grade, 0) / total * 100, 1) for grade in "ABC"
        },
        "average_metrics": {
            "color": avg("color_score"),
            "integrity": avg("integrity_score"),
            "shape": avg("shape_score"),
            "size": avg("size_score"),
            "defect": avg("defect_score"),
        },
        "score_distribution": dict(distribution),
        "defect_counts": dict(defect_counts.most_common(8)),
    }

