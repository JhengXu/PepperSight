from dataclasses import dataclass
from typing import Any

from app.models.entities import GradingRule


METRIC_LABELS = {
    "color": "色泽",
    "integrity": "完整度",
    "shape": "形态",
    "size": "尺寸",
    "defect": "缺陷控制",
}


@dataclass(frozen=True)
class QualityResult:
    quality_score: float
    grade: str
    grade_reason: str


def calculate_quality(
    scores: dict[str, float], defects: list[dict[str, Any]], rule: GradingRule
) -> QualityResult:
    """Convert objective vision features into an explainable business grade."""
    # SQLAlchemy applies column defaults on INSERT. The fallbacks also make this
    # pure service safe to unit-test with a transient rule instance.
    color_weight = rule.color_weight if rule.color_weight is not None else 0.30
    integrity_weight = rule.integrity_weight if rule.integrity_weight is not None else 0.30
    shape_weight = rule.shape_weight if rule.shape_weight is not None else 0.15
    size_weight = rule.size_weight if rule.size_weight is not None else 0.10
    defect_weight = rule.defect_weight if rule.defect_weight is not None else 0.15
    a_min_score = rule.a_min_score if rule.a_min_score is not None else 85.0
    b_min_score = rule.b_min_score if rule.b_min_score is not None else 65.0
    weighted_score = round(
        scores["color"] * color_weight
        + scores["integrity"] * integrity_weight
        + scores["shape"] * shape_weight
        + scores["size"] * size_weight
        + scores["defect"] * defect_weight,
        1,
    )

    severe_defect = next(
        (
            item
            for item in defects
            if str(item.get("severity", "mild")).lower() == "severe"
        ),
        None,
    )
    if severe_defect:
        defect_type = severe_defect.get("type", "unknown")
        return QualityResult(
            weighted_score,
            "C",
            f"检测到严重缺陷（{defect_type}），触发强制降级",
        )

    if weighted_score >= a_min_score:
        grade = "A"
    elif weighted_score >= b_min_score:
        grade = "B"
    else:
        grade = "C"

    return QualityResult(
        weighted_score,
        grade,
        f"综合评分{weighted_score:.1f}，判定为{grade}级",
    )
