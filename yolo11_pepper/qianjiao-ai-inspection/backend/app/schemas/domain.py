from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class DefectIn(BaseModel):
    type: str
    name: str
    severity: Literal["mild", "moderate", "severe"] = "mild"
    confidence: float = Field(default=0.8, ge=0, le=1)
    area_ratio: float | None = Field(default=None, ge=0, le=1)


class DetectionCreate(BaseModel):
    batch_id: str = "BATCH-DEMO-001"
    image_url: str | None = None
    annotated_image_url: str | None = None
    variety: str = "贵州辣椒"
    length: float | None = Field(default=None, gt=0)
    width: float | None = Field(default=None, gt=0)
    color_score: float = Field(ge=0, le=100)
    integrity_score: float = Field(ge=0, le=100)
    shape_score: float = Field(ge=0, le=100)
    size_score: float = Field(ge=0, le=100)
    defect_score: float = Field(ge=0, le=100)
    confidence: float = Field(default=0.95, ge=0, le=1)
    defects: list[DefectIn] = Field(default_factory=list)
    processing_time: float | None = Field(default=None, ge=0)
    model_grade: Literal["A", "B", "C"] | None = None
    model_grade_reason: str | None = None
    model_quality_score: float | None = Field(default=None, ge=0, le=100)


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sample_code: str
    batch_id: str
    source: str
    timestamp: datetime
    image_url: str
    annotated_image_url: str
    variety: str
    length: float | None
    width: float | None
    color_score: float
    integrity_score: float
    shape_score: float
    size_score: float
    defect_score: float
    quality_score: float
    grade: str
    confidence: float
    defects: list[dict]
    processing_time: float
    grade_reason: str

    @computed_field
    @property
    def result_type(self) -> Literal["hierarchical_model", "legacy"]:
        return (
            "hierarchical_model"
            if self.grade_reason.startswith("YOLO11识别：")
            else "legacy"
        )

    @computed_field
    @property
    def grade_label(self) -> Literal["一级", "二级"] | None:
        if self.result_type != "hierarchical_model":
            return None
        return "一级" if self.grade == "A" else "二级"

    @computed_field
    @property
    def species_confidence(self) -> float | None:
        return (
            round(self.color_score / 100, 4)
            if self.result_type == "hierarchical_model"
            else None
        )

    @computed_field
    @property
    def grade_confidence(self) -> float | None:
        return (
            round(self.integrity_score / 100, 4)
            if self.result_type == "hierarchical_model"
            else None
        )

    @computed_field
    @property
    def detector_confidence(self) -> float | None:
        return (
            round(self.size_score / 100, 4)
            if self.result_type == "hierarchical_model"
            else None
        )


class DetectionList(BaseModel):
    items: list[DetectionOut]
    total: int
    page: int
    limit: int
    pages: int


class BatchOut(BaseModel):
    id: str
    start_time: datetime
    source: str
    status: str
    note: str
    total: int = 0
    average_score: float = 0
    grade_percentages: dict[str, float] = Field(default_factory=dict)


class GradingRuleUpdate(BaseModel):
    a_min_score: float = Field(ge=0, le=100)
    b_min_score: float = Field(ge=0, le=100)
    color_weight: float = Field(ge=0, le=1)
    integrity_weight: float = Field(ge=0, le=1)
    shape_weight: float = Field(ge=0, le=1)
    size_weight: float = Field(ge=0, le=1)
    defect_weight: float = Field(ge=0, le=1)
    severe_mold_to_c: bool = True
    severe_damage_to_c: bool = True

    @model_validator(mode="after")
    def validate_rules(self):
        total = (
            self.color_weight
            + self.integrity_weight
            + self.shape_weight
            + self.size_weight
            + self.defect_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError("评分权重之和必须等于 100%")
        if self.a_min_score <= self.b_min_score:
            raise ValueError("A级最低分必须高于B级最低分")
        return self


class GradingRuleOut(GradingRuleUpdate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    updated_at: datetime
