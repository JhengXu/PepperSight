from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.db import Base


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    source: Mapped[str] = mapped_column(String(32), default="demo")
    status: Mapped[str] = mapped_column(String(24), default="active")
    note: Mapped[str] = mapped_column(String(255), default="贵州辣椒演示批次")

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batches.id"), index=True)
    source: Mapped[str] = mapped_column(String(24), default="legacy", index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    image_url: Mapped[str] = mapped_column(String(255), default="/images/pepper-a.svg")
    annotated_image_url: Mapped[str] = mapped_column(String(255), default="/images/pepper-a-annotated.svg")
    variety: Mapped[str] = mapped_column(String(64), default="贵州辣椒")
    length: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    color_score: Mapped[float] = mapped_column(Float)
    integrity_score: Mapped[float] = mapped_column(Float)
    shape_score: Mapped[float] = mapped_column(Float)
    size_score: Mapped[float] = mapped_column(Float)
    defect_score: Mapped[float] = mapped_column(Float)
    quality_score: Mapped[float] = mapped_column(Float, index=True)
    grade: Mapped[str] = mapped_column(String(1), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.95)
    defects: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    processing_time: Mapped[float] = mapped_column(Float, default=80.0)
    grade_reason: Mapped[str] = mapped_column(Text)

    batch: Mapped[Batch] = relationship(back_populates="detections")


class GradingRule(Base):
    __tablename__ = "grading_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    a_min_score: Mapped[float] = mapped_column(Float, default=85.0)
    b_min_score: Mapped[float] = mapped_column(Float, default=65.0)
    color_weight: Mapped[float] = mapped_column(Float, default=0.30)
    integrity_weight: Mapped[float] = mapped_column(Float, default=0.30)
    shape_weight: Mapped[float] = mapped_column(Float, default=0.15)
    size_weight: Mapped[float] = mapped_column(Float, default=0.10)
    defect_weight: Mapped[float] = mapped_column(Float, default=0.15)
    severe_mold_to_c: Mapped[bool] = mapped_column(Boolean, default=True)
    severe_damage_to_c: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
