import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import datetime

from app.schemas.domain import DetectionCreate


DEFECT_LIBRARY = {
    "black_spot": "表面黑斑",
    "damage": "机械破损",
    "color_abnormal": "色泽异常",
    "shape_abnormal": "形态异常",
    "mold": "霉变",
    "insect_damage": "虫害",
}


class MockService:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.running = False
        self.batch_id = f"BATCH-{datetime.now():%Y%m%d}-01"

    async def start(self, create_callback: Callable[[DetectionCreate], Awaitable[None]], batch_id: str | None = None) -> None:
        if batch_id:
            self.batch_id = batch_id
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop(create_callback))

    async def pause(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self, create_callback: Callable[[DetectionCreate], Awaitable[None]]) -> None:
        try:
            while self.running:
                await create_callback(self.generate(self.batch_id))
                await asyncio.sleep(random.uniform(2.0, 4.0))
        except asyncio.CancelledError:
            raise

    def generate(self, batch_id: str | None = None) -> DetectionCreate:
        # A fixed severe probability keeps the hard override easy to demonstrate,
        # independent of the otherwise realistic A/B/C score distribution.
        is_severe = random.random() < 0.10
        target_grade = "C" if is_severe else random.choices(["A", "B", "C"], weights=[55, 30, 15], k=1)[0]
        ranges = {"A": (87, 98), "B": (67, 83), "C": (38, 62)}
        low, high = ranges[target_grade]
        center = random.uniform(low, high)
        scores = [max(20, min(100, center + random.uniform(-5, 5))) for _ in range(5)]

        defects: list[dict] = []
        if is_severe:
            key = random.choice(list(DEFECT_LIBRARY))
            defects.append(self._defect(key, "severe"))
        elif target_grade == "A" and random.random() < 0.35:
            key = "black_spot"
            defects.append(self._defect(key, "mild"))
        elif target_grade == "B":
            key = random.choice(["black_spot", "damage", "color_abnormal", "shape_abnormal"])
            defects.append(self._defect(key, "moderate"))
        elif target_grade == "C":
            key = random.choice(list(DEFECT_LIBRARY))
            defects.append(self._defect(key, "moderate"))

        grade_suffix = target_grade.lower()
        return DetectionCreate(
            batch_id=batch_id or self.batch_id,
            image_url=f"/images/pepper-{grade_suffix}.svg",
            annotated_image_url=f"/images/pepper-{grade_suffix}-annotated.svg",
            length=round(random.uniform(8.5, 15.8), 1),
            width=round(random.uniform(1.7, 3.8), 1),
            color_score=round(scores[0], 1),
            integrity_score=round(scores[1], 1),
            shape_score=round(scores[2], 1),
            size_score=round(scores[3], 1),
            defect_score=round(scores[4], 1),
            confidence=round(random.uniform(0.91, 0.99), 3),
            defects=defects,
            processing_time=round(random.uniform(54, 118), 1),
        )

    @staticmethod
    def _defect(defect_type: str, severity: str) -> dict:
        return {
            "type": defect_type,
            "name": DEFECT_LIBRARY[defect_type],
            "severity": severity,
            "confidence": round(random.uniform(0.78, 0.96), 3),
            "area_ratio": round(random.uniform(0.01, 0.16), 3),
        }


mock_service = MockService()
