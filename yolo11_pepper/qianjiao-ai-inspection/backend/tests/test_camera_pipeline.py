import asyncio

import numpy as np

from app.services import camera_pipeline
from app.services.model_service import PepperPrediction


def make_prediction(species: str, grade: str, species_confidence: float):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    return PepperPrediction(
        species=species,
        grade=grade,
        label=f"{species}_{grade}",
        species_confidence=species_confidence,
        grade_confidence=0.9,
        joint_confidence=species_confidence * 0.9,
        good_probability=0.1,
        bad_probability=0.9,
        detector_confidence=0.8,
        bbox=(2, 2, 28, 28),
        sharpness=200,
        processing_time_ms=50,
        annotated_frame=frame,
    )


def test_predict_group_keeps_all_peppers(monkeypatch):
    predictions = [
        make_prediction("条子", "二级", 0.92),
        make_prediction("子弹头", "一级", 0.88),
    ]
    calls = []

    def predict_full_frame(frame, roi):
        calls.append((frame.shape, roi))
        return predictions

    monkeypatch.setattr(camera_pipeline.model_service, "predict_all", predict_full_frame)
    monkeypatch.setattr(
        camera_pipeline.camera_service,
        "get_frame",
        lambda: np.zeros((32, 32, 3), dtype=np.uint8),
    )

    result = asyncio.run(
        camera_pipeline._predict_group(np.zeros((32, 32, 3), dtype=np.uint8))
    )

    assert [item.species for item in result] == ["条子", "子弹头"]
    assert [item.grade for item in result] == ["二级", "一级"]
    assert calls == [((32, 32, 3), None)]


def test_empty_roi_unlocks_next_target_and_broadcasts(monkeypatch):
    events = []

    async def no_pepper(_frame):
        return []

    async def capture_event(event):
        events.append(event)

    monkeypatch.setattr(camera_pipeline, "_predict_group", no_pepper)
    monkeypatch.setattr(camera_pipeline.manager, "broadcast", capture_event)
    cleared = []
    monkeypatch.setattr(
        camera_pipeline.camera_service,
        "clear_live_annotations",
        lambda: cleared.append(True),
    )
    camera_pipeline._target_present = True
    camera_pipeline._missing_scans = 0
    try:
        async def run_until_confirmed_missing():
            results = []
            for _ in range(camera_pipeline.CLEAR_AFTER_MISSES):
                results.append(
                    await camera_pipeline.process_camera_frame(
                        np.zeros((32, 32, 3), dtype=np.uint8)
                    )
                )
            return results

        results = asyncio.run(run_until_confirmed_missing())
    finally:
        camera_pipeline._target_present = False
        camera_pipeline._missing_scans = 0

    assert results == [[]] * camera_pipeline.CLEAR_AFTER_MISSES
    assert events == [{"type": "target_cleared", "data": {}}]
    assert cleared == [True]
