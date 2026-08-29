import random

import numpy as np

from app.models.entities import GradingRule
from app.services.camera_service import CameraService
from app.services.mock_service import MockService
from app.services.quality_service import calculate_quality


def test_weighted_score_and_grade_a():
    rule = GradingRule()
    result = calculate_quality(
        {"color": 92, "integrity": 95, "shape": 86, "size": 88, "defect": 90},
        [],
        rule,
    )
    assert result.quality_score == 91.3
    assert result.grade == "A"


def test_any_severe_defect_overrides_high_score():
    rule = GradingRule()
    result = calculate_quality(
        {"color": 95, "integrity": 95, "shape": 95, "size": 95, "defect": 95},
        [{"type": "color_abnormal", "name": "色泽异常", "severity": "severe"}],
        rule,
    )
    assert result.grade == "C"
    assert result.grade_reason == "检测到严重缺陷（color_abnormal），触发强制降级"


def test_normal_grade_reason_is_score_based():
    result = calculate_quality(
        {"color": 80, "integrity": 80, "shape": 80, "size": 80, "defect": 80},
        [{"type": "black_spot", "name": "表面黑斑", "severity": "moderate"}],
        GradingRule(),
    )
    assert result.grade == "B"
    assert result.grade_reason == "综合评分80.0，判定为B级"


def test_mock_service_generates_severe_samples_near_ten_percent():
    state = random.getstate()
    random.seed(20260828)
    try:
        service = MockService()
        samples = [service.generate() for _ in range(500)]
    finally:
        random.setstate(state)
    severe_count = sum(
        any(defect.severity == "severe" for defect in sample.defects)
        for sample in samples
    )
    assert 30 <= severe_count <= 70


def test_central_roi_frame_difference_detects_large_change():
    previous = np.zeros((120, 160), dtype=np.uint8)
    current = previous.copy()
    current[20:100, 40:120] = 255
    assert CameraService.changed_pixel_ratio(previous, current) > 0.30


def test_default_camera_selection_is_bound_to_current_usb_index(monkeypatch):
    monkeypatch.delenv("QJ_CAMERA_INDEX", raising=False)
    service = CameraService()
    assert service.selection_mode == "explicit-external"
    assert service.camera_indices == [0]
    assert service.camera_name == "Web Camera"
    assert service.camera_serial == "202604081837"


def test_only_configured_external_index_is_ever_attempted(monkeypatch):
    monkeypatch.setenv("QJ_CAMERA_INDEX", "3")
    service = CameraService()
    assert service.selection_mode == "explicit-external"
    assert service.camera_indices == [3]
