import cv2
import numpy as np
import pytest

from app.services.camera_candidate_gate import (
    CameraGateConfig,
    RedComponentConfig,
    gate_camera_proposals,
    has_white_detection_surface,
    merge_detector_and_component_proposals,
    proposal_median_red_saturation,
    red_component_proposals,
    white_detection_surface_mask,
)
from app.services.model_service import PepperModelService


def red_frame() -> np.ndarray:
    frame = np.full((200, 300, 3), 170, dtype=np.uint8)
    cv2.rectangle(frame, (60, 55), (150, 120), (0, 0, 190), -1)
    return frame


def test_fixed_red_gate_accepts_saturated_red_candidate():
    results = gate_camera_proposals(red_frame(), [((60, 55, 140, 115), 0.8)])

    assert len(results) == 1
    assert results[0].accepted is True
    assert results[0].red_core > 0.30
    assert results[0].rejection_reason is None


def test_fixed_red_gate_rejects_non_red_background_candidate():
    results = gate_camera_proposals(red_frame(), [((180, 60, 260, 130), 0.9)])

    assert results[0].accepted is False
    assert "low_red_core" in results[0].rejection_reason


def test_fixed_red_gate_rejects_low_confidence_even_if_red():
    results = gate_camera_proposals(red_frame(), [((60, 55, 140, 115), 0.14)])

    assert results[0].accepted is False
    assert "low_confidence" in results[0].rejection_reason


def test_fixed_red_gate_class_agnostic_nms_removes_duplicate_boxes():
    results = gate_camera_proposals(
        red_frame(),
        [((60, 55, 140, 115), 0.8), ((62, 57, 138, 113), 0.7)],
    )

    accepted = [result for result in results if result.accepted]
    rejected = [result for result in results if not result.accepted]
    assert len(accepted) == 1
    assert accepted[0].confidence == pytest.approx(0.8)
    assert len(rejected) == 1
    assert rejected[0].rejection_reason == "class_agnostic_nms_duplicate"


def test_fixed_red_gate_rejects_oversized_red_region():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 190)
    results = gate_camera_proposals(frame, [((0, 0, 300, 200), 0.99)])

    assert results[0].accepted is False
    assert "too_large_area" in results[0].rejection_reason


def test_fixed_red_gate_requires_uint8_bgr():
    with pytest.raises(ValueError, match="uint8 BGR"):
        gate_camera_proposals(np.zeros((20, 20, 3), dtype=np.float32), [])


def test_legacy_red_gate_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("QJ_ENABLE_LEGACY_RED_GATE", raising=False)
    service = PepperModelService()

    assert service.enable_legacy_red_gate is False
    assert service.status()["camera_detector_mode"] == "strict_default"
    assert service.status()["legacy_red_gate_warning"] is None


def test_legacy_red_gate_status_warns_when_enabled(monkeypatch):
    monkeypatch.setenv("QJ_ENABLE_LEGACY_RED_GATE", "1")
    service = PepperModelService()

    assert service.enable_legacy_red_gate is True
    assert service.status()["camera_detector_mode"] == "legacy_proposal_fixed_red_gate"
    assert service.status()["legacy_red_gate_warning"]


def test_frame_border_filter_rejects_partial_edge_boxes_only():
    assert PepperModelService._touches_frame_border((10, 10, 99, 90), 100, 100, 2)
    assert PepperModelService._touches_frame_border((1, 20, 40, 70), 100, 100, 2)
    assert not PepperModelService._touches_frame_border((3, 3, 97, 97), 100, 100, 2)


def test_gate_parameters_match_frozen_camera_audit():
    config = CameraGateConfig()

    assert config.minimum_confidence == pytest.approx(0.15)
    assert config.red_core_minimum == pytest.approx(0.30)
    assert config.nms_iou == pytest.approx(0.45)
    assert config.inference_size == 960


def test_red_component_fallback_refuses_full_frame_without_roi():
    assert red_component_proposals(red_frame(), None) == []


def test_white_detection_surface_requires_a_large_bright_neutral_region():
    white = np.full((200, 300, 3), 245, dtype=np.uint8)
    dark_wood = np.full((200, 300, 3), (25, 45, 75), dtype=np.uint8)

    assert has_white_detection_surface(white) is True
    assert has_white_detection_surface(dark_wood) is False


def test_adaptive_white_surface_mask_finds_rotated_gray_board():
    frame = np.full((300, 400, 3), (30, 45, 70), dtype=np.uint8)
    board = np.array([[50, 40], [340, 20], [380, 250], [80, 280]], dtype=np.int32)
    cv2.fillConvexPoly(frame, board, (145, 145, 145))

    mask = white_detection_surface_mask(frame)

    assert mask is not None
    assert mask[150, 200] == 255
    assert mask[10, 10] == 0


def test_surface_mask_excludes_red_components_outside_board():
    frame = np.full((300, 400, 3), (20, 40, 70), dtype=np.uint8)
    cv2.rectangle(frame, (80, 30), (360, 270), (150, 150, 150), -1)
    cv2.ellipse(frame, (190, 150), (20, 55), 0, 0, 360, (0, 0, 150), -1)
    cv2.ellipse(frame, (35, 150), (18, 50), 0, 0, 360, (0, 0, 150), -1)
    mask = white_detection_surface_mask(frame)

    proposals = red_component_proposals(
        frame,
        {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
        component_config=RedComponentConfig(minimum_aspect_ratio=1.0),
        surface_mask=mask,
    )

    assert len(proposals) == 1
    assert proposals[0][0][0] > 80


def test_white_surface_components_reject_low_saturation_skin_colour():
    frame = np.full((300, 400, 3), 150, dtype=np.uint8)
    cv2.ellipse(frame, (130, 150), (20, 55), 0, 0, 360, (0, 0, 145), -1)
    cv2.ellipse(frame, (280, 150), (20, 55), 0, 0, 360, (70, 90, 120), -1)
    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)

    proposals = red_component_proposals(
        frame,
        {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
        CameraGateConfig(red_saturation_min=70, red_value_min=25),
        RedComponentConfig(
            minimum_aspect_ratio=1.0,
            minimum_median_saturation=110,
        ),
        surface_mask=mask,
    )

    assert len(proposals) == 1
    assert proposals[0][0][0] < 200


def test_larger_opening_kernel_disconnects_thin_board_edge_artifact():
    frame = np.full((300, 400, 3), 150, dtype=np.uint8)
    cv2.ellipse(frame, (190, 170), (25, 65), 0, 0, 360, (0, 0, 145), -1)
    cv2.line(frame, (212, 225), (399, 299), (20, 35, 70), 2)
    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)

    proposals = red_component_proposals(
        frame,
        {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
        CameraGateConfig(red_saturation_min=70, red_value_min=25),
        RedComponentConfig(
            minimum_aspect_ratio=1.0,
            minimum_fill_ratio=0.10,
            minimum_median_saturation=110,
            opening_kernel_size=7,
        ),
        surface_mask=mask,
    )

    assert len(proposals) == 1
    assert proposals[0][0][2] < 260


def test_white_surface_component_settings_keep_sparse_reflective_pepper():
    frame = np.full((300, 400, 3), 245, dtype=np.uint8)
    cv2.ellipse(frame, (200, 150), (20, 50), 0, 0, 360, (0, 0, 190), 12)
    roi = {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0}

    proposals = red_component_proposals(
        frame,
        roi,
        component_config=RedComponentConfig(
            minimum_aspect_ratio=1.0,
            minimum_fill_ratio=0.20,
        ),
    )

    assert len(proposals) == 1


def test_red_component_fallback_only_returns_components_inside_roi():
    frame = np.full((300, 400, 3), 170, dtype=np.uint8)
    cv2.ellipse(frame, (100, 150), (25, 60), 0, 0, 360, (0, 0, 190), -1)
    cv2.ellipse(frame, (300, 150), (25, 60), 0, 0, 360, (0, 0, 190), -1)
    roi = {"left": 0.0, "top": 0.0, "width": 0.5, "height": 1.0}

    proposals = red_component_proposals(frame, roi)

    assert len(proposals) == 1
    assert proposals[0][0][0] < 200


def test_component_fusion_never_duplicates_existing_detector_box():
    detector = [((60, 50, 140, 120), 0.8)]
    components = [((62, 52, 138, 118), 0.15), ((200, 50, 270, 120), 0.15)]

    merged = merge_detector_and_component_proposals(detector, components)

    assert merged == [detector[0], components[1]]


def test_component_fusion_removes_contained_duplicate_with_low_iou():
    detector = [((50, 20, 160, 190), 0.8)]
    components = [((75, 50, 120, 165), 0.15)]

    merged = merge_detector_and_component_proposals(detector, components)

    assert merged == detector


def test_component_fusion_keeps_two_detector_instances_inside_one_component():
    detector = [((40, 40, 100, 160), 0.8), ((110, 45, 170, 165), 0.75)]
    components = [((35, 35, 175, 170), 0.15)]

    merged = merge_detector_and_component_proposals(detector, components)

    assert merged == detector


def test_red_saturation_separates_pepper_from_skin_colour():
    frame = np.full((120, 240, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (100, 110), (0, 0, 150), -1)
    cv2.rectangle(frame, (130, 10), (220, 110), (70, 90, 120), -1)
    config = CameraGateConfig(red_saturation_min=70, red_value_min=25)

    pepper = proposal_median_red_saturation(frame, (10, 10, 100, 110), config)
    skin = proposal_median_red_saturation(frame, (130, 10, 220, 110), config)

    assert pepper > 110
    assert skin < 110


def test_red_component_fallback_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("QJ_ENABLE_RED_COMPONENT_FALLBACK", raising=False)
    service = PepperModelService()

    assert service.enable_red_component_fallback is False
    assert service.status()["red_component_fallback_requires_roi"] is True
