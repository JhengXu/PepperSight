import json
from types import SimpleNamespace

import cv2
import numpy as np

from app.services.camera_service import CameraService


def test_camera_selection_never_scans_or_falls_back(monkeypatch):
    monkeypatch.setenv("QJ_CAMERA_INDEX", "0")
    service = CameraService()

    assert service.camera_indices == [0]
    assert service.selection_mode == "explicit-external"


def test_macos_external_camera_identity_is_verified(monkeypatch):
    monkeypatch.setenv("QJ_CAMERA_NAME", "Web Camera")
    monkeypatch.setenv("QJ_CAMERA_SERIAL", "202604081837")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    payload = {
        "SPUSBDataType": [
            {
                "_name": "USB31Bus",
                "_items": [
                    {"_name": "Web Camera", "serial_num": "202604081837"}
                ],
            }
        ]
    }
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    assert CameraService()._verify_external_device() == (True, "")


def test_macos_missing_external_camera_fails_closed(monkeypatch):
    monkeypatch.setenv("QJ_CAMERA_NAME", "Web Camera")
    monkeypatch.setenv("QJ_CAMERA_SERIAL", "202604081837")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"SPUSBDataType": []})
        ),
    )

    verified, message = CameraService()._verify_external_device()
    assert verified is False
    assert "不会启用内置摄像头" in message


def test_exposure_request_and_readback_are_reported(monkeypatch):
    monkeypatch.setenv("QJ_CAMERA_AUTO_EXPOSURE", "0")
    monkeypatch.setenv("QJ_CAMERA_EXPOSURE", "-2.0")
    service = CameraService()

    class FakeCapture:
        def __init__(self):
            self.settings = []

        def set(self, prop, value):
            self.settings.append((prop, value))
            return prop == cv2.CAP_PROP_AUTO_EXPOSURE

        def get(self, prop):
            assert prop == cv2.CAP_PROP_EXPOSURE
            return -3.0

    capture = FakeCapture()
    service._configure_exposure(capture)
    status = service.status()

    assert (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) in capture.settings
    assert (cv2.CAP_PROP_EXPOSURE, -2.0) in capture.settings
    assert status["auto_exposure_applied"] is True
    assert status["exposure_applied"] is False
    assert status["exposure_readback"] == -3.0


def test_frame_brightness_fallback_affects_inference_frame(monkeypatch):
    monkeypatch.setenv("QJ_ADAPTIVE_GAMMA", "0")
    monkeypatch.setenv("QJ_CAMERA_FRAME_GAIN", "1.8")
    monkeypatch.setenv("QJ_CAMERA_FRAME_OFFSET", "12")
    service = CameraService()
    frame = np.full((4, 4, 3), 20, dtype=np.uint8)

    enhanced = service._enhance_frame(frame)

    assert enhanced.dtype == np.uint8
    assert np.all(enhanced == 48)


def test_adaptive_gamma_brightens_dark_and_darkens_bright_frames(monkeypatch):
    monkeypatch.setenv("QJ_ADAPTIVE_GAMMA", "1")
    monkeypatch.setenv("QJ_GAMMA_SMOOTHING", "1")
    monkeypatch.setenv("QJ_CAMERA_FRAME_GAIN", "1")
    monkeypatch.setenv("QJ_CAMERA_FRAME_OFFSET", "0")
    dark_service = CameraService()
    bright_service = CameraService()
    dark = np.full((20, 20, 3), 35, dtype=np.uint8)
    bright = np.full((20, 20, 3), 225, dtype=np.uint8)

    dark_result = dark_service._enhance_frame(dark)
    bright_result = bright_service._enhance_frame(bright)

    assert dark_result.mean() > dark.mean()
    assert bright_result.mean() < bright.mean()
    assert dark_service.status()["adaptive_gamma_current"] < 1.0
    assert bright_service.status()["adaptive_gamma_current"] > 1.0


def test_live_tracking_averages_boxes_and_ignores_one_frame_label_flip(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "2")
    monkeypatch.setenv("QJ_TRACKING_ALPHA", "0.35")
    monkeypatch.setenv("QJ_TRACKING_BBOX_ALPHA", "0.75")
    service = CameraService()

    def prediction(bbox, grade, confidence):
        return SimpleNamespace(
            bbox=bbox,
            species="条子",
            grade=grade,
            species_confidence=0.98,
            grade_confidence=confidence,
            joint_confidence=confidence,
        )

    assert service.update_live_annotations(
        [prediction((10, 10, 60, 100), "二级", 0.9)]
    ) == 0
    assert service.update_live_annotations(
        [prediction((14, 12, 64, 102), "一级", 0.9)]
    ) == 1

    annotation = service._live_annotations[0]
    assert annotation["bbox"] == (13, 12, 63, 102)
    assert annotation["species"] == "条子"
    assert annotation["grade"] == "二级"


def test_live_tracking_keeps_track_during_short_partial_miss(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_TRACKING_MAX_MISSES", "2")
    service = CameraService()
    prediction = SimpleNamespace(
        bbox=(10, 10, 60, 100),
        species="子弹头",
        grade="一级",
        species_confidence=0.9,
        grade_confidence=0.8,
        joint_confidence=0.72,
    )

    assert service.update_live_annotations([prediction]) == 1
    assert service.update_live_annotations([]) == 1
    assert service.update_live_annotations([]) == 1
    assert service.update_live_annotations([]) == 0


def test_live_tracking_matches_large_shape_change_by_nearby_center(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_TRACKING_CENTER_DISTANCE", "120")
    service = CameraService()

    def prediction(bbox):
        return SimpleNamespace(
            bbox=bbox,
            species="条子",
            grade="二级",
            species_confidence=0.99,
            grade_confidence=0.9,
            joint_confidence=0.89,
        )

    service.update_live_annotations([prediction((100, 40, 150, 300))])
    service.update_live_annotations([prediction((120, 125, 165, 300))])

    assert len(service._annotation_tracks) == 1


def test_fast_blurred_motion_updates_box_but_freezes_grade(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_TRACKING_BBOX_ALPHA", "0.75")
    monkeypatch.setenv("QJ_CLASSIFICATION_MIN_SHARPNESS", "90")
    monkeypatch.setenv("QJ_CLASSIFICATION_MAX_DISPLACEMENT", "35")
    service = CameraService()

    def prediction(bbox, grade, sharpness):
        return SimpleNamespace(
            bbox=bbox,
            species="条子",
            grade=grade,
            species_confidence=0.99,
            grade_confidence=0.9,
            joint_confidence=0.89,
            sharpness=sharpness,
        )

    service.update_live_annotations(
        [prediction((100, 100, 160, 260), "二级", 250)]
    )
    service.update_live_annotations(
        [prediction((180, 100, 240, 260), "一级", 30)]
    )

    annotation = service._live_annotations[0]
    assert annotation["bbox"][0] == 160
    assert annotation["grade"] == "二级"


def test_grade_switch_requires_three_clear_stable_frames(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_GRADE_SWITCH_FRAMES", "3")
    service = CameraService()

    def prediction(grade):
        return SimpleNamespace(
            bbox=(100, 100, 160, 260),
            species="条子",
            grade=grade,
            species_confidence=0.99,
            grade_confidence=0.9,
            joint_confidence=0.89,
            sharpness=250,
        )

    service.update_live_annotations([prediction("二级")])
    service.update_live_annotations([prediction("一级")])
    service.update_live_annotations([prediction("一级")])
    assert service._live_annotations[0]["grade"] == "二级"

    service.update_live_annotations([prediction("一级")])
    assert service._live_annotations[0]["grade"] == "一级"


def test_static_low_texture_pepper_is_not_hidden_as_blur(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "2")
    monkeypatch.setenv("QJ_CLASSIFICATION_MIN_SHARPNESS", "25")
    monkeypatch.setenv("QJ_CLASSIFICATION_CLEAR_SHARPNESS", "90")
    monkeypatch.setenv("QJ_CLASSIFICATION_LOW_TEXTURE_MAX_DISPLACEMENT", "10")
    service = CameraService()

    def prediction(bbox):
        return SimpleNamespace(
            bbox=bbox,
            species="条子",
            grade="二级",
            species_confidence=0.99,
            grade_confidence=0.8,
            joint_confidence=0.79,
            sharpness=33,
        )

    assert service.update_live_annotations([prediction((100, 100, 160, 300))]) == 0
    assert service.update_live_annotations([prediction((103, 101, 163, 301))]) == 1


def test_merged_observation_temporarily_preserves_two_existing_tracks(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_TRACKING_OCCLUSION_MAX_FRAMES", "4")
    service = CameraService()

    def prediction(bbox):
        return SimpleNamespace(
            bbox=bbox,
            species="条子",
            grade="二级",
            species_confidence=0.99,
            grade_confidence=0.9,
            joint_confidence=0.89,
            sharpness=200,
        )

    service.update_live_annotations(
        [prediction((100, 100, 160, 260)), prediction((170, 100, 230, 260))]
    )
    count = service.update_live_annotations([prediction((110, 100, 250, 260))])

    assert count == 2
    assert len(service._annotation_tracks) == 2
    assert all(track["misses"] == 0 for track in service._annotation_tracks)


def test_velocity_prediction_keeps_fast_track_identity(monkeypatch):
    monkeypatch.setenv("QJ_TRACKING_CONFIRMATION_FRAMES", "1")
    monkeypatch.setenv("QJ_TRACKING_CENTER_DISTANCE", "70")
    service = CameraService()

    def prediction(bbox):
        return SimpleNamespace(
            bbox=bbox,
            species="子弹头",
            grade="二级",
            species_confidence=0.95,
            grade_confidence=0.9,
            joint_confidence=0.86,
            sharpness=200,
        )

    service.update_live_annotations([prediction((20, 100, 70, 180))])
    service.update_live_annotations([prediction((80, 100, 130, 180))])
    service.update_live_annotations([prediction((150, 100, 200, 180))])

    assert len(service._annotation_tracks) == 1
