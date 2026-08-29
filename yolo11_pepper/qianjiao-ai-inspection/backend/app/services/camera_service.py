import json
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Missing device indices are expected while waiting for the USB camera. Keep
# OpenCV backend probe noise out of the hackathon console; status is exposed by API.
# Some minimal/headless OpenCV builds do not expose ``setLogLevel``; camera
# startup must remain importable in those environments (including CI).
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


TriggerCallback = Callable[[Any], None]


@lru_cache(maxsize=1)
def _annotation_fonts():
    font_path = next(
        (
            path
            for path in (
                Path("/System/Library/Fonts/STHeiti Medium.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            )
            if path.exists()
        ),
        None,
    )
    if font_path:
        return (
            ImageFont.truetype(str(font_path), 23),
            ImageFont.truetype(str(font_path), 15),
        )
    fallback = ImageFont.load_default()
    return fallback, fallback


class CameraService:
    """Single-process webcam capture, MJPEG publishing and stable live tracking."""

    ROI = {"left": 0.37, "top": 0.31, "width": 0.31, "height": 0.40}

    def __init__(self) -> None:
        # Never scan camera indices: AVFoundation index ordering is machine-specific
        # and scanning can silently open the built-in FaceTime camera. This project is
        # currently bound to the attached USB conveyor camera.
        self.camera_index = int(os.getenv("QJ_CAMERA_INDEX", "0"))
        self.camera_indices = [self.camera_index]
        self.camera_name = os.getenv("QJ_CAMERA_NAME", "Web Camera").strip()
        self.camera_serial = os.getenv("QJ_CAMERA_SERIAL", "202604081837").strip()
        self.require_external = os.getenv("QJ_REQUIRE_EXTERNAL_CAMERA", "1") == "1"
        self.selection_mode = "explicit-external"
        self.motion_threshold = float(os.getenv("QJ_MOTION_THRESHOLD", "0.06"))
        self.cooldown_seconds = float(os.getenv("QJ_TRIGGER_COOLDOWN", "0.8"))
        self.inference_interval = float(os.getenv("QJ_INFERENCE_INTERVAL", "0.35"))
        self.tracking_alpha = float(os.getenv("QJ_TRACKING_ALPHA", "0.35"))
        self.tracking_bbox_alpha = float(
            os.getenv("QJ_TRACKING_BBOX_ALPHA", "0.75")
        )
        self.tracking_iou_threshold = float(
            os.getenv("QJ_TRACKING_IOU_THRESHOLD", "0.18")
        )
        self.tracking_center_distance = float(
            os.getenv("QJ_TRACKING_CENTER_DISTANCE", "120")
        )
        self.tracking_confirmation_frames = int(
            os.getenv("QJ_TRACKING_CONFIRMATION_FRAMES", "2")
        )
        self.tracking_max_misses = int(os.getenv("QJ_TRACKING_MAX_MISSES", "3"))
        self.tracking_velocity_alpha = float(
            os.getenv("QJ_TRACKING_VELOCITY_ALPHA", "0.55")
        )
        self.tracking_occlusion_max_frames = int(
            os.getenv("QJ_TRACKING_OCCLUSION_MAX_FRAMES", "12")
        )
        self.classification_min_sharpness = float(
            os.getenv("QJ_CLASSIFICATION_MIN_SHARPNESS", "25")
        )
        self.classification_clear_sharpness = float(
            os.getenv("QJ_CLASSIFICATION_CLEAR_SHARPNESS", "90")
        )
        self.classification_max_displacement = float(
            os.getenv("QJ_CLASSIFICATION_MAX_DISPLACEMENT", "35")
        )
        self.classification_low_texture_max_displacement = float(
            os.getenv("QJ_CLASSIFICATION_LOW_TEXTURE_MAX_DISPLACEMENT", "10")
        )
        self.grade_switch_frames = int(os.getenv("QJ_GRADE_SWITCH_FRAMES", "3"))
        # OpenCV exposure controls are backend-specific.  The bound UVC camera is
        # asked for a longer manual exposure, while deterministic frame gain keeps
        # both inference and MJPEG usable when AVFoundation rejects that control.
        self.auto_exposure = os.getenv("QJ_CAMERA_AUTO_EXPOSURE", "0") == "1"
        self.exposure = float(os.getenv("QJ_CAMERA_EXPOSURE", "-2.0"))
        self.frame_brightness_gain = float(
            os.getenv("QJ_CAMERA_FRAME_GAIN", "1.8")
        )
        self.frame_brightness_offset = float(
            os.getenv("QJ_CAMERA_FRAME_OFFSET", "12")
        )
        self.adaptive_gamma = os.getenv("QJ_ADAPTIVE_GAMMA", "1") == "1"
        self.gamma_target_luminance = float(
            os.getenv("QJ_GAMMA_TARGET_LUMINANCE", "0.48")
        )
        self.gamma_minimum = float(os.getenv("QJ_GAMMA_MINIMUM", "0.65"))
        self.gamma_maximum = float(os.getenv("QJ_GAMMA_MAXIMUM", "1.45"))
        self.gamma_smoothing = float(os.getenv("QJ_GAMMA_SMOOTHING", "0.12"))
        self.disabled = os.getenv("QJ_CAMERA_DISABLED", "0") == "1"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame = None
        self._latest_jpeg: bytes = b""
        self._overlay_frame = None
        self._overlay_until = 0.0
        self._live_annotations: list[dict[str, Any]] = []
        self._annotation_tracks: list[dict[str, Any]] = []
        self._next_track_id = 1
        self._capture = None
        self._trigger_callback: TriggerCallback | None = None
        self._online = False
        self._fps = 0.0
        self._motion_ratio = 0.0
        self._trigger_count = 0
        self._external_verified = False
        self._auto_exposure_applied: bool | None = None
        self._exposure_applied: bool | None = None
        self._exposure_readback: float | None = None
        self._adaptive_gamma_value = 1.0
        self._frame_luminance = 0.0
        self._error = "摄像头尚未启动"
        self._publish_diagnostic(self._error)

    def start(self, trigger_callback: TriggerCallback | None = None) -> None:
        if self._thread and self._thread.is_alive():
            self._trigger_callback = trigger_callback
            return
        self._trigger_callback = trigger_callback
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="qianjiao-camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
        self._thread = None

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def show_annotation(self, frame, seconds: float = 1.8) -> None:
        """Temporarily publish a model-annotated frame without polluting inference input."""
        with self._lock:
            self._overlay_frame = frame.copy()
            self._overlay_until = time.monotonic() + seconds

    def update_live_annotations(self, predictions: list[Any]) -> int:
        """Match detections across frames and publish EMA-smoothed boxes/results."""
        observations = [
            {
                "bbox": tuple(float(value) for value in prediction.bbox),
                "species": prediction.species,
                "grade": prediction.grade,
                "species_confidence": float(prediction.species_confidence),
                "grade_confidence": float(prediction.grade_confidence),
                "joint_confidence": float(prediction.joint_confidence),
                "sharpness": float(getattr(prediction, "sharpness", float("inf"))),
            }
            for prediction in predictions
        ]
        with self._lock:
            available_tracks = set(range(len(self._annotation_tracks)))
            available_observations = set(range(len(observations)))
            matches = []
            scored_pairs = sorted(
                (
                    (
                        self._tracking_match_score(
                            self._predicted_track_bbox(track), observation["bbox"]
                        ),
                        track_index,
                        observation_index,
                    )
                    for track_index, track in enumerate(self._annotation_tracks)
                    for observation_index, observation in enumerate(observations)
                ),
                reverse=True,
            )
            for match_score, track_index, observation_index in scored_pairs:
                if match_score < 0:
                    break
                if track_index not in available_tracks or observation_index not in available_observations:
                    continue
                matches.append((track_index, observation_index))
                available_tracks.remove(track_index)
                available_observations.remove(observation_index)

            alpha = min(1.0, max(0.01, self.tracking_alpha))
            bbox_alpha = min(1.0, max(0.01, self.tracking_bbox_alpha))
            velocity_alpha = min(1.0, max(0.01, self.tracking_velocity_alpha))
            observation_motion: dict[int, tuple[float, float]] = {}
            for track_index, observation_index in matches:
                track = self._annotation_tracks[track_index]
                observation = observations[observation_index]
                old_center = (
                    (track["bbox"][0] + track["bbox"][2]) / 2,
                    (track["bbox"][1] + track["bbox"][3]) / 2,
                )
                new_center = (
                    (observation["bbox"][0] + observation["bbox"][2]) / 2,
                    (observation["bbox"][1] + observation["bbox"][3]) / 2,
                )
                displacement = float(
                    np.hypot(
                        old_center[0] - new_center[0],
                        old_center[1] - new_center[1],
                    )
                )
                delta = (
                    new_center[0] - old_center[0],
                    new_center[1] - old_center[1],
                )
                old_velocity = track.get("velocity", (0.0, 0.0))
                track["velocity"] = tuple(
                    (1.0 - velocity_alpha) * old + velocity_alpha * new
                    for old, new in zip(old_velocity, delta)
                )
                observation_motion[observation_index] = delta
                track["bbox"] = tuple(
                    (1.0 - bbox_alpha) * old + bbox_alpha * new
                    for old, new in zip(track["bbox"], observation["bbox"])
                )
                label = (observation["species"], observation["grade"])
                classification_usable = self._classification_is_usable(
                    observation["sharpness"], displacement
                )
                if classification_usable:
                    for key in ("species_confidence", "grade_confidence", "joint_confidence"):
                        track[key] = (1.0 - alpha) * track[key] + alpha * observation[key]
                    track["classification_hits"] += 1
                    if track["committed_label"] is None:
                        initial_scores = track["initial_label_scores"]
                        initial_scores[label] = (
                            initial_scores.get(label, 0.0)
                            + observation["joint_confidence"]
                        )
                        if (
                            track["classification_hits"]
                            >= self.tracking_confirmation_frames
                        ):
                            track["committed_label"] = max(
                                initial_scores, key=initial_scores.get
                            )
                    elif label == track["committed_label"]:
                        track["pending_label"] = None
                        track["pending_label_hits"] = 0
                    else:
                        if label == track["pending_label"]:
                            track["pending_label_hits"] += 1
                        else:
                            track["pending_label"] = label
                            track["pending_label_hits"] = 1
                        if track["pending_label_hits"] >= self.grade_switch_frames:
                            track["committed_label"] = label
                            track["pending_label"] = None
                            track["pending_label_hits"] = 0
                track["hits"] += 1
                track["misses"] = 0
                track["occluded_frames"] = 0

            for track_index in available_tracks:
                track = self._annotation_tracks[track_index]
                occluding_observation = self._find_occluding_observation(
                    track["bbox"], observations, observation_motion
                )
                if occluding_observation is not None:
                    dx, dy = observation_motion[occluding_observation]
                    track["bbox"] = self._translate_bbox(track["bbox"], dx, dy)
                    track["velocity"] = (dx, dy)
                    track["occluded_frames"] += 1
                    if track["occluded_frames"] <= self.tracking_occlusion_max_frames:
                        track["misses"] = 0
                        continue
                track["occluded_frames"] = 0
                track["misses"] += 1
                vx, vy = track.get("velocity", (0.0, 0.0))
                track["bbox"] = self._translate_bbox(track["bbox"], 0.6 * vx, 0.6 * vy)
                track["velocity"] = (0.75 * vx, 0.75 * vy)

            for observation_index in sorted(available_observations):
                observation = observations[observation_index]
                label = (observation["species"], observation["grade"])
                classification_usable = (
                    observation["sharpness"] >= self.classification_min_sharpness
                )
                initial_scores = (
                    {label: observation["joint_confidence"]}
                    if classification_usable
                    else {}
                )
                committed_label = (
                    label
                    if classification_usable
                    and self.tracking_confirmation_frames <= 1
                    else None
                )
                self._annotation_tracks.append(
                    {
                        **observation,
                        "track_id": self._next_track_id,
                        "hits": 1,
                        "misses": 0,
                        "classification_hits": 1 if classification_usable else 0,
                        "committed_label": committed_label,
                        "initial_label_scores": initial_scores,
                        "pending_label": None,
                        "pending_label_hits": 0,
                        "velocity": (0.0, 0.0),
                        "occluded_frames": 0,
                    }
                )
                self._next_track_id += 1

            self._annotation_tracks = [
                track
                for track in self._annotation_tracks
                if track["misses"] <= self.tracking_max_misses
            ]
            annotations = []
            for track in self._annotation_tracks:
                if (
                    track["hits"] < self.tracking_confirmation_frames
                    or track["classification_hits"] < self.tracking_confirmation_frames
                    or track["committed_label"] is None
                ):
                    continue
                species, grade = track["committed_label"]
                annotations.append(
                    {
                        "track_id": track["track_id"],
                        "bbox": tuple(int(round(value)) for value in track["bbox"]),
                        "species": species,
                        "grade": grade,
                        "species_confidence": track["species_confidence"],
                        "grade_confidence": track["grade_confidence"],
                        "joint_confidence": track["joint_confidence"],
                    }
                )
            annotations.sort(key=lambda item: item["bbox"][0])
            self._live_annotations = annotations
            return len(annotations)

    def clear_live_annotations(self) -> None:
        with self._lock:
            self._live_annotations = []
            self._annotation_tracks = []

    def live_summary(self) -> dict[str, Any]:
        with self._lock:
            annotations = [item.copy() for item in self._live_annotations]
        return {
            "count": len(annotations),
            "species": {
                "条子": sum(item["species"] == "条子" for item in annotations),
                "子弹头": sum(item["species"] == "子弹头" for item in annotations),
            },
            "grades": {
                "一级": sum(item["grade"] == "一级" for item in annotations),
                "二级": sum(item["grade"] == "二级" for item in annotations),
            },
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "online": self._online,
                "camera_index": self.camera_index,
                "camera_name": self.camera_name,
                "camera_serial": self.camera_serial or None,
                "external_verified": self._external_verified,
                "selection_mode": self.selection_mode,
                "candidate_indices": self.camera_indices,
                "fps": round(self._fps, 1),
                "motion_ratio": round(self._motion_ratio, 4),
                "motion_threshold": self.motion_threshold,
                "inference_interval": self.inference_interval,
                "tracking_window": self.tracking_confirmation_frames,
                "tracking_alpha": self.tracking_alpha,
                "tracking_bbox_alpha": self.tracking_bbox_alpha,
                "tracking_center_distance": self.tracking_center_distance,
                "tracking_max_misses": self.tracking_max_misses,
                "tracking_velocity_alpha": self.tracking_velocity_alpha,
                "tracking_occlusion_max_frames": self.tracking_occlusion_max_frames,
                "classification_min_sharpness": self.classification_min_sharpness,
                "classification_clear_sharpness": self.classification_clear_sharpness,
                "classification_max_displacement": self.classification_max_displacement,
                "classification_low_texture_max_displacement": (
                    self.classification_low_texture_max_displacement
                ),
                "grade_switch_frames": self.grade_switch_frames,
                "detection_scope": "full_frame",
                "auto_exposure_requested": self.auto_exposure,
                "auto_exposure_applied": self._auto_exposure_applied,
                "exposure_requested": self.exposure,
                "exposure_applied": self._exposure_applied,
                "exposure_readback": self._exposure_readback,
                "software_brightness_gain": self.frame_brightness_gain,
                "software_brightness_offset": self.frame_brightness_offset,
                "adaptive_gamma_enabled": self.adaptive_gamma,
                "adaptive_gamma_current": round(self._adaptive_gamma_value, 3),
                "frame_luminance": round(self._frame_luminance, 3),
                "trigger_count": self._trigger_count,
                "active_detections": len(self._live_annotations),
                "roi": self.ROI,
                "error": self._error or None,
            }

    def mjpeg_stream(self) -> Iterator[bytes]:
        while not self._stop.is_set():
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(1 / 24)

    def _capture_loop(self) -> None:
        if self.disabled:
            self._publish_diagnostic("摄像头已通过 QJ_CAMERA_DISABLED 禁用")
            self._stop.wait()
            return

        previous_roi = None
        last_inference = 0.0
        frame_counter = 0
        fps_started = time.monotonic()

        while not self._stop.is_set():
            external_ok, external_error = self._verify_external_device()
            with self._lock:
                self._external_verified = external_ok
            if not external_ok:
                self._publish_diagnostic(external_error)
                self._stop.wait(2.5)
                continue

            capture = self._open_capture()
            if capture is None:
                self._publish_diagnostic(
                    f"外接摄像头 {self.camera_name} 无法打开 · 不会回退到内置摄像头"
                )
                self._stop.wait(2.5)
                continue
            self._capture = capture
            previous_roi = None
            try:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._set_offline("摄像头读取中断，正在自动重连")
                        break
                    frame = self._enhance_frame(frame)

                    now = time.monotonic()
                    frame_counter += 1
                    elapsed = now - fps_started
                    if elapsed >= 1.0:
                        measured_fps = frame_counter / elapsed
                        with self._lock:
                            self._fps = measured_fps
                        frame_counter = 0
                        fps_started = now

                    height, width = frame.shape[:2]
                    left = int(width * self.ROI["left"])
                    top = int(height * self.ROI["top"])
                    right = int(width * (self.ROI["left"] + self.ROI["width"]))
                    bottom = int(height * (self.ROI["top"] + self.ROI["height"]))
                    roi = frame[top:bottom, left:right]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (15, 15), 0)

                    motion_ratio = 0.0
                    if previous_roi is not None:
                        motion_ratio = self.changed_pixel_ratio(previous_roi, gray)
                    previous_roi = gray

                    with self._lock:
                        self._online = True
                        self._error = ""
                        self._motion_ratio = motion_ratio

                    # Continuous OpenCV inference replaces the old one-shot motion
                    # trigger. The scheduler drops a frame when the model is busy,
                    # so capture/MJPEG publishing remains smooth at camera FPS.
                    if now - last_inference >= self.inference_interval:
                        last_inference = now
                        with self._lock:
                            self._trigger_count += 1
                        if self._trigger_callback:
                            self._trigger_callback(frame.copy())

                    self._publish_frame(frame)
            finally:
                capture.release()
                self._capture = None

        self._set_offline("摄像头服务已停止")

    def _open_capture(self):
        backends = [cv2.CAP_ANY]
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for camera_index in self.camera_indices:
            for backend in backends:
                capture = cv2.VideoCapture(camera_index, backend)
                if capture.isOpened():
                    with self._lock:
                        self.camera_index = camera_index
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    capture.set(cv2.CAP_PROP_FPS, 30)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._configure_exposure(capture)
                    return capture
                capture.release()
        return None

    def _configure_exposure(self, capture) -> None:
        """Request hardware exposure and record whether the backend accepted it."""
        auto_value = 0.75 if self.auto_exposure else 0.25
        auto_applied = bool(
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
        )
        exposure_applied = bool(
            capture.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        )
        readback = float(capture.get(cv2.CAP_PROP_EXPOSURE))
        with self._lock:
            self._auto_exposure_applied = auto_applied
            self._exposure_applied = exposure_applied
            self._exposure_readback = readback

    def _enhance_frame(self, frame):
        """Apply temporally smoothed adaptive gamma and optional linear trim."""
        enhanced = frame
        if self.adaptive_gamma:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminance = float(np.median(gray)) / 255.0
            safe_luminance = min(0.95, max(0.05, luminance))
            target = min(0.85, max(0.15, self.gamma_target_luminance))
            requested_gamma = float(np.log(target) / np.log(safe_luminance))
            requested_gamma = min(
                self.gamma_maximum,
                max(self.gamma_minimum, requested_gamma),
            )
            smoothing = min(1.0, max(0.01, self.gamma_smoothing))
            gamma = (
                (1.0 - smoothing) * self._adaptive_gamma_value
                + smoothing * requested_gamma
            )
            lookup = np.clip(
                ((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            enhanced = cv2.LUT(frame, lookup)
            with self._lock:
                self._adaptive_gamma_value = gamma
                self._frame_luminance = luminance
        if self.frame_brightness_gain != 1.0 or self.frame_brightness_offset != 0.0:
            enhanced = cv2.convertScaleAbs(
                enhanced,
                alpha=self.frame_brightness_gain,
                beta=self.frame_brightness_offset,
            )
        return enhanced

    def _verify_external_device(self) -> tuple[bool, str]:
        if not self.require_external:
            return True, ""
        if platform.system() != "Darwin":
            # Other platforms still use one explicitly configured index. They never
            # scan or fall back, but USB identity verification is macOS-specific.
            return True, ""

        try:
            completed = subprocess.run(
                ["system_profiler", "SPUSBDataType", "-json"],
                check=False,
                capture_output=True,
                text=True,
                timeout=6,
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return False, "无法验证 USB 外接摄像头 · 为安全起见不打开任何摄像头"

        devices = self._flatten_usb_items(payload.get("SPUSBDataType", []))
        for device in devices:
            name_matches = not self.camera_name or device.get("_name") == self.camera_name
            serial_matches = (
                not self.camera_serial
                or device.get("serial_num") == self.camera_serial
            )
            if name_matches and serial_matches:
                return True, ""

        identity = self.camera_name
        if self.camera_serial:
            identity += f" / {self.camera_serial}"
        return False, f"未检测到指定 USB 摄像头 {identity} · 不会启用内置摄像头"

    @classmethod
    def _flatten_usb_items(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for item in items:
            flattened.append(item)
            children = item.get("_items", [])
            if isinstance(children, list):
                flattened.extend(cls._flatten_usb_items(children))
        return flattened

    @staticmethod
    def _bbox_iou(first, second) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return overlap / max(first_area + second_area - overlap, 1e-9)

    def _tracking_match_score(self, first, second) -> float:
        overlap = self._bbox_iou(first, second)
        if overlap >= self.tracking_iou_threshold:
            return 2.0 + overlap
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
        distance = float(
            np.hypot(
                first_center[0] - second_center[0],
                first_center[1] - second_center[1],
            )
        )
        if distance > self.tracking_center_distance:
            return -1.0
        return 1.0 - distance / max(self.tracking_center_distance, 1.0)

    def _classification_is_usable(self, sharpness: float, displacement: float) -> bool:
        """Accept crisp crops in motion and low-texture crops only when nearly still."""
        if sharpness < self.classification_min_sharpness:
            return False
        if displacement > self.classification_max_displacement:
            return False
        return (
            sharpness >= self.classification_clear_sharpness
            or displacement <= self.classification_low_texture_max_displacement
        )

    @staticmethod
    def _predicted_track_bbox(track: dict[str, Any]):
        vx, vy = track.get("velocity", (0.0, 0.0))
        return CameraService._translate_bbox(track["bbox"], vx, vy)

    @staticmethod
    def _translate_bbox(bbox, dx: float, dy: float):
        return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)

    @staticmethod
    def _bbox_area(bbox) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @classmethod
    def _bbox_coverage(cls, inner, outer) -> float:
        x1 = max(inner[0], outer[0])
        y1 = max(inner[1], outer[1])
        x2 = min(inner[2], outer[2])
        y2 = min(inner[3], outer[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return intersection / max(cls._bbox_area(inner), 1e-9)

    def _find_occluding_observation(
        self,
        track_bbox,
        observations: list[dict[str, Any]],
        observation_motion: dict[int, tuple[float, float]],
    ) -> int | None:
        """Find a merged detection that temporarily covers this unmatched track."""
        candidates = []
        track_area = self._bbox_area(track_bbox)
        for index in observation_motion:
            observation_bbox = observations[index]["bbox"]
            if self._bbox_area(observation_bbox) < 1.25 * track_area:
                continue
            coverage = self._bbox_coverage(track_bbox, observation_bbox)
            if coverage >= 0.45:
                candidates.append((coverage, index))
        return max(candidates)[1] if candidates else None

    @staticmethod
    def changed_pixel_ratio(previous_gray, current_gray) -> float:
        difference = cv2.absdiff(previous_gray, current_gray)
        return float(np.count_nonzero(difference > 24)) / difference.size

    def _publish_frame(self, frame) -> None:
        with self._lock:
            live_annotations = [item.copy() for item in self._live_annotations]
            display_frame = (
                self._overlay_frame.copy()
                if self._overlay_frame is not None
                and time.monotonic() < self._overlay_until
                else frame.copy()
            )
            if time.monotonic() >= self._overlay_until:
                self._overlay_frame = None
        if live_annotations:
            display_frame = self._draw_live_annotations(display_frame, live_annotations)
        ok, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        with self._lock:
            self._latest_frame = frame.copy()
            self._latest_jpeg = encoded.tobytes()

    @staticmethod
    def _draw_live_annotations(frame, annotations: list[dict[str, Any]]):
        """Draw stable Chinese boxes and labels on the OpenCV stream."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        title_font, detail_font = _annotation_fonts()
        for item in annotations:
            x1, y1, x2, y2 = item["bbox"]
            # Velocity prediction may briefly place a box just outside the camera
            # frame. Clamp before drawing so a transient track never stops capture.
            x1 = max(0, min(frame.shape[1] - 2, x1))
            y1 = max(0, min(frame.shape[0] - 2, y1))
            x2 = max(x1 + 1, min(frame.shape[1] - 1, x2))
            y2 = max(y1 + 1, min(frame.shape[0] - 1, y2))
            first_grade = item["grade"] == "一级"
            color = (129, 200, 68) if first_grade else (224, 94, 65)
            title = f"{item['species']} | {item['grade']}"
            detail = (
                f"联合 {item['joint_confidence']:.0%}  "
                f"品种 {item['species_confidence']:.0%}  品级 {item['grade_confidence']:.0%}"
            )
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            title_width = draw.textbbox((0, 0), title, font=title_font)[2]
            detail_width = draw.textbbox((0, 0), detail, font=detail_font)[2]
            label_width = max(title_width, detail_width) + 18
            label_top = max(0, y1 - 55)
            label_bottom = y1
            draw.rectangle(
                (x1, label_top, min(frame.shape[1] - 1, x1 + label_width), label_bottom),
                fill=color,
            )
            draw.text((x1 + 8, label_top + 3), title, font=title_font, fill=(8, 15, 14))
            draw.text((x1 + 8, label_top + 31), detail, font=detail_font, fill=(8, 15, 14))
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def _publish_diagnostic(self, message: str) -> None:
        canvas = np.full((720, 1280, 3), (13, 20, 20), dtype=np.uint8)
        cv2.rectangle(canvas, (470, 220), (810, 500), (51, 73, 68), 2)
        cv2.putText(canvas, "EXTERNAL CAMERA OFFLINE", (400, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (94, 118, 110), 2)
        cv2.putText(canvas, "CONNECT USB CAMERA", (480, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (116, 139, 131), 2)
        cv2.putText(canvas, "BUILT-IN CAMERA BLOCKED", (460, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (65, 151, 115), 1)
        self._publish_frame(canvas)
        self._set_offline(message)

    def _set_offline(self, message: str) -> None:
        with self._lock:
            self._online = False
            self._fps = 0.0
            self._error = message


camera_service = CameraService()
