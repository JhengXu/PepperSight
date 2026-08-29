"""Fixed, auditable gate for optional legacy camera-domain proposals.

This is deliberately a detector post-filter, never a labelling function.  The
gate parameters are immutable defaults matching the camera-adaptation audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraGateConfig:
    proposal_confidence: float = 0.05
    minimum_confidence: float = 0.15
    red_hue_low: int = 15
    red_hue_high: int = 170
    red_saturation_min: int = 90
    red_value_min: int = 45
    red_core_minimum: float = 0.30
    minimum_area_ratio: float = 0.0015
    maximum_area_ratio: float = 0.12
    maximum_aspect_ratio: float = 5.0
    minimum_side_pixels: int = 24
    nms_iou: float = 0.45
    inference_size: int = 960


@dataclass(frozen=True)
class GatedProposal:
    bbox: tuple[int, int, int, int]
    confidence: float
    red_core: float
    area_ratio: float
    aspect_ratio: float
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class RedComponentConfig:
    minimum_component_pixels: int = 500
    minimum_area_ratio: float = 0.003
    maximum_area_ratio: float = 0.06
    minimum_aspect_ratio: float = 1.25
    maximum_aspect_ratio: float = 5.0
    minimum_fill_ratio: float = 0.30
    minimum_median_saturation: float = 0.0
    opening_kernel_size: int = 3
    border_margin_pixels: int = 4
    merge_iou: float = 0.30
    fallback_confidence: float = 0.15


def has_white_detection_surface(
    frame: np.ndarray,
    minimum_ratio: float = 0.35,
    saturation_maximum: int = 55,
    value_minimum: int = 180,
) -> bool:
    """Return whether enough of the full frame is a bright, neutral surface."""
    return white_detection_surface_mask(frame, minimum_ratio) is not None


def white_detection_surface_mask(
    frame: np.ndarray,
    minimum_ratio: float = 0.22,
) -> np.ndarray | None:
    """Return the convex mask of the largest neutral surface using adaptive brightness."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("white surface detection expects an HxWx3 uint8 BGR frame")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    otsu_value, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    value_floor = int(max(85, min(190, otsu_value)))
    neutral = (
        (hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= value_floor)
    ).astype(np.uint8) * 255
    neutral = cv2.morphologyEx(
        neutral,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    neutral = cv2.morphologyEx(
        neutral,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    contours, _ = cv2.findContours(neutral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    if cv2.contourArea(hull) / float(frame.shape[0] * frame.shape[1]) < minimum_ratio:
        return None
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return overlap / max(area_first + area_second - overlap, 1e-9)


def _intersection_over_smaller(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return overlap / max(min(first_area, second_area), 1e-9)


def proposal_median_red_saturation(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: CameraGateConfig = CameraGateConfig(),
) -> float:
    """Measure saturation only over red pixels inside a proposal."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    if crop.size == 0:
        return 0.0
    red = (
        ((crop[:, :, 0] < config.red_hue_low) | (crop[:, :, 0] > config.red_hue_high))
        & (crop[:, :, 1] > config.red_saturation_min)
        & (crop[:, :, 2] > config.red_value_min)
    )
    return float(np.median(crop[:, :, 1][red])) if np.any(red) else 0.0


def gate_camera_proposals(
    frame: np.ndarray,
    proposals: list[tuple[tuple[int, int, int, int], float]],
    config: CameraGateConfig = CameraGateConfig(),
) -> list[GatedProposal]:
    """Apply scalar gates then deterministic class-agnostic NMS.

    The input frame must be uint8 BGR, matching OpenCV and the camera service.
    Accepted rows are returned first, ordered by descending proposal confidence.
    Rejected rows remain available for logging and diagnosis.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("camera gate expects an HxWx3 uint8 BGR frame")
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    evaluated: list[GatedProposal] = []
    for raw_bbox, raw_confidence in proposals:
        x1, y1, x2, y2 = (int(round(value)) for value in raw_bbox)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        box_width, box_height = x2 - x1, y2 - y1
        crop = hsv[y1:y2, x1:x2]
        if crop.size:
            red = (
                ((crop[:, :, 0] < config.red_hue_low) | (crop[:, :, 0] > config.red_hue_high))
                & (crop[:, :, 1] > config.red_saturation_min)
                & (crop[:, :, 2] > config.red_value_min)
            )
            red_core = float(red.mean())
        else:
            red_core = 0.0
        area_ratio = box_width * box_height / float(width * height)
        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        reasons = []
        if raw_confidence < config.minimum_confidence:
            reasons.append("low_confidence")
        if red_core < config.red_core_minimum:
            reasons.append("low_red_core")
        if area_ratio < config.minimum_area_ratio:
            reasons.append("too_small_area")
        if area_ratio > config.maximum_area_ratio:
            reasons.append("too_large_area")
        if aspect_ratio > config.maximum_aspect_ratio:
            reasons.append("extreme_aspect")
        if min(box_width, box_height) < config.minimum_side_pixels:
            reasons.append("short_side")
        evaluated.append(
            GatedProposal(
                bbox=(x1, y1, x2, y2),
                confidence=float(raw_confidence),
                red_core=red_core,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                accepted=not reasons,
                rejection_reason=";".join(reasons) or None,
            )
        )

    kept: list[GatedProposal] = []
    duplicates: list[GatedProposal] = []
    scalar_passed = sorted(
        (proposal for proposal in evaluated if proposal.accepted),
        key=lambda proposal: proposal.confidence,
        reverse=True,
    )
    for proposal in scalar_passed:
        if any(_iou(proposal.bbox, other.bbox) > config.nms_iou for other in kept):
            duplicates.append(
                GatedProposal(
                    **{
                        **proposal.__dict__,
                        "accepted": False,
                        "rejection_reason": "class_agnostic_nms_duplicate",
                    }
                )
            )
        else:
            kept.append(proposal)
    scalar_rejected = [proposal for proposal in evaluated if not proposal.accepted]
    return kept + duplicates + scalar_rejected


def red_component_proposals(
    frame: np.ndarray,
    roi: dict[str, float] | None,
    gate_config: CameraGateConfig = CameraGateConfig(),
    component_config: RedComponentConfig = RedComponentConfig(),
    surface_mask: np.ndarray | None = None,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return conservative red connected components inside an explicit ROI.

    Requiring a caller-supplied conveyor ROI is intentional: the camera holdout
    showed that colour and shape alone cannot reliably separate hands from red
    peppers.  The function therefore refuses full-frame fallback.
    """
    if roi is None:
        return []
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("red component fallback expects an HxWx3 uint8 BGR frame")
    height, width = frame.shape[:2]
    roi_x1 = max(0, min(width, round(width * roi["left"])))
    roi_y1 = max(0, min(height, round(height * roi["top"])))
    roi_x2 = max(0, min(width, round(width * (roi["left"] + roi["width"]))))
    roi_y2 = max(0, min(height, round(height * (roi["top"] + roi["height"]))))
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < gate_config.red_hue_low) | (hsv[:, :, 0] > gate_config.red_hue_high))
        & (hsv[:, :, 1] > gate_config.red_saturation_min)
        & (hsv[:, :, 2] > gate_config.red_value_min)
    ).astype(np.uint8)
    roi_mask = np.zeros_like(red)
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
    if surface_mask is not None:
        if surface_mask.shape != red.shape:
            raise ValueError("surface mask must match the frame height and width")
        roi_mask *= (surface_mask > 0).astype(np.uint8)
    mask = red * roi_mask * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                component_config.opening_kernel_size,
                component_config.opening_kernel_size,
            ),
        ),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    proposals = []
    for x, y, box_width, box_height, component_area in stats[1:]:
        box_area = int(box_width) * int(box_height)
        area_ratio = box_area / float(width * height)
        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        fill_ratio = float(component_area) / max(box_area, 1)
        component_pixels = mask[y : y + box_height, x : x + box_width] > 0
        saturation_crop = hsv[y : y + box_height, x : x + box_width, 1]
        median_saturation = (
            float(np.median(saturation_crop[component_pixels]))
            if np.any(component_pixels)
            else 0.0
        )
        margin = component_config.border_margin_pixels
        if component_area < component_config.minimum_component_pixels:
            continue
        if not component_config.minimum_area_ratio <= area_ratio <= component_config.maximum_area_ratio:
            continue
        if not component_config.minimum_aspect_ratio <= aspect_ratio <= component_config.maximum_aspect_ratio:
            continue
        if fill_ratio < component_config.minimum_fill_ratio:
            continue
        if median_saturation < component_config.minimum_median_saturation:
            continue
        if x <= margin or y <= margin or x + box_width >= width - margin or y + box_height >= height - margin:
            continue
        proposals.append(
            (
                (int(x), int(y), int(x + box_width), int(y + box_height)),
                component_config.fallback_confidence,
            )
        )
    return proposals


def merge_detector_and_component_proposals(
    detector: list[tuple[tuple[int, int, int, int], float]],
    components: list[tuple[tuple[int, int, int, int], float]],
    config: RedComponentConfig = RedComponentConfig(),
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Keep detector boxes and append only non-overlapping component boxes."""
    merged = list(detector)
    for bbox, confidence in components:
        if any(
            _iou(bbox, existing_bbox) > config.merge_iou
            or _intersection_over_smaller(bbox, existing_bbox) > 0.65
            for existing_bbox, _ in merged
        ):
            continue
        merged.append((bbox, confidence))
    return merged
