#!/usr/bin/env python3
"""Extract deterministic, leakage-safe pepper quality features.

The learned YOLO features used by the hierarchical classifier are strong at
species recognition, but subtle grade cues can be under-represented.  This
utility builds a complementary feature block from a standardized 256x256
render of each pepper.  It deliberately uses no labels while extracting:

* RGB/HSV/Lab histograms, robust quantiles, and colour ratios;
* dark, yellow/green, pale, and highlight region statistics;
* gradient, Canny, Laplacian, local-contrast, LBP, and neighbour texture;
* contour geometry, convexity, moments, and an axis-aligned width profile.

Leakage protocol
----------------
The input must be a physically separated, single-split manifest.  ``train`` is
the default.  A test manifest is rejected before it is opened unless both
``--split test`` and ``--allow-test`` are explicitly supplied.  The script
does not discover manifests, infer other splits, train a model, or inspect
predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


ALGORITHM_VERSION = "quality_features_v1"
DEFAULT_BACKGROUND = (64, 68, 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic pepper colour/defect/texture/shape features "
            "from one physically separated manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "yolo11_pepper/datasets/pepper_ssl_v4_merged/train_manifest.csv"
        ),
        help="Single-split CSV manifest (default: merged v4 training manifest).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="The one split the physically separated manifest must contain.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/runs/hierarchical_v5_clean/features_quality"),
        help="Output directory; the cache is saved as quality_<split>.pt.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-fraction", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required explicit authorization when --split test is requested.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache with the same destination name.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_no_test_protocol(manifest: Path, split: str, allow_test: bool) -> None:
    manifest_tokens = {part.lower() for part in manifest.resolve().parts}
    test_named = any("test" in token for token in manifest_tokens)
    if split == "test" and not allow_test:
        raise ValueError(
            "STRICT NO-TEST protocol: --split test requires the explicit "
            "--allow-test flag after model selection is frozen."
        )
    if test_named and not (split == "test" and allow_test):
        raise ValueError(
            f"STRICT NO-TEST protocol rejected test-named manifest: {manifest}"
        )


def read_pure_manifest(path: Path, requested_split: str) -> list[dict[str, str]]:
    """Read a single-split manifest and validate identity/selection metadata."""
    rows: list[dict[str, str]] = []
    observed: Counter[str] = Counter()
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "split", "group_id", "class_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        for row_number, raw in enumerate(reader, start=2):
            split = (raw.get("split") or "").strip().lower()
            observed[split] += 1
            if split != requested_split:
                # Fail closed without materializing labels/paths/groups from the
                # unrequested row.  A physically separated manifest is required.
                continue
            path_value = (raw.get("path") or "").strip()
            group_id = (raw.get("group_id") or "").strip()
            if not path_value or not group_id:
                raise ValueError(f"Empty path/group_id at manifest row {row_number}")
            class_id = int(raw["class_id"])
            if not 0 <= class_id < 4:
                raise ValueError(f"Invalid class_id={class_id} at row {row_number}")
            resolved_path = Path(path_value).resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Image referenced at manifest row {row_number}: {resolved_path}"
                )
            selection_role = (raw.get("selection_role") or "").strip().lower()
            expected_roles = {
                "train": {"", "training"},
                "val": {"", "model_selection"},
                "test": {"", "strict_test", "final_test"},
            }[requested_split]
            if selection_role not in expected_roles:
                raise ValueError(
                    f"Unexpected selection_role={selection_role!r} for "
                    f"{requested_split} at row {row_number}"
                )
            rows.append(
                {
                    **raw,
                    "path": str(resolved_path),
                    "split": split,
                    "group_id": group_id,
                    "class_id": str(class_id),
                }
            )
    foreign = {name: count for name, count in observed.items() if name != requested_split}
    if foreign:
        raise ValueError(
            "Physically separated manifest required; found unrequested splits: "
            f"{foreign}"
        )
    if not rows:
        raise ValueError(f"No {requested_split!r} rows in manifest: {path}")
    return rows


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    height, width = mask.shape
    centre = np.array([width / 2.0, height / 2.0])
    candidates: list[tuple[float, int]] = []
    retained = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(12, round(height * width * 0.0005)):
            continue
        retained += 1
        distance = np.linalg.norm(centroids[label] - centre) / max(height, width, 1)
        # Area dominates; the mild centre prior resolves similarly sized blobs.
        score = area * max(0.70, 1.0 - 0.35 * distance)
        candidates.append((float(score), label))
    if not candidates:
        return np.zeros_like(mask, dtype=bool), retained
    selected = max(candidates)[1]
    return labels == selected, retained


def fill_holes(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, 8)
    if count <= 1:
        return mask
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    holes = inverse.astype(bool) & ~np.isin(labels, border_labels)
    return mask | holes


def clean_candidate(mask: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    kernel_size = max(3, round(min(height, width) * 0.018))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, small_kernel)
    selected, component_count = largest_component(cleaned.astype(bool))
    return fill_holes(selected), component_count


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray
    method: str
    component_count: int
    initial_ratio: float
    transparent: bool


def segment_opaque(rgb: np.ndarray) -> Segmentation:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)

    border_width = max(2, round(min(height, width) * 0.045))
    border = np.zeros((height, width), dtype=bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    background_lab = np.median(lab[border], axis=0)
    colour_distance = np.linalg.norm(lab - background_lab, axis=2)
    border_distance = colour_distance[border]
    distance_threshold = max(11.0, float(np.quantile(border_distance, 0.98)) + 3.0)

    red_or_orange = ((hue <= 27) | (hue >= 163)) & (saturation >= 42) & (value >= 18)
    red_dominant = (red >= green * 0.91) & (red >= blue * 0.86)
    chromatic_body = (
        (colour_distance >= distance_threshold)
        & (saturation >= 35)
        & red_dominant
        & (value >= 15)
    )
    initial = red_or_orange | chromatic_body
    mask, components = clean_candidate(initial)
    ratio = float(mask.mean())
    if 0.008 <= ratio <= 0.94:
        return Segmentation(mask, "opaque_red_colour", components, ratio, False)

    # A less hue-specific fallback catches unusually dark/desaturated peppers
    # while still suppressing a neutral conveyor/background.
    channel_range = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
    fallback = (
        (colour_distance >= max(8.0, distance_threshold - 3.0))
        & ((saturation >= 24) | (channel_range >= 22))
        & (value >= 10)
    )
    mask, fallback_components = clean_candidate(fallback)
    fallback_ratio = float(mask.mean())
    if 0.008 <= fallback_ratio <= 0.94:
        return Segmentation(
            mask,
            "opaque_colour_distance",
            fallback_components,
            ratio,
            False,
        )

    # Deterministic GrabCut is a final image-driven fallback.  The one-pixel
    # border is definite background; red pixels are definite foreground.
    cv2.setRNGSeed(2041)
    grab = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)
    grab[0] = cv2.GC_BGD
    grab[-1] = cv2.GC_BGD
    grab[:, 0] = cv2.GC_BGD
    grab[:, -1] = cv2.GC_BGD
    similar_background = colour_distance <= max(6.0, float(np.quantile(border_distance, 0.90)) + 2)
    grab[similar_background] = cv2.GC_PR_BGD
    grab[red_or_orange & ~border] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            grab,
            None,
            bg_model,
            fg_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
        grab_mask, grab_components = clean_candidate(
            (grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)
        )
        grab_ratio = float(grab_mask.mean())
        if 0.008 <= grab_ratio <= 0.94:
            return Segmentation(
                grab_mask, "opaque_grabcut", grab_components, ratio, False
            )
    except cv2.error:
        pass

    # Fail visibly rather than silently returning an empty tensor.  A centred
    # ellipse is only a numerical last resort and is counted in cache metadata.
    yy, xx = np.ogrid[:height, :width]
    ellipse = (
        ((xx - (width - 1) / 2) / max(width * 0.45, 1)) ** 2
        + ((yy - (height - 1) / 2) / max(height * 0.45, 1)) ** 2
        <= 1
    )
    return Segmentation(ellipse, "opaque_central_fallback", 1, ratio, False)


def load_and_segment(path: Path) -> tuple[np.ndarray, Segmentation, dict[str, int | float]]:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[..., :3])
    alpha = rgba[..., 3]
    alpha_pixels = alpha >= 4
    transparent = bool(alpha.min() < 250 and alpha_pixels.sum() >= 12)
    if transparent:
        mask, components = clean_candidate(alpha_pixels)
        if mask.sum() < 12:
            mask = alpha_pixels
            components = 1
            method = "alpha_raw_fallback"
        else:
            method = "alpha"
        segmentation = Segmentation(
            mask=mask,
            method=method,
            component_count=components,
            initial_ratio=float(alpha_pixels.mean()),
            transparent=True,
        )
    else:
        segmentation = segment_opaque(rgb)
    diagnostics = {
        "source_width": int(rgb.shape[1]),
        "source_height": int(rgb.shape[0]),
        "source_mask_ratio": float(segmentation.mask.mean()),
    }
    return rgb, segmentation, diagnostics


def canonical_render(
    rgb: np.ndarray,
    mask: np.ndarray,
    size: int,
    target_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot render an empty foreground mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped_rgb = rgb[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    target_edge = max(1, round(size * target_fraction))
    scale = target_edge / max(cropped_rgb.shape[0], cropped_rgb.shape[1], 1)
    width = max(1, round(cropped_rgb.shape[1] * scale))
    height = max(1, round(cropped_rgb.shape[0] * scale))
    resized_rgb = cv2.resize(cropped_rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    resized_mask = resized_mask.astype(bool)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = np.asarray(DEFAULT_BACKGROUND, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), dtype=bool)
    x = (size - width) // 2
    y = (size - height) // 2
    region = canvas[y : y + height, x : x + width]
    region[resized_mask] = resized_rgb[resized_mask]
    canvas_mask[y : y + height, x : x + width] = resized_mask
    return np.ascontiguousarray(canvas), canvas_mask


class FeatureBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.values: list[float] = []

    def add(self, name: str, value: float) -> None:
        self.names.append(name)
        self.values.append(float(value))

    def add_many(self, prefix: str, values: Iterable[float]) -> None:
        for index, value in enumerate(values):
            self.add(f"{prefix}_{index:02d}", float(value))


def normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    return histogram / max(histogram.sum(), 1.0)


def add_distribution(
    builder: FeatureBuilder,
    name: str,
    values: np.ndarray,
    histogram_bins: int,
    include_quantiles: bool = True,
) -> None:
    values = np.clip(values.astype(np.float64), 0.0, 1.0)
    histogram = normalized_histogram(values, histogram_bins)
    builder.add_many(f"{name}_hist", histogram)
    builder.add(f"{name}_mean", values.mean())
    builder.add(f"{name}_std", values.std())
    median = float(np.median(values))
    builder.add(f"{name}_mad", np.median(np.abs(values - median)))
    entropy = -np.sum(histogram * np.log(histogram + 1e-12)) / math.log(histogram_bins)
    builder.add(f"{name}_entropy", entropy)
    if include_quantiles:
        quantiles = np.quantile(values, (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
        for label, value in zip((5, 10, 25, 50, 75, 90, 95), quantiles):
            builder.add(f"{name}_q{label:02d}", value)


def component_features(mask: np.ndarray, body: np.ndarray) -> tuple[float, float, float]:
    candidate = (mask & body).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    body_area = max(int(body.sum()), 1)
    areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= max(3, round(body_area * 0.001))
    ]
    return (
        len(areas) / max(body_area / 1000.0, 1.0),
        max(areas, default=0) / body_area,
        np.mean(areas, dtype=np.float64) / body_area if areas else 0.0,
    )


def uniform_lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    centre = gray[1:-1, 1:-1]
    neighbours = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    bits = np.stack([item >= centre for item in neighbours], axis=-1).astype(np.uint8)
    transitions = np.sum(bits != np.roll(bits, 1, axis=-1), axis=-1)
    ones = bits.sum(axis=-1)
    codes = np.where(transitions <= 2, ones, 9)
    valid = mask[1:-1, 1:-1]
    histogram = np.bincount(codes[valid].astype(np.int64), minlength=10).astype(np.float64)
    return histogram / max(histogram.sum(), 1.0)


def add_texture_features(
    builder: FeatureBuilder,
    gray: np.ndarray,
    body: np.ndarray,
) -> dict[str, np.ndarray]:
    body_u8 = body.astype(np.uint8)
    inner = cv2.erode(body_u8, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if inner.sum() < 32:
        inner = body

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    for name, image in (("gradient", gradient), ("laplacian", laplacian)):
        values = image[inner].astype(np.float64)
        builder.add(f"texture_{name}_mean", values.mean())
        builder.add(f"texture_{name}_std", values.std())
        for label, value in zip(
            (50, 75, 90, 95, 99), np.quantile(values, (0.50, 0.75, 0.90, 0.95, 0.99))
        ):
            builder.add(f"texture_{name}_q{label}", value)

    builder.add("texture_gradient_gt_010", np.mean(gradient[inner] > 0.10))
    builder.add("texture_gradient_gt_020", np.mean(gradient[inner] > 0.20))
    builder.add("texture_laplacian_gt_010", np.mean(laplacian[inner] > 0.10))
    builder.add("texture_laplacian_gt_020", np.mean(laplacian[inner] > 0.20))
    edges_low = cv2.Canny(np.round(gray * 255).astype(np.uint8), 40, 100) > 0
    edges_high = cv2.Canny(np.round(gray * 255).astype(np.uint8), 80, 160) > 0
    builder.add("texture_canny_40_100", np.mean(edges_low[inner]))
    builder.add("texture_canny_80_160", np.mean(edges_high[inner]))

    for kernel_size in (3, 7, 15):
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        contrast = np.abs(gray - blurred)[inner]
        builder.add(f"texture_local_{kernel_size}_mean", contrast.mean())
        builder.add(f"texture_local_{kernel_size}_std", contrast.std())
        builder.add(f"texture_local_{kernel_size}_q90", np.quantile(contrast, 0.90))

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for index, (dy, dx) in enumerate(directions):
        shifted_gray = np.roll(gray, shift=(dy, dx), axis=(0, 1))
        shifted_mask = np.roll(inner, shift=(dy, dx), axis=(0, 1))
        valid = inner & shifted_mask
        # Suppress wrapped rows/columns introduced by np.roll.
        if dy > 0:
            valid[:dy] = False
        if dx > 0:
            valid[:, :dx] = False
        elif dx < 0:
            valid[:, dx:] = False
        difference = np.abs(gray - shifted_gray)[valid]
        builder.add(f"texture_pair_{index}_mean", difference.mean())
        builder.add(f"texture_pair_{index}_std", difference.std())
        builder.add(f"texture_pair_{index}_q75", np.quantile(difference, 0.75))
        builder.add(f"texture_pair_{index}_q90", np.quantile(difference, 0.90))

    builder.add_many("texture_lbp_uniform", uniform_lbp_histogram(gray, inner))
    angles = np.mod(np.arctan2(sobel_y, sobel_x), np.pi)
    orientation_hist, _ = np.histogram(
        angles[inner], bins=8, range=(0.0, np.pi), weights=gradient[inner]
    )
    orientation_hist = orientation_hist.astype(np.float64)
    orientation_hist /= max(orientation_hist.sum(), 1e-12)
    builder.add_many("texture_gradient_orientation", orientation_hist)
    return {"gradient": gradient, "inner": inner}


def add_shape_features(builder: FeatureBuilder, body: np.ndarray) -> None:
    size_y, size_x = body.shape
    area = int(body.sum())
    contours, _ = cv2.findContours(
        body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No foreground contour after canonical rendering")
    contour = max(contours, key=cv2.contourArea)
    contour_area = max(float(cv2.contourArea(contour)), 1e-9)
    perimeter = max(float(cv2.arcLength(contour, True)), 1e-9)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1e-9)
    hull_perimeter = max(float(cv2.arcLength(hull, True)), 1e-9)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    centroid_x = moments["m10"] / max(moments["m00"], 1e-9)
    centroid_y = moments["m01"] / max(moments["m00"], 1e-9)

    builder.add("shape_area_fraction", area / (size_x * size_y))
    builder.add("shape_bbox_width", width / size_x)
    builder.add("shape_bbox_height", height / size_y)
    builder.add("shape_aspect_log", math.log(max(width, 1) / max(height, 1)))
    builder.add("shape_extent", contour_area / max(width * height, 1))
    builder.add("shape_mask_to_contour_area", area / contour_area)
    builder.add("shape_perimeter_sqrt_area", perimeter / math.sqrt(contour_area))
    builder.add("shape_circularity", 4 * math.pi * contour_area / (perimeter**2))
    builder.add("shape_solidity", contour_area / hull_area)
    builder.add("shape_convexity", hull_perimeter / perimeter)
    builder.add("shape_boundary_roughness", perimeter / hull_perimeter)
    builder.add("shape_equivalent_diameter", math.sqrt(4 * contour_area / math.pi) / size_x)
    builder.add("shape_centroid_x", centroid_x / size_x)
    builder.add("shape_centroid_y", centroid_y / size_y)

    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    major_rect = max(rect_width, rect_height, 1e-9)
    minor_rect = max(min(rect_width, rect_height), 1e-9)
    builder.add("shape_minrect_major", major_rect / size_x)
    builder.add("shape_minrect_minor", minor_rect / size_x)
    builder.add("shape_minrect_aspect_log", math.log(major_rect / minor_rect))
    builder.add("shape_minrect_fill", contour_area / max(rect_width * rect_height, 1e-9))

    for epsilon_fraction in (0.01, 0.02, 0.05):
        polygon = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
        label = round(epsilon_fraction * 100)
        builder.add(f"shape_polygon_vertices_{label:02d}", len(polygon) / 100.0)

    hu = cv2.HuMoments(moments).flatten()
    transformed_hu = [
        0.0 if abs(value) < 1e-30 else float(np.clip(-np.sign(value) * np.log10(abs(value)), -20, 20))
        for value in hu
    ]
    builder.add_many("shape_hu", transformed_hu)

    ys, xs = np.nonzero(body)
    coordinates = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    centred = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-12)
    eigenvectors = eigenvectors[:, order]
    projected = centred @ eigenvectors
    major_coordinate = projected[:, 0]
    minor_coordinate = projected[:, 1]
    builder.add("shape_pca_eigen_ratio", eigenvalues[1] / eigenvalues[0])
    builder.add("shape_pca_eccentricity", math.sqrt(max(0.0, 1 - eigenvalues[1] / eigenvalues[0])))
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    builder.add("shape_axis_cos2", math.cos(2 * angle))
    builder.add("shape_axis_sin2", math.sin(2 * angle))

    edges = np.linspace(major_coordinate.min(), major_coordinate.max() + 1e-9, 9)
    widths: list[float] = []
    offsets: list[float] = []
    global_minor_span = max(float(np.ptp(minor_coordinate)), 1e-9)
    for index in range(8):
        included = (major_coordinate >= edges[index]) & (major_coordinate < edges[index + 1])
        values = minor_coordinate[included]
        if len(values):
            widths.append(float(np.ptp(values)) / global_minor_span)
            offsets.append(float(np.median(values)) / global_minor_span)
        else:
            widths.append(0.0)
            offsets.append(0.0)
    # Resolve the arbitrary PCA sign by placing the narrower end first.
    if np.mean(widths[:2]) > np.mean(widths[-2:]):
        widths.reverse()
        offsets = [-value for value in reversed(offsets)]
    builder.add_many("shape_axis_width", widths)
    builder.add_many("shape_axis_offset", offsets)


def extract_features(rgb: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, list[str]]:
    builder = FeatureBuilder()
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = np.stack(
        (
            hsv_u8[..., 0].astype(np.float32) / 179.0,
            hsv_u8[..., 1].astype(np.float32) / 255.0,
            hsv_u8[..., 2].astype(np.float32) / 255.0,
        ),
        axis=-1,
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    eroded = cv2.erode(body.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    colour_mask = eroded if eroded.sum() >= 32 else body

    colour_channels: Sequence[tuple[str, np.ndarray, int, bool]] = (
        ("rgb_r", rgb_float[..., 0], 8, True),
        ("rgb_g", rgb_float[..., 1], 8, True),
        ("rgb_b", rgb_float[..., 2], 8, True),
        ("hsv_h", hsv[..., 0], 18, False),
        ("hsv_s", hsv[..., 1], 8, True),
        ("hsv_v", hsv[..., 2], 8, True),
        ("lab_l", lab[..., 0], 8, True),
        ("lab_a", lab[..., 1], 8, True),
        ("lab_b", lab[..., 2], 8, True),
    )
    for name, channel, bins, quantiles in colour_channels:
        add_distribution(builder, name, channel[colour_mask], bins, quantiles)

    hue_degrees = hsv[..., 0] * 360.0
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    valid_hue = saturation >= 0.18
    ratio_masks = {
        "dark_v15": value < 0.15,
        "dark_v25": value < 0.25,
        "dark_v40": value < 0.40,
        "bright_v85": value > 0.85,
        "bright_v95": value > 0.95,
        "low_s20": saturation < 0.20,
        "high_s60": saturation > 0.60,
        "red": valid_hue & ((hue_degrees < 18) | (hue_degrees >= 342)),
        "orange": valid_hue & (hue_degrees >= 18) & (hue_degrees < 45),
        "yellow": valid_hue & (hue_degrees >= 45) & (hue_degrees < 75),
        "green": valid_hue & (hue_degrees >= 75) & (hue_degrees < 170),
        "brown_darkred": valid_hue & ((hue_degrees < 45) | (hue_degrees >= 342)) & (value < 0.45),
        "charred": (value < 0.25) & (saturation > 0.25),
        "pale": (saturation < 0.25) & (value > 0.55),
        "highlight": (value > 0.85) & (saturation < 0.35),
    }
    body_count = max(int(colour_mask.sum()), 1)
    for name, candidate in ratio_masks.items():
        builder.add(f"colour_ratio_{name}", (candidate & colour_mask).sum() / body_count)

    pixels = rgb_float[colour_mask]
    for first, second, label in ((0, 1, "rg"), (0, 2, "rb"), (1, 2, "gb")):
        if pixels[:, first].std() < 1e-9 or pixels[:, second].std() < 1e-9:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(pixels[:, first], pixels[:, second])[0, 1])
        builder.add(f"colour_correlation_{label}", correlation)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = add_texture_features(builder, gray, body)

    defect_masks = {
        "dark": (value < 0.32) & colour_mask,
        "yellow_green": (
            valid_hue & (hue_degrees >= 45) & (hue_degrees < 170) & colour_mask
        ),
        "pale": (saturation < 0.25) & (value > 0.55) & colour_mask,
        "high_gradient": (texture["gradient"] > 0.18) & texture["inner"],
    }
    for name, candidate in defect_masks.items():
        count_density, largest_ratio, mean_ratio = component_features(candidate, colour_mask)
        builder.add(f"region_{name}_component_density", count_density)
        builder.add(f"region_{name}_largest_ratio", largest_ratio)
        builder.add(f"region_{name}_mean_ratio", mean_ratio)

    ys, xs = np.nonzero(colour_mask)
    centre_x, centre_y = float(xs.mean()), float(ys.mean())
    radial = np.sqrt((np.indices(body.shape)[1] - centre_x) ** 2 + (np.indices(body.shape)[0] - centre_y) ** 2)
    radial_scale = max(float(np.quantile(radial[colour_mask], 0.98)), 1e-9)
    radial /= radial_scale
    rings = ((0.0, 0.34), (0.34, 0.67), (0.67, 1.20))
    for region_name, candidate in defect_masks.items():
        for ring_index, (lower, upper) in enumerate(rings):
            ring = colour_mask & (radial >= lower) & (radial < upper)
            builder.add(
                f"spatial_{region_name}_ring_{ring_index}",
                (candidate & ring).sum() / max(int(ring.sum()), 1),
            )

    add_shape_features(builder, body)
    vector = np.asarray(builder.values, dtype=np.float32)
    if not np.isfinite(vector).all():
        invalid = [builder.names[index] for index in np.flatnonzero(~np.isfinite(vector))]
        raise ValueError(f"Non-finite handcrafted features: {invalid[:8]}")
    return vector, builder.names


def row_metadata(
    row: dict[str, str],
    segmentation: Segmentation,
    diagnostics: dict[str, int | float],
    canonical_mask: np.ndarray,
) -> dict[str, str | int | float | bool]:
    return {
        "path": row["path"],
        "class_id": int(row["class_id"]),
        "class_name": row.get("class_name", ""),
        "group_id": row["group_id"],
        "source_id": row.get("source_id", ""),
        "origin": row.get("origin", ""),
        "view_type": row.get("view_type", ""),
        "pair_id": row.get("pair_id", ""),
        "selection_role": row.get("selection_role", ""),
        "mask_method": segmentation.method,
        "mask_component_count": int(segmentation.component_count),
        "mask_initial_ratio": float(segmentation.initial_ratio),
        "source_mask_ratio": float(diagnostics["source_mask_ratio"]),
        "canonical_mask_ratio": float(canonical_mask.mean()),
        "source_width": int(diagnostics["source_width"]),
        "source_height": int(diagnostics["source_height"]),
        "has_transparency": segmentation.transparent,
    }


def main() -> None:
    args = parse_args()
    if args.image_size < 96:
        raise ValueError("--image-size must be at least 96")
    if not 0.50 <= args.target_fraction <= 0.98:
        raise ValueError("--target-fraction must be between 0.50 and 0.98")
    manifest = args.manifest.resolve()
    extractor_source = Path(__file__).resolve()
    extractor_sha256 = sha256_file(extractor_source)
    enforce_no_test_protocol(manifest, args.split, args.allow_test)
    rows = read_pure_manifest(manifest, args.split)
    destination = args.output.resolve() / f"quality_{args.split}.pt"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Destination exists; pass --overwrite: {destination}")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    print(
        json.dumps(
            {
                "algorithm": ALGORITHM_VERSION,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "split": args.split,
                "rows": len(rows),
                "image_size": args.image_size,
                "target_fraction": args.target_fraction,
                "test_explicitly_allowed": bool(args.split == "test" and args.allow_test),
            },
            ensure_ascii=False,
        )
    )

    feature_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, str | int | float | bool]] = []
    feature_names: list[str] | None = None
    mask_methods: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        rgb, segmentation, diagnostics = load_and_segment(Path(row["path"]))
        canonical_rgb, canonical_mask = canonical_render(
            rgb,
            segmentation.mask,
            args.image_size,
            args.target_fraction,
        )
        vector, current_names = extract_features(canonical_rgb, canonical_mask)
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise RuntimeError("Feature schema changed between rows")
        feature_rows.append(vector)
        metadata_rows.append(
            row_metadata(row, segmentation, diagnostics, canonical_mask)
        )
        mask_methods[segmentation.method] += 1
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"  extracted {index}/{len(rows)}")

    if feature_names is None:
        raise RuntimeError("No features extracted")
    features = torch.from_numpy(np.stack(feature_rows)).unsqueeze(1).contiguous()
    class_ids = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    class_names = [row.get("class_name", "") for row in rows]
    fallback_methods = {"alpha_raw_fallback", "opaque_colour_distance", "opaque_grabcut", "opaque_central_fallback"}
    fallback_count = sum(mask_methods[name] for name in fallback_methods)
    payload = {
        "features": features,
        "paths": [row["path"] for row in rows],
        "groups": [row["group_id"] for row in rows],
        "class_ids": class_ids,
        "class_names": class_names,
        "source_ids": [row.get("source_id", "") for row in rows],
        "row_metadata": metadata_rows,
        "metadata": {
            "algorithm": ALGORITHM_VERSION,
            # The handcrafted extractor source is the reproducible equivalent
            # of a learned backbone checkpoint.  Selection code can therefore
            # attest the exact implementation just like a .pt backbone.
            "backbone_name": "quality_handcrafted",
            "kind": "handcrafted",
            "checkpoint": str(extractor_source),
            "checkpoint_sha256": extractor_sha256,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "split": args.split,
            "image_size": args.image_size,
            "target_fraction": args.target_fraction,
            "background_rgb": list(DEFAULT_BACKGROUND),
            "seed": args.seed,
            "views": 1,
            "feature_dim": features.shape[-1],
            "feature_names": feature_names,
            "mask_method_counts": dict(sorted(mask_methods.items())),
            "mask_fallback_count": fallback_count,
            "scale_normalized": True,
            "label_independent_extraction": True,
            "physically_separated_manifest_required": True,
            "test_requested_explicitly": bool(args.split == "test" and args.allow_test),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "saved": str(destination),
                "shape": list(features.shape),
                "mask_method_counts": dict(sorted(mask_methods.items())),
                "mask_fallback_count": fallback_count,
                "size_mib": round(destination.stat().st_size / (1024**2), 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
