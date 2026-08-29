#!/usr/bin/env python3
"""Prepare the small ``辣椒_优质`` increment without touching test data.

The premium photographs are one opaque-background acquisition batch.  This
utility segments each pepper, normalizes EXIF orientation, writes a transparent
PNG, and emits a training-only manifest suitable for the existing frozen
YOLO11 feature extractor.  The 62 accepted rows receive weight 0.50 and the
one blur-review row receives 0.25, for 31.25 effective samples before
source-group normalization.

No automatic train/validation split is made inside this batch.  Treating
correlated photographs from the same table/camera as independent validation
data would make the model-selection result optimistic.  Existing physically
separated clean-v5 validation data remains the only validation source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from extract_quality_features import Segmentation, canonical_render, largest_component


SCRIPT_VERSION = "premium-feature-strategy-v1"
CLASS_MAP = {
    "子弹头_好": (0, "子弹头_一级", 0, 0),
    "子弹头_差": (1, "子弹头_二级", 0, 1),
    "条子_好": (2, "条子_一级", 1, 0),
    "条子_差": (3, "条子_二级", 1, 1),
}
ACCEPTED_INCREMENT_WEIGHT = 0.50
REVIEW_INCREMENT_WEIGHT = 0.25
REVIEW_FLAGS = {
    "19e15d37a97750a4e7e043dee72f2160.jpg": "possible_blur_requires_human_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("辣椒_优质"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/runs/premium_feature_strategy_v1"),
    )
    parser.add_argument(
        "--clean-train-manifest",
        type=Path,
        default=Path(
            "yolo11_pepper/datasets/pepper_ssl_v5_clean_audit/"
            "train_label_audit_paired.csv"
        ),
    )
    parser.add_argument(
        "--clean-val-manifest",
        type=Path,
        default=Path(
            "yolo11_pepper/datasets/pepper_ssl_v4_merged/"
            "model_selection_manifest.csv"
        ),
    )
    parser.add_argument("--render-size", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_known_hashes(paths: list[Path]) -> set[str]:
    """Read only train/validation manifest metadata, never test manifests."""
    hashes: set[str] = set()
    for path in paths:
        with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = (row.get("content_sha256") or "").strip().lower()
                if len(value) == 64:
                    hashes.add(value)
    return hashes


def perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    transformed = cv2.dct(small)
    block = transformed[:8, :8].copy()
    median = float(np.median(block.flatten()[1:]))
    bits = block.flatten() >= median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def transparent_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty pepper mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rgba = np.dstack((rgb, np.where(mask, 255, 0).astype(np.uint8)))
    return Image.fromarray(rgba[y0:y1, x0:x1], mode="RGBA")


def load_and_segment_premium(
    path: Path,
) -> tuple[np.ndarray, Segmentation, dict[str, int | float]]:
    """Segment this controlled white-table domain without retaining shadows.

    The general quality-feature segmenter intentionally tolerates heterogeneous
    opaque backgrounds.  On this acquisition batch that tolerance can attach a
    pale cast-shadow region to the pepper.  A narrower chromatic mask is safer:
    saturated red/brown forms the body and saturated green retains the stem.
    """
    with Image.open(path) as source:
        normalized = ImageOps.exif_transpose(source).convert("RGBA")
        if max(normalized.size) > 1600:
            scale = 1600 / max(normalized.size)
            normalized = normalized.resize(
                (
                    max(1, round(normalized.width * scale)),
                    max(1, round(normalized.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        rgba = np.asarray(normalized, dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[..., :3])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)

    red_hue = (hue <= 28) | (hue >= 164)
    chroma_gap = red - np.minimum(green, blue)
    red_body = (
        red_hue
        & (saturation >= 58)
        & (value >= 13)
        & (red >= green * 0.93)
        & (red >= blue * 0.91)
        & (chroma_gap >= 13)
    )
    dark_brown = (
        (saturation >= 48)
        & (value >= 10)
        & (red >= green * 1.07)
        & (red >= blue * 1.06)
        & (chroma_gap >= 16)
    )
    green_stem = (
        (hue >= 28)
        & (hue <= 78)
        & (saturation >= 56)
        & (value >= 15)
        & (green >= red * 0.82)
        & (green >= blue * 1.07)
    )
    initial = red_body | dark_brown | green_stem
    # A small close joins highlight gaps without bridging a curved pepper's
    # concavity.  The general segmenter fills every enclosed hole, which can
    # incorrectly turn the pale region inside a curved pepper into foreground.
    kernel_size = max(3, round(min(initial.shape) * 0.006))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(initial.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask, components = largest_component(cleaned.astype(bool))

    # Recover genuinely black/charred tissue touching the chromatic body.  It
    # is a grade cue and must not become a transparent hole; white/grey cast
    # shadows are excluded by the low-value and minimum-saturation conditions.
    attach_size = max(5, round(min(mask.shape) * 0.018))
    if attach_size % 2 == 0:
        attach_size += 1
    near_body = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (attach_size, attach_size)),
    ).astype(bool)
    dark_attached = (value <= 92) & (saturation >= 24) & near_body
    mask, _ = largest_component((mask | dark_attached).astype(bool))

    # Fill only tiny internal highlight holes.  Large holes/concavities remain
    # background so a cast shadow cannot become a large artificial appendage.
    inverse = (~mask).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    border_labels = set(
        np.unique(
            np.concatenate(
                (
                    hole_labels[0],
                    hole_labels[-1],
                    hole_labels[:, 0],
                    hole_labels[:, -1],
                )
            )
        ).tolist()
    )
    maximum_hole = max(48, round(mask.sum() * 0.0025))
    for label in range(1, hole_count):
        if label in border_labels:
            continue
        if int(hole_stats[label, cv2.CC_STAT_AREA]) <= maximum_hole:
            mask[hole_labels == label] = True
    ratio = float(mask.mean())
    if not 0.008 <= ratio <= 0.35:
        raise ValueError(f"unsafe premium segmentation ratio={ratio:.6f}: {path}")
    segmentation = Segmentation(
        mask=mask,
        method="premium_chromatic_no_shadow",
        component_count=components,
        initial_ratio=float(initial.mean()),
        transparent=False,
    )
    diagnostics = {
        "source_width": int(rgb.shape[1]),
        "source_height": int(rgb.shape[0]),
        "source_mask_ratio": ratio,
    }
    return rgb, segmentation, diagnostics


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def contact_sheet(items: list[tuple[str, str, np.ndarray]], destination: Path) -> None:
    columns = 8
    tile_width, tile_height = 256, 304
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    label_font = font(18)
    for index, (class_name, stem, rendered) in enumerate(items):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(Image.fromarray(rendered), (x, y))
        draw.rectangle((x, y + 256, x + tile_width, y + tile_height), fill=(245, 245, 245))
        draw.text((x + 8, y + 260), class_name, fill=(20, 20, 20), font=label_font)
        draw.text((x + 8, y + 282), stem[:12], fill=(90, 90, 90), font=font(13))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    segmented_root = output / "segmented"
    output.mkdir(parents=True, exist_ok=True)
    known_hashes = read_known_hashes(
        [args.clean_train_manifest.resolve(), args.clean_val_manifest.resolve()]
    )

    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise ValueError(f"no images under {source}")
    unknown_classes = sorted({path.parent.name for path in images} - set(CLASS_MAP))
    if unknown_classes:
        raise ValueError(f"unknown class folders: {unknown_classes}")

    records: list[dict[str, Any]] = []
    preview: list[tuple[str, str, np.ndarray]] = []
    perceptual: list[tuple[Path, int, int]] = []
    original_hashes: list[str] = []
    orientation_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    mask_methods: Counter[str] = Counter()
    mask_ratios: list[float] = []

    for path in images:
        class_id, model_class_name, species_id, grade_id = CLASS_MAP[path.parent.name]
        original_sha = sha256_file(path)
        original_hashes.append(original_sha)
        with Image.open(path) as raw:
            orientation_counts[str(raw.getexif().get(274, "missing"))] += 1
            normalized = ImageOps.exif_transpose(raw)
            size_counts[f"{normalized.width}x{normalized.height}"] += 1
        rgb, segmentation, diagnostics = load_and_segment_premium(path)
        mask_methods[segmentation.method] += 1
        mask_ratios.append(float(diagnostics["source_mask_ratio"]))

        destination = segmented_root / path.parent.name / f"{path.stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        transparent_crop(rgb, segmentation.mask).save(destination, optimize=True)
        segmented_sha = sha256_file(destination)
        rendered, rendered_mask = canonical_render(
            rgb, segmentation.mask, args.render_size, 0.88
        )
        phash = perceptual_hash(rendered)
        perceptual.append((path, class_id, phash))
        preview_name = path.parent.name
        if path.name in REVIEW_FLAGS:
            preview_name += " [LOW-WEIGHT: BLUR]"
        preview.append((preview_name, path.stem, rendered))

        pair_id = f"premium-instance:{original_sha[:20]}"
        increment_weight = (
            REVIEW_INCREMENT_WEIGHT
            if path.name in REVIEW_FLAGS
            else ACCEPTED_INCREMENT_WEIGHT
        )
        records.append(
            {
                "path": str(destination.resolve()),
                "source_path": str(path.resolve()),
                "split": "train",
                "class_id": class_id,
                "class_name": path.parent.name,
                "model_class_name": model_class_name,
                "species_id": species_id,
                "grade_id": grade_id,
                "group_id": "premium:single_acquisition_batch:v1",
                "source_id": "premium:single_acquisition_batch:v1",
                "pair_id": pair_id,
                "content_sha256": segmented_sha,
                "source_content_sha256": original_sha,
                "origin": "premium_opaque_segmented",
                "view_type": "canonical",
                "record_role": "premium_incremental_train",
                "selection_role": "training",
                "eligible_for_model_training": "true",
                "eligible_for_model_selection": "false",
                "safe_for_training": "true",
                "species_weight": f"{increment_weight:.6f}",
                "grade_weight": f"{increment_weight:.6f}",
                "grade_label_source": "premium_manual_folder",
                "label_action": (
                    "hard_keep_low_increment_weight_review_blur"
                    if path.name in REVIEW_FLAGS
                    else "hard_keep_low_increment_weight"
                ),
                "quality_review": REVIEW_FLAGS.get(path.name, "accepted"),
                "mask_method": segmentation.method,
                "source_mask_ratio": f"{diagnostics['source_mask_ratio']:.8f}",
                "canonical_mask_ratio": f"{rendered_mask.mean():.8f}",
            }
        )

    exact_duplicates = len(original_hashes) - len(set(original_hashes))
    overlap_count = sum(item in known_hashes for item in original_hashes)
    near_pairs: list[dict[str, Any]] = []
    nearest_distances: list[int] = []
    for index, (first_path, first_class, first_hash) in enumerate(perceptual):
        distances = []
        for second_path, second_class, second_hash in perceptual:
            if second_path == first_path:
                continue
            distances.append(hamming(first_hash, second_hash))
        nearest_distances.append(min(distances))
        for second_path, second_class, second_hash in perceptual[index + 1 :]:
            distance = hamming(first_hash, second_hash)
            if distance <= 4:
                near_pairs.append(
                    {
                        "first": str(first_path.resolve()),
                        "second": str(second_path.resolve()),
                        "distance": distance,
                        "same_class": first_class == second_class,
                    }
                )

    manifest_path = output / "premium_train_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    contact_sheet(preview, output / "segmentation_contact_sheet.jpg")

    class_counts = Counter(record["model_class_name"] for record in records)
    report = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "strict_test_opened": False,
            "splits_emitted": ["train"],
            "validation_policy": (
                "do not split the single acquisition batch; retain the existing "
                "physically separated clean-v5 validation set"
            ),
        },
        "source": str(source),
        "source_rows": len(images),
        "training_rows": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "low_weight_review_flags": [
            {
                "path": str((source / "条子_好" / name).resolve()),
                "reason": reason,
                "included_at_fixed_weight": REVIEW_INCREMENT_WEIGHT,
            }
            for name, reason in REVIEW_FLAGS.items()
        ],
        "capture_domain": {
            "opaque_background": True,
            "normalized_sizes": dict(size_counts),
            "exif_orientation_before_normalization": dict(orientation_counts),
            "single_acquisition_group": True,
            "group_id": "premium:single_acquisition_batch:v1",
        },
        "segmentation": {
            "methods": dict(mask_methods),
            "mask_ratio_min": min(mask_ratios),
            "mask_ratio_median": float(np.median(mask_ratios)),
            "mask_ratio_max": max(mask_ratios),
            "transparent_pngs": len(images),
        },
        "identity_audit": {
            "exact_duplicate_files": exact_duplicates,
            "exact_sha_overlap_with_clean_train_or_validation": overlap_count,
            "phash_distance_threshold": 4,
            "near_duplicate_pairs": near_pairs,
            "nearest_phash_distance_min": min(nearest_distances),
            "nearest_phash_distance_median": float(np.median(nearest_distances)),
        },
        "increment_policy": {
            "accepted_weight_per_image": ACCEPTED_INCREMENT_WEIGHT,
            "review_weight_per_image": REVIEW_INCREMENT_WEIGHT,
            "effective_total_weight": (
                (len(records) - len(REVIEW_FLAGS)) * ACCEPTED_INCREMENT_WEIGHT
                + len(REVIEW_FLAGS) * REVIEW_INCREMENT_WEIGHT
            ),
            "rationale": (
                "63 images come from one camera/table batch. The 62 accepted rows "
                "receive 0.50 and the one blur-review row receives 0.25, for 31.25 "
                "effective samples before the XGBoost source-group normalization. "
                "This limits batch-specific colour and lighting influence."
            ),
        },
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "contact_sheet": str((output / "segmentation_contact_sheet.jpg").resolve()),
            "segmented_root": str(segmented_root.resolve()),
        },
    }
    report_path = output / "audit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "audit_report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
        "contact_sheet": {
            "path": str((output / "segmentation_contact_sheet.jpg").resolve()),
            "sha256": sha256_file(output / "segmentation_contact_sheet.jpg"),
        },
    }
    (output / "hash_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
