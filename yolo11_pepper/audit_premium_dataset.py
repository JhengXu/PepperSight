#!/usr/bin/env python3
"""Audit the newly supplied premium pepper photos without mutating source data.

The audit is deliberately label-blind for strict-test references: strict-test
images are read only to compute leakage fingerprints. No model inference or
metric calculation is performed on that split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_IDS = {
    "子弹头_好": 0,
    "子弹头_差": 1,
    "条子_好": 2,
    "条子_差": 3,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Fingerprint:
    sha256: str
    dhash64: str
    phash64: str
    object_dhash64: str
    object_phash64: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def dhash64(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash64(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    flat = dct.reshape(-1)
    threshold = float(np.median(flat[1:]))
    bits = flat > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def load_oriented_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def pepper_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment a red/dark pepper and green stem from a near-white background."""
    scale = min(1.0, 1200.0 / max(rgb.shape[:2]))
    small = cv2.resize(
        rgb,
        (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    # Red pepper skin, green/brown stem, and deep folds. Bright white paper and
    # the soft grey cast shadow are intentionally excluded.
    red = ((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 168)) & (saturation > 45) & (value > 28)
    green_brown = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 100) & (saturation > 35) & (value > 25)
    dark_red = (a_channel > 138) & (value < 170)
    mask = ((red | green_brown | dark_red).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
    # Prefer a central, large component. This suppresses isolated crumbs.
    h, w = mask.shape
    candidates: list[tuple[float, int]] = []
    for idx in range(1, count):
        x, y, cw, ch, area = stats[idx]
        cx, cy = x + cw / 2, y + ch / 2
        centre_penalty = math.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        candidates.append((float(area) * max(0.25, 1.0 - centre_penalty), idx))
    best = max(candidates)[1]
    selected = (labels == best).astype(np.uint8) * 255
    if scale != 1.0:
        selected = cv2.resize(selected, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return selected


def bbox_from_mask(mask: np.ndarray, padding: float = 0.14) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padx = round((x2 - x1) * padding)
    pady = round((y2 - y1) * padding)
    return (
        max(0, x1 - padx),
        max(0, y1 - pady),
        min(mask.shape[1], x2 + padx),
        min(mask.shape[0], y2 + pady),
    )


def standard_object_view(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    mask = pepper_mask(rgb)
    bbox = bbox_from_mask(mask)
    canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
    if bbox is None:
        fallback = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        return fallback, mask, None
    x1, y1, x2, y2 = bbox
    crop = rgb[y1:y2, x1:x2]
    # Use the mask only to find a tight bounding box. Preserve all original
    # pixels inside that box so specular highlights and dark folds are not
    # punched into artificial white holes in the training crop.
    scale = min(226 / max(1, crop.shape[1]), 226 / max(1, crop.shape[0]))
    target_w = max(1, round(crop.shape[1] * scale))
    target_h = max(1, round(crop.shape[0] * scale))
    fitted = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    offset_x = (256 - target_w) // 2
    offset_y = (256 - target_h) // 2
    canvas[offset_y : offset_y + target_h, offset_x : offset_x + target_w] = fitted
    return canvas, mask, bbox


def fingerprints(path: Path) -> tuple[Fingerprint, dict, np.ndarray, tuple[int, int, int, int] | None]:
    rgb = load_oriented_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    object_view, mask, bbox = standard_object_view(rgb)
    object_gray = cv2.cvtColor(object_view, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    object_variance = float(cv2.Laplacian(object_gray, cv2.CV_64F).var())
    metrics = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "megapixels": round(rgb.shape[0] * rgb.shape[1] / 1_000_000, 4),
        "brightness_mean": round(float(gray.mean()), 4),
        "contrast_std": round(float(gray.std()), 4),
        "laplacian_variance": round(variance, 4),
        "object_laplacian_variance": round(object_variance, 4),
        "object_area_fraction": round(float((mask > 0).mean()), 6),
    }
    fp = Fingerprint(
        sha256=sha256_file(path),
        dhash64=dhash64(gray),
        phash64=phash64(gray),
        object_dhash64=dhash64(object_gray),
        object_phash64=phash64(object_gray),
    )
    return fp, metrics, object_view, bbox


def iter_manifest_paths(manifest: Path) -> Iterable[Path]:
    frame = pd.read_csv(manifest)
    seen: set[Path] = set()
    for column in ("path", "source_path", "reference_instance_path", "paired_view_path"):
        if column not in frame.columns:
            continue
        for raw in frame[column].dropna().astype(str):
            path = Path(raw)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in seen:
                seen.add(path)
                yield path


def manifest_source_identifiers(manifest: Path) -> set[str]:
    frame = pd.read_csv(manifest)
    identifiers: set[str] = set()
    if "source_id" in frame.columns:
        identifiers.update(frame["source_id"].dropna().astype(str))
    for column in ("path", "source_path"):
        if column in frame.columns:
            identifiers.update(Path(raw).stem for raw in frame[column].dropna().astype(str))
    return identifiers


def reference_fingerprints(manifests: dict[str, Path]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    cache: dict[Path, dict] = {}
    for split_name, manifest in manifests.items():
        rows: list[dict] = []
        for path in iter_manifest_paths(manifest):
            if path not in cache:
                try:
                    fp, metrics, _, _ = fingerprints(path)
                    cache[path] = {**fp.__dict__, **metrics}
                except Exception as exc:  # pragma: no cover - audit must record, not hide, failures
                    cache[path] = {"decode_error": str(exc)}
            # Strict-test filenames encode labels in their parent directory. Do
            # not export that path; a hash identifier is sufficient for a
            # leakage audit and makes accidental label use impossible downstream.
            display_path = "" if split_name == "strict_test_hash_audit_only" else str(path)
            rows.append(
                {
                    "reference_scope": split_name,
                    "reference_path": display_path,
                    "reference_id": f"sha256:{cache[path].get('sha256', '')}",
                    **cache[path],
                }
            )
        output[split_name] = rows
    return output


def union_find_groups(paths: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    parent = {path: path for path in paths}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)
    roots: dict[str, str] = {}
    for path in paths:
        root = find(path)
        roots.setdefault(root, f"near_duplicate_cluster_{len(roots) + 1:03d}")
    return {path: roots[find(path)] for path in paths}


def contact_sheet(rows: list[dict], output_path: Path, title: str) -> None:
    cell_w, cell_h, columns = 300, 340, 4
    rows_n = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (cell_w * columns, 64 + cell_h * rows_n), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/System/Library/Fonts/PingFang.ttc")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    draw.text((16, 18), title, fill="black", font=font)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = 64 + (index // columns) * cell_h
        with Image.open(row["crop_path"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (280, 270))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        label = f"{index + 1:02d} {Path(row['path']).stem[:10]}\n{row['class_name']}  group:{row['group_id'].split(':')[-1]}"
        draw.multiline_text((x + 8, y + 284), label, fill="black", font=font, spacing=3)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("辣椒_优质"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_premium_v1_audit"),
    )
    parser.add_argument(
        "--clean-train",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v5_clean_audit/manifest.csv"),
    )
    parser.add_argument(
        "--model-selection",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"),
    )
    parser.add_argument(
        "--strict-test",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_merged/strict_test_manifest.csv"),
    )
    args = parser.parse_args()

    output = args.output.resolve()
    crops_dir = output / "crops"
    review_dir = output / "review"
    crops_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        (path.resolve() for path in args.input.glob("*/*") if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (path.parent.name, path.stat().st_birthtime, path.name),
    )
    rows: list[dict] = []
    decode_failures: list[dict] = []
    object_views: dict[str, np.ndarray] = {}
    for path in paths:
        class_name = path.parent.name
        if class_name not in CLASS_IDS:
            continue
        try:
            with Image.open(path) as verify_image:
                verify_image.verify()
            fp, metrics, object_view, bbox = fingerprints(path)
        except Exception as exc:
            decode_failures.append({"path": str(path), "error": str(exc)})
            continue
        object_views[str(path)] = object_view
        crop_class_dir = crops_dir / class_name
        crop_class_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_class_dir / f"{path.stem}__object.jpg"
        Image.fromarray(object_view).save(crop_path, quality=95, subsampling=0)
        bbox_text = "" if bbox is None else json.dumps(list(bbox), ensure_ascii=False)
        review_reasons: list[str] = []
        if bbox is None:
            review_reasons.append("object_segmentation_failed")
        if metrics["object_area_fraction"] < 0.01:
            review_reasons.append("very_small_object")
        if metrics["object_area_fraction"] > 0.55:
            review_reasons.append("object_area_suspiciously_large")
        # Full-frame Laplacian variance is artificially low because most pixels
        # are blank white paper. Blur must be judged on the standardized object.
        if metrics["object_laplacian_variance"] < 150:
            review_reasons.append("possible_blur")
        scene_id = fp.sha256[:20]
        birth_timestamp = float(getattr(path.stat(), "st_birthtime", path.stat().st_mtime))
        # All images came from one short capture session per folder. Keeping the
        # whole class-session in one group prevents background/lighting leakage
        # during any future group-aware cross-validation.
        group_id = f"premium_session:{class_name}:20260829"
        rows.append(
            {
                "path": str(path),
                "crop_path": str(crop_path),
                "class_id": CLASS_IDS[class_name],
                "class_name": class_name,
                "species": class_name.split("_")[0],
                "grade": class_name.split("_")[1],
                "scene_id": scene_id,
                "pair_id": f"premium-photo:{scene_id}",
                "group_id": "premium_session:20260829_1546",
                "capture_session_id": "premium_session:20260829_1546",
                "filesystem_birth_time": datetime.fromtimestamp(birth_timestamp).astimezone().isoformat(),
                "filesystem_birth_timestamp": birth_timestamp,
                "exif_capture_time_available": False,
                "physical_instance_independence_proven": False,
                "label_provenance": "human_folder_label_unadjudicated",
                "bbox_xyxy": bbox_text,
                **metrics,
                **fp.__dict__,
                "review_required": bool(review_reasons),
                "review_reasons": ";".join(review_reasons),
                "safe_against_exact_leakage": True,
                "safe_for_training": False,
                "eligible_for_model_selection": False,
                "eligible_for_strict_test": False,
                "species_weight": 0.0,
                "grade_weight": 0.0,
            }
        )

    # The white background makes object-only hashes spuriously close for peppers
    # with similar silhouettes. A scene/view is therefore called a near
    # duplicate only when *both full-frame* dHash and pHash are extremely close.
    within_pairs: list[dict] = []
    within_object_hash_review: list[dict] = []
    near_edges: list[tuple[str, str]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            distances = {
                "dhash_distance": hamming_hex(left["dhash64"], right["dhash64"]),
                "phash_distance": hamming_hex(left["phash64"], right["phash64"]),
                "object_dhash_distance": hamming_hex(left["object_dhash64"], right["object_dhash64"]),
                "object_phash_distance": hamming_hex(left["object_phash64"], right["object_phash64"]),
            }
            exact = left["sha256"] == right["sha256"]
            votes = int(distances["dhash_distance"] <= 6) + int(
                distances["phash_distance"] <= 6
            )
            near = exact or votes == 2
            object_shape_candidate = (
                distances["object_dhash_distance"] <= 5
                and distances["object_phash_distance"] <= 6
            )
            if object_shape_candidate and not near:
                within_object_hash_review.append(
                    {
                        "left_path": left["path"],
                        "right_path": right["path"],
                        "left_class": left["class_name"],
                        "right_class": right["class_name"],
                        **distances,
                        "training_exclusion": False,
                        "reason": "object-only 8x8 hashes are shape-biased; review warning only",
                    }
                )
            if near:
                within_pairs.append(
                    {
                        "left_path": left["path"],
                        "right_path": right["path"],
                        "left_class": left["class_name"],
                        "right_class": right["class_name"],
                        "exact_sha256": exact,
                        "near_duplicate": near,
                        "hash_vote_count": votes,
                        **distances,
                    }
                )
                near_edges.append((left["path"], right["path"]))

    duplicate_groups = union_find_groups([row["path"] for row in rows], near_edges)
    for row in rows:
        row["near_duplicate_group"] = duplicate_groups[row["path"]]

    manifests = {
        "clean_train": args.clean_train.resolve(),
        "model_selection": args.model_selection.resolve(),
        "strict_test_hash_audit_only": args.strict_test.resolve(),
    }
    references = reference_fingerprints(manifests)
    cross_matches: list[dict] = []
    cross_object_hash_review: list[dict] = []
    for row in rows:
        for scope, reference_rows in references.items():
            best: dict | None = None
            for reference in reference_rows:
                if "decode_error" in reference:
                    continue
                distances = {
                    "dhash_distance": hamming_hex(row["dhash64"], reference["dhash64"]),
                    "phash_distance": hamming_hex(row["phash64"], reference["phash64"]),
                    "object_dhash_distance": hamming_hex(row["object_dhash64"], reference["object_dhash64"]),
                    "object_phash_distance": hamming_hex(row["object_phash64"], reference["object_phash64"]),
                }
                score = sum(distances.values())
                candidate = {**reference, **distances, "distance_sum": score}
                if best is None or candidate["distance_sum"] < best["distance_sum"]:
                    best = candidate
                exact = row["sha256"] == reference["sha256"]
                votes = int(distances["dhash_distance"] <= 6) + int(
                    distances["phash_distance"] <= 6
                )
                if exact or votes == 2:
                    cross_matches.append(
                        {
                            "new_path": row["path"],
                            "reference_scope": scope,
                            "reference_path": reference["reference_path"],
                            "reference_id": reference["reference_id"],
                            "exact_sha256": exact,
                            "near_duplicate": exact or votes == 2,
                            "hash_vote_count": votes,
                            **distances,
                        }
                    )
                object_shape_candidate = (
                    distances["object_dhash_distance"] <= 5
                    and distances["object_phash_distance"] <= 6
                )
                if object_shape_candidate and not (exact or votes == 2):
                    cross_object_hash_review.append(
                        {
                            "new_path": row["path"],
                            "reference_scope": scope,
                            "reference_path": reference["reference_path"],
                            "reference_id": reference["reference_id"],
                            **distances,
                            "training_exclusion": False,
                            "reason": "object-only 8x8 hashes are shape-biased; review warning only",
                        }
                    )
            if best is not None:
                row[f"nearest_{scope}_reference"] = (
                    best["reference_id"]
                    if scope == "strict_test_hash_audit_only"
                    else best["reference_path"]
                )
                row[f"nearest_{scope}_distance_sum"] = best["distance_sum"]
                row[f"nearest_{scope}_object_phash_distance"] = best["object_phash_distance"]

    unsafe_paths = {match["new_path"] for match in cross_matches}
    exact_unsafe_paths = {
        match["new_path"] for match in cross_matches if bool(match["exact_sha256"])
    }
    # Near matches are conservatively withheld pending review even if they are
    # same-class lookalikes. Exact overlap is always unsafe.
    for row in rows:
        row["safe_against_exact_leakage"] = row["path"] not in exact_unsafe_paths
        fatal_review = "object_segmentation_failed" in row["review_reasons"]
        row["safe_for_training"] = row["path"] not in unsafe_paths and not fatal_review
        if row["safe_for_training"]:
            # The 63 photos are only ~9% of clean-v5 rows. A half-weight warm
            # start lets XGBoost learn the clean white-background domain without
            # allowing one short capture session to dominate historical data.
            row["species_weight"] = 0.25 if row["review_required"] else 0.5
            row["grade_weight"] = 0.25 if row["review_required"] else 0.5

    fieldnames = list(rows[0].keys()) if rows else []
    with (output / "premium_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "safe_training_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([row for row in rows if row["safe_for_training"]])
    pd.DataFrame(within_pairs).to_csv(output / "within_near_duplicates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cross_matches).to_csv(output / "cross_split_overlap_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(within_object_hash_review).to_csv(
        output / "within_object_hash_review_candidates.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cross_object_hash_review).to_csv(
        output / "cross_object_hash_review_candidates.csv", index=False, encoding="utf-8-sig"
    )

    for class_name in CLASS_IDS:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        contact_sheet(class_rows, review_dir / f"contact_{class_name}.jpg", f"Premium audit: {class_name}")
    contact_sheet(rows, review_dir / "contact_all.jpg", "Premium pepper dataset - all images")

    exact_within = sum(bool(item["exact_sha256"]) for item in within_pairs)
    near_within = len(within_pairs) - exact_within
    class_counts = Counter(row["class_name"] for row in rows)
    birth_timestamps = [float(row["filesystem_birth_timestamp"]) for row in rows]
    new_source_identifiers = {Path(row["path"]).stem for row in rows}
    source_identifier_overlap_counts = {
        scope: len(new_source_identifiers & manifest_source_identifiers(manifest))
        for scope, manifest in manifests.items()
    }
    summary = {
        "schema_version": 1,
        "source_directory": str(args.input.resolve()),
        "output_directory": str(output),
        "image_count": len(rows),
        "class_counts": dict(class_counts),
        "decode_failure_count": len(decode_failures),
        "decode_failures": decode_failures,
        "resolution_counts": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "exact_duplicate_pairs_within_new": exact_within,
        "near_duplicate_pairs_within_new": near_within,
        "capture_session_count": len({row["capture_session_id"] for row in rows}),
        "capture_session_inference": {
            "exif_capture_timestamps_available": False,
            "filesystem_birth_time_span_seconds": round(max(birth_timestamps) - min(birth_timestamps), 3),
            "basis": "filesystem birth times plus shared white background; conservatively grouped as one batch",
        },
        "object_hash_warning_counts": {
            "within_new": len(within_object_hash_review),
            "cross_reference": len(cross_object_hash_review),
            "policy": "warning only; not used to exclude training because 8x8 object hashes are shape-biased",
        },
        "cross_reference_image_counts": {
            scope: len(reference_rows) for scope, reference_rows in references.items()
        },
        "source_identifier_overlap_counts": source_identifier_overlap_counts,
        "cross_exact_overlap_counts": dict(
            Counter(match["reference_scope"] for match in cross_matches if match["exact_sha256"])
        ),
        "cross_near_overlap_counts": dict(
            Counter(match["reference_scope"] for match in cross_matches if not match["exact_sha256"])
        ),
        "safe_training_image_count": sum(bool(row["safe_for_training"]) for row in rows),
        "review_required_count": sum(bool(row["review_required"]) for row in rows),
        "strict_test_policy": {
            "read_for_hash_leakage_audit_only": True,
            "test_labels_used": False,
            "model_inference_performed": False,
            "test_metrics_computed": False,
            "test_images_rendered_or_exported": False,
        },
        "grouping_recommendation": {
            "group_id": "one group for the entire six-minute capture session across all four folders",
            "pair_id": "one id per source photo; each source photo yields exactly one object crop",
            "cross_validation_rule": "training-only batch: never place any of this capture_session_id in model-selection or test",
        },
        "weighting_recommendation": {
            "initial_species_weight": 0.5,
            "initial_grade_weight": 0.5,
            "reason": "new images are high resolution but share one white-background capture session and labels are not independently adjudicated",
        },
        "caveat": "Perceptual hashes screen scene/view duplication; they cannot prove that two differently posed photos are distinct physical peppers. Contact sheets require human review before claiming physical-instance independence.",
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    receipt_paths = [
        output / "premium_manifest.csv",
        output / "safe_training_manifest.csv",
        output / "within_near_duplicates.csv",
        output / "cross_split_overlap_candidates.csv",
        output / "within_object_hash_review_candidates.csv",
        output / "cross_object_hash_review_candidates.csv",
        output / "audit_summary.json",
    ]
    receipt = {
        "files": {str(path): sha256_file(path) for path in receipt_paths if path.exists()},
        "source_images": {row["path"]: row["sha256"] for row in rows},
    }
    (output / "receipt.sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
