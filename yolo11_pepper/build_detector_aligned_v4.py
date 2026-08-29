#!/usr/bin/env python3
"""Build a leakage-safe detector-aligned pepper crop dataset.

The source identity, class and split are inherited from scene groups in the v3
manifest.  The current production pepper detector is then run on each original
scene photograph and every accepted box is cropped with the same padding used
by ``PepperModelService``.

Only matched crops from v3 *training* groups are eligible for training.  Crops
from validation and test groups are always marked ``evaluation_only`` and are
written below a physically separate directory.  The script refuses to replace
an existing output directory and publishes the completed dataset atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import torch
from ultralytics import YOLO


CLASS_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")
MANIFEST_FIELDS = (
    "path",
    "source_path",
    "origin",
    "split",
    "usage",
    "eligible_for_training",
    "group_id",
    "source_id",
    "source_class_id",
    "source_class_name",
    "source_species",
    "source_grade",
    "source_label_provenance",
    "label_state",
    "detector_checkpoint",
    "detector_checkpoint_sha256",
    "detector_class_id",
    "detector_class_name",
    "detector_confidence",
    "detector_bbox_xyxy",
    "crop_bbox_xyxy",
    "box_width",
    "box_height",
    "elongation",
    "detector_rank_x",
    "reference_instance_path",
    "reference_bbox_xyxy",
    "reference_iou",
    "reference_match",
    "image_width",
    "image_height",
    "crop_sha256",
)


@dataclass(frozen=True)
class ReferenceInstance:
    path: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SceneGroup:
    group_id: str
    source_id: str
    source_path: Path
    split: str
    class_id: int
    class_name: str
    references: tuple[ReferenceInstance, ...]


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    elongation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"),
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("yolo11_pepper/runs/yolo11n_pepper/weights/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_detector_aligned_v4"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--min-elongation", type=float, default=1.30)
    parser.add_argument("--max-peppers", type=int, default=12)
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    parser.add_argument("--reference-min-iou", type=float, default=0.30)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_box(value: str) -> tuple[int, int, int, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    return tuple(int(round(float(item))) for item in parsed)


def read_scene_groups(manifest: Path) -> list[SceneGroup]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("origin") == "scene"]
    if not rows:
        raise RuntimeError(f"No scene rows found in v3 manifest: {manifest}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)

    groups: list[SceneGroup] = []
    source_owner: dict[str, tuple[str, str, str]] = {}
    for group_id, members in sorted(grouped.items()):
        identity = {
            (row["source_id"], row["source_path"], row["split"], row["class_id"], row["class_name"])
            for row in members
        }
        if len(identity) != 1:
            raise RuntimeError(f"Inconsistent v3 identity within group {group_id}: {sorted(identity)}")
        source_id, source_path_text, split, class_id_text, class_name = identity.pop()
        if split not in {"train", "val", "test"}:
            raise RuntimeError(f"Unsupported split {split!r} for {group_id}")
        if class_name not in CLASS_NAMES or int(class_id_text) != CLASS_NAMES.index(class_name):
            raise RuntimeError(f"Invalid class mapping for {group_id}: {class_id_text}/{class_name}")
        owner = (split, class_name, str(Path(source_path_text).resolve()))
        prior_owner = source_owner.setdefault(source_id, owner)
        if prior_owner != owner:
            raise RuntimeError(f"Source {source_id} has conflicting v3 ownership: {prior_owner}, {owner}")
        source_path = Path(source_path_text).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Original scene is missing: {source_path}")
        references = tuple(
            ReferenceInstance(path=row["path"], bbox=parse_box(row["bbox_xyxy"]))
            for row in members
        )
        groups.append(
            SceneGroup(
                group_id=group_id,
                source_id=source_id,
                source_path=source_path,
                split=split,
                class_id=int(class_id_text),
                class_name=class_name,
                references=references,
            )
        )
    return groups


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clamp_box(
    box: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    x2 = min(max(x1 + 1, x2), width)
    y2 = min(max(y1 + 1, y2), height)
    return x1, y1, x2, y2


def padded_box(
    box: tuple[int, int, int, int], width: int, height: int, ratio: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = max(4, round((x2 - x1) * ratio))
    pad_y = max(4, round((y2 - y1) * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def box_iou(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_references(
    detections: list[Detection], references: tuple[ReferenceInstance, ...], minimum_iou: float
) -> dict[int, tuple[int, float]]:
    """Greedily create a one-to-one highest-IoU detection/reference matching."""
    candidates = sorted(
        (
            (box_iou(detection.bbox, reference.bbox), detection_index, reference_index)
            for detection_index, detection in enumerate(detections)
            for reference_index, reference in enumerate(references)
        ),
        reverse=True,
    )
    matches: dict[int, tuple[int, float]] = {}
    used_references: set[int] = set()
    for overlap, detection_index, reference_index in candidates:
        if overlap < minimum_iou:
            break
        if detection_index in matches or reference_index in used_references:
            continue
        matches[detection_index] = (reference_index, overlap)
        used_references.add(reference_index)
    return matches


def detector_boxes(
    model: YOLO,
    image,
    confidence: float,
    iou: float,
    min_elongation: float,
    max_peppers: int,
    device: str,
) -> list[Detection]:
    result = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        agnostic_nms=True,
        max_det=max_peppers * 3,
        verbose=False,
        device=device,
    )[0]
    height, width = image.shape[:2]
    detections: list[Detection] = []
    if result.boxes is not None:
        class_names = model.names
        for box, score, class_id_value in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
        ):
            bbox = clamp_box(box, width, height)
            box_width = max(1, bbox[2] - bbox[0])
            box_height = max(1, bbox[3] - bbox[1])
            elongation = max(box_width / box_height, box_height / box_width)
            if elongation < min_elongation:
                continue
            class_id = int(class_id_value)
            class_name = (
                class_names.get(class_id, class_id)
                if isinstance(class_names, dict)
                else class_names[class_id]
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(score),
                    class_id=class_id,
                    class_name=str(class_name),
                    elongation=float(elongation),
                )
            )
    detections.sort(key=lambda item: item.bbox[0])
    return detections[:max_peppers]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(groups: list[SceneGroup], rows: list[dict[str, object]]) -> dict[str, object]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
        source_splits[str(row["source_id"])].add(str(row["split"]))
        hash_splits[str(row["crop_sha256"])].add(str(row["split"]))
    crossed_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    crossed_sources = {key: sorted(value) for key, value in source_splits.items() if len(value) > 1}
    crossed_hashes = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    training_violations = [
        str(row["path"])
        for row in rows
        if str(row["usage"]) == "training" and str(row["split"]) != "train"
    ]
    group_source_splits: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        group_source_splits[group.source_id].add(group.split)
    source_identity_violations = {
        source: sorted(splits)
        for source, splits in group_source_splits.items()
        if len(splits) > 1
    }
    passed = not any(
        (crossed_groups, crossed_sources, crossed_hashes, training_violations, source_identity_violations)
    )
    return {
        "passed": passed,
        "group_cross_split": crossed_groups,
        "source_cross_split": crossed_sources,
        "exact_crop_hash_cross_split": crossed_hashes,
        "non_train_marked_training": training_violations,
        "source_identity_cross_split": source_identity_violations,
        "unique_crop_hashes": len(hash_splits),
        "total_crop_rows": len(rows),
    }


def nested_counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    manifest = args.v3_manifest.resolve()
    detector_path = args.detector.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {output}. Choose a new --output path."
        )
    if not detector_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint is missing: {detector_path}")
    if args.max_peppers < 1:
        raise ValueError("--max-peppers must be positive")
    if not 0.0 <= args.padding_ratio <= 1.0:
        raise ValueError("--padding-ratio must be between 0 and 1")

    groups = read_scene_groups(manifest)
    device = choose_device(args.device)
    detector_sha = sha256_file(detector_path)
    detector = YOLO(detector_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    print(f"device={device} groups={len(groups)} building={building}")

    rows: list[dict[str, object]] = []
    source_reports: list[dict[str, object]] = []
    for group in groups:
        image = cv2.imread(str(group.source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read source image: {group.source_path}")
        height, width = image.shape[:2]
        detections = detector_boxes(
            detector,
            image,
            args.confidence,
            args.iou,
            args.min_elongation,
            args.max_peppers,
            device,
        )
        matches = match_references(detections, group.references, args.reference_min_iou)
        matched_count = 0
        for index, detection in enumerate(detections, 1):
            match = matches.get(index - 1)
            reference = group.references[match[0]] if match else None
            reference_iou = match[1] if match else 0.0
            matched_count += int(match is not None)
            if group.split == "train" and match is not None:
                usage = "training"
                prefix = Path("crops") / "train"
            elif group.split in {"val", "test"}:
                usage = "evaluation_only"
                prefix = Path("crops") / "evaluation_only" / group.split
            else:
                usage = "review_only"
                prefix = Path("crops") / "review_only"
            relative = (
                prefix
                / group.class_name
                / group.source_id
                / f"{group.source_id}__detector_{index:03d}.png"
            )
            temporary_path = building / relative
            final_path = output / relative
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            crop_box = padded_box(detection.bbox, width, height, args.padding_ratio)
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or not cv2.imwrite(str(temporary_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Could not save crop: {temporary_path}")
            species, grade = group.class_name.split("_", 1)
            rows.append(
                {
                    "path": str(final_path),
                    "source_path": str(group.source_path),
                    "origin": "detector_aligned_scene",
                    "split": group.split,
                    "usage": usage,
                    "eligible_for_training": str(usage == "training").lower(),
                    "group_id": group.group_id,
                    "source_id": group.source_id,
                    "source_class_id": group.class_id,
                    "source_class_name": group.class_name,
                    "source_species": species,
                    "source_grade": grade,
                    "source_label_provenance": "v3_manifest_scene_group",
                    "label_state": (
                        "source_group_label_matched_to_v3_instance"
                        if match is not None
                        else "source_group_label_unverified_detection"
                    ),
                    "detector_checkpoint": str(detector_path),
                    "detector_checkpoint_sha256": detector_sha,
                    "detector_class_id": detection.class_id,
                    "detector_class_name": detection.class_name,
                    "detector_confidence": f"{detection.confidence:.8f}",
                    "detector_bbox_xyxy": json.dumps(detection.bbox),
                    "crop_bbox_xyxy": json.dumps(crop_box),
                    "box_width": detection.bbox[2] - detection.bbox[0],
                    "box_height": detection.bbox[3] - detection.bbox[1],
                    "elongation": f"{detection.elongation:.8f}",
                    "detector_rank_x": index,
                    "reference_instance_path": reference.path if reference else "",
                    "reference_bbox_xyxy": json.dumps(reference.bbox) if reference else "",
                    "reference_iou": f"{reference_iou:.8f}",
                    "reference_match": str(match is not None).lower(),
                    "image_width": width,
                    "image_height": height,
                    "crop_sha256": sha256_file(temporary_path),
                }
            )
        source_reports.append(
            {
                "group_id": group.group_id,
                "source_id": group.source_id,
                "source_path": str(group.source_path),
                "split": group.split,
                "class_id": group.class_id,
                "class_name": group.class_name,
                "v3_reference_instances": len(group.references),
                "detector_crops": len(detections),
                "matched_references": matched_count,
                "reference_coverage": matched_count / len(group.references),
            }
        )
        print(
            f"{group.split:5s} {group.class_name}/{group.source_id}: "
            f"detector={len(detections)} matched={matched_count}/{len(group.references)}"
        )

    audit = audit_dataset(groups, rows)
    if not audit["passed"]:
        raise RuntimeError(f"Leakage audit failed; unpublished build remains at {building}: {audit}")
    training_rows = [row for row in rows if row["usage"] == "training"]
    evaluation_rows = [row for row in rows if row["usage"] == "evaluation_only"]
    review_rows = [row for row in rows if row["usage"] == "review_only"]
    write_csv(building / "manifest.csv", rows)
    write_csv(building / "train_manifest.csv", training_rows)
    write_csv(building / "evaluation_manifest.csv", evaluation_rows)

    total_references = sum(len(group.references) for group in groups)
    total_matches = sum(int(row["reference_match"] == "true") for row in rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "output": str(output),
        "detector": {
            "checkpoint": str(detector_path),
            "checkpoint_sha256": detector_sha,
            "device": device,
            "confidence": args.confidence,
            "iou": args.iou,
            "agnostic_nms": True,
            "min_elongation": args.min_elongation,
            "max_peppers": args.max_peppers,
            "padding_ratio": args.padding_ratio,
            "reference_min_iou": args.reference_min_iou,
        },
        "policy": {
            "training": "matched detector crops inherited only from v3 train groups",
            "validation_test": "evaluation_only; never emitted in train_manifest.csv",
            "unmatched_train_detections": "review_only; never eligible for training",
            "label_provenance": "scene-level label inherited from the immutable v3 group",
        },
        "totals": {
            "source_groups": len(groups),
            "v3_reference_instances": total_references,
            "detector_crops": len(rows),
            "reference_matches": total_matches,
            "reference_coverage": total_matches / total_references,
            "training_eligible": len(training_rows),
            "evaluation_only": len(evaluation_rows),
            "review_only": len(review_rows),
        },
        "crops_by_split": nested_counts(rows, "split"),
        "crops_by_usage": nested_counts(rows, "usage"),
        "crops_by_source_class": nested_counts(rows, "source_class_name"),
        "groups_by_split": dict(sorted(Counter(group.split for group in groups).items())),
        "source_reports": source_reports,
        "leakage_audit": audit,
    }
    (building / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (building / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(building, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
