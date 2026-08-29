#!/usr/bin/env python3
"""Build a leakage-safe YOLO pepper detector dataset.

This builder deliberately accepts two physically separated inputs only:

* the merged v4 training manifest (expected split: ``train``), and
* the merged v4 model-selection manifest (expected split: ``val``).

Only canonical ``origin=scene`` rows with an original ``source_path`` and
``bbox_xyxy`` are selected.  Each original scene is copied once and every
selected bbox row is emitted into its YOLO label file.  The default
``class_agnostic`` mode maps every pepper to class 0 (``pepper``), while
``four_class`` preserves the canonical four-way class id/name.  The
strict-test manifest is never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2


CLASS_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")
REQUIRED_COLUMNS = {
    "source_path",
    "origin",
    "split",
    "group_id",
    "source_id",
    "class_id",
    "class_name",
    "record_role",
    "selection_role",
    "view_type",
    "bbox_xyxy",
}
IMAGE_MANIFEST_FIELDS = (
    "split",
    "image_path",
    "label_path",
    "source_path",
    "source_id",
    "group_id",
    "source_class_id",
    "source_class_name",
    "source_sha256",
    "image_width",
    "image_height",
    "box_count",
)
BOX_MANIFEST_FIELDS = (
    "split",
    "input_manifest",
    "input_row_number",
    "image_path",
    "label_path",
    "source_path",
    "source_id",
    "group_id",
    "source_class_id",
    "source_class_name",
    "source_bbox_xyxy",
    "yolo_class_id",
    "yolo_xywh_normalized",
)


@dataclass(frozen=True)
class BoxRow:
    split: str
    input_manifest: Path
    input_row_number: int
    source_path: Path
    source_id: str
    group_id: str
    source_class_id: int
    source_class_name: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ReadReport:
    manifest: Path
    manifest_sha256: str
    expected_split: str
    expected_selection_role: str
    total_rows: int
    selected_scene_rows: int
    ignored_non_scene_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_merged/train_manifest.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(
            "yolo11_pepper/datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_detector_strict_v5"),
    )
    parser.add_argument(
        "--label-mode",
        choices=("class_agnostic", "four_class"),
        default="class_agnostic",
        help="Emit one pepper class (default) or preserve the canonical four classes.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bbox(value: str, *, manifest: Path, row_number: int) -> tuple[float, float, float, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid bbox JSON at {manifest}:{row_number}: {value!r}") from error
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid bbox at {manifest}:{row_number}: {value!r}")
    try:
        bbox = tuple(float(item) for item in parsed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric bbox at {manifest}:{row_number}: {value!r}") from error
    if not all(value == value and abs(value) != float("inf") for value in bbox):
        raise ValueError(f"Non-finite bbox at {manifest}:{row_number}: {value!r}")
    return bbox  # type: ignore[return-value]


def read_scene_rows(
    manifest: Path,
    *,
    expected_split: str,
    expected_selection_role: str,
) -> tuple[list[BoxRow], ReadReport]:
    """Read one physically separated split manifest and select canonical scenes."""
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {manifest}")
    selected: list[BoxRow] = []
    ignored_non_scene = 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing required columns in {manifest}: {sorted(missing)}")
        total_rows = 0
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            if row["split"] != expected_split:
                raise RuntimeError(
                    f"Physical split violation at {manifest}:{row_number}: "
                    f"expected {expected_split!r}, got {row['split']!r}"
                )
            if row["selection_role"] != expected_selection_role:
                raise RuntimeError(
                    f"Selection-role violation at {manifest}:{row_number}: expected "
                    f"{expected_selection_role!r}, got {row['selection_role']!r}"
                )
            is_canonical_scene = (
                row["origin"] == "scene"
                and row["view_type"] == "canonical"
                and row["record_role"] == "canonical_audited"
            )
            if not is_canonical_scene:
                ignored_non_scene += 1
                continue
            if not row["source_path"].strip() or not row["bbox_xyxy"].strip():
                raise RuntimeError(
                    f"Canonical scene row lacks source_path/bbox_xyxy at {manifest}:{row_number}"
                )
            try:
                source_class_id = int(row["class_id"])
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid canonical class_id at {manifest}:{row_number}: "
                    f"{row['class_id']!r}"
                ) from error
            if not 0 <= source_class_id < len(CLASS_NAMES):
                raise RuntimeError(
                    f"Canonical class_id out of range at {manifest}:{row_number}: "
                    f"{source_class_id}"
                )
            if row["class_name"] != CLASS_NAMES[source_class_id]:
                raise RuntimeError(
                    f"Canonical class mapping mismatch at {manifest}:{row_number}: "
                    f"{source_class_id}/{row['class_name']!r}"
                )
            source_path = Path(row["source_path"]).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Scene source referenced at {manifest}:{row_number} is missing: {source_path}"
                )
            selected.append(
                BoxRow(
                    split=expected_split,
                    input_manifest=manifest,
                    input_row_number=row_number,
                    source_path=source_path,
                    source_id=row["source_id"],
                    group_id=row["group_id"],
                    source_class_id=source_class_id,
                    source_class_name=row["class_name"],
                    bbox=parse_bbox(row["bbox_xyxy"], manifest=manifest, row_number=row_number),
                )
            )
    if not selected:
        raise RuntimeError(f"No canonical scene boxes selected from {manifest}")
    return selected, ReadReport(
        manifest=manifest,
        manifest_sha256=sha256_file(manifest),
        expected_split=expected_split,
        expected_selection_role=expected_selection_role,
        total_rows=total_rows,
        selected_scene_rows=len(selected),
        ignored_non_scene_rows=ignored_non_scene,
    )


def validate_source_identity(rows: Iterable[BoxRow]) -> dict[Path, list[BoxRow]]:
    grouped: dict[Path, list[BoxRow]] = defaultdict(list)
    for row in rows:
        grouped[row.source_path].append(row)
    for source_path, members in grouped.items():
        identities = {
            (
                row.split,
                row.source_id,
                row.group_id,
                row.source_class_id,
                row.source_class_name,
            )
            for row in members
        }
        if len(identities) != 1:
            raise RuntimeError(
                f"Conflicting identity for source {source_path}: {sorted(identities)}"
            )
    return dict(grouped)


def cross_split_values(rows: Iterable[BoxRow], getter) -> dict[str, list[str]]:
    split_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_sets[str(getter(row))].add(row.split)
    return {
        value: sorted(splits)
        for value, splits in sorted(split_sets.items())
        if len(splits) > 1
    }


def safe_stem(source_id: str, source_sha256: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._-") or "scene"
    return f"{sanitized}_{source_sha256[:16]}"


def normalized_yolo_box(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    source_path: Path,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(
            f"Out-of-bounds bbox {bbox} for {source_path} with dimensions {width}x{height}"
        )
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_read_report(report: ReadReport) -> dict[str, object]:
    return {
        "manifest": str(report.manifest),
        "manifest_sha256": report.manifest_sha256,
        "expected_split": report.expected_split,
        "expected_selection_role": report.expected_selection_role,
        "total_rows": report.total_rows,
        "selected_scene_rows": report.selected_scene_rows,
        "ignored_non_scene_rows": report.ignored_non_scene_rows,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    class_mapping = (
        {0: "pepper"}
        if args.label_mode == "class_agnostic"
        else {index: name for index, name in enumerate(CLASS_NAMES)}
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {output}. Choose a new --output path."
        )
    if args.train_manifest.resolve() == args.val_manifest.resolve():
        raise RuntimeError("Train and validation manifests must be physically separate files")

    train_rows, train_report = read_scene_rows(
        args.train_manifest,
        expected_split="train",
        expected_selection_role="training",
    )
    val_rows, val_report = read_scene_rows(
        args.val_manifest,
        expected_split="val",
        expected_selection_role="model_selection",
    )
    all_rows = train_rows + val_rows
    grouped_sources = validate_source_identity(all_rows)

    source_hashes = {path: sha256_file(path) for path in grouped_sources}
    group_overlap = cross_split_values(all_rows, lambda row: row.group_id)
    source_id_overlap = cross_split_values(all_rows, lambda row: row.source_id)
    source_path_overlap = cross_split_values(all_rows, lambda row: row.source_path)
    source_hash_overlap = cross_split_values(all_rows, lambda row: source_hashes[row.source_path])
    leakage_audit = {
        "passed": not any(
            (group_overlap, source_id_overlap, source_path_overlap, source_hash_overlap)
        ),
        "group_id_cross_split": group_overlap,
        "source_id_cross_split": source_id_overlap,
        "source_path_cross_split": source_path_overlap,
        "exact_source_sha256_cross_split": source_hash_overlap,
        "strict_test_manifest_opened": False,
    }
    if not leakage_audit["passed"]:
        raise RuntimeError(f"Cross-split leakage audit failed: {leakage_audit}")

    output.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    image_manifest_rows: list[dict[str, object]] = []
    box_manifest_rows: list[dict[str, object]] = []
    try:
        for split in ("train", "val"):
            (building / "images" / split).mkdir(parents=True, exist_ok=True)
            (building / "labels" / split).mkdir(parents=True, exist_ok=True)

        relative_path_owners: dict[Path, Path] = {}
        for source_path, members in sorted(
            grouped_sources.items(), key=lambda item: (item[1][0].split, str(item[0]))
        ):
            first = members[0]
            # The scene annotations were created after EXIF orientation was applied
            # (see prepare_ssl_dataset.py).  IMREAD_COLOR follows that orientation;
            # IMREAD_UNCHANGED would expose raw sensor dimensions and invalidate the
            # bbox coordinates for portrait images.
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV could not read source image: {source_path}")
            height, width = image.shape[:2]
            suffix = source_path.suffix.lower() or ".jpg"
            stem = safe_stem(first.source_id, source_hashes[source_path])
            image_relative = Path("images") / first.split / f"{stem}{suffix}"
            label_relative = Path("labels") / first.split / f"{stem}.txt"
            prior = relative_path_owners.setdefault(image_relative, source_path)
            if prior != source_path:
                raise RuntimeError(
                    f"Output filename collision between {prior} and {source_path}: {image_relative}"
                )

            yolo_lines: list[str] = []
            for member in members:
                yolo_class_id = (
                    0 if args.label_mode == "class_agnostic" else member.source_class_id
                )
                normalized = normalized_yolo_box(
                    member.bbox,
                    width=width,
                    height=height,
                    source_path=source_path,
                )
                yolo_lines.append(
                    f"{yolo_class_id} "
                    + " ".join(f"{value:.10f}" for value in normalized)
                )
                box_manifest_rows.append(
                    {
                        "split": first.split,
                        "input_manifest": str(member.input_manifest),
                        "input_row_number": member.input_row_number,
                        "image_path": str(output / image_relative),
                        "label_path": str(output / label_relative),
                        "source_path": str(source_path),
                        "source_id": first.source_id,
                        "group_id": first.group_id,
                        "source_class_id": first.source_class_id,
                        "source_class_name": first.source_class_name,
                        "source_bbox_xyxy": json.dumps(member.bbox),
                        "yolo_class_id": yolo_class_id,
                        "yolo_xywh_normalized": json.dumps(normalized),
                    }
                )
            shutil.copy2(source_path, building / image_relative)
            (building / label_relative).write_text(
                "\n".join(yolo_lines) + "\n", encoding="utf-8"
            )
            image_manifest_rows.append(
                {
                    "split": first.split,
                    "image_path": str(output / image_relative),
                    "label_path": str(output / label_relative),
                    "source_path": str(source_path),
                    "source_id": first.source_id,
                    "group_id": first.group_id,
                    "source_class_id": first.source_class_id,
                    "source_class_name": first.source_class_name,
                    "source_sha256": source_hashes[source_path],
                    "image_width": width,
                    "image_height": height,
                    "box_count": len(members),
                }
            )

        if len(box_manifest_rows) != len(all_rows):
            raise AssertionError(
                f"Lost bbox rows: selected={len(all_rows)}, emitted={len(box_manifest_rows)}"
            )
        if sum(int(row["box_count"]) for row in image_manifest_rows) != len(all_rows):
            raise AssertionError("Image-level box counts do not sum to all selected rows")

        write_csv(building / "manifest.csv", IMAGE_MANIFEST_FIELDS, image_manifest_rows)
        write_csv(building / "box_manifest.csv", BOX_MANIFEST_FIELDS, box_manifest_rows)
        yaml_names = "".join(
            f"  {class_id}: {json.dumps(name, ensure_ascii=False)}\n"
            for class_id, name in class_mapping.items()
        )
        data_yaml = (
            f"path: {output}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"{yaml_names}"
        )
        (building / "data.yaml").write_text(data_yaml, encoding="utf-8")

        images_by_split = Counter(str(row["split"]) for row in image_manifest_rows)
        boxes_by_split = Counter(str(row["split"]) for row in box_manifest_rows)
        boxes_by_source_class = Counter(
            f"{row['split']}:{row['source_class_name']}" for row in box_manifest_rows
        )
        boxes_by_yolo_class = Counter(
            f"{row['split']}:{row['yolo_class_id']}" for row in box_manifest_rows
        )
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(output),
            "label_mode": args.label_mode,
            "class_mapping": {str(key): value for key, value in class_mapping.items()},
            "input_policy": {
                "manifests_opened": [str(train_report.manifest), str(val_report.manifest)],
                "strict_test_manifest_opened": False,
                "unified_v3_manifest_opened": False,
                "selection": (
                    "canonical rows only: origin=scene, view_type=canonical, "
                    "record_role=canonical_audited, non-empty source_path+bbox_xyxy"
                ),
                "source_image_policy": "copy each unique resolved source_path once per split",
                "bbox_policy": (
                    "emit every selected manifest bbox row as YOLO class 0"
                    if args.label_mode == "class_agnostic"
                    else "emit every selected manifest bbox row with canonical class_id 0..3"
                ),
            },
            "input_reports": {
                "train": as_read_report(train_report),
                "val": as_read_report(val_report),
            },
            "counts": {
                "images_total": len(image_manifest_rows),
                "boxes_total": len(box_manifest_rows),
                "images_by_split": dict(sorted(images_by_split.items())),
                "boxes_by_split": dict(sorted(boxes_by_split.items())),
                "boxes_by_split_and_source_class": dict(sorted(boxes_by_source_class.items())),
                "boxes_by_split_and_yolo_class": dict(sorted(boxes_by_yolo_class.items())),
                "unique_group_ids": len({row.group_id for row in all_rows}),
                "unique_source_ids": len({row.source_id for row in all_rows}),
                "unique_source_paths": len(grouped_sources),
                "unique_source_hashes": len(set(source_hashes.values())),
            },
            "leakage_audit": leakage_audit,
        }
        (building / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (building / "leakage_audit.json").write_text(
            json.dumps(leakage_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(building, output)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
