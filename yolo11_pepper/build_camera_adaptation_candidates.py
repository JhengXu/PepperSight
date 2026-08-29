#!/usr/bin/env python3
"""Build an auditable camera-domain detector candidate set.

This tool intentionally does *not* turn pseudo detections into YOLO labels.  It
uses the legacy detector only as a proposal generator, applies fixed colour and
geometry gates, assigns whole capture sessions to train-pool/holdout, and emits
a review sheet.  Only rows explicitly changed to ``human_accept`` are eligible
for a later training export.

No classification manifests or feature caches are read by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


TIMESTAMP_RE = re.compile(r"(?:camera|snapshot)-(\d{8})-(\d{6})-")


@dataclass(frozen=True)
class GateConfig:
    proposal_conf: float = 0.05
    min_conf: float = 0.15
    red_hue_low: int = 15
    red_hue_high: int = 170
    red_saturation_min: int = 90
    red_value_min: int = 45
    red_core_min: float = 0.30
    min_area_ratio: float = 0.0015
    max_area_ratio: float = 0.12
    max_aspect_ratio: float = 5.0
    min_side_px: int = 24
    class_agnostic_nms_iou: float = 0.45
    session_gap_seconds: int = 120
    holdout_latest_sessions: int = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> str:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def parse_timestamp(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse capture time from {path.name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def discover_raw_images(upload_dir: Path) -> list[Path]:
    files = []
    for path in upload_dir.glob("*.jpg"):
        if path.name.startswith("camera-") or (
            path.name.startswith("snapshot-") and not path.name.startswith("snapshot-result-")
        ):
            files.append(path.resolve())
    return sorted(files, key=lambda path: (parse_timestamp(path), path.name))


def assign_sessions(paths: list[Path], gap_seconds: int) -> dict[Path, str]:
    sessions: dict[Path, str] = {}
    previous: datetime | None = None
    session_index = 0
    for path in paths:
        timestamp = parse_timestamp(path)
        if previous is None or (timestamp - previous).total_seconds() > gap_seconds:
            session_index += 1
        sessions[path] = f"camera_session_{session_index:02d}"
        previous = timestamp
    return sessions


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def class_agnostic_nms(rows: list[dict], threshold: float) -> set[int]:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["confidence"], reverse=True)
    keep: list[int] = []
    for index in order:
        box = np.asarray([rows[index][key] for key in ("x1", "y1", "x2", "y2")])
        if all(
            iou(
                box,
                np.asarray([rows[other][key] for key in ("x1", "y1", "x2", "y2")]),
            )
            <= threshold
            for other in keep
        ):
            keep.append(index)
    return set(keep)


def proposal_rows(image_path: Path, result, config: GateConfig) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rows: list[dict] = []
    if result.boxes is None:
        return rows
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    for proposal_index, (box, confidence, source_class) in enumerate(zip(boxes, confidences, classes)):
        x1, y1, x2, y2 = box.round().astype(int)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
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
        box_width, box_height = x2 - x1, y2 - y1
        area_ratio = (box_width * box_height) / float(width * height)
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        reasons = []
        if confidence < config.min_conf:
            reasons.append("low_confidence")
        if red_core < config.red_core_min:
            reasons.append("low_red_core")
        if area_ratio < config.min_area_ratio:
            reasons.append("too_small_area")
        if area_ratio > config.max_area_ratio:
            reasons.append("too_large_area")
        if aspect > config.max_aspect_ratio:
            reasons.append("extreme_aspect")
        if min(box_width, box_height) < config.min_side_px:
            reasons.append("short_side")
        rows.append(
            {
                "proposal_index": proposal_index,
                "source_class": int(source_class),
                "confidence": float(confidence),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "red_core": red_core,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect,
                "gate_reasons": ";".join(reasons),
                "passed_scalar_gates": not reasons,
            }
        )
    passed = [row for row in rows if row["passed_scalar_gates"]]
    kept_local = class_agnostic_nms(passed, config.class_agnostic_nms_iou)
    kept_ids = {id(passed[index]) for index in kept_local}
    for row in rows:
        row["passed_gate"] = bool(row["passed_scalar_gates"] and id(row) in kept_ids)
        if row["passed_scalar_gates"] and not row["passed_gate"]:
            row["gate_reasons"] = "class_agnostic_nms_duplicate"
    return rows


def copy_raw_images(paths: Iterable[Path], output_dir: Path) -> None:
    raw_dir = output_dir / "images" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        destination = raw_dir / path.name
        if not destination.exists() or sha256(destination) != sha256(path):
            shutil.copy2(path, destination)


def render_review_pages(image_rows: list[dict], candidates: list[dict], output_dir: Path) -> list[str]:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[dict]] = {}
    for row in candidates:
        if row["passed_gate"]:
            by_image.setdefault(row["image_name"], []).append(row)
    thumbs = []
    font = ImageFont.load_default()
    for image_row in image_rows:
        name = image_row["image_name"]
        if name not in by_image:
            continue
        image = Image.open(image_row["source_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in by_image[name]:
            box = (row["x1"], row["y1"], row["x2"], row["y2"])
            draw.rectangle(box, outline=(40, 230, 90), width=4)
            label = f"{row['candidate_id']} c={row['confidence']:.2f} red={row['red_core']:.2f}"
            draw.rectangle((box[0], max(0, box[1] - 18), min(image.width, box[0] + 245), box[1]), fill=(0, 0, 0))
            draw.text((box[0] + 2, max(0, box[1] - 16)), label, fill=(255, 255, 255), font=font)
        image.thumbnail((480, 270), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (500, 310), "white")
        tile.paste(image, ((500 - image.width) // 2, 24))
        tile_draw = ImageDraw.Draw(tile)
        title = f"{image_row['session_id']} | {image_row['split']} | {name} | n={len(by_image[name])}"
        tile_draw.text((8, 6), title, fill=(0, 0, 0), font=font)
        thumbs.append(tile)
    pages = []
    per_page = 12
    for page_index in range(math.ceil(len(thumbs) / per_page)):
        subset = thumbs[page_index * per_page : (page_index + 1) * per_page]
        page = Image.new("RGB", (1500, 1240), (230, 230, 230))
        for index, tile in enumerate(subset):
            page.paste(tile, ((index % 3) * 500, (index // 3) * 310))
        name = f"candidate_review_{page_index + 1:02d}.jpg"
        page.save(review_dir / name, quality=90)
        pages.append(str((review_dir / name).resolve()))
    return pages


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uploads", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--strict-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-image", default="snapshot-20260828-224109-2d631e45.jpg")
    parser.add_argument("--smoke-ground-truth-count", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    config = GateConfig()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_raw_images(args.uploads.resolve())
    if not paths:
        raise SystemExit("No raw camera images found")
    sessions = assign_sessions(paths, config.session_gap_seconds)
    content_hashes = {path: dhash(path) for path in paths}
    ordered_sessions = list(dict.fromkeys(sessions[path] for path in paths))
    holdout_sessions = set(ordered_sessions[-config.holdout_latest_sessions :])
    smoke_path = next((path for path in paths if path.name == args.smoke_image), None)
    if smoke_path is None:
        raise SystemExit(f"Smoke image not found: {args.smoke_image}")
    holdout_sessions.add(sessions[smoke_path])

    copy_raw_images(paths, output_dir)
    legacy = YOLO(str(args.legacy_model.resolve()))
    strict = YOLO(str(args.strict_model.resolve()))
    sources = [str(path) for path in paths]
    legacy_results = legacy.predict(
        source=sources,
        conf=config.proposal_conf,
        iou=0.5,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        stream=False,
    )
    strict_results = strict.predict(
        source=sources,
        conf=0.25,
        iou=0.5,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        stream=False,
    )

    image_rows: list[dict] = []
    candidate_rows: list[dict] = []
    smoke_metrics = {}
    for image_index, (path, legacy_result, strict_result) in enumerate(zip(paths, legacy_results, strict_results)):
        session_id = sessions[path]
        split = "holdout" if session_id in holdout_sessions else "candidate_train_pool"
        rows = proposal_rows(path, legacy_result, config)
        accepted_count = sum(row["passed_gate"] for row in rows)
        strict_count = 0 if strict_result.boxes is None else len(strict_result.boxes)
        image_row = {
            "image_id": f"camera_{image_index:04d}",
            "image_name": path.name,
            "source_path": str(path),
            "copied_path": str((output_dir / "images" / "raw" / path.name).resolve()),
            "capture_time": parse_timestamp(path).isoformat(),
            "session_id": session_id,
            "content_group_id": f"dhash_{content_hashes[path]}",
            "split": split,
            "sha256": sha256(path),
            "dhash64": content_hashes[path],
            "legacy_proposals": len(rows),
            "legacy_gate_candidates": accepted_count,
            "strict_predictions_conf_0_25": strict_count,
        }
        image_rows.append(image_row)
        accepted_index = 0
        for row in rows:
            if row["passed_gate"]:
                accepted_index += 1
                candidate_id = f"{image_row['image_id']}_candidate_{accepted_index:02d}"
            else:
                candidate_id = ""
            candidate_rows.append(
                {
                    **image_row,
                    **row,
                    "candidate_id": candidate_id,
                    "review_status": "pending_human_review" if row["passed_gate"] else "rule_rejected",
                    "human_label": "",
                    "review_notes": "",
                    "eligible_for_training": False,
                }
            )
        if path == smoke_path:
            smoke_metrics = {
                "image": path.name,
                "session_id": session_id,
                "split": split,
                "human_visible_pepper_count": args.smoke_ground_truth_count,
                "strict_prediction_count_conf_0_25": strict_count,
                "legacy_gate_candidate_count": accepted_count,
                "strict_count_recall_proxy": min(strict_count, args.smoke_ground_truth_count)
                / args.smoke_ground_truth_count,
                "legacy_gate_count_recall_proxy": min(accepted_count, args.smoke_ground_truth_count)
                / args.smoke_ground_truth_count,
                "warning": "Count recall is a smoke-test proxy, not IoU-matched AP; candidate identities require review.",
            }

    image_fields = list(image_rows[0].keys())
    candidate_fields = list(candidate_rows[0].keys())
    write_csv(output_dir / "image_manifest.csv", image_rows, image_fields)
    write_csv(output_dir / "candidate_manifest.csv", candidate_rows, candidate_fields)
    review_rows = [row for row in candidate_rows if row["passed_gate"]]
    write_csv(output_dir / "human_review.csv", review_rows, candidate_fields)
    review_pages = render_review_pages(image_rows, candidate_rows, output_dir)

    train_images = [row for row in image_rows if row["split"] == "candidate_train_pool"]
    holdout_images = [row for row in image_rows if row["split"] == "holdout"]
    content_splits: dict[str, set[str]] = {}
    content_counts: dict[str, int] = {}
    for row in image_rows:
        content_splits.setdefault(row["content_group_id"], set()).add(row["split"])
        content_counts[row["content_group_id"]] = content_counts.get(row["content_group_id"], 0) + 1
    content_split_leaks = sorted(group for group, splits in content_splits.items() if len(splits) > 1)
    report = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "camera-domain class-agnostic pepper detector adaptation candidate audit",
        "safety_status": "review_required_no_training_labels_exported",
        "classification_data_accessed": False,
        "models": {
            "legacy_proposal_model": {
                "path": str(args.legacy_model.resolve()),
                "sha256": sha256(args.legacy_model.resolve()),
            },
            "strict_comparator": {
                "path": str(args.strict_model.resolve()),
                "sha256": sha256(args.strict_model.resolve()),
            },
        },
        "gate_config": asdict(config),
        "grouping": {
            "method": "capture-time sessions; new session when adjacent raw captures are over 120 seconds apart",
            "content_identity": "SHA-256 exact identity plus 64-bit dHash recorded per image",
            "content_group_method": "exact dHash64 equality; IDs are written to every manifest row",
            "content_group_count": len(content_counts),
            "duplicate_content_groups": {
                group: count for group, count in content_counts.items() if count > 1
            },
            "content_group_split_leaks": content_split_leaks,
            "session_count": len(ordered_sessions),
            "sessions": ordered_sessions,
            "holdout_sessions": sorted(holdout_sessions),
            "split_rule": "latest two sessions plus the smoke-image session are immutable holdout",
        },
        "counts": {
            "raw_images": len(image_rows),
            "candidate_train_pool_images": len(train_images),
            "holdout_images": len(holdout_images),
            "all_legacy_proposals": len(candidate_rows),
            "gate_candidates_pending_review": len(review_rows),
            "training_eligible_boxes": 0,
        },
        "split_audit_passed": not content_split_leaks,
        "smoke_test": smoke_metrics,
        "review_pages": review_pages,
        "next_step": (
            "A human must mark each row human_accept/human_reject and supply corrected boxes. "
            "Only then may an exporter create YOLO labels from non-holdout sessions."
        ),
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    receipt = {
        path.name: sha256(path)
        for path in (
            output_dir / "image_manifest.csv",
            output_dir / "candidate_manifest.csv",
            output_dir / "human_review.csv",
            output_dir / "audit_report.json",
        )
    }
    (output_dir / "receipt.sha256.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
