#!/usr/bin/env python3
"""Evaluate detector recall strategies on the sealed camera holdout only.

This diagnostic never creates labels and never reads classification data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BACKEND_ROOT = Path(__file__).resolve().parent / "qianjiao-ai-inspection" / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
from app.services import camera_candidate_gate as gate_api  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def service_geometry_filter(pairs, width: int, height: int, roi=None):
    kept = []
    for bbox, confidence in pairs:
        x1, y1, x2, y2 = bbox
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        elongation = max(box_width / box_height, box_height / box_width)
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        inside = roi is None or (
            roi["left"] <= center_x <= roi["left"] + roi["width"]
            and roi["top"] <= center_y <= roi["top"] + roi["height"]
        )
        if elongation >= 1.30 and inside:
            kept.append((bbox, confidence))
    return kept


def model_pairs(model, frame, *, confidence, device, legacy=False):
    kwargs = {
        "source": frame,
        "conf": confidence,
        "iou": 0.45,
        "device": device,
        "verbose": False,
    }
    if legacy:
        kwargs.update({"imgsz": 960, "agnostic_nms": False, "max_det": 120})
    else:
        kwargs.update({"agnostic_nms": True, "max_det": 36})
    result = model.predict(**kwargs)[0]
    if result.boxes is None:
        return []
    return [
        (tuple(int(round(value)) for value in box), float(confidence_value))
        for box, confidence_value in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
        )
    ]


def render_overlay(rows, output: Path):
    font = ImageFont.load_default()
    tiles = []
    colours = {"strict35": (245, 80, 80), "legacy": (40, 220, 90), "component": (40, 180, 240)}
    for row in rows:
        image = Image.open(row["source_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for kind in ("component", "strict35", "legacy"):
            for bbox, _ in row[f"{kind}_pairs"]:
                draw.rectangle(bbox, outline=colours[kind], width=3)
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (660, 405), "white")
        tile.paste(image, (10, 35))
        label = (
            f"{row['image_name']} | strict.35={row['strict35_count']} "
            f"strict.25={row['strict25_count']} legacy+gate={row['legacy_count']} "
            f"strict+CC={row['component_union35_count']}"
        )
        ImageDraw.Draw(tile).text((10, 10), label, fill=(0, 0, 0), font=font)
        tiles.append(tile)
    sheet = Image.new("RGB", (1320, ((len(tiles) + 1) // 2) * 405), (230, 230, 230))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 660, (index // 2) * 405))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict-model", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke-image", default="snapshot-20260828-224109-2d631e45.jpg")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        holdout = [row for row in csv.DictReader(handle) if row["split"] == "holdout"]
    if not holdout:
        raise SystemExit("No holdout rows")
    strict = YOLO(args.strict_model)
    legacy = YOLO(args.legacy_model)
    fixed_gate = gate_api.CameraGateConfig()
    # This ROI was examined after the holdout was opened, so it is a diagnostic
    # only and may not be presented as an unbiased benchmark.
    diagnostic_roi = {"left": 0.30, "top": 0.10, "width": 0.65, "height": 0.85}

    evaluated = []
    smoke_candidates = []
    for manifest_row in holdout:
        image_path = Path(manifest_row["source_path"])
        frame = cv2.imread(str(image_path))
        height, width = frame.shape[:2]
        strict35 = service_geometry_filter(
            model_pairs(strict, frame, confidence=0.35, device=args.device), width, height
        )
        strict25 = service_geometry_filter(
            model_pairs(strict, frame, confidence=0.25, device=args.device), width, height
        )
        raw_legacy = model_pairs(
            legacy, frame, confidence=fixed_gate.proposal_confidence, device=args.device, legacy=True
        )
        legacy_results = gate_api.gate_camera_proposals(frame, raw_legacy, fixed_gate)
        legacy_pairs = service_geometry_filter(
            [(item.bbox, item.confidence) for item in legacy_results if item.accepted],
            width,
            height,
        )
        components = gate_api.red_component_proposals(frame, diagnostic_roi)
        strict_component = service_geometry_filter(
            gate_api.merge_detector_and_component_proposals(strict35, components),
            width,
            height,
            diagnostic_roi,
        )
        row = {
            **manifest_row,
            "strict35_count": len(strict35),
            "strict25_count": len(strict25),
            "legacy_count": len(legacy_pairs),
            "component_count": len(components),
            "component_union35_count": len(strict_component),
            "strict35_pairs": strict35,
            "strict25_pairs": strict25,
            "legacy_pairs": legacy_pairs,
            "component_pairs": components,
        }
        evaluated.append(row)
        if image_path.name == args.smoke_image:
            for index, item in enumerate(
                sorted((item for item in legacy_results if item.accepted), key=lambda item: item.bbox[0]),
                1,
            ):
                smoke_candidates.append(
                    {
                        "candidate_index": index,
                        "bbox": list(item.bbox),
                        "confidence": item.confidence,
                        "red_core": item.red_core,
                        "area_ratio": item.area_ratio,
                        "aspect_ratio": item.aspect_ratio,
                        "visual_status": "pepper_on_known_10_pepper_smoke_frame",
                    }
                )

    overlay_path = output / "holdout_detector_comparison.jpg"
    render_overlay(evaluated, overlay_path)
    csv_fields = list(smoke_candidates[0].keys())
    with (output / "smoke_10_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(smoke_candidates)

    compact_rows = [
        {key: value for key, value in row.items() if not key.endswith("_pairs")}
        for row in evaluated
    ]
    smoke_row = next(row for row in compact_rows if row["image_name"] == args.smoke_image)
    report = {
        "schema_version": 1,
        "classification_data_accessed": False,
        "holdout_only": True,
        "model_hashes": {
            "strict": sha256(args.strict_model.resolve()),
            "legacy": sha256(args.legacy_model.resolve()),
        },
        "thresholds": {
            "strict_default": [0.35, 0.25],
            "legacy_fixed_gate": fixed_gate.__dict__,
            "service_min_elongation": 1.30,
        },
        "holdout_rows": compact_rows,
        "independent_content_groups": sorted({row["content_group_id"] for row in compact_rows}),
        "smoke_10_peppers": {
            "image": args.smoke_image,
            "known_visible_count": 10,
            "strict_0_35_count": smoke_row["strict35_count"],
            "strict_0_25_count": smoke_row["strict25_count"],
            "legacy_fixed_gate_count": smoke_row["legacy_count"],
            "legacy_fixed_gate_apparent_misses": 0,
            "legacy_fixed_gate_apparent_false_boxes": 0,
            "candidate_details": smoke_candidates,
            "metric_warning": "visual count/box audit, not coordinate-IoU AP",
        },
        "holdout_failure": {
            "image": "camera-20260828-224539-21e5c4b4.jpg",
            "legacy_fixed_gate_count": next(
                row["legacy_count"]
                for row in compact_rows
                if row["image_name"] == "camera-20260828-224539-21e5c4b4.jpg"
            ),
            "provisional_visual_interpretation": "2 pepper boxes plus 1 retained hand false box after service elongation; full gate before elongation had 4 pepper + 2 hand candidates",
            "warning": "not human-labelled; cannot support an unbiased precision claim",
        },
        "opencv_component_diagnostic": {
            "roi": diagnostic_roi,
            "posthoc_holdout_tuning": True,
            "decision": "do_not_enable_by_default",
            "reason": "did not recover all ten peppers and can form hand/desk components",
        },
        "deployment_decision": "strict remains default; both recall fallbacks are opt-in experiments",
        "overlay": str(overlay_path),
    }
    report_path = output / "holdout_evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "holdout_evaluation.receipt.json").write_text(
        json.dumps(
            {
                "report_sha256": sha256(report_path),
                "overlay_sha256": sha256(overlay_path),
                "smoke_csv_sha256": sha256(output / "smoke_10_candidates.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
