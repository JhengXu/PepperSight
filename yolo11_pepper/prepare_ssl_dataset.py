#!/usr/bin/env python3
"""Build a leakage-safe pepper dataset from transparent cutouts and scene photos.

Scene photographs are split into individual peppers with a conservative red-body
detector and Apple's Vision foreground matte.  All crops from one source photo
remain in the same split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


CLASS_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=Path("辣椒_优质"))
    parser.add_argument(
        "--transparent-root", type=Path, default=Path("辣椒单体_透明PNG/成品")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("yolo11_pepper/datasets/pepper_ssl_v3")
    )
    parser.add_argument("--vision-mask", type=Path, default=Path("vision_mask"))
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def detect_red_bodies(image: Image.Image, work_side: int = 1200) -> list[tuple[int, int, int, int]]:
    rgb = np.asarray(image)
    scale = min(1.0, work_side / max(rgb.shape[:2]))
    work = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    red = work[:, :, 0].astype(np.int16)
    green = work[:, :, 1].astype(np.int16)
    red_body = (
        ((red - green > 12) & (red > 45) & (green * 100 < (red + 1) * 88))
        | ((hsv[:, :, 0] < 12) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 30))
    )
    mask = red_body.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    boxes: list[tuple[int, int, int, int, float, float]] = []
    for (x, y, box_w, box_h, area), (cx, cy) in zip(stats[1:], centroids[1:]):
        if area < 120 or area > height * width * 0.03:
            continue
        if box_w < 8 or box_h < 8 or y < height * 0.03 or y + box_h > height * 0.99:
            continue
        density = area / max(box_w * box_h, 1)
        if density < 0.08:
            continue
        factor = 1.0 / scale
        boxes.append(
            (
                max(0, math.floor(x * factor)),
                max(0, math.floor(y * factor)),
                min(image.width, math.ceil((x + box_w) * factor)),
                min(image.height, math.ceil((y + box_h) * factor)),
                cx,
                cy,
            )
        )
    boxes.sort(key=lambda value: (round(value[5] / 100), value[4]))
    return [tuple(int(v) for v in item[:4]) for item in boxes]


def padded_box(
    box: tuple[int, int, int, int], size: tuple[int, int], ratio: float = 0.38
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad = max(18, round(max(x1 - x0, y1 - y0) * ratio))
    return max(0, x0 - pad), max(0, y0 - pad), min(size[0], x1 + pad), min(size[1], y1 + pad)


def trim_matte(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    alpha = rgba[:, :, 3]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha >= 12).astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError(f"Vision returned an empty matte: {path}")
    primary = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary_area = int(stats[primary, cv2.CC_STAT_AREA])
    keep = labels == primary
    # Retain small detached stem fragments near the main subject.
    px, py, pw, ph = stats[primary, :4]
    for component in range(1, count):
        if component == primary:
            continue
        x, y, w, h, area = stats[component]
        near = x < px + pw + 12 and x + w > px - 12 and y < py + ph + 12 and y + h > py - 12
        if near and area >= primary_area * 0.008:
            keep |= labels == component
    rgba[~keep, 3] = 0
    rgba[~keep, :3] = 0
    ys, xs = np.nonzero(rgba[:, :, 3] >= 4)
    if not len(xs):
        raise RuntimeError(f"No foreground remains after matte cleanup: {path}")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    object_w, object_h = int(x1 - x0), int(y1 - y0)
    margin = max(8, round(max(object_w, object_h) * 0.07))
    cropped = Image.fromarray(rgba, "RGBA").crop((x0, y0, x1, y1))
    canvas = Image.new("RGBA", (object_w + 2 * margin, object_h + 2 * margin), (0, 0, 0, 0))
    canvas.alpha_composite(cropped, (margin, margin))
    canvas.save(path, format="PNG", optimize=True)
    return object_w, object_h


def split_sources(
    transparent: dict[int, list[str]], scenes: dict[int, list[str]], seed: int
) -> dict[tuple[str, int, str], str]:
    rng = random.Random(seed)
    owners: dict[tuple[str, int, str], str] = {}
    for class_id in range(4):
        groups = sorted(transparent[class_id])
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * 0.15))
        n_val = max(1, round(len(groups) * 0.15))
        for index, source in enumerate(groups):
            split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            owners[("transparent", class_id, source)] = split

        scene_groups = sorted(scenes[class_id])
        rng.shuffle(scene_groups)
        # Every class contributes one entire new scene to the strict test set.
        # The only class with four scenes also contributes one to validation.
        for index, source in enumerate(scene_groups):
            if index == len(scene_groups) - 1:
                split = "test"
            elif len(scene_groups) >= 4 and index == len(scene_groups) - 2:
                split = "val"
            else:
                split = "train"
            owners[("scene", class_id, source)] = split
    return owners


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(source.read_bytes())


def contact_sheet(paths: list[Path], destination: Path) -> None:
    tile_w, tile_h, columns = 180, 230, 6
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
        background = Image.new("RGBA", (tile_w - 12, tile_h - 30), (245, 245, 245, 255))
        rgba.thumbnail((tile_w - 28, tile_h - 48), Image.Resampling.LANCZOS)
        background.alpha_composite(rgba, ((background.width - rgba.width) // 2, (background.height - rgba.height) // 2))
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        sheet.paste(background.convert("RGB"), (x + 6, y + 24))
        draw.text((x + 8, y + 6), f"{index + 1:03d}", fill=(20, 20, 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    transparent_root = args.transparent_root.resolve()
    output = args.output.resolve()
    work = output / "_segmentation_work"
    final_scene_root = output / "segmented_scenes"
    output.mkdir(parents=True, exist_ok=args.resume)

    records: list[dict[str, object]] = []
    scene_sources: dict[int, list[str]] = defaultdict(list)
    jobs: list[tuple[Path, Path]] = []
    scene_records: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = scene_root / class_name
        for source_path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            image = oriented_rgb(source_path)
            boxes = detect_red_bodies(image)
            if not boxes:
                raise RuntimeError(f"No peppers detected: {source_path}")
            source_id = source_path.stem
            scene_sources[class_id].append(source_id)
            overlay = image.copy()
            overlay_draw = ImageDraw.Draw(overlay)
            cutout_paths: list[Path] = []
            for index, box in enumerate(boxes, 1):
                crop_box = padded_box(box, image.size)
                filename = f"{source_id}__pepper_{index:03d}.png"
                crop_path = work / "crops" / class_name / source_id / filename
                mask_path = final_scene_root / class_name / source_id / filename
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop(crop_box).save(crop_path, "PNG")
                jobs.append((crop_path, mask_path))
                cutout_paths.append(mask_path)
                overlay_draw.rectangle(box, outline=(0, 255, 0), width=max(3, image.width // 1000))
                overlay_draw.text((box[0], max(0, box[1] - 24)), str(index), fill=(0, 100, 255))
                scene_records.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_id": source_id,
                        "source_path": str(source_path.resolve()),
                        "path": str(mask_path.resolve()),
                        "bbox_xyxy": json.dumps(box),
                        "crop_xyxy": json.dumps(crop_box),
                    }
                )
            overlay.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            overlay_path = output / "qc" / "detections" / f"{class_name}__{source_id}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(overlay_path, quality=90)
            print(f"detected {class_name}/{source_path.name}: {len(boxes)}")

    jobs_path = work / "vision_jobs.tsv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    missing_jobs = [(source, target) for source, target in jobs if not target.exists()]
    jobs_path.write_text(
        "".join(f"{source}\t{target}\n" for source, target in missing_jobs), encoding="utf-8"
    )
    if missing_jobs:
        subprocess.run([str(args.vision_mask.resolve()), "--manifest", str(jobs_path)], check=True)
    for record in scene_records:
        object_size = trim_matte(Path(str(record["path"])))
        record["object_size"] = json.dumps(object_size)
    for class_name in CLASS_NAMES:
        for source_dir in sorted((final_scene_root / class_name).iterdir()):
            contact_sheet(sorted(source_dir.glob("*.png")), output / "qc" / "cutouts" / f"{class_name}__{source_dir.name}.jpg")

    transparent_sources: dict[int, list[str]] = defaultdict(list)
    transparent_records: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for source_dir in sorted(p for p in (transparent_root / class_name).iterdir() if p.is_dir()):
            transparent_sources[class_id].append(source_dir.name)
            for path in sorted(source_dir.rglob("*.png")):
                transparent_records.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_id": source_dir.name,
                        "source_path": str(path.resolve()),
                        "path": str(path.resolve()),
                        "bbox_xyxy": "",
                        "crop_xyxy": "",
                        "object_size": "",
                    }
                )

    owners = split_sources(transparent_sources, scene_sources, args.seed)
    final_rows: list[dict[str, object]] = []
    for origin, source_records in (("transparent", transparent_records), ("scene", scene_records)):
        for record in source_records:
            class_id = int(record["class_id"])
            source_id = str(record["source_id"])
            split = owners[(origin, class_id, source_id)]
            source = Path(str(record["path"]))
            destination = output / "images" / split / str(record["class_name"]) / f"{origin}_{source_id}" / source.name
            hardlink_or_copy(source, destination)
            final_rows.append(
                {
                    **record,
                    "path": str(destination.resolve()),
                    "origin": origin,
                    "split": split,
                    "group_id": f"{origin}:{record['class_name']}:{source_id}",
                    "original_class_id": class_id,
                    "label_state": "human",
                    "label_weight": 1.0,
                }
            )

    fields = [
        "path", "source_path", "origin", "split", "group_id", "source_id",
        "class_id", "class_name", "original_class_id", "label_state", "label_weight",
        "bbox_xyxy", "crop_xyxy", "object_size",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)
    summary: dict[str, object] = {"total": len(final_rows), "segmented_scene_total": len(scene_records), "splits": {}}
    for split in ("train", "val", "test"):
        split_rows = [row for row in final_rows if row["split"] == split]
        summary["splits"][split] = {
            "total": len(split_rows),
            "by_class": {name: sum(row["class_name"] == name for row in split_rows) for name in CLASS_NAMES},
            "scene": sum(row["origin"] == "scene" for row in split_rows),
            "transparent": sum(row["origin"] == "transparent" for row in split_rows),
            "groups": len({row["group_id"] for row in split_rows}),
        }
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
