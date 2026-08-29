#!/usr/bin/env python3
"""Detect one rough bounding box per pepper in the lajiao photo set.

This stage intentionally uses only conservative red-body pixels.  The boxes are
later expanded and passed to a separate foreground-matting stage, so green stems
and pale highlights do not need to be part of the detection mask.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


@dataclass
class Component:
    area: int
    bbox: tuple[int, int, int, int]
    cx: float
    cy: float
    wood_fraction: float = 0.0
    density: float = 0.0
    score: float = 0.0


class DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def make(self) -> int:
        label = len(self.parent)
        self.parent.append(label)
        return label

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


COUNT_OVERRIDES = {
    "77a86cd0fa361a91bcd24e76ba7cd18f": 20,
    "5d00e384853dde5c9fb89cef6543dd03": 25,
    "cb45b866ba653632a88bd98b13bc196f": 11,
    "5d0ec913260d2044ca22aca4f4fbb077": 12,
}


def verified_count(path: Path) -> int:
    if path.stem in COUNT_OVERRIDES:
        return COUNT_OVERRIDES[path.stem]
    return 20 if path.parent.name == "子弹头_差" else 15


def load_oriented(path: Path) -> tuple[Image.Image, str]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
    applied = "EXIF transpose"
    # This one source has no EXIF orientation even though its stored pixels are
    # sideways relative to every other tabletop image in the set.
    if path.stem == "2b49ff4440bcd8a70b72064ba522bfb8":
        oriented = oriented.transpose(Image.Transpose.ROTATE_270)
        applied += "+manual CW90"
    return oriented, applied


def label_runs(mask: np.ndarray) -> list[Component]:
    """8-connected component labeling using row runs (fast without scipy)."""
    height, width = mask.shape
    dsu = DisjointSet()
    runs: list[tuple[int, int, int, int]] = []  # y, x0, x1 inclusive, label
    previous: list[tuple[int, int, int]] = []  # x0, x1, label

    for y in range(height):
        row = mask[y]
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        current: list[tuple[int, int, int]] = []
        p = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            while p < len(previous) and previous[p][1] < x0 - 1:
                p += 1
            overlapping: list[int] = []
            q = p
            while q < len(previous) and previous[q][0] <= x1 + 1:
                overlapping.append(previous[q][2])
                q += 1
            label = dsu.make()
            for other in overlapping:
                dsu.union(label, other)
            current.append((x0, x1, label))
            runs.append((y, x0, x1, label))
        previous = current

    stats: dict[int, list[float]] = {}
    for y, x0, x1, label in runs:
        root = dsu.find(label)
        length = x1 - x0 + 1
        if root not in stats:
            # area, minx, miny, maxx, maxy, xsum, ysum
            stats[root] = [0, x0, y, x1, y, 0, 0]
        item = stats[root]
        item[0] += length
        item[1] = min(item[1], x0)
        item[2] = min(item[2], y)
        item[3] = max(item[3], x1)
        item[4] = max(item[4], y)
        item[5] += (x0 + x1) * length / 2.0
        item[6] += y * length

    components: list[Component] = []
    for item in stats.values():
        area = int(item[0])
        components.append(
            Component(
                area=area,
                bbox=(int(item[1]), int(item[2]), int(item[3]) + 1, int(item[4]) + 1),
                cx=float(item[5]) / area,
                cy=float(item[6]) / area,
            )
        )
    return components


def reading_order(items: list[Component]) -> list[Component]:
    if not items:
        return []
    median_height = float(np.median([item.bbox[3] - item.bbox[1] for item in items]))
    row_tolerance = max(12.0, median_height * 0.75)
    rows: list[list[Component]] = []
    for item in sorted(items, key=lambda value: value.cy):
        if not rows or abs(item.cy - float(np.median([member.cy for member in rows[-1]]))) > row_tolerance:
            rows.append([item])
        else:
            rows[-1].append(item)
    ordered: list[Component] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda value: value.cx))
    return ordered


def detect(path: Path, work_max_side: int = 1200) -> tuple[Image.Image, list[Component], dict]:
    oriented, orientation_applied = load_oriented(path)
    original_width, original_height = oriented.size
    scale = min(1.0, work_max_side / max(original_width, original_height))
    work_size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
    work = oriented.resize(work_size, Image.Resampling.LANCZOS)
    rgb = np.asarray(work, dtype=np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # Strong pepper-body seed.  Wood is warm too, but its G/R ratio is much
    # higher; keeping this seed conservative avoids following wood grain.
    mask = (
        (red > 38)
        & ((red - green) > 17)
        & (green * 100 < red * 74)
        & (blue * 100 < red * 112)
    )
    mask_image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    # Join highlight-separated red areas without noticeably merging neighbors.
    mask_image = mask_image.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    closed = np.asarray(mask_image) >= 128

    work_height, work_width = closed.shape
    total = work_width * work_height
    min_area = max(24, int(total * 0.000025))
    max_area = int(total * 0.035)
    raw = label_runs(closed)
    plausible: list[Component] = []
    for item in raw:
        x0, y0, x1, y1 = item.bbox
        width = x1 - x0
        height = y1 - y0
        if not (min_area <= item.area <= max_area and width >= 5 and height >= 5):
            continue
        if item.cx < work_width * 0.02 or item.cx > work_width * 0.98:
            continue
        # All peppers sit below the tabletop's upper edge.  The only strong red
        # candidates above this line are chair backs in two high-resolution
        # frames; excluding them avoids treating furniture as produce.
        if item.cy < work_height * 0.10 or item.cy > work_height * 0.98:
            continue
        aspect = max(width / max(height, 1), height / max(width, 1))
        if aspect > 9.5:
            continue

        pad = max(8, round(max(width, height) * 0.55))
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(work_width, x1 + pad), min(work_height, y1 + pad)
        region = rgb[ry0:ry1, rx0:rx1]
        rr = region[:, :, 0]
        rg = region[:, :, 1]
        rb = region[:, :, 2]
        # Pale/warm surrounding pixels are a strong cue that the component is
        # on the tabletop rather than a red chair or the bag below the table.
        wood = (rr > 70) & (rg > 55) & ((rr - rb) > 10) & ((rg - rb) > -2)
        item.wood_fraction = float(wood.mean())
        item.density = item.area / max(1, width * height)
        item.score = (
            math.log1p(item.area)
            + 3.0 * item.wood_fraction
            + 0.4 * min(item.density, 0.8)
            - 0.15 * max(0.0, aspect - 5.0)
        )
        if item.wood_fraction >= 0.42:
            plausible.append(item)

    # Counts were visually verified after inspecting every anomalous candidate
    # frame.  Most are 15 or 20; four explicit exceptions contain 11, 12, 20
    # and 25 respectively.  Ranking removes only chair/pad/debris candidates.
    areas = sorted((item.area for item in plausible), reverse=True)
    reference_areas = areas[:25]
    median_area = float(np.median(reference_areas)) if reference_areas else 0.0
    fragment_floor = max(min_area, median_area * 0.10)
    target = verified_count(path)
    selected = sorted(plausible, key=lambda item: item.score, reverse=True)[:target]
    selected = reading_order(selected)

    factor_x = original_width / work_width
    factor_y = original_height / work_height
    scaled: list[Component] = []
    for item in selected:
        x0, y0, x1, y1 = item.bbox
        scaled.append(
            Component(
                area=round(item.area * factor_x * factor_y),
                bbox=(
                    max(0, math.floor(x0 * factor_x)),
                    max(0, math.floor(y0 * factor_y)),
                    min(original_width, math.ceil(x1 * factor_x)),
                    min(original_height, math.ceil(y1 * factor_y)),
                ),
                cx=item.cx * factor_x,
                cy=item.cy * factor_y,
                wood_fraction=item.wood_fraction,
                density=item.density,
                score=item.score,
            )
        )

    # One pepper overlaps a bright-red tabletop pad in the source.  Their red
    # seeds connect, so the coarse component includes the pad.  The coordinates
    # below are the visually verified visible pepper extent in the normalized
    # (EXIF-transposed) source; the matting stage applies an additional guard.
    if path.stem == "2b55121f261ea04965cf3f9ca0f8b0fb":
        for item in scaled:
            x0, y0, x1, y1 = item.bbox
            if x0 == 0 and y1 - y0 > 1000:
                item.bbox = (170, 2650, 470, 3310)
                item.cx = 320.0
                item.cy = 2980.0
        scaled = reading_order(scaled)
    if path.stem == "dcb35000f647f77db0db77e6c00b2309":
        for item in scaled:
            x0, y0, x1, y1 = item.bbox
            if x0 == 0 and y1 - y0 > 1000:
                item.bbox = (185, 2820, 440, 3335)
                item.cx = 312.0
                item.cy = 3075.0
        scaled = reading_order(scaled)

    metadata = {
        "source": str(path),
        "group": path.parent.name,
        "source_size": [original_width, original_height],
        "work_size": [work_width, work_height],
        "orientation_applied": orientation_applied,
        "verified_count": target,
        "plausible_candidates": len(plausible),
        "selected": len(scaled),
        "count_status": "ok" if len(scaled) == target else "candidate_shortfall",
        "fragment_floor_work_pixels": round(fragment_floor, 2),
        "candidate_area_ratios": [
            round(item.area / median_area, 4) if median_area else 0.0
            for item in sorted(plausible, key=lambda value: value.area, reverse=True)
        ],
    }
    return oriented, scaled, metadata


def diagnostic_image(image: Image.Image, items: list[Component]) -> Image.Image:
    preview = image.copy()
    max_side = 1800
    scale = min(1.0, max_side / max(preview.size))
    if scale < 1.0:
        preview = preview.resize((round(preview.width * scale), round(preview.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    line_width = max(2, round(max(preview.size) / 700))
    for index, item in enumerate(items, start=1):
        x0, y0, x1, y1 = item.bbox
        box = tuple(round(value * scale) for value in (x0, y0, x1, y1))
        draw.rectangle(box, outline=(0, 255, 80), width=line_width)
        label_x = box[0]
        label_y = max(0, box[1] - 18)
        draw.rectangle((label_x, label_y, label_x + 36, label_y + 18), fill=(0, 0, 0))
        draw.text((label_x + 3, label_y + 2), f"{index:02d}", fill=(255, 255, 255))
    return preview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--only", action="append", default=[], help="Only process source stems containing this text")
    args = parser.parse_args()

    images = sorted(path for path in args.input_root.glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.only:
        images = [path for path in images if any(token in path.stem for token in args.only)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for path in images:
        image, items, metadata = detect(path)
        group_dir = args.output_root / path.parent.name
        group_dir.mkdir(parents=True, exist_ok=True)
        preview = diagnostic_image(image, items)
        preview.save(group_dir / f"{path.stem}_检测.jpg", quality=90)
        metadata["detections"] = [
            {
                "index": index,
                "bbox": list(item.bbox),
                "center": [round(item.cx, 1), round(item.cy, 1)],
                "seed_area": item.area,
                "wood_fraction": round(item.wood_fraction, 4),
                "score": round(item.score, 4),
            }
            for index, item in enumerate(items, start=1)
        ]
        manifest.append(metadata)
        print(
            f"{path.parent.name}/{path.name}: {len(items)}/{metadata['verified_count']} "
            f"(candidates={metadata['plausible_candidates']}, status={metadata['count_status']})"
        )

    (args.output_root / "detections.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
