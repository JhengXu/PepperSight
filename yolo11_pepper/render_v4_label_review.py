#!/usr/bin/env python3
"""Render high-priority train-only label-audit samples for human adjudication."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_audit/train_label_audit.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/runs/hierarchical_v4/label_review"),
    )
    parser.add_argument("--priority", default="1")
    parser.add_argument("--per-sheet", type=int, default=15)
    return parser.parse_args()


def font(size: int):
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def fit_rgba(path: Path, side: int) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    ratio = min((side - 20) / max(rgba.width, 1), (side - 20) / max(rgba.height, 1))
    rgba = rgba.resize(
        (max(1, round(rgba.width * ratio)), max(1, round(rgba.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("RGBA", (side, side), (68, 72, 72, 255))
    tile.alpha_composite(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
    return tile.convert("RGB")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("review_priority") == str(args.priority)
        ]
    rows.sort(key=lambda row: (row["class_id"], float(row["oof_p_original_grade"])))
    # Keep the sheet number reversible: the image caption is intentionally
    # compact, while this index preserves the full path and model suggestion for
    # expert adjudication without changing any training label automatically.
    index_path = args.output / f"priority_{args.priority}_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = (
            "review_number",
            "path",
            "human_class_id",
            "human_class_name",
            "suggested_class_id",
            "suggested_grade",
            "oof_p_good",
            "oof_p_bad",
            "audit_status",
            "human_decision_class_id",
            "reviewer_note",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            species_id = int(row["class_id"]) // 2
            suggested_grade_id = int(float(row["oof_p_bad"]) >= 0.5)
            writer.writerow(
                {
                    "review_number": number,
                    "path": str(Path(row["path"]).resolve()),
                    "human_class_id": row["class_id"],
                    "human_class_name": row["class_name"],
                    "suggested_class_id": species_id * 2 + suggested_grade_id,
                    "suggested_grade": "二级" if suggested_grade_id else "一级",
                    "oof_p_good": row["oof_p_good"],
                    "oof_p_bad": row["oof_p_bad"],
                    "audit_status": row["audit_status"],
                    "human_decision_class_id": "",
                    "reviewer_note": "",
                }
            )
    print(index_path)
    columns, image_side, caption_height = 5, 250, 86
    for sheet_index, start in enumerate(range(0, len(rows), args.per_sheet), 1):
        page_rows = rows[start : start + args.per_sheet]
        grid_rows = math.ceil(len(page_rows) / columns)
        sheet = Image.new(
            "RGB",
            (columns * image_side, grid_rows * (image_side + caption_height)),
            (245, 245, 243),
        )
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(page_rows):
            x = (local % columns) * image_side
            y = (local // columns) * (image_side + caption_height)
            sheet.paste(fit_rgba(Path(row["path"]), image_side), (x, y))
            name = Path(row["path"]).stem
            caption = (
                f"#{start + local + 1} {row['class_name']}\n"
                f"p一级={float(row['oof_p_good']):.2f} "
                f"p二级={float(row['oof_p_bad']):.2f}\n"
                f"{name[-24:]}"
            )
            draw.multiline_text(
                (x + 6, y + image_side + 4),
                caption,
                fill=(25, 25, 25),
                font=font(15),
                spacing=2,
            )
        destination = args.output / f"priority_{args.priority}_{sheet_index:02d}.jpg"
        sheet.save(destination, quality=92)
        print(destination)


if __name__ == "__main__":
    main()
