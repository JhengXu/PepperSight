#!/usr/bin/env python3
"""Extract leakage-safe, scale-normalized multi-view YOLO11 features.

This utility deliberately does not train or load any pepper classification head.
It may therefore be used before label auditing without carrying predictions from
an older (and potentially leaked) classifier into the v4 experiment.

Important defaults:
* only ``train,val`` are extracted; ``test`` must be requested explicitly after
  model selection;
* transparent peppers are tightly alpha-cropped and *always* resized, including
  upscaling small inputs, so source resolution cannot trivially encode grade;
* features are pooled from several backbone stages to retain both local defect
  cues and global shape cues.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    checkpoint: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/runs/hierarchical_v4/features"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-views", type=int, default=9)
    parser.add_argument("--eval-views", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=(
            "generic_det",
            "pepper_det",
            "strict_det",
            "strict_four_det",
            "imagenet_cls",
        ),
        default=("generic_det", "pepper_det", "imagenet_cls"),
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def alpha_crop(image: Image.Image) -> tuple[Image.Image, dict[str, int | bool]]:
    """Crop only genuinely transparent images; retain RGB detector context."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    transparent = bool(alpha.min() < 250)
    if transparent:
        ys, xs = np.nonzero(alpha >= 4)
        if len(xs):
            rgba = rgba.crop(
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            )
    return rgba, {
        "source_width": int(image.width),
        "source_height": int(image.height),
        "object_width": int(rgba.width),
        "object_height": int(rgba.height),
        "has_transparency": transparent,
    }


def resize_to_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    ratio = long_edge / max(image.width, image.height, 1)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    # Unlike PIL.thumbnail(), resize also enlarges small peppers.
    return image.resize((width, height), Image.Resampling.LANCZOS)


def render_view(
    path: Path,
    size: int,
    view: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    rng = random.Random(seed)
    with Image.open(path) as source:
        pepper, quality = alpha_crop(source)

    if view == 0:
        background = (64, 68, 68)
        target_fraction = 0.88
        angle = 0.0
        offset_x = offset_y = 0
    else:
        palettes = (
            (28, 30, 32),
            (62, 66, 67),
            (105, 101, 94),
            (177, 175, 168),
            (224, 223, 218),
        )
        base = rng.choice(palettes)
        background = tuple(
            max(0, min(255, channel + rng.randint(-10, 10))) for channel in base
        )
        # Narrow scale jitter prevents reintroducing the old size shortcut.
        target_fraction = rng.uniform(0.80, 0.92)
        angle = rng.uniform(-16.0, 16.0)
        offset_x = rng.randint(-round(size * 0.045), round(size * 0.045))
        offset_y = rng.randint(-round(size * 0.045), round(size * 0.045))
        if rng.random() < 0.5:
            pepper = ImageOps.mirror(pepper)
        if rng.random() < 0.12:
            pepper = ImageOps.flip(pepper)
        pepper = ImageEnhance.Brightness(pepper).enhance(rng.uniform(0.88, 1.12))
        pepper = ImageEnhance.Contrast(pepper).enhance(rng.uniform(0.88, 1.14))
        pepper = ImageEnhance.Color(pepper).enhance(rng.uniform(0.90, 1.10))

    pepper = resize_to_long_edge(pepper, max(1, round(size * target_fraction)))
    if view > 0:
        # Apply label-independent resolution degradation to every class.  This
        # makes sharpness a less useful shortcut while preserving view 0 for
        # evaluation and fine defects when the source actually contains them.
        if rng.random() < 0.45:
            down_ratio = rng.uniform(0.55, 0.88)
            down = (
                max(1, round(pepper.width * down_ratio)),
                max(1, round(pepper.height * down_ratio)),
            )
            original_size = pepper.size
            pepper = pepper.resize(down, Image.Resampling.BILINEAR).resize(
                original_size, Image.Resampling.BICUBIC
            )
        if rng.random() < 0.25:
            buffer = io.BytesIO()
            pepper.convert("RGB").save(buffer, format="JPEG", quality=rng.randint(62, 90))
            buffer.seek(0)
            jpeg = Image.open(buffer).convert("RGBA")
            jpeg.putalpha(pepper.getchannel("A"))
            pepper = jpeg.copy()
        if rng.random() < 0.18:
            pepper = pepper.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.75)))

    if angle:
        pepper = pepper.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (size, size), (*background, 255))
    x = (size - pepper.width) // 2 + offset_x
    y = (size - pepper.height) // 2 + offset_y
    canvas.alpha_composite(pepper, (x, y))
    rgb = canvas.convert("RGB")
    if view > 0:
        rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.95, 1.05))
        rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.95, 1.07))
    array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array.copy()), quality


class ManifestViews(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        size: int,
        views: int,
        seed: int,
    ) -> None:
        self.rows = rows
        self.size = size
        self.views = views
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows) * self.views

    def __getitem__(self, index: int):
        row_index, view = divmod(index, self.views)
        view_seed = self.seed + row_index * 100_003 + view * 10_007
        image, quality = render_view(
            Path(self.rows[row_index]["path"]), self.size, view, view_seed
        )
        quality_vector = torch.tensor(
            [
                quality["source_width"],
                quality["source_height"],
                quality["object_width"],
                quality["object_height"],
                int(quality["has_transparency"]),
            ],
            dtype=torch.int32,
        )
        return image, row_index, view, quality_vector


class MultiScaleYOLO(nn.Module):
    """YOLO11 trunk with stage-wise avg/max/std and optional cls projection."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        yolo = YOLO(spec.checkpoint.resolve())
        self.kind = spec.kind
        self.capture = {4, 6, 8, 9 if spec.kind == "cls" else 10}
        stop = 10 if spec.kind == "cls" else 11
        self.layers = nn.ModuleList(list(yolo.model.model[:stop]))
        self.classify = yolo.model.model[10] if spec.kind == "cls" else None
        for parameter in self.parameters():
            parameter.requires_grad = False

    @staticmethod
    def pool(feature: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(feature, 1).flatten(1)
        std = feature.flatten(2).std(2, unbiased=False)
        return torch.cat((avg, maximum, std), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        x = images
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                x = layer(x)
                if index in self.capture:
                    outputs.append(self.pool(x))
            if self.classify is not None:
                projected = self.classify.conv(x)
                projected = self.classify.pool(projected).flatten(1)
                outputs.append(projected)
        return torch.cat(outputs, dim=1)


def backbone_specs(root: Path) -> dict[str, BackboneSpec]:
    return {
        "generic_det": BackboneSpec("generic_det", root / "yolo11n.pt", "det"),
        "pepper_det": BackboneSpec(
            "pepper_det",
            root / "yolo11_pepper/runs/yolo11n_pepper/weights/best.pt",
            "det",
        ),
        "strict_det": BackboneSpec(
            "strict_det",
            root
            / "yolo11_pepper/runs/yolo11n_pepper_strict_v5_f4/weights/best.pt",
            "det",
        ),
        "strict_four_det": BackboneSpec(
            "strict_four_det",
            root
            / "yolo11_pepper/runs/yolo11n_pepper_strict_v5_four_f4/weights/best.pt",
            "det",
        ),
        "imagenet_cls": BackboneSpec("imagenet_cls", root / "yolo11n-cls.pt", "cls"),
    }


def extract_one(
    model: MultiScaleYOLO,
    rows: list[dict[str, str]],
    size: int,
    views: int,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = ManifestViews(rows, size, views, seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[torch.Tensor] = []
    quality_by_row = torch.zeros((len(rows), 5), dtype=torch.int32)
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, row_indices, view_indices, quality) in enumerate(loader, 1):
            chunks.append(model(images.to(device)).cpu())
            canonical = view_indices == 0
            if canonical.any():
                quality_by_row[row_indices[canonical]] = quality[canonical]
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                print(f"  batches {batch_index}/{len(loader)}")
    features = torch.cat(chunks).reshape(len(rows), views, -1)
    return features, quality_by_row


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    rows = read_manifest(args.manifest)
    selected_specs = backbone_specs(root)
    invalid = sorted(set(args.splits) - {row["split"] for row in rows})
    if invalid:
        raise ValueError(f"unknown/empty splits: {invalid}")
    print(
        json.dumps(
            {
                "device": str(device),
                "manifest": str(args.manifest.resolve()),
                "splits": args.splits,
                "backbones": args.backbones,
                "image_size": args.image_size,
                "train_views": args.train_views,
            },
            ensure_ascii=False,
        )
    )

    for backbone_name in args.backbones:
        spec = selected_specs[backbone_name]
        if not spec.checkpoint.exists():
            raise FileNotFoundError(spec.checkpoint)
        print(f"loading {backbone_name}: {spec.checkpoint}")
        model = MultiScaleYOLO(spec).to(device)
        for split in args.splits:
            split_rows = [row for row in rows if row["split"] == split]
            views = args.train_views if split == "train" else args.eval_views
            print(f"extracting {backbone_name}/{split}: {len(split_rows)} x {views}")
            features, quality = extract_one(
                model,
                split_rows,
                args.image_size,
                views,
                args.batch_size,
                args.num_workers,
                args.seed + {"train": 0, "val": 1, "test": 2}.get(split, 3) * 1_000_003,
                device,
            )
            destination = output / f"{backbone_name}_{split}.pt"
            torch.save(
                {
                    "features": features,
                    "quality": quality,
                    "paths": [str(Path(row["path"]).resolve()) for row in split_rows],
                    "groups": [row["group_id"] for row in split_rows],
                    "class_ids": torch.tensor(
                        [int(row["class_id"]) for row in split_rows], dtype=torch.long
                    ),
                    "metadata": {
                        "backbone_name": backbone_name,
                        "checkpoint": str(spec.checkpoint.resolve()),
                        "kind": spec.kind,
                        "split": split,
                        "image_size": args.image_size,
                        "views": views,
                        "seed": args.seed,
                        "manifest": str(args.manifest.resolve()),
                        "scale_normalized": True,
                        "test_requested_explicitly": "test" in args.splits,
                    },
                },
                destination,
            )
            print(
                f"saved {destination} shape={tuple(features.shape)} "
                f"size={destination.stat().st_size / (1024**2):.1f} MiB"
            )
        del model


if __name__ == "__main__":
    main()
