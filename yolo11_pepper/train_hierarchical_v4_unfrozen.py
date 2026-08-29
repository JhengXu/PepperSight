#!/usr/bin/env python3
"""Strict image-level hierarchical training with a partially unfrozen YOLO11.

This is the expensive companion to ``train_hierarchical_v4.py``.  It consumes
the merged v4 manifest directly, materializes only ``selection_role=training``
and ``selection_role=model_selection``, and never loads a historical pepper
classification head.  A public classification-pretrained YOLO11 checkpoint or
a pepper detector checkpoint supplies only visual backbone weights.  The
hierarchical species/conditional-grade head is randomly initialized per seed.

Training uses:
* class -> source-group -> pepper-instance balanced sampling;
* canonical/detector paired views without counting the pair as two samples;
* scale-normalized alpha or detector-crop preprocessing with dynamic weak/strong
  augmentation and resolution degradation;
* independent species/grade weights and grade soft targets from the v4 audit;
* label smoothing, a head-only warm-up, last-stage differential-LR fine tuning,
  EMA teacher supervision, and FixMatch consistency.

The strict-test role is discarded before its path or label fields are accessed.
There is intentionally no CLI argument for test data or an old classification
head.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from ultralytics import YOLO

from train_hierarchical_v4 import (
    FINAL_NAMES,
    GRADE_NAMES,
    SPECIES_NAMES,
    HierarchicalHead,
    choose_device,
    classification_metrics,
    conditional_grade_logits,
    initialize_head,
    joint_probabilities,
    parse_seeds,
    selection_tuple,
    smooth_targets,
    soft_cross_entropy,
    weighted_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict partial-unfreeze YOLO11 hierarchical pepper training."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_merged/manifest.csv"),
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=Path("yolo11n-cls.pt"),
        help="Public YOLO11 classification checkpoint or pepper detector checkpoint.",
    )
    parser.add_argument(
        "--backbone-kind",
        choices=("auto", "classification", "detection"),
        default="auto",
    )
    parser.add_argument(
        "--unfreeze-from",
        type=int,
        default=8,
        help="First YOLO backbone layer to fine-tune after head warm-up.",
    )
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--object-scale", type=float, default=0.86)
    parser.add_argument("--output", type=Path, default=Path("yolo11_pepper/runs/hierarchical_v4_unfrozen"))
    parser.add_argument("--seeds", default="3041")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-warmup-epochs", type=int, default=4)
    parser.add_argument(
        "--clean-warmup-epochs",
        type=int,
        default=3,
        help="Initially supervise grade only on audit high-consistency instances.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--grade-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grade-dropout", type=float, default=0.10)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--projection-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--species-loss-weight", type=float, default=0.70)
    parser.add_argument("--grade-loss-weight", type=float, default=1.30)
    parser.add_argument("--species-label-smoothing", type=float, default=0.02)
    parser.add_argument("--grade-label-smoothing", type=float, default=0.04)
    parser.add_argument("--ssl-weight", type=float, default=0.35)
    parser.add_argument("--ssl-ramp-epochs", type=int, default=10)
    parser.add_argument("--pseudo-threshold", type=float, default=0.88)
    parser.add_argument("--warmup-pseudo-threshold", type=float, default=0.94)
    parser.add_argument("--paired-cross-view-prob", type=float, default=0.60)
    parser.add_argument("--degrade-prob", type=float, default=0.55)
    parser.add_argument("--degrade-min-scale", type=float, default=0.35)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "group_accuracy", "balanced"),
        default="balanced",
    )
    parser.add_argument(
        "--train-bn",
        action="store_true",
        help="Update trainable-stage BatchNorm statistics; default freezes all BN stats.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_soft_pair(value: str | None, hard_grade: int) -> tuple[float, float]:
    if not value:
        target = [0.0, 0.0]
        target[hard_grade] = 1.0
        return tuple(target)
    try:
        parsed = json.loads(value)
        first, second = float(parsed[0]), float(parsed[1])
    except (ValueError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid grade_soft_target={value!r}") from exc
    if first < 0 or second < 0 or first + second <= 1e-8:
        raise ValueError(f"Invalid grade_soft_target={value!r}")
    total = first + second
    return first / total, second / total


@dataclass(frozen=True)
class ImageView:
    path: str
    view_type: str


@dataclass(frozen=True)
class PepperInstance:
    pair_id: str
    group_id: str
    class_id: int
    species_target: tuple[float, float]
    grade_target: tuple[float, float]
    species_weight: float
    grade_weight: float
    high_consistency: bool
    views: tuple[ImageView, ...]


def read_role_instances(path: Path) -> tuple[list[PepperInstance], list[PepperInstance], dict[str, object]]:
    """Materialize training/model-selection roles only; strict-test is ignored early."""
    if "test" in path.name.lower():
        raise ValueError(f"Strict protocol rejected test-named manifest: {path}")
    role_rows: dict[str, list[dict[str, str]]] = {"training": [], "model_selection": []}
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            role = (raw.get("selection_role") or "").strip()
            if role not in role_rows:
                # Do not access path, class_id, group_id, or labels for strict-test/diagnostic rows.
                continue
            if role == "training" and not parse_bool(raw.get("eligible_for_model_training")):
                raise ValueError("A training-role row is not eligible_for_model_training")
            role_rows[role].append(raw)

    def materialize(rows: Sequence[dict[str, str]], training: bool) -> list[PepperInstance]:
        by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            pair_id = row.get("pair_id") or f"single:{row['content_sha256']}"
            by_pair[pair_id].append(row)
        instances: list[PepperInstance] = []
        for pair_id, pair_rows in by_pair.items():
            canonical = [row for row in pair_rows if row.get("view_type") == "canonical"]
            if len(canonical) != 1:
                raise ValueError(f"Expected exactly one canonical row for {pair_id}, got {len(canonical)}")
            label_row = canonical[0]
            class_id = int(label_row["class_id"])
            species_id, grade_id = divmod(class_id, 2)
            species_target = [0.0, 0.0]
            species_target[species_id] = 1.0
            grade_target = (
                parse_soft_pair(label_row.get("grade_soft_target"), grade_id)
                if training
                else ((1.0, 0.0) if grade_id == 0 else (0.0, 1.0))
            )
            species_weight = float(label_row.get("species_weight") or 1.0) if training else 1.0
            grade_weight = float(label_row.get("grade_weight") or 0.0) if training else 1.0
            audit_status = label_row.get("audit_status") or ""
            high_consistency = bool(
                not training
                or (
                    grade_weight > 0
                    and audit_status in {"hard_label_retained", "auto_relabelled_high_conf"}
                )
            )
            views = tuple(
                ImageView(
                    path=str(Path(row["path"]).resolve()),
                    view_type=row.get("view_type") or "canonical",
                )
                for row in pair_rows
            )
            if not all(Path(view.path).exists() for view in views):
                missing = [view.path for view in views if not Path(view.path).exists()]
                raise FileNotFoundError(f"Missing image view(s): {missing[:3]}")
            if any(int(row["class_id"]) != class_id for row in pair_rows):
                raise ValueError(f"Paired views disagree on class for {pair_id}")
            instances.append(
                PepperInstance(
                    pair_id=pair_id,
                    group_id=label_row["group_id"],
                    class_id=class_id,
                    species_target=tuple(species_target),
                    grade_target=grade_target,
                    species_weight=species_weight,
                    grade_weight=grade_weight,
                    high_consistency=high_consistency,
                    views=views,
                )
            )
        return instances

    train = materialize(role_rows["training"], training=True)
    val = materialize(role_rows["model_selection"], training=False)
    train_groups = {item.group_id for item in train}
    val_groups = {item.group_id for item in val}
    if train_groups & val_groups:
        raise ValueError("Source-group leakage between training and model_selection")
    summary = {
        "training_rows": len(role_rows["training"]),
        "training_instances": len(train),
        "training_groups": len(train_groups),
        "training_paired_instances": sum(len(item.views) > 1 for item in train),
        "model_selection_rows": len(role_rows["model_selection"]),
        "model_selection_instances": len(val),
        "model_selection_groups": len(val_groups),
        "training_class_counts": dict(
            zip(
                FINAL_NAMES,
                [sum(item.class_id == class_id for item in train) for class_id in range(4)],
            )
        ),
        "zero_grade_weight": sum(item.grade_weight <= 0 for item in train),
        "soft_grade_targets": sum(max(item.grade_target) < 0.999 for item in train),
        "high_consistency": sum(item.high_consistency for item in train),
    }
    return train, val, summary


def oriented_image(path: str) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).copy()


def alpha_crop(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.nonzero(alpha >= 4)
    if len(xs):
        rgba = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return rgba


def random_background(size: int, rng: random.Random, strong: bool) -> Image.Image:
    if strong:
        palettes = ((28, 30, 32), (62, 66, 67), (108, 104, 96), (182, 180, 172), (226, 225, 220))
        base = rng.choice(palettes)
        color = tuple(max(0, min(255, channel + rng.randint(-14, 14))) for channel in base)
    else:
        color = (64, 68, 68)
    return Image.new("RGBA", (size, size), (*color, 255))


def degrade_resolution(image: Image.Image, rng: random.Random, minimum: float) -> Image.Image:
    ratio = rng.uniform(minimum, 0.82)
    small = image.resize(
        (max(8, round(image.width * ratio)), max(8, round(image.height * ratio))),
        Image.Resampling.BILINEAR,
    )
    return small.resize(image.size, Image.Resampling.BILINEAR)


def render_view(
    view: ImageView,
    size: int,
    object_scale: float,
    strong: bool,
    rng: random.Random,
    degrade_prob: float,
    degrade_min_scale: float,
) -> torch.Tensor:
    source = oriented_image(view.path)
    is_alpha = view.view_type == "canonical" and "A" in source.getbands()
    if is_alpha:
        pepper = alpha_crop(source)
        if strong:
            if rng.random() < 0.5:
                pepper = ImageOps.mirror(pepper)
            pepper = ImageEnhance.Brightness(pepper).enhance(rng.uniform(0.86, 1.14))
            pepper = ImageEnhance.Contrast(pepper).enhance(rng.uniform(0.88, 1.14))
            pepper = ImageEnhance.Color(pepper).enhance(rng.uniform(0.90, 1.10))
            scale = rng.uniform(max(0.62, object_scale - 0.18), min(0.97, object_scale + 0.08))
            angle = rng.uniform(-18, 18)
            offset = round(size * 0.07)
        else:
            scale = object_scale
            angle = 0.0
            offset = 0
        pepper.thumbnail((max(1, round(size * scale)), max(1, round(size * scale))), Image.Resampling.LANCZOS)
        if angle:
            pepper = pepper.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        canvas = random_background(size, rng, strong)
        x = (size - pepper.width) // 2 + (rng.randint(-offset, offset) if offset else 0)
        y = (size - pepper.height) // 2 + (rng.randint(-offset, offset) if offset else 0)
        canvas.alpha_composite(pepper, (x, y))
        rgb = canvas.convert("RGB")
    else:
        rgb_crop = source.convert("RGB")
        if strong and rng.random() < 0.5:
            rgb_crop = ImageOps.mirror(rgb_crop)
        if strong:
            rgb_crop = ImageEnhance.Brightness(rgb_crop).enhance(rng.uniform(0.88, 1.12))
            rgb_crop = ImageEnhance.Contrast(rgb_crop).enhance(rng.uniform(0.90, 1.12))
            rgb_crop = ImageEnhance.Color(rgb_crop).enhance(rng.uniform(0.92, 1.08))
            if rng.random() < 0.35:
                rgb_crop = rgb_crop.rotate(
                    rng.uniform(-8, 8),
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=(64, 68, 68),
                )
            scale = rng.uniform(max(0.70, object_scale - 0.10), min(0.94, object_scale + 0.06))
        else:
            scale = object_scale
        rgb_crop.thumbnail((max(1, round(size * scale)), max(1, round(size * scale))), Image.Resampling.LANCZOS)
        canvas = random_background(size, rng, strong).convert("RGB")
        canvas.paste(rgb_crop, ((size - rgb_crop.width) // 2, (size - rgb_crop.height) // 2))
        rgb = canvas

    if strong:
        if rng.random() < degrade_prob:
            rgb = degrade_resolution(rgb, rng, degrade_min_scale)
        if rng.random() < 0.18:
            rgb = rgb.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.9)))
    array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array.copy())
    if strong and rng.random() < 0.25:
        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        noise = torch.randn(tensor.shape, generator=generator) * rng.uniform(0.004, 0.018)
        tensor = (tensor + noise).clamp(0, 1)
    return tensor


class PepperTrainDataset(Dataset):
    def __init__(self, instances: Sequence[PepperInstance], args: argparse.Namespace, seed: int) -> None:
        self.instances = instances
        self.args = args
        self.seed = seed
        self.epoch = 0
        self.calls: dict[int, int] = defaultdict(int)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.calls.clear()

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, object]:
        instance = self.instances[index]
        occurrence = self.calls[index]
        self.calls[index] += 1
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 10_007 + occurrence * 97)
        canonical = next(view for view in instance.views if view.view_type == "canonical")
        detector = next((view for view in instance.views if view.view_type == "detector_aligned"), None)
        if detector is not None and rng.random() < self.args.paired_cross_view_prob:
            weak_source, strong_source = canonical, detector
        else:
            weak_source = rng.choice(instance.views)
            strong_source = weak_source
        weak = render_view(
            weak_source,
            self.args.image_size,
            self.args.object_scale,
            False,
            rng,
            self.args.degrade_prob,
            self.args.degrade_min_scale,
        )
        strong = render_view(
            strong_source,
            self.args.image_size,
            self.args.object_scale,
            True,
            rng,
            self.args.degrade_prob,
            self.args.degrade_min_scale,
        )
        return {
            "weak": weak,
            "strong": strong,
            "species_target": torch.tensor(instance.species_target, dtype=torch.float32),
            "grade_target": torch.tensor(instance.grade_target, dtype=torch.float32),
            "species_weight": torch.tensor(instance.species_weight, dtype=torch.float32),
            "grade_weight": torch.tensor(instance.grade_weight, dtype=torch.float32),
            "high_consistency": torch.tensor(instance.high_consistency, dtype=torch.bool),
            "hard_species": torch.tensor(instance.class_id // 2, dtype=torch.long),
            "hard_grade": torch.tensor(instance.class_id % 2, dtype=torch.long),
            "class_id": torch.tensor(instance.class_id, dtype=torch.long),
        }


class PepperValidationDataset(Dataset):
    def __init__(self, instances: Sequence[PepperInstance], args: argparse.Namespace) -> None:
        self.instances, self.args = instances, args

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        instance = self.instances[index]
        canonical = next(view for view in instance.views if view.view_type == "canonical")
        image = render_view(
            canonical,
            self.args.image_size,
            self.args.object_scale,
            False,
            random.Random(100_003 + index),
            0.0,
            self.args.degrade_min_scale,
        )
        return image, torch.tensor(instance.class_id, dtype=torch.long)


class BalancedInstanceBatchSampler(Sampler[list[int]]):
    """Equal final classes; equal source groups within each class; then instance."""

    def __init__(
        self,
        instances: Sequence[PepperInstance],
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size < 4:
            raise ValueError("batch_size must be at least four")
        self.batch_size, self.batches_per_epoch, self.seed = batch_size, batches_per_epoch, seed
        self.epoch = 0
        nested: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, instance in enumerate(instances):
            nested[instance.class_id][instance.group_id].append(index)
        if any(not nested[class_id] for class_id in range(4)):
            raise ValueError("All four final classes must exist in training")
        self.nested = nested

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + self.epoch * 999_983)
        for _ in range(self.batches_per_epoch):
            slots = [index % 4 for index in range(self.batch_size)]
            rng.shuffle(slots)
            batch = []
            for class_id in slots:
                group = rng.choice(tuple(self.nested[class_id]))
                batch.append(rng.choice(self.nested[class_id][group]))
            rng.shuffle(batch)
            yield batch


class GeMPool(nn.Module):
    def __init__(self, initial: float = 3.0) -> None:
        super().__init__()
        self.log_p = nn.Parameter(torch.tensor(math.log(initial)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        p = self.log_p.exp().clamp(1.0, 6.0)
        return feature.clamp_min(1e-6).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)


class PartialYOLOEncoder(nn.Module):
    def __init__(
        self,
        checkpoint: Path,
        kind: str,
        unfreeze_from: int,
        projection_dim: int,
        train_bn: bool,
    ) -> None:
        super().__init__()
        yolo = YOLO(checkpoint.resolve())
        source_layers = list(yolo.model.model)
        detected_kind = "classification" if type(source_layers[-1]).__name__ == "Classify" else "detection"
        self.kind = detected_kind if kind == "auto" else kind
        if self.kind != detected_kind:
            raise ValueError(
                f"--backbone-kind={kind} conflicts with checkpoint model type {detected_kind}"
            )
        if self.kind == "classification":
            classifier = source_layers[-1]
            self.layers = nn.ModuleList(source_layers[:-1])
            self.projection = copy.deepcopy(classifier.conv)
            self.feature_dim = int(classifier.conv.conv.out_channels)
            self.projection_pretrained = True
        else:
            self.layers = nn.ModuleList(source_layers[:11])
            output_channels = self._infer_output_channels()
            self.projection = nn.Sequential(
                nn.Conv2d(output_channels, projection_dim, 1, bias=False),
                nn.BatchNorm2d(projection_dim),
                nn.SiLU(inplace=True),
            )
            self.feature_dim = projection_dim
            self.projection_pretrained = False
        if not 0 <= unfreeze_from < len(self.layers):
            raise ValueError(f"unfreeze_from={unfreeze_from} outside 0:{len(self.layers)}")
        self.unfreeze_from = unfreeze_from
        self.train_bn = train_bn
        self.pool = GeMPool()
        self.fine_tune_enabled = False
        for index, layer in enumerate(self.layers):
            for parameter in layer.parameters():
                parameter.requires_grad = index >= unfreeze_from
        for parameter in self.projection.parameters():
            parameter.requires_grad = True

    def _infer_output_channels(self) -> int:
        # Nested YOLO blocks may end with an internal bottleneck convolution whose
        # channel count differs from the block output.  A tiny eval forward is the
        # reliable architecture-agnostic way to resolve the actual output shape.
        self.layers.eval()
        with torch.no_grad():
            feature = torch.zeros(1, 3, 128, 128)
            for layer in self.layers:
                feature = layer(feature)
        if feature.ndim != 4:
            raise RuntimeError(f"Unexpected YOLO backbone output shape: {tuple(feature.shape)}")
        return int(feature.shape[1])

    def stage_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for index, layer in enumerate(self.layers)
            if index >= self.unfreeze_from
            for parameter in layer.parameters()
        ]

    def projection_parameters(self) -> list[nn.Parameter]:
        return list(self.projection.parameters()) + list(self.pool.parameters())

    def set_training_mode(self, enabled: bool) -> None:
        self.fine_tune_enabled = enabled
        self.train()
        for index, layer in enumerate(self.layers):
            if index < self.unfreeze_from or not enabled:
                layer.eval()
        if not self.train_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = image
        for index, layer in enumerate(self.layers):
            if index < self.unfreeze_from or not self.fine_tune_enabled:
                with torch.no_grad():
                    feature = layer(feature)
            else:
                feature = layer(feature)
        feature = self.projection(feature)
        return self.pool(feature)


class EndToEndHierarchicalModel(nn.Module):
    def __init__(self, encoder: PartialYOLOEncoder, head: HierarchicalHead) -> None:
        super().__init__()
        self.encoder, self.head = encoder, head

    def set_training_mode(self, fine_tune: bool) -> None:
        self.train()
        self.encoder.set_training_mode(fine_tune)
        self.head.train()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.encoder(image))


def ema_update_model(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)


def set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    progress = (epoch - 1) / max(args.epochs - 1, 1)
    cosine = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    values = {
        "backbone": 0.0 if epoch <= args.head_warmup_epochs else args.backbone_lr * cosine,
        "projection": args.projection_lr * cosine,
        "head": args.head_lr * cosine,
    }
    for group in optimizer.param_groups:
        group["lr"] = values[group["name"]]
    return values


def fixmatch_loss(
    student: EndToEndHierarchicalModel,
    teacher: EndToEndHierarchicalModel,
    weak: torch.Tensor,
    strong: torch.Tensor,
    hard_species: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        teacher_species_logits, teacher_grade_logits = teacher(weak)
        species_probability = teacher_species_logits.softmax(1)
        species_confidence, species_label = species_probability.max(1)
        teacher_grade = conditional_grade_logits(teacher_grade_logits, hard_species).softmax(1)
        grade_confidence, grade_label = teacher_grade.max(1)
        species_mask = species_confidence >= threshold
        grade_mask = grade_confidence >= threshold
    student_species_logits, student_grade_logits = student(strong)
    species_each = F.cross_entropy(student_species_logits, species_label, reduction="none")
    grade_each = F.cross_entropy(
        conditional_grade_logits(student_grade_logits, hard_species), grade_label, reduction="none"
    )
    species_loss = species_each[species_mask].mean() if species_mask.any() else species_each.sum() * 0
    grade_loss = grade_each[grade_mask].mean() if grade_mask.any() else grade_each.sum() * 0
    return 0.25 * species_loss + 0.75 * grade_loss, {
        "pseudo_species": float(species_mask.float().mean()),
        "pseudo_grade": float(grade_mask.float().mean()),
    }


def validation_probabilities(
    model: EndToEndHierarchicalModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            species_logits, grade_logits = model(images.to(device))
            probabilities.append(joint_probabilities(species_logits, grade_logits).cpu())
            labels.append(batch_labels)
    return torch.cat(probabilities), torch.cat(labels)


@dataclass
class Candidate:
    seed: int
    best_epoch: int
    validation: dict[str, object]
    history: list[dict[str, object]]
    state_dict: dict[str, torch.Tensor]
    encoder_metadata: dict[str, object]


def build_model(args: argparse.Namespace, seed: int) -> EndToEndHierarchicalModel:
    encoder = PartialYOLOEncoder(
        args.backbone,
        args.backbone_kind,
        args.unfreeze_from,
        args.projection_dim,
        args.train_bn,
    )
    # The only pepper classification-head construction path is random initialization.
    head = initialize_head(encoder.feature_dim, args, seed)
    return EndToEndHierarchicalModel(encoder, head)


def train_seed(
    seed: int,
    train_instances: Sequence[PepperInstance],
    val_instances: Sequence[PepperInstance],
    args: argparse.Namespace,
    device: torch.device,
) -> Candidate:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    student = build_model(args, seed).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            {
                "params": student.encoder.stage_parameters(),
                "lr": 0.0,
                "name": "backbone",
                "weight_decay": args.weight_decay,
            },
            {
                "params": student.encoder.projection_parameters(),
                "lr": args.projection_lr,
                "name": "projection",
                "weight_decay": args.weight_decay,
            },
            {
                "params": student.head.parameters(),
                "lr": args.head_lr,
                "name": "head",
                "weight_decay": args.weight_decay,
            },
        ]
    )
    train_dataset = PepperTrainDataset(train_instances, args, seed)
    batches = args.batches_per_epoch or math.ceil(len(train_instances) / args.batch_size)
    batch_sampler = BalancedInstanceBatchSampler(
        train_instances, args.batch_size, batches, seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_dataset = PepperValidationDataset(val_instances, args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(args.batch_size, 16),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_groups = [item.group_id for item in val_instances]
    best_score: tuple[float, float, float] | None = None
    best_state = copy.deepcopy(teacher.state_dict())
    best_validation: dict[str, object] = {}
    best_epoch, stale = 0, 0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        fine_tune = epoch > args.head_warmup_epochs
        student.set_training_mode(fine_tune)
        teacher.eval()
        learning_rates = set_optimizer_lrs(optimizer, epoch, args)
        threshold = (
            args.warmup_pseudo_threshold
            if epoch <= args.head_warmup_epochs
            else args.pseudo_threshold
        )
        ssl_ramp = min(1.0, max(0.0, (epoch - 1) / max(args.ssl_ramp_epochs, 1)))
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        for batch in train_loader:
            weak = batch["weak"].to(device)
            strong = batch["strong"].to(device)
            hard_species = batch["hard_species"].to(device)
            species_target = smooth_targets(
                batch["species_target"].to(device), args.species_label_smoothing
            )
            grade_target = smooth_targets(
                batch["grade_target"].to(device), args.grade_label_smoothing
            )
            species_weight = batch["species_weight"].to(device)
            grade_weight = batch["grade_weight"].to(device)
            if epoch <= args.clean_warmup_epochs:
                grade_weight = grade_weight * batch["high_consistency"].to(device).float()
            species_logits, grade_logits = student(weak)
            species_each = soft_cross_entropy(species_logits, species_target)
            grade_each = soft_cross_entropy(
                conditional_grade_logits(grade_logits, hard_species), grade_target
            )
            species_loss = weighted_mean(species_each, species_weight)
            grade_loss = weighted_mean(grade_each, grade_weight)
            ssl_loss, pseudo = fixmatch_loss(
                student, teacher, weak, strong, hard_species, threshold
            )
            loss = (
                args.species_loss_weight * species_loss
                + args.grade_loss_weight * grade_loss
                + args.ssl_weight * ssl_ramp * ssl_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            ema_update_model(teacher, student, args.ema_decay)
            count = len(weak)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["species_loss"] += float(species_loss.detach()) * count
            totals["grade_loss"] += float(grade_loss.detach()) * count
            totals["ssl_loss"] += float(ssl_loss.detach()) * count
            totals["pseudo_species"] += pseudo["pseudo_species"] * count
            totals["pseudo_grade"] += pseudo["pseudo_grade"] * count

        probability, val_labels = validation_probabilities(teacher, val_loader, device)
        validation = classification_metrics(val_labels, probability, val_groups)
        row: dict[str, object] = {
            "epoch": epoch,
            **{name: value / max(seen, 1) for name, value in totals.items()},
            "fine_tune": fine_tune,
            "pseudo_threshold": threshold,
            "ssl_ramp": ssl_ramp,
            "learning_rates": learning_rates,
            "val_accuracy": validation["accuracy"],
            "val_species_accuracy": validation["species_accuracy"],
            "val_grade_accuracy": validation["grade_accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_group_accuracy": validation["group_accuracy"],
            "val_nll": validation["nll"],
        }
        history.append(row)
        score = selection_tuple(validation, args.selection_metric)
        if best_score is None or score > best_score:
            best_score = score
            best_state = copy.deepcopy(teacher.state_dict())
            best_validation = validation
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        if not args.quiet and (epoch == 1 or epoch % 2 == 0):
            print(json.dumps({"seed": seed, **row}, ensure_ascii=False))
        if stale >= args.patience:
            break

    encoder = student.encoder
    metadata = {
        "backbone_kind": encoder.kind,
        "backbone_layers": len(encoder.layers),
        "unfreeze_from": encoder.unfreeze_from,
        "projection_pretrained": encoder.projection_pretrained,
        "feature_dim": encoder.feature_dim,
        "train_bn": encoder.train_bn,
        "trainable_stage_parameters": sum(
            parameter.numel() for parameter in encoder.stage_parameters()
        ),
        "projection_parameters": sum(
            parameter.numel() for parameter in encoder.projection_parameters()
        ),
        "head_parameters": sum(parameter.numel() for parameter in student.head.parameters()),
    }
    return Candidate(
        seed=seed,
        best_epoch=best_epoch,
        validation=best_validation,
        history=history,
        state_dict={name: value.cpu() for name, value in best_state.items()},
        encoder_metadata=metadata,
    )


def checkpoint_payload(
    candidate: Candidate,
    args: argparse.Namespace,
    manifest_summary: dict[str, object],
) -> dict[str, object]:
    head_prefix = "head."
    head_state = {
        name[len(head_prefix) :]: value
        for name, value in candidate.state_dict.items()
        if name.startswith(head_prefix)
    }
    return {
        "model_state_dict": candidate.state_dict,
        "head_state_dict": head_state,
        "backbone_checkpoint": str(args.backbone.resolve()),
        "image_size": args.image_size,
        "object_scale": args.object_scale,
        "species_names": SPECIES_NAMES,
        "grade_names": GRADE_NAMES,
        "final_names": FINAL_NAMES,
        "architecture": {
            "type": "partially_unfrozen_yolo11_hierarchical_v4",
            **candidate.encoder_metadata,
            "hidden_dim": args.hidden_dim,
            "grade_hidden_dim": args.grade_hidden_dim,
            "decision": "argmax p(species) * p(grade|species)",
        },
        "preprocessing": (
            "scale-normalized alpha/detector crop; dynamic weak/strong views; "
            "resolution degradation; canonical neutral validation"
        ),
        "strict_protocol": {
            "historical_pepper_classification_head_loaded": False,
            "hierarchical_head_randomly_initialized": True,
            "test_rows_materialized": False,
            "test_images_loaded": False,
            "test_metrics_computed": False,
            "selection_data": "selection_role=model_selection only",
        },
        "training": {
            "seed": candidate.seed,
            "best_epoch": candidate.best_epoch,
            "selection_metric": args.selection_metric,
            "sampling": "class -> source_group -> pepper_instance",
            "ssl": "single EMA teacher + FixMatch weak/strong consistency",
            "manifest_summary": manifest_summary,
        },
        "validation_metrics": candidate.validation,
    }


def public_candidate(candidate: Candidate) -> dict[str, object]:
    return {
        "seed": candidate.seed,
        "best_epoch": candidate.best_epoch,
        "validation": candidate.validation,
        "encoder_metadata": candidate.encoder_metadata,
        "history": candidate.history,
    }


def main() -> None:
    args = parse_args()
    if not args.manifest.exists() or not args.backbone.exists():
        raise FileNotFoundError("Manifest/backbone checkpoint does not exist")
    if not 0 < args.object_scale <= 1:
        raise ValueError("--object-scale must be in (0, 1]")
    if not 0 <= args.degrade_prob <= 1 or not 0.1 <= args.degrade_min_scale <= 0.82:
        raise ValueError("Invalid resolution-degradation settings")
    if not 0 <= args.paired_cross_view_prob <= 1:
        raise ValueError("--paired-cross-view-prob must be in [0, 1]")
    if not 0 < args.pseudo_threshold <= args.warmup_pseudo_threshold <= 1:
        raise ValueError("Invalid pseudo-label thresholds")
    device = choose_device(args.device)
    seeds = parse_seeds(args.seeds)
    train_instances, val_instances, manifest_summary = read_role_instances(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "strict_protocol": "TRAIN + MODEL_SELECTION ONLY; TEST NOT READ",
                "device": str(device),
                "manifest": str(args.manifest.resolve()),
                "backbone": str(args.backbone.resolve()),
                "seeds": seeds,
                "manifest_summary": manifest_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    candidates = []
    for seed in seeds:
        candidate = train_seed(seed, train_instances, val_instances, args, device)
        candidates.append(candidate)
        torch.save(
            checkpoint_payload(candidate, args, manifest_summary),
            output / f"candidate_seed_{seed}.pt",
        )
        print(
            json.dumps(
                {
                    "seed_complete": seed,
                    "best_epoch": candidate.best_epoch,
                    "validation": candidate.validation,
                },
                ensure_ascii=False,
            )
        )
    best = max(
        candidates,
        key=lambda item: selection_tuple(item.validation, args.selection_metric),
    )
    best_path = output / "best_hierarchical_v4_unfrozen_strict.pt"
    torch.save(checkpoint_payload(best, args, manifest_summary), best_path)
    report = {
        "strict_protocol": checkpoint_payload(best, args, manifest_summary)["strict_protocol"],
        "manifest": str(args.manifest.resolve()),
        "backbone": str(args.backbone.resolve()),
        "test_manifest": None,
        "test_metrics": None,
        "configuration": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "manifest_summary": manifest_summary,
        "candidates": [public_candidate(candidate) for candidate in candidates],
        "selected_seed": best.seed,
        "selected_validation": best.validation,
        "checkpoint": str(best_path),
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_seed": best.seed,
                "selected_validation": best.validation,
                "checkpoint": str(best_path),
                "report": str(report_path),
                "test_was_not_read": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
