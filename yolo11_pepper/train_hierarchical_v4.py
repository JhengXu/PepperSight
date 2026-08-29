#!/usr/bin/env python3
"""Strict hierarchical pepper-head training from cached multi-view features.

Protocol guarantees
-------------------
* The species/grade head is always randomly initialized.  This file has no
  argument or code path that can load a historical classification head.
* Only ``train`` and ``val`` rows are materialized from the manifest.  No test
  feature path is accepted and no test image, label, feature, or metric is read.
* Hyper-parameter/seed selection uses validation metrics only.

Training combines class -> source-group -> sample balanced batches, independent
species/grade soft targets and weights, true batch co-teaching (two students
exchange their small-loss grade samples), label smoothing, a clean-sample
warm-up, and cross-EMA FixMatch consistency over cached weak/strong views.

The script intentionally trains only a classification head over pre-extracted
features.  It is therefore fast enough to compare several seeds locally before
running a more expensive partially-unfrozen image model.
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
from torch import nn


SPECIES_NAMES = ("子弹头", "条子")
GRADE_NAMES = ("一级", "二级")
FINAL_NAMES = ("子弹头_一级", "子弹头_二级", "条子_一级", "条子_二级")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict no-test hierarchical co-teaching on cached features."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"),
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        nargs="+",
        default=[Path("yolo11_pepper/runs/hierarchical_ssl_v3/train_features.pt")],
    )
    parser.add_argument(
        "--val-features",
        type=Path,
        nargs="+",
        default=[Path("yolo11_pepper/runs/hierarchical_ssl_v3/val_features.pt")],
    )
    parser.add_argument(
        "--label-decisions",
        type=Path,
        default=None,
        help=(
            "Optional TRAIN-ONLY audit CSV keyed by path. Supported fields include "
            "effective_class_id, species_weight, grade_weight/label_weight, "
            "species_soft_0/1, grade_soft_0/1 or p_good/p_bad, high_consistency, "
            "and label_state. Validation labels are never overridden."
        ),
    )
    parser.add_argument(
        "--use-provided-soft-labels",
        action="store_true",
        help="Use probability columns from the TRAIN-ONLY audit CSV as soft targets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/runs/hierarchical_v4_strict"),
    )
    parser.add_argument("--seeds", default="2041,2053,2069")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=0,
        help="0 means ceil(number_of_train_items / batch_size).",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--grade-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grade-dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--ema-decay", type=float, default=0.992)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--species-loss-weight", type=float, default=0.75)
    parser.add_argument("--grade-loss-weight", type=float, default=1.25)
    parser.add_argument("--species-label-smoothing", type=float, default=0.02)
    parser.add_argument("--grade-label-smoothing", type=float, default=0.04)
    parser.add_argument("--species-class-weight-power", type=float, default=0.0)
    parser.add_argument("--grade-class-weight-power", type=float, default=0.0)
    parser.add_argument(
        "--forget-rate",
        type=float,
        default=0.12,
        help="Estimated grade-label noise fraction removed by peer small-loss selection.",
    )
    parser.add_argument("--forget-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--clean-warmup-epochs",
        type=int,
        default=5,
        help="Initially supervise grade only on high-consistency audited samples.",
    )
    parser.add_argument("--ssl-weight", type=float, default=0.45)
    parser.add_argument("--ssl-ramp-epochs", type=int, default=10)
    parser.add_argument("--pseudo-threshold", type=float, default=0.85)
    parser.add_argument("--warmup-pseudo-threshold", type=float, default=0.92)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Conditional-grade supervised contrastive weight on the shared embedding.",
    )
    parser.add_argument("--contrastive-temperature", type=float, default=0.15)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "group_accuracy", "balanced"),
        default="balanced",
        help="All choices are computed on validation only.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def reject_test_feature_path(path: Path) -> None:
    """Fail closed when someone accidentally passes a test cache."""
    tokens = {part.lower() for part in path.parts}
    if "test" in tokens or "test_features.pt" in path.name.lower():
        raise ValueError(f"STRICT NO-TEST protocol rejected feature path: {path}")


def read_train_val_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read only train/val records; test records are discarded before field access."""
    result: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            split = (raw.get("split") or "").strip().lower()
            if split not in result:
                # Strictly do not materialize labels, paths, or groups for test/other rows.
                continue
            # The merged v4 manifest physically includes detector-aligned
            # validation crops for diagnostics.  They are explicitly excluded
            # from model selection and must never silently enlarge val.
            selection_role = (raw.get("selection_role") or "").strip().lower()
            if split == "train" and selection_role not in {"", "training"}:
                continue
            if split == "val" and selection_role not in {"", "model_selection"}:
                continue
            class_id = int(raw["class_id"])
            if not 0 <= class_id < 4:
                raise ValueError(f"Invalid class_id={class_id} in {split}")
            result[split].append(
                {
                    **raw,
                    "split": split,
                    "path": str(Path(raw["path"]).resolve()),
                    "group_id": raw["group_id"],
                    "class_id": str(class_id),
                }
            )
    if not result["train"] or not result["val"]:
        raise ValueError("Manifest must provide non-empty train and val splits")
    train_groups = {row["group_id"] for row in result["train"]}
    val_groups = {row["group_id"] for row in result["val"]}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"Group leakage between train and val: {sorted(overlap)[:5]}")
    return result


def load_feature_cache(path: Path, expected_rows: Sequence[dict[str, str]]) -> torch.Tensor:
    reject_test_feature_path(path)
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    features = payload["features"] if isinstance(payload, dict) else payload
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Feature cache does not contain a tensor: {path}")
    if features.ndim == 2:
        features = features.unsqueeze(1)
    if features.ndim != 3:
        raise ValueError(f"Expected [items, views, dim], got {tuple(features.shape)}")
    if len(features) != len(expected_rows):
        raise ValueError(
            f"Feature/manifest length mismatch for {path}: {len(features)} != {len(expected_rows)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"Feature cache contains NaN/Inf: {path}")
    cached_paths = payload.get("paths") if isinstance(payload, dict) else None
    if cached_paths is not None:
        normalized = [str(Path(item).resolve()) for item in cached_paths]
        expected = [row["path"] for row in expected_rows]
        if normalized != expected:
            raise ValueError(f"Feature path order does not match manifest: {path}")
    return features.float().contiguous()


def load_feature_caches(
    paths: Sequence[Path], expected_rows: Sequence[dict[str, str]]
) -> torch.Tensor:
    """Load, independently normalize and concatenate complementary backbones."""
    if not paths:
        raise ValueError("At least one feature cache is required")
    blocks = [load_feature_cache(path, expected_rows) for path in paths]
    expected_views = blocks[0].shape[1]
    if any(block.shape[1] != expected_views for block in blocks[1:]):
        shapes = [tuple(block.shape) for block in blocks]
        raise ValueError(f"Feature caches have different view counts: {shapes}")
    normalized = [
        F.normalize(block, dim=-1) * (block.shape[-1] ** 0.5) for block in blocks
    ]
    return torch.cat(normalized, dim=-1).contiguous()


def optional_float(row: dict[str, str], names: Iterable[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return None


def optional_bool(row: dict[str, str], names: Iterable[str]) -> bool | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        return str(value).strip().lower() in {"1", "true", "yes", "y", "clean", "high"}
    return None


def normalized_pair(first: float, second: float, label: str) -> tuple[float, float]:
    if first < 0 or second < 0 or not math.isfinite(first + second):
        raise ValueError(f"Invalid {label} soft target: {(first, second)}")
    total = first + second
    if total <= 1e-8:
        raise ValueError(f"Zero-mass {label} soft target")
    return first / total, second / total


def read_label_decisions(
    path: Path | None, allowed_train_paths: set[str]
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    decisions: dict[str, dict[str, str]] = {}
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("path"):
                raise ValueError("Every label-decision row must contain path")
            key = str(Path(row["path"]).resolve())
            if key not in allowed_train_paths:
                raise ValueError(
                    "Strict protocol accepts label decisions for TRAIN paths only; "
                    f"rejected: {key}"
                )
            if key in decisions:
                raise ValueError(f"Duplicate label-decision path: {key}")
            decisions[key] = row
    return decisions


@dataclass
class LabelBundle:
    species_targets: torch.Tensor
    grade_targets: torch.Tensor
    species_weights: torch.Tensor
    grade_weights: torch.Tensor
    high_consistency: torch.Tensor
    hard_species: torch.Tensor
    hard_grade: torch.Tensor
    final_classes: torch.Tensor
    group_ids: list[str]
    paths: list[str]
    audit_summary: dict[str, object]


def build_labels(
    rows: Sequence[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    use_soft_labels: bool,
    allow_overrides: bool,
) -> LabelBundle:
    species_targets: list[tuple[float, float]] = []
    grade_targets: list[tuple[float, float]] = []
    species_weights: list[float] = []
    grade_weights: list[float] = []
    high_consistency: list[bool] = []
    groups, paths = [], []
    used_decisions = 0
    state_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        original_class = int(row["class_id"])
        decision = decisions.get(row["path"], {}) if allow_overrides else {}
        if decision:
            used_decisions += 1
        effective_class = int(decision.get("effective_class_id") or original_class)
        if not 0 <= effective_class < 4:
            raise ValueError(f"Invalid effective_class_id for {row['path']}")
        species_id, grade_id = divmod(effective_class, 2)
        species_target = [0.0, 0.0]
        grade_target = [0.0, 0.0]
        species_target[species_id] = 1.0
        grade_target[grade_id] = 1.0

        if use_soft_labels and decision:
            species_0 = optional_float(
                decision, ("species_soft_0", "p_species_0", "p_子弹头")
            )
            species_1 = optional_float(
                decision, ("species_soft_1", "p_species_1", "p_条子")
            )
            if species_0 is not None and species_1 is not None:
                species_target = list(normalized_pair(species_0, species_1, "species"))
            grade_0 = optional_float(
                decision, ("grade_soft_0", "p_grade_0", "p_good", "p_一级")
            )
            grade_1 = optional_float(
                decision, ("grade_soft_1", "p_grade_1", "p_bad", "p_二级")
            )
            serialized_grade_target = decision.get("grade_soft_target")
            if serialized_grade_target not in (None, ""):
                parsed_target = json.loads(serialized_grade_target)
                if not isinstance(parsed_target, list) or len(parsed_target) != 2:
                    raise ValueError(
                        f"Invalid grade_soft_target for {row['path']}: "
                        f"{serialized_grade_target!r}"
                    )
                grade_0, grade_1 = float(parsed_target[0]), float(parsed_target[1])
            if grade_0 is not None and grade_1 is not None:
                grade_target = list(normalized_pair(grade_0, grade_1, "grade"))

        species_weight = optional_float(decision, ("species_weight",))
        if species_weight is None:
            species_weight = optional_float(row, ("species_weight",))
        if species_weight is None:
            species_weight = 1.0
        grade_weight = optional_float(decision, ("grade_weight", "label_weight"))
        if grade_weight is None:
            grade_weight = optional_float(row, ("grade_weight", "label_weight"))
        if grade_weight is None:
            grade_weight = 1.0
        if species_weight < 0 or grade_weight < 0:
            raise ValueError(f"Negative sample weight for {row['path']}")

        clean = optional_bool(decision, ("high_consistency", "is_clean"))
        state = (
            decision.get("audit_status")
            or decision.get("label_state")
            or "human_default"
        ).strip()
        state_counts[state] += 1
        if clean is None:
            clean = state in {
                "human_verified",
                "retained",
                "auto_relabelled_high_conf",
                "clean",
                "hard_label_retained",
            }
            if not decision:
                clean = grade_weight >= 0.999
        clean = bool(clean and grade_weight > 0)

        species_targets.append(tuple(species_target))
        grade_targets.append(tuple(grade_target))
        species_weights.append(float(species_weight))
        grade_weights.append(float(grade_weight))
        high_consistency.append(clean)
        groups.append(row["group_id"])
        paths.append(row["path"])

    species_tensor = torch.tensor(species_targets, dtype=torch.float32)
    grade_tensor = torch.tensor(grade_targets, dtype=torch.float32)
    hard_species = species_tensor.argmax(1)
    hard_grade = grade_tensor.argmax(1)
    final_classes = hard_species * 2 + hard_grade
    return LabelBundle(
        species_targets=species_tensor,
        grade_targets=grade_tensor,
        species_weights=torch.tensor(species_weights, dtype=torch.float32),
        grade_weights=torch.tensor(grade_weights, dtype=torch.float32),
        high_consistency=torch.tensor(high_consistency, dtype=torch.bool),
        hard_species=hard_species,
        hard_grade=hard_grade,
        final_classes=final_classes,
        group_ids=groups,
        paths=paths,
        audit_summary={
            "rows": len(rows),
            "decisions_applied": used_decisions,
            "high_consistency": int(sum(high_consistency)),
            "zero_grade_weight": int(sum(value <= 0 for value in grade_weights)),
            "soft_species_targets": int(sum(max(value) < 0.999 for value in species_targets)),
            "soft_grade_targets": int(sum(max(value) < 0.999 for value in grade_targets)),
            "label_states": dict(state_counts),
        },
    )


class ClassGroupBalancedBatches:
    """Draw class uniformly, then source group uniformly, then sample uniformly."""

    def __init__(
        self,
        classes: torch.Tensor,
        group_ids: Sequence[str],
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size < 4:
            raise ValueError("batch_size must be at least four")
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        nested: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (class_id, group_id) in enumerate(zip(classes.tolist(), group_ids)):
            nested[int(class_id)][group_id].append(index)
        missing = [class_id for class_id in range(4) if not nested[class_id]]
        if missing:
            raise ValueError(f"Training labels are missing final classes: {missing}")
        self.groups = {
            class_id: {group: tuple(indices) for group, indices in group_map.items()}
            for class_id, group_map in nested.items()
        }

    def iter_epoch(self, epoch: int) -> Iterable[torch.Tensor]:
        rng = random.Random(self.seed + epoch * 1_000_003)
        for batch_index in range(self.batches_per_epoch):
            class_slots = [slot % 4 for slot in range(self.batch_size)]
            rng.shuffle(class_slots)
            batch: list[int] = []
            for class_id in class_slots:
                group_map = self.groups[class_id]
                group_id = rng.choice(tuple(group_map))
                batch.append(rng.choice(group_map[group_id]))
            # Break any residual ordering while preserving the exact class quota.
            rng.shuffle(batch)
            yield torch.tensor(batch, dtype=torch.long)


class HierarchicalHead(nn.Module):
    """p(species) plus one p(grade | species) head per species."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        grade_hidden_dim: int,
        dropout: float,
        grade_dropout: float,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.species_head = nn.Linear(hidden_dim, 2)
        self.grade_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, grade_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(grade_dropout),
                    nn.Linear(grade_hidden_dim, 2),
                )
                for _ in range(2)
            ]
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        species_logits = self.species_head(shared)
        grade_logits = torch.stack([head(shared) for head in self.grade_heads], dim=1)
        return species_logits, grade_logits

    def forward_with_embedding(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        species_logits = self.species_head(shared)
        grade_logits = torch.stack([head(shared) for head in self.grade_heads], dim=1)
        return species_logits, grade_logits, shared


def initialize_head(feature_dim: int, args: argparse.Namespace, seed: int) -> HierarchicalHead:
    """The only head-construction path: deterministic random initialization."""
    torch.manual_seed(seed)
    return HierarchicalHead(
        feature_dim,
        args.hidden_dim,
        args.grade_hidden_dim,
        args.dropout,
        args.grade_dropout,
    )


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if not 0 <= smoothing < 1:
        raise ValueError("label smoothing must be in [0, 1)")
    return targets * (1 - smoothing) + smoothing / targets.shape[-1]


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def conditional_grade_contrastive_loss(
    embedding: torch.Tensor,
    species: torch.Tensor,
    grade: torch.Tensor,
    eligible: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Pull equal grades together and separate grades only within a species."""
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    selected = torch.nonzero(eligible, as_tuple=False).flatten()
    if len(selected) < 4:
        return embedding.sum() * 0
    z = F.normalize(embedding[selected], dim=1)
    selected_species = species[selected]
    selected_grade = grade[selected]
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    identity = torch.eye(len(selected), dtype=torch.bool, device=embedding.device)
    denominator_mask = (selected_species[:, None] == selected_species[None, :]) & ~identity
    positive_mask = denominator_mask & (
        selected_grade[:, None] == selected_grade[None, :]
    )
    valid = positive_mask.any(1) & denominator_mask.any(1)
    if not valid.any():
        return embedding.sum() * 0
    exponent = logits.exp()
    numerator = (exponent * positive_mask).sum(1).clamp_min(1e-9)
    denominator = (exponent * denominator_mask).sum(1).clamp_min(1e-9)
    each = -(numerator / denominator).clamp_min(1e-9).log()
    selected_weight = sample_weight[selected]
    return weighted_mean(each[valid], selected_weight[valid])


def group_count_class_weights(
    hard_labels: torch.Tensor,
    group_ids: Sequence[str],
    classes: int,
    power: float,
) -> torch.Tensor:
    if power <= 0:
        return torch.ones(classes)
    groups_by_class = [set() for _ in range(classes)]
    for label, group_id in zip(hard_labels.tolist(), group_ids):
        groups_by_class[int(label)].add(group_id)
    counts = torch.tensor([max(len(groups), 1) for groups in groups_by_class], dtype=torch.float32)
    weights = (counts.sum() / (classes * counts)).pow(power)
    return weights / weights.mean()


def conditional_grade_logits(
    grade_logits: torch.Tensor, hard_species: torch.Tensor
) -> torch.Tensor:
    return grade_logits[torch.arange(len(hard_species), device=grade_logits.device), hard_species]


def peer_small_loss_mask(
    losses: torch.Tensor,
    eligible: torch.Tensor,
    final_classes: torch.Tensor,
    remember_rate: float,
) -> torch.Tensor:
    """Select low-loss samples independently inside each final class."""
    selected = torch.zeros_like(eligible)
    for class_id in range(4):
        indices = torch.nonzero(eligible & (final_classes == class_id), as_tuple=False).flatten()
        if not len(indices):
            continue
        keep = max(1, math.ceil(len(indices) * remember_rate))
        local = losses[indices]
        selected[indices[local.argsort()[:keep]]] = True
    return selected


def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)


def confidence_weight(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    scaled = ((confidence - threshold) / max(1 - threshold, 1e-6)).clamp(0, 1)
    return 0.5 + 0.5 * scaled


def cross_ema_fixmatch(
    student: HierarchicalHead,
    peer_teacher: HierarchicalHead,
    own_teacher: HierarchicalHead,
    weak_features: torch.Tensor,
    strong_features: torch.Tensor,
    hard_species: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Peer EMA proposes labels; both EMA teachers must agree before use."""
    with torch.no_grad():
        peer_species_logits, peer_grade_logits = peer_teacher(weak_features)
        own_species_logits, own_grade_logits = own_teacher(weak_features)
        peer_species_probability = peer_species_logits.softmax(1)
        own_species_probability = own_species_logits.softmax(1)
        peer_species_conf, peer_species_label = peer_species_probability.max(1)
        own_species_conf, own_species_label = own_species_probability.max(1)
        species_mask = (
            (peer_species_label == own_species_label)
            & (peer_species_conf >= threshold)
            & (own_species_conf >= threshold)
        )
        selected_peer_grade = conditional_grade_logits(peer_grade_logits, hard_species).softmax(1)
        selected_own_grade = conditional_grade_logits(own_grade_logits, hard_species).softmax(1)
        peer_grade_conf, peer_grade_label = selected_peer_grade.max(1)
        own_grade_conf, own_grade_label = selected_own_grade.max(1)
        grade_mask = (
            (peer_grade_label == own_grade_label)
            & (peer_grade_conf >= threshold)
            & (own_grade_conf >= threshold)
        )

    student_species_logits, student_grade_logits = student(strong_features)
    species_each = F.cross_entropy(
        student_species_logits, peer_species_label, reduction="none"
    )
    grade_each = F.cross_entropy(
        conditional_grade_logits(student_grade_logits, hard_species),
        peer_grade_label,
        reduction="none",
    )
    species_weights = species_mask.float() * confidence_weight(peer_species_conf, threshold)
    grade_weights = grade_mask.float() * confidence_weight(peer_grade_conf, threshold)
    species_loss = weighted_mean(species_each, species_weights) if species_mask.any() else species_each.sum() * 0
    grade_loss = weighted_mean(grade_each, grade_weights) if grade_mask.any() else grade_each.sum() * 0
    return 0.25 * species_loss + 0.75 * grade_loss, {
        "pseudo_species": float(species_mask.float().mean()),
        "pseudo_grade": float(grade_mask.float().mean()),
    }


def joint_probabilities(
    species_logits: torch.Tensor, grade_logits: torch.Tensor
) -> torch.Tensor:
    species = species_logits.softmax(1)
    grade = grade_logits.softmax(2)
    return (species.unsqueeze(2) * grade).reshape(-1, 4)


def predict_probabilities(
    models: Sequence[HierarchicalHead],
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for model in models:
        model = model.to(device).eval()
        view_outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for view in range(features.shape[1]):
                pieces = []
                for start in range(0, len(features), 512):
                    species_logits, grade_logits = model(
                        features[start : start + 512, view].to(device)
                    )
                    pieces.append(joint_probabilities(species_logits, grade_logits).cpu())
                view_outputs.append(torch.cat(pieces))
        outputs.append(torch.stack(view_outputs).mean(0))
    probability = torch.stack(outputs).mean(0)
    return probability / probability.sum(1, keepdim=True).clamp_min(1e-8)


def classification_metrics(
    hard_labels: torch.Tensor,
    probability: torch.Tensor,
    group_ids: Sequence[str],
) -> dict[str, object]:
    prediction = probability.argmax(1)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for truth, predicted in zip(hard_labels.tolist(), prediction.tolist()):
        confusion[truth, predicted] += 1
    per_class = []
    for class_id, name in enumerate(FINAL_NAMES):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[class_id].sum()),
            }
        )
    group_correct: dict[str, list[float]] = defaultdict(list)
    for group_id, correct in zip(group_ids, (prediction == hard_labels).tolist()):
        group_correct[group_id].append(float(correct))
    group_accuracies = {group: float(np.mean(values)) for group, values in group_correct.items()}
    confidence = probability.max(1).values
    correct = prediction == hard_labels
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-7)
        if mask.any():
            ece += float(
                mask.float().mean()
                * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    return {
        "samples": len(hard_labels),
        "groups": len(group_accuracies),
        "accuracy": float(correct.float().mean()),
        "species_accuracy": float(((prediction // 2) == (hard_labels // 2)).float().mean()),
        "grade_accuracy": float(((prediction % 2) == (hard_labels % 2)).float().mean()),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "macro_precision": float(np.mean([item["precision"] for item in per_class])),
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "group_accuracy": float(np.mean(list(group_accuracies.values()))),
        "group_accuracy_min": float(min(group_accuracies.values())),
        "nll": float(F.nll_loss(probability.clamp_min(1e-9).log(), hard_labels)),
        "ece": ece,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def selection_tuple(metrics: dict[str, object], rule: str) -> tuple[float, float, float]:
    macro_f1 = float(metrics["macro_f1"])
    group_accuracy = float(metrics["group_accuracy"])
    if rule == "macro_f1":
        primary = macro_f1
    elif rule == "group_accuracy":
        primary = group_accuracy
    else:
        primary = 0.5 * macro_f1 + 0.5 * group_accuracy
    return primary, float(metrics["accuracy"]), -float(metrics["nll"])


@dataclass
class CandidateResult:
    seed: int
    best_epoch: int
    validation: dict[str, object]
    member_validation: list[dict[str, object]]
    history: list[dict[str, object]]
    state_a: dict[str, torch.Tensor]
    state_b: dict[str, torch.Tensor]
    best_member: int


def train_candidate(
    seed: int,
    train_features: torch.Tensor,
    train_labels: LabelBundle,
    val_features: torch.Tensor,
    val_labels: LabelBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> CandidateResult:
    # Two genuinely independent random initializations; no copied historical head.
    student_a = initialize_head(train_features.shape[-1], args, seed).to(device)
    student_b = initialize_head(train_features.shape[-1], args, seed + 104_729).to(device)
    teacher_a = copy.deepcopy(student_a).eval()
    teacher_b = copy.deepcopy(student_b).eval()
    for parameter in teacher_a.parameters():
        parameter.requires_grad = False
    for parameter in teacher_b.parameters():
        parameter.requires_grad = False

    optimizer_a = torch.optim.AdamW(
        student_a.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_b = torch.optim.AdamW(
        student_b.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, T_max=args.epochs, eta_min=args.min_lr
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=args.epochs, eta_min=args.min_lr
    )

    batches_per_epoch = args.batches_per_epoch or math.ceil(len(train_features) / args.batch_size)
    sampler = ClassGroupBalancedBatches(
        train_labels.final_classes,
        train_labels.group_ids,
        args.batch_size,
        batches_per_epoch,
        seed,
    )
    species_class_weights = group_count_class_weights(
        train_labels.hard_species,
        train_labels.group_ids,
        2,
        args.species_class_weight_power,
    ).to(device)
    grade_class_weights = torch.stack(
        [
            group_count_class_weights(
                train_labels.hard_grade[train_labels.hard_species == species_id],
                [
                    group
                    for group, is_species in zip(
                        train_labels.group_ids,
                        (train_labels.hard_species == species_id).tolist(),
                    )
                    if is_species
                ],
                2,
                args.grade_class_weight_power,
            )
            for species_id in range(2)
        ]
    ).to(device)

    best_score: tuple[float, float, float] | None = None
    best_epoch = 0
    best_state_a = copy.deepcopy(teacher_a.state_dict())
    best_state_b = copy.deepcopy(teacher_b.state_dict())
    stale = 0
    history: list[dict[str, object]] = []
    strong_generator = torch.Generator().manual_seed(seed + 9001)

    for epoch in range(1, args.epochs + 1):
        student_a.train()
        student_b.train()
        forget_rate = args.forget_rate * min(1.0, epoch / max(args.forget_warmup_epochs, 1))
        remember_rate = 1.0 - forget_rate
        pseudo_threshold = (
            args.warmup_pseudo_threshold
            if epoch <= args.clean_warmup_epochs
            else args.pseudo_threshold
        )
        ssl_ramp = min(1.0, max(0.0, (epoch - 1) / max(args.ssl_ramp_epochs, 1)))
        totals: dict[str, float] = defaultdict(float)
        seen = 0

        for indices in sampler.iter_epoch(epoch):
            weak = train_features[indices, 0].to(device)
            if train_features.shape[1] > 1:
                strong_view = torch.randint(
                    1,
                    train_features.shape[1],
                    (len(indices),),
                    generator=strong_generator,
                )
                strong = train_features[indices, strong_view].to(device)
            else:
                strong = weak
            species_target = smooth_targets(
                train_labels.species_targets[indices].to(device),
                args.species_label_smoothing,
            )
            grade_target = smooth_targets(
                train_labels.grade_targets[indices].to(device),
                args.grade_label_smoothing,
            )
            hard_species = train_labels.hard_species[indices].to(device)
            hard_grade = train_labels.hard_grade[indices].to(device)
            final_classes = train_labels.final_classes[indices].to(device)
            species_sample_weight = train_labels.species_weights[indices].to(device)
            grade_sample_weight = train_labels.grade_weights[indices].to(device)
            high_consistency = train_labels.high_consistency[indices].to(device)

            species_logits_a, grade_logits_a, embedding_a = student_a.forward_with_embedding(weak)
            species_logits_b, grade_logits_b, embedding_b = student_b.forward_with_embedding(weak)
            selected_grade_a = conditional_grade_logits(grade_logits_a, hard_species)
            selected_grade_b = conditional_grade_logits(grade_logits_b, hard_species)
            species_each_a = soft_cross_entropy(species_logits_a, species_target)
            species_each_b = soft_cross_entropy(species_logits_b, species_target)
            grade_each_a = soft_cross_entropy(selected_grade_a, grade_target)
            grade_each_b = soft_cross_entropy(selected_grade_b, grade_target)

            species_weight = (
                species_sample_weight * species_class_weights[hard_species]
            )
            grade_weight = (
                grade_sample_weight * grade_class_weights[hard_species, hard_grade]
            )
            eligible = grade_weight > 0
            if epoch <= args.clean_warmup_epochs:
                eligible &= high_consistency
            with torch.no_grad():
                selected_by_a = peer_small_loss_mask(
                    grade_each_a.detach(), eligible, final_classes, remember_rate
                )
                selected_by_b = peer_small_loss_mask(
                    grade_each_b.detach(), eligible, final_classes, remember_rate
                )

            # True co-teaching: A is updated on B's clean subset and vice versa.
            species_loss_a = weighted_mean(species_each_a, species_weight)
            species_loss_b = weighted_mean(species_each_b, species_weight)
            if selected_by_b.any():
                grade_loss_a = weighted_mean(
                    grade_each_a[selected_by_b], grade_weight[selected_by_b]
                )
            else:
                grade_loss_a = grade_each_a.sum() * 0
            if selected_by_a.any():
                grade_loss_b = weighted_mean(
                    grade_each_b[selected_by_a], grade_weight[selected_by_a]
                )
            else:
                grade_loss_b = grade_each_b.sum() * 0

            contrastive_loss_a = conditional_grade_contrastive_loss(
                embedding_a,
                hard_species,
                hard_grade,
                eligible,
                grade_weight,
                args.contrastive_temperature,
            )
            contrastive_loss_b = conditional_grade_contrastive_loss(
                embedding_b,
                hard_species,
                hard_grade,
                eligible,
                grade_weight,
                args.contrastive_temperature,
            )

            ssl_loss_a, pseudo_a = cross_ema_fixmatch(
                student_a,
                teacher_b,
                teacher_a,
                weak,
                strong,
                hard_species,
                pseudo_threshold,
            )
            ssl_loss_b, pseudo_b = cross_ema_fixmatch(
                student_b,
                teacher_a,
                teacher_b,
                weak,
                strong,
                hard_species,
                pseudo_threshold,
            )
            loss_a = (
                args.species_loss_weight * species_loss_a
                + args.grade_loss_weight * grade_loss_a
                + args.ssl_weight * ssl_ramp * ssl_loss_a
                + args.contrastive_weight * contrastive_loss_a
            )
            loss_b = (
                args.species_loss_weight * species_loss_b
                + args.grade_loss_weight * grade_loss_b
                + args.ssl_weight * ssl_ramp * ssl_loss_b
                + args.contrastive_weight * contrastive_loss_b
            )

            optimizer_a.zero_grad(set_to_none=True)
            loss_a.backward()
            nn.utils.clip_grad_norm_(student_a.parameters(), args.grad_clip)
            optimizer_a.step()
            optimizer_b.zero_grad(set_to_none=True)
            loss_b.backward()
            nn.utils.clip_grad_norm_(student_b.parameters(), args.grad_clip)
            optimizer_b.step()
            ema_update(teacher_a, student_a, args.ema_decay)
            ema_update(teacher_b, student_b, args.ema_decay)

            count = len(indices)
            seen += count
            totals["loss_a"] += float(loss_a.detach()) * count
            totals["loss_b"] += float(loss_b.detach()) * count
            totals["species_loss"] += float(
                0.5 * (species_loss_a.detach() + species_loss_b.detach())
            ) * count
            totals["grade_loss"] += float(
                0.5 * (grade_loss_a.detach() + grade_loss_b.detach())
            ) * count
            totals["ssl_loss"] += float(
                0.5 * (ssl_loss_a.detach() + ssl_loss_b.detach())
            ) * count
            totals["contrastive_loss"] += float(
                0.5 * (contrastive_loss_a.detach() + contrastive_loss_b.detach())
            ) * count
            totals["selected_grade"] += float(
                0.5 * (selected_by_a.float().mean() + selected_by_b.float().mean())
            ) * count
            totals["pseudo_species"] += 0.5 * (
                pseudo_a["pseudo_species"] + pseudo_b["pseudo_species"]
            ) * count
            totals["pseudo_grade"] += 0.5 * (
                pseudo_a["pseudo_grade"] + pseudo_b["pseudo_grade"]
            ) * count

        scheduler_a.step()
        scheduler_b.step()
        validation_probability = predict_probabilities(
            [teacher_a, teacher_b], val_features, device
        )
        validation = classification_metrics(
            val_labels.final_classes, validation_probability, val_labels.group_ids
        )
        row: dict[str, object] = {
            "epoch": epoch,
            **{name: value / max(seen, 1) for name, value in totals.items()},
            "remember_rate": remember_rate,
            "pseudo_threshold": pseudo_threshold,
            "ssl_ramp": ssl_ramp,
            "lr": scheduler_a.get_last_lr()[0],
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
            best_epoch = epoch
            best_state_a = copy.deepcopy(teacher_a.state_dict())
            best_state_b = copy.deepcopy(teacher_b.state_dict())
            stale = 0
        else:
            stale += 1
        if not args.quiet and (epoch == 1 or epoch % 5 == 0):
            print(json.dumps({"seed": seed, **row}, ensure_ascii=False))
        if stale >= args.patience:
            if not args.quiet:
                print(f"seed={seed} early_stop={epoch} best_epoch={best_epoch}")
            break

    teacher_a.load_state_dict(best_state_a)
    teacher_b.load_state_dict(best_state_b)
    ensemble_probability = predict_probabilities([teacher_a, teacher_b], val_features, device)
    validation = classification_metrics(
        val_labels.final_classes, ensemble_probability, val_labels.group_ids
    )
    member_validation = []
    for teacher in (teacher_a, teacher_b):
        probability = predict_probabilities([teacher], val_features, device)
        member_validation.append(
            classification_metrics(
                val_labels.final_classes, probability, val_labels.group_ids
            )
        )
    best_member = max(
        range(2),
        key=lambda index: selection_tuple(member_validation[index], args.selection_metric),
    )
    return CandidateResult(
        seed=seed,
        best_epoch=best_epoch,
        validation=validation,
        member_validation=member_validation,
        history=history,
        state_a={name: value.cpu() for name, value in best_state_a.items()},
        state_b={name: value.cpu() for name, value in best_state_b.items()},
        best_member=best_member,
    )


def public_candidate(result: CandidateResult) -> dict[str, object]:
    return {
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "validation": result.validation,
        "member_validation": result.member_validation,
        "best_member": result.best_member,
        "history": result.history,
    }


def checkpoint_payload(
    result: CandidateResult,
    feature_dim: int,
    args: argparse.Namespace,
    train_labels: LabelBundle,
    val_labels: LabelBundle,
) -> dict[str, object]:
    member_states = [result.state_a, result.state_b]
    return {
        # Compatibility field for a single-head runtime. Prefer member_state_dicts ensemble.
        "head_state_dict": member_states[result.best_member],
        "member_state_dicts": member_states,
        "feature_dim": feature_dim,
        "species_names": SPECIES_NAMES,
        "grade_names": GRADE_NAMES,
        "final_names": FINAL_NAMES,
        "architecture": {
            "type": "hierarchical_shared_species_conditional_grade_v4",
            "hidden_dim": args.hidden_dim,
            "grade_hidden_dim": args.grade_hidden_dim,
            "dropout": args.dropout,
            "grade_dropout": args.grade_dropout,
            "decision": "argmax p(species) * p(grade|species)",
        },
        "recommended_inference": (
            "average joint probabilities from both member_state_dicts, then Bayes argmax"
        ),
        "strict_protocol": {
            "random_head_initialization": True,
            "historical_classification_head_loaded": False,
            "test_features_loaded": False,
            "test_labels_read": False,
            "selection_data": "validation only",
        },
        "training": {
            "seed": result.seed,
            "best_epoch": result.best_epoch,
            "co_teaching": "two independent students exchange class-stratified small-loss grade samples",
            "sampling": "class -> source_group -> sample balanced",
            "ssl": "cross-EMA FixMatch with dual-teacher agreement",
            "selection_metric": args.selection_metric,
            "train_label_audit": train_labels.audit_summary,
            "validation_label_audit": val_labels.audit_summary,
        },
        "validation_metrics": result.validation,
        "member_validation_metrics": result.member_validation,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.forget_rate < 1:
        raise ValueError("--forget-rate must be in [0, 1)")
    if not 0 < args.pseudo_threshold <= args.warmup_pseudo_threshold <= 1:
        raise ValueError("Require 0 < pseudo_threshold <= warmup_pseudo_threshold <= 1")
    for path in (*args.train_features, *args.val_features):
        reject_test_feature_path(path)
    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    rows = read_train_val_manifest(args.manifest)
    if len(args.train_features) != len(args.val_features):
        raise ValueError("Train and validation must provide the same feature families")
    train_features = load_feature_caches(args.train_features, rows["train"])
    val_features = load_feature_caches(args.val_features, rows["val"])
    if train_features.shape[-1] != val_features.shape[-1]:
        raise ValueError("Train and validation feature dimensions differ")
    if train_features.shape[1] < 2 and args.ssl_weight > 0:
        print("warning: train cache has one view; FixMatch strong view equals weak view")
    decisions = read_label_decisions(
        args.label_decisions, {row["path"] for row in rows["train"]}
    )
    train_labels = build_labels(
        rows["train"], decisions, args.use_provided_soft_labels, allow_overrides=True
    )
    # Validation is immutable: no decision rows and no soft-label override.
    val_labels = build_labels(rows["val"], {}, False, allow_overrides=False)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "strict_protocol": "NO TEST DATA WILL BE READ",
                "device": str(device),
                "train": len(rows["train"]),
                "val": len(rows["val"]),
                "train_groups": len(set(train_labels.group_ids)),
                "val_groups": len(set(val_labels.group_ids)),
                "feature_shape_train": list(train_features.shape),
                "feature_shape_val": list(val_features.shape),
                "seeds": seeds,
                "train_label_audit": train_labels.audit_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    candidates: list[CandidateResult] = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        result = train_candidate(
            seed,
            train_features,
            train_labels,
            val_features,
            val_labels,
            args,
            device,
        )
        candidates.append(result)
        payload = checkpoint_payload(
            result, train_features.shape[-1], args, train_labels, val_labels
        )
        torch.save(payload, output / f"candidate_seed_{seed}.pt")
        print(
            json.dumps(
                {
                    "seed_complete": seed,
                    "best_epoch": result.best_epoch,
                    "validation": result.validation,
                },
                ensure_ascii=False,
            )
        )

    best = max(
        candidates,
        key=lambda item: selection_tuple(item.validation, args.selection_metric),
    )
    best_payload = checkpoint_payload(
        best, train_features.shape[-1], args, train_labels, val_labels
    )
    best_path = output / "best_hierarchical_v4_strict.pt"
    torch.save(best_payload, best_path)
    report = {
        "strict_protocol": best_payload["strict_protocol"],
        "manifest": str(args.manifest.resolve()),
        "train_features": [str(path.resolve()) for path in args.train_features],
        "val_features": [str(path.resolve()) for path in args.val_features],
        "test_features": None,
        "test_metrics": None,
        "selection_rule": f"{args.selection_metric} on validation only",
        "configuration": {
            key: (
                [str(item) if isinstance(item, Path) else item for item in value]
                if isinstance(value, list)
                else str(value) if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "train_label_audit": train_labels.audit_summary,
        "validation_label_audit": val_labels.audit_summary,
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
