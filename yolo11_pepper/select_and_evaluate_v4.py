#!/usr/bin/env python3
"""Leakage-safe v4 ensemble selection, calibration, and one-shot test evaluation.

There are deliberately two disjoint modes:

Selection mode (the default)
    Accepts physically separate train/validation manifests, train/validation
    feature caches, and v4 head checkpoints.  It validates the training inputs,
    predicts validation only, selects a small finite ensemble by the validation
    balanced score, then fits three validation-only temperatures: one for
    p(species) and one for each p(grade | species) branch.  The frozen decision
    is written to ``selection.json``.

Final-test mode (requires ``--final-test``)
    Loads only a previously frozen selection plus explicitly supplied test
    manifest/features.  It never performs model selection or temperature
    fitting.  A receipt next to the selection prevents accidentally evaluating
    the same frozen selection more than once.

The feature transformation exactly matches ``train_hierarchical_v4.py``:
each backbone block is independently L2-normalized and multiplied by
sqrt(block_dim), then the blocks are concatenated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SPECIES_NAMES = ("子弹头", "条子")
GRADE_NAMES = ("一级", "二级")
FINAL_NAMES = ("子弹头_一级", "子弹头_二级", "条子_一级", "条子_二级")
SELECTION_SCHEMA = "pepper-v4-ensemble-selection-v1"
SVM_SELECTION_SCHEMA = "pepper-v4-svm-selection-v1"
REPORT_SCHEMA = "pepper-v4-combined-final-test-report-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict validation-only v4 selection, or explicit one-shot final test."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--final-test",
        action="store_true",
        help="Enable the separate frozen-selection final-test stage.",
    )
    mode.add_argument(
        "--calibrate-svm",
        action="store_true",
        help="Calibrate the already-frozen hierarchical SVM on validation only.",
    )
    parser.add_argument("--device", default=None)

    # Selection-only inputs have no parser defaults.  Defaults are assigned
    # inside run_selection(), so final-test mode cannot accidentally touch them.
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--train-features", type=Path, nargs="+")
    parser.add_argument("--val-features", type=Path, nargs="+")
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--svm-model", type=Path)
    parser.add_argument("--svm-selection-output", type=Path)
    parser.add_argument("--reference-neural-selection", type=Path)
    parser.add_argument(
        "--weight-grid",
        default="0.25,0.50,0.75",
        help="Weight assigned to the first checkpoint in every two-checkpoint ensemble.",
    )
    parser.add_argument(
        "--temperature-bounds",
        default="0.05,10.0",
        help="Inclusive lower,upper bounds for validation temperature fitting.",
    )

    # Final-test-only inputs are mandatory only after --final-test is present.
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--svm-selection", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--test-features", type=Path, nargs="+")
    parser.add_argument("--test-report", type=Path)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any, *, refuse_existing: bool = True) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing and path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def path_mentions_test(path: Path) -> bool:
    for part in path.parts:
        tokens = [token for token in re.split(r"[^a-z0-9]+", part.lower()) if token]
        if "test" in tokens:
            return True
    return False


def reject_test_path(path: Path, label: str) -> None:
    if path_mentions_test(path):
        raise ValueError(f"STRICT VALIDATION-ONLY protocol rejected {label}: {path}")


def parse_float_list(value: str, label: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid {label}: {value}") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"Invalid {label}: {value}")
    return result


def canonical_class_name(row: dict[str, str], class_id: int) -> str:
    name = (row.get("class_name") or row.get("label") or "").strip()
    return name or FINAL_NAMES[class_id]


def read_single_split_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    """Require a physically split-specific manifest; do not silently filter rows."""
    path = path.resolve()
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), 2):
            split = (raw.get("split") or "").strip().lower()
            if split != expected_split:
                raise ValueError(
                    f"{path}:{line_number} has split={split!r}; expected only {expected_split!r}"
                )
            class_id = int(raw["class_id"])
            if not 0 <= class_id < 4:
                raise ValueError(f"{path}:{line_number} invalid class_id={class_id}")
            image_path = str(Path(raw["path"]).resolve())
            group_id = (raw.get("group_id") or "").strip()
            if not group_id:
                raise ValueError(f"{path}:{line_number} missing group_id")
            rows.append(
                {
                    **raw,
                    "path": image_path,
                    "split": split,
                    "class_id": str(class_id),
                    "class_name": canonical_class_name(raw, class_id),
                    "group_id": group_id,
                    "origin": (raw.get("origin") or "unknown").strip() or "unknown",
                    "source_id": (raw.get("source_id") or "").strip(),
                    "pair_id": (raw.get("pair_id") or image_path).strip(),
                    "content_sha256": (raw.get("content_sha256") or "").strip(),
                }
            )
    if not rows:
        raise ValueError(f"Empty {expected_split} manifest: {path}")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate image paths in {path}")
    return rows


def values_cross_split(
    first: Sequence[dict[str, str]],
    second: Sequence[dict[str, str]],
    field: str,
    *,
    ignore_empty: bool = True,
) -> list[str]:
    left = {row.get(field, "") for row in first}
    right = {row.get(field, "") for row in second}
    overlap = left & right
    if ignore_empty:
        overlap.discard("")
    return sorted(overlap)


def leakage_audit(
    train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]
) -> dict[str, Any]:
    audit = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "group_cross_split": values_cross_split(train_rows, val_rows, "group_id"),
        "source_cross_split": values_cross_split(train_rows, val_rows, "source_id"),
        "pair_cross_split": values_cross_split(train_rows, val_rows, "pair_id"),
        "content_hash_cross_split": values_cross_split(
            train_rows, val_rows, "content_sha256"
        ),
    }
    audit["passed"] = not any(
        audit[name]
        for name in (
            "group_cross_split",
            "source_cross_split",
            "pair_cross_split",
            "content_hash_cross_split",
        )
    )
    return audit


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    kind: str
    dim: int
    checkpoint: str

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dim": self.dim,
            "checkpoint": self.checkpoint,
            "normalization": "per-vector L2 * sqrt(family_dim)",
        }


@dataclass
class LoadedFeatures:
    tensor: torch.Tensor
    families: list[FeatureFamily]
    files: list[dict[str, Any]]


def load_feature_families(
    paths: Sequence[Path],
    rows: Sequence[dict[str, str]],
    expected_split: str,
    *,
    validation_only: bool,
) -> LoadedFeatures:
    if not paths:
        raise ValueError(f"No feature caches supplied for {expected_split}")
    blocks: list[torch.Tensor] = []
    families: list[FeatureFamily] = []
    files: list[dict[str, Any]] = []
    expected_paths = [row["path"] for row in rows]
    expected_classes = torch.tensor([int(row["class_id"]) for row in rows])
    for raw_path in paths:
        path = raw_path.resolve()
        if validation_only:
            reject_test_path(path, f"{expected_split} feature cache")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "features" not in payload:
            raise TypeError(f"Feature cache must be a dict with 'features': {path}")
        block = payload["features"]
        if not isinstance(block, torch.Tensor):
            raise TypeError(f"Feature tensor missing from {path}")
        if block.ndim == 2:
            block = block.unsqueeze(1)
        if block.ndim != 3 or len(block) != len(rows):
            raise ValueError(
                f"Feature shape/manifest mismatch for {path}: {tuple(block.shape)} vs {len(rows)} rows"
            )
        if not torch.isfinite(block).all():
            raise ValueError(f"NaN/Inf in feature cache: {path}")
        cached_paths = payload.get("paths")
        if cached_paths is None:
            raise ValueError(f"Strict path-order verification requires payload['paths']: {path}")
        normalized_paths = [str(Path(item).resolve()) for item in cached_paths]
        if normalized_paths != expected_paths:
            raise ValueError(f"Feature path order does not match {expected_split} manifest: {path}")
        cached_classes = payload.get("class_ids")
        if isinstance(cached_classes, torch.Tensor) and not torch.equal(
            cached_classes.cpu().long(), expected_classes
        ):
            raise ValueError(f"Feature class_ids do not match manifest: {path}")
        metadata = payload.get("metadata") or {}
        metadata_split = str(metadata.get("split", expected_split)).strip().lower()
        if metadata_split != expected_split:
            raise ValueError(
                f"Feature metadata split={metadata_split!r}, expected {expected_split!r}: {path}"
            )
        if validation_only and bool(metadata.get("test_requested_explicitly", False)):
            raise ValueError(f"Selection refuses a cache extracted in a test-requesting run: {path}")
        family = FeatureFamily(
            name=str(metadata.get("backbone_name") or path.stem.rsplit("_", 1)[0]),
            kind=str(metadata.get("kind") or "unknown"),
            dim=int(block.shape[-1]),
            checkpoint=str(metadata.get("checkpoint") or ""),
        )
        if family.name in {item.name for item in families}:
            raise ValueError(f"Duplicate feature family {family.name!r}")
        block = block.float().contiguous()
        normalized = F.normalize(block, dim=-1) * math.sqrt(block.shape[-1])
        blocks.append(normalized)
        families.append(family)
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": list(block.shape),
                "family": family.public(),
            }
        )
    view_counts = {block.shape[1] for block in blocks}
    if len(view_counts) != 1:
        raise ValueError(f"Feature families have inconsistent view counts: {sorted(view_counts)}")
    return LoadedFeatures(torch.cat(blocks, dim=-1).contiguous(), families, files)


def compare_feature_families(
    expected: Sequence[FeatureFamily], actual: Sequence[FeatureFamily], label: str
) -> None:
    expected_signature = [(item.name, item.kind, item.dim) for item in expected]
    actual_signature = [(item.name, item.kind, item.dim) for item in actual]
    if actual_signature != expected_signature:
        raise ValueError(
            f"{label} feature families do not match: {actual_signature} != {expected_signature}"
        )


class HierarchicalHead(nn.Module):
    """Current v4 p(species), p(grade | species) classification head."""

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


@dataclass
class CheckpointPrediction:
    checkpoint_id: str
    path: Path
    sha256: str
    seed: int | None
    member_count: int
    architecture: dict[str, Any]
    species_probability: torch.Tensor
    grade_probability: torch.Tensor
    member_metrics: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "seed": self.seed,
            "member_count": self.member_count,
            "architecture": self.architecture,
            "member_validation_metrics": self.member_metrics,
        }


def checkpoint_identity(path: Path, digest: str) -> str:
    return f"{path.parent.name}/{path.stem}:{digest[:12]}"


def predict_head_probabilities(
    head: HierarchicalHead,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    head = head.to(device).eval()
    joint_by_view: list[torch.Tensor] = []
    with torch.inference_mode():
        for view in range(features.shape[1]):
            joint_parts: list[torch.Tensor] = []
            for start in range(0, len(features), batch_size):
                species_logits, grade_logits = head(
                    features[start : start + batch_size, view].to(device)
                )
                species = species_logits.softmax(1)
                grade = grade_logits.softmax(2)
                joint_parts.append((species.unsqueeze(2) * grade).cpu())
            joint_by_view.append(torch.cat(joint_parts))
    # Match the v4 training evaluator exactly: average each member/view's joint
    # distribution, then refactor that mixture into p(s) and p(g|s).  Refactoring
    # preserves the averaged joint while still permitting three branch-specific
    # temperatures.
    joint = torch.stack(joint_by_view).mean(0)
    species = joint.sum(2)
    grade = joint / species.unsqueeze(2).clamp_min(1e-12)
    return species, grade


def load_checkpoint_prediction(
    path: Path,
    features: torch.Tensor,
    rows: Sequence[dict[str, str]],
    device: torch.device,
) -> CheckpointPrediction:
    path = path.resolve()
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict: {path}")
    feature_dim = int(payload.get("feature_dim", -1))
    if feature_dim != features.shape[-1]:
        raise ValueError(
            f"Checkpoint feature_dim={feature_dim} does not match {features.shape[-1]}: {path}"
        )
    architecture = dict(payload.get("architecture") or {})
    if architecture.get("type") != "hierarchical_shared_species_conditional_grade_v4":
        raise ValueError(f"Unsupported checkpoint architecture in {path}: {architecture}")
    state_dicts = payload.get("member_state_dicts")
    if not isinstance(state_dicts, (list, tuple)) or len(state_dicts) < 2:
        raise ValueError(f"v4 ensemble selection requires >=2 member_state_dicts: {path}")
    member_species: list[torch.Tensor] = []
    member_grade: list[torch.Tensor] = []
    member_metrics: list[dict[str, Any]] = []
    for state in state_dicts:
        head = HierarchicalHead(
            feature_dim,
            int(architecture["hidden_dim"]),
            int(architecture["grade_hidden_dim"]),
            float(architecture.get("dropout", 0.0)),
            float(architecture.get("grade_dropout", 0.0)),
        )
        head.load_state_dict(state, strict=True)
        species_probability, grade_probability = predict_head_probabilities(
            head, features, device
        )
        member_species.append(species_probability)
        member_grade.append(grade_probability)
        member_metrics.append(
            metrics_from_branches(rows, species_probability, grade_probability)
        )
        del head
    member_joint = torch.stack(
        [
            joint_probability(species, grade).reshape(-1, 2, 2)
            for species, grade in zip(member_species, member_grade)
        ]
    ).mean(0)
    checkpoint_species = member_joint.sum(2)
    checkpoint_grade = member_joint / checkpoint_species.unsqueeze(2).clamp_min(1e-12)
    return CheckpointPrediction(
        checkpoint_id=checkpoint_identity(path, digest),
        path=path,
        sha256=digest,
        seed=(
            int(payload["training"]["seed"])
            if isinstance(payload.get("training"), dict)
            and payload["training"].get("seed") is not None
            else None
        ),
        member_count=len(state_dicts),
        architecture=architecture,
        species_probability=checkpoint_species,
        grade_probability=checkpoint_grade,
        member_metrics=member_metrics,
    )


def mix_branches(
    predictions: Sequence[CheckpointPrediction],
    components: Sequence[tuple[int, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    total = sum(weight for _, weight in components)
    if total <= 0:
        raise ValueError("Ensemble weights must have positive mass")
    joint = sum(
        joint_probability(
            predictions[index].species_probability,
            predictions[index].grade_probability,
        ).reshape(-1, 2, 2)
        * (weight / total)
        for index, weight in components
    )
    joint = joint / joint.sum((1, 2), keepdim=True).clamp_min(1e-12)
    species = joint.sum(2)
    grade = joint / species.unsqueeze(2).clamp_min(1e-12)
    return species, grade


def joint_probability(
    species_probability: torch.Tensor, grade_probability: torch.Tensor
) -> torch.Tensor:
    return (species_probability.unsqueeze(2) * grade_probability).reshape(-1, 4)


def ece_score(
    probability: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> float:
    confidence, prediction = probability.max(1)
    correct = prediction.eq(labels)
    ece = probability.new_tensor(0.0)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if mask.any():
            ece += mask.float().mean() * (
                correct[mask].float().mean() - confidence[mask].mean()
            ).abs()
    return float(ece)


def confusion_matrix(labels: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for truth, predicted in zip(labels.tolist(), prediction.tolist()):
        confusion[int(truth), int(predicted)] += 1
    return confusion


def per_class_metrics(confusion: torch.Tensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(FINAL_NAMES):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - true_positive
        false_negative = int(confusion[class_id].sum()) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        result.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[class_id].sum()),
            }
        )
    return result


def metrics_from_branches(
    rows: Sequence[dict[str, str]],
    species_probability: torch.Tensor,
    grade_probability: torch.Tensor,
    *,
    include_per_class: bool = True,
) -> dict[str, Any]:
    labels = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    species_labels = labels // 2
    grade_labels = labels % 2
    probability = joint_probability(species_probability, grade_probability)
    prediction = probability.argmax(1)
    species_prediction = species_probability.argmax(1)
    conditional_grade = grade_probability[
        torch.arange(len(labels)), species_labels
    ].argmax(1)
    confusion = confusion_matrix(labels, prediction)
    per_class = per_class_metrics(confusion)
    group_correct: dict[str, list[float]] = defaultdict(list)
    for row, correct in zip(rows, prediction.eq(labels).tolist()):
        group_correct[row["group_id"]].append(float(correct))
    group_accuracies = [float(np.mean(values)) for values in group_correct.values()]
    macro_f1 = float(np.mean([item["f1"] for item in per_class]))
    group_accuracy = float(np.mean(group_accuracies))
    result: dict[str, Any] = {
        "samples": len(rows),
        "groups": len(group_correct),
        "species_accuracy": float(species_prediction.eq(species_labels).float().mean()),
        "conditional_grade_accuracy": float(
            conditional_grade.eq(grade_labels).float().mean()
        ),
        "joint_accuracy": float(prediction.eq(labels).float().mean()),
        "joint_macro_f1": macro_f1,
        "group_joint_accuracy": group_accuracy,
        "balanced_selection_score": 0.5 * (macro_f1 + group_accuracy),
        "joint_nll": float(F.nll_loss(probability.clamp_min(1e-12).log(), labels)),
        "joint_ece_15bin": ece_score(probability, labels),
        "confusion": confusion.tolist(),
    }
    if include_per_class:
        result["per_class"] = per_class
    return result


def selection_key(metrics: dict[str, Any], component_count: int) -> tuple[float, ...]:
    return (
        float(metrics["balanced_selection_score"]),
        float(metrics["joint_accuracy"]),
        -float(metrics["joint_nll"]),
        -float(component_count),
    )


def fit_scalar_temperature(
    probability: torch.Tensor,
    labels: torch.Tensor,
    lower: float,
    upper: float,
    iterations: int = 100,
) -> tuple[float, float]:
    """Bounded golden-section minimization of NLL in log-temperature space."""
    if len(probability) == 0:
        raise ValueError("Cannot fit a temperature on an empty branch")
    log_probability = probability.double().clamp_min(1e-15).log()
    labels = labels.long()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        log_scaled = F.log_softmax(log_probability / temperature, dim=1)
        return float(F.nll_loss(log_scaled, labels))

    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value, second_value = objective(first), objective(second)
    for _ in range(iterations):
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = objective(first)
        else:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = objective(second)
    candidates = [
        (left, objective(left)),
        (right, objective(right)),
        (first, first_value),
        (second, second_value),
        (0.0, objective(0.0)) if lower <= 1.0 <= upper else (first, first_value),
    ]
    best_log_temperature, best_nll = min(candidates, key=lambda item: item[1])
    return math.exp(best_log_temperature), best_nll


def fit_conditional_temperatures(
    rows: Sequence[dict[str, str]],
    species_probability: torch.Tensor,
    grade_probability: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[list[float], dict[str, float]]:
    labels = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    species_labels, grade_labels = labels // 2, labels % 2
    species_temperature, species_nll = fit_scalar_temperature(
        species_probability, species_labels, lower, upper
    )
    temperatures = [species_temperature]
    objectives = {"species_nll": species_nll}
    for species, name in enumerate(SPECIES_NAMES):
        mask = species_labels == species
        temperature, nll = fit_scalar_temperature(
            grade_probability[mask, species], grade_labels[mask], lower, upper
        )
        temperatures.append(temperature)
        objectives[f"grade_given_{name}_nll"] = nll
    return temperatures, objectives


def apply_temperatures(
    species_probability: torch.Tensor,
    grade_probability: torch.Tensor,
    temperatures: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(temperatures) != 3 or any(value <= 0 for value in temperatures):
        raise ValueError(f"Expected three positive temperatures, got {temperatures}")
    species = F.softmax(
        species_probability.clamp_min(1e-15).log() / float(temperatures[0]), dim=1
    )
    grade = torch.empty_like(grade_probability)
    for species_id in range(2):
        grade[:, species_id] = F.softmax(
            grade_probability[:, species_id].clamp_min(1e-15).log()
            / float(temperatures[species_id + 1]),
            dim=1,
        )
    return species, grade


def enumerate_ensemble_specs(
    predictions: Sequence[CheckpointPrediction], grid: Sequence[float]
) -> list[dict[str, Any]]:
    for value in grid:
        if not 0 < value < 1:
            raise ValueError("--weight-grid values must be strictly between 0 and 1")
    specifications: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        specifications.append(
            {
                "ensemble_id": f"dual_member::{prediction.checkpoint_id}",
                "components": [(index, 1.0)],
                "kind": "checkpoint_dual_member_equal",
                "cross_seed": False,
            }
        )
    for first in range(len(predictions)):
        for second in range(first + 1, len(predictions)):
            cross_seed = (
                predictions[first].seed is not None
                and predictions[second].seed is not None
                and predictions[first].seed != predictions[second].seed
            )
            for weight in sorted(set(grid)):
                specifications.append(
                    {
                        "ensemble_id": (
                            f"pair::{predictions[first].checkpoint_id}@{weight:.3f}::"
                            f"{predictions[second].checkpoint_id}@{1-weight:.3f}"
                        ),
                        "components": [(first, weight), (second, 1.0 - weight)],
                        "kind": "two_checkpoint_weight_grid",
                        "cross_seed": cross_seed,
                    }
                )
    if len(predictions) >= 3:
        equal = 1.0 / len(predictions)
        specifications.append(
            {
                "ensemble_id": "all_checkpoints_equal",
                "components": [(index, equal) for index in range(len(predictions))],
                "kind": "all_checkpoint_equal",
                "cross_seed": len({item.seed for item in predictions}) > 1,
            }
        )
    return specifications


def public_components(
    predictions: Sequence[CheckpointPrediction], components: Sequence[tuple[int, float]]
) -> list[dict[str, Any]]:
    total = sum(weight for _, weight in components)
    return [
        {
            "checkpoint_id": predictions[index].checkpoint_id,
            "path": str(predictions[index].path),
            "sha256": predictions[index].sha256,
            "seed": predictions[index].seed,
            "checkpoint_weight": weight / total,
            "member_policy": "equal",
            "member_count": predictions[index].member_count,
        }
        for index, weight in components
    ]


def input_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def default_selection_inputs(args: argparse.Namespace) -> None:
    base = Path("yolo11_pepper")
    args.train_manifest = args.train_manifest or base / "datasets/pepper_ssl_v4_merged/train_manifest.csv"
    args.val_manifest = args.val_manifest or base / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
    args.train_features = args.train_features or [
        base / "runs/hierarchical_v4/features_merged/pepper_det_train.pt",
        base / "runs/hierarchical_v4/features_merged/imagenet_cls_train.pt",
    ]
    args.val_features = args.val_features or [
        base / "runs/hierarchical_v4/features/pepper_det_val.pt",
        base / "runs/hierarchical_v4/features/imagenet_cls_val.pt",
    ]
    args.selection_output = args.selection_output or base / "runs/hierarchical_v4/v4_selection/selection.json"


def reject_final_only_args_during_selection(args: argparse.Namespace) -> None:
    supplied = [
        name
        for name in (
            "selection",
            "svm_selection",
            "test_manifest",
            "test_features",
            "test_report",
            "svm_model",
            "svm_selection_output",
            "reference_neural_selection",
        )
        if getattr(args, name) is not None
    ]
    if supplied:
        raise ValueError(
            f"Selection mode refuses final-test arguments: {', '.join(supplied)}"
        )


def run_selection(args: argparse.Namespace) -> None:
    reject_final_only_args_during_selection(args)
    default_selection_inputs(args)
    if not args.checkpoints:
        raise ValueError("Selection mode requires --checkpoints with at least two v4 checkpoints")
    if len(args.checkpoints) < 2:
        raise ValueError("Use at least two checkpoints for cross-checkpoint ensemble selection")
    for path, label in (
        (args.train_manifest, "train manifest"),
        (args.val_manifest, "validation manifest"),
        (args.selection_output, "selection output"),
    ):
        reject_test_path(path, label)
    for path in (*args.train_features, *args.val_features, *args.checkpoints):
        reject_test_path(path, "selection input")

    train_rows = read_single_split_manifest(args.train_manifest, "train")
    val_rows = read_single_split_manifest(args.val_manifest, "val")
    audit = leakage_audit(train_rows, val_rows)
    if not audit["passed"]:
        raise ValueError(f"Train/validation leakage audit failed: {audit}")

    # Training features are loaded for schema/order/provenance validation only.
    train_features = load_feature_families(
        args.train_features, train_rows, "train", validation_only=True
    )
    val_features = load_feature_families(
        args.val_features, val_rows, "val", validation_only=True
    )
    compare_feature_families(train_features.families, val_features.families, "validation")
    if train_features.tensor.shape[-1] != val_features.tensor.shape[-1]:
        raise ValueError("Train and validation combined feature dimensions differ")
    del train_features.tensor

    checkpoint_paths = [path.resolve() for path in args.checkpoints]
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("Duplicate checkpoint paths supplied")
    device = choose_device(args.device)
    predictions: list[CheckpointPrediction] = []
    seen_digests: set[str] = set()
    for path in checkpoint_paths:
        prediction = load_checkpoint_prediction(
            path, val_features.tensor, val_rows, device
        )
        if prediction.sha256 in seen_digests:
            raise ValueError(f"Duplicate checkpoint content supplied: {path}")
        seen_digests.add(prediction.sha256)
        predictions.append(prediction)
        print(
            json.dumps(
                {
                    "checkpoint": prediction.checkpoint_id,
                    "seed": prediction.seed,
                    "dual_member_validation": metrics_from_branches(
                        val_rows,
                        prediction.species_probability,
                        prediction.grade_probability,
                    ),
                },
                ensure_ascii=False,
            )
        )

    weight_grid = parse_float_list(args.weight_grid, "--weight-grid")
    specifications = enumerate_ensemble_specs(predictions, weight_grid)
    candidate_reports: list[dict[str, Any]] = []
    best_specification: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for specification in specifications:
        species_probability, grade_probability = mix_branches(
            predictions, specification["components"]
        )
        metrics = metrics_from_branches(val_rows, species_probability, grade_probability)
        report = {
            "ensemble_id": specification["ensemble_id"],
            "kind": specification["kind"],
            "cross_seed": specification["cross_seed"],
            "components": public_components(predictions, specification["components"]),
            "validation_metrics_raw": metrics,
        }
        candidate_reports.append(report)
        if best_metrics is None or selection_key(
            metrics, len(specification["components"])
        ) > selection_key(best_metrics, len(best_specification["components"])):
            best_specification = specification
            best_metrics = metrics
    assert best_specification is not None and best_metrics is not None

    selected_species, selected_grade = mix_branches(
        predictions, best_specification["components"]
    )
    bounds = parse_float_list(args.temperature_bounds, "--temperature-bounds")
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
        raise ValueError("--temperature-bounds must be two positive increasing values")
    temperatures, temperature_objectives = fit_conditional_temperatures(
        val_rows, selected_species, selected_grade, bounds[0], bounds[1]
    )
    calibrated_species, calibrated_grade = apply_temperatures(
        selected_species, selected_grade, temperatures
    )
    calibrated_metrics = metrics_from_branches(
        val_rows, calibrated_species, calibrated_grade
    )

    selected_components = public_components(
        predictions, best_specification["components"]
    )
    frozen_core = {
        "ensemble_id": best_specification["ensemble_id"],
        "components": selected_components,
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
        "feature_families": [item.public() for item in val_features.families],
        "combined_feature_dim": int(val_features.tensor.shape[-1]),
    }
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at_utc": utc_now(),
        "selection_id": f"pepper-v4-selection-{sha256_json(frozen_core)[:20]}",
        "protocol": {
            "stage": "validation_selection_and_temperature_calibration",
            "test_arguments_accepted": False,
            "test_manifest_read": False,
            "test_features_read": False,
            "test_metrics_computed": False,
            "selection_metric": (
                "0.5 * validation joint macro-F1 + 0.5 * validation source-group mean joint accuracy"
            ),
            "temperature_fit_data": "validation only",
            "temperature_fit_targets": (
                "species plus true-species conditional grade branch"
            ),
            "final_test_requires_explicit_flag": "--final-test",
        },
        "input_fingerprints": {
            "train_manifest": input_record(args.train_manifest),
            "val_manifest": input_record(args.val_manifest),
            "train_features": train_features.files,
            "val_features": val_features.files,
        },
        "leakage_audit": audit,
        "feature_schema": {
            "families": [item.public() for item in val_features.families],
            "combined_dim": int(val_features.tensor.shape[-1]),
            "concatenation_order": [item.name for item in val_features.families],
        },
        "checkpoint_inventory": [prediction.public() for prediction in predictions],
        "enumeration": {
            "checkpoint_member_policy": "equal average over all member_state_dicts",
            "single_checkpoint_candidates": len(predictions),
            "pair_weight_grid": sorted(set(weight_grid)),
            "all_checkpoint_equal_candidate": len(predictions) >= 3,
            "candidate_count": len(candidate_reports),
        },
        "candidates": candidate_reports,
        "selected": {
            **frozen_core,
            "kind": best_specification["kind"],
            "cross_seed": best_specification["cross_seed"],
            "validation_metrics_raw": best_metrics,
            "temperature_fit_objectives": temperature_objectives,
            "validation_metrics_calibrated": calibrated_metrics,
        },
        "final_test_status": "not_run",
    }
    atomic_write_json(args.selection_output, selection, refuse_existing=True)
    print(
        json.dumps(
            {
                "selection_saved": str(args.selection_output.resolve()),
                "selection_id": selection["selection_id"],
                "candidate_count": len(candidate_reports),
                "selected_ensemble": best_specification["ensemble_id"],
                "temperatures": selection["selected"]["temperatures"],
                "validation_raw": best_metrics,
                "validation_calibrated": calibrated_metrics,
                "strict_protocol": "NO TEST MANIFEST, FEATURE, LABEL, OR METRIC WAS READ",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def svm_family_schema(families: Sequence[FeatureFamily]) -> list[dict[str, Any]]:
    return [
        {
            "name": family.name,
            "kind": family.kind,
            "dim": family.dim,
            "checkpoint": family.checkpoint,
            "normalization": "per-vector L2 only",
        }
        for family in families
    ]


def svm_feature_matrix(
    features: LoadedFeatures, view_mode: str
) -> np.ndarray:
    """Recover the exact feature scaling used by benchmark_v4_classifiers.py."""
    blocks: list[torch.Tensor] = []
    offset = 0
    for family in features.families:
        block = features.tensor[:, :, offset : offset + family.dim]
        # load_feature_families uses L2 * sqrt(dim), while the frozen SVM was
        # fit on L2-only blocks.  Undo only that family-specific multiplier.
        block = block / math.sqrt(family.dim)
        if view_mode == "canonical":
            block = block[:, 0]
        elif view_mode == "view_mean":
            block = block.mean(1)
        else:
            raise ValueError(f"Unsupported frozen SVM view_mode={view_mode!r}")
        blocks.append(block)
        offset += family.dim
    if offset != features.tensor.shape[-1]:
        raise ValueError("SVM feature-family dimensions do not cover the tensor")
    return torch.cat(blocks, dim=1).cpu().numpy()


def ordered_binary_probability(model: Any, matrix: np.ndarray) -> torch.Tensor:
    probability = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    if probability.ndim != 2 or probability.shape[1] != len(classes):
        raise ValueError("Frozen binary estimator returned an invalid probability shape")
    if set(classes) != {0, 1}:
        raise ValueError(f"Frozen binary estimator classes must be [0,1], got {classes}")
    ordered = np.empty((len(matrix), 2), dtype=np.float64)
    for source_column, class_id in enumerate(classes):
        ordered[:, class_id] = probability[:, source_column]
    ordered /= np.clip(ordered.sum(1, keepdims=True), 1e-15, None)
    return torch.from_numpy(ordered).float()


def predict_svm_branches(
    payload: dict[str, Any], features: LoadedFeatures
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_names = [str(item) for item in payload.get("feature_families", ())]
    actual_names = [family.name for family in features.families]
    if expected_names != actual_names:
        raise ValueError(
            f"Frozen SVM feature families do not match: {expected_names} != {actual_names}"
        )
    species_model = payload.get("species_model")
    grade_models = payload.get("grade_models")
    if species_model is None or not isinstance(grade_models, (list, tuple)) or len(grade_models) != 2:
        raise ValueError("Frozen hierarchical SVM payload lacks species_model/two grade_models")
    matrix = svm_feature_matrix(features, str(payload.get("view_mode") or ""))
    species_probability = ordered_binary_probability(species_model, matrix)
    grade_probability = torch.stack(
        [ordered_binary_probability(model, matrix) for model in grade_models], dim=1
    )
    return species_probability, grade_probability


def default_svm_calibration_inputs(args: argparse.Namespace) -> None:
    base = Path("yolo11_pepper")
    args.val_manifest = args.val_manifest or base / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
    args.val_features = args.val_features or [
        base / "runs/hierarchical_v4/features/pepper_det_val.pt",
        base / "runs/hierarchical_v4/features/imagenet_cls_val.pt",
    ]
    args.svm_model = (
        args.svm_model
        or base / "runs/hierarchical_v4/best_hierarchical_v4_svm_strict.joblib"
    )
    args.svm_selection_output = (
        args.svm_selection_output
        or base / "runs/hierarchical_v4/v4_selection/svm_selection_strict.json"
    )
    args.reference_neural_selection = (
        args.reference_neural_selection
        or base / "runs/hierarchical_v4/v4_selection/selection.json"
    )


def verify_strict_svm_training_provenance(
    svm_payload: dict[str, Any], expected_val_manifest: Path
) -> dict[str, Any]:
    """Fail closed unless the estimator was built from physical train/val files."""
    protocol = svm_payload.get("protocol") or {}
    required_booleans = {
        "test_loaded": False,
        "test_metadata_read": False,
    }
    for field, expected in required_booleans.items():
        if protocol.get(field) is not expected:
            raise ValueError(
                f"Strict SVM provenance requires protocol.{field}={expected}"
            )
    train_value = protocol.get("train_manifest")
    val_value = protocol.get("validation_manifest")
    if not train_value or not val_value:
        raise ValueError("Strict SVM provenance lacks physical train/validation manifests")
    train_manifest = Path(str(train_value)).resolve()
    val_manifest = Path(str(val_value)).resolve()
    if train_manifest.name != "train_manifest.csv":
        raise ValueError(f"SVM train input was not the physical train manifest: {train_manifest}")
    if val_manifest.name != "model_selection_manifest.csv":
        raise ValueError(
            f"SVM validation input was not the physical model-selection manifest: {val_manifest}"
        )
    if val_manifest != expected_val_manifest.resolve():
        raise ValueError(
            f"SVM validation provenance does not match calibration manifest: {val_manifest}"
        )
    reject_test_path(train_manifest, "strict SVM train provenance")
    reject_test_path(val_manifest, "strict SVM validation provenance")
    train_rows = read_single_split_manifest(train_manifest, "train")
    provenance_val_rows = read_single_split_manifest(val_manifest, "val")
    if len(train_rows) != 697 or len(provenance_val_rows) != 146:
        raise ValueError(
            "Strict SVM provenance expected physical train=697 and validation=146 rows"
        )
    if protocol.get("paired_views") != "canonical/detector pair has total sample mass 1":
        raise ValueError("Strict SVM provenance lacks paired-view total-mass-one policy")
    return {
        "verified": True,
        "unified_manifest_opened": False,
        "test_metadata_read": False,
        "test_data_loaded": False,
        "physical_train_manifest": input_record(train_manifest),
        "physical_validation_manifest": input_record(val_manifest),
        "train_rows": len(train_rows),
        "validation_rows": len(provenance_val_rows),
        "paired_view_policy": protocol["paired_views"],
    }


def reject_non_svm_args_during_svm_calibration(args: argparse.Namespace) -> None:
    supplied = [
        name
        for name in (
            "train_manifest",
            "train_features",
            "checkpoints",
            "selection_output",
            "selection",
            "svm_selection",
            "test_manifest",
            "test_features",
            "test_report",
        )
        if getattr(args, name) is not None
    ]
    if supplied:
        raise ValueError(
            f"SVM calibration refuses neural/final-test arguments: {', '.join(supplied)}"
        )


def run_svm_calibration(args: argparse.Namespace) -> None:
    """Calibrate a frozen hierarchical SVM using validation and nothing else."""
    reject_non_svm_args_during_svm_calibration(args)
    default_svm_calibration_inputs(args)
    for path, label in (
        (args.val_manifest, "validation manifest"),
        (args.svm_model, "frozen SVM"),
        (args.svm_selection_output, "SVM selection output"),
        (args.reference_neural_selection, "reference neural selection"),
    ):
        reject_test_path(path, label)
    for path in args.val_features:
        reject_test_path(path, "validation feature")

    val_rows = read_single_split_manifest(args.val_manifest, "val")
    val_features = load_feature_families(
        args.val_features, val_rows, "val", validation_only=True
    )
    import joblib
    import sklearn

    svm_model_path = args.svm_model.resolve()
    svm_payload = joblib.load(svm_model_path)
    if not isinstance(svm_payload, dict):
        raise TypeError(f"Frozen SVM artifact must contain a dict: {svm_model_path}")
    strict_provenance = verify_strict_svm_training_provenance(
        svm_payload, args.val_manifest
    )
    species_probability, grade_probability = predict_svm_branches(
        svm_payload, val_features
    )
    raw_metrics = metrics_from_branches(
        val_rows, species_probability, grade_probability
    )
    recorded_validation = svm_payload.get("validation") or {}
    if recorded_validation:
        if abs(float(recorded_validation["accuracy"]) - raw_metrics["joint_accuracy"]) > 1e-7:
            raise ValueError("Frozen SVM validation accuracy was not exactly reproduced")
        if abs(float(recorded_validation["macro_f1"]) - raw_metrics["joint_macro_f1"]) > 1e-7:
            raise ValueError("Frozen SVM validation macro-F1 was not exactly reproduced")
        if recorded_validation.get("confusion") != raw_metrics["confusion"]:
            raise ValueError("Frozen SVM validation confusion was not exactly reproduced")

    bounds = parse_float_list(args.temperature_bounds, "--temperature-bounds")
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
        raise ValueError("--temperature-bounds must be two positive increasing values")
    temperatures, objectives = fit_conditional_temperatures(
        val_rows,
        species_probability,
        grade_probability,
        bounds[0],
        bounds[1],
    )
    calibrated_species, calibrated_grade = apply_temperatures(
        species_probability, grade_probability, temperatures
    )
    calibrated_metrics = metrics_from_branches(
        val_rows, calibrated_species, calibrated_grade
    )

    neural_path = args.reference_neural_selection.resolve()
    neural_selection = json.loads(neural_path.read_text(encoding="utf-8"))
    if neural_selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"Invalid reference neural selection: {neural_path}")
    neural_metrics = neural_selection["selected"]["validation_metrics_calibrated"]
    if (
        raw_metrics["joint_accuracy"] < neural_metrics["joint_accuracy"]
        or raw_metrics["joint_macro_f1"] < neural_metrics["joint_macro_f1"]
    ):
        raise ValueError(
            "Cannot precommit SVM as primary: it is not at least as good on both frozen validation criteria"
        )

    family_schema = svm_family_schema(val_features.families)
    frozen_core = {
        "model": {
            "path": str(svm_model_path),
            "sha256": sha256_file(svm_model_path),
            "family": "hierarchical_rbf_svc",
            "C": float(svm_payload["C"]),
            "view_mode": str(svm_payload["view_mode"]),
        },
        "feature_families": family_schema,
        "combined_feature_dim": int(val_features.tensor.shape[-1]),
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
    }
    primary_commitment = {
        "primary_model": "svm",
        "committed_before_test": True,
        "basis": (
            "SVM exceeded the frozen neural selection on validation joint accuracy "
            "and joint macro-F1"
        ),
        "test_metric_switching_allowed": False,
        "svm_validation": {
            "joint_accuracy": raw_metrics["joint_accuracy"],
            "joint_macro_f1": raw_metrics["joint_macro_f1"],
            "balanced_selection_score": raw_metrics["balanced_selection_score"],
        },
        "reference_neural_selection": {
            "path": str(neural_path),
            "sha256": sha256_file(neural_path),
            "selection_id": neural_selection["selection_id"],
            "validation": {
                "joint_accuracy": neural_metrics["joint_accuracy"],
                "joint_macro_f1": neural_metrics["joint_macro_f1"],
                "balanced_selection_score": neural_metrics[
                    "balanced_selection_score"
                ],
            },
        },
    }
    svm_selection = {
        "schema": SVM_SELECTION_SCHEMA,
        "created_at_utc": utc_now(),
        "selection_id": f"pepper-v4-svm-selection-{sha256_json({**frozen_core, 'primary_commitment': primary_commitment})[:20]}",
        "protocol": {
            "stage": "validation_temperature_calibration_of_frozen_svm",
            "model_training_or_selection_performed": False,
            "strict_training_provenance_verified": True,
            "unified_manifest_opened_by_training": False,
            "test_arguments_accepted": False,
            "test_manifest_read": False,
            "test_features_read": False,
            "test_metrics_computed": False,
            "temperature_fit_data": "validation only",
            "final_test_requires_explicit_flag": "--final-test",
        },
        "input_fingerprints": {
            "val_manifest": input_record(args.val_manifest),
            "val_features": val_features.files,
            "svm_model": input_record(svm_model_path),
            "reference_neural_selection": input_record(neural_path),
            "strict_train_manifest": strict_provenance[
                "physical_train_manifest"
            ],
        },
        "runtime": {
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
        "feature_schema": {
            "families": family_schema,
            "combined_dim": int(val_features.tensor.shape[-1]),
            "concatenation_order": [family.name for family in val_features.families],
        },
        "selected": {
            **frozen_core,
            "validation_metrics_raw": raw_metrics,
            "temperature_fit_objectives": objectives,
            "validation_metrics_calibrated": calibrated_metrics,
        },
        "strict_training_provenance": strict_provenance,
        "primary_commitment": primary_commitment,
        "final_test_status": "not_run",
    }
    atomic_write_json(
        args.svm_selection_output, svm_selection, refuse_existing=True
    )
    print(
        json.dumps(
            {
                "svm_selection_saved": str(args.svm_selection_output.resolve()),
                "selection_id": svm_selection["selection_id"],
                "primary_precommitted": "svm",
                "validation_raw": raw_metrics,
                "temperatures": svm_selection["selected"]["temperatures"],
                "validation_calibrated": calibrated_metrics,
                "strict_protocol": "NO TEST MANIFEST, FEATURE, LABEL, OR METRIC WAS READ",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def reject_selection_only_args_during_final_test(args: argparse.Namespace) -> None:
    supplied = [
        name
        for name in (
            "train_manifest",
            "val_manifest",
            "train_features",
            "val_features",
            "checkpoints",
            "selection_output",
            "svm_model",
            "svm_selection_output",
            "reference_neural_selection",
        )
        if getattr(args, name) is not None
    ]
    if supplied:
        raise ValueError(
            f"Final-test mode refuses selection-stage arguments: {', '.join(supplied)}"
        )


def metrics_for_subsets(
    rows: Sequence[dict[str, str]],
    species_probability: torch.Tensor,
    grade_probability: torch.Tensor,
    field: str,
) -> dict[str, Any]:
    indices_by_value: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_value[row.get(field) or "unknown"].append(index)
    result: dict[str, Any] = {}
    for value, indices in sorted(indices_by_value.items()):
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        result[value] = metrics_from_branches(
            [rows[index] for index in indices],
            species_probability[tensor_indices],
            grade_probability[tensor_indices],
            include_per_class=False,
        )
    return result


def families_from_selection(selection: dict[str, Any]) -> list[FeatureFamily]:
    result = []
    for item in selection["feature_schema"]["families"]:
        result.append(
            FeatureFamily(
                name=str(item["name"]),
                kind=str(item["kind"]),
                dim=int(item["dim"]),
                checkpoint=str(item.get("checkpoint") or ""),
            )
        )
    return result


def run_final_test(args: argparse.Namespace) -> None:
    reject_selection_only_args_during_final_test(args)
    if (
        args.selection is None
        or args.svm_selection is None
        or args.test_manifest is None
        or not args.test_features
    ):
        raise ValueError(
            "--final-test requires --selection, --svm-selection, --test-manifest, and --test-features"
        )
    selection_path = args.selection.resolve()
    svm_selection_path = args.svm_selection.resolve()
    neural_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    svm_selection = json.loads(svm_selection_path.read_text(encoding="utf-8"))
    if neural_selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"Unsupported or non-frozen selection: {selection_path}")
    if svm_selection.get("schema") != SVM_SELECTION_SCHEMA:
        raise ValueError(f"Unsupported or non-frozen SVM selection: {svm_selection_path}")
    if neural_selection.get("final_test_status") != "not_run":
        raise ValueError("Neural selection does not declare final_test_status=not_run")
    if svm_selection.get("final_test_status") != "not_run":
        raise ValueError("SVM selection does not declare final_test_status=not_run")

    strict_provenance = svm_selection.get("strict_training_provenance") or {}
    svm_protocol = svm_selection.get("protocol") or {}
    if svm_protocol.get("strict_training_provenance_verified") is not True:
        raise ValueError("Combined final test refuses a non-strict SVM selection")
    if svm_protocol.get("unified_manifest_opened_by_training") is not False:
        raise ValueError("Combined final test requires unified-manifest-free SVM training")
    if strict_provenance.get("verified") is not True:
        raise ValueError("Strict SVM provenance is not verified")
    if strict_provenance.get("unified_manifest_opened") is not False:
        raise ValueError("Strict SVM provenance indicates unified manifest access")
    if strict_provenance.get("test_metadata_read") is not False:
        raise ValueError("Strict SVM provenance indicates test metadata access")
    if strict_provenance.get("test_data_loaded") is not False:
        raise ValueError("Strict SVM provenance indicates test data access")
    if strict_provenance.get("train_rows") != 697 or strict_provenance.get(
        "validation_rows"
    ) != 146:
        raise ValueError("Strict SVM physical train/validation row counts changed")
    if strict_provenance.get("paired_view_policy") != (
        "canonical/detector pair has total sample mass 1"
    ):
        raise ValueError("Strict SVM paired-view weighting policy is absent")
    for manifest_key in (
        "physical_train_manifest",
        "physical_validation_manifest",
    ):
        record = strict_provenance.get(manifest_key) or {}
        manifest_path = Path(str(record.get("path") or "")).resolve()
        if not manifest_path.is_file() or sha256_file(manifest_path) != record.get(
            "sha256"
        ):
            raise ValueError(f"Strict SVM provenance fingerprint changed: {manifest_key}")

    commitment = svm_selection.get("primary_commitment") or {}
    reference = commitment.get("reference_neural_selection") or {}
    if commitment.get("primary_model") != "svm":
        raise ValueError("Frozen primary commitment is not SVM")
    if commitment.get("committed_before_test") is not True:
        raise ValueError("SVM was not explicitly committed as primary before test")
    if commitment.get("test_metric_switching_allowed") is not False:
        raise ValueError("Frozen primary commitment does not forbid test-time switching")
    if reference.get("selection_id") != neural_selection.get("selection_id"):
        raise ValueError("SVM commitment references a different neural selection id")
    if reference.get("sha256") != sha256_file(selection_path):
        raise ValueError("SVM commitment references a different neural selection hash")

    report_path = (
        args.test_report.resolve()
        if args.test_report is not None
        else selection_path.with_name("combined_final_test_report.json")
    )
    receipt_path = selection_path.with_name("combined_final_test_receipt.json")
    if report_path.exists() or receipt_path.exists():
        raise FileExistsError(
            f"One-shot final test already has an artifact: {report_path} or {receipt_path}"
        )

    # Check both frozen selections and load both model artifacts before touching
    # any test input.  No comparison below can change the precommitted primary.
    selected_components = neural_selection["selected"]["components"]
    for component in selected_components:
        checkpoint_path = Path(component["path"]).resolve()
        actual_digest = sha256_file(checkpoint_path)
        if actual_digest != component["sha256"]:
            raise ValueError(f"Frozen checkpoint hash mismatch: {checkpoint_path}")
    svm_model_record = svm_selection["selected"]["model"]
    svm_model_path = Path(svm_model_record["path"]).resolve()
    if sha256_file(svm_model_path) != svm_model_record["sha256"]:
        raise ValueError(f"Frozen SVM hash mismatch: {svm_model_path}")
    import joblib

    svm_payload = joblib.load(svm_model_path)
    if not isinstance(svm_payload, dict):
        raise TypeError(f"Frozen SVM artifact must contain a dict: {svm_model_path}")

    # This is the only code path permitted to unseal test.  The rows/features
    # are materialized once and the identical in-memory object feeds both models.
    test_rows = read_single_split_manifest(args.test_manifest, "test")
    test_features = load_feature_families(
        args.test_features, test_rows, "test", validation_only=False
    )
    neural_families = families_from_selection(neural_selection)
    svm_families = families_from_selection(svm_selection)
    compare_feature_families(neural_families, test_features.families, "neural test")
    compare_feature_families(svm_families, test_features.families, "SVM test")
    if test_features.tensor.shape[-1] != int(
        neural_selection["feature_schema"]["combined_dim"]
    ):
        raise ValueError("Test combined feature dimension does not match neural selection")
    if test_features.tensor.shape[-1] != int(
        svm_selection["feature_schema"]["combined_dim"]
    ):
        raise ValueError("Test combined feature dimension does not match SVM selection")

    # Frozen neural comparator.
    device = choose_device(args.device)
    predictions: list[CheckpointPrediction] = []
    components: list[tuple[int, float]] = []
    for component in selected_components:
        prediction = load_checkpoint_prediction(
            Path(component["path"]), test_features.tensor, test_rows, device
        )
        if prediction.sha256 != component["sha256"]:
            raise ValueError(f"Checkpoint changed during final evaluation: {prediction.path}")
        predictions.append(prediction)
        components.append((len(predictions) - 1, float(component["checkpoint_weight"])))
    neural_species, neural_grade = mix_branches(predictions, components)
    neural_temperature_dict = neural_selection["selected"]["temperatures"]
    neural_temperatures = [
        float(neural_temperature_dict["species"]),
        float(neural_temperature_dict["grade_given_子弹头"]),
        float(neural_temperature_dict["grade_given_条子"]),
    ]
    neural_species, neural_grade = apply_temperatures(
        neural_species, neural_grade, neural_temperatures
    )

    # Frozen, validation-precommitted SVM primary.  This branch is always the
    # primary in the report; no test metric is inspected to decide that role.
    svm_species, svm_grade = predict_svm_branches(svm_payload, test_features)
    svm_temperature_dict = svm_selection["selected"]["temperatures"]
    svm_temperatures = [
        float(svm_temperature_dict["species"]),
        float(svm_temperature_dict["grade_given_子弹头"]),
        float(svm_temperature_dict["grade_given_条子"]),
    ]
    svm_species, svm_grade = apply_temperatures(
        svm_species, svm_grade, svm_temperatures
    )

    neural_overall = metrics_from_branches(test_rows, neural_species, neural_grade)
    svm_overall = metrics_from_branches(test_rows, svm_species, svm_grade)
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": utc_now(),
        "protocol": {
            "stage": "one_shot_combined_final_test",
            "test_opened_once_for_both_frozen_models": True,
            "both_selections_frozen_before_test": True,
            "selection_calibration_or_threshold_refit_on_test": False,
            "primary_model": "svm",
            "primary_committed_on_validation_before_test": True,
            "test_metrics_may_change_primary": False,
        },
        "precommitted_primary": commitment,
        "selections": {
            "primary_svm": {
                "path": str(svm_selection_path),
                "sha256": sha256_file(svm_selection_path),
                "selection_id": svm_selection["selection_id"],
            },
            "comparator_neural": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "selection_id": neural_selection["selection_id"],
            },
        },
        "test_inputs": {
            "manifest": input_record(args.test_manifest),
            "features": test_features.files,
            "samples": len(test_rows),
            "shared_in_memory_between_models": True,
        },
        "models": {
            "primary_svm": {
                "role_locked_before_test": "primary",
                "selected_model": svm_selection["selected"],
                "test_metrics": svm_overall,
                "test_metrics_by_origin": metrics_for_subsets(
                    test_rows, svm_species, svm_grade, "origin"
                ),
                "test_metrics_by_group": metrics_for_subsets(
                    test_rows, svm_species, svm_grade, "group_id"
                ),
            },
            "comparator_neural": {
                "role_locked_before_test": "comparator",
                "selected_model": neural_selection["selected"],
                "test_metrics": neural_overall,
                "test_metrics_by_origin": metrics_for_subsets(
                    test_rows, neural_species, neural_grade, "origin"
                ),
                "test_metrics_by_group": metrics_for_subsets(
                    test_rows, neural_species, neural_grade, "group_id"
                ),
            },
        },
    }
    atomic_write_json(report_path, report, refuse_existing=True)
    receipt = {
        "schema": "pepper-v4-combined-final-test-receipt-v1",
        "created_at_utc": utc_now(),
        "primary_model": "svm",
        "primary_was_committed_before_test": True,
        "test_metric_switching_allowed": False,
        "selections": {
            "svm": {
                "selection_id": svm_selection["selection_id"],
                "sha256": sha256_file(svm_selection_path),
            },
            "neural": {
                "selection_id": neural_selection["selection_id"],
                "sha256": sha256_file(selection_path),
            },
        },
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status": "both frozen models evaluated in one sealed run; repeat refused",
    }
    atomic_write_json(receipt_path, receipt, refuse_existing=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.final_test:
        run_final_test(args)
    elif args.calibrate_svm:
        run_svm_calibration(args)
    else:
        run_selection(args)


if __name__ == "__main__":
    main()
