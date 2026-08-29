#!/usr/bin/env python3
"""Train and freeze the clean-v5 hierarchical SVM using train/validation only.

This program deliberately has no test-data command-line arguments and no final-test
mode.  It performs the complete *model-selection* stage for the clean v5 feature
experiment:

1. load the physically separated 697-row training manifest and 146-row validation
   manifest;
2. verify path order, labels, groups and split metadata in every feature cache;
3. independently L2-normalize each feature family, then compare the predeclared
   ``canonical`` and ``view_mean`` training representations;
4. fit ``p(species)`` and two ``p(grade | species)`` RBF-SVC branches, with every
   canonical/detector ``pair_id`` receiving total base mass one;
5. select from the fixed C grid by validation four-class macro-F1, then accuracy;
6. fit three scalar temperatures on validation only and freeze a joblib model plus
   an auditable JSON selection record.

The strict test manifest is neither named nor opened anywhere in the executable
path.  A later, separate evaluator must consume the frozen artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC


SCRIPT_VERSION = "pepper-clean-v5-validation-selection-v1"
CLASS_NAMES = ("子弹头_一级", "子弹头_二级", "条子_一级", "条子_二级")
SPECIES_NAMES = ("子弹头", "条子")

# This grid is intentionally a source-code constant: it cannot be expanded after
# looking at validation results through a command-line flag.
PREDECLARED_C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
PREDECLARED_VIEW_MODES = ("canonical", "view_mean")
TEMPERATURE_BOUNDS = (0.05, 10.0)
RANDOM_SEED = 2041
EXPECTED_TRAIN_ROWS = 697
EXPECTED_VAL_ROWS = 146

PROJECT = Path(__file__).resolve().parent
DEFAULT_TRAIN_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/train_manifest.csv"
)
DEFAULT_VAL_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
)
DEFAULT_TRAIN_FEATURES = (
    PROJECT / "runs/hierarchical_v5_clean/features/strict_det_train.pt",
    PROJECT / "runs/hierarchical_v5_clean/features_cls384/imagenet_cls_train.pt",
)
DEFAULT_VAL_FEATURES = (
    PROJECT / "runs/hierarchical_v5_clean/features/strict_det_val.pt",
    PROJECT / "runs/hierarchical_v5_clean/features_cls384/imagenet_cls_val.pt",
)
DEFAULT_MODEL_OUTPUT = (
    PROJECT / "runs/hierarchical_v5_clean/best_hierarchical_clean_v5_svm.joblib"
)
DEFAULT_SELECTION_OUTPUT = PROJECT / "runs/hierarchical_v5_clean/selection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict train/validation-only clean-v5 hierarchical SVM selection. "
            "There is intentionally no test argument or test mode."
        )
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument(
        "--train-features",
        type=Path,
        nargs="+",
        default=list(DEFAULT_TRAIN_FEATURES),
        help="Ordered train feature families; defaults to strict_det + imagenet_cls.",
    )
    parser.add_argument(
        "--val-features",
        type=Path,
        nargs="+",
        default=list(DEFAULT_VAL_FEATURES),
        help="Ordered validation feature families matching --train-features.",
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT
    )
    return parser.parse_args()


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


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def path_has_test_token(path: Path) -> bool:
    """Match a complete case-insensitive token, not substrings such as 'latest'."""
    return any(
        "test"
        in {
            token
            for token in re.split(r"[^a-z0-9]+", part.lower())
            if token
        }
        for part in path.parts
    )


def reject_test_path(path: Path, label: str) -> None:
    if path_has_test_token(path):
        raise ValueError(
            f"STRICT VALIDATION-ONLY protocol rejected {label} containing a test token: {path}"
        )


def parse_finite_weight(value: str, *, path: Path, line: int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}:{line} invalid {field}={value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{path}:{line} invalid {field}={value!r}")
    return result


def read_manifest(
    path: Path, *, expected_split: str, expected_rows: int
) -> list[dict[str, Any]]:
    """Read a physically split-specific manifest without silently filtering it."""
    path = path.resolve()
    reject_test_path(path, f"{expected_split} manifest")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path",
            "split",
            "group_id",
            "source_id",
            "class_id",
            "pair_id",
            "content_sha256",
            "selection_role",
            "eligible_for_model_training",
            "view_type",
            "record_role",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        if expected_split == "train":
            weight_missing = {"species_weight", "grade_weight"} - set(
                reader.fieldnames or ()
            )
            if weight_missing:
                raise ValueError(
                    f"{path} missing training-weight columns: {sorted(weight_missing)}"
                )
        for line, raw in enumerate(reader, 2):
            split = (raw.get("split") or "").strip().lower()
            if split != expected_split:
                raise ValueError(
                    f"{path}:{line} split={split!r}; expected only {expected_split!r}"
                )
            try:
                class_id = int(raw["class_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line} invalid class_id") from error
            if not 0 <= class_id < 4:
                raise ValueError(f"{path}:{line} class_id={class_id} is outside [0,3]")
            image_value = (raw.get("path") or "").strip()
            group_id = (raw.get("group_id") or "").strip()
            source_id = (raw.get("source_id") or "").strip()
            pair_id = (raw.get("pair_id") or "").strip()
            content_sha256 = (raw.get("content_sha256") or "").strip().lower()
            if not image_value or not group_id or not source_id or not pair_id:
                raise ValueError(
                    f"{path}:{line} requires non-empty path/group_id/source_id/pair_id"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
                raise ValueError(f"{path}:{line} invalid content_sha256")
            image_path = Path(image_value).resolve()
            reject_test_path(image_path, f"{expected_split} image path")
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
            selection_role = (raw.get("selection_role") or "").strip().lower()
            eligible = (
                raw.get("eligible_for_model_training") or ""
            ).strip().lower()
            view_type = (raw.get("view_type") or "").strip().lower()
            record_role = (raw.get("record_role") or "").strip().lower()
            expected_role = "training" if expected_split == "train" else "model_selection"
            expected_eligible = "true" if expected_split == "train" else "false"
            if selection_role != expected_role:
                raise ValueError(
                    f"{path}:{line} selection_role={selection_role!r}; expected {expected_role!r}"
                )
            if eligible != expected_eligible:
                raise ValueError(
                    f"{path}:{line} eligible_for_model_training={eligible!r}; "
                    f"expected {expected_eligible!r}"
                )
            permitted_views = (
                {"canonical", "detector_aligned"}
                if expected_split == "train"
                else {"canonical"}
            )
            if view_type not in permitted_views:
                raise ValueError(
                    f"{path}:{line} invalid {expected_split} view_type={view_type!r}"
                )
            expected_record_role = (
                "paired_detector_train_view"
                if view_type == "detector_aligned"
                else "canonical_audited"
            )
            if record_role != expected_record_role:
                raise ValueError(
                    f"{path}:{line} record_role={record_role!r}; "
                    f"expected {expected_record_role!r} for view_type={view_type!r}"
                )
            record: dict[str, Any] = {
                **raw,
                "path": str(image_path),
                "split": split,
                "group_id": group_id,
                "pair_id": pair_id,
                "source_id": source_id,
                "content_sha256": content_sha256,
                "selection_role": selection_role,
                "eligible_for_model_training": eligible,
                "view_type": view_type,
                "record_role": record_role,
                "class_id": class_id,
            }
            if expected_split == "train":
                record["species_weight"] = parse_finite_weight(
                    raw.get("species_weight", ""),
                    path=path,
                    line=line,
                    field="species_weight",
                )
                record["grade_weight"] = parse_finite_weight(
                    raw.get("grade_weight", ""),
                    path=path,
                    line=line,
                    field="grade_weight",
                )
            records.append(record)
    if len(records) != expected_rows:
        raise ValueError(
            f"{path} has {len(records)} rows; clean-v5 protocol expects {expected_rows}"
        )
    paths = [row["path"] for row in records]
    if len(paths) != len(set(paths)):
        duplicate = next(item for item, count in Counter(paths).items() if count > 1)
        raise ValueError(f"Duplicate image path in {path}: {duplicate}")
    class_counts = Counter(int(row["class_id"]) for row in records)
    if set(class_counts) != {0, 1, 2, 3}:
        raise ValueError(f"{path} does not contain all four classes: {class_counts}")
    return records


def overlap(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    field: str,
    *,
    ignore_empty: bool = True,
) -> list[str]:
    shared = {str(row.get(field, "")) for row in first} & {
        str(row.get(field, "")) for row in second
    }
    if ignore_empty:
        shared.discard("")
    return sorted(shared)


def leakage_audit(
    train_rows: Sequence[dict[str, Any]], val_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    audit = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "image_path_overlap": overlap(train_rows, val_rows, "path"),
        "group_id_overlap": overlap(train_rows, val_rows, "group_id"),
        "source_id_overlap": overlap(train_rows, val_rows, "source_id"),
        "pair_id_overlap": overlap(train_rows, val_rows, "pair_id"),
        "content_sha256_overlap": overlap(
            train_rows, val_rows, "content_sha256"
        ),
    }
    audit["passed"] = not any(
        audit[key]
        for key in (
            "image_path_overlap",
            "group_id_overlap",
            "source_id_overlap",
            "pair_id_overlap",
            "content_sha256_overlap",
        )
    )
    return audit


def pair_weight_audit(
    train_rows: Sequence[dict[str, Any]], tolerance: float = 1e-12
) -> tuple[np.ndarray, dict[str, Any]]:
    pair_counts = Counter(str(row["pair_id"]) for row in train_rows)
    base_weight = np.asarray(
        [1.0 / pair_counts[str(row["pair_id"])] for row in train_rows],
        dtype=np.float64,
    )
    total_by_pair: dict[str, float] = defaultdict(float)
    labels_by_pair: dict[str, set[int]] = defaultdict(set)
    groups_by_pair: dict[str, set[str]] = defaultdict(set)
    sources_by_pair: dict[str, set[str]] = defaultdict(set)
    views_by_pair: dict[str, list[str]] = defaultdict(list)
    species_weights_by_pair: dict[str, list[float]] = defaultdict(list)
    grade_weights_by_pair: dict[str, list[float]] = defaultdict(list)
    for row, weight in zip(train_rows, base_weight):
        pair_id = str(row["pair_id"])
        total_by_pair[pair_id] += float(weight)
        labels_by_pair[pair_id].add(int(row["class_id"]))
        groups_by_pair[pair_id].add(str(row["group_id"]))
        sources_by_pair[pair_id].add(str(row["source_id"]))
        views_by_pair[pair_id].append(str(row["view_type"]))
        species_weights_by_pair[pair_id].append(float(row["species_weight"]))
        grade_weights_by_pair[pair_id].append(float(row["grade_weight"]))
    inconsistent = sorted(
        pair_id for pair_id, labels in labels_by_pair.items() if len(labels) != 1
    )
    if inconsistent:
        raise ValueError(f"pair_id has inconsistent labels: {inconsistent[:10]}")
    inconsistent_identity = sorted(
        pair_id
        for pair_id in pair_counts
        if len(groups_by_pair[pair_id]) != 1 or len(sources_by_pair[pair_id]) != 1
    )
    if inconsistent_identity:
        raise ValueError(
            f"pair_id crosses group/source identity: {inconsistent_identity[:10]}"
        )
    invalid_views: dict[str, list[str]] = {}
    for pair_id, views in views_by_pair.items():
        expected = (
            ["canonical"]
            if pair_counts[pair_id] == 1
            else ["canonical", "detector_aligned"]
        )
        if sorted(views) != sorted(expected):
            invalid_views[pair_id] = views
    if invalid_views:
        raise ValueError(
            "Each pair must be one canonical view or exactly canonical+detector_aligned: "
            f"{dict(list(invalid_views.items())[:10])}"
        )
    inconsistent_weights = sorted(
        pair_id
        for pair_id in pair_counts
        if (
            max(species_weights_by_pair[pair_id])
            - min(species_weights_by_pair[pair_id])
            > tolerance
            or max(grade_weights_by_pair[pair_id])
            - min(grade_weights_by_pair[pair_id])
            > tolerance
        )
    )
    if inconsistent_weights:
        raise ValueError(
            f"Paired views have inconsistent species/grade weights: {inconsistent_weights[:10]}"
        )
    deviations = {
        pair_id: total
        for pair_id, total in total_by_pair.items()
        if abs(total - 1.0) > tolerance
    }
    if deviations:
        raise ValueError(f"pair_id total base mass is not one: {deviations}")
    return base_weight, {
        "pair_count": len(pair_counts),
        "single_view_pairs": sum(count == 1 for count in pair_counts.values()),
        "multi_view_pairs": sum(count > 1 for count in pair_counts.values()),
        "maximum_views_per_pair": max(pair_counts.values()),
        "base_mass_policy": "each row receives 1 / count(pair_id)",
        "minimum_total_base_mass": min(total_by_pair.values()),
        "maximum_total_base_mass": max(total_by_pair.values()),
        "all_pair_total_base_mass_exactly_one_within_tolerance": True,
        "all_pairs_have_consistent_label_group_source_and_weights": True,
        "multi_view_schema": "exactly one canonical plus one detector_aligned view",
        "tolerance": tolerance,
    }


def pair_aware_balanced_weight(
    labels: np.ndarray,
    raw_weight: np.ndarray,
    *,
    label_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Balance classes from pair-normalized effective mass, not duplicate rows."""
    result = raw_weight.copy()
    positive = raw_weight > 0
    observed = sorted(set(labels[positive].tolist()))
    if observed != list(range(len(label_names))):
        raise ValueError(
            f"Positive effective weights do not cover {list(range(len(label_names)))}: {observed}"
        )
    mass_before = {
        label_names[class_id]: float(raw_weight[(labels == class_id) & positive].sum())
        for class_id in observed
    }
    total = sum(mass_before.values())
    factors = {
        class_id: total / (len(observed) * mass_before[label_names[class_id]])
        for class_id in observed
    }
    for class_id, factor in factors.items():
        result[(labels == class_id) & positive] *= factor
    return result, {
        "mass_before_balance": mass_before,
        "class_factors": {
            label_names[class_id]: float(factor)
            for class_id, factor in factors.items()
        },
        "mass_after_balance": {
            label_names[class_id]: float(result[(labels == class_id) & positive].sum())
            for class_id in observed
        },
        "policy": "balance from pair-normalized confidence mass; sklearn class_weight=None",
    }


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    kind: str
    dim: int
    image_size: int
    scale_normalized: bool
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_hash_attestation: str

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dim": self.dim,
            "image_size": self.image_size,
            "scale_normalized": self.scale_normalized,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_hash_attestation": self.checkpoint_hash_attestation,
            "normalization": "independent per-view L2 before concatenation",
        }


@dataclass
class LoadedFeatureSet:
    blocks: list[torch.Tensor]
    families: list[FeatureFamily]
    files: list[dict[str, Any]]


def checkpoint_digest(
    checkpoint_value: str,
    *,
    cache_path: Path,
    metadata_digest: str,
) -> tuple[str, str]:
    if not checkpoint_value:
        raise ValueError(f"Feature cache lacks metadata.checkpoint: {cache_path}")
    checkpoint = Path(checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Feature provenance checkpoint no longer exists: {checkpoint}"
        )
    reject_test_path(checkpoint, "feature provenance checkpoint")
    current_digest = sha256_file(checkpoint)
    normalized_metadata_digest = metadata_digest.strip().lower()
    if normalized_metadata_digest:
        if normalized_metadata_digest != current_digest:
            raise ValueError(
                f"Feature cache checkpoint digest no longer matches {checkpoint}: "
                f"{normalized_metadata_digest} != {current_digest}"
            )
        return current_digest, "cache_metadata_sha256_verified_against_current_file"
    # Older extractor schema did not persist a digest.  Fail if the checkpoint
    # is newer than this cache; otherwise record the weaker, explicit attestation
    # rather than pretending that a selection-time hash was extraction-bound.
    if checkpoint.stat().st_mtime_ns > cache_path.stat().st_mtime_ns:
        raise ValueError(
            f"Checkpoint is newer than feature cache and cache has no checkpoint_sha256: {cache_path}"
        )
    return (
        current_digest,
        "selection_time_sha256; checkpoint mtime verified not newer than cache",
    )


def load_feature_set(
    paths: Sequence[Path],
    rows: Sequence[dict[str, Any]],
    *,
    expected_split: str,
    expected_manifest: Path,
) -> LoadedFeatureSet:
    if not paths:
        raise ValueError(f"No {expected_split} feature files were supplied")
    expected_paths = [str(row["path"]) for row in rows]
    expected_classes = torch.tensor(
        [int(row["class_id"]) for row in rows], dtype=torch.long
    )
    expected_groups = [str(row["group_id"]) for row in rows]
    blocks: list[torch.Tensor] = []
    families: list[FeatureFamily] = []
    files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        reject_test_path(path, f"{expected_split} feature cache")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Feature cache is not a dictionary: {path}")
        feature = payload.get("features")
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Feature cache lacks a tensor named 'features': {path}")
        if feature.ndim == 2:
            feature = feature.unsqueeze(1)
        if feature.ndim != 3 or feature.shape[0] != len(rows):
            raise ValueError(
                f"{path} shape {tuple(feature.shape)} does not match {len(rows)} manifest rows"
            )
        if feature.shape[1] < 1 or feature.shape[2] < 1:
            raise ValueError(f"Feature cache has an empty dimension: {path}")
        feature = feature.float().contiguous()
        if not torch.isfinite(feature).all():
            raise ValueError(f"Feature cache contains NaN/Inf: {path}")
        if (torch.linalg.vector_norm(feature, dim=-1) <= 0).any():
            raise ValueError(f"Feature cache contains an all-zero vector: {path}")

        cached_paths = payload.get("paths")
        if cached_paths is None:
            raise ValueError(f"Strict path-order verification needs payload['paths']: {path}")
        resolved_cached_paths = [str(Path(value).resolve()) for value in cached_paths]
        if resolved_cached_paths != expected_paths:
            raise ValueError(f"Feature path order differs from manifest: {path}")
        cached_classes = payload.get("class_ids")
        if not isinstance(cached_classes, torch.Tensor) or not torch.equal(
            cached_classes.cpu().long(), expected_classes
        ):
            raise ValueError(f"Feature class_ids differ from manifest: {path}")
        cached_groups = payload.get("groups")
        if cached_groups is None:
            raise ValueError(f"Strict group-order verification needs payload['groups']: {path}")
        if [str(value) for value in cached_groups] != expected_groups:
            raise ValueError(f"Feature groups differ from manifest: {path}")

        metadata = payload.get("metadata") or {}
        metadata_split = str(metadata.get("split", "")).strip().lower()
        if metadata_split != expected_split:
            raise ValueError(
                f"{path} metadata split={metadata_split!r}; expected {expected_split!r}"
            )
        if bool(metadata.get("test_requested_explicitly", False)):
            raise ValueError(
                f"Validation-only selection rejects a cache produced in a test-requesting extraction: {path}"
            )
        metadata_manifest = str(metadata.get("manifest") or "").strip()
        if not metadata_manifest:
            raise ValueError(f"Feature cache lacks metadata.manifest: {path}")
        if Path(metadata_manifest).resolve() != expected_manifest.resolve():
            raise ValueError(
                f"{path} was not extracted from the physical {expected_split} manifest: "
                f"{metadata_manifest} != {expected_manifest.resolve()}"
            )
        name = str(metadata.get("backbone_name") or path.stem.rsplit("_", 1)[0])
        if name in {family.name for family in families}:
            raise ValueError(f"Duplicate feature family {name!r}")
        checkpoint = str(metadata.get("checkpoint") or "")
        image_size = int(metadata.get("image_size") or 0)
        if image_size <= 0:
            raise ValueError(f"Feature cache lacks a positive metadata.image_size: {path}")
        if not bool(metadata.get("scale_normalized", False)):
            raise ValueError(f"Feature cache is not marked scale_normalized: {path}")
        checkpoint_sha256, checkpoint_attestation = checkpoint_digest(
            checkpoint,
            cache_path=path,
            metadata_digest=str(metadata.get("checkpoint_sha256") or ""),
        )
        family = FeatureFamily(
            name=name,
            kind=str(metadata.get("kind") or "unknown"),
            dim=int(feature.shape[-1]),
            image_size=image_size,
            scale_normalized=True,
            checkpoint=str(Path(checkpoint).resolve()) if checkpoint else "",
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_hash_attestation=checkpoint_attestation,
        )
        blocks.append(feature)
        families.append(family)
        cache_record = fingerprint(path)
        cache_record.update(
            {
                "shape": list(feature.shape),
                "family": family.public(),
                "metadata_manifest_reference": metadata_manifest,
                "metadata_manifest_reference_opened_by_this_script": False,
                "test_requested_explicitly": False,
            }
        )
        files.append(cache_record)
    return LoadedFeatureSet(blocks=blocks, families=families, files=files)


def compare_feature_schemas(
    train: LoadedFeatureSet, validation: LoadedFeatureSet
) -> None:
    train_signature = [
        (
            f.name,
            f.kind,
            f.dim,
            f.image_size,
            f.scale_normalized,
            f.checkpoint_sha256,
        )
        for f in train.families
    ]
    val_signature = [
        (
            f.name,
            f.kind,
            f.dim,
            f.image_size,
            f.scale_normalized,
            f.checkpoint_sha256,
        )
        for f in validation.families
    ]
    if train_signature != val_signature:
        raise ValueError(
            "Train and validation feature-family schemas/checkpoints differ: "
            f"{train_signature} != {val_signature}"
        )


def feature_matrix(features: LoadedFeatureSet, view_mode: str) -> np.ndarray:
    if view_mode not in PREDECLARED_VIEW_MODES:
        raise ValueError(f"Unsupported view_mode={view_mode!r}")
    selected: list[torch.Tensor] = []
    for raw in features.blocks:
        normalized = F.normalize(raw, p=2, dim=-1)
        if view_mode == "canonical":
            block = normalized[:, 0]
        else:
            # Mean aggregation reduces the norm according to view disagreement;
            # normalize again so inter-family scale is not an accidental stability
            # feature and every selected family vector is genuinely unit length.
            block = F.normalize(normalized.mean(1), p=2, dim=-1)
        selected.append(block)
    matrix = torch.cat(selected, dim=1).cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Combined feature matrix contains NaN/Inf")
    return matrix


def make_svc(c_value: float) -> SVC:
    return SVC(
        C=c_value,
        kernel="rbf",
        gamma="scale",
        # Pair-aware balancing is folded into sample_weight.  LibSVM's built-in
        # "balanced" mode counts duplicate rows and would undo pair mass control.
        class_weight=None,
        probability=True,
        random_state=RANDOM_SEED,
    )


def ordered_binary_probability(model: SVC, matrix: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    if set(classes) != {0, 1}:
        raise ValueError(f"Expected binary classes {{0,1}}, got {classes}")
    ordered = np.empty((len(matrix), 2), dtype=np.float64)
    for source_column, class_id in enumerate(classes):
        ordered[:, class_id] = probability[:, source_column]
    ordered /= np.clip(ordered.sum(1, keepdims=True), 1e-15, None)
    return ordered


def fit_candidate(
    train_x: np.ndarray,
    val_x: np.ndarray,
    labels: np.ndarray,
    species_weight: np.ndarray,
    grade_weight: np.ndarray,
    c_value: float,
) -> tuple[SVC, list[SVC], np.ndarray, np.ndarray]:
    species_labels = labels // 2
    grade_labels = labels % 2
    species_keep = species_weight > 0
    if set(species_labels[species_keep].tolist()) != {0, 1}:
        raise ValueError("Positive species_weight rows do not cover both species")
    species_model = make_svc(c_value)
    species_model.fit(
        train_x[species_keep],
        species_labels[species_keep],
        sample_weight=species_weight[species_keep],
    )
    species_probability = ordered_binary_probability(species_model, val_x)

    grade_models: list[SVC] = []
    grade_probabilities: list[np.ndarray] = []
    for species_id in range(2):
        keep = (species_labels == species_id) & (grade_weight > 0)
        if set(grade_labels[keep].tolist()) != {0, 1}:
            raise ValueError(
                f"Positive grade_weight rows for {SPECIES_NAMES[species_id]} do not cover both grades"
            )
        model = make_svc(c_value)
        model.fit(
            train_x[keep],
            grade_labels[keep],
            sample_weight=grade_weight[keep],
        )
        grade_models.append(model)
        grade_probabilities.append(ordered_binary_probability(model, val_x))
    return (
        species_model,
        grade_models,
        species_probability,
        np.stack(grade_probabilities, axis=1),
    )


def ece_score(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == labels
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def branch_metrics(
    rows: Sequence[dict[str, Any]],
    species_probability: np.ndarray,
    grade_probability: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    species_labels = labels // 2
    grade_labels = labels % 2
    joint = species_probability[:, :, None] * grade_probability
    joint = joint.reshape(-1, 4)
    joint /= np.clip(joint.sum(1, keepdims=True), 1e-15, None)
    prediction = joint.argmax(1)
    species_prediction = species_probability.argmax(1)
    conditional_grade_prediction = grade_probability[
        np.arange(len(labels)), species_labels
    ].argmax(1)
    confusion = confusion_matrix(labels, prediction, labels=[0, 1, 2, 3])
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        truth_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - truth_positive
        false_negative = int(confusion[class_id].sum()) - truth_positive
        precision = truth_positive / max(truth_positive + false_positive, 1)
        recall = truth_positive / max(truth_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-15)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[class_id].sum()),
            }
        )
    group_correct: dict[str, list[float]] = defaultdict(list)
    for row, is_correct in zip(rows, prediction == labels):
        group_correct[str(row["group_id"])].append(float(is_correct))
    return {
        "samples": len(labels),
        "groups": len(group_correct),
        "species_accuracy": float(accuracy_score(species_labels, species_prediction)),
        "conditional_grade_accuracy": float(
            accuracy_score(grade_labels, conditional_grade_prediction)
        ),
        "joint_accuracy": float(accuracy_score(labels, prediction)),
        "joint_macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "group_joint_accuracy": float(
            np.mean([np.mean(values) for values in group_correct.values()])
        ),
        "joint_nll": float(
            -np.log(np.clip(joint[np.arange(len(labels)), labels], 1e-15, 1.0)).mean()
        ),
        "joint_ece_15bin": ece_score(joint, labels),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def fit_scalar_temperature(
    probability: np.ndarray,
    labels: np.ndarray,
    *,
    lower: float,
    upper: float,
    iterations: int = 100,
) -> tuple[float, float]:
    """Fit one temperature with bounded golden-section NLL minimization."""
    if len(probability) == 0:
        raise ValueError("Cannot fit temperature on an empty branch")
    log_probability = torch.from_numpy(probability).double().clamp_min(1e-15).log()
    target = torch.from_numpy(labels).long()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        log_scaled = F.log_softmax(log_probability / temperature, dim=1)
        return float(F.nll_loss(log_scaled, target))

    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value = objective(first)
    second_value = objective(second)
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
    ]
    if lower <= 1.0 <= upper:
        candidates.append((0.0, objective(0.0)))
    best_log_temperature, best_nll = min(candidates, key=lambda item: item[1])
    return math.exp(best_log_temperature), best_nll


def fit_temperatures(
    rows: Sequence[dict[str, Any]],
    species_probability: np.ndarray,
    grade_probability: np.ndarray,
) -> tuple[list[float], dict[str, float]]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    species_labels = labels // 2
    grade_labels = labels % 2
    lower, upper = TEMPERATURE_BOUNDS
    species_temperature, species_nll = fit_scalar_temperature(
        species_probability,
        species_labels,
        lower=lower,
        upper=upper,
    )
    temperatures = [species_temperature]
    objectives = {"species_nll": species_nll}
    for species_id, species_name in enumerate(SPECIES_NAMES):
        keep = species_labels == species_id
        temperature, nll = fit_scalar_temperature(
            grade_probability[keep, species_id],
            grade_labels[keep],
            lower=lower,
            upper=upper,
        )
        temperatures.append(temperature)
        objectives[f"grade_given_{species_name}_nll"] = nll
    return temperatures, objectives


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def apply_temperatures(
    species_probability: np.ndarray,
    grade_probability: np.ndarray,
    temperatures: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(temperatures) != 3 or any(value <= 0 for value in temperatures):
        raise ValueError(f"Expected three positive temperatures, got {temperatures}")
    species = softmax_numpy(
        np.log(np.clip(species_probability, 1e-15, 1.0)) / temperatures[0]
    )
    grade = np.empty_like(grade_probability)
    for species_id in range(2):
        grade[:, species_id] = softmax_numpy(
            np.log(np.clip(grade_probability[:, species_id], 1e-15, 1.0))
            / temperatures[species_id + 1]
        )
    return species, grade


def atomic_joblib_dump(value: Any, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen model: {destination}")
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        joblib.dump(value, temporary, compress=3)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite frozen model: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_dump(value: Any, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {destination}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite frozen selection: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    all_paths = (
        args.train_manifest,
        args.val_manifest,
        *args.train_features,
        *args.val_features,
        args.model_output,
        args.selection_output,
    )
    for path in all_paths:
        reject_test_path(path.resolve(), "input/output path")
    if len(args.train_features) != len(args.val_features):
        raise ValueError("Train and validation feature-family counts differ")
    model_output = args.model_output.resolve()
    selection_output = args.selection_output.resolve()
    receipt_output = selection_output.with_name(
        f"{selection_output.name}.sha256.json"
    )
    if model_output == selection_output:
        raise ValueError("Model and selection outputs must be different files")
    for output in (model_output, selection_output, receipt_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {output}")

    train_rows = read_manifest(
        args.train_manifest,
        expected_split="train",
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    val_rows = read_manifest(
        args.val_manifest,
        expected_split="val",
        expected_rows=EXPECTED_VAL_ROWS,
    )
    audit = leakage_audit(train_rows, val_rows)
    if not audit["passed"]:
        raise ValueError(f"Train/validation leakage audit failed: {audit}")
    pair_base_weight, pair_audit = pair_weight_audit(train_rows)

    train_features = load_feature_set(
        args.train_features,
        train_rows,
        expected_split="train",
        expected_manifest=args.train_manifest,
    )
    val_features = load_feature_set(
        args.val_features,
        val_rows,
        expected_split="val",
        expected_manifest=args.val_manifest,
    )
    compare_feature_schemas(train_features, val_features)

    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["class_id"]) for row in val_rows], dtype=np.int64)
    species_confidence = np.asarray(
        [float(row["species_weight"]) for row in train_rows], dtype=np.float64
    )
    grade_confidence = np.asarray(
        [float(row["grade_weight"]) for row in train_rows], dtype=np.float64
    )
    species_labels = labels // 2
    grade_labels = labels % 2
    species_weight_raw = species_confidence * pair_base_weight
    grade_weight_raw = grade_confidence * pair_base_weight
    species_weight, species_balance_audit = pair_aware_balanced_weight(
        species_labels,
        species_weight_raw,
        label_names=SPECIES_NAMES,
    )
    grade_weight = np.zeros_like(grade_weight_raw)
    grade_balance_audit: dict[str, Any] = {}
    for species_id, species_name in enumerate(SPECIES_NAMES):
        branch = species_labels == species_id
        branch_weight, branch_audit = pair_aware_balanced_weight(
            grade_labels[branch],
            grade_weight_raw[branch],
            label_names=("一级", "二级"),
        )
        grade_weight[branch] = branch_weight
        grade_balance_audit[f"grade_given_{species_name}"] = branch_audit
    balancing_audit = {
        "species": species_balance_audit,
        "conditional_grade": grade_balance_audit,
    }

    candidates: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_models: tuple[SVC, list[SVC]] | None = None
    best_probabilities: tuple[np.ndarray, np.ndarray] | None = None
    best_specification: dict[str, Any] | None = None
    for view_mode in PREDECLARED_VIEW_MODES:
        train_x = feature_matrix(train_features, view_mode)
        val_x = feature_matrix(val_features, view_mode)
        if train_x.shape[1] != val_x.shape[1]:
            raise ValueError("Train/validation combined feature dimensions differ")
        for c_value in PREDECLARED_C_GRID:
            (
                species_model,
                grade_models,
                species_probability,
                grade_probability,
            ) = fit_candidate(
                train_x,
                val_x,
                labels,
                species_weight,
                grade_weight,
                c_value,
            )
            metrics = branch_metrics(val_rows, species_probability, grade_probability)
            candidate = {
                "view_mode": view_mode,
                "C": c_value,
                "validation_metrics_raw": metrics,
            }
            candidates.append(candidate)
            # Deliberately only the requested two validation criteria.  Exact
            # ties retain the earlier predeclared candidate: canonical first,
            # then the lower C value.
            key = (metrics["joint_macro_f1"], metrics["joint_accuracy"])
            if key > best_key:
                best_key = key
                best_models = (species_model, grade_models)
                best_probabilities = (species_probability, grade_probability)
                best_specification = candidate
            print(json.dumps(candidate, ensure_ascii=False))

    if best_models is None or best_probabilities is None or best_specification is None:
        raise RuntimeError("No clean-v5 candidate was trained")
    temperatures, temperature_objectives = fit_temperatures(
        val_rows, *best_probabilities
    )
    calibrated_probabilities = apply_temperatures(
        *best_probabilities, temperatures
    )
    calibrated_metrics = branch_metrics(val_rows, *calibrated_probabilities)

    feature_schema = {
        "families": [family.public() for family in train_features.families],
        "concatenation_order": [family.name for family in train_features.families],
        "combined_dim": int(sum(family.dim for family in train_features.families)),
        "normalization": (
            "each family/view independently L2-normalized; view_mean is L2-normalized "
            "again after averaging; no cross-family scaling"
        ),
    }
    protocol = {
        "stage": "strict_train_validation_model_selection_and_calibration",
        "selection_data": str(args.val_manifest.resolve()),
        "selection_metric": "validation joint macro-F1, then validation joint accuracy",
        "exact_tie_policy": "predeclared order: canonical before view_mean, then lower C",
        "temperature_fit_data": "validation only",
        "temperature_application": "softmax(log(branch_probability) / T)",
        "strict_test_manifest_opened": False,
        "test_feature_cache_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "test_arguments_supported": False,
        "test_token_paths_rejected": True,
    }
    model_payload = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "species_model": best_models[0],
        "grade_models": best_models[1],
        "feature_families": [family.name for family in train_features.families],
        "feature_schema": feature_schema,
        "view_mode": best_specification["view_mode"],
        "C": best_specification["C"],
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
        "validation_metrics_raw": best_specification["validation_metrics_raw"],
        "validation_metrics_calibrated": calibrated_metrics,
        "protocol": protocol,
        "training_weight_policy": {
            "species": (
                "species_weight * (1 / count(pair_id)), then pair-aware species mass balance"
            ),
            "grade": (
                "grade_weight * (1 / count(pair_id)), then pair-aware grade mass balance "
                "inside each species; rows with grade_weight=0 excluded"
            ),
            "pair_audit": pair_audit,
            "balancing_audit": balancing_audit,
        },
    }
    atomic_joblib_dump(model_payload, model_output)
    model_record = fingerprint(model_output)

    frozen_core = {
        "model": model_record,
        "view_mode": best_specification["view_mode"],
        "C": best_specification["C"],
        "feature_schema": feature_schema,
        "temperatures": model_payload["temperatures"],
    }
    selection = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "selection_id": f"pepper-clean-v5-{sha256_json(frozen_core)[:20]}",
        "protocol": protocol,
        "strict_test_manifest_opened": False,
        "input_fingerprints": {
            "script": fingerprint(Path(__file__)),
            "train_manifest": fingerprint(args.train_manifest),
            "validation_manifest": fingerprint(args.val_manifest),
            "train_features": train_features.files,
            "validation_features": val_features.files,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "leakage_audit": audit,
        "pair_weight_audit": pair_audit,
        "training_summary": {
            "train_rows": len(train_rows),
            "train_groups": len({row["group_id"] for row in train_rows}),
            "train_class_counts": {
                CLASS_NAMES[class_id]: int((labels == class_id).sum())
                for class_id in range(4)
            },
            "positive_species_weight_rows": int((species_weight > 0).sum()),
            "positive_grade_weight_rows": int((grade_weight > 0).sum()),
            "pair_normalized_species_confidence_sum_before_balance": float(
                species_weight_raw.sum()
            ),
            "pair_normalized_grade_confidence_sum_before_balance": float(
                grade_weight_raw.sum()
            ),
            "effective_species_weight_sum": float(species_weight.sum()),
            "effective_grade_weight_sum": float(grade_weight.sum()),
            "balancing_audit": balancing_audit,
        },
        "validation_summary": {
            "rows": len(val_rows),
            "groups": len({row["group_id"] for row in val_rows}),
            "class_counts": {
                CLASS_NAMES[class_id]: int((val_labels == class_id).sum())
                for class_id in range(4)
            },
        },
        "candidate_space": {
            "family": "hierarchical RBF-SVC",
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": None,
            "sample_weight_balance": (
                "pair-aware species balance and within-species grade balance"
            ),
            "probability": True,
            "random_seed": RANDOM_SEED,
            "view_modes": list(PREDECLARED_VIEW_MODES),
            "C_grid": list(PREDECLARED_C_GRID),
            "candidate_count": len(candidates),
            "grid_is_not_cli_configurable": True,
        },
        "candidates": candidates,
        "selected": {
            **frozen_core,
            "validation_metrics_raw": best_specification["validation_metrics_raw"],
            "temperature_fit_bounds": list(TEMPERATURE_BOUNDS),
            "temperature_fit_objectives": temperature_objectives,
            "validation_metrics_calibrated": calibrated_metrics,
        },
        "final_test_status": "not_run",
    }
    atomic_json_dump(selection, selection_output)
    selection_record = fingerprint(selection_output)
    receipt = {
        "schema": "pepper-clean-v5-selection-receipt-v1",
        "created_at_utc": utc_now(),
        "selection_id": selection["selection_id"],
        "selection": selection_record,
        "model": model_record,
        "strict_test_manifest_opened": False,
    }
    atomic_json_dump(receipt, receipt_output)
    print(
        json.dumps(
            {
                "model_saved": str(model_output),
                "selection_saved": str(selection_output),
                "selection_receipt_saved": str(receipt_output),
                "selection_sha256": selection_record["sha256"],
                "selection_id": selection["selection_id"],
                "selected_view_mode": best_specification["view_mode"],
                "selected_C": best_specification["C"],
                "validation_raw": best_specification["validation_metrics_raw"],
                "temperatures": model_payload["temperatures"],
                "validation_calibrated": calibrated_metrics,
                "strict_protocol": "NO TEST MANIFEST, FEATURE, LABEL OR METRIC WAS OPENED",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
