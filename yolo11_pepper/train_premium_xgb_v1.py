#!/usr/bin/env python3
"""Leakage-safe XGBoost selection for clean-v5 plus audited premium crops.

This executable is deliberately a *selection* program.  It has no strict-test
argument and rejects any path containing a complete ``test`` token.  Model
architecture and hyperparameters are selected with source-group out-of-fold
predictions from rows explicitly eligible for model selection.  Audited data
whose physical independence is not proven can be marked final-fit-only: it is
added only after selection is frozen.  The physically separated model-selection
split is opened only after selection, to report validation metrics and fit
post-hoc probability temperatures.

Two heads are compared:

* direct four-class XGBoost;
* hierarchical XGBoost: p(species) and two p(grade | species) branches.

Every physical source group receives total training mass one per supervised
task before class balancing.  Consequently a scene containing many detected
peppers cannot dominate a scene containing only a few peppers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Avoid the macOS OpenMP collision between PyTorch and XGBoost and keep the
# small-data benchmark deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import xgboost
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import train_select_clean_v5 as core


SCRIPT_VERSION = "pepper-premium-xgb-selection-v1"
PROJECT = Path(__file__).resolve().parent
CLASS_NAMES = core.CLASS_NAMES
SPECIES_NAMES = core.SPECIES_NAMES
RANDOM_SEED = 3061
MAX_CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 800

# Small, source-code-fixed search space.  It intentionally favours shallow,
# regularized trees because the feature width is much larger than the number
# of independent physical source groups.  The grid cannot be expanded through
# the CLI after looking at validation results.
PREDECLARED_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "d2_k512_conservative",
        "top_variance_features": 512,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5.0,
        "gamma": 0.10,
        "reg_alpha": 0.30,
        "reg_lambda": 8.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
    },
    {
        "name": "d2_k1024_conservative",
        "top_variance_features": 1024,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5.0,
        "gamma": 0.10,
        "reg_alpha": 0.30,
        "reg_lambda": 8.0,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
    },
    {
        "name": "d3_k512_strong_regularization",
        "top_variance_features": 512,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 8.0,
        "gamma": 0.20,
        "reg_alpha": 0.50,
        "reg_lambda": 10.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
    },
    {
        "name": "d3_k1024_balanced",
        "top_variance_features": 1024,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 5.0,
        "gamma": 0.15,
        "reg_alpha": 0.30,
        "reg_lambda": 8.0,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
    },
)
PREDECLARED_ARCHITECTURES = ("direct_four_class", "hierarchical")

DEFAULT_CLEAN_TRAIN_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
)
DEFAULT_CLEAN_TRAIN_FEATURE = (
    PROJECT
    / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt"
)
DEFAULT_VAL_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
)
DEFAULT_VAL_FEATURE = (
    PROJECT / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT / "runs/premium_xgb_v1"


@dataclass(frozen=True)
class LoadedBlock:
    manifest: Path
    feature_cache: Path
    rows: list[dict[str, Any]]
    features: np.ndarray
    feature_record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only source-group CV selection of direct and hierarchical "
            "XGBoost heads; strict test access is intentionally unsupported."
        )
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "One training-only manifest per feature cache. Repeat for clean and "
            "audited premium data. Defaults to the clean-v5 manifest only."
        ),
    )
    parser.add_argument(
        "--train-feature",
        type=Path,
        action="append",
        default=None,
        help="Feature cache matching each --train-manifest, in identical order.",
    )
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--val-feature", type=Path, default=DEFAULT_VAL_FEATURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def _read_manifest(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
    path = path.resolve()
    core.reject_test_path(path, f"{expected_split} manifest")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "split", "group_id", "class_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for line, raw in enumerate(reader, 2):
            split = str(raw.get("split") or "").strip().lower()
            if split != expected_split:
                raise ValueError(
                    f"{path}:{line} split={split!r}; expected only {expected_split!r}"
                )
            image = Path(str(raw.get("path") or "")).resolve()
            core.reject_test_path(image, f"{expected_split} image")
            if not image.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image}")
            group_id = str(raw.get("group_id") or "").strip()
            if not group_id:
                raise ValueError(f"{path}:{line} has an empty group_id")
            try:
                class_id = int(raw.get("class_id", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line} has invalid class_id") from error
            if class_id not in range(4):
                raise ValueError(f"{path}:{line} class_id={class_id} is outside [0,3]")

            def finite_weight(field: str, default: float) -> float:
                value = raw.get(field)
                result = default if value in (None, "") else float(value)
                if not math.isfinite(result) or result < 0:
                    raise ValueError(f"{path}:{line} invalid {field}={value!r}")
                return result

            def boolean(field: str, default: bool) -> bool:
                value = raw.get(field)
                if value in (None, ""):
                    return default
                normalized = str(value).strip().lower()
                if normalized in {"true", "1", "yes"}:
                    return True
                if normalized in {"false", "0", "no"}:
                    return False
                raise ValueError(f"{path}:{line} invalid {field}={value!r}")

            if "safe_for_training" in (reader.fieldnames or ()) and not boolean(
                "safe_for_training", False
            ):
                raise ValueError(f"{path}:{line} is not marked safe_for_training")

            record = dict(raw)
            record.update(
                {
                    "path": str(image),
                    "split": split,
                    "group_id": group_id,
                    "source_id": str(raw.get("source_id") or group_id).strip(),
                    "pair_id": str(raw.get("pair_id") or image).strip(),
                    "class_id": class_id,
                    "species_weight": finite_weight("species_weight", 1.0),
                    "grade_weight": finite_weight("grade_weight", 1.0),
                    # Premium data from one acquisition session is deliberately
                    # final-fit-only.  It must not be split image-by-image across
                    # CV folds and thereby masquerade as independent evidence.
                    "eligible_for_model_selection_bool": boolean(
                        "eligible_for_model_selection", True
                    ),
                    "dataset_origin": str(
                        raw.get("dataset_origin") or raw.get("origin") or path.stem
                    ).strip(),
                }
            )
            rows.append(record)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        duplicate = next(path for path, count in Counter(paths).items() if count > 1)
        raise ValueError(f"Duplicate image path inside {path}: {duplicate}")
    return rows


def _cache_manifest_matches(metadata: dict[str, Any], manifest: Path) -> bool:
    value = str(metadata.get("manifest") or "").strip()
    return bool(value) and Path(value).resolve() == manifest.resolve()


def _load_block(manifest: Path, cache: Path, *, split: str) -> LoadedBlock:
    manifest = manifest.resolve()
    cache = cache.resolve()
    core.reject_test_path(cache, f"{split} feature cache")
    rows = _read_manifest(manifest, expected_split=split)
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), torch.Tensor):
        raise TypeError(f"Feature cache has no tensor payload: {cache}")
    tensor = payload["features"].float().cpu()
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3 or tensor.shape[0] != len(rows):
        raise ValueError(
            f"Feature shape {tuple(tensor.shape)} does not match {len(rows)} rows: {cache}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Feature cache contains NaN/Inf: {cache}")
    cached_paths = [str(Path(value).resolve()) for value in payload.get("paths", [])]
    expected_paths = [row["path"] for row in rows]
    if cached_paths != expected_paths:
        raise ValueError(f"Feature paths differ from manifest order: {cache}")
    if [str(value) for value in payload.get("groups", [])] != [
        row["group_id"] for row in rows
    ]:
        raise ValueError(f"Feature groups differ from manifest: {cache}")
    cached_classes = payload.get("class_ids")
    expected_classes = torch.tensor(
        [row["class_id"] for row in rows], dtype=torch.long
    )
    if not isinstance(cached_classes, torch.Tensor) or not torch.equal(
        cached_classes.cpu().long(), expected_classes
    ):
        raise ValueError(f"Feature labels differ from manifest: {cache}")
    metadata = dict(payload.get("metadata") or {})
    if str(metadata.get("split") or "").lower() != split:
        raise ValueError(f"Feature split metadata is not {split}: {cache}")
    if bool(metadata.get("test_requested_explicitly", False)):
        raise ValueError(f"Feature cache was produced by a test-requesting run: {cache}")
    if not _cache_manifest_matches(metadata, manifest):
        raise ValueError(
            f"Feature metadata manifest does not match {manifest}: "
            f"{metadata.get('manifest')!r}"
        )
    if str(metadata.get("kind") or "").lower() != "cls":
        raise ValueError(f"Expected a YOLO classification feature cache: {cache}")
    if int(metadata.get("image_size") or 0) != 256:
        raise ValueError(f"Expected 256 px feature extraction: {cache}")

    # The first view is the deterministic canonical view.  Validation caches
    # contain exactly one view; augmented train views are not expanded, so each
    # image/source keeps its intended mass.
    canonical = F.normalize(tensor[:, 0], p=2, dim=1).numpy().astype(np.float32)
    record = core.fingerprint(cache)
    record.update(
        {
            "shape": list(tensor.shape),
            "canonical_shape": list(canonical.shape),
            "metadata": metadata,
            "manifest": core.fingerprint(manifest),
        }
    )
    return LoadedBlock(manifest, cache, rows, canonical, record)


def _overlap(
    first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]], field: str
) -> list[str]:
    left = {str(row.get(field) or "").strip() for row in first}
    right = {str(row.get(field) or "").strip() for row in second}
    return sorted((left & right) - {""})


def _leakage_audit(
    train_rows: Sequence[dict[str, Any]], val_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    fields = (
        "path",
        "group_id",
        "source_id",
        "pair_id",
        "content_sha256",
        "crop_sha256",
        "source_sha256",
    )
    overlap = {field: _overlap(train_rows, val_rows, field) for field in fields}
    # Empty hashes are ignored.  Source/group identity and exact crop hashes are
    # all checked when the manifest provides them.
    passed = not any(overlap.values())
    return {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "checked_fields": list(fields),
        "overlap": overlap,
        "passed": passed,
    }


def _training_integrity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    path_counts = Counter(str(row["path"]) for row in rows)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    digest_fields = ("content_sha256", "crop_sha256")
    duplicate_hashes: dict[str, list[str]] = {}
    for field in digest_fields:
        counts = Counter(str(row.get(field) or "").strip() for row in rows)
        counts.pop("", None)
        duplicate_hashes[field] = sorted(key for key, count in counts.items() if count > 1)
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_group[str(row["group_id"])].add(int(row["class_id"]))
    mixed_groups = {
        group: sorted(labels)
        for group, labels in labels_by_group.items()
        if len(labels) != 1
    }
    # A final-fit-only acquisition session may legitimately contain all four
    # folder labels.  Only the subset used for group CV must have pure-label
    # groups; _build_group_folds enforces that stronger requirement later.
    passed = not duplicate_paths and not any(duplicate_hashes.values())
    return {
        "passed": passed,
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": duplicate_hashes,
        "mixed_label_groups_allowed_only_outside_model_selection": mixed_groups,
        "groups": len(labels_by_group),
    }


def _build_group_folds(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_group[str(row["group_id"])].add(int(row["class_id"]))
    mixed = {group: values for group, values in labels_by_group.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"Physical source groups cross class labels: {mixed}")
    group_ids = np.asarray(sorted(labels_by_group), dtype=object)
    group_labels = np.asarray(
        [next(iter(labels_by_group[str(group)])) for group in group_ids],
        dtype=np.int64,
    )
    group_counts = np.bincount(group_labels, minlength=4)
    folds = min(MAX_CV_FOLDS, int(group_counts.min()))
    if folds < 3:
        raise ValueError(
            f"Need at least three source groups per class for group CV: {group_counts.tolist()}"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    row_groups = np.asarray([row["group_id"] for row in rows], dtype=object)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    public: list[dict[str, Any]] = []
    for fold_id, (fit_group_i, hold_group_i) in enumerate(
        splitter.split(group_ids, group_labels)
    ):
        fit_groups = set(group_ids[fit_group_i].tolist())
        hold_groups = set(group_ids[hold_group_i].tolist())
        fit = np.flatnonzero(np.isin(row_groups, list(fit_groups)))
        hold = np.flatnonzero(np.isin(row_groups, list(hold_groups)))
        if fit_groups & hold_groups:
            raise RuntimeError("Group CV construction leaked a source group")
        result.append((fit, hold))
        public.append(
            {
                "fold": fold_id,
                "fit_rows": int(len(fit)),
                "holdout_rows": int(len(hold)),
                "fit_groups": len(fit_groups),
                "holdout_groups": len(hold_groups),
                "holdout_group_class_counts": {
                    CLASS_NAMES[class_id]: int(
                        sum(group_labels[hold_group_i] == class_id)
                    )
                    for class_id in range(4)
                },
            }
        )
    return result, {
        "folds": folds,
        "group_class_counts": {
            CLASS_NAMES[class_id]: int(group_counts[class_id])
            for class_id in range(4)
        },
        "fold_audit": public,
    }


def _balanced_source_weights(
    rows: Sequence[dict[str, Any]],
    labels: np.ndarray,
    confidence: np.ndarray,
    indices: np.ndarray,
    *,
    classes: Sequence[int],
) -> np.ndarray:
    """Give each active source total mass one, then balance task classes."""
    result = np.zeros(len(rows), dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        if confidence[index] > 0:
            groups[str(rows[index]["group_id"])].append(index)
    for members in groups.values():
        mass = float(confidence[members].sum())
        if mass > 0:
            result[members] = confidence[members] / mass
    class_mass = {class_id: float(result[labels == class_id].sum()) for class_id in classes}
    if any(mass <= 0 for mass in class_mass.values()):
        raise ValueError(f"A supervised task class has zero mass: {class_mass}")
    target = sum(class_mass.values()) / len(classes)
    for class_id, mass in class_mass.items():
        result[labels == class_id] *= target / mass
    active = result > 0
    result[active] /= result[active].mean()
    return result.astype(np.float32)


def _variance_indices(matrix: np.ndarray, fit: np.ndarray, top_k: int) -> np.ndarray:
    count = min(int(top_k), matrix.shape[1])
    variance = matrix[fit].var(axis=0, dtype=np.float64)
    # Stable descending variance, with feature index resolving exact ties.
    order = np.lexsort((np.arange(len(variance)), -variance))
    return np.sort(order[:count]).astype(np.int64)


def _make_model(
    config: dict[str, Any],
    *,
    binary: bool,
    n_estimators: int,
    n_jobs: int,
    early_stopping: bool,
    seed_offset: int,
) -> XGBClassifier:
    kwargs: dict[str, Any] = {
        "objective": "binary:logistic" if binary else "multi:softprob",
        "eval_metric": "logloss" if binary else "mlogloss",
        "tree_method": "hist",
        "max_bin": 128,
        "n_estimators": int(n_estimators),
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "min_child_weight": float(config["min_child_weight"]),
        "gamma": float(config["gamma"]),
        "reg_alpha": float(config["reg_alpha"]),
        "reg_lambda": float(config["reg_lambda"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "random_state": RANDOM_SEED + seed_offset,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if not binary:
        kwargs["num_class"] = 4
    if early_stopping:
        kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**kwargs)


def _fit_early_stopped(
    model: XGBClassifier,
    x: np.ndarray,
    labels: np.ndarray,
    fit_weights: np.ndarray,
    hold_weights: np.ndarray,
    fit: np.ndarray,
    hold: np.ndarray,
) -> int:
    fit_active = fit[fit_weights[fit] > 0]
    hold_active = hold[hold_weights[hold] > 0]
    if len(fit_active) == 0 or len(hold_active) == 0:
        raise ValueError("Early-stopping fold has no active rows")
    model.fit(
        x[fit_active],
        labels[fit_active],
        sample_weight=fit_weights[fit_active],
        eval_set=[(x[hold_active], labels[hold_active])],
        sample_weight_eval_set=[hold_weights[hold_active]],
        verbose=False,
    )
    best = getattr(model, "best_iteration", None)
    return int(best) + 1 if best is not None else int(model.n_estimators)


def _joint_to_branches(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if joint.shape[1] != 4:
        raise ValueError(f"Expected four-class probabilities, got {joint.shape}")
    species = np.column_stack((joint[:, :2].sum(1), joint[:, 2:].sum(1)))
    grade = np.empty((len(joint), 2, 2), dtype=np.float64)
    grade[:, 0] = joint[:, :2] / np.clip(species[:, 0, None], 1e-15, None)
    grade[:, 1] = joint[:, 2:] / np.clip(species[:, 1, None], 1e-15, None)
    return species, grade


def _branches_to_joint(species: np.ndarray, grade: np.ndarray) -> np.ndarray:
    return (species[:, :, None] * grade).reshape(len(species), 4)


def _cv_direct(
    rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    *,
    n_jobs: int,
) -> tuple[np.ndarray, list[int]]:
    labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    confidence = np.asarray([row["grade_weight"] for row in rows], dtype=np.float64)
    oof = np.zeros((len(rows), 4), dtype=np.float64)
    estimators: list[int] = []
    for fold_id, (fit, hold) in enumerate(folds):
        fit_weights = _balanced_source_weights(
            rows, labels, confidence, fit, classes=(0, 1, 2, 3)
        )
        hold_weights = _balanced_source_weights(
            rows, labels, confidence, hold, classes=(0, 1, 2, 3)
        )
        selected = _variance_indices(
            matrix, fit, int(config["top_variance_features"])
        )
        x = matrix[:, selected]
        model = _make_model(
            config,
            binary=False,
            n_estimators=MAX_ESTIMATORS,
            n_jobs=n_jobs,
            early_stopping=True,
            seed_offset=fold_id,
        )
        estimators.append(
            _fit_early_stopped(
                model, x, labels, fit_weights, hold_weights, fit, hold
            )
        )
        oof[hold] = model.predict_proba(x[hold])
    if not np.allclose(oof.sum(1), 1.0, atol=1e-5):
        raise RuntimeError("Direct OOF probabilities are incomplete")
    return oof, estimators


def _cv_hierarchical(
    rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    *,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    final_labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    species_labels = final_labels // 2
    grade_labels = final_labels % 2
    species_confidence = np.asarray(
        [row["species_weight"] for row in rows], dtype=np.float64
    )
    grade_confidence = np.asarray(
        [row["grade_weight"] for row in rows], dtype=np.float64
    )
    species_oof = np.zeros((len(rows), 2), dtype=np.float64)
    grade_oof = np.zeros((len(rows), 2, 2), dtype=np.float64)
    iterations: dict[str, list[int]] = {
        "species": [],
        "grade_given_子弹头": [],
        "grade_given_条子": [],
    }
    for fold_id, (fit, hold) in enumerate(folds):
        selected = _variance_indices(
            matrix, fit, int(config["top_variance_features"])
        )
        x = matrix[:, selected]
        species_fit_weight = _balanced_source_weights(
            rows,
            species_labels,
            species_confidence,
            fit,
            classes=(0, 1),
        )
        species_hold_weight = _balanced_source_weights(
            rows,
            species_labels,
            species_confidence,
            hold,
            classes=(0, 1),
        )
        species_model = _make_model(
            config,
            binary=True,
            n_estimators=MAX_ESTIMATORS,
            n_jobs=n_jobs,
            early_stopping=True,
            seed_offset=fold_id,
        )
        iterations["species"].append(
            _fit_early_stopped(
                species_model,
                x,
                species_labels,
                species_fit_weight,
                species_hold_weight,
                fit,
                hold,
            )
        )
        species_oof[hold] = species_model.predict_proba(x[hold])

        for species_id, species_name in enumerate(SPECIES_NAMES):
            branch_indices = np.flatnonzero(species_labels == species_id)
            branch_fit = np.intersect1d(fit, branch_indices, assume_unique=True)
            branch_hold = np.intersect1d(hold, branch_indices, assume_unique=True)
            branch_fit_weight = _balanced_source_weights(
                rows,
                grade_labels,
                grade_confidence,
                branch_fit,
                classes=(0, 1),
            )
            branch_hold_weight = _balanced_source_weights(
                rows,
                grade_labels,
                grade_confidence,
                branch_hold,
                classes=(0, 1),
            )
            grade_model = _make_model(
                config,
                binary=True,
                n_estimators=MAX_ESTIMATORS,
                n_jobs=n_jobs,
                early_stopping=True,
                seed_offset=100 + 10 * species_id + fold_id,
            )
            key = f"grade_given_{species_name}"
            iterations[key].append(
                _fit_early_stopped(
                    grade_model,
                    x,
                    grade_labels,
                    branch_fit_weight,
                    branch_hold_weight,
                    branch_fit,
                    branch_hold,
                )
            )
            # A conditional branch is defined for every input, including inputs
            # whose true species is the other branch.
            grade_oof[hold, species_id] = grade_model.predict_proba(x[hold])
    if not np.allclose(species_oof.sum(1), 1.0, atol=1e-5) or not np.allclose(
        grade_oof.sum(2), 1.0, atol=1e-5
    ):
        raise RuntimeError("Hierarchical OOF probabilities are incomplete")
    return species_oof, grade_oof, iterations


def _iteration_summary(values: dict[str, list[int]] | list[int]) -> dict[str, Any]:
    if isinstance(values, list):
        return {
            "per_fold": values,
            "median_for_final_fit": int(max(1, round(float(np.median(values))))),
        }
    return {key: _iteration_summary(item) for key, item in values.items()}


def _fit_final_direct(
    rows: Sequence[dict[str, Any]],
    train_x: np.ndarray,
    val_x: np.ndarray,
    config: dict[str, Any],
    estimators: int,
    *,
    n_jobs: int,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray]:
    labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    confidence = np.asarray([row["grade_weight"] for row in rows], dtype=np.float64)
    indices = np.arange(len(rows))
    weights = _balanced_source_weights(
        rows, labels, confidence, indices, classes=(0, 1, 2, 3)
    )
    selected = _variance_indices(
        train_x, indices, int(config["top_variance_features"])
    )
    model = _make_model(
        config,
        binary=False,
        n_estimators=estimators,
        n_jobs=n_jobs,
        early_stopping=False,
        seed_offset=900,
    )
    active = weights > 0
    model.fit(train_x[active][:, selected], labels[active], sample_weight=weights[active])
    probability = model.predict_proba(val_x[:, selected]).astype(np.float64)
    return model, selected, probability


def _fit_final_hierarchical(
    rows: Sequence[dict[str, Any]],
    train_x: np.ndarray,
    val_x: np.ndarray,
    config: dict[str, Any],
    estimators: dict[str, int],
    *,
    n_jobs: int,
) -> tuple[XGBClassifier, list[XGBClassifier], np.ndarray, np.ndarray, np.ndarray]:
    final_labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    species_labels = final_labels // 2
    grade_labels = final_labels % 2
    species_confidence = np.asarray(
        [row["species_weight"] for row in rows], dtype=np.float64
    )
    grade_confidence = np.asarray(
        [row["grade_weight"] for row in rows], dtype=np.float64
    )
    all_indices = np.arange(len(rows))
    selected = _variance_indices(
        train_x, all_indices, int(config["top_variance_features"])
    )
    train_selected = train_x[:, selected]
    val_selected = val_x[:, selected]

    species_weight = _balanced_source_weights(
        rows, species_labels, species_confidence, all_indices, classes=(0, 1)
    )
    species_model = _make_model(
        config,
        binary=True,
        n_estimators=estimators["species"],
        n_jobs=n_jobs,
        early_stopping=False,
        seed_offset=901,
    )
    active = species_weight > 0
    species_model.fit(
        train_selected[active],
        species_labels[active],
        sample_weight=species_weight[active],
    )
    species_probability = species_model.predict_proba(val_selected).astype(np.float64)

    grade_models: list[XGBClassifier] = []
    grade_probability = np.empty((len(val_x), 2, 2), dtype=np.float64)
    for species_id, species_name in enumerate(SPECIES_NAMES):
        branch = all_indices[species_labels == species_id]
        weights = _balanced_source_weights(
            rows, grade_labels, grade_confidence, branch, classes=(0, 1)
        )
        model = _make_model(
            config,
            binary=True,
            n_estimators=estimators[f"grade_given_{species_name}"],
            n_jobs=n_jobs,
            early_stopping=False,
            seed_offset=910 + species_id,
        )
        active = weights > 0
        model.fit(
            train_selected[active],
            grade_labels[active],
            sample_weight=weights[active],
        )
        grade_models.append(model)
        grade_probability[:, species_id] = model.predict_proba(val_selected)
    return species_model, grade_models, selected, species_probability, grade_probability


def _temperature_scale_joint(
    probability: np.ndarray, labels: np.ndarray
) -> tuple[float, float, np.ndarray]:
    temperature, objective = core.fit_scalar_temperature(
        probability,
        labels,
        lower=core.TEMPERATURE_BOUNDS[0],
        upper=core.TEMPERATURE_BOUNDS[1],
    )
    calibrated = core.softmax_numpy(
        np.log(np.clip(probability, 1e-15, 1.0)) / temperature
    )
    return temperature, objective, calibrated


def _write_validation_predictions(
    path: Path,
    rows: Sequence[dict[str, Any]],
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    fieldnames = ["path", "group_id", "class_id", "class_name"]
    for architecture in PREDECLARED_ARCHITECTURES:
        for state in ("raw", "calibrated"):
            fieldnames.extend(
                f"{architecture}_{state}_p_{class_name}"
                for class_name in CLASS_NAMES
            )
            fieldnames.append(f"{architecture}_{state}_prediction")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output: dict[str, Any] = {
                "path": row["path"],
                "group_id": row["group_id"],
                "class_id": row["class_id"],
                "class_name": CLASS_NAMES[int(row["class_id"])],
            }
            for architecture in PREDECLARED_ARCHITECTURES:
                for state, matrix in zip(
                    ("raw", "calibrated"), probabilities[architecture]
                ):
                    for class_id, class_name in enumerate(CLASS_NAMES):
                        output[f"{architecture}_{state}_p_{class_name}"] = (
                            f"{float(matrix[index, class_id]):.10f}"
                        )
                    output[f"{architecture}_{state}_prediction"] = int(
                        matrix[index].argmax()
                    )
            writer.writerow(output)


def main() -> None:
    args = parse_args()
    train_manifests = args.train_manifest or [DEFAULT_CLEAN_TRAIN_MANIFEST]
    train_features = args.train_feature or [DEFAULT_CLEAN_TRAIN_FEATURE]
    if len(train_manifests) != len(train_features):
        raise ValueError("Repeat --train-manifest and --train-feature equally")
    output_dir = args.output_dir.resolve()
    for path in (
        *train_manifests,
        *train_features,
        args.val_manifest,
        args.val_feature,
        output_dir,
    ):
        core.reject_test_path(Path(path).resolve(), "selection input/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "model": output_dir / "best_premium_xgb_v1.joblib",
        "selection": output_dir / "selection.json",
        "predictions": output_dir / "validation_predictions.csv",
        "receipt": output_dir / "sha256_receipt.json",
    }
    for path in artifact_paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")

    blocks = [
        _load_block(manifest, cache, split="train")
        for manifest, cache in zip(train_manifests, train_features)
    ]
    val_block = _load_block(args.val_manifest, args.val_feature, split="val")
    feature_dims = {block.features.shape[1] for block in (*blocks, val_block)}
    if len(feature_dims) != 1:
        raise ValueError(f"Feature dimensions differ across caches: {feature_dims}")
    train_rows = [row for block in blocks for row in block.rows]
    train_x = np.concatenate([block.features for block in blocks], axis=0)
    val_rows = val_block.rows
    val_x = val_block.features

    integrity = _training_integrity(train_rows)
    if not integrity["passed"]:
        raise ValueError(f"Combined training integrity check failed: {integrity}")
    leakage = _leakage_audit(train_rows, val_rows)
    if not leakage["passed"]:
        raise ValueError(f"Train/validation leakage check failed: {leakage}")
    selection_indices = np.asarray(
        [
            index
            for index, row in enumerate(train_rows)
            if bool(row["eligible_for_model_selection_bool"])
        ],
        dtype=np.int64,
    )
    final_fit_only_indices = np.asarray(
        [
            index
            for index, row in enumerate(train_rows)
            if not bool(row["eligible_for_model_selection_bool"])
        ],
        dtype=np.int64,
    )
    if len(selection_indices) == 0:
        raise ValueError("No training rows are eligible for model selection")
    selection_rows = [train_rows[index] for index in selection_indices]
    selection_x = train_x[selection_indices]
    folds, fold_audit = _build_group_folds(selection_rows)

    candidates: list[dict[str, Any]] = []
    candidate_state: dict[tuple[str, str], dict[str, Any]] = {}
    for architecture in PREDECLARED_ARCHITECTURES:
        for config in PREDECLARED_CONFIGS:
            if architecture == "direct_four_class":
                joint, iterations = _cv_direct(
                    selection_rows,
                    selection_x,
                    folds,
                    config,
                    n_jobs=args.n_jobs,
                )
                species, grade = _joint_to_branches(joint)
            else:
                species, grade, iterations = _cv_hierarchical(
                    selection_rows,
                    selection_x,
                    folds,
                    config,
                    n_jobs=args.n_jobs,
                )
                joint = _branches_to_joint(species, grade)
            metrics = core.branch_metrics(selection_rows, species, grade)
            item = {
                "architecture": architecture,
                "config": config,
                "train_group_oof_metrics": metrics,
                "early_stopping": _iteration_summary(iterations),
            }
            candidates.append(item)
            candidate_state[(architecture, config["name"])] = {
                "joint": joint,
                "species": species,
                "grade": grade,
                "iterations": iterations,
            }
            print(
                json.dumps(
                    {
                        "architecture": architecture,
                        "config": config["name"],
                        "oof_macro_f1": metrics["joint_macro_f1"],
                        "oof_accuracy": metrics["joint_accuracy"],
                    },
                    ensure_ascii=False,
                )
            )

    # Select each architecture by source-group OOF macro-F1 then accuracy.  Exact
    # ties keep the earlier predeclared, more conservative configuration.
    winners: dict[str, dict[str, Any]] = {}
    for architecture in PREDECLARED_ARCHITECTURES:
        pool = [row for row in candidates if row["architecture"] == architecture]
        winners[architecture] = max(
            enumerate(pool),
            key=lambda item: (
                item[1]["train_group_oof_metrics"]["joint_macro_f1"],
                item[1]["train_group_oof_metrics"]["joint_accuracy"],
                -item[0],
            ),
        )[1]
    # Only this architecture becomes eligible for the later one-shot strict
    # test evaluator.  Validation results below do not alter the decision.
    architecture_winner = max(
        enumerate(PREDECLARED_ARCHITECTURES),
        key=lambda item: (
            winners[item[1]]["train_group_oof_metrics"]["joint_macro_f1"],
            winners[item[1]]["train_group_oof_metrics"]["joint_accuracy"],
            -item[0],
        ),
    )[1]

    val_labels = np.asarray([row["class_id"] for row in val_rows], dtype=np.int64)
    fitted: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    probability_output: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    direct_winner = winners["direct_four_class"]
    direct_config = direct_winner["config"]
    direct_state = candidate_state[("direct_four_class", direct_config["name"])]
    direct_iteration = int(
        max(1, round(float(np.median(direct_state["iterations"]))))
    )
    direct_model, direct_features, direct_raw = _fit_final_direct(
        train_rows,
        train_x,
        val_x,
        direct_config,
        direct_iteration,
        n_jobs=args.n_jobs,
    )
    direct_species_raw, direct_grade_raw = _joint_to_branches(direct_raw)
    direct_raw_metrics = core.branch_metrics(
        val_rows, direct_species_raw, direct_grade_raw
    )
    direct_temperature, direct_temperature_nll, direct_calibrated = (
        _temperature_scale_joint(direct_raw, val_labels)
    )
    direct_species_cal, direct_grade_cal = _joint_to_branches(direct_calibrated)
    direct_cal_metrics = core.branch_metrics(
        val_rows, direct_species_cal, direct_grade_cal
    )
    validation["direct_four_class"] = {
        "raw": direct_raw_metrics,
        "calibrated": direct_cal_metrics,
        "temperature": direct_temperature,
        "temperature_fit_nll": direct_temperature_nll,
    }
    probability_output["direct_four_class"] = (direct_raw, direct_calibrated)
    fitted["direct_four_class"] = {
        "model": direct_model,
        "feature_indices": direct_features,
        "n_estimators": direct_iteration,
        "temperature": direct_temperature,
    }

    hierarchy_winner = winners["hierarchical"]
    hierarchy_config = hierarchy_winner["config"]
    hierarchy_state = candidate_state[("hierarchical", hierarchy_config["name"])]
    hierarchy_estimators = {
        key: int(max(1, round(float(np.median(values)))))
        for key, values in hierarchy_state["iterations"].items()
    }
    (
        species_model,
        grade_models,
        hierarchy_features,
        hierarchy_species_raw,
        hierarchy_grade_raw,
    ) = _fit_final_hierarchical(
        train_rows,
        train_x,
        val_x,
        hierarchy_config,
        hierarchy_estimators,
        n_jobs=args.n_jobs,
    )
    hierarchy_raw = _branches_to_joint(hierarchy_species_raw, hierarchy_grade_raw)
    hierarchy_raw_metrics = core.branch_metrics(
        val_rows, hierarchy_species_raw, hierarchy_grade_raw
    )
    hierarchy_temperatures, hierarchy_temperature_nll = core.fit_temperatures(
        val_rows, hierarchy_species_raw, hierarchy_grade_raw
    )
    hierarchy_species_cal, hierarchy_grade_cal = core.apply_temperatures(
        hierarchy_species_raw, hierarchy_grade_raw, hierarchy_temperatures
    )
    hierarchy_calibrated = _branches_to_joint(
        hierarchy_species_cal, hierarchy_grade_cal
    )
    hierarchy_cal_metrics = core.branch_metrics(
        val_rows, hierarchy_species_cal, hierarchy_grade_cal
    )
    validation["hierarchical"] = {
        "raw": hierarchy_raw_metrics,
        "calibrated": hierarchy_cal_metrics,
        "temperatures": {
            "species": hierarchy_temperatures[0],
            "grade_given_子弹头": hierarchy_temperatures[1],
            "grade_given_条子": hierarchy_temperatures[2],
        },
        "temperature_fit_nll": hierarchy_temperature_nll,
    }
    probability_output["hierarchical"] = (
        hierarchy_raw,
        hierarchy_calibrated,
    )
    fitted["hierarchical"] = {
        "species_model": species_model,
        "grade_models": grade_models,
        "feature_indices": hierarchy_features,
        "n_estimators": hierarchy_estimators,
        "temperatures": hierarchy_temperatures,
    }

    _write_validation_predictions(
        artifact_paths["predictions"], val_rows, probability_output
    )
    model_payload = {
        "schema": SCRIPT_VERSION,
        "selected_architecture_for_future_strict_test": architecture_winner,
        "feature_preprocessing": {
            "backbone": "YOLO11 ImageNet classification checkpoint",
            "image_size": 256,
            "view": "canonical view index 0",
            "normalization": "row-wise L2",
            "feature_dim_before_variance_selection": int(train_x.shape[1]),
        },
        "class_names": CLASS_NAMES,
        "species_names": SPECIES_NAMES,
        "winners": winners,
        "models": fitted,
        "protocol": {
            "architecture_and_hyperparameter_selection": (
                "train-only physical-source-group OOF macro-F1, then accuracy"
            ),
            "validation_role": "one-time reporting and post-hoc temperature calibration",
            "validation_did_not_select_architecture_or_hyperparameters": True,
            "strict_test_opened": False,
        },
    }
    core.atomic_joblib_dump(model_payload, artifact_paths["model"])

    origin_counts = Counter(str(row["dataset_origin"]) for row in train_rows)
    group_origin: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        group_origin[str(row["dataset_origin"])].add(str(row["group_id"]))
    selection = {
        "schema": SCRIPT_VERSION,
        "protocol": {
            "selection_scope": "combined training manifests only",
            "selection_metric": "source-group OOF four-class macro-F1, then accuracy",
            "exact_tie_policy": "predeclared architecture/config order",
            "validation_opened_after_selection_frozen": True,
            "temperature_fit_data": str(args.val_manifest.resolve()),
            "strict_test_manifest_opened": False,
            "strict_test_feature_opened": False,
            "strict_test_labels_read": False,
            "strict_test_metrics_computed": False,
            "strict_test_arguments_supported": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "input_fingerprints": {
            "script": core.fingerprint(Path(__file__)),
            "train_blocks": [block.feature_record for block in blocks],
            "validation_block": val_block.feature_record,
        },
        "training_summary": {
            "rows": len(train_rows),
            "groups": len({row["group_id"] for row in train_rows}),
            "model_selection_rows": int(len(selection_indices)),
            "model_selection_groups": len(
                {train_rows[index]["group_id"] for index in selection_indices}
            ),
            "final_fit_only_rows": int(len(final_fit_only_indices)),
            "final_fit_only_groups": len(
                {train_rows[index]["group_id"] for index in final_fit_only_indices}
            ),
            "final_fit_only_policy": (
                "included after architecture/hyperparameters are frozen; excluded "
                "from OOF selection because physical independence is not proven"
            ),
            "class_rows": {
                CLASS_NAMES[class_id]: int(
                    sum(int(row["class_id"]) == class_id for row in train_rows)
                )
                for class_id in range(4)
            },
            "class_groups": fold_audit["group_class_counts"],
            "origin_rows": dict(origin_counts),
            "origin_groups": {
                origin: len(groups) for origin, groups in group_origin.items()
            },
            "positive_species_weight_rows": int(
                sum(float(row["species_weight"]) > 0 for row in train_rows)
            ),
            "positive_grade_weight_rows": int(
                sum(float(row["grade_weight"]) > 0 for row in train_rows)
            ),
            "source_mass_policy": (
                "within each supervised task, confidence weights sum to one per "
                "physical group, followed by class-mass balancing"
            ),
        },
        "training_integrity": integrity,
        "train_validation_leakage_audit": leakage,
        "group_cv": fold_audit,
        "candidate_space": {
            "architectures": list(PREDECLARED_ARCHITECTURES),
            "configs": list(PREDECLARED_CONFIGS),
            "candidate_count": len(candidates),
            "max_estimators": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "fixed_in_source_code": True,
        },
        "candidates": candidates,
        "architecture_winners_from_train_oof": winners,
        "selected_architecture_for_future_strict_test": architecture_winner,
        "validation": validation,
        "validation_probabilities": core.fingerprint(artifact_paths["predictions"]),
        "model": core.fingerprint(artifact_paths["model"]),
        "final_test_status": "not_run",
    }
    core.atomic_json_dump(selection, artifact_paths["selection"])
    receipt = {
        "schema": "pepper-premium-xgb-receipt-v1",
        "selection": core.fingerprint(artifact_paths["selection"]),
        "model": core.fingerprint(artifact_paths["model"]),
        "validation_predictions": core.fingerprint(artifact_paths["predictions"]),
        "strict_test_opened": False,
    }
    core.atomic_json_dump(receipt, artifact_paths["receipt"])
    print(
        json.dumps(
            {
                "selected_architecture_for_future_strict_test": architecture_winner,
                "architecture_winners": {
                    key: value["config"]["name"] for key, value in winners.items()
                },
                "validation": validation,
                "model": str(artifact_paths["model"]),
                "selection": str(artifact_paths["selection"]),
                "strict_protocol": "NO STRICT TEST FILE, LABEL, OR METRIC WAS OPENED",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
