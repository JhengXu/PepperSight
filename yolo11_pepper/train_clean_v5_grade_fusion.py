#!/usr/bin/env python3
"""Select clean-v5 conditional-grade feature fusion without held-out test access.

The species head always consumes the canonical ImageNet-256 feature only.  Each
conditional grade head independently chooses an RBF-SVC C and a fixed weight for
handcrafted quality features.  All choices are made with five-fold, source-group
stratified cross-validation inside the training split.  The physically separated
model-selection split is opened only after those choices are frozen; it is used
once for validation reporting and scalar temperature calibration.

This executable intentionally exposes no final-test argument or evaluation mode.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_select_clean_v5 as core


SCRIPT_VERSION = "pepper-clean-v5-grade-fusion-selection-v1"
PROJECT = Path(__file__).resolve().parent
RANDOM_SEED = 2161
CV_FOLDS = 5

# These grids are deliberately constants rather than CLI flags.  They are small
# enough to keep the source-group CV search auditable and resistant to tuning on
# the physical validation split.
PREDECLARED_C_GRID = (1.0, 3.0, 10.0, 30.0)
PREDECLARED_QUALITY_ALPHA_GRID = (0.0, 0.25, 0.5, 1.0)
# A positive value adds evidence for grade index 1 (二级).  Zero comes first so
# exact metric ties retain the unadjusted posterior; later ties prefer less bias.
PREDECLARED_GRADE_LOGIT_BIAS_GRID = (0.0, -0.25, 0.25, -0.5, 0.5)

DEFAULT_TRAIN_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
)
DEFAULT_VAL_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
)
DEFAULT_DEEP_TRAIN = (
    PROJECT
    / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt"
)
DEFAULT_DEEP_VAL = (
    PROJECT / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt"
)
DEFAULT_QUALITY_TRAIN = (
    PROJECT / "runs/hierarchical_v5_clean/features_quality/quality_train.pt"
)
DEFAULT_QUALITY_VAL = (
    PROJECT / "runs/hierarchical_v5_clean/features_quality/quality_val.pt"
)
DEFAULT_QUALITY_TRAIN_SOURCE_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/train_manifest.csv"
)
DEFAULT_BASELINE_SELECTION = (
    PROJECT / "runs/hierarchical_v5_clean/selection_reaudit.json"
)
DEFAULT_MODEL_OUTPUT = (
    PROJECT / "runs/hierarchical_v5_clean/best_clean_v5_grade_fusion_svm.joblib"
)
DEFAULT_SELECTION_OUTPUT = (
    PROJECT / "runs/hierarchical_v5_clean/selection_grade_fusion.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-group CV selection of ImageNet-256 + handcrafted-quality "
            "conditional grade heads; no final-test interface exists."
        )
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--deep-train", type=Path, default=DEFAULT_DEEP_TRAIN)
    parser.add_argument("--deep-val", type=Path, default=DEFAULT_DEEP_VAL)
    parser.add_argument("--quality-train", type=Path, default=DEFAULT_QUALITY_TRAIN)
    parser.add_argument("--quality-val", type=Path, default=DEFAULT_QUALITY_VAL)
    parser.add_argument(
        "--quality-train-source-manifest",
        type=Path,
        default=DEFAULT_QUALITY_TRAIN_SOURCE_MANIFEST,
        help=(
            "Original physical train manifest recorded by the label-independent "
            "quality cache.  Paths/classes/groups must still match the audited manifest."
        ),
    )
    parser.add_argument(
        "--baseline-selection", type=Path, default=DEFAULT_BASELINE_SELECTION
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT
    )
    return parser.parse_args()


def _make_svc(c_value: float, *, probability: bool) -> SVC:
    return SVC(
        C=c_value,
        kernel="rbf",
        gamma="scale",
        class_weight=None,
        probability=probability,
        random_state=RANDOM_SEED,
    )


def _load_quality_cache(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    expected_split: str,
    recorded_manifest: Path,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Load label-independent quality features with strict row/provenance checks."""
    path = path.resolve()
    recorded_manifest = recorded_manifest.resolve()
    core.reject_test_path(path, f"{expected_split} quality cache")
    core.reject_test_path(recorded_manifest, f"{expected_split} quality source manifest")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Quality cache is not a dictionary: {path}")
    feature = payload.get("features")
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"Quality cache lacks features tensor: {path}")
    if feature.ndim == 3 and feature.shape[1] == 1:
        feature = feature[:, 0]
    if feature.ndim != 2 or feature.shape[0] != len(rows):
        raise ValueError(
            f"Quality shape {tuple(feature.shape)} does not match {len(rows)} rows"
        )
    feature = feature.float().cpu().numpy()
    if not np.isfinite(feature).all():
        raise ValueError(f"Quality cache contains NaN/Inf: {path}")

    expected_paths = [str(Path(row["path"]).resolve()) for row in rows]
    cached_paths = [str(Path(value).resolve()) for value in payload.get("paths", [])]
    if cached_paths != expected_paths:
        raise ValueError(f"Quality feature path order differs from manifest: {path}")
    expected_groups = [str(row["group_id"]) for row in rows]
    if [str(value) for value in payload.get("groups", [])] != expected_groups:
        raise ValueError(f"Quality feature groups differ from manifest: {path}")
    expected_classes = torch.tensor(
        [int(row["class_id"]) for row in rows], dtype=torch.long
    )
    cached_classes = payload.get("class_ids")
    if not isinstance(cached_classes, torch.Tensor) or not torch.equal(
        cached_classes.cpu().long(), expected_classes
    ):
        raise ValueError(f"Quality feature classes differ from manifest: {path}")

    metadata = payload.get("metadata") or {}
    if str(metadata.get("split", "")).strip().lower() != expected_split:
        raise ValueError(f"Quality cache split differs from {expected_split}: {path}")
    if bool(metadata.get("test_requested_explicitly", False)):
        raise ValueError(f"Quality cache was made by a held-out-requesting run: {path}")
    metadata_manifest = Path(str(metadata.get("manifest") or "")).resolve()
    if metadata_manifest != recorded_manifest:
        raise ValueError(
            f"Quality cache manifest differs from recorded physical manifest: "
            f"{metadata_manifest} != {recorded_manifest}"
        )
    manifest_digest = str(metadata.get("manifest_sha256") or "").lower()
    current_manifest_digest = core.sha256_file(recorded_manifest)
    if manifest_digest != current_manifest_digest:
        raise ValueError(
            f"Quality source-manifest digest mismatch: {manifest_digest} != "
            f"{current_manifest_digest}"
        )
    checkpoint_digest, checkpoint_attestation = core.checkpoint_digest(
        str(metadata.get("checkpoint") or ""),
        cache_path=path,
        metadata_digest=str(metadata.get("checkpoint_sha256") or ""),
    )
    feature_names = [str(value) for value in metadata.get("feature_names", [])]
    if len(feature_names) != feature.shape[1]:
        raise ValueError(
            f"Quality feature-name count differs from width in {path}: "
            f"{len(feature_names)} != {feature.shape[1]}"
        )
    record = core.fingerprint(path)
    record.update(
        {
            "shape": list(feature.shape),
            "algorithm": str(metadata.get("algorithm") or ""),
            "kind": str(metadata.get("kind") or ""),
            "feature_dim": int(feature.shape[1]),
            "feature_names_sha256": core.sha256_json(feature_names),
            "recorded_physical_manifest": core.fingerprint(recorded_manifest),
            "row_order_reverified_against_current_audited_manifest": True,
            "feature_values_are_label_independent": True,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_hash_attestation": checkpoint_attestation,
        }
    )
    return feature, record, feature_names


def _deep_canonical(feature_set: core.LoadedFeatureSet) -> np.ndarray:
    if len(feature_set.blocks) != 1:
        raise ValueError("Fusion protocol expects exactly one ImageNet feature family")
    family = feature_set.families[0]
    if family.name != "imagenet_cls" or family.image_size != 256:
        raise ValueError(
            f"Fusion protocol requires imagenet_cls at 256 px, got {family.public()}"
        )
    matrix = F.normalize(feature_set.blocks[0][:, 0], p=2, dim=-1)
    return matrix.cpu().numpy().astype(np.float32, copy=False)


def _normalized_scaled_quality(
    scaler: StandardScaler, matrix: np.ndarray
) -> np.ndarray:
    scaled = scaler.transform(matrix).astype(np.float32, copy=False)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.clip(norms, 1e-12, None)


def _fused_matrix(
    deep: np.ndarray,
    quality: np.ndarray,
    *,
    scaler: StandardScaler | None,
    alpha: float,
) -> np.ndarray:
    if alpha == 0.0:
        return deep
    if scaler is None:
        raise ValueError("A fitted StandardScaler is required when alpha > 0")
    quality_block = _normalized_scaled_quality(scaler, quality)
    return np.concatenate([deep, quality_block * alpha], axis=1)


def _build_group_folds(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    label_by_group: dict[str, set[int]] = defaultdict(set)
    pair_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label_by_group[str(row["group_id"])].add(int(row["class_id"]))
        pair_groups[str(row["pair_id"])].add(str(row["group_id"]))
    mixed = {group: labels for group, labels in label_by_group.items() if len(labels) != 1}
    if mixed:
        raise ValueError(f"Source groups cross class labels: {dict(list(mixed.items())[:5])}")
    crossed_pairs = [pair for pair, groups in pair_groups.items() if len(groups) != 1]
    if crossed_pairs:
        raise ValueError(f"Pairs cross source groups: {crossed_pairs[:5]}")

    group_ids = np.asarray(sorted(label_by_group), dtype=object)
    group_labels = np.asarray(
        [next(iter(label_by_group[str(group)])) for group in group_ids],
        dtype=np.int64,
    )
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    row_groups = np.asarray([str(row["group_id"]) for row in rows], dtype=object)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_audit: list[dict[str, Any]] = []
    for fold_id, (fit_group_index, hold_group_index) in enumerate(
        splitter.split(group_ids, group_labels)
    ):
        fit_groups = set(group_ids[fit_group_index].tolist())
        hold_groups = set(group_ids[hold_group_index].tolist())
        if fit_groups & hold_groups:
            raise RuntimeError("Group overlap inside CV split")
        fit_index = np.flatnonzero(np.isin(row_groups, list(fit_groups)))
        hold_index = np.flatnonzero(np.isin(row_groups, list(hold_groups)))
        folds.append((fit_index, hold_index))
        fit_pairs = {str(rows[index]["pair_id"]) for index in fit_index}
        hold_pairs = {str(rows[index]["pair_id"]) for index in hold_index}
        if fit_pairs & hold_pairs:
            raise RuntimeError("Pair overlap inside CV split")
        fold_audit.append(
            {
                "fold": fold_id,
                "fit_rows": len(fit_index),
                "hold_rows": len(hold_index),
                "fit_groups": len(fit_groups),
                "hold_groups": len(hold_groups),
                "fit_group_class_counts": dict(
                    Counter(core.CLASS_NAMES[value] for value in group_labels[fit_group_index])
                ),
                "hold_group_class_counts": dict(
                    Counter(core.CLASS_NAMES[value] for value in group_labels[hold_group_index])
                ),
                "group_overlap": [],
                "pair_overlap": [],
            }
        )
    return folds, {
        "method": "StratifiedKFold over unique source_group IDs",
        "folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "groups": len(group_ids),
        "group_class_counts": dict(Counter(core.CLASS_NAMES[value] for value in group_labels)),
        "fold_audit": fold_audit,
        "all_rows_from_each_pair_and_group_stay_in_one_fold": True,
    }


def _balanced_fit_weights(
    labels: np.ndarray,
    raw_weight: np.ndarray,
    fit_index: np.ndarray,
    *,
    names: Sequence[str],
) -> np.ndarray:
    balanced, _ = core.pair_aware_balanced_weight(
        labels[fit_index], raw_weight[fit_index], label_names=names
    )
    return balanced


def _weighted_binary_metrics(
    truth: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    return {
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1],
                average="macro",
                sample_weight=weight,
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(truth, prediction, sample_weight=weight)),
    }


def _apply_grade_logit_bias(
    probability: np.ndarray, logit_bias: float
) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-15, 1.0))
    logits[:, 1] += logit_bias
    return core.softmax_numpy(logits)


def _select_species_c(
    deep: np.ndarray,
    rows: Sequence[dict[str, Any]],
    labels: np.ndarray,
    species_raw_weight: np.ndarray,
    species_confidence: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[dict[str, Any]]]:
    species_labels = labels // 2
    canonical = np.asarray([row["view_type"] == "canonical" for row in rows])
    candidates: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    selected = PREDECLARED_C_GRID[0]
    for c_value in PREDECLARED_C_GRID:
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold_id, (fit_index, hold_index) in enumerate(folds):
            fit_keep = fit_index[species_raw_weight[fit_index] > 0]
            fit_weight = _balanced_fit_weights(
                species_labels,
                species_raw_weight,
                fit_keep,
                names=core.SPECIES_NAMES,
            )
            model = _make_svc(c_value, probability=False)
            model.fit(deep[fit_keep], species_labels[fit_keep], sample_weight=fit_weight)
            score_keep = hold_index[
                canonical[hold_index] & (species_confidence[hold_index] > 0)
            ]
            prediction = model.predict(deep[score_keep]).astype(np.int64)
            truth = species_labels[score_keep]
            weight = species_confidence[score_keep]
            metrics = _weighted_binary_metrics(truth, prediction, weight)
            fold_metrics.append({"fold": fold_id, "rows": len(score_keep), **metrics})
            truths.append(truth)
            predictions.append(prediction)
            weights.append(weight)
        pooled = _weighted_binary_metrics(
            np.concatenate(truths), np.concatenate(predictions), np.concatenate(weights)
        )
        candidate = {"C": c_value, "pooled": pooled, "fold_metrics": fold_metrics}
        candidates.append(candidate)
        key = (pooled["macro_f1"], pooled["accuracy"])
        if key > best_key:
            best_key = key
            selected = c_value
    return selected, candidates


def _select_grade_head(
    species_id: int,
    deep: np.ndarray,
    quality: np.ndarray,
    rows: Sequence[dict[str, Any]],
    labels: np.ndarray,
    grade_raw_weight: np.ndarray,
    grade_confidence: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    species_labels = labels // 2
    grade_labels = labels % 2
    canonical = np.asarray([row["view_type"] == "canonical" for row in rows])
    candidates: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    selected = {
        "alpha": PREDECLARED_QUALITY_ALPHA_GRID[0],
        "C": PREDECLARED_C_GRID[0],
        "logit_bias": PREDECLARED_GRADE_LOGIT_BIAS_GRID[0],
    }
    for alpha in PREDECLARED_QUALITY_ALPHA_GRID:
        for c_value in PREDECLARED_C_GRID:
            truths: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            fold_chunks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            for fold_id, (fit_index, hold_index) in enumerate(folds):
                fit_keep = fit_index[
                    (species_labels[fit_index] == species_id)
                    & (grade_raw_weight[fit_index] > 0)
                ]
                fit_weight = _balanced_fit_weights(
                    grade_labels,
                    grade_raw_weight,
                    fit_keep,
                    names=("一级", "二级"),
                )
                scaler = None
                if alpha > 0:
                    scaler = StandardScaler().fit(quality[fit_keep])
                fit_x = _fused_matrix(
                    deep[fit_keep], quality[fit_keep], scaler=scaler, alpha=alpha
                )
                model = _make_svc(c_value, probability=True)
                model.fit(fit_x, grade_labels[fit_keep], sample_weight=fit_weight)
                score_keep = hold_index[
                    (species_labels[hold_index] == species_id)
                    & canonical[hold_index]
                    & (grade_confidence[hold_index] > 0)
                ]
                score_x = _fused_matrix(
                    deep[score_keep], quality[score_keep], scaler=scaler, alpha=alpha
                )
                probability = core.ordered_binary_probability(model, score_x)
                truth = grade_labels[score_keep]
                weight = grade_confidence[score_keep]
                truths.append(truth)
                probabilities.append(probability)
                weights.append(weight)
                fold_chunks.append((fold_id, truth, probability, weight))
            pooled_truth = np.concatenate(truths)
            pooled_probability = np.concatenate(probabilities)
            pooled_weight = np.concatenate(weights)
            for logit_bias in PREDECLARED_GRADE_LOGIT_BIAS_GRID:
                biased_probability = _apply_grade_logit_bias(
                    pooled_probability, logit_bias
                )
                pooled = _weighted_binary_metrics(
                    pooled_truth, biased_probability.argmax(1), pooled_weight
                )
                fold_metrics: list[dict[str, Any]] = []
                for fold_id, truth, probability, weight in fold_chunks:
                    prediction = _apply_grade_logit_bias(
                        probability, logit_bias
                    ).argmax(1)
                    metrics = _weighted_binary_metrics(truth, prediction, weight)
                    fold_metrics.append(
                        {"fold": fold_id, "rows": len(truth), **metrics}
                    )
                candidate = {
                    "alpha": alpha,
                    "C": c_value,
                    "logit_bias": logit_bias,
                    "pooled": pooled,
                    "fold_metrics": fold_metrics,
                }
                candidates.append(candidate)
                key = (pooled["macro_f1"], pooled["accuracy"])
                if key > best_key:
                    best_key = key
                    selected = {
                        "alpha": alpha,
                        "C": c_value,
                        "logit_bias": logit_bias,
                    }
    return selected, candidates


def _fit_full_weights(
    labels: np.ndarray,
    pair_base_weight: np.ndarray,
    species_confidence: np.ndarray,
    grade_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    species_labels = labels // 2
    grade_labels = labels % 2
    species_raw = pair_base_weight * species_confidence
    grade_raw = pair_base_weight * grade_confidence
    species_weight, species_audit = core.pair_aware_balanced_weight(
        species_labels, species_raw, label_names=core.SPECIES_NAMES
    )
    grade_weight = np.zeros_like(grade_raw)
    grade_audits: dict[str, Any] = {}
    for species_id, species_name in enumerate(core.SPECIES_NAMES):
        branch = species_labels == species_id
        branch_weight, branch_audit = core.pair_aware_balanced_weight(
            grade_labels[branch], grade_raw[branch], label_names=("一级", "二级")
        )
        grade_weight[branch] = branch_weight
        grade_audits[f"grade_given_{species_name}"] = branch_audit
    return species_weight, grade_weight, {
        "species": species_audit,
        "conditional_grade": grade_audits,
    }


def _load_baseline(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    core.reject_test_path(path, "baseline validation selection")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("protocol") or {}
    if bool(protocol.get("strict_test_manifest_opened", False)):
        raise ValueError("Baseline selection does not attest clean held-out isolation")
    selected = payload.get("selected") or {}
    raw = selected.get("validation_metrics_raw")
    calibrated = selected.get("validation_metrics_calibrated")
    if not isinstance(raw, dict) or not isinstance(calibrated, dict):
        raise ValueError("Baseline selection lacks validation metrics")
    return {"raw": raw, "calibrated": calibrated}, core.fingerprint(path)


def main() -> None:
    args = parse_args()
    all_paths = (
        args.train_manifest,
        args.val_manifest,
        args.deep_train,
        args.deep_val,
        args.quality_train,
        args.quality_val,
        args.quality_train_source_manifest,
        args.baseline_selection,
        args.model_output,
        args.selection_output,
    )
    for path in all_paths:
        core.reject_test_path(path.resolve(), "fusion input/output")
    model_output = args.model_output.resolve()
    selection_output = args.selection_output.resolve()
    receipt_output = selection_output.with_name(f"{selection_output.name}.sha256.json")
    for output in (model_output, selection_output, receipt_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {output}")

    train_rows = core.read_manifest(
        args.train_manifest,
        expected_split="train",
        expected_rows=core.EXPECTED_TRAIN_ROWS,
    )
    val_rows = core.read_manifest(
        args.val_manifest,
        expected_split="val",
        expected_rows=core.EXPECTED_VAL_ROWS,
    )
    leakage = core.leakage_audit(train_rows, val_rows)
    if not leakage["passed"]:
        raise ValueError(f"Train/validation leakage audit failed: {leakage}")
    pair_base_weight, pair_audit = core.pair_weight_audit(train_rows)

    deep_train_set = core.load_feature_set(
        [args.deep_train],
        train_rows,
        expected_split="train",
        expected_manifest=args.train_manifest,
    )
    quality_train, quality_train_record, quality_names = _load_quality_cache(
        args.quality_train,
        train_rows,
        expected_split="train",
        recorded_manifest=args.quality_train_source_manifest,
    )
    deep_train = _deep_canonical(deep_train_set)
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    species_confidence = np.asarray(
        [float(row["species_weight"]) for row in train_rows], dtype=np.float64
    )
    grade_confidence = np.asarray(
        [float(row["grade_weight"]) for row in train_rows], dtype=np.float64
    )
    species_raw_weight = species_confidence * pair_base_weight
    grade_raw_weight = grade_confidence * pair_base_weight
    folds, fold_audit = _build_group_folds(train_rows)

    # All model/hyperparameter choices below use the training folds only.
    species_c, species_candidates = _select_species_c(
        deep_train,
        train_rows,
        labels,
        species_raw_weight,
        species_confidence,
        folds,
    )
    grade_selections: list[dict[str, float]] = []
    grade_candidates: dict[str, list[dict[str, Any]]] = {}
    for species_id, species_name in enumerate(core.SPECIES_NAMES):
        selected, candidates = _select_grade_head(
            species_id,
            deep_train,
            quality_train,
            train_rows,
            labels,
            grade_raw_weight,
            grade_confidence,
            folds,
        )
        grade_selections.append(selected)
        grade_candidates[f"grade_given_{species_name}"] = candidates

    # Only now load physical validation features.  The selected C/alpha values are
    # immutable for the rest of this run.
    deep_val_set = core.load_feature_set(
        [args.deep_val],
        val_rows,
        expected_split="val",
        expected_manifest=args.val_manifest,
    )
    core.compare_feature_schemas(deep_train_set, deep_val_set)
    quality_val, quality_val_record, quality_val_names = _load_quality_cache(
        args.quality_val,
        val_rows,
        expected_split="val",
        recorded_manifest=args.val_manifest,
    )
    if quality_names != quality_val_names:
        raise ValueError("Train/validation handcrafted feature-name schemas differ")
    deep_val = _deep_canonical(deep_val_set)

    species_weight, grade_weight, balancing_audit = _fit_full_weights(
        labels,
        pair_base_weight,
        species_confidence,
        grade_confidence,
    )
    species_labels = labels // 2
    grade_labels = labels % 2
    species_keep = species_weight > 0
    species_model = _make_svc(species_c, probability=True)
    species_model.fit(
        deep_train[species_keep],
        species_labels[species_keep],
        sample_weight=species_weight[species_keep],
    )
    species_probability = core.ordered_binary_probability(species_model, deep_val)

    grade_models: list[SVC] = []
    grade_scalers: list[StandardScaler | None] = []
    grade_probability_blocks: list[np.ndarray] = []
    for species_id, selected in enumerate(grade_selections):
        keep = (species_labels == species_id) & (grade_weight > 0)
        alpha = float(selected["alpha"])
        scaler = StandardScaler().fit(quality_train[keep]) if alpha > 0 else None
        train_x = _fused_matrix(
            deep_train[keep], quality_train[keep], scaler=scaler, alpha=alpha
        )
        val_x = _fused_matrix(deep_val, quality_val, scaler=scaler, alpha=alpha)
        model = _make_svc(float(selected["C"]), probability=True)
        model.fit(train_x, grade_labels[keep], sample_weight=grade_weight[keep])
        grade_models.append(model)
        grade_scalers.append(scaler)
        grade_probability_blocks.append(
            _apply_grade_logit_bias(
                core.ordered_binary_probability(model, val_x),
                float(selected["logit_bias"]),
            )
        )
    grade_probability = np.stack(grade_probability_blocks, axis=1)
    raw_metrics = core.branch_metrics(val_rows, species_probability, grade_probability)
    temperatures, temperature_objectives = core.fit_temperatures(
        val_rows, species_probability, grade_probability
    )
    calibrated_probability = core.apply_temperatures(
        species_probability, grade_probability, temperatures
    )
    calibrated_metrics = core.branch_metrics(val_rows, *calibrated_probability)
    baseline_metrics, baseline_record = _load_baseline(args.baseline_selection)

    comparison = {}
    for stage, metrics in (("raw", raw_metrics), ("calibrated", calibrated_metrics)):
        reference = baseline_metrics[stage]
        comparison[stage] = {
            key: float(metrics[key]) - float(reference[key])
            for key in (
                "species_accuracy",
                "conditional_grade_accuracy",
                "joint_accuracy",
                "joint_macro_f1",
                "group_joint_accuracy",
                "joint_nll",
                "joint_ece_15bin",
            )
        }
    improved = (
        calibrated_metrics["joint_macro_f1"], calibrated_metrics["joint_accuracy"]
    ) > (
        baseline_metrics["calibrated"]["joint_macro_f1"],
        baseline_metrics["calibrated"]["joint_accuracy"],
    )

    protocol = {
        "stage": "train_group_cv_selection_then_single_physical_validation",
        "hyperparameter_selection_data": "training split only",
        "selection_metric": (
            "audit-confidence-weighted pooled canonical-row macro-F1, then accuracy"
        ),
        "tie_policy": (
            "predeclared lower alpha, then lower C, then zero/smaller-magnitude "
            "grade logit bias"
        ),
        "physical_validation_access": (
            "loaded once after all C and alpha choices were frozen; used for reporting "
            "and scalar temperature calibration"
        ),
        "strict_test_manifest_opened": False,
        "test_feature_cache_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "test_arguments_supported": False,
        "test_token_paths_rejected": True,
    }
    feature_schema = {
        "species_head": {
            "family": "imagenet_cls",
            "image_size": 256,
            "view": "canonical augmentation index 0",
            "normalization": "row L2",
            "quality_features_used": False,
            "dim": int(deep_train.shape[1]),
        },
        "conditional_grade_heads": [
            {
                "species": core.SPECIES_NAMES[species_id],
                "deep_family": "imagenet_cls",
                "deep_dim": int(deep_train.shape[1]),
                "quality_family": "quality_handcrafted" if selected["alpha"] > 0 else None,
                "quality_dim": int(quality_train.shape[1]) if selected["alpha"] > 0 else 0,
                "quality_transform": (
                    "branch-train-only StandardScaler, then row L2, then alpha"
                    if selected["alpha"] > 0
                    else "disabled"
                ),
                "combined_dim": int(
                    deep_train.shape[1]
                    + (quality_train.shape[1] if selected["alpha"] > 0 else 0)
                ),
                **selected,
            }
            for species_id, selected in enumerate(grade_selections)
        ],
    }
    model_payload = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": core.utc_now(),
        "species_model": species_model,
        "species_C": species_c,
        "grade_models": grade_models,
        "grade_quality_scalers": grade_scalers,
        "grade_selections": grade_selections,
        "feature_schema": feature_schema,
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
        "validation_metrics_raw": raw_metrics,
        "validation_metrics_calibrated": calibrated_metrics,
        "protocol": protocol,
    }
    core.atomic_joblib_dump(model_payload, model_output)
    model_record = core.fingerprint(model_output)

    frozen_core = {
        "model": model_record,
        "species_C": species_c,
        "grade_selections": grade_selections,
        "feature_schema": feature_schema,
        "temperatures": model_payload["temperatures"],
    }
    selection = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": core.utc_now(),
        "selection_id": f"pepper-clean-v5-grade-fusion-{core.sha256_json(frozen_core)[:20]}",
        "protocol": protocol,
        "strict_test_manifest_opened": False,
        "input_fingerprints": {
            "script": core.fingerprint(Path(__file__)),
            "train_manifest": core.fingerprint(args.train_manifest),
            "validation_manifest": core.fingerprint(args.val_manifest),
            "deep_train": deep_train_set.files[0],
            "deep_validation": deep_val_set.files[0],
            "quality_train": quality_train_record,
            "quality_validation": quality_val_record,
            "baseline_selection": baseline_record,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "leakage_audit": leakage,
        "pair_weight_audit": pair_audit,
        "group_cv_audit": fold_audit,
        "training_summary": {
            "rows": len(train_rows),
            "groups": len({row["group_id"] for row in train_rows}),
            "positive_species_weight_rows": int((species_confidence > 0).sum()),
            "positive_grade_weight_rows": int((grade_confidence > 0).sum()),
            "paired_views_receive_total_base_mass_one": True,
            "balancing_audit": balancing_audit,
        },
        "candidate_space": {
            "family": "hierarchical RBF-SVC",
            "species_input": "ImageNet-256 deep only",
            "grade_input": "ImageNet-256 deep plus optional standardized quality",
            "C_grid": list(PREDECLARED_C_GRID),
            "quality_alpha_grid": list(PREDECLARED_QUALITY_ALPHA_GRID),
            "grade_logit_bias_grid": list(PREDECLARED_GRADE_LOGIT_BIAS_GRID),
            "grade_logit_bias_selection": (
                "training OOF only; add bias to grade-index-1 log posterior before "
                "conditional softmax"
            ),
            "grid_is_not_cli_configurable": True,
            "species_candidate_count": len(species_candidates),
            "grade_candidate_count_per_branch": len(next(iter(grade_candidates.values()))),
        },
        "training_cv_candidates": {
            "species": species_candidates,
            **grade_candidates,
        },
        "selected": {
            **frozen_core,
            "validation_metrics_raw": raw_metrics,
            "temperature_fit_bounds": list(core.TEMPERATURE_BOUNDS),
            "temperature_fit_objectives": temperature_objectives,
            "validation_metrics_calibrated": calibrated_metrics,
        },
        "baseline_comparison": {
            "baseline_metrics": baseline_metrics,
            "fusion_minus_baseline": comparison,
            "improved_by_calibrated_macro_f1_then_accuracy": improved,
        },
        "final_test_status": "not_run",
    }
    core.atomic_json_dump(selection, selection_output)
    selection_record = core.fingerprint(selection_output)
    receipt = {
        "schema": "pepper-clean-v5-grade-fusion-receipt-v1",
        "created_at_utc": core.utc_now(),
        "selection_id": selection["selection_id"],
        "selection": selection_record,
        "model": model_record,
        "strict_test_manifest_opened": False,
    }
    core.atomic_json_dump(receipt, receipt_output)
    print(
        json.dumps(
            {
                "selection_id": selection["selection_id"],
                "model_saved": str(model_output),
                "selection_saved": str(selection_output),
                "receipt_saved": str(receipt_output),
                "species_C": species_c,
                "grade_selections": grade_selections,
                "validation_raw": raw_metrics,
                "validation_calibrated": calibrated_metrics,
                "fusion_minus_baseline": comparison,
                "improved": improved,
                "strict_protocol": "NO TEST MANIFEST, FEATURE, LABEL OR METRIC WAS OPENED",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
