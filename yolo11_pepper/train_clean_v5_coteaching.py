#!/usr/bin/env python3
"""Strict clean-v5 co-teaching selection on frozen YOLO11n-cls features.

The executable deliberately accepts no test manifest or test feature argument.
It trains a randomly initialized two-member hierarchical head over a frozen,
ImageNet-pretrained YOLO11n-cls feature cache.  The head is

    p(species) and {p(grade | species=0), p(grade | species=1)},

and inference is Bayes composition followed by argmax.  Training uses the
train-only audit weights/soft grade targets, label smoothing, a clean-sample
curriculum, class-stratified small-loss co-teaching and cross-EMA FixMatch.

Only one predeclared configuration and seed are run.  Validation is used for
early stopping and post-hoc scalar temperatures, not for a configuration grid.
Canonical/detector-aligned rows sharing a pair_id are collapsed before training
so paired views cannot receive duplicate sample mass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from train_hierarchical_v4 import (
    FINAL_NAMES,
    GRADE_NAMES,
    SPECIES_NAMES,
    CandidateResult,
    LabelBundle,
    build_labels,
    classification_metrics,
    initialize_head,
    load_feature_caches,
    predict_probabilities,
    read_label_decisions,
    train_candidate,
)
from train_select_clean_v5 import (
    apply_temperatures,
    branch_metrics,
    fit_temperatures,
)


SCRIPT_VERSION = "pepper-clean-v5-coteaching-validation-v2"
SEED = 3179
EXPECTED_TRAIN_ROWS = 697
EXPECTED_VAL_ROWS = 146
PROJECT = Path(__file__).resolve().parent
DEFAULT_TRAIN_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
)
DEFAULT_VAL_MANIFEST = (
    PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
)
DEFAULT_TRAIN_FEATURE = (
    PROJECT
    / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt"
)
DEFAULT_VAL_FEATURE = (
    PROJECT / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt"
)
DEFAULT_OUTPUT = PROJECT / "runs/hierarchical_v5_clean/coteaching"
BASELINE_SELECTION = PROJECT / "runs/hierarchical_v5_clean/selection_reaudit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict validation-only clean-v5 hierarchical co-teaching."
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--train-feature", type=Path, default=DEFAULT_TRAIN_FEATURE)
    parser.add_argument("--val-feature", type=Path, default=DEFAULT_VAL_FEATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def reject_test_path(path: Path, label: str) -> None:
    for part in path.resolve().parts:
        tokens = {token for token in re.split(r"[^a-z0-9]+", part.lower()) if token}
        if "test" in tokens:
            raise ValueError(f"STRICT NO-TEST protocol rejected {label}: {path}")


def read_physical_manifest(
    path: Path, expected_split: str, expected_rows: int
) -> list[dict[str, str]]:
    path = path.resolve()
    reject_test_path(path, f"{expected_split} manifest")
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "split", "group_id", "source_id", "class_id", "pair_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing fields {sorted(missing)}")
        for line, raw in enumerate(reader, 2):
            split = (raw.get("split") or "").strip().lower()
            if split != expected_split:
                raise ValueError(f"{path}:{line} expected split={expected_split}, got {split}")
            image_path = Path(raw["path"]).resolve()
            reject_test_path(image_path, f"{expected_split} image")
            class_id = int(raw["class_id"])
            if class_id not in range(4):
                raise ValueError(f"{path}:{line} invalid class_id={class_id}")
            selection_role = (raw.get("selection_role") or "").strip().lower()
            expected_role = "training" if expected_split == "train" else "model_selection"
            if selection_role != expected_role:
                raise ValueError(
                    f"{path}:{line} selection_role={selection_role!r}, expected {expected_role!r}"
                )
            rows.append(
                {
                    **raw,
                    "path": str(image_path),
                    "class_id": str(class_id),
                    "split": split,
                    "group_id": raw["group_id"].strip(),
                    "source_id": raw["source_id"].strip(),
                    "pair_id": raw["pair_id"].strip(),
                }
            )
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate paths in {path}")
    if set(int(row["class_id"]) for row in rows) != set(range(4)):
        raise ValueError(f"All four classes are required in {path}")
    return rows


def verify_no_overlap(
    train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for field in ("path", "group_id", "source_id", "pair_id", "content_sha256"):
        left = {row.get(field, "") for row in train_rows} - {""}
        right = {row.get(field, "") for row in val_rows} - {""}
        shared = sorted(left & right)
        audit[f"{field}_overlap"] = shared
        if shared:
            raise ValueError(f"Train/validation leakage on {field}: {shared[:5]}")
    audit["passed"] = True
    return audit


def verify_feature_metadata(path: Path, manifest: Path, split: str) -> dict[str, Any]:
    reject_test_path(path, f"{split} feature")
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") or {}
    if str(metadata.get("split", "")).lower() != split:
        raise ValueError(f"{path} metadata split mismatch")
    if bool(metadata.get("test_requested_explicitly", False)):
        raise ValueError(f"{path} was produced by a test-requesting extraction")
    if Path(str(metadata.get("manifest", ""))).resolve() != manifest.resolve():
        raise ValueError(f"{path} metadata manifest does not match physical {split} manifest")
    return {
        **fingerprint(path),
        "shape": list(payload["features"].shape),
        "metadata": metadata,
    }


def collapse_pairs(
    features: torch.Tensor,
    labels: LabelBundle,
    rows: Sequence[dict[str, str]],
) -> tuple[torch.Tensor, LabelBundle, dict[str, Any]]:
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_pair[row["pair_id"]].append(index)
    collapsed_features: list[torch.Tensor] = []
    keep: list[int] = []
    pair_sizes: list[int] = []
    for pair_id, indices in by_pair.items():
        first = indices[0]
        for index in indices[1:]:
            if (
                rows[index]["class_id"] != rows[first]["class_id"]
                or rows[index]["group_id"] != rows[first]["group_id"]
                or not torch.allclose(
                    labels.species_targets[index], labels.species_targets[first], atol=1e-7
                )
                or not torch.allclose(
                    labels.grade_targets[index], labels.grade_targets[first], atol=1e-7
                )
                or abs(float(labels.species_weights[index] - labels.species_weights[first]))
                > 1e-7
                or abs(float(labels.grade_weights[index] - labels.grade_weights[first])) > 1e-7
            ):
                raise ValueError(f"Inconsistent label/weight within pair_id={pair_id}")
        collapsed_features.append(features[indices].mean(0))
        keep.append(first)
        pair_sizes.append(len(indices))
    selected = torch.tensor(keep, dtype=torch.long)
    result_labels = LabelBundle(
        species_targets=labels.species_targets[selected],
        grade_targets=labels.grade_targets[selected],
        species_weights=labels.species_weights[selected],
        grade_weights=labels.grade_weights[selected],
        high_consistency=labels.high_consistency[selected],
        hard_species=labels.hard_species[selected],
        hard_grade=labels.hard_grade[selected],
        final_classes=labels.final_classes[selected],
        group_ids=[labels.group_ids[index] for index in keep],
        paths=[labels.paths[index] for index in keep],
        audit_summary={
            **labels.audit_summary,
            "rows_before_pair_collapse": len(rows),
            "unique_pairs_after_collapse": len(keep),
            "multi_view_pairs": sum(size > 1 for size in pair_sizes),
            "maximum_rows_per_pair": max(pair_sizes),
            "pair_feature_policy": "mean each corresponding cached augmentation view",
        },
    )
    return torch.stack(collapsed_features), result_labels, result_labels.audit_summary


def pseudo_relabel_strict_contradictions(
    labels: LabelBundle,
    rows: Sequence[dict[str, str]],
) -> tuple[LabelBundle, dict[str, Any]]:
    """Create, without mutating the manifest, the predeclared OOF pseudo-label arm.

    Only grade is changed.  The train-only audit already imposed group-excluded
    OOF and a strict contradiction gate.  We retain its soft OOF distribution,
    set a deliberately small weight (0.20, below the minimum retained hard-label
    weight of 0.35), and never mark these samples high-consistency.  This remains
    circular self-training evidence rather than a substitute for expert review.
    """
    species_targets = labels.species_targets.clone()
    grade_targets = labels.grade_targets.clone()
    species_weights = labels.species_weights.clone()
    grade_weights = labels.grade_weights.clone()
    high_consistency = labels.high_consistency.clone()
    hard_species = labels.hard_species.clone()
    hard_grade = labels.hard_grade.clone()
    affected_pairs: set[str] = set()
    affected_rows = 0
    original_classes: Counter[int] = Counter()
    pseudo_classes: Counter[int] = Counter()
    confidences: list[float] = []
    for index, row in enumerate(rows):
        if row.get("audit_status") != "manual_review_strict_grade_contradiction":
            continue
        p_good = float(row["oof_p_good"])
        p_bad = float(row["oof_p_bad"])
        total = p_good + p_bad
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid OOF grade probability for {row['path']}")
        probability = torch.tensor([p_good / total, p_bad / total], dtype=torch.float32)
        pseudo_grade = int(probability.argmax())
        original_grade = int(labels.hard_grade[index])
        if pseudo_grade == original_grade:
            raise ValueError(
                "Strict contradiction audit row does not contradict its original grade: "
                f"{row['path']}"
            )
        grade_targets[index] = probability
        grade_weights[index] = 0.20
        high_consistency[index] = False
        hard_grade[index] = pseudo_grade
        affected_rows += 1
        affected_pairs.add(row["pair_id"])
        original_classes[int(labels.final_classes[index])] += 1
        pseudo_classes[int(hard_species[index] * 2 + pseudo_grade)] += 1
        confidences.append(float(probability.max()))
    final_classes = hard_species * 2 + hard_grade
    audit = {
        "candidate": "strict_oof_pseudo_relabel",
        "affected_rows": affected_rows,
        "affected_unique_pairs": len(affected_pairs),
        "expected_unique_pairs": 33,
        "grade_only": True,
        "species_preserved": bool(torch.equal(hard_species, labels.hard_species)),
        "pseudo_grade_weight": 0.20,
        "minimum_retained_hard_grade_weight": 0.35,
        "mean_oof_confidence": float(np.mean(confidences)),
        "minimum_oof_confidence": float(min(confidences)),
        "original_class_row_counts": dict(sorted(original_classes.items())),
        "pseudo_class_row_counts": dict(sorted(pseudo_classes.items())),
        "manifest_modified": False,
        "risk": (
            "circular self-training: the pseudo label comes from group-excluded OOF "
            "predictions of the same frozen feature family; expert adjudication is preferred"
        ),
    }
    if len(affected_pairs) != 33:
        raise ValueError(
            f"Predeclared pseudo-label arm expected 33 unique pairs, got {len(affected_pairs)}"
        )
    return (
        LabelBundle(
            species_targets=species_targets,
            grade_targets=grade_targets,
            species_weights=species_weights,
            grade_weights=grade_weights,
            high_consistency=high_consistency,
            hard_species=hard_species,
            hard_grade=hard_grade,
            final_classes=final_classes,
            group_ids=list(labels.group_ids),
            paths=list(labels.paths),
            audit_summary={**labels.audit_summary, "pseudo_relabel": audit},
        ),
        audit,
    )


def restore_original_targets_for_unlabelled_grade(
    labels: LabelBundle, rows: Sequence[dict[str, str]]
) -> LabelBundle:
    """Prevent zero-weight OOF guesses from silently changing batch strata."""
    grade_targets = labels.grade_targets.clone()
    hard_grade = labels.hard_grade.clone()
    high_consistency = labels.high_consistency.clone()
    restored = 0
    for index, row in enumerate(rows):
        if float(labels.grade_weights[index]) > 0:
            continue
        original_grade = int(row["class_id"]) % 2
        grade_targets[index].zero_()
        grade_targets[index, original_grade] = 1.0
        hard_grade[index] = original_grade
        high_consistency[index] = False
        restored += 1
    return LabelBundle(
        species_targets=labels.species_targets.clone(),
        grade_targets=grade_targets,
        species_weights=labels.species_weights.clone(),
        grade_weights=labels.grade_weights.clone(),
        high_consistency=high_consistency,
        hard_species=labels.hard_species.clone(),
        hard_grade=hard_grade,
        final_classes=labels.hard_species * 2 + hard_grade,
        group_ids=list(labels.group_ids),
        paths=list(labels.paths),
        audit_summary={
            **labels.audit_summary,
            "zero_weight_grade_targets_restored_to_original_for_sampling": restored,
        },
    )


def supervision_mass(labels: LabelBundle) -> dict[str, Any]:
    final_counts = Counter(int(value) for value in labels.final_classes.tolist())
    high_counts = Counter(
        int(label)
        for label, high in zip(labels.final_classes.tolist(), labels.high_consistency.tolist())
        if high
    )
    grade_mass = defaultdict(float)
    for label, weight in zip(labels.final_classes.tolist(), labels.grade_weights.tolist()):
        grade_mass[int(label)] += float(weight)
    return {
        "pair_counts_by_effective_class": {
            FINAL_NAMES[index]: final_counts[index] for index in range(4)
        },
        "high_consistency_pairs_by_effective_class": {
            FINAL_NAMES[index]: high_counts[index] for index in range(4)
        },
        "grade_weight_mass_by_effective_class": {
            FINAL_NAMES[index]: grade_mass[index] for index in range(4)
        },
        "mitigation_for_low_mass_子弹头_一级": (
            "every batch has an exact four-class quota; sampling is class -> source-group "
            "-> pair, then confidence weight and co-teaching are applied"
        ),
    }


def make_training_args(device: str | None) -> argparse.Namespace:
    # One source-controlled configuration.  No CLI hyperparameter knobs are
    # offered, which prevents validation-driven grid expansion after inspection.
    return argparse.Namespace(
        hidden_dim=128,
        grade_hidden_dim=64,
        dropout=0.18,
        grade_dropout=0.12,
        lr=6e-4,
        weight_decay=1.5e-4,
        min_lr=2e-5,
        ema_decay=0.992,
        grad_clip=5.0,
        species_loss_weight=0.75,
        grade_loss_weight=1.25,
        species_label_smoothing=0.02,
        grade_label_smoothing=0.05,
        species_class_weight_power=0.25,
        grade_class_weight_power=0.25,
        forget_rate=0.12,
        forget_warmup_epochs=18,
        clean_warmup_epochs=6,
        ssl_weight=0.30,
        ssl_ramp_epochs=12,
        pseudo_threshold=0.86,
        warmup_pseudo_threshold=0.94,
        contrastive_weight=0.0,
        contrastive_temperature=0.15,
        selection_metric="balanced",
        epochs=110,
        patience=24,
        batch_size=128,
        batches_per_epoch=0,
        device=device,
        quiet=False,
    )


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def branch_probabilities(
    result: CandidateResult,
    features: torch.Tensor,
    train_args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    members = []
    for index, state in enumerate((result.state_a, result.state_b)):
        model = initialize_head(features.shape[-1], train_args, SEED + index).to(device)
        model.load_state_dict(state)
        model.eval()
        members.append(model)
    species_outputs: list[torch.Tensor] = []
    grade_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for model in members:
            for view in range(features.shape[1]):
                species_pieces, grade_pieces = [], []
                for start in range(0, len(features), 512):
                    species_logits, grade_logits = model(
                        features[start : start + 512, view].to(device)
                    )
                    species_pieces.append(species_logits.softmax(1).cpu())
                    grade_pieces.append(grade_logits.softmax(2).cpu())
                species_outputs.append(torch.cat(species_pieces))
                grade_outputs.append(torch.cat(grade_pieces))
    species = torch.stack(species_outputs).mean(0).numpy().astype(np.float64)
    grade = torch.stack(grade_outputs).mean(0).numpy().astype(np.float64)
    species /= np.clip(species.sum(1, keepdims=True), 1e-15, None)
    grade /= np.clip(grade.sum(2, keepdims=True), 1e-15, None)
    return species, grade


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def baseline_metrics() -> dict[str, Any]:
    if not BASELINE_SELECTION.exists():
        return {"available": False}
    payload = json.loads(BASELINE_SELECTION.read_text(encoding="utf-8"))
    selected = payload["selected"]
    return {
        "available": True,
        "artifact": fingerprint(BASELINE_SELECTION),
        "validation_metrics_raw": selected["validation_metrics_raw"],
        "validation_metrics_calibrated": selected["validation_metrics_calibrated"],
    }


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.train_manifest, "train manifest"),
        (args.val_manifest, "validation manifest"),
        (args.train_feature, "train feature"),
        (args.val_feature, "validation feature"),
        (args.output, "output"),
    ):
        reject_test_path(path, label)
    train_rows = read_physical_manifest(args.train_manifest, "train", EXPECTED_TRAIN_ROWS)
    val_rows = read_physical_manifest(args.val_manifest, "val", EXPECTED_VAL_ROWS)
    leakage_audit = verify_no_overlap(train_rows, val_rows)
    train_feature_record = verify_feature_metadata(
        args.train_feature, args.train_manifest, "train"
    )
    val_feature_record = verify_feature_metadata(args.val_feature, args.val_manifest, "val")
    train_features = load_feature_caches([args.train_feature], train_rows)
    val_features = load_feature_caches([args.val_feature], val_rows)
    if train_features.shape[-1] != val_features.shape[-1]:
        raise ValueError("Train/validation feature dimensions differ")

    decisions = read_label_decisions(
        args.train_manifest, {row["path"] for row in train_rows}
    )
    removal_labels = restore_original_targets_for_unlabelled_grade(
        build_labels(train_rows, decisions, True, allow_overrides=True), train_rows
    )
    pseudo_labels, pseudo_relabel_audit = pseudo_relabel_strict_contradictions(
        removal_labels, train_rows
    )
    val_labels = build_labels(val_rows, {}, False, allow_overrides=False)
    collapsed_features, removal_labels, pair_audit = collapse_pairs(
        train_features, removal_labels, train_rows
    )
    pseudo_features, pseudo_labels, pseudo_pair_audit = collapse_pairs(
        train_features, pseudo_labels, train_rows
    )
    if not torch.equal(collapsed_features, pseudo_features):
        raise ValueError("Candidate arms unexpectedly produced different feature tensors")

    train_args = make_training_args(args.device)
    device = choose_device(args.device)
    print(
        json.dumps(
            {
                "strict_protocol": "NO TEST DATA WILL BE READ",
                "device": str(device),
                "train_pairs": len(collapsed_features),
                "validation_rows": len(val_features),
                "feature_shape_train": list(collapsed_features.shape),
                "removal_supervision": supervision_mass(removal_labels),
                "pseudo_relabel_supervision": supervision_mass(pseudo_labels),
                "pseudo_relabel_audit": pseudo_relabel_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # The physical model-selection validation is deliberately not passed into
    # train_candidate.  Each arm is optimized/early-stopped only against its own
    # training monitor (canonical view), then both frozen arms are compared once
    # on the physical validation set with the predeclared ranking below.
    candidate_results: list[dict[str, Any]] = []
    for candidate_index, (candidate_name, candidate_labels) in enumerate(
        (
            ("remove_strict_contradictions", removal_labels),
            ("strict_oof_pseudo_relabel", pseudo_labels),
        )
    ):
        candidate_seed = SEED + candidate_index * 10_007
        np.random.seed(candidate_seed)
        torch.manual_seed(candidate_seed)
        result = train_candidate(
            candidate_seed,
            collapsed_features,
            candidate_labels,
            # Training-only monitor; physical validation remains unopened here.
            collapsed_features[:, :1],
            candidate_labels,
            train_args,
            device,
        )
        species_probability, grade_probability = branch_probabilities(
            result, val_features, train_args, device
        )
        validation_raw = branch_metrics(val_rows, species_probability, grade_probability)
        candidate_results.append(
            {
                "name": candidate_name,
                "seed": candidate_seed,
                "result": result,
                "validation_metrics_raw": validation_raw,
                "species_probability": species_probability,
                "grade_probability": grade_probability,
                "supervision_mass": supervision_mass(candidate_labels),
                "optimization_monitor": {
                    "data": "training pairs, canonical cached view only",
                    "best_epoch": result.best_epoch,
                    "epochs_executed": len(result.history),
                    "metrics_at_selected_epoch": result.validation,
                },
            }
        )
        print(
            json.dumps(
                {
                    "candidate_complete": candidate_name,
                    "physical_validation_metrics_raw": validation_raw,
                    "physical_validation_was_evaluated_once": True,
                },
                ensure_ascii=False,
            )
        )

    # Predeclared rule: joint macro-F1, then joint accuracy, then conditional
    # grade accuracy; exact tie favors removal (the lower-circularity arm).
    selected_index = max(
        range(len(candidate_results)),
        key=lambda index: (
            candidate_results[index]["validation_metrics_raw"]["joint_macro_f1"],
            candidate_results[index]["validation_metrics_raw"]["joint_accuracy"],
            candidate_results[index]["validation_metrics_raw"][
                "conditional_grade_accuracy"
            ],
            -index,
        ),
    )
    selected_candidate = candidate_results[selected_index]
    result = selected_candidate["result"]
    species_probability = selected_candidate["species_probability"]
    grade_probability = selected_candidate["grade_probability"]
    validation_raw = selected_candidate["validation_metrics_raw"]
    temperatures, temperature_objectives = fit_temperatures(
        val_rows, species_probability, grade_probability
    )
    calibrated_species, calibrated_grade = apply_temperatures(
        species_probability, grade_probability, temperatures
    )
    validation_calibrated = branch_metrics(
        val_rows, calibrated_species, calibrated_grade
    )
    baseline = baseline_metrics()
    if baseline["available"]:
        reference = baseline["validation_metrics_calibrated"]
        comparison = {
            "joint_accuracy_delta": validation_calibrated["joint_accuracy"]
            - reference["joint_accuracy"],
            "joint_macro_f1_delta": validation_calibrated["joint_macro_f1"]
            - reference["joint_macro_f1"],
            "species_accuracy_delta": validation_calibrated["species_accuracy"]
            - reference["species_accuracy"],
            "conditional_grade_accuracy_delta": validation_calibrated[
                "conditional_grade_accuracy"
            ]
            - reference["conditional_grade_accuracy"],
        }
    else:
        comparison = {"available": False}

    output = args.output.resolve()
    model_path = output / "best_hierarchical_clean_v5_coteaching.pt"
    selection_path = output / "selection.json"
    model_payload = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "member_state_dicts": [result.state_a, result.state_b],
        "feature_dim": int(collapsed_features.shape[-1]),
        "feature_family": "frozen_yolo11n_cls_imagenet_256_multiscale",
        "species_names": SPECIES_NAMES,
        "grade_names": GRADE_NAMES,
        "final_names": FINAL_NAMES,
        "architecture": {
            "type": "shared_embedding_plus_species_and_two_conditional_grade_heads",
            "hidden_dim": train_args.hidden_dim,
            "grade_hidden_dim": train_args.grade_hidden_dim,
            "decision": "argmax flatten(p(species) * p(grade | species))",
        },
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
        "training": {
            "candidate": selected_candidate["name"],
            "seed": selected_candidate["seed"],
            "best_epoch": result.best_epoch,
            "configuration": vars(train_args),
            "co_teaching": "two independently initialized students exchange per-class small-loss samples",
            "label_smoothing": {
                "species": train_args.species_label_smoothing,
                "grade": train_args.grade_label_smoothing,
            },
            "low_confidence_weighting": "species_weight and grade_weight from train-only paired audit",
            "curriculum": f"first {train_args.clean_warmup_epochs} epochs use high-consistency supervised samples",
            "semi_supervision": "cross-EMA dual-teacher agreement over cached weak/strong views",
            "paired_view_policy": pair_audit["pair_feature_policy"],
            "supervision_mass": selected_candidate["supervision_mass"],
            "pseudo_relabel_audit": (
                pseudo_relabel_audit
                if selected_candidate["name"] == "strict_oof_pseudo_relabel"
                else None
            ),
        },
        "strict_protocol": {
            "historical_classification_head_loaded": False,
            "head_randomly_initialized": True,
            "frozen_backbone_features_only": True,
            "test_manifest_opened": False,
            "test_feature_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
        },
    }
    atomic_torch_save(model_payload, model_path)
    model_record = fingerprint(model_path)
    selection = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "protocol": {
            "stage": "strict_train_validation_model_selection_and_calibration",
            "configuration_count": 2,
            "seed_count_per_configuration": 1,
            "candidate_arms": [
                "remove_strict_contradictions",
                "strict_oof_pseudo_relabel",
            ],
            "selection_metric": (
                "physical validation joint macro-F1, then joint accuracy, then "
                "conditional grade accuracy; exact tie favors removal"
            ),
            "optimization_monitor_data": "training pairs, canonical cached view only",
            "physical_validation_evaluations_per_candidate": 1,
            "temperature_fit_data": str(args.val_manifest.resolve()),
            "strict_test_manifest_opened": False,
            "test_feature_cache_opened": False,
            "test_labels_read": False,
            "test_metrics_computed": False,
            "test_arguments_supported": False,
        },
        "input_fingerprints": {
            "script": fingerprint(Path(__file__)),
            "train_manifest": fingerprint(args.train_manifest),
            "validation_manifest": fingerprint(args.val_manifest),
            "train_feature": train_feature_record,
            "validation_feature": val_feature_record,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "device": str(device),
        },
        "leakage_audit": leakage_audit,
        "pair_audit": pair_audit,
        "pseudo_pair_audit": pseudo_pair_audit,
        "pseudo_relabel_audit": pseudo_relabel_audit,
        "candidate_results": [
            {
                "name": candidate["name"],
                "seed": candidate["seed"],
                "validation_metrics_raw": candidate["validation_metrics_raw"],
                "supervision_mass": candidate["supervision_mass"],
                "optimization_monitor": candidate["optimization_monitor"],
            }
            for candidate in candidate_results
        ],
        "training": {
            "selected_candidate": selected_candidate["name"],
            "seed": selected_candidate["seed"],
            "best_epoch": result.best_epoch,
            "epochs_executed": len(result.history),
            "configuration": vars(train_args),
            "history": result.history,
            "member_validation": result.member_validation,
            "ensemble_training_monitor_at_selected_epoch": result.validation,
        },
        "selected": {
            "candidate": selected_candidate["name"],
            "model": model_record,
            "temperatures": {
                "species": temperatures[0],
                "grade_given_子弹头": temperatures[1],
                "grade_given_条子": temperatures[2],
            },
            "temperature_fit_objectives": temperature_objectives,
            "validation_metrics_raw": validation_raw,
            "validation_metrics_calibrated": validation_calibrated,
        },
        "clean_svm_baseline": baseline,
        "comparison_to_clean_svm_calibrated": comparison,
        "final_test_status": {
            "evaluated": False,
            "reason": "strict test remains unopened; a future frozen-artifact evaluator is required",
        },
    }
    atomic_json(selection, selection_path)
    receipt_path = output / "selection.json.sha256.json"
    atomic_json(
        {
            "selection": fingerprint(selection_path),
            "model": model_record,
            "strict_test_manifest_opened": False,
        },
        receipt_path,
    )
    print(
        json.dumps(
            {
                "model": model_record,
                "selection": fingerprint(selection_path),
                "validation_raw": validation_raw,
                "validation_calibrated": validation_calibrated,
                "comparison_to_clean_svm_calibrated": comparison,
                "test_was_not_read": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
