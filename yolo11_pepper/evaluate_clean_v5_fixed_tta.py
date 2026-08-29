#!/usr/bin/env python3
"""One-shot clean-validation evaluation of one predeclared two-view TTA.

The rule is intentionally fixed and has no command-line controls for views,
weights, temperatures, thresholds, checkpoints, or manifests:

* view 1: the exact neutral canonical validation tensor used during training;
* view 2: a horizontal flip of that already-rendered tensor;
* each view: p(species, grade) = p(species) * p(grade | species);
* aggregate: arithmetic mean of the two four-class joint probabilities.

Only the physical train/validation manifest used to train the selected model is
accepted.  It contains no strict-test rows, as attested by the frozen training
receipt.  This script has no test input argument and never opens a test file.
The TTA is recommended only when both joint accuracy and four-class macro-F1
strictly improve over the recomputed canonical baseline.  Even then it remains
a validation-only candidate because repeated experiments have already used the
same nine physical validation groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_select_clean_v5 as clean_core
from train_hierarchical_v4 import HierarchicalHead, joint_probabilities
from train_hierarchical_v4_unfrozen import (
    EndToEndHierarchicalModel,
    PartialYOLOEncoder,
    PepperInstance,
    PepperValidationDataset,
    read_role_instances,
)


SCRIPT_VERSION = "pepper-clean-v5-fixed-two-view-tta-validation-v1"
PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "datasets/pepper_ssl_v5_clean_audit/train_val_manifest.csv"
MODEL = (
    PROJECT
    / "runs/hierarchical_v5_clean_unfrozen_cls_s3041"
    / "best_hierarchical_v4_unfrozen_strict.pt"
)
TRAINING_REPORT = (
    PROJECT
    / "runs/hierarchical_v5_clean_unfrozen_cls_s3041"
    / "training_report.json"
)
TRAINING_RECEIPT = TRAINING_REPORT.with_name("training_report.json.sha256.json")
DEFAULT_OUTPUT = (
    PROJECT
    / "runs/hierarchical_v5_clean_unfrozen_cls_s3041"
    / "fixed_tta_2view_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen original-plus-horizontal-flip 0.5/0.5 TTA "
            "on clean physical validation only."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser.parse_args()


def require_frozen_inputs() -> dict[str, Any]:
    for path in (MANIFEST, MODEL, TRAINING_REPORT, TRAINING_RECEIPT):
        clean_core.reject_test_path(path.resolve(), "fixed TTA input")
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {path.resolve()}")

    receipt = json.loads(TRAINING_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("strict_test_opened") is not False:
        raise ValueError("Training receipt does not attest strict_test_opened=false")
    manifest_record = receipt.get("physical_train_val_manifest") or {}
    if int(manifest_record.get("strict_test_rows", -1)) != 0:
        raise ValueError("Frozen training manifest receipt does not attest zero test rows")
    expected = {
        MANIFEST.resolve(): str(manifest_record.get("sha256") or ""),
        MODEL.resolve(): str((receipt.get("model") or {}).get("sha256") or ""),
        TRAINING_REPORT.resolve(): str((receipt.get("report") or {}).get("sha256") or ""),
    }
    for path, digest in expected.items():
        if not digest or clean_core.sha256_file(path) != digest:
            raise ValueError(f"Frozen training receipt hash mismatch: {path}")
    return receipt


def build_model(checkpoint: dict[str, Any], device: torch.device) -> EndToEndHierarchicalModel:
    architecture = checkpoint.get("architecture") or {}
    if architecture.get("type") != "partially_unfrozen_yolo11_hierarchical_v4":
        raise ValueError(f"Unexpected model architecture: {architecture.get('type')!r}")
    strict = checkpoint.get("strict_protocol") or {}
    required = {
        "historical_pepper_classification_head_loaded": False,
        "hierarchical_head_randomly_initialized": True,
        "test_rows_materialized": False,
        "test_images_loaded": False,
        "test_metrics_computed": False,
    }
    for key, expected in required.items():
        if strict.get(key) is not expected:
            raise ValueError(f"Checkpoint violates strict_protocol.{key}")

    backbone = Path(str(checkpoint["backbone_checkpoint"])).resolve()
    clean_core.reject_test_path(backbone, "fixed TTA backbone")
    if not backbone.is_file():
        raise FileNotFoundError(f"Backbone checkpoint is missing: {backbone}")
    encoder = PartialYOLOEncoder(
        backbone,
        str(architecture["backbone_kind"]),
        int(architecture["unfreeze_from"]),
        int(architecture.get("feature_dim") or 512),
        bool(architecture["train_bn"]),
    )
    if int(architecture["feature_dim"]) != encoder.feature_dim:
        raise ValueError("Checkpoint feature dimension does not match reconstructed encoder")
    head = HierarchicalHead(
        encoder.feature_dim,
        int(architecture["hidden_dim"]),
        int(architecture["grade_hidden_dim"]),
        dropout=0.0,
        grade_dropout=0.0,
    )
    model = EndToEndHierarchicalModel(encoder, head)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def rows_for_metrics(instances: Sequence[PepperInstance]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": instance.class_id,
            "group_id": instance.group_id,
            "pair_id": instance.pair_id,
            "path": next(
                view.path for view in instance.views if view.view_type == "canonical"
            ),
        }
        for instance in instances
    ]


def probabilities_from_joint(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if joint.ndim != 2 or joint.shape[1] != 4:
        raise ValueError(f"Expected [items, 4] joint probability, got {joint.shape}")
    if not np.isfinite(joint).all() or np.any(joint < 0):
        raise ValueError("Joint probabilities contain invalid values")
    normalized = joint / np.clip(joint.sum(1, keepdims=True), 1e-15, None)
    matrix = normalized.reshape(-1, 2, 2)
    species = matrix.sum(2)
    grade = matrix / np.clip(species[:, :, None], 1e-15, None)
    return species, grade


def evaluate_joint(
    rows: Sequence[dict[str, Any]], joint: np.ndarray
) -> dict[str, Any]:
    species, grade = probabilities_from_joint(joint)
    return clean_core.branch_metrics(rows, species, grade)


def infer_fixed_views(
    model: EndToEndHierarchicalModel,
    instances: Sequence[PepperInstance],
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = PepperValidationDataset(
        instances,
        SimpleNamespace(
            image_size=int(checkpoint["image_size"]),
            object_scale=float(checkpoint["object_scale"]),
            degrade_min_scale=0.35,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    canonical_parts: list[torch.Tensor] = []
    flipped_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            canonical_logits = model(images)
            flipped_logits = model(torch.flip(images, dims=(-1,)))
            canonical_parts.append(joint_probabilities(*canonical_logits).cpu())
            flipped_parts.append(joint_probabilities(*flipped_logits).cpu())
            label_parts.append(labels.cpu())
    loaded_labels = torch.cat(label_parts).long()
    expected_labels = torch.tensor([item.class_id for item in instances], dtype=torch.long)
    if not torch.equal(loaded_labels, expected_labels):
        raise ValueError("Validation loader changed manifest label order")
    canonical = torch.cat(canonical_parts).double().numpy()
    flipped = torch.cat(flipped_parts).double().numpy()
    averaged = 0.5 * canonical + 0.5 * flipped
    return canonical, flipped, averaged


def validate_recomputed_baseline(
    metrics: dict[str, Any],
    joint: np.ndarray,
    rows: Sequence[dict[str, Any]],
    training_report: dict[str, Any],
) -> dict[str, Any]:
    stored = training_report.get("selected_validation") or {}
    probability = torch.from_numpy(joint).float()
    labels = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    prediction = probability.argmax(1)
    confidence = probability.max(1).values
    correct = prediction == labels
    training_style_ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-7)
        if mask.any():
            training_style_ece += float(
                mask.float().mean()
                * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    comparisons = {
        "species_accuracy": (metrics["species_accuracy"], stored.get("species_accuracy")),
        "conditional_grade_accuracy": (
            metrics["conditional_grade_accuracy"],
            stored.get("grade_accuracy"),
        ),
        "joint_accuracy": (metrics["joint_accuracy"], stored.get("accuracy")),
        "joint_macro_f1": (metrics["joint_macro_f1"], stored.get("macro_f1")),
        "joint_nll": (metrics["joint_nll"], stored.get("nll")),
        "training_style_ece_10bin": (training_style_ece, stored.get("ece")),
    }
    differences: dict[str, float] = {}
    for name, (recomputed, original) in comparisons.items():
        if original is None:
            raise ValueError(f"Training report is missing baseline metric {name}")
        differences[name] = float(recomputed) - float(original)
    # CPU inference can differ in the last few floating-point bits from the MPS
    # training-time pass.  Decisions and discrete metrics must be exact; proper
    # scoring rules are allowed only negligible backend arithmetic differences.
    exact_names = (
        "species_accuracy",
        "conditional_grade_accuracy",
        "joint_accuracy",
        "joint_macro_f1",
    )
    if any(abs(differences[name]) > 5e-8 for name in exact_names):
        raise ValueError(f"Recomputed baseline discrete metrics changed: {differences}")
    if any(
        abs(differences[name]) > 2e-6
        for name in ("joint_nll", "training_style_ece_10bin")
    ):
        raise ValueError(f"Recomputed baseline probability metrics changed: {differences}")
    return {
        "stored": stored,
        "recomputed_training_style_ece_10bin": training_style_ece,
        "note": (
            "The comparison metrics use 15-bin ECE; this 10-bin value exists only "
            "to reproduce the original training report's implementation."
        ),
        "recomputed_minus_stored": differences,
        "passed": True,
    }


def public_probability(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def make_predictions(
    rows: Sequence[dict[str, Any]],
    canonical: np.ndarray,
    flipped: np.ndarray,
    averaged: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        truth = int(row["class_id"])
        baseline_prediction = int(canonical[index].argmax())
        tta_prediction = int(averaged[index].argmax())
        result.append(
            {
                "row_index": index,
                "path": str(row["path"]),
                "group_id": str(row["group_id"]),
                "pair_id": str(row["pair_id"]),
                "truth_class_id": truth,
                "truth_class_name": clean_core.CLASS_NAMES[truth],
                "canonical_joint_probability": public_probability(canonical[index]),
                "horizontal_flip_joint_probability": public_probability(flipped[index]),
                "fixed_tta_joint_probability": public_probability(averaged[index]),
                "baseline_prediction": baseline_prediction,
                "baseline_prediction_name": clean_core.CLASS_NAMES[baseline_prediction],
                "fixed_tta_prediction": tta_prediction,
                "fixed_tta_prediction_name": clean_core.CLASS_NAMES[tta_prediction],
                "prediction_changed": bool(tta_prediction != baseline_prediction),
                "baseline_correct": bool(baseline_prediction == truth),
                "fixed_tta_correct": bool(tta_prediction == truth),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    receipt = require_frozen_inputs()
    output = args.output.resolve()
    clean_core.reject_test_path(output, "fixed TTA output")
    report_path = output / "validation_report.json"
    predictions_path = output / "sample_predictions.json"
    receipt_path = output / "sha256_receipt.json"
    for path in (report_path, predictions_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite one-shot artifact: {path}")

    _, val_instances, manifest_summary = read_role_instances(MANIFEST)
    if manifest_summary["model_selection_instances"] != 146:
        raise ValueError("Expected exactly 146 physical validation instances")
    if manifest_summary["model_selection_groups"] != 9:
        raise ValueError("Expected exactly nine physical validation groups")
    rows = rows_for_metrics(val_instances)
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Neural checkpoint is not a dictionary")
    training_report = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    if Path(str(training_report.get("manifest") or "")).resolve() != MANIFEST.resolve():
        raise ValueError("Training report names a different manifest")
    if Path(str(training_report.get("checkpoint") or "")).resolve() != MODEL.resolve():
        raise ValueError("Training report names a different checkpoint")

    device = torch.device(args.device)
    model = build_model(checkpoint, device)
    canonical, flipped, averaged = infer_fixed_views(
        model, val_instances, checkpoint, device, args.batch_size
    )
    baseline_metrics = evaluate_joint(rows, canonical)
    flip_metrics = evaluate_joint(rows, flipped)
    tta_metrics = evaluate_joint(rows, averaged)
    baseline_consistency = validate_recomputed_baseline(
        baseline_metrics, canonical, rows, training_report
    )

    joint_improved = bool(
        float(tta_metrics["joint_accuracy"]) > float(baseline_metrics["joint_accuracy"])
    )
    macro_improved = bool(
        float(tta_metrics["joint_macro_f1"])
        > float(baseline_metrics["joint_macro_f1"])
    )
    accepted = joint_improved and macro_improved
    status = (
        "recommended_as_validation_candidate_only_dual_metric_improvement"
        if accepted
        else "rejected_no_strict_dual_metric_improvement"
    )

    predictions = {
        "schema": f"{SCRIPT_VERSION}-sample-predictions",
        "selection_scope": "same clean physical model_selection split used by training",
        "strict_test_manifest_opened": False,
        "rows": make_predictions(rows, canonical, flipped, averaged),
    }
    output.mkdir(parents=True, exist_ok=True)
    clean_core.atomic_json_dump(predictions, predictions_path)
    prediction_record = clean_core.fingerprint(predictions_path)

    report = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": clean_core.utc_now(),
        "status": status,
        "accepted": accepted,
        "deployment_changed": False,
        "fixed_rule": {
            "views": ["canonical neutral validation tensor", "horizontal flip"],
            "per_view_joint": "p(species) * p(grade | species)",
            "aggregation": "0.5 * canonical joint probability + 0.5 * flipped joint probability",
            "alternative_views_evaluated": 0,
            "weights_searched": False,
            "temperature_applied_or_fitted": None,
            "threshold_applied_or_fitted": None,
        },
        "acceptance_rule_predeclared_in_source": {
            "joint_accuracy": "strictly greater than canonical baseline",
            "joint_macro_f1": "strictly greater than canonical baseline",
            "both_required": True,
            "deployment_on_acceptance": False,
        },
        "metrics": {
            "canonical_baseline": baseline_metrics,
            "horizontal_flip_audit_only": flip_metrics,
            "fixed_two_view_tta": tta_metrics,
            "tta_minus_baseline": {
                key: float(tta_metrics[key]) - float(baseline_metrics[key])
                for key in (
                    "species_accuracy",
                    "conditional_grade_accuracy",
                    "joint_accuracy",
                    "joint_macro_f1",
                    "group_joint_accuracy",
                    "joint_nll",
                    "joint_ece_15bin",
                )
            },
            "strict_acceptance_checks": {
                "joint_accuracy_improved": joint_improved,
                "joint_macro_f1_improved": macro_improved,
            },
        },
        "baseline_reproduction": baseline_consistency,
        "selection_bias_disclosure": {
            "same_validation_groups_used_by_prior_experiments": True,
            "physical_validation_groups": 9,
            "risk": (
                "many candidate methods have already been compared on these same groups; "
                "even a passing result is a validation-only hypothesis, not an unbiased "
                "generalization estimate"
            ),
            "required_confirmation": "new untouched external camera/conveyor holdout",
        },
        "inputs": {
            "manifest": clean_core.fingerprint(MANIFEST),
            "model": clean_core.fingerprint(MODEL),
            "training_report": clean_core.fingerprint(TRAINING_REPORT),
            "training_receipt": clean_core.fingerprint(TRAINING_RECEIPT),
            "backbone": clean_core.fingerprint(
                Path(str(checkpoint["backbone_checkpoint"]))
            ),
            "training_receipt_attestation": receipt,
        },
        "preprocessing": {
            "checkpoint_description": checkpoint["preprocessing"],
            "image_size": int(checkpoint["image_size"]),
            "object_scale": float(checkpoint["object_scale"]),
            "canonical_render": "PepperValidationDataset exact training-time implementation",
            "flip_stage": "after canonical render, tensor width dimension",
        },
        "manifest_summary": manifest_summary,
        "sample_predictions": prediction_record,
        "protocol": {
            "selection_data": str(MANIFEST.resolve()),
            "physical_validation_rows": len(rows),
            "physical_validation_groups": len({str(row["group_id"]) for row in rows}),
            "strict_test_rows_in_frozen_manifest": 0,
            "strict_test_manifest_opened": False,
            "test_image_loaded": False,
            "test_label_read": False,
            "test_metrics_computed": False,
            "test_arguments_supported": False,
            "fixed_tta_rules_evaluated": 1,
        },
    }
    clean_core.atomic_json_dump(report, report_path)
    report_record = clean_core.fingerprint(report_path)
    hash_receipt = {
        "schema": f"{SCRIPT_VERSION}-sha256-receipt",
        "created_at_utc": clean_core.utc_now(),
        "status": status,
        "script": clean_core.fingerprint(Path(__file__)),
        "inputs": {
            "manifest": clean_core.fingerprint(MANIFEST),
            "model": clean_core.fingerprint(MODEL),
            "training_report": clean_core.fingerprint(TRAINING_REPORT),
            "training_receipt": clean_core.fingerprint(TRAINING_RECEIPT),
            "backbone": clean_core.fingerprint(
                Path(str(checkpoint["backbone_checkpoint"]))
            ),
        },
        "outputs": {
            "report": report_record,
            "sample_predictions": prediction_record,
        },
        "strict_test_manifest_opened": False,
        "deployment_changed": False,
    }
    clean_core.atomic_json_dump(hash_receipt, receipt_path)
    print(
        json.dumps(
            {
                "status": status,
                "accepted": accepted,
                "canonical_baseline": baseline_metrics,
                "fixed_two_view_tta": tta_metrics,
                "report": str(report_path),
                "predictions": str(predictions_path),
                "receipt": str(receipt_path),
                "strict_test_manifest_opened": False,
                "deployment_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
