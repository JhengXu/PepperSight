#!/usr/bin/env python3
"""One-shot, no-test evaluation of the predeclared clean-v5 branch hybrid.

The fixed hierarchy is deliberately not tunable from the command line:

* p(species): calibrated clean-v5 SVM;
* p(grade | 子弹头): calibrated clean-v5 co-teaching ensemble;
* p(grade | 条子): native partially-unfrozen YOLO11 neural head;
* p(species, grade) = p(species) * p(grade | species), then argmax.

This branch choice was motivated by already-known validation subclass behavior,
so the report explicitly records validation-overfitting risk. There is no test
argument, no blend coefficient, and no alternative rule evaluated by this file.
The fixed hybrid is a candidate only if both joint accuracy and macro-F1 are
strictly greater than every complete frozen parent on physical validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_select_clean_v5 as svm_core
from train_clean_v5_coteaching import SCRIPT_VERSION as COTEACHING_SCHEMA
from train_hierarchical_v4 import HierarchicalHead
from train_hierarchical_v4_unfrozen import (
    EndToEndHierarchicalModel,
    ImageView,
    PartialYOLOEncoder,
    PepperInstance,
    PepperValidationDataset,
)


SCRIPT_VERSION = "pepper-clean-v5-fixed-species-branch-hybrid-validation-v2"
PROJECT = Path(__file__).resolve().parent

VAL_MANIFEST = PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
SVM_MODEL = PROJECT / "runs/hierarchical_v5_clean/best_hierarchical_clean_v5_reaudit_svm.joblib"
SVM_SELECTION = PROJECT / "runs/hierarchical_v5_clean/selection_reaudit.json"
VAL_FEATURE = PROJECT / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt"
COTEACHING_MODEL = (
    PROJECT / "runs/hierarchical_v5_clean/coteaching/best_hierarchical_clean_v5_coteaching.pt"
)
COTEACHING_SELECTION = PROJECT / "runs/hierarchical_v5_clean/coteaching/selection.json"
NEURAL_MODEL = (
    PROJECT
    / "runs/hierarchical_v5_clean_unfrozen_cls_s3041"
    / "best_hierarchical_v4_unfrozen_strict.pt"
)
DEFAULT_OUTPUT = PROJECT / "runs/hierarchical_v5_clean_hybrid_fixed_species_branches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly one frozen clean-v5 SVM/co-teaching/YOLO11 branch "
            "combination on physical validation. No test option exists."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser.parse_args()


def require_frozen_inputs() -> None:
    for path in (
        VAL_MANIFEST,
        SVM_MODEL,
        SVM_SELECTION,
        VAL_FEATURE,
        COTEACHING_MODEL,
        COTEACHING_SELECTION,
        NEURAL_MODEL,
    ):
        resolved = path.resolve()
        svm_core.reject_test_path(resolved, "fixed branch hybrid input")
        if not resolved.is_file():
            raise FileNotFoundError(f"Required frozen input does not exist: {resolved}")
    if VAL_MANIFEST.name != "model_selection_manifest.csv":
        raise ValueError("Only physical model_selection_manifest.csv is permitted")


def probability_temperature_scale(
    probability: np.ndarray, temperature: float
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return svm_core.softmax_numpy(
        np.log(np.clip(probability, 1e-15, 1.0)) / float(temperature)
    )


def load_svm(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], svm_core.LoadedFeatureSet]:
    selection = json.loads(SVM_SELECTION.read_text(encoding="utf-8"))
    selected = selection.get("selected") or {}
    selected_model = selected.get("model") or {}
    expected_digest = str(selected_model.get("sha256") or "")
    if expected_digest != svm_core.sha256_file(SVM_MODEL):
        raise ValueError("SVM selection/model hash mismatch")
    if Path(str(selected_model.get("path") or "")).resolve() != SVM_MODEL.resolve():
        raise ValueError("SVM selection names a different model")
    payload = joblib.load(SVM_MODEL)
    if payload.get("schema") != "pepper-clean-v5-validation-selection-v1":
        raise ValueError(f"Unexpected SVM schema: {payload.get('schema')!r}")
    if payload.get("view_mode") != "canonical":
        raise ValueError("Fixed hybrid requires canonical-view SVM")
    if list(payload.get("feature_families") or ()) != ["imagenet_cls"]:
        raise ValueError("Fixed hybrid requires one-family clean ImageNet SVM")

    loaded = svm_core.load_feature_set(
        [VAL_FEATURE], rows, expected_split="val", expected_manifest=VAL_MANIFEST
    )
    matrix = svm_core.feature_matrix(loaded, "canonical")
    expected_dim = int((payload.get("feature_schema") or {}).get("combined_dim") or 0)
    if matrix.shape != (len(rows), expected_dim):
        raise ValueError(f"SVM feature shape {matrix.shape} != {(len(rows), expected_dim)}")
    raw_species = svm_core.ordered_binary_probability(payload["species_model"], matrix)
    raw_grade = np.stack(
        [svm_core.ordered_binary_probability(model, matrix) for model in payload["grade_models"]],
        axis=1,
    )
    temperatures = selected.get("temperatures") or {}
    payload_temperatures = payload.get("temperatures") or {}
    keys = ("species", "grade_given_子弹头", "grade_given_条子")
    if any(
        not np.isclose(
            float(temperatures[key]), float(payload_temperatures[key]), atol=0, rtol=1e-12
        )
        for key in keys
    ):
        raise ValueError("SVM payload and selection temperatures differ")
    calibrated_species, calibrated_grade = svm_core.apply_temperatures(
        raw_species, raw_grade, [float(temperatures[key]) for key in keys]
    )
    provenance = {
        "model": svm_core.fingerprint(SVM_MODEL),
        "selection": svm_core.fingerprint(SVM_SELECTION),
        "validation_feature": svm_core.fingerprint(VAL_FEATURE),
        "feature_schema": payload["feature_schema"],
        "view_mode": payload["view_mode"],
        "temperatures": temperatures,
    }
    return calibrated_species, calibrated_grade, provenance, payload, loaded


def load_coteaching(
    loaded_features: svm_core.LoadedFeatureSet, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    selection = json.loads(COTEACHING_SELECTION.read_text(encoding="utf-8"))
    selected = selection.get("selected") or {}
    model_record = selected.get("model") or {}
    if str(model_record.get("sha256") or "") != svm_core.sha256_file(COTEACHING_MODEL):
        raise ValueError("Co-teaching selection/model hash mismatch")
    if Path(str(model_record.get("path") or "")).resolve() != COTEACHING_MODEL.resolve():
        raise ValueError("Co-teaching selection names a different model")
    payload = torch.load(COTEACHING_MODEL, map_location="cpu", weights_only=False)
    if payload.get("schema") != COTEACHING_SCHEMA:
        raise ValueError(f"Unexpected co-teaching schema: {payload.get('schema')!r}")
    strict = payload.get("strict_protocol") or {}
    required = {
        "historical_classification_head_loaded": False,
        "head_randomly_initialized": True,
        "frozen_backbone_features_only": True,
        "test_manifest_opened": False,
        "test_feature_opened": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
    }
    for key, expected in required.items():
        if strict.get(key) is not expected:
            raise ValueError(f"Co-teaching checkpoint violates strict_protocol.{key}")
    architecture = payload.get("architecture") or {}
    feature_dim = int(payload["feature_dim"])
    if len(loaded_features.blocks) != 1 or loaded_features.blocks[0].shape[-1] != feature_dim:
        raise ValueError("Co-teaching feature family/dimension mismatch")
    # Match train_hierarchical_v4.load_feature_caches exactly.
    feature = F.normalize(loaded_features.blocks[0], dim=-1) * (feature_dim**0.5)
    models: list[HierarchicalHead] = []
    for state in payload["member_state_dicts"]:
        model = HierarchicalHead(
            feature_dim,
            int(architecture["hidden_dim"]),
            int(architecture["grade_hidden_dim"]),
            dropout=0.0,
            grade_dropout=0.0,
        ).to(device)
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
    species_parts: list[torch.Tensor] = []
    grade_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for model in models:
            species_logits, grade_logits = model(feature[:, 0].to(device))
            species_parts.append(species_logits.softmax(1).cpu())
            grade_parts.append(grade_logits.softmax(2).cpu())
    raw_species = torch.stack(species_parts).mean(0).numpy().astype(np.float64)
    raw_grade = torch.stack(grade_parts).mean(0).numpy().astype(np.float64)
    temperatures = selected.get("temperatures") or {}
    payload_temperatures = payload.get("temperatures") or {}
    keys = ("species", "grade_given_子弹头", "grade_given_条子")
    if any(
        not np.isclose(
            float(temperatures[key]), float(payload_temperatures[key]), atol=0, rtol=1e-12
        )
        for key in keys
    ):
        raise ValueError("Co-teaching payload and selection temperatures differ")
    calibrated_species, calibrated_grade = svm_core.apply_temperatures(
        raw_species, raw_grade, [float(temperatures[key]) for key in keys]
    )
    provenance = {
        "model": svm_core.fingerprint(COTEACHING_MODEL),
        "selection": svm_core.fingerprint(COTEACHING_SELECTION),
        "feature": svm_core.fingerprint(VAL_FEATURE),
        "architecture": architecture,
        "strict_protocol": strict,
        "temperatures": temperatures,
        "selected_candidate": selected.get("candidate"),
    }
    return calibrated_species, calibrated_grade, provenance, payload


def validation_instances(rows: Sequence[dict[str, Any]]) -> list[PepperInstance]:
    instances: list[PepperInstance] = []
    for row in rows:
        class_id = int(row["class_id"])
        species_id, grade_id = divmod(class_id, 2)
        instances.append(
            PepperInstance(
                pair_id=str(row["pair_id"]),
                group_id=str(row["group_id"]),
                class_id=class_id,
                species_target=(1.0, 0.0) if species_id == 0 else (0.0, 1.0),
                grade_target=(1.0, 0.0) if grade_id == 0 else (0.0, 1.0),
                species_weight=1.0,
                grade_weight=1.0,
                high_consistency=True,
                views=(ImageView(path=str(row["path"]), view_type="canonical"),),
            )
        )
    return instances


def build_neural_model(
    checkpoint: dict[str, Any], device: torch.device
) -> EndToEndHierarchicalModel:
    architecture = checkpoint.get("architecture") or {}
    if architecture.get("type") != "partially_unfrozen_yolo11_hierarchical_v4":
        raise ValueError(f"Unexpected neural architecture: {architecture.get('type')!r}")
    strict = checkpoint.get("strict_protocol") or {}
    required = {
        "historical_pepper_classification_head_loaded": False,
        "test_rows_materialized": False,
        "test_images_loaded": False,
        "test_metrics_computed": False,
    }
    for key, expected in required.items():
        if strict.get(key) is not expected:
            raise ValueError(f"Neural checkpoint violates strict_protocol.{key}")
    backbone = Path(str(checkpoint["backbone_checkpoint"])).resolve()
    svm_core.reject_test_path(backbone, "neural backbone")
    if not backbone.is_file():
        raise FileNotFoundError(f"Neural backbone is missing: {backbone}")
    encoder = PartialYOLOEncoder(
        backbone,
        str(architecture["backbone_kind"]),
        int(architecture["unfreeze_from"]),
        int(architecture.get("feature_dim") or 512),
        bool(architecture["train_bn"]),
    )
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


def load_neural(
    rows: Sequence[dict[str, Any]], device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(NEURAL_MODEL, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Neural checkpoint is not a dictionary")
    model = build_neural_model(checkpoint, device)
    dataset = PepperValidationDataset(
        validation_instances(rows),
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
    species_parts: list[torch.Tensor] = []
    grade_parts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            species_logits, grade_logits = model(images.to(device))
            species_parts.append(species_logits.softmax(1).cpu())
            grade_parts.append(grade_logits.softmax(2).cpu())
            labels.append(batch_labels.cpu())
    if not torch.equal(
        torch.cat(labels).long(),
        torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long),
    ):
        raise ValueError("Neural validation loader changed manifest label order")
    species_probability = torch.cat(species_parts).numpy().astype(np.float64)
    grade_probability = torch.cat(grade_parts).numpy().astype(np.float64)
    provenance = {
        "model": svm_core.fingerprint(NEURAL_MODEL),
        "backbone_checkpoint": svm_core.fingerprint(
            Path(str(checkpoint["backbone_checkpoint"]))
        ),
        "architecture": checkpoint["architecture"],
        "strict_protocol": checkpoint["strict_protocol"],
        "preprocessing": checkpoint["preprocessing"],
        "image_size": int(checkpoint["image_size"]),
        "object_scale": float(checkpoint["object_scale"]),
        "conditional_grade_temperature": None,
    }
    return species_probability, grade_probability, provenance, checkpoint


def joint_probability(
    species_probability: np.ndarray, grade_probability: np.ndarray
) -> np.ndarray:
    probability = (species_probability[:, :, None] * grade_probability).reshape(-1, 4)
    return probability / np.clip(probability.sum(1, keepdims=True), 1e-15, None)


def public_probabilities(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def make_predictions(
    rows: Sequence[dict[str, Any]],
    svm_species: np.ndarray,
    svm_grade: np.ndarray,
    coteaching_grade: np.ndarray,
    neural_grade: np.ndarray,
    hybrid_grade: np.ndarray,
) -> list[dict[str, Any]]:
    joint = joint_probability(svm_species, hybrid_grade)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        truth = int(row["class_id"])
        prediction = int(joint[index].argmax())
        result.append(
            {
                "row_index": index,
                "path": str(row["path"]),
                "group_id": str(row["group_id"]),
                "source_id": str(row["source_id"]),
                "pair_id": str(row["pair_id"]),
                "truth_class_id": truth,
                "truth_class_name": svm_core.CLASS_NAMES[truth],
                "svm_species_probability_calibrated": public_probabilities(svm_species[index]),
                "svm_grade_given_子弹头_probability_audit_only": public_probabilities(
                    svm_grade[index, 0]
                ),
                "coteaching_grade_given_子弹头_probability_calibrated": public_probabilities(
                    coteaching_grade[index, 0]
                ),
                "neural_grade_given_条子_probability_native": public_probabilities(
                    neural_grade[index, 1]
                ),
                "fixed_hybrid_grade_probability": [
                    public_probabilities(hybrid_grade[index, 0]),
                    public_probabilities(hybrid_grade[index, 1]),
                ],
                "fixed_hybrid_joint_probability": public_probabilities(joint[index]),
                "fixed_hybrid_prediction": prediction,
                "fixed_hybrid_prediction_name": svm_core.CLASS_NAMES[prediction],
                "fixed_hybrid_correct": bool(prediction == truth),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    require_frozen_inputs()
    output = args.output.resolve()
    svm_core.reject_test_path(output, "fixed hybrid output")
    report_path = output / "validation_report.json"
    predictions_path = output / "branch_predictions.json"
    receipt_path = output / "hash_receipt.json"
    for path in (report_path, predictions_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite one-shot artifact: {path}")

    rows = svm_core.read_manifest(
        VAL_MANIFEST, expected_split="val", expected_rows=svm_core.EXPECTED_VAL_ROWS
    )
    device = torch.device(args.device)
    svm_species, svm_grade, svm_provenance, svm_payload, loaded_features = load_svm(rows)
    coteaching_species, coteaching_grade, coteaching_provenance, _ = load_coteaching(
        loaded_features, device
    )
    neural_species, neural_grade, neural_provenance, neural_checkpoint = load_neural(
        rows, device, args.batch_size
    )

    hybrid_grade = np.empty_like(svm_grade)
    hybrid_grade[:, 0] = coteaching_grade[:, 0]
    hybrid_grade[:, 1] = neural_grade[:, 1]
    parent_metrics = {
        "clean_svm": svm_core.branch_metrics(rows, svm_species, svm_grade),
        "clean_coteaching": svm_core.branch_metrics(rows, coteaching_species, coteaching_grade),
        "partially_unfrozen_yolo11": svm_core.branch_metrics(
            rows, neural_species, neural_grade
        ),
    }
    hybrid_metrics = svm_core.branch_metrics(rows, svm_species, hybrid_grade)
    reference_accuracy = max(
        float(metrics["joint_accuracy"]) for metrics in parent_metrics.values()
    )
    reference_macro_f1 = max(
        float(metrics["joint_macro_f1"]) for metrics in parent_metrics.values()
    )
    accepted = bool(
        float(hybrid_metrics["joint_accuracy"]) > reference_accuracy
        and float(hybrid_metrics["joint_macro_f1"]) > reference_macro_f1
    )
    status = (
        "accepted_validation_candidate_not_for_default_deployment"
        if accepted
        else "rejected_no_strict_dual_metric_improvement"
    )

    predictions = {
        "schema": f"{SCRIPT_VERSION}-branch-predictions",
        "selection_scope": "physical model_selection manifest only",
        "strict_test_manifest_opened": False,
        "rows": make_predictions(
            rows, svm_species, svm_grade, coteaching_grade, neural_grade, hybrid_grade
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    svm_core.atomic_json_dump(predictions, predictions_path)
    prediction_record = svm_core.fingerprint(predictions_path)

    report = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": svm_core.utc_now(),
        "status": status,
        "accepted": accepted,
        "deployment_changed": False,
        "fixed_rule": {
            "species": "calibrated clean-v5 SVM p(species)",
            "grade_given_子弹头": "calibrated selected clean-v5 co-teaching p(grade|子弹头)",
            "grade_given_条子": "native partially-unfrozen YOLO11 p(grade|条子)",
            "joint": "p(species) * p(grade|species), flatten to four classes, argmax",
            "continuous_blend_weights": None,
            "alternative_hybrid_rules_evaluated": 0,
        },
        "selection_disclosure": {
            "validation_subclass_results_previously_known": True,
            "reason_for_子弹头_branch": (
                "co-teaching previously had stronger 子弹头_一级 validation recall/F1; "
                "this low-effective-mass class benefits from the anti-noise branch"
            ),
            "reason_for_条子_branch": (
                "partially-unfrozen YOLO11 previously had stronger validation F1 for both 条子 grades"
            ),
            "risk": (
                "the discrete branch rule was inspired by known validation subclass performance "
                "and may overfit the nine physical validation groups"
            ),
            "claim_scope": "validation candidate only; requires a new untouched external holdout",
        },
        "acceptance_rule_predeclared_in_source": {
            "joint_accuracy": "strictly greater than every complete frozen parent",
            "joint_macro_f1": "strictly greater than every complete frozen parent",
            "both_required": True,
            "deployment_on_acceptance": False,
        },
        "metrics": {
            "parents_recomputed": parent_metrics,
            "fixed_hybrid": hybrid_metrics,
            "strongest_parent_reference": {
                "joint_accuracy": reference_accuracy,
                "joint_macro_f1": reference_macro_f1,
            },
            "fixed_hybrid_delta_vs_strongest_parent": {
                "joint_accuracy": float(hybrid_metrics["joint_accuracy"])
                - reference_accuracy,
                "joint_macro_f1": float(hybrid_metrics["joint_macro_f1"])
                - reference_macro_f1,
            },
        },
        "stored_metric_consistency": {
            "svm_parent": svm_payload["validation_metrics_calibrated"],
            "coteaching_parent": json.loads(COTEACHING_SELECTION.read_text(encoding="utf-8"))[
                "selected"
            ]["validation_metrics_calibrated"],
            "neural_parent": neural_checkpoint["validation_metrics"],
        },
        "inputs": {
            "manifest": svm_core.fingerprint(VAL_MANIFEST),
            "svm": svm_provenance,
            "coteaching": coteaching_provenance,
            "partially_unfrozen_yolo11": neural_provenance,
        },
        "branch_predictions": prediction_record,
        "protocol": {
            "selection_data": str(VAL_MANIFEST.resolve()),
            "physical_manifest_rows": len(rows),
            "physical_manifest_groups": len({str(row["group_id"]) for row in rows}),
            "strict_test_manifest_opened": False,
            "test_image_loaded": False,
            "test_label_read": False,
            "test_metrics_computed": False,
            "test_arguments_supported": False,
            "fixed_hybrid_rule_evaluations": 1,
        },
    }
    svm_core.atomic_json_dump(report, report_path)
    report_record = svm_core.fingerprint(report_path)
    receipt = {
        "schema": f"{SCRIPT_VERSION}-hash-receipt",
        "created_at_utc": svm_core.utc_now(),
        "status": status,
        "script": svm_core.fingerprint(Path(__file__)),
        "inputs": {
            "manifest": svm_core.fingerprint(VAL_MANIFEST),
            "validation_feature": svm_core.fingerprint(VAL_FEATURE),
            "svm_model": svm_core.fingerprint(SVM_MODEL),
            "svm_selection": svm_core.fingerprint(SVM_SELECTION),
            "coteaching_model": svm_core.fingerprint(COTEACHING_MODEL),
            "coteaching_selection": svm_core.fingerprint(COTEACHING_SELECTION),
            "neural_model": svm_core.fingerprint(NEURAL_MODEL),
            "neural_backbone": neural_provenance["backbone_checkpoint"],
        },
        "outputs": {"report": report_record, "branch_predictions": prediction_record},
        "strict_test_manifest_opened": False,
    }
    svm_core.atomic_json_dump(receipt, receipt_path)
    print(
        json.dumps(
            {
                "status": status,
                "accepted": accepted,
                "parents": parent_metrics,
                "fixed_hybrid": hybrid_metrics,
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
