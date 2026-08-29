#!/usr/bin/env python3
"""Freeze the post-validation, pre-test premium XGBoost candidate decision.

The train-only OOF experiment compared direct and hierarchical XGBoost.  This
program records the business-architecture constraint that the strict test will
evaluate the hierarchical branch only, and freezes a clean-only hierarchical
model with the exact same config/tree counts for a one-pass premium ablation.
It deliberately exposes no test manifest or test feature argument.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import train_premium_xgb_v1 as xgb
import train_select_clean_v5 as core


PROJECT = Path(__file__).resolve().parent
EXPECTED_CONFIG = "d2_k512_conservative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", type=Path, default=PROJECT / "runs/premium_xgb_v1/selection.json"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT / "runs/premium_xgb_v1/best_premium_xgb_v1.joblib",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=PROJECT / "runs/premium_xgb_v1/premium_effect_ablation.json",
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=(
            PROJECT
            / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
        ),
    )
    parser.add_argument(
        "--clean-feature",
        type=Path,
        default=(
            PROJECT
            / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt"
        ),
    )
    parser.add_argument("--val-manifest", type=Path, default=xgb.DEFAULT_VAL_MANIFEST)
    parser.add_argument("--val-feature", type=Path, default=xgb.DEFAULT_VAL_FEATURE)
    parser.add_argument(
        "--clean-model-output",
        type=Path,
        default=(
            PROJECT
            / "runs/premium_xgb_v1/clean_only_hierarchical_fixed.joblib"
        ),
    )
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=PROJECT / "runs/premium_xgb_v1/test_candidate_decision.json",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=(
            PROJECT / "runs/premium_xgb_v1/test_candidate_decision.receipt.json"
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def _no_test_attested(selection: dict[str, Any]) -> None:
    protocol = selection.get("protocol") or {}
    keys = (
        "strict_test_manifest_opened",
        "strict_test_feature_opened",
        "strict_test_labels_read",
        "strict_test_metrics_computed",
        "strict_test_arguments_supported",
    )
    if any(bool(protocol.get(key, True)) for key in keys):
        raise ValueError("Selection does not attest the required no-test boundary")


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError(f"Frozen ablation mismatch for {name}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    # Data/model inputs must remain test-free.  The two required governance
    # output filenames intentionally contain the word "test"; they are JSON
    # decision records, not data inputs.
    input_paths = (
        args.selection,
        args.model,
        args.ablation,
        args.clean_manifest,
        args.clean_feature,
        args.val_manifest,
        args.val_feature,
        args.clean_model_output,
    )
    for path in input_paths:
        core.reject_test_path(Path(path).resolve(), "pre-test decision input/output")
    for output in (
        args.clean_model_output,
        args.decision_output,
        args.receipt_output,
    ):
        if output.resolve().exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {output}")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    model_payload = joblib.load(args.model.resolve())
    if selection.get("schema") != xgb.SCRIPT_VERSION:
        raise ValueError("Unexpected selection schema")
    if model_payload.get("schema") != xgb.SCRIPT_VERSION:
        raise ValueError("Unexpected model schema")
    _no_test_attested(selection)
    model_record = core.fingerprint(args.model.resolve())
    if model_record["sha256"] != selection["model"]["sha256"]:
        raise ValueError("Frozen model hash differs from selection")
    if ablation.get("protocol", {}).get("strict_test_opened") is not False:
        raise ValueError("Ablation does not attest strict_test_opened=false")

    winner = selection["architecture_winners_from_train_oof"]["hierarchical"]
    if winner["config"]["name"] != EXPECTED_CONFIG:
        raise ValueError(
            f"Hierarchical winner is {winner['config']['name']}, not {EXPECTED_CONFIG}"
        )
    estimators = {
        key: int(value["median_for_final_fit"])
        for key, value in winner["early_stopping"].items()
    }
    clean = xgb._load_block(args.clean_manifest, args.clean_feature, split="train")
    val = xgb._load_block(args.val_manifest, args.val_feature, split="val")
    leakage = xgb._leakage_audit(clean.rows, val.rows)
    if not leakage["passed"]:
        raise ValueError(f"Clean-only train/validation leakage: {leakage}")
    (
        species_model,
        grade_models,
        feature_indices,
        species_probability,
        grade_probability,
    ) = xgb._fit_final_hierarchical(
        clean.rows,
        clean.features,
        val.features,
        winner["config"],
        estimators,
        n_jobs=args.n_jobs,
    )
    temperatures, temperature_objectives = core.fit_temperatures(
        val.rows, species_probability, grade_probability
    )
    species_calibrated, grade_calibrated = core.apply_temperatures(
        species_probability, grade_probability, temperatures
    )
    metrics = core.branch_metrics(val.rows, species_calibrated, grade_calibrated)
    expected = ablation["clean_only_validation"]["hierarchical"]["calibrated"]
    for key in (
        "species_accuracy",
        "conditional_grade_accuracy",
        "joint_accuracy",
        "joint_macro_f1",
        "joint_nll",
        "joint_ece_15bin",
    ):
        _assert_close(metrics[key], expected[key], key)

    clean_model_payload = {
        "schema": "pepper-premium-xgb-clean-only-fixed-hierarchical-v1",
        "architecture": "hierarchical",
        "config": winner["config"],
        "n_estimators": estimators,
        "species_model": species_model,
        "grade_models": grade_models,
        "feature_indices": feature_indices,
        "temperatures": temperatures,
        "temperature_fit_objectives": temperature_objectives,
        "feature_preprocessing": model_payload["feature_preprocessing"],
        "class_names": model_payload["class_names"],
        "species_names": model_payload["species_names"],
        "validation_metrics_calibrated": metrics,
        "provenance": {
            "selection": core.fingerprint(args.selection.resolve()),
            "ablation": core.fingerprint(args.ablation.resolve()),
            "clean_manifest": core.fingerprint(args.clean_manifest.resolve()),
            "clean_feature": core.fingerprint(args.clean_feature.resolve()),
            "validation_manifest": core.fingerprint(args.val_manifest.resolve()),
            "validation_feature": core.fingerprint(args.val_feature.resolve()),
            "strict_test_opened": False,
        },
    }
    core.atomic_joblib_dump(clean_model_payload, args.clean_model_output.resolve())
    clean_model_record = core.fingerprint(args.clean_model_output.resolve())

    direct_oof = selection["architecture_winners_from_train_oof"][
        "direct_four_class"
    ]["train_group_oof_metrics"]
    hierarchy_oof = winner["train_group_oof_metrics"]
    direct_val = selection["validation"]["direct_four_class"]["calibrated"]
    hierarchy_val = selection["validation"]["hierarchical"]["calibrated"]
    premium_delta = ablation["premium_minus_clean_only"]["hierarchical"]
    decision = {
        "schema": "pepper-premium-xgb-test-candidate-decision-v1",
        "created_at_utc": core.utc_now(),
        "decision_stage": "post-validation and before any strict-test access",
        "strict_test_opened_at_decision_time": False,
        "primary_test_candidate": {
            "architecture": "hierarchical",
            "config": winner["config"],
            "model": model_record,
            "model_payload_branch": "models.hierarchical",
            "feature_indices": int(
                len(model_payload["models"]["hierarchical"]["feature_indices"])
            ),
            "n_estimators": estimators,
        },
        "fixed_clean_only_ablation_candidate": {
            "architecture": "hierarchical",
            "config": winner["config"],
            "model": clean_model_record,
            "role": (
                "same-specification clean-only control scored in the same one-pass "
                "strict-test evaluation"
            ),
        },
        "decision_rationale": [
            (
                "The requested production architecture is hierarchical "
                "p(species) * p(grade|species); direct four-class XGBoost is an ablation."
            ),
            (
                "Direct four-class led clean source-group OOF macro-F1 by only "
                f"{100 * (direct_oof['joint_macro_f1'] - hierarchy_oof['joint_macro_f1']):.3f} "
                "percentage points."
            ),
            (
                "On the physically separated validation split, hierarchical exceeded "
                f"direct by {100 * (hierarchy_val['joint_accuracy'] - direct_val['joint_accuracy']):.3f} "
                "points joint accuracy and "
                f"{100 * (hierarchy_val['joint_macro_f1'] - direct_val['joint_macro_f1']):.3f} "
                "points macro-F1."
            ),
            (
                "The fixed clean-only ablation showed positive premium increments only "
                f"for hierarchical: joint {100 * premium_delta['joint_accuracy']:+.3f} "
                f"points and macro-F1 {100 * premium_delta['joint_macro_f1']:+.3f} points."
            ),
        ],
        "selection_transparency": {
            "decision_uses_validation_evidence": True,
            "decision_was_not_the_original_train_oof_architecture_winner": True,
            "original_train_oof_winner": selection[
                "selected_architecture_for_future_strict_test"
            ],
            "business_architecture_constraint_applied_before_test": True,
        },
        "strict_test_precommitment": {
            "single_test_data_open": True,
            "models_scored_in_that_pass": [
                "primary premium hierarchical",
                "fixed same-config clean-only hierarchical control",
            ],
            "direct_four_class_will_not_be_scored": True,
            "no_post_test_candidate_switching": True,
            "no_test_based_tuning_or_recalibration": True,
            "temperatures_are_frozen_from_validation": True,
        },
        "evidence": {
            "selection": core.fingerprint(args.selection.resolve()),
            "ablation": core.fingerprint(args.ablation.resolve()),
            "main_model": model_record,
            "clean_only_model": clean_model_record,
        },
    }
    core.atomic_json_dump(decision, args.decision_output.resolve())
    receipt = {
        "schema": "pepper-premium-xgb-test-candidate-receipt-v1",
        "created_at_utc": core.utc_now(),
        "script": core.fingerprint(Path(__file__)),
        "decision": core.fingerprint(args.decision_output.resolve()),
        "main_model": model_record,
        "selection": core.fingerprint(args.selection.resolve()),
        "ablation": core.fingerprint(args.ablation.resolve()),
        "clean_only_model": clean_model_record,
        "strict_test_opened": False,
    }
    core.atomic_json_dump(receipt, args.receipt_output.resolve())
    print(json.dumps({"decision": decision, "receipt": receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
