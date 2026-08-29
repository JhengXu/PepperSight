#!/usr/bin/env python3
"""Fixed-specification clean-only ablation for the premium XGBoost run.

The configurations and tree counts are read from the already frozen train-only
OOF selection.  This program only removes the final-fit-only premium block and
refits those exact specifications, so it performs no new model selection.  It
has no strict-test interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_premium_xgb_v1 as xgb
import train_select_clean_v5 as core


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT / "runs/premium_xgb_v1/selection.json",
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
        "--output",
        type=Path,
        default=PROJECT / "runs/premium_xgb_v1/premium_effect_ablation.json",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.selection,
        args.clean_manifest,
        args.clean_feature,
        args.val_manifest,
        args.val_feature,
        args.output,
    ):
        core.reject_test_path(Path(path).resolve(), "ablation input/output")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen ablation: {output}")
    frozen = json.loads(args.selection.read_text(encoding="utf-8"))
    if frozen.get("schema") != xgb.SCRIPT_VERSION:
        raise ValueError("Selection schema does not match the training program")
    protocol = frozen.get("protocol") or {}
    forbidden = (
        "strict_test_manifest_opened",
        "strict_test_feature_opened",
        "strict_test_labels_read",
        "strict_test_metrics_computed",
        "strict_test_arguments_supported",
    )
    if any(bool(protocol.get(key, True)) for key in forbidden):
        raise ValueError("Frozen selection does not attest a no-test protocol")

    clean = xgb._load_block(args.clean_manifest, args.clean_feature, split="train")
    val = xgb._load_block(args.val_manifest, args.val_feature, split="val")
    if not xgb._leakage_audit(clean.rows, val.rows)["passed"]:
        raise ValueError("Clean-only ablation train/validation leakage audit failed")
    winners = frozen["architecture_winners_from_train_oof"]
    val_labels = np.asarray([row["class_id"] for row in val.rows], dtype=np.int64)

    direct = winners["direct_four_class"]
    direct_trees = int(direct["early_stopping"]["median_for_final_fit"])
    _, _, direct_probability = xgb._fit_final_direct(
        clean.rows,
        clean.features,
        val.features,
        direct["config"],
        direct_trees,
        n_jobs=args.n_jobs,
    )
    direct_species, direct_grade = xgb._joint_to_branches(direct_probability)
    direct_raw = core.branch_metrics(val.rows, direct_species, direct_grade)
    direct_temperature, _, direct_calibrated = xgb._temperature_scale_joint(
        direct_probability, val_labels
    )
    direct_species_cal, direct_grade_cal = xgb._joint_to_branches(direct_calibrated)
    direct_cal = core.branch_metrics(val.rows, direct_species_cal, direct_grade_cal)

    hierarchy = winners["hierarchical"]
    hierarchy_trees = {
        key: int(value["median_for_final_fit"])
        for key, value in hierarchy["early_stopping"].items()
    }
    _, _, _, species, grade = xgb._fit_final_hierarchical(
        clean.rows,
        clean.features,
        val.features,
        hierarchy["config"],
        hierarchy_trees,
        n_jobs=args.n_jobs,
    )
    hierarchy_raw = core.branch_metrics(val.rows, species, grade)
    hierarchy_temperatures, _ = core.fit_temperatures(val.rows, species, grade)
    species_cal, grade_cal = core.apply_temperatures(
        species, grade, hierarchy_temperatures
    )
    hierarchy_cal = core.branch_metrics(val.rows, species_cal, grade_cal)

    premium_validation = frozen["validation"]
    clean_validation = {
        "direct_four_class": {
            "raw": direct_raw,
            "calibrated": direct_cal,
            "temperature": direct_temperature,
        },
        "hierarchical": {
            "raw": hierarchy_raw,
            "calibrated": hierarchy_cal,
            "temperatures": hierarchy_temperatures,
        },
    }
    delta: dict[str, dict[str, float]] = {}
    for architecture in xgb.PREDECLARED_ARCHITECTURES:
        before = clean_validation[architecture]["calibrated"]
        after = premium_validation[architecture]["calibrated"]
        delta[architecture] = {
            key: float(after[key] - before[key])
            for key in (
                "species_accuracy",
                "conditional_grade_accuracy",
                "joint_accuracy",
                "joint_macro_f1",
                "joint_nll",
                "joint_ece_15bin",
            )
        }
    report = {
        "schema": "pepper-premium-xgb-fixed-increment-ablation-v1",
        "protocol": {
            "architecture_and_hyperparameters_reselected": False,
            "tree_counts_reselected": False,
            "temperature_fit": "validation-only reporting for each fixed refit",
            "strict_test_opened": False,
        },
        "frozen_selection": core.fingerprint(args.selection.resolve()),
        "clean_only_validation": clean_validation,
        "clean_plus_premium_validation": premium_validation,
        "premium_minus_clean_only": delta,
    }
    core.atomic_json_dump(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
