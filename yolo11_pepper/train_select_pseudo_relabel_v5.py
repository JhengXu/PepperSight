#!/usr/bin/env python3
"""Strict train-only pseudo-grade relabel experiment for pepper clean-v5.

Only rows already marked as high-confidence, cross-fitted grade contradictions are
eligible.  Species labels are immutable.  A small source-code-fixed policy grid is
selected with five-fold source-group out-of-fold predictions on reliable *training*
labels.  The winning policy is then frozen, fitted on all training groups and
evaluated/calibrated on the physically separate model-selection split exactly once.

There is deliberately no final-holdout command line argument or execution path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score

from train_select_clean_v5 import (
    CLASS_NAMES,
    SPECIES_NAMES,
    apply_temperatures,
    atomic_joblib_dump,
    atomic_json_dump,
    branch_metrics,
    compare_feature_schemas,
    feature_matrix,
    fingerprint,
    fit_candidate,
    fit_temperatures,
    leakage_audit,
    load_feature_set,
    pair_aware_balanced_weight,
    pair_weight_audit,
    read_manifest,
    reject_test_path,
    sha256_json,
    utc_now,
)


SCRIPT_VERSION = "pepper-clean-v5-pseudo-relabel-selection-v1"
PROJECT = Path(__file__).resolve().parent
EXPECTED_TRAIN_ROWS = 697
EXPECTED_VAL_ROWS = 146
RANDOM_SEED = 2041

SOURCE_TRAIN_MANIFEST = (
    PROJECT
    / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
)
VAL_MANIFEST = PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
TRAIN_FEATURES = (
    PROJECT
    / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt"
)
VAL_FEATURES = (
    PROJECT / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt"
)
OUTPUT_DIR = PROJECT / "runs/hierarchical_v5_pseudo_relabel"
DATASET_DIR = PROJECT / "datasets/pepper_ssl_v5_pseudo_relabel"

# Small, immutable grid declared before this script opens any validation labels.
# Every pseudo weight is below the minimum 0.35 retained human hard-label weight.
POLICIES: tuple[dict[str, Any], ...] = (
    {
        "name": "no_pseudo_flip",
        "minimum_opposite_probability": None,
        "pseudo_grade_weight": 0.0,
    },
    {
        "name": "opposite_p80_weight015",
        "minimum_opposite_probability": 0.80,
        "pseudo_grade_weight": 0.15,
    },
    {
        "name": "opposite_p90_weight015",
        "minimum_opposite_probability": 0.90,
        "pseudo_grade_weight": 0.15,
    },
    {
        "name": "opposite_p90_weight025",
        "minimum_opposite_probability": 0.90,
        "pseudo_grade_weight": 0.25,
    },
)
PREDECLARED_C_GRID = (3.0, 10.0, 30.0)
PREDECLARED_VIEW_MODES = ("canonical", "view_mean")
STRICT_SPECIES_PROBABILITY = 0.90
STRICT_VIEW_STABILITY = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only grouped-OOF pseudo-grade relabel selection."
    )
    parser.add_argument("--source-train-manifest", type=Path, default=SOURCE_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--train-features", type=Path, default=TRAIN_FEATURES)
    parser.add_argument("--validation-features", type=Path, default=VAL_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    return parser.parse_args()


def as_float(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for {row['path']}")
    return value


def verify_audit_columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "audit_status",
        "oof_fold",
        "oof_p_original_species",
        "oof_p_original_grade",
        "oof_species_view_stability",
        "oof_grade_view_stability",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Audit manifest misses pseudo-label evidence: {sorted(missing)}")
    group_folds: dict[str, int] = {}
    for row in rows:
        fold = int(row["oof_fold"])
        if fold not in range(5):
            raise ValueError(f"Invalid oof_fold={fold} for {row['path']}")
        previous = group_folds.setdefault(str(row["group_id"]), fold)
        if previous != fold:
            raise ValueError(f"Source group crosses OOF folds: {row['group_id']}")
    if set(group_folds.values()) != set(range(5)):
        raise ValueError("Expected all five source-group OOF folds")
    return {
        "groups": len(group_folds),
        "fold_group_counts": dict(Counter(group_folds.values())),
        "group_to_fold": group_folds,
    }


def eligible_pairs(
    rows: Sequence[dict[str, Any]], minimum_opposite_probability: float | None
) -> dict[str, dict[str, Any]]:
    if minimum_opposite_probability is None:
        return {}
    canonical_by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["view_type"] == "canonical":
            canonical_by_pair[str(row["pair_id"])] = row
    result: dict[str, dict[str, Any]] = {}
    for pair_id, row in canonical_by_pair.items():
        opposite_probability = 1.0 - as_float(row, "oof_p_original_grade")
        gates = {
            "audit_status": row["audit_status"]
            == "manual_review_strict_grade_contradiction",
            "species_probability": as_float(row, "oof_p_original_species")
            >= STRICT_SPECIES_PROBABILITY,
            "opposite_grade_probability": opposite_probability
            >= minimum_opposite_probability,
            "species_view_stability": as_float(row, "oof_species_view_stability")
            >= STRICT_VIEW_STABILITY,
            "grade_view_stability": as_float(row, "oof_grade_view_stability")
            >= STRICT_VIEW_STABILITY,
        }
        if all(gates.values()):
            old_class = int(row["class_id"])
            result[pair_id] = {
                "old_class_id": old_class,
                "new_class_id": old_class ^ 1,
                "opposite_probability": opposite_probability,
                "group_id": str(row["group_id"]),
                "source_id": str(row["source_id"]),
                "gates": gates,
            }
    return result


def apply_policy(
    source_rows: Sequence[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = eligible_pairs(source_rows, policy["minimum_opposite_probability"])
    output: list[dict[str, Any]] = []
    for source in source_rows:
        row = deepcopy(source)
        pair_id = str(row["pair_id"])
        evidence = selected.get(pair_id)
        row["pseudo_relabel_eligible"] = "true" if evidence else "false"
        row["pseudo_relabel_applied"] = "false"
        row["pseudo_relabel_policy"] = str(policy["name"])
        row["pseudo_original_class_id"] = str(source["class_id"])
        row["pseudo_opposite_probability"] = (
            f"{evidence['opposite_probability']:.8f}" if evidence else ""
        )
        row["pseudo_grade_weight"] = "0.0000"
        if evidence:
            new_class = int(evidence["new_class_id"])
            row["class_id"] = new_class
            row["class_name"] = CLASS_NAMES[new_class].replace("一级", "好").replace("二级", "差")
            row["grade_weight"] = f"{float(policy['pseudo_grade_weight']):.4f}"
            row["pseudo_grade_weight"] = row["grade_weight"]
            row["pseudo_relabel_applied"] = "true"
            row["label_state"] = "pseudo_grade_relabel_oof"
            row["audit_status"] = "pseudo_grade_relabel_oof"
            row["audit_reasons"] = (
                f"{source.get('audit_reasons', '')};pseudo_flip_from_strict_crossfit_evidence"
            ).strip(";")
            grade = new_class % 2
            row["grade_soft_target"] = "[0.96, 0.04]" if grade == 0 else "[0.04, 0.96]"
        output.append(row)

    labels_by_pair: dict[str, set[int]] = {}
    weights_by_pair: dict[str, set[float]] = {}
    for row in output:
        pair_id = str(row["pair_id"])
        labels_by_pair.setdefault(pair_id, set()).add(int(row["class_id"]))
        weights_by_pair.setdefault(pair_id, set()).add(float(row["grade_weight"]))
    if any(len(values) != 1 for values in labels_by_pair.values()):
        raise ValueError("Pseudo policy produced inconsistent labels across paired views")
    if any(len(values) != 1 for values in weights_by_pair.values()):
        raise ValueError("Pseudo policy produced inconsistent weights across paired views")
    return output, selected


def balanced_weights(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    base, _ = pair_weight_audit(rows)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    species = labels // 2
    grade = labels % 2
    raw_species = np.asarray([float(row["species_weight"]) for row in rows]) * base
    raw_grade = np.asarray([float(row["grade_weight"]) for row in rows]) * base
    species_weight, _ = pair_aware_balanced_weight(
        species, raw_species, label_names=SPECIES_NAMES
    )
    grade_weight = np.zeros_like(raw_grade)
    for species_id in range(2):
        keep = species == species_id
        grade_weight[keep], _ = pair_aware_balanced_weight(
            grade[keep], raw_grade[keep], label_names=("一级", "二级")
        )
    return species_weight, grade_weight


def oof_metrics(
    source_rows: Sequence[dict[str, Any]],
    policy_rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    c_value: float,
) -> dict[str, Any]:
    all_species: list[np.ndarray] = []
    all_grade: list[np.ndarray] = []
    evaluation_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    folds = np.asarray([int(row["oof_fold"]) for row in source_rows])
    for fold in range(5):
        train_indices = np.flatnonzero(folds != fold)
        # Canonical-only scoring prevents paired detector crops from being counted
        # as independent evidence. Disputed/unlabelled grades never score a policy.
        eval_indices = np.asarray(
            [
                i
                for i, row in enumerate(source_rows)
                if int(row["oof_fold"]) == fold
                and row["view_type"] == "canonical"
                and float(row["species_weight"]) > 0
                and float(row["grade_weight"]) > 0
            ],
            dtype=np.int64,
        )
        train_subset = [policy_rows[i] for i in train_indices]
        species_weight, grade_weight = balanced_weights(train_subset)
        labels = np.asarray([int(row["class_id"]) for row in train_subset])
        _, _, species_probability, grade_probability = fit_candidate(
            matrix[train_indices],
            matrix[eval_indices],
            labels,
            species_weight,
            grade_weight,
            c_value,
        )
        all_species.append(species_probability)
        all_grade.append(grade_probability)
        evaluation_rows.extend(source_rows[i] for i in eval_indices)
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "train_groups": len({source_rows[i]["group_id"] for i in train_indices}),
                "scored_canonical_rows": int(len(eval_indices)),
                "scored_groups": len({source_rows[i]["group_id"] for i in eval_indices}),
            }
        )
    metrics = branch_metrics(
        evaluation_rows,
        np.concatenate(all_species),
        np.concatenate(all_grade),
    )
    metrics["folds"] = fold_summaries
    metrics["score_population"] = (
        "canonical rows with positive original species and grade audit weight"
    )
    return metrics


def atomic_csv_dump(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {destination}")
    fieldnames = list(rows[0])
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_dump(value: Any, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen feature cache: {destination}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    paths = (
        args.source_train_manifest,
        args.validation_manifest,
        args.train_features,
        args.validation_features,
        args.output_dir,
        args.dataset_dir,
    )
    for path in paths:
        reject_test_path(path.resolve(), "input/output path")
    output_dir = args.output_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    model_path = output_dir / "best_hierarchical_pseudo_relabel_v5_svm.joblib"
    selection_path = output_dir / "selection.json"
    receipt_path = output_dir / "selection.json.sha256.json"
    manifest_path = dataset_dir / "train_manifest.csv"
    cache_path = output_dir / "imagenet_cls_train_pseudo_relabel.pt"
    for output in (model_path, selection_path, receipt_path, manifest_path, cache_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {output}")

    source_rows = read_manifest(
        args.source_train_manifest,
        expected_split="train",
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    val_rows = read_manifest(
        args.validation_manifest, expected_split="val", expected_rows=EXPECTED_VAL_ROWS
    )
    group_audit = verify_audit_columns(source_rows)
    leak_audit = leakage_audit(source_rows, val_rows)
    if not leak_audit["passed"]:
        raise ValueError(f"Physical split leakage audit failed: {leak_audit}")

    train_features = load_feature_set(
        [args.train_features],
        source_rows,
        expected_split="train",
        expected_manifest=args.source_train_manifest,
    )
    val_features = load_feature_set(
        [args.validation_features],
        val_rows,
        expected_split="val",
        expected_manifest=args.validation_manifest,
    )
    compare_feature_schemas(train_features, val_features)

    candidates: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf, -math.inf)
    selected: dict[str, Any] | None = None
    for view_mode in PREDECLARED_VIEW_MODES:
        matrix = feature_matrix(train_features, view_mode)
        for policy in POLICIES:
            policy_rows, evidence = apply_policy(source_rows, policy)
            for c_value in PREDECLARED_C_GRID:
                metrics = oof_metrics(source_rows, policy_rows, matrix, c_value)
                candidate = {
                    "policy": policy,
                    "eligible_unique_pairs": len(evidence),
                    "eligible_groups": len({item["group_id"] for item in evidence.values()}),
                    "view_mode": view_mode,
                    "C": c_value,
                    "training_oof_metrics": metrics,
                }
                candidates.append(candidate)
                # Grade accuracy is primary because species labels never change;
                # macro-F1 and joint accuracy resolve ties. Exact ties retain the
                # earlier fixed grid order, which starts with the no-flip control.
                key = (
                    metrics["conditional_grade_accuracy"],
                    metrics["joint_macro_f1"],
                    metrics["joint_accuracy"],
                )
                if key > best_key:
                    best_key = key
                    selected = candidate
                print(json.dumps(candidate, ensure_ascii=False))
    if selected is None:
        raise RuntimeError("No grouped-OOF candidate was selected")

    selected_rows, evidence = apply_policy(source_rows, selected["policy"])
    selected_matrix = feature_matrix(train_features, selected["view_mode"])
    validation_matrix = feature_matrix(val_features, selected["view_mode"])
    species_weight, grade_weight = balanced_weights(selected_rows)
    selected_labels = np.asarray([int(row["class_id"]) for row in selected_rows])
    species_model, grade_models, species_probability, grade_probability = fit_candidate(
        selected_matrix,
        validation_matrix,
        selected_labels,
        species_weight,
        grade_weight,
        float(selected["C"]),
    )
    # This is the sole physical validation scoring pass for the already frozen
    # train-only policy/hyperparameter specification.
    validation_raw = branch_metrics(val_rows, species_probability, grade_probability)
    temperatures, temperature_objectives = fit_temperatures(
        val_rows, species_probability, grade_probability
    )
    calibrated = apply_temperatures(species_probability, grade_probability, temperatures)
    validation_calibrated = branch_metrics(val_rows, *calibrated)

    atomic_csv_dump(selected_rows, manifest_path)
    original_cache = torch.load(args.train_features, map_location="cpu", weights_only=True)
    derived_cache = dict(original_cache)
    derived_cache["class_ids"] = torch.tensor(selected_labels, dtype=torch.long)
    derived_cache["metadata"] = {
        **dict(original_cache.get("metadata") or {}),
        "manifest": str(manifest_path),
        "source_feature_cache": str(args.train_features.resolve()),
        "source_feature_cache_sha256": fingerprint(args.train_features)["sha256"],
        "pseudo_relabel_policy": selected["policy"],
        "pseudo_relabel_selection": "five-fold source-group OOF on training only",
    }
    atomic_torch_dump(derived_cache, cache_path)

    feature_schema = {
        "families": [family.public() for family in train_features.families],
        "concatenation_order": [family.name for family in train_features.families],
        "combined_dim": int(sum(family.dim for family in train_features.families)),
        "normalization": "per-view L2; view_mean re-normalized after averaging",
    }
    protocol = {
        "stage": "train_only_grouped_oof_pseudo_relabel_then_single_validation",
        "selection_metric": "training OOF conditional grade accuracy, then joint macro-F1, then joint accuracy",
        "oof_score_population": "canonical, positive original audited species+grade weight only",
        "disputed_rows_scored_in_oof": False,
        "species_labels_mutated": False,
        "validation_used_for_policy_or_C_selection": False,
        "physical_validation_scoring_passes_after_freeze": 1,
        "final_holdout_arguments_supported": False,
        "final_holdout_paths_opened": False,
    }
    model_payload = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "species_model": species_model,
        "grade_models": grade_models,
        "feature_families": [family.name for family in train_features.families],
        "feature_schema": feature_schema,
        "view_mode": selected["view_mode"],
        "C": selected["C"],
        "pseudo_relabel_policy": selected["policy"],
        "pseudo_relabel_unique_pairs": len(evidence),
        "temperatures": {
            "species": temperatures[0],
            "grade_given_子弹头": temperatures[1],
            "grade_given_条子": temperatures[2],
            "application": "softmax(log(branch_probability) / temperature)",
        },
        "validation_metrics_raw": validation_raw,
        "validation_metrics_calibrated": validation_calibrated,
        "protocol": protocol,
    }
    atomic_joblib_dump(model_payload, model_path)
    model_record = fingerprint(model_path)
    frozen_core = {
        "model": model_record,
        "policy": selected["policy"],
        "view_mode": selected["view_mode"],
        "C": selected["C"],
        "feature_schema": feature_schema,
        "temperatures": model_payload["temperatures"],
    }
    selection_record = {
        "schema": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "selection_id": f"pepper-clean-v5-pseudo-{sha256_json(frozen_core)[:20]}",
        "protocol": protocol,
        "guardrails": {
            "physical_split_leakage_audit": leak_audit,
            "group_oof_audit": group_audit,
            "species_labels_mutated": False,
            "automatic_candidates_limited_to_prior_strict_grade_contradictions": True,
            "pseudo_weight_below_minimum_human_hard_weight": True,
            "final_holdout_paths_opened": False,
        },
        "input_fingerprints": {
            "script": fingerprint(Path(__file__)),
            "source_train_manifest": fingerprint(args.source_train_manifest),
            "validation_manifest": fingerprint(args.validation_manifest),
            "train_features": fingerprint(args.train_features),
            "validation_features": fingerprint(args.validation_features),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "candidate_space": {
            "policies": list(POLICIES),
            "C_grid": list(PREDECLARED_C_GRID),
            "view_modes": list(PREDECLARED_VIEW_MODES),
            "candidate_count": len(candidates),
            "strict_species_probability": STRICT_SPECIES_PROBABILITY,
            "strict_view_stability": STRICT_VIEW_STABILITY,
        },
        "candidates": candidates,
        "selected": {
            **frozen_core,
            "eligible_unique_pairs": len(evidence),
            "eligible_groups": len({item["group_id"] for item in evidence.values()}),
            "evidence_by_pair": evidence,
            "training_oof_metrics": selected["training_oof_metrics"],
            "temperature_fit_objectives": temperature_objectives,
            "validation_metrics_raw": validation_raw,
            "validation_metrics_calibrated": validation_calibrated,
        },
        "derived_artifacts": {
            "train_manifest": fingerprint(manifest_path),
            "train_feature_cache": fingerprint(cache_path),
        },
        "deployment_decision": "pending_comparison_with_clean_v5_control",
        "final_holdout_status": "not_run",
    }
    atomic_json_dump(selection_record, selection_path)
    selection_fingerprint = fingerprint(selection_path)
    atomic_json_dump(
        {
            "schema": "pepper-clean-v5-pseudo-selection-receipt-v1",
            "created_at_utc": utc_now(),
            "selection_id": selection_record["selection_id"],
            "selection": selection_fingerprint,
            "model": model_record,
            "manifest": fingerprint(manifest_path),
            "feature_cache": fingerprint(cache_path),
            "final_holdout_paths_opened": False,
        },
        receipt_path,
    )
    print(
        json.dumps(
            {
                "selected": {
                    "policy": selected["policy"],
                    "view_mode": selected["view_mode"],
                    "C": selected["C"],
                    "eligible_unique_pairs": len(evidence),
                },
                "training_oof_metrics": selected["training_oof_metrics"],
                "validation_raw": validation_raw,
                "validation_calibrated": validation_calibrated,
                "model": model_record,
                "selection": selection_fingerprint,
                "strict_protocol": "NO FINAL HOLDOUT PATH, CACHE OR LABEL WAS OPENED",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
