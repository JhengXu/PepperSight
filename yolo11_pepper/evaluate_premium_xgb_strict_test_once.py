#!/usr/bin/env python3
"""One-shot strict-test evaluation of a frozen premium XGBoost selection.

This executable is intentionally separate from model selection.  Before it opens
any strict-test input it verifies both frozen selections, both model artifacts,
their receipts, every train/validation fingerprint, and the precommitted XGBoost
architecture winner.  It then creates an irreversible per-selection seal.  A
failed run therefore consumes the one allowed test opening just like a successful
run, and a second invocation is refused.

The test split cannot select an architecture, hyperparameter, threshold,
temperature, feature subset, ensemble weight, or deployment winner.  It only
reports the already selected XGBoost head and the already selected clean-v5 SVM
on the same frozen 3,584-dimensional YOLO11 classification features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


SCRIPT_VERSION = "pepper-premium-xgb-one-shot-strict-test-v1"
PREMIUM_SCHEMA = "pepper-premium-xgb-selection-v1"
PREMIUM_RECEIPT_SCHEMA = "pepper-premium-xgb-receipt-v1"
DECISION_SCHEMA = "pepper-premium-xgb-test-candidate-decision-v1"
DECISION_RECEIPT_SCHEMA = "pepper-premium-xgb-test-candidate-receipt-v1"
CLEAN_ONLY_SCHEMA = "pepper-premium-xgb-clean-only-fixed-hierarchical-v1"
BASELINE_SCHEMA = "pepper-clean-v5-validation-selection-v1"
BASELINE_RECEIPT_SCHEMA = "pepper-clean-v5-selection-receipt-v1"
REPORT_SCHEMA = "pepper-premium-xgb-strict-test-report-v1"
RECEIPT_SCHEMA = "pepper-premium-xgb-strict-test-receipt-v1"

CLASS_NAMES = ("子弹头_一级", "子弹头_二级", "条子_一级", "条子_二级")
SPECIES_NAMES = ("子弹头", "条子")
EXPECTED_TEST_ROWS = 180
EXPECTED_FEATURE_DIM = 3584
EXPECTED_IMAGE_SIZE = 256
EXPECTED_TEST_VIEWS = 1
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 3089
CONFIRMATION_TEXT = "OPEN_FROZEN_STRICT_TEST_ONCE"
PREDECLARED_ARCHITECTURES = ("direct_four_class", "hierarchical")
OVERLAP_FIELDS = (
    "path",
    "content_sha256",
    "group_id",
    "source_id",
    "pair_id",
)


@dataclass(frozen=True)
class FrozenInputs:
    premium_selection_path: Path
    premium_selection: dict[str, Any]
    premium_model_path: Path
    premium_model: dict[str, Any]
    premium_receipt_path: Path
    decision_path: Path
    decision: dict[str, Any]
    decision_receipt_path: Path
    clean_only_model_path: Path
    clean_only_model: dict[str, Any]
    baseline_selection_path: Path
    baseline_selection: dict[str, Any]
    baseline_model_path: Path
    baseline_model: dict[str, Any]
    baseline_receipt_path: Path
    train_manifest_records: tuple[dict[str, Any], ...]
    validation_manifest_records: tuple[dict[str, Any], ...]
    reference_feature_metadata: dict[str, Any]
    frozen_fingerprints: dict[str, Any]


@dataclass(frozen=True)
class ProbabilityState:
    species: np.ndarray
    grade: np.ndarray
    joint: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen premium XGBoost selection and one frozen clean-v5 "
            "SVM baseline on a strict test split exactly once."
        )
    )
    parser.add_argument("--premium-selection", type=Path, required=True)
    parser.add_argument("--premium-model", type=Path, required=True)
    parser.add_argument("--premium-receipt", type=Path, required=True)
    parser.add_argument("--test-candidate-decision", type=Path, required=True)
    parser.add_argument("--test-candidate-decision-receipt", type=Path, required=True)
    parser.add_argument("--clean-only-hierarchical-model", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--strict-test-manifest", type=Path, required=True)
    parser.add_argument("--imagenet-cls-test-feature", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--historical-report",
        type=Path,
        help=(
            "Optional already-existing historical JSON. It is copied only as a "
            "labelled observation and can never alter the frozen comparison."
        ),
    )
    parser.add_argument(
        "--confirm-one-shot",
        required=True,
        help=f"Must equal {CONFIRMATION_TEXT!r}.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify frozen non-test artifacts only. Does not stat, hash, or open "
            "the declared strict-test manifest/cache and does not create a seal."
        ),
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
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise ValueError(f"{label}.{key} must be exactly false")


def verify_record(record: Mapping[str, Any], label: str) -> Path:
    required = {"path", "sha256", "bytes"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{label} fingerprint lacks {sorted(missing)}")
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != int(record["bytes"]):
        raise ValueError(
            f"{label} byte size changed: {actual_size} != {record['bytes']}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != str(record["sha256"]):
        raise ValueError(f"{label} SHA-256 changed: {path}")
    return path


def same_resolved_path(first: Path, second: Path, label: str) -> None:
    if first.resolve() != second.resolve():
        raise ValueError(f"{label} path differs: {first.resolve()} != {second.resolve()}")


def _metric_pair(candidate: Mapping[str, Any]) -> tuple[float, float]:
    metrics = candidate.get("train_group_oof_metrics") or {}
    return float(metrics["joint_macro_f1"]), float(metrics["joint_accuracy"])


def verify_precommitted_architecture(selection: Mapping[str, Any]) -> str:
    candidate_space = selection.get("candidate_space") or {}
    architectures = tuple(candidate_space.get("architectures") or ())
    if architectures != PREDECLARED_ARCHITECTURES:
        raise ValueError(
            "Premium selection architecture order changed or was not predeclared"
        )
    if candidate_space.get("fixed_in_source_code") is not True:
        raise ValueError("Premium candidate space is not declared source-code-fixed")
    candidates = selection.get("candidates")
    winners = selection.get("architecture_winners_from_train_oof")
    if not isinstance(candidates, list) or not isinstance(winners, dict):
        raise ValueError("Premium selection lacks candidates/OOF architecture winners")

    recomputed: dict[str, Mapping[str, Any]] = {}
    for architecture in PREDECLARED_ARCHITECTURES:
        pool = [
            item
            for item in candidates
            if isinstance(item, dict) and item.get("architecture") == architecture
        ]
        if not pool:
            raise ValueError(f"No frozen OOF candidates for {architecture}")
        winner = max(
            enumerate(pool),
            key=lambda indexed: (*_metric_pair(indexed[1]), -indexed[0]),
        )[1]
        frozen = winners.get(architecture)
        if not isinstance(frozen, dict) or sha256_json(frozen) != sha256_json(winner):
            raise ValueError(f"Frozen OOF winner mismatch for {architecture}")
        recomputed[architecture] = winner

    expected = max(
        enumerate(PREDECLARED_ARCHITECTURES),
        key=lambda indexed: (*_metric_pair(recomputed[indexed[1]]), -indexed[0]),
    )[1]
    selected = str(selection.get("selected_architecture_for_future_strict_test") or "")
    if selected != expected:
        raise ValueError(
            f"Frozen architecture {selected!r} is not the train-OOF winner {expected!r}"
        )
    return selected


def verify_test_candidate_decision(
    *,
    decision_path: Path,
    decision_receipt_path: Path,
    selection_path: Path,
    premium_model_path: Path,
    clean_only_model_path: Path,
    original_oof_winner: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the transparent post-validation/pre-test business precommitment."""
    decision = read_json(decision_path, "test candidate decision")
    receipt = read_json(decision_receipt_path, "test candidate decision receipt")
    if decision.get("schema") != DECISION_SCHEMA:
        raise ValueError("Unsupported test-candidate decision schema")
    if receipt.get("schema") != DECISION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported test-candidate decision receipt schema")
    if decision.get("decision_stage") != "post-validation and before any strict-test access":
        raise ValueError("Test candidate was not frozen at the declared pre-test stage")
    if decision.get("strict_test_opened_at_decision_time") is not False:
        raise ValueError("Decision does not attest zero strict-test access")
    if receipt.get("strict_test_opened") is not False:
        raise ValueError("Decision receipt does not attest zero strict-test access")

    receipt_decision = verify_record(receipt.get("decision") or {}, "decision receipt")
    receipt_selection = verify_record(
        receipt.get("selection") or {}, "decision receipt selection"
    )
    receipt_main = verify_record(
        receipt.get("main_model") or {}, "decision receipt main model"
    )
    receipt_clean = verify_record(
        receipt.get("clean_only_model") or {}, "decision receipt clean-only model"
    )
    verify_record(receipt.get("ablation") or {}, "decision receipt ablation")
    same_resolved_path(receipt_decision, decision_path, "test candidate decision")
    same_resolved_path(receipt_selection, selection_path, "decision selection")
    same_resolved_path(receipt_main, premium_model_path, "decision main model")
    same_resolved_path(receipt_clean, clean_only_model_path, "decision clean-only model")

    primary = decision.get("primary_test_candidate") or {}
    control = decision.get("fixed_clean_only_ablation_candidate") or {}
    if primary.get("architecture") != "hierarchical":
        raise ValueError("Decision primary must be the requested hierarchical architecture")
    if primary.get("model_payload_branch") != "models.hierarchical":
        raise ValueError("Decision does not lock the primary to models.hierarchical")
    if int(primary.get("feature_indices", -1)) != 512:
        raise ValueError("Decision primary does not freeze 512 feature indices")
    if control.get("architecture") != "hierarchical":
        raise ValueError("Decision clean-only control is not hierarchical")
    if control.get("role") != (
        "same-specification clean-only control scored in the same one-pass strict-test evaluation"
    ):
        raise ValueError("Decision clean-only control role changed")
    primary_model = verify_record(primary.get("model") or {}, "decision primary model")
    control_model = verify_record(control.get("model") or {}, "decision control model")
    same_resolved_path(primary_model, premium_model_path, "decision primary model")
    same_resolved_path(control_model, clean_only_model_path, "decision control model")
    if sha256_json(primary.get("config")) != sha256_json(control.get("config")):
        raise ValueError("Premium and clean-only decision configs are not identical")

    transparency = decision.get("selection_transparency") or {}
    if transparency.get("decision_uses_validation_evidence") is not True:
        raise ValueError("Decision does not transparently declare validation evidence")
    if transparency.get("decision_was_not_the_original_train_oof_architecture_winner") is not True:
        raise ValueError("Decision hides its departure from the original OOF winner")
    if transparency.get("original_train_oof_winner") != original_oof_winner:
        raise ValueError("Decision records the wrong original train-OOF winner")
    if transparency.get("business_architecture_constraint_applied_before_test") is not True:
        raise ValueError("Hierarchical business constraint was not frozen before test")

    commitment = decision.get("strict_test_precommitment") or {}
    if commitment.get("single_test_data_open") is not True:
        raise ValueError("Decision does not require a single test-data opening")
    if commitment.get("models_scored_in_that_pass") != [
        "primary premium hierarchical",
        "fixed same-config clean-only hierarchical control",
    ]:
        raise ValueError("Decision model list changed")
    for key in (
        "direct_four_class_will_not_be_scored",
        "no_post_test_candidate_switching",
        "no_test_based_tuning_or_recalibration",
        "temperatures_are_frozen_from_validation",
    ):
        if commitment.get(key) is not True:
            raise ValueError(f"Decision precommitment {key!r} is not true")

    evidence = decision.get("evidence") or {}
    for key, expected_path in (
        ("selection", selection_path),
        ("main_model", premium_model_path),
        ("clean_only_model", clean_only_model_path),
    ):
        observed = verify_record(evidence.get(key) or {}, f"decision evidence {key}")
        same_resolved_path(observed, expected_path, f"decision evidence {key}")
    verify_record(evidence.get("ablation") or {}, "decision evidence ablation")
    return decision, receipt


def _manifest_record_from_feature_record(
    feature_record: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    record = feature_record.get("manifest")
    if not isinstance(record, dict):
        raise ValueError(f"{label} lacks an embedded manifest fingerprint")
    return record


def preflight_frozen_inputs(args: argparse.Namespace) -> FrozenInputs:
    """Complete all non-test checks before the one-shot seal is created."""
    if args.confirm_one_shot != CONFIRMATION_TEXT:
        raise ValueError(
            f"--confirm-one-shot must equal exactly {CONFIRMATION_TEXT!r}"
        )

    premium_selection_path = args.premium_selection.resolve()
    premium_model_path = args.premium_model.resolve()
    premium_receipt_path = args.premium_receipt.resolve()
    premium_selection = read_json(premium_selection_path, "premium selection")
    premium_receipt = read_json(premium_receipt_path, "premium receipt")
    if premium_selection.get("schema") != PREMIUM_SCHEMA:
        raise ValueError("Unsupported premium selection schema")
    if premium_receipt.get("schema") != PREMIUM_RECEIPT_SCHEMA:
        raise ValueError("Unsupported premium receipt schema")
    if premium_selection.get("final_test_status") != "not_run":
        raise ValueError("Premium selection does not declare final_test_status=not_run")
    premium_protocol = premium_selection.get("protocol") or {}
    require_false(
        premium_protocol,
        (
            "strict_test_manifest_opened",
            "strict_test_feature_opened",
            "strict_test_labels_read",
            "strict_test_metrics_computed",
            "strict_test_arguments_supported",
        ),
        "premium_selection.protocol",
    )
    if premium_protocol.get("validation_opened_after_selection_frozen") is not True:
        raise ValueError("Premium selection did not freeze architecture before validation")
    if premium_receipt.get("strict_test_opened") is not False:
        raise ValueError("Premium receipt does not attest strict_test_opened=false")
    receipt_selection_path = verify_record(
        premium_receipt.get("selection") or {}, "premium receipt selection"
    )
    receipt_model_path = verify_record(
        premium_receipt.get("model") or {}, "premium receipt model"
    )
    same_resolved_path(receipt_selection_path, premium_selection_path, "premium selection")
    same_resolved_path(receipt_model_path, premium_model_path, "premium model")
    selected_model_path = verify_record(
        premium_selection.get("model") or {}, "premium selected model"
    )
    same_resolved_path(selected_model_path, premium_model_path, "premium model")

    original_oof_architecture = verify_precommitted_architecture(premium_selection)
    premium_model = joblib.load(premium_model_path)
    if not isinstance(premium_model, dict) or premium_model.get("schema") != PREMIUM_SCHEMA:
        raise TypeError("Frozen premium model payload has the wrong schema")
    if premium_model.get("selected_architecture_for_future_strict_test") != original_oof_architecture:
        raise ValueError("Premium model and selection disagree on architecture winner")
    if sha256_json(premium_model.get("winners")) != sha256_json(
        premium_selection.get("architecture_winners_from_train_oof")
    ):
        raise ValueError("Premium model and selection contain different OOF winners")
    premium_model_protocol = premium_model.get("protocol") or {}
    if premium_model_protocol.get("strict_test_opened") is not False:
        raise ValueError("Premium model does not attest strict_test_opened=false")
    if (
        premium_model_protocol.get(
            "validation_did_not_select_architecture_or_hyperparameters"
        )
        is not True
    ):
        raise ValueError("Premium model permits validation architecture switching")
    preprocessing = premium_model.get("feature_preprocessing") or {}
    if int(preprocessing.get("feature_dim_before_variance_selection", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError("Premium model does not use 3,584-dimensional features")
    if int(preprocessing.get("image_size", -1)) != EXPECTED_IMAGE_SIZE:
        raise ValueError("Premium model feature image size is not 256")
    if preprocessing.get("view") != "canonical view index 0":
        raise ValueError("Premium model does not precommit canonical view zero")
    if preprocessing.get("normalization") != "row-wise L2":
        raise ValueError("Premium model does not precommit row-wise L2 normalization")
    fitted = premium_model.get("models") or {}
    if original_oof_architecture not in fitted:
        raise ValueError(
            f"Premium model lacks original OOF architecture {original_oof_architecture}"
        )
    _verify_premium_model_branch(
        fitted[original_oof_architecture], original_oof_architecture
    )

    # The original train-OOF winner was direct four-class.  A separate,
    # transparent post-validation/pre-test decision applies the user's fixed
    # hierarchical architecture requirement.  It must be verified rather than
    # silently rewriting the earlier selection record.
    decision_path = args.test_candidate_decision.resolve()
    decision_receipt_path = args.test_candidate_decision_receipt.resolve()
    clean_only_model_path = args.clean_only_hierarchical_model.resolve()
    decision, _decision_receipt = verify_test_candidate_decision(
        decision_path=decision_path,
        decision_receipt_path=decision_receipt_path,
        selection_path=premium_selection_path,
        premium_model_path=premium_model_path,
        clean_only_model_path=clean_only_model_path,
        original_oof_winner=original_oof_architecture,
    )
    primary_decision = decision["primary_test_candidate"]
    premium_hierarchical = fitted.get("hierarchical")
    _verify_premium_model_branch(premium_hierarchical, "hierarchical")
    if sha256_json(primary_decision.get("config")) != sha256_json(
        (premium_model.get("winners") or {}).get("hierarchical", {}).get("config")
    ):
        raise ValueError("Decision primary config differs from the frozen hierarchical head")
    if int(primary_decision.get("feature_indices", -1)) != len(
        np.asarray(premium_hierarchical["feature_indices"])
    ):
        raise ValueError("Decision primary feature count differs from the frozen head")
    if sha256_json(primary_decision.get("n_estimators")) != sha256_json(
        premium_hierarchical.get("n_estimators")
    ):
        raise ValueError("Decision primary tree counts differ from the frozen head")

    clean_only_model = joblib.load(clean_only_model_path)
    if not isinstance(clean_only_model, dict) or clean_only_model.get("schema") != CLEAN_ONLY_SCHEMA:
        raise TypeError("Frozen clean-only hierarchical control has the wrong schema")
    if clean_only_model.get("architecture") != "hierarchical":
        raise ValueError("Clean-only control is not frozen to hierarchical")
    _verify_premium_model_branch(clean_only_model, "hierarchical")
    control_decision = decision["fixed_clean_only_ablation_candidate"]
    if sha256_json(clean_only_model.get("config")) != sha256_json(
        primary_decision.get("config")
    ):
        raise ValueError("Clean-only and premium configurations differ")
    if sha256_json(clean_only_model.get("n_estimators")) != sha256_json(
        primary_decision.get("n_estimators")
    ):
        raise ValueError("Clean-only and premium tree counts differ")
    if int(control_decision["config"]["top_variance_features"]) != len(
        np.asarray(clean_only_model["feature_indices"])
    ):
        raise ValueError("Clean-only decision feature count differs from its model")
    clean_preprocessing = clean_only_model.get("feature_preprocessing") or {}
    if sha256_json(clean_preprocessing) != sha256_json(preprocessing):
        raise ValueError("Premium and clean-only feature preprocessing differ")
    clean_provenance = clean_only_model.get("provenance") or {}
    if clean_provenance.get("strict_test_opened") is not False:
        raise ValueError("Clean-only control does not attest strict_test_opened=false")
    for key in (
        "selection",
        "ablation",
        "clean_manifest",
        "clean_feature",
        "validation_manifest",
        "validation_feature",
    ):
        verify_record(clean_provenance.get(key) or {}, f"clean-only provenance {key}")

    baseline_selection_path = args.baseline_selection.resolve()
    baseline_model_path = args.baseline_model.resolve()
    baseline_receipt_path = args.baseline_receipt.resolve()
    baseline_selection = read_json(baseline_selection_path, "baseline selection")
    baseline_receipt = read_json(baseline_receipt_path, "baseline receipt")
    if baseline_selection.get("schema") != BASELINE_SCHEMA:
        raise ValueError("Unsupported clean-v5 baseline selection schema")
    if baseline_receipt.get("schema") != BASELINE_RECEIPT_SCHEMA:
        raise ValueError("Unsupported clean-v5 baseline receipt schema")
    if baseline_selection.get("final_test_status") != "not_run":
        raise ValueError("Baseline selection does not declare final_test_status=not_run")
    baseline_protocol = baseline_selection.get("protocol") or {}
    require_false(
        baseline_protocol,
        (
            "strict_test_manifest_opened",
            "test_feature_cache_opened",
            "test_labels_read",
            "test_metrics_computed",
            "test_arguments_supported",
        ),
        "baseline_selection.protocol",
    )
    if baseline_selection.get("strict_test_manifest_opened") is not False:
        raise ValueError("Baseline top-level strict test attestation is not false")
    if baseline_receipt.get("strict_test_manifest_opened") is not False:
        raise ValueError("Baseline receipt strict test attestation is not false")
    receipt_baseline_selection = verify_record(
        baseline_receipt.get("selection") or {}, "baseline receipt selection"
    )
    receipt_baseline_model = verify_record(
        baseline_receipt.get("model") or {}, "baseline receipt model"
    )
    same_resolved_path(
        receipt_baseline_selection, baseline_selection_path, "baseline selection"
    )
    same_resolved_path(receipt_baseline_model, baseline_model_path, "baseline model")
    selected_baseline_model = verify_record(
        (baseline_selection.get("selected") or {}).get("model") or {},
        "baseline selected model",
    )
    same_resolved_path(selected_baseline_model, baseline_model_path, "baseline model")
    baseline_model = joblib.load(baseline_model_path)
    if not isinstance(baseline_model, dict) or baseline_model.get("schema") != BASELINE_SCHEMA:
        raise TypeError("Frozen baseline model payload has the wrong schema")
    if baseline_model.get("view_mode") != "canonical":
        raise ValueError("Baseline is not frozen to canonical-view features")
    baseline_model_protocol = baseline_model.get("protocol") or {}
    require_false(
        baseline_model_protocol,
        (
            "strict_test_manifest_opened",
            "test_feature_cache_opened",
            "test_labels_read",
            "test_metrics_computed",
            "test_arguments_supported",
        ),
        "baseline_model.protocol",
    )
    baseline_feature_schema = baseline_model.get("feature_schema") or {}
    if int(baseline_feature_schema.get("combined_dim", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError("Baseline model feature dimension is not 3,584")
    families = baseline_feature_schema.get("families") or []
    if len(families) != 1 or families[0].get("name") != "imagenet_cls":
        raise ValueError("Baseline model must use only imagenet_cls features")
    if int(families[0].get("dim", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError("Baseline imagenet_cls family dimension changed")
    if int(families[0].get("image_size", -1)) != EXPECTED_IMAGE_SIZE:
        raise ValueError("Baseline imagenet_cls image size changed")
    checkpoint_path = verify_record(
        {
            "path": families[0].get("checkpoint"),
            "sha256": families[0].get("checkpoint_sha256"),
            "bytes": Path(str(families[0].get("checkpoint"))).resolve().stat().st_size,
        },
        "baseline feature checkpoint",
    )
    if len(baseline_model.get("grade_models") or []) != 2:
        raise ValueError("Baseline model must contain two conditional grade heads")

    premium_inputs = premium_selection.get("input_fingerprints") or {}
    train_blocks = premium_inputs.get("train_blocks")
    validation_block = premium_inputs.get("validation_block")
    if not isinstance(train_blocks, list) or not train_blocks:
        raise ValueError("Premium selection has no frozen train blocks")
    if not isinstance(validation_block, dict):
        raise ValueError("Premium selection has no frozen validation block")
    train_manifest_records: list[dict[str, Any]] = []
    for index, block in enumerate(train_blocks):
        if not isinstance(block, dict):
            raise TypeError(f"Premium train block {index} is not an object")
        verify_record(block, f"premium train feature {index}")
        manifest_record = dict(
            _manifest_record_from_feature_record(block, f"premium train block {index}")
        )
        verify_record(manifest_record, f"premium train manifest {index}")
        train_manifest_records.append(manifest_record)
    verify_record(validation_block, "premium validation feature")
    validation_manifest_record = dict(
        _manifest_record_from_feature_record(validation_block, "premium validation block")
    )
    verify_record(validation_manifest_record, "premium validation manifest")

    baseline_inputs = baseline_selection.get("input_fingerprints") or {}
    baseline_train_record = baseline_inputs.get("train_manifest") or {}
    baseline_val_record = baseline_inputs.get("validation_manifest") or {}
    verify_record(baseline_train_record, "baseline train manifest")
    verify_record(baseline_val_record, "baseline validation manifest")
    # The baseline can be a subset of premium training, but its physical
    # validation split must be exactly the same frozen split.
    if str(baseline_val_record.get("sha256")) != str(
        validation_manifest_record.get("sha256")
    ):
        raise ValueError("Premium and baseline validation manifests differ")

    reference_metadata = dict(validation_block.get("metadata") or {})
    for key, expected in (
        ("backbone_name", "imagenet_cls"),
        ("kind", "cls"),
        ("image_size", EXPECTED_IMAGE_SIZE),
        ("scale_normalized", True),
    ):
        if reference_metadata.get(key) != expected:
            raise ValueError(f"Frozen validation feature metadata {key!r} changed")
    same_resolved_path(
        Path(str(reference_metadata.get("checkpoint"))).resolve(),
        checkpoint_path,
        "premium/baseline feature checkpoint",
    )

    frozen_fingerprints = {
        "script": fingerprint(Path(__file__)),
        "premium_selection": fingerprint(premium_selection_path),
        "premium_model": fingerprint(premium_model_path),
        "premium_receipt": fingerprint(premium_receipt_path),
        "test_candidate_decision": fingerprint(decision_path),
        "test_candidate_decision_receipt": fingerprint(decision_receipt_path),
        "clean_only_hierarchical_model": fingerprint(clean_only_model_path),
        "baseline_selection": fingerprint(baseline_selection_path),
        "baseline_model": fingerprint(baseline_model_path),
        "baseline_receipt": fingerprint(baseline_receipt_path),
        "baseline_feature_checkpoint": fingerprint(checkpoint_path),
    }
    return FrozenInputs(
        premium_selection_path=premium_selection_path,
        premium_selection=premium_selection,
        premium_model_path=premium_model_path,
        premium_model=premium_model,
        premium_receipt_path=premium_receipt_path,
        decision_path=decision_path,
        decision=decision,
        decision_receipt_path=decision_receipt_path,
        clean_only_model_path=clean_only_model_path,
        clean_only_model=clean_only_model,
        baseline_selection_path=baseline_selection_path,
        baseline_selection=baseline_selection,
        baseline_model_path=baseline_model_path,
        baseline_model=baseline_model,
        baseline_receipt_path=baseline_receipt_path,
        train_manifest_records=tuple(train_manifest_records),
        validation_manifest_records=(validation_manifest_record,),
        reference_feature_metadata=reference_metadata,
        frozen_fingerprints=frozen_fingerprints,
    )


def _verify_premium_model_branch(branch: Any, architecture: str) -> None:
    if not isinstance(branch, dict):
        raise TypeError(f"Premium {architecture} model payload must be an object")
    indices = np.asarray(branch.get("feature_indices"))
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"Premium {architecture} has no frozen feature indices")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Premium {architecture} feature indices are not integers")
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= EXPECTED_FEATURE_DIM:
        raise ValueError(f"Premium {architecture} feature indices are invalid")
    if architecture == "direct_four_class":
        if not hasattr(branch.get("model"), "predict_proba"):
            raise TypeError("Premium direct head lacks predict_proba")
        if not math.isfinite(float(branch.get("temperature", 0))) or float(
            branch.get("temperature", 0)
        ) <= 0:
            raise ValueError("Premium direct temperature is not frozen/positive")
    elif architecture == "hierarchical":
        if not hasattr(branch.get("species_model"), "predict_proba"):
            raise TypeError("Premium species head lacks predict_proba")
        grade_models = branch.get("grade_models") or []
        if len(grade_models) != 2 or any(
            not hasattr(model, "predict_proba") for model in grade_models
        ):
            raise TypeError("Premium hierarchical head lacks two grade predict_proba heads")
        temperatures = branch.get("temperatures") or []
        if len(temperatures) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in temperatures
        ):
            raise ValueError("Premium hierarchical temperatures are invalid")
    else:
        raise ValueError(f"Unsupported frozen architecture: {architecture}")


def create_one_shot_seal(
    frozen: FrozenInputs, output_dir: Path, args: argparse.Namespace
) -> tuple[Path, Path]:
    """Consume the single allowed test opening before any test byte is read."""
    output_dir = output_dir.resolve()
    seal_path = frozen.premium_selection_path.with_name("strict_test_once_seal.json")
    if output_dir.exists():
        raise FileExistsError(f"Strict-test output directory already exists: {output_dir}")
    if seal_path.exists():
        raise FileExistsError(
            f"This frozen premium selection has already consumed its strict test: {seal_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    seal = {
        "schema": "pepper-premium-xgb-strict-test-seal-v1",
        "created_at_utc": utc_now(),
        "status": (
            "one-shot strict-test opening consumed before reading test inputs; "
            "failure is terminal and rerun is forbidden"
        ),
        "selection": frozen.frozen_fingerprints["premium_selection"],
        "model": frozen.frozen_fingerprints["premium_model"],
        "test_candidate_decision": frozen.frozen_fingerprints[
            "test_candidate_decision"
        ],
        "clean_only_hierarchical_model": frozen.frozen_fingerprints[
            "clean_only_hierarchical_model"
        ],
        "clean_v5_svm_model": frozen.frozen_fingerprints["baseline_model"],
        "output_dir": str(output_dir),
        "strict_test_manifest_path_declared_but_not_opened_at_seal_time": str(
            args.strict_test_manifest.resolve()
        ),
        "strict_test_feature_path_declared_but_not_opened_at_seal_time": str(
            args.imagenet_cls_test_feature.resolve()
        ),
    }
    encoded = (json.dumps(seal, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(seal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except Exception:
        # Keep the newly-created output directory as an additional failure marker.
        raise
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    write_new_json(output_dir / "RUN_STARTED.json", seal)
    return output_dir, seal_path


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def read_physical_manifest(
    path: Path, expected_split: str, *, expected_rows: int | None = None
) -> list[dict[str, Any]]:
    path = path.resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "path",
            "split",
            "class_id",
            "group_id",
            "source_id",
            "pair_id",
            "content_sha256",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks strict identity columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for line, raw in enumerate(reader, 2):
            split = str(raw.get("split") or "").strip().lower()
            if split != expected_split:
                raise ValueError(
                    f"{path}:{line} split={split!r}; expected only {expected_split!r}"
                )
            image_path = Path(str(raw.get("path") or "")).resolve()
            try:
                class_id = int(raw.get("class_id", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line} invalid class_id") from error
            if class_id not in range(4):
                raise ValueError(f"{path}:{line} class_id outside [0,3]")
            identity = {
                key: str(raw.get(key) or "").strip()
                for key in ("group_id", "source_id", "pair_id")
            }
            if not all(identity.values()):
                raise ValueError(f"{path}:{line} has an empty group/source/pair id")
            content_sha256 = str(raw.get("content_sha256") or "").strip().lower()
            if not _valid_sha256(content_sha256):
                raise ValueError(f"{path}:{line} has invalid content_sha256")
            if expected_split == "test":
                if not image_path.is_file():
                    raise FileNotFoundError(f"Strict-test image is missing: {image_path}")
                if "selection_role" in (reader.fieldnames or ()) and str(
                    raw.get("selection_role") or ""
                ).strip().lower() != "strict_test":
                    raise ValueError(f"{path}:{line} is not marked selection_role=strict_test")
                if "eligible_for_model_training" in (reader.fieldnames or ()) and str(
                    raw.get("eligible_for_model_training") or ""
                ).strip().lower() not in {"false", "0", "no"}:
                    raise ValueError(f"{path}:{line} is eligible for model training")
            row = dict(raw)
            row.update(
                {
                    "path": str(image_path),
                    "split": split,
                    "class_id": class_id,
                    "group_id": identity["group_id"],
                    "source_id": identity["source_id"],
                    "pair_id": identity["pair_id"],
                    "content_sha256": content_sha256,
                }
            )
            rows.append(row)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows; expected {expected_rows}")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{path} contains duplicate resolved image paths")
    classes = Counter(int(row["class_id"]) for row in rows)
    if set(classes) != {0, 1, 2, 3}:
        raise ValueError(f"{path} does not contain all four classes: {classes}")
    return rows


def _identity_overlap(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    field: str,
) -> list[str]:
    left = {str(row.get(field) or "").strip() for row in first} - {""}
    right = {str(row.get(field) or "").strip() for row in second} - {""}
    return sorted(left & right)


def leakage_audit(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons = {
        "train_vs_validation": (train, validation),
        "train_vs_test": (train, test),
        "validation_vs_test": (validation, test),
    }
    overlaps: dict[str, dict[str, list[str]]] = {}
    for name, (left, right) in comparisons.items():
        overlaps[name] = {
            field: _identity_overlap(left, right, field) for field in OVERLAP_FIELDS
        }
    passed = not any(
        values
        for comparison in overlaps.values()
        for values in comparison.values()
    )
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "checked_fields": list(OVERLAP_FIELDS),
        "overlap": overlaps,
        "passed": passed,
    }


def load_test_feature(
    path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    reference_metadata: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = path.resolve()
    cache_record = fingerprint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), torch.Tensor):
        raise TypeError(f"Strict-test feature cache has no feature tensor: {path}")
    tensor = payload["features"].float().cpu()
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    expected_shape = (EXPECTED_TEST_ROWS, EXPECTED_TEST_VIEWS, EXPECTED_FEATURE_DIM)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Strict-test feature shape {tuple(tensor.shape)} != {expected_shape}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Strict-test features contain NaN/Inf")
    if (torch.linalg.vector_norm(tensor[:, 0], dim=1) <= 0).any():
        raise ValueError("Strict-test features contain an all-zero canonical vector")

    expected_paths = [str(row["path"]) for row in rows]
    cached_paths = [str(Path(str(value)).resolve()) for value in payload.get("paths", [])]
    if cached_paths != expected_paths:
        raise ValueError("Strict-test cache path order differs from the 180-row manifest")
    cached_classes = payload.get("class_ids")
    expected_classes = torch.tensor([int(row["class_id"]) for row in rows])
    if not isinstance(cached_classes, torch.Tensor) or not torch.equal(
        cached_classes.cpu().long(), expected_classes.long()
    ):
        raise ValueError("Strict-test cache class order differs from the manifest")
    expected_groups = [str(row["group_id"]) for row in rows]
    if [str(value) for value in payload.get("groups", [])] != expected_groups:
        raise ValueError("Strict-test cache group order differs from the manifest")
    if "source_ids" in payload and [str(value) for value in payload["source_ids"]] != [
        str(row["source_id"]) for row in rows
    ]:
        raise ValueError("Strict-test cache source order differs from the manifest")
    if "pair_ids" in payload and [str(value) for value in payload["pair_ids"]] != [
        str(row["pair_id"]) for row in rows
    ]:
        raise ValueError("Strict-test cache pair order differs from the manifest")

    metadata = dict(payload.get("metadata") or {})
    if str(metadata.get("split") or "").strip().lower() != "test":
        raise ValueError("Strict-test feature metadata split is not test")
    if metadata.get("test_requested_explicitly") is not True:
        raise ValueError("Strict-test feature was not extracted with explicit test authorization")
    metadata_manifest_value = str(metadata.get("manifest") or "").strip()
    if not metadata_manifest_value:
        raise ValueError("Strict-test feature metadata has no source manifest")
    metadata_manifest = Path(metadata_manifest_value).resolve()
    # Older frozen test caches may reference the immutable unified source
    # manifest rather than its physically separated strict-test projection.
    # Exact path/class/group order below is the authoritative row-level check;
    # preserve and hash either metadata source instead of silently rewriting it.
    metadata_manifest_exact = metadata_manifest == manifest_path.resolve()
    metadata_manifest_record = fingerprint(metadata_manifest)
    checks = {
        "backbone_name": "imagenet_cls",
        "kind": "cls",
        "image_size": EXPECTED_IMAGE_SIZE,
        "views": EXPECTED_TEST_VIEWS,
        "scale_normalized": True,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Strict-test metadata {key}={metadata.get(key)!r}; expected {expected!r}"
            )
    for key in ("backbone_name", "kind", "image_size", "scale_normalized"):
        if metadata.get(key) != reference_metadata.get(key):
            raise ValueError(f"Strict-test and frozen validation metadata differ on {key}")
    same_resolved_path(
        Path(str(metadata.get("checkpoint"))).resolve(),
        Path(str(reference_metadata.get("checkpoint"))).resolve(),
        "test/reference feature checkpoint",
    )
    matrix = F.normalize(tensor[:, 0], p=2, dim=1).numpy().astype(np.float32)
    cache_record.update(
        {
            "shape": list(tensor.shape),
            "metadata": metadata,
            "metadata_manifest": metadata_manifest_record,
            "metadata_manifest_is_exact_strict_test_projection": metadata_manifest_exact,
            "path_class_group_order_verified": True,
            "optional_source_order_verified": "source_ids" in payload,
            "optional_pair_order_verified": "pair_ids" in payload,
            "canonical_rowwise_l2_applied_once_for_both_models": True,
        }
    )
    return matrix, cache_record


def ordered_probability(model: Any, matrix: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    model_classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    if set(model_classes) != set(classes):
        raise ValueError(f"Model classes {model_classes} != expected {list(classes)}")
    result = np.empty((len(matrix), len(classes)), dtype=np.float64)
    index_by_class = {class_id: index for index, class_id in enumerate(classes)}
    for source_column, class_id in enumerate(model_classes):
        result[:, index_by_class[class_id]] = raw[:, source_column]
    result /= np.clip(result.sum(1, keepdims=True), 1e-15, None)
    return result


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def joint_from_branches(species: np.ndarray, grade: np.ndarray) -> np.ndarray:
    joint = (species[:, :, None] * grade).reshape(-1, 4)
    joint /= np.clip(joint.sum(1, keepdims=True), 1e-15, None)
    return joint


def branches_from_joint(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint = np.asarray(joint, dtype=np.float64)
    species = np.stack((joint[:, :2].sum(1), joint[:, 2:].sum(1)), axis=1)
    grade = np.empty((len(joint), 2, 2), dtype=np.float64)
    grade[:, 0] = joint[:, :2] / np.clip(species[:, 0, None], 1e-15, None)
    grade[:, 1] = joint[:, 2:] / np.clip(species[:, 1, None], 1e-15, None)
    return species, grade


def calibrate_hierarchical(
    species: np.ndarray, grade: np.ndarray, temperatures: Sequence[float]
) -> ProbabilityState:
    values = [float(value) for value in temperatures]
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Expected three frozen positive branch temperatures")
    calibrated_species = softmax_numpy(
        np.log(np.clip(species, 1e-15, 1.0)) / values[0]
    )
    calibrated_grade = np.empty_like(grade)
    for species_id in range(2):
        calibrated_grade[:, species_id] = softmax_numpy(
            np.log(np.clip(grade[:, species_id], 1e-15, 1.0))
            / values[species_id + 1]
        )
    return ProbabilityState(
        calibrated_species,
        calibrated_grade,
        joint_from_branches(calibrated_species, calibrated_grade),
    )


def predict_hierarchical_branch(
    branch: Mapping[str, Any], matrix: np.ndarray
) -> tuple[ProbabilityState, ProbabilityState]:
    """Score one already-frozen hierarchical branch without model switching."""
    indices = np.asarray(branch["feature_indices"], dtype=np.int64)
    selected = matrix[:, indices]
    species = ordered_probability(branch["species_model"], selected, (0, 1))
    grade = np.stack(
        [
            ordered_probability(model, selected, (0, 1))
            for model in branch["grade_models"]
        ],
        axis=1,
    )
    raw = ProbabilityState(species, grade, joint_from_branches(species, grade))
    calibrated = calibrate_hierarchical(species, grade, branch["temperatures"])
    return raw, calibrated


def predict_baseline(
    payload: Mapping[str, Any], matrix: np.ndarray
) -> tuple[ProbabilityState, ProbabilityState]:
    species = ordered_probability(payload["species_model"], matrix, (0, 1))
    grade = np.stack(
        [ordered_probability(model, matrix, (0, 1)) for model in payload["grade_models"]],
        axis=1,
    )
    raw = ProbabilityState(species, grade, joint_from_branches(species, grade))
    temperature_record = payload["temperatures"]
    temperatures = (
        float(temperature_record["species"]),
        float(temperature_record["grade_given_子弹头"]),
        float(temperature_record["grade_given_条子"]),
    )
    calibrated = calibrate_hierarchical(species, grade, temperatures)
    return raw, calibrated


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
    return float(result)


def metrics(
    rows: Sequence[Mapping[str, Any]], state: ProbabilityState
) -> dict[str, Any]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    species_labels = labels // 2
    grade_labels = labels % 2
    species_prediction = state.species.argmax(1)
    grade_prediction = state.grade[np.arange(len(rows)), species_labels].argmax(1)
    joint_prediction = state.joint.argmax(1)
    confusion = confusion_matrix(labels, joint_prediction, labels=[0, 1, 2, 3])
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - true_positive
        false_negative = int(confusion[class_id].sum()) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-15)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(confusion[class_id].sum()),
            }
        )
    return {
        "samples": len(rows),
        "groups": len({str(row["group_id"]) for row in rows}),
        "species_accuracy": float(np.mean(species_prediction == species_labels)),
        "conditional_grade_accuracy": float(np.mean(grade_prediction == grade_labels)),
        "joint_accuracy": float(np.mean(joint_prediction == labels)),
        "joint_macro_f1": float(
            f1_score(
                labels,
                joint_prediction,
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "joint_nll": float(
            -np.log(
                np.clip(state.joint[np.arange(len(rows)), labels], 1e-15, 1.0)
            ).mean()
        ),
        "joint_ece_15bin": ece_score(state.joint, labels),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def group_metrics(
    rows: Sequence[Mapping[str, Any]], state: ProbabilityState
) -> dict[str, Any]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[str(row["group_id"])].append(index)
    by_group: dict[str, Any] = {}
    for group_id, indices in sorted(members.items()):
        subset_rows = [rows[index] for index in indices]
        subset = ProbabilityState(
            state.species[indices], state.grade[indices], state.joint[indices]
        )
        by_group[group_id] = metrics(subset_rows, subset)
    aggregate_keys = (
        "species_accuracy",
        "conditional_grade_accuracy",
        "joint_accuracy",
        "joint_macro_f1",
        "joint_nll",
        "joint_ece_15bin",
    )
    return {
        "groups": len(by_group),
        "unweighted_group_mean": {
            key: float(np.mean([value[key] for value in by_group.values()]))
            for key in aggregate_keys
        },
        "by_group": by_group,
    }


def _state_predictions(
    rows: Sequence[Mapping[str, Any]], state: ProbabilityState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    species = state.species.argmax(1)
    grade = state.grade[np.arange(len(rows)), labels // 2].argmax(1)
    joint = state.joint.argmax(1)
    return species, grade, joint


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    premium: ProbabilityState,
    baseline: ProbabilityState,
    *,
    comparison_label: str,
) -> dict[str, Any]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    groups: dict[str, np.ndarray] = {}
    for group_id in sorted({str(row["group_id"]) for row in rows}):
        groups[group_id] = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["group_id"]) == group_id
            ],
            dtype=np.int64,
        )
    group_ids = list(groups)
    premium_predictions = _state_predictions(rows, premium)
    baseline_predictions = _state_predictions(rows, baseline)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {
        "species_accuracy": np.empty(BOOTSTRAP_REPEATS),
        "conditional_grade_accuracy": np.empty(BOOTSTRAP_REPEATS),
        "joint_accuracy": np.empty(BOOTSTRAP_REPEATS),
        "joint_macro_f1": np.empty(BOOTSTRAP_REPEATS),
    }
    true_species = labels // 2
    true_grade = labels % 2
    for iteration in range(BOOTSTRAP_REPEATS):
        sampled = rng.integers(0, len(group_ids), size=len(group_ids))
        indices = np.concatenate([groups[group_ids[index]] for index in sampled])
        values["species_accuracy"][iteration] = np.mean(
            premium_predictions[0][indices] == true_species[indices]
        ) - np.mean(baseline_predictions[0][indices] == true_species[indices])
        values["conditional_grade_accuracy"][iteration] = np.mean(
            premium_predictions[1][indices] == true_grade[indices]
        ) - np.mean(baseline_predictions[1][indices] == true_grade[indices])
        values["joint_accuracy"][iteration] = np.mean(
            premium_predictions[2][indices] == labels[indices]
        ) - np.mean(baseline_predictions[2][indices] == labels[indices])
        values["joint_macro_f1"][iteration] = f1_score(
            labels[indices],
            premium_predictions[2][indices],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        ) - f1_score(
            labels[indices],
            baseline_predictions[2][indices],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
    result: dict[str, Any] = {}
    for key, distribution in values.items():
        point_premium = metrics(rows, premium)[key]
        point_baseline = metrics(rows, baseline)[key]
        result[key] = {
            "point_delta_premium_minus_baseline": float(
                point_premium - point_baseline
            ),
            "group_bootstrap_95_percentile_interval": [
                float(np.quantile(distribution, 0.025)),
                float(np.quantile(distribution, 0.975)),
            ],
            "two_sided_tail_fraction": float(
                min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(distribution <= 0)),
                        float(np.mean(distribution >= 0)),
                    ),
                )
            ),
        }
    return {
        "resampling_unit": "physical group_id",
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "comparison": comparison_label,
        "metrics": result,
    }


def exact_mcnemar(
    rows: Sequence[Mapping[str, Any]],
    premium: ProbabilityState,
    baseline: ProbabilityState,
) -> dict[str, Any]:
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    premium_correct = premium.joint.argmax(1) == labels
    baseline_correct = baseline.joint.argmax(1) == labels
    both_correct = int(np.sum(premium_correct & baseline_correct))
    premium_only = int(np.sum(premium_correct & ~baseline_correct))
    baseline_only = int(np.sum(~premium_correct & baseline_correct))
    both_wrong = int(np.sum(~premium_correct & ~baseline_correct))
    discordant = premium_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(premium_only, baseline_only)
        probability = sum(math.comb(discordant, value) for value in range(lower + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2.0 * probability)
    return {
        "both_correct": both_correct,
        "premium_correct_baseline_wrong": premium_only,
        "premium_wrong_baseline_correct": baseline_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "exact_two_sided_binomial_p": float(p_value),
    }


def write_predictions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, ProbabilityState],
) -> None:
    identity_fields = (
        "row_index",
        "path",
        "group_id",
        "source_id",
        "pair_id",
        "content_sha256",
        "class_id",
        "class_name",
    )
    state_fields: list[str] = []
    for name in states:
        state_fields.extend([f"{name}_prediction", f"{name}_correct"])
        state_fields.extend(f"{name}_p_{class_name}" for class_name in CLASS_NAMES)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*identity_fields, *state_fields])
        writer.writeheader()
        for index, row in enumerate(rows):
            output: dict[str, Any] = {
                "row_index": index,
                "path": row["path"],
                "group_id": row["group_id"],
                "source_id": row["source_id"],
                "pair_id": row["pair_id"],
                "content_sha256": row["content_sha256"],
                "class_id": row["class_id"],
                "class_name": CLASS_NAMES[int(row["class_id"])],
            }
            for name, state in states.items():
                prediction = int(state.joint[index].argmax())
                output[f"{name}_prediction"] = prediction
                output[f"{name}_correct"] = prediction == int(row["class_id"])
                for class_id, class_name in enumerate(CLASS_NAMES):
                    output[f"{name}_p_{class_name}"] = f"{state.joint[index, class_id]:.12f}"
            writer.writerow(output)


def historical_observation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = read_json(path, "historical report")
    # Preserve the source exactly instead of interpreting it as a candidate.
    return {
        "role": "historical_observation_only_not_used_for_selection_or_switching",
        "source": fingerprint(path),
        "source_schema": payload.get("schema"),
        "payload": payload,
    }


def write_new_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite JSON artifact: {path}")
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
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite JSON artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> None:
    # No strict-test path is opened above or inside this preflight call.
    frozen = preflight_frozen_inputs(args)
    output_dir, seal_path = create_one_shot_seal(frozen, args.output_dir, args)

    # The selection is now irreversibly sealed.  This is the first permitted
    # point at which any strict-test byte, label, metadata, or feature is read.
    test_manifest_path = args.strict_test_manifest.resolve()
    test_feature_path = args.imagenet_cls_test_feature.resolve()
    test_manifest_record = fingerprint(test_manifest_path)
    test_rows = read_physical_manifest(
        test_manifest_path, "test", expected_rows=EXPECTED_TEST_ROWS
    )

    train_rows: list[dict[str, Any]] = []
    for record in frozen.train_manifest_records:
        train_rows.extend(
            read_physical_manifest(Path(str(record["path"])), "train")
        )
    validation_rows: list[dict[str, Any]] = []
    for record in frozen.validation_manifest_records:
        validation_rows.extend(
            read_physical_manifest(Path(str(record["path"])), "val")
        )
    leakage = leakage_audit(train_rows, validation_rows, test_rows)
    if not leakage["passed"]:
        raise ValueError(f"Strict train/validation/test leakage audit failed: {leakage}")

    test_matrix, test_cache_record = load_test_feature(
        test_feature_path,
        test_manifest_path,
        test_rows,
        frozen.reference_feature_metadata,
    )
    # The decision explicitly selects models.hierarchical.  The original direct
    # OOF winner is intentionally not scored in this pass.
    premium_raw, premium_calibrated = predict_hierarchical_branch(
        frozen.premium_model["models"]["hierarchical"], test_matrix
    )
    clean_only_raw, clean_only_calibrated = predict_hierarchical_branch(
        frozen.clean_only_model, test_matrix
    )
    baseline_raw, baseline_calibrated = predict_baseline(
        frozen.baseline_model, test_matrix
    )
    states = {
        "premium_xgb_raw": premium_raw,
        "premium_xgb_calibrated": premium_calibrated,
        "clean_only_xgb_raw": clean_only_raw,
        "clean_only_xgb_calibrated": clean_only_calibrated,
        "clean_v5_svm_raw": baseline_raw,
        "clean_v5_svm_calibrated": baseline_calibrated,
    }
    predictions_path = output_dir / "sample_predictions.csv"
    write_predictions(predictions_path, test_rows, states)

    result_models: dict[str, Any] = {}
    for name, state in states.items():
        result_models[name] = {
            "metrics": metrics(test_rows, state),
            "group_metrics": group_metrics(test_rows, state),
        }
    comparison = {
        "roles_frozen_before_test": {
            "primary": "premium hierarchical XGBoost",
            "same_specification_ablation_control": "clean-only hierarchical XGBoost",
            "current_model_comparator": "clean-v5 SVM",
            "direct_four_class_scored": False,
            "test_metric_switching_allowed": False,
        },
        "premium_minus_clean_only_fixed_hierarchical": {
            "calibrated_group_bootstrap": paired_bootstrap(
                test_rows,
                premium_calibrated,
                clean_only_calibrated,
                comparison_label=(
                    "frozen premium hierarchical XGBoost calibrated minus frozen "
                    "same-config clean-only hierarchical XGBoost calibrated"
                ),
            ),
            "calibrated_joint_mcnemar": exact_mcnemar(
                test_rows, premium_calibrated, clean_only_calibrated
            ),
        },
        "premium_minus_current_clean_v5_svm": {
            "calibrated_group_bootstrap": paired_bootstrap(
                test_rows,
                premium_calibrated,
                baseline_calibrated,
                comparison_label=(
                    "frozen premium hierarchical XGBoost calibrated minus frozen "
                    "clean-v5 SVM calibrated"
                ),
            ),
            "calibrated_joint_mcnemar": exact_mcnemar(
                test_rows, premium_calibrated, baseline_calibrated
            ),
        },
    }
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": utc_now(),
        "protocol": {
            "stage": "one_shot_frozen_strict_test",
            "one_shot_seal_created_before_test_open": True,
            "selection_and_models_verified_before_test_open": True,
            "selected_xgb_architecture": "hierarchical",
            "original_train_oof_architecture_winner": "direct_four_class",
            "xgb_test_candidate_selected_by": (
                "transparent post-validation business architecture constraint, "
                "frozen before strict-test access"
            ),
            "decision_uses_validation_evidence": True,
            "validation_used_for_frozen_temperature_calibration": True,
            "direct_four_class_not_scored_by_precommitment": True,
            "test_used_for_model_or_architecture_selection": False,
            "test_used_for_threshold_or_temperature_fit": False,
            "test_used_for_feature_selection_or_ensemble_weighting": False,
            "deployment_switching_from_test_allowed": False,
            "same_in_memory_feature_matrix_used_for_all_three_models": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "frozen_inputs": frozen.frozen_fingerprints,
        "test_candidate_decision": frozen.decision,
        "strict_test_inputs": {
            "manifest": test_manifest_record,
            "feature_cache": test_cache_record,
            "rows": len(test_rows),
            "groups": len({str(row["group_id"]) for row in test_rows}),
            "class_counts": {
                CLASS_NAMES[class_id]: int(
                    sum(int(row["class_id"]) == class_id for row in test_rows)
                )
                for class_id in range(4)
            },
        },
        "leakage_audit": leakage,
        "models": result_models,
        "paired_comparison": comparison,
        "sample_predictions": fingerprint(predictions_path),
        "historical_observation": historical_observation(args.historical_report),
    }
    report_path = output_dir / "strict_test_report.json"
    write_new_json(report_path, report)
    completed = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "completed; the frozen strict test cannot be opened again",
        "one_shot_seal": fingerprint(seal_path),
        "selection": frozen.frozen_fingerprints["premium_selection"],
        "model": frozen.frozen_fingerprints["premium_model"],
        "test_candidate_decision": frozen.frozen_fingerprints[
            "test_candidate_decision"
        ],
        "clean_only_hierarchical_model": frozen.frozen_fingerprints[
            "clean_only_hierarchical_model"
        ],
        "clean_v5_svm_model": frozen.frozen_fingerprints["baseline_model"],
        "strict_test_manifest": test_manifest_record,
        "strict_test_feature": test_cache_record,
        "report": fingerprint(report_path),
        "sample_predictions": fingerprint(predictions_path),
        "selection_calibration_threshold_or_weight_changed_after_test": False,
    }
    write_new_json(output_dir / "sha256_receipt.json", completed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        frozen = preflight_frozen_inputs(args)
        print(
            json.dumps(
                {
                    "schema": SCRIPT_VERSION,
                    "status": "preflight_passed_without_strict_test_access",
                    "frozen_inputs": frozen.frozen_fingerprints,
                    "decision_primary": frozen.decision.get(
                        "primary_test_candidate"
                    ),
                    "declared_test_paths_not_statted_hashed_or_opened": [
                        str(args.strict_test_manifest.resolve()),
                        str(args.imagenet_cls_test_feature.resolve()),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run(args)


if __name__ == "__main__":
    main()
