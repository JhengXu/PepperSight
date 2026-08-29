#!/usr/bin/env python3
"""Compare premium-v1 with clean-v5 feature domains without opening test data."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "runs/premium_feature_strategy_v1"
CLASS_NAMES = ("子弹头_一级", "子弹头_二级", "条子_一级", "条子_二级")

MANIFESTS = {
    "train": PROJECT
    / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv",
    "val": PROJECT
    / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv",
    "premium": OUTPUT / "premium_train_manifest.csv",
}
FEATURES = {
    "imagenet_cls": {
        "train": PROJECT
        / "runs/hierarchical_v5_clean/features_cls256_reaudit/imagenet_cls_train.pt",
        "val": PROJECT
        / "runs/hierarchical_v5_clean/features_cls256/imagenet_cls_val.pt",
        "premium": OUTPUT / "features_cls256/imagenet_cls_train.pt",
    },
    "strict_det": {
        "train": PROJECT / "runs/hierarchical_v5_clean/features/strict_det_train.pt",
        "val": PROJECT / "runs/hierarchical_v5_clean/features/strict_det_val.pt",
        "premium": OUTPUT / "features_strict_det256/strict_det_train.pt",
    },
    "quality_handcrafted": {
        "train": PROJECT / "runs/hierarchical_v5_clean/features_quality/quality_train.pt",
        "val": PROJECT / "runs/hierarchical_v5_clean/features_quality/quality_val.pt",
        "premium": OUTPUT / "features_quality/quality_train.pt",
    },
}
SCHEMES = {
    "imagenet_cls": ("imagenet_cls",),
    "strict_det": ("strict_det",),
    "quality_handcrafted": ("quality_handcrafted",),
    "imagenet_plus_strict": ("imagenet_cls", "strict_det"),
    "imagenet_plus_quality": ("imagenet_cls", "quality_handcrafted"),
    "all_three": ("imagenet_cls", "strict_det", "quality_handcrafted"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def l2(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(denominator, 1e-12, None)


def load_family(name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split, path in FEATURES[name].items():
        cache = torch.load(path.resolve(), map_location="cpu", weights_only=False)
        manifest_rows = rows(MANIFESTS[split])
        expected_paths = [str(Path(row["path"]).resolve()) for row in manifest_rows]
        if list(cache["paths"]) != expected_paths:
            raise ValueError(f"{name}/{split}: cache path order != manifest")
        current_labels = np.asarray([int(row["class_id"]) for row in manifest_rows])
        if not np.array_equal(cache["class_ids"].numpy(), current_labels):
            raise ValueError(f"{name}/{split}: cache labels != manifest")
        feature = cache["features"].numpy().astype(np.float32)[:, 0, :]
        matrices[split] = feature
        labels[split] = current_labels

    if name == "quality_handcrafted":
        mean = matrices["train"].mean(axis=0, dtype=np.float64)
        std = matrices["train"].std(axis=0, dtype=np.float64)
        retained = std > 1e-8
        for split in matrices:
            matrices[split] = l2(
                ((matrices[split][:, retained] - mean[retained]) / std[retained]).astype(
                    np.float32
                )
            )
    else:
        matrices = {split: l2(value) for split, value in matrices.items()}
    return matrices, labels


def scheme_metrics(
    matrices: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        train = matrices["train"][labels["train"] == class_id]
        val = matrices["val"][labels["val"] == class_id]
        premium = matrices["premium"][labels["premium"] == class_id]
        train_centroid = l2(train.mean(axis=0, keepdims=True))[0]
        val_centroid = l2(val.mean(axis=0, keepdims=True))[0]
        premium_centroid = l2(premium.mean(axis=0, keepdims=True))[0]
        clean_spread = np.median(1.0 - train @ train_centroid)
        val_shift = 1.0 - float(val_centroid @ train_centroid)
        premium_shift = 1.0 - float(premium_centroid @ train_centroid)

        same = premium @ train.T
        wrong = premium @ matrices["train"][labels["train"] != class_id].T
        same_nearest = np.max(same, axis=1)
        wrong_nearest = np.max(wrong, axis=1)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "premium_rows": len(premium),
                "clean_train_rows": len(train),
                "clean_val_rows": len(val),
                "clean_train_spread_cosine_distance_median": float(clean_spread),
                "clean_val_to_train_centroid_cosine_distance": val_shift,
                "premium_to_train_centroid_cosine_distance": premium_shift,
                "premium_shift_over_clean_val_shift": float(
                    premium_shift / max(val_shift, 1e-12)
                ),
                "premium_nearest_same_class_cosine_median": float(
                    np.median(same_nearest)
                ),
                "premium_nearest_wrong_class_cosine_median": float(
                    np.median(wrong_nearest)
                ),
                "premium_nearest_class_margin_median": float(
                    np.median(same_nearest - wrong_nearest)
                ),
            }
        )
    return {
        "dimension": int(matrices["train"].shape[1]),
        "per_class": per_class,
        "aggregate": {
            "premium_shift_median": float(
                np.median(
                    [item["premium_to_train_centroid_cosine_distance"] for item in per_class]
                )
            ),
            "clean_val_shift_median": float(
                np.median(
                    [item["clean_val_to_train_centroid_cosine_distance"] for item in per_class]
                )
            ),
            "premium_over_clean_val_shift_median": float(
                np.median([item["premium_shift_over_clean_val_shift"] for item in per_class])
            ),
            "premium_nearest_class_margin_median": float(
                np.median(
                    [item["premium_nearest_class_margin_median"] for item in per_class]
                )
            ),
        },
    }


def main() -> None:
    family_matrices: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] | None = None
    for family in FEATURES:
        loaded, current_labels = load_family(family)
        family_matrices[family] = loaded
        if labels is None:
            labels = current_labels
        elif any(
            not np.array_equal(labels[split], current_labels[split]) for split in labels
        ):
            raise ValueError(f"label order differs for {family}")
    assert labels is not None

    results: dict[str, Any] = {}
    for scheme, families in SCHEMES.items():
        scale = np.sqrt(len(families))
        combined = {
            split: np.concatenate(
                [family_matrices[family][split] / scale for family in families], axis=1
            )
            for split in ("train", "val", "premium")
        }
        results[scheme] = scheme_metrics(combined, labels)

    existing_validation = {}
    for label, path in {
        "imagenet_cls_256_only": PROJECT
        / "runs/hierarchical_v5_clean/selection_reaudit.json",
        "strict_det_plus_imagenet_cls_384": PROJECT
        / "runs/hierarchical_v5_clean/selection.json",
        "imagenet_cls_plus_handcrafted_quality": PROJECT
        / "runs/hierarchical_v5_clean/selection_grade_fusion.json",
    }.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["selected"]["validation_metrics_raw"]
        existing_validation[label] = {
            "source": str(path.resolve()),
            "species_accuracy": metrics["species_accuracy"],
            "conditional_grade_accuracy": metrics["conditional_grade_accuracy"],
            "joint_accuracy": metrics["joint_accuracy"],
            "joint_macro_f1": metrics["joint_macro_f1"],
            "note": "historical clean train/validation selection result; not a strict-test result",
        }

    report = {
        "schema": "premium-feature-domain-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "strict_test_opened": False,
            "data_used": ["clean_v5_train", "clean_v5_model_selection", "premium_train"],
            "analysis_role": "descriptive feature-domain comparison; no test selection",
        },
        "preprocessing": {
            "deep": "canonical view 0; row-wise L2",
            "quality": "clean-train StandardScaler; retained nonconstant columns; row-wise L2",
            "combination": "each family independently normalized, then divided by sqrt(number of families)",
        },
        "feature_domain_metrics": results,
        "existing_clean_validation_evidence": existing_validation,
        "recommendation": {
            "primary_xgboost_features": "imagenet_cls",
            "dimension": 3584,
            "reason": (
                "The 256px ImageNet-cls family has the strongest existing clean-v5 "
                "validation result. Adding strict-detector or handcrafted-quality "
                "features previously reduced clean validation joint accuracy and/or "
                "macro-F1. The premium batch is a single photographic domain, so extra "
                "colour/texture features increase shortcut risk."
            ),
            "strict_det_role": "diagnostic ablation only; not the primary XGBoost block",
            "quality_role": "diagnostic ablation only; do not deploy unless independent camera validation improves",
            "premium_background_policy": (
                "segmentation and standardized recompositing are required because raw "
                "pepper occupancy is only about 3-12.5% of the photograph and all images "
                "share the same white-table/light domain"
            ),
        },
        "input_fingerprints": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "manifests": {
                split: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for split, path in MANIFESTS.items()
            },
            "feature_caches": {
                family: {
                    split: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for split, path in paths.items()
                }
                for family, paths in FEATURES.items()
            },
        },
    }
    destination = OUTPUT / "feature_domain_report.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"saved": str(destination), "recommendation": report["recommendation"], "aggregates": {name: item["aggregate"] for name, item in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
