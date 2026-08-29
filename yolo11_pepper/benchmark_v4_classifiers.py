#!/usr/bin/env python3
"""Validation-only classical benchmark on leakage-safe v4 feature families."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "yolo11_pepper/runs/hierarchical_v4"
TRAIN_MANIFEST = (
    ROOT / "yolo11_pepper/datasets/pepper_ssl_v4_merged/train_manifest.csv"
)
VAL_MANIFEST = (
    ROOT / "yolo11_pepper/datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load(name: str, split: str) -> tuple[np.ndarray, list[str]]:
    feature_directory = "features_merged" if split == "train" else "features"
    payload = torch.load(
        RUN / feature_directory / f"{name}_{split}.pt",
        map_location="cpu",
        weights_only=True,
    )
    feature = torch.nn.functional.normalize(payload["features"].float(), dim=-1)
    return feature.numpy(), [str(Path(path).resolve()) for path in payload["paths"]]


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion": confusion_matrix(labels, prediction, labels=[0, 1, 2, 3]).tolist(),
    }


def main() -> None:
    # Read physically separated train/validation manifests.  The unified
    # manifest (which also contains strict-test metadata) is never opened.
    train_rows = rows(TRAIN_MANIFEST)
    val_rows = rows(VAL_MANIFEST)
    loaded_train = [load(name, "train") for name in ("pepper_det", "imagenet_cls")]
    loaded_val = [load(name, "val") for name in ("pepper_det", "imagenet_cls")]
    train_blocks = [item[0] for item in loaded_train]
    val_blocks = [item[0] for item in loaded_val]
    expected_train_paths = [str(Path(row["path"]).resolve()) for row in train_rows]
    expected_val_paths = [str(Path(row["path"]).resolve()) for row in val_rows]
    if any(item[1] != expected_train_paths for item in loaded_train):
        raise ValueError("Train feature order does not match explicit train manifest")
    if any(item[1] != expected_val_paths for item in loaded_val):
        raise ValueError("Validation feature order does not match explicit model-selection manifest")
    labels = np.array([int(row["class_id"]) for row in train_rows])
    val_labels = np.array([int(row["class_id"]) for row in val_rows])
    pair_counts: dict[str, int] = {}
    for row in train_rows:
        pair_counts[row["pair_id"]] = pair_counts.get(row["pair_id"], 0) + 1
    pair_weight = np.array([1.0 / pair_counts[row["pair_id"]] for row in train_rows])
    grade_weight = np.array([float(row["grade_weight"]) for row in train_rows]) * pair_weight
    species_weight = np.array([float(row["species_weight"]) for row in train_rows]) * pair_weight
    results: list[dict[str, object]] = []
    best_score = (-1.0, -1.0)
    best_payload: dict[str, object] | None = None

    for view_mode in ("canonical", "view_mean"):
        if view_mode == "canonical":
            x = np.concatenate([block[:, 0] for block in train_blocks], axis=1)
        else:
            x = np.concatenate([block.mean(1) for block in train_blocks], axis=1)
        val_x = np.concatenate([block[:, 0] for block in val_blocks], axis=1)
        for family, parameters in (
            ("logistic", (0.01, 0.03, 0.1, 0.3, 1.0)),
            ("rbf_svc", (0.3, 1.0, 3.0, 10.0, 30.0)),
        ):
            for c in parameters:
                if family == "logistic":
                    constructor = lambda: LogisticRegression(
                        C=c, max_iter=4000, class_weight="balanced", random_state=2041
                    )
                else:
                    constructor = lambda: SVC(
                        C=c,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=2041,
                    )
                species_model = constructor()
                species_model.fit(x, labels // 2, sample_weight=species_weight)
                species_probability = species_model.predict_proba(val_x)
                grade_probability = []
                grade_models = []
                for species in range(2):
                    keep = ((labels // 2) == species) & (grade_weight > 0)
                    model = constructor()
                    model.fit(
                        x[keep], labels[keep] % 2, sample_weight=grade_weight[keep]
                    )
                    grade_models.append(model)
                    grade_probability.append(model.predict_proba(val_x))
                grade_probability = np.stack(grade_probability, axis=1)
                joint = species_probability[:, :, None] * grade_probability
                prediction = joint.reshape(len(val_x), 4).argmax(1)
                result = {
                    "family": family,
                    "C": c,
                    "view_mode": view_mode,
                    **metrics(val_labels, prediction),
                }
                results.append(result)
                score = (float(result["macro_f1"]), float(result["accuracy"]))
                if score > best_score:
                    best_score = score
                    best_payload = {
                        "species_model": species_model,
                        "grade_models": grade_models,
                        "feature_families": ("pepper_det", "imagenet_cls"),
                        "view_mode": view_mode,
                        "C": c,
                        "validation": result,
                        "protocol": {
                            "test_loaded": False,
                            "test_metadata_read": False,
                            "selection": "validation macro-F1 then accuracy",
                            "train_manifest": str(TRAIN_MANIFEST.resolve()),
                            "validation_manifest": str(VAL_MANIFEST.resolve()),
                            "species_training": "all train rows with species_weight",
                            "grade_training": "only train rows with grade_weight > 0",
                            "paired_views": "canonical/detector pair has total sample mass 1",
                        },
                    }
                print(json.dumps(result, ensure_ascii=False))

    results.sort(key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)
    output = RUN / "classical_validation_benchmark_strict.json"
    output.write_text(
        json.dumps(
            {"scope": "train/validation only; test never loaded", "results": results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if best_payload is None:
        raise RuntimeError("No benchmark model was trained")
    joblib.dump(best_payload, RUN / "best_hierarchical_v4_svm_strict.joblib")
    print("BEST", json.dumps(results[:5], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
