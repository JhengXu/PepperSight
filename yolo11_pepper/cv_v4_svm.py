#!/usr/bin/env python3
"""Train-only grouped OOF robustness check for the v4 YOLO-feature SVM head."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "yolo11_pepper/datasets/pepper_ssl_v4_audit/train_label_audit.csv"
FEATURE_ROOT = ROOT / "yolo11_pepper/runs/hierarchical_v4/features"
OUTPUT = ROOT / "yolo11_pepper/runs/hierarchical_v4/svm_group_oof.json"


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_features(name: str) -> np.ndarray:
    payload = torch.load(
        FEATURE_ROOT / f"{name}_train.pt", map_location="cpu", weights_only=True
    )
    block = torch.nn.functional.normalize(payload["features"][:, 0].float(), dim=1)
    return block.numpy()


def main() -> None:
    rows = read_rows()
    x = np.concatenate(
        [load_features("pepper_det"), load_features("imagenet_cls")], axis=1
    )
    y = np.array([int(row["class_id"]) for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    # Use only image-resolution reliability here, not OOF model disagreement,
    # so a held fold cannot influence fit-fold weights through an audit teacher.
    resolution = np.array([float(row["resolution_factor"]) for row in rows])
    all_results = []
    for seed in (2041, 2053, 2069):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for c in (3.0, 10.0, 30.0):
            prediction = np.full(len(y), -1, dtype=np.int64)
            for fit, held in splitter.split(x, y, groups):
                species_model = SVC(
                    C=c, kernel="rbf", gamma="scale", class_weight="balanced"
                ).fit(x[fit], y[fit] // 2, sample_weight=resolution[fit])
                species_score = species_model.decision_function(x[held])
                species_probability = 1.0 / (1.0 + np.exp(-species_score))
                species_probability = np.stack(
                    [1.0 - species_probability, species_probability], axis=1
                )
                grade_probability = []
                for species in range(2):
                    local = fit[(y[fit] // 2) == species]
                    grade_model = SVC(
                        C=c, kernel="rbf", gamma="scale", class_weight="balanced"
                    ).fit(x[local], y[local] % 2, sample_weight=resolution[local])
                    score = grade_model.decision_function(x[held])
                    probability_bad = 1.0 / (1.0 + np.exp(-score))
                    grade_probability.append(
                        np.stack([1.0 - probability_bad, probability_bad], axis=1)
                    )
                grade_probability = np.stack(grade_probability, axis=1)
                joint = species_probability[:, :, None] * grade_probability
                prediction[held] = joint.reshape(len(held), 4).argmax(1)
            if (prediction < 0).any():
                raise AssertionError("OOF prediction incomplete")
            group_accuracy = [
                float((prediction[groups == group] == y[groups == group]).mean())
                for group in sorted(set(groups))
            ]
            result = {
                "seed": seed,
                "C": c,
                "samples": len(y),
                "groups": len(set(groups)),
                "accuracy": float(accuracy_score(y, prediction)),
                "species_accuracy": float(((prediction // 2) == (y // 2)).mean()),
                "conditional_grade_accuracy": float(
                    ((prediction % 2) == (y % 2)).mean()
                ),
                "macro_f1": float(f1_score(y, prediction, average="macro")),
                "group_accuracy_mean": float(np.mean(group_accuracy)),
                "group_accuracy_min": float(np.min(group_accuracy)),
                "confusion": confusion_matrix(y, prediction, labels=[0, 1, 2, 3]).tolist(),
            }
            all_results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    aggregate = {}
    for c in (3.0, 10.0, 30.0):
        selected = [row for row in all_results if row["C"] == c]
        aggregate[str(c)] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "accuracy",
                "species_accuracy",
                "conditional_grade_accuracy",
                "macro_f1",
                "group_accuracy_mean",
                "group_accuracy_min",
            )
        }
    report = {
        "scope": "train-only repeated StratifiedGroupKFold; val/test never loaded",
        "weight_policy": "resolution_factor only; no model-derived audit weights",
        "runs": all_results,
        "aggregate": aggregate,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
