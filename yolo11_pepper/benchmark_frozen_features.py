#!/usr/bin/env python3
"""Validation-only benchmark of classifiers on frozen YOLO11 pepper features.

The strict test split is deliberately never loaded here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "yolo11_pepper/runs/hierarchical_ssl_v3"
MANIFEST = ROOT / "yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"
OUT = RUN / "frozen_feature_benchmark.json"


def read_rows():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def report(labels, prediction):
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "confusion": confusion_matrix(labels, prediction, labels=[0, 1, 2, 3]).tolist(),
    }


def hierarchical_predict(species_model, grade_models, features):
    species_p = species_model.predict_proba(features)
    grade_p = np.stack([model.predict_proba(features) for model in grade_models], axis=1)
    joint = species_p[:, :, None] * grade_p
    return joint.reshape(len(features), 4).argmax(1)


def main():
    rows = read_rows()
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    train_features = torch.load(RUN / "train_features.pt", map_location="cpu", weights_only=True)["features"].numpy()
    val_features = torch.load(RUN / "val_features.pt", map_location="cpu", weights_only=True)["features"][:, 0].numpy()
    train_labels = np.array([int(row["class_id"]) for row in train_rows])
    val_labels = np.array([int(row["class_id"]) for row in val_rows])

    with (RUN / "relabel_decisions.csv").open(encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    effective = np.array([int(row["effective_class_id"]) for row in decisions])
    label_weight = np.array([float(row["label_weight"]) for row in decisions])
    group_sizes = {}
    for row in train_rows:
        group_sizes[row["group_id"]] = group_sizes.get(row["group_id"], 0) + 1
    group_weight = np.array([1.0 / np.sqrt(group_sizes[row["group_id"]]) for row in train_rows])
    group_weight /= group_weight.mean()

    candidates = []
    for views in ("base", "all"):
        if views == "base":
            x = train_features[:, 0]
            y = effective.copy()
            w = label_weight * group_weight
        else:
            x = train_features.reshape(-1, train_features.shape[-1])
            y = np.repeat(effective, train_features.shape[1])
            w = np.repeat(label_weight * group_weight, train_features.shape[1])
        keep = w > 0
        x, y, w = x[keep], y[keep], w[keep]

        direct_models = []
        for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            direct_models.append((f"logreg_C{c}", make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=2031),
            )))
        for components in (32, 64, 128):
            direct_models.append((f"pca{components}_logreg", make_pipeline(
                StandardScaler(), PCA(n_components=components, whiten=True, random_state=2031),
                LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=2031),
            )))
        direct_models.extend([
            ("lda_shrink", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
            ("extra_trees", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=2031, n_jobs=-1,
            )),
        ])
        for gamma in ("scale", 0.001, 0.003, 0.01):
            for c in (0.3, 1.0, 3.0, 10.0):
                direct_models.append((f"rbf_C{c}_g{gamma}", make_pipeline(
                    StandardScaler(), SVC(C=c, gamma=gamma, class_weight="balanced", probability=True, random_state=2031)
                )))

        for name, model in direct_models:
            try:
                model.fit(x, y, **({"logisticregression__sample_weight": w} if name.startswith("logreg") else {}))
            except TypeError:
                model.fit(x, y)
            pred = model.predict(val_features)
            row = {"kind": "direct", "views": views, "name": name, **report(val_labels, pred)}
            candidates.append(row)
            print(row)

        # Hierarchical logistic models; fit species and grade heads separately.
        for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            base = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=3000, class_weight="balanced", random_state=2031
            ))
            species_model = clone(base).fit(x, y // 2, logisticregression__sample_weight=w)
            grade_models = []
            for species in range(2):
                mask = y // 2 == species
                grade_models.append(clone(base).fit(
                    x[mask], y[mask] % 2,
                    logisticregression__sample_weight=w[mask],
                ))
            pred = hierarchical_predict(species_model, grade_models, val_features)
            row = {"kind": "hierarchical", "views": views, "name": f"logreg_C{c}", **report(val_labels, pred)}
            candidates.append(row)
            print(row)

    candidates.sort(key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)
    OUT.write_text(json.dumps({"selection_scope": "validation only", "candidates": candidates}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BEST", json.dumps(candidates[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
