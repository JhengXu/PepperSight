#!/usr/bin/env python3
"""Train and evaluate a four-class XGBoost head on frozen YOLO11 features.

The script follows the same split protocol as ``train_ssl_hierarchical.py``:

* the 256-dimensional YOLO11 backbone features are loaded from the existing
  cache and the backbone remains frozen;
* XGBoost hyperparameters are selected using validation macro-F1 only;
* test features and labels are loaded only after model selection;
* the pre-change deployed SSL head is read from ``metrics.json`` for a directly
  comparable per-class test F1 report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

# PyTorch and XGBoost both load OpenMP on macOS.  Keeping this small head
# single-threaded avoids a native runtime collision while remaining fast on a
# 643-row training set.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "yolo11_pepper/runs/hierarchical_ssl_v3"
MANIFEST = ROOT / "yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"
RELABEL = RUN / "relabel_decisions.csv"
BASELINE_METRICS = RUN / "metrics.json"
FINAL_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-selected XGBoost head on frozen 256-D YOLO11 features."
    )
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--relabel-decisions", type=Path, default=RELABEL)
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_METRICS)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="CPU threads used by each XGBoost fit.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_feature_cache(run: Path, split: str) -> np.ndarray:
    payload = torch.load(
        run / f"{split}_features.pt", map_location="cpu", weights_only=True
    )
    features = payload["features"]
    if features.ndim != 3 or features.shape[-1] != 256:
        raise ValueError(
            f"Expected {split} features shaped [samples, views, 256], "
            f"got {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"Non-finite values found in {split} features")
    return features.float().numpy()


def class_balanced_weights(labels: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    active = base_weights > 0
    counts = np.bincount(labels[active], minlength=len(FINAL_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"At least one class has no active training rows: {counts.tolist()}")
    class_weights = counts.sum() / (len(FINAL_NAMES) * counts)
    weights = base_weights * class_weights[labels]
    return (weights / weights[active].mean()).astype(np.float32)


def classification_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(4), zero_division=0
    )
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "per_class": [
            {
                "class_id": class_id,
                "class_name": FINAL_NAMES[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in range(4)
        ],
        "confusion": confusion_matrix(labels, prediction, labels=np.arange(4)).tolist(),
    }


def candidate_grid() -> list[dict[str, Any]]:
    """A compact, deterministic grid suitable for the small cached dataset."""
    candidates: list[dict[str, Any]] = []
    for view_mode in ("canonical", "all_train_views"):
        for max_depth in (2, 3, 4):
            for learning_rate in (0.03, 0.07):
                for min_child_weight in (1.0, 5.0):
                    for n_estimators in (120, 260):
                        candidates.append(
                            {
                                "view_mode": view_mode,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "min_child_weight": min_child_weight,
                                "n_estimators": n_estimators,
                            }
                        )
    return candidates


def make_model(config: dict[str, Any], seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        n_estimators=int(config["n_estimators"]),
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )


def training_matrix(
    train_features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    view_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if view_mode == "canonical":
        x, y, w = train_features[:, 0], labels, weights
    elif view_mode == "all_train_views":
        views = train_features.shape[1]
        x = train_features.reshape(-1, train_features.shape[-1])
        y = np.repeat(labels, views)
        # Keep each image's total contribution constant when its views are expanded.
        w = np.repeat(weights / views, views)
    else:
        raise ValueError(f"Unknown view mode: {view_mode}")
    active = w > 0
    return x[active], y[active], w[active]


def write_f1_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class", "f1"))
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({"class": row["class_name"], "f1": row["f1"]})


def load_current_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # This was the deployed head immediately before the XGBoost replacement.
    return payload["new_ssl_head"]["test_calibrated"]


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(args.manifest.resolve())
    rows = {
        split: [row for row in manifest_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    train_features = load_feature_cache(run, "train")
    val_features = load_feature_cache(run, "val")
    if len(train_features) != len(rows["train"]) or len(val_features) != len(rows["val"]):
        raise ValueError("Feature caches do not match manifest split sizes")

    relabel_rows = read_csv(args.relabel_decisions.resolve())
    if len(relabel_rows) != len(rows["train"]):
        raise ValueError("Relabel decisions do not match training rows")
    if [row["path"] for row in relabel_rows] != [row["path"] for row in rows["train"]]:
        raise ValueError("Relabel decisions and training manifest are not in the same order")
    train_labels = np.array(
        [int(row["effective_class_id"]) for row in relabel_rows], dtype=np.int64
    )
    base_weights = np.array(
        [float(row["label_weight"]) for row in relabel_rows], dtype=np.float32
    )
    train_weights = class_balanced_weights(train_labels, base_weights)
    val_labels = np.array([int(row["class_id"]) for row in rows["val"]], dtype=np.int64)
    val_x = val_features[:, 0]

    results: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], dict[str, Any], XGBClassifier] | None = None
    grid = candidate_grid()
    for index, config in enumerate(grid, start=1):
        x, y, weights = training_matrix(
            train_features, train_labels, train_weights, config["view_mode"]
        )
        model = make_model(config, args.seed, args.n_jobs)
        model.fit(x, y, sample_weight=weights)
        validation = classification_metrics(val_labels, model.predict(val_x))
        result = {**config, "validation": validation}
        results.append(result)
        score = (float(validation["macro_f1"]), float(validation["accuracy"]))
        print(
            f"[{index:02d}/{len(grid)}] {config} "
            f"val_macro_f1={score[0]:.6f} val_accuracy={score[1]:.6f}"
        )
        if best is None or score > best[0]:
            best = (score, config, model)

    if best is None:
        raise RuntimeError("No XGBoost candidate was trained")
    _, selected_config, selected_model = best
    results.sort(
        key=lambda row: (
            float(row["validation"]["macro_f1"]),
            float(row["validation"]["accuracy"]),
        ),
        reverse=True,
    )
    (run / "xgboost_validation_candidates.json").write_text(
        json.dumps(
            {
                "selection_scope": "train/validation only; test cache not loaded",
                "selection_rule": "maximum validation macro-F1, then accuracy",
                "feature_dim": 256,
                "candidates": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Strict test boundary: test features are first opened after selection is final.
    test_features = load_feature_cache(run, "test")
    if len(test_features) != len(rows["test"]):
        raise ValueError("Test feature cache does not match manifest split size")
    test_labels = np.array([int(row["class_id"]) for row in rows["test"]], dtype=np.int64)
    test_metrics = classification_metrics(test_labels, selected_model.predict(test_features[:, 0]))
    baseline = load_current_baseline(args.baseline_metrics.resolve())

    # Save the native booster directly.  This remains portable across
    # scikit-learn releases (the local sklearn version is newer than the
    # compatibility metadata expected by XGBoost 2.1.x).
    selected_model.get_booster().save_model(str(run / "best_xgboost_head.ubj"))
    comparison = []
    for xgb_row, baseline_row in zip(test_metrics["per_class"], baseline["per_class"]):
        if xgb_row["class_name"] != baseline_row["class_name"]:
            raise ValueError("XGBoost and baseline class orders differ")
        comparison.append(
            {
                "class": xgb_row["class_name"],
                "xgboost_f1": xgb_row["f1"],
                "baseline_f1": baseline_row["f1"],
                "delta": xgb_row["f1"] - baseline_row["f1"],
            }
        )
    report = {
        "architecture": "frozen YOLO11 256-D backbone -> direct four-class XGBoost head",
        "protocol": {
            "selection": "validation macro-F1 then accuracy",
            "test_evaluations": 1,
            "test_used_for_model_selection": False,
            "training_labels": "SSL relabel decisions and label weights, matching current head",
            "baseline": "pre-change deployed best_hierarchical_ssl_calibrated.pt",
        },
        "split_sizes": {split: len(rows[split]) for split in rows},
        "selected_config": selected_config,
        "validation": results[0]["validation"],
        "test": test_metrics,
        "current_baseline_test": baseline,
        "per_class_comparison": comparison,
        "macro_f1_delta": test_metrics["macro_f1"] - baseline["macro_f1"],
        "better_on_test_macro_f1": (
            "xgboost" if test_metrics["macro_f1"] > baseline["macro_f1"] else "current_baseline"
        ),
        "model": str((run / "best_xgboost_head.ubj").resolve()),
    }
    (run / "xgboost_test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_f1_csv(run / "xgboost_test_f1.csv", test_metrics)
    write_f1_csv(run / "current_baseline_test_f1.csv", baseline)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
