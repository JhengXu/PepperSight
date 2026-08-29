#!/usr/bin/env python3
"""Validation-only hyperparameter selection for the cached SSL head features."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_ssl_hierarchical import (
    FINAL_NAMES,
    GRADE_NAMES,
    SPECIES_NAMES,
    HierarchicalHead,
    evaluate_model,
    metrics,
    probabilities_for_features,
    read_manifest,
    train_head,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "yolo11_pepper/runs/hierarchical_ssl_v3"
MANIFEST = ROOT / "yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"
INITIAL = ROOT / (
    "yolo11_pepper/runs/hierarchical_combined_single_relabelled_v2_kornia_hybrid/"
    "best_hierarchical_kornia_calibrated.pt"
)


def main() -> None:
    device = torch.device("cpu")
    all_rows = read_manifest(MANIFEST)
    rows = {split: [row for row in all_rows if row["split"] == split] for split in ("train", "val", "test")}
    features = {
        split: torch.load(RUN / f"{split}_features.pt", map_location="cpu", weights_only=True)["features"]
        for split in rows
    }
    labels = {split: torch.tensor([int(row["class_id"]) for row in rows[split]]) for split in rows}
    initial_payload = torch.load(INITIAL, map_location="cpu", weights_only=True)
    initial = HierarchicalHead(int(initial_payload["feature_dim"]))
    initial.load_state_dict(initial_payload["head_state_dict"])

    with (RUN / "relabel_decisions.csv").open(encoding="utf-8-sig", newline="") as handle:
        relabel = list(csv.DictReader(handle))
    effective_labels = torch.tensor([int(row["effective_class_id"]) for row in relabel])
    base_weights = torch.tensor([float(row["label_weight"]) for row in relabel])
    states = [row["label_state"] for row in relabel]

    configs = [
        # name, lr, ssl, smoothing, class-weight power, uncertain weight scale
        ("c01", 2e-4, 0.20, 0.03, 0.50, 0.60),
        ("c02", 3e-4, 0.25, 0.04, 0.50, 0.75),
        ("c03", 4e-4, 0.30, 0.05, 0.50, 1.00),
        ("c04", 5e-4, 0.40, 0.05, 0.50, 0.75),
        ("c05", 3e-4, 0.50, 0.03, 0.75, 0.75),
        ("c06", 4e-4, 0.55, 0.05, 0.75, 1.00),
        ("c07", 5e-4, 0.65, 0.06, 0.75, 0.60),
        ("c08", 2e-4, 0.35, 0.06, 1.00, 1.00),
        ("c09", 3e-4, 0.45, 0.05, 1.00, 0.75),
        ("c10", 4e-4, 0.60, 0.04, 1.00, 0.60),
        ("c11", 6e-4, 0.30, 0.03, 0.35, 0.75),
        ("c12", 7e-4, 0.50, 0.05, 0.35, 1.00),
    ]
    candidates = []
    best = None
    for config_index, (name, lr, ssl, smoothing, power, uncertain_scale) in enumerate(configs):
        weights = base_weights.clone()
        for index, state in enumerate(states):
            if state in {"ambiguous_unlabelled", "model_disagrees_downweighted"}:
                weights[index] *= uncertain_scale
        args = SimpleNamespace(
            device_object=device,
            lr=lr,
            weight_decay=1e-4,
            ema_decay=0.992,
            head_batch=128,
            label_smoothing=smoothing,
            ssl_weight=ssl,
            patience=14,
            class_weight_power=power,
            quiet=True,
        )
        head, history = train_head(
            copy.deepcopy(initial), features["train"], effective_labels, weights,
            features["val"][:, 0], labels["val"], args, 60,
            30_000 + config_index * 997, select_best=True,
        )
        probability = probabilities_for_features(head, features["val"][:, 0], device)
        validation = metrics(labels["val"], probability)
        row = {
            "name": name, "lr": lr, "ssl_weight": ssl, "label_smoothing": smoothing,
            "class_weight_power": power, "uncertain_weight_scale": uncertain_scale,
            "epochs": len(history), "validation": validation,
        }
        candidates.append(row)
        score = (validation["macro_f1"], validation["accuracy"], -validation["nll"])
        print(name, f"val_acc={validation['accuracy']:.4f}", f"val_F1={validation['macro_f1']:.4f}")
        if best is None or score > best[0]:
            best = (score, name, head, row)

    assert best is not None
    _, selected_name, selected_head, selected_row = best
    final_report, temperatures = evaluate_model(
        selected_head, features["val"][:, 0], labels["val"],
        features["test"][:, 0], labels["test"], rows["test"], device,
    )
    checkpoint = RUN / "best_hierarchical_ssl_tuned_calibrated.pt"
    torch.save(
        {
            "head_state_dict": selected_head.state_dict(),
            "feature_dim": features["train"].shape[-1],
            "backbone_checkpoint": str((ROOT / "yolo11n.pt").resolve()),
            "image_size": 224,
            "species_names": SPECIES_NAMES,
            "grade_names": GRADE_NAMES,
            "final_names": FINAL_NAMES,
            "joint_temperatures": tuple(float(value) for value in temperatures),
            "recommended_inference": "hierarchical temperature calibration then joint Bayes argmax",
            "training": {
                "backbone": "frozen YOLO11 layers 0:11",
                "head_only": True,
                "ssl": "EMA teacher + FixMatch + adaptive thresholds + SoftMatch weighting",
                "selected_by": "validation macro-F1; test evaluated once after selection",
                "selected_config": selected_row,
            },
            "strict_test_metrics": final_report["test_calibrated"],
        },
        checkpoint,
    )
    result = {
        "selection_rule": "maximum validation macro-F1; no test-set tuning",
        "selected": selected_row,
        "candidates": candidates,
        "final_report": final_report,
        "checkpoint": str(checkpoint),
    }
    (RUN / "tuning_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_row, "final_report": final_report, "checkpoint": str(checkpoint)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
