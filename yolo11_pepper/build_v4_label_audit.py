#!/usr/bin/env python3
"""Build a train-only, group-OOF label audit for the pepper dataset.

This script deliberately separates *auditing* from *evaluation*:

* only rows with ``split == train`` are used to fit audit models or derive
  label decisions;
* every prediction used for an audit decision is out-of-fold by ``group_id``;
* validation and test rows are copied through to the new manifest unchanged,
  and their labels are never read by the audit logic;
* no label is automatically flipped.  Very strong contradictions are queued
  for human review, while uncertain grades become unlabeled/soft supervision;
* the v3 dataset and manifest are never modified.

The cached features must come from label-independent frozen backbones.  One or
more feature files may be supplied; independently normalized feature blocks are
concatenated so agreement is informed by complementary detection- and
classification-pretrained YOLO11 representations rather than one model family.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


FINAL_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")
AUDIT_FIELDS = (
    "species_weight",
    "grade_weight",
    "grade_soft_target",
    "audit_status",
    "review_priority",
    "audit_reasons",
    "resolution_factor",
    "object_width",
    "object_height",
    "object_max_dimension",
    "oof_fold",
    "oof_p_original_species",
    "oof_p_good",
    "oof_p_bad",
    "oof_p_original_grade",
    "oof_species_vote_agreement",
    "oof_grade_vote_agreement",
    "oof_species_view_stability",
    "oof_grade_view_stability",
    "oof_model_count",
    "group_excluded_knn_grade",
    "group_excluded_knn_margin",
)


@dataclass(frozen=True)
class Group:
    group_id: str
    class_id: int
    indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.indices)


class HierarchicalAuditHead(nn.Module):
    """Small head trained from scratch inside each OOF fold.

    It intentionally does not load an existing pepper head because such a head
    may already have seen the held-out audit group.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.18),
        )
        self.species_head = nn.Linear(128, 2)
        self.grade_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Dropout(0.12),
                    nn.Linear(64, 2),
                )
                for _ in range(2)
            ]
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        species_logits = self.species_head(shared)
        grade_logits = torch.stack([head(shared) for head in self.grade_heads], dim=1)
        return species_logits, grade_logits


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a group-OOF, train-only v4 pepper label audit manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="+",
        default=[root / "yolo11_pepper/runs/hierarchical_ssl_v3/train_features.pt"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "yolo11_pepper/datasets/pepper_ssl_v4_audit",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models-per-fold", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--ambiguous-low", type=float, default=0.40)
    parser.add_argument("--ambiguous-high", type=float, default=0.60)
    parser.add_argument("--hard-keep-probability", type=float, default=0.75)
    parser.add_argument("--strict-opposite-probability", type=float, default=0.80)
    parser.add_argument("--strict-species-probability", type=float, default=0.90)
    parser.add_argument("--strict-view-stability", type=float, default=0.80)
    parser.add_argument("--knn-margin", type=float, default=0.005)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the requested v4 output directory; v3 is always protected",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    # Deliberately filter by split before class_id/class_name is accessed.
    train_rows = [row for row in all_rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError(f"no train rows found in {path}")
    return all_rows, train_rows


def validate_train_rows(rows: list[dict[str, str]]) -> list[Group]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        class_id = int(row["class_id"])
        if class_id not in range(4):
            raise ValueError(f"unsupported class_id={class_id} at train row {index}")
        grouped[row["group_id"]].append(index)

    groups: list[Group] = []
    for group_id, indices in grouped.items():
        class_ids = {int(rows[index]["class_id"]) for index in indices}
        if len(class_ids) != 1:
            raise ValueError(f"group {group_id!r} contains multiple labels: {sorted(class_ids)}")
        groups.append(Group(group_id, class_ids.pop(), tuple(indices)))
    return groups


def stratified_group_kfold(
    groups: list[Group], n_splits: int, seed: int
) -> tuple[list[list[int]], dict[str, int]]:
    """Deterministic StratifiedGroupKFold for single-label groups.

    The project environment intentionally has no scikit-learn dependency.  In
    this dataset every group has exactly one class, so the SGKF objective can be
    implemented directly: within each class, larger groups are greedily placed
    into the fold with the smallest current sample count for that class, then
    the smallest total fold size.  No group can cross folds.
    """

    if n_splits < 2:
        raise ValueError("folds must be at least 2")
    by_class: dict[int, list[Group]] = defaultdict(list)
    for group in groups:
        by_class[group.class_id].append(group)
    for class_id in range(4):
        if len(by_class[class_id]) < n_splits:
            raise ValueError(
                f"class {FINAL_NAMES[class_id]} has {len(by_class[class_id])} groups, "
                f"fewer than folds={n_splits}"
            )

    rng = random.Random(seed)
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_class_sizes = [[0] * 4 for _ in range(n_splits)]
    fold_total_sizes = [0] * n_splits
    group_to_fold: dict[str, int] = {}

    for class_id in range(4):
        class_groups = list(by_class[class_id])
        rng.shuffle(class_groups)
        class_groups.sort(key=lambda group: group.size, reverse=True)
        fold_tie_order = list(range(n_splits))
        rng.shuffle(fold_tie_order)
        tie_rank = {fold: rank for rank, fold in enumerate(fold_tie_order)}
        for group in class_groups:
            fold = min(
                range(n_splits),
                key=lambda candidate: (
                    fold_class_sizes[candidate][class_id],
                    fold_total_sizes[candidate],
                    tie_rank[candidate],
                ),
            )
            fold_indices[fold].extend(group.indices)
            fold_class_sizes[fold][class_id] += group.size
            fold_total_sizes[fold] += group.size
            group_to_fold[group.group_id] = fold

    assigned = sorted(index for fold in fold_indices for index in fold)
    expected = sorted(index for group in groups for index in group.indices)
    if assigned != expected:
        raise AssertionError("SGKF assignment is incomplete or duplicated")
    return [sorted(indices) for indices in fold_indices], group_to_fold


def group_balanced_weights(rows: list[dict[str, str]], indices: Iterable[int]) -> torch.Tensor:
    """Equal class mass, equal source-group mass within each class."""

    selected = list(indices)
    group_sizes = Counter(rows[index]["group_id"] for index in selected)
    class_groups: dict[int, set[str]] = defaultdict(set)
    for index in selected:
        class_groups[int(rows[index]["class_id"])].add(rows[index]["group_id"])
    weights = []
    for index in selected:
        row = rows[index]
        class_id = int(row["class_id"])
        group_id = row["group_id"]
        weights.append(1.0 / (len(class_groups[class_id]) * group_sizes[group_id]))
    result = torch.tensor(weights, dtype=torch.float32)
    return result / result.mean().clamp_min(1e-8)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def train_oof_head(
    features: torch.Tensor,
    rows: list[dict[str, str]],
    train_indices: list[int],
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> HierarchicalAuditHead:
    seed_everything(seed)
    model = HierarchicalAuditHead(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.025, 1e-6)
    )
    labels = torch.tensor([int(rows[index]["class_id"]) for index in train_indices])
    weights = group_balanced_weights(rows, train_indices)
    generator = torch.Generator().manual_seed(seed + 17)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)

    for _epoch in range(args.epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        model.train()
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            global_indices = train_index_tensor[positions]
            view_indices = torch.randint(
                0,
                features.shape[1],
                (len(positions),),
                generator=generator,
            )
            batch = features[global_indices, view_indices].to(device)
            batch_labels = labels[positions].to(device)
            batch_weights = weights[positions].to(device)
            species_labels = batch_labels // 2
            grade_labels = batch_labels % 2

            species_logits, grade_logits = model(batch)
            species_each = F.cross_entropy(
                species_logits,
                species_labels,
                reduction="none",
                label_smoothing=args.label_smoothing,
            )
            selected_grade = grade_logits[
                torch.arange(len(batch_labels), device=device), species_labels
            ]
            grade_each = F.cross_entropy(
                selected_grade,
                grade_labels,
                reduction="none",
                label_smoothing=args.label_smoothing,
            )
            loss = ((species_each + grade_each) * batch_weights).sum()
            loss /= batch_weights.sum().clamp_min(1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
    return model.cpu().eval()


def predict_fold(
    model: HierarchicalAuditHead,
    features: torch.Tensor,
    rows: list[dict[str, str]],
    held_indices: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model = model.to(device).eval()
    held = torch.tensor(held_indices, dtype=torch.long)
    held_features = features[held]
    shape = held_features.shape
    with torch.no_grad():
        species_logits, grade_logits = model(held_features.reshape(-1, shape[-1]).to(device))
        species_probs = species_logits.softmax(1).reshape(shape[0], shape[1], 2).cpu()
        grade_probs = grade_logits.softmax(2).reshape(shape[0], shape[1], 2, 2).cpu()

    original_species = torch.tensor([int(rows[index]["class_id"]) // 2 for index in held_indices])
    conditional_grade = grade_probs[
        torch.arange(len(held_indices))[:, None],
        torch.arange(shape[1])[None, :],
        original_species[:, None],
    ]
    species_mean = species_probs.mean(1)
    grade_mean = conditional_grade.mean(1)
    species_prediction = species_mean.argmax(1)
    grade_prediction = grade_mean.argmax(1)
    species_stability = (
        species_probs.argmax(2) == species_prediction[:, None]
    ).float().mean(1)
    grade_stability = (
        conditional_grade.argmax(2) == grade_prediction[:, None]
    ).float().mean(1)
    return {
        "species_probs": species_mean,
        "grade_probs": grade_mean,
        "species_prediction": species_prediction,
        "grade_prediction": grade_prediction,
        "species_stability": species_stability,
        "grade_stability": grade_stability,
    }


def object_dimensions(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        ys, xs = np.nonzero(alpha >= 4)
        if len(xs):
            width = int(xs.max() - xs.min() + 1)
            height = int(ys.max() - ys.min() + 1)
        else:
            width, height = image.size
    return width, height, max(width, height)


def resolution_factor(max_dimension: int) -> tuple[float, str | None]:
    if max_dimension < 112:
        return 0.35, "very_low_resolution"
    if max_dimension < 160:
        return 0.60, "low_resolution"
    return 1.0, None


def group_excluded_knn(
    features: torch.Tensor, rows: list[dict[str, str]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced prototype support that never uses the sample's own group."""

    sample_features = F.normalize(features.mean(1), dim=1)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group_id"]].append(index)

    group_ids: list[str] = []
    group_species: list[int] = []
    group_grades: list[int] = []
    centroids: list[torch.Tensor] = []
    for group_id, indices in grouped.items():
        class_id = int(rows[indices[0]]["class_id"])
        group_ids.append(group_id)
        group_species.append(class_id // 2)
        group_grades.append(class_id % 2)
        centroids.append(F.normalize(sample_features[indices].mean(0), dim=0))
    centroid_tensor = torch.stack(centroids)
    species_tensor = torch.tensor(group_species)
    grade_tensor = torch.tensor(group_grades)

    predictions, margins = [], []
    for index, row in enumerate(rows):
        similarities = sample_features[index] @ centroid_tensor.T
        original_species = int(row["class_id"]) // 2
        external_group = torch.tensor([group_id != row["group_id"] for group_id in group_ids])
        scores = []
        for grade in range(2):
            values = similarities[
                external_group & (species_tensor == original_species) & (grade_tensor == grade)
            ]
            if not len(values):
                raise ValueError("group-excluded KNN has no comparison group for a grade")
            top_k = min(3, len(values))
            scores.append(float(values.topk(top_k).values.mean()))
        predictions.append(int(scores[1] > scores[0]))
        margins.append(abs(scores[1] - scores[0]))
    return torch.tensor(predictions), torch.tensor(margins)


def rounded_probability(value: float) -> str:
    return f"{value:.8f}"


def grade_target_text(target: torch.Tensor) -> str:
    return json.dumps([round(float(target[0]), 8), round(float(target[1]), 8)])


def audit_train_rows(
    rows: list[dict[str, str]],
    folds: list[list[int]],
    group_to_fold: dict[str, int],
    model_outputs: dict[int, list[dict[str, torch.Tensor]]],
    knn_prediction: torch.Tensor,
    knn_margin: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    output_rows: list[dict[str, str] | None] = [None] * len(rows)
    status_counter: Counter[str] = Counter()
    priority_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    human_species_matches = 0
    human_grade_matches = 0

    for fold_id, held_indices in enumerate(folds):
        outputs = model_outputs[fold_id]
        if len(outputs) != args.models_per_fold:
            raise AssertionError("missing OOF model output")
        for local_index, global_index in enumerate(held_indices):
            row = rows[global_index]
            original_class = int(row["class_id"])
            original_species, original_grade = divmod(original_class, 2)
            species_probs = torch.stack(
                [output["species_probs"][local_index] for output in outputs]
            )
            grade_probs = torch.stack(
                [output["grade_probs"][local_index] for output in outputs]
            )
            species_predictions = torch.stack(
                [output["species_prediction"][local_index] for output in outputs]
            )
            grade_predictions = torch.stack(
                [output["grade_prediction"][local_index] for output in outputs]
            )
            species_stabilities = torch.stack(
                [output["species_stability"][local_index] for output in outputs]
            )
            grade_stabilities = torch.stack(
                [output["grade_stability"][local_index] for output in outputs]
            )

            mean_species = species_probs.mean(0)
            mean_grade = grade_probs.mean(0)
            predicted_species = int(mean_species.argmax())
            predicted_grade = int(mean_grade.argmax())
            p_original_species = float(mean_species[original_species])
            p_original_grade = float(mean_grade[original_grade])
            species_vote_agreement = float((species_predictions == predicted_species).float().mean())
            grade_vote_agreement = float((grade_predictions == predicted_grade).float().mean())
            species_view_stability = float(species_stabilities.min())
            grade_view_stability = float(grade_stabilities.min())
            human_species_matches += predicted_species == original_species
            human_grade_matches += predicted_grade == original_grade

            width, height, max_dimension = object_dimensions(Path(row["path"]))
            res_factor, res_reason = resolution_factor(max_dimension)
            if res_reason:
                resolution_counter[res_reason] += 1
            else:
                resolution_counter["normal_resolution"] += 1

            reasons: list[str] = []
            if res_reason:
                reasons.append(res_reason)

            all_species_models_opposite = bool((species_predictions != original_species).all())
            strict_species_review = (
                all_species_models_opposite
                and p_original_species <= 1.0 - args.strict_opposite_probability
                and species_view_stability >= args.strict_view_stability
            )
            if strict_species_review:
                species_weight = 0.0
                reasons.append("strict_oof_species_contradiction")
            elif p_original_species < 0.60 or species_view_stability < 0.60:
                species_weight = 0.50
                reasons.append("weak_oof_species_support")
            else:
                species_weight = 1.0

            ambiguous = args.ambiguous_low <= float(mean_grade[0]) <= args.ambiguous_high
            all_grade_models_opposite = bool((grade_predictions != original_grade).all())
            strict_opposite = (
                all_grade_models_opposite
                and (1.0 - p_original_grade) >= args.strict_opposite_probability
                and grade_view_stability >= args.strict_view_stability
                and p_original_species >= args.strict_species_probability
                and species_vote_agreement == 1.0
                and int(knn_prediction[global_index]) != original_grade
                and float(knn_margin[global_index]) >= args.knn_margin
            )

            if strict_species_review:
                audit_status = "manual_review_species"
                review_priority = 1
                grade_weight = 0.0
                soft_target = mean_grade
            elif strict_opposite:
                # Never auto-flip.  A human adjudicator must decide whether this
                # is a true label issue or a shared model blind spot.
                audit_status = "manual_review_strict_grade_contradiction"
                review_priority = 1
                grade_weight = 0.0
                soft_target = mean_grade
                reasons.append("strict_oof_grade_contradiction")
            elif ambiguous:
                audit_status = "grade_unlabelled_40_60"
                review_priority = 3
                grade_weight = 0.0
                soft_target = mean_grade
                reasons.append("oof_grade_probability_40_60")
            elif predicted_grade != original_grade:
                audit_status = "grade_unlabelled_oof_disagrees"
                review_priority = 2 if all_grade_models_opposite else 3
                grade_weight = 0.0
                soft_target = mean_grade
                reasons.append("oof_grade_disagrees_without_strict_review_gate")
            elif (
                p_original_grade >= args.hard_keep_probability
                and grade_vote_agreement == 1.0
                and grade_view_stability >= args.strict_view_stability
                and p_original_species >= 0.80
            ):
                audit_status = "hard_label_retained"
                review_priority = 0
                grade_weight = res_factor
                soft_target = torch.full((2,), args.label_smoothing)
                soft_target[original_grade] = 1.0 - args.label_smoothing
            else:
                audit_status = "soft_label_low_confidence"
                review_priority = 0
                evidence = max(0.0, min(1.0, (p_original_grade - 0.50) / 0.25))
                grade_weight = res_factor * min(0.75, 0.25 + 0.50 * evidence)
                human_target = torch.zeros(2)
                human_target[original_grade] = 1.0
                soft_target = 0.70 * human_target + 0.30 * mean_grade
                soft_target /= soft_target.sum()
                reasons.append("soft_human_oof_blend")

            status_counter[audit_status] += 1
            priority_counter[str(review_priority)] += 1
            audited = {
                **row,
                # Preserve the original label.  There is intentionally no
                # effective/relabelled class field in this audit output.
                "species_weight": f"{species_weight:.4f}",
                "grade_weight": f"{grade_weight:.4f}",
                "grade_soft_target": grade_target_text(soft_target),
                "audit_status": audit_status,
                "review_priority": str(review_priority),
                "audit_reasons": ";".join(reasons),
                "resolution_factor": f"{res_factor:.4f}",
                "object_width": str(width),
                "object_height": str(height),
                "object_max_dimension": str(max_dimension),
                "oof_fold": str(fold_id),
                "oof_p_original_species": rounded_probability(p_original_species),
                "oof_p_good": rounded_probability(float(mean_grade[0])),
                "oof_p_bad": rounded_probability(float(mean_grade[1])),
                "oof_p_original_grade": rounded_probability(p_original_grade),
                "oof_species_vote_agreement": rounded_probability(species_vote_agreement),
                "oof_grade_vote_agreement": rounded_probability(grade_vote_agreement),
                "oof_species_view_stability": rounded_probability(species_view_stability),
                "oof_grade_view_stability": rounded_probability(grade_view_stability),
                "oof_model_count": str(args.models_per_fold),
                "group_excluded_knn_grade": str(int(knn_prediction[global_index])),
                "group_excluded_knn_margin": rounded_probability(float(knn_margin[global_index])),
            }
            output_rows[global_index] = audited

    if any(row is None for row in output_rows):
        raise AssertionError("some train rows did not receive an OOF audit")
    final_rows = [row for row in output_rows if row is not None]
    summary = {
        "train_samples": len(rows),
        "train_groups": len({row["group_id"] for row in rows}),
        "folds": len(folds),
        "models_per_fold": args.models_per_fold,
        "status_counts": dict(status_counter),
        "review_priority_counts": dict(priority_counter),
        "resolution_counts": dict(resolution_counter),
        "oof_human_species_agreement": human_species_matches / len(rows),
        "oof_human_grade_agreement": human_grade_matches / len(rows),
        "grade_supervised_count": sum(float(row["grade_weight"]) > 0 for row in final_rows),
        "grade_unlabelled_count": sum(float(row["grade_weight"]) == 0 for row in final_rows),
        "species_supervised_weight_sum": sum(float(row["species_weight"]) for row in final_rows),
        "grade_supervised_weight_sum": sum(float(row["grade_weight"]) for row in final_rows),
        "fold_train_sample_counts": [len(rows) - len(fold) for fold in folds],
        "fold_held_out_sample_counts": [len(fold) for fold in folds],
        "fold_held_out_group_counts": [
            len({rows[index]["group_id"] for index in fold}) for fold in folds
        ],
        "group_to_fold": dict(sorted(group_to_fold.items())),
    }
    return final_rows, summary


def safe_prepare_output(output: Path, source_manifest: Path, force: bool) -> None:
    source_root = source_manifest.resolve().parent
    resolved_output = output.resolve()
    if resolved_output == source_root or source_root in resolved_output.parents:
        # The intended v4 sibling is safe; a descendant of v3 is not.
        if resolved_output == source_root or source_root.name == "pepper_ssl_v3":
            raise ValueError(f"refusing to write audit output inside v3: {resolved_output}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists: {output}; pass --force to replace it")
        marker = output / "audit_summary.json"
        if not marker.is_file():
            raise ValueError(
                f"refusing to delete unrecognized output directory {output}; "
                "audit_summary.json is missing"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ambiguous_low < args.ambiguous_high < 1.0:
        raise ValueError("invalid ambiguity interval")
    safe_prepare_output(args.output, args.manifest, args.force)
    all_rows, train_rows = read_manifest(args.manifest)
    groups = validate_train_rows(train_rows)
    feature_blocks: list[torch.Tensor] = []
    expected_paths = [str(Path(row["path"]).resolve()) for row in train_rows]
    expected_views: int | None = None
    for feature_path in args.features:
        features_payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        block = features_payload["features"].float()
        if block.ndim != 3 or len(block) != len(train_rows):
            raise ValueError(
                f"feature shape {tuple(block.shape)} from {feature_path} does not match "
                f"{len(train_rows)} train rows"
            )
        if expected_views is None:
            expected_views = block.shape[1]
        elif block.shape[1] != expected_views:
            raise ValueError(
                f"feature view mismatch: {feature_path} has {block.shape[1]}, "
                f"expected {expected_views}"
            )
        payload_paths = features_payload.get("paths")
        if payload_paths is not None and list(payload_paths) != expected_paths:
            raise ValueError(f"feature path order does not match manifest: {feature_path}")
        # Equalize feature-family scale before concatenation.  Multiplying by
        # sqrt(dim) keeps per-coordinate variance numerically useful to the
        # following LayerNorm while preventing a max-pooled block from
        # overwhelming another backbone.
        block = F.normalize(block, dim=-1) * (block.shape[-1] ** 0.5)
        feature_blocks.append(block)
    features = torch.cat(feature_blocks, dim=-1)

    device = choose_device(args.device)
    folds, group_to_fold = stratified_group_kfold(groups, args.folds, args.seed)
    all_indices = set(range(len(train_rows)))
    model_outputs: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
    for fold_id, held_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(held_indices))
        held_groups = {train_rows[index]["group_id"] for index in held_indices}
        fit_groups = {train_rows[index]["group_id"] for index in train_indices}
        if held_groups & fit_groups:
            raise AssertionError("group leakage detected in OOF split")
        print(
            f"fold {fold_id + 1}/{args.folds}: fit={len(train_indices)} samples/"
            f"{len(fit_groups)} groups, audit={len(held_indices)} samples/"
            f"{len(held_groups)} groups"
        )
        for model_index in range(args.models_per_fold):
            model_seed = args.seed + fold_id * 10_007 + model_index * 997
            model = train_oof_head(
                features, train_rows, train_indices, args, model_seed, device
            )
            model_outputs[fold_id].append(
                predict_fold(model, features, train_rows, held_indices, device)
            )
            print(f"  OOF model {model_index + 1}/{args.models_per_fold} complete")

    knn_prediction, knn_margin = group_excluded_knn(features, train_rows)
    audited_train, summary = audit_train_rows(
        train_rows,
        folds,
        group_to_fold,
        model_outputs,
        knn_prediction,
        knn_margin,
        args,
    )

    base_fields = list(all_rows[0].keys())
    output_fields = base_fields + [field for field in AUDIT_FIELDS if field not in base_fields]
    train_by_path = {row["path"]: row for row in audited_train}
    combined_rows: list[dict[str, str]] = []
    for row in all_rows:
        if row.get("split") == "train":
            combined_rows.append(train_by_path[row["path"]])
        else:
            # Blind passthrough: do not inspect or derive anything from a
            # validation/test label.  Empty weights prevent accidental use in
            # the training-label audit.
            combined_rows.append(
                {
                    **row,
                    **{field: "" for field in AUDIT_FIELDS},
                    "audit_status": "held_out_untouched",
                    "review_priority": "",
                }
            )

    write_csv(args.output / "train_label_audit.csv", audited_train, output_fields)
    write_csv(args.output / "manifest.csv", combined_rows, output_fields)
    summary.update(
        {
            "source_manifest": str(args.manifest.resolve()),
            "source_features": [str(path.resolve()) for path in args.features],
            "output_manifest": str((args.output / "manifest.csv").resolve()),
            "train_audit_manifest": str((args.output / "train_label_audit.csv").resolve()),
            "guardrails": {
                "validation_labels_used_for_audit": False,
                "test_labels_used_for_audit": False,
                "existing_head_used_for_oof": False,
                "automatic_hard_label_flips": 0,
                "group_leakage": False,
            },
            "thresholds": {
                "ambiguous_grade_interval": [args.ambiguous_low, args.ambiguous_high],
                "hard_keep_probability": args.hard_keep_probability,
                "strict_opposite_probability": args.strict_opposite_probability,
                "strict_species_probability": args.strict_species_probability,
                "strict_view_stability": args.strict_view_stability,
                "knn_margin": args.knn_margin,
            },
        }
    )
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
