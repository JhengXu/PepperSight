#!/usr/bin/env python3
"""Retrain only the hierarchical pepper classification head with robust SSL.

The YOLO11 feature extractor is frozen.  Weak/strong views are embedded once,
then an EMA teacher and student head are trained with FixMatch-style consistency,
FreeMatch-style class-adaptive thresholds, and SoftMatch confidence weights.
Only ambiguous 40%-60% grade predictions that pass two-model/multi-view
agreement are relabelled. Validation and test labels are never changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


SPECIES_NAMES = ("子弹头", "条子")
GRADE_NAMES = ("好", "差")
FINAL_NAMES = ("子弹头_好", "子弹头_差", "条子_好", "条子_差")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v3/manifest.csv"),
    )
    parser.add_argument("--backbone", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument(
        "--initial-model", type=Path,
        default=Path(
            "yolo11_pepper/runs/hierarchical_combined_single_relabelled_v2_"
            "kornia_hybrid/best_hierarchical_kornia_calibrated.pt"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("yolo11_pepper/runs/hierarchical_ssl_v3"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-views", type=int, default=5)
    parser.add_argument("--embedding-batch", type=int, default=32)
    parser.add_argument("--head-batch", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.992)
    parser.add_argument("--ssl-weight", type=float, default=0.65)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "path": Path(row["path"]),
                    "class_id": int(row["class_id"]),
                    "original_class_id": int(row["original_class_id"]),
                }
            )
    return rows


def rgba_object(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.nonzero(alpha >= 4)
    if len(xs):
        rgba = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return rgba


def render_view(path: Path, size: int, view: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    with Image.open(path) as source:
        pepper = rgba_object(source)
    if view == 0:
        background = (64, 68, 68)
        scale = 0.86
        angle = 0.0
        offset_x = offset_y = 0
    else:
        palettes = ((30, 32, 34), (62, 66, 67), (105, 101, 94), (180, 178, 170), (225, 224, 218))
        base = rng.choice(palettes)
        background = tuple(max(0, min(255, channel + rng.randint(-12, 12))) for channel in base)
        scale = rng.uniform(0.68, 0.94)
        angle = rng.uniform(-18, 18)
        offset_x = rng.randint(-round(size * 0.07), round(size * 0.07))
        offset_y = rng.randint(-round(size * 0.07), round(size * 0.07))
        if rng.random() < 0.5:
            pepper = ImageOps.mirror(pepper)
        if rng.random() < 0.18:
            pepper = ImageOps.flip(pepper)
        pepper = ImageEnhance.Brightness(pepper).enhance(rng.uniform(0.88, 1.12))
        pepper = ImageEnhance.Contrast(pepper).enhance(rng.uniform(0.90, 1.12))
        pepper = ImageEnhance.Color(pepper).enhance(rng.uniform(0.92, 1.08))
    pepper.thumbnail((max(1, round(size * scale)), max(1, round(size * scale))), Image.Resampling.LANCZOS)
    if angle:
        pepper = pepper.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (size, size), (*background, 255))
    x = (size - pepper.width) // 2 + offset_x
    y = (size - pepper.height) // 2 + offset_y
    canvas.alpha_composite(pepper, (x, y))
    rgb = canvas.convert("RGB")
    if view > 0:
        if rng.random() < 0.18:
            rgb = rgb.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.8)))
        rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.94, 1.06))
        rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.94, 1.08))
    array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array.copy())


class ViewDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], size: int, views: int, seed: int) -> None:
        self.rows, self.size, self.views, self.seed = rows, size, views, seed

    def __len__(self) -> int:
        return len(self.rows) * self.views

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row_index, view = divmod(index, self.views)
        path = Path(self.rows[row_index]["path"])
        view_seed = self.seed + row_index * 100_003 + view * 10_007
        return render_view(path, self.size, view, view_seed), row_index, view


class FrozenYOLOBackbone(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        yolo = YOLO(checkpoint.resolve())
        self.layers = nn.ModuleList(list(yolo.model.model[:11]))
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature = image
            for layer in self.layers:
                feature = layer(feature)
            return F.adaptive_avg_pool2d(feature, 1).flatten(1)


class HierarchicalHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 128), nn.SiLU(), nn.Dropout(0.15)
        )
        self.species_head = nn.Linear(128, 2)
        self.grade_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 2)),
                nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 2)),
            ]
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        return self.species_head(shared), torch.stack([head(shared) for head in self.grade_heads], 1)


def extract_embeddings(
    backbone: FrozenYOLOBackbone,
    rows: list[dict[str, object]],
    size: int,
    views: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = ViewDataset(rows, size, views, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features: list[torch.Tensor] = []
    backbone.eval()
    with torch.no_grad():
        for batch_index, (images, _, _) in enumerate(loader, 1):
            features.append(backbone(images.to(device)).cpu())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"embedding {batch_index}/{len(loader)}")
    return torch.cat(features).reshape(len(rows), views, -1)


def joint_log_probs(
    species_logits: torch.Tensor,
    grade_logits: torch.Tensor,
    temperatures: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperatures is None:
        temperatures = torch.ones(3, device=species_logits.device)
    species_log = F.log_softmax(species_logits / temperatures[0], dim=1)
    grade_scaled = torch.stack(
        [grade_logits[:, species] / temperatures[species + 1] for species in range(2)], dim=1
    )
    grade_log = F.log_softmax(grade_scaled, dim=2)
    return (species_log.unsqueeze(2) + grade_log).reshape(-1, 4)


def model_view_probabilities(head: HierarchicalHead, features: torch.Tensor, device: torch.device):
    head.eval()
    joint, species, grade = [], [], []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            batch = features[start : start + 256].to(device)
            shape = batch.shape
            species_logits, grade_logits = head(batch.reshape(-1, shape[-1]))
            joint.append(joint_log_probs(species_logits, grade_logits).exp().reshape(shape[0], shape[1], 4).cpu())
            species.append(species_logits.softmax(1).reshape(shape[0], shape[1], 2).cpu())
            grade.append(grade_logits.softmax(2).reshape(shape[0], shape[1], 2, 2).cpu())
    return torch.cat(joint), torch.cat(species), torch.cat(grade)


def class_weights(labels: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=4).float().clamp_min(1)
    weights = (counts.sum() / (4 * counts)).pow(power)
    return weights / weights.mean()


def supervised_loss(
    head: HierarchicalHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    joint_class_weights: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    species_logits, grade_logits = head(features)
    species = labels // 2
    grade = labels % 2
    species_loss = F.cross_entropy(species_logits, species, reduction="none", label_smoothing=smoothing)
    selected_grade = grade_logits[torch.arange(len(labels), device=labels.device), species]
    grade_loss = F.cross_entropy(selected_grade, grade, reduction="none", label_smoothing=smoothing)
    weights = sample_weights * joint_class_weights[labels]
    return ((species_loss + grade_loss) * weights).sum() / weights.sum().clamp_min(1e-6)


def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)


def metrics(labels: torch.Tensor, probabilities: torch.Tensor) -> dict[str, object]:
    prediction = probabilities.argmax(1)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for truth, predicted in zip(labels.tolist(), prediction.tolist()):
        confusion[truth, predicted] += 1
    per_class = []
    for class_id, name in enumerate(FINAL_NAMES):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class_id": class_id, "class_name": name,
                "precision": precision, "recall": recall, "f1": f1,
                "support": int(confusion[class_id].sum()),
            }
        )
    confidence = probabilities.max(1).values
    correct = prediction == labels
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-7)
        if mask.any():
            ece += float(mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs())
    nll = float(F.nll_loss(probabilities.clamp_min(1e-9).log(), labels))
    return {
        "samples": len(labels),
        "accuracy": float(correct.float().mean()),
        "macro_precision": float(np.mean([item["precision"] for item in per_class])),
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "nll": nll,
        "ece": ece,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def probabilities_for_features(
    head: HierarchicalHead,
    features: torch.Tensor,
    device: torch.device,
    temperatures: torch.Tensor | None = None,
) -> torch.Tensor:
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            species_logits, grade_logits = head(features[start : start + 512].to(device))
            local_temperatures = temperatures.to(device) if temperatures is not None else None
            outputs.append(joint_log_probs(species_logits, grade_logits, local_temperatures).exp().cpu())
    return torch.cat(outputs)


def train_head(
    initial: HierarchicalHead,
    train_features: torch.Tensor,
    labels: torch.Tensor,
    label_weights: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    args: argparse.Namespace,
    epochs: int,
    seed: int,
    select_best: bool,
) -> tuple[HierarchicalHead, list[dict[str, object]]]:
    torch.manual_seed(seed)
    student = copy.deepcopy(initial).to(args.device_object)
    teacher = copy.deepcopy(initial).to(args.device_object)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    weights = class_weights(labels, getattr(args, "class_weight_power", 1.0)).to(args.device_object)
    moving_distribution = torch.full((4,), 0.25, device=args.device_object)
    moving_class_conf = torch.full((4,), 0.75, device=args.device_object)
    global_conf = torch.tensor(0.75, device=args.device_object)
    generator = torch.Generator().manual_seed(seed)
    best_score, stale, best_state = -1.0, 0, copy.deepcopy(teacher.state_dict())
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(labels), generator=generator)
        running_loss, running_supervised, running_ssl, used = 0.0, 0.0, 0.0, 0
        student.train()
        for start in range(0, len(order), args.head_batch):
            indices = order[start : start + args.head_batch]
            weak = train_features[indices, 0].to(args.device_object)
            strong_views = torch.randint(1, train_features.shape[1], (len(indices),), generator=generator)
            strong = train_features[indices, strong_views].to(args.device_object)
            batch_labels = labels[indices].to(args.device_object)
            batch_weights = label_weights[indices].to(args.device_object)
            loss_supervised = supervised_loss(
                student, weak, batch_labels, batch_weights, weights, args.label_smoothing
            )
            with torch.no_grad():
                teacher_species, teacher_grade = teacher(weak)
                raw_probability = joint_log_probs(teacher_species, teacher_grade).exp()
                moving_distribution.mul_(0.98).add_(raw_probability.mean(0), alpha=0.02)
                aligned = raw_probability * (0.25 / moving_distribution.clamp_min(1e-5))
                aligned /= aligned.sum(1, keepdim=True)
                confidence, pseudo_label = aligned.max(1)
                global_conf.mul_(0.98).add_(confidence.mean(), alpha=0.02)
                for class_id in range(4):
                    class_mask = pseudo_label == class_id
                    if class_mask.any():
                        moving_class_conf[class_id].mul_(0.98).add_(confidence[class_mask].mean(), alpha=0.02)
                thresholds = (global_conf * moving_class_conf / moving_class_conf.max()).clamp(0.62, 0.92)
                if epoch <= 5:
                    thresholds = thresholds.clamp_min(0.84)
                selected_threshold = thresholds[pseudo_label]
                mu = confidence.mean()
                sigma = confidence.std().clamp_min(0.06)
                soft_weight = torch.where(
                    confidence >= mu,
                    torch.ones_like(confidence),
                    torch.exp(-((confidence - mu) ** 2) / (2 * sigma**2)),
                )
                pseudo_weight = torch.where(
                    confidence >= selected_threshold,
                    torch.ones_like(confidence),
                    soft_weight * 0.22,
                )
                pseudo_weight *= (confidence >= 0.50).float()
            student_species, student_grade = student(strong)
            ssl_each = F.nll_loss(
                joint_log_probs(student_species, student_grade), pseudo_label, reduction="none"
            )
            loss_ssl = (ssl_each * pseudo_weight).sum() / pseudo_weight.sum().clamp_min(1.0)
            ramp = min(1.0, max(0.0, (epoch - 2) / 8))
            loss = loss_supervised + args.ssl_weight * ramp * loss_ssl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            ema_update(teacher, student, args.ema_decay)
            count = len(indices)
            running_loss += float(loss.detach()) * count
            running_supervised += float(loss_supervised.detach()) * count
            running_ssl += float(loss_ssl.detach()) * count
            used += count
        scheduler.step()
        val_probability = probabilities_for_features(teacher, val_features, args.device_object)
        validation = metrics(val_labels, val_probability)
        row = {
            "epoch": epoch,
            "loss": running_loss / used,
            "supervised_loss": running_supervised / used,
            "ssl_loss": running_ssl / used,
            "lr": scheduler.get_last_lr()[0],
            "adaptive_thresholds": [float(value) for value in thresholds.cpu()],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        if not getattr(args, "quiet", False) and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(json.dumps(row, ensure_ascii=False))
        score = float(validation["macro_f1"])
        if score > best_score + 1e-6:
            best_score, stale, best_state = score, 0, copy.deepcopy(teacher.state_dict())
        else:
            stale += 1
        if select_best and stale >= args.patience:
            print(f"early stop at epoch {epoch}, best val macro-F1={best_score:.4f}")
            break
    teacher.load_state_dict(best_state if select_best else teacher.state_dict())
    return teacher.cpu(), history


def initial_reliability(
    rows: list[dict[str, object]], head: HierarchicalHead, features: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, object]]:
    joint, _, _ = model_view_probabilities(head.to(device), features, device)
    mean = joint.mean(1)
    labels = torch.tensor([int(row["class_id"]) for row in rows])
    prediction = mean.argmax(1)
    stability = (joint.argmax(2) == prediction[:, None]).float().mean(1)
    label_probability = mean[torch.arange(len(rows)), labels]
    reliable = (prediction == labels) & (stability >= 0.8) & (label_probability >= 0.52)
    per_class = {FINAL_NAMES[c]: int((reliable & (labels == c)).sum()) for c in range(4)}
    return reliable.float(), {
        "reliable": int(reliable.sum()), "total": len(rows), "per_class": per_class,
        "mean_view_stability": float(stability.mean()),
    }


def relabel_training_rows(
    rows: list[dict[str, object]],
    features: torch.Tensor,
    head_a: HierarchicalHead,
    head_b: HierarchicalHead,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]], dict[str, object]]:
    outputs = [model_view_probabilities(head.to(device), features, device) for head in (head_a, head_b)]
    labels, weights, decisions = [], [], []
    counters = Counter()
    for index, row in enumerate(rows):
        original = int(row["class_id"])
        species_id, original_grade = divmod(original, 2)
        p_good_models, grade_predictions, stabilities, species_probs = [], [], [], []
        for _, species_probability, grade_probability in outputs:
            grade_views = grade_probability[index, :, species_id]
            mean_grade = grade_views.mean(0)
            p_good_models.append(float(mean_grade[0]))
            grade_prediction = int(mean_grade.argmax())
            grade_predictions.append(grade_prediction)
            stabilities.append(float((grade_views.argmax(1) == grade_prediction).float().mean()))
            species_probs.append(float(species_probability[index, :, species_id].mean()))
        p_good = float(np.mean(p_good_models))
        predicted_grade = 0 if p_good >= 0.5 else 1
        ambiguous = 0.40 <= p_good <= 0.60
        disagreement = predicted_grade != original_grade
        dual_agreement = grade_predictions[0] == grade_predictions[1] == predicted_grade
        stable = min(stabilities) >= 0.75
        species_consistent = min(species_probs) >= 0.60
        accepted = ambiguous and disagreement and dual_agreement and stable and species_consistent
        if accepted:
            effective = species_id * 2 + predicted_grade
            weight, state = 0.65, "auto_relabelled_40_60"
            counters["accepted"] += 1
        elif ambiguous and disagreement:
            effective = original
            weight, state = 0.0, "ambiguous_unlabelled"
            counters["ambiguous_unlabelled"] += 1
        elif disagreement and dual_agreement:
            effective = original
            weight, state = 0.30, "model_disagrees_downweighted"
            counters["downweighted"] += 1
        else:
            effective = original
            weight, state = 1.0, "retained"
            counters["retained"] += 1
        labels.append(effective)
        weights.append(weight)
        decisions.append(
            {
                "path": str(row["path"]), "group_id": row["group_id"],
                "original_class_id": original, "original_class_name": FINAL_NAMES[original],
                "effective_class_id": effective, "effective_class_name": FINAL_NAMES[effective],
                "p_good": p_good, "p_bad": 1 - p_good,
                "ambiguous_40_60": ambiguous, "models_agree": dual_agreement,
                "min_view_stability": min(stabilities), "min_species_probability": min(species_probs),
                "label_state": state, "label_weight": weight,
            }
        )
    return torch.tensor(labels), torch.tensor(weights), decisions, dict(counters)


def fit_temperatures(
    head: HierarchicalHead, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    head = head.to(device).eval()
    with torch.no_grad():
        species_logits, grade_logits = head(features.to(device))
    log_temperatures = nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.LBFGS([log_temperatures], lr=0.15, max_iter=120, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperatures = log_temperatures.exp().clamp(0.05, 10.0)
        loss = F.nll_loss(joint_log_probs(species_logits, grade_logits, temperatures), labels.to(device))
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperatures.detach().exp().clamp(0.05, 10.0).cpu()


def evaluate_model(
    head: HierarchicalHead,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    test_rows: list[dict[str, object]],
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor]:
    temperatures = fit_temperatures(head, val_features, val_labels, device)
    raw_val = probabilities_for_features(head.to(device), val_features, device)
    calibrated_val = probabilities_for_features(head, val_features, device, temperatures)
    raw_test = probabilities_for_features(head, test_features, device)
    calibrated_test = probabilities_for_features(head, test_features, device, temperatures)
    report: dict[str, object] = {
        "temperatures": {
            "species": float(temperatures[0]),
            "grade_given_子弹头": float(temperatures[1]),
            "grade_given_条子": float(temperatures[2]),
        },
        "validation_raw": metrics(val_labels, raw_val),
        "validation_calibrated": metrics(val_labels, calibrated_val),
        "test_raw": metrics(test_labels, raw_test),
        "test_calibrated": metrics(test_labels, calibrated_test),
    }
    by_origin = {}
    for origin in ("transparent", "scene"):
        mask = torch.tensor([row["origin"] == origin for row in test_rows])
        by_origin[origin] = metrics(test_labels[mask], calibrated_test[mask])
    report["test_calibrated_by_origin"] = by_origin
    return report, temperatures


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device_object = choose_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = read_manifest(args.manifest.resolve())
    rows = {split: [row for row in all_rows if row["split"] == split] for split in ("train", "val", "test")}
    print(f"device={args.device_object} split_sizes={{k: len(v) for k,v in rows.items()}}")
    backbone = FrozenYOLOBackbone(args.backbone).to(args.device_object)
    embeddings: dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        cache = output / f"{split}_features.pt"
        views = args.train_views if split == "train" else 1
        if cache.exists():
            embeddings[split] = torch.load(cache, map_location="cpu", weights_only=True)["features"]
        else:
            print(f"extracting {split}: {len(rows[split])} images x {views} views")
            embeddings[split] = extract_embeddings(
                backbone, rows[split], args.image_size, views, args.embedding_batch,
                args.seed + {"train": 0, "val": 1, "test": 2}[split] * 1_000_003,
                args.device_object,
            )
            torch.save({"features": embeddings[split]}, cache)
    del backbone
    labels = {split: torch.tensor([int(row["class_id"]) for row in rows[split]]) for split in rows}

    initial_payload = torch.load(args.initial_model.resolve(), map_location="cpu", weights_only=True)
    initial = HierarchicalHead(int(initial_payload["feature_dim"]))
    initial.load_state_dict(initial_payload["head_state_dict"])
    initial_report, _ = evaluate_model(
        copy.deepcopy(initial), embeddings["val"][:, 0], labels["val"],
        embeddings["test"][:, 0], labels["test"], rows["test"], args.device_object,
    )
    reliable_weights, reliability_report = initial_reliability(
        rows["train"], copy.deepcopy(initial), embeddings["train"], args.device_object
    )
    print("initial reliability", json.dumps(reliability_report, ensure_ascii=False))

    warm_a, history_a = train_head(
        initial, embeddings["train"], labels["train"], reliable_weights,
        embeddings["val"][:, 0], labels["val"], args, args.warmup_epochs,
        args.seed + 11, select_best=False,
    )
    initial_b = copy.deepcopy(initial)
    torch.manual_seed(args.seed + 29)
    with torch.no_grad():
        for parameter in initial_b.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.006)
    warm_b, history_b = train_head(
        initial_b, embeddings["train"], labels["train"], reliable_weights,
        embeddings["val"][:, 0], labels["val"], args, args.warmup_epochs,
        args.seed + 37, select_best=False,
    )
    effective_labels, effective_weights, decisions, relabel_summary = relabel_training_rows(
        rows["train"], embeddings["train"], warm_a, warm_b, args.device_object
    )
    relabel_fields = list(decisions[0])
    with (output / "relabel_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=relabel_fields)
        writer.writeheader()
        writer.writerows(decisions)
    print("relabel", json.dumps(relabel_summary, ensure_ascii=False))

    final_head, final_history = train_head(
        warm_a, embeddings["train"], effective_labels, effective_weights,
        embeddings["val"][:, 0], labels["val"], args, args.epochs,
        args.seed + 101, select_best=True,
    )
    new_report, temperatures = evaluate_model(
        final_head, embeddings["val"][:, 0], labels["val"],
        embeddings["test"][:, 0], labels["test"], rows["test"], args.device_object,
    )
    checkpoint = output / "best_hierarchical_ssl_calibrated.pt"
    torch.save(
        {
            "head_state_dict": final_head.state_dict(),
            "feature_dim": embeddings["train"].shape[-1],
            "backbone_checkpoint": str(args.backbone.resolve()),
            "image_size": args.image_size,
            "species_names": SPECIES_NAMES,
            "grade_names": GRADE_NAMES,
            "final_names": FINAL_NAMES,
            "joint_temperatures": tuple(float(value) for value in temperatures),
            "recommended_inference": "hierarchical temperature calibration then joint Bayes argmax",
            "training": {
                "backbone": "frozen YOLO11 layers 0:11",
                "head_only": True,
                "ssl": "EMA teacher + FixMatch weak/strong consistency + adaptive class thresholds + SoftMatch weights",
                "relabel_summary": relabel_summary,
            },
            "strict_test_metrics": new_report["test_calibrated"],
        },
        checkpoint,
    )
    result = {
        "architecture": "frozen YOLO11 backbone -> species head -> two species-conditional grade heads -> joint Bayes argmax",
        "dataset": str(args.manifest.resolve()),
        "split_sizes": {split: len(rows[split]) for split in rows},
        "trainable_parameters": sum(parameter.numel() for parameter in final_head.parameters()),
        "frozen_backbone": True,
        "initial_reliability": reliability_report,
        "relabel_summary": relabel_summary,
        "baseline_existing_head": initial_report,
        "new_ssl_head": new_report,
        "checkpoint": str(checkpoint),
        "histories": {"warmup_a": history_a, "warmup_b": history_b, "final": final_history},
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "histories"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
