from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Avoid a native OpenMP collision when PyTorch and XGBoost share the macOS
# inference process. The classifier is tiny, so one XGBoost thread is ample.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image

from app.services.camera_candidate_gate import (
    CameraGateConfig,
    RedComponentConfig,
    gate_camera_proposals,
    merge_detector_and_component_proposals,
    proposal_median_red_saturation,
    red_component_proposals,
    white_detection_surface_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DETECTOR = (
    PROJECT_ROOT
    / "runs"
    / "yolo11n_pepper_strict_v5_f4"
    / "weights"
    / "best.pt"
)
DEFAULT_CLASSIFIER = (
    PROJECT_ROOT
    / "runs"
    / "hierarchical_v5_clean"
    / "best_hierarchical_clean_v5_reaudit_svm.joblib"
)
DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "runs"
    / "hierarchical_v5_clean"
    / "selection_reaudit.json"
)
DEFAULT_BACKBONE = PROJECT_ROOT.parent / "yolo11n.pt"
DEFAULT_CLS_BACKBONE = PROJECT_ROOT.parent / "yolo11n-cls.pt"
DEFAULT_LEGACY_PROPOSAL_DETECTOR = (
    PROJECT_ROOT / "runs" / "yolo11n_pepper" / "weights" / "best.pt"
)
DEFAULT_DETECTOR_SHA256 = "caf31c3ad8ed267ad6af94c8cdd8d279bb94beba46802cab2ea67846a2322f35"
LEGACY_V4_SELECTION_SHA256 = "5019a8e679493c58566c217a7ca3defec07d6c7f91397f6f45b85f99ca325480"
LEGACY_V4_DETECTOR_SHA256 = "f12193ee787b7d56dd6e06422d7b6069d179d1650e9f6ae0e957cb6b148faead"
IMAGENET_CLS_SHA256 = "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7"
SUPPORTED_SVM_SCHEMAS = {
    "pepper-clean-v5-validation-selection-v1",
    "pepper-v4-svm-selection-v1",
}
SPECIES_NAMES = ("子弹头", "条子")
GRADE_NAMES = ("一级", "二级")
ASCII_LABELS = (
    "ZIDANTOU LEVEL 1",
    "ZIDANTOU LEVEL 2",
    "TIAOZI LEVEL 1",
    "TIAOZI LEVEL 2",
)


class NoPepperDetectedError(RuntimeError):
    """Raised when the detector finds no pepper in the conveyor trigger zone."""


@dataclass(frozen=True)
class PepperPrediction:
    species: str
    grade: str
    label: str
    species_confidence: float
    grade_confidence: float
    joint_confidence: float
    good_probability: float
    bad_probability: float
    detector_confidence: float
    bbox: tuple[int, int, int, int]
    sharpness: float
    processing_time_ms: float
    annotated_frame: Any


class HierarchicalHeadProxy:
    """p(species) head plus one p(grade|species) head for each species."""

    @staticmethod
    def build(torch, nn, feature_dim: int):
        class HierarchicalHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, 128),
                    nn.SiLU(),
                    nn.Dropout(0.15),
                )
                self.species_head = nn.Linear(128, 2)
                self.grade_heads = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 2)),
                        nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 2)),
                    ]
                )

            def forward(self, feature):
                shared = self.shared(feature)
                species_logits = self.species_head(shared)
                grade_logits = torch.stack([head(shared) for head in self.grade_heads], dim=1)
                return species_logits, grade_logits

        return HierarchicalHead()


class PepperModelService:
    """YOLO detector plus leakage-safe hierarchical pepper classification."""

    def __init__(self) -> None:
        self.detector_path = Path(os.getenv("QJ_DETECTOR_MODEL", DEFAULT_DETECTOR)).resolve()
        self.classifier_path = Path(os.getenv("QJ_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER)).resolve()
        self.selection_path = Path(
            os.getenv("QJ_CLASSIFIER_SELECTION", DEFAULT_SELECTION)
        ).resolve()
        self.backbone_path = Path(
            os.getenv("QJ_CLASSIFIER_BACKBONE", DEFAULT_BACKBONE)
        ).resolve()
        self.cls_backbone_path = Path(
            os.getenv("QJ_CLASSIFIER_CLS_BACKBONE", DEFAULT_CLS_BACKBONE)
        ).resolve()
        self.detector_confidence = float(os.getenv("QJ_DETECTOR_CONF", "0.35"))
        self.detector_iou = float(os.getenv("QJ_DETECTOR_IOU", "0.45"))
        # Optional camera-domain recall experiment. It is off by default because
        # the immutable holdout exposed hand false positives after colour gating.
        self.enable_legacy_red_gate = os.getenv("QJ_ENABLE_LEGACY_RED_GATE", "0") == "1"
        self.enable_red_component_fallback = (
            os.getenv("QJ_ENABLE_RED_COMPONENT_FALLBACK", "0") == "1"
        )
        self.legacy_proposal_path = Path(
            os.getenv("QJ_LEGACY_PROPOSAL_MODEL", DEFAULT_LEGACY_PROPOSAL_DETECTOR)
        ).resolve()
        default_camera_gate = CameraGateConfig()
        self._camera_gate_config = CameraGateConfig(
            red_core_minimum=float(
                os.getenv(
                    "QJ_CAMERA_GATE_RED_CORE_MINIMUM",
                    str(default_camera_gate.red_core_minimum),
                )
            )
        )
        self.min_elongation = float(os.getenv("QJ_MIN_PEPPER_ELONGATION", "1.30"))
        self.reject_frame_border_proposals = (
            os.getenv("QJ_REJECT_FRAME_BORDER_PROPOSALS", "1") == "1"
        )
        self.frame_border_margin = max(
            0, int(os.getenv("QJ_FRAME_BORDER_MARGIN", "2"))
        )
        self.max_peppers = int(os.getenv("QJ_MAX_PEPPERS", "12"))
        # Production inference must fail closed. Classifying an arbitrary center
        # crop when YOLO finds nothing turns people and desk objects into peppers.
        self.allow_roi_fallback = os.getenv("QJ_ALLOW_ROI_FALLBACK", "0") == "1"
        self._lock = threading.Lock()
        self._loaded = False
        self._loading = False
        self._error = "模型尚未加载"
        self._device = "cpu"
        self._torch = None
        self._detector = None
        self._legacy_proposal_detector = None
        self._backbone_layers = None
        self._pepper_feature_layers = None
        self._cls_feature_layers = None
        self._cls_projection = None
        self._head = None
        suffix = self.classifier_path.suffix.lower()
        self._head_type = {
            ".joblib": "hierarchical_svm",
            ".pkl": "hierarchical_svm",
            ".ubj": "xgboost_direct_4class",
        }.get(suffix, "hierarchical_neural")
        self._xgboost = None
        self._image_size = 224
        self._thresholds = (0.5, 0.5)
        self._joint_temperatures = (1.0, 1.0, 1.0)
        self._selection_id: str | None = None
        self._selection_schema: str | None = None
        self._selection_sha256: str | None = None
        self._classifier_sha256: str | None = None
        self._detector_sha256: str | None = None
        self._feature_families: tuple[str, ...] = ()
        self._validation_metrics: dict[str, Any] | None = None
        self._provenance: dict[str, Any] | None = None

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loading = True
            try:
                import torch
                from torch import nn
                from ultralytics import YOLO

                if not self.detector_path.exists():
                    raise FileNotFoundError(f"检测模型不存在：{self.detector_path}")
                if not self.classifier_path.exists():
                    raise FileNotFoundError(f"分类模型不存在：{self.classifier_path}")
                self._detector_sha256 = self._validate_detector_checkpoint()
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                self._torch = torch
                self._detector = YOLO(self.detector_path)
                if self.enable_legacy_red_gate:
                    if not self.legacy_proposal_path.exists():
                        raise FileNotFoundError(
                            f"旧版候选检测模型不存在：{self.legacy_proposal_path}"
                        )
                    actual_legacy_hash = self._sha256_file(self.legacy_proposal_path)
                    expected_legacy_hash = os.getenv(
                        "QJ_LEGACY_PROPOSAL_SHA256", LEGACY_V4_DETECTOR_SHA256
                    ).strip().lower()
                    if actual_legacy_hash != expected_legacy_hash:
                        raise ValueError("旧版候选检测模型 SHA256 不匹配，已拒绝加载")
                    self._legacy_proposal_detector = YOLO(self.legacy_proposal_path)
                suffix = self.classifier_path.suffix.lower()
                if suffix in {".joblib", ".pkl"}:
                    self._load_hierarchical_svm(torch, nn, YOLO)
                    self._loaded = True
                    self._error = ""
                    return
                if suffix == ".ubj":
                    import xgboost

                    if not self.backbone_path.exists():
                        raise FileNotFoundError(
                            f"分类骨干模型不存在：{self.backbone_path}"
                        )
                    backbone_path = self.backbone_path
                    self._head = xgboost.Booster()
                    self._head.load_model(str(self.classifier_path))
                    self._xgboost = xgboost
                    self._head_type = "xgboost_direct_4class"
                    self._image_size = 224
                else:
                    payload = torch.load(
                        self.classifier_path,
                        map_location=self._device,
                        weights_only=True,
                    )
                    backbone_path = Path(payload["backbone_checkpoint"]).resolve()
                    self._head = HierarchicalHeadProxy.build(
                        torch, nn, int(payload["feature_dim"])
                    ).to(self._device)
                    self._head.load_state_dict(payload["head_state_dict"])
                    self._head.eval()
                    self._head_type = "hierarchical_neural"
                    self._image_size = int(payload.get("image_size", 224))
                    self._thresholds = tuple(
                        float(value)
                        for value in payload.get("grade_bad_thresholds", (0.5, 0.5))
                    )
                    self._joint_temperatures = tuple(
                        float(value)
                        for value in payload.get(
                            "joint_temperatures", (1.0, 1.0, 1.0)
                        )
                    )
                backbone = YOLO(backbone_path)
                self._backbone_layers = nn.ModuleList(list(backbone.model.model[:11])).to(self._device).eval()
                for parameter in self._backbone_layers.parameters():
                    parameter.requires_grad = False
                self._loaded = True
                self._error = ""
            except Exception as exc:
                self._error = str(exc)
                raise
            finally:
                self._loading = False

    def _load_hierarchical_svm(self, torch, nn, yolo_class) -> None:
        """Load a sealed hierarchical SVM and exactly its declared trunks."""
        import joblib

        if not self.selection_path.exists():
            raise FileNotFoundError(f"分类校准文件不存在：{self.selection_path}")
        if not self.cls_backbone_path.exists():
            raise FileNotFoundError(f"ImageNet 分类骨干不存在：{self.cls_backbone_path}")
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        schema = str(selection.get("schema") or "")
        if schema not in SUPPORTED_SVM_SCHEMAS:
            raise ValueError(f"不支持的 SVM selection schema：{schema or '<missing>'}")
        self._selection_sha256 = self._validate_selection_fingerprint(selection)
        selected = selection.get("selected") or {}
        model_record = selected.get("model") or {}
        expected_hash = str(model_record.get("sha256") or "")
        actual_hash = self._sha256_file(self.classifier_path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError("分类器与验证集封存记录不一致，已拒绝加载")
        self._classifier_sha256 = actual_hash
        selected_family_records = selected.get("feature_families") or (
            (selected.get("feature_schema") or {}).get("families")
        ) or []
        expected_families = [
            (item.get("name"), int(item.get("dim", 0)))
            for item in selected_family_records
        ]
        supported_families = {
            (("imagenet_cls", 3584),),
            (("pepper_det", 2304), ("imagenet_cls", 3584)),
        }
        family_tuple = tuple(expected_families)
        if family_tuple not in supported_families:
            raise ValueError(f"不支持的 SVM 特征结构：{expected_families}")
        if schema == "pepper-clean-v5-validation-selection-v1" and family_tuple != (
            ("imagenet_cls", 3584),
        ):
            raise ValueError("clean v5 selection 只允许 imagenet_cls 3584 维特征")
        if schema == "pepper-clean-v5-validation-selection-v1":
            family_record = selected_family_records[0]
            if (
                int(family_record.get("image_size", 0)) != 256
                or family_record.get("scale_normalized") is not True
                or family_record.get("normalization")
                != "independent per-view L2 before concatenation"
            ):
                raise ValueError("clean v5 canonical 特征预处理声明不受支持")
            feature_schema = selected.get("feature_schema") or {}
            if (
                int(feature_schema.get("combined_dim", 0)) != 3584
                or feature_schema.get("concatenation_order") != ["imagenet_cls"]
            ):
                raise ValueError("clean v5 特征拼接 schema 不一致")

        payload = joblib.load(self.classifier_path)
        if not isinstance(payload, dict):
            raise TypeError("SVM 模型必须是分层分类器字典")
        feature_names = tuple(name for name, _ in expected_families)
        if tuple(payload.get("feature_families", ())) != feature_names:
            raise ValueError("SVM 模型与 selection 特征家族不一致")
        payload_schema = payload.get("schema")
        if schema == "pepper-clean-v5-validation-selection-v1" and payload_schema != schema:
            raise ValueError("SVM 模型与 selection schema 不一致")
        if schema == "pepper-v4-svm-selection-v1" and payload_schema not in {
            None,
            schema,
        }:
            raise ValueError("legacy v4 SVM 模型 schema 不受支持")
        if payload.get("view_mode") != "canonical":
            raise ValueError("在线推理仅支持已选定的 canonical SVM")
        grade_models = payload.get("grade_models")
        if payload.get("species_model") is None or not isinstance(
            grade_models, (list, tuple)
        ) or len(grade_models) != 2:
            raise ValueError("SVM 模型缺少品种分支或两个条件品级分支")
        combined_dim = sum(dim for _, dim in expected_families)
        for name, estimator in (
            ("species_model", payload["species_model"]),
            ("grade_model_0", grade_models[0]),
            ("grade_model_1", grade_models[1]),
        ):
            if int(getattr(estimator, "n_features_in_", -1)) != combined_dim:
                raise ValueError(
                    f"{name} 维度与 selection 不一致："
                    f"{getattr(estimator, 'n_features_in_', None)} != {combined_dim}"
                )

        cls_record = next(
            (
                item
                for item in selected_family_records
                if item.get("name") == "imagenet_cls"
            ),
            None,
        )
        expected_cls_hash = str((cls_record or {}).get("checkpoint_sha256") or "")
        if not expected_cls_hash and schema == "pepper-v4-svm-selection-v1":
            expected_cls_hash = IMAGENET_CLS_SHA256
        if not expected_cls_hash or self._sha256_file(self.cls_backbone_path) != expected_cls_hash:
            raise ValueError("ImageNet 分类骨干与 selection 封存记录不一致")

        temperatures = selected.get("temperatures") or {}
        self._joint_temperatures = (
            float(temperatures["species"]),
            float(temperatures["grade_given_子弹头"]),
            float(temperatures["grade_given_条子"]),
        )
        if any(value <= 0 for value in self._joint_temperatures):
            raise ValueError("分支温度必须为正数")
        self._head = payload
        self._head_type = (
            "hierarchical_svm_clean_v5"
            if schema == "pepper-clean-v5-validation-selection-v1"
            else "hierarchical_svm_v4"
        )
        self._image_size = 256
        self._selection_id = str(selection.get("selection_id") or "")
        self._selection_schema = schema
        self._feature_families = feature_names
        self._validation_metrics = selected.get("validation_metrics_calibrated")
        protocol = selection.get("protocol") or {}
        self._provenance = {
            "artifact": "clean_v5" if self._head_type.endswith("clean_v5") else "legacy_v4",
            "selection_scope": "strict_train_and_validation_only",
            "validation_only": True,
            "strict_test_evaluated": bool(protocol.get("test_metrics_computed", False)),
            "test_metrics": None,
            "warning": (
                "当前展示的准确率只是物理隔离验证集结果；"
                "clean v5 尚未在新的独立盲测集上评估。"
                if self._head_type.endswith("clean_v5")
                else "legacy v4 依赖历史 pepper 特征骨干，仅作环境变量兼容。"
            ),
        }

        if "pepper_det" in feature_names:
            # Keep a separate copy because prediction may fuse the detector.
            pepper_backbone = yolo_class(self.detector_path)
            self._pepper_feature_layers = nn.ModuleList(
                list(pepper_backbone.model.model[:11])
            ).to(self._device).eval()
        cls_backbone = yolo_class(self.cls_backbone_path)
        self._cls_feature_layers = nn.ModuleList(
            list(cls_backbone.model.model[:10])
        ).to(self._device).eval()
        self._cls_projection = cls_backbone.model.model[10].to(self._device).eval()
        modules = [self._cls_feature_layers, self._cls_projection]
        if self._pepper_feature_layers is not None:
            modules.append(self._pepper_feature_layers)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_detector_checkpoint(self) -> str:
        """Fail closed when a detector checkpoint cannot be fingerprinted."""
        actual = self._sha256_file(self.detector_path)
        expected = os.getenv("QJ_DETECTOR_SHA256", "").strip().lower()
        if not expected and self.detector_path == DEFAULT_DETECTOR.resolve():
            expected = DEFAULT_DETECTOR_SHA256
        if not expected and self.selection_path.exists():
            try:
                schema = json.loads(
                    self.selection_path.read_text(encoding="utf-8")
                ).get("schema")
            except (OSError, ValueError):
                schema = None
            if schema == "pepper-v4-svm-selection-v1":
                expected = LEGACY_V4_DETECTOR_SHA256
        if not expected:
            raise ValueError(
                "自定义检测模型必须同时设置 QJ_DETECTOR_SHA256"
            )
        if actual != expected:
            raise ValueError("检测模型 SHA256 与封存记录不一致，已拒绝加载")
        return actual

    def _validate_selection_fingerprint(self, selection: dict[str, Any]) -> str:
        """Verify the immutable selection receipt (or the known legacy digest)."""
        actual = self._sha256_file(self.selection_path)
        schema = selection.get("schema")
        explicit = os.getenv("QJ_CLASSIFIER_SELECTION_SHA256", "").strip().lower()
        if explicit:
            if actual != explicit:
                raise ValueError("selection SHA256 与环境变量封存值不一致")
            return actual
        if schema == "pepper-v4-svm-selection-v1":
            if actual != LEGACY_V4_SELECTION_SHA256:
                raise ValueError("legacy v4 selection SHA256 不匹配")
            return actual

        receipt_path = Path(
            os.getenv(
                "QJ_CLASSIFIER_SELECTION_RECEIPT",
                f"{self.selection_path}.sha256.json",
            )
        ).resolve()
        if not receipt_path.exists():
            raise FileNotFoundError(f"clean v5 selection 封存回执不存在：{receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        selection_record = receipt.get("selection") or {}
        model_record = receipt.get("model") or {}
        if receipt.get("selection_id") != selection.get("selection_id"):
            raise ValueError("selection 与封存回执的 selection_id 不一致")
        if str(selection_record.get("sha256") or "") != actual:
            raise ValueError("clean v5 selection SHA256 与封存回执不一致")
        expected_model_hash = str(model_record.get("sha256") or "")
        selected_model_hash = str(
            ((selection.get("selected") or {}).get("model") or {}).get("sha256")
            or ""
        )
        if not expected_model_hash or expected_model_hash != selected_model_hash:
            raise ValueError("selection 与封存回执的分类器 SHA256 不一致")
        return actual

    def status(self) -> dict[str, Any]:
        if self._head_type.startswith("hierarchical_svm"):
            feature_text = "+".join(self._feature_families) or "unloaded"
            architecture = (
                f"YOLO11 detector + {feature_text} multiscale features + "
                "p(species) 1x2 + p(grade|species) 2x2 SVM"
            )
            decision_rule = "argmax[p(species) * p(grade|species)]"
        elif self._head_type == "xgboost_direct_4class":
            architecture = "YOLO11 detector + frozen 256-D YOLO11 backbone + XGBoost 4-class head"
            decision_rule = "argmax[p(class)]"
        else:
            architecture = "YOLO11 detector + p(species) 1x2 + p(grade|species) 2x2"
            decision_rule = "argmax[p(species) * p(grade|species)]"
        return {
            "online": self._loaded,
            "loading": self._loading,
            "device": self._device,
            "detector": str(self.detector_path),
            "classifier": str(self.classifier_path),
            "selection": str(self.selection_path) if self._selection_id else None,
            "selection_id": self._selection_id,
            "selection_schema": self._selection_schema,
            "selection_sha256": self._selection_sha256,
            "classifier_sha256": self._classifier_sha256,
            "detector_sha256": self._detector_sha256,
            "camera_detector_mode": (
                "legacy_proposal_fixed_red_gate"
                if self.enable_legacy_red_gate
                else "strict_default"
            ),
            "legacy_red_gate_enabled": self.enable_legacy_red_gate,
            "camera_gate_red_core_minimum": self._camera_gate_config.red_core_minimum,
            "reject_frame_border_proposals": self.reject_frame_border_proposals,
            "frame_border_margin": self.frame_border_margin,
            "legacy_red_gate_warning": (
                "实验功能：摄像头 holdout 中发现手部误框，不应默认开启"
                if self.enable_legacy_red_gate
                else None
            ),
            "red_component_fallback_enabled": self.enable_red_component_fallback,
            "red_component_fallback_requires_roi": True,
            "red_component_fallback_warning": (
                "实验功能：holdout 中手部仍可形成红色连通域，仅允许在显式传送带 ROI 内运行"
                if self.enable_red_component_fallback
                else None
            ),
            "feature_families": list(self._feature_families),
            "architecture": architecture,
            "decision_rule": decision_rule,
            "joint_temperatures": (
                None
                if self._head_type == "xgboost_direct_4class"
                else self._joint_temperatures
            ),
            "validation_metrics": self._validation_metrics,
            "provenance": self._provenance,
            "error": self._error or None,
        }

    def predict(self, frame: np.ndarray, roi: dict[str, float] | None = None) -> PepperPrediction:
        predictions = self.predict_all(frame, roi)
        return max(predictions, key=lambda item: item.detector_confidence)

    def predict_all(
        self, frame: np.ndarray, roi: dict[str, float] | None = None
    ) -> list[PepperPrediction]:
        """Classify every YOLO pepper box in the trigger zone in one frame."""
        self.load()
        with self._lock:
            started = time.perf_counter()
            detections = self._detect_all(frame, roi)
            classified = []
            annotated = frame.copy()
            crops = [self._crop_with_padding(frame, bbox) for bbox, _ in detections]
            sharpness = [
                float(
                    cv2.Laplacian(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                    ).var()
                )
                for crop in crops
            ]
            batch = self._torch.cat(
                [self._prepare_crop(crop) for crop in crops],
                dim=0,
            )
            species_batch, grade_batch, joint_batch = self._classify_batch(batch)
            for index, (bbox, detector_confidence) in enumerate(detections):
                species_probability = species_batch[index]
                grade_probability = grade_batch[index]
                joint_probability = joint_batch[index]
                flat_class_id = int(joint_probability.reshape(-1).argmax())
                species_id, grade_id = divmod(flat_class_id, 2)
                selected_grade = grade_probability[species_id]
                bad_probability = float(selected_grade[1])
                grade_confidence = float(selected_grade[grade_id])
                joint_confidence = float(joint_probability[species_id, grade_id])
                class_id = species_id * 2 + grade_id
                label = f"{SPECIES_NAMES[species_id]}_{GRADE_NAMES[grade_id]}"
                annotated = self._annotate(
                    annotated, bbox, ASCII_LABELS[class_id], joint_confidence
                )
                classified.append(
                    {
                        "species": SPECIES_NAMES[species_id],
                        "grade": GRADE_NAMES[grade_id],
                        "label": label,
                        "species_confidence": float(species_probability[species_id]),
                        "grade_confidence": grade_confidence,
                        "joint_confidence": joint_confidence,
                        "good_probability": float(selected_grade[0]),
                        "bad_probability": bad_probability,
                        "detector_confidence": detector_confidence,
                        "bbox": bbox,
                        "sharpness": sharpness[index],
                    }
                )
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            return [
                PepperPrediction(
                    **item,
                    processing_time_ms=elapsed,
                    annotated_frame=annotated,
                )
                for item in classified
            ]

    def _classify_batch(self, tensor):
        """Return calibrated species, conditional-grade and joint probabilities."""
        torch = self._torch
        with torch.inference_mode():
            tensor = tensor.to(self._device)
            if self._head_type.startswith("hierarchical_svm"):
                matrix = self._extract_svm_feature_matrix(tensor)
                species_probability = self._ordered_binary_probability(
                    self._head["species_model"], matrix
                )
                grade_probability = np.stack(
                    [
                        self._ordered_binary_probability(model, matrix)
                        for model in self._head["grade_models"]
                    ],
                    axis=1,
                )
                species_probability = self._probability_temperature_scale(
                    species_probability, self._joint_temperatures[0]
                )
                for species_id in range(2):
                    grade_probability[:, species_id] = self._probability_temperature_scale(
                        grade_probability[:, species_id],
                        self._joint_temperatures[species_id + 1],
                    )
            else:
                feature = tensor
                for layer in self._backbone_layers:
                    feature = layer(feature)
                feature = torch.nn.functional.adaptive_avg_pool2d(
                    feature, 1
                ).flatten(1)
                if self._head_type == "xgboost_direct_4class":
                    matrix = self._xgboost.DMatrix(feature.float().cpu().numpy())
                    direct = np.asarray(self._head.predict(matrix), dtype=np.float64)
                    joint = direct.reshape(-1, 2, 2)
                    joint /= np.clip(joint.sum(axis=(1, 2), keepdims=True), 1e-15, None)
                    species_probability = joint.sum(axis=2)
                    grade_probability = joint / np.clip(
                        species_probability[:, :, None], 1e-15, None
                    )
                    return species_probability, grade_probability, joint
                species_logits, grade_logits = self._head(feature)
                species_tensor, grade_tensor = self._temperature_calibrated_probabilities(
                    species_logits,
                    grade_logits,
                    self._joint_temperatures,
                )
                species_probability = species_tensor.float().cpu().numpy()
                grade_probability = grade_tensor.float().cpu().numpy()
        joint_probability = species_probability[:, :, None] * grade_probability
        return species_probability, grade_probability, joint_probability

    def _extract_svm_feature_matrix(self, tensor) -> np.ndarray:
        """Reproduce the declared canonical feature matrix used at selection."""
        torch = self._torch
        tensor = tensor.to(self._device)
        family_features = []
        if "pepper_det" in self._feature_families:
            pepper_outputs = []
            feature = tensor
            for index, layer in enumerate(self._pepper_feature_layers):
                feature = layer(feature)
                if index in {4, 6, 8, 10}:
                    pepper_outputs.append(self._multiscale_pool(feature))
            pepper = torch.cat(pepper_outputs, dim=1)
            if pepper.shape[1] != 2304:
                raise RuntimeError(f"pepper_det 特征维度错误：{pepper.shape[1]}")
            family_features.append(torch.nn.functional.normalize(pepper.float(), dim=1))

        cls_outputs = []
        feature = tensor
        for index, layer in enumerate(self._cls_feature_layers):
            feature = layer(feature)
            if index in {4, 6, 8, 9}:
                cls_outputs.append(self._multiscale_pool(feature))
        projected = self._cls_projection.conv(feature)
        cls_outputs.append(self._cls_projection.pool(projected).flatten(1))
        imagenet = torch.cat(cls_outputs, dim=1)
        if imagenet.shape[1] != 3584:
            raise RuntimeError(f"imagenet_cls 特征维度错误：{imagenet.shape[1]}")
        family_features.append(torch.nn.functional.normalize(imagenet.float(), dim=1))
        combined = torch.cat(family_features, dim=1)
        expected = sum(
            2304 if name == "pepper_det" else 3584
            for name in self._feature_families
        )
        if combined.shape[1] != expected:
            raise RuntimeError(f"SVM 特征总维度错误：{combined.shape[1]} != {expected}")
        return combined.cpu().numpy()

    @staticmethod
    def _multiscale_pool(feature):
        import torch
        import torch.nn.functional as functional

        average = functional.adaptive_avg_pool2d(feature, 1).flatten(1)
        maximum = functional.adaptive_max_pool2d(feature, 1).flatten(1)
        standard_deviation = feature.flatten(2).std(2, unbiased=False)
        return torch.cat((average, maximum, standard_deviation), dim=1)

    @staticmethod
    def _ordered_binary_probability(model, matrix: np.ndarray) -> np.ndarray:
        probability = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        classes = [int(value) for value in np.asarray(model.classes_).tolist()]
        if probability.ndim != 2 or probability.shape[1] != len(classes):
            raise ValueError("SVM 返回了无效的概率形状")
        if set(classes) != {0, 1}:
            raise ValueError(f"SVM 二分类标签必须是 [0,1]，实际为 {classes}")
        ordered = np.empty((len(matrix), 2), dtype=np.float64)
        for source_column, class_id in enumerate(classes):
            ordered[:, class_id] = probability[:, source_column]
        ordered /= np.clip(ordered.sum(axis=1, keepdims=True), 1e-15, None)
        return ordered

    @staticmethod
    def _probability_temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("温度必须为正数")
        log_probability = np.log(np.clip(probability, 1e-15, None)) / temperature
        log_probability -= log_probability.max(axis=1, keepdims=True)
        scaled = np.exp(log_probability)
        return scaled / scaled.sum(axis=1, keepdims=True)

    @staticmethod
    def _temperature_calibrated_probabilities(species_logits, grade_logits, temperatures):
        """Calibrate p(species) and both conditional grade branches."""
        temperature = species_logits.new_tensor(temperatures).clamp_min(0.05)
        species_probability = (species_logits / temperature[0]).softmax(1)
        grade_scaled = species_logits.new_empty(grade_logits.shape)
        grade_scaled[:, 0] = grade_logits[:, 0] / temperature[1]
        grade_scaled[:, 1] = grade_logits[:, 1] / temperature[2]
        return species_probability, grade_scaled.softmax(2)

    @staticmethod
    def _bayesian_argmax(species_probability, grade_probability):
        """Return argmax over p(species) * p(grade | species), shape 2x2."""
        joint_probability = species_probability.unsqueeze(-1) * grade_probability
        flat_class_id = int(joint_probability.reshape(-1).argmax())
        species_id, grade_id = divmod(flat_class_id, 2)
        return species_id, grade_id, joint_probability

    @staticmethod
    def _direct_class_probabilities(class_probability):
        """Decompose direct four-class probabilities for the existing API."""
        joint_probability = np.asarray(class_probability, dtype=np.float64).reshape(2, 2)
        total = float(joint_probability.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("XGBoost returned invalid class probabilities")
        joint_probability /= total
        species_probability = joint_probability.sum(axis=1)
        grade_probability = joint_probability / np.clip(
            species_probability[:, None], 1e-12, None
        )
        flat_class_id = int(joint_probability.reshape(-1).argmax())
        species_id, grade_id = divmod(flat_class_id, 2)
        return (
            species_id,
            grade_id,
            joint_probability,
            species_probability,
            grade_probability,
        )

    def _detect(
        self, frame: np.ndarray, roi: dict[str, float] | None
    ) -> tuple[tuple[int, int, int, int], float]:
        return max(
            self._detect_all(frame, roi),
            key=lambda item: item[1],
        )

    def _detect_all(
        self, frame: np.ndarray, roi: dict[str, float] | None
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        proposal_detector = (
            self._legacy_proposal_detector
            if self.enable_legacy_red_gate
            else self._detector
        )
        proposal_confidence = (
            self._camera_gate_config.proposal_confidence
            if self.enable_legacy_red_gate
            else self.detector_confidence
        )
        predict_options = {}
        if self.enable_legacy_red_gate:
            predict_options = {
                "imgsz": self._camera_gate_config.inference_size,
                # Keep cross-class duplicates until the fixed gate performs
                # deterministic class-agnostic NMS.
                "agnostic_nms": False,
                "max_det": max(100, self.max_peppers * 10),
            }
        else:
            predict_options = {
                "agnostic_nms": True,
                "max_det": self.max_peppers * 3,
            }
        result = proposal_detector.predict(
            source=frame,
            conf=proposal_confidence,
            iou=self.detector_iou,
            verbose=False,
            device=self._device,
            **predict_options,
        )[0]
        height, width = frame.shape[:2]
        candidates = []
        proposal_pairs = []
        if result.boxes is not None:
            proposal_pairs = [
                (tuple(int(round(value)) for value in box), float(confidence))
                for box, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            ]
        raw_proposal_pairs = list(proposal_pairs)
        if self.enable_legacy_red_gate:
            gated = gate_camera_proposals(frame, proposal_pairs, self._camera_gate_config)
            proposal_pairs = [
                (proposal.bbox, proposal.confidence)
                for proposal in gated
                if proposal.accepted
            ]
            surface_mask = (
                white_detection_surface_mask(frame)
                if self.enable_red_component_fallback
                else None
            )
            if surface_mask is not None:
                white_gate_config = CameraGateConfig(
                    minimum_confidence=0.20,
                    red_saturation_min=70,
                    red_value_min=25,
                    red_core_minimum=0.10,
                    maximum_area_ratio=0.10,
                    maximum_aspect_ratio=7.0,
                )
                white_detector_pairs = []
                for proposal in gate_camera_proposals(
                    frame, raw_proposal_pairs, white_gate_config
                ):
                    if not proposal.accepted:
                        continue
                    center_x = (proposal.bbox[0] + proposal.bbox[2]) // 2
                    center_y = (proposal.bbox[1] + proposal.bbox[3]) // 2
                    if surface_mask[center_y, center_x] == 0:
                        continue
                    if proposal_median_red_saturation(
                        frame, proposal.bbox, white_gate_config
                    ) < 110:
                        continue
                    white_detector_pairs.append(
                        (proposal.bbox, proposal.confidence)
                    )
                component_pairs = red_component_proposals(
                    frame,
                    {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
                    CameraGateConfig(
                        red_saturation_min=70,
                        red_value_min=25,
                    ),
                    RedComponentConfig(
                        minimum_aspect_ratio=1.0,
                        maximum_area_ratio=0.08,
                        minimum_fill_ratio=0.10,
                        minimum_median_saturation=110,
                        opening_kernel_size=7,
                    ),
                    surface_mask=surface_mask,
                )
                proposal_pairs = merge_detector_and_component_proposals(
                    white_detector_pairs,
                    component_pairs,
                )
        elif self.enable_red_component_fallback and roi is not None:
            component_pairs = red_component_proposals(
                frame,
                roi,
                self._camera_gate_config,
            )
            proposal_pairs = merge_detector_and_component_proposals(
                proposal_pairs,
                component_pairs,
            )
        for box, confidence in proposal_pairs:
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            if self.reject_frame_border_proposals and self._touches_frame_border(
                (x1, y1, x2, y2),
                width,
                height,
                self.frame_border_margin,
            ):
                continue
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            inside_roi = not roi or (
                roi["left"] <= center_x <= roi["left"] + roi["width"]
                and roi["top"] <= center_y <= roi["top"] + roi["height"]
            )
            area = max(0, x2 - x1) * max(0, y2 - y1)
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            elongation = max(box_width / box_height, box_height / box_width)
            if elongation < self.min_elongation:
                continue
            candidates.append((inside_roi, float(confidence), area, (x1, y1, x2, y2)))
        if roi:
            candidates = [candidate for candidate in candidates if candidate[0]]
        if candidates:
            candidates.sort(key=lambda item: item[3][0])
            return [
                (bbox, confidence)
                for _, confidence, _, bbox in candidates[: self.max_peppers]
            ]
        if not self.allow_roi_fallback:
            raise NoPepperDetectedError("触发区域内未检测到辣椒")
        if roi:
            x1 = int(width * roi["left"])
            y1 = int(height * roi["top"])
            x2 = int(width * (roi["left"] + roi["width"]))
            y2 = int(height * (roi["top"] + roi["height"]))
        else:
            x1, y1, x2, y2 = int(width * 0.3), int(height * 0.2), int(width * 0.7), int(height * 0.8)
        return [((x1, y1, x2, y2), 0.0)]

    @staticmethod
    def _touches_frame_border(
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        margin: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= width - margin
            or y2 >= height - margin
        )

    @staticmethod
    def _crop_with_padding(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = max(4, round((x2 - x1) * 0.12))
        pad_y = max(4, round((y2 - y1) * 0.12))
        return frame[max(0, y1 - pad_y) : min(height, y2 + pad_y), max(0, x1 - pad_x) : min(width, x2 + pad_x)]

    def _prepare_crop(self, crop: np.ndarray):
        if crop.size == 0:
            raise RuntimeError("辣椒检测框为空")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pepper = Image.fromarray(rgb).convert("RGBA")
        target_fraction = 0.88 if self._head_type.startswith("hierarchical_svm") else 0.86
        target = round(self._image_size * target_fraction)
        scale = target / max(pepper.width, pepper.height, 1)
        pepper = pepper.resize(
            (
                max(1, round(pepper.width * scale)),
                max(1, round(pepper.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new(
            "RGBA", self._image_size_tuple, (64, 68, 68, 255)
        )
        x = (self._image_size - pepper.width) // 2
        y = (self._image_size - pepper.height) // 2
        canvas.alpha_composite(pepper, (x, y))
        array = (
            np.asarray(canvas.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
            / 255.0
        )
        return self._torch.from_numpy(array.copy()).unsqueeze(0)

    @property
    def _image_size_tuple(self) -> tuple[int, int]:
        return self._image_size, self._image_size

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        confidence: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox
        color = (68, 200, 129) if "LEVEL 1" in label else (65, 94, 224)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {confidence:.1%}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y1 - text_height - 12)
        cv2.rectangle(annotated, (x1, top), (x1 + text_width + 12, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 18, 18), 2)
        return annotated


model_service = PepperModelService()
