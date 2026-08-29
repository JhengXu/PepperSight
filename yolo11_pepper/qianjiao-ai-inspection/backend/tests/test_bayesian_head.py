import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.model_service import PepperModelService


def test_bayesian_joint_probabilities_sum_to_one():
    species = torch.tensor([0.6, 0.4])
    conditional_grade = torch.tensor([[0.7, 0.3], [0.2, 0.8]])

    species_id, grade_id, joint = PepperModelService._bayesian_argmax(
        species,
        conditional_grade,
    )

    assert float(joint.sum()) == pytest.approx(1.0)
    assert (species_id, grade_id) == (0, 0)
    assert float(joint[species_id, grade_id]) == pytest.approx(0.42)


def test_joint_argmax_can_choose_non_argmax_species():
    # Hard routing would select species 0 first. Joint Bayes correctly selects
    # species 1 because p(species=1) * p(level1|species=1) is larger.
    species = torch.tensor([0.55, 0.45])
    conditional_grade = torch.tensor([[0.51, 0.49], [0.99, 0.01]])

    species_id, grade_id, joint = PepperModelService._bayesian_argmax(
        species,
        conditional_grade,
    )

    assert (species_id, grade_id) == (1, 0)
    assert float(joint[1, 0]) == pytest.approx(0.4455)


def test_hierarchical_temperature_calibration_preserves_probability_sums():
    species_logits = torch.tensor([[2.0, 0.0]])
    grade_logits = torch.tensor([[[3.0, 0.0], [0.0, 1.0]]])

    species, grade = PepperModelService._temperature_calibrated_probabilities(
        species_logits,
        grade_logits,
        (2.0, 3.0, 0.5),
    )

    assert float(species.sum()) == pytest.approx(1.0)
    assert np.allclose(grade.sum(2).cpu().numpy(), np.ones((1, 2)))
    assert float(species[0, 0]) < float(species_logits.softmax(1)[0, 0])
    assert float(grade[0, 0, 0]) < float(grade_logits.softmax(2)[0, 0, 0])


def test_direct_xgboost_probabilities_are_decomposed_for_existing_api():
    (
        species_id,
        grade_id,
        joint,
        species,
        conditional_grade,
    ) = PepperModelService._direct_class_probabilities([0.10, 0.20, 0.55, 0.15])

    assert (species_id, grade_id) == (1, 0)
    assert float(joint.sum()) == pytest.approx(1.0)
    assert np.allclose(species, [0.30, 0.70])
    assert np.allclose(conditional_grade.sum(axis=1), [1.0, 1.0])
    assert np.allclose(joint, species[:, None] * conditional_grade)


class _ReverseBinaryEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, matrix):
        return np.tile(np.array([[0.8, 0.2]]), (len(matrix), 1))


def test_svm_probability_columns_follow_estimator_classes():
    probability = PepperModelService._ordered_binary_probability(
        _ReverseBinaryEstimator(), np.zeros((3, 4), dtype=np.float32)
    )

    assert probability.shape == (3, 2)
    assert np.allclose(probability, np.array([[0.2, 0.8]] * 3))


def test_svm_probability_temperature_scaling_is_normalized():
    probability = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)

    sharper = PepperModelService._probability_temperature_scale(probability, 0.5)
    softer = PepperModelService._probability_temperature_scale(probability, 2.0)

    assert np.allclose(sharper.sum(axis=1), [1.0, 1.0])
    assert np.allclose(softer.sum(axis=1), [1.0, 1.0])
    assert sharper[0, 0] > probability[0, 0]
    assert softer[0, 0] < probability[0, 0]


def test_multiscale_pool_concatenates_average_maximum_and_std():
    feature = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])

    pooled = PepperModelService._multiscale_pool(feature)

    assert pooled.shape == (1, 3)
    assert np.allclose(pooled[0].numpy(), [4.0, 7.0, np.sqrt(5.0)])


def test_clean_v5_canonical_crop_matches_offline_extractor(tmp_path):
    """Online BGR crops must exactly reproduce extractor view=0 pixels."""
    root = Path(__file__).resolve().parents[3]
    extractor_path = root / "extract_v4_multiscale_features.py"
    spec = importlib.util.spec_from_file_location("pepper_feature_extractor", extractor_path)
    assert spec and spec.loader
    extractor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extractor
    spec.loader.exec_module(extractor)

    height, width = 91, 173
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 3 + yy) % 256,
            (xx + yy * 5) % 256,
            (xx * 7 + yy * 2) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    image_path = tmp_path / "opaque-crop.png"
    Image.fromarray(rgb, mode="RGB").save(image_path)

    expected, _ = extractor.render_view(image_path, size=256, view=0, seed=2041)
    service = PepperModelService()
    service._torch = torch
    service._head_type = "hierarchical_svm_clean_v5"
    service._image_size = 256
    actual = service._prepare_crop(rgb[:, :, ::-1].copy()).squeeze(0)

    assert actual.shape == (3, 256, 256)
    assert torch.equal(actual, expected)


def test_clean_v5_status_never_presents_validation_as_test():
    service = PepperModelService()
    service._head_type = "hierarchical_svm_clean_v5"
    service._selection_schema = "pepper-clean-v5-validation-selection-v1"
    service._feature_families = ("imagenet_cls",)
    service._validation_metrics = {"joint_accuracy": 0.856}
    service._provenance = {
        "validation_only": True,
        "strict_test_evaluated": False,
        "test_metrics": None,
    }

    status = service.status()

    assert status["validation_metrics"] == {"joint_accuracy": 0.856}
    assert status["provenance"]["validation_only"] is True
    assert status["provenance"]["strict_test_evaluated"] is False
    assert status["provenance"]["test_metrics"] is None
    assert status["feature_families"] == ["imagenet_cls"]
