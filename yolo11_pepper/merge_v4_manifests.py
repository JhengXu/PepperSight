#!/usr/bin/env python3
"""Merge the audited v4 manifest with detector-aligned production views.

The audited manifest remains the canonical dataset.  Matched detector crops
from training groups are appended as a second view of the same pepper instance,
not as an independent label.  Detector crops from validation and test groups
are isolated in a diagnostic-only manifest and can never enter the ordinary
validation/model-selection or strict-test manifests.

Both source datasets are immutable inputs.  The merger refuses to overwrite an
existing output directory and publishes a completed, audited dataset atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


MERGE_FIELDS = (
    "record_role",
    "selection_role",
    "evaluation_scope",
    "eligible_for_model_training",
    "pair_id",
    "view_type",
    "paired_view_path",
    "reference_instance_path",
    "pair_consistency_eligible",
    "independent_sample",
    "content_sha256",
    "detector_label_state",
)

PAIR_FIELDS = (
    "pair_id",
    "split",
    "group_id",
    "source_id",
    "class_id",
    "class_name",
    "canonical_path",
    "detector_aligned_path",
    "reference_iou",
    "detector_confidence",
    "training_pair_eligible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_audit/manifest.csv"),
        help="label-audit v4 main manifest; may be supplied later with this option",
    )
    parser.add_argument(
        "--detector-manifest",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_detector_aligned_v4/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo11_pepper/datasets/pepper_ssl_v4_merged"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def require_fields(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Manifest {path} is missing required fields: {missing}")


def normalized_path(value: str) -> str:
    return str(Path(value).resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_id_for_path(path: str) -> str:
    digest = hashlib.sha256(normalized_path(path).encode("utf-8")).hexdigest()[:20]
    return f"pepper-instance:{digest}"


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ordered_union(*field_groups: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for group in field_groups:
        for field in group:
            if field not in result:
                result.append(field)
    return result


def validate_audit_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    group_identity: dict[str, tuple[str, str, str]] = {}
    for index, row in enumerate(rows, 1):
        path = normalized_path(row["path"])
        if path in by_path:
            raise ValueError(f"Duplicate canonical path at audit row {index}: {path}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported audit split at row {index}: {row['split']!r}")
        if int(row["class_id"]) not in range(4):
            raise ValueError(f"Unsupported audit class_id at row {index}: {row['class_id']!r}")
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Canonical image is missing: {image_path}")
        identity = (row["split"], row["class_id"], row["class_name"])
        prior = group_identity.setdefault(row["group_id"], identity)
        if prior != identity:
            raise ValueError(f"Canonical group has conflicting ownership: {row['group_id']}: {prior}, {identity}")
        normalized = dict(row)
        normalized["path"] = path
        normalized["source_path"] = normalized_path(row["source_path"])
        by_path[path] = normalized
    return by_path


def validate_detector_rows(rows: list[dict[str, str]]) -> None:
    seen_paths: set[str] = set()
    for index, row in enumerate(rows, 1):
        path = normalized_path(row["path"])
        if path in seen_paths:
            raise ValueError(f"Duplicate detector crop at row {index}: {path}")
        seen_paths.add(path)
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported detector split at row {index}: {row['split']!r}")
        if not Path(path).is_file():
            raise FileNotFoundError(f"Detector crop is missing: {path}")
        eligible = row["eligible_for_training"].strip().lower() == "true"
        if eligible and not (
            row["split"] == "train"
            and row["usage"] == "training"
            and row["reference_match"].strip().lower() == "true"
        ):
            raise ValueError(f"Unsafe eligible detector row {index}: {row}")
        if row["split"] in {"val", "test"} and (
            eligible or row["usage"] != "evaluation_only"
        ):
            raise ValueError(f"Held-out detector crop is not evaluation_only at row {index}")


def copy_detector_metadata(target: dict[str, object], detector: dict[str, str]) -> None:
    """Copy detector evidence without replacing the audited label decision."""
    protected = {
        "path",
        "source_path",
        "origin",
        "split",
        "group_id",
        "source_id",
        "class_id",
        "class_name",
        "original_class_id",
        "label_state",
        "label_weight",
    }
    for key, value in detector.items():
        if key not in protected:
            target[key] = value
    target["detector_label_state"] = detector.get("label_state", "")


def build_rows(
    audit_rows: list[dict[str, str]],
    audit_by_path: dict[str, dict[str, str]],
    detector_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matched_detector_by_reference: dict[str, dict[str, str]] = {}
    pair_rows: list[dict[str, object]] = []
    for detector in detector_rows:
        reference_text = detector.get("reference_instance_path", "").strip()
        reference_path = normalized_path(reference_text) if reference_text else ""
        matched = detector.get("reference_match", "").lower() == "true"
        if matched:
            if reference_path not in audit_by_path:
                raise ValueError(f"Detector reference is absent from audited manifest: {reference_path}")
            if reference_path in matched_detector_by_reference:
                raise ValueError(f"Multiple detector views claim one reference instance: {reference_path}")
            base = audit_by_path[reference_path]
            expected = (base["split"], base["group_id"], base["class_id"], base["class_name"])
            actual = (
                detector["split"],
                detector["group_id"],
                detector["source_class_id"],
                detector["source_class_name"],
            )
            if expected != actual:
                raise ValueError(
                    f"Detector/reference ownership mismatch for {reference_path}: {actual} != {expected}"
                )
            matched_detector_by_reference[reference_path] = detector

    merged: list[dict[str, object]] = []
    for raw in audit_rows:
        canonical = audit_by_path[normalized_path(raw["path"])]
        split = canonical["split"]
        detector = matched_detector_by_reference.get(canonical["path"])
        pair_id = pair_id_for_path(canonical["path"])
        row: dict[str, object] = dict(canonical)
        row.update(
            {
                "usage": {
                    "train": "canonical_training",
                    "val": "model_selection",
                    "test": "strict_test",
                }[split],
                "eligible_for_training": str(split == "train").lower(),
                "record_role": "canonical_audited",
                "selection_role": {
                    "train": "training",
                    "val": "model_selection",
                    "test": "strict_test",
                }[split],
                "evaluation_scope": "standard",
                "eligible_for_model_training": str(split == "train").lower(),
                "pair_id": pair_id,
                "view_type": "canonical",
                "paired_view_path": normalized_path(detector["path"]) if detector else "",
                "reference_instance_path": canonical["path"],
                "pair_consistency_eligible": str(split == "train" and detector is not None).lower(),
                "independent_sample": "true",
                "detector_label_state": "",
            }
        )
        merged.append(row)

    for detector in detector_rows:
        crop_path = normalized_path(detector["path"])
        reference_text = detector.get("reference_instance_path", "").strip()
        reference_path = normalized_path(reference_text) if reference_text else ""
        matched = detector.get("reference_match", "").lower() == "true"
        base = audit_by_path.get(reference_path) if matched else None
        if base:
            row = dict(base)
            class_id, class_name = base["class_id"], base["class_name"]
            original_class_id = base["original_class_id"]
            label_state = base["label_state"]
            pair_id = pair_id_for_path(reference_path)
        else:
            row = {field: "" for field in audit_rows[0]}
            class_id, class_name = detector["source_class_id"], detector["source_class_name"]
            original_class_id = class_id
            label_state = detector.get("label_state", "source_group_label_unverified_detection")
            pair_id = pair_id_for_path(crop_path)
            row.update(
                {
                    "label_weight": "1.0",
                    "audit_status": "detector_evaluation_source_group_only",
                    "review_priority": "3",
                    "audit_reasons": "detector_box_has_no_unique_v3_reference_match",
                }
            )

        split = detector["split"]
        input_eligible = detector["eligible_for_training"].lower() == "true"
        training_eligible = split == "train" and input_eligible and matched
        if split == "train" and not training_eligible:
            selection_role = "review_only"
            evaluation_scope = "detector_domain_review"
            record_role = "detector_train_review_only"
        elif split == "train":
            selection_role = "training"
            evaluation_scope = "detector_domain_training_view"
            record_role = "paired_detector_train_view"
        else:
            selection_role = "diagnostic_only"
            evaluation_scope = "detector_domain"
            record_role = "detector_evaluation_only"

        row.update(
            {
                "path": crop_path,
                "source_path": normalized_path(detector["source_path"]),
                "origin": detector["origin"],
                "split": split,
                "group_id": detector["group_id"],
                "source_id": detector["source_id"],
                "class_id": class_id,
                "class_name": class_name,
                "original_class_id": original_class_id,
                "label_state": label_state,
                "usage": detector["usage"],
                "eligible_for_training": str(input_eligible).lower(),
                "record_role": record_role,
                "selection_role": selection_role,
                "evaluation_scope": evaluation_scope,
                "eligible_for_model_training": str(training_eligible).lower(),
                "pair_id": pair_id,
                "view_type": "detector_aligned",
                "paired_view_path": reference_path,
                "reference_instance_path": reference_path,
                "pair_consistency_eligible": str(training_eligible and base is not None).lower(),
                "independent_sample": "false",
            }
        )
        copy_detector_metadata(row, detector)
        merged.append(row)
        if base is not None:
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "group_id": base["group_id"],
                    "source_id": base["source_id"],
                    "class_id": base["class_id"],
                    "class_name": base["class_name"],
                    "canonical_path": reference_path,
                    "detector_aligned_path": crop_path,
                    "reference_iou": detector["reference_iou"],
                    "detector_confidence": detector["detector_confidence"],
                    "training_pair_eligible": str(training_eligible).lower(),
                }
            )
    return merged, pair_rows


def add_and_verify_hashes(rows: list[dict[str, object]]) -> None:
    hash_cache: dict[str, str] = {}
    for row in rows:
        path = str(row["path"])
        if path not in hash_cache:
            hash_cache[path] = sha256_file(Path(path))
        content_hash = hash_cache[path]
        recorded_detector_hash = str(row.get("crop_sha256", "")).strip()
        if recorded_detector_hash and recorded_detector_hash != content_hash:
            raise ValueError(f"Detector crop hash mismatch: {path}")
        row["content_sha256"] = content_hash


def leakage_audit(
    merged: list[dict[str, object]],
    train: list[dict[str, object]],
    model_selection: list[dict[str, object]],
    strict_test: list[dict[str, object]],
    detector_evaluation: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> dict[str, object]:
    def cross_split(field: str) -> dict[str, list[str]]:
        owners: dict[str, set[str]] = defaultdict(set)
        for row in merged:
            owners[str(row[field])].add(str(row["split"]))
        return {key: sorted(value) for key, value in owners.items() if key and len(value) > 1}

    path_counts = Counter(str(row["path"]) for row in merged)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    pair_counts = Counter(str(row["pair_id"]) for row in merged)
    oversized_pairs = {key: value for key, value in pair_counts.items() if key and value > 2}
    violations = {
        "group_cross_split": cross_split("group_id"),
        "source_cross_split": cross_split("source_id"),
        "pair_cross_split": cross_split("pair_id"),
        "exact_content_hash_cross_split": cross_split("content_sha256"),
        "duplicate_paths": duplicate_paths,
        "pairs_with_more_than_two_views": oversized_pairs,
        "train_contains_non_train_split": [
            str(row["path"]) for row in train if row["split"] != "train"
        ],
        "train_contains_ineligible_row": [
            str(row["path"])
            for row in train
            if row["eligible_for_model_training"] != "true"
        ],
        "model_selection_contains_detector_view": [
            str(row["path"]) for row in model_selection if row["view_type"] != "canonical"
        ],
        "strict_test_contains_detector_view": [
            str(row["path"]) for row in strict_test if row["view_type"] != "canonical"
        ],
        "detector_evaluation_contains_training_role": [
            str(row["path"])
            for row in detector_evaluation
            if row["selection_role"] != "diagnostic_only"
            or row["eligible_for_model_training"] != "false"
            or row["split"] not in {"val", "test"}
        ],
        "train_pair_not_train_split": [
            str(row["pair_id"])
            for row in pair_rows
            if row["training_pair_eligible"] == "true" and row["split"] != "train"
        ],
    }
    passed = not any(bool(value) for value in violations.values())
    return {
        "passed": passed,
        **violations,
        "counts": {
            "merged_rows": len(merged),
            "unique_paths": len(path_counts),
            "unique_content_hashes": len({str(row["content_sha256"]) for row in merged}),
            "unique_instances": len(pair_counts),
            "paired_instances": sum(count == 2 for count in pair_counts.values()),
        },
    }


def count_by(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    audit_manifest = args.audit_manifest.resolve()
    detector_manifest = args.detector_manifest.resolve()
    output = args.output.resolve()
    for manifest in (audit_manifest, detector_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(f"Source manifest is missing: {manifest}")
    for source_directory in (audit_manifest.parent, detector_manifest.parent):
        if path_is_within(output, source_directory):
            raise ValueError(f"Output must not be inside immutable source directory: {source_directory}")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing merged dataset: {output}. Choose a new --output path."
        )

    audit_fields, audit_rows = read_csv(audit_manifest)
    detector_fields, detector_rows = read_csv(detector_manifest)
    require_fields(
        audit_manifest,
        audit_fields,
        {"path", "source_path", "origin", "split", "group_id", "source_id", "class_id", "class_name", "original_class_id", "label_state"},
    )
    require_fields(
        detector_manifest,
        detector_fields,
        {"path", "source_path", "origin", "split", "usage", "eligible_for_training", "group_id", "source_id", "source_class_id", "source_class_name", "reference_instance_path", "reference_match", "reference_iou", "detector_confidence", "crop_sha256"},
    )
    if not audit_rows or not detector_rows:
        raise ValueError("Both source manifests must contain data rows")
    audit_by_path = validate_audit_rows(audit_rows)
    validate_detector_rows(detector_rows)
    merged, pair_rows = build_rows(audit_rows, audit_by_path, detector_rows)
    add_and_verify_hashes(merged)

    train = [row for row in merged if row["eligible_for_model_training"] == "true"]
    model_selection = [row for row in merged if row["selection_role"] == "model_selection"]
    strict_test = [row for row in merged if row["selection_role"] == "strict_test"]
    detector_evaluation = [
        row
        for row in merged
        if row["view_type"] == "detector_aligned" and row["split"] in {"val", "test"}
    ]
    train_pairs = [row for row in pair_rows if row["training_pair_eligible"] == "true"]
    audit = leakage_audit(
        merged, train, model_selection, strict_test, detector_evaluation, pair_rows
    )
    if not audit["passed"]:
        raise RuntimeError(f"Merged leakage audit failed: {json.dumps(audit, ensure_ascii=False)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    fields = ordered_union(
        ["path", "source_path", "origin", "split", "group_id", "source_id", "class_id", "class_name"],
        MERGE_FIELDS,
        audit_fields,
        detector_fields,
    )
    write_csv(building / "manifest.csv", fields, merged)
    write_csv(building / "train_manifest.csv", fields, train)
    write_csv(building / "model_selection_manifest.csv", fields, model_selection)
    write_csv(building / "strict_test_manifest.csv", fields, strict_test)
    write_csv(building / "detector_evaluation_manifest.csv", fields, detector_evaluation)
    write_csv(building / "paired_views.csv", PAIR_FIELDS, pair_rows)
    write_csv(building / "train_pairs.csv", PAIR_FIELDS, train_pairs)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "audit_manifest": str(audit_manifest),
            "detector_manifest": str(detector_manifest),
        },
        "output": str(output),
        "policy": {
            "canonical_dataset": "all rows from the label-audit v4 manifest",
            "detector_train": "only eligible matched crops from train groups; represented as paired views",
            "detector_val_test": "diagnostic-only; excluded from model-selection and strict-test manifests",
            "pair_semantics": "same pair_id means the canonical and detector crop are two views of one pepper, not independent labels",
        },
        "counts": {
            "canonical_rows": len(audit_rows),
            "detector_rows": len(detector_rows),
            "merged_rows": len(merged),
            "train_rows": len(train),
            "train_unique_instances": len({str(row["pair_id"]) for row in train}),
            "train_detector_views": sum(row["record_role"] == "paired_detector_train_view" for row in train),
            "model_selection_rows": len(model_selection),
            "strict_test_rows": len(strict_test),
            "detector_evaluation_rows": len(detector_evaluation),
            "all_matched_pairs": len(pair_rows),
            "training_pairs": len(train_pairs),
        },
        "merged_by_split": count_by(merged, "split"),
        "merged_by_record_role": count_by(merged, "record_role"),
        "train_by_view_type": count_by(train, "view_type"),
        "train_by_class": count_by(train, "class_name"),
        "detector_evaluation_by_split": count_by(detector_evaluation, "split"),
        "leakage_audit": audit,
    }
    (building / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (building / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(building, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
