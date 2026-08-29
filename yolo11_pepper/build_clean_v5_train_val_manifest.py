#!/usr/bin/env python3
"""Build a physically test-free image-training manifest for clean v5.

The image fine-tuner historically accepted one manifest containing several
selection roles.  This builder preserves that interface while ensuring the
result contains *only* the audited training rows and the physically separated
model-selection rows.  It also fails on path, source, group, pair or content
hash overlap before writing anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DEFAULT_TRAIN = (
    PROJECT
    / "datasets/pepper_ssl_v5_clean_audit/train_label_audit_paired.csv"
)
DEFAULT_VAL = PROJECT / "datasets/pepper_ssl_v4_merged/model_selection_manifest.csv"
DEFAULT_OUTPUT = (
    PROJECT / "datasets/pepper_ssl_v5_clean_audit/train_val_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.resolve().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values(rows: list[dict[str, str]], key: str) -> set[str]:
    return {str(row.get(key) or "").strip() for row in rows if row.get(key)}


def validate_role(
    rows: list[dict[str, str]], *, split: str, role: str, eligible: str
) -> None:
    if not rows:
        raise ValueError(f"No {split} rows")
    for line, row in enumerate(rows, 2):
        observed = {
            "split": str(row.get("split") or "").strip().lower(),
            "selection_role": str(row.get("selection_role") or "").strip().lower(),
            "eligible_for_model_training": str(
                row.get("eligible_for_model_training") or ""
            ).strip().lower(),
        }
        expected = {
            "split": split,
            "selection_role": role,
            "eligible_for_model_training": eligible,
        }
        if observed != expected:
            raise ValueError(
                f"row {line} role mismatch in {split}: {observed} != {expected}"
            )
        image = Path(str(row.get("path") or "")).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    val_path = args.val.resolve()
    output = args.output.resolve()
    if any("test" in part.lower().replace("latest", "") for part in output.parts):
        raise ValueError(f"Output must be physically test-free: {output}")
    train_fields, train = read_rows(train_path)
    val_fields, val = read_rows(val_path)
    validate_role(train, split="train", role="training", eligible="true")
    validate_role(val, split="val", role="model_selection", eligible="false")

    overlap: dict[str, list[str]] = {}
    for key in ("path", "group_id", "source_id", "pair_id", "content_sha256"):
        shared = sorted(values(train, key) & values(val, key))
        overlap[key] = shared
        if shared:
            raise ValueError(f"Train/validation {key} overlap: {shared[:3]}")

    fields = list(train_fields)
    fields.extend(field for field in val_fields if field not in fields)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(train)
            writer.writerows(val)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    receipt = {
        "schema": "pepper-clean-v5-physical-train-val-manifest-v1",
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "validation_manifest": str(val_path),
        "validation_manifest_sha256": sha256_file(val_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "train_rows": len(train),
        "validation_rows": len(val),
        "strict_test_rows": 0,
        "overlap": overlap,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
