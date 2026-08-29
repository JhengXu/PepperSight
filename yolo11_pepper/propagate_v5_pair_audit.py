#!/usr/bin/env python3
"""Make detector-aligned training views inherit their canonical label audit.

The two images in a ``pair_id`` are alternative views of one pepper, not two
independent labels.  A train-only OOF audit may assign slightly different
confidence weights to those pixels, so this utility freezes the canonical
view's label decision onto its detector-aligned partner before model fitting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DECISION_FIELDS = (
    "species_weight",
    "grade_weight",
    "grade_soft_target",
    "audit_status",
    "review_priority",
    "audit_reasons",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    missing = {"pair_id", "view_type", "class_id", *DECISION_FIELDS} - set(fields)
    if missing:
        raise ValueError(f"Input audit lacks fields: {sorted(missing)}")

    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    propagated = 0
    for pair_id, members in by_pair.items():
        canonical = [row for row in members if row["view_type"] == "canonical"]
        if len(canonical) != 1:
            raise ValueError(f"Pair {pair_id} has {len(canonical)} canonical views")
        source = canonical[0]
        if any(row["class_id"] != source["class_id"] for row in members):
            raise ValueError(f"Pair {pair_id} has inconsistent class labels")
        for row in members:
            if row is source:
                continue
            if row["view_type"] != "detector_aligned":
                raise ValueError(f"Pair {pair_id} has unsupported view {row['view_type']!r}")
            for field in DECISION_FIELDS:
                row[field] = source[field]
            propagated += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "rows": len(rows),
            "pairs": len(by_pair),
            "detector_views_inheriting_canonical_audit": propagated,
            "output": str(args.output.resolve()),
        }
    )


if __name__ == "__main__":
    main()
