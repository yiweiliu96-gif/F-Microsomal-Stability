#!/usr/bin/env python3
"""Synchronize author-verified pair review status across release tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = ROOT / "reports" / "assay_matched_fluorination_pairs"
PROJECT_RELEASE_DIR = ROOT / "github_release" / "fluorination_cross_species_clint" / "data" / "derived"
SOURCE_DIR = PROJECT_SOURCE_DIR if PROJECT_SOURCE_DIR.exists() else ROOT / "data" / "derived"
RELEASE_DIR = PROJECT_RELEASE_DIR if PROJECT_SOURCE_DIR.exists() else SOURCE_DIR
QUEUE = SOURCE_DIR / "cross_species_manual_review_queue.csv"
TARGETS = [
    SOURCE_DIR / "cross_species_assay_matched_pairs.csv",
    SOURCE_DIR / "cross_species_source_row_audit.csv",
]


def update_table(path: Path, review: pd.DataFrame) -> None:
    frame = pd.read_csv(path)
    if frame["pair_id"].duplicated().any() or set(frame["pair_id"]) != set(review["pair_id"]):
        raise RuntimeError(f"Pair IDs do not match the verified review table: {path}")
    status = review.set_index("pair_id")
    frame["manual_review_status"] = frame["pair_id"].map(status["manual_review_status"])
    frame["manual_review_note"] = frame["pair_id"].map(status["manual_review_note"])
    frame.to_csv(path, index=False)


def main() -> None:
    review = pd.read_csv(QUEUE)
    if len(review) != 97 or review["pair_id"].duplicated().any():
        raise RuntimeError("Expected 97 unique manually reviewed pairs")
    if not review["manual_review_status"].eq("verified").all():
        raise RuntimeError("Manual review is not complete; refusing to publish verified status")

    for target in TARGETS:
        update_table(target, review)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    review.to_csv(RELEASE_DIR / QUEUE.name, index=False)
    for target in TARGETS:
        pd.read_csv(target).to_csv(RELEASE_DIR / target.name, index=False)
    print("Synchronized verified review status for 97 pairs across working and release tables")


if __name__ == "__main__":
    main()
