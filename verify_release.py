#!/usr/bin/env python3
"""Verify integrity and manuscript-facing values in the public release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(value: str, expected: float, tolerance: float = 5e-4) -> bool:
    return abs(float(value) - expected) <= tolerance


required = [
    "README.md",
    "CITATION.cff",
    "MANIFEST.sha256",
    "DATA_AND_REVIEW_NOTICE.md",
    "docs/DATA_AVAILABILITY.md",
    "data/README.md",
    "models/MODEL_CARD.md",
    "models/lgbm_models.joblib",
    "results/paired_identical_molecule_metrics.csv",
    "results/strict_external_baseline_comparison.csv",
    "data/derived/cross_species_assay_matched_pairs.csv",
    "data/derived/same_assay_species_effect_summary.csv",
]
for name in required:
    check((ROOT / name).is_file(), f"missing required file: {name}")

prohibited_suffixes = {".pdf", ".docx", ".xlsx", ".xls"}
prohibited = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in prohibited_suffixes]
check(not prohibited, f"prohibited publisher/office files present: {prohibited}")

text_suffixes = {".md", ".py", ".txt", ".json", ".csv", ".cff", ".yml", ".yaml"}
absolute_path_hits: list[str] = []
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in text_suffixes and path.stat().st_size < 5_000_000:
        try:
            if ("/" + "Users/") in path.read_text(encoding="utf-8"):
                absolute_path_hits.append(str(path.relative_to(ROOT)))
        except UnicodeDecodeError:
            pass
check(not absolute_path_hits, f"local absolute paths found: {absolute_path_hits}")

pair_rows = rows("data/derived/cross_species_assay_matched_pairs.csv")
check(len(pair_rows) == 97, f"cross-species pair count is {len(pair_rows)}, expected 97")
check(all(row.get("manual_review_status") == "verified" for row in pair_rows), "not all cross-species pairs are manually verified")

species_rows = rows("data/derived/same_assay_species_effect_summary.csv")
overall = {(row["species"], row["transformation"]): row for row in species_rows}
check(overall.get(("HLM", "all"), {}).get("n_pairs") == "476", "HLM same-assay pair count mismatch")
check(overall.get(("RLM", "all"), {}).get("n_pairs") == "209", "RLM same-assay pair count mismatch")

summary = rows("data/derived/cross_species_assay_matched_summary.csv")[0]
check(summary.get("n_pairs") == "97", "cross-species summary pair count mismatch")
check(close(summary.get("delta_hlm_delta_rlm_spearman", "nan"), 0.430522), "cross-species Spearman mismatch")
check(close(summary.get("species_opposite_direction", "nan"), 0.381443), "cross-species reversal fraction mismatch")

internal = rows("results/paired_identical_molecule_metrics.csv")
internal_index = {(row["scope"], row["model"]): row for row in internal}
check(close(internal_index[("fluorinated", "structure-only")]["interval_rmse"], 0.453226), "internal structure-only RMSE mismatch")
check(close(internal_index[("fluorinated", "RLM-anchored residual")]["interval_rmse"], 0.303264), "internal residual RMSE mismatch")

external = rows("results/strict_external_baseline_comparison.csv")
external_index = {(row["analysis_set"], row["model"]): row for row in external}
analysis = "strict_total_no_unit_or_stereo_flags"
check(external_index[(analysis, "RLM-anchored residual")]["n"] == "598", "strict external n mismatch")
check(close(external_index[(analysis, "structure-only")]["rmse"], 1.002877), "external structure-only RMSE mismatch")
check(close(external_index[(analysis, "RLM-only linear")]["rmse"], 0.705031), "external linear RLM RMSE mismatch")
check(close(external_index[(analysis, "RLM-anchored residual")]["rmse"], 0.615317), "external residual RMSE mismatch")
check(close(external_index[(analysis, "RLM-anchored residual")]["low_clearance_auc"], 0.863819), "external residual ROC-AUC mismatch")

model = ROOT / "models/lgbm_models.joblib"
if model.is_file():
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    check(digest == "9e06ffa724c5de4eb2d830bf20a982d0e249ec36e9b925b868338ca57811c14a", "model SHA-256 mismatch")

manifest = ROOT / "MANIFEST.sha256"
if manifest.is_file():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / relative
        check(target.is_file(), f"manifest target missing: {relative}")
        if target.is_file():
            observed = hashlib.sha256(target.read_bytes()).hexdigest()
            check(observed == expected, f"manifest checksum mismatch: {relative}")

if FAILURES:
    print(f"Release audit failed: {len(FAILURES)} issue(s)")
    for failure in FAILURES:
        print(f"- {failure}")
    raise SystemExit(1)

print("Release audit passed: required files, provenance, pair review, primary metrics, and model checksum verified.")
