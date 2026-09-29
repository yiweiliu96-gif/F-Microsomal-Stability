#!/usr/bin/env python3
"""Verify integrity of the public release (data + code + model)."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


required = [
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "data/README.md",
    "models/MODEL_CARD.md",
    "models/lgbm_models.joblib",
    "data/development/ADME_HLM_train.csv",
    "data/development/ADME_HLM_test.csv",
    "data/development/ADME_RLM_train.csv",
    "data/development/ADME_RLM_test.csv",
    "data/derived/biogen_paired_modelling_table.csv",
    "data/derived/external_predictions.csv",
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

print("Release audit passed: required files, model checksum, prohibited files, and absolute paths verified.")
