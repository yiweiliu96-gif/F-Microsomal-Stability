"""Shared modelling helpers used by publication analysis modules."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


IDENTIFIER_COLS = {
    "compound_id",
    "name",
    "smiles",
    "canonical_smiles",
    "species",
    "article_class_unstable",
    "label",
    "split",
    "source",
    "valid_smiles",
    "descriptor_error",
    "descriptor_backend",
    "scaffold_split",
    "murcko_scaffold",
}


def normalize_species(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"HLM", "HUMAN", "HUMAN LIVER MICROSOME", "HUMAN LIVER MICROSOMES"}:
        return "HLM"
    if text in {"RLM", "RAT", "RAT LIVER MICROSOME", "RAT LIVER MICROSOMES"}:
        return "RLM"
    return text


def normalize_split_name(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "training": "train",
        "train": "train",
        "tr": "train",
        "test": "test",
        "external": "external",
        "validation": "validation",
        "valid": "validation",
        "val": "validation",
    }
    return aliases.get(text, text)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in IDENTIFIER_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def select_feature_set(cols: list[str], feature_set: str) -> list[str]:
    morgan = [c for c in cols if c.startswith("morgan_")]
    fluoro = [c for c in cols if c.startswith("fluoro_")]
    rdkit = [c for c in cols if c.startswith("rdkit_")]
    if feature_set == "rdkit_only":
        return rdkit
    if feature_set == "rdkit_fluoro":
        return rdkit + fluoro
    if feature_set == "fingerprint_only":
        return morgan
    if feature_set in {"all", "fluorine_aware_all"}:
        return rdkit + fluoro + morgan
    if feature_set == "fluoro_only":
        return fluoro
    if feature_set == "interpretable_fluoro":
        return rdkit + fluoro
    raise ValueError(f"Unsupported feature set: {feature_set}")


def _safe_metric(fn, y_true, values) -> float:
    try:
        return float(fn(y_true, values))
    except ValueError:
        return math.nan


def safe_metrics(y_true, pred, prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, prob),
        "pr_auc": _safe_metric(average_precision_score, y_true, prob),
    }
