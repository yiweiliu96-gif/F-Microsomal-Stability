#!/usr/bin/env python3
"""Evaluate whether measured RLM fluorination changes predict HLM changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "assay_matched_fluorination_pairs" / "cross_species_assay_matched_pairs.csv"
OUT = ROOT / "reports" / "cross_species_fluorination_delta_translation"
SEED = 20260824
BOOTSTRAPS = 10000


def add_network_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left, right in frame[["base_smiles", "fluorinated_smiles"]].itertuples(index=False):
        union(left, right)
    roots = {find(value) for value in parent}
    labels = {root: f"PAIRNET_{index:04d}" for index, root in enumerate(sorted(roots), start=1)}
    result = frame.copy()
    result["pair_network_id"] = result["base_smiles"].map(lambda value: labels[find(value)])
    return result


def model_definitions() -> dict[str, object]:
    return {
        "no_change": None,
        "training_mean": "mean",
        "delta_RLM_OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "delta_RLM_Huber": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.01)),
        "delta_RLM_plus_transform_Ridge": make_pipeline(
            ColumnTransformer(
                [
                    ("delta", StandardScaler(), ["delta_rlm"]),
                    ("transform", OneHotEncoder(handle_unknown="ignore"), ["transformation"]),
                ]
            ),
            Ridge(alpha=10.0),
        ),
    }


def predictions(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    x = frame[["delta_rlm", "transformation"]]
    y = frame["delta_hlm"].to_numpy(dtype=float)
    groups = frame[group_column].to_numpy()
    splitter = GroupKFold(n_splits=5)
    output = frame[["pair_id", "document_id", "pair_network_id", "transformation", "delta_hlm", "delta_rlm"]].copy()
    output["split_strategy"] = group_column
    fold_assignment = np.zeros(len(frame), dtype=int)
    models = model_definitions()
    model_predictions = {name: np.zeros(len(frame), dtype=float) for name in models}
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        fold_assignment[test] = fold
        for name, model in models.items():
            if model is None:
                model_predictions[name][test] = 0.0
            elif model == "mean":
                model_predictions[name][test] = float(np.mean(y[train]))
            else:
                train_x = x.iloc[train] if "transform" in name else x[["delta_rlm"]].iloc[train]
                test_x = x.iloc[test] if "transform" in name else x[["delta_rlm"]].iloc[test]
                model.fit(train_x, y[train])
                model_predictions[name][test] = model.predict(test_x)
    output["fold"] = fold_assignment
    for name, values in model_predictions.items():
        output[f"prediction__{name}"] = values
    return output


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rho = np.nan
    if np.unique(prediction).size > 1:
        rho = float(spearmanr(y, prediction).statistic)
    actual_improved = y < 0
    predicted_improved = prediction < 0
    return {
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "spearman": rho,
        "improvement_direction_accuracy": float(np.mean(actual_improved == predicted_improved)),
        "predicted_improved_fraction": float(np.mean(predicted_improved)),
    }


def summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = oof["delta_hlm"].to_numpy(dtype=float)
    models = [column.removeprefix("prediction__") for column in oof if column.startswith("prediction__")]
    metrics = []
    for model in models:
        metrics.append(
            {
                "split_strategy": oof["split_strategy"].iloc[0],
                "model": model,
                "n_pairs": len(oof),
                "n_groups": oof[oof["split_strategy"].iloc[0]].nunique(),
                **metric_row(y, oof[f"prediction__{model}"].to_numpy(dtype=float)),
            }
        )
    metric_table = pd.DataFrame(metrics)

    cluster_column = oof["split_strategy"].iloc[0]
    clusters = [group.index.to_numpy() for _, group in oof.groupby(cluster_column, sort=False)]
    reference = oof["prediction__no_change"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    rows = []
    for model in models:
        if model == "no_change":
            continue
        pred = oof[f"prediction__{model}"].to_numpy(dtype=float)
        deltas = []
        for _ in range(BOOTSTRAPS):
            selected = rng.integers(0, len(clusters), len(clusters))
            indices = np.concatenate([clusters[index] for index in selected])
            model_rmse = mean_squared_error(y[indices], pred[indices]) ** 0.5
            reference_rmse = mean_squared_error(y[indices], reference[indices]) ** 0.5
            deltas.append(model_rmse - reference_rmse)
        rows.append(
            {
                "split_strategy": cluster_column,
                "model": model,
                "reference": "no_change",
                "rmse_delta_model_minus_reference": float(np.mean((y - pred) ** 2) ** 0.5 - np.mean(y**2) ** 0.5),
                "rmse_delta_ci95_low": float(np.quantile(deltas, 0.025)),
                "rmse_delta_ci95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_probability_model_better": float(np.mean(np.asarray(deltas) < 0)),
            }
        )
    return metric_table, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = add_network_ids(pd.read_csv(SOURCE))
    data = data.rename(
        columns={
            "assay_matched_delta_hlm": "delta_hlm",
            "assay_matched_delta_rlm": "delta_rlm",
        }
    )
    all_oof, all_metrics, all_bootstrap = [], [], []
    for group_column in ["document_id", "pair_network_id"]:
        oof = predictions(data.reset_index(drop=True), group_column)
        metrics, bootstrap = summarize(oof)
        all_oof.append(oof)
        all_metrics.append(metrics)
        all_bootstrap.append(bootstrap)
    predictions_table = pd.concat(all_oof, ignore_index=True)
    metrics_table = pd.concat(all_metrics, ignore_index=True)
    bootstrap_table = pd.concat(all_bootstrap, ignore_index=True)
    predictions_table.to_csv(OUT / "cross_validated_predictions.csv", index=False)
    metrics_table.to_csv(OUT / "cross_validated_metrics.csv", index=False)
    bootstrap_table.to_csv(OUT / "paired_cluster_bootstrap.csv", index=False)
    metadata = {
        "target": "Delta HLM = log10 CLint(fluorinated) - log10 CLint(parent).",
        "primary_predictor": "Measured Delta RLM for the identical fluorination transform.",
        "validation": "Five-fold grouped cross-validation, repeated separately by ChEMBL document and connected molecular-pair network.",
        "baselines": "No-change prediction and training-fold mean.",
        "scope": "Exploratory low-capacity translation analysis; 98 pairs are insufficient for a defensible GNN benchmark.",
        "bootstrap": f"{BOOTSTRAPS} resamples clustered by the corresponding validation grouping.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metrics_table.to_string(index=False))
    print("\nRMSE comparisons against no-change baseline")
    print(bootstrap_table.to_string(index=False))


if __name__ == "__main__":
    main()
