#!/usr/bin/env python3
"""Train prespecified paired HLM/RLM models on the Biogen public ADME set."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from fms.fluoro_features import calculate_fluoro_rdkit_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "biogen_paired_study" / "biogen_paired_modelling_table.csv"
OUT = ROOT / "reports" / "biogen_paired_models"
CACHE = OUT / "molecular_features.joblib"
HLM = "LOG HLM_CLint (mL/min/kg)"
RLM = "LOG RLM_CLint (mL/min/kg)"
THRESHOLD = 1.6651264089380924
SEEDS = (20260819, 20260820, 20260821)
BOOTSTRAPS = 10000


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def molecular_features(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    row = {f"rdkit_{name}": safe_float(function(mol)) for name, function in Descriptors.descList}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    bits = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    row.update({f"ecfp4_{index}": int(value) for index, value in enumerate(bits)})
    row.update(calculate_fluoro_rdkit_features(mol))
    return row


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = joblib.load(CACHE) if CACHE.exists() else {}
    missing = [value for value in data["canonical_smiles"] if value not in cache]
    for index, smiles in enumerate(missing, start=1):
        cache[smiles] = molecular_features(smiles)
        if index % 500 == 0:
            print(f"feature generation {index}/{len(missing)}")
    joblib.dump(cache, CACHE, compress=3)
    features = pd.DataFrame([cache[value] for value in data["canonical_smiles"]])
    features = features.drop(columns=[column for column in features if column in data], errors="ignore")
    return pd.concat([data.reset_index(drop=True), features], axis=1)


def representation_columns(data: pd.DataFrame, fluorine_augmented: bool) -> list[str]:
    columns = [column for column in data if column.startswith("rdkit_") or column.startswith("ecfp4_")]
    if fluorine_augmented:
        columns += [column for column in data if column.startswith("fluoro_")]
    return columns


def prepare_matrices(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[columns].replace([np.inf, -np.inf], np.nan)
    test_x = test[columns].replace([np.inf, -np.inf], np.nan)
    medians = train_x.median(axis=0).fillna(0.0)
    return train_x.fillna(medians), test_x.fillna(medians)


def lgbm(seed: int = SEEDS[0]) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=550,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=25,
        reg_lambda=3.0,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


class TobitMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(x)
        mu = output[:, 0]
        log_sigma = output[:, 1].clamp(-3.0, 1.5)
        return mu, log_sigma


def tobit_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    censored: torch.Tensor,
    boundary: float,
) -> torch.Tensor:
    sigma = torch.exp(log_sigma)
    exact_nll = 0.5 * ((target - mu) / sigma) ** 2 + log_sigma
    z_boundary = (boundary - mu) / sigma
    censored_nll = -torch.special.log_ndtr(z_boundary)
    return torch.where(censored, censored_nll, exact_nll).mean()


def fit_tobit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    x_train = pipeline.fit_transform(train[columns].replace([np.inf, -np.inf], np.nan)).astype("float32")
    x_validation = pipeline.transform(validation[columns].replace([np.inf, -np.inf], np.nan)).astype("float32")
    x_test = pipeline.transform(test[columns].replace([np.inf, -np.inf], np.nan)).astype("float32")

    exact_train = train.loc[train["hlm_left_censored"].eq(0), HLM]
    target_mean = float(exact_train.mean())
    target_sd = float(exact_train.std())
    boundary = float((train[HLM].min() - target_mean) / target_sd)
    y_train = ((train[HLM].to_numpy() - target_mean) / target_sd).astype("float32")
    y_validation = ((validation[HLM].to_numpy() - target_mean) / target_sd).astype("float32")

    tensors = {
        "x_train": torch.tensor(x_train),
        "y_train": torch.tensor(y_train),
        "c_train": torch.tensor(train["hlm_left_censored"].astype(bool).to_numpy()),
        "x_validation": torch.tensor(x_validation),
        "y_validation": torch.tensor(y_validation),
        "c_validation": torch.tensor(validation["hlm_left_censored"].astype(bool).to_numpy()),
    }
    model = TobitMLP(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    for epoch in range(1, 301):
        model.train()
        permutation = torch.randperm(len(train))
        train_losses = []
        for start in range(0, len(train), 128):
            index = permutation[start : start + 128]
            mu, log_sigma = model(tensors["x_train"][index])
            loss = tobit_loss(
                mu, log_sigma, tensors["y_train"][index], tensors["c_train"][index], boundary
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            mu, log_sigma = model(tensors["x_validation"])
            validation_loss = float(
                tobit_loss(
                    mu,
                    log_sigma,
                    tensors["y_validation"],
                    tensors["c_validation"],
                    boundary,
                )
            )
        history.append({"epoch": epoch, "train_loss": np.mean(train_losses), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 30:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled, log_sigma = model(torch.tensor(x_test))
    prediction = prediction_scaled.numpy() * target_sd + target_mean
    uncertainty = np.exp(log_sigma.numpy()) * target_sd
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_tobit_loss": best_loss,
        "target_mean": target_mean,
        "target_sd": target_sd,
        "prediction_sigma_mean": float(np.mean(uncertainty)),
        "history": history,
    }
    return prediction, metadata


def interval_rmse(y: np.ndarray, prediction: np.ndarray, censored: np.ndarray, boundary: float) -> float:
    residual = np.where(censored, np.maximum(prediction - boundary, 0.0), prediction - y)
    return float(np.sqrt(np.mean(residual**2)))


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    y = frame[HLM].to_numpy()
    censored = frame["hlm_left_censored"].astype(bool).to_numpy()
    exact = ~censored
    stable = (y <= THRESHOLD).astype(int)
    top_n = max(1, int(np.ceil(0.2 * len(frame))))
    top_index = np.argsort(prediction)[:top_n]
    return {
        "n": len(frame),
        "n_uncensored": int(exact.sum()),
        "uncensored_rmse": float(np.sqrt(mean_squared_error(y[exact], prediction[exact]))),
        "uncensored_mae": float(mean_absolute_error(y[exact], prediction[exact])),
        "uncensored_spearman": float(spearmanr(y[exact], prediction[exact]).statistic),
        "interval_rmse": interval_rmse(y, prediction, censored, float(frame[HLM].min())),
        "low_clearance_roc_auc": float(roc_auc_score(stable, -prediction)),
        "low_clearance_pr_auc": float(average_precision_score(stable, -prediction)),
        "top20_low_clearance_fraction": float(stable[top_index].mean()),
        "low_clearance_prevalence": float(stable.mean()),
        "top20_enrichment": float(stable[top_index].mean() / stable.mean()),
    }


def metric_value(frame: pd.DataFrame, prediction: np.ndarray, metric: str) -> float:
    y = frame[HLM].to_numpy()
    censored = frame["hlm_left_censored"].astype(bool).to_numpy()
    if metric == "uncensored_rmse":
        exact = ~censored
        return float(np.sqrt(mean_squared_error(y[exact], prediction[exact])))
    if metric == "interval_rmse":
        return interval_rmse(y, prediction, censored, float(frame[HLM].min()))
    if metric == "low_clearance_roc_auc":
        return float(roc_auc_score((y <= THRESHOLD).astype(int), -prediction))
    raise ValueError(metric)


def bootstrap_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
) -> dict[str, float]:
    rng = np.random.default_rng(SEEDS[0])
    values = []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[index].reset_index(drop=True)
        try:
            delta = metric_value(sample, candidate[index], metric) - metric_value(sample, baseline[index], metric)
        except ValueError:
            continue
        values.append(delta)
    values = np.asarray(values)
    observed = metric_value(frame, candidate, metric) - metric_value(frame, baseline, metric)
    return {
        "metric": metric,
        "delta_candidate_minus_baseline": float(observed),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
    }


def main() -> None:
    data = add_features(pd.read_csv(DATA))
    if data.columns.duplicated().any():
        raise RuntimeError(f"Duplicate feature columns: {data.columns[data.columns.duplicated()].tolist()}")
    print(f"loaded feature matrix: {data.shape}", flush=True)
    train_all = data[data["hash_hlm_split"].eq("train") & data[HLM].notna()].copy()
    test_all = data[data["hash_hlm_split"].eq("test") & data[HLM].notna()].copy().reset_index(drop=True)
    validation_mask = train_all["scaffold_fold"].eq(0)
    fit_all = train_all[~validation_mask].copy()
    validation_all = train_all[validation_mask].copy()

    general_columns = representation_columns(data, False)
    fluorine_columns = representation_columns(data, True)
    predictions = test_all[
        [
            "Internal ID",
            "canonical_smiles",
            "murcko_scaffold",
            "is_fluorinated",
            "fluorine_subgroup",
            "hash_hlm_split",
            "hlm_left_censored",
            "rlm_left_censored",
            HLM,
            RLM,
        ]
    ].copy()
    models = {}

    for representation, columns in (("general", general_columns), ("fluorine_augmented", fluorine_columns)):
        print(f"fitting LightGBM representation: {representation} ({len(columns)} features)", flush=True)
        x_train, x_test = prepare_matrices(train_all, test_all, columns)
        model = lgbm()
        model.fit(x_train, train_all[HLM])
        predictions[f"structure_LGBM__{representation}"] = model.predict(x_test)
        models[f"structure_LGBM__{representation}"] = {"model": model, "columns": columns}

        paired_train = train_all[train_all[RLM].notna()].copy()
        paired_test_mask = test_all[RLM].notna()
        paired_test = test_all[paired_test_mask].copy()
        augmented_columns = columns + [RLM, "rlm_left_censored"]
        x_train, x_test = prepare_matrices(paired_train, paired_test, augmented_columns)
        direct = lgbm()
        direct.fit(x_train, paired_train[HLM])
        direct_prediction = direct.predict(x_test)
        predictions.loc[paired_test_mask, f"measured_RLM_LGBM__{representation}"] = direct_prediction
        models[f"measured_RLM_LGBM__{representation}"] = {"model": direct, "columns": augmented_columns}

        residual = lgbm()
        residual.fit(x_train, paired_train[HLM] - paired_train[RLM])
        residual_prediction = paired_test[RLM].to_numpy() + residual.predict(x_test)
        predictions.loc[paired_test_mask, f"RLM_anchor_residual_LGBM__{representation}"] = residual_prediction
        models[f"RLM_anchor_residual_LGBM__{representation}"] = {"model": residual, "columns": augmented_columns}
        print(f"completed LightGBM representation: {representation}", flush=True)

    paired_train = train_all[train_all[RLM].notna()].copy()
    paired_test_mask = test_all[RLM].notna()
    paired_test = test_all[paired_test_mask].copy()
    linear = Ridge(alpha=1.0).fit(paired_train[[RLM]], paired_train[HLM])
    predictions.loc[paired_test_mask, "measured_RLM_linear"] = linear.predict(paired_test[[RLM]])

    tobit_metadata = []
    for task, task_train, task_validation, task_test, columns in (
        ("structure_Tobit_general", fit_all, validation_all, test_all, general_columns),
        (
            "measured_RLM_Tobit_general",
            fit_all[fit_all[RLM].notna()],
            validation_all[validation_all[RLM].notna()],
            paired_test,
            general_columns + [RLM, "rlm_left_censored"],
        ),
    ):
        print(f"fitting Tobit task: {task}", flush=True)
        seed_predictions = []
        for seed in SEEDS:
            prediction, metadata = fit_tobit(task_train, task_validation, task_test, columns, seed)
            seed_predictions.append(prediction)
            metadata["task"] = task
            tobit_metadata.append(metadata)
        ensemble = np.mean(seed_predictions, axis=0)
        if task.startswith("measured"):
            predictions.loc[paired_test_mask, task] = ensemble
        else:
            predictions[task] = ensemble
        print(f"completed Tobit task: {task}", flush=True)

    predictions.to_csv(OUT / "hash_test_predictions.csv", index=False)
    joblib.dump(models, OUT / "lgbm_models.joblib", compress=3)
    (OUT / "tobit_training_metadata.json").write_text(
        json.dumps(tobit_metadata, indent=2) + "\n", encoding="utf-8"
    )

    prediction_columns = [
        column
        for column in predictions
        if column.startswith(("structure_", "measured_", "RLM_anchor_"))
    ]
    metric_rows = []
    for column in prediction_columns:
        eligible = predictions[column].notna()
        for scope, scope_mask in (
            ("all", np.ones(len(predictions), dtype=bool)),
            ("fluorinated", predictions["is_fluorinated"].eq(1).to_numpy()),
            ("nonfluorinated", predictions["is_fluorinated"].eq(0).to_numpy()),
        ):
            mask = eligible & scope_mask
            frame = predictions.loc[mask].reset_index(drop=True)
            if len(frame) < 10:
                continue
            metric_rows.append({"model": column, "scope": scope, **metrics(frame, frame[column].to_numpy())})
    pd.DataFrame(metric_rows).to_csv(OUT / "hash_test_metrics.csv", index=False)

    comparisons = []
    paired_mask = predictions[RLM].notna()
    prespecified = (
        ("structure_LGBM__general", "measured_RLM_LGBM__general"),
        ("structure_LGBM__fluorine_augmented", "measured_RLM_LGBM__fluorine_augmented"),
        ("measured_RLM_LGBM__general", "measured_RLM_LGBM__fluorine_augmented"),
        ("measured_RLM_LGBM__general", "RLM_anchor_residual_LGBM__general"),
        ("structure_LGBM__general", "structure_Tobit_general"),
        ("measured_RLM_LGBM__general", "measured_RLM_Tobit_general"),
    )
    for baseline, candidate in prespecified:
        eligible = paired_mask if baseline.startswith("measured") or candidate.startswith(("measured", "RLM_anchor")) else np.ones(len(predictions), dtype=bool)
        eligible &= predictions[baseline].notna() & predictions[candidate].notna()
        for scope, scope_mask in (
            ("all", np.ones(len(predictions), dtype=bool)),
            ("fluorinated", predictions["is_fluorinated"].eq(1).to_numpy()),
            ("nonfluorinated", predictions["is_fluorinated"].eq(0).to_numpy()),
        ):
            mask = eligible & scope_mask
            frame = predictions.loc[mask].reset_index(drop=True)
            for metric in ("uncensored_rmse", "interval_rmse", "low_clearance_roc_auc"):
                comparisons.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "scope": scope,
                        "n": len(frame),
                        **bootstrap_delta(
                            frame,
                            frame[baseline].to_numpy(),
                            frame[candidate].to_numpy(),
                            metric,
                        ),
                    }
                )
    pd.DataFrame(comparisons).to_csv(OUT / "paired_bootstrap_comparisons.csv", index=False)
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
