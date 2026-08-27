#!/usr/bin/env python3
"""Post hoc algorithm robustness benchmark on the frozen strict external set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_biogen_paired_models as core


TRAIN = ROOT / "reports" / "biogen_paired_study" / "biogen_paired_modelling_table.csv"
EXTERNAL_PREDICTIONS = ROOT / "reports" / "openadmet_chembl35_paired_external" / "external_predictions.csv"
EXTERNAL_AUDIT = ROOT / "reports" / "openadmet_chembl35_paired_external" / "external_endpoint_semantics_audit.csv"
OUT = ROOT / "reports" / "rlm_anchor_algorithm_benchmark"
SEEDS = (20260819, 20260820, 20260821)


def models(seed: int) -> dict[str, object]:
    def bounded_scaling() -> tuple[object, ...]:
        return (
            VarianceThreshold(threshold=1e-12),
            StandardScaler(),
            FunctionTransformer(np.clip, kw_args={"a_min": -10.0, "a_max": 10.0}),
        )

    return {
        "Ridge": make_pipeline(*bounded_scaling(), Ridge(alpha=10.0, solver="lsqr")),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=5, max_features=0.33,
            random_state=seed, n_jobs=4,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=5, max_features=0.5,
            random_state=seed, n_jobs=4,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=550, learning_rate=0.025, num_leaves=31,
            min_child_samples=25, reg_lambda=3.0, subsample=0.9,
            colsample_bytree=0.8, random_state=seed, n_jobs=4, verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=550, learning_rate=0.025, max_depth=6,
            min_child_weight=25, reg_lambda=3.0, subsample=0.9,
            colsample_bytree=0.8, objective="reg:squarederror",
            random_state=seed, n_jobs=4,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=550, learning_rate=0.025, depth=6,
            l2_leaf_reg=3.0, random_seed=seed, verbose=False,
            thread_count=4, allow_writing_files=False,
        ),
        "descriptor_MLP": make_pipeline(
            *bounded_scaling(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-3,
                batch_size=128, learning_rate_init=1e-3, max_iter=250,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed,
            ),
        ),
    }


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    low = (observed <= core.THRESHOLD).astype(int)
    return {
        "rmse": float(mean_squared_error(observed, prediction) ** 0.5),
        "mae": float(mean_absolute_error(observed, prediction)),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "low_clearance_auc": float(roc_auc_score(low, -prediction)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN)
    external = pd.read_csv(EXTERNAL_PREDICTIONS)
    audit = pd.read_csv(EXTERNAL_AUDIT)[
        ["external_record_id", "strict_total_intrinsic_pair", "source_unit_per_ug_flag", "full_inchikey_overlap"]
    ]
    external = external.merge(audit, on="external_record_id", validate="one_to_one")
    external = external.loc[
        external["is_fluorinated"].eq(1)
        & external["strict_total_intrinsic_pair"].eq(1)
        & external["source_unit_per_ug_flag"].eq(0)
        & external["full_inchikey_overlap"].eq(1)
    ].copy()
    if len(external) != 598:
        raise RuntimeError(f"Frozen strict fluorinated external set changed: expected 598, found {len(external)}")
    external[core.HLM] = external["external_hlm"]
    external[core.RLM] = external["external_rlm"]
    external["rlm_left_censored"] = 0
    combined = pd.concat(
        [train.assign(_dataset="train"), external.assign(_dataset="external")],
        ignore_index=True, sort=False,
    )
    combined = core.add_features(combined)
    train = combined.loc[
        combined["_dataset"].eq("train") & combined[core.HLM].notna() & combined[core.RLM].notna()
    ].copy()
    external = combined.loc[combined["_dataset"].eq("external")].copy().reset_index(drop=True)
    columns = core.representation_columns(combined, fluorine_augmented=False) + [core.RLM, "rlm_left_censored"]
    x_train, x_external = core.prepare_matrices(train, external, columns)
    target = train[core.HLM] - train[core.RLM]
    observed = external[core.HLM].to_numpy(dtype=float)

    prediction_table = external[
        ["external_record_id", "canonical_smiles", "doc_id", "doc_doi", core.HLM, core.RLM]
    ].copy()
    seed_rows, timing_rows = [], []
    ensembles: dict[str, list[np.ndarray]] = {}
    for seed in SEEDS:
        for name, model in models(seed).items():
            started = time.perf_counter()
            model.fit(x_train, target)
            prediction = external[core.RLM].to_numpy(dtype=float) + model.predict(x_external)
            elapsed = time.perf_counter() - started
            ensembles.setdefault(name, []).append(prediction)
            seed_rows.append({"model": name, "seed": seed, **metrics(observed, prediction)})
            timing_rows.append({"model": name, "seed": seed, "fit_and_predict_seconds": elapsed})
            print(name, seed, seed_rows[-1], flush=True)

    ensemble_rows = []
    for name, values in ensembles.items():
        prediction = np.mean(values, axis=0)
        prediction_table[f"prediction__{name}"] = prediction
        per_seed = pd.DataFrame([row for row in seed_rows if row["model"] == name])
        ensemble_rows.append(
            {
                "model": name,
                "seeds": ";".join(map(str, SEEDS)),
                **metrics(observed, prediction),
                "seed_rmse_min": float(per_seed["rmse"].min()),
                "seed_rmse_max": float(per_seed["rmse"].max()),
            }
        )
    pd.DataFrame(seed_rows).to_csv(OUT / "seed_metrics.csv", index=False)
    pd.DataFrame(ensemble_rows).sort_values("rmse").to_csv(OUT / "ensemble_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT / "runtime.csv", index=False)
    prediction_table.to_csv(OUT / "external_predictions.csv", index=False)
    metadata = {
        "status": "Post hoc algorithm robustness analysis; not used to select the prespecified primary model.",
        "training_set": "All paired Biogen HLM/RLM records.",
        "test_set": "Frozen 598-compound strict total-intrinsic fluorinated ChEMBL35 set.",
        "representation": "ECFP4 plus RDKit descriptors, measured RLM, and RLM censoring indicator; no handcrafted fluorine descriptors.",
        "target": "HLM minus measured RLM residual; prediction equals measured RLM plus fitted residual.",
        "external_label_use": "No hyperparameter tuning, sample filtering, or model selection used external endpoint values.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nEnsembles")
    print(pd.DataFrame(ensemble_rows).sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
