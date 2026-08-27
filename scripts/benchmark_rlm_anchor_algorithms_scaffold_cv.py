#!/usr/bin/env python3
"""Benchmark RLM-anchored residual algorithms on identical scaffold folds."""

from __future__ import annotations

import json
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_biogen_paired_models as core
from benchmark_rlm_anchor_algorithms_external import SEEDS, models


DATA = ROOT / "reports" / "biogen_paired_study" / "biogen_paired_modelling_table.csv"
OUT = ROOT / "reports" / "rlm_anchor_algorithm_scaffold_cv"
ITERATIONS = 10000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = core.add_features(pd.read_csv(DATA))
    data = data.loc[data[core.HLM].notna() & data[core.RLM].notna()].copy().reset_index(drop=True)
    if len(data) != 3049 or int(data["is_fluorinated"].sum()) != 557:
        raise RuntimeError("The frozen paired Biogen analysis set has changed.")
    columns = core.representation_columns(data, fluorine_augmented=False) + [core.RLM, "rlm_left_censored"]
    model_names = [
        "LightGBM", "XGBoost", "CatBoost", "RandomForest",
        "ExtraTrees", "Ridge", "descriptor_MLP",
    ]
    predictions = data[
        [
            "Internal ID", "canonical_smiles", "murcko_scaffold", "scaffold_fold",
            "is_fluorinated", "fluorine_subgroup", "hlm_left_censored", "rlm_left_censored",
            core.HLM, core.RLM,
        ]
    ].copy()
    runtime_rows = []
    for name in model_names:
        for seed in SEEDS:
            predictions[f"prediction__{name}__seed_{seed}"] = np.nan

    for fold in sorted(data["scaffold_fold"].dropna().astype(int).unique()):
        train = data.loc[data["scaffold_fold"].ne(fold)].copy()
        test = data.loc[data["scaffold_fold"].eq(fold)].copy()
        x_train, x_test = core.prepare_matrices(train, test, columns)
        target = train[core.HLM] - train[core.RLM]
        for seed in SEEDS:
            for name in model_names:
                # Instantiate one estimator at a time so fitted forests are not retained in memory.
                model = models(seed)[name]
                parameters = model.get_params(deep=False)
                if "n_jobs" in parameters:
                    model.set_params(n_jobs=1)
                if "thread_count" in parameters:
                    model.set_params(thread_count=1)
                # Ridge is deterministic; duplicate columns are retained for a uniform output schema.
                started = time.perf_counter()
                model.fit(x_train, target)
                prediction = test[core.RLM].to_numpy(dtype=float) + model.predict(x_test)
                predictions.loc[test.index, f"prediction__{name}__seed_{seed}"] = prediction
                runtime_rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "seed": seed,
                        "n_train": len(train),
                        "n_test": len(test),
                        "fit_and_predict_seconds": time.perf_counter() - started,
                    }
                )
                print(f"fold={fold} model={name} seed={seed}", flush=True)
                del model
                gc.collect()
        predictions.to_csv(OUT / "scaffold_oof_predictions.checkpoint.csv", index=False)

    metric_rows = []
    for name in model_names:
        seed_columns = [f"prediction__{name}__seed_{seed}" for seed in SEEDS]
        predictions[f"prediction__{name}__ensemble"] = predictions[seed_columns].mean(axis=1)
        for seed, column in [(str(seed), f"prediction__{name}__seed_{seed}") for seed in SEEDS] + [
            ("ensemble", f"prediction__{name}__ensemble")
        ]:
            for scope, mask in [
                ("all", np.ones(len(predictions), dtype=bool)),
                ("fluorinated", predictions["is_fluorinated"].eq(1).to_numpy()),
                ("nonfluorinated", predictions["is_fluorinated"].eq(0).to_numpy()),
            ]:
                frame = predictions.loc[mask].reset_index(drop=True)
                metric_rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "scope": scope,
                        "n": len(frame),
                        "n_scaffolds": frame["murcko_scaffold"].nunique(),
                        **core.metrics(frame, frame[column].to_numpy(dtype=float)),
                    }
                )

    ensemble_metrics = pd.DataFrame(metric_rows)
    comparison_frame = predictions.loc[predictions["is_fluorinated"].eq(1)].reset_index(drop=True)
    groups = comparison_frame["murcko_scaffold"].drop_duplicates().to_numpy()
    indices = {group: comparison_frame.index[comparison_frame["murcko_scaffold"].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEEDS[0])
    observed = comparison_frame[core.HLM].to_numpy(dtype=float)
    censored = comparison_frame["hlm_left_censored"].astype(bool).to_numpy()
    boundary = float(comparison_frame[core.HLM].min())

    def interval_rmse(index: np.ndarray, prediction: np.ndarray) -> float:
        residual = np.where(
            censored[index],
            np.maximum(prediction[index] - boundary, 0.0),
            prediction[index] - observed[index],
        )
        return float(np.sqrt(np.mean(residual**2)))

    baseline = comparison_frame["prediction__LightGBM__ensemble"].to_numpy(dtype=float)
    bootstrap_rows = []
    for name in model_names:
        if name == "LightGBM":
            continue
        candidate = comparison_frame[f"prediction__{name}__ensemble"].to_numpy(dtype=float)
        draws = []
        for _ in range(ITERATIONS):
            sampled = rng.choice(groups, len(groups), replace=True)
            selected = np.concatenate([indices[group] for group in sampled])
            draws.append(interval_rmse(selected, candidate) - interval_rmse(selected, baseline))
        draws = np.asarray(draws)
        full_index = np.arange(len(comparison_frame))
        bootstrap_rows.append(
            {
                "baseline": "LightGBM",
                "candidate": name,
                "scope": "fluorinated",
                "metric": "interval_rmse",
                "delta_candidate_minus_baseline": interval_rmse(full_index, candidate) - interval_rmse(full_index, baseline),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "two_sided_p": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
                "bootstrap_unit": "Murcko scaffold",
                "bootstrap_iterations": ITERATIONS,
            }
        )

    predictions.to_csv(OUT / "scaffold_oof_predictions.csv", index=False)
    ensemble_metrics.to_csv(OUT / "scaffold_oof_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(OUT / "runtime.csv", index=False)
    pd.DataFrame(bootstrap_rows).sort_values("delta_candidate_minus_baseline").to_csv(
        OUT / "scaffold_bootstrap_vs_lightgbm.csv", index=False
    )
    metadata = {
        "task": "Predict HLM-RLM residual and add the prediction to measured RLM.",
        "split": "Frozen five-fold GroupKFold by standardized Bemis-Murcko scaffold.",
        "training_population": "All 3049 paired Biogen compounds; metrics reported on the identical 557 fluorinated compounds and 2492 nonfluorinated compounds.",
        "representation": "ECFP4 plus RDKit descriptors, measured RLM, and RLM censoring indicator; no handcrafted fluorine descriptors.",
        "hyperparameters": "Identical to the frozen external algorithm robustness benchmark; no external labels used for tuning.",
        "seeds": list(SEEDS),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nFluorinated ensemble metrics")
    print(
        ensemble_metrics.loc[
            ensemble_metrics["scope"].eq("fluorinated") & ensemble_metrics["seed"].eq("ensemble")
        ].sort_values("interval_rmse").to_string(index=False)
    )
    print("\nScaffold bootstrap")
    print(pd.DataFrame(bootstrap_rows).sort_values("delta_candidate_minus_baseline").to_string(index=False))


if __name__ == "__main__":
    main()
