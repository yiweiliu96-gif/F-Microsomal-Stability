#!/usr/bin/env python3
"""Recalculate internal model comparisons on identical paired molecules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "biogen_paired_scaffold_cv"
PREDICTIONS = REPORT / "scaffold_oof_predictions.csv"
OUT = REPORT / "paired_identical_molecule_metrics.csv"
BOOTSTRAP_OUT = REPORT / "paired_scaffold_bootstrap_comparisons.csv"
HLM = "LOG HLM_CLint (mL/min/kg)"
BOUNDARY = 0.675686709
THRESHOLD = 1.6651264089380924
MODELS = {
    "structure-only": "structure_LGBM__general",
    "direct RLM-assisted": "measured_RLM_LGBM__general",
    "RLM-anchored residual": "RLM_anchor_residual_LGBM__general",
}


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    observed = frame[HLM].to_numpy()
    predicted = frame[column].to_numpy()
    censored = frame["hlm_left_censored"].astype(bool).to_numpy()
    exact = ~censored
    residual = np.where(censored, np.maximum(predicted - BOUNDARY, 0.0), predicted - observed)
    return {
        "uncensored_rmse": float(np.sqrt(mean_squared_error(observed[exact], predicted[exact]))),
        "uncensored_mae": float(mean_absolute_error(observed[exact], predicted[exact])),
        "uncensored_spearman": float(spearmanr(observed[exact], predicted[exact]).statistic),
        "interval_rmse": float(np.sqrt(np.mean(residual**2))),
        "low_clearance_auc": float(roc_auc_score((observed <= THRESHOLD).astype(int), -predicted)),
    }


def scaffold_bootstrap(frame: pd.DataFrame, baseline: str, candidate: str, iterations: int = 10000) -> list[dict[str, object]]:
    scaffolds = frame["murcko_scaffold"].drop_duplicates().to_numpy()
    rows = {scaffold: frame.index[frame["murcko_scaffold"].eq(scaffold)].to_numpy() for scaffold in scaffolds}
    rng = np.random.default_rng(20260824)
    observed_baseline = metrics(frame, baseline)
    observed_candidate = metrics(frame, candidate)
    samples = {name: [] for name in observed_baseline}
    for _ in range(iterations):
        sampled_scaffolds = rng.choice(scaffolds, len(scaffolds), replace=True)
        index = np.concatenate([rows[scaffold] for scaffold in sampled_scaffolds])
        sample = frame.loc[index]
        try:
            baseline_values = metrics(sample, baseline)
            candidate_values = metrics(sample, candidate)
        except ValueError:
            continue
        for name in samples:
            samples[name].append(candidate_values[name] - baseline_values[name])
    output = []
    for name, values in samples.items():
        values = np.asarray(values)
        output.append(
            {
                "metric": name,
                "delta_candidate_minus_baseline": observed_candidate[name] - observed_baseline[name],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "two_sided_p": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
                "bootstrap_unit": "Bemis-Murcko scaffold",
                "bootstrap_iterations": iterations,
            }
        )
    return output


def main() -> None:
    data = pd.read_csv(PREDICTIONS)
    paired = data[list(MODELS.values())].notna().all(axis=1)
    rows = []
    bootstrap_rows = []
    for scope, scope_mask in (
        ("fluorinated", data["is_fluorinated"].eq(1)),
        ("nonfluorinated", data["is_fluorinated"].eq(0)),
    ):
        frame = data.loc[paired & scope_mask].reset_index(drop=True)
        for label, column in MODELS.items():
            rows.append(
                {
                    "scope": scope,
                    "model": label,
                    "prediction_column": column,
                    "n": len(frame),
                    "n_scaffolds": frame["murcko_scaffold"].nunique(),
                    **metrics(frame, column),
                }
            )
        for candidate_label in ("direct RLM-assisted", "RLM-anchored residual"):
            for result in scaffold_bootstrap(frame, MODELS["structure-only"], MODELS[candidate_label]):
                bootstrap_rows.append(
                    {
                        "scope": scope,
                        "n": len(frame),
                        "n_scaffolds": frame["murcko_scaffold"].nunique(),
                        "baseline_label": "structure-only",
                        "candidate_label": candidate_label,
                        **result,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
