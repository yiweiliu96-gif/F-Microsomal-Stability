#!/usr/bin/env python3
"""Clustered GINE versus LightGBM comparisons on identical molecules."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
GINE = ROOT / "reports" / "rlm_anchor_gine_benchmark"
TABULAR_INTERNAL = ROOT / "reports" / "rlm_anchor_algorithm_scaffold_cv" / "scaffold_oof_predictions.csv"
TABULAR_EXTERNAL = ROOT / "reports" / "rlm_anchor_algorithm_benchmark" / "external_predictions.csv"
OUT = GINE / "cluster_bootstrap_vs_lightgbm.csv"
HLM = "LOG HLM_CLint (mL/min/kg)"
SEED = 20260824
ITERATIONS = 10000


def bootstrap_delta(frame: pd.DataFrame, cluster: str, interval: bool) -> dict[str, float]:
    observed = frame[HLM].to_numpy(dtype=float)
    baseline = frame["prediction__LightGBM__ensemble"].to_numpy(dtype=float)
    candidate = frame["prediction__GINE__ensemble"].to_numpy(dtype=float)
    censored = frame["hlm_left_censored"].astype(bool).to_numpy()
    boundary = float(frame[HLM].min())

    def rmse(index: np.ndarray, prediction: np.ndarray) -> float:
        if interval:
            residual = np.where(
                censored[index],
                np.maximum(prediction[index] - boundary, 0.0),
                prediction[index] - observed[index],
            )
            return float(np.sqrt(np.mean(residual**2)))
        return float(mean_squared_error(observed[index], prediction[index]) ** 0.5)

    groups = frame[cluster].drop_duplicates().to_numpy()
    indices = {group: frame.index[frame[cluster].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(groups, len(groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        draws.append(rmse(selected, candidate) - rmse(selected, baseline))
    draws = np.asarray(draws)
    full = np.arange(len(frame))
    return {
        "delta_gine_minus_lightgbm": rmse(full, candidate) - rmse(full, baseline),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "bootstrap_unit": cluster,
        "bootstrap_iterations": ITERATIONS,
    }


def main() -> None:
    internal_gine = pd.read_csv(GINE / "scaffold_oof_predictions.csv")
    internal_tabular = pd.read_csv(TABULAR_INTERNAL)[
        ["Internal ID", "prediction__LightGBM__ensemble"]
    ]
    internal = internal_gine.merge(internal_tabular, on="Internal ID", validate="one_to_one")
    internal = internal.loc[internal["is_fluorinated"].eq(1)].reset_index(drop=True)

    external_gine = pd.read_csv(GINE / "strict_external_predictions.csv")
    external_tabular = pd.read_csv(TABULAR_EXTERNAL)[
        ["external_record_id", "prediction__LightGBM"]
    ].rename(columns={"prediction__LightGBM": "prediction__LightGBM__ensemble"})
    external = external_gine.merge(external_tabular, on="external_record_id", validate="one_to_one")

    rows = [
        {
            "evaluation": "scaffold_cv",
            "scope": "fluorinated",
            "metric": "interval_rmse",
            "n": len(internal),
            **bootstrap_delta(internal, "murcko_scaffold", interval=True),
        },
        {
            "evaluation": "strict_external",
            "scope": "fluorinated",
            "metric": "rmse",
            "n": len(external),
            **bootstrap_delta(external, "doc_id", interval=False),
        },
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
