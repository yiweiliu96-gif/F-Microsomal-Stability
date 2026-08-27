#!/usr/bin/env python3
"""Document-clustered comparisons for the external algorithm benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "rlm_anchor_algorithm_benchmark" / "external_predictions.csv"
OUT = ROOT / "reports" / "rlm_anchor_algorithm_benchmark" / "document_bootstrap_vs_lightgbm.csv"
SEED = 20260824
ITERATIONS = 10000
HLM = "LOG HLM_CLint (mL/min/kg)"


def main() -> None:
    data = pd.read_csv(SOURCE)
    observed = data[HLM].to_numpy(dtype=float)
    groups = data["doc_id"].drop_duplicates().to_numpy()
    indices = {group: data.index[data["doc_id"].eq(group)].to_numpy() for group in groups}
    rng = np.random.default_rng(SEED)
    rows = []
    candidates = [
        column.removeprefix("prediction__")
        for column in data
        if column.startswith("prediction__") and column != "prediction__LightGBM"
    ]
    baseline = data["prediction__LightGBM"].to_numpy(dtype=float)
    for candidate in candidates:
        prediction = data[f"prediction__{candidate}"].to_numpy(dtype=float)
        observed_delta = (
            mean_squared_error(observed, prediction) ** 0.5
            - mean_squared_error(observed, baseline) ** 0.5
        )
        draws = []
        for _ in range(ITERATIONS):
            sampled = rng.choice(groups, len(groups), replace=True)
            selected = np.concatenate([indices[group] for group in sampled])
            draws.append(
                mean_squared_error(observed[selected], prediction[selected]) ** 0.5
                - mean_squared_error(observed[selected], baseline[selected]) ** 0.5
            )
        draws = np.asarray(draws)
        rows.append(
            {
                "baseline": "LightGBM",
                "candidate": candidate,
                "metric": "RMSE",
                "delta_candidate_minus_baseline": observed_delta,
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "two_sided_p": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
                "bootstrap_unit": "ChEMBL document",
                "bootstrap_iterations": ITERATIONS,
            }
        )
    result = pd.DataFrame(rows).sort_values("delta_candidate_minus_baseline")
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
