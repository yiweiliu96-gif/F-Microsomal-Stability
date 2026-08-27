#!/usr/bin/env python3
"""Five-fold Murcko-scaffold evaluation for paired Biogen HLM/RLM models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_biogen_paired_models as benchmark


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "biogen_paired_study" / "biogen_paired_modelling_table.csv"
OUT = ROOT / "reports" / "biogen_paired_scaffold_cv"
HLM = benchmark.HLM
RLM = benchmark.RLM


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = benchmark.add_features(pd.read_csv(DATA))
    eligible = data[data[HLM].notna()].copy().reset_index(drop=True)
    general = benchmark.representation_columns(data, False)
    fluorine = benchmark.representation_columns(data, True)
    prediction_rows = []

    for fold in range(5):
        train = eligible[eligible["scaffold_fold"].ne(fold)].copy()
        test = eligible[eligible["scaffold_fold"].eq(fold)].copy()
        output = test[
            [
                "Internal ID",
                "canonical_smiles",
                "murcko_scaffold",
                "scaffold_fold",
                "is_fluorinated",
                "fluorine_subgroup",
                "hlm_left_censored",
                "rlm_left_censored",
                HLM,
                RLM,
            ]
        ].copy()
        for representation, columns in (("general", general), ("fluorine_augmented", fluorine)):
            x_train, x_test = benchmark.prepare_matrices(train, test, columns)
            model = benchmark.lgbm(benchmark.SEEDS[fold % len(benchmark.SEEDS)])
            model.fit(x_train, train[HLM])
            output[f"structure_LGBM__{representation}"] = model.predict(x_test)

            train_paired = train[train[RLM].notna()].copy()
            test_paired_mask = test[RLM].notna()
            test_paired = test[test_paired_mask].copy()
            augmented = columns + [RLM, "rlm_left_censored"]
            x_train, x_test = benchmark.prepare_matrices(train_paired, test_paired, augmented)
            direct = benchmark.lgbm(benchmark.SEEDS[fold % len(benchmark.SEEDS)])
            direct.fit(x_train, train_paired[HLM])
            output.loc[test_paired_mask, f"measured_RLM_LGBM__{representation}"] = direct.predict(x_test)

            residual = benchmark.lgbm(benchmark.SEEDS[fold % len(benchmark.SEEDS)])
            residual.fit(x_train, train_paired[HLM] - train_paired[RLM])
            output.loc[test_paired_mask, f"RLM_anchor_residual_LGBM__{representation}"] = (
                test_paired[RLM].to_numpy() + residual.predict(x_test)
            )
        prediction_rows.append(output)
        print(f"completed scaffold fold {fold}", flush=True)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(OUT / "scaffold_oof_predictions.csv", index=False)
    prediction_columns = [column for column in predictions if column.startswith(("structure_", "measured_", "RLM_anchor_"))]
    metric_rows = []
    for column in prediction_columns:
        for scope, mask in (
            ("all", predictions[column].notna()),
            ("fluorinated", predictions[column].notna() & predictions["is_fluorinated"].eq(1)),
            ("nonfluorinated", predictions[column].notna() & predictions["is_fluorinated"].eq(0)),
        ):
            frame = predictions.loc[mask].reset_index(drop=True)
            metric_rows.append({"model": column, "scope": scope, **benchmark.metrics(frame, frame[column].to_numpy())})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "scaffold_oof_metrics.csv", index=False)

    comparisons = []
    for baseline, candidate in (
        ("structure_LGBM__general", "measured_RLM_LGBM__general"),
        ("measured_RLM_LGBM__general", "measured_RLM_LGBM__fluorine_augmented"),
        ("RLM_anchor_residual_LGBM__general", "RLM_anchor_residual_LGBM__fluorine_augmented"),
    ):
        available = predictions[baseline].notna() & predictions[candidate].notna()
        for scope, scope_mask in (
            ("all", np.ones(len(predictions), dtype=bool)),
            ("fluorinated", predictions["is_fluorinated"].eq(1).to_numpy()),
            ("nonfluorinated", predictions["is_fluorinated"].eq(0).to_numpy()),
        ):
            frame = predictions.loc[available & scope_mask].reset_index(drop=True)
            for metric in ("uncensored_rmse", "interval_rmse", "low_clearance_roc_auc"):
                comparisons.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "scope": scope,
                        "n": len(frame),
                        **benchmark.bootstrap_delta(
                            frame,
                            frame[baseline].to_numpy(),
                            frame[candidate].to_numpy(),
                            metric,
                        ),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUT / "scaffold_oof_bootstrap.csv", index=False)
    metadata = {
        "split": "Five-fold GroupKFold by standardized Bemis-Murcko scaffold",
        "selection": "No scaffold-fold predictions were used for hash-test model selection.",
        "models": prediction_columns,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
