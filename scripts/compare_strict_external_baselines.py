#!/usr/bin/env python3
"""Compare frozen external models on endpoint-audited fluorinated subsets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "openadmet_chembl35_paired_external"
PREDICTIONS = REPORT / "external_predictions.csv"
AUDIT = REPORT / "external_endpoint_semantics_audit.csv"
OUT = REPORT / "strict_external_baseline_comparison.csv"
BOOTSTRAP_OUT = REPORT / "strict_external_baseline_bootstrap.csv"
THRESHOLD = 1.6651264089380924
MODELS = {
    "structure-only": "structure_LGBM__general",
    "RLM-only linear": "measured_RLM_ridge",
    "RLM constant offset": "RLM_anchor_constant_offset",
    "RLM-anchored residual": "RLM_anchor_residual_LGBM__general",
    "RLM-anchored residual + F descriptors": "RLM_anchor_residual_LGBM__fluorine_augmented",
}


def metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    observed = frame["external_hlm"].to_numpy()
    predicted = frame[prediction_column].to_numpy()
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "low_clearance_auc": float(roc_auc_score((observed <= THRESHOLD).astype(int), -predicted)),
    }


def bootstrap(frame: pd.DataFrame, baseline: str, candidate: str, iterations: int = 5000) -> list[dict[str, object]]:
    documents = frame["doc_id"].drop_duplicates().to_numpy()
    rows = {document: frame.index[frame["doc_id"].eq(document)].to_numpy() for document in documents}
    rng = np.random.default_rng(20260824)
    observed_baseline = metrics(frame, baseline)
    observed_candidate = metrics(frame, candidate)
    samples = {name: [] for name in observed_baseline}
    for _ in range(iterations):
        sampled_documents = rng.choice(documents, len(documents), replace=True)
        index = np.concatenate([rows[document] for document in sampled_documents])
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
                "baseline": baseline,
                "candidate": candidate,
                "metric": name,
                "delta_candidate_minus_baseline": observed_candidate[name] - observed_baseline[name],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "two_sided_p": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
                "bootstrap_unit": "ChEMBL document",
                "bootstrap_iterations": iterations,
            }
        )
    return output


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    audit = pd.read_csv(AUDIT)[
        [
            "external_record_id",
            "strict_total_intrinsic_pair",
            "source_unit_per_ug_flag",
            "full_inchikey_overlap",
        ]
    ]
    data = predictions.merge(audit, on="external_record_id", how="left", validate="one_to_one")
    subsets = {
        "strict_total_intrinsic": data["strict_total_intrinsic_pair"].eq(1),
        "strict_total_no_unit_or_stereo_flags": (
            data["strict_total_intrinsic_pair"].eq(1)
            & data["source_unit_per_ug_flag"].eq(0)
            & data["full_inchikey_overlap"].eq(1)
        ),
    }
    rows = []
    bootstraps = []
    for subset_name, subset_mask in subsets.items():
        frame = data.loc[subset_mask & data["is_fluorinated"].eq(1)].reset_index(drop=True)
        for label, column in MODELS.items():
            rows.append(
                {
                    "analysis_set": subset_name,
                    "scope": "fluorinated",
                    "model": label,
                    "prediction_column": column,
                    "n": len(frame),
                    "n_documents": frame["doc_id"].nunique(),
                    **metrics(frame, column),
                }
            )
        residual = MODELS["RLM-anchored residual"]
        for baseline_label in ("structure-only", "RLM-only linear", "RLM constant offset"):
            for row in bootstrap(frame, MODELS[baseline_label], residual):
                bootstraps.append(
                    {
                        "analysis_set": subset_name,
                        "scope": "fluorinated",
                        "n": len(frame),
                        "n_documents": frame["doc_id"].nunique(),
                        "baseline_label": baseline_label,
                        "candidate_label": "RLM-anchored residual",
                        **row,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(bootstraps).to_csv(BOOTSTRAP_OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(bootstraps).to_string(index=False))


if __name__ == "__main__":
    main()
