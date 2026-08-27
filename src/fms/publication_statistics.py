"""Paired bootstrap comparisons and similarity-domain performance for V3 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="reports/v3_publication/statistics")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def bootstrap_difference(
    merged: pd.DataFrame, prob_a: str, prob_b: str, n_bootstrap: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    y = merged["label"].to_numpy()
    a = merged[prob_a].to_numpy()
    b = merged[prob_b].to_numpy()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(merged), len(merged))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    arr = np.asarray(diffs)
    return {
        "roc_auc_diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci_low_95": float(np.quantile(arr, 0.025)),
        "ci_high_95": float(np.quantile(arr, 0.975)),
        "p_two_sided": float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))),
        "n_bootstrap_valid": int(len(arr)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)
    id_cols = ["species", "compound_id", "smiles", "label", "scaffold_split"]
    comparisons = []

    # Same feature/model, specialized training versus general training.
    for (species, feature_set, model), group in pred.groupby(
        ["species", "feature_set", "model"]
    ):
        wide = group.pivot_table(
            index=id_cols,
            columns="training_scope",
            values="pred_prob_stable",
            aggfunc="first",
        ).reset_index()
        if {"fluorinated_only", "general_all_compounds"}.issubset(wide.columns):
            result = bootstrap_difference(
                wide,
                "fluorinated_only",
                "general_all_compounds",
                args.n_bootstrap,
                args.random_state,
            )
            comparisons.append(
                {
                    "comparison_type": "training_scope",
                    "species": species,
                    "feature_set": feature_set,
                    "model": model,
                    "a": "fluorinated_only",
                    "b": "general_all_compounds",
                    **result,
                }
            )

    # Within each training scope/model, all features versus fingerprint only.
    for (species, scope, model), group in pred.groupby(
        ["species", "training_scope", "model"]
    ):
        wide = group.pivot_table(
            index=id_cols,
            columns="feature_set",
            values="pred_prob_stable",
            aggfunc="first",
        ).reset_index()
        if {"all", "fingerprint_only"}.issubset(wide.columns):
            result = bootstrap_difference(
                wide, "all", "fingerprint_only", args.n_bootstrap, args.random_state
            )
            comparisons.append(
                {
                    "comparison_type": "feature_ablation",
                    "species": species,
                    "feature_set": "",
                    "model": model,
                    "training_scope": scope,
                    "a": "all",
                    "b": "fingerprint_only",
                    **result,
                }
            )
    pd.DataFrame(comparisons).to_csv(
        output_dir / "paired_bootstrap_comparisons.csv", index=False
    )

    # Similarity-based applicability domain for every fitted benchmark.
    domain_rows = []
    bins = [-np.inf, 0.4, 0.6, 0.8, np.inf]
    labels = ["<0.4", "0.4-<0.6", "0.6-<0.8", ">=0.8"]
    pred["similarity_bin"] = pd.cut(
        pred["max_tanimoto_to_training"], bins=bins, labels=labels, right=False
    )
    for keys, group in pred.groupby(
        ["species", "training_scope", "feature_set", "model", "similarity_bin"],
        observed=True,
    ):
        species, scope, feature_set, model, similarity_bin = keys
        if len(group) < 10:
            continue
        y = group["label"].astype(int)
        prob = group["pred_prob_stable"]
        predicted = (prob >= 0.5).astype(int)
        domain_rows.append(
            {
                "species": species,
                "training_scope": scope,
                "feature_set": feature_set,
                "model": model,
                "similarity_bin": similarity_bin,
                "n": len(group),
                "roc_auc": roc_auc_score(y, prob) if y.nunique() == 2 else np.nan,
                "mcc": matthews_corrcoef(y, predicted),
                "accuracy": np.mean(y.to_numpy() == predicted.to_numpy()),
                "mean_similarity": group["max_tanimoto_to_training"].mean(),
            }
        )
    pd.DataFrame(domain_rows).to_csv(
        output_dir / "similarity_domain_performance.csv", index=False
    )


if __name__ == "__main__":
    main()
