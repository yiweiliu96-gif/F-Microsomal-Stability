"""Fair general-vs-fluorinated benchmark on the same fluorinated scaffold holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from .train import numeric_feature_columns, safe_metrics, select_feature_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-input", required=True)
    parser.add_argument("--fluoro-input", required=True)
    parser.add_argument("--output-dir", default="reports/v3_publication/benchmark")
    parser.add_argument("--split-col", default="scaffold_split")
    parser.add_argument("--models", nargs="+", default=["extra_trees", "rf"])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["rdkit_only", "rdkit_fluoro", "fingerprint_only", "all"],
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def make_model(name: str, args: argparse.Namespace) -> Pipeline:
    cls = ExtraTreesClassifier if name == "extra_trees" else RandomForestClassifier
    if name not in {"extra_trees", "rf"}:
        raise ValueError(f"Unsupported model: {name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            (
                "clf",
                cls(
                    n_estimators=args.n_estimators,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                ),
            ),
        ]
    )


def fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


def nearest_similarity(eval_smiles: pd.Series, train_smiles: pd.Series) -> np.ndarray:
    train_fps = [fp_from_smiles(value) for value in train_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    values = []
    for smiles in eval_smiles:
        fp = fp_from_smiles(smiles)
        if fp is None or not train_fps:
            values.append(np.nan)
        else:
            values.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return np.asarray(values, dtype=float)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.read_csv(args.all_input)
    fluoro_df = pd.read_csv(args.fluoro_input)
    rows = []
    prediction_rows = []

    for species in ["HLM", "RLM"]:
        fsp = fluoro_df.loc[fluoro_df["species"].eq(species)].copy()
        asp = all_df.loc[all_df["species"].eq(species)].copy()
        eval_df = fsp.loc[fsp[args.split_col].ne("train")].copy()
        eval_scaffolds = set(eval_df["murcko_scaffold"].astype(str))
        specialized_train = fsp.loc[fsp[args.split_col].eq("train")].copy()
        general_train = asp.loc[~asp["murcko_scaffold"].astype(str).isin(eval_scaffolds)].copy()

        numeric_cols = numeric_feature_columns(fsp.copy())
        for scope, train_df in [
            ("fluorinated_only", specialized_train),
            ("general_all_compounds", general_train),
        ]:
            similarity = nearest_similarity(eval_df["smiles"], train_df["smiles"])
            for feature_set in args.feature_sets:
                feature_cols = select_feature_set(numeric_cols, feature_set)
                for model_name in args.models:
                    model = make_model(model_name, args)
                    model.fit(train_df[feature_cols], train_df["label"].astype(int))
                    prob = model.predict_proba(eval_df[feature_cols])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    metrics = safe_metrics(eval_df["label"].astype(int), pred, prob)
                    metrics["brier"] = brier_score_loss(eval_df["label"].astype(int), prob)
                    row = {
                        "species": species,
                        "training_scope": scope,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_train": len(train_df),
                        "n_eval": len(eval_df),
                        "n_features": len(feature_cols),
                    }
                    row.update(metrics)
                    rows.append(row)
                    ids = eval_df[
                        ["compound_id", "smiles", "label", args.split_col, "murcko_scaffold"]
                    ].copy()
                    ids["species"] = species
                    ids["training_scope"] = scope
                    ids["feature_set"] = feature_set
                    ids["model"] = model_name
                    ids["pred_prob_stable"] = prob
                    ids["pred_label"] = pred
                    ids["max_tanimoto_to_training"] = similarity
                    prediction_rows.append(ids)
                    print(
                        f"{species}/{scope}/{feature_set}/{model_name}: "
                        f"AUC={metrics['roc_auc']:.3f}"
                    )

    results = pd.DataFrame(rows).sort_values(
        ["species", "roc_auc"], ascending=[True, False]
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    results.to_csv(output_dir / "publication_benchmark_results.csv", index=False)
    predictions.to_csv(output_dir / "publication_benchmark_predictions.csv", index=False)

    comparisons = []
    keys = ["species", "feature_set", "model"]
    for key, group in results.groupby(keys):
        indexed = group.set_index("training_scope")
        if {"fluorinated_only", "general_all_compounds"}.issubset(indexed.index):
            comparisons.append(
                {
                    **dict(zip(keys, key)),
                    "roc_auc_fluorinated_only": indexed.loc["fluorinated_only", "roc_auc"],
                    "roc_auc_general": indexed.loc["general_all_compounds", "roc_auc"],
                    "roc_auc_fluorinated_minus_general": indexed.loc[
                        "fluorinated_only", "roc_auc"
                    ]
                    - indexed.loc["general_all_compounds", "roc_auc"],
                    "mcc_fluorinated_only": indexed.loc["fluorinated_only", "mcc"],
                    "mcc_general": indexed.loc["general_all_compounds", "mcc"],
                }
            )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_dir / "general_vs_fluorinated_comparison.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
