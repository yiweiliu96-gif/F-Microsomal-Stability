#!/usr/bin/env python3
"""Run prespecified quantitative analyses that strengthen the submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import train_biogen_paired_models as core




def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected input paths exists: " + ", ".join(map(str, paths)))


TRAIN = first_existing(
    ROOT / "reports/biogen_paired_study/biogen_paired_modelling_table.csv",
    ROOT / "data/derived/biogen_paired_modelling_table.csv",
)
EXTERNAL = first_existing(
    ROOT / "reports/openadmet_chembl35_paired_external/external_predictions.csv",
    ROOT / "data/derived/external_predictions.csv",
)
EXTERNAL_AUDIT = first_existing(
    ROOT / "reports/openadmet_chembl35_paired_external/external_endpoint_semantics_audit.csv",
    ROOT / "data/derived/external_endpoint_semantics_audit.csv",
)
PAIRS = first_existing(
    ROOT / "reports/assay_matched_fluorination_pairs/cross_species_assay_matched_pairs.csv",
    ROOT / "data/derived/cross_species_assay_matched_pairs.csv",
)
MANUAL = first_existing(
    ROOT / "reports/assay_matched_fluorination_pairs/cross_species_manual_review_queue.csv",
    ROOT / "data/derived/cross_species_manual_review_queue.csv",
)
OUT = (
    ROOT / "reports/continuous_submission/submission_strengthening"
    if (ROOT / "reports").exists()
    else ROOT / "results/submission_strengthening"
)
if not (ROOT / "reports/biogen_paired_study").exists():
    core.OUT = ROOT / "results/feature_cache"
    core.CACHE = core.OUT / "molecular_features.joblib"
HLM = core.HLM
RLM = core.RLM
SEED = 20260819
PERMUTATIONS = 200
BOOTSTRAPS = 5000


def model_metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    low = (observed <= core.THRESHOLD).astype(int)
    return {
        "rmse": float(mean_squared_error(observed, prediction) ** 0.5),
        "mae": float(mean_absolute_error(observed, prediction)),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "low_clearance_auc": float(roc_auc_score(low, -prediction)),
    }


def strict_external() -> pd.DataFrame:
    predictions = pd.read_csv(EXTERNAL)
    audit = pd.read_csv(EXTERNAL_AUDIT)[
        ["external_record_id", "strict_total_intrinsic_pair", "source_unit_per_ug_flag", "full_inchikey_overlap"]
    ]
    frame = predictions.merge(audit, on="external_record_id", validate="one_to_one")
    frame = frame.loc[
        frame["is_fluorinated"].eq(1)
        & frame["strict_total_intrinsic_pair"].eq(1)
        & frame["source_unit_per_ug_flag"].eq(0)
        & frame["full_inchikey_overlap"].eq(1)
    ].copy().reset_index(drop=True)
    if len(frame) != 598:
        raise RuntimeError(f"Strict external set changed: expected 598, found {len(frame)}")
    frame[HLM] = frame["external_hlm"]
    frame[RLM] = frame["external_rlm"]
    frame["rlm_left_censored"] = 0
    return frame


def prepare_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN)
    external = strict_external()
    combined = pd.concat(
        [train.assign(_dataset="train"), external.assign(_dataset="external")],
        ignore_index=True,
        sort=False,
    )
    combined = core.add_features(combined)
    paired_train = combined.loc[
        combined["_dataset"].eq("train") & combined[HLM].notna() & combined[RLM].notna()
    ].copy()
    external_features = combined.loc[combined["_dataset"].eq("external")].copy().reset_index(drop=True)
    columns = core.representation_columns(combined, fluorine_augmented=False) + [RLM, "rlm_left_censored"]
    return paired_train, external_features, columns


def predict_with_anchor(
    model: object,
    train: pd.DataFrame,
    external: pd.DataFrame,
    columns: list[str],
    anchor: np.ndarray,
) -> np.ndarray:
    variant = external.copy()
    variant[RLM] = anchor
    x_train, x_variant = core.prepare_matrices(train, variant, columns)
    return anchor + model.predict(x_variant)


def document_bootstrap_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    documents = frame["doc_id"].drop_duplicates().to_numpy()
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        indices = np.concatenate([np.flatnonzero(frame["doc_id"].to_numpy() == doc) for doc in sampled])
        observed = frame[HLM].to_numpy()[indices]
        if metric == "rmse":
            b = mean_squared_error(observed, baseline[indices]) ** 0.5
            c = mean_squared_error(observed, candidate[indices]) ** 0.5
        elif metric == "mae":
            b = mean_absolute_error(observed, baseline[indices])
            c = mean_absolute_error(observed, candidate[indices])
        else:
            raise ValueError(metric)
        values.append(float(c - b))
    values_array = np.asarray(values)
    observed = model_metrics(frame[HLM].to_numpy(), candidate)[metric] - model_metrics(
        frame[HLM].to_numpy(), baseline
    )[metric]
    return {
        "metric": metric,
        "candidate_minus_true_anchor": float(observed),
        "ci95_low": float(np.quantile(values_array, 0.025)),
        "ci95_high": float(np.quantile(values_array, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min(np.mean(values_array <= 0), np.mean(values_array >= 0)))),
    }


def nearest_training_rlm(train: pd.DataFrame, external: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train["canonical_smiles"]]
    proxy, similarities = [], []
    train_rlm = train[RLM].to_numpy(dtype=float)
    for smiles in external["canonical_smiles"]:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        scores = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        best = int(np.argmax(scores))
        proxy.append(train_rlm[best])
        similarities.append(scores[best])
    return np.asarray(proxy), np.asarray(similarities)


def anchor_specificity_analysis(
    train: pd.DataFrame, external: pd.DataFrame, columns: list[str]
) -> None:
    x_train, x_external = core.prepare_matrices(train, external, columns)
    model = core.lgbm(SEED)
    model.fit(x_train, train[HLM] - train[RLM])

    def predict_anchor(anchor: np.ndarray) -> np.ndarray:
        variant = x_external.copy()
        variant[RLM] = anchor
        variant["rlm_left_censored"] = 0
        return anchor + model.predict(variant)

    observed = external[HLM].to_numpy(dtype=float)
    true_rlm = external[RLM].to_numpy(dtype=float)
    true_prediction = predict_anchor(true_rlm)
    nearest_rlm, nearest_similarity = nearest_training_rlm(train, external)
    nearest_prediction = predict_anchor(nearest_rlm)

    prediction_table = external[
        ["external_record_id", "canonical_smiles", "doc_id", "fluorine_subgroup", HLM, RLM]
    ].copy()
    prediction_table["true_anchor_prediction"] = true_prediction
    prediction_table["nearest_training_analogue_rlm"] = nearest_rlm
    prediction_table["nearest_training_analogue_similarity"] = nearest_similarity
    prediction_table["nearest_analogue_anchor_prediction"] = nearest_prediction
    prediction_table.to_csv(OUT / "anchor_specificity_predictions.csv", index=False)

    rows = [
        {"control": "same-compound measured RLM", "n": len(external), **model_metrics(observed, true_prediction)},
        {"control": "nearest training analogue RLM", "n": len(external), **model_metrics(observed, nearest_prediction)},
    ]
    bootstrap_rows = []
    for metric in ("rmse", "mae"):
        bootstrap_rows.append(
            {
                "comparison": "nearest training analogue minus same-compound anchor",
                "n": len(external),
                "n_documents": external["doc_id"].nunique(),
                **document_bootstrap_delta(external, true_prediction, nearest_prediction, metric),
            }
        )

    rng = np.random.default_rng(SEED)
    permutation_rows = []
    for index in range(PERMUTATIONS):
        permuted_rlm = rng.permutation(true_rlm)
        prediction = predict_anchor(permuted_rlm)
        permutation_rows.append({"control": "global RLM permutation", "iteration": index, **model_metrics(observed, prediction)})

    grouped = external.groupby("doc_id").indices
    eligible_indices = np.sort(np.concatenate([np.asarray(v) for v in grouped.values() if len(v) >= 2]))
    eligible = external.iloc[eligible_indices].reset_index(drop=True)
    true_eligible = true_prediction[eligible_indices]
    observed_eligible = observed[eligible_indices]
    rows.append(
        {
            "control": "same-compound measured RLM in multi-compound documents",
            "n": len(eligible_indices),
            **model_metrics(observed_eligible, true_eligible),
        }
    )
    for index in range(PERMUTATIONS):
        proxy = true_rlm.copy()
        for values in grouped.values():
            values = np.asarray(values)
            if len(values) < 2:
                continue
            order = rng.permutation(values)
            proxy[order] = true_rlm[np.roll(order, 1)]
        prediction = predict_anchor(proxy)[eligible_indices]
        permutation_rows.append(
            {
                "control": "within-document deranged RLM",
                "iteration": index,
                **model_metrics(observed_eligible, prediction),
            }
        )

    permutations = pd.DataFrame(permutation_rows)
    permutations.to_csv(OUT / "anchor_permutation_metrics.csv", index=False)
    for control, group in permutations.groupby("control"):
        reference_observed = observed_eligible if control.startswith("within") else observed
        reference_prediction = true_eligible if control.startswith("within") else true_prediction
        reference = model_metrics(reference_observed, reference_prediction)
        row = {"control": control, "n": len(reference_observed)}
        for metric in ("rmse", "mae", "spearman", "low_clearance_auc"):
            values = group[metric].to_numpy()
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_p975"] = float(np.quantile(values, 0.975))
            if metric in ("rmse", "mae"):
                row[f"empirical_p_vs_true"] = float((1 + np.sum(values <= reference[metric])) / (1 + len(values)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "anchor_specificity_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "anchor_specificity_document_bootstrap.csv", index=False)


def cross_conformal_analysis(
    train: pd.DataFrame, external: pd.DataFrame, columns: list[str]
) -> None:
    calibration_scores: list[np.ndarray] = []
    external_fold_predictions: list[np.ndarray] = []
    calibration_folds: list[int] = []
    for fold in range(5):
        fit = train.loc[train["scaffold_fold"].ne(fold)].copy()
        calibration = train.loc[
            train["scaffold_fold"].eq(fold) & train["hlm_left_censored"].eq(0)
        ].copy()
        x_fit, x_calibration = core.prepare_matrices(fit, calibration, columns)
        _, x_external = core.prepare_matrices(fit, external, columns)
        model = core.lgbm(core.SEEDS[fold % len(core.SEEDS)])
        model.fit(x_fit, fit[HLM] - fit[RLM])
        calibration_prediction = calibration[RLM].to_numpy() + model.predict(x_calibration)
        external_prediction = external[RLM].to_numpy() + model.predict(x_external)
        calibration_scores.append(np.abs(calibration[HLM].to_numpy() - calibration_prediction))
        external_fold_predictions.append(external_prediction)
        calibration_folds.extend([fold] * len(calibration))

    score = np.concatenate(calibration_scores)
    fold_for_score = np.asarray(calibration_folds)
    fold_predictions = np.vstack(external_fold_predictions)
    lower_candidates = np.column_stack(
        [fold_predictions[fold_for_score[i]] - score[i] for i in range(len(score))]
    )
    upper_candidates = np.column_stack(
        [fold_predictions[fold_for_score[i]] + score[i] for i in range(len(score))]
    )
    point_prediction = fold_predictions.mean(axis=0)
    observed = external[HLM].to_numpy(dtype=float)
    interval_table = external[
        ["external_record_id", "doc_id", "fluorine_subgroup", "cf3_count", "cf2_count", "aryl_f_count", HLM, RLM]
    ].copy()
    coverage_rows = []
    for nominal in (0.80, 0.90):
        alpha = 1.0 - nominal
        lower = np.quantile(lower_candidates, alpha / 2, axis=1, method="lower")
        upper = np.quantile(upper_candidates, 1 - alpha / 2, axis=1, method="higher")
        interval_table[f"lower_{int(nominal * 100)}"] = lower
        interval_table[f"upper_{int(nominal * 100)}"] = upper
        interval_table[f"width_{int(nominal * 100)}"] = upper - lower
        for subgroup, group_index in [("all", np.arange(len(external))), *external.groupby("fluorine_subgroup").indices.items()]:
            index = np.asarray(group_index)
            covered = (observed[index] >= lower[index]) & (observed[index] <= upper[index])
            coverage_frame = external.iloc[index].copy()
            coverage_frame["covered"] = covered.astype(int)
            coverage, coverage_low, coverage_high = cluster_bootstrap_rate(
                coverage_frame.rename(columns={"doc_id": "document_id"}), "covered"
            )
            coverage_rows.append(
                {
                    "nominal_coverage": nominal,
                    "subgroup": subgroup,
                    "n": len(index),
                    "empirical_coverage": coverage,
                    "coverage_ci95_low": coverage_low,
                    "coverage_ci95_high": coverage_high,
                    "median_interval_width": float(np.median((upper - lower)[index])),
                    "mean_interval_width": float(np.mean((upper - lower)[index])),
                }
            )
    interval_table["cross_conformal_point_prediction"] = point_prediction
    interval_table["absolute_error"] = np.abs(point_prediction - observed)
    interval_table.to_csv(OUT / "cross_conformal_external_intervals.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "cross_conformal_coverage.csv", index=False)

    width = interval_table["width_90"].to_numpy()
    order = np.argsort(width)
    selective_rows = []
    for retained_fraction in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        n = int(np.floor(len(external) * retained_fraction))
        index = order[:n]
        covered = (
            (observed[index] >= interval_table["lower_90"].to_numpy()[index])
            & (observed[index] <= interval_table["upper_90"].to_numpy()[index])
        )
        selective_rows.append(
            {
                "retained_fraction": retained_fraction,
                "n": n,
                **model_metrics(observed[index], point_prediction[index]),
                "empirical_90_coverage": float(covered.mean()),
                "median_90_width": float(np.median(width[index])),
            }
        )
    pd.DataFrame(selective_rows).to_csv(OUT / "selective_prediction_curve.csv", index=False)
    summary = {
        "calibration_n_exact": int(len(score)),
        "external_n": int(len(external)),
        "width_absolute_error_spearman": float(spearmanr(width, np.abs(point_prediction - observed)).statistic),
        "point_metrics": model_metrics(observed, point_prediction),
    }
    (OUT / "cross_conformal_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def cluster_bootstrap_rate(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    grouped = frame.groupby("document_id")[column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    sampled = rng.integers(0, len(grouped), size=(10000, len(grouped)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return float(frame[column].mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def reversal_risk_analysis() -> None:
    pairs = pd.read_csv(PAIRS)
    manual = pd.read_csv(MANUAL)[["pair_id", "manual_review_status"]]
    pairs = pairs.drop(columns=["manual_review_status"], errors="ignore").merge(manual, on="pair_id", validate="one_to_one")
    pairs = pairs.loc[
        pairs["strict_both_species_assay_matched"].eq(1)
        & pairs["selected_assays_free_of_non_equality_records"].eq(1)
        & pairs["manual_review_status"].eq("verified")
    ].copy()
    if len(pairs) != 97:
        raise RuntimeError(f"Verified cross-species pair set changed: expected 97, found {len(pairs)}")
    pairs["delta_rlm_bin"] = pd.cut(
        pairs["assay_matched_delta_rlm"],
        bins=[-np.inf, -0.30, -0.10, 0.10, np.inf],
        labels=["< -0.30", "-0.30 to -0.10", "-0.10 to 0.10", "> 0.10"],
        right=False,
    )
    pairs["reversal"] = 1 - pairs["assay_matched_direction_agreement"].astype(int)
    pairs["hlm_improvement"] = pairs["assay_matched_delta_hlm"].lt(0).astype(int)
    rows = []
    for bin_name, group in pairs.groupby("delta_rlm_bin", observed=True):
        reversal, reversal_low, reversal_high = cluster_bootstrap_rate(group, "reversal")
        improvement, improvement_low, improvement_high = cluster_bootstrap_rate(group, "hlm_improvement")
        rows.append(
            {
                "delta_rlm_bin": str(bin_name),
                "n_pairs": len(group),
                "n_documents": group["document_id"].nunique(),
                "median_delta_rlm": float(group["assay_matched_delta_rlm"].median()),
                "median_delta_hlm": float(group["assay_matched_delta_hlm"].median()),
                "reversal_fraction": reversal,
                "reversal_ci95_low": reversal_low,
                "reversal_ci95_high": reversal_high,
                "hlm_improvement_fraction": improvement,
                "hlm_improvement_ci95_low": improvement_low,
                "hlm_improvement_ci95_high": improvement_high,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "cross_species_reversal_risk_bins.csv", index=False)

    transformation_rows = []
    for transformation, group in pairs.groupby("transformation"):
        reversal, low, high = cluster_bootstrap_rate(group, "reversal")
        transformation_rows.append(
            {
                "transformation": transformation,
                "n_pairs": len(group),
                "n_documents": group["document_id"].nunique(),
                "reversal_fraction": reversal,
                "reversal_ci95_low": low,
                "reversal_ci95_high": high,
            }
        )
    pd.DataFrame(transformation_rows).to_csv(OUT / "cross_species_reversal_by_transformation.csv", index=False)
    pairs.to_csv(OUT / "cross_species_verified_pairs_with_risk_bins.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, external, columns = prepare_feature_tables()
    anchor_specificity_analysis(train, external, columns)
    cross_conformal_analysis(train, external, columns)
    reversal_risk_analysis()
    metadata = {
        "status": "Post hoc submission-strengthening analyses. Controls and bins were fixed before these analyses were run; no external sample or endpoint threshold was selected by performance.",
        "strict_external_n": 598,
        "anchor_permutations": PERMUTATIONS,
        "document_bootstraps": BOOTSTRAPS,
        "uncertainty_calibration": "Five-fold scaffold cross-conformal analysis using exact development observations only.",
        "reversal_bins": ["< -0.30", "-0.30 to -0.10", "-0.10 to 0.10", "> 0.10 log10 CLint"],
        "random_seed": SEED,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote submission-strengthening analyses to {OUT}")


if __name__ == "__main__":
    main()
