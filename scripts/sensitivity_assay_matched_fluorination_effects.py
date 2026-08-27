#!/usr/bin/env python3
"""Prespecified robustness analyses for assay-matched fluorination effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "reports" / "assay_matched_fluorination_pairs"
OUT = SOURCE_DIR / "fluorination_effect_sensitivity.csv"
SEED = 20260824
ITERATIONS = 5000


def add_network(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(a: str, b: str) -> None:
        ra, rb = find(str(a)), find(str(b))
        if ra != rb:
            parent[rb] = ra

    for a, b in frame[[left, right]].itertuples(index=False):
        union(a, b)
    roots = {find(value) for value in parent}
    labels = {root: f"NET_{index:04d}" for index, root in enumerate(sorted(roots), start=1)}
    result = frame.copy()
    result["pair_network_id"] = result[left].map(lambda value: labels[find(str(value))])
    return result


def bootstrap_single(frame: pd.DataFrame, delta: str, cluster: str) -> dict[str, float]:
    groups = [group.index.to_numpy() for _, group in frame.groupby(cluster, sort=False)]
    rng = np.random.default_rng(SEED)
    medians, fractions = [], []
    for _ in range(ITERATIONS):
        selected = rng.integers(0, len(groups), len(groups))
        values = frame.loc[np.concatenate([groups[index] for index in selected]), delta].to_numpy(dtype=float)
        medians.append(float(np.median(values)))
        fractions.append(float(np.mean(values < 0)))
    values = frame[delta].to_numpy(dtype=float)
    return {
        "median_delta": float(np.median(values)),
        "median_delta_ci95_low": float(np.quantile(medians, 0.025)),
        "median_delta_ci95_high": float(np.quantile(medians, 0.975)),
        "improved_fraction": float(np.mean(values < 0)),
        "improved_fraction_ci95_low": float(np.quantile(fractions, 0.025)),
        "improved_fraction_ci95_high": float(np.quantile(fractions, 0.975)),
    }


def bootstrap_cross(frame: pd.DataFrame, cluster: str) -> dict[str, float]:
    groups = [group.index.to_numpy() for _, group in frame.groupby(cluster, sort=False)]
    rng = np.random.default_rng(SEED)
    correlations, agreements = [], []
    for _ in range(ITERATIONS):
        selected = rng.integers(0, len(groups), len(groups))
        draw = frame.loc[np.concatenate([groups[index] for index in selected])]
        if draw["assay_matched_delta_hlm"].nunique() > 1 and draw["assay_matched_delta_rlm"].nunique() > 1:
            correlations.append(
                float(spearmanr(draw["assay_matched_delta_hlm"], draw["assay_matched_delta_rlm"]).statistic)
            )
        agreements.append(float(draw["assay_matched_direction_agreement"].mean()))
    rho = float(spearmanr(frame["assay_matched_delta_hlm"], frame["assay_matched_delta_rlm"]).statistic)
    return {
        "delta_spearman": rho,
        "delta_spearman_ci95_low": float(np.quantile(correlations, 0.025)),
        "delta_spearman_ci95_high": float(np.quantile(correlations, 0.975)),
        "direction_agreement": float(frame["assay_matched_direction_agreement"].mean()),
        "direction_agreement_ci95_low": float(np.quantile(agreements, 0.025)),
        "direction_agreement_ci95_high": float(np.quantile(agreements, 0.975)),
    }


def main() -> None:
    single = pd.read_csv(SOURCE_DIR / "same_assay_species_fluorination_pairs_primary.csv")
    single = add_network(single, "base_connectivity_key", "fluorinated_connectivity_key").reset_index(drop=True)
    cross = pd.read_csv(SOURCE_DIR / "cross_species_assay_matched_pairs.csv")
    cross = add_network(cross, "base_smiles", "fluorinated_smiles").reset_index(drop=True)
    rows = []

    for species, species_frame in single.groupby("species", sort=False):
        delta = "delta_log10_clint_f_minus_base"
        scenarios = {
            "primary": np.ones(len(species_frame), dtype=bool),
            "exclude_exact_zero_delta": species_frame[delta].ne(0).to_numpy(),
            "exclude_absolute_delta_gt_1": species_frame[delta].abs().le(1).to_numpy(),
            "exclude_chembl_dataset_doi": ~species_frame["document_doi"].fillna("").str.contains(
                "10.6019/CHEMBL", case=False
            ).to_numpy(),
        }
        for scenario, mask in scenarios.items():
            frame = species_frame.loc[mask].reset_index(drop=True)
            for cluster in ["document_id", "pair_network_id"]:
                rows.append(
                    {
                        "analysis": "single_species",
                        "species": species,
                        "scenario": scenario,
                        "cluster_unit": cluster,
                        "n_pairs": len(frame),
                        "n_clusters": frame[cluster].nunique(),
                        **bootstrap_single(frame, delta, cluster),
                    }
                )

    cross_scenarios = {
        "primary": np.ones(len(cross), dtype=bool),
        "exclude_any_exact_zero_delta": cross[["assay_matched_delta_hlm", "assay_matched_delta_rlm"]].ne(0).all(axis=1).to_numpy(),
        "exclude_any_absolute_delta_gt_1": cross[["assay_matched_delta_hlm", "assay_matched_delta_rlm"]].abs().le(1).all(axis=1).to_numpy(),
        "aryl_H_to_F_only": cross["transformation"].eq("Ar-H_to_Ar-F").to_numpy(),
    }
    for scenario, mask in cross_scenarios.items():
        frame = cross.loc[mask].reset_index(drop=True)
        for cluster in ["document_id", "pair_network_id"]:
            row = {
                "analysis": "cross_species",
                "species": "HLM_RLM",
                "scenario": scenario,
                "cluster_unit": cluster,
                "n_pairs": len(frame),
                "n_clusters": frame[cluster].nunique(),
                **bootstrap_cross(frame, cluster),
            }
            for species, delta in [("hlm", "assay_matched_delta_hlm"), ("rlm", "assay_matched_delta_rlm")]:
                summary = bootstrap_single(frame, delta, cluster)
                row.update({f"{species}_{key}": value for key, value in summary.items()})
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
