#!/usr/bin/env python3
"""Build assay-matched fluorination pairs and audit cross-species pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_openadmet_chembl35_paired_external import (
    HLM_RAW,
    RLM_RAW,
    exclusion_sets,
    prepare_records,
)
from mine_strict_continuous_fluorination_pairs import candidate_defluorinations


ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "reports" / "strict_continuous_fluorination_pairs"
EXTERNAL_PRIMARY = ROOT / "reports" / "openadmet_chembl35_paired_external" / "paired_external_primary.csv"
OUT = ROOT / "reports" / "assay_matched_fluorination_pairs"
SEED = 20260824
BOOTSTRAPS = 10000
PERMUTATIONS = 10000


def assay_molecule_table(path: Path, species: str) -> pd.DataFrame:
    records = prepare_records(path, species)
    non_equality = (
        records.assign(_non_equality=records["standard_relation"].fillna("=").ne("="))
        .groupby(["assay_id", "connectivity_key"])["_non_equality"]
        .max()
        .to_dict()
    )
    records = records.loc[records["eligible_record"]].copy()
    records["has_traceable_publication"] = (
        records["doc_doi"].notna() | records["doc_pubmed_id"].notna()
    ).astype(int)
    rows = []
    for (assay_id, key), group in records.groupby(["assay_id", "connectivity_key"], sort=False):
        values = group["log10_clint_ml_min_kg"].to_numpy(dtype=float)
        first = group.iloc[0]
        rows.append(
            {
                "species": species.upper(),
                "assay_id": int(assay_id),
                "document_id": int(first["doc_id"]),
                "document_doi": first["doc_doi"],
                "document_title": first["doc_title"],
                "has_traceable_publication": int(first["has_traceable_publication"]),
                "connectivity_key": key,
                "canonical_smiles": first["canonical_smiles_audit"],
                "value_log10_clint": float(np.median(values)),
                "record_count": int(len(group)),
                "replicate_range_log10": float(values.max() - values.min()),
                "source_standard_types": ";".join(sorted(group["standard_type"].dropna().astype(str).unique())),
                "source_standard_units": ";".join(sorted(group["standard_units"].dropna().astype(str).unique())),
                "source_scaled_units": ";".join(sorted(group["scaled_units"].dropna().astype(str).unique())),
                "has_non_equality_record_in_assay": int(non_equality.get((assay_id, key), False)),
            }
        )
    return pd.DataFrame(rows)


def previous_dataset_keys() -> set[str]:
    sets = exclusion_sets()
    return set().union(*sets.values())


def mine_same_assay_pairs(table: pd.DataFrame) -> pd.DataFrame:
    lookups = {
        assay_id: group.set_index("canonical_smiles", drop=False)
        for assay_id, group in table.groupby("assay_id", sort=False)
    }
    excluded = previous_dataset_keys()
    rows = []
    for assay_id, assay in lookups.items():
        for _, fluorinated in assay.iterrows():
            for candidate in candidate_defluorinations(fluorinated["canonical_smiles"]):
                if candidate["base_smiles"] not in assay.index:
                    continue
                base = assay.loc[candidate["base_smiles"]]
                if isinstance(base, pd.DataFrame):
                    base = base.iloc[0]
                pair_range = max(base["replicate_range_log10"], fluorinated["replicate_range_log10"])
                delta = float(fluorinated["value_log10_clint"] - base["value_log10_clint"])
                rows.append(
                    {
                        "species": fluorinated["species"],
                        "assay_id": int(assay_id),
                        "document_id": int(fluorinated["document_id"]),
                        "document_doi": fluorinated["document_doi"],
                        "document_title": fluorinated["document_title"],
                        "base_connectivity_key": base["connectivity_key"],
                        "fluorinated_connectivity_key": fluorinated["connectivity_key"],
                        "base_smiles": base["canonical_smiles"],
                        "fluorinated_smiles": fluorinated["canonical_smiles"],
                        **candidate,
                        "base_log10_clint": float(base["value_log10_clint"]),
                        "fluorinated_log10_clint": float(fluorinated["value_log10_clint"]),
                        "delta_log10_clint_f_minus_base": delta,
                        "improved_after_fluorination": int(delta < 0),
                        "base_record_count": int(base["record_count"]),
                        "fluorinated_record_count": int(fluorinated["record_count"]),
                        "base_replicate_range_log10": float(base["replicate_range_log10"]),
                        "fluorinated_replicate_range_log10": float(fluorinated["replicate_range_log10"]),
                        "pair_max_replicate_range_log10": float(pair_range),
                        "base_source_standard_types": base["source_standard_types"],
                        "fluorinated_source_standard_types": fluorinated["source_standard_types"],
                        "base_source_standard_units": base["source_standard_units"],
                        "fluorinated_source_standard_units": fluorinated["source_standard_units"],
                        "base_source_scaled_units": base["source_scaled_units"],
                        "fluorinated_source_scaled_units": fluorinated["source_scaled_units"],
                        "source_endpoint_type_match": int(
                            base["source_standard_types"] == fluorinated["source_standard_types"]
                        ),
                        "source_unit_match": int(
                            base["source_standard_units"] == fluorinated["source_standard_units"]
                            and base["source_scaled_units"] == fluorinated["source_scaled_units"]
                        ),
                        "base_has_non_equality_record_in_assay": int(base["has_non_equality_record_in_assay"]),
                        "fluorinated_has_non_equality_record_in_assay": int(
                            fluorinated["has_non_equality_record_in_assay"]
                        ),
                        "has_traceable_publication": int(fluorinated["has_traceable_publication"]),
                        "overlap_any_previous_dataset": int(
                            base["connectivity_key"] in excluded
                            or fluorinated["connectivity_key"] in excluded
                        ),
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.drop_duplicates(
        ["species", "assay_id", "base_connectivity_key", "fluorinated_connectivity_key", "transformation"]
    )
    pairs["primary_quality_eligible"] = (
        pairs["has_traceable_publication"].eq(1)
        & pairs["pair_max_replicate_range_log10"].le(0.3)
        & pairs["overlap_any_previous_dataset"].eq(0)
        & pairs["formula_delta_ok"].eq(1)
        & pairs["source_endpoint_type_match"].eq(1)
        & pairs["source_unit_match"].eq(1)
        & pairs["base_has_non_equality_record_in_assay"].eq(0)
        & pairs["fluorinated_has_non_equality_record_in_assay"].eq(0)
    ).astype(int)
    # Select one assay per chemical transform using provenance and replicate quality only.
    pairs = pairs.sort_values(
        [
            "base_connectivity_key",
            "fluorinated_connectivity_key",
            "transformation",
            "primary_quality_eligible",
            "pair_max_replicate_range_log10",
            "document_id",
            "assay_id",
        ],
        ascending=[True, True, True, False, True, True, True],
    )
    pairs["selected_primary_observation"] = (
        pairs.groupby(
            ["base_connectivity_key", "fluorinated_connectivity_key", "transformation"], sort=False
        ).cumcount().eq(0)
        & pairs["primary_quality_eligible"].eq(1)
    ).astype(int)
    pairs["manual_review_status"] = "pending"
    pairs["manual_review_note"] = "Verify the parent/fluorinated identity and assay-table values in the cited source."
    return pairs.reset_index(drop=True)


def parse_ids(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(float(item.strip())) for item in str(value).split(";") if item.strip()}


def assay_value_lookup(table: pd.DataFrame) -> dict[tuple[int, str], pd.Series]:
    return {(int(row.assay_id), row.connectivity_key): row for row in table.itertuples(index=False)}


def audit_cross_species_pairs(hlm_table: pd.DataFrame, rlm_table: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(PAIR_DIR / "strict_continuous_fluorination_pairs.csv")
    pairs = pairs.loc[pairs["dataset"].eq("ChEMBL35_strict")].copy()
    primary = pd.read_csv(EXTERNAL_PRIMARY).set_index("external_record_id")
    hlookup = assay_value_lookup(hlm_table)
    rlookup = assay_value_lookup(rlm_table)
    rows = []
    for pair in pairs.itertuples(index=False):
        base = primary.loc[pair.base_id]
        fluorinated = primary.loc[pair.fluorinated_id]
        h_shared = sorted(parse_ids(base.hlm_assay_ids) & parse_ids(fluorinated.hlm_assay_ids))
        r_shared = sorted(parse_ids(base.rlm_assay_ids) & parse_ids(fluorinated.rlm_assay_ids))

        def choose(shared: list[int], lookup: dict[tuple[int, str], pd.Series]):
            candidates = []
            for assay_id in shared:
                left = lookup.get((assay_id, base.connectivity_key))
                right = lookup.get((assay_id, fluorinated.connectivity_key))
                if left is not None and right is not None:
                    candidates.append((max(left.replicate_range_log10, right.replicate_range_log10), assay_id, left, right))
            return min(candidates, key=lambda item: (item[0], item[1])) if candidates else None

        hs = choose(h_shared, hlookup)
        rs = choose(r_shared, rlookup)
        row = pair._asdict()
        row.update(
            {
                "hlm_shared_assay_ids": ";".join(map(str, h_shared)),
                "rlm_shared_assay_ids": ";".join(map(str, r_shared)),
                "has_shared_hlm_assay": int(bool(h_shared)),
                "has_shared_rlm_assay": int(bool(r_shared)),
                "strict_both_species_assay_matched": int(hs is not None and rs is not None),
                "selected_assays_free_of_non_equality_records": int(
                    hs is not None
                    and rs is not None
                    and not hs[2].has_non_equality_record_in_assay
                    and not hs[3].has_non_equality_record_in_assay
                    and not rs[2].has_non_equality_record_in_assay
                    and not rs[3].has_non_equality_record_in_assay
                ),
            }
        )
        if hs is not None:
            row.update(
                {
                    "selected_hlm_assay_id": hs[1],
                    "assay_matched_base_hlm": hs[2].value_log10_clint,
                    "assay_matched_fluorinated_hlm": hs[3].value_log10_clint,
                    "assay_matched_delta_hlm": hs[3].value_log10_clint - hs[2].value_log10_clint,
                    "hlm_pair_max_replicate_range_log10": hs[0],
                }
            )
        if rs is not None:
            row.update(
                {
                    "selected_rlm_assay_id": rs[1],
                    "assay_matched_base_rlm": rs[2].value_log10_clint,
                    "assay_matched_fluorinated_rlm": rs[3].value_log10_clint,
                    "assay_matched_delta_rlm": rs[3].value_log10_clint - rs[2].value_log10_clint,
                    "rlm_pair_max_replicate_range_log10": rs[0],
                }
            )
        rows.append(row)
    audited = pd.DataFrame(rows)
    audited["strict_both_species_assay_matched"] = (
        audited["strict_both_species_assay_matched"].eq(1)
        & audited["selected_assays_free_of_non_equality_records"].eq(1)
    ).astype(int)
    eligible = audited["strict_both_species_assay_matched"].eq(1)
    audited.loc[eligible, "assay_matched_hlm_improved"] = audited.loc[eligible, "assay_matched_delta_hlm"].lt(0).astype(int)
    audited.loc[eligible, "assay_matched_rlm_improved"] = audited.loc[eligible, "assay_matched_delta_rlm"].lt(0).astype(int)
    audited.loc[eligible, "assay_matched_direction_agreement"] = (
        np.sign(audited.loc[eligible, "assay_matched_delta_hlm"])
        == np.sign(audited.loc[eligible, "assay_matched_delta_rlm"])
    ).astype(int)
    audited["manual_review_status"] = "pending"
    audited["manual_review_note"] = "Verify both selected species-specific assays and molecular structures against the cited source."
    return audited


def network_count(frame: pd.DataFrame, left: str, right: str) -> int:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in frame[[left, right]].itertuples(index=False):
        union(str(a), str(b))
    return len({find(x) for x in parent})


def cluster_statistics(
    frame: pd.DataFrame,
    delta_column: str,
    cluster_column: str,
    left: str,
    right: str,
) -> dict[str, float | int]:
    values = frame[delta_column].to_numpy(dtype=float)
    cluster_groups = [g.index.to_numpy() for _, g in frame.groupby(cluster_column, sort=False)]
    rng = np.random.default_rng(SEED)
    medians, improved = [], []
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(cluster_groups), len(cluster_groups))
        indices = np.concatenate([cluster_groups[i] for i in sampled])
        draw = frame.loc[indices, delta_column].to_numpy(dtype=float)
        medians.append(np.median(draw))
        improved.append(np.mean(draw < 0))
    cluster_means = frame.groupby(cluster_column)[delta_column].mean().to_numpy(dtype=float)
    observed = abs(float(cluster_means.mean()))
    random_stats = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], len(cluster_means))
        random_stats[i] = abs(float(np.mean(cluster_means * signs)))
    return {
        "n_pairs": int(len(frame)),
        "n_documents": int(frame[cluster_column].nunique()),
        "n_assays": int(frame["assay_id"].nunique()) if "assay_id" in frame else int(frame[["selected_hlm_assay_id", "selected_rlm_assay_id"]].astype(str).agg("::".join, axis=1).nunique()),
        "n_pair_networks": network_count(frame, left, right),
        "median_delta": float(np.median(values)),
        "median_delta_ci95_low": float(np.quantile(medians, 0.025)),
        "median_delta_ci95_high": float(np.quantile(medians, 0.975)),
        "improved_fraction": float(np.mean(values < 0)),
        "improved_fraction_ci95_low": float(np.quantile(improved, 0.025)),
        "improved_fraction_ci95_high": float(np.quantile(improved, 0.975)),
        "cluster_signflip_p_two_sided": float((np.sum(random_stats >= observed) + 1) / (PERMUTATIONS + 1)),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    valid = values.dropna().sort_values()
    m = len(valid)
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    if not m:
        return adjusted
    raw = valid.to_numpy() * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1)
    adjusted.loc[valid.index] = corrected
    return adjusted


def species_summaries(all_pairs: pd.DataFrame) -> pd.DataFrame:
    primary = all_pairs.loc[all_pairs["selected_primary_observation"].eq(1)].copy().reset_index(drop=True)
    rows = []
    for species, sf in primary.groupby("species", sort=False):
        groups = [("all", sf), *sf.groupby("transformation", sort=False)]
        for transform, frame in groups:
            if frame.empty:
                continue
            rows.append(
                {
                    "analysis": "single_species_same_assay",
                    "species": species,
                    "transformation": transform,
                    **cluster_statistics(
                        frame,
                        "delta_log10_clint_f_minus_base",
                        "document_id",
                        "base_connectivity_key",
                        "fluorinated_connectivity_key",
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["cluster_signflip_q_bh"] = benjamini_hochberg(result["cluster_signflip_p_two_sided"])
    return result


def cross_species_summary(audited: pd.DataFrame) -> pd.DataFrame:
    frame = audited.loc[audited["strict_both_species_assay_matched"].eq(1)].copy().reset_index(drop=True)
    rows = []
    for transform, group in [("all", frame), *frame.groupby("transformation", sort=False)]:
        if group.empty:
            continue
        base = {
            "analysis": "cross_species_both_assays_matched",
            "transformation": transform,
            "n_pairs": len(group),
            "n_documents": group["document_id"].nunique(),
            "n_pair_networks": network_count(group, "base_smiles", "fluorinated_smiles"),
            "direction_agreement": group["assay_matched_direction_agreement"].mean(),
            "species_opposite_direction": 1 - group["assay_matched_direction_agreement"].mean(),
        }
        if len(group) >= 4 and group["assay_matched_delta_hlm"].nunique() > 1 and group["assay_matched_delta_rlm"].nunique() > 1:
            base["delta_hlm_delta_rlm_spearman"] = float(
                spearmanr(group["assay_matched_delta_hlm"], group["assay_matched_delta_rlm"]).statistic
            )
        else:
            base["delta_hlm_delta_rlm_spearman"] = np.nan
        document_groups = [item.index.to_numpy() for _, item in group.groupby("document_id", sort=False)]
        rng = np.random.default_rng(SEED)
        agreement_draws, correlation_draws = [], []
        for _ in range(BOOTSTRAPS):
            selected = rng.integers(0, len(document_groups), len(document_groups))
            indices = np.concatenate([document_groups[index] for index in selected])
            draw = group.loc[indices]
            agreement_draws.append(float(draw["assay_matched_direction_agreement"].mean()))
            if (
                len(draw) >= 4
                and draw["assay_matched_delta_hlm"].nunique() > 1
                and draw["assay_matched_delta_rlm"].nunique() > 1
            ):
                correlation_draws.append(
                    float(spearmanr(draw["assay_matched_delta_hlm"], draw["assay_matched_delta_rlm"]).statistic)
                )
        base["direction_agreement_ci95_low"] = float(np.quantile(agreement_draws, 0.025))
        base["direction_agreement_ci95_high"] = float(np.quantile(agreement_draws, 0.975))
        if correlation_draws:
            base["delta_spearman_ci95_low"] = float(np.quantile(correlation_draws, 0.025))
            base["delta_spearman_ci95_high"] = float(np.quantile(correlation_draws, 0.975))
        else:
            base["delta_spearman_ci95_low"] = np.nan
            base["delta_spearman_ci95_high"] = np.nan
        for species, column in [("HLM", "assay_matched_delta_hlm"), ("RLM", "assay_matched_delta_rlm")]:
            stats = cluster_statistics(group, column, "document_id", "base_smiles", "fluorinated_smiles")
            for key, value in stats.items():
                if key not in {"n_pairs", "n_documents", "n_assays", "n_pair_networks"}:
                    base[f"{species.lower()}_{key}"] = value
        rows.append(base)
    return pd.DataFrame(rows)


def manual_review_queue(audited: pd.DataFrame) -> pd.DataFrame:
    frame = audited.loc[audited["strict_both_species_assay_matched"].eq(1)].copy()
    rare_transform = frame["transformation"].ne("Ar-H_to_Ar-F")
    reversal = frame["assay_matched_direction_agreement"].eq(0)
    large_effect = frame[["assay_matched_delta_hlm", "assay_matched_delta_rlm"]].abs().max(axis=1).gt(0.5)
    rounded_zero = frame[["assay_matched_delta_hlm", "assay_matched_delta_rlm"]].eq(0).any(axis=1)
    frame["review_priority"] = np.select(
        [rare_transform | large_effect, reversal | rounded_zero],
        [1, 2],
        default=3,
    )
    reasons = []
    for index in frame.index:
        labels = []
        if rare_transform.loc[index]:
            labels.append("non_aryl_transform")
        if reversal.loc[index]:
            labels.append("cross_species_reversal")
        if large_effect.loc[index]:
            labels.append("absolute_delta_gt_0.5")
        if rounded_zero.loc[index]:
            labels.append("zero_or_rounded_delta")
        if not labels:
            labels.append("routine_random_audit")
        reasons.append(";".join(labels))
    frame["review_reason"] = reasons
    columns = [
        "review_priority", "review_reason", "manual_review_status", "manual_review_note",
        "pair_id", "document_id", "doi", "transformation", "base_id", "fluorinated_id",
        "base_smiles", "fluorinated_smiles", "selected_hlm_assay_id", "selected_rlm_assay_id",
        "assay_matched_base_hlm", "assay_matched_fluorinated_hlm", "assay_matched_delta_hlm",
        "assay_matched_base_rlm", "assay_matched_fluorinated_rlm", "assay_matched_delta_rlm",
        "assay_matched_direction_agreement",
    ]
    return frame[columns].sort_values(
        ["review_priority", "document_id", "transformation", "pair_id"]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hlm_table = assay_molecule_table(HLM_RAW, "hlm")
    rlm_table = assay_molecule_table(RLM_RAW, "rlm")
    hlm_pairs = mine_same_assay_pairs(hlm_table)
    rlm_pairs = mine_same_assay_pairs(rlm_table)
    all_species = pd.concat([hlm_pairs, rlm_pairs], ignore_index=True)
    audited = audit_cross_species_pairs(hlm_table, rlm_table)
    single_summary = species_summaries(all_species)
    cross_summary = cross_species_summary(audited)
    review_queue = manual_review_queue(audited)

    all_species.to_csv(OUT / "same_assay_species_fluorination_pairs_all_observations.csv", index=False)
    all_species.loc[all_species["selected_primary_observation"].eq(1)].to_csv(
        OUT / "same_assay_species_fluorination_pairs_primary.csv", index=False
    )
    audited.to_csv(OUT / "cross_species_pair_assay_audit.csv", index=False)
    audited.loc[audited["strict_both_species_assay_matched"].eq(1)].to_csv(
        OUT / "cross_species_assay_matched_pairs.csv", index=False
    )
    single_summary.to_csv(OUT / "same_assay_species_effect_summary.csv", index=False)
    cross_summary.to_csv(OUT / "cross_species_assay_matched_summary.csv", index=False)
    review_queue.to_csv(OUT / "cross_species_manual_review_queue.csv", index=False)

    metadata = {
        "effect_definition": "delta log10 CLint = fluorinated - nonfluorinated parent; negative values indicate lower clearance after fluorination.",
        "exact_transform_rule": "Only graph edits that replace one to three carbon-bound hydrogens by fluorine at a single carbon are retained.",
        "single_species_primary_rule": "Parent and fluorinated analogue share one ChEMBL assay; exact quantitative liver-microsome records; neither molecule has an additional non-equality record in the selected assay; traceable source; matching original endpoint types and unit provenance; pair replicate range <=0.3 log10; neither molecule overlaps prior project datasets.",
        "duplicate_pair_selection": "One assay observation per chemical pair and transform, selected by replicate range then document and assay identifiers without using endpoint magnitude.",
        "cross_species_primary_rule": "Previously curated same-document HLM/RLM pairs are retained only when parent and fluorinated analogue share at least one HLM assay and at least one RLM assay, with no additional non-equality records for either molecule in the selected assays; values are recomputed within selected assays.",
        "uncertainty": f"{BOOTSTRAPS} document-cluster bootstrap replicates; {PERMUTATIONS} document-cluster sign-flip permutations.",
        "multiplicity": "Benjamini-Hochberg adjustment across single-species all-transform and motif tests.",
        "manual_review": "Every primary pair remains pending source-table verification; no mechanistic or causal claim should be made before review.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Single-species primary effects")
    print(single_summary.to_string(index=False))
    print("\nCross-species assay audit")
    print(audited[["has_shared_hlm_assay", "has_shared_rlm_assay", "strict_both_species_assay_matched"]].sum().to_string())
    print("\nCross-species effects")
    print(cross_summary.to_string(index=False))


if __name__ == "__main__":
    main()
