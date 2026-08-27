#!/usr/bin/env python3
"""Build a traceable, molecule-paired continuous HLM/RLM external set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "external_data" / "openadmet_chembl35"
OUT = ROOT / "reports" / "openadmet_chembl35_paired_external"
HLM_RAW = RAW_DIR / "ChEMBL_HLM_CL_scaled_raw.parquet"
RLM_RAW = RAW_DIR / "ChEMBL_RLM_CL_scaled_raw.parquet"


def molecule_key(smiles: object) -> tuple[str, str, str, int, int, int, int]:
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return "", "", "", 0, 0, 0, 0
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    key = inchi.MolToInchiKey(mol)
    f_count = sum(atom.GetAtomicNum() == 9 for atom in mol.GetAtoms())
    aryl_f = 0
    cf2 = 0
    cf3 = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 9 and atom.GetNeighbors()[0].GetIsAromatic():
            aryl_f += 1
        if atom.GetAtomicNum() == 6:
            attached_f = sum(neighbor.GetAtomicNum() == 9 for neighbor in atom.GetNeighbors())
            cf2 += int(attached_f == 2)
            cf3 += int(attached_f == 3)
    return canonical, key, key.split("-")[0], f_count, aryl_f, cf2, cf3


def source_keys(path: Path, key_column: str | None = None, smiles_column: str | None = None) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if key_column and key_column in frame:
        return set(frame[key_column].dropna().astype(str).str.split("-").str[0])
    if not smiles_column or smiles_column not in frame:
        return set()
    keys = set()
    for value in frame[smiles_column].dropna():
        _, _, connectivity, *_ = molecule_key(value)
        if connectivity:
            keys.add(connectivity)
    return keys


def exclusion_sets() -> dict[str, set[str]]:
    return {
        "biogen": source_keys(
            ROOT / "reports/public_resource_audit/biogen_paired_hlm_rlm_standardized_audit.csv",
            key_column="inchikey",
        ),
        "kcb": source_keys(
            ROOT / "reports/kcb_public_dataset_audit/kcb_standardized_audit_table.csv",
            key_column="inchikey",
        ),
        "ncats": source_keys(
            ROOT / "reports/public_resource_audit/ncats_pubchem_standardized_audit.csv",
            key_column="inchikey",
        ),
        "previous_chembl_external": source_keys(
            ROOT / "reports/v7_external_validation/v7_chembl_external_validation_set.csv",
            smiles_column="canonical_smiles",
        ),
        "reference_hlm": source_keys(
            ROOT / "reports/v4_high_level_sci/ref_si/hlm_source_data_with_fluorine_features.csv",
            smiles_column="canonical_smiles",
        ),
        "reference_rlm": source_keys(
            ROOT / "reports/v4_high_level_sci/ref_si/rlm_source_data_with_fluorine_features.csv",
            smiles_column="canonical_smiles",
        ),
    }


def prepare_records(path: Path, species: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    structure = frame["OPENADMET_CANONICAL_SMILES"].fillna(frame["canonical_smiles"])
    parsed = pd.DataFrame(
        [molecule_key(value) for value in structure],
        columns=[
            "canonical_smiles_audit",
            "inchikey_audit",
            "connectivity_key",
            "f_count",
            "aryl_f_count",
            "cf2_count",
            "cf3_count",
        ],
        index=frame.index,
    )
    frame = pd.concat([frame, parsed], axis=1)
    description = frame["assay_description"].fillna("").str.lower()
    microsome = frame["assay_subcellular_fraction"].fillna("").str.lower().eq("microsome")
    liver = frame["assay_tissue"].fillna("").str.lower().eq("liver")
    scaled_unit_ok = frame["scaled_units"].fillna("").eq("mL.min-1.kg-1")
    already_scaled_unit = frame["standard_units"].fillna("").eq("mL.min-1.kg-1")
    positive_value = pd.to_numeric(frame["standard_value_scaled"], errors="coerce").gt(0)
    frame["eligible_record"] = (
        frame["connectivity_key"].ne("")
        & frame["standard_relation"].fillna("=").eq("=")
        & positive_value
        & (scaled_unit_ok | already_scaled_unit)
        & (microsome | description.str.contains("microsom", regex=False))
        & (liver | description.str.contains("liver", regex=False))
    )
    frame["species"] = species
    frame["log10_clint_ml_min_kg"] = np.log10(frame["standard_value_scaled"].where(positive_value))
    return frame


def aggregate_document_pairs(frame: pd.DataFrame, species: str) -> pd.DataFrame:
    eligible = frame.loc[frame["eligible_record"]].copy()
    rows = []
    for (key, doc_id), group in eligible.groupby(["connectivity_key", "doc_id"], sort=False):
        values = group["log10_clint_ml_min_kg"].to_numpy(dtype=float)
        first = group.iloc[0]
        rows.append(
            {
                "connectivity_key": key,
                "doc_id": int(doc_id),
                "canonical_smiles": first["canonical_smiles_audit"],
                "inchikey": first["inchikey_audit"],
                "f_count": int(first["f_count"]),
                "aryl_f_count": int(first["aryl_f_count"]),
                "cf2_count": int(first["cf2_count"]),
                "cf3_count": int(first["cf3_count"]),
                f"log10_{species}_clint_ml_min_kg": float(np.median(values)),
                f"{species}_record_count": int(len(group)),
                f"{species}_replicate_range_log10": float(values.max() - values.min()),
                f"{species}_assay_ids": ";".join(map(str, sorted(group["assay_id"].unique()))),
                "doc_doi": first["doc_doi"],
                "doc_pubmed_id": first["doc_pubmed_id"],
                "doc_year": first["doc_year"],
                "doc_title": first["doc_title"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hlm_raw = prepare_records(HLM_RAW, "hlm")
    rlm_raw = prepare_records(RLM_RAW, "rlm")
    hlm = aggregate_document_pairs(hlm_raw, "hlm")
    rlm = aggregate_document_pairs(rlm_raw, "rlm")
    document_pairs = hlm.merge(
        rlm.drop(
            columns=[
                "canonical_smiles",
                "inchikey",
                "f_count",
                "aryl_f_count",
                "cf2_count",
                "cf3_count",
                "doc_doi",
                "doc_pubmed_id",
                "doc_year",
                "doc_title",
            ]
        ),
        on=["connectivity_key", "doc_id"],
        how="inner",
        validate="one_to_one",
    )
    document_pairs["combined_replicate_range_log10"] = document_pairs[
        ["hlm_replicate_range_log10", "rlm_replicate_range_log10"]
    ].max(axis=1)
    document_pairs["has_traceable_publication"] = (
        document_pairs["doc_doi"].notna() | document_pairs["doc_pubmed_id"].notna()
    ).astype(int)

    # Select one source document per molecule without using endpoint magnitude.
    document_pairs = document_pairs.sort_values(
        [
            "connectivity_key",
            "has_traceable_publication",
            "combined_replicate_range_log10",
            "doc_id",
        ],
        ascending=[True, False, True, True],
    )
    selected = document_pairs.drop_duplicates("connectivity_key", keep="first").copy()

    exclusions = exclusion_sets()
    for name, keys in exclusions.items():
        selected[f"overlap_{name}"] = selected["connectivity_key"].isin(keys).astype(int)
    overlap_columns = [column for column in selected if column.startswith("overlap_")]
    selected["overlap_any_previous_dataset"] = selected[overlap_columns].max(axis=1)
    selected["is_fluorinated"] = selected["f_count"].gt(0).astype(int)
    selected["primary_external_eligible"] = (
        selected["overlap_any_previous_dataset"].eq(0)
        & selected["has_traceable_publication"].eq(1)
        & selected["combined_replicate_range_log10"].le(0.3)
    ).astype(int)
    selected["external_record_id"] = [f"OA35PAIR_{index:05d}" for index in range(1, len(selected) + 1)]
    selected = selected.sort_values("external_record_id").reset_index(drop=True)

    selected.to_csv(OUT / "paired_external_candidates.csv", index=False)
    selected.loc[selected["primary_external_eligible"].eq(1)].to_csv(
        OUT / "paired_external_primary.csv", index=False
    )
    selected.loc[selected["overlap_any_previous_dataset"].eq(1)].to_csv(
        OUT / "paired_external_excluded_overlap.csv", index=False
    )

    flow = [
        {"stage": "OpenADMET ChEMBL35 raw HLM records", "n": len(hlm_raw)},
        {"stage": "OpenADMET ChEMBL35 raw RLM records", "n": len(rlm_raw)},
        {"stage": "Eligible quantitative HLM records", "n": int(hlm_raw["eligible_record"].sum())},
        {"stage": "Eligible quantitative RLM records", "n": int(rlm_raw["eligible_record"].sum())},
        {"stage": "Same-molecule, same-document pairs", "n": len(document_pairs)},
        {"stage": "Unique paired molecules", "n": len(selected)},
        {"stage": "Unique fluorinated paired molecules", "n": int(selected["is_fluorinated"].sum())},
        {
            "stage": "Independent fluorinated pairs before quality sensitivity filter",
            "n": int((selected["is_fluorinated"].eq(1) & selected["overlap_any_previous_dataset"].eq(0)).sum()),
        },
        {
            "stage": "Primary independent fluorinated external pairs",
            "n": int((selected["is_fluorinated"].eq(1) & selected["primary_external_eligible"].eq(1)).sum()),
        },
    ]
    pd.DataFrame(flow).to_csv(OUT / "curation_flow.csv", index=False)
    summary = {
        "source": "OpenADMET ChEMBL35 microsomal-clearance raw records",
        "source_url": "https://github.com/OpenADMET/data-catalogs/tree/main/catalogs/activities/ChEMBL_Microsomal/ChEMBL35_Microsome",
        "endpoint": "log10 intrinsic clearance scaled to mL min-1 kg-1",
        "pairing_rule": "same standardized connectivity and same ChEMBL document",
        "primary_quality_rule": "exact quantitative values, liver-microsome evidence, traceable DOI/PubMed source, within-species replicate range <=0.3 log10 units",
        "selection_rule_for_multiple_documents": "prefer traceable publication, then smallest combined replicate range, then lowest ChEMBL document ID; endpoint magnitude is not used",
        "counts": {row["stage"]: int(row["n"]) for row in flow},
        "overlap_counts": {column: int(selected[column].sum()) for column in overlap_columns},
        "software": {
            "pandas": pd.__version__,
            "rdkit": getattr(Chem, "__version__", "2022.09.5 project environment"),
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(flow).to_string(index=False))
    print(json.dumps(summary["overlap_counts"], indent=2))


if __name__ == "__main__":
    main()
