#!/usr/bin/env python3
"""Trace assay-matched fluorination pairs back to raw ChEMBL activity rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_openadmet_chembl35_paired_external import HLM_RAW, RLM_RAW, molecule_key


PAIR_FILE = ROOT / "reports" / "assay_matched_fluorination_pairs" / "cross_species_manual_review_queue.csv"
OUT = ROOT / "reports" / "assay_matched_fluorination_pairs"


def text_set(frame: pd.DataFrame, column: str) -> str:
    return ";".join(sorted(frame[column].dropna().astype(str).unique()))


def prepare_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    structure = frame["OPENADMET_CANONICAL_SMILES"].fillna(frame["canonical_smiles"])
    frame["connectivity_key_audit"] = [molecule_key(value)[2] for value in structure]
    values = pd.to_numeric(frame["standard_value_scaled"], errors="coerce")
    frame["log10_scaled_audit"] = np.log10(values.where(values > 0))
    return frame


def endpoint_audit(
    raw: pd.DataFrame,
    assay_id: int,
    base_key: str,
    fluorinated_key: str,
    expected_base: float,
    expected_fluorinated: float,
    species: str,
) -> dict[str, object]:
    assay = raw.loc[raw["assay_id"].eq(assay_id)].copy()
    base = assay.loc[assay["connectivity_key_audit"].eq(base_key)].copy()
    fluorinated = assay.loc[assay["connectivity_key_audit"].eq(fluorinated_key)].copy()
    prefix = species.lower()
    base_value = float(base["log10_scaled_audit"].median()) if len(base) else np.nan
    fluorinated_value = float(fluorinated["log10_scaled_audit"].median()) if len(fluorinated) else np.nan
    descriptions = text_set(assay, "assay_description").lower()
    tissue = text_set(assay, "assay_tissue").lower()
    fraction = text_set(assay, "assay_subcellular_fraction").lower()
    organism = text_set(assay, "assay_organism").lower()
    source_type_match = text_set(base, "standard_type") == text_set(fluorinated, "standard_type")
    source_unit_match = (
        text_set(base, "standard_units") == text_set(fluorinated, "standard_units")
        and text_set(base, "scaled_units") == text_set(fluorinated, "scaled_units")
    )
    document_match = (
        len(base) > 0
        and len(fluorinated) > 0
        and set(base["doc_id"].dropna().astype(int)) == set(fluorinated["doc_id"].dropna().astype(int))
    )
    exact_relation = base["standard_relation"].fillna("=").eq("=").all() and fluorinated[
        "standard_relation"
    ].fillna("=").eq("=").all()
    semantic_ok = (
        ("microsom" in descriptions or "microsome" in fraction)
        and ("liver" in descriptions or "liver" in tissue)
        and ((species == "HLM" and "homo sapiens" in organism) or (species == "RLM" and "rattus norvegicus" in organism))
    )
    value_match = (
        np.isfinite(base_value)
        and np.isfinite(fluorinated_value)
        and abs(base_value - expected_base) <= 1e-10
        and abs(fluorinated_value - expected_fluorinated) <= 1e-10
    )
    passed = all(
        [len(base) > 0, len(fluorinated) > 0, source_type_match, source_unit_match, document_match, exact_relation, semantic_ok, value_match]
    )
    return {
        f"{prefix}_raw_base_rows": len(base),
        f"{prefix}_raw_fluorinated_rows": len(fluorinated),
        f"{prefix}_raw_document_ids": text_set(assay, "doc_id"),
        f"{prefix}_raw_standard_types": text_set(assay, "standard_type"),
        f"{prefix}_raw_standard_units": text_set(assay, "standard_units"),
        f"{prefix}_raw_scaled_units": text_set(assay, "scaled_units"),
        f"{prefix}_raw_assay_description": text_set(assay, "assay_description"),
        f"{prefix}_raw_organism": text_set(assay, "assay_organism"),
        f"{prefix}_raw_tissue": text_set(assay, "assay_tissue"),
        f"{prefix}_raw_subcellular_fraction": text_set(assay, "assay_subcellular_fraction"),
        f"{prefix}_recomputed_base_log10_clint": base_value,
        f"{prefix}_recomputed_fluorinated_log10_clint": fluorinated_value,
        f"{prefix}_source_endpoint_type_match": int(source_type_match),
        f"{prefix}_source_unit_match": int(source_unit_match),
        f"{prefix}_source_document_match": int(document_match),
        f"{prefix}_quantitative_equality_relations": int(exact_relation),
        f"{prefix}_liver_microsome_species_semantics": int(semantic_ok),
        f"{prefix}_aggregated_values_reproduced": int(value_match),
        f"{prefix}_automated_source_row_pass": int(passed),
    }


def main() -> None:
    pairs = pd.read_csv(PAIR_FILE)
    hlm = prepare_raw(HLM_RAW)
    rlm = prepare_raw(RLM_RAW)
    rows = []
    for pair in pairs.itertuples(index=False):
        base_key = molecule_key(pair.base_smiles)[2]
        fluorinated_key = molecule_key(pair.fluorinated_smiles)[2]
        row = pair._asdict()
        row["base_connectivity_key_recomputed"] = base_key
        row["fluorinated_connectivity_key_recomputed"] = fluorinated_key
        row.update(
            endpoint_audit(
                hlm, int(pair.selected_hlm_assay_id), base_key, fluorinated_key,
                float(pair.assay_matched_base_hlm), float(pair.assay_matched_fluorinated_hlm), "HLM",
            )
        )
        row.update(
            endpoint_audit(
                rlm, int(pair.selected_rlm_assay_id), base_key, fluorinated_key,
                float(pair.assay_matched_base_rlm), float(pair.assay_matched_fluorinated_rlm), "RLM",
            )
        )
        row["automated_source_row_audit_pass"] = int(
            row["hlm_automated_source_row_pass"] and row["rlm_automated_source_row_pass"]
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["review_priority", "document_id", "pair_id"])
    result.to_csv(OUT / "cross_species_source_row_audit.csv", index=False)
    failures = result.loc[result["automated_source_row_audit_pass"].eq(0)].copy()
    failures.to_csv(OUT / "cross_species_source_row_audit_failures.csv", index=False)
    summary = {
        "n_pairs": len(result),
        "n_automated_pass": int(result["automated_source_row_audit_pass"].sum()),
        "n_automated_fail": int((1 - result["automated_source_row_audit_pass"]).sum()),
        "checks": [
            "selected assay contains both molecular connectivity keys",
            "quantitative equality relations",
            "matching parent/fluorinated endpoint types and unit provenance",
            "matching source document within species",
            "species-specific liver-microsome semantics",
            "exact reproduction of aggregated log10 CLint values from raw scaled rows",
        ],
        "remaining_manual_check": "Automated passing does not verify chemical structures and values against the original article or Supporting Information table.",
    }
    (OUT / "cross_species_source_row_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if len(failures):
        print(failures[["pair_id", "document_id", "hlm_automated_source_row_pass", "rlm_automated_source_row_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
