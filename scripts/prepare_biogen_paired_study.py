#!/usr/bin/env python3
"""Prepare leakage-audited paired Biogen HLM/RLM modelling records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

from fms.fluoro_features import calculate_fluoro_rdkit_features


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "public_candidates" / "biogen_adme_public_set_3521.csv"
SPLITS = ROOT / "data" / "public_candidates" / "biogen_official_splits"
PREREGISTRATION = ROOT / "docs" / "BIOGEN_PAIRED_STUDY_PREREGISTRATION_20260819.md"
OUT = ROOT / "reports" / "biogen_paired_study"
HLM_COLUMN = "LOG HLM_CLint (mL/min/kg)"
RLM_COLUMN = "LOG RLM_CLint (mL/min/kg)"
Q_HUMAN = 20.7
PCLH_THRESHOLD = 14.3
CLINT_THRESHOLD = PCLH_THRESHOLD * Q_HUMAN / (Q_HUMAN - PCLH_THRESHOLD)
LOG_CLINT_THRESHOLD = math.log10(CLINT_THRESHOLD)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_record(smiles: str) -> dict[str, object]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"valid_structure": False}
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = (
        Chem.MolToSmiles(scaffold_mol, canonical=True)
        if scaffold_mol.GetNumAtoms()
        else canonical
    )
    features = calculate_fluoro_rdkit_features(mol)
    cf3 = features["fluoro_CF3_carbon_count"] > 0
    aryl_f = features["fluoro_aryl_F_count"] > 0
    alkyl_f = features["fluoro_alkyl_F_count"] > 0
    fluoroalkoxy = any(
        atom.GetAtomicNum() == 6
        and any(neighbor.GetAtomicNum() == 9 for neighbor in atom.GetNeighbors())
        and any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors())
        for atom in mol.GetAtoms()
    )
    if cf3:
        subgroup = "CF3-containing"
    elif features["fluoro_CF2_carbon_count"] > 0:
        subgroup = "CF2-containing"
    elif aryl_f:
        subgroup = "aryl-F without CF3"
    elif fluoroalkoxy:
        subgroup = "fluoroalkoxy"
    elif alkyl_f:
        subgroup = "other alkyl-F"
    else:
        subgroup = "nonfluorinated"
    return {
        "valid_structure": True,
        "canonical_smiles": canonical,
        "murcko_scaffold": scaffold,
        "is_fluorinated": int(features["fluoro_has_fluorine"]),
        "fluorine_subgroup": subgroup,
        **features,
    }


def canonical_set(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    output = set()
    for value in frame["smiles"]:
        record = molecule_record(value)
        if record["valid_structure"]:
            output.add(record["canonical_smiles"])
    return output


def assign_scaffold_folds(frame: pd.DataFrame) -> pd.Series:
    eligible = frame[frame[HLM_COLUMN].notna()].copy()
    folds = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    splitter = GroupKFold(n_splits=5)
    for fold, (_, test_index) in enumerate(
        splitter.split(eligible, groups=eligible["murcko_scaffold"])
    ):
        folds.loc[eligible.index[test_index]] = fold
    return folds


def hash_split(smiles: str) -> str:
    value = int(hashlib.sha256(("20260819:" + smiles).encode("utf-8")).hexdigest()[:12], 16)
    return "test" if value % 100 < 20 else "train"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW)
    chemistry = pd.DataFrame([molecule_record(value) for value in raw["SMILES"]])
    data = pd.concat([raw.reset_index(drop=True), chemistry], axis=1)
    data[HLM_COLUMN] = pd.to_numeric(data[HLM_COLUMN], errors="coerce")
    data[RLM_COLUMN] = pd.to_numeric(data[RLM_COLUMN], errors="coerce")

    hlm_train = canonical_set(SPLITS / "ADME_HLM_train.csv")
    hlm_test = canonical_set(SPLITS / "ADME_HLM_test.csv")
    data["official_hlm_split"] = np.select(
        [data["canonical_smiles"].isin(hlm_train), data["canonical_smiles"].isin(hlm_test)],
        ["train", "test"],
        default="not_in_hlm_endpoint",
    )
    data["hash_hlm_split"] = np.where(
        data[HLM_COLUMN].notna(), data["canonical_smiles"].map(hash_split), "not_in_hlm_endpoint"
    )
    data["scaffold_fold"] = assign_scaffold_folds(data)

    hlm_boundary = data[HLM_COLUMN].min()
    rlm_boundary = data[RLM_COLUMN].min()
    data["hlm_left_censored"] = data[HLM_COLUMN].eq(hlm_boundary).astype(int)
    data["rlm_left_censored"] = data[RLM_COLUMN].eq(rlm_boundary).astype(int)
    data["paired_endpoint_available"] = data[[HLM_COLUMN, RLM_COLUMN]].notna().all(axis=1).astype(int)
    data["log_clint_human_minus_rat"] = data[HLM_COLUMN] - data[RLM_COLUMN]
    data["hlm_low_clearance"] = np.where(
        data[HLM_COLUMN].notna(), (data[HLM_COLUMN] <= LOG_CLINT_THRESHOLD).astype(int), np.nan
    )
    data.to_csv(OUT / "biogen_paired_modelling_table.csv", index=False)

    eligible = data[data[HLM_COLUMN].notna()]
    paired = data[data["paired_endpoint_available"].eq(1)]
    exact_split_overlap = hlm_train & hlm_test
    scaffold_overlap = set(
        eligible.loc[eligible["official_hlm_split"].eq("train"), "murcko_scaffold"]
    ) & set(eligible.loc[eligible["official_hlm_split"].eq("test"), "murcko_scaffold"])
    hash_scaffold_overlap = set(
        eligible.loc[eligible["hash_hlm_split"].eq("train"), "murcko_scaffold"]
    ) & set(eligible.loc[eligible["hash_hlm_split"].eq("test"), "murcko_scaffold"])
    summary = {
        "raw_records": len(data),
        "valid_structures": int(data["valid_structure"].sum()),
        "unique_canonical_structures": int(data["canonical_smiles"].nunique()),
        "hlm_nonmissing": int(data[HLM_COLUMN].notna().sum()),
        "rlm_nonmissing": int(data[RLM_COLUMN].notna().sum()),
        "paired_nonmissing": len(paired),
        "fluorinated_paired_nonmissing": int(paired["is_fluorinated"].sum()),
        "official_hlm_train": int(eligible["official_hlm_split"].eq("train").sum()),
        "official_hlm_test": int(eligible["official_hlm_split"].eq("test").sum()),
        "official_hlm_fluorinated_test": int(
            (eligible["official_hlm_split"].eq("test") & eligible["is_fluorinated"].eq(1)).sum()
        ),
        "official_hlm_fluorinated_paired_test": int(
            (
                eligible["official_hlm_split"].eq("test")
                & eligible["is_fluorinated"].eq(1)
                & eligible["paired_endpoint_available"].eq(1)
            ).sum()
        ),
        "hash_hlm_train": int(eligible["hash_hlm_split"].eq("train").sum()),
        "hash_hlm_test": int(eligible["hash_hlm_split"].eq("test").sum()),
        "hash_hlm_fluorinated_test": int(
            (eligible["hash_hlm_split"].eq("test") & eligible["is_fluorinated"].eq(1)).sum()
        ),
        "hash_hlm_fluorinated_paired_test": int(
            (
                eligible["hash_hlm_split"].eq("test")
                & eligible["is_fluorinated"].eq(1)
                & eligible["paired_endpoint_available"].eq(1)
            ).sum()
        ),
        "hash_shared_scaffolds": len(hash_scaffold_overlap),
        "hash_test_fraction_on_shared_scaffolds": float(
            eligible.loc[eligible["hash_hlm_split"].eq("test"), "murcko_scaffold"]
            .isin(hash_scaffold_overlap)
            .mean()
        ),
        "official_exact_structure_overlap": len(exact_split_overlap),
        "official_shared_scaffolds": len(scaffold_overlap),
        "official_test_fraction_on_shared_scaffolds": float(
            eligible.loc[eligible["official_hlm_split"].eq("test"), "murcko_scaffold"]
            .isin(scaffold_overlap)
            .mean()
        ),
        "hlm_left_censor_boundary": float(hlm_boundary),
        "hlm_left_censored": int(data["hlm_left_censored"].sum()),
        "rlm_left_censor_boundary": float(rlm_boundary),
        "rlm_left_censored": int(data["rlm_left_censored"].sum()),
        "hlm_log_clint_low_clearance_threshold": LOG_CLINT_THRESHOLD,
        "preregistration_sha256": sha256(PREREGISTRATION),
        "raw_data_sha256": sha256(RAW),
    }
    (OUT / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
