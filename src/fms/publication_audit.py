"""Publication-focused data integrity and split leakage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, inchi

from .scaffold_split import murcko_scaffold
from .train import normalize_species, normalize_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports/v3_publication/data_audit")
    parser.add_argument("--split-col", default="scaffold_split")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--species-col", default="species")
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    return parser.parse_args()


def structure_keys(smiles: object) -> dict[str, object]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "audit_valid_smiles": False,
            "audit_canonical_smiles": "",
            "audit_inchikey_connectivity": "",
            "audit_scaffold": "",
            "audit_fp": None,
        }
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        connectivity = inchi.MolToInchiKey(mol).split("-")[0]
    except Exception:
        connectivity = ""
    fp = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    return {
        "audit_valid_smiles": True,
        "audit_canonical_smiles": canonical,
        "audit_inchikey_connectivity": connectivity,
        "audit_scaffold": murcko_scaffold(canonical),
        "audit_fp": fp,
    }


def overlap_rows(df: pd.DataFrame, key: str, split_col: str, species_col: str) -> pd.DataFrame:
    usable = df.loc[df[key].fillna("").astype(str).ne("")].copy()
    grouped = (
        usable.groupby([species_col, key], dropna=False)[split_col]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="splits")
    )
    grouped["n_splits"] = grouped["splits"].map(len)
    return grouped.loc[grouped["n_splits"].gt(1)].copy()


def nearest_cross_split(df: pd.DataFrame, species_col: str, split_col: str) -> pd.DataFrame:
    rows = []
    for species, sdf in df.groupby(species_col):
        train = sdf.loc[sdf[split_col].eq("train") & sdf["audit_fp"].notna()]
        train_fps = train["audit_fp"].tolist()
        train_ids = train.index.tolist()
        for idx, row in sdf.loc[sdf[split_col].ne("train") & sdf["audit_fp"].notna()].iterrows():
            similarities = DataStructs.BulkTanimotoSimilarity(row["audit_fp"], train_fps)
            if not similarities:
                continue
            best_pos = int(np.argmax(similarities))
            rows.append(
                {
                    "row_index": idx,
                    "species": species,
                    "eval_split": row[split_col],
                    "nearest_train_row_index": train_ids[best_pos],
                    "max_tanimoto_to_train": float(similarities[best_pos]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df["_audit_species"] = df[args.species_col].map(normalize_species)
    df["_audit_split"] = df[args.split_col].map(normalize_split_name)

    keys = df[args.smiles_col].map(structure_keys).apply(pd.Series)
    audited = pd.concat([df, keys], axis=1)

    canonical_overlap = overlap_rows(
        audited, "audit_canonical_smiles", "_audit_split", "_audit_species"
    )
    connectivity_overlap = overlap_rows(
        audited, "audit_inchikey_connectivity", "_audit_split", "_audit_species"
    )
    scaffold_overlap = overlap_rows(audited, "audit_scaffold", "_audit_split", "_audit_species")

    label_conflicts = (
        audited.loc[audited["audit_inchikey_connectivity"].ne("")]
        .groupby(["_audit_species", "audit_inchikey_connectivity"])[args.label_col]
        .agg(["nunique", "count", "min", "max"])
        .reset_index()
    )
    label_conflicts = label_conflicts.loc[label_conflicts["nunique"].gt(1)].copy()

    nearest = nearest_cross_split(audited, "_audit_species", "_audit_split")
    if not nearest.empty:
        nearest["above_similarity_threshold"] = nearest["max_tanimoto_to_train"].ge(
            args.similarity_threshold
        )

    canonical_overlap.to_csv(output_dir / "canonical_smiles_cross_split.csv", index=False)
    connectivity_overlap.to_csv(output_dir / "connectivity_cross_split.csv", index=False)
    scaffold_overlap.to_csv(output_dir / "scaffold_cross_split.csv", index=False)
    label_conflicts.to_csv(output_dir / "label_conflicts.csv", index=False)
    nearest.to_csv(output_dir / "nearest_train_similarity.csv", index=False)

    split_counts = (
        audited.groupby(["_audit_species", "_audit_split", args.label_col])
        .size()
        .reset_index(name="n")
    )
    split_counts.to_csv(output_dir / "split_class_counts.csv", index=False)

    similarity_summary = []
    if not nearest.empty:
        for (species, split), part in nearest.groupby(["species", "eval_split"]):
            similarity_summary.append(
                {
                    "species": species,
                    "eval_split": split,
                    "n": len(part),
                    "mean_max_tanimoto": part["max_tanimoto_to_train"].mean(),
                    "median_max_tanimoto": part["max_tanimoto_to_train"].median(),
                    "p90_max_tanimoto": part["max_tanimoto_to_train"].quantile(0.9),
                    "fraction_ge_threshold": part["above_similarity_threshold"].mean(),
                }
            )
    similarity_summary_df = pd.DataFrame(similarity_summary)
    similarity_summary_df.to_csv(output_dir / "nearest_similarity_summary.csv", index=False)

    summary = {
        "input": args.input,
        "n_rows": int(len(audited)),
        "n_invalid_smiles": int((~audited["audit_valid_smiles"]).sum()),
        "n_canonical_cross_split_groups": int(len(canonical_overlap)),
        "n_connectivity_cross_split_groups": int(len(connectivity_overlap)),
        "n_scaffold_cross_split_groups": int(len(scaffold_overlap)),
        "n_label_conflict_groups": int(len(label_conflicts)),
        "similarity_threshold": args.similarity_threshold,
        "label_semantics": "label=1 means stable; imported article Class=1 means unstable",
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# V3 Publication Data Audit",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Invalid SMILES: {summary['n_invalid_smiles']}",
        f"- Canonical-SMILES groups crossing splits: {summary['n_canonical_cross_split_groups']}",
        f"- Connectivity groups crossing splits: {summary['n_connectivity_cross_split_groups']}",
        f"- Murcko-scaffold groups crossing splits: {summary['n_scaffold_cross_split_groups']}",
        f"- Conflicting-label connectivity groups: {summary['n_label_conflict_groups']}",
        "",
        "Label convention: the source article uses Class=1 for unstable compounds; this project uses "
        "`label=1` for stable compounds.",
        "",
        "Near-neighbor similarity is diagnostic, not automatically treated as leakage. See "
        "`nearest_similarity_summary.csv` and `nearest_train_similarity.csv`.",
    ]
    (output_dir / "audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
