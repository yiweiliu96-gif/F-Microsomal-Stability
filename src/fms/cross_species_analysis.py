"""Analyze compounds shared between HLM and RLM and their label concordance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .fluoro_features import calculate_fluoro_rdkit_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports/v3_publication/cross_species")
    return parser.parse_args()


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else ""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df["_canonical"] = df["smiles"].map(canonical)
    slim = df[["_canonical", "species", "label", "compound_id", "smiles"]].copy()
    duplicate_counts = slim.groupby(["_canonical", "species"]).size()
    unambiguous = duplicate_counts.loc[duplicate_counts.eq(1)].index
    slim = slim.set_index(["_canonical", "species"]).loc[unambiguous].reset_index()
    pivot = slim.pivot(index="_canonical", columns="species", values="label").dropna()
    shared = pivot.rename(columns={"HLM": "hlm_label", "RLM": "rlm_label"}).reset_index()
    shared["concordant"] = shared["hlm_label"].eq(shared["rlm_label"])
    shared["pattern"] = shared.apply(
        lambda row: f"HLM_{int(row.hlm_label)}_RLM_{int(row.rlm_label)}", axis=1
    )
    smiles_map = slim.drop_duplicates("_canonical").set_index("_canonical")["smiles"]
    shared["smiles"] = shared["_canonical"].map(smiles_map)

    feature_rows = []
    for smiles in shared["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        feature_rows.append(calculate_fluoro_rdkit_features(mol, compute_3d=False))
    features = pd.DataFrame(feature_rows)
    shared = pd.concat([shared.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    shared.to_csv(output_dir / "shared_hlm_rlm_compounds.csv", index=False)

    summary = (
        shared.groupby("pattern")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    summary["fraction"] = summary["n"] / len(shared) if len(shared) else 0
    summary.to_csv(output_dir / "label_concordance_summary.csv", index=False)

    fluoro_cols = [
        col
        for col in shared.columns
        if col.startswith("fluoro_") and pd.api.types.is_numeric_dtype(shared[col])
    ]
    effect_rows = []
    for col in fluoro_cols:
        for pattern, part in shared.groupby("pattern"):
            effect_rows.append(
                {
                    "feature": col,
                    "pattern": pattern,
                    "n": len(part),
                    "mean": part[col].mean(),
                    "prevalence_gt_0": part[col].gt(0).mean(),
                }
            )
    pd.DataFrame(effect_rows).to_csv(
        output_dir / "cross_species_fluoro_feature_summary.csv", index=False
    )

    md = [
        "# HLM-RLM Cross-Species Analysis",
        "",
        f"- Unambiguous structures shared by HLM and RLM: {len(shared)}",
        f"- Concordant labels: {int(shared['concordant'].sum())}",
        f"- Discordant labels: {int((~shared['concordant']).sum())}",
        f"- Concordance fraction: {shared['concordant'].mean():.3f}" if len(shared) else "",
        "",
        "Patterns use project labels: 1=stable and 0=unstable.",
    ]
    (output_dir / "cross_species_summary.md").write_text(
        "\n".join(line for line in md if line != "") + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
