"""Murcko scaffold helpers for split audits."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: object) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
