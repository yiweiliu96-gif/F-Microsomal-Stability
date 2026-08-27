"""Fluorine-specific RDKit descriptors used in publication analyses."""

from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _safe_min(values: list[float]) -> float:
    return float(np.min(values)) if values else math.nan


def _safe_max(values: list[float]) -> float:
    return float(np.max(values)) if values else math.nan


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def _atom_charge(atom: Chem.Atom) -> float | None:
    if not atom.HasProp("_GasteigerCharge"):
        return None
    try:
        value = float(atom.GetProp("_GasteigerCharge"))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def calculate_fluoro_rdkit_features(mol: Chem.Mol | None, compute_3d: bool = False) -> dict[str, float]:
    """Return fluorine motif descriptors for an RDKit molecule.

    The ``compute_3d`` argument is accepted for API compatibility with older
    project modules; this lightweight portable implementation only uses 2D
    graph descriptors.
    """

    if mol is None:
        return {
            "fluoro_F_count": 0,
            "fluoro_has_fluorine": 0,
            "fluoro_F_fraction_heavy": 0.0,
            "fluoro_fluorinated_carbon_count": 0,
            "fluoro_max_F_per_carbon": 0,
            "fluoro_single_F_carbon_count": 0,
            "fluoro_CF2_carbon_count": 0,
            "fluoro_CF3_carbon_count": 0,
            "fluoro_CHF_carbon_count": 0,
            "fluoro_CH2F_carbon_count": 0,
            "fluoro_trifluoromethyl_count": 0,
            "fluoro_aryl_F_count": 0,
            "fluoro_alkyl_F_count": 0,
            "fluoro_vinyl_F_count": 0,
            "fluoro_ring_bound_F_count": 0,
            "fluoro_fluorinated_aromatic_ring_count": 0,
            "fluoro_aryl_CF3_count": 0,
            "fluoro_alpha_hetero_F_count": 0,
            "fluoro_beta_hetero_F_count": 0,
            "fluoro_F_within_2_bonds_hetero_count": 0,
            "fluoro_F_within_3_bonds_hetero_count": 0,
            "fluoro_F_to_hetero_min_distance": math.nan,
            "fluoro_F_to_hetero_mean_min_distance": math.nan,
            "fluoro_ortho_hetero_aromatic_F_count": 0,
            "fluoro_meta_hetero_aromatic_F_count": 0,
            "fluoro_para_hetero_aromatic_F_count": 0,
            "fluoro_F_atom_gasteiger_charge_mean": math.nan,
            "fluoro_F_atom_gasteiger_charge_min": math.nan,
            "fluoro_F_atom_gasteiger_charge_max": math.nan,
            "fluoro_CF_carbon_gasteiger_charge_mean": math.nan,
            "fluoro_CF_carbon_degree_mean": math.nan,
            "fluoro_CF_carbon_total_h_mean": math.nan,
            "fluoro_CF_bond_length_mean_A": math.nan,
            "fluoro_CF_bond_length_min_A": math.nan,
            "fluoro_CF_bond_length_max_A": math.nan,
            "fluoro_F_substituent_entropy": math.nan,
        }

    f_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9]
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    fluorinated_carbons: dict[int, list[Chem.Atom]] = {}
    aryl_f = alkyl_f = vinyl_f = ring_bound_f = 0

    for f_atom in f_atoms:
        if not f_atom.GetNeighbors():
            continue
        nbr = f_atom.GetNeighbors()[0]
        if nbr.GetAtomicNum() != 6:
            continue
        fluorinated_carbons.setdefault(nbr.GetIdx(), []).append(f_atom)
        if nbr.GetIsAromatic():
            aryl_f += 1
        elif nbr.GetHybridization().name == "SP2":
            vinyl_f += 1
        else:
            alkyl_f += 1
        if nbr.IsInRing():
            ring_bound_f += 1

    f_per_carbon = {idx: len(values) for idx, values in fluorinated_carbons.items()}
    cf3 = sum(1 for count in f_per_carbon.values() if count == 3)
    cf2 = sum(1 for count in f_per_carbon.values() if count == 2)
    single_f = sum(1 for count in f_per_carbon.values() if count == 1)
    chf = 0
    ch2f = 0
    aryl_cf3 = 0
    trifluoromethyl = 0
    fluorinated_aromatic_rings = set()
    cf_carbon_degrees = []
    cf_carbon_hs = []
    cf_carbon_charges = []
    for carbon_idx, count in f_per_carbon.items():
        atom = mol.GetAtomWithIdx(carbon_idx)
        cf_carbon_degrees.append(atom.GetDegree())
        cf_carbon_hs.append(atom.GetTotalNumHs())
        charge = _atom_charge(atom)
        if charge is not None:
            cf_carbon_charges.append(charge)
        if count == 1 and atom.GetTotalNumHs() >= 1:
            chf += 1
        if count == 1 and atom.GetTotalNumHs() == 2:
            ch2f += 1
        if atom.GetIsAromatic():
            atom_ring_info = mol.GetRingInfo().AtomRings()
            for ring in atom_ring_info:
                if carbon_idx in ring and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    fluorinated_aromatic_rings.add(tuple(sorted(ring)))
        if count == 3:
            if any(nbr.GetAtomicNum() == 6 for nbr in atom.GetNeighbors()):
                trifluoromethyl += 1
            if any(nbr.GetAtomicNum() == 6 and nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                aryl_cf3 += 1

    hetero_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in {7, 8, 15, 16}
    ]
    f_to_hetero_minima = []
    within_2 = within_3 = alpha_hetero = beta_hetero = 0
    ortho_hetero = meta_hetero = para_hetero = 0
    for f_atom in f_atoms:
        if not hetero_indices:
            continue
        distances = []
        for hetero_idx in hetero_indices:
            path = Chem.GetShortestPath(mol, f_atom.GetIdx(), hetero_idx)
            if path:
                distances.append((len(path) - 1, mol.GetAtomWithIdx(hetero_idx).GetAtomicNum()))
        if not distances:
            continue
        min_dist = min(distance for distance, _ in distances)
        f_to_hetero_minima.append(min_dist)
        if min_dist <= 2:
            within_2 += 1
        if min_dist <= 3:
            within_3 += 1
        if min_dist == 2:
            alpha_hetero += 1
        if min_dist == 3 or (
            min_dist == 2 and any(atom_num in {15, 16} for distance, atom_num in distances if distance == min_dist)
        ):
            beta_hetero += 1

        nbrs = f_atom.GetNeighbors()
        if nbrs and nbrs[0].GetIsAromatic():
            carbon_idx = nbrs[0].GetIdx()
            aromatic_position_flags = {"ortho": False, "meta": False, "para": False}
            for hetero_idx in hetero_indices:
                hetero = mol.GetAtomWithIdx(hetero_idx)
                if not hetero.GetIsAromatic():
                    continue
                path = Chem.GetShortestPath(mol, carbon_idx, hetero_idx)
                bond_distance = len(path) - 1
                if bond_distance == 1:
                    aromatic_position_flags["ortho"] = True
                elif bond_distance == 2:
                    aromatic_position_flags["meta"] = True
                elif bond_distance == 3:
                    aromatic_position_flags["para"] = True
            ortho_hetero += int(aromatic_position_flags["ortho"])
            meta_hetero += int(aromatic_position_flags["meta"])
            para_hetero += int(aromatic_position_flags["para"])

    f_atom_charges = []
    for atom in f_atoms:
        charge = _atom_charge(atom)
        if charge is not None:
            f_atom_charges.append(charge)

    return {
        "fluoro_F_count": len(f_atoms),
        "fluoro_has_fluorine": int(len(f_atoms) > 0),
        "fluoro_F_fraction_heavy": float(len(f_atoms) / heavy) if heavy else 0.0,
        "fluoro_fluorinated_carbon_count": len(fluorinated_carbons),
        "fluoro_max_F_per_carbon": max(f_per_carbon.values()) if f_per_carbon else 0,
        "fluoro_single_F_carbon_count": single_f,
        "fluoro_CF2_carbon_count": cf2,
        "fluoro_CF3_carbon_count": cf3,
        "fluoro_CHF_carbon_count": chf,
        "fluoro_CH2F_carbon_count": ch2f,
        "fluoro_trifluoromethyl_count": trifluoromethyl,
        "fluoro_aryl_F_count": aryl_f,
        "fluoro_alkyl_F_count": alkyl_f,
        "fluoro_vinyl_F_count": vinyl_f,
        "fluoro_ring_bound_F_count": ring_bound_f,
        "fluoro_fluorinated_aromatic_ring_count": len(fluorinated_aromatic_rings),
        "fluoro_aryl_CF3_count": aryl_cf3,
        "fluoro_alpha_hetero_F_count": alpha_hetero,
        "fluoro_beta_hetero_F_count": beta_hetero,
        "fluoro_F_within_2_bonds_hetero_count": within_2,
        "fluoro_F_within_3_bonds_hetero_count": within_3,
        "fluoro_F_to_hetero_min_distance": _safe_min(f_to_hetero_minima),
        "fluoro_F_to_hetero_mean_min_distance": _safe_mean(f_to_hetero_minima),
        "fluoro_ortho_hetero_aromatic_F_count": ortho_hetero,
        "fluoro_meta_hetero_aromatic_F_count": meta_hetero,
        "fluoro_para_hetero_aromatic_F_count": para_hetero,
        "fluoro_F_atom_gasteiger_charge_mean": _safe_mean(f_atom_charges),
        "fluoro_F_atom_gasteiger_charge_min": _safe_min(f_atom_charges),
        "fluoro_F_atom_gasteiger_charge_max": _safe_max(f_atom_charges),
        "fluoro_CF_carbon_gasteiger_charge_mean": _safe_mean(cf_carbon_charges),
        "fluoro_CF_carbon_degree_mean": _safe_mean(cf_carbon_degrees),
        "fluoro_CF_carbon_total_h_mean": _safe_mean(cf_carbon_hs),
        "fluoro_CF_bond_length_mean_A": math.nan,
        "fluoro_CF_bond_length_min_A": math.nan,
        "fluoro_CF_bond_length_max_A": math.nan,
        "fluoro_F_substituent_entropy": _entropy(list(f_per_carbon.values())) if f_per_carbon else math.nan,
    }
