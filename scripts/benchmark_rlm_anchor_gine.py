#!/usr/bin/env python3
"""Same-task GINE benchmark for measured-RLM-anchored HLM residual prediction."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_biogen_paired_models as core


TRAIN_FILE = ROOT / "reports" / "biogen_paired_study" / "biogen_paired_modelling_table.csv"
EXTERNAL_FILE = ROOT / "reports" / "openadmet_chembl35_paired_external" / "external_predictions.csv"
AUDIT_FILE = ROOT / "reports" / "openadmet_chembl35_paired_external" / "external_endpoint_semantics_audit.csv"
OUT = ROOT / "reports" / "rlm_anchor_gine_benchmark"
SEEDS = (20260819, 20260820, 20260821)
EPOCHS = 60
BATCH_SIZE = 128


def one_hot(value: object, choices: list[object]) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot(atom.GetAtomicNum(), [1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + one_hot(
            atom.GetHybridization(),
            [
                Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D,
                Chem.HybridizationType.SP3D2,
            ],
        )
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        one_hot(
            bond.GetBondType(),
            [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC],
        )
        + one_hot(
            bond.GetStereo(),
            [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE],
        )
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def graph_from_smiles(smiles: str, row_index: int) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid standardized SMILES at row {row_index}: {smiles}")
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edges, attributes = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        edges.extend([[a, b], [b, a]])
        attributes.extend([feature, feature])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 11), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, record_id=torch.tensor([row_index]))


class ResidualGINE(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 128):
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(3):
            network = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(network, edge_dim=bond_dim))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden + 2, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1),
        )

    def forward(self, batch: Data, rlm_scaled: torch.Tensor, censored: torch.Tensor) -> torch.Tensor:
        x = self.atom_encoder(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, batch.edge_index, batch.edge_attr)) + x)
        pooled = global_mean_pool(x, batch.batch)
        return self.head(torch.cat([pooled, rlm_scaled[:, None], censored[:, None]], dim=1)).squeeze(1)


def fit_predict(
    graphs: list[Data],
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rlm_mean = float(frame.loc[train_index, core.RLM].mean())
    rlm_sd = float(frame.loc[train_index, core.RLM].std())
    residual = frame[core.HLM] - frame[core.RLM]
    target_mean = float(residual.loc[train_index].mean())
    target_sd = float(residual.loc[train_index].std())
    model = ResidualGINE(graphs[0].x.shape[1], graphs[0].edge_attr.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader([graphs[index] for index in train_index], batch_size=BATCH_SIZE, shuffle=True)
    history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in train_loader:
            index = batch.record_id.view(-1).numpy()
            rlm = torch.tensor((frame.loc[index, core.RLM].to_numpy() - rlm_mean) / rlm_sd, dtype=torch.float32)
            censored = torch.tensor(frame.loc[index, "rlm_left_censored"].to_numpy(), dtype=torch.float32)
            target = torch.tensor((residual.loc[index].to_numpy() - target_mean) / target_sd, dtype=torch.float32)
            prediction = model(batch, rlm, censored)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "training_mse_scaled": float(np.mean(losses))})
    model.eval()
    output = np.zeros(len(test_index), dtype=float)
    loader = DataLoader([graphs[index] for index in test_index], batch_size=BATCH_SIZE, shuffle=False)
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            index = batch.record_id.view(-1).numpy()
            rlm = torch.tensor((frame.loc[index, core.RLM].to_numpy() - rlm_mean) / rlm_sd, dtype=torch.float32)
            censored = torch.tensor(frame.loc[index, "rlm_left_censored"].to_numpy(), dtype=torch.float32)
            residual_prediction = model(batch, rlm, censored).numpy() * target_sd + target_mean
            values = frame.loc[index, core.RLM].to_numpy() + residual_prediction
            output[cursor : cursor + len(values)] = values
            cursor += len(values)
    return output, history


def load_external() -> pd.DataFrame:
    external = pd.read_csv(EXTERNAL_FILE)
    audit = pd.read_csv(AUDIT_FILE)[
        ["external_record_id", "strict_total_intrinsic_pair", "source_unit_per_ug_flag", "full_inchikey_overlap"]
    ]
    external = external.merge(audit, on="external_record_id", validate="one_to_one")
    external = external.loc[
        external["is_fluorinated"].eq(1)
        & external["strict_total_intrinsic_pair"].eq(1)
        & external["source_unit_per_ug_flag"].eq(0)
        & external["full_inchikey_overlap"].eq(1)
    ].copy().reset_index(drop=True)
    external[core.HLM] = external["external_hlm"]
    external[core.RLM] = external["external_rlm"]
    external["rlm_left_censored"] = 0
    return external


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_FILE)
    train = train.loc[train[core.HLM].notna() & train[core.RLM].notna()].copy().reset_index(drop=True)
    external = load_external()
    combined = pd.concat([train.assign(_dataset="train"), external.assign(_dataset="external")], ignore_index=True, sort=False)
    graphs = [graph_from_smiles(smiles, index) for index, smiles in enumerate(combined["canonical_smiles"])]
    train_indices = np.arange(len(train))
    external_indices = np.arange(len(train), len(combined))
    internal = train[
        ["Internal ID", "canonical_smiles", "murcko_scaffold", "scaffold_fold", "is_fluorinated",
         "hlm_left_censored", "rlm_left_censored", core.HLM, core.RLM]
    ].copy()
    external_output = external[["external_record_id", "canonical_smiles", "doc_id", "doc_doi", core.HLM, core.RLM]].copy()
    external_output["hlm_left_censored"] = 0
    histories = []
    for seed in SEEDS:
        internal_column = f"prediction__GINE__seed_{seed}"
        internal[internal_column] = np.nan
        for fold in range(5):
            fit_index = train_indices[train["scaffold_fold"].ne(fold).to_numpy()]
            test_index = train_indices[train["scaffold_fold"].eq(fold).to_numpy()]
            prediction, history = fit_predict(graphs, combined, fit_index, test_index, seed)
            internal.loc[test_index, internal_column] = prediction
            histories.append({"scope": "scaffold_cv", "fold": fold, "seed": seed, "history": history})
            print(f"GINE fold={fold} seed={seed}", flush=True)
        external_prediction, history = fit_predict(graphs, combined, train_indices, external_indices, seed)
        external_output[f"prediction__GINE__seed_{seed}"] = external_prediction
        histories.append({"scope": "strict_external", "fold": None, "seed": seed, "history": history})
        print(f"GINE external seed={seed}", flush=True)

    internal_seed_columns = [f"prediction__GINE__seed_{seed}" for seed in SEEDS]
    external_seed_columns = [f"prediction__GINE__seed_{seed}" for seed in SEEDS]
    internal["prediction__GINE__ensemble"] = internal[internal_seed_columns].mean(axis=1)
    external_output["prediction__GINE__ensemble"] = external_output[external_seed_columns].mean(axis=1)
    metric_rows = []
    for seed, column in [(str(seed), f"prediction__GINE__seed_{seed}") for seed in SEEDS] + [("ensemble", "prediction__GINE__ensemble")]:
        for scope, mask in [
            ("all", np.ones(len(internal), dtype=bool)),
            ("fluorinated", internal["is_fluorinated"].eq(1).to_numpy()),
            ("nonfluorinated", internal["is_fluorinated"].eq(0).to_numpy()),
        ]:
            frame = internal.loc[mask].reset_index(drop=True)
            metric_rows.append({"evaluation": "scaffold_cv", "scope": scope, "seed": seed, **core.metrics(frame, frame[column].to_numpy())})
        frame = external_output.reset_index(drop=True)
        metric_rows.append({"evaluation": "strict_external", "scope": "fluorinated", "seed": seed, **core.metrics(frame, frame[column].to_numpy())})
    internal.to_csv(OUT / "scaffold_oof_predictions.csv", index=False)
    external_output.to_csv(OUT / "strict_external_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False)
    (OUT / "training_history.json").write_text(json.dumps(histories) + "\n", encoding="utf-8")
    metadata = {
        "role": "Same-task graph neural network robustness baseline; not selected using external labels.",
        "architecture": "Three-layer GINE with mean graph pooling; measured RLM and RLM censoring indicator concatenated before the regression head.",
        "target": "HLM-RLM residual; predicted HLM equals measured RLM plus predicted residual.",
        "training": f"Fixed {EPOCHS} epochs, AdamW, three seeds; no external hyperparameter tuning.",
        "internal_split": "The same frozen five-fold Murcko scaffold assignment as the tabular benchmark.",
        "external_set": "The same frozen 598-compound strict total-intrinsic fluorinated ChEMBL35 set.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(metric_rows).loc[lambda x: x["seed"].eq("ensemble")].to_string(index=False))


if __name__ == "__main__":
    main()
