# Assay-matched fluorination and RLM-anchored HLM prediction

This repository accompanies the manuscript:

> Assay-Matched Fluorination Reveals Species-Discordant Clearance Changes and Enables RLM-Anchored HLM Prediction

The study asks two linked medicinal-chemistry questions:

1. When an exact fluorination edit is measured under matched assay conditions, does its clearance effect transfer between rat and human liver microsomes?
2. After quantitative rat liver microsomal intrinsic clearance (RLM CLint) is available for the same compound, can it improve prediction of human liver microsomal intrinsic clearance (HLM CLint)?

The model predicts an in vitro continuous HLM endpoint. It does not predict human pharmacokinetics, clinical exposure, or in vivo species translation.

## Frozen findings

- Same-assay fluorination pairs: 476 HLM pairs and 209 RLM pairs.
- Cross-species exact edits: 97 manually verified pairs from 42 documents; Spearman correlation between Delta HLM and Delta RLM = 0.431; direction reversal = 38.1%.
- Internal fluorinated scaffold validation: structure-only interval RMSE = 0.453; RLM-anchored residual interval RMSE = 0.303.
- Strict external fluorinated set: structure-only RMSE = 1.003; linear RLM-to-HLM RMSE = 0.705; RLM-anchored residual RMSE = 0.615.
- Residual versus linear RLM translation: RMSE difference = -0.090 (document-bootstrap 95% CI, -0.177 to -0.007).
- Handcrafted fluorine descriptors and neural representations did not improve overall accuracy. Fluorine annotations are retained for subgroup and applicability-domain analysis.

## Repository map

- `data/development/`: public HLM and RLM split files used to reconstruct paired model-development compounds.
- `data/derived/`: standardized modeling tables, assay-matched fluorination pairs, source-row audits, manual review status, endpoint-semantics audits, and molecule-level predictions.
- `models/`: frozen LightGBM model bundle and model card.
- `results/`: machine-readable internal, external, algorithm, bootstrap, motif, falsification, and uncertainty results.
- `scripts/`: data preparation, model fitting, external evaluation, matched-pair audit, sensitivity, and figure scripts.
- `src/fms/`: shared molecular-feature, splitting, statistics, and audit utilities.
- `figures/`: editable SVG versions of the five main-text figures and Supplementary Figure S1.
- `docs/`: data-availability and reproducibility notes.
- `verify_release.py`: dependency-light integrity and claim audit for the release.

See `data/README.md`, `models/MODEL_CARD.md`, `results/README.md`, and `DATA_AND_REVIEW_NOTICE.md` before reuse.

## Quick integrity check

```bash
python verify_release.py
```

The check verifies required files, manually reviewed pair counts, principal manuscript-facing metrics, prohibited publisher files, local absolute paths, and the frozen model checksum.

## Environment

The primary tabular workflow used Python 3.10.9. Install the environment with:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric are required only for the GINE robustness benchmark. Platform-specific wheels may be necessary.

## Core reproduction sequence

Run commands from the repository root. OpenADMET ChEMBL35 catalog inputs that are not redistributed must be obtained from the cited public source and placed in the paths documented by the scripts.

```bash
python scripts/prepare_biogen_paired_study.py
python scripts/train_biogen_paired_models.py
python scripts/evaluate_biogen_scaffold_cv.py
python scripts/compare_paired_internal_baselines.py
python scripts/build_openadmet_chembl35_paired_external.py
python scripts/compare_strict_external_baselines.py
python scripts/audit_assay_matched_fluorination_pairs.py
python scripts/audit_cross_species_pair_source_rows.py
python scripts/sensitivity_assay_matched_fluorination_effects.py
python scripts/evaluate_cross_species_fluorination_delta.py
python scripts/analyze_submission_strengthening.py
```

Algorithm and graph-network analyses were robustness controls and were not used to replace the prespecified LightGBM model after external endpoints were examined. Anchor-falsification, cross-conformal, and magnitude-stratified analyses are labeled as post hoc analyses in their metadata.

## Data provenance and licensing

The study reuses public data reported by Fang et al. and ChEMBL35 records curated through OpenADMET. Source publications, database identifiers, versions, and access conditions are documented in `DATA_AND_REVIEW_NOTICE.md`. The MIT licence applies to repository software only. Third-party and derived data retain their original provenance and applicable source terms.

Publisher PDFs and Supporting Information files are intentionally excluded.

## Citation and versioned release

Use `CITATION.cff` to cite this software and data package. The public repository is available at
<https://github.com/yiweiliu96-gif/F-Microsomal-Stability>, and the exact manuscript-associated
snapshot is tagged as `v1.0.0`. A repository DOI will be added if the tagged release is archived in
Zenodo; no DOI is claimed in the current release.
