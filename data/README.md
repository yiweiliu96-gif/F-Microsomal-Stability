# Data dictionary and provenance

## Endpoint convention

HLM and RLM values are log10 intrinsic-clearance measurements. The source unit used by the public industrial paired data is mL min-1 kg-1. ChEMBL-derived records are retained only under the endpoint and unit rules documented in the manuscript and scripts. A negative fluorination-pair delta means lower clearance after fluorination.

## Development data (training/test sets)

- `development/ADME_HLM_train.csv` and `development/ADME_HLM_test.csv`: structure and HLM activity split files from the cited public paired ADME release.
- `development/ADME_RLM_train.csv` and `development/ADME_RLM_test.csv`: corresponding RLM split files.

Columns: `smiles` is the source molecular structure string; `activity` is log10 CLint.

## Principal derived files

- `derived/biogen_paired_modelling_table.csv`: standardized paired modeling table, scaffold assignments, censoring indicators, fluorine annotations, and continuous HLM/RLM endpoints.
- `derived/external_predictions.csv`: external validation compounds with observed HLM/RLM endpoints and predictions from the manuscript-facing baselines and ablations.

## Key fields

- `canonical_smiles`: standardized isomeric molecular structure.
- `connectivity_key`: stereochemistry-independent connectivity identifier used for overlap checks.
- `murcko_scaffold`: canonical Bemis-Murcko scaffold used for grouped validation.
- `document_id` or `doc_id`: ChEMBL document provenance identifier.
- `assay_id`: species-specific ChEMBL assay identifier.
- `base_*` and `fluorinated_*`: nonfluorinated parent and fluorinated analogue values.
- `delta_*_f_minus_base`: fluorinated value minus parent value; negative values denote lower clearance after fluorination.
- `manual_review_status`: `verified` only after source-publication review; `pending` is not manually verified.
- `*_left_censored`: operational censoring indicator for repeated minima in the public paired data.
- `strict_total_intrinsic_pair`: both species meet the total intrinsic-clearance endpoint definition.
- `source_unit_per_ug_flag`: unresolved per-microgram source-unit metadata flag.
- `full_inchikey_overlap`: full stereochemical identity agreement between species records.

Blank values represent unavailable or inapplicable fields unless a file-specific script defines another convention. Consult the generating script before changing missing-value treatment.
