# Data provenance, use, and manual-review notice

## Reused public data

The model-development split files derive from the public paired ADME data reported by Fang et al.:

> Fang, C. et al. J. Chem. Inf. Model. 2023, 63, 3263-3274. DOI: 10.1021/acs.jcim.3c00160.

The assay-matched fluorination pairs and external HLM/RLM evaluation records derive from ChEMBL35 microsomal-clearance records curated through the OpenADMET data catalog. Users must cite ChEMBL, OpenADMET, and the original experimental publication identified by each ChEMBL document or DOI field, as applicable.

The original publications and database records remain authoritative. Repository tables support audit and reproduction; they do not replace the original experimental reports.

## Data categories

- `data/development/` contains source-derived public split tables used to reconstruct model-development compounds.
- `data/derived/` contains standardized structures, calculated molecular annotations, pair definitions, endpoint-semantics classifications, source identifiers, model predictions, and statistical-analysis inputs created for this study.
- `results/` contains computed summaries and uncertainty estimates.
- `models/` contains fitted model objects generated in this study.

No new in vitro or in vivo measurements were generated.

## Manual review

All 97 retained cross-species fluorination pairs have `manual_review_status = verified`. The status records author confirmation against the original article or Supporting Information for molecular identity, fluorination edit, endpoint, units, and reported HLM/RLM values.

Automated source-row reconstruction also checked assay, document, endpoint, unit, species semantics, and aggregation consistency. Single-species pair tables may contain `pending` entries and must not be described as manually verified unless their row-level status is updated after source review.

## Selection safeguards

No external endpoint was used to tune hyperparameters, select the primary model, remove high-error compounds, or choose fluorination pairs according to effect direction. Source-level exclusions were based only on documented endpoint, unit, molecular-identity, assay, overlap, replicate-consistency, or censoring criteria.

## Licensing

The repository `LICENSE` applies to software code. It does not override rights or conditions attached to Fang et al., ChEMBL, OpenADMET, or original experimental publications. Users are responsible for complying with those source terms.
