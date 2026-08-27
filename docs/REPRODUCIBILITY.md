# Reproducibility guide

## Scope

This release supports inspection and reproduction of the manuscript-facing continuous HLM/RLM analyses. It includes standardized analysis inputs, derived tables, molecule-level predictions, fitted primary models, statistical summaries, and executable scripts.

## Primary analysis contract

- Primary endpoint: continuous HLM log10 CLint.
- Experimental anchor: quantitative RLM log10 CLint for the same compound.
- Primary model: LightGBM residual prediction, with HLM prediction equal to measured RLM plus the fitted HLM-minus-RLM residual.
- Internal validation: five-fold GroupKFold by canonical Bemis-Murcko scaffold.
- External validation: endpoint-audited fluorinated ChEMBL35 set with 598 compounds from 174 documents.
- Primary uncertainty: paired scaffold bootstrap internally and ChEMBL-document bootstrap externally.

## Recommended order

1. Run `python verify_release.py`.
2. Inspect `data/README.md` and `DATA_AND_REVIEW_NOTICE.md`.
3. Reconstruct paired development data.
4. Fit the frozen model specification.
5. Reproduce scaffold validation.
6. Rebuild the external set from the cited OpenADMET/ChEMBL35 inputs.
7. Reproduce external metrics and bootstrap comparisons.
8. Reproduce matched-pair and source-row audits.
9. Regenerate figures from machine-readable outputs.

## External inputs

Publisher PDFs are not redistributed. OpenADMET/ChEMBL35 source catalogs must be obtained from their public source when a script requires raw activity records. Source versions and identifiers must not be silently replaced with later releases.

## Expected variation

Small numerical differences may occur across operating systems or library builds. The manuscript-facing comparisons should remain within reported precision. Any change to molecular standardization, endpoint membership, scaffold grouping, censoring treatment, or document clustering constitutes a new analysis and should be reported as such.
