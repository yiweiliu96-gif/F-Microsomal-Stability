# Model card: RLM-anchored HLM residual model

## Intended use

Estimate continuous HLM log10 CLint after a quantitative RLM log10 CLint measurement is available for the same fluorinated or nonfluorinated drug-like molecule. The primary manuscript evaluation emphasizes fluorinated chemical space.

## Model formulation

The model predicts the residual `HLM - RLM` from a general molecular representation, measured RLM, and an RLM censoring indicator. The final HLM estimate equals measured RLM plus the fitted residual.

The general representation combines a 2,048-bit ECFP4 fingerprint (Morgan radius 2) with RDKit descriptors. The frozen bundle also contains matched structure-only, direct RLM-assisted, and fluorine-augmented ablation models.

## Bundle

- File: `lgbm_models.joblib`
- Format: Python joblib dictionary
- Primary key: `RLM_anchor_residual_LGBM__general`
- SHA-256: `9e06ffa724c5de4eb2d830bf20a982d0e249ec36e9b925b868338ca57811c14a`

## Validation boundary

- Internal fluorinated scaffold validation: n = 557, interval RMSE = 0.303, low-clearance ROC-AUC = 0.944.
- Strict external fluorinated validation: n = 598 from 174 ChEMBL documents, RMSE = 0.615, Spearman rho = 0.771, low-clearance ROC-AUC = 0.864.
- The external comparison with linear RLM translation primarily supports lower continuous error; ranking improvement beyond the linear baseline was not significant.

## Limitations

- The model requires an exact quantitative RLM measurement for the same compound.
- It does not predict in vivo pharmacokinetics, clinical exposure, CYP contribution, metabolites, or site of metabolism.
- Inputs outside the observed endpoint range or chemical domain require direct HLM confirmation.
- CF3-containing compounds showed weaker external continuous calibration than other examined fluorinated subgroups.
- A censored or boundary-reported RLM value should not be interpreted as an exact continuous anchor.

## Software compatibility

The model was created with Python 3.10.9, LightGBM 4.7.0, scikit-learn 1.5.2, RDKit 2022.09.5, NumPy 1.26.4, pandas 2.2.3, and joblib 1.5 or later. Loading untrusted joblib files is unsafe; use only the archived release checksum.
