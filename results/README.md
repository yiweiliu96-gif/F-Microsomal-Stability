# Result-file map

- `paired_identical_molecule_metrics.csv`: structure-only, direct RLM-assisted, and residual-model internal scaffold metrics on identical compound sets.
- `paired_scaffold_bootstrap_comparisons.csv`: paired internal uncertainty estimates.
- `strict_external_baseline_comparison.csv`: strict 598-compound and broader 610-compound external baseline metrics.
- `strict_external_baseline_bootstrap.csv`: document-clustered external uncertainty and p values.
- `scaffold_oof_metrics.csv`: algorithm benchmark under the same scaffold folds.
- `document_bootstrap_vs_lightgbm.csv`: external algorithm comparisons against prespecified LightGBM.
- `external_motif_metrics.csv`: motif-resolved external error and ranking metrics.
- `cross_validated_metrics.csv`: edit-level Delta RLM to Delta HLM translation controls.
- `submission_strengthening/`: post hoc anchor falsification, cross-conformal uncertainty, reversal-risk, and selective-prediction analyses.

`metrics.csv` and `ensemble_metrics.csv` are retained as robustness outputs from auxiliary model families. They are not the source of the manuscript's primary LightGBM residual-model values; use the files named above for manuscript-facing claims.
