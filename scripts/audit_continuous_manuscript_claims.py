#!/usr/bin/env python3
"""Audit manuscript-facing continuous-endpoint claims against frozen outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "JCIM_CONTINUOUS_RLM_ANCHORED_MANUSCRIPT.md"
SI = ROOT / "docs" / "JCIM_CONTINUOUS_SUPPORTING_INFORMATION.md"
OUT = ROOT / "reports" / "continuous_manuscript_audit"


@dataclass
class Check:
    name: str
    expected: str
    location: str


def row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def claim_checks() -> list[Check]:
    pair_effects = pd.read_csv(
        ROOT / "reports/assay_matched_fluorination_pairs/same_assay_species_effect_summary.csv"
    )
    cross = pd.read_csv(
        ROOT / "reports/assay_matched_fluorination_pairs/cross_species_assay_matched_summary.csv"
    )
    delta_models = pd.read_csv(
        ROOT / "reports/cross_species_fluorination_delta_translation/cross_validated_metrics.csv"
    )
    internal = pd.read_csv(
        ROOT / "reports/biogen_paired_scaffold_cv/paired_identical_molecule_metrics.csv"
    )
    external = pd.read_csv(
        ROOT / "reports/openadmet_chembl35_paired_external/strict_external_baseline_comparison.csv"
    )
    algorithms = pd.read_csv(ROOT / "reports/rlm_anchor_algorithm_benchmark/ensemble_metrics.csv")
    gine = pd.read_csv(ROOT / "reports/rlm_anchor_gine_benchmark/metrics.csv")
    motif = pd.read_csv(
        ROOT / "reports/openadmet_chembl35_paired_external/external_motif_metrics.csv"
    )

    hlm_all = row(pair_effects, species="HLM", transformation="all")
    rlm_all = row(pair_effects, species="RLM", transformation="all")
    cross_all = row(cross, transformation="all")
    delta_doc = row(delta_models, split_strategy="document_id", model="delta_RLM_OLS")
    delta_zero = row(delta_models, split_strategy="document_id", model="no_change")
    int_structure = row(internal, scope="fluorinated", model="structure-only")
    int_anchor = row(internal, scope="fluorinated", model="RLM-anchored residual")
    ext_structure = row(
        external,
        analysis_set="strict_total_no_unit_or_stereo_flags",
        model="structure-only",
    )
    ext_linear = row(
        external,
        analysis_set="strict_total_no_unit_or_stereo_flags",
        model="RLM-only linear",
    )
    ext_anchor = row(
        external,
        analysis_set="strict_total_no_unit_or_stereo_flags",
        model="RLM-anchored residual",
    )
    catboost = row(algorithms, model="CatBoost")
    gine_external = row(gine, evaluation="strict_external", scope="fluorinated", seed="ensemble")
    cf3 = row(motif, motif="CF3", model="RLM_anchor_residual_LGBM__general")

    return [
        Check("HLM pair count", str(int(hlm_all.n_pairs)), "both"),
        Check("RLM pair count", str(int(rlm_all.n_pairs)), "both"),
        Check("cross-species pair count", str(int(cross_all.n_pairs)), "both"),
        Check("cross-species Spearman", fmt(cross_all.delta_hlm_delta_rlm_spearman), "both"),
        Check("direction agreement percent", f"{100 * cross_all.direction_agreement:.1f}%", "both"),
        Check("direction reversal percent", f"{100 * cross_all.species_opposite_direction:.1f}%", "both"),
        Check("delta no-change RMSE", fmt(delta_zero.rmse), "both"),
        Check("delta RLM OLS RMSE", fmt(delta_doc.rmse), "both"),
        Check("internal fluorinated n", str(int(int_structure.n)), "both"),
        Check("internal structure interval RMSE", fmt(int_structure.interval_rmse), "both"),
        Check("internal anchor interval RMSE", fmt(int_anchor.interval_rmse), "both"),
        Check("internal anchor AUC", fmt(int_anchor.low_clearance_auc), "both"),
        Check("external strict fluorinated n", str(int(ext_anchor.n)), "both"),
        Check("external structure RMSE", fmt(ext_structure.rmse), "both"),
        Check("external linear RMSE", fmt(ext_linear.rmse), "both"),
        Check("external anchor RMSE", fmt(ext_anchor.rmse), "both"),
        Check("external anchor Spearman", fmt(ext_anchor.spearman), "manuscript"),
        Check("external anchor AUC", fmt(ext_anchor.low_clearance_auc), "manuscript"),
        Check("external CatBoost RMSE", fmt(catboost.rmse), "both"),
        Check("external GINE RMSE", fmt(gine_external.uncensored_rmse), "both"),
        Check("CF3 subgroup n", str(int(cf3.n)), "manuscript"),
        Check("CF3 subgroup RMSE", fmt(cf3.rmse), "manuscript"),
    ]


def first_mentions(text: str, label: str) -> list[int]:
    return [match.start() for match in re.finditer(rf"\b{re.escape(label)}\b", text)]


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    si = SI.read_text(encoding="utf-8")
    results: list[dict[str, object]] = []
    for check in claim_checks():
        targets = {"manuscript": manuscript, "si": si}
        locations = targets if check.location == "both" else {check.location: targets[check.location]}
        for location, text in locations.items():
            passed = check.expected in text
            results.append(
                {
                    "category": "numeric_claim",
                    "check": check.name,
                    "location": location,
                    "expected_text": check.expected,
                    "passed": passed,
                    "detail": "present" if passed else "missing or differently rounded",
                }
            )

    forbidden = [
        r"\bV[0-9]+\b",
        r"source study",
        r"source record",
        r"reference data",
        r"development records",
        r"JCIM-level",
        r"\bMLM\b",
    ]
    for pattern in forbidden:
        matches = re.findall(pattern, manuscript, flags=re.IGNORECASE)
        results.append(
            {
                "category": "terminology",
                "check": pattern,
                "location": "manuscript",
                "expected_text": "no match",
                "passed": not matches,
                "detail": "; ".join(matches[:5]),
            }
        )

    abstract = manuscript.split("## Abstract", 1)[1].split("## Keywords", 1)[0]
    abstract_words = re.findall(r"\b[\w'-]+\b", abstract)
    results.append(
        {
            "category": "format",
            "check": "abstract word count <= 200",
            "location": "manuscript",
            "expected_text": "<=200",
            "passed": len(abstract_words) <= 200,
            "detail": str(len(abstract_words)),
        }
    )

    body = manuscript.split("## Figure Captions", 1)[0]
    figure_firsts = []
    for number in range(1, 6):
        mentions = first_mentions(body, f"Figure {number}")
        figure_firsts.append(mentions[0] if mentions else -1)
    ordered = all(value >= 0 for value in figure_firsts) and figure_firsts == sorted(figure_firsts)
    results.append(
        {
            "category": "cross_reference",
            "check": "Figures 1-5 first cited in order",
            "location": "manuscript",
            "expected_text": "1,2,3,4,5",
            "passed": ordered,
            "detail": ",".join(map(str, figure_firsts)),
        }
    )

    frame = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "claim_audit.csv", index=False)
    failures = frame.loc[~frame.passed]
    summary = [
        "# Continuous Manuscript Claim Audit",
        "",
        f"Checks: {len(frame)}",
        f"Passed: {int(frame.passed.sum())}",
        f"Failed: {len(failures)}",
        "",
    ]
    if failures.empty:
        summary.append("All automated manuscript-facing checks passed.")
    else:
        summary.append("## Failures")
        summary.extend(
            f"- {item.location}: {item.check} (expected `{item.expected_text}`; {item.detail})"
            for item in failures.itertuples()
        )
    (OUT / "claim_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
