#!/usr/bin/env python3
"""Generate five manuscript figures for the continuous RLM-anchored story."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fms_continuous_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "continuous_submission" / "figures"

BLUE = "#3B6FB6"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE5"
DARK = "#202124"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def annotate_bars(ax: plt.Axes, bars, digits: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )


def row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_axis_off()

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="white", edgecolor=color, linewidth=1.35,
        )
        ax.add_patch(patch)
        ax.text(x + 0.014, y + h - 0.035, title, fontweight="bold", fontsize=8.2, color=color, va="top", linespacing=1.08)
        ax.text(x + 0.014, y + h - 0.115, body, fontsize=7.2, color=DARK, va="top", linespacing=1.30)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1, color=GRAY))

    ax.text(0.02, 0.96, "Chemical question: does an exact fluorination edit transfer between species?", fontsize=11, fontweight="bold", va="top")
    box(0.02, 0.59, 0.205, 0.23, "Same-assay\nedits", "HLM: 476 pairs\nRLM: 209 pairs\nOutcome-blind selection", BLUE)
    box(0.275, 0.59, 0.205, 0.23, "Cross-species\npairs", "97 identical edits\n42 documents\nRaw-row audit", PURPLE)
    box(0.53, 0.59, 0.205, 0.23, "Transfer\nboundary", "Spearman 0.431\n38.1% reversals\nMotifs exploratory", ORANGE)
    box(0.785, 0.59, 0.195, 0.23, "Decision", "RLM edit SAR does\nnot replace direct\nHLM measurement", DARK)
    arrow(0.225, 0.705, 0.275, 0.705); arrow(0.48, 0.705, 0.53, 0.705); arrow(0.735, 0.705, 0.785, 0.705)

    ax.text(0.02, 0.49, "Prediction question: after RLM is measured, can HLM estimation be improved?", fontsize=11, fontweight="bold", va="top")
    box(0.02, 0.11, 0.205, 0.23, "Paired data", "3,049 HLM/RLM pairs\n557 fluorinated\nFive scaffold folds", BLUE)
    box(0.275, 0.11, 0.205, 0.23, "Competing\ninputs", "Structure only\nRLM-only linear\nRLM-anchored residual", TEAL)
    box(0.53, 0.11, 0.205, 0.23, "Frozen external\ntest", "598 fluorinated\n174 documents\nTotal CLint semantics", PURPLE)
    box(0.785, 0.11, 0.195, 0.23, "Use case", "Refine absolute HLM\nCLint after quantitative\nRLM measurement", DARK)
    arrow(0.225, 0.225, 0.275, 0.225); arrow(0.48, 0.225, 0.53, 0.225); arrow(0.735, 0.225, 0.785, 0.225)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "figure_1_study_design")


def effect_forest(ax: plt.Axes, data: pd.DataFrame, species: str, color: str) -> None:
    order = ["all", "Ar-H_to_Ar-F", "C-H_to_C-F", "CH2_to_CF2", "CH3_to_CF3", "OCH3_to_OCF3"]
    labels = ["All", "Ar-H→Ar-F", "C-H→C-F", r"CH$_2$→CF$_2$", r"CH$_3$→CF$_3$", r"OCH$_3$→OCF$_3$"]
    subset = data.loc[data.species.eq(species)].set_index("transformation").reindex(order)
    y = np.arange(len(order))[::-1]
    values = subset.median_delta.to_numpy()
    lower = values - subset.median_delta_ci95_low.to_numpy()
    upper = subset.median_delta_ci95_high.to_numpy() - values
    ax.errorbar(values, y, xerr=[lower, upper], fmt="o", color=color, ecolor=color, capsize=2.5, lw=1.2, ms=4.5)
    ax.axvline(0, color=DARK, lw=0.8, ls="--")
    ax.set_yticks(y, [f"{label}  (n={int(n)})" for label, n in zip(labels, subset.n_pairs)])
    ax.set_xlabel(r"Median Δlog$_{10}$ CLint (fluorinated − parent)", fontsize=8.2)
    ax.set_title(f"{species} same-assay effects", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6)


def figure_2() -> None:
    effects = pd.read_csv(ROOT / "reports/assay_matched_fluorination_pairs/same_assay_species_effect_summary.csv")
    pairs = pd.read_csv(ROOT / "reports/assay_matched_fluorination_pairs/cross_species_assay_matched_pairs.csv")
    sensitivity = pd.read_csv(ROOT / "reports/assay_matched_fluorination_pairs/fluorination_effect_sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), gridspec_kw={"hspace": 0.46, "wspace": 0.72})
    effect_forest(axes[0, 0], effects, "HLM", BLUE); panel(axes[0, 0], "A")
    effect_forest(axes[0, 1], effects, "RLM", ORANGE); panel(axes[0, 1], "B")

    ax = axes[1, 0]; panel(ax, "C")
    x = pairs.assay_matched_delta_rlm.to_numpy(); y = pairs.assay_matched_delta_hlm.to_numpy()
    agree = pairs.assay_matched_direction_agreement.eq(1).to_numpy()
    ax.scatter(x[agree], y[agree], s=22, color=TEAL, alpha=0.75, label="same direction", edgecolor="white", linewidth=0.3)
    ax.scatter(x[~agree], y[~agree], s=25, color=ORANGE, alpha=0.82, label="opposite direction", edgecolor="white", linewidth=0.3)
    lim = max(1.05, np.nanmax(np.abs(np.r_[x, y])))
    ax.axhline(0, color=GRAY, lw=0.7); ax.axvline(0, color=GRAY, lw=0.7)
    ax.plot([-lim, lim], [-lim, lim], color=DARK, lw=0.8, ls="--")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"ΔRLM log$_{10}$ CLint", ylabel=r"ΔHLM log$_{10}$ CLint")
    ax.set_title("Identical edits in both species", loc="left", fontweight="bold")
    ax.text(0.04, 0.96, "ρ = 0.431\n38.1% reversals", transform=ax.transAxes, va="top", fontsize=8.5)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]; panel(ax, "D")
    scenarios = ["primary", "exclude_any_exact_zero_delta", "exclude_any_absolute_delta_gt_1", "aryl_H_to_F_only"]
    labels = ["Primary", "No exact-zero Δ", "|Δ| ≤ 1", "Ar-H→Ar-F only"]
    subset = sensitivity.loc[
        sensitivity.analysis.eq("cross_species")
        & sensitivity.cluster_unit.eq("document_id")
        & sensitivity.scenario.isin(scenarios)
    ].set_index("scenario").reindex(scenarios)
    yy = np.arange(len(labels))[::-1]
    vals = subset.direction_agreement.to_numpy()
    ax.errorbar(vals, yy, xerr=[vals-subset.direction_agreement_ci95_low, subset.direction_agreement_ci95_high-vals], fmt="o", color=PURPLE, capsize=2.5, lw=1.2)
    ax.axvline(0.5, color=DARK, ls="--", lw=0.8)
    ax.set_yticks(yy, [f"{label}  (n={int(n)})" for label, n in zip(labels, subset.n_pairs)])
    ax.set_xlim(0.42, 0.80); ax.set_xlabel("Direction agreement")
    ax.set_title("Sensitivity of cross-species agreement", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6)
    save(fig, "figure_2_fluorination_effects")


def figure_3() -> None:
    primary = pd.read_csv(ROOT / "reports/biogen_paired_scaffold_cv/paired_identical_molecule_metrics.csv")
    interaction = pd.read_csv(ROOT / "reports/biogen_fluorine_insights/measured_rlm_fluorine_interaction.csv")
    algorithms = pd.read_csv(ROOT / "reports/rlm_anchor_algorithm_scaffold_cv/scaffold_oof_metrics.csv")
    gine = pd.read_csv(ROOT / "reports/rlm_anchor_gine_benchmark/metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), gridspec_kw={"hspace": 0.48, "wspace": 0.36})
    fluoro = primary.loc[primary.scope.eq("fluorinated")]
    labels = ["Structure only", "Direct RLM", "RLM residual"]
    colors = [GRAY, TEAL, BLUE]
    ax = axes[0, 0]; panel(ax, "A")
    bars = ax.bar(labels, fluoro.interval_rmse, color=colors, width=0.68); annotate_bars(ax, bars)
    ax.set_ylabel("Interval RMSE"); ax.set_ylim(0, 0.53); ax.set_title("Fluorinated scaffold holdout", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax = axes[0, 1]; panel(ax, "B")
    bars = ax.bar(labels, fluoro.low_clearance_auc, color=colors, width=0.68); annotate_bars(ax, bars)
    ax.set_ylabel("Low-clearance ROC-AUC"); ax.set_ylim(0.75, 0.98); ax.set_title("Secondary ranking endpoint", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]; panel(ax, "C")
    order = ["interval_rmse", "uncensored_rmse", "low_clearance_roc_auc"]
    names = ["Interval RMSE", "Exact-value RMSE", "ROC-AUC"]
    subset = interaction.set_index("metric").reindex(order)
    y = np.arange(3)[::-1]; vals = subset.difference_in_differences.to_numpy()
    ax.errorbar(vals, y, xerr=[vals-subset.ci95_low, subset.ci95_high-vals], fmt="o", color=PURPLE, capsize=2.5, lw=1.2)
    ax.axvline(0, color=DARK, ls="--", lw=0.8); ax.set_yticks(y, names)
    ax.set_xlabel("Fluorinated minus nonfluorinated gain")
    ax.set_title("Subset interaction", loc="left", fontweight="bold"); ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6)

    ax = axes[1, 1]; panel(ax, "D")
    alg = algorithms.loc[algorithms.seed.eq("ensemble") & algorithms.scope.eq("fluorinated")]
    wanted = ["ExtraTrees", "LightGBM", "XGBoost", "RandomForest", "CatBoost", "descriptor_MLP"]
    values = [row(alg, model=name).interval_rmse for name in wanted]
    values.append(row(gine, evaluation="scaffold_cv", scope="fluorinated", seed="ensemble").interval_rmse)
    names = ["ExtraTrees", "LightGBM", "XGBoost", "Random forest", "CatBoost", "Descriptor MLP", "GINE"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=[BLUE]*5 + [ORANGE, PURPLE], height=0.65)
    ax.set_yticks(y, names); ax.set_xlim(0.27, 0.40); ax.set_xlabel("Interval RMSE")
    ax.set_title("Same-task algorithm benchmark", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6)
    save(fig, "figure_3_internal_anchor_benchmark")


def figure_4() -> None:
    pred = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/external_predictions.csv")
    audit = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/external_endpoint_semantics_audit.csv")
    strict = pred.merge(audit[["external_record_id", "strict_total_intrinsic_pair", "source_unit_per_ug_flag", "full_inchikey_overlap"]], on="external_record_id")
    strict = strict.loc[strict.is_fluorinated.eq(1) & strict.strict_total_intrinsic_pair.eq(1) & strict.source_unit_per_ug_flag.eq(0) & strict.full_inchikey_overlap.eq(1)]
    metrics = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/strict_external_baseline_comparison.csv")
    boot = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/strict_external_baseline_bootstrap.csv")
    algorithms = pd.read_csv(ROOT / "reports/rlm_anchor_algorithm_benchmark/ensemble_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={"hspace": 0.46, "wspace": 0.37})
    for ax, column, title, color, label in [
        (axes[0, 0], "structure_LGBM__general", "Structure-only transfer", GRAY, "A"),
        (axes[0, 1], "RLM_anchor_residual_LGBM__general", "RLM-anchored residual", BLUE, "B"),
    ]:
        panel(ax, label)
        ax.scatter(strict.external_hlm, strict[column], s=12, color=color, alpha=0.45, edgecolor="none")
        lo = min(strict.external_hlm.min(), strict[column].min()); hi = max(strict.external_hlm.max(), strict[column].max())
        ax.plot([lo, hi], [lo, hi], color=DARK, ls="--", lw=0.8)
        ax.set(xlabel=r"Observed HLM log$_{10}$ CLint", ylabel=r"Predicted HLM log$_{10}$ CLint")
        ax.set_title(title, loc="left", fontweight="bold")

    ax = axes[1, 0]; panel(ax, "C")
    subset = metrics.loc[metrics.analysis_set.eq("strict_total_no_unit_or_stereo_flags")]
    models = ["structure-only", "RLM constant offset", "RLM-only linear", "RLM-anchored residual"]
    labels = ["Structure only", "Constant offset", "RLM linear", "RLM residual"]
    vals = [row(subset, model=name).rmse for name in models]
    bars = ax.bar(labels, vals, color=[GRAY, "#A0A7B4", TEAL, BLUE], width=0.7); annotate_bars(ax, bars)
    ax.set_ylabel("External RMSE"); ax.set_ylim(0, 1.12); ax.tick_params(axis="x", rotation=22)
    ax.set_title("Translation controls", loc="left", fontweight="bold")

    ax = axes[1, 1]; panel(ax, "D")
    selected = boot.loc[
        boot.analysis_set.eq("strict_total_no_unit_or_stereo_flags")
        & boot.metric.eq("rmse")
        & boot.baseline_label.isin(["structure-only", "RLM-only linear"])
    ]
    selected = selected.set_index("baseline_label").reindex(["structure-only", "RLM-only linear"])
    vals = selected.delta_candidate_minus_baseline.to_numpy(); y = np.array([1, 0])
    ax.errorbar(vals, y, xerr=[vals-selected.ci95_low, selected.ci95_high-vals], fmt="o", color=ORANGE, capsize=3, lw=1.3)
    ax.axvline(0, color=DARK, ls="--", lw=0.8)
    ax.set_yticks(y, ["vs structure only", "vs RLM linear"]); ax.set_xlabel("Residual-model ΔRMSE")
    ax.set_title("Document-bootstrap improvement", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT_GRAY, lw=0.6)
    save(fig, "figure_4_external_validation")


def figure_5() -> None:
    motif = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/external_motif_metrics.csv")
    primary = pd.read_csv(ROOT / "reports/openadmet_chembl35_paired_external/strict_external_baseline_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw={"hspace": 0.5, "wspace": 0.4})
    motifs = ["CF3", "CF2_without_CF3", "aryl_F_without_CF3_CF2"]
    labels = [r"CF$_3$", r"CF$_2$ without CF$_3$", "Aryl-F only"]
    general = motif.loc[motif.model.eq("RLM_anchor_residual_LGBM__general")].set_index("motif").reindex(motifs)
    augmented = motif.loc[motif.model.eq("RLM_anchor_residual_LGBM__fluorine_augmented")].set_index("motif").reindex(motifs)
    x = np.arange(3); width = 0.36
    ax = axes[0, 0]; panel(ax, "A")
    ax.bar(x-width/2, general.rmse, width, color=BLUE, label="General representation")
    ax.bar(x+width/2, augmented.rmse, width, color=ORANGE, label="+ F descriptors")
    ax.set_xticks(x, [f"{name}\n(n={int(n)})" for name, n in zip(labels, general.n)])
    ax.set_ylabel("External RMSE"); ax.set_title("Motif-dependent calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax = axes[0, 1]; panel(ax, "B")
    ax.bar(x-width/2, general.low_clearance_auc, width, color=BLUE)
    ax.bar(x+width/2, augmented.low_clearance_auc, width, color=ORANGE)
    ax.set_xticks(x, labels); ax.set_ylim(0.82, 0.98); ax.set_ylabel("Low-clearance ROC-AUC")
    ax.set_title("Ranking remains useful", loc="left", fontweight="bold")

    ax = axes[1, 0]; panel(ax, "C")
    full_general = row(motif, motif="all_fluorinated", model="RLM_anchor_residual_LGBM__general")
    full_aug = row(motif, motif="all_fluorinated", model="RLM_anchor_residual_LGBM__fluorine_augmented")
    strict = primary.loc[primary.analysis_set.eq("strict_total_no_unit_or_stereo_flags")]
    strict_general = row(strict, model="RLM-anchored residual")
    strict_aug = row(strict, model="RLM-anchored residual + F descriptors")
    labels2 = ["Full traceable\n(n=879)", "Strict total CLint\n(n=598)"]
    g = [full_general.rmse, strict_general.rmse]; a = [full_aug.rmse, strict_aug.rmse]
    xx = np.arange(2)
    ax.bar(xx-width/2, g, width, color=BLUE, label="General")
    ax.bar(xx+width/2, a, width, color=ORANGE, label="+ F descriptors")
    ax.set_xticks(xx, labels2); ax.set_ylim(0.54, 0.66); ax.set_ylabel("External RMSE")
    ax.set_title("Fluorine-descriptor ablation", loc="left", fontweight="bold")

    ax = axes[1, 1]; panel(ax, "D"); ax.set_axis_off()
    stages = [
        (0.02, "1", "Structure-only\nprioritization", GRAY),
        (0.35, "2", "Measure\nquantitative\nRLM CLint", TEAL),
        (0.68, "3", "Anchor HLM estimate;\nconfirm high-risk\ncases", BLUE),
    ]
    for x0, number, text, color in stages:
        ax.add_patch(plt.Circle((x0+0.07, 0.62), 0.07, facecolor=color, edgecolor="none"))
        ax.text(x0+0.07, 0.62, number, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x0+0.07, 0.37, text, ha="center", va="top", fontsize=7.4, linespacing=1.18)
    ax.add_patch(FancyArrowPatch((0.18, 0.62), (0.35, 0.62), arrowstyle="-|>", mutation_scale=10, color=GRAY))
    ax.add_patch(FancyArrowPatch((0.51, 0.62), (0.68, 0.62), arrowstyle="-|>", mutation_scale=10, color=GRAY))
    ax.text(0.5, 0.06, r"CF$_3$-rich or out-of-domain compounds warrant", ha="center", fontsize=7.4, color=ORANGE)
    ax.text(0.5, -0.02, "direct HLM confirmation.", ha="center", fontsize=7.4, color=ORANGE)
    ax.set_title("Staged experimental use", loc="left", fontweight="bold")
    save(fig, "figure_5_fluorine_transfer_boundaries")


def main() -> None:
    style()
    figure_1(); figure_2(); figure_3(); figure_4(); figure_5()
    print(OUT)


if __name__ == "__main__":
    main()
