# =============================================================================
# CELL 31: STATISTICS, INTEGRITY CHECKS, FIGURES, AND FINAL EXPORTS
# =============================================================================

from pathlib import Path
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from scipy.stats import wilcoxon


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

if "DATA_DIR" not in globals():
    DATA_DIR = Path.cwd()
else:
    DATA_DIR = Path(DATA_DIR)

OUTPUT_DIR = DATA_DIR / "outputs"
AMJEPA_DIR = OUTPUT_DIR / "am_jepa_experiment"

OBJECTIVE_DIR = AMJEPA_DIR / "objective_ablation"
ROBUSTNESS_DIR = AMJEPA_DIR / "missing_modality_robustness"
SELECTOR_DIR = AMJEPA_DIR / "silver_sparse_selector"
DUAL_DIR = AMJEPA_DIR / "silver_dual_selector"
FINAL_DIR = AMJEPA_DIR / "silver_final_stance"
CELL30_DIR = AMJEPA_DIR / "final_ablation_robustness"

CELL31_DIR = AMJEPA_DIR / "final_statistics_exports"
FIGURE_DIR = CELL31_DIR / "figures"
TABLE_DIR = CELL31_DIR / "tables"

CELL31_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
FIGURE_DPI = 300
EXPECTED_SEEDS = [42, 52, 62, 72, 82]
EXPECTED_TEST_REPORTS = 89
EXPECTED_ABSTENTIONS = 12


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required output is missing:\n{path}"
        )
    return path


def load_csv(path, required=None):
    path = require_file(path)
    table = pd.read_csv(path)

    if required:
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {missing}"
            )

    return table


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_effect_size(differences):
    differences = np.asarray(differences, dtype=np.float64)
    standard_deviation = differences.std(ddof=1)

    if len(differences) < 2 or standard_deviation == 0:
        return np.nan

    return float(differences.mean() / standard_deviation)


def exact_wilcoxon(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]

    if len(differences) == 0 or np.allclose(differences, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running_maximum = 0.0
    number = len(values)

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (number - rank) * values[index],
        )
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum

    return adjusted


def save_figure(fig, stem):
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_latex_table(table, path, caption, label, columns=None):
    export = table.copy()
    if columns is not None:
        export = export[columns]

    latex = export.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.4f}",
        caption=caption,
        label=label,
        position="t",
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(latex)


# -----------------------------------------------------------------------------
# Load frozen results
# -----------------------------------------------------------------------------

objective_raw = load_csv(
    OBJECTIVE_DIR / "am_jepa_objective_ablation.csv",
    ["variant", "campaign_f1", "core3_macro_f1"],
)

multiseed_summary = load_csv(
    OBJECTIVE_DIR / "am_jepa_full_multiseed_summary.csv",
    ["seed", "campaign_f1", "core3_macro_f1"],
)

missing_detailed = load_csv(
    ROBUSTNESS_DIR / "missing_modality_detailed_results.csv",
    [
        "variant",
        "seed",
        "scenario",
        "campaign_f1",
        "core3_macro_f1",
    ],
)

selector_primary = load_csv(
    SELECTOR_DIR / "selector_primary_summary.csv"
)

dual_seed_deltas = load_csv(
    DUAL_DIR / "dual_single_seed_deltas.csv",
    ["feature_set", "seed", "budget", "map_delta"],
)

dual_primary = load_csv(
    DUAL_DIR / "dual_selector_primary_summary.csv"
)

stance_seed = load_csv(
    FINAL_DIR / "stance_test_seed_results.csv",
    ["method", "seed", "evaluation_set", "macro_f1", "mcc"],
)

stance_aggregate = load_csv(
    FINAL_DIR / "stance_test_aggregate.csv"
)

selective_aggregate = load_csv(
    FINAL_DIR / "stance_selective_aggregate.csv",
    [
        "method",
        "target_validation_coverage",
        "test_coverage_mean",
        "macro_f1_mean",
        "selective_risk_mean",
    ],
)

test_reports = load_csv(
    CELL30_DIR / "deterministic_evidence_linked_test_reports.csv",
    [
        "report_id",
        "annotation_id",
        "decision",
        "reason_code",
        "abstained",
        "identifier_integrity_passed",
    ],
)

report_integrity = load_csv(
    CELL30_DIR / "deterministic_report_integrity_summary.csv"
)

cell30_manifest_path = require_file(
    CELL30_DIR / "cell30_manifest.json"
)

with open(cell30_manifest_path, encoding="utf-8") as file:
    cell30_manifest = json.load(file)


# -----------------------------------------------------------------------------
# Corrected seed-42 objective-ablation table
# -----------------------------------------------------------------------------

if "seed" in objective_raw.columns:
    objective_seed42 = objective_raw[
        objective_raw["seed"] == 42
    ].copy()
else:
    # The objective variants were written first, followed by four extra
    # full-AM-JEPA seeds. Preserve the first occurrence of each variant.
    objective_seed42 = objective_raw.drop_duplicates(
        "variant",
        keep="first",
    ).copy()

if objective_seed42["variant"].nunique() != 6:
    raise AssertionError(
        "The corrected objective ablation must contain six seed-42 variants."
    )

objective_seed42 = (
    objective_seed42
    .sort_values("campaign_f1", ascending=False)
    .reset_index(drop=True)
)

objective_seed42.to_csv(
    TABLE_DIR / "objective_ablation_seed42_corrected.csv",
    index=False,
)

multiseed_summary.to_csv(
    TABLE_DIR / "full_am_jepa_five_seed_results.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Statistical test 1: paired stance-method comparisons across five seeds
# -----------------------------------------------------------------------------

stance_all = stance_seed[
    stance_seed["evaluation_set"] == "all_test"
].copy()

stance_methods = [
    "dense_am_jepa",
    "dual_text_only",
    "dual_full_multimodal",
]

stance_comparisons = [
    ("dual_text_only", "dense_am_jepa"),
    ("dual_full_multimodal", "dense_am_jepa"),
    ("dual_text_only", "dual_full_multimodal"),
]

stance_test_rows = []

for metric in [
    "macro_f1",
    "mcc",
    "macro_ovr_auroc",
]:
    if metric not in stance_all.columns:
        continue

    pivot = stance_all.pivot(
        index="seed",
        columns="method",
        values=metric,
    )

    for method_a, method_b in stance_comparisons:
        if not {method_a, method_b}.issubset(pivot.columns):
            continue

        paired = pivot[[method_a, method_b]].dropna()
        differences = (
            paired[method_a] - paired[method_b]
        ).to_numpy(dtype=np.float64)
        statistic, p_value = exact_wilcoxon(differences)

        stance_test_rows.append(
            {
                "family": "stance_method",
                "metric": metric,
                "comparison": f"{method_a} - {method_b}",
                "pairs": len(differences),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "cohen_dz": paired_effect_size(differences),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

stance_tests = pd.DataFrame(stance_test_rows)


# -----------------------------------------------------------------------------
# Statistical test 2: missing-modality degradation versus complete input
# -----------------------------------------------------------------------------

full_missing = missing_detailed[
    missing_detailed["variant"] == "full_am_jepa"
].copy()

missing_test_rows = []

for metric in ["campaign_f1", "core3_macro_f1"]:
    pivot = full_missing.pivot(
        index="seed",
        columns="scenario",
        values=metric,
    )

    if "complete" not in pivot.columns:
        raise AssertionError("Complete-input robustness reference is missing.")

    for scenario in pivot.columns:
        if scenario == "complete":
            continue

        paired = pivot[["complete", scenario]].dropna()
        differences = (
            paired[scenario] - paired["complete"]
        ).to_numpy(dtype=np.float64)
        statistic, p_value = exact_wilcoxon(differences)

        missing_test_rows.append(
            {
                "family": "missing_modality",
                "metric": metric,
                "comparison": f"{scenario} - complete",
                "pairs": len(differences),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "cohen_dz": paired_effect_size(differences),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

missing_tests = pd.DataFrame(missing_test_rows)


# -----------------------------------------------------------------------------
# Statistical test 3: matched dual-versus-single selector deltas
# -----------------------------------------------------------------------------

selector_test_rows = []

for (feature_set, budget), group in dual_seed_deltas.groupby(
    ["feature_set", "budget"],
    sort=False,
):
    for metric in [
        "map_delta",
        "mrr_delta",
        "f1_at_k_delta",
        "ndcg_at_k_delta",
    ]:
        if metric not in group.columns:
            continue

        differences = group[metric].to_numpy(dtype=np.float64)
        statistic, p_value = exact_wilcoxon(differences)

        selector_test_rows.append(
            {
                "family": "dual_vs_single_selector",
                "metric": metric,
                "comparison": f"{feature_set}, K={int(budget)}",
                "pairs": len(differences),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "cohen_dz": paired_effect_size(differences),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

selector_tests = pd.DataFrame(selector_test_rows)


# -----------------------------------------------------------------------------
# Holm correction within each hypothesis family
# -----------------------------------------------------------------------------

all_tests = pd.concat(
    [stance_tests, missing_tests, selector_tests],
    ignore_index=True,
)

all_tests["p_holm"] = np.nan

for family, indices in all_tests.groupby("family").groups.items():
    index_array = np.asarray(list(indices), dtype=np.int64)
    adjusted = holm_adjust(
        all_tests.loc[index_array, "p_value"].to_numpy()
    )
    all_tests.loc[index_array, "p_holm"] = adjusted

all_tests["significant_holm_0_05"] = (
    all_tests["p_holm"] < ALPHA
)

all_tests.to_csv(
    TABLE_DIR / "paired_statistical_tests_holm.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Integrity audit
# -----------------------------------------------------------------------------

forbidden_report_columns = {
    "route_to_deepseek",
    "deepseek_request",
    "verifier_response",
    "corrected_label",
}

integrity_checks = {
    "cell30_manifest_completed": (
        cell30_manifest.get("status") == "completed"
    ),
    "expected_test_report_count": (
        len(test_reports) == EXPECTED_TEST_REPORTS
    ),
    "expected_abstention_count": (
        int(test_reports["abstained"].astype(bool).sum())
        == EXPECTED_ABSTENTIONS
    ),
    "report_ids_unique": test_reports["report_id"].is_unique,
    "annotation_ids_unique": test_reports["annotation_id"].is_unique,
    "all_identifier_checks_passed": bool(
        test_reports["identifier_integrity_passed"].astype(bool).all()
    ),
    "no_forbidden_verifier_columns": not bool(
        forbidden_report_columns.intersection(test_reports.columns)
    ),
    "external_verifier_false": (
        "external_verifier_used" in report_integrity.columns
        and not bool(
            report_integrity["external_verifier_used"].astype(bool).any()
        )
    ),
    "five_full_am_jepa_seeds": (
        sorted(multiseed_summary["seed"].astype(int).tolist())
        == EXPECTED_SEEDS
    ),
    "six_seed42_objective_variants": (
        objective_seed42["variant"].nunique() == 6
    ),
    "finite_statistical_outputs": bool(
        np.isfinite(all_tests["p_value"]).all()
        and np.isfinite(all_tests["p_holm"]).all()
    ),
}

integrity_table = pd.DataFrame(
    [
        {
            "check": key,
            "passed": bool(value),
        }
        for key, value in integrity_checks.items()
    ]
)

if not integrity_table["passed"].all():
    failed = integrity_table.loc[
        ~integrity_table["passed"],
        "check",
    ].tolist()
    raise AssertionError(
        "Final integrity audit failed: " + ", ".join(failed)
    )

integrity_table.to_csv(
    TABLE_DIR / "final_integrity_checks.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Publication visual system
# -----------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "navy": "#173F5F",
    "blue": "#20639B",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D1495B",
    "gray": "#6C757D",
    "light": "#E9ECEF",
}


def polish_axis(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color="#D9DEE3",
        linewidth=0.6,
        alpha=0.7,
        zorder=0,
    )
    ax.tick_params(length=3, width=0.7)


figure_catalog_rows = []


# -----------------------------------------------------------------------------
# Figure 1: objective trade-space with collapse diagnostic
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.8, 5.6))

latent_values = objective_seed42.get(
    "mean_latent_std",
    pd.Series(np.ones(len(objective_seed42))),
).to_numpy(dtype=np.float64)

latent_min = float(latent_values.min())
latent_range = max(float(latent_values.max() - latent_min), 1e-12)
sizes = 90 + 260 * (latent_values - latent_min) / latent_range

for row_number, row in objective_seed42.reset_index(drop=True).iterrows():
    variant = str(row["variant"])
    is_full = variant == "full_am_jepa"
    is_random = variant == "random_encoder"
    marker = "*" if is_full else ("X" if is_random else "o")
    color = (
        COLORS["teal"]
        if is_full
        else COLORS["gray"]
        if is_random
        else COLORS["blue"]
    )

    point = ax.scatter(
        row["campaign_f1"],
        row["core3_macro_f1"],
        s=sizes[row_number],
        c=color,
        marker=marker,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.95,
        zorder=4,
    )
    point.set_path_effects(
        [path_effects.withStroke(linewidth=2.2, foreground="white")]
    )

    label = variant.replace("_", " ")
    offsets = {
        "full_am_jepa": (7, -13),
        "random_encoder": (7, -2),
        "masked_evidence_only": (7, 5),
        "evidence_plus_modality": (7, -12),
        "evidence_plus_future": (7, 6),
        "full_without_variance": (-105, -14),
    }
    offset = offsets.get(variant, (6, 5))
    ax.annotate(
        label,
        (row["campaign_f1"], row["core3_macro_f1"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.2,
        color="#263238",
        arrowprops=(
            dict(arrowstyle="-", color="#9AA0A6", lw=0.6)
            if abs(offset[0]) > 20
            else None
        ),
    )

random_row = objective_seed42[
    objective_seed42["variant"] == "random_encoder"
].iloc[0]

ax.axvline(
    random_row["campaign_f1"],
    color=COLORS["gray"],
    linestyle="--",
    linewidth=0.9,
    alpha=0.65,
)
ax.axhline(
    random_row["core3_macro_f1"],
    color=COLORS["gray"],
    linestyle="--",
    linewidth=0.9,
    alpha=0.65,
)

ax.text(
    0.012,
    0.975,
    "Upper-right is better\nBubble area $\\propto$ latent dispersion",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=8.3,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#CFD8DC",
        alpha=0.95,
    ),
)

ax.set_xlabel("Campaign F1")
ax.set_ylabel("Core-three aspect macro-F1")
ax.set_title("Objective Trade-space and Representation Dispersion")
ax.set_xlim(
    objective_seed42["campaign_f1"].min() - 0.006,
    objective_seed42["campaign_f1"].max() + 0.007,
)
ax.set_ylim(
    objective_seed42["core3_macro_f1"].min() - 0.012,
    objective_seed42["core3_macro_f1"].max() + 0.014,
)
polish_axis(ax)
fig.tight_layout()
save_figure(fig, "figure_1_objective_trade_space")
figure_catalog_rows.append(
    {
        "figure": "Figure 1",
        "file_stem": "figure_1_objective_trade_space",
        "design": "Annotated bubble trade-space",
        "message": "Objective balance and latent-dispersion behavior",
    }
)


# -----------------------------------------------------------------------------
# Figure 2: missing-modality degradation fingerprint
# -----------------------------------------------------------------------------

robustness_summary = load_csv(
    CELL30_DIR / "missing_modality_robustness_publication_table.csv"
)

degradation = robustness_summary[
    robustness_summary["scenario"] != "complete"
].copy()

degradation_labels = (
    degradation["scenario"]
    .str.replace("_", " ", regex=False)
    .str.replace("pct", "%", regex=False)
)

drop_matrix = degradation[
    ["campaign_f1_drop_mean", "core3_f1_drop_mean"]
].to_numpy(dtype=np.float64)

fig = plt.figure(figsize=(10.4, 6.2))
grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.25, 1.0],
    wspace=0.34,
)

ax_heat = fig.add_subplot(grid[0, 0])
image = ax_heat.imshow(
    drop_matrix,
    cmap="YlOrRd",
    aspect="auto",
    vmin=0.0,
    vmax=max(0.56, float(drop_matrix.max())),
)

ax_heat.set_yticks(np.arange(len(degradation)))
ax_heat.set_yticklabels(degradation_labels)
ax_heat.set_xticks([0, 1])
ax_heat.set_xticklabels(["Campaign F1", "Core-three F1"])
ax_heat.set_title("A. Degradation fingerprint")

for row in range(drop_matrix.shape[0]):
    for column in range(drop_matrix.shape[1]):
        value = drop_matrix[row, column]
        ax_heat.text(
            column,
            row,
            f"$-$ {value:.3f}",
            ha="center",
            va="center",
            color="white" if value > 0.27 else "#202124",
            fontsize=8.5,
            fontweight="bold" if value > 0.25 else "normal",
        )

colorbar = fig.colorbar(
    image,
    ax=ax_heat,
    fraction=0.046,
    pad=0.04,
)
colorbar.set_label("Mean loss from complete input")

ax_profile = fig.add_subplot(grid[0, 1])
y = np.arange(len(degradation))

ax_profile.errorbar(
    degradation["campaign_auroc_mean"],
    y - 0.12,
    xerr=degradation["campaign_auroc_sd"],
    fmt="o",
    color=COLORS["blue"],
    ecolor="#90A4AE",
    capsize=2.5,
    label="Campaign AUROC",
    zorder=3,
)
ax_profile.errorbar(
    degradation["core3_macro_f1_mean"],
    y + 0.12,
    xerr=degradation["core3_macro_f1_sd"],
    fmt="D",
    color=COLORS["orange"],
    ecolor="#B0BEC5",
    capsize=2.5,
    label="Core-three F1",
    zorder=3,
)

ax_profile.set_yticks(y)
ax_profile.set_yticklabels([])
ax_profile.invert_yaxis()
ax_profile.set_xlabel("Retained performance (mean $\\pm$ SD)")
ax_profile.set_title("B. Residual capability")
ax_profile.legend(frameon=False, loc="lower right")
polish_axis(ax_profile, grid_axis="x")

fig.suptitle(
    "Missing-Modality Robustness: Degradation and Residual Capability",
    y=1.01,
)
fig.subplots_adjust(left=0.19, right=0.97, bottom=0.10, top=0.90)
save_figure(fig, "figure_2_missing_modality_fingerprint")
figure_catalog_rows.append(
    {
        "figure": "Figure 2",
        "file_stem": "figure_2_missing_modality_fingerprint",
        "design": "Annotated heatmap plus estimation profile",
        "message": "Which missing inputs damage which task",
    }
)


# -----------------------------------------------------------------------------
# Figure 3: seed-aware selective risk--coverage frontier
# -----------------------------------------------------------------------------

selective_seed = load_csv(
    FINAL_DIR / "stance_selective_seed_results.csv",
    [
        "method",
        "seed",
        "target_validation_coverage",
        "test_coverage",
        "selective_risk",
    ],
)

method_styles = {
    "dense_am_jepa": (COLORS["navy"], "o"),
    "dual_text_only": (COLORS["teal"], "s"),
    "dual_full_multimodal": (COLORS["orange"], "D"),
}

fig, ax = plt.subplots(figsize=(8.2, 5.6))

for method, group in selective_seed.groupby("method", sort=False):
    color, marker = method_styles.get(method, (COLORS["gray"], "o"))

    for seed, seed_group in group.groupby("seed"):
        ordered_seed = seed_group.sort_values("test_coverage")
        ax.plot(
            ordered_seed["test_coverage"],
            ordered_seed["selective_risk"],
            color=color,
            linewidth=0.75,
            alpha=0.16,
            zorder=1,
        )

    summary = (
        group.groupby("target_validation_coverage", as_index=False)
        .agg(
            coverage_mean=("test_coverage", "mean"),
            coverage_sd=("test_coverage", "std"),
            risk_mean=("selective_risk", "mean"),
            risk_sd=("selective_risk", "std"),
        )
        .sort_values("coverage_mean")
    )

    ax.errorbar(
        summary["coverage_mean"],
        summary["risk_mean"],
        xerr=summary["coverage_sd"].fillna(0.0),
        yerr=summary["risk_sd"].fillna(0.0),
        marker=marker,
        markersize=6.0,
        linewidth=2.2,
        capsize=2.5,
        color=color,
        label=method.replace("_", " "),
        zorder=4,
    )

primary_point = selective_aggregate[
    (selective_aggregate["method"] == "dual_text_only")
    & np.isclose(
        selective_aggregate["target_validation_coverage"],
        0.7,
    )
].iloc[0]

ax.scatter(
    primary_point["test_coverage_mean"],
    primary_point["selective_risk_mean"],
    marker="*",
    s=220,
    color=COLORS["gold"],
    edgecolor="#5D4037",
    linewidth=0.8,
    zorder=6,
)
ax.annotate(
    "Best selective macro-F1\ncoverage 0.692, risk 0.272",
    (
        primary_point["test_coverage_mean"],
        primary_point["selective_risk_mean"],
    ),
    xytext=(34, 28),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->",
        color="#5D4037",
        lw=0.8,
    ),
    fontsize=8.4,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="#CFD8DC",
    ),
)

ax.annotate(
    "preferred direction",
    xy=(0.57, 0.16),
    xytext=(0.72, 0.23),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COLORS["gray"],
        lw=1.0,
    ),
    color=COLORS["gray"],
    fontsize=8.2,
)

ax.set_xlabel("Realized test coverage")
ax.set_ylabel("Selective risk (lower is better)")
ax.set_xlim(0.50, 1.025)
ax.set_ylim(0.12, 0.44)
ax.set_title("Seed-Aware Selective Risk--Coverage Frontier")
ax.legend(frameon=False, loc="upper left")
polish_axis(ax)
fig.tight_layout()
save_figure(fig, "figure_3_selective_risk_frontier")
figure_catalog_rows.append(
    {
        "figure": "Figure 3",
        "file_stem": "figure_3_selective_risk_frontier",
        "design": "Seed trajectories and uncertainty-aware frontier",
        "message": "Reliability gained by abstention",
    }
)


# -----------------------------------------------------------------------------
# Figure 4: multi-metric estimation plot for stance architectures
# -----------------------------------------------------------------------------

stance_plot = stance_aggregate[
    stance_aggregate["evaluation_set"] == "all_test"
].copy()

method_order = [
    "dense_am_jepa",
    "dual_text_only",
    "dual_full_multimodal",
]
method_labels = {
    "dense_am_jepa": "Dense AM-JEPA",
    "dual_text_only": "Dual text sparse",
    "dual_full_multimodal": "Dual multimodal sparse",
}
method_colors = {
    "dense_am_jepa": COLORS["navy"],
    "dual_text_only": COLORS["teal"],
    "dual_full_multimodal": COLORS["orange"],
}

metric_panels = [
    ("accuracy", "Accuracy", True),
    ("macro_f1", "Macro-F1", True),
    ("mcc", "MCC", True),
    ("macro_ovr_auroc", "Macro-AUROC", True),
    ("ece", "ECE", False),
]

fig, axes = plt.subplots(
    1,
    len(metric_panels),
    figsize=(13.4, 4.2),
    sharey=True,
)
y_positions = np.arange(len(method_order))[::-1]

for panel_number, (metric, title, higher_is_better) in enumerate(metric_panels):
    ax = axes[panel_number]

    for y_position, method in zip(y_positions, method_order):
        row = stance_plot[stance_plot["method"] == method].iloc[0]
        mean = float(row[f"{metric}_mean"])
        sd = float(row[f"{metric}_sd"])
        ax.errorbar(
            mean,
            y_position,
            xerr=sd,
            fmt="o",
            markersize=6.5,
            capsize=3,
            elinewidth=1.5,
            color=method_colors[method],
            zorder=3,
        )
        ax.text(
            mean,
            y_position + 0.18,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=method_colors[method],
        )

    ax.set_title(title)
    ax.set_xlabel("higher $\\rightarrow$" if higher_is_better else "$\\leftarrow$ lower")
    polish_axis(ax, grid_axis="x")

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels([method_labels[item] for item in method_order])
axes[0].set_ylim(-0.55, len(method_order) - 0.42)

fig.suptitle("Five-Seed Stance Architecture Estimation Plot", y=1.02)
fig.tight_layout()
save_figure(fig, "figure_4_stance_estimation_forest")
figure_catalog_rows.append(
    {
        "figure": "Figure 4",
        "file_stem": "figure_4_stance_estimation_forest",
        "design": "Five-panel estimation/forest plot",
        "message": "Accuracy, discrimination, association, and calibration",
    }
)


# -----------------------------------------------------------------------------
# Figure 5: dual-selector gain map
# -----------------------------------------------------------------------------

gain_map = (
    dual_seed_deltas
    .groupby(["feature_set", "budget"])["map_delta"]
    .mean()
    .unstack("budget")
    .sort_index()
)

gain_sd = (
    dual_seed_deltas
    .groupby(["feature_set", "budget"])["map_delta"]
    .std()
    .unstack("budget")
    .reindex(index=gain_map.index, columns=gain_map.columns)
)

values = gain_map.to_numpy(dtype=np.float64)
limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

fig, ax = plt.subplots(figsize=(7.7, 3.8))
image = ax.imshow(
    values,
    cmap="RdBu_r",
    norm=normalization,
    aspect="auto",
)

ax.set_xticks(np.arange(len(gain_map.columns)))
ax.set_xticklabels([f"K={int(item)}" for item in gain_map.columns])
ax.set_yticks(np.arange(len(gain_map.index)))
ax.set_yticklabels(
    [item.replace("_", " ") for item in gain_map.index]
)

for row in range(values.shape[0]):
    for column in range(values.shape[1]):
        mean_value = values[row, column]
        sd_value = gain_sd.iloc[row, column]
        ax.text(
            column,
            row,
            f"{mean_value:+.3f}\n$\\pm${sd_value:.3f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=("white" if abs(mean_value) > 0.65 * limit else "#202124"),
            fontweight="bold" if mean_value > 0 else "normal",
        )

colorbar = fig.colorbar(image, ax=ax, pad=0.025)
colorbar.set_label("Dual minus matched single-selector MAP")
ax.set_title("Where Does Explicit E$^+$/E$^-$ Separation Help?")
ax.set_xlabel("Evidence budget")
ax.set_ylabel("Feature view")
fig.tight_layout()
save_figure(fig, "figure_5_dual_selector_gain_map")
figure_catalog_rows.append(
    {
        "figure": "Figure 5",
        "file_stem": "figure_5_dual_selector_gain_map",
        "design": "Diverging gain heatmap",
        "message": "Budget- and modality-dependent dual-selector gains",
    }
)


figure_catalog = pd.DataFrame(figure_catalog_rows)
figure_catalog.to_csv(
    TABLE_DIR / "publication_figure_catalog.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# LaTeX-ready tables
# -----------------------------------------------------------------------------

objective_latex_columns = [
    column
    for column in [
        "variant",
        "campaign_f1",
        "campaign_auroc",
        "campaign_auprc",
        "core3_macro_f1",
        "core3_macro_mcc",
        "mean_latent_std",
    ]
    if column in objective_seed42.columns
]

write_latex_table(
    objective_seed42,
    TABLE_DIR / "objective_ablation_seed42.tex",
    "Seed-42 AM-JEPA objective ablation.",
    "tab:objective-ablation",
    objective_latex_columns,
)

write_latex_table(
    robustness_summary,
    TABLE_DIR / "missing_modality_robustness.tex",
    "Five-seed missing-modality robustness.",
    "tab:missing-modality",
    [
        "scenario",
        "campaign_f1_mean",
        "campaign_f1_sd",
        "core3_macro_f1_mean",
        "core3_macro_f1_sd",
    ],
)

write_latex_table(
    stance_plot,
    TABLE_DIR / "stance_architecture_ablation.tex",
    "Five-seed stance architecture comparison.",
    "tab:stance-architecture",
    [
        "method",
        "macro_f1_mean",
        "macro_f1_sd",
        "mcc_mean",
        "mcc_sd",
        "macro_ovr_auroc_mean",
        "ece_mean",
    ],
)


# -----------------------------------------------------------------------------
# Final result summary and reproducibility manifest
# -----------------------------------------------------------------------------

dense_row = stance_plot[
    stance_plot["method"] == "dense_am_jepa"
].iloc[0]

dual_text_row = stance_plot[
    stance_plot["method"] == "dual_text_only"
].iloc[0]

selective_primary = selective_aggregate[
    (selective_aggregate["method"] == "dual_text_only")
    & np.isclose(
        selective_aggregate["target_validation_coverage"],
        0.7,
    )
].iloc[0]

result_summary = {
    "label_status": (
        "AI-assisted single-reviewer silver pilot; not independent "
        "human gold-standard evidence."
    ),
    "campaign_f1_five_seed_mean": float(
        multiseed_summary["campaign_f1"].mean()
    ),
    "campaign_f1_five_seed_sd": float(
        multiseed_summary["campaign_f1"].std(ddof=1)
    ),
    "core3_macro_f1_five_seed_mean": float(
        multiseed_summary["core3_macro_f1"].mean()
    ),
    "core3_macro_f1_five_seed_sd": float(
        multiseed_summary["core3_macro_f1"].std(ddof=1)
    ),
    "dense_stance_macro_f1": float(dense_row["macro_f1_mean"]),
    "dual_text_full_coverage_macro_f1_mean": float(
        dual_text_row["macro_f1_mean"]
    ),
    "dual_text_full_coverage_macro_f1_sd": float(
        dual_text_row["macro_f1_sd"]
    ),
    "dual_text_selective_macro_f1": float(
        selective_primary["macro_f1_mean"]
    ),
    "dual_text_selective_test_coverage": float(
        selective_primary["test_coverage_mean"]
    ),
    "deterministic_reports": int(len(test_reports)),
    "deterministic_abstentions": int(
        test_reports["abstained"].astype(bool).sum()
    ),
    "external_verifier_used": False,
    "statistical_note": (
        "Five-seed Wilcoxon tests have low power; exact effect estimates "
        "and Holm-adjusted p-values are reported without overstating "
        "non-significance."
    ),
}

with open(
    CELL31_DIR / "final_result_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(result_summary, file, indent=2)


# Hash every final table and figure before packaging.
export_candidates = sorted(
    list(TABLE_DIR.glob("*"))
    + list(FIGURE_DIR.glob("*"))
    + [
        CELL30_DIR / "deterministic_evidence_linked_test_reports.csv",
        CELL30_DIR / "deterministic_report_integrity_summary.csv",
        CELL31_DIR / "final_result_summary.json",
    ]
)

hash_rows = []

for path in export_candidates:
    require_file(path)
    hash_rows.append(
        {
            "file": path.name,
            "relative_path": str(path.relative_to(AMJEPA_DIR)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    )

hash_manifest = pd.DataFrame(hash_rows)
hash_manifest.to_csv(
    CELL31_DIR / "final_export_hash_manifest.csv",
    index=False,
)


final_manifest = {
    "cell": 31,
    "status": "completed",
    "alpha": ALPHA,
    "multiple_testing": "Holm adjustment within hypothesis family",
    "paired_test": "two-sided Wilcoxon signed-rank",
    "seeds": EXPECTED_SEEDS,
    "integrity_checks": integrity_checks,
    "figures": [path.name for path in sorted(FIGURE_DIR.glob("*.png"))],
    "tables": [path.name for path in sorted(TABLE_DIR.glob("*"))],
    "external_verifier_used": False,
    "test_used_for_model_selection": False,
}

with open(
    CELL31_DIR / "cell31_final_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(final_manifest, file, indent=2)


# -----------------------------------------------------------------------------
# Final publication package
# -----------------------------------------------------------------------------

zip_path = CELL31_DIR / "SparseSentinel_APT_final_results_package.zip"

package_files = (
    export_candidates
    + [
        CELL31_DIR / "final_export_hash_manifest.csv",
        CELL31_DIR / "cell31_final_manifest.json",
    ]
)

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for path in package_files:
        # Do not include legacy routing queues or protected evaluation keys.
        lowered = path.name.lower()
        if "deepseek" in lowered or "do_not_load" in lowered:
            raise AssertionError(
                f"Forbidden legacy artifact entered final package: {path.name}"
            )

        archive.write(
            path,
            arcname=str(path.relative_to(AMJEPA_DIR)),
        )


# -----------------------------------------------------------------------------
# Display final results
# -----------------------------------------------------------------------------

print("=" * 80)
print("SPARSESENTINEL-APT FINAL STATISTICS AND EXPORT")
print("=" * 80)

print("\nCorrected seed-42 objective ablation:")
display(objective_seed42.round(6))

print("\nHolm-adjusted paired statistical tests:")
display(
    all_tests[
        [
            "family",
            "metric",
            "comparison",
            "pairs",
            "mean_difference",
            "cohen_dz",
            "p_value",
            "p_holm",
            "significant_holm_0_05",
        ]
    ].round(6)
)

print("\nFinal integrity checks:")
display(integrity_table)

print("\nFinal result summary:")
display(pd.DataFrame([result_summary]).T.rename(columns={0: "value"}))

print("\nCell 31 completed successfully.")
print("Final package:")
print(zip_path)
print("\nFinal package SHA-256:")
print(sha256_file(zip_path))
