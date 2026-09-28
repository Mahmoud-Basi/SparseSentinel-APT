# =============================================================================
# CELL 30: FINAL ABLATION, ROBUSTNESS, AND DETERMINISTIC REPORTING
# =============================================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


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
CELL30_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VALIDATION_COVERAGE = 0.80
EXPECTED_TEST_RECORDS = 89
EXPECTED_CANDIDATES_PER_RECORD = 32
REPORT_SCHEMA_VERSION = "1.0"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required preserved output is missing:\n{path}\n"
            "Run the relevant completed cell or restore its saved output."
        )
    return path


def read_csv(path, required_columns=None):
    path = require_file(path)
    table = pd.read_csv(path)

    if required_columns is not None:
        missing = sorted(set(required_columns) - set(table.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {missing}"
            )

    return table


def confidence_threshold_for_fixed_coverage(confidences, target_coverage):
    """Return a deterministic validation-only confidence threshold."""
    values = np.asarray(confidences, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite validation confidences were found.")

    number_to_retain = int(
        np.ceil(float(target_coverage) * len(values))
    )
    number_to_retain = min(max(number_to_retain, 1), len(values))

    return float(
        np.sort(values)[::-1][number_to_retain - 1]
    )


def split_identifiers(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def stable_report_id(annotation_id):
    digest = hashlib.sha256(
        str(annotation_id).encode("utf-8")
    ).hexdigest()[:16].upper()
    return f"SSR-{digest}"


def validate_identifier_list(value, ledger):
    identifiers = split_identifiers(value)
    invalid = [item for item in identifiers if item not in ledger]
    return len(invalid) == 0, invalid


# -----------------------------------------------------------------------------
# Load only preserved experiment outputs
# -----------------------------------------------------------------------------

objective_results = read_csv(
    OBJECTIVE_DIR / "am_jepa_objective_ablation.csv",
    [
        "variant",
        "campaign_f1",
        "campaign_auroc",
        "core3_macro_f1",
    ],
)

missing_modality_detailed = read_csv(
    ROBUSTNESS_DIR / "missing_modality_detailed_results.csv",
    [
        "variant",
        "seed",
        "scenario",
        "campaign_f1",
        "campaign_auroc",
        "core3_macro_f1",
        "campaign_f1_drop",
        "core3_f1_drop",
    ],
)

selector_primary = read_csv(
    SELECTOR_DIR / "selector_primary_summary.csv"
)

dual_delta_summary = read_csv(
    DUAL_DIR / "dual_single_delta_summary.csv"
)

stance_aggregate = read_csv(
    FINAL_DIR / "stance_test_aggregate.csv",
    [
        "method",
        "evaluation_set",
        "macro_f1_mean",
        "mcc_mean",
        "macro_ovr_auroc_mean",
        "ece_mean",
    ],
)

selective_aggregate = read_csv(
    FINAL_DIR / "stance_selective_aggregate.csv",
    [
        "method",
        "target_validation_coverage",
        "test_coverage_mean",
        "accuracy_mean",
        "macro_f1_mean",
        "mcc_mean",
        "selective_risk_mean",
    ],
)

validation_ensemble = read_csv(
    FINAL_DIR / "primary_validation_ensemble.csv",
    [
        "annotation_id",
        "predictive_confidence",
        "predicted_stance",
    ],
)

test_ensemble = read_csv(
    FINAL_DIR / "primary_test_ensemble_with_reference.csv",
    [
        "annotation_id",
        "window_id",
        "aspect",
        "predicted_stance",
        "predictive_confidence",
        "predictive_entropy",
    ],
)

consensus_evidence = read_csv(
    FINAL_DIR / "stance_consensus_selected_evidence.csv",
    [
        "annotation_id",
        "split",
        "channel",
        "rank",
        "evidence_id",
    ],
)

if len(test_ensemble) != EXPECTED_TEST_RECORDS:
    raise AssertionError(
        f"Expected {EXPECTED_TEST_RECORDS} test records, "
        f"but found {len(test_ensemble)}."
    )


# -----------------------------------------------------------------------------
# A. Objective ablation
# -----------------------------------------------------------------------------

objective_columns = [
    column
    for column in [
        "variant",
        "epochs",
        "campaign_f1",
        "campaign_mcc",
        "campaign_auroc",
        "campaign_auprc",
        "core3_macro_f1",
        "core3_macro_mcc",
        "mean_latent_std",
        "near_constant_dimensions",
    ]
    if column in objective_results.columns
]

objective_ablation_table = (
    objective_results[objective_columns]
    .sort_values("campaign_f1", ascending=False)
    .reset_index(drop=True)
)


# -----------------------------------------------------------------------------
# B. Five-seed missing-modality robustness
# -----------------------------------------------------------------------------

full_robustness = missing_modality_detailed[
    missing_modality_detailed["variant"] == "full_am_jepa"
].copy()

if full_robustness["seed"].nunique() != 5:
    raise AssertionError(
        "Missing-modality robustness does not contain five AM-JEPA seeds."
    )

robustness_rows = []

for scenario, group in full_robustness.groupby(
    "scenario",
    sort=False,
):
    row = {
        "scenario": scenario,
        "seeds": int(group["seed"].nunique()),
    }

    if "realized_drop_rate" in group.columns:
        row["realized_drop_rate"] = float(
            group["realized_drop_rate"].mean()
        )

    for metric in [
        "campaign_f1",
        "campaign_auroc",
        "core3_macro_f1",
        "campaign_f1_drop",
        "core3_f1_drop",
    ]:
        values = group[metric].to_numpy(dtype=np.float64)
        row[metric + "_mean"] = float(values.mean())
        row[metric + "_sd"] = float(values.std(ddof=1))

    robustness_rows.append(row)

missing_modality_summary = pd.DataFrame(robustness_rows)

scenario_order = [
    "complete",
    "missing_network",
    "missing_temporal",
    "missing_text",
    "network_only",
    "temporal_only",
    "text_only",
    "random_drop_25pct",
    "random_drop_50pct",
    "random_drop_75pct",
]

missing_modality_summary["scenario"] = pd.Categorical(
    missing_modality_summary["scenario"],
    categories=scenario_order,
    ordered=True,
)

missing_modality_summary = (
    missing_modality_summary
    .sort_values("scenario")
    .reset_index(drop=True)
)


# -----------------------------------------------------------------------------
# C. Sparse-selector and final stance ablations
# -----------------------------------------------------------------------------

stance_ablation_table = (
    stance_aggregate[
        stance_aggregate["evaluation_set"] == "all_test"
    ]
    .copy()
    .sort_values("macro_f1_mean", ascending=False)
    .reset_index(drop=True)
)

selective_robustness_table = (
    selective_aggregate
    .copy()
    .sort_values(
        [
            "method",
            "target_validation_coverage",
        ],
        ascending=[True, False],
    )
    .reset_index(drop=True)
)


# -----------------------------------------------------------------------------
# D. Deterministic evidence-linked reports and abstention
# -----------------------------------------------------------------------------

confidence_threshold = confidence_threshold_for_fixed_coverage(
    validation_ensemble["predictive_confidence"],
    TARGET_VALIDATION_COVERAGE,
)

test_reports = test_ensemble.drop(
    columns=[
        column
        for column in test_ensemble.columns
        if column.lower() in {
            "route_to_deepseek",
            "deepseek_request",
        }
    ],
    errors="ignore",
).copy()

test_evidence = consensus_evidence[
    consensus_evidence["split"] == "test"
].copy()

evidence_lists = (
    test_evidence
    .sort_values(
        [
            "annotation_id",
            "channel",
            "rank",
        ]
    )
    .groupby(
        [
            "annotation_id",
            "channel",
        ]
    )["evidence_id"]
    .apply(lambda values: ";".join(values.astype(str)))
    .unstack("channel")
    .reset_index()
    .rename(
        columns={
            "SUPPORTING": "supporting_evidence_ids",
            "CONTRADICTING": "contradicting_evidence_ids",
        }
    )
)

for column in [
    "supporting_evidence_ids",
    "contradicting_evidence_ids",
]:
    if column not in evidence_lists.columns:
        evidence_lists[column] = ""

test_reports = test_reports.merge(
    evidence_lists,
    on="annotation_id",
    how="left",
    validate="one_to_one",
)

test_reports[
    [
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    ]
] = test_reports[
    [
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    ]
].fillna("")

test_reports["abstained"] = (
    test_reports["predictive_confidence"]
    < confidence_threshold
)

test_reports["decision"] = np.where(
    test_reports["abstained"],
    "ABSTAIN",
    "REPORT_STANCE",
)

test_reports["reason_code"] = np.where(
    test_reports["abstained"],
    "LOW_VALIDATION_CALIBRATED_CONFIDENCE",
    "SUFFICIENT_VALIDATION_CALIBRATED_CONFIDENCE",
)

test_reports["report_id"] = test_reports[
    "annotation_id"
].map(stable_report_id)

test_reports["report_schema_version"] = REPORT_SCHEMA_VERSION
test_reports["confidence_threshold"] = confidence_threshold
test_reports["target_validation_coverage"] = TARGET_VALIDATION_COVERAGE


# Build the immutable identifier ledger from the complete candidate-event table.
if "all_stance_events" in globals():
    evidence_ledger = set(
        all_stance_events["evidence_id"].astype(str)
    )
else:
    # Safe fallback for a restarted kernel: validate against every identifier
    # preserved in the consensus-evidence file.
    evidence_ledger = set(
        consensus_evidence["evidence_id"].astype(str)
    )

support_validity = []
contradiction_validity = []
invalid_identifier_lists = []

for row in test_reports.itertuples(index=False):
    support_valid, invalid_support = validate_identifier_list(
        row.supporting_evidence_ids,
        evidence_ledger,
    )
    contradiction_valid, invalid_contradiction = validate_identifier_list(
        row.contradicting_evidence_ids,
        evidence_ledger,
    )

    support_validity.append(support_valid)
    contradiction_validity.append(contradiction_valid)
    invalid_identifier_lists.append(
        ";".join(invalid_support + invalid_contradiction)
    )

test_reports["supporting_ids_valid"] = support_validity
test_reports["contradicting_ids_valid"] = contradiction_validity
test_reports["invalid_evidence_ids"] = invalid_identifier_lists
test_reports["identifier_integrity_passed"] = (
    test_reports["supporting_ids_valid"]
    & test_reports["contradicting_ids_valid"]
)

# Fail closed: any ledger violation becomes an abstention.
integrity_failure = ~test_reports["identifier_integrity_passed"]
test_reports.loc[integrity_failure, "abstained"] = True
test_reports.loc[integrity_failure, "decision"] = "ABSTAIN"
test_reports.loc[
    integrity_failure,
    "reason_code",
] = "EVIDENCE_IDENTIFIER_INTEGRITY_FAILURE"


# -----------------------------------------------------------------------------
# E. Integrity and tamper-detection tests
# -----------------------------------------------------------------------------

required_report_columns = {
    "report_id",
    "annotation_id",
    "window_id",
    "aspect",
    "predicted_stance",
    "predictive_confidence",
    "predictive_entropy",
    "decision",
    "reason_code",
    "supporting_evidence_ids",
    "contradicting_evidence_ids",
    "identifier_integrity_passed",
}

schema_passed = required_report_columns.issubset(test_reports.columns)
unique_report_ids_passed = test_reports["report_id"].is_unique
finite_confidence_passed = np.isfinite(
    test_reports["predictive_confidence"]
).all()
finite_entropy_passed = np.isfinite(
    test_reports["predictive_entropy"]
).all()
identifier_integrity_passed = bool(
    test_reports["identifier_integrity_passed"].all()
)

# Deliberately inject one nonexistent identifier. The validator must reject it.
tampered_value = "NONEXISTENT-EVIDENCE-ID"
tamper_was_rejected = not validate_identifier_list(
    tampered_value,
    evidence_ledger,
)[0]

if not all(
    [
        schema_passed,
        unique_report_ids_passed,
        finite_confidence_passed,
        finite_entropy_passed,
        identifier_integrity_passed,
        tamper_was_rejected,
    ]
):
    raise AssertionError(
        "Deterministic report integrity verification failed."
    )

report_integrity_summary = pd.DataFrame(
    [
        {
            "records": int(len(test_reports)),
            "confidence_threshold": confidence_threshold,
            "target_validation_coverage": TARGET_VALIDATION_COVERAGE,
            "retained_records": int((~test_reports["abstained"]).sum()),
            "abstained_records": int(test_reports["abstained"].sum()),
            "realized_coverage": float((~test_reports["abstained"]).mean()),
            "schema_passed": bool(schema_passed),
            "unique_report_ids_passed": bool(unique_report_ids_passed),
            "finite_confidence_passed": bool(finite_confidence_passed),
            "finite_entropy_passed": bool(finite_entropy_passed),
            "identifier_integrity_passed": bool(identifier_integrity_passed),
            "tamper_detection_passed": bool(tamper_was_rejected),
            "external_verifier_used": False,
        }
    ]
)


# -----------------------------------------------------------------------------
# Save Cell 30 outputs
# -----------------------------------------------------------------------------

objective_ablation_table.to_csv(
    CELL30_DIR / "objective_ablation_publication_table.csv",
    index=False,
)

missing_modality_summary.to_csv(
    CELL30_DIR / "missing_modality_robustness_publication_table.csv",
    index=False,
)

selector_primary.to_csv(
    CELL30_DIR / "single_selector_ablation_publication_table.csv",
    index=False,
)

dual_delta_summary.to_csv(
    CELL30_DIR / "dual_selector_ablation_publication_table.csv",
    index=False,
)

stance_ablation_table.to_csv(
    CELL30_DIR / "stance_architecture_ablation_publication_table.csv",
    index=False,
)

selective_robustness_table.to_csv(
    CELL30_DIR / "selective_prediction_robustness_table.csv",
    index=False,
)

test_reports.to_csv(
    CELL30_DIR / "deterministic_evidence_linked_test_reports.csv",
    index=False,
)

report_integrity_summary.to_csv(
    CELL30_DIR / "deterministic_report_integrity_summary.csv",
    index=False,
)

cell30_manifest = {
    "cell": 30,
    "status": "completed",
    "policy": (
        "Synthesis of preserved Cells 22--29 outputs; no retraining, "
        "external verification, or post-hoc label correction."
    ),
    "target_validation_coverage": TARGET_VALIDATION_COVERAGE,
    "validation_selected_confidence_threshold": confidence_threshold,
    "test_records": int(len(test_reports)),
    "retained_records": int((~test_reports["abstained"]).sum()),
    "abstained_records": int(test_reports["abstained"].sum()),
    "report_schema_version": REPORT_SCHEMA_VERSION,
    "integrity_checks_passed": True,
    "label_limitation": (
        "AI-assisted single-reviewer silver pilot; not an independent "
        "human gold standard."
    ),
}

with open(
    CELL30_DIR / "cell30_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(cell30_manifest, file, indent=2)


# -----------------------------------------------------------------------------
# Display concise publication-facing output
# -----------------------------------------------------------------------------

print("=" * 80)
print("SPARSESENTINEL-APT FINAL ABLATION AND ROBUSTNESS")
print("=" * 80)

print("\nObjective ablation:")
display(objective_ablation_table.round(6))

print("\nFive-seed missing-modality robustness:")
display(missing_modality_summary.round(6))

print("\nFinal stance architecture ablation:")
display(stance_ablation_table.round(6))

print("\nSelective-prediction robustness:")
display(selective_robustness_table.round(6))

print("\nDeterministic report integrity:")
display(report_integrity_summary.round(6))

print("\nCell 30 completed successfully.")
print("No model was retrained and no external verifier was used.")
print("Saved outputs to:")
print(CELL30_DIR)
