# =============================================================================
# CELL 25A: DAPT2020-MAST THREAT-STANCE ANNOTATION PACKAGE
# =============================================================================

from pathlib import Path
import hashlib
import json
import math
import re

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

ANNOTATION_DIR = (
    OUTPUT_DIR
    / "dapt2020_mast_annotation"
)

ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WINDOWS_PATH = (
    OUTPUT_DIR
    / "dapt2020_mast_windows.parquet"
)

MEMBERSHIP_PATH = (
    OUTPUT_DIR
    / "dapt2020_mast_window_membership_enriched.parquet"
)

ASPECT_PAIRS_PATH = (
    OUTPUT_DIR
    / "dapt2020_mast_aspect_pairs.parquet"
)

# Training is sampled for expert supervision. Every validation and test
# window-aspect pair is retained to avoid evaluation-sampling bias.
TRAINING_REVIEW_TARGET = 1200
ANNOTATION_BATCH_SIZE = 240
ANNOTATION_SEED = 20260922

ASPECT_ORDER = [
    "reconnaissance",
    "initial_access_foothold",
    "lateral_movement",
    "data_exfiltration",
]

SPLIT_ORDER = [
    "train",
    "validation",
    "test",
]

VALID_STANCE_LABELS = [
    "SUPPORTING",
    "CONTRADICTING",
    "NEUTRAL",
]

ASPECT_GUIDE = {
    "reconnaissance": {
        "definition": (
            "Evidence indicating discovery, probing, enumeration, or "
            "information gathering that can prepare a later intrusion."
        ),
        "include": (
            "Scanning, service discovery, host discovery, systematic probing, "
            "or repeated exploratory communication."
        ),
        "exclude": (
            "Routine isolated service contact without an exploratory pattern."
        ),
    },
    "initial_access_foothold": {
        "definition": (
            "Evidence indicating an attempt to obtain or maintain an initial "
            "presence in the target environment."
        ),
        "include": (
            "Exploit-like access behavior, suspicious authentication or login "
            "activity, and communication consistent with foothold creation."
        ),
        "exclude": (
            "Ordinary authorized access or unrelated discovery behavior."
        ),
    },
    "lateral_movement": {
        "definition": (
            "Evidence indicating movement, access, or propagation from one "
            "internal system or trust boundary to another."
        ),
        "include": (
            "Abnormal internal-to-internal access, remote-service use, or "
            "multi-endpoint propagation consistent with campaign progression."
        ),
        "exclude": (
            "Expected internal service communication without movement evidence."
        ),
    },
    "data_exfiltration": {
        "definition": (
            "Evidence indicating unauthorized transfer, staging, or removal of "
            "data from the monitored environment."
        ),
        "include": (
            "Outbound transfer behavior, unusual volume or asymmetry, or a "
            "campaign-consistent channel for removing data."
        ),
        "exclude": (
            "Routine outbound communication without evidence of data removal."
        ),
    },
}


# -----------------------------------------------------------------------------
# Required-artifact checks
# -----------------------------------------------------------------------------

required_paths = [
    WINDOWS_PATH,
    MEMBERSHIP_PATH,
    ASPECT_PAIRS_PATH,
]

missing_paths = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    raise FileNotFoundError(
        "Missing required DAPT2020-MAST artifacts:\n"
        + "\n".join(missing_paths)
        + "\n\nRun the dataset-construction and verbalization cells first."
    )


# -----------------------------------------------------------------------------
# Load and validate source tables
# -----------------------------------------------------------------------------

windows = pd.read_parquet(WINDOWS_PATH)
membership = pd.read_parquet(MEMBERSHIP_PATH)
aspect_pairs = pd.read_parquet(ASPECT_PAIRS_PATH)

required_window_columns = {
    "window_id",
    "split",
    "day",
    "evidence_count",
}

required_membership_columns = {
    "window_id",
    "position",
    "evidence_id",
    "timestamp_parsed",
    "network_scope",
    "verbalized_event",
}

required_pair_columns = {
    "window_id",
    "split",
    "day",
    "aspect",
    "weak_stance",
    "support_count",
    "other_attack_count",
    "benign_count",
    "insufficient_context",
    "requires_expert_review",
}

for table_name, table, required_columns in [
    ("windows", windows, required_window_columns),
    ("membership", membership, required_membership_columns),
    ("aspect_pairs", aspect_pairs, required_pair_columns),
]:
    missing_columns = sorted(
        required_columns - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: {missing_columns}"
        )

assert windows["window_id"].is_unique
assert aspect_pairs[["window_id", "aspect"]].duplicated().sum() == 0
assert set(aspect_pairs["aspect"]) == set(ASPECT_ORDER)
assert set(aspect_pairs["split"]).issubset(set(SPLIT_ORDER))
assert membership["verbalized_event"].notna().all()

membership["timestamp_parsed"] = pd.to_datetime(
    membership["timestamp_parsed"],
    errors="raise",
)


# -----------------------------------------------------------------------------
# Candidate-context categories used only for balanced sampling
# -----------------------------------------------------------------------------

def candidate_context(row):
    if int(row["support_count"]) > 0:
        return "EXACT_ASPECT_SUPPORT"

    if int(row["other_attack_count"]) > 0:
        return "OTHER_ATTACK_CONTEXT"

    return "BENIGN_ONLY_CONTEXT"


aspect_pairs = aspect_pairs.copy()

aspect_pairs["candidate_context"] = aspect_pairs.apply(
    candidate_context,
    axis=1,
)


# -----------------------------------------------------------------------------
# Deterministic identifiers and balanced training sampling
# -----------------------------------------------------------------------------

def stable_digest(value, length=20):
    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:length].upper()


def deterministic_balanced_sample(
    dataframe,
    target_size,
    seed,
):
    """
    Round-robin sampling across aspect, candidate context, and review flag.
    This keeps rare strata represented without using evaluation outcomes.
    """
    dataframe = dataframe.copy()

    if target_size is None or target_size >= len(dataframe):
        return dataframe.copy()

    dataframe["sampling_stratum"] = (
        dataframe["aspect"].astype(str)
        + "|"
        + dataframe["candidate_context"].astype(str)
        + "|review="
        + dataframe["requires_expert_review"].astype(str)
    )

    dataframe["sampling_rank"] = [
        stable_digest(
            f"{seed}|{window_id}|{aspect}",
            length=64,
        )
        for window_id, aspect in zip(
            dataframe["window_id"],
            dataframe["aspect"],
        )
    ]

    groups = {
        stratum: group.sort_values("sampling_rank").copy()
        for stratum, group in dataframe.groupby(
            "sampling_stratum",
            sort=True,
        )
    }

    selected_indices = []
    offsets = {
        stratum: 0
        for stratum in groups
    }

    while len(selected_indices) < target_size:
        added = False

        for stratum in sorted(groups):
            offset = offsets[stratum]
            group = groups[stratum]

            if offset < len(group):
                selected_indices.append(
                    group.index[offset]
                )
                offsets[stratum] += 1
                added = True

                if len(selected_indices) == target_size:
                    break

        if not added:
            break

    return dataframe.loc[selected_indices].copy()


training_pairs = aspect_pairs[
    aspect_pairs["split"] == "train"
].copy()

validation_pairs = aspect_pairs[
    aspect_pairs["split"] == "validation"
].copy()

test_pairs = aspect_pairs[
    aspect_pairs["split"] == "test"
].copy()

selected_training_pairs = deterministic_balanced_sample(
    dataframe=training_pairs,
    target_size=min(
        TRAINING_REVIEW_TARGET,
        len(training_pairs),
    ),
    seed=ANNOTATION_SEED,
)

# Preserve every evaluation pair. The final model-development workflow must
# keep the test annotations sealed until all design choices are frozen.
selected_pairs = pd.concat(
    [
        selected_training_pairs,
        validation_pairs,
        test_pairs,
    ],
    ignore_index=True,
)

assert selected_pairs[["window_id", "aspect"]].duplicated().sum() == 0

selected_pairs["annotation_id"] = [
    "MAST-A-" + stable_digest(
        f"{ANNOTATION_SEED}|{window_id}|{aspect}",
        length=20,
    )
    for window_id, aspect in zip(
        selected_pairs["window_id"],
        selected_pairs["aspect"],
    )
]

assert selected_pairs["annotation_id"].is_unique


# -----------------------------------------------------------------------------
# Build blinded evidence packets
# -----------------------------------------------------------------------------

membership_groups = {
    window_id: group.sort_values(
        ["position", "timestamp_parsed", "evidence_id"]
    ).copy()
    for window_id, group in membership.groupby(
        "window_id",
        sort=False,
    )
}


def build_evidence_packet(window_id):
    if window_id not in membership_groups:
        raise KeyError(
            f"No evidence membership found for {window_id}."
        )

    group = membership_groups[window_id]
    packet = []

    for _, row in group.iterrows():
        packet.append(
            {
                "position": int(row["position"]),
                "evidence_id": str(row["evidence_id"]),
                "timestamp": row["timestamp_parsed"].isoformat(),
                "network_scope": str(row["network_scope"]),
                "evidence_text": str(row["verbalized_event"]),
            }
        )

    return json.dumps(
        packet,
        ensure_ascii=False,
        separators=(",", ":"),
    )


annotation_rows = []

for _, row in selected_pairs.iterrows():
    guide = ASPECT_GUIDE[row["aspect"]]

    annotation_rows.append(
        {
            "annotation_id": row["annotation_id"],
            "aspect": row["aspect"],
            "aspect_definition": guide["definition"],
            "inclusion_criteria": guide["include"],
            "exclusion_criteria": guide["exclude"],
            "evidence_packet_json": build_evidence_packet(
                row["window_id"]
            ),
            "annotator_a_stance": "",
            "annotator_a_insufficient_context": "",
            "annotator_a_confidence_1_to_5": "",
            "annotator_a_decisive_evidence_ids": "",
            "annotator_a_notes": "",
            "annotator_b_stance": "",
            "annotator_b_insufficient_context": "",
            "annotator_b_confidence_1_to_5": "",
            "annotator_b_decisive_evidence_ids": "",
            "annotator_b_notes": "",
        }
    )

annotation_table = pd.DataFrame(annotation_rows)

# Deterministic blind ordering prevents split and weak-label blocks from being
# visible to annotators.
annotation_table["blind_order"] = [
    stable_digest(
        f"blind|{ANNOTATION_SEED}|{annotation_id}",
        length=64,
    )
    for annotation_id in annotation_table["annotation_id"]
]

annotation_table = (
    annotation_table
    .sort_values("blind_order")
    .drop(columns="blind_order")
    .reset_index(drop=True)
)


# -----------------------------------------------------------------------------
# Leakage and integrity audit
# -----------------------------------------------------------------------------

forbidden_export_columns = {
    "split",
    "day",
    "window_id",
    "weak_stance",
    "support_count",
    "other_attack_count",
    "benign_count",
    "campaign_label",
    "aspect_label",
    "stage_label",
    "activity_label",
}

assert forbidden_export_columns.isdisjoint(
    set(annotation_table.columns)
)

assert len(annotation_table) == len(selected_pairs)
assert annotation_table["annotation_id"].is_unique

packet_text = "\n".join(
    annotation_table["evidence_packet_json"].astype(str)
)

# These patterns detect accidental field-name leakage, not ordinary words
# appearing naturally in controlled descriptions.
forbidden_field_patterns = [
    r"Activity_[A-Za-z0-9_]+",
    r"Stage_[A-Za-z0-9_]+",
    r"campaign_label",
    r"aspect_label",
    r"stage_label",
    r"activity_label",
]

detected_patterns = [
    pattern
    for pattern in forbidden_field_patterns
    if re.search(pattern, packet_text, flags=re.IGNORECASE)
]

if detected_patterns:
    raise ValueError(
        "Potential label-field leakage detected in annotation packets: "
        f"{detected_patterns}"
    )

# Every evidence identifier in every annotation packet must be immutable and
# traceable to the membership table.
known_evidence_ids = set(
    membership["evidence_id"].astype(str)
)

for packet in annotation_table["evidence_packet_json"]:
    for item in json.loads(packet):
        if item["evidence_id"] not in known_evidence_ids:
            raise ValueError(
                "Unknown evidence identifier in annotation packet: "
                f"{item['evidence_id']}"
            )


# -----------------------------------------------------------------------------
# Hidden provenance key and adjudication template
# -----------------------------------------------------------------------------

key_columns = [
    "annotation_id",
    "window_id",
    "split",
    "day",
    "aspect",
    "candidate_context",
    "weak_stance",
    "support_count",
    "other_attack_count",
    "benign_count",
    "insufficient_context",
    "requires_expert_review",
]

annotation_key = selected_pairs[key_columns].copy()

adjudication_template = annotation_table[
    [
        "annotation_id",
        "aspect",
    ]
].copy()

adjudication_template["adjudicated_stance"] = ""
adjudication_template["adjudicated_insufficient_context"] = ""
adjudication_template["adjudicated_decisive_evidence_ids"] = ""
adjudication_template["adjudication_notes"] = ""


# -----------------------------------------------------------------------------
# Batch creation
# -----------------------------------------------------------------------------

number_of_batches = math.ceil(
    len(annotation_table)
    / ANNOTATION_BATCH_SIZE
)

batch_manifest_rows = []

for batch_index in range(number_of_batches):
    start = batch_index * ANNOTATION_BATCH_SIZE
    stop = min(
        start + ANNOTATION_BATCH_SIZE,
        len(annotation_table),
    )

    batch = annotation_table.iloc[start:stop].copy()
    batch_number = batch_index + 1

    shared_columns = [
        "annotation_id",
        "aspect",
        "aspect_definition",
        "inclusion_criteria",
        "exclusion_criteria",
        "evidence_packet_json",
    ]

    annotator_a_columns = shared_columns + [
        column
        for column in batch.columns
        if column.startswith("annotator_a_")
    ]

    annotator_b_columns = shared_columns + [
        column
        for column in batch.columns
        if column.startswith("annotator_b_")
    ]

    batch_a_name = (
        f"annotator_A_batch_{batch_number:03d}.csv"
    )
    batch_b_name = (
        f"annotator_B_batch_{batch_number:03d}.csv"
    )

    batch[annotator_a_columns].to_csv(
        ANNOTATION_DIR / batch_a_name,
        index=False,
        encoding="utf-8-sig",
    )

    batch[annotator_b_columns].to_csv(
        ANNOTATION_DIR / batch_b_name,
        index=False,
        encoding="utf-8-sig",
    )

    batch_manifest_rows.append(
        {
            "batch_number": batch_number,
            "annotator_a_file": batch_a_name,
            "annotator_b_file": batch_b_name,
            "records": len(batch),
            "first_annotation_id": batch.iloc[0]["annotation_id"],
            "last_annotation_id": batch.iloc[-1]["annotation_id"],
        }
    )

batch_manifest = pd.DataFrame(
    batch_manifest_rows
)


# -----------------------------------------------------------------------------
# Annotation guidelines
# -----------------------------------------------------------------------------

guidelines = {
    "task": (
        "Determine the evidential stance of the complete window toward the "
        "specified cybersecurity aspect. Threat stance is not emotion."
    ),
    "labels": {
        "SUPPORTING": (
            "The observed evidence materially supports that the target aspect "
            "is present in this window."
        ),
        "CONTRADICTING": (
            "The observed evidence materially supports a benign or alternative "
            "interpretation that conflicts with the target aspect. Mere absence "
            "of attack evidence is not contradiction."
        ),
        "NEUTRAL": (
            "The evidence is unrelated, non-decisive, or compatible with both "
            "the target aspect and an alternative interpretation."
        ),
    },
    "insufficient_context": (
        "Mark true when the available window does not permit a defensible stance. "
        "This flag is separate from the three semantic labels."
    ),
    "decisive_evidence_ids": (
        "List only evidence identifiers that directly justify the selected stance, "
        "separated by semicolons. Do not create new identifiers."
    ),
    "rules": [
        "Judge only the supplied evidence and aspect definition.",
        "Do not infer an attack solely from unusual volume or timing.",
        "Do not label a benign-only window as CONTRADICTING solely because the "
        "target aspect is absent.",
        "Use NEUTRAL when evidence concerns a different attack aspect.",
        "Set insufficient_context=true when a reliable decision requires evidence "
        "outside the supplied window.",
        "Confidence must be an integer from 1 (very uncertain) to 5 (very certain).",
    ],
    "aspect_guide": ASPECT_GUIDE,
    "allowed_stance_labels": VALID_STANCE_LABELS,
}


# -----------------------------------------------------------------------------
# Save annotation artifacts
# -----------------------------------------------------------------------------

MASTER_ANNOTATION_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_blinded_annotation_master.csv"
)

HIDDEN_KEY_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_annotation_hidden_key.parquet"
)

ADJUDICATION_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_adjudication_template.csv"
)

BATCH_MANIFEST_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_annotation_batch_manifest.csv"
)

GUIDELINES_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_annotation_guidelines.json"
)

PACKAGE_MANIFEST_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_annotation_package_manifest.json"
)

annotation_table.to_csv(
    MASTER_ANNOTATION_PATH,
    index=False,
    encoding="utf-8-sig",
)

annotation_key.to_parquet(
    HIDDEN_KEY_PATH,
    index=False,
)

adjudication_template.to_csv(
    ADJUDICATION_PATH,
    index=False,
    encoding="utf-8-sig",
)

batch_manifest.to_csv(
    BATCH_MANIFEST_PATH,
    index=False,
)

with open(
    GUIDELINES_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        guidelines,
        file,
        ensure_ascii=False,
        indent=2,
    )

package_manifest = {
    "seed": ANNOTATION_SEED,
    "training_review_target": TRAINING_REVIEW_TARGET,
    "validation_policy": "all_window_aspect_pairs",
    "test_policy": "all_window_aspect_pairs_sealed_until_final_evaluation",
    "annotation_batch_size": ANNOTATION_BATCH_SIZE,
    "total_annotation_units": int(len(annotation_table)),
    "number_of_batches": int(number_of_batches),
    "stance_labels": VALID_STANCE_LABELS,
    "insufficient_context_is_separate_flag": True,
    "contradiction_is_never_automatically_inferred": True,
    "weak_labels_exposed_to_annotators": False,
    "split_exposed_to_annotators": False,
    "source_artifacts": {
        "windows": str(WINDOWS_PATH),
        "membership": str(MEMBERSHIP_PATH),
        "aspect_pairs": str(ASPECT_PAIRS_PATH),
    },
}

with open(
    PACKAGE_MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        package_manifest,
        file,
        ensure_ascii=False,
        indent=2,
    )


# -----------------------------------------------------------------------------
# Report package statistics without exposing test-label distributions
# -----------------------------------------------------------------------------

selection_summary = (
    annotation_key
    .groupby("split", observed=True)
    .size()
    .reindex(SPLIT_ORDER)
    .rename("annotation_units")
    .reset_index()
)

aspect_summary = (
    annotation_key
    .groupby(
        ["split", "aspect"],
        observed=True,
    )
    .size()
    .rename("annotation_units")
    .reset_index()
)

integrity_summary = pd.DataFrame(
    [
        {
            "annotation_units": len(annotation_table),
            "unique_annotation_ids": annotation_table[
                "annotation_id"
            ].nunique(),
            "batches": number_of_batches,
            "forbidden_export_columns_present": len(
                forbidden_export_columns.intersection(
                    annotation_table.columns
                )
            ),
            "forbidden_field_patterns_detected": len(
                detected_patterns
            ),
            "contradiction_auto_labels": 0,
        }
    ]
)

print("=" * 80)
print("DAPT2020-MAST THREAT-STANCE ANNOTATION PACKAGE")
print("=" * 80)

print("\nSelection summary:")
display(selection_summary)

print("\nAspect coverage:")
display(aspect_summary)

print("\nIntegrity summary:")
display(integrity_summary)

print("\nAnnotation batches:")
display(batch_manifest)

print("\nSaved blinded annotation master:")
print(MASTER_ANNOTATION_PATH)

print("\nSaved hidden provenance key:")
print(HIDDEN_KEY_PATH)

print("\nSaved adjudication template:")
print(ADJUDICATION_PATH)

print("\nSaved annotation guidelines:")
print(GUIDELINES_PATH)

print(
    "\nCell 25A completed successfully. "
    "Begin with annotation_batch_001.csv as the pilot batch. "
    "Do not open the hidden key while annotating."
)
