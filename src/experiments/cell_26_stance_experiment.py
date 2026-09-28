# =============================================================================
# CELL 26: SILVER-LABEL THREAT-STANCE EVALUATION
# =============================================================================

from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from scipy.optimize import minimize_scalar
except ImportError as error:
    raise ImportError(
        "Cell 26 requires SciPy for validation-only temperature scaling."
    ) from error


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

STANCE_SEED = 42
BOOTSTRAP_REPETITIONS = 2000

STANCE_CLASSES = [
    "CONTRADICTING",
    "NEUTRAL",
    "SUPPORTING",
]

ASPECT_ORDER = [
    "reconnaissance",
    "initial_access_foothold",
    "lateral_movement",
    "data_exfiltration",
]

C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

TARGET_COVERAGES = [
    1.00,
    0.90,
    0.80,
]

random.seed(STANCE_SEED)
np.random.seed(STANCE_SEED)

if "DATA_DIR" not in globals():
    DATA_DIR = Path.cwd()
else:
    DATA_DIR = Path(DATA_DIR)

OUTPUT_DIR = DATA_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "am_jepa_tensors"
AMJEPA_DIR = OUTPUT_DIR / "am_jepa_experiment"
ANNOTATION_DIR = OUTPUT_DIR / "dapt2020_mast_annotation"

STANCE_OUTPUT_DIR = (
    AMJEPA_DIR
    / "silver_stance_evaluation"
)

STANCE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HIDDEN_KEY_PATH = (
    ANNOTATION_DIR
    / "dapt2020_mast_annotation_hidden_key.parquet"
)

REPRESENTATION_PATH = (
    AMJEPA_DIR
    / "am_jepa_frozen_representations.npy"
)

WINDOW_IDS_PATH = (
    TENSOR_DIR
    / "window_ids.npy"
)


def locate_resolved_labels():
    filename = (
        "DAPT2020_MAST_batch_001_"
        "provisional_resolved_labels.xlsx"
    )

    candidates = [
        ANNOTATION_DIR / filename,
        OUTPUT_DIR / "provisional_resolved_labels" / filename,
        DATA_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches = list(
        OUTPUT_DIR.rglob(filename)
    )

    if len(recursive_matches) == 1:
        return recursive_matches[0]

    if len(recursive_matches) > 1:
        raise RuntimeError(
            "Multiple resolved-label workbooks were found. "
            "Keep one authoritative copy or set RESOLVED_LABEL_PATH "
            "manually.\n"
            + "\n".join(str(path) for path in recursive_matches)
        )

    raise FileNotFoundError(
        "Resolved-label workbook was not found. Expected:\n"
        + "\n".join(str(path) for path in candidates)
    )


RESOLVED_LABEL_PATH = locate_resolved_labels()

required_paths = [
    HIDDEN_KEY_PATH,
    REPRESENTATION_PATH,
    WINDOW_IDS_PATH,
    TENSOR_DIR / "X_network.npy",
    TENSOR_DIR / "X_temporal_final.npy",
    TENSOR_DIR / "attention_mask.npy",
    TENSOR_DIR / "text_embedding_lookup.npy",
    TENSOR_DIR / "window_text_ids.npy",
]

missing_paths = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    raise FileNotFoundError(
        "Missing required stance-evaluation artifacts:\n"
        + "\n".join(missing_paths)
        + "\n\nRun Cells 19–21 and 25A first, then place the "
          "resolved pilot workbook in the annotation output directory."
    )


# -----------------------------------------------------------------------------
# Load resolved labels and hidden provenance
# -----------------------------------------------------------------------------

resolved = pd.read_excel(
    RESOLVED_LABEL_PATH,
    sheet_name="Provisional labels",
)

resolved = resolved.rename(
    columns={
        "Annotation ID": "annotation_id",
        "Aspect": "aspect",
        "Resolved stance": "stance",
        "Resolved insufficient context": "insufficient_context",
        "Resolved confidence (1–5)": "confidence",
        "Resolved decisive evidence IDs": "decisive_evidence_ids",
        "Resolution source": "resolution_source",
        "Dataset status": "dataset_status",
    }
)

required_label_columns = {
    "annotation_id",
    "aspect",
    "stance",
    "insufficient_context",
    "confidence",
    "decisive_evidence_ids",
    "resolution_source",
    "dataset_status",
}

missing_label_columns = sorted(
    required_label_columns
    - set(resolved.columns)
)

if missing_label_columns:
    raise ValueError(
        "Resolved-label workbook is missing columns: "
        f"{missing_label_columns}"
    )

resolved["annotation_id"] = (
    resolved["annotation_id"]
    .astype(str)
    .str.strip()
)

resolved["aspect"] = (
    resolved["aspect"]
    .astype(str)
    .str.strip()
    .str.lower()
)

resolved["stance"] = (
    resolved["stance"]
    .astype(str)
    .str.strip()
    .str.upper()
)

resolved["insufficient_context"] = (
    resolved["insufficient_context"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "TRUE": True,
        "FALSE": False,
    })
)

resolved["confidence"] = pd.to_numeric(
    resolved["confidence"],
    errors="raise",
).astype(np.int64)

if resolved["annotation_id"].duplicated().any():
    raise ValueError(
        "Resolved labels contain duplicate annotation IDs."
    )

if not set(resolved["stance"]).issubset(
    set(STANCE_CLASSES)
):
    raise ValueError(
        "Unexpected stance labels: "
        f"{sorted(set(resolved['stance']) - set(STANCE_CLASSES))}"
    )

if not set(resolved["aspect"]).issubset(
    set(ASPECT_ORDER)
):
    raise ValueError(
        "Unexpected aspects: "
        f"{sorted(set(resolved['aspect']) - set(ASPECT_ORDER))}"
    )

if resolved["insufficient_context"].isna().any():
    raise ValueError(
        "Invalid insufficient-context value detected."
    )

if not resolved["confidence"].between(1, 5).all():
    raise ValueError(
        "Confidence must be an integer from 1 through 5."
    )

hidden_key = pd.read_parquet(
    HIDDEN_KEY_PATH
)

required_key_columns = {
    "annotation_id",
    "window_id",
    "split",
    "aspect",
}

missing_key_columns = sorted(
    required_key_columns
    - set(hidden_key.columns)
)

if missing_key_columns:
    raise ValueError(
        "Hidden provenance key is missing columns: "
        f"{missing_key_columns}"
    )

hidden_key["annotation_id"] = (
    hidden_key["annotation_id"]
    .astype(str)
    .str.strip()
)

hidden_key["window_id"] = (
    hidden_key["window_id"]
    .astype(str)
)

hidden_key["aspect"] = (
    hidden_key["aspect"]
    .astype(str)
    .str.strip()
    .str.lower()
)

hidden_key["split"] = (
    hidden_key["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

stance_table = resolved.merge(
    hidden_key[
        [
            "annotation_id",
            "window_id",
            "split",
            "aspect",
        ]
    ],
    on=[
        "annotation_id",
        "aspect",
    ],
    how="left",
    validate="one_to_one",
)

if stance_table[
    [
        "window_id",
        "split",
    ]
].isna().any().any():
    missing_ids = stance_table.loc[
        stance_table["window_id"].isna(),
        "annotation_id",
    ].tolist()

    raise ValueError(
        "Some resolved labels do not match the hidden key: "
        f"{missing_ids[:10]}"
    )

if len(stance_table) != 240:
    warnings.warn(
        "The pilot contains "
        f"{len(stance_table)} records rather than the expected 240."
    )


# -----------------------------------------------------------------------------
# Recover tensor order and build representations
# -----------------------------------------------------------------------------

try:
    window_ids = np.load(
        WINDOW_IDS_PATH,
        allow_pickle=False,
    )
except ValueError:
    window_ids = np.load(
        WINDOW_IDS_PATH,
        allow_pickle=True,
    )

window_ids = np.asarray(
    window_ids
).astype(str)

window_to_tensor_index = {
    window_id: index
    for index, window_id in enumerate(window_ids)
}

stance_table["tensor_index"] = (
    stance_table["window_id"]
    .map(window_to_tensor_index)
)

if stance_table["tensor_index"].isna().any():
    raise ValueError(
        "Some resolved windows were not found in window_ids.npy."
    )

stance_table["tensor_index"] = (
    stance_table["tensor_index"]
    .astype(np.int64)
)

frozen_representations = np.load(
    REPRESENTATION_PATH,
    allow_pickle=False,
).astype(np.float32)

X_network_stance = np.load(
    TENSOR_DIR / "X_network.npy",
    allow_pickle=False,
).astype(np.float32)

X_temporal_stance = np.load(
    TENSOR_DIR / "X_temporal_final.npy",
    allow_pickle=False,
).astype(np.float32)

attention_mask_stance = np.load(
    TENSOR_DIR / "attention_mask.npy",
    allow_pickle=False,
).astype(bool)

text_embedding_lookup_stance = np.load(
    TENSOR_DIR / "text_embedding_lookup.npy",
    allow_pickle=False,
).astype(np.float32)

window_text_ids_stance = np.load(
    TENSOR_DIR / "window_text_ids.npy",
    allow_pickle=False,
).astype(np.int64)

number_of_windows = len(window_ids)

if frozen_representations.shape[0] != number_of_windows:
    raise ValueError(
        "Frozen representation count does not match window_ids.npy."
    )


def masked_mean(array, mask):
    expanded_mask = mask.astype(
        np.float32
    )[..., None]

    denominator = expanded_mask.sum(
        axis=1
    ).clip(min=1.0)

    return (
        array
        * expanded_mask
    ).sum(axis=1) / denominator


network_pooled = masked_mean(
    X_network_stance,
    attention_mask_stance,
)

temporal_pooled = masked_mean(
    X_temporal_stance,
    attention_mask_stance,
)

text_sequence_stance = (
    text_embedding_lookup_stance[
        window_text_ids_stance
    ]
)

text_pooled = masked_mean(
    text_sequence_stance,
    attention_mask_stance,
)

aspect_to_index = {
    aspect: index
    for index, aspect in enumerate(
        ASPECT_ORDER
    )
}

aspect_one_hot = np.zeros(
    (
        len(stance_table),
        len(ASPECT_ORDER),
    ),
    dtype=np.float32,
)

aspect_one_hot[
    np.arange(len(stance_table)),
    stance_table["aspect"]
    .map(aspect_to_index)
    .to_numpy(dtype=np.int64),
] = 1.0

tensor_indices = stance_table[
    "tensor_index"
].to_numpy(dtype=np.int64)

representation_sets = {
    "aspect_prior": aspect_one_hot,
    "pooled_multimodal": np.concatenate(
        [
            network_pooled[tensor_indices],
            temporal_pooled[tensor_indices],
            text_pooled[tensor_indices],
            aspect_one_hot,
        ],
        axis=1,
    ),
    "frozen_am_jepa": np.concatenate(
        [
            frozen_representations[tensor_indices],
            aspect_one_hot,
        ],
        axis=1,
    ),
}

stance_to_index = {
    stance: index
    for index, stance in enumerate(
        STANCE_CLASSES
    )
}

y_stance = stance_table[
    "stance"
].map(stance_to_index).to_numpy(
    dtype=np.int64
)

sample_weights = (
    stance_table["confidence"]
    .to_numpy(dtype=np.float64)
    / 5.0
)

sample_weights *= np.where(
    stance_table[
        "insufficient_context"
    ].to_numpy(dtype=bool),
    0.5,
    1.0,
)


# -----------------------------------------------------------------------------
# Fixed leakage-aware partitions
# -----------------------------------------------------------------------------

split_masks = {
    split_name: (
        stance_table["split"]
        .to_numpy()
        == split_name
    )
    for split_name in [
        "train",
        "validation",
        "test",
    ]
}

for split_name, split_mask in split_masks.items():
    if split_mask.sum() == 0:
        raise ValueError(
            f"The pilot has no {split_name} labels."
        )

train_window_ids = set(
    stance_table.loc[
        split_masks["train"],
        "window_id",
    ]
)

validation_window_ids = set(
    stance_table.loc[
        split_masks["validation"],
        "window_id",
    ]
)

test_window_ids = set(
    stance_table.loc[
        split_masks["test"],
        "window_id",
    ]
)

if (
    train_window_ids & validation_window_ids
    or train_window_ids & test_window_ids
    or validation_window_ids & test_window_ids
):
    raise AssertionError(
        "A window appears in more than one partition."
    )

distribution_rows = []

for split_name, split_mask in split_masks.items():
    for stance_name in STANCE_CLASSES:
        distribution_rows.append(
            {
                "split": split_name,
                "stance": stance_name,
                "count": int(
                    (
                        stance_table.loc[
                            split_mask,
                            "stance",
                        ]
                        == stance_name
                    ).sum()
                ),
                "unique_windows": int(
                    stance_table.loc[
                        split_mask,
                        "window_id",
                    ].nunique()
                ),
                "insufficient_context": int(
                    stance_table.loc[
                        split_mask,
                        "insufficient_context",
                    ].sum()
                ),
            }
        )

split_distribution = pd.DataFrame(
    distribution_rows
)

for split_name in [
    "train",
    "validation",
    "test",
]:
    present_classes = set(
        stance_table.loc[
            split_masks[split_name],
            "stance",
        ]
    )

    if present_classes != set(STANCE_CLASSES):
        raise ValueError(
            f"The {split_name} pilot partition does not contain "
            "all three stance classes. This 240-unit pilot is too "
            "small for a fixed-split three-class evaluation. "
            "Complete additional blinded batches before proceeding. "
            f"Present classes: {sorted(present_classes)}"
        )


# -----------------------------------------------------------------------------
# Metrics and validation-only calibration
# -----------------------------------------------------------------------------

def build_probe(C_value):
    return Pipeline(
        steps=[
            (
                "standardizer",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C_value,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=10000,
                    random_state=STANCE_SEED,
                ),
            ),
        ]
    )


def align_probabilities(probabilities, observed_classes):
    aligned = np.full(
        (
            probabilities.shape[0],
            len(STANCE_CLASSES),
        ),
        1e-12,
        dtype=np.float64,
    )

    for source_column, class_index in enumerate(
        observed_classes
    ):
        aligned[
            :,
            int(class_index),
        ] = probabilities[
            :,
            source_column,
        ]

    aligned /= aligned.sum(
        axis=1,
        keepdims=True,
    )

    return aligned


def expected_calibration_error(
    labels,
    probabilities,
    number_of_bins=10,
):
    confidence = probabilities.max(
        axis=1
    )

    predictions = probabilities.argmax(
        axis=1
    )

    correct = (
        predictions == labels
    ).astype(np.float64)

    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    ece = 0.0

    for bin_index in range(
        number_of_bins
    ):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == 0:
            selected = (
                confidence >= lower
            ) & (
                confidence <= upper
            )
        else:
            selected = (
                confidence > lower
            ) & (
                confidence <= upper
            )

        if selected.any():
            ece += (
                selected.mean()
                * abs(
                    correct[selected].mean()
                    - confidence[selected].mean()
                )
            )

    return float(ece)


def multiclass_brier(labels, probabilities):
    one_hot = label_binarize(
        labels,
        classes=np.arange(
            len(STANCE_CLASSES)
        ),
    )

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def fit_temperature(labels, probabilities):
    logits = np.log(
        np.clip(
            probabilities,
            1e-12,
            1.0,
        )
    )

    def objective(log_temperature):
        temperature = np.exp(
            log_temperature
        )

        scaled_logits = logits / temperature

        scaled_logits -= scaled_logits.max(
            axis=1,
            keepdims=True,
        )

        scaled_probabilities = np.exp(
            scaled_logits
        )

        scaled_probabilities /= (
            scaled_probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )

        return log_loss(
            labels,
            scaled_probabilities,
            labels=np.arange(
                len(STANCE_CLASSES)
            ),
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(0.05),
            np.log(20.0),
        ),
        method="bounded",
    )

    return float(
        np.exp(result.x)
    )


def apply_temperature(probabilities, temperature):
    logits = np.log(
        np.clip(
            probabilities,
            1e-12,
            1.0,
        )
    )

    scaled_logits = logits / temperature

    scaled_logits -= scaled_logits.max(
        axis=1,
        keepdims=True,
    )

    scaled_probabilities = np.exp(
        scaled_logits
    )

    scaled_probabilities /= (
        scaled_probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    return scaled_probabilities


def calculate_stance_metrics(
    labels,
    probabilities,
):
    predictions = probabilities.argmax(
        axis=1
    )

    result = {
        "samples": int(len(labels)),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            labels,
            predictions,
        ),
        "nll": log_loss(
            labels,
            probabilities,
            labels=np.arange(
                len(STANCE_CLASSES)
            ),
        ),
        "brier": multiclass_brier(
            labels,
            probabilities,
        ),
        "ece": expected_calibration_error(
            labels,
            probabilities,
        ),
    }

    try:
        result["macro_ovr_auroc"] = (
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=np.arange(
                    len(STANCE_CLASSES)
                ),
            )
        )
    except ValueError:
        result["macro_ovr_auroc"] = np.nan

    return result


# -----------------------------------------------------------------------------
# Train on training labels, tune/calibrate on validation, test once
# -----------------------------------------------------------------------------

main_rows = []
per_class_rows = []
selective_rows = []
prediction_tables = []

train_mask = split_masks["train"]
validation_mask = split_masks["validation"]
test_mask = split_masks["test"]

for representation_name, features in (
    representation_sets.items()
):
    validation_search_rows = []

    if representation_name == "aspect_prior":
        candidate_C_values = C_GRID
    else:
        candidate_C_values = C_GRID

    for C_value in candidate_C_values:
        probe = build_probe(C_value)

        probe.fit(
            features[train_mask],
            y_stance[train_mask],
            classifier__sample_weight=
                sample_weights[train_mask],
        )

        validation_probabilities = (
            align_probabilities(
                probe.predict_proba(
                    features[
                        validation_mask
                    ]
                ),
                probe.named_steps[
                    "classifier"
                ].classes_,
            )
        )

        validation_metrics = (
            calculate_stance_metrics(
                y_stance[validation_mask],
                validation_probabilities,
            )
        )

        validation_search_rows.append(
            {
                "C": C_value,
                **validation_metrics,
            }
        )

    validation_search = pd.DataFrame(
        validation_search_rows
    )

    best_validation_row = (
        validation_search
        .sort_values(
            by=[
                "macro_f1",
                "mcc",
                "balanced_accuracy",
                "nll",
                "C",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    best_C = float(
        best_validation_row["C"]
    )

    final_probe = build_probe(
        best_C
    )

    final_probe.fit(
        features[train_mask],
        y_stance[train_mask],
        classifier__sample_weight=
            sample_weights[train_mask],
    )

    validation_probabilities = (
        align_probabilities(
            final_probe.predict_proba(
                features[validation_mask]
            ),
            final_probe.named_steps[
                "classifier"
            ].classes_,
        )
    )

    temperature = fit_temperature(
        y_stance[validation_mask],
        validation_probabilities,
    )

    test_probabilities_uncalibrated = (
        align_probabilities(
            final_probe.predict_proba(
                features[test_mask]
            ),
            final_probe.named_steps[
                "classifier"
            ].classes_,
        )
    )

    test_probabilities = apply_temperature(
        test_probabilities_uncalibrated,
        temperature,
    )

    evaluation_masks = {
        "all_test": np.ones(
            test_mask.sum(),
            dtype=bool,
        ),
        "sufficient_context_test": (
            ~stance_table.loc[
                test_mask,
                "insufficient_context",
            ].to_numpy(dtype=bool)
        ),
    }

    test_labels = y_stance[
        test_mask
    ]

    for evaluation_set, local_mask in (
        evaluation_masks.items()
    ):
        if local_mask.sum() == 0:
            continue

        metrics = calculate_stance_metrics(
            test_labels[local_mask],
            test_probabilities[local_mask],
        )

        main_rows.append(
            {
                "representation": representation_name,
                "evaluation_set": evaluation_set,
                "best_C": best_C,
                "temperature": temperature,
                "validation_macro_f1": float(
                    best_validation_row[
                        "macro_f1"
                    ]
                ),
                **metrics,
            }
        )

        local_predictions = (
            test_probabilities[
                local_mask
            ].argmax(axis=1)
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                test_labels[local_mask],
                local_predictions,
                labels=np.arange(
                    len(STANCE_CLASSES)
                ),
                zero_division=0,
            )
        )

        for class_index, class_name in enumerate(
            STANCE_CLASSES
        ):
            per_class_rows.append(
                {
                    "representation": representation_name,
                    "evaluation_set": evaluation_set,
                    "stance": class_name,
                    "precision": precision[class_index],
                    "recall": recall[class_index],
                    "f1": f1[class_index],
                    "support": int(
                        support[class_index]
                    ),
                }
            )

    validation_confidence = (
        apply_temperature(
            validation_probabilities,
            temperature,
        ).max(axis=1)
    )

    test_confidence = test_probabilities.max(
        axis=1
    )

    for target_coverage in TARGET_COVERAGES:
        if target_coverage >= 1.0:
            threshold = 0.0
        else:
            threshold = float(
                np.quantile(
                    validation_confidence,
                    1.0 - target_coverage,
                    method="lower",
                )
            )

        retained = (
            test_confidence
            >= threshold
        )

        if retained.sum() == 0:
            continue

        retained_metrics = (
            calculate_stance_metrics(
                test_labels[retained],
                test_probabilities[retained],
            )
        )

        selective_rows.append(
            {
                "representation": representation_name,
                "target_validation_coverage": target_coverage,
                "confidence_threshold": threshold,
                "test_coverage": retained.mean(),
                "abstention_rate": 1.0 - retained.mean(),
                **retained_metrics,
            }
        )

    prediction_table = stance_table.loc[
        test_mask,
        [
            "annotation_id",
            "window_id",
            "aspect",
            "stance",
            "insufficient_context",
            "confidence",
        ],
    ].copy()

    prediction_table[
        "representation"
    ] = representation_name

    prediction_table[
        "predicted_stance"
    ] = [
        STANCE_CLASSES[index]
        for index in test_probabilities.argmax(
            axis=1
        )
    ]

    prediction_table[
        "predictive_confidence"
    ] = test_confidence

    prediction_table[
        "predictive_entropy"
    ] = -np.sum(
        test_probabilities
        * np.log(
            np.clip(
                test_probabilities,
                1e-12,
                1.0,
            )
        ),
        axis=1,
    )

    for class_index, class_name in enumerate(
        STANCE_CLASSES
    ):
        prediction_table[
            "probability_"
            + class_name.lower()
        ] = test_probabilities[
            :,
            class_index,
        ]

    prediction_tables.append(
        prediction_table
    )


# -----------------------------------------------------------------------------
# Majority-class baseline
# -----------------------------------------------------------------------------

dummy = DummyClassifier(
    strategy="prior",
    random_state=STANCE_SEED,
)

dummy.fit(
    aspect_one_hot[train_mask],
    y_stance[train_mask],
    sample_weight=sample_weights[
        train_mask
    ],
)

dummy_test_probabilities = (
    align_probabilities(
        dummy.predict_proba(
            aspect_one_hot[test_mask]
        ),
        dummy.classes_,
    )
)

dummy_metrics = calculate_stance_metrics(
    y_stance[test_mask],
    dummy_test_probabilities,
)

main_rows.append(
    {
        "representation": "majority_prior",
        "evaluation_set": "all_test",
        "best_C": np.nan,
        "temperature": 1.0,
        "validation_macro_f1": np.nan,
        **dummy_metrics,
    }
)


# -----------------------------------------------------------------------------
# Group bootstrap confidence intervals for primary AM-JEPA test result
# -----------------------------------------------------------------------------

all_predictions = pd.concat(
    prediction_tables,
    ignore_index=True,
)

primary_predictions = all_predictions[
    all_predictions["representation"]
    == "frozen_am_jepa"
].copy()

bootstrap_source = primary_predictions[
    ~primary_predictions[
        "insufficient_context"
    ].astype(bool)
].copy()

bootstrap_groups = (
    bootstrap_source["window_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(
    STANCE_SEED
)

bootstrap_rows = []

for repetition in range(
    BOOTSTRAP_REPETITIONS
):
    sampled_groups = rng.choice(
        bootstrap_groups,
        size=len(bootstrap_groups),
        replace=True,
    )

    sampled_parts = [
        bootstrap_source[
            bootstrap_source["window_id"]
            == group_id
        ]
        for group_id in sampled_groups
    ]

    sampled = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    sampled_true = sampled[
        "stance"
    ].map(stance_to_index).to_numpy(
        dtype=np.int64
    )

    sampled_predicted = sampled[
        "predicted_stance"
    ].map(stance_to_index).to_numpy(
        dtype=np.int64
    )

    bootstrap_rows.append(
        {
            "repetition": repetition,
            "macro_f1": f1_score(
                sampled_true,
                sampled_predicted,
                average="macro",
                zero_division=0,
            ),
            "mcc": matthews_corrcoef(
                sampled_true,
                sampled_predicted,
            ),
        }
    )

bootstrap_results = pd.DataFrame(
    bootstrap_rows
)

bootstrap_summary = pd.DataFrame(
    [
        {
            "metric": metric,
            "mean": bootstrap_results[
                metric
            ].mean(),
            "ci_2_5": bootstrap_results[
                metric
            ].quantile(0.025),
            "ci_97_5": bootstrap_results[
                metric
            ].quantile(0.975),
            "bootstrap_repetitions": (
                BOOTSTRAP_REPETITIONS
            ),
            "resampling_unit": "window_id",
            "evaluation_set": (
                "sufficient_context_test"
            ),
        }
        for metric in [
            "macro_f1",
            "mcc",
        ]
    ]
)


# -----------------------------------------------------------------------------
# Decisive-evidence sparsity audit
# -----------------------------------------------------------------------------

def count_decisive_ids(value):
    text = str(value).strip()

    if (
        not text
        or text.upper() in {
            "NAN",
            "NONE",
            "NONE_RECORDED",
        }
    ):
        return 0

    return len(
        [
            item
            for item in text.split(";")
            if item.strip()
        ]
    )


stance_table[
    "decisive_evidence_count"
] = stance_table[
    "decisive_evidence_ids"
].map(count_decisive_ids)

sparsity_rows = []

for aspect_name in [
    "ALL",
    *ASPECT_ORDER,
]:
    if aspect_name == "ALL":
        subset = stance_table
    else:
        subset = stance_table[
            stance_table["aspect"]
            == aspect_name
        ]

    sparsity_rows.append(
        {
            "aspect": aspect_name,
            "records": len(subset),
            "records_with_decisive_evidence": int(
                (
                    subset[
                        "decisive_evidence_count"
                    ]
                    > 0
                ).sum()
            ),
            "decisive_evidence_coverage": float(
                (
                    subset[
                        "decisive_evidence_count"
                    ]
                    > 0
                ).mean()
            ),
            "mean_decisive_evidence_count": float(
                subset[
                    "decisive_evidence_count"
                ].mean()
            ),
            "median_decisive_evidence_count": float(
                subset[
                    "decisive_evidence_count"
                ].median()
            ),
            "mean_retained_fraction_of_32": float(
                (
                    subset[
                        "decisive_evidence_count"
                    ]
                    / 32.0
                ).mean()
            ),
        }
    )

sparsity_audit = pd.DataFrame(
    sparsity_rows
)


# -----------------------------------------------------------------------------
# Save results and manifest
# -----------------------------------------------------------------------------

main_results = pd.DataFrame(
    main_rows
)

per_class_results = pd.DataFrame(
    per_class_rows
)

selective_results = pd.DataFrame(
    selective_rows
)

main_results.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_main_results.csv",
    index=False,
)

per_class_results.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_per_class_results.csv",
    index=False,
)

selective_results.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_selective_results.csv",
    index=False,
)

all_predictions.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_test_predictions.csv",
    index=False,
)

split_distribution.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_split_distribution.csv",
    index=False,
)

bootstrap_results.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_bootstrap_samples.csv",
    index=False,
)

bootstrap_summary.to_csv(
    STANCE_OUTPUT_DIR
    / "stance_bootstrap_summary.csv",
    index=False,
)

sparsity_audit.to_csv(
    STANCE_OUTPUT_DIR
    / "decisive_evidence_sparsity_audit.csv",
    index=False,
)

manifest = {
    "status": "exploratory_silver_label_evaluation",
    "records": int(len(stance_table)),
    "unique_annotation_ids": int(
        stance_table["annotation_id"].nunique()
    ),
    "unique_windows": int(
        stance_table["window_id"].nunique()
    ),
    "classes": STANCE_CLASSES,
    "aspects": ASPECT_ORDER,
    "split_policy": (
        "Original leakage-aware train/validation/test partition "
        "recovered from the hidden provenance key."
    ),
    "validation_policy": (
        "Regularization and temperature selected using validation only."
    ),
    "test_policy": (
        "The test labels are evaluated once after model selection."
    ),
    "context_policy": (
        "Training weights are halved for insufficient-context records; "
        "test metrics are reported both with and without them."
    ),
    "label_status": (
        "AI-assisted, single-reviewer-resolved silver labels; "
        "not an independent human gold standard."
    ),
    "resolved_label_path": str(
        RESOLVED_LABEL_PATH
    ),
    "frozen_representation_path": str(
        REPRESENTATION_PATH
    ),
    "seed": STANCE_SEED,
    "bootstrap_repetitions": (
        BOOTSTRAP_REPETITIONS
    ),
}

with open(
    STANCE_OUTPUT_DIR
    / "stance_evaluation_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


# -----------------------------------------------------------------------------
# Display publication-facing summaries
# -----------------------------------------------------------------------------

print("=" * 80)
print("DAPT2020-MAST SILVER-LABEL THREAT-STANCE EVALUATION")
print("=" * 80)

print("\nResolved labels:")
print(RESOLVED_LABEL_PATH)

print("\nLabel status:")
print(manifest["label_status"])

print("\nLeakage-aware split distribution:")
display(
    split_distribution.pivot_table(
        index="split",
        columns="stance",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
)

print("\nPrimary and baseline results:")
display(
    main_results[
        [
            "representation",
            "evaluation_set",
            "samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "mcc",
            "macro_ovr_auroc",
            "nll",
            "brier",
            "ece",
        ]
    ]
    .sort_values(
        [
            "evaluation_set",
            "macro_f1",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(6)
)

print("\nPer-class test performance:")
display(
    per_class_results[
        per_class_results[
            "evaluation_set"
        ] == "all_test"
    ]
    .round(6)
)

print("\nValidation-selected abstention results:")
display(
    selective_results[
        [
            "representation",
            "target_validation_coverage",
            "confidence_threshold",
            "test_coverage",
            "macro_f1",
            "mcc",
            "ece",
        ]
    ].round(6)
)

print("\nAM-JEPA group-bootstrap confidence intervals:")
display(
    bootstrap_summary.round(6)
)

print("\nDecisive-evidence sparsity audit:")
display(
    sparsity_audit.round(6)
)

print("\nSaved stance-evaluation results to:")
print(STANCE_OUTPUT_DIR)

print(
    "\nCell 26 completed successfully. "
    "Interpret these results as exploratory silver-label evidence only; "
    "do not describe them as human gold-standard performance."
)
