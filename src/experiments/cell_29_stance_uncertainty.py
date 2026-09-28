# =============================================================================
# CELL 29: FINAL SPARSE STANCE, CALIBRATION, AND ABSTENTION EVALUATION
# =============================================================================

from pathlib import Path
import gc
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


# -----------------------------------------------------------------------------
# Required objects from Cells 26--28
# -----------------------------------------------------------------------------

required_objects = [
    "DATA_DIR",
    "AMJEPA_DIR",
    "STANCE_CLASSES",
    "ASPECT_ORDER",
    "C_GRID",
    "stance_table",
    "y_stance",
    "sample_weights",
    "split_masks",
    "frozen_representations",
    "aspect_one_hot",
    "window_to_tensor_index",
    "attention_mask_stance",
    "feature_sets",
    "membership",
    "membership_groups",
    "aspect_to_index_selector",
    "SELECTOR_SEEDS",
    "EVIDENCE_BUDGETS",
    "DUAL_FEATURE_SETS",
    "DUAL_OUTPUT_DIR",
    "DUAL_CHANNELS",
    "DUAL_HIDDEN_DIMENSION",
    "DualAspectConditionedSelector",
    "predict_dual_scores",
    "align_probabilities",
    "fit_temperature",
    "apply_temperature",
    "expected_calibration_error",
    "multiclass_brier",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        "Missing objects required by Cell 29:\n"
        + "\n".join(missing_objects)
        + "\n\nRun Cells 26, 27, and 28 first."
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FINAL_OUTPUT_DIR = (
    Path(AMJEPA_DIR)
    / "silver_final_stance"
)

FINAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_SEEDS = list(SELECTOR_SEEDS)
FINAL_BUDGETS = sorted(
    {
        int(value)
        for value in EVIDENCE_BUDGETS
    }
)

FINAL_TARGET_COVERAGES = [
    1.00,
    0.90,
    0.80,
    0.70,
]

FINAL_METHODS = [
    "dense_am_jepa",
    "dual_text_only",
    "dual_full_multimodal",
]

DEEPSEEK_TARGET_COVERAGE = 0.80
FINAL_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def set_final_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_final_probe(C_value, seed):
    return Pipeline(
        steps=[
            (
                "standardizer",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(C_value),
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=10000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def calculate_final_metrics(labels, probabilities):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = probabilities.argmax(
        axis=1
    )

    one_hot = label_binarize(
        labels,
        classes=np.arange(
            len(STANCE_CLASSES)
        ),
    )

    result = {
        "samples": int(len(labels)),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "nll": float(
            log_loss(
                labels,
                probabilities,
                labels=np.arange(
                    len(STANCE_CLASSES)
                ),
            )
        ),
        "brier": float(
            multiclass_brier(
                labels,
                probabilities,
            )
        ),
        "ece": float(
            expected_calibration_error(
                labels,
                probabilities,
            )
        ),
    }

    try:
        result["macro_ovr_auroc"] = float(
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

    try:
        result["macro_ovr_auprc"] = float(
            average_precision_score(
                one_hot,
                probabilities,
                average="macro",
            )
        )
    except ValueError:
        result["macro_ovr_auprc"] = np.nan

    return result


def normalized_entropy(probabilities):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            np.clip(
                probabilities,
                1e-12,
                1.0,
            )
        ),
        axis=1,
    )

    return entropy / math.log(
        probabilities.shape[1]
    )


def confidence_threshold_for_coverage(
    validation_confidence,
    target_coverage,
):
    if target_coverage >= 1.0:
        return 0.0

    return float(
        np.quantile(
            validation_confidence,
            1.0 - target_coverage,
            method="lower",
        )
    )


# -----------------------------------------------------------------------------
# Construct candidate events for every stance-annotation unit
# -----------------------------------------------------------------------------

all_event_rows = []

for record_index, record in stance_table.reset_index(
    drop=True
).iterrows():
    window_id = str(
        record["window_id"]
    )

    if window_id not in membership_groups:
        raise ValueError(
            f"No membership rows were found for {window_id}."
        )

    tensor_index = int(
        window_to_tensor_index[
            window_id
        ]
    )

    group = membership_groups[
        window_id
    ]

    evidence_by_position = dict(
        zip(
            group["position"].astype(int),
            group["evidence_id"].astype(str),
        )
    )

    valid_positions = np.where(
        attention_mask_stance[
            tensor_index
        ]
    )[0]

    for position in valid_positions:
        if int(position) not in evidence_by_position:
            raise ValueError(
                "A valid tensor position has no evidence ID: "
                f"{window_id}, position={position}."
            )

        all_event_rows.append(
            {
                "record_index": int(record_index),
                "annotation_id": record[
                    "annotation_id"
                ],
                "window_id": window_id,
                "split": record["split"],
                "aspect": record["aspect"],
                "aspect_index": int(
                    aspect_to_index_selector[
                        record["aspect"]
                    ]
                ),
                "tensor_index": tensor_index,
                "position": int(position),
                "evidence_id": evidence_by_position[
                    int(position)
                ],
            }
        )

all_stance_events = pd.DataFrame(
    all_event_rows
)

if all_stance_events.empty:
    raise ValueError(
        "No candidate events were constructed for Cell 29."
    )

record_event_counts = (
    all_stance_events
    .groupby("record_index")
    .size()
)

if len(record_event_counts) != len(stance_table):
    raise AssertionError(
        "At least one stance record has no candidate events."
    )

if (
    all_stance_events
    .groupby("window_id")["split"]
    .nunique()
    > 1
).any():
    raise AssertionError(
        "A window crosses the fixed train/validation/test partitions."
    )


# -----------------------------------------------------------------------------
# Apply frozen dual-selector checkpoints to every annotation unit
# -----------------------------------------------------------------------------

event_feature_cache = {}
selector_score_cache = {}

event_tensor_indices = all_stance_events[
    "tensor_index"
].to_numpy(dtype=np.int64)

event_positions = all_stance_events[
    "position"
].to_numpy(dtype=np.int64)

event_aspects = all_stance_events[
    "aspect_index"
].to_numpy(dtype=np.int64)

dummy_channels = np.zeros(
    len(all_stance_events),
    dtype=np.int64,
)

for feature_name in DUAL_FEATURE_SETS:
    raw_event_features = feature_sets[
        feature_name
    ][
        event_tensor_indices,
        event_positions,
    ].astype(np.float32)

    event_feature_cache[
        feature_name
    ] = raw_event_features

    for seed in FINAL_SEEDS:
        checkpoint_path = (
            Path(DUAL_OUTPUT_DIR)
            / (
                f"dual_selector_{feature_name}_"
                f"seed_{seed}.pt"
            )
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing Cell 28 checkpoint: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=FINAL_DEVICE,
            weights_only=False,
        )

        scaler_mean = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32,
        )

        scaler_scale = np.asarray(
            checkpoint["scaler_scale"],
            dtype=np.float32,
        )

        scaler_scale = np.where(
            scaler_scale > 0,
            scaler_scale,
            1.0,
        )

        scaled_event_features = (
            (
                raw_event_features
                - scaler_mean
            )
            / scaler_scale
        ).astype(np.float32)

        model = DualAspectConditionedSelector(
            input_dimension=int(
                checkpoint[
                    "input_dimension"
                ]
            ),
            number_of_aspects=len(
                ASPECT_ORDER
            ),
            hidden_dimension=int(
                checkpoint[
                    "hidden_dimension"
                ]
            ),
        ).to(FINAL_DEVICE)

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        _, all_logits = predict_dual_scores(
            model,
            scaled_event_features,
            event_aspects,
            dummy_channels,
        )

        all_probabilities = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    all_logits,
                    -30.0,
                    30.0,
                )
            )
        )

        selector_score_cache[
            (
                feature_name,
                int(seed),
            )
        ] = all_probabilities.astype(
            np.float32
        )

        del model
        del checkpoint

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Create sparse E+/E- stance representations
# -----------------------------------------------------------------------------

dense_context = np.concatenate(
    [
        frozen_representations[
            stance_table[
                "tensor_index"
            ].to_numpy(dtype=np.int64)
        ],
        aspect_one_hot,
    ],
    axis=1,
).astype(np.float32)

sparse_feature_cache = {}
selection_rows = []


def build_sparse_stance_features(
    feature_name,
    seed,
    budget,
    save_selections=False,
):
    cache_key = (
        feature_name,
        int(seed),
        int(budget),
    )

    if (
        cache_key in sparse_feature_cache
        and not save_selections
    ):
        return sparse_feature_cache[
            cache_key
        ]

    raw_features = event_feature_cache[
        feature_name
    ]

    scores = selector_score_cache[
        (
            feature_name,
            int(seed),
        )
    ]

    feature_rows = []

    for record_index in range(
        len(stance_table)
    ):
        event_indices = np.where(
            all_stance_events[
                "record_index"
            ].to_numpy(dtype=np.int64)
            == record_index
        )[0]

        local_features = raw_features[
            event_indices
        ]

        local_scores = scores[
            event_indices
        ]

        number_to_keep = min(
            int(budget),
            len(event_indices),
        )

        supporting_order = np.argsort(
            -local_scores[:, 0],
            kind="stable",
        )[:number_to_keep]

        contradicting_order = np.argsort(
            -local_scores[:, 1],
            kind="stable",
        )[:number_to_keep]

        supporting_features = local_features[
            supporting_order
        ].mean(axis=0)

        contradicting_features = local_features[
            contradicting_order
        ].mean(axis=0)

        supporting_scores = local_scores[
            supporting_order,
            0,
        ]

        contradicting_scores = local_scores[
            contradicting_order,
            1,
        ]

        overlap = len(
            set(
                supporting_order.tolist()
            )
            & set(
                contradicting_order.tolist()
            )
        ) / max(
            number_to_keep,
            1,
        )

        score_features = np.asarray(
            [
                supporting_scores.mean(),
                supporting_scores.max(),
                supporting_scores.std(),
                contradicting_scores.mean(),
                contradicting_scores.max(),
                contradicting_scores.std(),
                supporting_scores.mean()
                - contradicting_scores.mean(),
                overlap,
                number_to_keep
                / max(
                    len(event_indices),
                    1,
                ),
            ],
            dtype=np.float32,
        )

        feature_rows.append(
            np.concatenate(
                [
                    dense_context[
                        record_index
                    ],
                    supporting_features,
                    contradicting_features,
                    supporting_features
                    - contradicting_features,
                    score_features,
                ]
            )
        )

        if save_selections:
            record_events = all_stance_events.iloc[
                event_indices
            ]

            for channel_index, (
                channel_name,
                selected_order,
            ) in enumerate(
                [
                    (
                        "SUPPORTING",
                        supporting_order,
                    ),
                    (
                        "CONTRADICTING",
                        contradicting_order,
                    ),
                ]
            ):
                for rank, local_index in enumerate(
                    selected_order,
                    start=1,
                ):
                    selected_event = record_events.iloc[
                        int(local_index)
                    ]

                    selection_rows.append(
                        {
                            "method": (
                                "dual_"
                                + feature_name
                            ),
                            "feature_set": feature_name,
                            "seed": int(seed),
                            "budget": int(budget),
                            "record_index": int(
                                record_index
                            ),
                            "annotation_id": selected_event[
                                "annotation_id"
                            ],
                            "window_id": selected_event[
                                "window_id"
                            ],
                            "split": selected_event[
                                "split"
                            ],
                            "aspect": selected_event[
                                "aspect"
                            ],
                            "channel": channel_name,
                            "rank": int(rank),
                            "position": int(
                                selected_event[
                                    "position"
                                ]
                            ),
                            "evidence_id": selected_event[
                                "evidence_id"
                            ],
                            "selector_score": float(
                                local_scores[
                                    int(local_index),
                                    channel_index,
                                ]
                            ),
                        }
                    )

    features = np.stack(
        feature_rows
    ).astype(np.float32)

    sparse_feature_cache[
        cache_key
    ] = features

    return features


# -----------------------------------------------------------------------------
# Validation-only model and evidence-budget selection
# -----------------------------------------------------------------------------

train_mask = split_masks["train"]
validation_mask = split_masks[
    "validation"
]
test_mask = split_masks["test"]

validation_search_rows = []
selected_configuration_rows = []
test_metric_rows = []
per_class_rows = []
selective_rows = []
prediction_parts = []

for method in FINAL_METHODS:
    if method == "dense_am_jepa":
        feature_name = None
        candidate_budgets = [0]
    else:
        feature_name = method.replace(
            "dual_",
            "",
            1,
        )

        if feature_name not in DUAL_FEATURE_SETS:
            raise ValueError(
                f"Unsupported dual feature set: {feature_name}"
            )

        candidate_budgets = FINAL_BUDGETS

    for seed in FINAL_SEEDS:
        set_final_seed(seed)
        seed_search_rows = []

        for budget in candidate_budgets:
            if method == "dense_am_jepa":
                features = dense_context
            else:
                features = build_sparse_stance_features(
                    feature_name,
                    seed,
                    budget,
                )

            for C_value in C_GRID:
                probe = build_final_probe(
                    C_value,
                    seed,
                )

                probe.fit(
                    features[train_mask],
                    y_stance[train_mask],
                    classifier__sample_weight=
                        sample_weights[
                            train_mask
                        ],
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
                    calculate_final_metrics(
                        y_stance[
                            validation_mask
                        ],
                        validation_probabilities,
                    )
                )

                search_row = {
                    "method": method,
                    "feature_set": (
                        "frozen_am_jepa"
                        if feature_name is None
                        else feature_name
                    ),
                    "seed": int(seed),
                    "budget": int(budget),
                    "C": float(C_value),
                    **validation_metrics,
                }

                validation_search_rows.append(
                    search_row
                )

                seed_search_rows.append(
                    search_row
                )

        best_validation = (
            pd.DataFrame(
                seed_search_rows
            )
            .sort_values(
                by=[
                    "macro_f1",
                    "mcc",
                    "balanced_accuracy",
                    "macro_ovr_auroc",
                    "nll",
                    "budget",
                    "C",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        selected_budget = int(
            best_validation[
                "budget"
            ]
        )

        selected_C = float(
            best_validation["C"]
        )

        if method == "dense_am_jepa":
            selected_features = dense_context
        else:
            selected_features = (
                build_sparse_stance_features(
                    feature_name,
                    seed,
                    selected_budget,
                    save_selections=True,
                )
            )

        final_probe = build_final_probe(
            selected_C,
            seed,
        )

        final_probe.fit(
            selected_features[
                train_mask
            ],
            y_stance[train_mask],
            classifier__sample_weight=
                sample_weights[train_mask],
        )

        validation_probabilities_raw = (
            align_probabilities(
                final_probe.predict_proba(
                    selected_features[
                        validation_mask
                    ]
                ),
                final_probe.named_steps[
                    "classifier"
                ].classes_,
            )
        )

        temperature = fit_temperature(
            y_stance[validation_mask],
            validation_probabilities_raw,
        )

        validation_probabilities = (
            apply_temperature(
                validation_probabilities_raw,
                temperature,
            )
        )

        test_probabilities_raw = (
            align_probabilities(
                final_probe.predict_proba(
                    selected_features[
                        test_mask
                    ]
                ),
                final_probe.named_steps[
                    "classifier"
                ].classes_,
            )
        )

        test_probabilities = apply_temperature(
            test_probabilities_raw,
            temperature,
        )

        calibrated_validation_metrics = (
            calculate_final_metrics(
                y_stance[validation_mask],
                validation_probabilities,
            )
        )

        selected_configuration_rows.append(
            {
                "method": method,
                "feature_set": (
                    "frozen_am_jepa"
                    if feature_name is None
                    else feature_name
                ),
                "seed": int(seed),
                "selected_budget": int(
                    selected_budget
                ),
                "selected_C": selected_C,
                "temperature": temperature,
                "validation_macro_f1": (
                    calibrated_validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_mcc": (
                    calibrated_validation_metrics[
                        "mcc"
                    ]
                ),
                "validation_macro_ovr_auroc": (
                    calibrated_validation_metrics[
                        "macro_ovr_auroc"
                    ]
                ),
                "validation_macro_ovr_auprc": (
                    calibrated_validation_metrics[
                        "macro_ovr_auprc"
                    ]
                ),
                "validation_ece": (
                    calibrated_validation_metrics[
                        "ece"
                    ]
                ),
            }
        )

        evaluation_masks = {
            "all_test": np.ones(
                int(test_mask.sum()),
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

            metrics = calculate_final_metrics(
                test_labels[local_mask],
                test_probabilities[local_mask],
            )

            test_metric_rows.append(
                {
                    "method": method,
                    "feature_set": (
                        "frozen_am_jepa"
                        if feature_name is None
                        else feature_name
                    ),
                    "seed": int(seed),
                    "selected_budget": int(
                        selected_budget
                    ),
                    "selected_C": selected_C,
                    "temperature": temperature,
                    "evaluation_set": evaluation_set,
                    **metrics,
                }
            )

            predictions = test_probabilities[
                local_mask
            ].argmax(axis=1)

            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    test_labels[local_mask],
                    predictions,
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
                        "method": method,
                        "seed": int(seed),
                        "evaluation_set": evaluation_set,
                        "stance": class_name,
                        "precision": float(
                            precision[
                                class_index
                            ]
                        ),
                        "recall": float(
                            recall[
                                class_index
                            ]
                        ),
                        "f1": float(
                            f1[class_index]
                        ),
                        "support": int(
                            support[
                                class_index
                            ]
                        ),
                    }
                )

        validation_confidence = (
            validation_probabilities.max(
                axis=1
            )
        )

        test_confidence = (
            test_probabilities.max(
                axis=1
            )
        )

        for target_coverage in (
            FINAL_TARGET_COVERAGES
        ):
            threshold = (
                confidence_threshold_for_coverage(
                    validation_confidence,
                    target_coverage,
                )
            )

            retained = (
                test_confidence
                >= threshold
            )

            if retained.sum() == 0:
                continue

            retained_metrics = (
                calculate_final_metrics(
                    test_labels[retained],
                    test_probabilities[
                        retained
                    ],
                )
            )

            selective_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "selected_budget": int(
                        selected_budget
                    ),
                    "target_validation_coverage": float(
                        target_coverage
                    ),
                    "confidence_threshold": threshold,
                    "test_coverage": float(
                        retained.mean()
                    ),
                    "abstention_rate": float(
                        1.0
                        - retained.mean()
                    ),
                    "selective_risk": float(
                        1.0
                        - retained_metrics[
                            "accuracy"
                        ]
                    ),
                    **retained_metrics,
                }
            )

        for split_name, local_mask, probabilities in [
            (
                "validation",
                validation_mask,
                validation_probabilities,
            ),
            (
                "test",
                test_mask,
                test_probabilities,
            ),
        ]:
            prediction_table = stance_table.loc[
                local_mask,
                [
                    "annotation_id",
                    "window_id",
                    "aspect",
                    "stance",
                    "insufficient_context",
                    "confidence",
                ],
            ].copy()

            prediction_table["split"] = (
                split_name
            )
            prediction_table["method"] = method
            prediction_table["seed"] = int(
                seed
            )
            prediction_table[
                "selected_budget"
            ] = int(selected_budget)
            prediction_table[
                "predicted_stance"
            ] = [
                STANCE_CLASSES[index]
                for index in probabilities.argmax(
                    axis=1
                )
            ]
            prediction_table[
                "predictive_confidence"
            ] = probabilities.max(axis=1)
            prediction_table[
                "predictive_entropy"
            ] = normalized_entropy(
                probabilities
            )

            for class_index, class_name in enumerate(
                STANCE_CLASSES
            ):
                prediction_table[
                    "probability_"
                    + class_name.lower()
                ] = probabilities[
                    :,
                    class_index,
                ]

            prediction_parts.append(
                prediction_table
            )

        del final_probe


# -----------------------------------------------------------------------------
# Assemble and aggregate five-seed results
# -----------------------------------------------------------------------------

validation_search = pd.DataFrame(
    validation_search_rows
)

selected_configurations = pd.DataFrame(
    selected_configuration_rows
)

test_seed_results = pd.DataFrame(
    test_metric_rows
)

per_class_results = pd.DataFrame(
    per_class_rows
)

selective_results = pd.DataFrame(
    selective_rows
)

prediction_results = pd.concat(
    prediction_parts,
    ignore_index=True,
)

aggregate_metric_names = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "macro_ovr_auroc",
    "macro_ovr_auprc",
    "nll",
    "brier",
    "ece",
]

aggregate_rows = []

for (
    method,
    evaluation_set,
), group in test_seed_results.groupby(
    [
        "method",
        "evaluation_set",
    ],
    sort=False,
):
    row = {
        "method": method,
        "evaluation_set": evaluation_set,
        "seeds": int(
            group["seed"].nunique()
        ),
        "selected_budget_mode": int(
            group[
                "selected_budget"
            ].mode().iloc[0]
        ),
    }

    for metric in aggregate_metric_names:
        values = group[metric].to_numpy(
            dtype=np.float64
        )

        row[
            metric + "_mean"
        ] = float(values.mean())

        row[
            metric + "_sd"
        ] = float(
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )

    aggregate_rows.append(row)

aggregate_results = pd.DataFrame(
    aggregate_rows
)

selective_aggregate_rows = []

for (
    method,
    target_coverage,
), group in selective_results.groupby(
    [
        "method",
        "target_validation_coverage",
    ],
    sort=False,
):
    row = {
        "method": method,
        "target_validation_coverage": float(
            target_coverage
        ),
        "seeds": int(
            group["seed"].nunique()
        ),
    }

    for metric in [
        "test_coverage",
        "abstention_rate",
        "accuracy",
        "macro_f1",
        "mcc",
        "selective_risk",
        "ece",
    ]:
        values = group[metric].to_numpy(
            dtype=np.float64
        )

        row[metric + "_mean"] = float(
            values.mean()
        )

        row[metric + "_sd"] = float(
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )

    selective_aggregate_rows.append(row)

selective_aggregate = pd.DataFrame(
    selective_aggregate_rows
)


# -----------------------------------------------------------------------------
# Select the DeepSeek routing configuration using validation results only
# -----------------------------------------------------------------------------

dual_validation_summary = (
    selected_configurations[
        selected_configurations[
            "method"
        ].str.startswith("dual_")
    ]
    .groupby("method", as_index=False)
    .agg(
        validation_macro_f1_mean=(
            "validation_macro_f1",
            "mean",
        ),
        validation_mcc_mean=(
            "validation_mcc",
            "mean",
        ),
        validation_auroc_mean=(
            "validation_macro_ovr_auroc",
            "mean",
        ),
        validation_ece_mean=(
            "validation_ece",
            "mean",
        ),
    )
)

primary_validation_row = (
    dual_validation_summary
    .sort_values(
        by=[
            "validation_macro_f1_mean",
            "validation_mcc_mean",
            "validation_auroc_mean",
            "validation_ece_mean",
            "method",
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

primary_method = str(
    primary_validation_row[
        "method"
    ]
)

primary_feature_set = primary_method.replace(
    "dual_",
    "",
    1,
)

primary_seed_configs = selected_configurations[
    selected_configurations["method"]
    == primary_method
].copy()

primary_budget = int(
    primary_seed_configs[
        "selected_budget"
    ].mode().iloc[0]
)


def build_ensemble_table(split_name):
    subset = prediction_results[
        (
            prediction_results["method"]
            == primary_method
        )
        & (
            prediction_results["split"]
            == split_name
        )
    ].copy()

    probability_columns = [
        "probability_"
        + class_name.lower()
        for class_name in STANCE_CLASSES
    ]

    metadata_columns = [
        "annotation_id",
        "window_id",
        "aspect",
        "stance",
        "insufficient_context",
        "confidence",
    ]

    metadata = (
        subset[
            metadata_columns
        ]
        .drop_duplicates(
            "annotation_id"
        )
        .set_index("annotation_id")
    )

    probabilities = (
        subset
        .groupby("annotation_id")[
            probability_columns
        ]
        .mean()
    )

    ensemble = metadata.join(
        probabilities,
        how="inner",
    ).reset_index()

    probability_array = ensemble[
        probability_columns
    ].to_numpy(dtype=np.float64)

    ensemble[
        "predicted_stance"
    ] = [
        STANCE_CLASSES[index]
        for index in probability_array.argmax(
            axis=1
        )
    ]

    ensemble[
        "predictive_confidence"
    ] = probability_array.max(axis=1)

    ensemble[
        "predictive_entropy"
    ] = normalized_entropy(
        probability_array
    )

    ensemble["split"] = split_name

    return ensemble


validation_ensemble = build_ensemble_table(
    "validation"
)

test_ensemble = build_ensemble_table(
    "test"
)

deepseek_threshold = (
    confidence_threshold_for_coverage(
        validation_ensemble[
            "predictive_confidence"
        ].to_numpy(dtype=np.float64),
        DEEPSEEK_TARGET_COVERAGE,
    )
)

test_ensemble["route_to_deepseek"] = (
    test_ensemble[
        "predictive_confidence"
    ]
    < deepseek_threshold
)


# -----------------------------------------------------------------------------
# Build consensus sparse evidence IDs for the label-blind DeepSeek queue
# -----------------------------------------------------------------------------

primary_scores = np.mean(
    np.stack(
        [
            selector_score_cache[
                (
                    primary_feature_set,
                    int(seed),
                )
            ]
            for seed in FINAL_SEEDS
        ],
        axis=0,
    ),
    axis=0,
)

consensus_selection_rows = []

for record_index in range(
    len(stance_table)
):
    event_indices = np.where(
        all_stance_events[
            "record_index"
        ].to_numpy(dtype=np.int64)
        == record_index
    )[0]

    local_scores = primary_scores[
        event_indices
    ]

    record_events = all_stance_events.iloc[
        event_indices
    ]

    number_to_keep = min(
        primary_budget,
        len(event_indices),
    )

    for channel_index, channel_name in enumerate(
        DUAL_CHANNELS
    ):
        order = np.argsort(
            -local_scores[
                :,
                channel_index,
            ],
            kind="stable",
        )[:number_to_keep]

        for rank, local_index in enumerate(
            order,
            start=1,
        ):
            event = record_events.iloc[
                int(local_index)
            ]

            consensus_selection_rows.append(
                {
                    "record_index": int(
                        record_index
                    ),
                    "annotation_id": event[
                        "annotation_id"
                    ],
                    "window_id": event[
                        "window_id"
                    ],
                    "split": event["split"],
                    "aspect": event["aspect"],
                    "channel": channel_name,
                    "rank": int(rank),
                    "position": int(
                        event["position"]
                    ),
                    "evidence_id": event[
                        "evidence_id"
                    ],
                    "selector_score": float(
                        local_scores[
                            int(local_index),
                            channel_index,
                        ]
                    ),
                }
            )

consensus_selections = pd.DataFrame(
    consensus_selection_rows
)

test_consensus = consensus_selections[
    consensus_selections["split"]
    == "test"
].copy()

evidence_lists = (
    test_consensus
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
    .apply(
        lambda values: ";".join(
            values.astype(str)
        )
    )
    .unstack("channel")
    .reset_index()
    .rename(
        columns={
            "SUPPORTING": (
                "selected_supporting_evidence_ids"
            ),
            "CONTRADICTING": (
                "selected_contradicting_evidence_ids"
            ),
        }
    )
)

deepseek_blind_queue = (
    test_ensemble[
        test_ensemble[
            "route_to_deepseek"
        ]
    ][
        [
            "annotation_id",
            "window_id",
            "aspect",
            "predicted_stance",
            "predictive_confidence",
            "predictive_entropy",
        ]
    ]
    .merge(
        evidence_lists,
        on="annotation_id",
        how="left",
        validate="one_to_one",
    )
)

deepseek_blind_queue.insert(
    0,
    "request_id",
    [
        f"DEEPSEEK-{index:04d}"
        for index in range(
            1,
            len(deepseek_blind_queue) + 1,
        )
    ],
)

# Deliberately exclude the reference stance and all correctness fields.
for forbidden_column in [
    "stance",
    "gold_stance",
    "correct",
    "label",
]:
    if forbidden_column in deepseek_blind_queue.columns:
        raise AssertionError(
            "The label-blind DeepSeek queue contains a forbidden field: "
            f"{forbidden_column}"
        )

deepseek_evaluation_key = (
    test_ensemble[
        test_ensemble[
            "route_to_deepseek"
        ]
    ][
        [
            "annotation_id",
            "stance",
            "insufficient_context",
            "confidence",
        ]
    ]
    .merge(
        deepseek_blind_queue[
            [
                "request_id",
                "annotation_id",
            ]
        ],
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )
)


# -----------------------------------------------------------------------------
# Save all outputs
# -----------------------------------------------------------------------------

validation_search.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_validation_search.csv",
    index=False,
)

selected_configurations.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_selected_configurations.csv",
    index=False,
)

test_seed_results.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_test_seed_results.csv",
    index=False,
)

aggregate_results.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_test_aggregate.csv",
    index=False,
)

per_class_results.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_per_class_results.csv",
    index=False,
)

selective_results.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_selective_seed_results.csv",
    index=False,
)

selective_aggregate.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_selective_aggregate.csv",
    index=False,
)

prediction_results.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_validation_test_predictions.csv",
    index=False,
)

pd.DataFrame(selection_rows).to_csv(
    FINAL_OUTPUT_DIR
    / "stance_selected_evidence_by_seed.csv",
    index=False,
)

consensus_selections.to_csv(
    FINAL_OUTPUT_DIR
    / "stance_consensus_selected_evidence.csv",
    index=False,
)

validation_ensemble.to_csv(
    FINAL_OUTPUT_DIR
    / "primary_validation_ensemble.csv",
    index=False,
)

test_ensemble.to_csv(
    FINAL_OUTPUT_DIR
    / "primary_test_ensemble_with_reference.csv",
    index=False,
)

deepseek_blind_queue.to_csv(
    FINAL_OUTPUT_DIR
    / "deepseek_blind_request_queue.csv",
    index=False,
)

deepseek_evaluation_key.to_csv(
    FINAL_OUTPUT_DIR
    / "deepseek_evaluation_key_DO_NOT_LOAD_BEFORE_RESPONSES.csv",
    index=False,
)

manifest = {
    "study_type": (
        "exploratory_ai_assisted_silver_label_evaluation"
    ),
    "stance_classes": STANCE_CLASSES,
    "aspect_order": ASPECT_ORDER,
    "seeds": FINAL_SEEDS,
    "candidate_budgets": FINAL_BUDGETS,
    "methods": FINAL_METHODS,
    "selection_policy": (
        "budget and C selected independently per seed using validation "
        "macro-F1, MCC, balanced accuracy, AUROC, NLL, then smaller "
        "budget and C"
    ),
    "calibration_policy": (
        "temperature fitted on validation predictions only"
    ),
    "abstention_policy": (
        "confidence thresholds selected on validation for fixed target "
        "coverage; test retained when confidence >= threshold"
    ),
    "primary_method_selection": (
        "dual method selected using five-seed validation means only"
    ),
    "primary_method": primary_method,
    "primary_feature_set": primary_feature_set,
    "primary_budget_mode": primary_budget,
    "deepseek_target_validation_coverage": (
        DEEPSEEK_TARGET_COVERAGE
    ),
    "deepseek_confidence_threshold": (
        deepseek_threshold
    ),
    "deepseek_requests": int(
        len(deepseek_blind_queue)
    ),
    "deepseek_queue_is_label_blind": True,
    "test_used_for_configuration_selection": False,
    "limitations": [
        "The 240-unit pilot uses AI-assisted single-reviewer silver labels.",
        "The rationale-supervised dual selectors use only 94 annotated units.",
        "Results are exploratory and are not independent human gold-standard estimates.",
    ],
}

with open(
    FINAL_OUTPUT_DIR
    / "stance_experiment_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


# -----------------------------------------------------------------------------
# Display concise primary results
# -----------------------------------------------------------------------------

print("=" * 80)
print("SPARSESENTINEL-APT FINAL STANCE AND UNCERTAINTY EVALUATION")
print("=" * 80)

print(
    "\nLabel status: AI-assisted single-reviewer silver pilot "
    "(not an independent human gold standard)."
)

print("\nValidation-selected configurations:")

display(
    selected_configurations[
        [
            "method",
            "seed",
            "selected_budget",
            "selected_C",
            "temperature",
            "validation_macro_f1",
            "validation_mcc",
            "validation_ece",
        ]
    ].round(6)
)

print("\nFive-seed test summary:")

display(
    aggregate_results[
        aggregate_results[
            "evaluation_set"
        ] == "all_test"
    ][
        [
            "method",
            "selected_budget_mode",
            "accuracy_mean",
            "accuracy_sd",
            "macro_f1_mean",
            "macro_f1_sd",
            "mcc_mean",
            "mcc_sd",
            "macro_ovr_auroc_mean",
            "macro_ovr_auprc_mean",
            "ece_mean",
            "brier_mean",
        ]
    ]
    .sort_values(
        "macro_f1_mean",
        ascending=False,
    )
    .round(6)
)

print("\nSelective prediction summary:")

display(
    selective_aggregate[
        [
            "method",
            "target_validation_coverage",
            "test_coverage_mean",
            "accuracy_mean",
            "macro_f1_mean",
            "mcc_mean",
            "selective_risk_mean",
            "ece_mean",
        ]
    ]
    .sort_values(
        [
            "method",
            "target_validation_coverage",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(6)
)

print("\nDeepSeek routing package:")
print("Primary method:", primary_method)
print("Evidence budget:", primary_budget)
print(
    "Validation-selected confidence threshold:",
    round(deepseek_threshold, 6),
)
print(
    "Test cases routed to DeepSeek:",
    len(deepseek_blind_queue),
    "/",
    len(test_ensemble),
)

print("\nSaved Cell 29 results to:")
print(FINAL_OUTPUT_DIR)

print(
    "\nCell 29 completed successfully. The DeepSeek request queue is "
    "label-blind; do not load the evaluation key until all API responses "
    "have been cached."
)

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
