# =============================================================================
# CELL 28: DUAL SUPPORTING/CONTRADICTING EVIDENCE SELECTORS
# =============================================================================

from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Required Cell 27 objects
# -----------------------------------------------------------------------------

required_objects = [
    "AMJEPA_DIR",
    "ASPECT_ORDER",
    "selector_records",
    "event_table",
    "feature_sets",
    "EVIDENCE_BUDGETS",
    "SELECTOR_SEEDS",
    "evaluate_rankings",
    "average_precision_from_ranking",
    "gather_event_features",
    "record_results",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        "Missing objects from Cell 27:\n"
        + "\n".join(missing_objects)
        + "\n\nRun Cells 26 and 27 before Cell 28."
    )

DUAL_OUTPUT_DIR = (
    Path(AMJEPA_DIR)
    / "silver_dual_selector"
)

DUAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DUAL_CHANNELS = [
    "SUPPORTING",
    "CONTRADICTING",
]

DUAL_FEATURE_SETS = [
    "text_only",
    "full_multimodal",
]

DUAL_HIDDEN_DIMENSION = 64
DUAL_MAXIMUM_EPOCHS = 300
DUAL_PATIENCE = 30
DUAL_BATCH_SIZE = 256
DUAL_LEARNING_RATE = 1e-3
DUAL_WEIGHT_DECAY = 1e-4

DEVICE_DUAL = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

channel_to_index = {
    channel: index
    for index, channel in enumerate(
        DUAL_CHANNELS
    )
}


def set_dual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Verify that the decisive-ID records support the dual-channel interpretation
# -----------------------------------------------------------------------------

observed_rationale_stances = set(
    selector_records["stance"]
)

unexpected_rationale_stances = sorted(
    observed_rationale_stances
    - set(DUAL_CHANNELS)
)

if unexpected_rationale_stances:
    raise ValueError(
        "Some decisive-evidence records have a neutral or unknown stance. "
        "They cannot be assigned automatically to E+ or E-. "
        f"Unexpected labels: {unexpected_rationale_stances}"
    )

expected_channel_counts = {
    "SUPPORTING": 56,
    "CONTRADICTING": 38,
}

observed_channel_counts = (
    selector_records["stance"]
    .value_counts()
    .to_dict()
)

if observed_channel_counts != expected_channel_counts:
    print(
        "Warning: rationale-channel counts differ from the audited pilot."
    )
    print("Expected:", expected_channel_counts)
    print("Observed:", observed_channel_counts)

channel_lookup = (
    selector_records
    .set_index("annotation_id")[
        "stance"
    ]
    .to_dict()
)

dual_event_table = event_table.copy()

dual_event_table["channel"] = (
    dual_event_table["annotation_id"]
    .map(channel_lookup)
)

if dual_event_table["channel"].isna().any():
    raise ValueError(
        "At least one event could not be assigned to a dual channel."
    )

dual_event_table["channel_index"] = (
    dual_event_table["channel"]
    .map(channel_to_index)
    .astype(np.int64)
)


# -----------------------------------------------------------------------------
# Split and channel audit
# -----------------------------------------------------------------------------

dual_audit_rows = []

for split_name in [
    "train",
    "validation",
    "test",
]:
    for channel_name in DUAL_CHANNELS:
        subset = dual_event_table[
            (
                dual_event_table["split"]
                == split_name
            )
            & (
                dual_event_table["channel"]
                == channel_name
            )
        ]

        records = int(
            subset["annotation_id"]
            .nunique()
        )

        if records == 0:
            raise ValueError(
                f"No {channel_name} rationale records occur in "
                f"the {split_name} partition."
            )

        dual_audit_rows.append(
            {
                "split": split_name,
                "channel": channel_name,
                "records": records,
                "unique_windows": int(
                    subset["window_id"]
                    .nunique()
                ),
                "candidate_events": int(
                    len(subset)
                ),
                "decisive_events": int(
                    subset["target"].sum()
                ),
                "positive_prevalence": float(
                    subset["target"].mean()
                ),
            }
        )

dual_split_audit = pd.DataFrame(
    dual_audit_rows
)


# -----------------------------------------------------------------------------
# Shared encoder with separate positive and negative scoring heads
# -----------------------------------------------------------------------------

class DualAspectConditionedSelector(nn.Module):
    def __init__(
        self,
        input_dimension,
        number_of_aspects,
        hidden_dimension,
    ):
        super().__init__()

        self.event_projection = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.aspect_embedding = nn.Embedding(
            number_of_aspects,
            hidden_dimension,
        )

        interaction_dimension = (
            hidden_dimension * 3
        )

        self.supporting_head = nn.Sequential(
            nn.Linear(
                interaction_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                hidden_dimension,
                1,
            ),
        )

        self.contradicting_head = nn.Sequential(
            nn.Linear(
                interaction_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                hidden_dimension,
                1,
            ),
        )

    def encode(
        self,
        event_features,
        aspect_indices,
    ):
        event_state = self.event_projection(
            event_features
        )

        aspect_state = self.aspect_embedding(
            aspect_indices
        )

        interaction = (
            event_state
            * aspect_state
        )

        return torch.cat(
            [
                event_state,
                aspect_state,
                interaction,
            ],
            dim=-1,
        )

    def forward_all_channels(
        self,
        event_features,
        aspect_indices,
    ):
        combined = self.encode(
            event_features,
            aspect_indices,
        )

        supporting_scores = (
            self.supporting_head(
                combined
            ).squeeze(-1)
        )

        contradicting_scores = (
            self.contradicting_head(
                combined
            ).squeeze(-1)
        )

        return torch.stack(
            [
                supporting_scores,
                contradicting_scores,
            ],
            dim=1,
        )

    def forward(
        self,
        event_features,
        aspect_indices,
        channel_indices,
    ):
        all_scores = self.forward_all_channels(
            event_features,
            aspect_indices,
        )

        return all_scores.gather(
            1,
            channel_indices.unsqueeze(1),
        ).squeeze(1)


# -----------------------------------------------------------------------------
# Prediction and validation helpers
# -----------------------------------------------------------------------------

def predict_dual_scores(
    model,
    features,
    aspects,
    channels,
    batch_size=1024,
):
    model.eval()
    selected_parts = []
    all_channel_parts = []

    with torch.no_grad():
        for start in range(
            0,
            len(features),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(features),
            )

            batch_features = torch.from_numpy(
                features[start:end]
            ).float().to(DEVICE_DUAL)

            batch_aspects = torch.from_numpy(
                aspects[start:end]
            ).long().to(DEVICE_DUAL)

            batch_channels = torch.from_numpy(
                channels[start:end]
            ).long().to(DEVICE_DUAL)

            all_scores = model.forward_all_channels(
                batch_features,
                batch_aspects,
            )

            selected_scores = all_scores.gather(
                1,
                batch_channels.unsqueeze(1),
            ).squeeze(1)

            selected_parts.append(
                selected_scores.cpu().numpy()
            )

            all_channel_parts.append(
                all_scores.cpu().numpy()
            )

    return (
        np.concatenate(selected_parts),
        np.concatenate(all_channel_parts),
    )


def calculate_channel_map(
    table,
    score_column,
):
    rows = []

    for channel_name in DUAL_CHANNELS:
        channel_table = table[
            table["channel"]
            == channel_name
        ]

        values = []

        for _, group in channel_table.groupby(
            "annotation_id",
            sort=False,
        ):
            values.append(
                average_precision_from_ranking(
                    group["target"].to_numpy(
                        dtype=np.int64
                    ),
                    group[score_column].to_numpy(
                        dtype=np.float64
                    ),
                )
            )

        rows.append(
            {
                "channel": channel_name,
                "records": int(
                    channel_table[
                        "annotation_id"
                    ].nunique()
                ),
                "map": float(
                    np.nanmean(values)
                ),
            }
        )

    result = pd.DataFrame(rows)

    return (
        result,
        float(result["map"].mean()),
    )


# -----------------------------------------------------------------------------
# Five-seed training for text and full multimodal dual selectors
# -----------------------------------------------------------------------------

dual_record_result_parts = []
dual_training_history_rows = []
dual_validation_rows = []
dual_event_score_parts = []

train_mask_dual = (
    dual_event_table["split"].to_numpy()
    == "train"
)

validation_mask_dual = (
    dual_event_table["split"].to_numpy()
    == "validation"
)

test_mask_dual = (
    dual_event_table["split"].to_numpy()
    == "test"
)

aspects_dual = dual_event_table[
    "aspect_index"
].to_numpy(dtype=np.int64)

channels_dual = dual_event_table[
    "channel_index"
].to_numpy(dtype=np.int64)

targets_dual = dual_event_table[
    "target"
].to_numpy(dtype=np.float32)

weights_dual = dual_event_table[
    "sample_weight"
].to_numpy(dtype=np.float32)

for feature_name in DUAL_FEATURE_SETS:
    raw_features = gather_event_features(
        feature_sets[feature_name],
        dual_event_table,
    ).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(
        raw_features[
            train_mask_dual
        ]
    )

    scaled_features = scaler.transform(
        raw_features
    ).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(
            scaled_features[
                train_mask_dual
            ]
        ),
        torch.from_numpy(
            aspects_dual[
                train_mask_dual
            ]
        ),
        torch.from_numpy(
            channels_dual[
                train_mask_dual
            ]
        ),
        torch.from_numpy(
            targets_dual[
                train_mask_dual
            ]
        ),
        torch.from_numpy(
            weights_dual[
                train_mask_dual
            ]
        ),
    )

    validation_table_base = dual_event_table[
        validation_mask_dual
    ].copy()

    test_table_base = dual_event_table[
        test_mask_dual
    ].copy()

    for seed in SELECTOR_SEEDS:
        set_dual_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=DUAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model = DualAspectConditionedSelector(
            input_dimension=
                scaled_features.shape[1],
            number_of_aspects=
                len(ASPECT_ORDER),
            hidden_dimension=
                DUAL_HIDDEN_DIMENSION,
        ).to(DEVICE_DUAL)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=DUAL_LEARNING_RATE,
            weight_decay=DUAL_WEIGHT_DECAY,
        )

        loss_function = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        best_macro_map = -np.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(
            1,
            DUAL_MAXIMUM_EPOCHS + 1,
        ):
            model.train()
            weighted_loss_sum = 0.0
            weight_sum = 0.0

            for (
                batch_features,
                batch_aspects,
                batch_channels,
                batch_targets,
                batch_weights,
            ) in train_loader:
                batch_features = (
                    batch_features
                    .float()
                    .to(DEVICE_DUAL)
                )

                batch_aspects = (
                    batch_aspects
                    .long()
                    .to(DEVICE_DUAL)
                )

                batch_channels = (
                    batch_channels
                    .long()
                    .to(DEVICE_DUAL)
                )

                batch_targets = (
                    batch_targets
                    .float()
                    .to(DEVICE_DUAL)
                )

                batch_weights = (
                    batch_weights
                    .float()
                    .to(DEVICE_DUAL)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    batch_features,
                    batch_aspects,
                    batch_channels,
                )

                losses = loss_function(
                    logits,
                    batch_targets,
                )

                loss = (
                    losses
                    * batch_weights
                ).sum() / batch_weights.sum().clamp_min(
                    1e-8
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                weighted_loss_sum += float(
                    (
                        losses
                        * batch_weights
                    ).sum().item()
                )

                weight_sum += float(
                    batch_weights.sum().item()
                )

            validation_scores, _ = (
                predict_dual_scores(
                    model,
                    scaled_features[
                        validation_mask_dual
                    ],
                    aspects_dual[
                        validation_mask_dual
                    ],
                    channels_dual[
                        validation_mask_dual
                    ],
                )
            )

            validation_scored = (
                validation_table_base.copy()
            )

            validation_scored[
                "dual_score"
            ] = validation_scores

            channel_map_table, macro_map = (
                calculate_channel_map(
                    validation_scored,
                    "dual_score",
                )
            )

            dual_training_history_rows.append(
                {
                    "feature_set": feature_name,
                    "seed": seed,
                    "epoch": epoch,
                    "train_weighted_bce": (
                        weighted_loss_sum
                        / max(
                            weight_sum,
                            1e-8,
                        )
                    ),
                    "validation_supporting_map": float(
                        channel_map_table.loc[
                            channel_map_table[
                                "channel"
                            ] == "SUPPORTING",
                            "map",
                        ].iloc[0]
                    ),
                    "validation_contradicting_map": float(
                        channel_map_table.loc[
                            channel_map_table[
                                "channel"
                            ] == "CONTRADICTING",
                            "map",
                        ].iloc[0]
                    ),
                    "validation_macro_map": (
                        macro_map
                    ),
                }
            )

            if macro_map > best_macro_map + 1e-8:
                best_macro_map = macro_map
                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= DUAL_PATIENCE
            ):
                break

        if best_state is None:
            raise RuntimeError(
                "Dual selector failed to produce a checkpoint."
            )

        model.load_state_dict(
            best_state
        )

        best_history_row = next(
            row
            for row in dual_training_history_rows
            if (
                row["feature_set"]
                == feature_name
                and row["seed"] == seed
                and row["epoch"] == best_epoch
            )
        )

        dual_validation_rows.append(
            {
                "feature_set": feature_name,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_supporting_map": (
                    best_history_row[
                        "validation_supporting_map"
                    ]
                ),
                "best_validation_contradicting_map": (
                    best_history_row[
                        "validation_contradicting_map"
                    ]
                ),
                "best_validation_macro_map": (
                    best_macro_map
                ),
            }
        )

        test_scores, all_test_scores = (
            predict_dual_scores(
                model,
                scaled_features[
                    test_mask_dual
                ],
                aspects_dual[
                    test_mask_dual
                ],
                channels_dual[
                    test_mask_dual
                ],
            )
        )

        test_scored = test_table_base.copy()
        test_scored["dual_score"] = (
            test_scores
        )
        test_scored[
            "supporting_head_score"
        ] = all_test_scores[:, 0]
        test_scored[
            "contradicting_head_score"
        ] = all_test_scores[:, 1]
        test_scored["feature_set"] = (
            feature_name
        )
        test_scored["seed"] = seed

        dual_event_score_parts.append(
            test_scored
        )

        seed_results = evaluate_rankings(
            test_scored,
            score_column="dual_score",
            method_name=(
                "dual_" + feature_name
            ),
            seed=seed,
        )

        seed_results["channel"] = (
            seed_results[
                "annotation_id"
            ].map(channel_lookup)
        )

        dual_record_result_parts.append(
            seed_results
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "feature_set": feature_name,
                "input_dimension": int(
                    scaled_features.shape[1]
                ),
                "hidden_dimension": (
                    DUAL_HIDDEN_DIMENSION
                ),
                "aspect_order": ASPECT_ORDER,
                "channel_order": DUAL_CHANNELS,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_macro_map": (
                    best_macro_map
                ),
                "scaler_mean": (
                    scaler.mean_.astype(
                        np.float32
                    )
                ),
                "scaler_scale": (
                    scaler.scale_.astype(
                        np.float32
                    )
                ),
            },
            DUAL_OUTPUT_DIR
            / (
                f"dual_selector_{feature_name}_"
                f"seed_{seed}.pt"
            ),
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Add the corresponding single-selector results from Cell 27
# -----------------------------------------------------------------------------

single_comparison = record_results[
    record_results["method"].isin(
        DUAL_FEATURE_SETS
    )
].copy()

single_comparison["channel"] = (
    single_comparison["annotation_id"]
    .map(channel_lookup)
)

single_comparison["method"] = (
    "single_"
    + single_comparison["method"]
)

if single_comparison["channel"].isna().any():
    raise ValueError(
        "Cell 27 results could not be aligned with dual channels."
    )

dual_record_results = pd.concat(
    dual_record_result_parts,
    ignore_index=True,
)

comparison_record_results = pd.concat(
    [
        dual_record_results,
        single_comparison,
    ],
    ignore_index=True,
)


# -----------------------------------------------------------------------------
# Channel-specific, seed-level, and macro-channel summaries
# -----------------------------------------------------------------------------

ranking_metrics = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_at_k",
    "ndcg_at_k",
    "average_precision",
    "reciprocal_rank",
    "retained_fraction",
]

channel_seed_results = (
    comparison_record_results
    .groupby(
        [
            "method",
            "seed",
            "channel",
            "budget",
        ],
        as_index=False,
    )[ranking_metrics]
    .mean()
)

macro_seed_results = (
    channel_seed_results
    .groupby(
        [
            "method",
            "seed",
            "budget",
        ],
        as_index=False,
    )[ranking_metrics]
    .mean()
)


def aggregate_seed_table(
    table,
    group_columns,
):
    rows = []

    for keys, group in table.groupby(
        group_columns,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_values = dict(
            zip(
                group_columns,
                keys,
            )
        )

        for metric in ranking_metrics:
            values = group[
                metric
            ].to_numpy(dtype=np.float64)

            rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "mean": float(
                        np.nanmean(values)
                    ),
                    "standard_deviation": (
                        float(
                            np.nanstd(
                                values,
                                ddof=1,
                            )
                        )
                        if len(values) > 1
                        else 0.0
                    ),
                    "minimum": float(
                        np.nanmin(values)
                    ),
                    "maximum": float(
                        np.nanmax(values)
                    ),
                    "seeds": int(
                        len(values)
                    ),
                }
            )

    return pd.DataFrame(rows)


channel_aggregate = aggregate_seed_table(
    channel_seed_results,
    [
        "method",
        "channel",
        "budget",
    ],
)

macro_aggregate = aggregate_seed_table(
    macro_seed_results,
    [
        "method",
        "budget",
    ],
)


# -----------------------------------------------------------------------------
# Direct dual-versus-single deltas at matched seed and budget
# -----------------------------------------------------------------------------

delta_rows = []

for feature_name in DUAL_FEATURE_SETS:
    dual_method = (
        "dual_" + feature_name
    )
    single_method = (
        "single_" + feature_name
    )

    dual_subset = macro_seed_results[
        macro_seed_results["method"]
        == dual_method
    ]

    single_subset = macro_seed_results[
        macro_seed_results["method"]
        == single_method
    ]

    matched = dual_subset.merge(
        single_subset,
        on=[
            "seed",
            "budget",
        ],
        suffixes=(
            "_dual",
            "_single",
        ),
        validate="one_to_one",
    )

    for _, row in matched.iterrows():
        delta_rows.append(
            {
                "feature_set": feature_name,
                "seed": int(row["seed"]),
                "budget": int(row["budget"]),
                "map_delta": float(
                    row[
                        "average_precision_dual"
                    ]
                    - row[
                        "average_precision_single"
                    ]
                ),
                "mrr_delta": float(
                    row[
                        "reciprocal_rank_dual"
                    ]
                    - row[
                        "reciprocal_rank_single"
                    ]
                ),
                "precision_at_k_delta": float(
                    row[
                        "precision_at_k_dual"
                    ]
                    - row[
                        "precision_at_k_single"
                    ]
                ),
                "recall_at_k_delta": float(
                    row[
                        "recall_at_k_dual"
                    ]
                    - row[
                        "recall_at_k_single"
                    ]
                ),
                "f1_at_k_delta": float(
                    row[
                        "f1_at_k_dual"
                    ]
                    - row[
                        "f1_at_k_single"
                    ]
                ),
                "ndcg_at_k_delta": float(
                    row[
                        "ndcg_at_k_dual"
                    ]
                    - row[
                        "ndcg_at_k_single"
                    ]
                ),
            }
        )

dual_single_deltas = pd.DataFrame(
    delta_rows
)

delta_summary = (
    dual_single_deltas
    .groupby(
        [
            "feature_set",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        map_delta_mean=(
            "map_delta",
            "mean",
        ),
        map_delta_sd=(
            "map_delta",
            "std",
        ),
        mrr_delta_mean=(
            "mrr_delta",
            "mean",
        ),
        precision_at_k_delta_mean=(
            "precision_at_k_delta",
            "mean",
        ),
        recall_at_k_delta_mean=(
            "recall_at_k_delta",
            "mean",
        ),
        f1_at_k_delta_mean=(
            "f1_at_k_delta",
            "mean",
        ),
        ndcg_at_k_delta_mean=(
            "ndcg_at_k_delta",
            "mean",
        ),
    )
)


# -----------------------------------------------------------------------------
# Publication-facing wide tables
# -----------------------------------------------------------------------------

macro_primary = (
    macro_seed_results
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        precision_at_k_mean=(
            "precision_at_k",
            "mean",
        ),
        recall_at_k_mean=(
            "recall_at_k",
            "mean",
        ),
        f1_at_k_mean=(
            "f1_at_k",
            "mean",
        ),
        hit_at_k_mean=(
            "hit_at_k",
            "mean",
        ),
        ndcg_at_k_mean=(
            "ndcg_at_k",
            "mean",
        ),
        macro_map_mean=(
            "average_precision",
            "mean",
        ),
        macro_map_sd=(
            "average_precision",
            "std",
        ),
        macro_mrr_mean=(
            "reciprocal_rank",
            "mean",
        ),
        retained_fraction_mean=(
            "retained_fraction",
            "mean",
        ),
    )
)

channel_primary = (
    channel_seed_results
    .groupby(
        [
            "method",
            "channel",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        precision_at_k_mean=(
            "precision_at_k",
            "mean",
        ),
        recall_at_k_mean=(
            "recall_at_k",
            "mean",
        ),
        f1_at_k_mean=(
            "f1_at_k",
            "mean",
        ),
        ndcg_at_k_mean=(
            "ndcg_at_k",
            "mean",
        ),
        map_mean=(
            "average_precision",
            "mean",
        ),
        map_sd=(
            "average_precision",
            "std",
        ),
        mrr_mean=(
            "reciprocal_rank",
            "mean",
        ),
    )
)


# -----------------------------------------------------------------------------
# Save results and manifest
# -----------------------------------------------------------------------------

dual_record_results.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_record_results.csv",
    index=False,
)

comparison_record_results.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_single_comparison_records.csv",
    index=False,
)

channel_seed_results.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_channel_seed_results.csv",
    index=False,
)

macro_seed_results.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_macro_seed_results.csv",
    index=False,
)

channel_aggregate.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_channel_aggregate.csv",
    index=False,
)

macro_aggregate.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_macro_aggregate.csv",
    index=False,
)

dual_single_deltas.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_single_seed_deltas.csv",
    index=False,
)

delta_summary.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_single_delta_summary.csv",
    index=False,
)

macro_primary.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_primary_summary.csv",
    index=False,
)

channel_primary.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_channel_summary.csv",
    index=False,
)

dual_split_audit.to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_split_audit.csv",
    index=False,
)

pd.DataFrame(
    dual_training_history_rows
).to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_training_history.csv",
    index=False,
)

pd.DataFrame(
    dual_validation_rows
).to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_validation_selection.csv",
    index=False,
)

pd.concat(
    dual_event_score_parts,
    ignore_index=True,
).to_csv(
    DUAL_OUTPUT_DIR
    / "dual_selector_test_event_scores.csv",
    index=False,
)

dual_manifest = {
    "status": (
        "exploratory_silver_label_dual_rationale_retrieval"
    ),
    "channels": DUAL_CHANNELS,
    "channel_semantics": {
        "SUPPORTING": "E+ supporting-evidence selector",
        "CONTRADICTING": "E- contradicting-evidence selector",
    },
    "rationale_records": int(
        len(selector_records)
    ),
    "channel_record_counts": {
        key: int(value)
        for key, value in observed_channel_counts.items()
    },
    "neutral_policy": (
        "Neutral records contain no decisive evidence identifiers "
        "in this pilot and are excluded rather than treated as negatives."
    ),
    "architecture": (
        "Shared event projection and aspect embedding with separate "
        "supporting and contradicting scoring heads."
    ),
    "checkpoint_selection": (
        "Validation macro-MAP averaged equally across E+ and E-."
    ),
    "feature_sets": DUAL_FEATURE_SETS,
    "seeds": SELECTOR_SEEDS,
    "budgets": EVIDENCE_BUDGETS,
    "device": str(DEVICE_DUAL),
    "label_limitation": (
        "AI-assisted, single-reviewer-resolved silver rationales; "
        "not independent human gold-standard evidence channels."
    ),
}

with open(
    DUAL_OUTPUT_DIR
    / "dual_selector_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        dual_manifest,
        file,
        indent=2,
    )


# -----------------------------------------------------------------------------
# Display publication-facing results
# -----------------------------------------------------------------------------

print("=" * 80)
print("DAPT2020-MAST DUAL E+/E- SPARSE EVIDENCE SELECTORS")
print("=" * 80)

print("\nDevice:", DEVICE_DUAL)

print("\nChannel interpretation:")
print("SUPPORTING   -> E+ selector")
print("CONTRADICTING -> E- selector")

print("\nLeakage-aware dual-channel audit:")
display(
    dual_split_audit.round(6)
)

print("\nDual versus single selector: macro-channel results")
display(
    macro_primary
    .sort_values(
        [
            "budget",
            "macro_map_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(6)
)

print("\nDual-selector channel-specific results:")
display(
    channel_primary[
        channel_primary["method"]
        .str.startswith("dual_")
    ]
    .sort_values(
        [
            "method",
            "channel",
            "budget",
        ]
    )
    .round(6)
)

print("\nMatched-seed dual-minus-single deltas:")
display(
    delta_summary.round(6)
)

print("\nValidation-selected dual checkpoints:")
display(
    pd.DataFrame(
        dual_validation_rows
    ).round(6)
)

print("\nSaved dual-selector results to:")
print(DUAL_OUTPUT_DIR)

print(
    "\nCell 28 completed successfully. Positive deltas favor the "
    "dual E+/E- architecture. Interpret all results as exploratory "
    "silver-rationale evidence, not independent human gold-standard "
    "performance."
)
