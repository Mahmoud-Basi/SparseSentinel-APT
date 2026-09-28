# =============================================================================
# CELL 27: ASPECT-CONDITIONED SPARSE EVIDENCE SELECTOR
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
# Required Cell 26 objects
# -----------------------------------------------------------------------------

required_objects = [
    "DATA_DIR",
    "OUTPUT_DIR",
    "AMJEPA_DIR",
    "TENSOR_DIR",
    "STANCE_OUTPUT_DIR",
    "stance_table",
    "ASPECT_ORDER",
    "X_network_stance",
    "X_temporal_stance",
    "attention_mask_stance",
    "text_embedding_lookup_stance",
    "window_text_ids_stance",
    "window_to_tensor_index",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise NameError(
        "Missing objects from Cell 26:\n"
        + "\n".join(missing_objects)
        + "\n\nRun Cell 26 before Cell 27."
    )

DATA_DIR = Path(DATA_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

MEMBERSHIP_PATH = (
    OUTPUT_DIR
    / "dapt2020_mast_window_membership_enriched.parquet"
)

SELECTOR_OUTPUT_DIR = (
    Path(AMJEPA_DIR)
    / "silver_sparse_selector"
)

SELECTOR_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not MEMBERSHIP_PATH.exists():
    raise FileNotFoundError(
        f"Missing evidence membership table: {MEMBERSHIP_PATH}"
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SELECTOR_SEEDS = [
    42,
    52,
    62,
    72,
    82,
]

EVIDENCE_BUDGETS = [
    1,
    2,
    4,
    8,
]

HIDDEN_DIMENSION = 64
MAXIMUM_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_BASELINE_REPETITIONS = 200

DEVICE_SELECTOR = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def set_selector_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Select only records with explicit decisive-evidence supervision
# -----------------------------------------------------------------------------

def parse_evidence_ids(value):
    text = str(value).strip()

    if (
        not text
        or text.upper() in {
            "NAN",
            "NONE",
            "NONE_RECORDED",
        }
    ):
        return []

    return [
        item.strip()
        for item in text.split(";")
        if item.strip()
    ]


selector_records = stance_table.copy()

selector_records[
    "decisive_id_list"
] = selector_records[
    "decisive_evidence_ids"
].map(parse_evidence_ids)

selector_records[
    "decisive_evidence_count"
] = selector_records[
    "decisive_id_list"
].map(len)

selector_records = selector_records[
    selector_records[
        "decisive_evidence_count"
    ] > 0
].copy()

selector_records = selector_records.reset_index(
    drop=True
)

if len(selector_records) != 94:
    print(
        "Warning: expected 94 rationale-annotated records, "
        f"but found {len(selector_records)}."
    )

if len(selector_records) < 30:
    raise ValueError(
        "Too few records contain decisive evidence for a "
        "meaningful selector experiment."
    )


# -----------------------------------------------------------------------------
# Recover evidence ID to tensor-position mapping
# -----------------------------------------------------------------------------

membership = pd.read_parquet(
    MEMBERSHIP_PATH
)

required_membership_columns = {
    "window_id",
    "position",
    "evidence_id",
}

missing_membership_columns = sorted(
    required_membership_columns
    - set(membership.columns)
)

if missing_membership_columns:
    raise ValueError(
        "Membership table is missing columns: "
        f"{missing_membership_columns}"
    )

membership["window_id"] = (
    membership["window_id"]
    .astype(str)
)

membership["evidence_id"] = (
    membership["evidence_id"]
    .astype(str)
    .str.strip()
)

membership["position"] = pd.to_numeric(
    membership["position"],
    errors="raise",
).astype(np.int64)

if membership[
    [
        "window_id",
        "position",
    ]
].duplicated().any():
    raise ValueError(
        "Membership contains duplicate window-position pairs."
    )

membership_groups = {
    window_id: group.sort_values(
        "position"
    ).copy()
    for window_id, group in membership.groupby(
        "window_id",
        sort=False,
    )
}


# -----------------------------------------------------------------------------
# Event-level multimodal representations
# -----------------------------------------------------------------------------

text_event_features = (
    text_embedding_lookup_stance[
        window_text_ids_stance
    ]
)

feature_sets = {
    "network_temporal": np.concatenate(
        [
            X_network_stance,
            X_temporal_stance,
        ],
        axis=-1,
    ).astype(np.float32),
    "text_only": text_event_features.astype(
        np.float32
    ),
    "full_multimodal": np.concatenate(
        [
            X_network_stance,
            X_temporal_stance,
            text_event_features,
        ],
        axis=-1,
    ).astype(np.float32),
}

aspect_to_index_selector = {
    aspect: index
    for index, aspect in enumerate(
        ASPECT_ORDER
    )
}


# -----------------------------------------------------------------------------
# Construct candidate-event supervision
# -----------------------------------------------------------------------------

event_rows = []

for record_index, record in (
    selector_records.iterrows()
):
    window_id = str(
        record["window_id"]
    )

    if window_id not in membership_groups:
        raise ValueError(
            f"No membership rows found for {window_id}."
        )

    tensor_index = int(
        window_to_tensor_index[
            window_id
        ]
    )

    group = membership_groups[
        window_id
    ]

    decisive_ids = set(
        record["decisive_id_list"]
    )

    available_ids = set(
        group["evidence_id"]
    )

    unknown_ids = sorted(
        decisive_ids
        - available_ids
    )

    if unknown_ids:
        raise ValueError(
            "Decisive evidence is not present in its source window. "
            f"Annotation {record['annotation_id']}: {unknown_ids}"
        )

    valid_positions = np.where(
        attention_mask_stance[
            tensor_index
        ]
    )[0]

    group_positions = set(
        group["position"].astype(int)
    )

    for position in valid_positions:
        if int(position) not in group_positions:
            raise ValueError(
                "A valid tensor position has no evidence identifier: "
                f"{window_id}, position={position}."
            )

    evidence_by_position = dict(
        zip(
            group["position"].astype(int),
            group["evidence_id"],
        )
    )

    positive_count = len(
        decisive_ids
    )

    negative_count = (
        len(valid_positions)
        - positive_count
    )

    if negative_count <= 0:
        raise ValueError(
            "A selector record has no non-decisive candidates: "
            f"{record['annotation_id']}"
        )

    for position in valid_positions:
        evidence_id = evidence_by_position[
            int(position)
        ]

        target = int(
            evidence_id in decisive_ids
        )

        # Each annotation unit receives equal total weight. Within a unit,
        # positive and non-positive candidate sets each receive half the weight.
        if target == 1:
            within_window_weight = (
                0.5 / positive_count
            )
        else:
            within_window_weight = (
                0.5 / negative_count
            )

        annotation_weight = (
            float(record["confidence"])
            / 5.0
        )

        if bool(
            record["insufficient_context"]
        ):
            annotation_weight *= 0.5

        event_rows.append(
            {
                "record_index": record_index,
                "annotation_id": record[
                    "annotation_id"
                ],
                "window_id": window_id,
                "split": record["split"],
                "aspect": record["aspect"],
                "aspect_index": (
                    aspect_to_index_selector[
                        record["aspect"]
                    ]
                ),
                "tensor_index": tensor_index,
                "position": int(position),
                "evidence_id": evidence_id,
                "target": target,
                "sample_weight": (
                    within_window_weight
                    * annotation_weight
                ),
            }
        )

event_table = pd.DataFrame(
    event_rows
)

if event_table.empty:
    raise ValueError(
        "No selector candidates were constructed."
    )

positive_audit = (
    event_table
    .groupby("annotation_id")[
        "target"
    ]
    .sum()
)

if (positive_audit <= 0).any():
    raise AssertionError(
        "Every selector record must contain at least one positive event."
    )


# -----------------------------------------------------------------------------
# Split audit: no annotation or window may cross partitions
# -----------------------------------------------------------------------------

split_order_selector = [
    "train",
    "validation",
    "test",
]

split_audit_rows = []

for split_name in split_order_selector:
    split_events = event_table[
        event_table["split"]
        == split_name
    ]

    split_records = selector_records[
        selector_records["split"]
        == split_name
    ]

    if split_records.empty:
        raise ValueError(
            f"No rationale-annotated records occur in {split_name}."
        )

    split_audit_rows.append(
        {
            "split": split_name,
            "records": len(split_records),
            "unique_windows": int(
                split_records[
                    "window_id"
                ].nunique()
            ),
            "candidate_events": len(
                split_events
            ),
            "decisive_events": int(
                split_events[
                    "target"
                ].sum()
            ),
            "positive_prevalence": float(
                split_events[
                    "target"
                ].mean()
            ),
        }
    )

split_audit = pd.DataFrame(
    split_audit_rows
)

window_split_count = (
    event_table
    .groupby("window_id")["split"]
    .nunique()
)

if (window_split_count > 1).any():
    raise AssertionError(
        "At least one window crosses selector partitions."
    )


# -----------------------------------------------------------------------------
# Aspect-conditioned multiplicative selector
# -----------------------------------------------------------------------------

class AspectConditionedSelector(nn.Module):
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

        self.score_head = nn.Sequential(
            nn.Linear(
                hidden_dimension * 3,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                hidden_dimension,
                1,
            ),
        )

    def forward(
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

        combined = torch.cat(
            [
                event_state,
                aspect_state,
                interaction,
            ],
            dim=-1,
        )

        return self.score_head(
            combined
        ).squeeze(-1)


# -----------------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------------

def average_precision_from_ranking(
    targets,
    scores,
):
    order = np.argsort(-scores)
    ranked_targets = targets[order]
    positive_total = int(
        ranked_targets.sum()
    )

    if positive_total == 0:
        return np.nan

    cumulative = np.cumsum(
        ranked_targets
    )

    precision_at_rank = (
        cumulative
        / np.arange(
            1,
            len(ranked_targets) + 1,
        )
    )

    return float(
        (
            precision_at_rank
            * ranked_targets
        ).sum()
        / positive_total
    )


def reciprocal_rank(
    targets,
    scores,
):
    order = np.argsort(-scores)
    ranked_targets = targets[order]
    positive_positions = np.where(
        ranked_targets == 1
    )[0]

    if len(positive_positions) == 0:
        return np.nan

    return float(
        1.0
        / (
            positive_positions[0]
            + 1
        )
    )


def ndcg_at_k(
    targets,
    scores,
    budget,
):
    budget = min(
        budget,
        len(targets),
    )

    order = np.argsort(-scores)[
        :budget
    ]

    gains = targets[order]

    discounts = 1.0 / np.log2(
        np.arange(
            2,
            budget + 2,
        )
    )

    dcg = float(
        (gains * discounts).sum()
    )

    ideal_positive_count = min(
        int(targets.sum()),
        budget,
    )

    if ideal_positive_count == 0:
        return np.nan

    idcg = float(
        discounts[
            :ideal_positive_count
        ].sum()
    )

    return dcg / idcg


def evaluate_rankings(
    table,
    score_column,
    method_name,
    seed,
):
    rows = []

    for annotation_id, group in table.groupby(
        "annotation_id",
        sort=False,
    ):
        targets = group[
            "target"
        ].to_numpy(dtype=np.int64)

        scores = group[
            score_column
        ].to_numpy(dtype=np.float64)

        positive_total = int(
            targets.sum()
        )

        ap = average_precision_from_ranking(
            targets,
            scores,
        )

        mrr = reciprocal_rank(
            targets,
            scores,
        )

        for budget in EVIDENCE_BUDGETS:
            effective_budget = min(
                budget,
                len(group),
            )

            selected = np.argsort(
                -scores
            )[:effective_budget]

            true_positive = int(
                targets[selected].sum()
            )

            precision = (
                true_positive
                / effective_budget
            )

            recall = (
                true_positive
                / positive_total
            )

            if precision + recall > 0:
                f1 = (
                    2.0
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                    )
                )
            else:
                f1 = 0.0

            rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "annotation_id": annotation_id,
                    "window_id": group[
                        "window_id"
                    ].iloc[0],
                    "aspect": group[
                        "aspect"
                    ].iloc[0],
                    "budget": budget,
                    "candidate_events": len(
                        group
                    ),
                    "decisive_events": (
                        positive_total
                    ),
                    "selected_events": (
                        effective_budget
                    ),
                    "true_positive": (
                        true_positive
                    ),
                    "precision_at_k": (
                        precision
                    ),
                    "recall_at_k": recall,
                    "f1_at_k": f1,
                    "hit_at_k": float(
                        true_positive > 0
                    ),
                    "ndcg_at_k": ndcg_at_k(
                        targets,
                        scores,
                        budget,
                    ),
                    "average_precision": ap,
                    "reciprocal_rank": mrr,
                    "retained_fraction": (
                        effective_budget
                        / len(group)
                    ),
                }
            )

    return pd.DataFrame(rows)


def mean_average_precision(
    table,
    score_column,
):
    values = []

    for _, group in table.groupby(
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

    return float(
        np.nanmean(values)
    )


# -----------------------------------------------------------------------------
# Feature assembly
# -----------------------------------------------------------------------------

def gather_event_features(
    feature_tensor,
    table,
):
    return feature_tensor[
        table["tensor_index"].to_numpy(
            dtype=np.int64
        ),
        table["position"].to_numpy(
            dtype=np.int64
        ),
    ]


def predict_selector_scores(
    model,
    features,
    aspects,
    batch_size=1024,
):
    model.eval()
    score_parts = []

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
            ).float().to(
                DEVICE_SELECTOR
            )

            batch_aspects = torch.from_numpy(
                aspects[start:end]
            ).long().to(
                DEVICE_SELECTOR
            )

            scores = model(
                batch_features,
                batch_aspects,
            )

            score_parts.append(
                scores.detach().cpu().numpy()
            )

    return np.concatenate(
        score_parts
    )


# -----------------------------------------------------------------------------
# Train five seeds for each modality setting
# -----------------------------------------------------------------------------

all_record_results = []
training_history_rows = []
validation_selection_rows = []
event_prediction_tables = []

for feature_name, feature_tensor in (
    feature_sets.items()
):
    all_features = gather_event_features(
        feature_tensor,
        event_table,
    ).astype(np.float32)

    train_event_mask = (
        event_table["split"].to_numpy()
        == "train"
    )

    validation_event_mask = (
        event_table["split"].to_numpy()
        == "validation"
    )

    test_event_mask = (
        event_table["split"].to_numpy()
        == "test"
    )

    scaler = StandardScaler()

    scaler.fit(
        all_features[
            train_event_mask
        ]
    )

    scaled_features = scaler.transform(
        all_features
    ).astype(np.float32)

    aspects = event_table[
        "aspect_index"
    ].to_numpy(dtype=np.int64)

    targets = event_table[
        "target"
    ].to_numpy(dtype=np.float32)

    weights = event_table[
        "sample_weight"
    ].to_numpy(dtype=np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(
            scaled_features[
                train_event_mask
            ]
        ),
        torch.from_numpy(
            aspects[
                train_event_mask
            ]
        ),
        torch.from_numpy(
            targets[
                train_event_mask
            ]
        ),
        torch.from_numpy(
            weights[
                train_event_mask
            ]
        ),
    )

    validation_table = event_table[
        validation_event_mask
    ].copy()

    test_table_base = event_table[
        test_event_mask
    ].copy()

    for seed in SELECTOR_SEEDS:
        set_selector_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model = AspectConditionedSelector(
            input_dimension=
                scaled_features.shape[1],
            number_of_aspects=
                len(ASPECT_ORDER),
            hidden_dimension=
                HIDDEN_DIMENSION,
        ).to(DEVICE_SELECTOR)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        loss_function = (
            nn.BCEWithLogitsLoss(
                reduction="none"
            )
        )

        best_validation_map = -np.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(
            1,
            MAXIMUM_EPOCHS + 1,
        ):
            model.train()
            epoch_loss_sum = 0.0
            epoch_weight_sum = 0.0

            for (
                batch_features,
                batch_aspects,
                batch_targets,
                batch_weights,
            ) in train_loader:
                batch_features = (
                    batch_features
                    .float()
                    .to(DEVICE_SELECTOR)
                )

                batch_aspects = (
                    batch_aspects
                    .long()
                    .to(DEVICE_SELECTOR)
                )

                batch_targets = (
                    batch_targets
                    .float()
                    .to(DEVICE_SELECTOR)
                )

                batch_weights = (
                    batch_weights
                    .float()
                    .to(DEVICE_SELECTOR)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    batch_features,
                    batch_aspects,
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

                epoch_loss_sum += float(
                    (
                        losses
                        * batch_weights
                    ).sum().item()
                )

                epoch_weight_sum += float(
                    batch_weights.sum().item()
                )

            validation_scores = (
                predict_selector_scores(
                    model,
                    scaled_features[
                        validation_event_mask
                    ],
                    aspects[
                        validation_event_mask
                    ],
                )
            )

            validation_scored = (
                validation_table.copy()
            )

            validation_scored[
                "selector_score"
            ] = validation_scores

            validation_map = (
                mean_average_precision(
                    validation_scored,
                    "selector_score",
                )
            )

            training_history_rows.append(
                {
                    "feature_set": feature_name,
                    "seed": seed,
                    "epoch": epoch,
                    "train_weighted_bce": (
                        epoch_loss_sum
                        / max(
                            epoch_weight_sum,
                            1e-8,
                        )
                    ),
                    "validation_map": (
                        validation_map
                    ),
                }
            )

            if (
                validation_map
                > best_validation_map
                + 1e-8
            ):
                best_validation_map = (
                    validation_map
                )

                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                break

        if best_state is None:
            raise RuntimeError(
                "Selector training did not produce a valid checkpoint."
            )

        model.load_state_dict(
            best_state
        )

        validation_selection_rows.append(
            {
                "feature_set": feature_name,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_map": (
                    best_validation_map
                ),
            }
        )

        test_scores = predict_selector_scores(
            model,
            scaled_features[
                test_event_mask
            ],
            aspects[
                test_event_mask
            ],
        )

        test_scored = test_table_base.copy()
        test_scored["selector_score"] = (
            test_scores
        )
        test_scored["feature_set"] = (
            feature_name
        )
        test_scored["seed"] = seed

        event_prediction_tables.append(
            test_scored
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "feature_set": feature_name,
                "input_dimension": int(
                    scaled_features.shape[1]
                ),
                "number_of_aspects": len(
                    ASPECT_ORDER
                ),
                "hidden_dimension": (
                    HIDDEN_DIMENSION
                ),
                "aspect_order": ASPECT_ORDER,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_map": (
                    best_validation_map
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
            SELECTOR_OUTPUT_DIR
            / (
                f"selector_{feature_name}_"
                f"seed_{seed}.pt"
            ),
        )

        seed_results = evaluate_rankings(
            test_scored,
            score_column="selector_score",
            method_name=feature_name,
            seed=seed,
        )

        all_record_results.append(
            seed_results
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Non-learned test baselines
# -----------------------------------------------------------------------------

full_features = gather_event_features(
    feature_sets[
        "full_multimodal"
    ],
    event_table,
).astype(np.float32)

train_event_mask = (
    event_table["split"].to_numpy()
    == "train"
)

test_event_mask = (
    event_table["split"].to_numpy()
    == "test"
)

baseline_scaler = StandardScaler()
baseline_scaler.fit(
    full_features[
        train_event_mask
    ]
)

standardized_full_features = (
    baseline_scaler.transform(
        full_features
    )
)

baseline_test_table = event_table[
    test_event_mask
].copy()

baseline_test_table[
    "magnitude_score"
] = np.linalg.norm(
    standardized_full_features[
        test_event_mask
    ],
    axis=1,
)

all_record_results.append(
    evaluate_rankings(
        baseline_test_table,
        score_column="magnitude_score",
        method_name="magnitude_baseline",
        seed=-1,
    )
)

random_result_parts = []

for repetition in range(
    RANDOM_BASELINE_REPETITIONS
):
    rng = np.random.default_rng(
        10000 + repetition
    )

    random_table = (
        baseline_test_table.copy()
    )

    random_table[
        "random_score"
    ] = rng.random(
        len(random_table)
    )

    random_result_parts.append(
        evaluate_rankings(
            random_table,
            score_column="random_score",
            method_name="random_baseline",
            seed=repetition,
        )
    )

random_record_results = pd.concat(
    random_result_parts,
    ignore_index=True,
)

all_record_results.append(
    random_record_results
)


# -----------------------------------------------------------------------------
# Aggregate test results
# -----------------------------------------------------------------------------

record_results = pd.concat(
    all_record_results,
    ignore_index=True,
)

metric_columns = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_at_k",
    "ndcg_at_k",
    "average_precision",
    "reciprocal_rank",
    "retained_fraction",
]

seed_level_results = (
    record_results
    .groupby(
        [
            "method",
            "seed",
            "budget",
        ],
        as_index=False,
    )[metric_columns]
    .mean()
)

aggregate_rows = []

for (
    method_name,
    budget,
), group in seed_level_results.groupby(
    [
        "method",
        "budget",
    ],
    sort=False,
):
    for metric in metric_columns:
        values = group[
            metric
        ].to_numpy(dtype=np.float64)

        aggregate_rows.append(
            {
                "method": method_name,
                "budget": int(budget),
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

aggregate_results = pd.DataFrame(
    aggregate_rows
)

primary_summary = (
    seed_level_results[
        [
            "method",
            "seed",
            "budget",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "hit_at_k",
            "ndcg_at_k",
            "average_precision",
            "reciprocal_rank",
            "retained_fraction",
        ]
    ]
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
        map_mean=(
            "average_precision",
            "mean",
        ),
        mrr_mean=(
            "reciprocal_rank",
            "mean",
        ),
        retained_fraction_mean=(
            "retained_fraction",
            "mean",
        ),
    )
)


# -----------------------------------------------------------------------------
# Per-aspect audit for the full multimodal selector
# -----------------------------------------------------------------------------

full_selector_records = record_results[
    record_results["method"]
    == "full_multimodal"
].copy()

per_aspect_results = (
    full_selector_records
    .groupby(
        [
            "aspect",
            "budget",
        ],
        as_index=False,
    )[
        [
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "hit_at_k",
            "ndcg_at_k",
            "average_precision",
            "reciprocal_rank",
            "retained_fraction",
        ]
    ]
    .mean()
)


# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

record_results.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_record_level_results.csv",
    index=False,
)

seed_level_results.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_seed_level_results.csv",
    index=False,
)

aggregate_results.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_aggregate_results.csv",
    index=False,
)

primary_summary.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_primary_summary.csv",
    index=False,
)

per_aspect_results.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_per_aspect_results.csv",
    index=False,
)

split_audit.to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_split_audit.csv",
    index=False,
)

pd.DataFrame(
    training_history_rows
).to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_training_history.csv",
    index=False,
)

pd.DataFrame(
    validation_selection_rows
).to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_validation_selection.csv",
    index=False,
)

pd.concat(
    event_prediction_tables,
    ignore_index=True,
).to_csv(
    SELECTOR_OUTPUT_DIR
    / "selector_test_event_scores.csv",
    index=False,
)

selector_manifest = {
    "status": (
        "exploratory_silver_label_rationale_retrieval"
    ),
    "supervised_records": int(
        len(selector_records)
    ),
    "unlabeled_records_excluded": int(
        len(stance_table)
        - len(selector_records)
    ),
    "unlabeled_policy": (
        "Records without decisive evidence identifiers were excluded "
        "from selector supervision and were not treated as negatives."
    ),
    "within_record_negative_policy": (
        "Within explicitly rationale-annotated records, valid events "
        "not selected as decisive were treated as non-decisive ranking "
        "candidates."
    ),
    "split_policy": (
        "Original hidden-key train/validation/test split; no window "
        "crosses partitions."
    ),
    "selection_policy": (
        "Early stopping used validation MAP; test rankings were "
        "computed once for each frozen seed checkpoint."
    ),
    "feature_sets": {
        name: int(tensor.shape[-1])
        for name, tensor in feature_sets.items()
    },
    "seeds": SELECTOR_SEEDS,
    "budgets": EVIDENCE_BUDGETS,
    "random_baseline_repetitions": (
        RANDOM_BASELINE_REPETITIONS
    ),
    "device": str(DEVICE_SELECTOR),
    "label_limitation": (
        "AI-assisted, single-reviewer-resolved silver rationales; "
        "not independent human gold-standard evidence selection."
    ),
}

with open(
    SELECTOR_OUTPUT_DIR
    / "selector_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selector_manifest,
        file,
        indent=2,
    )


# -----------------------------------------------------------------------------
# Display publication-facing results
# -----------------------------------------------------------------------------

print("=" * 80)
print("DAPT2020-MAST ASPECT-CONDITIONED SPARSE EVIDENCE SELECTOR")
print("=" * 80)

print("\nDevice:", DEVICE_SELECTOR)
print(
    "Rationale-annotated records:",
    len(selector_records),
)
print(
    "Records without decisive IDs excluded as unlabeled:",
    len(stance_table)
    - len(selector_records),
)

print("\nLeakage-aware selector split audit:")
display(
    split_audit.round(6)
)

print("\nPrimary sparse-retrieval summary:")
display(
    primary_summary
    .sort_values(
        [
            "budget",
            "map_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(6)
)

print("\nFull-multimodal selector by aspect:")
display(
    per_aspect_results.round(6)
)

print("\nValidation-selected checkpoints:")
display(
    pd.DataFrame(
        validation_selection_rows
    ).round(6)
)

print("\nSaved selector results to:")
print(SELECTOR_OUTPUT_DIR)

print(
    "\nCell 27 completed successfully. These are exploratory "
    "silver-rationale results. Do not describe them as independently "
    "human-verified evidence-selection performance."
)
