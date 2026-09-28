# Reproducibility and leakage controls

- Preprocessing statistics are fitted on the training partition only.
- Validation data select checkpoints, evidence budgets, calibration temperatures, and abstention thresholds.
- Test labels are not used during model or policy selection.
- Five-seed summaries use seeds 42, 52, 62, 72, and 82.
- Missing-modality evaluations preserve each trained model and modify only the evaluation input availability.
- Reports are deterministic functions of predictions, uncertainty, selected immutable event identifiers, and fixed templates.
- The stance pilot is silver-labelled and is not an independently adjudicated human benchmark.

