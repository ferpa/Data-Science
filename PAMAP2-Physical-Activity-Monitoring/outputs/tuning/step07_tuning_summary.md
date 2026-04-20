# Step 07 Subject-aware tuning summary

## Tuning protocol
- Inner evaluation: LOSO within TRAIN (one fold per subject in TRAIN).
- Model selection: VALIDATION subject only (subject-independent).
- Test set: reported once; not used for selection.

## Best hyperparameters
- glmnet: alpha=0.25, lambda=0.242167
- ranger: trees=400, mtry=14, min.node.size=1

## Validation comparison (selection basis)
- Best model: ranger_rf_tuned
- Validation accuracy: 0.9089
- Validation macro-F1: 0.9176

## Artifacts
- glmnet_loso_tuning.csv
- ranger_loso_summary.csv (+ ranger_loso_raw.csv)
- validation_model_comparison.csv
- test_model_comparison.csv
- model_glmnet_tuned.rds
- model_ranger_rf_tuned.rds
