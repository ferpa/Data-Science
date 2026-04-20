# Baseline Models Summary (Protocol Only, No Leakage)

## Windowing
- Train windows: 13,904
- Validation windows: 2,623
- Test windows: 2,707
- Classes: ascending_stairs, cycling, descending_stairs, ironing, lying, nordic_walking, rope_jumping, running, sitting, standing, vacuum_cleaning, walking

## Models
- glmnet multinomial (ridge) with train-only scaling
- ranger random forest with optional class weights

## Selection Criterion
- Pick best model by validation Macro-F1 (tie-break by accuracy).

## Best on Validation
- Model: glmnet_ridge
- Accuracy: 0.9295
- Macro-F1: 0.9280

## Artifacts
- baseline_metrics.csv
- confusion matrices (csv + png) for each model/split
- scaler_train_only.rds
