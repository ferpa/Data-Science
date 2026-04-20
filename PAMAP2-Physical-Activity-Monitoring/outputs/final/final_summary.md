# Final Model Summary (PAMAP2 Protocol Only)

## Data & Split
- Protocol files only; optional sessions excluded.
- Subject-independent split:
  - Train subjects: 101, 103, 104, 106, 107, 108
  - Validation subject: 102
  - Test subject: 105

## Windowing
- Window length (seconds): 5
- Step / stride (seconds): 1
- Min rows per window: 250
- Label purity threshold: 0.9

## Final model
- Model: Random Forest (ranger)
- num.trees: 400
- mtry: 14
- min.node.size: 1
- class weights: YES (inverse frequency)
- importance: permutation

## Final TEST performance (one-shot report)
- Accuracy: 0.8685
- Macro-F1: 0.8580

## Artifacts
- final_model_ranger_rf_dev.rds
- final_test_metrics.csv
- final_test_per_class_metrics.csv
- final_test_confusion.csv + final_test_confusion.png
- final_feature_importance.csv + final_feature_importance_top30.png
