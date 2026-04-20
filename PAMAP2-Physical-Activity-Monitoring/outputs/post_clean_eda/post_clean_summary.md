# Post-clean Validation Summary (Protocol Only)

## Validation Checks
- Protocol-only confirmed (no optional sessions).
- activity_id == 0 removed.
- Orientation columns removed (dataset marks them invalid).
- acc6 columns removed.

## Subject-independent Split
- Eligible subjects (min rows 50000): 101, 102, 103, 104, 105, 106, 107, 108
- Train subjects: 101, 103, 104, 106, 107, 108
- Validation subject: 102
- Test subject: 105
- Train rows: 1,400,690 | Val rows: 263,349 | Test rows: 272,442

## Artifacts
- post_clean_global_summary.csv
- post_clean_missingness_by_feature.csv + figure
- post_clean_activity_distribution_overall.csv + figure
- post_clean_rows_by_subject.csv + figure
- post_clean_activity_coverage_by_subject.csv + figure
- split_manifest_protocol.csv
- loso_folds_train_protocol.rds
