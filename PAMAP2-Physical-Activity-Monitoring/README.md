# Human Activity Recognition with PAMAP2

HarvardX Data Science Capstone (PH125.9x) — *Choose Your Own* project.

A subject-independent classifier that recognises twelve everyday physical
activities from short windows of inertial sensor data. Trained, tuned and
evaluated under a strict Leave-One-Subject-Out protocol so that the
reported numbers reflect performance on a person the model has never seen
before, not on a held-out slice of a familiar user.

## Result

A tuned Random Forest on engineered window-level features reaches
**~87% accuracy** and **~0.86 macro-F1** on a held-out test subject,
beating a regularised multinomial baseline on every split.

## Repository layout

```
.
|-- Script.R                     # End-to-end pipeline (ingest -> model)
|-- Fernando_Parodi_Capstone.Rmd # Final report (knits to PDF)
|-- Fernando_Parodi_Capstone.pdf # Pre-built report
|-- README.md
|
|-- data/
|   `-- processed/               # Cleaned dataset + split manifest
|
`-- outputs/
    |-- eda_raw/                 # Raw exploratory analysis
    |-- post_clean_eda/          # Post-cleaning EDA
    |-- cleaning/                # Cleaning summary
    |-- baselines/               # Untuned baseline metrics
    |-- tuning/                  # LOSO tuning + model comparison
    `-- final/                   # Final test metrics + figures
```

The `data/` and `outputs/` trees are produced by `Script.R` and are not
checked in. Knitting the report from a fresh clone regenerates them
automatically.

## How to reproduce

1. Clone the repository.
2. Place the PAMAP2 protocol files (UCI ML Repository) in the expected
   raw-data path or it will be dowloaded automatically when running the script.
3. Knit `Fernando_Parodi_Capstone.Rmd`. If artifacts are missing, the
   setup chunk runs `Script.R` as a child process and streams its
   progress live before laying out the PDF.

## Dataset

PAMAP2 Physical Activity Monitoring (Reiss & Stricker, 2012),
distributed by the UCI Machine Learning Repository for research use.
Cite the original 2012 papers if you reuse it.

## Author

Fernando Marcelo Parodi
