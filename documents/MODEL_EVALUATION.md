# OpenCredit AI — Model Evaluation Report

**Generated:** 2026-09-03 16:50 UTC  
**Data source:** synthetic  
**Dataset:** 300 samples, 60 features  
**Split:** seed 42, test_size 0.2

Evaluates the saved production artifacts in `data/models/` (3 algorithms × 2 targets) on a held-out test set.

## Trust — `trust_label`

Ranking by ROC-AUC (best first):

| # | Model | Acc | Prec | Rec | F1 | AUC | LogLoss | Brier | CV Acc |
|---|-------|-----|------|-----|----|----|---------|-------|--------|
| 1 | `random_forest` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | 0.000 | 1.000±0.000 |
| 2 | `gradient_boosting` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000±0.000 |
| 3 | `xgboost` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | 0.000 | 0.990±0.013 |

**Winner:** `random_forest` (ROC-AUC 1.000)

Confusion matrices (held-out test set):

| Model | TP | TN | FP | FN |
|-------|----|----|----|----|
| `random_forest` | 39 | 21 | 0 | 0 |
| `gradient_boosting` | 39 | 21 | 0 | 0 |
| `xgboost` | 39 | 21 | 0 | 0 |

## Potential — `potential_label`

Ranking by ROC-AUC (best first):

| # | Model | Acc | Prec | Rec | F1 | AUC | LogLoss | Brier | CV Acc |
|---|-------|-----|------|-----|----|----|---------|-------|--------|
| 1 | `random_forest` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | 0.000 | 1.000±0.000 |
| 2 | `gradient_boosting` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000±0.000 |
| 3 | `xgboost` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | 0.000 | 0.997±0.007 |

**Winner:** `random_forest` (ROC-AUC 1.000)

Confusion matrices (held-out test set):

| Model | TP | TN | FP | FN |
|-------|----|----|----|----|
| `random_forest` | 24 | 36 | 0 | 0 |
| `gradient_boosting` | 24 | 36 | 0 | 0 |
| `xgboost` | 24 | 36 | 0 | 0 |

---

*Metrics: accuracy / precision / recall / F1 / ROC-AUC on the held-out test set; LogLoss and Brier measure probabilistic calibration (lower is better); CV Acc is 5-fold cross-validated accuracy on the full dataset (algorithm stability).*

> **Caveat — synthetic data.** The synthetic generator derives both labels as deterministic functions of the 60 features, so perfect metrics here only confirm that each algorithm can learn the generator's rules.  They are **not** evidence of real-world performance.  Re-run this evaluation on real labeled investigations before trusting any model in production decisions.
## Top features — winning models

### Trust — `random_forest`

| # | Feature | Importance |
|---|---------|------------|
| 1 | `sentiment_compound` | 0.1020 |
| 2 | `sentiment_positive` | 0.1000 |
| 3 | `features_total` | 0.0932 |
| 4 | `sentiment_negative` | 0.0800 |
| 5 | `features_cat_risk` | 0.0724 |
| 6 | `features_cat_audience` | 0.0700 |
| 7 | `feature_found_ratio` | 0.0604 |
| 8 | `confidence_max` | 0.0600 |
| 9 | `features_searched` | 0.0507 |
| 10 | `positive_signal_ratio` | 0.0464 |

### Potential — `random_forest`

| # | Feature | Importance |
|---|---------|------------|
| 1 | `missing_information_count` | 0.1332 |
| 2 | `features_cat_growth` | 0.1007 |
| 3 | `features_total` | 0.0933 |
| 4 | `features_searched` | 0.0800 |
| 5 | `confidence_max` | 0.0742 |
| 6 | `confidence_mean` | 0.0705 |
| 7 | `credibility_reliable_ratio` | 0.0655 |
| 8 | `positive_signal_ratio` | 0.0613 |
| 9 | `features_cat_demand` | 0.0522 |
| 10 | `credibility_confidence` | 0.0500 |
