# Corrected implementation results

The supplied implementation has been corrected and retrained. This run replaces the preliminary duplicate-expanded experiment. The model and threshold were selected on validation data, then frozen before test evaluation. The revised Word report has not been rewritten as part of this implementation task.

Run `20261116T181952Z` uses Gradient Boosting with threshold **0.26**. Predictions at or above the threshold are labelled Bad. This is a different experiment from the original Random Forest run, so changes in scores cannot be attributed to any single fix.

## Corrections made

- Deduplicated demographics by customer using field consensus and enforced a many-to-one join. Exact duplicate records no longer multiply applications.
- Aggregated previous loans separately for each application. Added approval-time filtering and censored future repayment events.
- Rejected unrecognized target labels and duplicate application IDs. Preserved unknown outcomes as missing rather than silently treating them as defaults.
- Used five stratified customer folds to create training, validation and test partitions. Kept preprocessing inside each training pipeline.
- Selected thresholds using validation F1, then ranked models using unrounded validation ROC-AUC with F1 as the tie-breaker. The selected fitted pipeline is retained without refitting.
- Added a majority-class baseline, full-precision metrics, saved predictions, source fingerprints, split positions and dependency versions.
- Fixed the app's percentage-to-fraction bug and stopped deriving current loan number from potentially incomplete history. Added validation for inconsistent input values.
- Added SHAP support for Logistic Regression, Random Forest and Gradient Boosting. The app reports output scale and reconstruction error. Failures are logged and shown to the user.
- Removed the hardcoded login password. Passwords use independent random salts. Databases, source records and private evidence are excluded from Git.

## Data audit

The source contains **4,368 applications**, **3,416 Good** and **952 Bad**. All 4,368 customers in the performance file are distinct. There are 4,346 demographic rows, including **12 exact duplicates**, representing 4,334 customers. Only eight duplicate demographic rows matched the performance table, which explains the previous inflated total of 4,376. There are **1,099 applications without a matching demographic record**. These are retained, with training-only imputation. The historical file contains 18,183 loans and no duplicate loan IDs. All 18,183 historical rows satisfy the earlier-loan checks in this supplied dataset, and none has a repayment after its matching current approval date. The cutoff fix protects against future-event leakage when other data is supplied, rather than demonstrating such leakage in these supplied rows.

| Partition | Applications | Customers | Good | Bad |
| --- | ---: | ---: | ---: | ---: |
| Train | 2620 | 2620 | 2049 | 571 |
| Validation | 874 | 874 | 684 | 190 |
| Test | 874 | 874 | 683 | 191 |

Customer intersections are zero for every partition pair. Each partition spans July 2017. This grouped holdout evaluates different customers from the same historical period. It does not establish future-period or external-lender performance.

## Validation comparison

| Candidate | Threshold | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.26 | 0.7278 | 0.5133 | 0.4753 | 0.5579 |
| Random Forest | 0.60 | 0.7147 | 0.4972 | 0.5298 | 0.4684 |
| Logistic Regression | 0.60 | 0.7050 | 0.4894 | 0.4946 | 0.4842 |

The selection rule identifies the winner for this split. It does not demonstrate statistically significant superiority.

## Held-out test results

| Metric | Selected model | Majority-class baseline |
| --- | ---: | ---: |
| accuracy | 0.7483 | 0.7815 |
| precision | 0.4338 | 0.0000 |
| recall | 0.4974 | 0.0000 |
| f1_score | 0.4634 | 0.0000 |
| roc_auc | 0.7076 | 0.5000 |
| balanced_accuracy | 0.6579 | 0.5000 |
| average_precision | 0.4226 | 0.2185 |
| pr_auc_trapezoid | 0.4181 | 0.6093 |

The confusion matrix is **TN 559, FP 124, FN 96, TP 95**. Of 191 actual Bad cases, 95 were identified and 96 were missed. Of 683 not-Bad cases, 559 were identified correctly and 124 were flagged incorrectly. Of 219 flagged applications, 124 were false positives, about 56.6%. Specificity is 81.8%. The model flags about 25.1% of applications.

The majority baseline achieves higher accuracy but detects no Bad cases. The selected threshold trades accuracy for Bad-class detection. ROC-AUC uses continuous scores. Average precision and trapezoidal PR-AUC are reported separately because their interpolation rules differ. A constant-score baseline has trapezoidal PR-AUC of about 0.609 because of the linear interpolation endpoint convention, while its average precision is the Bad prevalence, about 0.219. Use average precision for the primary baseline PR comparison.

## Explanation verification

SHAP was evaluated on all 874 held-out cases. Gradient Boosting explanations use raw log odds, a baseline of -1.4324641461 and tree-path-dependent feature dependence without a separate background dataset. The maximum all-feature additive reconstruction error was 5.78e-15. These are model attributions, not causal effects or probability percentage points.

Five private evidence cases cover true positive, true negative, false positive, false negative and threshold-adjacent predictions. Each records anonymous test case ID, raw predictor values, actual label, full-precision score, threshold rule and all contributions. The figure shows six factors, while the application shows five. Neither display normalizes contributions. Additivity is checked before reducing the display to the largest factors.

A global mean absolute SHAP figure covers all 874 test cases. The largest average attribution is observed historical repayment delay. This describes this model and sample, not a universal relationship.

## Verification performed

23 automated checks passed. These cover duplicate joins, multiple applications for one customer, future-event invariance, overdue-loan censoring, empty history, invalid dates and labels, duplicate IDs, explicit loan sequence, invalid inputs, threshold equality, customer separation, unseen categories and SHAP reconstruction for all three model types.

Streamlit AppTest exercised invalid and valid login, prediction with explanation, percentage conversion, invalid repayment totals, missing history, saved history and the model-information view. SQLite checks covered password verification and officer-specific assessment retrieval. Simulated explanation failure returns an empty explanation with a logged failure. This is functional testing, not a security audit or usability study. Browser screenshots were not captured.

All seven figures were regenerated from the saved model and split positions. The figure script verifies source fingerprints and exact test-metric reproduction before plotting.

## Remaining limitations and report work

- Demographic snapshots have no effective timestamps. The source cannot prove their historical availability. The model uses an explicit approval-time assumption for contracted terms.
- The target's observation window and label-availability timestamps are missing. A credible future-period evaluation requires those facts. A chronological split alone would not resolve label-maturity leakage.
- The archive does not establish the original dataset release version, licence or acquisition date. Source SHA-256 fingerprints identify the supplied CSV version without inventing provenance. Raw records remain excluded from the public repository.
- The separate OneFi competition test files are not an independent labelled external validation and were not used for tuning or evaluation.
- No calibration, confidence intervals, fairness audit, external evaluation or production readiness claim is made. New-to-credit and out-of-range lending remain poorly supported.
- Chapters 3 to 5, the abstract, tables and figures need to be synchronized with this new Gradient Boosting experiment. The report's original Random Forest and 876-row results must not be presented as the corrected experiment.
- Literature, references, equations, signatures and other editorial items from the supervisor review remain separate report work.

## Evidence and publication

`artifacts/nigeria_metrics.json` contains the full audit, partitions, metrics and source hashes. `artifacts/legacy_metrics.json` preserves the old aggregate results. `artifacts/shap_verification.json` records aggregate reconstruction checks. The saved model and regenerated figures accompany the code.

`artifacts/test_predictions.csv` and `artifacts/shap_cases.private.json` stay local and are ignored by Git. The raw CSV files and SQLite database are also ignored. No push or pull request has been made. Publishing requires the user's authorization.
