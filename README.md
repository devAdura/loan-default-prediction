# Explainable loan risk prototype

A Streamlit academic prototype using the supplied historical Data Science Nigeria / OneFi loan files. The corrected run selects Gradient Boosting at a validation threshold of 0.26. Read [RESULTS.md](RESULTS.md) for the audit, measured results, verification and remaining limitations. Read [FEATURES.md](FEATURES.md) for all 18 predictor definitions.

## Setup

Use Python 3.13. The exact tested package versions are in `requirements-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The saved model is included so the app and synthetic tests can run without publishing lending records. Only load model files from a trusted source. The supplied dataset and local database are excluded from Git.

Set your own local password, then run the application. The username is `officer`.

```powershell
$env:DEMO_OFFICER_PASSWORD = Read-Host 'Choose a local prototype password'
streamlit run app.py
```

This setting creates the officer account on first use. Changing the environment value later does not reset an existing account. For a fresh prototype account, use a new `CREDIT_RISK_DATABASE` path. The default local database is `credit_risk.db`. Do not commit it.

## Reproduce training and figures

Place these original supplied CSVs in `data/onefi/` locally:

- `traindemographics.csv`
- `trainperf.csv`
- `trainprevloans.csv`

Then run:

```powershell
python train_nigeria_model.py
python generate_report_figures.py
python -m pytest -q
```

The separate competition test files are not required and are not used as external labelled validation. Compare source hashes in the saved metrics to verify the exact CSV version. Retraining replaces the local model and metrics. Preserve a run separately if needed before rerunning.

Training fits three independent candidates with train-only preprocessing, selects each threshold on validation F1 and ranks models on unrounded validation ROC-AUC with F1 as tie-breaker. It retains the selected fitted pipeline without refitting, preserving the validation-tuned score distribution. The test partition does not select models or thresholds.

Customer folds create approximately 60% training, 20% validation and 20% test. There is no customer intersection between partitions. All current applications are from July 2017, so this is a historical grouped internal holdout. It is not a future-period lending simulation. The cutoff for historical features is the current loan approval timestamp, with future repayment events censored.

The saved metrics retain exact split positions and version information. The figure generator verifies CSV fingerprints and exact metric reproduction before drawing model-specific charts. SHAP checks include the positive Bad class, output scale, baseline and additive reconstruction. Global SHAP is an aggregate of all held-out cases. Local research figures display six factors, while the application displays five.

## Public repository contents

Code, tests, aggregate metrics, the trained pipeline and regenerated figures can be prepared for publication. Raw CSVs, passwords, SQLite databases, row-level predictions and raw SHAP case inputs remain local. The original archive is preserved separately and identified by `source_archive.json`.

`artifacts/legacy_metrics.json` is the old duplicate-expanded run for historical comparison. The current experiment uses `artifacts/nigeria_metrics.json`. The old and new experiments differ in data construction, split and refitting policy, so score differences cannot be assigned to one correction.

## Limits

The prototype estimates the dataset's Bad label. It is not an automatic approval system or calibrated guarantee of default. Demographic snapshot dates and the target observation horizon are unavailable, so fully verified historical availability and future-period validation remain unresolved. Dataset licence and original release metadata must be confirmed separately before redistributing source records. No raw source records are included in this repository.

The Word report must be synchronized with the corrected run before submission. See the remaining report work in `RESULTS.md`.
