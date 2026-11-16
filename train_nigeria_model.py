"""Train and evaluate the Nigerian OneFi credit-risk model.

The model predicts the source dataset's Bad class. It is an academic decision-support
prototype, not a production lending model.
"""
from pathlib import Path
import json
import hashlib
import platform
from importlib.metadata import version
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, balanced_accuracy_score, auc, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nigeria_features import FEATURES, build_training_frame

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "onefi"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
CATEGORICAL = ["bank_account_type", "employment_status", "education"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def make_preprocessor():
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])


def fit_pipeline(classifier):
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])


def choose_threshold(y_true, probabilities):
    thresholds = np.arange(10, 81) / 100
    scores = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        scores.append((f1_score(y_true, pred, zero_division=0), recall_score(y_true, pred, zero_division=0), threshold))
    return max(scores)[2]


def evaluate(model, threshold, X, y):
    p = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return score_predictions(y, p, threshold)


def score_predictions(y, p, threshold):
    pred = (p >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "average_precision": float(average_precision_score(y, p)),
        "pr_auc_trapezoid": float(auc(recall[::-1], precision[::-1])),
    }


def make_partitions(X, y, groups):
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    test = folds[0][1]
    val = folds[1][1]
    train = np.sort(np.concatenate([folds[i][1] for i in [2, 3, 4]]))
    parts = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        if set(groups.iloc[parts[a]]) & set(groups.iloc[parts[b]]):
            raise ValueError("Customer overlap between partitions")
    if any(y.iloc[v].nunique() != 2 for v in parts.values()):
        raise ValueError("Every partition must contain both classes")
    return parts


def data_audit(demo, perf, prev, X, y):
    fields = ["birthdate", "bank_account_type", "employment_status_clients", "level_of_education_clients"]
    conflicts = {f: int((demo.groupby("customerid")[f].nunique() > 1).sum()) for f in fields}
    return {
        "source_rows": {"demographics": len(demo), "performance": len(perf), "previous_loans": len(prev)},
        "demographic_exact_duplicates": int(demo.duplicated().sum()),
        "demographic_duplicate_keys": int(demo.customerid.duplicated().sum()),
        "demographic_conflicting_customers_by_field": conflicts,
        "performance_duplicate_ids": int(perf.systemloanid.duplicated().sum()),
        "history_duplicate_ids": int(prev.systemloanid.duplicated().sum()),
        "unmatched_demographic_applications": int((~perf.customerid.isin(demo.customerid)).sum()),
        "retained_applications": len(X), "good": int((y==0).sum()), "bad": int(y.sum()),
        "feature_missing_counts": {c: int(X[c].isna().sum()) for c in X},
        "assessment_time": "Current approveddate. Approval-time contracted-term prototype, not a pre-approval application model.",
        "history_date_missing": {f: int(pd.to_datetime(prev[f], errors="coerce", format="mixed").isna().sum()) for f in ["approveddate","firstduedate","firstrepaiddate"]},
        "limitations": ["Demographic snapshots have no effective timestamps. Their historical availability cannot be independently established.", "Target observation horizon and label-availability timestamps are not supplied. Chronological outcome-maturity validation is not established.", "Source dataset licence and original release version are not present in the supplied archive."]
    }


def main():
    demo = pd.read_csv(DATA_DIR / "traindemographics.csv")
    perf = pd.read_csv(DATA_DIR / "trainperf.csv")
    prev = pd.read_csv(DATA_DIR / "trainprevloans.csv")
    X, y = build_training_frame(demo, perf, prev)

    audit = data_audit(demo, perf, prev, X, y)
    partitions = make_partitions(X, y, perf.customerid)
    train, val, test = [partitions[k] for k in ["train", "validation", "test"]]
    X_train, X_val, X_test = [X.iloc[i] for i in [train, val, test]]
    y_train, y_val, y_test = [y.iloc[i] for i in [train, val, test]]
    audit["partitions"] = {}
    for name, positions in partitions.items():
        rows = perf.iloc[positions]
        audit["partitions"][name] = {"rows": len(rows), "unique_customers": int(rows.customerid.nunique()), "bad": int(y.iloc[positions].sum()), "good": int((1-y.iloc[positions]).sum()), "approval_start": str(rows.approveddate.min()), "approval_end": str(rows.approveddate.max())}
    audit["customer_overlap"] = {a+"_"+b: len(set(perf.iloc[partitions[a]].customerid) & set(perf.iloc[partitions[b]].customerid)) for a,b in [("train","validation"),("train","test"),("validation","test")]}

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=2, subsample=0.9, random_state=42),
    }

    validation = []
    fitted = {}
    for name, clf in candidates.items():
        model = fit_pipeline(clf)
        model.fit(X_train, y_train)
        fitted[name] = model
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)
        val_metrics = evaluate(model, threshold, X_val, y_val)
        validation.append({"model": name, "threshold": float(threshold), **val_metrics})

    # Prefer ROC-AUC for ranking discrimination; F1 is the tie-breaker.
    validation.sort(key=lambda r: (r["roc_auc"], r["f1_score"]), reverse=True)
    best = validation[0]
    selected_name = best["model"]
    selected_threshold = best["threshold"]

    # Keep the exact validation-tuned pipeline frozen. Refitting can move scores
    # and would invalidate the threshold's relationship to this fitted model.
    final_model = fitted[selected_name]
    test_metrics = evaluate(final_model, selected_threshold, X_test, y_test)
    baseline = fit_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, 0.5, X_test, y_test)

    metrics = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "evaluation_design": "Five stratified customer folds: folds 2-4 train, fold 1 validation, fold 0 test. Historical July 2017 population, not a future-period evaluation. No refit after selection.",
        "validation_rows": len(X_val),
        "baseline_test_metrics": baseline_metrics,
        "data_audit": audit,
        "split_positions": partitions,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("train*.csv"))},
        "versions": {**{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "shap", "streamlit", "joblib"]}, "python": platform.python_version()},
        "dataset": "Data Science Nigeria / OneFi Credit Risk Prediction",
        "target": "Bad = 1 (source dataset's Bad class)",
        "deployed_model": selected_name,
        "decision_threshold": selected_threshold,
        "threshold_selection": "Threshold chosen on validation data to maximise F1-score; final metrics use an untouched test set.",
        "validation_model_comparison": validation,
        "test_metrics": test_metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "bad_rate": float(y.mean()),
        "feature_names": FEATURES,
        "important_limitations": [
            "The dataset is historical Nigerian OneFi lending data and should not be treated as a current national credit bureau dataset.",
            "Most training applications are repeat-loan applications; current loan number starts at 2. True new-to-credit applicants are underrepresented.",
            "The model should support, not replace, human lending review and applicable Nigerian credit policies.",
        ],
    }
    probabilities = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"case_id": [f"test-{i:04d}" for i in range(len(test))], "actual": y_test.to_numpy(), "probability": probabilities, "predicted": (probabilities >= selected_threshold).astype(int)}).to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    # A mean vector, not customer records, is the interventional background for LR.
    final_model.shap_background_mean_ = np.asarray(final_model.named_steps["preprocessor"].transform(X_train)).mean(axis=0)
    joblib.dump(final_model, ARTIFACT_DIR / "credit_risk_nigeria_model.joblib")
    (ARTIFACT_DIR / "nigeria_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
