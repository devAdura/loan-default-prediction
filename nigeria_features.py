"""Feature engineering for the Nigerian OneFi credit-risk dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "age_years": "Applicant age",
    "bank_account_type": "Bank account type",
    "employment_status": "Employment status",
    "education": "Education level",
    "loan_number": "Current loan number",
    "loan_amount": "Loan amount",
    "loan_term_days": "Loan term",
    "total_due": "Total amount due",
    "loan_cost_ratio": "Loan cost relative to amount",
    "previous_loan_count": "Previous loan count",
    "previous_avg_amount": "Average previous loan amount",
    "previous_max_amount": "Largest previous loan amount",
    "previous_avg_term_days": "Average previous loan term",
    "previous_repayment_rate": "On-time repayment rate",
    "previous_avg_delay_days": "Average repayment delay",
    "previous_max_delay_days": "Largest repayment delay",
    "previous_total_amount": "Total previous borrowing",
    "previous_avg_cost_ratio": "Average previous loan cost",
}

FEATURES = list(DISPLAY_NAMES)


def _age_from_birthdate(series: pd.Series, reference_dates: pd.Series) -> pd.Series:
    birth = pd.to_datetime(series, errors="coerce")
    ref = pd.to_datetime(reference_dates, errors="coerce")
    return ((ref - birth).dt.days / 365.25).clip(18, 100)


HISTORY = [f for f in FEATURES if f.startswith("previous_")]
DEMO_FIELDS = ["birthdate", "bank_account_type", "employment_status_clients", "level_of_education_clients"]


def deduplicate_demographics(demographics):
    """One customer row. Conflicting field values become missing, never guessed."""
    if demographics["customerid"].isna().any():
        raise ValueError("Missing demographic customer key")
    def consensus(values):
        unique = values.dropna().unique()
        return unique[0] if len(unique) == 1 else np.nan
    return demographics.groupby("customerid", as_index=False)[DEMO_FIELDS].agg(consensus)


def assessment_dates(current):
    # Approval-time prototype: current contracted terms are assumed known here.
    dates = pd.to_datetime(current["approveddate"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Every application needs a valid approval assessment timestamp")
    return dates


def build_previous_loan_features(current, previous):
    """Per-application history, observed at the current approval timestamp.

    A loan must have a lower sequence number AND an earlier approval. Future
    repayments are censored. Unpaid overdue loans accrue observed delay up to
    assessment, while not-yet-due unpaid loans have unknown repayment outcomes.
    """
    result = pd.DataFrame(0.0, index=range(len(current)), columns=HISTORY)
    result.index = current.index
    if previous.empty:
        return result
    prev = previous.drop_duplicates().copy()
    if prev["systemloanid"].isna().any() or prev["systemloanid"].duplicated().any():
        raise ValueError("Historical loan IDs must be unique after exact deduplication")
    if prev["customerid"].isna().any():
        raise ValueError("Missing historical customer key")
    for field in ["approveddate", "firstduedate", "firstrepaiddate"]:
        prev[field] = pd.to_datetime(prev[field], errors="coerce", format="mixed")
    prev["cost_ratio"] = (prev["totaldue"] / prev["loanamount"].replace(0, np.nan) - 1).clip(lower=0)
    base = current[["customerid", "systemloanid", "loannumber"]].reset_index(drop=True).copy()
    base["application_row"] = np.arange(len(base))
    base["assessed_at"] = assessment_dates(current).to_numpy()
    merged = base.merge(prev, on="customerid", suffixes=("_current", "_prev"), validate="many_to_many")
    merged = merged.loc[
        (merged.loannumber_prev < merged.loannumber_current)
        & (merged.approveddate < merged.assessed_at)
        & (merged.systemloanid_prev != merged.systemloanid_current)
    ].copy()
    if merged.empty:
        return result
    repaid_known = merged.firstrepaiddate.notna() & (merged.firstrepaiddate <= merged.assessed_at)
    due_known = merged.firstduedate.notna()
    outcome_known = due_known & (repaid_known | (merged.firstduedate <= merged.assessed_at))
    merged["on_time"] = np.where(outcome_known, (repaid_known & (merged.firstrepaiddate <= merged.firstduedate)).astype(float), np.nan)
    end = merged.firstrepaiddate.where(repaid_known, merged.assessed_at)
    merged["delay_days"] = ((end - merged.firstduedate).dt.total_seconds() / 86400).clip(lower=0).where(outcome_known)
    agg = merged.groupby("application_row").agg(
        previous_loan_count=("systemloanid_prev", "count"),
        previous_avg_amount=("loanamount", "mean"),
        previous_max_amount=("loanamount", "max"),
        previous_avg_term_days=("termdays", "mean"),
        previous_repayment_rate=("on_time", "mean"),
        previous_avg_delay_days=("delay_days", "mean"),
        previous_max_delay_days=("delay_days", "max"),
        previous_total_amount=("loanamount", "sum"),
        previous_avg_cost_ratio=("cost_ratio", "mean"),
    )
    # Preserve unknown outcomes on observed history as NaN for train-only imputation.
    for position, row in agg.iterrows():
        result.iloc[int(position)] = row
    return result


def build_training_frame(demographics, performance, previous):
    """Exactly one feature row per unique labelled application, in source order."""
    perf = performance.copy()
    if perf[["customerid", "systemloanid"]].isna().any().any() or perf.systemloanid.duplicated().any():
        raise ValueError("Performance application IDs must be unique and customer keys present")
    labels = perf.good_bad_flag.astype("string").str.strip().str.lower()
    if not labels.isin(["good", "bad"]).all():
        raise ValueError("Unknown or missing target label")
    demo = deduplicate_demographics(demographics)
    merged = perf.merge(demo, on="customerid", how="left", validate="many_to_one", sort=False)
    merged.index = performance.index
    merged["age_years"] = _age_from_birthdate(merged.birthdate, assessment_dates(merged))
    history = build_previous_loan_features(merged, previous)
    for field in HISTORY:
        merged[field] = history[field].to_numpy()
    for target, source in [("employment_status", "employment_status_clients"), ("education", "level_of_education_clients"), ("bank_account_type", "bank_account_type")]:
        merged[target] = merged[source].fillna("Unknown")
    for target, source in [("loan_number", "loannumber"), ("loan_amount", "loanamount"), ("loan_term_days", "termdays"), ("total_due", "totaldue")]:
        merged[target] = pd.to_numeric(merged[source], errors="coerce")
    merged["loan_cost_ratio"] = (merged.total_due / merged.loan_amount.replace(0, np.nan) - 1).clip(lower=0)
    X = merged[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    y = labels.eq("bad").astype(int)
    return X, y


def build_prediction_frame(applicant: dict, previous_summary: dict | None = None):
    row = dict(applicant)
    row.update(previous_summary or {})
    if "loan_number" not in row:
        raise ValueError("Supply actual current loan number, not observed history count plus one")
    for field in HISTORY:
        row.setdefault(field, 0)
    numeric = [f for f in FEATURES if f not in {"bank_account_type", "employment_status", "education", "loan_cost_ratio"}]
    for field in numeric:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Provide a numeric {field}") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {field}")
        row[field] = value
    if not 18 <= row["age_years"] <= 100:
        raise ValueError("Applicant age must be between 18 and 100")
    if row["loan_number"] < 1 or not row["loan_number"].is_integer():
        raise ValueError("Current loan number must be a positive integer")
    if not row["previous_loan_count"].is_integer() or row["previous_loan_count"] >= row["loan_number"]:
        raise ValueError("Observed history count must be less than current loan number")
    if row["loan_amount"] <= 0 or row["loan_term_days"] <= 0 or row["total_due"] < row["loan_amount"]:
        raise ValueError("Loan amount and term must be positive, and repayment must cover principal")
    if not 0 <= row["previous_repayment_rate"] <= 1:
        raise ValueError("Repayment rate must be a fraction from zero to one")
    if row["previous_loan_count"] == 0:
        for field in HISTORY:
            row[field] = 0
    else:
        if row["previous_max_amount"] < row["previous_avg_amount"] or row["previous_max_delay_days"] < row["previous_avg_delay_days"]:
            raise ValueError("Historical maximum must be at least the average")
        if not np.isclose(row["previous_total_amount"], row["previous_avg_amount"] * row["previous_loan_count"], atol=1.0):
            raise ValueError("Previous total must equal observed count times average amount")
    row["loan_cost_ratio"] = max(row["total_due"] / row["loan_amount"] - 1, 0)
    return pd.DataFrame([row], columns=FEATURES)
