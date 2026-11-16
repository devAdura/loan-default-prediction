"""Streamlit dashboard for an explainable Nigerian credit-risk assessment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import hmac
import json
import os
import secrets
import logging
import sqlite3

import joblib
import pandas as pd
import streamlit as st

from nigeria_features import DISPLAY_NAMES, build_prediction_frame

ROOT = Path(__file__).parent
ARTIFACT_DIR = ROOT / "artifacts"
DATABASE = Path(os.getenv("CREDIT_RISK_DATABASE", str(ROOT / "credit_risk.db")))
MODEL_PATH = ARTIFACT_DIR / "credit_risk_nigeria_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "nigeria_metrics.json"

OPTIONS = {
    "bank_account_type": {"Savings": "Savings account", "Current": "Current account", "Other": "Other account type"},
    "employment_status": {"Permanent": "Permanent employee", "Self-Employed": "Self-employed", "Contract": "Contract worker", "Student": "Student", "Unemployed": "Unemployed", "Retired": "Retired", "Unknown": "Not provided"},
    "education": {"Primary": "Primary education", "Secondary": "Secondary education", "Graduate": "University graduate", "Post-Graduate": "Postgraduate", "Unknown": "Not provided"},
}

INPUT_FIELDS = [
    "age_years", "bank_account_type", "employment_status", "education",
    "loan_number", "loan_amount", "loan_term_days", "total_due",
    "previous_loan_count", "previous_avg_amount", "previous_max_amount",
    "previous_avg_term_days", "previous_repayment_rate", "previous_avg_delay_days",
    "previous_max_delay_days", "previous_total_amount", "previous_avg_cost_ratio",
]

PLAIN_NAMES = {
    "age_years": "Applicant age",
    "bank_account_type": "Bank account type",
    "employment_status": "Employment status",
    "education": "Education level",
    "loan_amount": "Requested loan amount",
    "loan_term_days": "Loan term (days)",
    "total_due": "Expected total repayment",
    "previous_loan_count": "Number of previous loans",
    "previous_avg_amount": "Average previous loan amount",
    "previous_max_amount": "Largest previous loan amount",
    "previous_avg_term_days": "Average previous loan term",
    "previous_repayment_rate": "On-time repayment rate",
    "previous_avg_delay_days": "Average repayment delay",
    "previous_max_delay_days": "Largest repayment delay",
    "previous_total_amount": "Total previous borrowing",
    "previous_avg_cost_ratio": "Average previous loan cost",
}


def db_connection():
    return sqlite3.connect(DATABASE, check_same_thread=False)


def initialise_database():
    with db_connection() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, officer TEXT NOT NULL,
            assessed_at TEXT NOT NULL, risk_label TEXT NOT NULL,
            default_probability REAL NOT NULL, input_data TEXT NOT NULL
        )""")
        default_password = os.getenv("DEMO_OFFICER_PASSWORD")
        if not default_password:
            raise ValueError("Set DEMO_OFFICER_PASSWORD before starting the prototype")
        connection.execute(
            "INSERT OR IGNORE INTO users VALUES (?, ?, ?)",
            ("officer", hash_password(default_password), "Loan Officer"),
        )


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return salt + "$" + digest


def verify_password(username: str, password: str) -> bool:
    with db_connection() as connection:
        row = connection.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row or "$" not in row[0]:
        return False
    salt = row[0].split("$", 1)[0]
    return hmac.compare_digest(row[0], hash_password(password, salt))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def money(value):
    return f"₦{value:,.0f}"


def select_option(field, label, help_text=None):
    value = st.selectbox(label, list(OPTIONS[field]), format_func=lambda x: OPTIONS[field][x], help=help_text)
    return value


def applicant_form():
    # Inputs rerun immediately so the history selection can lock its fields.
    with st.container(border=True):
        st.subheader("Applicant and loan details")
        st.caption("Use approved loan terms and history observable at the approval assessment timestamp. Do not treat the model output as an automatic approval or rejection.")
        st.markdown("**Applicant information**")
        columns = st.columns(2, gap="medium")
        with columns[0]:
            age_years = st.number_input("Applicant age (years)", min_value=18, max_value=100, value=35)
        with columns[1]:
            bank_account_type = select_option("bank_account_type", "Bank account type")
        columns = st.columns(2, gap="medium")
        with columns[0]:
            employment_status = select_option("employment_status", "Employment status")
        with columns[1]:
            education = select_option("education", "Education level")
        loan_number = st.number_input("Current loan number from lender records", min_value=1, max_value=1000, value=3, help="Use the actual loan sequence number. Observed history may be incomplete.")
        st.markdown("**Requested loan**")
        columns = st.columns(3, gap="medium")
        with columns[0]:
            loan_amount = st.number_input("Requested loan amount (₦)", min_value=1000, max_value=1_000_000, value=30000, step=1000, help="Amount the applicant wants to borrow.")
        with columns[1]:
            loan_term_days = st.number_input("Loan term (days)", min_value=7, max_value=365, value=30)
        with columns[2]:
            total_due = st.number_input("Expected total repayment (₦)", min_value=1000, max_value=1_500_000, value=34500, step=1000, help="Total amount expected to be repaid, including the loan cost shown by the lender.")
        st.divider()
        st.subheader("Previous loan history")
        st.caption("Previous repayment behaviour is one of the strongest sources of credit information.")
        has_previous = st.radio("Is prior-loan history available for this assessment?", ["Yes", "No"], horizontal=True, key="has_previous_loan")
        has_history = has_previous == "Yes"
        columns = st.columns(2, gap="medium")
        with columns[0]:
            previous_loan_count = st.number_input(
                "Number of previous loans", min_value=0, max_value=30,
                value=2 if has_history else 0, disabled=not has_history, key="previous_loan_count"
            )
        with columns[1]:
            previous_avg_term_days = st.number_input(
                "Average previous loan term (days)", min_value=0, max_value=365,
                value=30 if has_history else 0, disabled=not has_history, key="previous_avg_term_days"
            )
        columns = st.columns(2, gap="medium")
        with columns[0]:
            previous_avg_amount = st.number_input(
                "Average previous loan amount (₦)", min_value=0, max_value=1_000_000,
                value=20000 if has_history else 0, step=1000, disabled=not has_history, key="previous_avg_amount"
            )
        with columns[1]:
            previous_max_amount = st.number_input(
                "Largest previous loan amount (₦)", min_value=0, max_value=1_000_000,
                value=30000 if has_history else 0, step=1000, disabled=not has_history, key="previous_max_amount"
            )
        columns = st.columns(2, gap="medium")
        with columns[0]:
            previous_total_amount = st.number_input(
                "Total previous borrowing (₦)", min_value=0, max_value=5_000_000,
                value=int(previous_avg_amount * previous_loan_count) if has_history else 0,
                step=1000, disabled=not has_history, key="previous_total_amount"
            )
        with columns[1]:
            previous_avg_cost_ratio = st.number_input(
                "Average previous loan cost (%)", min_value=0.0, max_value=100.0,
                value=20.0 if has_history else 0.0, step=1.0,
                help="Average increase from the previous loan amount to the total amount due. For example, 20% means ₦10,000 became ₦12,000 due.",
                disabled=not has_history, key="previous_avg_cost_ratio"
            ) / 100
        columns = st.columns(2, gap="medium")
        with columns[0]:
            previous_avg_delay_days = st.number_input(
                "Average repayment delay (days)", min_value=0.0, max_value=365.0,
                value=2.0 if has_history else 0.0, step=1.0, disabled=not has_history, key="previous_avg_delay_days"
            )
        with columns[1]:
            previous_max_delay_days = st.number_input(
                "Largest repayment delay (days)", min_value=0.0, max_value=365.0,
                value=7.0 if has_history else 0.0, step=1.0, disabled=not has_history, key="previous_max_delay_days"
            )
        previous_repayment_rate = st.slider(
            "On-time repayment rate", 0, 100, 80 if has_history else 0, format="%d%%",
            help="Percentage of previous loans with an observable outcome at assessment that were repaid by the first due date.",
            disabled=not has_history, key="previous_repayment_rate"
        )
        previous_repayment_rate = previous_repayment_rate / 100
        if not has_history:
            st.info("Previous-loan fields are locked. The assessment will use zero observed history. This may mean incomplete records rather than no previous borrowing. Extra human review is recommended.")
        submitted = st.button("Assess credit risk", type="primary")
    values = locals()
    applicant = {key: values[key] for key in INPUT_FIELDS}
    if not has_history:
        applicant.update({key: 0 for key in INPUT_FIELDS if key.startswith("previous_")})
    return submitted, applicant


def explanation(model, applicant: pd.DataFrame):
    try:
        from explanations import explain
        grouped, metadata = explain(model, applicant)
        row = grouped.iloc[0].rename_axis("base_feature").reset_index(name="impact")
        row["feature"] = row.base_feature.map(PLAIN_NAMES).fillna(row.base_feature.map(DISPLAY_NAMES))
        result = row.sort_values("impact", key=lambda x: x.abs(), ascending=False).head(5)
        result.attrs["shap_metadata"] = metadata
        return result
    except Exception:
        logging.exception("SHAP explanation failed")
        return pd.DataFrame(columns=["feature", "impact"])


def save_assessment(officer, risk_label, probability, applicant):
    with db_connection() as connection:
        connection.execute(
            "INSERT INTO assessments (officer, assessed_at, risk_label, default_probability, input_data) VALUES (?, ?, ?, ?, ?)",
            (officer, datetime.now(timezone.utc).isoformat(), risk_label, float(probability), json.dumps(applicant)),
        )


def show_history(username):
    with db_connection() as connection:
        history = pd.read_sql_query(
            "SELECT id, assessed_at, officer, risk_label, default_probability FROM assessments WHERE officer = ? ORDER BY id DESC",
            connection, params=(username,),
        )
    st.subheader("Saved assessment history")
    if history.empty:
        st.info("No assessments have been saved yet.")
        return
    history = history.rename(columns={"assessed_at": "Date", "risk_label": "Risk classification", "default_probability": "Estimated risk"})
    history["Estimated risk"] = (history["Estimated risk"] * 100).round(1).astype(str) + "%"
    st.dataframe(history[["id", "Date", "Risk classification", "Estimated risk"]], width="stretch", hide_index=True)


def login_screen():
    st.title("Nigerian Credit Risk Assessment")
    st.caption("Explainable AI academic prototype for loan officers")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if verify_password(username, password):
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.info("Sign in as officer using the password configured by the person running this prototype.")


def model_information():
    st.header("About this model")
    st.write("This version uses the Data Science Nigeria / OneFi credit-risk dataset rather than the older German credit dataset. Loan amounts are displayed in Nigerian naira because the training data contains Nigerian lending records.")
    st.warning("This is a historical academic dataset, not a live Nigerian credit-bureau feed. Model performance can change when lending behaviour, prices, policies, or applicant populations change.")
    st.subheader("Important limitation: new-to-credit applicants")
    st.write("Most training records are repeat-loan applications. The dataset has very few applicants with no previous loan history. Therefore, the model should not be treated as highly reliable for someone who has never borrowed before.")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("Deployed model", metrics["deployed_model"])
        b.metric("Test ROC-AUC", f"{metrics['test_metrics']['roc_auc']:.3f}")
        c.metric("Bad-loan rate", f"{metrics['bad_rate']:.1%}")
        st.subheader("Final test results")
        tm = metrics["test_metrics"]
        st.dataframe(pd.DataFrame([{
            "Accuracy": tm["accuracy"], "Precision": tm["precision"], "Recall": tm["recall"], "F1-score": tm["f1_score"], "ROC-AUC": tm["roc_auc"],
        }]), width="stretch", hide_index=True)
        st.caption(f"Risk classification threshold: {metrics['decision_threshold']:.0%}. It was selected on validation data to balance precision and recall using F1-score.")
    st.subheader("What the model uses")
    st.write("Age, bank account type, employment status, education, current loan amount and term, expected repayment, and summaries of previous borrowing and repayment behaviour.")
    st.subheader("What the model does not mean")
    st.write("A higher model score is not proof that a person will default. It is a statistical estimate based on patterns in historical data and should be reviewed alongside the lender's normal checks and policies.")


def main():
    st.set_page_config(page_title="Nigerian Credit Risk XAI", page_icon="📊", layout="wide")
    try:
        initialise_database()
    except ValueError as exc:
        st.error(str(exc))
        return
    if not MODEL_PATH.exists():
        st.error("The Nigerian model has not been trained yet.")
        st.code("python train_nigeria_model.py\nstreamlit run app.py")
        return
    if "user" not in st.session_state:
        login_screen()
        return

    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {"decision_threshold": 0.5}
    threshold = float(metrics.get("decision_threshold", 0.5))

    st.sidebar.title("Nigerian Credit Risk XAI")
    st.sidebar.write(f"Signed in as **{st.session_state.user}**")
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    page = st.sidebar.radio("Navigate", ["New assessment", "Assessment history", "Model information"])
    if page == "Assessment history":
        show_history(st.session_state.user)
        return
    if page == "Model information":
        model_information()
        return

    st.header("New credit-risk assessment")
    st.caption("Decision-support only. Do not use the model as the sole basis for approving or rejecting a loan.")
    submitted, raw = applicant_form()
    if not submitted:
        return

    try:
        data = build_prediction_frame(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    probability = float(model.predict_proba(data)[0, 1])
    risk_label = "Higher risk" if probability >= threshold else "Lower risk"

    if raw["previous_loan_count"] == 0 and raw["loan_number"] > 1:
        st.warning("Missing-history warning: loan sequence indicates earlier borrowing, but no history was supplied. Zero observed history does not mean no previous loans.")
    if raw["loan_number"] == 1:
        st.warning("New-to-credit warning: the training data contains very few applicants with no previous loan history. Treat this result as limited evidence and carry out normal human review.")
    if raw["loan_amount"] > 60000:
        st.warning("Loan amount warning: the historical training records top out at about ₦60,000. Predictions for substantially larger amounts may be less reliable.")

    if probability >= threshold:
        st.error(f"Model classification: {risk_label}")
    else:
        st.success(f"Model classification: {risk_label}")
    st.metric("Estimated risk of a bad loan", f"{probability:.1%}")
    st.caption("This percentage is a model score based on historical data; it is not a guarantee that the applicant will or will not repay.")

    st.subheader("Why the model reached this result")
    contributors = explanation(model, data)
    if contributors.empty:
        st.warning("The explanation could not be generated in this environment. The prediction and history were still saved.")
    else:
        contributors["effect"] = contributors.impact.apply(lambda x: "Pushed the risk estimate higher" if x > 0 else ("Pushed the risk estimate lower" if x < 0 else "No change"))
        contributors["relative_influence"] = contributors.impact.abs().round(3)
        st.dataframe(contributors[["feature", "effect", "relative_influence"]].rename(columns={"feature": "Factor", "effect": "Effect on estimate", "relative_influence": "Relative influence"}), hide_index=True, width="stretch")
        metadata = contributors.attrs["shap_metadata"]
        st.caption(f"SHAP output scale: {metadata['scale']}. Baseline: {metadata['baseline']:.6f}. All-feature additive reconstruction error: {metadata['max_reconstruction_error']:.2g}.")
        st.caption("The relative influence values are absolute SHAP attributions without normalization. They show which submitted factors influenced this particular prediction most; they are not percentages of default probability.")

    save_assessment(st.session_state.user, risk_label, probability, raw)
    st.info("This assessment has been saved in the history page.")


if __name__ == "__main__":
    main()
