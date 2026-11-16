import os
import sqlite3
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
import app

ROOT=Path(__file__).resolve().parents[1]


def test_database_authentication_and_isolation(tmp_path,monkeypatch):
    monkeypatch.setenv('DEMO_OFFICER_PASSWORD','temporary-test-password')
    monkeypatch.setattr(app,'DATABASE',tmp_path/'test.db')
    app.initialise_database()
    assert app.verify_password('officer','temporary-test-password')
    assert not app.verify_password('officer','wrong')
    assert not app.verify_password("' OR 1=1 --",'temporary-test-password')
    assert app.hash_password('same')!=app.hash_password('same')
    app.save_assessment('officer','Higher risk',.7,{'loan_amount':100})
    app.save_assessment('other','Lower risk',.1,{})
    with app.db_connection() as db:
        rows=db.execute('SELECT risk_label FROM assessments WHERE officer=?',('officer',)).fetchall()
    assert rows==[('Higher risk',)]


def test_missing_shap_returns_empty_and_records_failure(monkeypatch):
    import explanations
    def fail(*args): raise RuntimeError('simulated unavailable SHAP')
    monkeypatch.setattr(explanations,'explain',fail)
    assert app.explanation(None,pd.DataFrame()).empty


def test_missing_password_fails_closed(tmp_path,monkeypatch):
    monkeypatch.delenv('DEMO_OFFICER_PASSWORD',raising=False)
    monkeypatch.setattr(app,'DATABASE',tmp_path/'test.db')
    with pytest.raises(ValueError): app.initialise_database()


@pytest.mark.skipif(not (ROOT/'artifacts/credit_risk_nigeria_model.joblib').exists(),reason='Train model before integration test')
def test_streamlit_login_assess_validate_history_and_model_info(tmp_path,monkeypatch):
    monkeypatch.setenv('DEMO_OFFICER_PASSWORD','temporary-functional-test')
    monkeypatch.setenv('CREDIT_RISK_DATABASE',str(tmp_path/'ui.db'))
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not at.exception
    at.text_input[0].set_value('officer')
    at.text_input[1].set_value('wrong')
    at.button[0].click().run()
    assert any('Invalid username' in e.value for e in at.error)
    at.text_input[1].set_value('temporary-functional-test')
    at.button[0].click().run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any('saved' in x.value for x in at.info)
    assert any('SHAP output scale' in x.value for x in at.caption)
    with sqlite3.connect(tmp_path/'ui.db') as db:
        saved=db.execute('SELECT input_data FROM assessments').fetchall()
    assert len(saved)==1
    import json
    assert json.loads(saved[0][0])['previous_repayment_rate']==.8
    # Invalid repayment total must not create a saved assessment.
    at.number_input[4].set_value(1000)
    at.button[0].click().run()
    assert any('repayment must cover principal' in e.value for e in at.error)
    at.number_input[4].set_value(34500)
    at.radio[0].set_value('No').run()
    at.button[0].click().run()
    assert any('Missing-history warning' in x.value for x in at.warning)
    with sqlite3.connect(tmp_path/'ui.db') as db:
        saved=db.execute('SELECT input_data FROM assessments ORDER BY id').fetchall()
    assert len(saved)==2
    assert json.loads(saved[-1][0])['previous_loan_count']==0
    at.sidebar.radio[0].set_value('Assessment history').run()
    assert not at.exception
    assert len(at.dataframe[0].value)==2
    at.sidebar.radio[0].set_value('Model information').run()
    assert not at.exception
    assert any(x.label=='Deployed model' for x in at.metric)
