import numpy as np
import pandas as pd
import pytest
from nigeria_features import build_previous_loan_features, build_training_frame, build_prediction_frame
from train_nigeria_model import make_partitions, choose_threshold, score_predictions, fit_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def current():
    return pd.DataFrame([dict(customerid='c', systemloanid=3, loannumber=3, approveddate='2017-03-01'), dict(customerid='c', systemloanid=4, loannumber=4, approveddate='2017-04-01')], index=[91, 13])


def history():
    return pd.DataFrame([
        dict(customerid='c',systemloanid=1,loannumber=1,approveddate='2017-01-01', firstduedate='2017-02-01',firstrepaiddate='2017-02-03', loanamount=100,totaldue=120,termdays=30),
        dict(customerid='c',systemloanid=2,loannumber=2,approveddate='2017-03-15', firstduedate='2017-04-15',firstrepaiddate='2017-04-10', loanamount=200,totaldue=240,termdays=30),
    ])


def applicant():
    return dict(age_years=35, bank_account_type='Savings', employment_status='Permanent',education='Graduate',loan_number=3,loan_amount=30000,loan_term_days=30,total_due=34500,previous_loan_count=2,previous_avg_amount=10000,previous_max_amount=10000,previous_avg_term_days=30,previous_total_amount=20000,previous_avg_cost_ratio=.2,previous_repayment_rate=.8,previous_avg_delay_days=2,previous_max_delay_days=7)


def test_history_is_per_application_and_preserves_index():
    got=build_previous_loan_features(current(),history())
    assert got.index.tolist()==[91,13]
    assert got.previous_loan_count.tolist()==[1,2]
    assert got.previous_total_amount.tolist()==[100,300]
    assert got.previous_repayment_rate.tolist()==[0,0] # April loan not yet due, repayment still future


def test_future_repayment_mutation_cannot_change_features():
    prev=history()
    before=build_previous_loan_features(current(),prev)
    prev.loc[1,'firstrepaiddate']='2030-01-01'
    pd.testing.assert_frame_equal(before,build_previous_loan_features(current(),prev))


def test_overdue_unpaid_delay_is_censored_at_assessment():
    prev=history().iloc[:1].copy()
    prev['firstrepaiddate']='2030-01-01'
    got=build_previous_loan_features(current(),prev)
    assert got.iloc[0].previous_avg_delay_days==28
    assert got.iloc[0].previous_repayment_rate==0


def test_invalid_approval_excluded_and_empty_schema():
    prev=history(); prev['approveddate']='invalid'
    got=build_previous_loan_features(current(),prev)
    assert got.previous_loan_count.eq(0).all()
    assert build_previous_loan_features(current(),prev.iloc[:0]).shape==(2,9)


def test_duplicate_history_exact_dedup_conflicts_rejected():
    prev=history()
    pd.testing.assert_frame_equal(build_previous_loan_features(current(),prev),build_previous_loan_features(current(),pd.concat([prev,prev])))
    extra=prev.iloc[:1].copy(); extra['loanamount']=999
    with pytest.raises(ValueError):
        build_previous_loan_features(current(),pd.concat([prev,extra]))


def source_frames():
    perf=current().copy(); perf['good_bad_flag']=['Good','Bad']; perf['loanamount']=100; perf['totaldue']=120; perf['termdays']=30
    demo=pd.DataFrame([dict(customerid='c',birthdate='1980-01-01',bank_account_type='Savings',employment_status_clients='Permanent',level_of_education_clients='Graduate')])
    return demo,perf,history()


def test_demo_join_does_not_multiply_rows_and_conflict_is_missing():
    demo,perf,prev=source_frames()
    X,y=build_training_frame(pd.concat([demo,demo]),perf,prev)
    assert len(X)==len(y)==2
    assert y.tolist()==[0,1]
    conflicting=demo.copy(); conflicting['birthdate']='1985-01-01'
    X,_=build_training_frame(pd.concat([demo,conflicting]),perf,prev)
    assert X.age_years.isna().all()


def test_unknown_targets_and_duplicate_applications_rejected():
    demo,perf,prev=source_frames()
    perf.loc[91,'good_bad_flag']='unknown'
    with pytest.raises(ValueError): build_training_frame(demo,perf,prev)
    perf['good_bad_flag']='Good'
    with pytest.raises(ValueError): build_training_frame(demo,pd.concat([perf,perf]),prev)


@pytest.mark.parametrize('update',[{'previous_repayment_rate':80},{'loan_amount':0},{'total_due':1},{'loan_number':2},{'age_years':np.nan},{'previous_max_amount':1},{'previous_total_amount':42}])
def test_prediction_rejects_inconsistent_inputs(update):
    row=applicant(); row.update(update)
    with pytest.raises(ValueError): build_prediction_frame(row)


def test_explicit_loan_number_and_zero_history():
    row=applicant(); row['loan_number']=12
    assert build_prediction_frame(row).loan_number.iloc[0]==12
    del row['loan_number']
    with pytest.raises(ValueError): build_prediction_frame(row)
    row=applicant(); row['previous_loan_count']=0
    got=build_prediction_frame(row)
    assert got.filter(like='previous_').eq(0).all().all()


def test_customer_split_and_threshold_boundary():
    X=pd.DataFrame({'x':np.arange(100)})
    y=pd.Series(np.tile([0,0,1,1],25))
    groups=pd.Series(np.repeat(np.arange(50),2))
    parts=make_partitions(X,y,groups)
    assert len(set(sum(parts.values(),[])))==100
    assert not set(groups.iloc[parts['train']]) & set(groups.iloc[parts['test']])
    result=score_predictions([0,1],np.array([.44,.44]),.44)
    assert result['confusion_matrix']==[[0,1],[0,1]]
    assert choose_threshold([0,1],np.array([.1,.7]))<=.7


@pytest.mark.parametrize('classifier',[LogisticRegression(),RandomForestClassifier(n_estimators=10,random_state=42),GradientBoostingClassifier(n_estimators=10,random_state=42)])
def test_shap_all_classifiers_additivity_and_unknown_category(classifier):
    from explanations import explain
    X=pd.concat([build_prediction_frame(applicant())]*30,ignore_index=True)
    X['loan_amount']=np.arange(30)*1000+1000
    y=pd.Series(np.arange(30)%2)
    model=fit_pipeline(classifier).fit(X,y)
    model.shap_background_mean_=model.named_steps['preprocessor'].transform(X).mean(axis=0)
    X.loc[0,'education']='Unseen category'
    grouped,metadata=explain(model,X.iloc[:2])
    assert grouped.shape==(2,18)
    assert metadata['max_reconstruction_error']<1e-6
