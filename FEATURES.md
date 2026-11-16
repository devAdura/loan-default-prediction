# Feature definitions

The assessment timestamp is the current loan approval date. This is an approval-time prototype using contracted terms. The data cannot establish suitability for decisions before approval. Demographic snapshots lack effective timestamps, so their historical availability remains an assumption.

A historical loan must have both a lower loan number and an approval strictly earlier than the current assessment. Its ID must differ from the current loan. Aggregation is by application row, preserving source order and row index. Exact historical duplicates are removed. Conflicting historical IDs and duplicate current application IDs are rejected.

Repayments after assessment cannot influence features. An outcome is observable if a repayment has occurred by assessment or its first due date has passed. Not-yet-due unpaid loans contribute borrowing terms but have unknown repayment outcomes. Zero observable history uses zeros. Unknown outcomes on observed loans remain missing and are imputed using training data only. No observed history does not prove no prior borrowing.

| Code name | Type | Unit | Source | Calculation |
| --- | --- | --- | --- | --- |
| age_years | float | years | birthdate | Age at current approveddate, days divided by 365.25, clipped to 18 to 100 |
| bank_account_type | category | text | demographics | Customer consensus value, Unknown if missing or conflicting |
| employment_status | category | text | employment_status_clients | Customer consensus value, Unknown if missing or conflicting |
| education | category | text | level_of_education_clients | Customer consensus value, Unknown if missing or conflicting |
| loan_number | float | sequence | current loannumber | Actual lender sequence, never inferred from observed history |
| loan_amount | float | NGN | current loanamount | Contracted principal |
| loan_term_days | float | days | current termdays | Contracted term |
| total_due | float | NGN | current totaldue | Contracted amount due, assumed known at approval |
| loan_cost_ratio | float | fraction | current principal and total due | max(totaldue / loanamount - 1, 0), zero principal becomes missing |
| previous_loan_count | float | loans | historical systemloanid | Count distinct valid earlier loans |
| previous_avg_amount | float | NGN | historical loanamount | Mean earlier principal |
| previous_max_amount | float | NGN | historical loanamount | Maximum earlier principal |
| previous_avg_term_days | float | days | historical termdays | Mean earlier term |
| previous_repayment_rate | float | fraction | historical first due and repayment dates | Mean on-time indicator over observable outcomes only |
| previous_avg_delay_days | float | days | historical first due and repayment dates | Mean nonnegative first-repayment delay known at assessment, unpaid overdue delays censored at assessment |
| previous_max_delay_days | float | days | historical first due and repayment dates | Maximum observed or censored delay |
| previous_total_amount | float | NGN | historical loanamount | Sum earlier principal borrowed, not repayments received |
| previous_avg_cost_ratio | float | fraction | historical totaldue and loanamount | Mean max(totaldue / loanamount - 1, 0) |

All historical terms are assumed fixed at original approval. Current outcomes and direct identifiers are excluded from predictors. Invalid birthdates become missing. Missing ages and numeric values are median-imputed inside the fitted training pipeline. Missing categories become Unknown. The fitted encoder tolerates unseen categories. Unknown target labels raise an error.

The app requires actual current loan sequence separately from observed history count. It converts the repayment percentage to a fraction and checks amounts, rates, history counts, averages and maxima before predicting. Zero observed history clears submitted history values. Input history must be measured at the same approval-time cutoff and with the same outcome definitions as training.
