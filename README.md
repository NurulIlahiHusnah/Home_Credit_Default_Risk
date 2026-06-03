# 🏦 Credit Scoring Model — Home Credit Default Risk

**Objective:** Reduce false rejection of creditworthy applicants by building a robust credit default prediction model aligned with OJK regulatory standards.

---

## Problem Statement

Home Credit Indonesia faced a selection bias problem: the existing system systematically rejected creditworthy applicants whose profiles had never been seen in training data. This project addresses that by:

- Reducing the **False Rejection Rate** — good applicants wrongly denied credit
- Maintaining credit risk control through regulatory-grade metrics (KS Statistic, per OJK requirements)
- Correcting historical selection bias via **Reject Inference**

---

## Dataset

Source: [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

| Table | Description |
|-------|-------------|
| `application_train / test` | Primary applicant data; target: 0 = on-time, 1 = payment difficulties |
| `bureau` + `bureau_balance` | External credit bureau history (joined on SK_ID_BUREAU) |
| `previous_application` | Previous loan applications at Home Credit |
| `installments_payments` | Historical installment payment records |
| `POS_CASH_balance` | Monthly balances for POS and cash installment loans |
| `credit_card_balance` | Credit card utilization and payment history |

- **Primary key:** `SK_ID_CURR` (unique applicant identifier across all tables)
- **Class imbalance:** ~8% default (TARGET = 1) vs ~92% non-default
- **Scale:** Tens of millions of records across all tables

---

## Methodology — 9-Phase Pipeline

```
PHASE 1 — Leakage Audit         : Identify and remove post-decision columns
PHASE 2 — Preprocessing         : Clean, encode, impute, scale (anti-leakage)
PHASE 3 — Reject Inference      : Label Propagation to correct selection bias
PHASE 4 — Baseline Model        : Logistic Regression
PHASE 5 — Main Model            : XGBoost + SHAP
PHASE 6 — Regulatory Evaluation : KS, AUC, Precision@R70, Log Loss, F1
PHASE 7 — Threshold Tuning      : Minimize False Rejection Rate
PHASE 8 — SHAP Explanation      : Feature importance for stakeholders & regulators
PHASE 9 — Submission + Summary  : Final artifacts and results
```

---

## Key Results

| Metric | Logistic Regression | XGBoost | Target | Status |
|--------|:-------------------:|:-------:|:------:|:------:|
| ROC-AUC | 0.7622 | **0.7679** | > 0.75 | ✅ |
| KS Statistic | 0.3866 | **0.4026** | > 0.35 | ✅ |
| Precision@R70 | 0.8930 | **0.8931** | > 0.30 | ✅ |
| Log Loss | 0.5795 | **0.4573** | Lower is better | ✅ |
| F1 Score | 0.2682 | 0.3043 | — | — |

> **Note on F1:** F1 is computed at the default 0.5 threshold, which is not the operational decision threshold for imbalanced credit data (~8% positive rate). The operational threshold is optimized in Phase 7 to minimize False Rejection Rate while maintaining Precision@approved ≥ 30%. AUC, KS, and Precision@R70 are the primary evaluation metrics for this use case.

**Recommended model:** XGBoost with Reject Inference — consistently outperforms Logistic Regression across all metrics. Logistic Regression is retained as an interpretable baseline for regulatory reporting and KS statistic computation.

---

## Technical Highlights

### Reject Inference (Label Propagation)
Training data contains only approved applicants, introducing selection bias: the model never learns from rejected profiles and therefore keeps rejecting unfamiliar but potentially creditworthy applicants. To correct this, Label Propagation identifies applicants from the unlabeled test set whose feature profiles are similar to known non-defaults, and adds them to training with a reduced sample weight (0.3). This expands the model's knowledge beyond the historically approved population and reduces false rejection of novel profiles.

### Anti-Leakage Preprocessing
Post-decision features — installment performance, active credit card balances, current DPD from running loans — are explicitly identified and removed before training. These features are only observable after a loan is approved and active, meaning their presence in a model would simulate perfect foresight and cause significant data leakage.

### Regulatory Compliance (OJK)
- **KS Statistic** computed and reported per OJK credit scoring requirements (target > 0.35; achieved 0.4026 — Excellent range)
- **Logistic Regression** provides interpretable coefficients for regulatory and stakeholder review
- **SHAP values** explain individual XGBoost predictions and global feature importance

### Threshold Tuning
The decision threshold is optimized on the validation set to minimize False Rejection Rate (FRR) while maintaining approval quality — directly answering the business objective of reducing false rejections without sacrificing acceptable credit risk levels.

---

## Tech Stack

```
Python · pandas · numpy · scikit-learn · XGBoost · SHAP
imbalanced-learn · category_encoders · matplotlib · seaborn · joblib
```

---

## How to Run

1. Place all raw data tables in the `/content/` directory
2. Run the aggregation notebook first to generate `app_train_final.parquet` and `app_test_final.parquet`
3. Open `Preprocessing_Modeling_HomeCredit.ipynb` in Google Colab or Jupyter Notebook
4. Run all cells sequentially — all artifacts (models, scaler, threshold, SHAP) are saved automatically

---

## File Structure

```
├── Preprocessing_Modeling_HomeCredit.ipynb   # Main pipeline notebook
├── app_train_final.parquet                   # Preprocessed training data
├── app_test_final.parquet                    # Preprocessed test data
├── model_xgb.pkl                             # Trained XGBoost model
├── model_lr.pkl                              # Trained Logistic Regression model
├── scaler.pkl                                # Fitted StandardScaler
├── target_encoder.pkl                        # Fitted TargetEncoder
├── imputer.pkl                               # Fitted median imputer
├── optimal_threshold.pkl                     # Optimized decision threshold
├── feature_cols.pkl                          # Final feature column list
├── shap_importance.csv                       # SHAP feature importance scores
├── leakage_audit.csv                         # Documented leakage columns
└── submission.csv                            # Kaggle submission file
```

---

*[Portfolio](https://nurulilahihusnah.github.io/portfolio-data-science/) · [LinkedIn](https://www.linkedin.com/in/nurul-ilahi-husnah/) · [Tableau Public](https://public.tableau.com/app/profile/nurul-ilahi-husnah)*
