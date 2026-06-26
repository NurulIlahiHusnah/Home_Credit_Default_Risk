# 🏦 Credit Scoring Model — Home Credit Default Risk

> **Objective:** Reduce false rejection of creditworthy applicants by building a robust credit default prediction model aligned with OJK regulatory standards and IFRS 9 compliance requirements.

---

## Problem Statement

Home Credit Indonesia faced a selection bias problem: the existing system systematically rejected creditworthy applicants whose profiles had never been seen in training data. This project addresses that by:

- Reducing the **False Rejection Rate** — good applicants wrongly denied credit
- Maintaining credit risk control through regulatory-grade metrics (KS Statistic, per OJK requirements)
- Correcting historical selection bias via **Reject Inference**
- Producing **calibrated PD estimates** suitable for ECL calculation under IFRS 9

---

## Dataset

**Source:** [Home Credit Default Risk — HCI](https://www.kaggle.com/c/home-credit-default-risk)

| Table | Description |
|---|---|
| `application_train / test` | Primary applicant data; target: 0 = on-time, 1 = payment difficulties |
| `bureau + bureau_balance` | External credit bureau history (joined on SK_ID_BUREAU) |
| `previous_application` | Previous loan applications at Home Credit |
| `installments_payments` | Historical installment payment records |
| `POS_CASH_balance` | Monthly balances for POS and cash installment loans |
| `credit_card_balance` | Credit card utilization and payment history |

- **Primary key:** `SK_ID_CURR` (unique applicant identifier across all tables)
- **Class imbalance:** ~8% default (TARGET = 1) vs ~92% non-default
- **Scale:** Tens of millions of records across all tables

---

## Methodology — 10-Phase Pipeline

```
PHASE 1  — Leakage Audit           : Identify and remove post-decision columns
PHASE 2  — Preprocessing           : Clean, encode, impute, scale (anti-leakage)
PHASE 3  — Reject Inference        : Label Propagation to correct selection bias
PHASE 4  — Baseline Model          : Logistic Regression
PHASE 5  — Main Model              : XGBoost + SHAP
PHASE 6  — Regulatory Evaluation   : KS, AUC, Precision@R70, Log Loss, F1
PHASE 7  — Threshold Tuning        : Minimize False Rejection Rate
PHASE 8  — Calibration PD          : Isotonic calibration for IFRS 9 compliance
PHASE 9  — SHAP Explanation        : Feature importance for stakeholders & regulators
PHASE 10 — Submission + Summary    : Final artifacts and results
```

---

## Key Results

| Metric | Logistic Regression | XGBoost | Target | Status |
|---|---|---|---|---|
| ROC-AUC | 0.7622 | **0.7679** | > 0.75 | ✅ |
| KS Statistic | 0.3866 | **0.4026** | > 0.35 | ✅ |
| Precision@R70 | 0.8930 | **0.8931** | > 0.30 | ✅ |
| Log Loss | 0.5795 | **0.4573** | Lower is better | ✅ |
| Brier Score (before calibration) | — | 0.1486 | — | — |
| Brier Score (after calibration) | — | **0.0668** | Lower is better | ✅ |
| Calibration Improvement | — | **~55%** | — | ✅ |

> **Note on F1:** F1 is computed at the default 0.5 threshold, which is not the operational decision threshold for imbalanced credit data (~8% positive rate). AUC, KS, and Precision@R70 are the primary evaluation metrics for this use case.

> **Recommended model:** XGBoost with Reject Inference — consistently outperforms Logistic Regression across all metrics. Logistic Regression is retained as an interpretable baseline for regulatory reporting.

---

## IFRS 9 Framing

This project incorporates IFRS 9 considerations for Expected Credit Loss (ECL) estimation.

### ECL Formula

```
ECL = PD × LGD × EAD
```

| Component | Description |
|---|---|
| **PD** (Probability of Default) | Output dari model ini — wajib terkalibrasi |
| **LGD** (Loss Given Default) | Estimasi kerugian jika terjadi default |
| **EAD** (Exposure at Default) | Total eksposur saat terjadi default |

### Kenapa Kalibrasi PD Wajib untuk IFRS 9

Model dengan AUC tinggi belum tentu menghasilkan PD yang akurat secara absolut. Resampling (Label Propagation, SMOTE, dll.) mengubah proporsi kelas di training data sehingga PD mentah model menjadi overestimate atau underestimate terhadap actual default rate.

Dalam proyek ini:
- **Sebelum kalibrasi:** Mean PD ~4x lebih tinggi dari actual default rate (~8.1%)
- **Sesudah kalibrasi isotonic:** Mean PD mendekati actual default rate
- **Implikasi:** PD yang tidak terkalibrasi akan menyebabkan ECL overestimate — pencadangan (provisioning) berlebih di semua skenario IFRS 9 secara bersamaan

### PIT vs TTC

| Jenis | Deskripsi | Model ini |
|---|---|---|
| **PIT** (Point-in-Time) | PD berubah sesuai kondisi ekonomi saat ini; memasukkan MEV | ❌ Tidak diimplementasikan |
| **TTC** (Through-the-Cycle) | PD stabil sepanjang siklus ekonomi; tidak sensitif terhadap MEV | Mendekati, tapi bukan TTC murni |
| **Pseudo-TTC** | Model statis tanpa MEV; framing paling akurat untuk model ini | ✅ |

> Model ini adalah **static PIT model tanpa variabel makroekonomi (MEV)**. Untuk full IFRS 9 compliance, diperlukan integrasi MEV dan forward-looking scenario weighting (base, optimistic, pessimistic).

### Keterbatasan Model (IFRS 9 Context)

- Tidak ada MEV (GDP growth, unemployment rate, inflation)
- Tidak ada term structure (PD per tenor/maturity)
- Tidak ada scenario weighting (3 skenario IFRS 9)
- Model ini memberikan **fondasi PD yang valid dan terkalibrasi** — langkah selanjutnya adalah integrasi MEV untuk full compliance

---

## Technical Highlights

### Reject Inference (Label Propagation)

Training data contains only approved applicants, introducing selection bias: the model never learns from rejected profiles and therefore keeps rejecting unfamiliar but potentially creditworthy applicants. To correct this, Label Propagation identifies applicants from the unlabeled test set whose feature profiles are similar to known non-defaults, and adds them to training with a reduced sample weight (0.3).

### Anti-Leakage Preprocessing

Post-decision features — installment performance, active credit card balances, current DPD from running loans — are explicitly identified and removed before training. These features are only observable after a loan is approved and active; their presence in a model would simulate perfect foresight and cause significant data leakage.

### Regulatory Compliance (OJK)

- KS Statistic computed and reported per OJK credit scoring requirements (target > 0.35; achieved 0.4026 — Excellent range)
- Logistic Regression provides interpretable coefficients for regulatory and stakeholder review
- SHAP values explain individual XGBoost predictions and global feature importance

### Probability Calibration (Isotonic)

- `CalibratedClassifierCV(cv='prefit', method='isotonic')` fitted on held-out validation set
- Brier Score improved ~55% (0.1486 → 0.0668)
- AUC unchanged — calibration corrects absolute probability accuracy without affecting ranking ability
- Calibrated model saved as `models/model_xgb_calibrated.pkl`

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
3. Open `notebooks/Preprocessing_Modeling_HomeCredit.ipynb` in Google Colab or Jupyter Notebook
4. Run all cells sequentially — all artifacts (models, scaler, threshold, SHAP) are saved automatically

---

## File Structure

```
Home_Credit_Default_Risk/
├── notebooks/
│   └── Preprocessing_Modeling_HomeCredit.ipynb  # Main pipeline notebook (10 phases)
├── models/
│   ├── model_xgb_calibrated.pkl                 # Trained & calibrated XGBoost model
│   └── pd_calibration.png                       # Reliability diagram (before vs after)
├── outputs/
│   └── submission.csv                           # Kaggle submission file
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Portfolio · [LinkedIn](https://www.linkedin.com/in/nurul-ilahi-husnah27/) · [GitHub](https://github.com/NurulIlahiHusnah)
