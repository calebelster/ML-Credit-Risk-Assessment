# ML Credit Risk Assessment

Credit risk assessment project that trains multiple machine learning models (including a Stacking ensemble) on a public credit risk dataset and exposes them through a Streamlit web app for:

- Single application risk scoring
- Batch (CSV/Excel) risk scoring
- Model analysis and visualization

Dataset source: [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/code)

***

## 1. Project Structure

```text
ML-Credit-Risk-Assessment/
├── data/
│   └── credit_risk_dataset.csv
├── src/
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── grad_boost.py
│   │   ├── log_reg.py
│   │   └── neural_net.py
│   ├── analysis/
│   │   ├── ensemble_methods.py
│   │   ├── cross_validation.py
│   │   ├── compare_models.py
│   │   ├── feature_importance.py
│   │   └── plot_calibration.py
│   └── process_data.py
├── output/
│   ├── random_forest_preds.csv
│   ├── grad_boost_preds.csv
│   ├── log_reg_preds.csv
│   ├── neural_net_preds.csv
│   ├── ensemble_bagging_preds.csv
│   ├── ensemble_voting_preds.csv
│   ├── ensemble_stacking_preds.csv
│   ├── ensemble_blending_preds.csv
│   ├── random_forest_cv_metrics.csv
│   ├── grad_boost_cv_metrics.csv
│   ├── log_reg_cv_metrics.csv
│   ├── neural_net_cv_metrics.csv
│   ├── ensemble_bagging_cv_metrics.csv
│   ├── ensemble_voting_cv_metrics.csv
│   ├── ensemble_stacking_cv_metrics.csv
│   └── ensemble_blending_cv_metrics.csv
├── app/
│   ├── app.py
│   ├── saved_models/
│   │   └── stacking_model.pkl
│   ├── pages/
│   │   ├── 1_Risk_Calculator.py
│   │   └── 2_Model_Analysis.py
│   └── utils/
│       ├── predictor.py
│       ├── data_processor.py
│       └── visualizations.py
├── .streamlit/
│   └── config.toml
├── main.py
├── requirements.txt
└── README.md
```


***

## 2. Dataset

- Source: [https://www.kaggle.com/datasets/laotse/credit-risk-dataset/code](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/code)
- Target column: `loan_status`
    - `0` = no default
    - `1` = default
- Some columns (e.g., `loan_grade`) are removed from modeling because the user does not know them at application time.

Place `credit_risk_dataset.csv` in the `data/` directory.

***

## 3. Setup

### 3.1 Create Environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```


### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```


***

## 4. Training \& Evaluation Pipeline

### 4.1 Train Base Models (Optional)

From the project root:

```bash
python src/models/random_forest.py
python src/models/grad_boost.py
python src/models/log_reg.py
python src/models/neural_net.py
```

These scripts:

- Load `data/credit_risk_dataset.csv`
- Train a model with a train/test split
- Save test predictions to `output/*_preds.csv`
- Optionally save feature importance (Random Forest)


### 4.2 Train Ensemble Models \& Save Deployment Model

```bash
python src/analysis/ensemble_methods.py
```

`ensemble_methods.py`:

- Drops `loan_grade` (not user-provided)
- Performs one-hot encoding, median imputation, and scaling on the training set
- Trains base models:
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    - Neural Net (MLP)
- Builds four ensembles:
    - Soft Voting
    - Bagging (average of base probabilities)
    - Stacking (Logistic Regression meta-learner)
    - Blending (inner holdout on the train set)
- Evaluates on a held-out test set and writes:
    - `output/ensemble_voting_preds.csv`
    - `output/ensemble_bagging_preds.csv`
    - `output/ensemble_stacking_preds.csv`
    - `output/ensemble_blending_preds.csv`
- Saves a deployment-ready stacking model artifact to:
    - `app/saved_models/stacking_model.pkl`

The stacking artifact contains:

- Base models: RF, GB, LR, NN
- Meta-learner: Logistic Regression
- Preprocessors: imputer, feature scaler, meta-feature scaler
- Feature names (after one-hot encoding)


### 4.3 K-Fold Cross-Validation for All Models

```bash
python src/analysis/cross_validation.py
```

`cross_validation.py`:

- Drops `loan_grade`
- Applies the same preprocessing (one-hot, impute, scale)
- Runs Stratified K-fold CV for 8 models:
    - Base:
        - Random Forest → `random_forest_cv_metrics.csv`
        - Gradient Boost → `grad_boost_cv_metrics.csv`
        - Logistic Regression → `log_reg_cv_metrics.csv`
        - Neural Net → `neural_net_cv_metrics.csv`
    - Ensembles:
        - Bagging → `ensemble_bagging_cv_metrics.csv`
        - Voting → `ensemble_voting_cv_metrics.csv`
        - Stacking → `ensemble_stacking_cv_metrics.csv`
        - Blending → `ensemble_blending_cv_metrics.csv`
- Metrics per fold:
    - `AUC`
    - `PR_AUC`
    - `Brier`
    - `KS`
    - `LogLoss`

These CSVs are used by the Model Analysis page.

***

## 5. Running the Streamlit App

From the project root:

1. Ensure `app/saved_models/stacking_model.pkl` exists (from `ensemble_methods.py`).
2. Run Streamlit:
```bash
cd app
streamlit run Home.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

***

## 6. Risk Calculator (Page 1)

The Risk Calculator page has two tabs.

### 6.1 Single Application

Fill in:

- `person_age` (integer): Age in years (≥ 18)
- `person_income` (float): Annual income in USD
- `person_emp_length` (integer): Years in current employment
- `loan_amnt` (float): Requested loan amount (USD)
- `loan_int_rate` (float): Nominal interest rate (%)
- `cb_person_cred_hist_length` (integer): Years since first credit line
- `person_home_ownership` (categorical): `RENT`, `OWN`, `MORTGAGE`, `OTHER`
- `loan_intent` (categorical): `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`
- `cb_person_default_on_file` (categorical): `'Y'` (prior default) or `'N'` (no prior default)

Sidebar “Decision Threshold” (0.30–0.70):

- Lower threshold (e.g., 0.30–0.40)
    - More conservative
    - Flags more applications as high risk
- Higher threshold (e.g., 0.60–0.70)
    - More lenient
    - Approves more loans

Click “Calculate Risk” to see:

- Risk Score (%)
- Default Probability
- Risk Category
- Threshold used

Then the app provides tailored feedback based on the input:

- “What looks good” (e.g., strong income relative to loan, long credit history, no prior default)
- “Where you could improve” (e.g., large loan vs income, high interest rate, short employment)
- “Overall assessment” (low / moderate / high risk narrative)


### 6.2 Batch Upload

The batch tab lets you upload many applications at once (CSV or Excel).

#### Required Columns

Your file must contain these columns (case-sensitive):

- `person_age` (integer): Age in years (≥ 18)
- `person_income` (float): Annual gross income in USD
- `person_emp_length` (integer): Years in current employment
- `loan_amnt` (float): Requested loan amount in USD
- `loan_int_rate` (float): Nominal interest rate (%)
- `cb_person_cred_hist_length` (integer): Years since first credit line
- `person_home_ownership` (string): one of `RENT`, `OWN`, `MORTGAGE`, `OTHER`
- `loan_intent` (string): one of `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`
- `cb_person_default_on_file` (string): `'Y'` if prior default, `'N'` otherwise

Notes:

- Extra columns are ignored.
- `loan_grade` is ignored and not used by the model.
- Basic validation checks for missing columns and invalid ages (< 18).

Steps:

1. Open the “Batch Upload” tab.
2. Review the “Required File Format” section and example schema table.
3. Upload a `.csv`, `.xlsx`, or `.xls` file with the required columns.
4. Click “Calculate Risk for All”.

The app returns:

- Summary counts and percentages of Low / Medium / High risk applications
- Full results table with:
    - `default_probability`
    - `risk_score_percent`
    - `predicted_default`
    - `risk_category`
- Download button for a results CSV (`credit_risk_results.csv`)

Per-application feedback:

- Enter a row index (0-based) to inspect a specific application.
- The app shows:
    - Default probability and risk score
    - “What looks good”
    - “What needs improvement”
    - “Overall assessment”

***

## 7. Model Analysis (Page 2)

The Model Analysis page has three tabs.

### 7.1 Cross-Validation

Loads CV metrics from:

- `random_forest_cv_metrics.csv`
- `grad_boost_cv_metrics.csv`
- `log_reg_cv_metrics.csv`
- `neural_net_cv_metrics.csv`
- `ensemble_bagging_cv_metrics.csv`
- `ensemble_voting_cv_metrics.csv`
- `ensemble_stacking_cv_metrics.csv`
- `ensemble_blending_cv_metrics.csv`

Features:

- Dropdown to select model (defaults to **Ensemble Stacking** if available)
- Plotly line chart of AUC, PR_AUC, Brier, KS across folds
- X-axis shows integer fold numbers (1…K)
- Mean ± standard deviation of metrics
- Per-fold metrics table


### 7.2 Test Performance

Uses test prediction CSVs:

- `ensemble_stacking_preds.csv`
- `random_forest_preds.csv`
- `ensemble_bagging_preds.csv`
- `ensemble_voting_preds.csv`
- `ensemble_blending_preds.csv`

For the selected model, shows:

- Confusion matrix heatmap
- AUC-ROC
- PR-AUC
- F1-Score
- Brier Score


### 7.3 Comparison

Compares key test-set metrics (AUC, PR-AUC, F1, LogLoss) across:

- Random Forest
- Gradient Boost
- Logistic Regression
- Neural Net
- Ensemble Bagging
- Ensemble Voting
- Ensemble Stacking
- Ensemble Blending

Highlights **Ensemble Stacking** as the primary model:

- AUC ≈ 0.928 (tied best)
- PR-AUC ≈ 0.871
- F1 ≈ 0.794
- LogLoss ≈ 0.231 (best calibration)

Explains why `loan_grade` is excluded:

- Not known by the applicant
- May act as proxy for internal decisions
- Removing it reduces leakage and makes the tool realistic

***

## 8. Deployment / WordPress Embedding (Optional)

You can deploy the Streamlit app (e.g., Streamlit Cloud, VM, or container) and embed it in WordPress.

Example WordPress embed (Custom HTML block):

```html
<iframe
  src="https://your-app.streamlit.app/"
  width="100%"
  height="1200"
  frameborder="0"
  style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>
```


***

## 9. Notes \& Assumptions

- `loan_status = 1` is treated as a “default”; model outputs are default probabilities.
- The app provides risk estimates and educational feedback; it does not make binding credit decisions or provide legal/financial advice.
- Feedback is heuristic and based on standard credit risk drivers (income vs loan, prior default, credit history, employment, interest burden).