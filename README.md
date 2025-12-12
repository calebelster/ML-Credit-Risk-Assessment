# ML Credit Risk Assessment

Website: **Credit Risk Assessment Platform**
[https://ml-credit-risk-assessment-e6owtb8xlwbjtjjnccc57a.streamlit.app/](https://ml-credit-risk-assessment-e6owtb8xlwbjtjjnccc57a.streamlit.app/)

Credit risk assessment project that trains multiple machine learning models (including a Stacking ensemble) on a public credit risk dataset and exposes them through a Streamlit web app for:

* Single application risk scoring
* Batch (CSV/Excel) risk scoring
* Model analysis and visualization

Dataset source: [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/code)

---

# 1. Project Structure

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
│   └── prediction files
├── app/
│   ├── Home.py
│   ├── __init__.py
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
├── QUICKSTART.md
└── README.md
```

---

# 2. Dataset

* **Source:** Kaggle Credit Risk Dataset
* **Target column:** `loan_status`

  * `0` = non-default
  * `1` = default

Some features (e.g., `loan_grade`) are removed, both because they would not be known by the user at application time, and due to overfitting issues.

---

# 3. Setup

# 3.1 Note on UV and Reproducibility

The program runs through a **web-hosted Streamlit app** which is linked at the top of this file. It does not inherently require the user to develop a virtual environment with UV.
However, the project can be run locally and provides full UV compatibility.

# 3.2 Running Locally

Check out the QUICKSTART.md file for instructions on how to begin your credit risk analysis locally.

---

# 4. Training & Evaluation Pipeline

## 4.1 Train Base Models (Optional)

From the project root:

```bash
python src/models/random_forest.py
python src/models/grad_boost.py
python src/models/log_reg.py
python src/models/neural_net.py
```

These scripts:

* Load and process the dataset
* Train the model
* Save predictions
* Save optional feature importances (Random Forest)

---

## 4.2 Train Ensemble Models & Save Deployment Artifact

```bash
python src/analysis/ensemble_methods.py
```

This script:

* Drops columns not known at prediction time
* Applies one-hot encoding, imputation, and scaling
* Trains:

  * Random Forest
  * Gradient Boosting
  * Logistic Regression
  * Neural Net
* Builds ensembles:

  * Soft Voting
  * Bagging
  * Stacking (meta-logistic)
  * Blending
* Evaluates each on test set
* Saves the final deployment model:

```
app/saved_models/stacking_model.pkl
```

---

## 4.3 K-Fold Cross-Validation

```bash
python src/analysis/cross_validation.py
```

Generates CV metrics for all base models and all ensemble methods.
Metrics saved include:

* AUC
* PR-AUC
* Brier Score
* KS statistic
* LogLoss

Used for visualization on the Model Analysis page.

---

# 5. Risk Calculator (Page 1)

Two modes:

### **5.1 Single Application**

User enters fields such as:

* Income
* Age
* Loan amount
* Employment length
* Loan intent
* Home ownership
* Credit history length
* Prior default history

A threshold slider (0.30–0.70) adjusts risk strictness.

Outputs include:

* Default probability
* Risk score
* Risk category
* Explanatory feedback
* What helps / What hurts the risk score

---

### **5.2 Batch Upload**

Upload `.csv` files.

Must include required columns:

* `person_age`
* `person_income`
* `person_emp_length`
* `loan_amnt`
* `loan_int_rate`
* `cb_person_cred_hist_length`
* `person_home_ownership`
* `loan_intent`
* `cb_person_default_on_file`

Outputs:

* Summary report of low/medium/high risk
* Full prediction table
* Downloadable results CSV
* Per-row detailed analysis

---

# 6. Model Analysis (Page 2)

Three tabs:

## 6.1 Cross-Validation

Displays metrics per fold for all models, including:

* AUC
* PR-AUC
* Brier
* KS
* LogLoss

## 6.2 Test Performance

Metrics on the held-out test set:

* Confusion matrix
* ROC curve
* PR curve
* F1, LogLoss

## 6.3 Comparison

Compares:

* Base models
* Bagging
* Voting
* Stacking
* Blending

The **Stacking Ensemble** achieves:

* **AUC ≈ 0.928**
* **PR-AUC ≈ 0.871**
* **F1 ≈ 0.794**
* **LogLoss ≈ 0.231**

---

# 7. Validation & Class Balance

## 7.1 Class Imbalance Strategy

The original dataset is heavily imbalanced, with far fewer defaults than non-defaults.
To avoid bias toward predicting “no default,” we rebalanced the **training set only**:

* **All default cases** were kept
* **A random sample of non-default cases** was drawn
* Final training set used a **50/50 split**

This improves learning of default-related patterns while keeping evaluation unbiased.

---

## 7.2 Validation Criteria

The project includes several measurable validation objectives:

### **Model-Level Validation**

* AUC, PR-AUC, LogLoss, KS, and Brier Score
* Per-fold cross-validation metrics
* Test-set confusion matrices and curves

### **Application-Level Validation**

* Warnings when required columns are missing
* Checks for invalid ages or malformed entries
* Detection of improperly labeled categorical values
* User-facing error messages when files cannot be processed

### **Performance Validation**

* End-to-end prediction for a single case executes in under one second
* Batch uploads (hundreds of rows) process in under a few seconds

---

# 8. Handling Incorrect or Missing Data

We implemented several safeguards:

* **Missing Columns:** App displays an error and lists missing fields
* **Invalid Data Types:** Automatic conversion attempts, fallback to error message
* **Unrecognized Categories:** User receives a message with allowed options
* **Age Restrictions:** Entries under age 18 are rejected
* **Worst-Case Prevention:**
  The system *never* runs predictions on structurally invalid data.
  Errors are shown immediately to the user with corrective instructions.

---

# 9. Automation Summary

Our project satisfies the automation goals by:

### Fully automated end-to-end workflow

* Preprocessing
* Training
* Cross-validation
* Ensemble generation
* Saving final model
* Web inference
* Batch processing

### Only minimal human input required

* Selecting threshold
* Uploading a file
* Optional selection of model for analysis

Everything else runs with **one click**.

---

# 10. Success Criteria

Our project meets the following measurable targets:

* **Model performance:**

  * AUC ≥ 0.92
  * PR-AUC ≥ 0.85
  * LogLoss ≤ 0.25
* **Usability criteria:**

  * Users can receive a full risk evaluation in ≤ 2 clicks
  * Batch scoring works with 100% valid files
  * Invalid files trigger clear error messages
* **Reproducibility:**

  * All results reproducible with included scripts and environment instructions
* **Automation:**

  * Entire workflow (train → evaluate → deploy) is automated end to end
