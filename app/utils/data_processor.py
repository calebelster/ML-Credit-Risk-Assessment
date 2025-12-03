import pandas as pd
from typing import Dict, List, Optional


class DataProcessor:
    """
    Process loan application data from forms and CSVs.
    Expected raw input columns (before one-hot):
      - person_age (int)
      - person_income (float, annual income)
      - person_emp_length (int, years)
      - loan_amnt (float)
      - loan_int_rate (float, %)
      - cb_person_cred_hist_length (int, years)
      - cb_person_default_on_file (str: 'Y' or 'N')
      - person_home_ownership (str: RENT / OWN / MORTGAGE / OTHER)
      - loan_intent (str: PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION)
    """

    CATEGORICAL_COLS = [
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]

    NUMERIC_COLS = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "cb_person_cred_hist_length",
    ]

    def __init__(self):
        self.all_features = self.CATEGORICAL_COLS + self.NUMERIC_COLS

    def get_expected_schema(self) -> pd.DataFrame:
        """
        Return a small DataFrame describing expected columns and example values,
        for display on the UI.
        """
        data = {
            "Column": self.all_features,
            "Type": [
                "categorical",  # person_home_ownership
                "categorical",  # loan_intent
                "categorical",  # cb_person_default_on_file
                "numeric (int)",   # person_age
                "numeric (float)", # person_income
                "numeric (int)",   # person_emp_length
                "numeric (float)", # loan_amnt
                "numeric (float)", # loan_int_rate
                "numeric (int)",   # cb_person_cred_hist_length
            ],
            "Example": [
                "RENT",
                "3.5",
                "N",
                35,
                60000,
                5,
                12000,
                8.5,
                10,
            ],
            "Notes": [
                "One of: RENT, OWN, MORTGAGE, OTHER",
                "One of: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION",
                "One of: 'Y' (prior default) or 'N' (no prior default)",
                "Age in years (>=18)",
                "Annual gross income in USD",
                "Years in current employment",
                "Requested loan amount in USD",
                "Nominal interest rate in percent",
                "Years since first credit line",
            ],
        }
        return pd.DataFrame(data)

    def form_to_dataframe(self, form_data: Dict) -> pd.DataFrame:
        df = pd.DataFrame([form_data])

        missing = [col for col in self.all_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        for col in self.NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.CATEGORICAL_COLS:
            df[col] = df[col].astype(str)

        return df[self.all_features]

    def csv_to_dataframe(self, uploaded_file) -> pd.DataFrame:
        """
        Convert uploaded CSV/Excel to DataFrame and validate schema.
        """
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Only CSV and Excel files supported")
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")

        # Drop loan_grade if present
        if "loan_grade" in df.columns:
            df = df.drop("loan_grade", axis=1)

        missing = [col for col in self.all_features if col not in df.columns]
        if missing:
            raise ValueError(
                f"File is missing required columns: {missing}. "
                f"Required columns are: {self.all_features}"
            )

        df = df[self.all_features].copy()

        for col in self.NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.CATEGORICAL_COLS:
            df[col] = df[col].astype(str)

        # Basic sanity checks
        if (df["person_age"] < 18).any():
            raise ValueError("Some rows have person_age < 18, which is invalid.")

        return df

    def preprocess_for_prediction(self, X, feature_names):
        """
        One-hot encode and align features to match training data.
        """
        X_enc = pd.get_dummies(X, columns=self.CATEGORICAL_COLS, drop_first=False)

        for col in feature_names:
            if col not in X_enc.columns:
                X_enc[col] = 0

        X_enc = X_enc[feature_names]

        return X_enc

    from typing import Dict, List, Optional
import pandas as pd

def generate_application_feedback(
    row: pd.Series,
    default_prob: float,
    feature_contribs: Optional[Dict[str, float]] = None
) -> Dict[str, object]:
    """
    Generate feedback with bullet lists:
      - 'positives': List[str] (reassurance / strengths)
      - 'negatives': List[str] (constructive suggestions)
      - 'overall': str (short summary)
    Optionally accepts `feature_contribs` mapping feature_name -> contribution (positive means raises risk).
    """

    positives: List[str] = []
    negatives: List[str] = []

    # helper to add unique bullets
    def add_pos(msg: str):
        if msg not in positives:
            positives.append(msg)

    def add_neg(msg: str):
        if msg not in negatives:
            negatives.append(msg)

    # compute loan pct income robustly
    try:
        person_income = float(row.get("person_income", 0) or 0)
    except Exception:
        person_income = 0.0
    loan_amnt = float(row.get("loan_amnt", 0) or 0)
    loan_pct_income = loan_amnt / person_income if person_income > 0 else 1.0

    # Credit history
    hist = row.get("cb_person_cred_hist_length", None)
    if hist is not None and hist >= 8:
        add_pos("You have a relatively long credit history — this is a strong positive. Keep maintaining on-time payments.")
    else:
        add_neg("Building a longer credit history (keep accounts open, avoid gaps) will help future scores.")

    # Loan vs income
    if loan_pct_income <= 0.2:
        add_pos("Your requested loan is modest relative to income — this reduces repayment pressure.")
    elif loan_pct_income <= 0.4:
        add_neg("Your loan is a moderate share of income; lowering the amount or increasing income would reduce risk.")
    else:
        add_neg("Your loan is a large share of income; consider reducing the request or increasing income/savings.")

    # Interest rate
    ir = row.get("loan_int_rate", None)
    if ir is not None:
        try:
            ir = float(ir)
            if ir <= 10:
                add_pos("Your interest rate is in a reasonable range; keep shopping for better rates periodically.")
            else:
                add_neg("A lower interest rate would reduce monthly burden; consider refinancing or negotiating if possible.")
        except Exception:
            pass

    # Employment length
    emp = row.get("person_emp_length", None)
    if emp is not None and emp >= 2:
        add_pos("Employment tenure suggests stability — maintain steady employment where possible.")
    else:
        add_neg("Longer tenure at your job or documenting stable income (contracts/pay stubs) will strengthen applications.")

    # Prior default
    default_flag = row.get("cb_person_default_on_file", "N")
    if str(default_flag).upper() == "N":
        add_pos("No prior default on file — good track record. Continue punctual payments.")
    else:
        add_neg("A prior default increases risk; consistently on-time payments over time will improve the score.")

    # Home ownership
    home = row.get("person_home_ownership", None)
    if home in ["OWN", "MORTGAGE"]:
        add_pos("Owning or having a mortgage indicates asset stability — keep building equity and savings.")
    else:
        add_neg("Building savings and reducing unsecured debt can improve your profile if you currently rent.")

    # Optional: incorporate feature_contribs (e.g. SHAP)
    if feature_contribs:
        # We want the top features that *increase* risk and top features that *decrease* risk
        # feature_contribs: feature -> contribution (positive means increases prob / risk)
        contrib_items = sorted(feature_contribs.items(), key=lambda kv: kv[1], reverse=True)
        # positive contributors to risk
        for feat, val in contrib_items[:3]:
            if val > 0:
                add_neg(f"Factor: {feat} is contributing to higher risk (impact ≈ {val:.3f}).")
        # negative contributors (i.e., protective features)
        for feat, val in reversed(contrib_items[-3:]):
            if val < 0:
                add_pos(f"Factor: {feat} is helping reduce risk (impact ≈ {val:.3f}).")

    # Decide final messaging by risk band
    if default_prob < 0.3:
        overall = "Low risk — this application looks favorable."
        # low risk: reassure, but provide light suggestions to keep improving
        # move most negatives to "suggestions" (they are already constructive)
        # ensure positives appear first in UI
    elif default_prob < 0.6:
        overall = "Moderate risk — you have strengths and areas to improve."
    else:
        overall = "High risk — significant improvements are recommended."

    return {
        "positives": positives or ["No strong positives identified."],
        "negatives": negatives or ["No critical weaknesses identified."],
        "overall": overall,
        "default_prob": float(default_prob)
    }
