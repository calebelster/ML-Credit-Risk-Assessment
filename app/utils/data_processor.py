import pandas as pd
import shap
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

    def generate_application_feedback(
            self,
            row: pd.Series,
            default_prob: float,
            feature_contribs: Optional[Dict[str, float]] = None
        ) -> Dict[str, object]:
        """
        Generate feedback with bullet lists:
        - 'good': strengths
        - 'improve': suggestions
        - 'overall': summary
        """
        positives: List[str] = []
        negatives: List[str] = []

        # helper functions
        def add_pos(msg: str):
            if msg not in positives:
                positives.append(msg)

        def add_neg(msg: str):
            if msg not in negatives:
                negatives.append(msg)

        # compute loan percent income
        try:
            person_income = float(row.get("person_income", 0) or 0)
        except Exception:
            person_income = 0.0

        loan_amnt = float(row.get("loan_amnt", 0) or 0)
        loan_pct_income = loan_amnt / person_income if person_income > 0 else 1.0

        # Credit history
        hist = row.get("cb_person_cred_hist_length", None)
        if hist is not None and hist >= 8:
            add_pos("You have a relatively long credit history — this is a strong positive.")
        else:
            add_neg("Building a longer credit history will help your future applications.")

        # Loan vs income
        if loan_pct_income <= 0.2:
            add_pos("Your requested loan is modest relative to income.")
        elif loan_pct_income <= 0.4:
            add_neg("Your loan is a moderate share of income; reducing it would lower risk.")
        else:
            add_neg("Your loan is a large share of income; consider reducing the amount requested.")

        # Interest rate
        ir = row.get("loan_int_rate", None)
        if ir is not None:
            try:
                ir = float(ir)
                if ir <= 10:
                    add_pos("Your interest rate is in a reasonable range.")
                else:
                    add_neg("A lower interest rate would reduce monthly burden.")
            except Exception:
                pass

        # Employment length
        emp = row.get("person_emp_length", None)
        if emp is not None and emp >= 2:
            add_pos("Employment tenure suggests stability.")
        else:
            add_neg("Longer job tenure or consistent income documentation would improve your profile.")

        # Prior default
        default_flag = row.get("cb_person_default_on_file", "N")
        if str(default_flag).upper() == "N":
            add_pos("No prior default on file — good track record.")
        else:
            add_neg("A prior default increases risk; consistent on-time payments will improve future scores.")

        # Home ownership
        home = row.get("person_home_ownership", None)
        if home in ["OWN", "MORTGAGE"]:
            add_pos("Home ownership signals financial stability.")
        else:
            add_neg("Building savings and reducing unsecured debt can strengthen your profile.")

        # Optional SHAP-style contributions
        if feature_contribs:
            contrib_items = sorted(feature_contribs.items(), key=lambda kv: kv[1], reverse=True)

            # top risk-increasing factors
            for feat, val in contrib_items[:3]:
                if val > 0:
                    add_neg(f"Factor: {feat} increases risk (impact ≈ {val:.3f}).")

            # most protective features
            for feat, val in reversed(contrib_items[-3:]):
                if val < 0:
                    add_pos(f"Factor: {feat} helps reduce risk (impact ≈ {val:.3f}).")

        # Final text summary
        if default_prob < 0.1:
            overall = "Low risk — this application looks favorable."
        elif default_prob < 0.25:
            overall = "Moderate risk — you have strengths and areas to improve."
        else:
            overall = "High risk — significant improvements are recommended."

        return {
            "good": positives or ["No major strengths identified."],
            "improve": negatives or ["No major risk factors identified."],
            "overall": overall,
            "default_prob": float(default_prob),
        }
    
    def compute_shap_values(self, model, X: pd.DataFrame, feature_names: List[str]):
        """
        Compute SHAP values for each row in X using the trained model.

        Parameters:
        - model: trained scikit-learn or tree-based model
        - X: input features DataFrame (raw or preprocessed)
        - feature_names: list of features expected by the model

        Returns:
        - List[Dict]: list of dictionaries mapping feature -> contribution per row
        """
        # Preprocess to one-hot encode and align with training
        X_enc = self.preprocess_for_prediction(X, feature_names)

        # Choose the right explainer
        # For tree-based models (RandomForest, XGBoost, LightGBM, etc.)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_enc)  # returns array-like for each row

        # Convert to list of dicts per row
        contributions = []
        for i in range(len(X_enc)):
            row_contrib = dict(zip(X_enc.columns, shap_values[i]))
            contributions.append(row_contrib)

        return contributions
