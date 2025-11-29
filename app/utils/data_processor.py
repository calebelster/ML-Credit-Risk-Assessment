import pandas as pd
from typing import Dict


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

    def generate_application_feedback(self, row: pd.Series, default_prob: float) -> Dict[str, str]:
        """
        Generate simple textual feedback for one application based on key factors.
        """
        msgs_good = []
        msgs_improve = []

        # Debt-to-income style signal: loan_percent_income
        if row["person_income"] > 0:
            loan_pct_income = row["loan_amnt"] / row["person_income"]
        else:
            loan_pct_income = 1.0

        # Age / history
        if row["cb_person_cred_hist_length"] >= 8:
            msgs_good.append("You have a relatively long credit history, which is positive.")
        else:
            msgs_improve.append("A longer credit history would improve your profile over time.")

        # Income vs loan amount
        if loan_pct_income <= 0.2:
            msgs_good.append("Your requested loan is modest relative to your income.")
        elif loan_pct_income <= 0.4:
            msgs_improve.append("Your loan is a moderate share of your income; keeping it lower can reduce risk.")
        else:
            msgs_improve.append("Your loan is a large share of your income; lowering the amount or increasing income would help.")

        # Interest rate
        if row["loan_int_rate"] <= 10:
            msgs_good.append("Your interest rate is in a reasonable range.")
        else:
            msgs_improve.append("A lower interest rate would reduce your monthly burden.")

        # Employment length
        if row["person_emp_length"] >= 2:
            msgs_good.append("Your employment length suggests some income stability.")
        else:
            msgs_improve.append("A longer employment history at your job would strengthen your application.")

        # Prior default
        if row["cb_person_default_on_file"] == "N":
            msgs_good.append("No prior default on file is a strong positive factor.")
        else:
            msgs_improve.append("A previous default increases risk; maintaining clean repayment behavior over time will help.")

        # Home ownership
        if row["person_home_ownership"] in ["OWN", "MORTGAGE"]:
            msgs_good.append("Owning or mortgaging a home is typically seen as more stable than renting.")
        else:
            msgs_improve.append("Building more financial stability (savings, assets) can improve your profile.")

        # Overall risk
        if default_prob < 0.3:
            overall = "Overall, this application looks low risk."
        elif default_prob < 0.6:
            overall = "Overall, this application is moderate risk; a few improvements could help."
        else:
            overall = "Overall, this application is high risk; significant improvements are recommended."

        return {
            "good": " ".join(msgs_good) if msgs_good else "No strong positives identified.",
            "improve": " ".join(msgs_improve) if msgs_improve else "No critical weaknesses identified.",
            "overall": overall,
        }