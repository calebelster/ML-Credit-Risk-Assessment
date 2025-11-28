import pandas as pd
import numpy as np
from typing import Dict, List
import io


class DataProcessor:
    """Process loan application data from forms and CSVs"""

    CATEGORICAL_COLS = [
        'person_home_ownership',
        'loan_intent',
        'cb_person_default_on_file'
    ]

    NUMERIC_COLS = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'cb_person_cred_hist_length'
    ]

    def __init__(self):
        self.all_features = self.CATEGORICAL_COLS + self.NUMERIC_COLS

    def form_to_dataframe(self, form_data: Dict) -> pd.DataFrame:
        """
        Convert form inputs to DataFrame.

        Parameters
        ----------
        form_data : dict
            Keys match column names from CATEGORICAL_COLS + NUMERIC_COLS

        Returns
        -------
        pd.DataFrame
            Single row with loan application
        """
        df = pd.DataFrame([form_data])

        # Validate required fields
        missing = [col for col in self.all_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Ensure correct dtypes
        for col in self.NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in self.CATEGORICAL_COLS:
            df[col] = df[col].astype(str)

        return df[self.all_features]

    def csv_to_dataframe(self, uploaded_file) -> pd.DataFrame:
        """
        Convert uploaded CSV/Excel to DataFrame.

        Parameters
        ----------
        uploaded_file : streamlit.UploadedFile
            CSV or Excel file

        Returns
        -------
        pd.DataFrame
            Validated and preprocessed data
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Only CSV and Excel files supported")
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")

        # Drop loan_grade if present
        if 'loan_grade' in df.columns:
            df = df.drop('loan_grade', axis=1)

        # Validate columns
        missing = [col for col in self.all_features if col not in df.columns]
        if missing:
            raise ValueError(f"File missing columns: {missing}")

        # Clean and convert
        df = df[self.all_features]
        for col in self.NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in self.CATEGORICAL_COLS:
            df[col] = df[col].astype(str)

        return df

    def preprocess_for_prediction(self, X, feature_names):
        """
        One-hot encode and align features to match training data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features with categorical and numeric columns
        feature_names : list
            Expected feature names after one-hot encoding (from model)

        Returns
        -------
        pd.DataFrame
            One-hot encoded and aligned to training feature space
        """
        # One-hot encode categorical columns
        X_enc = pd.get_dummies(X, columns=self.CATEGORICAL_COLS, drop_first=False)

        # Align columns with training data
        for col in feature_names:
            if col not in X_enc.columns:
                X_enc[col] = 0

        # Keep only the features that were in training
        X_enc = X_enc[feature_names]

        return X_enc