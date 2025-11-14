import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_data(filepath):
    """Load CSV data and return DataFrame."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found.")

def clean_data(df):
    """Drop rows with missing values."""
    return df.dropna()

def encode_categoricals(df, categorical_cols):
    """One-hot encode categorical features."""
    return pd.get_dummies(df, columns=categorical_cols)

def scale_numeric(X):
    """Scale features for neural net."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

if __name__ == "__main__":
    # Example usage for standalone testing
    DATAPATH = "../data/credit_risk_dataset.csv"
    df = read_data(DATAPATH)
    df = clean_data(df)
    cats = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
    df = encode_categoricals(df, cats)
    print(df.head())