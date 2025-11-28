from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_raw():
    """
    Load the raw credit risk dataset and return X, y (loan_status).
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop("loan_grade", axis=1)  # Remove loan_grade - user doesn't know this value
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    return X, y


def make_train_test(test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: bool = True):
    """
    Create a single train/test split on the raw features.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = load_raw()
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return X_train, X_test, y_train, y_test


def preprocess_tabular(X_train, X_test):
    """
    Apply one-hot encoding + median imputation + standard scaling.

    Returns
    -------
    X_train_imp : pd.DataFrame  # imputed, not scaled (for trees)
    X_test_imp  : pd.DataFrame
    X_train_scaled : pd.DataFrame  # imputed + scaled (for LR/NN)
    X_test_scaled  : pd.DataFrame
    enc_columns : list[str]  # column names after one-hot encoding
    imputer : fitted SimpleImputer
    scaler : fitted StandardScaler
    """
    # One-hot encode train and test, then align columns
    X_train_enc = pd.get_dummies(X_train)
    X_test_enc = pd.get_dummies(X_test)

    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    # Median imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_enc),
        columns=X_train_enc.columns,
        index=X_train_enc.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_enc),
        columns=X_train_enc.columns,
        index=X_test_enc.index,
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        columns=X_train_imp.columns,
        index=X_train_imp.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp),
        columns=X_train_imp.columns,
        index=X_test_imp.index,
    )

    return X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, list(X_train_enc.columns), imputer, scaler