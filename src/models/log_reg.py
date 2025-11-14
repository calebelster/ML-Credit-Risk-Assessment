import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
y = df["loan_status"]
X = df.drop("loan_status", axis=1)
numeric_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({'y_true': y_test, 'y_pred_prob': probs, 'y_pred': preds})
results.to_csv(OUTPUT_PATH / "log_reg_preds.csv", index=False)

coefs = model.named_steps["classifier"].coef_[0]
features = model.named_steps["preprocessor"].get_feature_names_out()
coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
coef_df.to_csv(OUTPUT_PATH / "log_reg_coefs.csv", index=False)

print("ROC-AUC:", roc_auc_score(y_test, probs))
print("Classification report:\n", classification_report(y_test, preds))