import pandas as pd
import numpy as np
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Load the data
df = pd.read_csv("data/credit_risk_dataset.csv")

y = df["loan_status"]
X = df.drop("loan_status", axis=1)


# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# Define transformers
numeric_transformer = Pipeline(steps=[
    # Impute missing numeric values with the median
    ("imputer", SimpleImputer(strategy="median")),
    # Standardize the numeric features
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # Impute missing categorical values with the most frequent value
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # One-Hot Encode categorical features
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Define the model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train the model
model.fit(X_train, y_train)


# Make predictions
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Model Eval

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
class_labels = sorted(list(set(y_test)))

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Default', 'Default'], 
    yticklabels=['Not Default', 'Default']
)
plt.title('Confusion Matrix (Balanced Model)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, preds))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, preds))

print("\nROC-AUC SCORE:")
print(roc_auc_score(y_test, probs))


# Feature Importance Visualization

# Extract feature names and coefficients
coefs = model.named_steps["classifier"].coef_[0]
features = model.named_steps["preprocessor"].get_feature_names_out()

# Create the DataFrame 
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefs,
    "Abs_Coefficient": np.abs(coefs)
})

# Sort the full DataFrame by abs value in descending order
top_10_abs_sorted = coef_df.sort_values("Abs_Coefficient", ascending=False).head(10)

# Create plot
print("\nTOP 10 FEATURES BY ABSOLUTE COEFFICIENT:")
print(tabulate(top_10_abs_sorted, headers="keys", tablefmt="psql"))

sorted_features_for_plot = top_10_abs_sorted['Feature']

plt.figure(figsize=(14, 10))
sns.barplot(
    x='Coefficient',       
    y='Feature',
    data=top_10_abs_sorted,
    order=sorted_features_for_plot,

    palette='vlag'
)

plt.title('Top 10 Logistic Regression Feature Coefficients (Sorted by Absolute Magnitude)')
plt.xlabel('Coefficient Value (Effect on Log-Odds)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()