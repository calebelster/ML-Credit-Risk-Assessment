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
# Changed from LogisticRegression to HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Load the data
df = pd.read_csv("data/credit_risk_dataset.csv")

# Separate features (X) and target (y)
y = df["loan_status"]
X = df.drop("loan_status", axis=1)


# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# --- Preprocessing Pipeline (Same as LogReg, but StandardScaler is not strictly necessary for tree models) ---

numeric_transformer = Pipeline(steps=[
    # Impute missing numeric values with the median
    ("imputer", SimpleImputer(strategy="median")),
    # Scaling is optional for tree models but kept for consistency/safety
    ("scaler", StandardScaler()) 
])

categorical_transformer = Pipeline(steps=[
    # Impute missing categorical values with the most frequent value
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # One-Hot Encode categorical features. HistGBM can handle categories natively, 
    # but OneHot is safer with the ColumnTransformer setup.
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --- Model Definition (Swapped Classifier) ---
# HistGradientBoostingClassifier is used. It handles class imbalance via sample weights
# or class_weight, but it's simpler to use the 'auto' or default behavior for the first run.

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # Change classifier here:
    ("classifier", GradientBoostingClassifier(random_state=42)) 
    # Note: For imbalanced data, you might later tune 'sample_weight' 
    # during fitting, as it doesn't have a direct 'class_weight' parameter.
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

# --- Model Evaluation Outputs ---

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
plt.title('Confusion Matrix (Gradient Boost Model)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, preds))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, preds))

print("\nROC-AUC SCORE:")
print(roc_auc_score(y_test, probs))


# --- Feature Importance Visualization (Adapted for Tree Models) ---

# 1. Extract feature names and importances
# Tree models use feature_importances_ instead of coef_
importances = model.named_steps["classifier"].feature_importances_
features = model.named_steps["preprocessor"].get_feature_names_out()


# 2. Create the DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances,
})

# 3. Sort by Importance value in descending order
top_10_sorted = importance_df.sort_values("Importance", ascending=False).head(10)

print("\nTOP 10 FEATURES BY IMPURITY-BASED IMPORTANCE:")
print(tabulate(top_10_sorted, headers="keys", tablefmt="psql"))

# 4. Prepare plot order
sorted_features_for_plot = top_10_sorted['Feature']


# 5. Create the plot
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance',               # Use the Importance value on the X-axis
    y='Feature',                  # Feature names on the Y-axis
    data=top_10_sorted,           # Use the DataFrame sorted by importance
    order=sorted_features_for_plot, # Explicitly set the y-axis order
    color='skyblue'               # Use a single color since importance is always positive
)

plt.title('Top 10 Gradient Boosting Feature Importances (Impurity-Based)')
plt.xlabel('Feature Importance (Gini/Impurity)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()