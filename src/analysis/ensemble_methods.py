import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"

# Load and preprocess
df = pd.read_csv(DATA_PATH)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# One-hot encode + impute missing values
X_enc = pd.get_dummies(X)
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

# Standardize features for models that need it
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

# --- Fit base models for ensembling ---
lr = LogisticRegression(max_iter=2500, solver="lbfgs", random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=2000, random_state=42)

# Note: neural net and logistic regression use scaled data, tree models use imputed
lr.fit(X_scaled, y)
nn.fit(X_scaled, y)
rf.fit(X_imp, y)
gb.fit(X_imp, y)

base_models = [
    ("lr", lr, X_scaled),
    ("rf", rf, X_imp),
    ("gb", gb, X_imp),
    ("nn", nn, X_scaled)
]

# SOFT VOTING (with correct X per model)
voting = VotingClassifier(
    estimators=[
        ("lr", lr), ("rf", rf), ("gb", gb), ("nn", nn)
    ],
    voting="soft"
)
# For VotingClassifier: all X must be aligned. Use all scaled for classification.
voting.fit(X_scaled, y)
voting_probs = voting.predict_proba(X_scaled)[:, 1]
voting_preds = voting.predict(X_scaled)
pd.DataFrame({
    "y_true": y, "y_pred_prob": voting_probs, "y_pred": voting_preds
}).to_csv(OUTPUT_DIR / "ensemble_voting_preds.csv", index=False)

# BAGGING (average probabilities, use model-specific X)
bagging_probs = np.mean([
    base.predict_proba(X_use)[:, 1] for (_, base, X_use) in base_models
], axis=0)
bagging_preds = (bagging_probs > 0.5).astype(int)
pd.DataFrame({
    "y_true": y, "y_pred_prob": bagging_probs, "y_pred": bagging_preds
}).to_csv(OUTPUT_DIR / "ensemble_bagging_preds.csv", index=False)

# STACKING (logreg meta, 5-fold OOF, use correct preprocessing for each base)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features = np.zeros((X_imp.shape[0], len(base_models)))

for i, (name, base, X_use) in enumerate(base_models):
    preds = np.zeros(X_use.shape[0])
    for train_idx, test_idx in kf.split(X_use):
        base.fit(X_use.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = base.predict_proba(X_use.iloc[test_idx])[:, 1]
    meta_features[:, i] = preds

# Use logistic regression as the meta-learner (requires scaling)
stack_scaler = StandardScaler()
meta_features_scaled = stack_scaler.fit_transform(meta_features)
stacker = LogisticRegression(max_iter=2500, random_state=42)
stacker.fit(meta_features_scaled, y)
stack_probs = stacker.predict_proba(meta_features_scaled)[:, 1]
stack_preds = (stack_probs > 0.5).astype(int)
pd.DataFrame({
    "y_true": y, "y_pred_prob": stack_probs, "y_pred": stack_preds
}).to_csv(OUTPUT_DIR / "ensemble_stacking_preds.csv", index=False)

# BLENDING (logreg meta-model on random 20% holdout, predicts all)
# Meta-features: full base model probability matrix
meta_X = np.column_stack([
    base.predict_proba(X_use)[:, 1] for (_, base, X_use) in base_models
])
X_train, X_blend, y_train, y_blend = train_test_split(meta_X, y, test_size=0.2, random_state=42)
blend_scaler = StandardScaler()
X_blend_scaled = blend_scaler.fit_transform(X_blend)
X_full_scaled = blend_scaler.transform(meta_X)
meta_model = LogisticRegression(max_iter=2500, random_state=42)
meta_model.fit(X_blend_scaled, y_blend)
blend_probs = meta_model.predict_proba(X_full_scaled)[:, 1]
blend_preds = (blend_probs > 0.5).astype(int)
pd.DataFrame({
    "y_true": y, "y_pred_prob": blend_probs, "y_pred": blend_preds
}).to_csv(OUTPUT_DIR / "ensemble_blending_preds.csv", index=False)

print("All ensemble outputs (Voting, Bagging, Stacking, Blending) written with imputation and scaling.")