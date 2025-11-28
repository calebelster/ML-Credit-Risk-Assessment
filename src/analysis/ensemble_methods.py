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
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Load data and single stratified train/test split
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -------------------------------------------------------------------
# One-hot encode + impute + scale (fit on TRAIN only)
# -------------------------------------------------------------------
# One-hot
X_train_enc = pd.get_dummies(X_train_full)
X_test_enc = pd.get_dummies(X_test_full)

# Align columns
X_train_enc, X_test_enc = X_train_enc.align(
    X_test_enc, join="left", axis=1, fill_value=0
)

# Impute
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

# Scale
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

# -------------------------------------------------------------------
# Base models (fit ONLY on training data)
# -------------------------------------------------------------------
lr = LogisticRegression(max_iter=2500, solver="lbfgs", random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=2000, random_state=42)

# LR + NN on scaled; trees on imputed
lr.fit(X_train_scaled, y_train_full)
nn.fit(X_train_scaled, y_train_full)
rf.fit(X_train_imp, y_train_full)
gb.fit(X_train_imp, y_train_full)

# Store both train and test views per model
base_models = [
    ("lr", lr, X_train_scaled, X_test_scaled),
    ("rf", rf, X_train_imp, X_test_imp),
    ("gb", gb, X_train_imp, X_test_imp),
    ("nn", nn, X_train_scaled, X_test_scaled),
]

# -------------------------------------------------------------------
# SOFT VOTING (predict on TEST only)
# -------------------------------------------------------------------
voting = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("nn", nn)],
    voting="soft",
)

# VotingClassifier works on a single feature matrix; use scaled version
voting.fit(X_train_scaled, y_train_full)
voting_probs = voting.predict_proba(X_test_scaled)[:, 1]
voting_preds = voting.predict(X_test_scaled)

pd.DataFrame(
    {
        "y_true": y_test_full,
        "y_pred_prob": voting_probs,
        "y_pred": voting_preds,
    }
).to_csv(OUTPUT_DIR / "ensemble_voting_preds.csv", index=False)

# -------------------------------------------------------------------
# BAGGING (average probabilities from base models on TEST only)
# -------------------------------------------------------------------
bagging_probs = np.mean(
    [
        model.predict_proba(X_test_use)[:, 1]
        for (_, model, _, X_test_use) in base_models
    ],
    axis=0,
)
bagging_preds = (bagging_probs > 0.5).astype(int)

pd.DataFrame(
    {
        "y_true": y_test_full,
        "y_pred_prob": bagging_probs,
        "y_pred": bagging_preds,
    }
).to_csv(OUTPUT_DIR / "ensemble_bagging_preds.csv", index=False)

# -------------------------------------------------------------------
# STACKING (OOF on TRAIN, evaluate on TEST)
# -------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_train = X_train_imp.shape[0]
n_test = X_test_imp.shape[0]

meta_features_train = np.zeros((n_train, len(base_models)))
meta_features_test = np.zeros((n_test, len(base_models)))

for i, (name, base, Xtr_use, Xte_use) in enumerate(base_models):
    # Out-of-fold predictions for TRAIN
    oof_preds = np.zeros(n_train)
    for train_idx, val_idx in kf.split(Xtr_use, y_train_full):
        # clone the base estimator
        base_clone = type(base)(**base.get_params())
        base_clone.fit(Xtr_use.iloc[train_idx], y_train_full.iloc[train_idx])
        oof_preds[val_idx] = base_clone.predict_proba(Xtr_use.iloc[val_idx])[:, 1]
    meta_features_train[:, i] = oof_preds

    # Fit on all TRAIN, predict TEST
    base_final = type(base)(**base.get_params())
    base_final.fit(Xtr_use, y_train_full)
    meta_features_test[:, i] = base_final.predict_proba(Xte_use)[:, 1]

stack_scaler = StandardScaler()
meta_train_scaled = stack_scaler.fit_transform(meta_features_train)
meta_test_scaled = stack_scaler.transform(meta_features_test)

stacker = LogisticRegression(max_iter=2500, random_state=42)
stacker.fit(meta_train_scaled, y_train_full)

stack_probs = stacker.predict_proba(meta_test_scaled)[:, 1]
stack_preds = (stack_probs > 0.5).astype(int)

pd.DataFrame(
    {
        "y_true": y_test_full,
        "y_pred_prob": stack_probs,
        "y_pred": stack_preds,
    }
).to_csv(OUTPUT_DIR / "ensemble_stacking_preds.csv", index=False)

# -------------------------------------------------------------------
# BLENDING (outer train/test; inner train/blend on TRAIN)
# -------------------------------------------------------------------
# Meta-features from base models on TRAIN and TEST
meta_train_full = np.column_stack(
    [model.predict_proba(Xtr_use)[:, 1] for (_, model, Xtr_use, _) in base_models]
)
meta_test_full = np.column_stack(
    [model.predict_proba(Xte_use)[:, 1] for (_, model, _, Xte_use) in base_models]
)

# Inner split on TRAIN meta-features for blending
X_meta_train, X_meta_blend, y_meta_train, y_meta_blend = train_test_split(
    meta_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full,
)

blend_scaler = StandardScaler()
X_meta_blend_scaled = blend_scaler.fit_transform(X_meta_blend)
X_meta_test_scaled = blend_scaler.transform(meta_test_full)

meta_model = LogisticRegression(max_iter=2500, random_state=42)
meta_model.fit(X_meta_blend_scaled, y_meta_blend)

blend_probs = meta_model.predict_proba(X_meta_test_scaled)[:, 1]
blend_preds = (blend_probs > 0.5).astype(int)

pd.DataFrame(
    {
        "y_true": y_test_full,
        "y_pred_prob": blend_probs,
        "y_pred": blend_preds,
    }
).to_csv(OUTPUT_DIR / "ensemble_blending_preds.csv", index=False)

print("All ensemble outputs (Voting, Bagging, Stacking, Blending) written using a clean train/test split.")