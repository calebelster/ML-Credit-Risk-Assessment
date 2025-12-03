import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sk_train_test_split

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "app" / "saved_models"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Model save directory: {MODEL_SAVE_DIR}")

# -------------------------------------------------------------------
# Load data and REMOVE loan_grade
# -------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
if "loan_grade" in df.columns:
    df = df.drop("loan_grade", axis=1)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train set: {X_train_full.shape[0]} samples")
print(f"Test set: {X_test_full.shape[0]} samples")
print(f"Features: {X_train_full.shape[1]} (loan_grade removed)")

# -------------------------------------------------------------------
# One-hot encode + impute + scale (fit on TRAIN only)
# -------------------------------------------------------------------

X_train_enc = pd.get_dummies(X_train_full)
X_test_enc = pd.get_dummies(X_test_full)

X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

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

print(f"Features after preprocessing: {X_train_scaled.shape[1]}")

# -------------------------------------------------------------------
# Base models
# -------------------------------------------------------------------

lr = LogisticRegression(max_iter=2500, solver="lbfgs", random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=2000, random_state=42)

print("\nFitting base models...")
lr.fit(X_train_scaled, y_train_full)
nn.fit(X_train_scaled, y_train_full)
rf.fit(X_train_imp, y_train_full)
gb.fit(X_train_imp, y_train_full)
print("Base models fitted.")

# Full and reduced base-model sets
base_models_full = [
    ("lr", lr, X_train_scaled, X_test_scaled),
    ("rf", rf, X_train_imp, X_test_imp),
    ("gb", gb, X_train_imp, X_test_imp),
    ("nn", nn, X_train_scaled, X_test_scaled),
]

base_models_no_nn = [
    ("lr", lr, X_train_scaled, X_test_scaled),
    ("rf", rf, X_train_imp, X_test_imp),
    ("gb", gb, X_train_imp, X_test_imp),
]

base_models_rf_gb = [
    ("rf", rf, X_train_imp, X_test_imp),
    ("gb", gb, X_train_imp, X_test_imp),
]

# -------------------------------------------------------------------
# Voting ensembles
# -------------------------------------------------------------------

print("\nBuilding Voting ensembles...")

# full
voting_full = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("nn", nn)],
    voting="soft",
)
voting_full.fit(X_train_scaled, y_train_full)
v_full_probs = voting_full.predict_proba(X_test_scaled)[:, 1]
v_full_preds = voting_full.predict(X_test_scaled)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": v_full_probs, "y_pred": v_full_preds}
).to_csv(OUTPUT_DIR / "ensemble_voting_full_preds.csv", index=False)

# no NN
voting_no_nn = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    voting="soft",
)
voting_no_nn.fit(X_train_scaled, y_train_full)
v_nonn_probs = voting_no_nn.predict_proba(X_test_scaled)[:, 1]
v_nonn_preds = voting_no_nn.predict(X_test_scaled)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": v_nonn_probs, "y_pred": v_nonn_preds}
).to_csv(OUTPUT_DIR / "ensemble_voting_no_nn_preds.csv", index=False)

# RF + GB
voting_rf_gb = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft",
)
voting_rf_gb.fit(X_train_imp, y_train_full)
v_rf_gb_probs = voting_rf_gb.predict_proba(X_test_imp)[:, 1]
v_rf_gb_preds = voting_rf_gb.predict(X_test_imp)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": v_rf_gb_probs, "y_pred": v_rf_gb_preds}
).to_csv(OUTPUT_DIR / "ensemble_voting_rf_gb_preds.csv", index=False)

print("Voting ensembles saved.")

# -------------------------------------------------------------------
# Bagging ensembles
# -------------------------------------------------------------------

print("\nBuilding Bagging ensembles...")


def bagging_from_base_list(base_list):
    return np.mean(
        [model.predict_proba(X_test_use)[:, 1] for (_, model, _, X_test_use) in base_list],
        axis=0,
    )


# full
b_full_probs = bagging_from_base_list(base_models_full)
b_full_preds = (b_full_probs > 0.5).astype(int)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": b_full_probs, "y_pred": b_full_preds}
).to_csv(OUTPUT_DIR / "ensemble_bagging_full_preds.csv", index=False)

# no NN
b_nonn_probs = bagging_from_base_list(base_models_no_nn)
b_nonn_preds = (b_nonn_probs > 0.5).astype(int)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": b_nonn_probs, "y_pred": b_nonn_preds}
).to_csv(OUTPUT_DIR / "ensemble_bagging_no_nn_preds.csv", index=False)

# RF + GB
b_rf_gb_probs = bagging_from_base_list(base_models_rf_gb)
b_rf_gb_preds = (b_rf_gb_probs > 0.5).astype(int)
pd.DataFrame(
    {"y_true": y_test_full.values, "y_pred_prob": b_rf_gb_probs, "y_pred": b_rf_gb_preds}
).to_csv(OUTPUT_DIR / "ensemble_bagging_rf_gb_preds.csv", index=False)

print("Bagging ensembles saved.")

# -------------------------------------------------------------------
# Stacking ensembles
# -------------------------------------------------------------------

print("\nBuilding Stacking ensembles...")


def build_stacking_from_base_list(base_list, suffix):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_train = X_train_imp.shape[0]
    n_test = X_test_imp.shape[0]
    meta_train = np.zeros((n_train, len(base_list)))
    meta_test = np.zeros((n_test, len(base_list)))

    for i, (name, base, Xtr_use, Xte_use) in enumerate(base_list):
        print(f" OOF predictions for {name} ({suffix})...")
        oof = np.zeros(n_train)
        for inner_train_idx, inner_val_idx in kf.split(Xtr_use, y_train_full):
            base_clone = type(base)(**base.get_params())
            base_clone.fit(Xtr_use.iloc[inner_train_idx], y_train_full.iloc[inner_train_idx])
            oof[inner_val_idx] = base_clone.predict_proba(Xtr_use.iloc[inner_val_idx])[:, 1]
        meta_train[:, i] = oof

        base_final = type(base)(**base.get_params())
        base_final.fit(Xtr_use, y_train_full)
        meta_test[:, i] = base_final.predict_proba(Xte_use)[:, 1]

    stack_scaler = StandardScaler()
    meta_train_scaled = stack_scaler.fit_transform(meta_train)
    meta_test_scaled = stack_scaler.transform(meta_test)

    stacker = LogisticRegression(max_iter=2500, random_state=42)
    stacker.fit(meta_train_scaled, y_train_full)
    probs = stacker.predict_proba(meta_test_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)

    out = OUTPUT_DIR / f"ensemble_stacking_{suffix}_preds.csv"
    pd.DataFrame(
        {"y_true": y_test_full.values, "y_pred_prob": probs, "y_pred": preds}
    ).to_csv(out, index=False)
    print(f"Stacking ({suffix}) predictions saved to {out}")
    return stacker, stack_scaler


stacker_full, stack_scaler_full = build_stacking_from_base_list(base_models_full, "full")
stacker_no_nn, stack_scaler_no_nn = build_stacking_from_base_list(base_models_no_nn, "no_nn")
stacker_rf_gb, stack_scaler_rf_gb = build_stacking_from_base_list(base_models_rf_gb, "rf_gb")

# -------------------------------------------------------------------
# Blending ensembles
# -------------------------------------------------------------------

print("\nBuilding Blending ensembles...")


def build_blending_from_base_list(base_list, suffix):
    meta_train_full = np.column_stack(
        [model.predict_proba(Xtr_use)[:, 1] for (_, model, Xtr_use, _) in base_list]
    )
    meta_test_full = np.column_stack(
        [model.predict_proba(Xte_use)[:, 1] for (_, model, _, Xte_use) in base_list]
    )

    X_meta_train, X_meta_blend, y_meta_train, y_meta_blend = sk_train_test_split(
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

    probs = meta_model.predict_proba(X_meta_test_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)

    out = OUTPUT_DIR / f"ensemble_blending_{suffix}_preds.csv"
    pd.DataFrame(
        {"y_true": y_test_full.values, "y_pred_prob": probs, "y_pred": preds}
    ).to_csv(out, index=False)
    print(f"Blending ({suffix}) predictions saved to {out}")


build_blending_from_base_list(base_models_full, "full")
build_blending_from_base_list(base_models_no_nn, "no_nn")
build_blending_from_base_list(base_models_rf_gb, "rf_gb")

# -------------------------------------------------------------------
# Save deployment stacking model (RF+GB variant)
# -------------------------------------------------------------------

print("\nSaving Stacking model for deployment (RF+GB base set)...")

stacking_artifacts = {
    "base_models": {
        "rf": rf,
        "gb": gb,
    },
    "meta_learner": stacker_rf_gb,
    "meta_scaler": stack_scaler_rf_gb,
    "feature_imputer": imputer,
    "feature_scaler": scaler,
    "feature_names": list(X_train_imp.columns),
}

model_path = MODEL_SAVE_DIR / "stacking_model.pkl"
joblib.dump(stacking_artifacts, model_path)
print(f"Model saved to: {model_path}")

print("\n" + "=" * 70)
print("All ensemble outputs written using a clean train/test split.")
print(f"Model saved for deployment at: {model_path}")
print("=" * 70)