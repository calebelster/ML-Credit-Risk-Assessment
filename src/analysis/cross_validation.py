from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    average_precision_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def ks_statistic(y_true, y_scores):
    return ks_2samp(y_scores[y_true == 0], y_scores[y_true == 1]).statistic


def load_and_preprocess():
    """
    Load data, drop loan_grade, one-hot encode, impute, scale.

    Returns
    -------
    X_imp : pd.DataFrame
        Imputed (not scaled) features for tree models
    X_scaled : pd.DataFrame
        Imputed + scaled features for LR/NN
    y : np.ndarray
        Target (0 = no default, 1 = default)
    """
    df = pd.read_csv(DATA_PATH)

    # Remove loan_grade (user doesn’t know this value)
    if "loan_grade" in df.columns:
        df = df.drop("loan_grade", axis=1)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"].values

    # One-hot encode
    X_enc = pd.get_dummies(X)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_enc),
        columns=X_enc.columns,
        index=X_enc.index,
    )

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imp),
        columns=X_imp.columns,
        index=X_imp.index,
    )

    return X_imp, X_scaled, y


def cross_validate_all(n_splits=5, random_state=42):
    """
    Perform K-fold CV for:
      - Random Forest
      - Gradient Boost
      - Logistic Regression
      - Neural Net
      - Ensemble Bagging
      - Ensemble Voting
      - Ensemble Stacking
      - Ensemble Blending

    Writes one CSV per model into OUTPUT_DIR.
    """
    X_imp, X_scaled, y = load_and_preprocess()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Containers for metrics
    base_model_metrics = {
        "random_forest": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "grad_boost": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "log_reg": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "neural_net": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
    }

    ensemble_metrics = {
        "bagging": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "voting": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "stacking": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "blending": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
    }

    fold_idx = 0
    for train_idx, test_idx in skf.split(X_imp, y):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")

        X_imp_train, X_imp_test = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
        X_scaled_train, X_scaled_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ------------------------------
        # Base models
        # ------------------------------
        lr = LogisticRegression(max_iter=2500, solver="lbfgs", random_state=random_state)
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        nn = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=2000, random_state=random_state)

        # Fit base models (LR/NN on scaled, RF/GB on imp)
        lr.fit(X_scaled_train, y_train)
        nn.fit(X_scaled_train, y_train)
        rf.fit(X_imp_train, y_train)
        gb.fit(X_imp_train, y_train)

        # Predict probabilities for each base model
        # Random Forest
        rf_prob = rf.predict_proba(X_imp_test)[:, 1]
        base_model_metrics["random_forest"]["AUC"].append(roc_auc_score(y_test, rf_prob))
        base_model_metrics["random_forest"]["PR_AUC"].append(average_precision_score(y_test, rf_prob))
        base_model_metrics["random_forest"]["Brier"].append(brier_score_loss(y_test, rf_prob))
        base_model_metrics["random_forest"]["KS"].append(ks_statistic(y_test, rf_prob))
        base_model_metrics["random_forest"]["LogLoss"].append(log_loss(y_test, rf_prob))

        # Gradient Boost
        gb_prob = gb.predict_proba(X_imp_test)[:, 1]
        base_model_metrics["grad_boost"]["AUC"].append(roc_auc_score(y_test, gb_prob))
        base_model_metrics["grad_boost"]["PR_AUC"].append(average_precision_score(y_test, gb_prob))
        base_model_metrics["grad_boost"]["Brier"].append(brier_score_loss(y_test, gb_prob))
        base_model_metrics["grad_boost"]["KS"].append(ks_statistic(y_test, gb_prob))
        base_model_metrics["grad_boost"]["LogLoss"].append(log_loss(y_test, gb_prob))

        # Logistic Regression
        lr_prob = lr.predict_proba(X_scaled_test)[:, 1]
        base_model_metrics["log_reg"]["AUC"].append(roc_auc_score(y_test, lr_prob))
        base_model_metrics["log_reg"]["PR_AUC"].append(average_precision_score(y_test, lr_prob))
        base_model_metrics["log_reg"]["Brier"].append(brier_score_loss(y_test, lr_prob))
        base_model_metrics["log_reg"]["KS"].append(ks_statistic(y_test, lr_prob))
        base_model_metrics["log_reg"]["LogLoss"].append(log_loss(y_test, lr_prob))

        # Neural Net
        nn_prob = nn.predict_proba(X_scaled_test)[:, 1]
        base_model_metrics["neural_net"]["AUC"].append(roc_auc_score(y_test, nn_prob))
        base_model_metrics["neural_net"]["PR_AUC"].append(average_precision_score(y_test, nn_prob))
        base_model_metrics["neural_net"]["Brier"].append(brier_score_loss(y_test, nn_prob))
        base_model_metrics["neural_net"]["KS"].append(ks_statistic(y_test, nn_prob))
        base_model_metrics["neural_net"]["LogLoss"].append(log_loss(y_test, nn_prob))

        # ------------------------------
        # Build base_models list used by ensembles
        # ------------------------------
        base_models = [
            ("lr", lr, X_scaled_train, X_scaled_test),
            ("rf", rf, X_imp_train, X_imp_test),
            ("gb", gb, X_imp_train, X_imp_test),
            ("nn", nn, X_scaled_train, X_scaled_test),
        ]

        # ------------------------------
        # Voting (soft)
        # ------------------------------
        voting = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("nn", nn)],
            voting="soft",
        )
        voting.fit(X_scaled_train, y_train)
        y_prob_v = voting.predict_proba(X_scaled_test)[:, 1]

        ensemble_metrics["voting"]["AUC"].append(roc_auc_score(y_test, y_prob_v))
        ensemble_metrics["voting"]["PR_AUC"].append(average_precision_score(y_test, y_prob_v))
        ensemble_metrics["voting"]["Brier"].append(brier_score_loss(y_test, y_prob_v))
        ensemble_metrics["voting"]["KS"].append(ks_statistic(y_test, y_prob_v))
        ensemble_metrics["voting"]["LogLoss"].append(log_loss(y_test, y_prob_v))

        # ------------------------------
        # Bagging (average of base probs)
        # ------------------------------
        bagging_probs = np.mean(
            [model.predict_proba(Xte)[:, 1] for (_, model, _, Xte) in base_models],
            axis=0,
        )
        ensemble_metrics["bagging"]["AUC"].append(roc_auc_score(y_test, bagging_probs))
        ensemble_metrics["bagging"]["PR_AUC"].append(average_precision_score(y_test, bagging_probs))
        ensemble_metrics["bagging"]["Brier"].append(brier_score_loss(y_test, bagging_probs))
        ensemble_metrics["bagging"]["KS"].append(ks_statistic(y_test, bagging_probs))
        ensemble_metrics["bagging"]["LogLoss"].append(log_loss(y_test, bagging_probs))

        # ------------------------------
        # Stacking (OOF on train in this fold)
        # ------------------------------
        inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        n_train_fold = X_imp_train.shape[0]
        meta_train_fold = np.zeros((n_train_fold, len(base_models)))
        meta_test_fold = np.zeros((X_imp_test.shape[0], len(base_models)))

        for i, (name, model, Xtr_use, Xte_use) in enumerate(base_models):
            # OOF on TRAIN
            oof_preds = np.zeros(n_train_fold)
            for inner_train_idx, inner_val_idx in inner_kf.split(Xtr_use, y_train):
                m_clone = type(model)(**model.get_params())
                m_clone.fit(Xtr_use.iloc[inner_train_idx], y_train[inner_train_idx])
                oof_preds[inner_val_idx] = m_clone.predict_proba(Xtr_use.iloc[inner_val_idx])[:, 1]
            meta_train_fold[:, i] = oof_preds

            # Fit on all TRAIN in this fold, predict TEST
            m_final = type(model)(**model.get_params())
            m_final.fit(Xtr_use, y_train)
            meta_test_fold[:, i] = m_final.predict_proba(Xte_use)[:, 1]

        # Scale meta-features and fit meta-learner
        stack_scaler = StandardScaler()
        meta_train_scaled = stack_scaler.fit_transform(meta_train_fold)
        meta_test_scaled = stack_scaler.transform(meta_test_fold)

        stacker = LogisticRegression(max_iter=2500, random_state=random_state)
        stacker.fit(meta_train_scaled, y_train)
        y_prob_stack = stacker.predict_proba(meta_test_scaled)[:, 1]

        ensemble_metrics["stacking"]["AUC"].append(roc_auc_score(y_test, y_prob_stack))
        ensemble_metrics["stacking"]["PR_AUC"].append(average_precision_score(y_test, y_prob_stack))
        ensemble_metrics["stacking"]["Brier"].append(brier_score_loss(y_test, y_prob_stack))
        ensemble_metrics["stacking"]["KS"].append(ks_statistic(y_test, y_prob_stack))
        ensemble_metrics["stacking"]["LogLoss"].append(log_loss(y_test, y_prob_stack))

        # ------------------------------
        # Blending (inner split on TRAIN, evaluate on TEST)
        # ------------------------------
        meta_train_full = np.column_stack(
            [model.predict_proba(Xtr)[:, 1] for (_, model, Xtr, _) in base_models]
        )
        meta_test_full = np.column_stack(
            [model.predict_proba(Xte)[:, 1] for (_, model, _, Xte) in base_models]
        )

        X_meta_train, X_meta_blend, y_meta_train, y_meta_blend = train_test_split(
            meta_train_full,
            y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=y_train,
        )

        blend_scaler = StandardScaler()
        X_meta_blend_scaled = blend_scaler.fit_transform(X_meta_blend)
        X_meta_test_scaled = blend_scaler.transform(meta_test_full)

        meta_model = LogisticRegression(max_iter=2500, random_state=random_state)
        meta_model.fit(X_meta_blend_scaled, y_meta_blend)

        y_prob_blend = meta_model.predict_proba(X_meta_test_scaled)[:, 1]

        ensemble_metrics["blending"]["AUC"].append(roc_auc_score(y_test, y_prob_blend))
        ensemble_metrics["blending"]["PR_AUC"].append(average_precision_score(y_test, y_prob_blend))
        ensemble_metrics["blending"]["Brier"].append(brier_score_loss(y_test, y_prob_blend))
        ensemble_metrics["blending"]["KS"].append(ks_statistic(y_test, y_prob_blend))
        ensemble_metrics["blending"]["LogLoss"].append(log_loss(y_test, y_prob_blend))

    # --------------------------------
    # After all folds, write CSVs
    # --------------------------------
    # Base models
    for name, m in base_model_metrics.items():
        df_metrics = pd.DataFrame(m)
        out_file = OUTPUT_DIR / f"{name}_cv_metrics.csv"
        df_metrics.to_csv(out_file, index=False)
        print(f"Saved CV metrics for {name} to {out_file}")

    # Ensembles
    for name, m in ensemble_metrics.items():
        df_metrics = pd.DataFrame(m)
        out_file = OUTPUT_DIR / f"ensemble_{name}_cv_metrics.csv"
        df_metrics.to_csv(out_file, index=False)
        print(f"Saved CV metrics for ensemble {name} to {out_file}")


if __name__ == "__main__":
    print("Running K-fold CV for base and ensemble models...")
    cross_validate_all()
    print("Done.")