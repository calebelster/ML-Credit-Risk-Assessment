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
    df = pd.read_csv(DATA_PATH)

    if "loan_grade" in df.columns:
        df = df.drop("loan_grade", axis=1)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"].values

    X_enc = pd.get_dummies(X)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_enc),
        columns=X_enc.columns,
        index=X_enc.index,
    )

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
      - Base models (RF, GB, LR, NN)
      - Ensemble Bagging/Voting/Stacking/Blending for:
          * full base set (LR+RF+GB+NN)
          * no NN (LR+RF+GB)
          * RF+GB only
    """
    X_imp, X_scaled, y = load_and_preprocess()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    base_model_metrics = {
        "random_forest": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "grad_boost": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "log_reg": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        "neural_net": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
    }

    # ensemble_metrics[ensemble_type][variant] -> dict of lists
    ensemble_metrics = {
        "bagging": {
            "full": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "no_nn": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "rf_gb": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        },
        "voting": {
            "full": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "no_nn": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "rf_gb": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        },
        "stacking": {
            "full": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "no_nn": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "rf_gb": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        },
        "blending": {
            "full": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "no_nn": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
            "rf_gb": {"AUC": [], "PR_AUC": [], "Brier": [], "KS": [], "LogLoss": []},
        },
    }

    fold_idx = 0
    for train_idx, test_idx in skf.split(X_imp, y):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")

        X_imp_train, X_imp_test = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
        X_scaled_train, X_scaled_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Base models
        lr = LogisticRegression(max_iter=2500, solver="lbfgs", random_state=random_state)
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        nn = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=2000, random_state=random_state)

        lr.fit(X_scaled_train, y_train)
        nn.fit(X_scaled_train, y_train)
        rf.fit(X_imp_train, y_train)
        gb.fit(X_imp_train, y_train)

        # Base metrics
        rf_prob = rf.predict_proba(X_imp_test)[:, 1]
        base_model_metrics["random_forest"]["AUC"].append(roc_auc_score(y_test, rf_prob))
        base_model_metrics["random_forest"]["PR_AUC"].append(average_precision_score(y_test, rf_prob))
        base_model_metrics["random_forest"]["Brier"].append(brier_score_loss(y_test, rf_prob))
        base_model_metrics["random_forest"]["KS"].append(ks_statistic(y_test, rf_prob))
        base_model_metrics["random_forest"]["LogLoss"].append(log_loss(y_test, rf_prob))

        gb_prob = gb.predict_proba(X_imp_test)[:, 1]
        base_model_metrics["grad_boost"]["AUC"].append(roc_auc_score(y_test, gb_prob))
        base_model_metrics["grad_boost"]["PR_AUC"].append(average_precision_score(y_test, gb_prob))
        base_model_metrics["grad_boost"]["Brier"].append(brier_score_loss(y_test, gb_prob))
        base_model_metrics["grad_boost"]["KS"].append(ks_statistic(y_test, gb_prob))
        base_model_metrics["grad_boost"]["LogLoss"].append(log_loss(y_test, gb_prob))

        lr_prob = lr.predict_proba(X_scaled_test)[:, 1]
        base_model_metrics["log_reg"]["AUC"].append(roc_auc_score(y_test, lr_prob))
        base_model_metrics["log_reg"]["PR_AUC"].append(average_precision_score(y_test, lr_prob))
        base_model_metrics["log_reg"]["Brier"].append(brier_score_loss(y_test, lr_prob))
        base_model_metrics["log_reg"]["KS"].append(ks_statistic(y_test, lr_prob))
        base_model_metrics["log_reg"]["LogLoss"].append(log_loss(y_test, lr_prob))

        nn_prob = nn.predict_proba(X_scaled_test)[:, 1]
        base_model_metrics["neural_net"]["AUC"].append(roc_auc_score(y_test, nn_prob))
        base_model_metrics["neural_net"]["PR_AUC"].append(average_precision_score(y_test, nn_prob))
        base_model_metrics["neural_net"]["Brier"].append(brier_score_loss(y_test, nn_prob))
        base_model_metrics["neural_net"]["KS"].append(ks_statistic(y_test, nn_prob))
        base_model_metrics["neural_net"]["LogLoss"].append(log_loss(y_test, nn_prob))

        # Base lists for ensembles
        base_full = [
            ("lr", lr, X_scaled_train, X_scaled_test),
            ("rf", rf, X_imp_train, X_imp_test),
            ("gb", gb, X_imp_train, X_imp_test),
            ("nn", nn, X_scaled_train, X_scaled_test),
        ]
        base_no_nn = [
            ("lr", lr, X_scaled_train, X_scaled_test),
            ("rf", rf, X_imp_train, X_imp_test),
            ("gb", gb, X_imp_train, X_imp_test),
        ]
        base_rf_gb = [
            ("rf", rf, X_imp_train, X_imp_test),
            ("gb", gb, X_imp_train, X_imp_test),
        ]

        # Voting
        voting_full = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("nn", nn)],
            voting="soft",
        )
        voting_full.fit(X_scaled_train, y_train)
        pv = voting_full.predict_proba(X_scaled_test)[:, 1]
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pv)),
            ("PR_AUC", average_precision_score(y_test, pv)),
            ("Brier", brier_score_loss(y_test, pv)),
            ("KS", ks_statistic(y_test, pv)),
            ("LogLoss", log_loss(y_test, pv)),
        ]:
            ensemble_metrics["voting"]["full"][mname].append(val)

        voting_no_nn = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
            voting="soft",
        )
        voting_no_nn.fit(X_scaled_train, y_train)
        pv2 = voting_no_nn.predict_proba(X_scaled_test)[:, 1]
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pv2)),
            ("PR_AUC", average_precision_score(y_test, pv2)),
            ("Brier", brier_score_loss(y_test, pv2)),
            ("KS", ks_statistic(y_test, pv2)),
            ("LogLoss", log_loss(y_test, pv2)),
        ]:
            ensemble_metrics["voting"]["no_nn"][mname].append(val)

        voting_rf_gb = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
        )
        voting_rf_gb.fit(X_imp_train, y_train)
        pv3 = voting_rf_gb.predict_proba(X_imp_test)[:, 1]
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pv3)),
            ("PR_AUC", average_precision_score(y_test, pv3)),
            ("Brier", brier_score_loss(y_test, pv3)),
            ("KS", ks_statistic(y_test, pv3)),
            ("LogLoss", log_loss(y_test, pv3)),
        ]:
            ensemble_metrics["voting"]["rf_gb"][mname].append(val)

        # Bagging helper
        def bag_probs(base_list):
            return np.mean(
                [m.predict_proba(Xte)[:, 1] for (_, m, _, Xte) in base_list],
                axis=0,
            )

        # Bagging full
        pb = bag_probs(base_full)
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pb)),
            ("PR_AUC", average_precision_score(y_test, pb)),
            ("Brier", brier_score_loss(y_test, pb)),
            ("KS", ks_statistic(y_test, pb)),
            ("LogLoss", log_loss(y_test, pb)),
        ]:
            ensemble_metrics["bagging"]["full"][mname].append(val)

        # Bagging no NN
        pb2 = bag_probs(base_no_nn)
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pb2)),
            ("PR_AUC", average_precision_score(y_test, pb2)),
            ("Brier", brier_score_loss(y_test, pb2)),
            ("KS", ks_statistic(y_test, pb2)),
            ("LogLoss", log_loss(y_test, pb2)),
        ]:
            ensemble_metrics["bagging"]["no_nn"][mname].append(val)

        # Bagging RF+GB
        pb3 = bag_probs(base_rf_gb)
        for mname, val in [
            ("AUC", roc_auc_score(y_test, pb3)),
            ("PR_AUC", average_precision_score(y_test, pb3)),
            ("Brier", brier_score_loss(y_test, pb3)),
            ("KS", ks_statistic(y_test, pb3)),
            ("LogLoss", log_loss(y_test, pb3)),
        ]:
            ensemble_metrics["bagging"]["rf_gb"][mname].append(val)

        # Stacking helper
        def stacking_from_base_list(base_list, key):
            inner_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            n_train_fold = X_imp_train.shape[0]
            meta_tr = np.zeros((n_train_fold, len(base_list)))
            meta_te = np.zeros((X_imp_test.shape[0], len(base_list)))

            for i, (name, model, Xtr_use, Xte_use) in enumerate(base_list):
                oof = np.zeros(n_train_fold)
                for inner_train_idx, inner_val_idx in inner_kf.split(Xtr_use, y_train):
                    m_clone = type(model)(**model.get_params())
                    m_clone.fit(Xtr_use.iloc[inner_train_idx], y_train[inner_train_idx])
                    oof[inner_val_idx] = m_clone.predict_proba(Xtr_use.iloc[inner_val_idx])[:, 1]
                meta_tr[:, i] = oof

                m_final = type(model)(**model.get_params())
                m_final.fit(Xtr_use, y_train)
                meta_te[:, i] = m_final.predict_proba(Xte_use)[:, 1]

            stack_scaler = StandardScaler()
            meta_tr_s = stack_scaler.fit_transform(meta_tr)
            meta_te_s = stack_scaler.transform(meta_te)

            stacker = LogisticRegression(max_iter=2500, random_state=random_state)
            stacker.fit(meta_tr_s, y_train)
            p = stacker.predict_proba(meta_te_s)[:, 1]

            for mname, val in [
                ("AUC", roc_auc_score(y_test, p)),
                ("PR_AUC", average_precision_score(y_test, p)),
                ("Brier", brier_score_loss(y_test, p)),
                ("KS", ks_statistic(y_test, p)),
                ("LogLoss", log_loss(y_test, p)),
            ]:
                ensemble_metrics["stacking"][key][mname].append(val)

        stacking_from_base_list(base_full, "full")
        stacking_from_base_list(base_no_nn, "no_nn")
        stacking_from_base_list(base_rf_gb, "rf_gb")

        # Blending helper
        def blending_from_base_list(base_list, key):
            meta_tr_full = np.column_stack(
                [m.predict_proba(Xtr)[:, 1] for (_, m, Xtr, _) in base_list]
            )
            meta_te_full = np.column_stack(
                [m.predict_proba(Xte)[:, 1] for (_, m, _, Xte) in base_list]
            )

            X_meta_train, X_meta_blend, y_meta_train, y_meta_blend = train_test_split(
                meta_tr_full,
                y_train,
                test_size=0.2,
                random_state=random_state,
                stratify=y_train,
            )

            blend_scaler = StandardScaler()
            X_meta_blend_s = blend_scaler.fit_transform(X_meta_blend)
            X_meta_test_s = blend_scaler.transform(meta_te_full)

            meta_model = LogisticRegression(max_iter=2500, random_state=random_state)
            meta_model.fit(X_meta_blend_s, y_meta_blend)
            p = meta_model.predict_proba(X_meta_test_s)[:, 1]

            for mname, val in [
                ("AUC", roc_auc_score(y_test, p)),
                ("PR_AUC", average_precision_score(y_test, p)),
                ("Brier", brier_score_loss(y_test, p)),
                ("KS", ks_statistic(y_test, p)),
                ("LogLoss", log_loss(y_test, p)),
            ]:
                ensemble_metrics["blending"][key][mname].append(val)

        blending_from_base_list(base_full, "full")
        blending_from_base_list(base_no_nn, "no_nn")
        blending_from_base_list(base_rf_gb, "rf_gb")

    # Save base model CV metrics
    for name, m in base_model_metrics.items():
        df_metrics = pd.DataFrame(m)
        out_file = OUTPUT_DIR / f"{name}_cv_metrics.csv"
        df_metrics.to_csv(out_file, index=False)
        print(f"Saved CV metrics for {name} to {out_file}")

    # Save ensemble CV metrics (12 variants)
    for ens_name, variants in ensemble_metrics.items():
        for variant, m in variants.items():
            df_metrics = pd.DataFrame(m)
            out_file = OUTPUT_DIR / f"ensemble_{ens_name}_{variant}_cv_metrics.csv"
            df_metrics.to_csv(out_file, index=False)
            print(f"Saved CV metrics for ensemble {ens_name} ({variant}) to {out_file}")


if __name__ == "__main__":
    print("Running K-fold CV for base and ensemble models (with variants)...")
    cross_validate_all()
    print("Done.")