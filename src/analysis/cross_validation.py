from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def ks_statistic(y_true, y_scores):
    return ks_2samp(y_scores[y_true == 0], y_scores[y_true == 1]).statistic

# Load raw data
df = pd.read_csv(DATA_PATH)
cats = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=cats)

X = df.drop("loan_status", axis=1).values
y = df["loan_status"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, pr_aucs, briers, ks_stats, loglosses = [], [], [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    aucs.append(roc_auc_score(y_test, y_prob))
    pr_aucs.append(average_precision_score(y_test, y_prob))
    briers.append(brier_score_loss(y_test, y_prob))
    ks_stats.append(ks_statistic(y_test, y_prob))
    loglosses.append(log_loss(y_test, y_prob))

summary = pd.DataFrame({
    'AUC': aucs,
    'PR-AUC': pr_aucs,
    'Brier': briers,
    'KS': ks_stats,
    'LogLoss': loglosses,
})
print("RF CV Metrics (mean over folds):\n", summary.mean())
summary.to_csv(OUTPUT_DIR / "random_forest_cv_metrics.csv", index=False)