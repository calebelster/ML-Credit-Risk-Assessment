from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from scipy.stats import ks_2samp

def ks_statistic(y_true, y_scores):
    return ks_2samp(y_scores[y_true == 0], y_scores[y_true == 1]).statistic

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

rf_file = OUTPUT_DIR / "random_forest_preds.csv"
df = pd.read_csv(rf_file)
y_true = df['y_true'].values
y_pred_prob = df['y_pred_prob'].values

skf = StratifiedKFold(n_splits=5)
aucs, pr_aucs, briers, ks_stats, loglosses = [], [], [], [], []

for train, test in skf.split(y_true, y_true):
    yt = y_true[test]
    yp = y_pred_prob[test]
    aucs.append(roc_auc_score(yt, yp))
    pr_aucs.append(average_precision_score(yt, yp))
    briers.append(brier_score_loss(yt, yp))
    ks_stats.append(ks_statistic(yt, yp))
    loglosses.append(log_loss(yt, yp))

summary = pd.DataFrame({
    'AUC': aucs,
    'PR-AUC': pr_aucs,
    'Brier': briers,
    'KS': ks_stats,
    'LogLoss': loglosses,
})
print("RF CV Metrics (mean over folds):\n", summary.mean())
summary.to_csv(OUTPUT_DIR / "random_forest_cv_metrics.csv", index=False)