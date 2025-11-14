from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

rf_file = OUTPUT_DIR / "random_forest_preds.csv"
df = pd.read_csv(rf_file)
y_true = df['y_true']
y_pred_prob = df['y_pred_prob']

skf = StratifiedKFold(n_splits=5)
aucs = []
for train, test in skf.split(y_true, y_true):
    auc = roc_auc_score(y_true.iloc[test], y_pred_prob.iloc[test])
    aucs.append(auc)
print("RF CV Mean AUC:", np.mean(aucs))
pd.Series(aucs, name='RandomForestCV_AUC').to_csv(OUTPUT_DIR / "random_forest_cv_auc.csv", index=False)