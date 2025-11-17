from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    brier_score_loss, average_precision_score, log_loss
)
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold

def gini_coefficient(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    return 2 * auc - 1

def ks_statistic(y_true, y_scores):
    return ks_2samp(y_scores[y_true == 0], y_scores[y_true == 1]).statistic

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = {
    'Random Forest': OUTPUT_DIR / "random_forest_preds.csv",
    'Gradient Boost': OUTPUT_DIR / "grad_boost_preds.csv",
    'Logistic Regression': OUTPUT_DIR / "log_reg_preds.csv",
    'Neural Net': OUTPUT_DIR / "neural_net_preds.csv",
    'Ensemble Bagging': OUTPUT_DIR / "ensemble_bagging_preds.csv",
    'Ensemble Voting': OUTPUT_DIR / "ensemble_voting_preds.csv",
    'Ensemble Stacking': OUTPUT_DIR / "ensemble_stacking_preds.csv",
    'Ensemble Blending': OUTPUT_DIR / "ensemble_blending_preds.csv",
}

metrics = {}
plt.figure(figsize=(8, 6))

for model, file in files.items():
    if not file.exists():
        print(f"Skipping {file} (not found)")
        continue

    df = pd.read_csv(file)
    y_true = df['y_true']
    y_pred_prob = df['y_pred_prob']
    y_pred = df['y_pred'] if 'y_pred' in df.columns else (y_pred_prob > 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    gini = gini_coefficient(y_true, y_pred_prob)
    ks = ks_statistic(y_true, y_pred_prob)
    logloss = log_loss(y_true, y_pred_prob)
    conf = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model} (AUC={auc:.2f})')

    metrics[model] = {
        'AUC': auc,
        'GINI': gini,
        'PR-AUC': pr_auc,
        'KS': ks,
        'Brier': brier,
        'LogLoss': logloss,
        'Precision': report['1']['precision'] if '1' in report else None,
        'Recall': report['1']['recall'] if '1' in report else None,
        'F1-Score': report['1']['f1-score'] if '1' in report else None,
        'Confusion Matrix': conf.tolist()
    }

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png")

df_metrics = pd.DataFrame(metrics).T
df_metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv")
print(df_metrics)

md_file = OUTPUT_DIR / "model_comparison_metrics.md"
header = (
    "| Model | AUC | GINI | PR-AUC | KS | Brier | LogLoss | Precision | Recall | F1-Score |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)
lines = ["# Model and Ensemble Metrics Comparison\n", header]
for idx, row in df_metrics.iterrows():
    lines.append(
        f"| {idx} | "
        f"{row['AUC']:.3f} | {row['GINI']:.3f} | {row['PR-AUC']:.3f} | {row['KS']:.3f} | "
        f"{row['Brier']:.3f} | {row['LogLoss']:.3f} | "
        f"{row['Precision'] if pd.notna(row['Precision']) else '-'} | "
        f"{row['Recall'] if pd.notna(row['Recall']) else '-'} | {row['F1-Score'] if pd.notna(row['F1-Score']) else '-'} |\n"
    )

lines.append("\n# Confusion Matrices\n")
for model, row in df_metrics.iterrows():
    conf = np.array(row['Confusion Matrix'])
    lines.append(f"## {model}\n")
    lines.append("|       | Pred 0 | Pred 1 |")
    lines.append("|-------|--------|--------|")
    lines.append(f"| Actual 0 | {conf[0][0]} | {conf[0][1]} |")
    lines.append(f"| Actual 1 | {conf[1][0]} | {conf[1][1]} |\n")

lines.append("\n# K-Fold CV Metrics\n")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for model, file in files.items():
    if not file.exists():
        continue
    df = pd.read_csv(file)
    aucs, pr_aucs, briers, ks_stats, loglosses = [], [], [], [], []
    y_true = df['y_true'].values
    y_pred_prob = df['y_pred_prob'].values
    for train, test in kf.split(y_true):
        yt, yp = y_true[test], y_pred_prob[test]
        aucs.append(roc_auc_score(yt, yp))
        pr_aucs.append(average_precision_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))
        ks_stats.append(ks_statistic(yt, yp))
        loglosses.append(log_loss(yt, yp))
    lines.append(f"## {model} KFold CV\n")
    lines.append("| Fold | AUC | PR_AUC | Brier | KS | LogLoss |")
    lines.append("|------|-----|--------|--------|-----|---------|")
    for i in range(5):
        lines.append(f"| {i+1} | {aucs[i]:.3f} | {pr_aucs[i]:.3f} | {briers[i]:.3f} | {ks_stats[i]:.3f} | {loglosses[i]:.3f} |")
    lines.append("")

md_file.write_text("".join(lines))
print(f"\nMetrics markdown table, confusion matrices (in tables), and kfold results written to {md_file}")