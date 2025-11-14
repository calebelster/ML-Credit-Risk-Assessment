from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = {
    'Random Forest': OUTPUT_DIR / "random_forest_preds.csv",
    'Gradient Boost': OUTPUT_DIR / "grad_boost_preds.csv",
    'Logistic Regression': OUTPUT_DIR / "log_reg_preds.csv",
    'Neural Net': OUTPUT_DIR / "neural_net_preds.csv"
}

metrics = {}
plt.figure(figsize=(8, 6))

for model, file in files.items():
    df = pd.read_csv(file)
    y_true = df['y_true']
    y_pred_prob = df['y_pred_prob']
    y_pred = df['y_pred'] if 'y_pred' in df.columns else (y_pred_prob > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model} (AUC={auc:.2f})')
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics[model] = {
        'AUC': auc,
        'Precision': report['1']['precision'] if '1' in report else None,
        'Recall': report['1']['recall'] if '1' in report else None,
        'F1-Score': report['1']['f1-score'] if '1' in report else None,
        'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist()
    }

plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png")
pd.DataFrame(metrics).T.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv")
print(pd.DataFrame(metrics).T)
