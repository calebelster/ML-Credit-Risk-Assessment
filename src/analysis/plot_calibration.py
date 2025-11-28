from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = {
    'Random Forest': OUTPUT_DIR / "random_forest_preds.csv",
    'Gradient Boost': OUTPUT_DIR / "grad_boost_preds.csv",
    'Logistic Regression': OUTPUT_DIR / "log_reg_preds.csv",
    'Neural Net': OUTPUT_DIR / "neural_net_preds.csv"
}

plt.figure(figsize=(8,6))
brier_scores = {}

for model, file in files.items():
    df = pd.read_csv(file)
    y_true = df['y_true']
    y_pred_prob = df['y_pred_prob']
    brier = brier_score_loss(y_true, y_pred_prob)
    brier_scores[model] = brier
    disp = CalibrationDisplay.from_predictions(y_true, y_pred_prob, n_bins=10, name=model)
    disp.plot(ax=plt.gca())

plt.title('Calibration Curves')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_curves.png")
pd.Series(brier_scores, name='Brier Score').to_csv(OUTPUT_DIR / "model_brier_scores.csv")
print(pd.Series(brier_scores, name='Brier Score'))