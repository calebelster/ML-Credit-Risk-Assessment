from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

fi_files = {
    'Random Forest': OUTPUT_DIR / "random_forest_importances.csv",
    'Gradient Boost': OUTPUT_DIR / "grad_boost_importances.csv",
    'Logistic Regression': OUTPUT_DIR / "log_reg_coefs.csv"
}

for model, fi_file in fi_files.items():
    fi = pd.read_csv(fi_file)
    plt.figure(figsize=(8, 5))
    if 'Coefficient' in fi.columns:
        fi['abs_coef'] = fi['Coefficient'].abs()
        fi = fi.sort_values('abs_coef', ascending=False).head(10)
        plt.barh(fi['Feature'], fi['Coefficient'])
        plt.xlabel('Coefficient')
    else:
        fi = fi.sort_values('Importance', ascending=False).head(10)
        plt.barh(fi['Feature'], fi['Importance'])
        plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'{model} - Top 10 Features')
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Most important at top
    plt.savefig(OUTPUT_DIR / f"{model.lower().replace(' ', '_')}_feature_importance.png")
    plt.close()
