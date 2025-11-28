import subprocess
import os
from pathlib import Path

# Paths for scripts (adapt as necessary for your project structure)
MODELSCRIPTS = {
    "gradboost": "src/models/grad_boost.py",
    "logreg": "src/models/log_reg.py",
    "randomforest": "src/models/random_forest.py",
    "neuralnet": "src/models/neural_net.py"
}

ANALYSISSCRIPTS = [
    "src/analysis/feature_importance.py",
    "src/analysis/plot_calibration.py",
    "src/analysis/cross_validation.py",
]

ENSEMBLESCRIPT = "src/analysis/ensemble_methods.py"
COMPAREMODELS = "src/analysis/compare_models.py"

MODELOUTPUTS = {
    "gradboost": "output/grad_boost_preds.csv",
    "logreg": "output/log_reg_preds.csv",
    "randomforest": "output/random_forest_preds.csv",
    "neuralnet": "output/neural_net_preds.csv"
}

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Run all base model scripts
    for key, script in MODELSCRIPTS.items():
        output_file = MODELOUTPUTS[key]
        if not os.path.exists(output_file):
            print(f"{output_file} not found, running {script}.")
            subprocess.run(["python", script], check=True)
        else:
            print(f"{output_file} exists, skipping {script}.")

    # 2. Run analysis scripts for feature importance, calibration, and cross-validation
    for script in ANALYSISSCRIPTS:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)

    # 3. Run ensemble method script to produce ensemble CSVs
    print(f"Running ensemble methods: {ENSEMBLESCRIPT}...")
    subprocess.run(["python", ENSEMBLESCRIPT], check=True)

    # 4. Run compare_models to analyze all outputs (including ensembles)
    print(f"Running model comparisons: {COMPAREMODELS}...")
    subprocess.run(["python", COMPAREMODELS], check=True)