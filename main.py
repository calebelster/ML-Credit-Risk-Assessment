import subprocess
import os

# Paths to model and analysis scripts (from project root)
MODEL_SCRIPTS = {
    "grad_boost": "./src/models/grad_boost.py",
    "log_reg": "./src/models/log_reg.py",
    "random_forest": "./src/models/random_forest.py",
    "neural_net": "./src/models/neural_net.py"
}
MODEL_OUTPUTS = {
    "grad_boost": "./output/grad_boost_preds.csv",
    "log_reg": "./output/log_reg_preds.csv",
    "random_forest": "./output/random_forest_preds.csv",
    "neural_net": "./output/neural_net_preds.csv"
}
ANALYSIS_SCRIPTS = [
    "./src/analysis/compare_models.py",
    "./src/analysis/plot_calibration.py",
    "./src/analysis/feature_importance.py",
    "./src/analysis/cross_validation.py"
]

def run_script(filepath):
    print(f"\nRunning {filepath} ...")
    result = subprocess.run(['python', filepath], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Script error output:\n", result.stderr)

if __name__ == "__main__":
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    print("Checking model outputs and running models if needed:")
    for model_key, script_path in MODEL_SCRIPTS.items():
        output_file = MODEL_OUTPUTS[model_key]
        if not os.path.exists(output_file):
            print(f"{output_file} not found, running {script_path}.")
            run_script(script_path)
        else:
            print(f"{output_file} already exists. Skipping {script_path}.")

    print("\nRunning analysis scripts:")
    for script in ANALYSIS_SCRIPTS:
        run_script(script)

    print("\nAll tasks completed.")