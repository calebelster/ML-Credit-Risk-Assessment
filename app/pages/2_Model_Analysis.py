import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from utils.visualizations import ModelVisualizations

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("📊 Model Analysis & Performance")
st.markdown("**Explore model metrics, cross-validation stability, and test performance for all models.**")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


@st.cache_data
def load_cv_metrics():
    metrics_files = {
        "Random Forest": "random_forest_cv_metrics.csv",
        "Gradient Boost": "grad_boost_cv_metrics.csv",
        "Logistic Regression": "log_reg_cv_metrics.csv",
        "Neural Net": "neural_net_cv_metrics.csv",
        "Ensemble Bagging (full)": "ensemble_bagging_full_cv_metrics.csv",
        "Ensemble Bagging (no NN)": "ensemble_bagging_no_nn_cv_metrics.csv",
        "Ensemble Bagging (RF+GB)": "ensemble_bagging_rf_gb_cv_metrics.csv",
        "Ensemble Voting (full)": "ensemble_voting_full_cv_metrics.csv",
        "Ensemble Voting (no NN)": "ensemble_voting_no_nn_cv_metrics.csv",
        "Ensemble Voting (RF+GB)": "ensemble_voting_rf_gb_cv_metrics.csv",
        "Ensemble Stacking (full)": "ensemble_stacking_full_cv_metrics.csv",
        "Ensemble Stacking (no NN)": "ensemble_stacking_no_nn_cv_metrics.csv",
        "Ensemble Stacking (RF+GB)": "ensemble_stacking_rf_gb_cv_metrics.csv",
        "Ensemble Blending (full)": "ensemble_blending_full_cv_metrics.csv",
        "Ensemble Blending (no NN)": "ensemble_blending_no_nn_cv_metrics.csv",
        "Ensemble Blending (RF+GB)": "ensemble_blending_rf_gb_cv_metrics.csv",
    }

    metrics = {}
    for name, file in metrics_files.items():
        path = OUTPUT_DIR / file
        if path.exists():
            metrics[name] = pd.read_csv(path)
    return metrics


@st.cache_data
def load_test_results():
    results = {}
    out = OUTPUT_DIR

    paths = {
        "Ensemble Voting (full)": out / "ensemble_voting_full_preds.csv",
        "Ensemble Voting (no NN)": out / "ensemble_voting_no_nn_preds.csv",
        "Ensemble Voting (RF+GB)": out / "ensemble_voting_rf_gb_preds.csv",
        "Ensemble Bagging (full)": out / "ensemble_bagging_full_preds.csv",
        "Ensemble Bagging (no NN)": out / "ensemble_bagging_no_nn_preds.csv",
        "Ensemble Bagging (RF+GB)": out / "ensemble_bagging_rf_gb_preds.csv",
        "Ensemble Stacking (full)": out / "ensemble_stacking_full_preds.csv",
        "Ensemble Stacking (no NN)": out / "ensemble_stacking_no_nn_preds.csv",
        "Ensemble Stacking (RF+GB)": out / "ensemble_stacking_rf_gb_preds.csv",
        "Ensemble Blending (full)": out / "ensemble_blending_full_preds.csv",
        "Ensemble Blending (no NN)": out / "ensemble_blending_no_nn_preds.csv",
        "Ensemble Blending (RF+GB)": out / "ensemble_blending_rf_gb_preds.csv",
        "Random Forest": out / "random_forest_preds.csv",
        "Ensemble Stacking (legacy)": out / "ensemble_stacking_preds.csv",
    }

    for name, path in paths.items():
        if path.exists():
            results[name] = pd.read_csv(path)

    return results


cv_metrics = load_cv_metrics()
test_results = load_test_results()

tab1, tab2, tab3 = st.tabs(["Cross-Validation", "Test Performance", "Comparison"])

# TAB 1
with tab1:
    st.subheader("K-Fold Cross-Validation Metrics")
    if not cv_metrics:
        st.warning("No CV metrics found. Run src/analysis/cross_validation.py first.")
    else:
        model_names = list(cv_metrics.keys())
        default_label_cv = "Ensemble Stacking (RF+GB)"
        default_index = model_names.index(default_label_cv) if default_label_cv in model_names else 0

        selected_model = st.selectbox("Select Model", model_names, index=default_index)
        cv_df = cv_metrics[selected_model]

        fig = ModelVisualizations.plot_cv_stability(cv_df)
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean AUC", f"{cv_df['AUC'].mean():.3f}", f"±{cv_df['AUC'].std():.3f}")
        with col2:
            st.metric("Mean PR-AUC", f"{cv_df['PR_AUC'].mean():.3f}", f"±{cv_df['PR_AUC'].std():.3f}")
        with col3:
            st.metric("Mean Brier", f"{cv_df['Brier'].mean():.3f}", f"±{cv_df['Brier'].std():.3f}")

        st.markdown("#### Per-Fold Metrics")
        st.dataframe(cv_df, width="stretch")

# TAB 2
with tab2:
    st.subheader("Test Set Performance")

    if not test_results:
        st.warning("No test prediction files found. Run src/analysis/ensemble_methods.py first.")
    else:
        test_model_names = list(test_results.keys())
        default_label = "Ensemble Stacking (RF+GB)"
        default_index_test = test_model_names.index(default_label) if default_label in test_model_names else 0

        selected_test_model = st.selectbox(
            "Select Model",
            test_model_names,
            index=default_index_test,
            key="model_select_test",
        )
        test_df = test_results[selected_test_model]

        y_true = test_df["y_true"].values
        y_proba = test_df["y_pred_prob"].values
        y_pred = test_df["y_pred"].values

        fig_cm = ModelVisualizations.plot_confusion_matrix(y_true, y_pred, selected_test_model)
        st.plotly_chart(fig_cm, width="stretch")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC-ROC", f"{roc_auc_score(y_true, y_proba):.3f}")
        with col2:
            st.metric("PR-AUC", f"{average_precision_score(y_true, y_proba):.3f}")
        with col3:
            st.metric("F1-Score", f"{f1_score(y_true, y_pred):.3f}")
        with col4:
            st.metric("Brier Score", f"{brier_score_loss(y_true, y_proba):.3f}")

        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with col6:
            st.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        with col7:
            st.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")

# TAB 3
with tab3:
    st.subheader("Model Comparison (Test Set)")

    if not test_results:
        st.warning("No test prediction files found. Run src/analysis/ensemble_methods.py first.")
    else:
        rows = []
        for model_name, df in test_results.items():
            y_true = df["y_true"].values
            y_pred = df["y_pred"].values
            y_proba = df["y_pred_prob"].values

            rows.append(
                {
                    "Model": model_name,
                    "AUC": roc_auc_score(y_true, y_proba),
                    "PR-AUC": average_precision_score(y_true, y_proba),
                    "F1-Score": f1_score(y_true, y_pred),
                    "LogLoss": brier_score_loss(y_true, y_proba),
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                }
            )

        comp_df = pd.DataFrame(rows)

        fig_comp = ModelVisualizations.plot_metrics_comparison(
            comp_df[["Model", "AUC", "PR-AUC", "F1-Score", "LogLoss"]]
        )
        st.plotly_chart(fig_comp, width="stretch")

        st.markdown("---")

        apr_df = comp_df[["Model", "Accuracy", "Precision", "Recall"]]
        apr_long = apr_df.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score",
        )

        fig_apr = px.bar(
            apr_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Accuracy, Precision, Recall by Model",
        )
        st.plotly_chart(fig_apr, width="stretch")

        st.markdown("---")
        st.markdown(
            """
            ### Model Selection Rationale

            **Primary model: Ensemble Stacking (RF+GB)**
            - Top-tier discrimination: AUC ≈ 0.928 with one of the highest KS values, meaning strong separation between good and bad borrowers.
            - Best probability quality: lowest LogLoss and among the lowest Brier scores, so predicted default probabilities are well calibrated.
            - Strong balance of precision and recall: very high precision with solid recall, yielding a robust F1-score.
            - Stable across folds: cross-validation metrics are consistently strong on every fold, indicating robustness.
            - Simpler ensemble: uses only Random Forest and Gradient Boosting as base models, making it easier to maintain and interpret than ensembles that also include Logistic Regression and Neural Nets.

            Alternative ensembles (full/no-NN variants and blending/voting/bagging) are kept for comparison
            but show slightly worse calibration or KS and do not clearly outperform the RF+GB stacking model.
            """
        )