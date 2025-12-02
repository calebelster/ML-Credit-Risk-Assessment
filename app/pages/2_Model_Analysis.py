import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# Add app directory to path (absolute import)
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from utils.visualizations import ModelVisualizations

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("📊 Model Analysis & Performance")
st.markdown("**Explore model metrics, cross-validation stability, and test performance for all models.**")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


# ------------------------------
# Data loading helpers
# ------------------------------
@st.cache_data
def load_cv_metrics():
    """
    Load CV metrics for base and ensemble models.
    """
    metrics_files = {
        'Random Forest': 'random_forest_cv_metrics.csv',
        'Gradient Boost': 'grad_boost_cv_metrics.csv',
        'Logistic Regression': 'log_reg_cv_metrics.csv',
        'Neural Net': 'neural_net_cv_metrics.csv',
        'Ensemble Bagging': 'ensemble_bagging_cv_metrics.csv',
        'Ensemble Voting': 'ensemble_voting_cv_metrics.csv',
        'Ensemble Stacking': 'ensemble_stacking_cv_metrics.csv',
        'Ensemble Blending': 'ensemble_blending_cv_metrics.csv',
    }

    metrics = {}
    for name, file in metrics_files.items():
        path = OUTPUT_DIR / file
        if path.exists():
            df = pd.read_csv(path)
            metrics[name] = df
    return metrics


@st.cache_data
def load_test_results():
    """
    Load test-set prediction CSVs.
    """
    results = {}
    stacking_path = OUTPUT_DIR / "ensemble_stacking_preds.csv"
    rf_path = OUTPUT_DIR / "random_forest_preds.csv"
    bag_path = OUTPUT_DIR / "ensemble_bagging_preds.csv"
    vote_path = OUTPUT_DIR / "ensemble_voting_preds.csv"
    blend_path = OUTPUT_DIR / "ensemble_blending_preds.csv"

    if stacking_path.exists():
        results["Ensemble Stacking"] = pd.read_csv(stacking_path)
    if rf_path.exists():
        results["Random Forest"] = pd.read_csv(rf_path)
    if bag_path.exists():
        results["Ensemble Bagging"] = pd.read_csv(bag_path)
    if vote_path.exists():
        results["Ensemble Voting"] = pd.read_csv(vote_path)
    if blend_path.exists():
        results["Ensemble Blending"] = pd.read_csv(blend_path)

    return results


cv_metrics = load_cv_metrics()
test_results = load_test_results()

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["Cross-Validation", "Test Performance", "Comparison"])


# ===================== TAB 1: CV STABILITY =====================
with tab1:
    st.subheader("K-Fold Cross-Validation Metrics")
    st.markdown("Select a model to inspect how its metrics vary across folds.")

    if not cv_metrics:
        st.warning("No CV metrics found. Run src/analysis/cross_validation.py first.")
    else:
        # Default to Ensemble Stacking if available
        model_names = list(cv_metrics.keys())
        default_index = model_names.index("Ensemble Stacking") if "Ensemble Stacking" in model_names else 0

        selected_model = st.selectbox("Select Model", model_names, index=default_index)
        cv_df = cv_metrics[selected_model]

        # Plot CV stability
        fig = ModelVisualizations.plot_cv_stability(cv_df)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean AUC", f"{cv_df['AUC'].mean():.3f}", f"±{cv_df['AUC'].std():.3f}")
        with col2:
            st.metric("Mean PR-AUC", f"{cv_df['PR_AUC'].mean():.3f}", f"±{cv_df['PR_AUC'].std():.3f}")
        with col3:
            st.metric("Mean Brier", f"{cv_df['Brier'].mean():.3f}", f"±{cv_df['Brier'].std():.3f}")

        st.markdown("#### Per-Fold Metrics")
        st.dataframe(cv_df, use_container_width=True)


# ===================== TAB 2: TEST PERFORMANCE =====================
with tab2:
    st.subheader("Test Set Performance")

    if not test_results:
        st.warning("No test prediction files found. Run src/analysis/ensemble_methods.py first.")
    else:
        # Default to Ensemble Stacking if available
        test_model_names = list(test_results.keys())
        default_index_test = (
            test_model_names.index("Ensemble Stacking")
            if "Ensemble Stacking" in test_model_names
            else 0
        )

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

        # Confusion matrix + plot
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = ModelVisualizations.plot_confusion_matrix(
            y_true, y_pred, selected_test_model
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Global metrics (same as before)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC-ROC", f"{roc_auc_score(y_true, y_proba):.3f}")
        with col2:
            st.metric("PR-AUC", f"{average_precision_score(y_true, y_proba):.3f}")
        with col3:
            st.metric("F1-Score", f"{f1_score(y_true, y_pred):.3f}")
        with col4:
            st.metric("Brier Score", f"{brier_score_loss(y_true, y_proba):.3f}")

        # NEW: Accuracy, Precision, Recall
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with col6:
            st.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
        with col7:
            st.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")


# ===================== TAB 3: COMPARISON =====================
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

        # First chart: reuse your existing helper
        fig_comp = ModelVisualizations.plot_metrics_comparison(
            comp_df[["Model", "AUC", "PR-AUC", "F1-Score", "LogLoss"]]
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # Second chart: build APR comparison directly with Plotly
        apr_df = comp_df[["Model", "Accuracy", "Precision", "Recall"]]

        # Melt to long form: columns -> Metric, value -> Score
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
        st.plotly_chart(fig_apr, use_container_width=True)


    st.markdown("---")
    st.markdown(
        """
        ### Model Selection Rationale

        **Ensemble Stacking** is the primary model (after removing `loan_grade`):
        - Tied for highest **AUC** (0.928) – excellent discrimination between defaults and non-defaults.
        - Best **calibration** – lowest LogLoss (0.231), meaning probabilities are well aligned with actual outcomes.
        - Highest **F1-Score** among robust models (≈0.794) – strong balance of precision and recall.
        - Consistent **CV performance** – stable metrics across folds, indicating robustness.
        - Combines strengths of Random Forest, Gradient Boost, and Logistic Regression.

        **Random Forest** remains an excellent benchmark / fallback:
        - Same AUC (0.928) but slightly worse calibration (LogLoss 0.252).
        - More interpretable and faster to score.
        - Useful for feature importance and business explanations.

        Removing `loan_grade`:
        - Users don’t know their loan grade ahead of time; it is an internal lender feature.
        - Removing it avoids potential leakage and makes the tool realistic for applicants.
        - Performance remained strong after removal (Stacking AUC only dropped slightly to 0.928).

        You can use the **Risk Calculator** page to adjust the decision threshold and see how approval risk changes in real time.
        """
    )