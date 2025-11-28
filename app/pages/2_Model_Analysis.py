import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from utils.visualizations import ModelVisualizations

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("📊 Model Analysis & Performance")
st.markdown("**Explore model metrics, cross-validation stability, and feature importance**")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


# Load data
@st.cache_data
def load_metrics():
    metrics_files = {
        'Ensemble Stacking': 'ensemble_stacking_cv_metrics.csv',
    }

    metrics = {}
    for name, file in metrics_files.items():
        path = OUTPUT_DIR / file
        if path.exists():
            metrics[name] = pd.read_csv(path)

    return metrics


@st.cache_data
def load_main_results():
    stacking_preds = OUTPUT_DIR / "ensemble_stacking_preds.csv"

    results = {}
    if stacking_preds.exists():
        results['Ensemble Stacking'] = pd.read_csv(stacking_preds)

    return results


try:
    cv_metrics = load_metrics()
    test_results = load_main_results()

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Cross-Validation", "Test Performance", "Comparison"])

    # ===================== TAB 1: CV STABILITY =====================
    with tab1:
        st.subheader("K-Fold Cross-Validation Metrics (Ensemble Stacking)")
        st.markdown("Shows model stability across 5 folds. Consistent performance = robust model.")

        if cv_metrics:
            cv_df = cv_metrics['Ensemble Stacking']

            # Plot CV stability
            fig = ModelVisualizations.plot_cv_stability(cv_df)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean AUC", f"{cv_df['AUC'].mean():.3f}", f"±{cv_df['AUC'].std():.3f}")
            with col2:
                st.metric("Mean PR-AUC", f"{cv_df['PR_AUC'].mean():.3f}", f"±{cv_df['PR_AUC'].std():.3f}")
            with col3:
                st.metric("Mean Brier", f"{cv_df['Brier'].mean():.3f}", f"±{cv_df['Brier'].std():.3f}")

            st.dataframe(cv_df, use_container_width=True)
        else:
            st.warning("No CV metrics found. Run ensemble_methods.py first.")

    # ===================== TAB 2: TEST PERFORMANCE =====================
    with tab2:
        st.subheader("Test Set Performance (Ensemble Stacking)")

        if test_results:
            test_df = test_results['Ensemble Stacking']
            y_true = test_df['y_true'].values
            y_proba = test_df['y_pred_prob'].values
            y_pred = test_df['y_pred'].values

            from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss, \
                f1_score

            cm = confusion_matrix(y_true, y_pred)
            fig_cm = ModelVisualizations.plot_confusion_matrix(y_true, y_pred, "Ensemble Stacking")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AUC-ROC", f"{roc_auc_score(y_true, y_proba):.3f}")
            with col2:
                st.metric("PR-AUC", f"{average_precision_score(y_true, y_proba):.3f}")
            with col3:
                st.metric("F1-Score", f"{f1_score(y_true, y_pred):.3f}")
            with col4:
                st.metric("Brier Score", f"{brier_score_loss(y_true, y_proba):.3f}")
        else:
            st.warning("No test results found.")

    # ===================== TAB 3: COMPARISON =====================
    with tab3:
        st.subheader("Model Comparison")

        comparison_data = {
            'Model': ['Random Forest', 'Ensemble Stacking'],
            'AUC': [0.928, 0.928],
            'PR-AUC': [0.868, 0.871],
            'F1-Score': [0.788, 0.794],
            'LogLoss': [0.252, 0.231],
        }

        comp_df = pd.DataFrame(comparison_data)

        fig_comp = ModelVisualizations.plot_metrics_comparison(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown("""
        ### Model Selection Rationale

        **Ensemble Stacking** is the primary model (after removing loan_grade):
        - ✅ **Tied for highest AUC** (0.928) - best discrimination
        - ✅ **Best calibration** - lowest LogLoss (0.231)
        - ✅ **Consistent CV performance** - stable across all 5 folds
        - ✅ **Best F1-Score** (0.794) - balanced precision/recall
        - ✅ **Combines strengths** of multiple base learners (RF, GB, LR, NN)

        ### Key Performance Metrics
        - **Precision:** 93.4% - Only 6.6% of approved loans predicted to default actually don't
        - **Recall:** 68.9% - Catches ~69% of actual defaults
        - **Decision Threshold:** Adjustable in Risk Calculator (0.30-0.70)
          - Lower threshold (0.30-0.40): More conservative, catches more defaults
          - Higher threshold (0.60-0.70): More lenient, approves more loans

        ### Why We Removed loan_grade
        - Users don't know their loan grade before applying
        - It's a derived feature from internal scoring, not user-provided input
        - Removing it made predictions more practical and reduced potential leakage
        - Model performance remained strong (AUC 0.928 vs previous 0.931)
        """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
