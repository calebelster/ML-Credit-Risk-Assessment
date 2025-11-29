import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import streamlit as st


class ModelVisualizations:
    """Generate interactive performance plots"""

    @staticmethod
    def plot_roc_curves(metrics_dict):
        """
        Plot ROC curves for multiple models.

        Parameters
        ----------
        metrics_dict : dict
            Keys: model names, Values: {'y_true', 'y_proba'}
        """
        fig = go.Figure()

        for model_name, data in metrics_dict.items():
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC={roc_auc:.3f})',
                hovertemplate='<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>'
            ))

        # Diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray'),
            showlegend=True
        ))

        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest',
            height=500
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name):
        """Confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Default', 'Predicted Default'],
            y=['Actual No Default', 'Actual Default'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))

        fig.update_layout(
            title=f'Confusion Matrix: {model_name}',
            height=400
        )
        return fig

    @staticmethod
    def plot_metrics_comparison(metrics_df):
        """Bar chart comparing metrics across models"""
        metrics_to_plot = ['AUC', 'PR-AUC', 'F1-Score']

        fig = px.bar(
            metrics_df,
            x='Model',
            y=metrics_to_plot,
            barmode='group',
            title='Model Metrics Comparison',
            height=500
        )

        fig.update_layout(yaxis_title='Score', xaxis_title='Model')
        return fig

    @staticmethod
    def plot_feature_importance(importances_df, top_n=15):
        """Feature importance bar chart"""
        top_features = importances_df.nlargest(top_n, 'Importance')

        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            height=500
        )

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig

    @staticmethod
    def plot_cv_stability(cv_metrics_df):
        """Plot CV metrics across folds to show stability"""
        metrics = ["AUC", "PR-AUC", "Brier", "KS"]

        fig = go.Figure()

        # x values are fold numbers starting at 1
        x_vals = np.arange(1, len(cv_metrics_df) + 1)

        for metric in metrics:
            if metric in cv_metrics_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=cv_metrics_df[metric],
                        mode="lines+markers",
                        name=metric,
                    )
                )

        fig.update_layout(
            title="CV Metrics Across Folds",
            xaxis_title="Fold",
            yaxis_title="Metric Value",
            height=500,
            hovermode="x unified",
        )

        # Ensure x-axis shows only integer fold numbers
        fig.update_xaxes(
            tickmode="array",
            tickvals=x_vals,
            ticktext=[str(i) for i in x_vals],
        )

        return fig