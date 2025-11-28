import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .data_processor import DataProcessor

MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "stacking_model.pkl"


class CreditRiskPredictor:
    def __init__(self):
        self.base_models = {}
        self.meta_learner = None
        self.meta_scaler = None
        self.feature_imputer = None
        self.feature_scaler = None
        self.feature_names = None
        self.data_processor = DataProcessor()
        self.load_model()

    def load_model(self):
        """Load pre-trained stacking model and preprocessors"""
        try:
            artifacts = joblib.load(MODEL_PATH)
            self.base_models = artifacts['base_models']
            self.meta_learner = artifacts['meta_learner']
            self.meta_scaler = artifacts['meta_scaler']
            self.feature_imputer = artifacts['feature_imputer']
            self.feature_scaler = artifacts['feature_scaler']
            self.feature_names = artifacts['feature_names']
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict_probability(self, X):
        """
        Predict probability of default using stacking ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features with categorical and numeric columns

        Returns
        -------
        np.ndarray
            Probability of default [0, 1]
        """
        try:
            # One-hot encode and align to training features
            X_enc = self.data_processor.preprocess_for_prediction(X, self.feature_names)

            # Impute missing values
            X_imp = pd.DataFrame(
                self.feature_imputer.transform(X_enc),
                columns=X_enc.columns,
                index=X_enc.index
            )

            # Scale features
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(X_imp),
                columns=X_enc.columns,
                index=X_enc.index
            )

            # Generate meta-features from base models
            meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))

            # LR and NN use scaled features
            meta_features[:, 0] = self.base_models['lr'].predict_proba(X_scaled)[:, 1]
            meta_features[:, 3] = self.base_models['nn'].predict_proba(X_scaled)[:, 1]

            # RF and GB use imputed (non-scaled) features
            meta_features[:, 1] = self.base_models['rf'].predict_proba(X_imp)[:, 1]
            meta_features[:, 2] = self.base_models['gb'].predict_proba(X_imp)[:, 1]

            # Scale meta-features
            meta_features_scaled = self.meta_scaler.transform(meta_features)

            # Predict with meta-learner
            probs = self.meta_learner.predict_proba(meta_features_scaled)[:, 1]
            return probs
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, X, threshold=0.5):
        """
        Predict on batch of applications.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features
        threshold : float, default 0.5
            Decision threshold for binary classification

        Returns
        -------
        pd.DataFrame
            Original X with added risk columns
        """
        probs = self.predict_probability(X)
        risk_scores = probs * 100  # Convert to percentage

        # Apply threshold
        risk_preds = (probs >= threshold).astype(int)

        # Risk categories based on probability
        risk_category = pd.cut(
            risk_scores,
            bins=[0, 30, 60, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )

        # Override category if below threshold (approval)
        risk_category = pd.Series(
            ['Approved ✅' if pred == 0 else risk_cat
             for pred, risk_cat in zip(risk_preds, risk_category)],
            index=X.index
        )

        result = X.copy()
        result['default_probability'] = probs
        result['risk_score_percent'] = risk_scores
        result['predicted_default'] = risk_preds
        result['risk_category'] = risk_category

        return result