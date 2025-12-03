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
        try:
            artifacts = joblib.load(MODEL_PATH)
            self.base_models = artifacts["base_models"]  # expects 'rf' and 'gb'
            self.meta_learner = artifacts["meta_learner"]
            self.meta_scaler = artifacts["meta_scaler"]
            self.feature_imputer = artifacts["feature_imputer"]
            self.feature_scaler = artifacts["feature_scaler"]
            self.feature_names = artifacts["feature_names"]
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default using RF+GB stacking ensemble.
        """
        try:
            # One-hot + align
            X_enc = self.data_processor.preprocess_for_prediction(X, self.feature_names)

            # Impute
            X_imp = pd.DataFrame(
                self.feature_imputer.transform(X_enc),
                columns=X_enc.columns,
                index=X_enc.index,
            )

            # Scale base features (for consistency, even though RF/GB use X_imp)
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(X_imp),
                columns=X_enc.columns,
                index=X_enc.index,
            )

            # Meta-features: RF and GB probabilities on imputed features
            meta_features = np.zeros((X_imp.shape[0], 2))
            meta_features[:, 0] = self.base_models["rf"].predict_proba(X_imp)[:, 1]
            meta_features[:, 1] = self.base_models["gb"].predict_proba(X_imp)[:, 1]

            meta_features_scaled = self.meta_scaler.transform(meta_features)
            probs = self.meta_learner.predict_proba(meta_features_scaled)[:, 1]
            return probs
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        probs = self.predict_probability(X)
        risk_scores = probs * 100
        risk_preds = (probs >= threshold).astype(int)

        # Pure bin-based categories
        risk_category = pd.cut(
            risk_scores,
            bins=[0, 10, 25, 100],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

        result = X.copy()
        result["default_probability"] = probs
        result["risk_score_percent"] = risk_scores
        result["predicted_default"] = risk_preds
        result["risk_category"] = risk_category

        return result