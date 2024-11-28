"""Anomaly detection models for credit card fraud detection."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_scores(self, X):
        pass


class IsolationForestDetector(BaseDetector):
    """Isolation Forest based anomaly detector."""

    def __init__(self, n_estimators=100, contamination=0.01, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "Isolation Forest"

    def fit(self, X):
        """Fit the Isolation Forest model."""
        self.model.fit(X)
        print(f"{self.name} fitted on {X.shape[0]} samples")
        return self

    def predict(self, X):
        """Predict anomalies. Returns 1 for fraud, 0 for normal."""
        predictions = self.model.predict(X)
        return np.where(predictions == -1, 1, 0)

    def get_scores(self, X):
        """Get anomaly scores (lower = more anomalous)."""
        return -self.model.score_samples(X)


class LOFDetector(BaseDetector):
    """Local Outlier Factor based anomaly detector."""

    def __init__(self, n_neighbors=20, contamination=0.01):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.name = "Local Outlier Factor"

    def fit(self, X):
        """Fit the LOF model (LOF is transductive, stores training data)."""
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )
        self.model.fit(X)
        print(f"{self.name} fitted on {X.shape[0]} samples")
        return self

    def predict(self, X):
        """Predict anomalies. Returns 1 for fraud, 0 for normal."""
        predictions = self.model.predict(X)
        return np.where(predictions == -1, 1, 0)

    def get_scores(self, X):
        """Get anomaly scores (lower = more anomalous)."""
        return -self.model.score_samples(X)


class ModelFactory:
    """Factory class to create anomaly detection models."""

    _models = {
        "isolation_forest": IsolationForestDetector,
        "lof": LOFDetector,
    }

    @classmethod
    def create(cls, model_name, **kwargs):
        """Create a model instance by name."""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        return cls._models[model_name](**kwargs)

    @classmethod
    def list_models(cls):
        """List all available model names."""
        return list(cls._models.keys())
