
"""
ML Models Definition

Defines the architecture for:
1. Binary Classifier (Detection Probability)
2. Regressor (Apparent Magnitude Prediction)
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator

class NeoClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            random_state=42, 
            class_weight="balanced",
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class NeoRegressor(BaseEstimator):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=None, 
            random_state=42,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
