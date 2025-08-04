import numpy as np
from typing import Optional
from sklearn.metrics import mean_squared_error
from src.ml_plugins.base_metric_plugin import MetricPlugin

class RMSEMetric(MetricPlugin):
    """Root Mean Squared Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Root Mean Squared Error"
        self._description = "Square root of MSE. More interpretable as it's in the same units as the target variable."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower RMSE is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        except Exception as e:
            raise ValueError(f"Error calculating RMSE: {str(e)}")
    
    def get_category(self) -> str:
        return self._category
    
    def supports_classification(self) -> bool:
        return self._supports_classification
    
    def supports_regression(self) -> bool:
        return self._supports_regression
    
    def requires_probabilities(self) -> bool:
        return self._requires_probabilities
    
    def higher_is_better(self) -> bool:
        return self._higher_is_better
    
    def get_range(self) -> tuple:
        return self._range


def get_metric_plugin() -> MetricPlugin:
    return RMSEMetric()
