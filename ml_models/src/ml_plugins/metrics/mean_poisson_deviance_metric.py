import numpy as np
from typing import Optional
from sklearn.metrics import mean_poisson_deviance
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MeanPoissonDevianceMetric(MetricPlugin):
    """Mean Poisson Deviance metric for count data regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Mean Poisson Deviance"
        self._description = "Ideal for count data regression. Measures deviance assuming Poisson distribution."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower deviance is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Ensure non-negative values for Poisson deviance
            if np.any(y_true < 0) or np.any(y_pred < 0):
                raise ValueError("Poisson deviance requires non-negative values")
            
            return float(mean_poisson_deviance(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Mean Poisson Deviance: {str(e)}")
    
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
    return MeanPoissonDevianceMetric()
