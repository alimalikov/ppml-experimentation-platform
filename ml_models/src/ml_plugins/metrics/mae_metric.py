import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MAEMetric(MetricPlugin):
    """Mean Absolute Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Mean Absolute Error"
        self._description = "Average of absolute differences between predicted and actual values"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower MAE is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(mean_absolute_error(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating MAE: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of MAE value."""
        # MAE interpretation depends on data scale, so provide general guidance
        if value < 0.1:
            return "Excellent (Very Low Error)"
        elif value < 1.0:
            return "Good (Low Error)"
        elif value < 5.0:
            return "Fair (Moderate Error)"
        elif value < 20.0:
            return "Poor (High Error)"
        else:
            return "Very Poor (Very High Error)"

def get_metric_plugin() -> MetricPlugin:
    return MAEMetric()
