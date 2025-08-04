import numpy as np
from typing import Optional
from sklearn.metrics import max_error
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MaxErrorMetric(MetricPlugin):
    """Maximum Residual Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Maximum Error"
        self._description = "Maximum absolute difference between predicted and actual values - worst-case error"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower max error is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(max_error(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Maximum Error: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of Maximum Error."""
        # Max error interpretation depends on data scale and tolerance:
        if value < 0.5:
            return "Excellent (Very Low Worst-Case Error)"
        elif value < 2.0:
            return "Good (Low Worst-Case Error)"
        elif value < 10.0:
            return "Fair (Moderate Worst-Case Error)"
        elif value < 50.0:
            return "Poor (High Worst-Case Error)"
        else:
            return "Very Poor (Extremely High Worst-Case Error)"

def get_metric_plugin() -> MetricPlugin:
    return MaxErrorMetric()
