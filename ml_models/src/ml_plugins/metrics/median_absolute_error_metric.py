import numpy as np
from typing import Optional
from sklearn.metrics import median_absolute_error
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MedianAbsoluteErrorMetric(MetricPlugin):
    """Median Absolute Error metric for regression tasks - robust to outliers."""
    
    def __init__(self):
        super().__init__()
        self._name = "Median Absolute Error"
        self._description = "Median of absolute differences between predicted and actual values - robust to outliers"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower MedAE is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(median_absolute_error(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Median Absolute Error: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of Median Absolute Error."""
        # MedAE interpretation depends on data scale, but generally:
        if value < 0.1:
            return "Excellent (Very Low Median Error)"
        elif value < 1.0:
            return "Good (Low Median Error)"
        elif value < 5.0:
            return "Fair (Moderate Median Error)"
        elif value < 20.0:
            return "Poor (High Median Error)"
        else:
            return "Very Poor (Very High Median Error)"

def get_metric_plugin() -> MetricPlugin:
    return MedianAbsoluteErrorMetric()
