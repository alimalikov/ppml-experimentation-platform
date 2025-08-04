import numpy as np
from typing import Optional
from sklearn.metrics import log_loss
from src.ml_plugins.base_metric_plugin import MetricPlugin

class LogLossMetric(MetricPlugin):
    """Logarithmic Loss - measures prediction uncertainty."""
    
    def __init__(self):
        super().__init__()
        self._name = "Log Loss"
        self._description = "Logarithmic loss measures prediction uncertainty - lower values indicate better calibrated probabilities"
        self._category = "Probabilistic"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = True
        self._higher_is_better = False  # Lower is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        if y_proba is None:
            raise ValueError("Log Loss requires prediction probabilities")
        
        return log_loss(y_true, y_proba)
    
    def get_interpretation(self, value: float) -> str:
        if value <= 0.1:
            return "Excellent (Very confident and accurate)"
        elif value <= 0.3:
            return "Good (Confident predictions)"
        elif value <= 0.5:
            return "Fair (Moderately confident)"
        elif value <= 1.0:
            return "Poor (Low confidence)"
        else:
            return "Very Poor (Very uncertain predictions)"

def get_metric_plugin() -> MetricPlugin:
    return LogLossMetric()