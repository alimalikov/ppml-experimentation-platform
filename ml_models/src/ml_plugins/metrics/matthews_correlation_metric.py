import numpy as np
from typing import Optional
from sklearn.metrics import matthews_corrcoef
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MatthewsCorrelationMetric(MetricPlugin):
    """Matthews Correlation Coefficient - balanced measure for imbalanced datasets."""
    
    def __init__(self):
        super().__init__()
        self._name = "Matthews Correlation Coefficient"
        self._description = "Balanced measure that works well with imbalanced datasets, ranges from -1 to +1"
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (-1, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        return matthews_corrcoef(y_true, y_pred)
    
    def get_interpretation(self, value: float) -> str:
        if value >= 0.8:
            return "Excellent (Very strong positive correlation)"
        elif value >= 0.6:
            return "Good (Strong positive correlation)"
        elif value >= 0.3:
            return "Fair (Moderate positive correlation)"
        elif value >= 0.1:
            return "Weak (Weak positive correlation)"
        elif value >= -0.1:
            return "No correlation"
        elif value >= -0.3:
            return "Weak negative correlation"
        else:
            return "Strong negative correlation"

def get_metric_plugin() -> MetricPlugin:
    return MatthewsCorrelationMetric()