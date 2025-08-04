import numpy as np
from typing import Optional
from sklearn.metrics import hamming_loss
from src.ml_plugins.base_metric_plugin import MetricPlugin

class HammingLossMetric(MetricPlugin):
    """Hamming Loss metric for multi-label and multi-class classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Hamming Loss"
        self._description = "Fraction of labels that are incorrectly predicted. Useful for multi-label classification where partial correctness matters."
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower loss is better
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(hamming_loss(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Hamming Loss: {str(e)}")
    
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
    return HammingLossMetric()
