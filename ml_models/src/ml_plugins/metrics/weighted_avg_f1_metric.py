import numpy as np
from typing import Optional
from sklearn.metrics import f1_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class WeightedAvgF1Metric(MetricPlugin):
    """Weighted-Averaged F1 Score metric for multi-class classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Weighted-Averaged F1 Score"
        self._description = "F1 scores for each class weighted by the number of true instances for each class. Better for imbalanced datasets."
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        except Exception as e:
            raise ValueError(f"Error calculating Weighted-Averaged F1 Score: {str(e)}")
    
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
    return WeightedAvgF1Metric()
