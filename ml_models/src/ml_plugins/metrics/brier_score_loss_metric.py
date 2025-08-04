import numpy as np
from typing import Optional
from sklearn.metrics import brier_score_loss
from src.ml_plugins.base_metric_plugin import MetricPlugin

class BrierScoreLossMetric(MetricPlugin):
    """Brier Score Loss metric for probabilistic binary classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Brier Score Loss"
        self._description = "Mean squared difference between predicted probabilities and actual binary outcomes. Measures both calibration and discrimination."
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = True  # Needs prediction probabilities
        self._higher_is_better = False  # Lower loss is better
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            if y_proba is None:
                raise ValueError("Brier Score Loss requires prediction probabilities")
            
            # For binary classification, use probabilities of positive class
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                # Multi-class case - use probability of positive class (class 1)
                if y_proba.shape[1] == 2:
                    y_prob_pos = y_proba[:, 1]
                else:
                    raise ValueError("Brier Score Loss is primarily designed for binary classification")
            else:
                y_prob_pos = y_proba.flatten()
            
            return float(brier_score_loss(y_true, y_prob_pos))
        except Exception as e:
            raise ValueError(f"Error calculating Brier Score Loss: {str(e)}")
    
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
    return BrierScoreLossMetric()
