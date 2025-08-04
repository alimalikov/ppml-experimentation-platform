import numpy as np
from typing import Optional
from sklearn.metrics import average_precision_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class AveragePrecisionMetric(MetricPlugin):
    """Average Precision Score metric for binary and multi-class classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Average Precision Score"
        self._description = "Area under the Precision-Recall curve. Particularly useful for imbalanced datasets where positive class is rare."
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = True  # Needs prediction probabilities
        self._higher_is_better = True
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            if y_proba is None:
                raise ValueError("Average Precision Score requires prediction probabilities")
            
            average = kwargs.get('average', 'macro')  # Default to macro averaging
            
            # Handle binary vs multi-class
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
                # Multi-class case
                return float(average_precision_score(y_true, y_proba, average=average))
            else:
                # Binary case - use probabilities of positive class
                if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                    y_prob_pos = y_proba[:, 1]
                else:
                    y_prob_pos = y_proba.flatten()
                
                return float(average_precision_score(y_true, y_prob_pos))
        except Exception as e:
            raise ValueError(f"Error calculating Average Precision Score: {str(e)}")
    
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
    return AveragePrecisionMetric()
