import numpy as np
from typing import Optional
from sklearn.metrics import precision_recall_fscore_support
from src.ml_plugins.base_metric_plugin import MetricPlugin

class GeometricMeanScoreMetric(MetricPlugin):
    """Geometric Mean Score metric for imbalanced classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Geometric Mean Score"
        self._description = "Geometric mean of class-wise sensitivity (recall). Particularly useful for imbalanced datasets as it's sensitive to poor performance on minority classes."
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
            # Get recall (sensitivity) for each class
            _, recalls, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            
            # Handle zero recalls to avoid issues with geometric mean
            # Replace zeros with a small value to prevent geometric mean from being zero
            recalls = np.where(recalls == 0, 1e-10, recalls)
            
            # Calculate geometric mean
            if len(recalls) == 0:
                return 0.0
            
            geometric_mean = np.prod(recalls) ** (1.0 / len(recalls))
            return float(geometric_mean)
        except Exception as e:
            raise ValueError(f"Error calculating Geometric Mean Score: {str(e)}")
    
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
    return GeometricMeanScoreMetric()
