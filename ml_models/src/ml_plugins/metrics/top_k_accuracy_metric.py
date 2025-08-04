import numpy as np
from typing import Optional
from sklearn.metrics import top_k_accuracy_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class TopKAccuracyMetric(MetricPlugin):
    """Top-K Accuracy metric for multi-class classification tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Top-K Accuracy"
        self._description = "Fraction of samples whose true label is among the top-k predicted labels. Useful for multi-class problems with many classes."
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
                raise ValueError("Top-K Accuracy requires prediction probabilities")
            
            k = kwargs.get('k', 3)  # Default to top-3 accuracy
            
            # Ensure k is valid
            n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
            k = min(k, n_classes)
            
            return float(top_k_accuracy_score(y_true, y_proba, k=k))
        except Exception as e:
            raise ValueError(f"Error calculating Top-K Accuracy: {str(e)}")
    
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
    return TopKAccuracyMetric()
