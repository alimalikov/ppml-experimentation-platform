import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class ROCAUCMetric(MetricPlugin):
    """ROC AUC Score metric for binary and multiclass classification."""
    
    def __init__(self):
        super().__init__()
        self._name = "ROC AUC Score"
        self._description = "Area Under the Receiver Operating Characteristic Curve - measures classifier's ability to distinguish between classes"
        self._category = "Probabilistic"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = True
        self._higher_is_better = True
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        if y_proba is None:
            raise ValueError("ROC AUC requires prediction probabilities")
        
        try:
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                # Binary classification - use probabilities of positive class
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    y_proba = y_proba[:, 1]
                return roc_auc_score(y_true, y_proba)
            else:
                # Multiclass classification
                return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception as e:
            raise ValueError(f"Failed to calculate ROC AUC: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        if value >= 0.9:
            return "Excellent (Outstanding discrimination)"
        elif value >= 0.8:
            return "Good (Acceptable discrimination)"
        elif value >= 0.7:
            return "Fair (Some discrimination ability)"
        elif value >= 0.6:
            return "Poor (Poor discrimination)"
        else:
            return "Very Poor (No discrimination ability)"

def get_metric_plugin() -> MetricPlugin:
    return ROCAUCMetric()