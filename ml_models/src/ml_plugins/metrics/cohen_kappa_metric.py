import numpy as np
from typing import Optional
from sklearn.metrics import cohen_kappa_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class CohenKappaMetric(MetricPlugin):
    """Cohen's Kappa Score - measures inter-rater agreement for categorical items."""
    
    def __init__(self):
        super().__init__()
        self._name = "Cohen's Kappa Score"
        self._description = "Measures classification agreement accounting for chance agreement, useful for multi-class problems"
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
        """Calculate Cohen's Kappa Score"""
        try:
            return float(cohen_kappa_score(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Cohen's Kappa: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation based on Landis and Koch (1977) scale"""
        if value >= 0.81:
            return "Almost Perfect Agreement (0.81-1.00)"
        elif value >= 0.61:
            return "Substantial Agreement (0.61-0.80)"
        elif value >= 0.41:
            return "Moderate Agreement (0.41-0.60)"
        elif value >= 0.21:
            return "Fair Agreement (0.21-0.40)"
        elif value >= 0.01:
            return "Slight Agreement (0.01-0.20)"
        elif value >= 0:
            return "Poor Agreement (0.00)"
        else:
            return "Less than Chance Agreement (< 0)"

def get_metric_plugin() -> MetricPlugin:
    return CohenKappaMetric()