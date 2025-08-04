import numpy as np
from typing import Optional
from sklearn.metrics import r2_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class R2ScoreMetric(MetricPlugin):
    """R² (Coefficient of Determination) metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "R² Score"
        self._description = "Coefficient of determination - proportion of variance in target explained by features"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = True  # Higher R² is better
        self._range = (-float('inf'), 1.0)  # Can be negative for very poor models
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(r2_score(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating R² Score: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of R² Score value."""
        if value >= 0.9:
            return "Excellent (Explains >90% of variance)"
        elif value >= 0.8:
            return "Good (Explains 80-90% of variance)"
        elif value >= 0.6:
            return "Fair (Explains 60-80% of variance)"
        elif value >= 0.3:
            return "Poor (Explains 30-60% of variance)"
        elif value >= 0:
            return "Very Poor (Explains <30% of variance)"
        else:
            return "Extremely Poor (Worse than mean baseline)"

def get_metric_plugin() -> MetricPlugin:
    return R2ScoreMetric()
