import numpy as np
from typing import Optional
from sklearn.metrics import explained_variance_score
from src.ml_plugins.base_metric_plugin import MetricPlugin

class ExplainedVarianceMetric(MetricPlugin):
    """Explained Variance Score metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Explained Variance Score"
        self._description = "Proportion of variance in the target variable that is predictable from the features"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = True  # Higher explained variance is better
        self._range = (-float('inf'), 1.0)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            return float(explained_variance_score(y_true, y_pred))
        except Exception as e:
            raise ValueError(f"Error calculating Explained Variance Score: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of Explained Variance Score."""
        if value >= 0.95:
            return "Excellent (Explains >95% of variance)"
        elif value >= 0.85:
            return "Good (Explains 85-95% of variance)"
        elif value >= 0.70:
            return "Fair (Explains 70-85% of variance)"
        elif value >= 0.50:
            return "Poor (Explains 50-70% of variance)"
        elif value >= 0:
            return "Very Poor (Explains <50% of variance)"
        else:
            return "Extremely Poor (Negative explained variance)"

def get_metric_plugin() -> MetricPlugin:
    return ExplainedVarianceMetric()
