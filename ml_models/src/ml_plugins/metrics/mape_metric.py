import numpy as np
from typing import Optional
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MeanAbsolutePercentageErrorMetric(MetricPlugin):
    """Mean Absolute Percentage Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Mean Absolute Percentage Error"
        self._description = "Mean percentage error between predicted and actual values - scale-independent"
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower MAPE is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Handle division by zero
            mask = y_true != 0
            if not np.any(mask):
                raise ValueError("Cannot calculate MAPE when all true values are zero")
            
            # Calculate MAPE only for non-zero true values
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return float(mape)
        except Exception as e:
            raise ValueError(f"Error calculating MAPE: {str(e)}")
    
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of MAPE value."""
        if value < 5:
            return "Excellent (<5% average error)"
        elif value < 10:
            return "Good (5-10% average error)"
        elif value < 20:
            return "Fair (10-20% average error)"
        elif value < 50:
            return "Poor (20-50% average error)"
        else:
            return "Very Poor (>50% average error)"

def get_metric_plugin() -> MetricPlugin:
    return MeanAbsolutePercentageErrorMetric()
