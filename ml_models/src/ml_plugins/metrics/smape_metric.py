import numpy as np
from typing import Optional
from src.ml_plugins.base_metric_plugin import MetricPlugin

class SymmetricMAPEMetric(MetricPlugin):
    """Symmetric Mean Absolute Percentage Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Symmetric Mean Absolute Percentage Error"
        self._description = "Symmetric version of MAPE that treats over- and under-forecasts equally. Less biased than MAPE."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower SMAPE is better
        self._range = (0, 200)  # SMAPE ranges from 0% to 200%
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Calculate symmetric MAPE
            numerator = np.abs(y_pred - y_true)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            
            # Handle division by zero
            mask = denominator != 0
            if not np.any(mask):
                return 0.0  # All predictions and actual values are zero
            
            smape = np.mean(numerator[mask] / denominator[mask]) * 100
            return float(smape)
        except Exception as e:
            raise ValueError(f"Error calculating SMAPE: {str(e)}")
    
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
    return SymmetricMAPEMetric()
