import numpy as np
from typing import Optional
from scipy.stats import pearsonr
from src.ml_plugins.base_metric_plugin import MetricPlugin

class PearsonCorrelationMetric(MetricPlugin):
    """Pearson Correlation Coefficient metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Pearson Correlation Coefficient"
        self._description = "Measures linear correlation between predicted and actual values. Values close to 1 indicate strong positive correlation."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = True  # Higher correlation is better
        self._range = (-1, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Handle edge cases
            if len(y_true) < 2:
                raise ValueError("Pearson correlation requires at least 2 data points")
            
            # Check for constant arrays
            if np.var(y_true) == 0 or np.var(y_pred) == 0:
                return 0.0  # No correlation possible with constant values
            
            correlation, _ = pearsonr(y_true, y_pred)
            
            # Handle NaN result
            if np.isnan(correlation):
                return 0.0
                
            return float(correlation)
        except Exception as e:
            raise ValueError(f"Error calculating Pearson correlation: {str(e)}")
    
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
    return PearsonCorrelationMetric()
