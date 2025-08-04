import numpy as np
from typing import Optional
from sklearn.metrics import mean_tweedie_deviance
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MeanTweedieDevianceMetric(MetricPlugin):
    """Mean Tweedie Deviance metric for regression tasks with continuous positive targets."""
    
    def __init__(self):
        super().__init__()
        self._name = "Mean Tweedie Deviance"
        self._description = "Flexible metric for positive continuous targets. Generalizes gamma and Poisson deviances (power=1.5)."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower deviance is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Use power=1.5 as a balanced compromise between Poisson (1.0) and Gamma (2.0)
            power = kwargs.get('power', 1.5)
            
            # Ensure positive values for Tweedie deviance when power > 1
            if power > 1 and (np.any(y_true <= 0) or np.any(y_pred <= 0)):
                raise ValueError(f"Tweedie deviance with power={power} requires strictly positive values")
            elif power <= 1 and (np.any(y_true < 0) or np.any(y_pred < 0)):
                raise ValueError(f"Tweedie deviance with power={power} requires non-negative values")
            
            return float(mean_tweedie_deviance(y_true, y_pred, power=power))
        except Exception as e:
            raise ValueError(f"Error calculating Mean Tweedie Deviance: {str(e)}")
    
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
    return MeanTweedieDevianceMetric()
