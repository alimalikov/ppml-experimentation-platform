import numpy as np
from typing import Optional
from src.ml_plugins.base_metric_plugin import MetricPlugin

class MASEMetric(MetricPlugin):
    """Mean Absolute Scaled Error metric for regression tasks."""
    
    def __init__(self):
        super().__init__()
        self._name = "Mean Absolute Scaled Error"
        self._description = "Scale-independent metric that compares MAE to naive forecast MAE. Values < 1 indicate better than naive forecast."
        self._category = "Regression"
        self._supports_classification = False
        self._supports_regression = True
        self._requires_probabilities = False
        self._higher_is_better = False  # Lower MASE is better
        self._range = (0, float('inf'))
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Calculate MAE of predictions
            mae = np.mean(np.abs(y_true - y_pred))
            
            # Calculate MAE of naive forecast (using previous value as prediction)
            if len(y_true) < 2:
                raise ValueError("MASE requires at least 2 data points")
            
            # For time series, use lag-1 as naive forecast
            # For non-time series, use mean as naive forecast
            if 'seasonal_period' in kwargs:
                # Time series case with seasonality
                seasonal_period = kwargs['seasonal_period']
                if len(y_true) <= seasonal_period:
                    # Fall back to lag-1 if not enough data for seasonal naive
                    naive_errors = np.abs(np.diff(y_true))
                else:
                    naive_errors = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period])
            else:
                # Simple lag-1 naive forecast for time series or mean for cross-sectional
                if kwargs.get('is_time_series', False):
                    naive_errors = np.abs(np.diff(y_true))
                else:
                    # For cross-sectional data, use mean as naive forecast
                    mean_naive = np.mean(y_true)
                    naive_errors = np.abs(y_true - mean_naive)
            
            naive_mae = np.mean(naive_errors)
            
            # Avoid division by zero
            if naive_mae == 0:
                return float('inf') if mae > 0 else 0.0
            
            return float(mae / naive_mae)
        except Exception as e:
            raise ValueError(f"Error calculating MASE: {str(e)}")
    
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
    return MASEMetric()
