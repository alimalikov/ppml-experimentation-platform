from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

class MetricPlugin(ABC):
    """
    Abstract base class for custom evaluation metric plugins.
    Each plugin represents a specific evaluation metric with its calculation logic.
    """
    
    def __init__(self):
        self._name: str = ""
        self._description: str = ""
        self._category: str = "General"  # e.g., "Classification", "Regression", "Ranking", "Probabilistic"
        self._supports_classification: bool = True
        self._supports_regression: bool = False
        self._requires_probabilities: bool = False  # Does this metric need prediction probabilities?
        self._requires_multiclass: bool = False  # Does this metric work only with multiclass?
        self._higher_is_better: bool = True  # Direction of optimization
        self._range: tuple = (0, 1)  # Expected range of metric values
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of the metric."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a brief description of what this metric measures."""
        pass
    
    def get_category(self) -> str:
        """Return the category/group this metric belongs to."""
        return self._category
    
    def supports_task_type(self, task_type: str) -> bool:
        """Check if this metric supports the given task type."""
        if task_type.lower() == "classification":
            return self._supports_classification
        elif task_type.lower() == "regression":
            return self._supports_regression
        return False
    
    def requires_probabilities(self) -> bool:
        """Check if this metric requires prediction probabilities."""
        return self._requires_probabilities
    
    def is_higher_better(self) -> bool:
        """Return True if higher values indicate better performance."""
        return self._higher_is_better
    
    def get_value_range(self) -> tuple:
        """Return the expected range of metric values (min, max)."""
        return self._range
    
    def is_compatible_with_data(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> tuple[bool, str]:
        """
        Check if this metric is compatible with the given predictions.
        Returns (is_compatible, reason_if_not)
        """
        if self._requires_probabilities and y_proba is None:
            return False, "This metric requires prediction probabilities"
        
        if self._requires_multiclass and len(np.unique(y_true)) < 3:
            return False, "This metric requires multiclass data (3+ classes)"
        
        return True, "Compatible"
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        Calculate the metric value.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (if available and needed)
            **kwargs: Additional parameters for metric calculation
            
        Returns:
            Calculated metric value
        """
        pass
    
    def get_interpretation(self, value: float) -> str:
        """
        Provide interpretation guidance for the metric value.
        Returns a human-readable interpretation of the metric value.
        """
        if self._higher_is_better:
            if value >= 0.9:
                return "Excellent"
            elif value >= 0.8:
                return "Good"
            elif value >= 0.7:
                return "Fair"
            elif value >= 0.6:
                return "Poor"
            else:
                return "Very Poor"
        else:
            if value <= 0.1:
                return "Excellent"
            elif value <= 0.2:
                return "Good"
            elif value <= 0.3:
                return "Fair"
            elif value <= 0.4:
                return "Poor"
            else:
                return "Very Poor"

# Factory function that metric plugins must implement
def get_metric_plugin() -> MetricPlugin:
    """Factory function that returns an instance of the metric plugin."""
    raise NotImplementedError("Each metric plugin must implement get_metric_plugin() function")