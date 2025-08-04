import streamlit as st

import numpy as np

from typing import Any, Optional

from sklearn.metrics import balanced_accuracy_score



# Import for plugin system - will be auto-fixed during save

try:

    from src.ml_plugins.base_metric_plugin import MetricPlugin

except ImportError:

    # Fallback for testing

    import sys

    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

    sys.path.append(project_root)

    from src.ml_plugins.base_metric_plugin import MetricPlugin



class BalancedAccuracyMetric(MetricPlugin):

    def __init__(self):

        super().__init__()

        self._name = "Balanced Accuracy Score"

        self._description = "Average of recall obtained on each class - useful for imbalanced datasets"

        self._category = "Classification"

        self._supports_classification = True

        self._supports_regression = False

        self._requires_probabilities = False

        self._higher_is_better = True

        self._range = (0.0, 1.0)

        

    def get_name(self) -> str:

        return self._name

        

    def get_description(self) -> str:

        return self._description

        

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:

        """Calculate the balanced accuracy score"""

        try:

            return float(balanced_accuracy_score(y_true, y_pred))

        except Exception as e:

            raise ValueError(f"Error calculating balanced accuracy: {str(e)}")

        

    def get_interpretation(self, value: float) -> str:

        """Provide interpretation of the metric value"""

        if value >= 0.95:

            return "Excellent (Nearly perfect balance across classes)"

        elif value >= 0.85:

            return "Very Good (Strong performance on all classes)"

        elif value >= 0.75:

            return "Good (Decent balance across classes)"

        elif value >= 0.65:

            return "Fair (Some class imbalance issues)"

        elif value >= 0.55:

            return "Poor (Significant class imbalance problems)"

        else:

            return "Very Poor (Severe class imbalance issues)"



def get_metric_plugin() -> MetricPlugin:

    return BalancedAccuracyMetric()