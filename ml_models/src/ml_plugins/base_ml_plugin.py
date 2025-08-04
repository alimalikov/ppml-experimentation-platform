from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator
import numpy as np # Added for type hinting y_proba

class MLPlugin(ABC):
    """
    Abstract base class for ML algorithm plugins.
    Each plugin represents a specific machine learning algorithm with its configuration.
    """
    
    def __init__(self):
        self._name: str = ""
        self._description: str = ""
        self._category: str = "General"  # e.g., "Ensemble", "Linear", "Tree-based", "Neural Network"
        self._supports_classification: bool = True
        self._supports_regression: bool = False
        self._min_samples_required: int = 10
        self._max_features_recommended: Optional[int] = None
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of the algorithm."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a brief description of the algorithm."""
        pass
    
    def get_category(self) -> str:
        """Return the category/group this algorithm belongs to."""
        return self._category
    
    def supports_task_type(self, task_type: str) -> bool:
        """Check if this plugin supports the given task type."""
        if task_type.lower() == "classification":
            return self._supports_classification
        elif task_type.lower() == "regression":
            return self._supports_regression
        return False
    
    def is_compatible_with_data(self, df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
        """
        Check if this algorithm is compatible with the given dataset.
        Returns (is_compatible, reason_if_not)
        """
        n_samples = len(df)
        n_features = len(df.columns) - 1  # Exclude target column
        
        if n_samples < self._min_samples_required:
            return False, f"Requires at least {self._min_samples_required} samples, got {n_samples}"
        
        if self._max_features_recommended and n_features > self._max_features_recommended:
            return False, f"Recommended for datasets with ≤{self._max_features_recommended} features, got {n_features}"
        
        return True, "Compatible"
    
    @abstractmethod
    def get_hyperparameter_config(self, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Generate Streamlit UI elements for hyperparameter configuration.
        Returns a dictionary of parameter names and their selected values.
        """
        pass
    
    @abstractmethod
    def create_model_instance(self, hyperparameters: Dict[str, Any]) -> BaseEstimator:
        """
        Create and return a configured instance of the ML model.
        """
        pass
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Optional preprocessing specific to this algorithm.
        Default implementation returns data unchanged.
        """
        return X, y
    
    def get_feature_importance(self, trained_model: BaseEstimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from the trained model if available.
        Returns None if feature importance is not available for this algorithm.
        """
        if hasattr(trained_model, 'feature_importances_'):
            return dict(zip(feature_names, trained_model.feature_importances_))
        elif hasattr(trained_model, 'coef_'):
            # For linear models, use absolute coefficient values
            import numpy as np
            coef = trained_model.coef_
            if coef.ndim > 1:  # Multi-class classification
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            return dict(zip(feature_names, coef))
        return None
    
    def get_model_specific_metrics(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Calculate any algorithm-specific metrics beyond the standard ones.
        Returns a dictionary of metric names and values.
        Note: Consider using get_algorithm_specific_metrics for consistency with train_and_evaluate_model.
        """
        return {}

    def get_algorithm_specific_metrics(self, 
                                       y_true: Union[pd.Series, np.ndarray], 
                                       y_pred: Union[pd.Series, np.ndarray], 
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate any algorithm-specific metrics using true labels, predictions, and (optionally) probabilities.
        This method is intended to be called by the evaluation pipeline (e.g., train_and_evaluate_model).
        
        Args:
            y_true: Ground truth target values.
            y_pred: Predicted target values.
            y_proba: Predicted probabilities (for classification tasks), if available.
                     Should be an array of shape (n_samples, n_classes).

        Returns:
            A dictionary where keys are metric names (str) and values are the calculated metric values (Any).
            Default implementation returns an empty dictionary. Individual plugins should override 
            this method if they have specific metrics to compute.
        """
        return {} # Default implementation returns an empty dictionary
    
    def export_model_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export the model configuration for saving/loading experiments.
        """
        return {
            "plugin_name": self.get_name(),
            "hyperparameters": hyperparameters,
            "plugin_category": self.get_category()
        }
    
    def import_model_config(self, config: Dict[str, Any], unique_key_prefix: str) -> None:
        """
        Import and apply a saved model configuration to the UI.
        """
        # Default implementation - individual plugins should override this
        # to set their specific UI element values in st.session_state
        pass
    
    def get_available_visualizations(self) -> Dict[str, str]:
        """Return available visualizations for this plugin."""
        return {
            "feature_importance": "📊 Feature Importance Analysis",
            "class_analysis": "🎯 Class-Specific Feature Analysis", 
            "probability_dist": "📈 Probability Distribution Analysis"
        }

    def render_visualization(self, viz_type: str, **kwargs):
        """Render a specific visualization type."""
        if viz_type == "feature_importance":
            return self.plot_feature_importance(**kwargs)
        elif viz_type == "class_analysis":
            return self.plot_class_feature_analysis(**kwargs)
        elif viz_type == "probability_dist":
            return self.plot_probability_distribution(**kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")

    def plot_feature_importance(self, **kwargs):
        """Default feature importance plot - plugins should override this."""
        import matplotlib.pyplot as plt
        import streamlit as st
        
        st.info("Feature importance visualization not implemented for this algorithm.")
        return None

    def plot_class_feature_analysis(self, **kwargs):
        """Default class analysis plot - plugins should override this."""
        import streamlit as st
        
        st.info("Class-specific feature analysis not implemented for this algorithm.")
        return None

    def plot_probability_distribution(self, **kwargs):
        """Default probability distribution plot - plugins should override this."""
        import streamlit as st
        
        st.info("Probability distribution analysis not implemented for this algorithm.")
        return None

# Factory function that plugins must implement
def get_plugin() -> MLPlugin:
    """Factory function that returns an instance of the plugin."""
    raise NotImplementedError("Each plugin must implement get_plugin() function")