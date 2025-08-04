import os
import sys
import importlib.util
import glob
from typing import List, Dict, Optional, Any
import streamlit as st
import numpy as np
from .base_metric_plugin import MetricPlugin
from .base_ml_plugin import MLPlugin # Ensure MLPlugin (base for algorithm plugins) is imported

class MetricPluginManager:
    """
    Manages discovery, loading, and organization of metric plugins.
    """
    
    def __init__(self, metrics_directory: str = None):
        if metrics_directory is None:
            # Default to metrics subdirectory
            metrics_directory = os.path.join(os.path.dirname(__file__), "metrics")
        
        self.metrics_directory = metrics_directory
        self._loaded_metrics: Dict[str, MetricPlugin] = {}
        self._metric_categories: Dict[str, List[str]] = {}
        
        # Ensure metrics directory exists
        os.makedirs(self.metrics_directory, exist_ok=True)
        
        # Load all metric plugins
        self._discover_and_load_metrics()
    
    def _discover_and_load_metrics(self):
        """Discover and load all metric plugins from the metrics directory."""
        metric_files = glob.glob(os.path.join(self.metrics_directory, "*_metric.py"))
        
        for metric_file in metric_files:
            try:
                self._load_metric(metric_file)
            except Exception as e:
                st.warning(f"Failed to load metric {os.path.basename(metric_file)}: {e}")
    
    def _load_metric(self, metric_file_path: str):
        """Load a single metric plugin from a file."""
        metric_name = os.path.basename(metric_file_path)[:-3]  # Remove .py extension
        
        # Load the module
        spec = importlib.util.spec_from_file_location(metric_name, metric_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load metric spec from {metric_file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the metric instance
        if hasattr(module, 'get_metric_plugin'):
            metric_instance = module.get_metric_plugin()
            if isinstance(metric_instance, MetricPlugin):
                metric_key = metric_instance.get_name()
                self._loaded_metrics[metric_key] = metric_instance
                
                # Organize by category
                category = metric_instance.get_category()
                if category not in self._metric_categories:
                    self._metric_categories[category] = []
                self._metric_categories[category].append(metric_key)
                
            else:
                raise TypeError(f"Metric {metric_name} does not return a valid MetricPlugin instance")
        else:
            raise AttributeError(f"Metric {metric_name} does not have a get_metric_plugin() function")
    
    def get_available_metrics(self, task_type: str = "classification") -> Dict[str, MetricPlugin]:
        """Get all metrics that support the specified task type."""
        compatible_metrics = {}
        for name, metric in self._loaded_metrics.items():
            if metric.supports_task_type(task_type):
                compatible_metrics[name] = metric
        return compatible_metrics
    
    def get_metrics_by_category(self, task_type: str = "classification") -> Dict[str, List[str]]:
        """Get metrics organized by category for the specified task type."""
        categorized = {}
        available_metrics = self.get_available_metrics(task_type)
        
        for metric_name in available_metrics:
            metric = self._loaded_metrics[metric_name]
            category = metric.get_category()
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(metric_name)
        
        return categorized
    
    def get_metric(self, metric_name: str) -> Optional[MetricPlugin]:
        """Get a specific metric by name."""
        return self._loaded_metrics.get(metric_name)
    
    def calculate_metrics(self,
                          selected_metrics: List[str],
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None,
                          trained_algorithm_plugin: Optional[MLPlugin] = None  # <-- ADD THIS NEW ARGUMENT
                         ) -> Dict[str, Any]: # Return type can be Any to accommodate errors or float values
        """Calculate multiple metrics for the given predictions."""
        results = {}
        
        for metric_name in selected_metrics:
            metric = self.get_metric(metric_name) # metric is an instance of a metric plugin
            if metric is None:
                results[metric_name] = "Error: Metric not found" # Provide error message
                continue
                
            try:
                # Check compatibility
                is_compatible, reason = metric.is_compatible_with_data(y_true, y_pred, y_proba)
                if not is_compatible:
                    results[metric_name] = f"Error: {reason}"
                    continue
                
                # Calculate metric
                if isinstance(metric, AlgorithmMetricExtractor):
                    # Pass the trained_algorithm_plugin to AlgorithmMetricExtractor's calculate method
                    value = metric.calculate(y_true, y_pred, y_proba, trained_model=trained_algorithm_plugin)
                else:
                    # For standard metrics that don't need the full trained_model
                    value = metric.calculate(y_true, y_pred, y_proba)
                
                results[metric_name] = value
                
            except Exception as e:
                results[metric_name] = f"Error calculating {metric_name}: {str(e)}" # More informative error
        
        return results
    
    def get_metric_count(self) -> int:
        """Get the total number of loaded metrics."""
        return len(self._loaded_metrics)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded metrics.
        
        Returns:
            Dictionary containing metric statistics
        """
        # Adjust this based on your actual metric manager structure
        return {
            'total_metrics': len(getattr(self, 'metrics', {})),
            'categories': {},
            'load_time': getattr(self, '_load_time', None),
            'errors': len(getattr(self, '_metric_load_errors', {})),
            'error_details': getattr(self, '_metric_load_errors', {}).copy()
        }

# Global metric manager instance
@st.cache_resource
def get_metric_manager() -> MetricPluginManager:
    """Get or create the global metric manager instance."""
    return MetricPluginManager()