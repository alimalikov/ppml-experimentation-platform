"""ML Plugins package for extensible machine learning functionality."""

# Make imports available at package level
try:
    from .plugin_manager import get_plugin_manager, MLPluginManager
    from .metric_manager import get_metric_manager, MetricPluginManager  
    from .visualization_manager import get_visualization_manager, VisualizationManager
    from .base_visualization_plugin import BaseVisualizationPlugin, VisualizationCategory, DataType
    PLUGINS_AVAILABLE = True
except ImportError as e:
    PLUGINS_AVAILABLE = False
    print(f"Some plugins not available: {e}")

__all__ = [
    'get_plugin_manager', 'MLPluginManager',
    'get_metric_manager', 'MetricPluginManager',
    'get_visualization_manager', 'VisualizationManager', 
    'BaseVisualizationPlugin', 'VisualizationCategory', 'DataType',
    'PLUGINS_AVAILABLE'
]