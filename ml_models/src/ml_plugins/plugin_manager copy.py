import os
import sys
import importlib.util
import glob
from typing import List, Dict, Optional, Any
import streamlit as st
import pandas as pd
from .base_ml_plugin import MLPlugin

class MLPluginManager:
    """
    Manages discovery, loading, and organization of ML algorithm plugins.
    """
    
    def __init__(self, plugins_directory: str = None):
        if plugins_directory is None:
            # Default to plugins subdirectory
            plugins_directory = os.path.join(os.path.dirname(__file__), "algorithms")
        
        self.plugins_directory = plugins_directory
        self._loaded_plugins: Dict[str, MLPlugin] = {}
        self._plugin_categories: Dict[str, List[str]] = {}
        
        # Ensure plugins directory exists
        os.makedirs(self.plugins_directory, exist_ok=True)
        
        # Load all plugins
        self._discover_and_load_plugins()
    
    def _discover_and_load_plugins(self):
        """Discover and load all plugins from the plugins directory."""
        plugin_files = glob.glob(os.path.join(self.plugins_directory, "*_plugin.py"))
        
        for plugin_file in plugin_files:
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                st.warning(f"Failed to load plugin {os.path.basename(plugin_file)}: {e}")
    
    def _load_plugin(self, plugin_file_path: str):
        """Load a single plugin from a file."""
        plugin_name = os.path.basename(plugin_file_path)[:-3]  # Remove .py extension
        
        # Load the module
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin spec from {plugin_file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the plugin instance
        if hasattr(module, 'get_plugin'):
            plugin_instance = module.get_plugin()
            if isinstance(plugin_instance, MLPlugin):
                plugin_key = plugin_instance.get_name()
                self._loaded_plugins[plugin_key] = plugin_instance
                
                # Organize by category
                category = plugin_instance.get_category()
                if category not in self._plugin_categories:
                    self._plugin_categories[category] = []
                self._plugin_categories[category].append(plugin_key)
                
                # st.success(f"✅ Loaded plugin: {plugin_key}")
            else:
                raise TypeError(f"Plugin {plugin_name} does not return a valid MLPlugin instance")
        else:
            raise AttributeError(f"Plugin {plugin_name} does not have a get_plugin() function")
    
    def get_available_plugins(self, task_type: str = "classification") -> Dict[str, MLPlugin]:
        """Get all plugins that support the specified task type."""
        compatible_plugins = {}
        for name, plugin in self._loaded_plugins.items():
            if plugin.supports_task_type(task_type):
                compatible_plugins[name] = plugin
        return compatible_plugins
    
    def get_plugins_by_category(self, task_type: str = "classification") -> Dict[str, List[str]]:
        """Get plugins organized by category for the specified task type."""
        categorized = {}
        available_plugins = self.get_available_plugins(task_type)
        
        for plugin_name in available_plugins:
            plugin = self._loaded_plugins[plugin_name]
            category = plugin.get_category()
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(plugin_name)
        
        return categorized
    
    def get_plugin(self, plugin_name: str) -> Optional[MLPlugin]:
        """Get a specific plugin by name."""
        return self._loaded_plugins.get(plugin_name)
    
    def get_compatible_plugins(self, df: pd.DataFrame, target_col: str, task_type: str = "classification") -> Dict[str, MLPlugin]:
        """Get plugins that are compatible with the given dataset."""
        compatible = {}
        available_plugins = self.get_available_plugins(task_type)
        
        for name, plugin in available_plugins.items():
            is_compatible, reason = plugin.is_compatible_with_data(df, target_col)
            if is_compatible:
                compatible[name] = plugin
        
        return compatible
    
    def get_plugin_count(self) -> int:
        """Get the total number of loaded plugins."""
        return len(self._loaded_plugins)
    
    def reload_plugins(self):
        """Reload all plugins (useful for development)."""
        self._loaded_plugins.clear()
        self._plugin_categories.clear()
        self._discover_and_load_plugins()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded plugins.
        
        Returns:
            Dictionary containing plugin statistics
        """
        if not self.plugins:
            return {
                'total_plugins': 0,
                'categories': {},
                'data_types': {},
                'load_time': self._load_time if hasattr(self, '_load_time') else None,
                'errors': len(getattr(self, '_plugin_load_errors', {})),
                'error_details': getattr(self, '_plugin_load_errors', {}).copy()
            }
        
        # Count by category (if plugins have categories)
        categories = {}
        for plugin in self.plugins.values():
            # Adjust this based on your plugin structure
            cat = getattr(plugin, 'category', 'Unknown')
            if hasattr(cat, 'value'):
                cat = cat.value
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_plugins': len(self.plugins),
            'categories': categories,
            'data_types': {},  # Add data type counting if needed
            'load_time': getattr(self, '_load_time', None),
            'errors': len(getattr(self, '_plugin_load_errors', {})),
            'error_details': getattr(self, '_plugin_load_errors', {}).copy()
        }

# Global plugin manager instance
@st.cache_resource
def get_plugin_manager() -> MLPluginManager:
    """Get or create the global plugin manager instance."""
    return MLPluginManager()