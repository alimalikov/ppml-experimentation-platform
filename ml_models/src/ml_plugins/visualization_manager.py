import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import traceback

from .base_visualization_plugin import (
    BaseVisualizationPlugin, 
    VisualizationCategory, 
    DataType, 
    VisualizationError
)

class VisualizationManager:
    """
    Manager class for visualization plugins.
    
    Handles plugin discovery, loading, validation, and execution.
    Provides a unified interface for managing all visualization plugins.
    """
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize the visualization manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.plugins: Dict[str, BaseVisualizationPlugin] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self.plugin_directories = plugin_directories or []
        self.logger = logging.getLogger("VisualizationManager")
        
        # Performance tracking
        self._load_time = None
        self._plugin_load_errors: Dict[str, str] = {}
        
        # Cache for expensive operations
        self._compatibility_cache: Dict[str, bool] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Default plugin directories
        current_dir = Path(__file__).parent
        default_dirs = [
            str(current_dir / "visualizations"),
            str(current_dir / "custom_visualizations"),
        ]
        self.plugin_directories.extend(default_dirs)
        
        # Ensure plugin directories exist
        self._ensure_plugin_directories()
        
        # Load plugins on initialization
        self.reload_plugins()
    
    def _ensure_plugin_directories(self) -> None:
        """Create plugin directories if they don't exist."""
        for directory in self.plugin_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files to make directories Python packages
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Visualization plugins directory\n")
    
    def reload_plugins(self) -> Tuple[int, int]:
        """
        Reload all plugins from plugin directories.
        
        Returns:
            Tuple of (loaded_count, error_count)
        """
        start_time = time.time()
        self.logger.info("Starting plugin reload...")
        
        # DEBUG: Print plugin directories
        print(f"DEBUG: Plugin directories: {self.plugin_directories}")
        for directory in self.plugin_directories:
            print(f"DEBUG: Checking directory: {directory}")
            if os.path.exists(directory):
                files = list(Path(directory).glob("*.py"))
                print(f"DEBUG: Found Python files: {[f.name for f in files]}")
            else:
                print(f"DEBUG: Directory does not exist: {directory}")
        
        # Clear existing plugins
        self.plugins.clear()
        self.plugin_metadata.clear()
        self._plugin_load_errors.clear()
        self._compatibility_cache.clear()
        
        loaded_count = 0
        error_count = 0
        
        for directory in self.plugin_directories:
            if not os.path.exists(directory):
                continue
                
            try:
                # Add directory to Python path if not already there
                if directory not in sys.path:
                    sys.path.insert(0, directory)
                
                # Find all Python files in the directory
                for file_path in Path(directory).glob("*.py"):
                    if file_path.name.startswith("_"):
                        continue  # Skip private files
                    
                    try:
                        plugin_count, plugin_errors = self._load_plugin_file(file_path)
                        loaded_count += plugin_count
                        error_count += plugin_errors
                        
                    except Exception as e:
                        self.logger.error(f"Error loading plugin file {file_path}: {str(e)}")
                        self._plugin_load_errors[str(file_path)] = str(e)
                        error_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error scanning directory {directory}: {str(e)}")
                error_count += 1
        
        self._load_time = time.time() - start_time
        
        self.logger.info(f"Plugin reload completed: {loaded_count} loaded, {error_count} errors in {self._load_time:.2f}s")
        
        # Store metadata
        self._update_plugin_metadata()
        
        return loaded_count, error_count
    
    def _load_plugin_file(self, file_path: Path) -> Tuple[int, int]:
        """
        Load plugins from a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (loaded_count, error_count)
        """
        module_name = file_path.stem
        loaded_count = 0
        error_count = 0
        
        print(f"DEBUG: Attempting to load plugin file: {file_path}")
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                print(f"DEBUG: Could not create spec for {file_path}")
                raise ImportError(f"Could not load spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            print(f"DEBUG: Module created for {module_name}")
            
            spec.loader.exec_module(module)
            print(f"DEBUG: Module executed for {module_name}")
            
            # Find all classes that inherit from BaseVisualizationPlugin
            for name, obj in inspect.getmembers(module, inspect.isclass):
                print(f"DEBUG: Found class {name} in {module_name}")
                
                # Check if it's a class defined in this module (not imported)
                if obj.__module__ != module.__name__:
                    print(f"DEBUG: {name} is imported from {obj.__module__}, skipping")
                    continue
                
                try:
                    # More robust inheritance check
                    if (hasattr(obj, '__bases__') and 
                        any('BaseVisualizationPlugin' in str(base.__name__ if hasattr(base, '__name__') else str(base)) for base in obj.__bases__)): # Using base.__name__ if available
                        
                        print(f"DEBUG: {name} appears to inherit from BaseVisualizationPlugin")
                        
                        # Additional check - try to instantiate
                        try:
                            plugin_instance = obj()
                            print(f"DEBUG: Successfully instantiated {name}")
                            
                            # Check if it has required methods
                            required_methods = ['can_visualize', 'render', 'get_config_options']
                            has_required_methods = all(hasattr(plugin_instance, method) for method in required_methods)
                            
                            if has_required_methods:
                                print(f"DEBUG: {name} has all required methods")
                                
                                # Validate plugin
                                if self._validate_plugin(plugin_instance):
                                    plugin_key = f"{module_name}.{name}"
                                    self.plugins[plugin_key] = plugin_instance
                                    loaded_count += 1
                                    self.logger.info(f"Loaded plugin: {plugin_instance.name}")
                                    print(f"DEBUG: Successfully loaded plugin: {plugin_instance.name}")
                                else:
                                    self.logger.warning(f"Plugin validation failed: {name}")
                                    print(f"DEBUG: Plugin validation failed for {name}")
                                    self._plugin_load_errors[f"{module_name}.{name}"] = "Validation failed"
                                    error_count += 1
                            else:
                                missing_methods_list = [m for m in required_methods if not hasattr(plugin_instance, m)]
                                error_message = f"Missing required methods: {missing_methods_list}"
                                self.logger.warning(f"Plugin {name} {error_message.lower()}")
                                print(f"DEBUG: {name} is missing required methods: {missing_methods_list}")
                                self._plugin_load_errors[f"{module_name}.{name}"] = error_message
                                error_count += 1
                                
                        except Exception as e_instantiate:
                            instantiation_error_msg = f"Error instantiating plugin {name}: {str(e_instantiate)}"
                            self.logger.error(instantiation_error_msg)
                            print(f"DEBUG: Error instantiating {name}: {str(e_instantiate)}")
                            print(f"DEBUG: Traceback: {traceback.format_exc()}")
                            self._plugin_load_errors[f"{module_name}.{name}"] = f"Instantiation error: {str(e_instantiate)}"
                            error_count += 1
                    else: # This corresponds to lines 220-221 of your selection
                        print(f"DEBUG: {name} does not inherit from BaseVisualizationPlugin")
                    
                except Exception as e_check: # This corresponds to lines 222-224 of your selection
                    check_error_msg = f"Error during plugin compatibility check for class {name}: {str(e_check)}"
                    self.logger.error(check_error_msg)
                    print(f"DEBUG: Error checking class {name}: {str(e_check)}")
                    print(f"DEBUG: Traceback for error checking class {name}: {traceback.format_exc()}")
                    self._plugin_load_errors[f"{module_name}.{name}"] = f"Compatibility check error: {str(e_check)}"
                    error_count += 1
                    continue
                
    
        except Exception as e:
            self.logger.error(f"Error loading module {module_name}: {str(e)}")
            print(f"DEBUG: Error loading module {module_name}: {str(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            self._plugin_load_errors[module_name] = str(e)
            error_count += 1
        
        print(f"DEBUG: Plugin loading complete for {file_path}: {loaded_count} loaded, {error_count} errors")
        return loaded_count, error_count
    
    def _validate_plugin(self, plugin: BaseVisualizationPlugin) -> bool:
        """
        Validate that a plugin meets requirements.
        
        Args:
            plugin: Plugin instance to validate
            
        Returns:
            bool: True if plugin is valid, False otherwise
        """
        try:
            # Check required attributes
            required_attrs = ['name', 'description', 'category']
            for attr in required_attrs:
                if not hasattr(plugin, attr) or not getattr(plugin, attr):
                    self.logger.error(f"Plugin missing required attribute: {attr}")
                    return False
            
            # Check required methods are implemented
            required_methods = ['can_visualize', 'render', 'get_config_options']
            for method in required_methods:
                if not hasattr(plugin, method):
                    self.logger.error(f"Plugin missing required method: {method}")
                    return False
            
            # Test basic functionality
            config_options = plugin.get_config_options()
            if not isinstance(config_options, dict):
                self.logger.error("get_config_options must return a dictionary")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating plugin: {str(e)}")
            return False
    
    def _update_plugin_metadata(self) -> None:
        """Update metadata for all loaded plugins."""
        for plugin_key, plugin in self.plugins.items():
            self.plugin_metadata[plugin_key] = {
                **plugin.get_plugin_info(),
                'plugin_key': plugin_key,
                'loaded_at': datetime.now().isoformat(),
                'module_path': plugin.__class__.__module__
            }
    
    def get_available_plugins(self, 
                            data_type: Optional[Union[str, DataType]] = None,
                            category: Optional[Union[str, VisualizationCategory]] = None,
                            model_results: Optional[List[Dict[str, Any]]] = None,
                            data: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get list of available plugins that match the given criteria.
        
        Args:
            data_type: Filter by data type
            category: Filter by visualization category
            model_results: Model results to check compatibility
            data: Dataset to check compatibility
            
        Returns:
            List of plugin keys that match the criteria
        """
        available = []
        
        print(f"DEBUG get_available_plugins: data_type={data_type}, category={category}")
        print(f"DEBUG get_available_plugins: model_results count={len(model_results) if model_results else 0}")
        print(f"DEBUG get_available_plugins: data shape={data.shape if data is not None else 'None'}")
        
        for plugin_key, plugin in self.plugins.items():
            try:
                print(f"DEBUG: Checking plugin {plugin_key}")
                
                # Check category filter
                if category:
                    if isinstance(category, str):
                        try:
                            category_enum = VisualizationCategory(category.lower())
                        except ValueError:
                            print(f"DEBUG: Invalid category '{category}', skipping category filter")
                            category_enum = None
                    else:
                        category_enum = category
                    
                    if category_enum and hasattr(plugin, 'category') and plugin.category != category_enum:
                        print(f"DEBUG: Plugin {plugin_key} category mismatch: {plugin.category} != {category_enum}")
                        continue
                
                # Check if plugin can handle the data type and results
                # Use the plugin's can_visualize method instead of enum conversion
                if data_type is not None or model_results is not None:
                    try:
                        can_handle = plugin.can_visualize(data_type, model_results, data)
                        print(f"DEBUG: Plugin {plugin_key}.can_visualize({data_type}, {len(model_results) if model_results else 0} results, data) = {can_handle}")
                        
                        if can_handle:
                            available.append(plugin_key)
                            print(f"DEBUG: Added {plugin_key} to available plugins")
                        else:
                            print(f"DEBUG: Plugin {plugin_key} cannot handle the data")
                            
                    except Exception as e:
                        self.logger.warning(f"Error checking plugin {plugin_key} compatibility: {str(e)}")
                        print(f"DEBUG: Error checking {plugin_key}: {str(e)}")
                        import traceback
                        print(f"DEBUG: Traceback: {traceback.format_exc()}")
                        continue
                else:
                    # No specific requirements, add all plugins
                    available.append(plugin_key)
                    print(f"DEBUG: No requirements specified, added {plugin_key}")
            
            except Exception as e:
                self.logger.error(f"Error processing plugin {plugin_key}: {str(e)}")
                print(f"DEBUG: Error processing {plugin_key}: {str(e)}")
                continue
        
        print(f"DEBUG get_available_plugins: Final available plugins: {available}")
        return available
    
    def get_plugin(self, plugin_key: str) -> Optional[BaseVisualizationPlugin]:
        """
        Get a specific plugin by key.
        
        Args:
            plugin_key: Key of the plugin to retrieve
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_key)
    
    def get_plugin_by_name(self, name: str) -> Optional[BaseVisualizationPlugin]:
        """
        Get a plugin by its display name.
        
        Args:
            name: Display name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        for plugin in self.plugins.values():
            if plugin.name == name:
                return plugin
        return None
    
    def render_visualization(self, 
                           plugin_key: str,
                           data: pd.DataFrame,
                           model_results: List[Dict[str, Any]],
                           config: Optional[Dict[str, Any]] = None,
                           **kwargs) -> bool:
        """
        Render a visualization using the specified plugin.
        
        Args:
            plugin_key: Key of the plugin to use
            data: Dataset to visualize
            model_results: Model results to visualize
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        plugin = self.get_plugin(plugin_key)
        if plugin is None:
            st.error(f"❌ Plugin not found: {plugin_key}")
            return False
        
        try:
            return plugin.safe_render(data, model_results, config, **kwargs)
        except Exception as e:
            self.logger.error(f"Error rendering with plugin {plugin_key}: {str(e)}")
            st.error(f"❌ Error rendering visualization: {str(e)}")
            return False
    
    def create_plugin_ui(self, 
                        data_type: Optional[Union[str, DataType]] = None,
                        category: Optional[Union[str, VisualizationCategory]] = None,
                        model_results: Optional[List[Dict[str, Any]]] = None,
                        data: Optional[pd.DataFrame] = None,
                        key_prefix: str = "viz") -> Optional[str]:
        """
        Create a Streamlit UI for plugin selection and configuration.
        
        Args:
            data_type: Filter plugins by data type
            category: Filter plugins by category
            model_results: Model results for compatibility checking
            data: Dataset for compatibility checking
            key_prefix: Prefix for Streamlit component keys
            
        Returns:
            Selected plugin key or None
        """
        available_plugins = self.get_available_plugins(data_type, category, model_results, data)
        
        if not available_plugins:
            st.info("🔍 No compatible visualization plugins found for your data.")
            return None
        
        # Create plugin selection UI
        st.markdown("### 📊 **Select Visualization**")
        
        # Group plugins by category for better organization
        plugins_by_category = {}
        for plugin_key in available_plugins:
            plugin = self.get_plugin(plugin_key)
            if plugin:
                cat = plugin.category.value
                if cat not in plugins_by_category:
                    plugins_by_category[cat] = []
                plugins_by_category[cat].append((plugin_key, plugin.name))
        
        # Create selectbox options
        options = []
        option_mapping = {}
        
        for category_name, plugins in plugins_by_category.items():
            for plugin_key, plugin_name in plugins:
                display_name = f"📈 {plugin_name} ({category_name})"
                options.append(display_name)
                option_mapping[display_name] = plugin_key
        
        if not options:
            return None
        
        selected_display = st.selectbox(
            "Choose a visualization:",
            options,
            key=f"{key_prefix}_plugin_select"
        )
        
        selected_plugin_key = option_mapping.get(selected_display)
        if not selected_plugin_key:
            return None
        
        # Show plugin information
        plugin = self.get_plugin(selected_plugin_key)
        if plugin:
            with st.expander("ℹ️ Plugin Information", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {plugin.name}")
                    st.write(f"**Category:** {plugin.category.value}")
                    st.write(f"**Version:** {plugin.version}")
                    
                with col2:
                    st.write(f"**Author:** {plugin.author or 'Unknown'}")
                    st.write(f"**Interactive:** {'Yes' if plugin.interactive else 'No'}")
                    st.write(f"**Export Formats:** {', '.join(plugin.export_formats)}")
                
                st.write(f"**Description:** {plugin.description}")
            
            # Create configuration UI if needed
            config_options = plugin.get_config_options()
            if config_options:
                st.markdown("### ⚙️ **Configuration**")
                user_config = plugin.create_config_ui(f"{key_prefix}_{selected_plugin_key}")
                
                # Store config in session state
                config_key = f"viz_config_{selected_plugin_key}"
                st.session_state[config_key] = user_config
        
        return selected_plugin_key
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
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
                'load_time': self._load_time,
                'errors': len(self._plugin_load_errors),
                'error_details': self._plugin_load_errors.copy()  # Always include error_details
            }
        
        # Count by category
        categories = {}
        for plugin in self.plugins.values():
            cat = plugin.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        # Count by supported data types
        data_types = {}
        for plugin in self.plugins.values():
            for dt in plugin.supported_data_types:
                dt_name = dt.value
                data_types[dt_name] = data_types.get(dt_name, 0) + 1
        
        return {
            'total_plugins': len(self.plugins),
            'categories': categories,
            'data_types': data_types,
            'load_time': self._load_time,
            'errors': len(self._plugin_load_errors),
            'error_details': self._plugin_load_errors.copy(),  # Always include error_details
            'plugin_directories': self.plugin_directories
        }
    
    def export_plugin_config(self, filepath: str) -> bool:
        """
        Export plugin configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            config_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_plugins': len(self.plugins),
                    'manager_version': '1.0.0'
                },
                'plugins': self.plugin_metadata,
                'statistics': self.get_plugin_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Plugin configuration exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting plugin config: {str(e)}")
            return False
    
    def create_management_ui(self) -> None:
        """Create a Streamlit UI for plugin management."""
        st.markdown("## 🔧 **Visualization Plugin Management**")
        
        # Plugin statistics
        stats = self.get_plugin_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Plugins", stats['total_plugins'])
        with col2:
            st.metric("Categories", len(stats['categories']))
        with col3:
            st.metric("Load Time", f"{stats['load_time']:.2f}s" if stats['load_time'] else "N/A")
        with col4:
            st.metric("Errors", stats['errors'])
        
        # Plugin list
        if self.plugins:
            st.markdown("### 📋 **Loaded Plugins**")
            
            plugin_data = []
            for plugin_key, plugin in self.plugins.items():
                plugin_data.append({
                    'Name': plugin.name,
                    'Category': plugin.category.value,
                    'Version': plugin.version,
                    'Author': plugin.author or 'Unknown',
                    'Data Types': ', '.join([dt.value for dt in plugin.supported_data_types]),
                    'Renders': plugin._render_count
                })
            
            df = pd.DataFrame(plugin_data)
            st.dataframe(df, use_container_width=True)
        
        # Error details - with safety check
        error_details = stats.get('error_details', {})
        if stats['errors'] > 0 and error_details:
            st.markdown("### ❌ **Load Errors**")
            for plugin_name, error in error_details.items():
                st.error(f"**{plugin_name}**: {error}")
        elif stats['errors'] > 0:
            st.markdown("### ❌ **Load Errors**")
            st.error(f"Found {stats['errors']} error(s) but no detailed information available.")
        
        # Management actions
        st.markdown("### 🔄 **Actions**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Plugins", help="Reload all plugins from directories"):
                with st.spinner("Reloading plugins..."):
                    loaded, errors = self.reload_plugins()
                st.success(f"✅ Reloaded: {loaded} plugins loaded, {errors} errors")
                st.rerun()
        
        with col2:
            if st.button("📥 Export Config", help="Export plugin configuration to JSON"):
                try:
                    from datetime import datetime
                    filepath = f"plugin_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    if self.export_plugin_config(filepath):
                        st.success(f"✅ Configuration exported to {filepath}")
                    else:
                        st.error("❌ Failed to export configuration")
                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")

# Global instance for easy access
_visualization_manager = None

def get_visualization_manager(plugin_directories: Optional[List[str]] = None) -> VisualizationManager:
    """
    Get the global visualization manager instance.
    
    Args:
        plugin_directories: Optional list of plugin directories
        
    Returns:
        VisualizationManager instance
    """
    global _visualization_manager
    
    if _visualization_manager is None:
        _visualization_manager = VisualizationManager(plugin_directories)
    
    return _visualization_manager

def reset_visualization_manager() -> None:
    """Reset the global visualization manager instance."""
    global _visualization_manager
    _visualization_manager = None