# base_visualization_plugin.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from enum import Enum
import logging

class VisualizationCategory(Enum):
    """Enumeration of visualization categories"""
    PERFORMANCE = "performance"
    COMPARISON = "comparison"
    INTERPRETATION = "interpretation"
    DATA_EXPLORATION = "data_exploration"
    MODEL_ANALYSIS = "model_analysis"
    FEATURE_ANALYSIS = "feature_analysis"
    ERROR_ANALYSIS = "error_analysis"

class DataType(Enum):
    """Enumeration of supported data types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    MULTICLASS = "multiclass"
    BINARY = "binary"

class VisualizationError(Exception):
    """Custom exception for visualization-related errors"""
    pass

class BaseVisualizationPlugin(ABC):
    """
    Abstract base class for all visualization plugins.
    
    This class provides the foundation for creating visualization plugins
    that can be dynamically loaded and used within the ML application.
    Each plugin should inherit from this class and implement the required methods.
    """
    
    def __init__(self):
        """Initialize the base visualization plugin"""
        self.name: str = ""
        self.description: str = ""
        self.version: str = "1.0.0"
        self.author: str = ""
        self.category: VisualizationCategory = VisualizationCategory.PERFORMANCE
        self.supported_data_types: List[DataType] = []
        self.required_columns: List[str] = []
        self.optional_columns: List[str] = []
        self.min_samples: int = 1
        self.max_samples: Optional[int] = None
        self.requires_trained_model: bool = False
        self.requires_predictions: bool = False
        self.requires_probabilities: bool = False
        self.interactive: bool = True
        self.export_formats: List[str] = ["png", "pdf", "html"]
        
        # Initialize logger
        self.logger = logging.getLogger(f"visualization.{self.__class__.__name__}")
        
        # Performance tracking
        self._render_count = 0
        self._last_render_time = None
        
        # Configuration cache
        self._config_cache = {}
    
    @abstractmethod
    def can_visualize(self, data_type: Union[str, DataType], model_results: List[Dict[str, Any]], 
                     data: Optional[pd.DataFrame] = None) -> bool:
        """
        Check if this plugin can handle the given data and model results.
        
        Args:
            data_type: Type of machine learning problem
            model_results: List of model result dictionaries
            data: Optional dataframe containing the dataset
            
        Returns:
            bool: True if plugin can handle the data, False otherwise
        """
        raise NotImplementedError("Subclasses must implement can_visualize method")
    
    @abstractmethod
    def render(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
               config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Render the visualization using Streamlit components.
        
        Args:
            data: Dataset used for training/testing
            model_results: List of model result dictionaries
            config: Configuration dictionary for the visualization
            **kwargs: Additional keyword arguments
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement render method")
    
    @abstractmethod
    def get_config_options(self) -> Dict[str, Dict[str, Any]]:
        """
        Return configuration options for this visualization.
        
        Returns:
            Dict containing configuration options with their properties:
            {
                'option_name': {
                    'type': 'select|multiselect|slider|checkbox|text|number',
                    'label': 'Human readable label',
                    'default': default_value,
                    'options': list_of_options (for select/multiselect),
                    'min': min_value (for slider/number),
                    'max': max_value (for slider/number),
                    'help': 'Help text',
                    'required': True/False
                }
            }
        """
        raise NotImplementedError("Subclasses must implement get_config_options method")
    
    def validate_data(self, data: pd.DataFrame, model_results: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that the provided data meets plugin requirements.
        
        Args:
            data: Dataset to validate
            model_results: Model results to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        try:
            # Check if data is provided and not empty
            if data is None or data.empty:
                errors.append("Data cannot be None or empty")
                return False, errors
            
            # Check minimum sample requirements
            if len(data) < self.min_samples:
                errors.append(f"Data must have at least {self.min_samples} samples, got {len(data)}")
            
            # Check maximum sample requirements
            if self.max_samples and len(data) > self.max_samples:
                errors.append(f"Data cannot have more than {self.max_samples} samples, got {len(data)}")
            
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check model results
            if not model_results:
                if self.requires_trained_model:
                    errors.append("This visualization requires trained model results")
            else:
                # Validate model results structure
                for i, result in enumerate(model_results):
                    if not isinstance(result, dict):
                        errors.append(f"Model result {i} must be a dictionary")
                        continue
                    
                    if self.requires_predictions and 'predictions' not in result:
                        errors.append(f"Model result {i} missing required 'predictions'")
                    
                    if self.requires_probabilities and 'probabilities' not in result:
                        errors.append(f"Model result {i} missing required 'probabilities'")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this plugin.
        
        Returns:
            Dictionary containing plugin metadata
        """
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'category': self.category.value,
            'supported_data_types': [dt.value for dt in self.supported_data_types],
            'required_columns': self.required_columns,
            'optional_columns': self.optional_columns,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples,
            'requires_trained_model': self.requires_trained_model,
            'requires_predictions': self.requires_predictions,
            'requires_probabilities': self.requires_probabilities,
            'interactive': self.interactive,
            'export_formats': self.export_formats,
            'render_count': self._render_count,
            'last_render_time': self._last_render_time
        }
    
    def create_config_ui(self, key_prefix: str = "") -> Dict[str, Any]:
        """
        Create Streamlit UI components for plugin configuration.
        
        Args:
            key_prefix: Prefix for Streamlit component keys to avoid conflicts
            
        Returns:
            Dictionary containing user-selected configuration values
        """
        config_options = self.get_config_options()
        user_config = {}
        
        if not config_options:
            return user_config
        
        st.markdown(f"**⚙️ {self.name} Configuration**")
        
        for option_name, option_props in config_options.items():
            key = f"{key_prefix}_{self.name}_{option_name}" if key_prefix else f"{self.name}_{option_name}"
            
            option_type = option_props.get('type', 'text')
            label = option_props.get('label', option_name)
            default = option_props.get('default')
            help_text = option_props.get('help', '')
            required = option_props.get('required', False)
            
            try:
                if option_type == 'select':
                    options = option_props.get('options', [])
                    value = st.selectbox(
                        label, options, 
                        index=options.index(default) if default in options else 0,
                        key=key, help=help_text
                    )
                    
                elif option_type == 'multiselect':
                    options = option_props.get('options', [])
                    default_list = default if isinstance(default, list) else [default] if default else []
                    value = st.multiselect(
                        label, options, default=default_list,
                        key=key, help=help_text
                    )
                    
                elif option_type == 'slider':
                    min_val = option_props.get('min', 0)
                    max_val = option_props.get('max', 100)
                    value = st.slider(
                        label, min_val, max_val, default or min_val,
                        key=key, help=help_text
                    )
                    
                elif option_type == 'checkbox':
                    value = st.checkbox(
                        label, value=default or False,
                        key=key, help=help_text
                    )
                    
                elif option_type == 'number':
                    min_val = option_props.get('min')
                    max_val = option_props.get('max')
                    value = st.number_input(
                        label, min_value=min_val, max_value=max_val, value=default or 0,
                        key=key, help=help_text
                    )
                    
                else:  # text
                    value = st.text_input(
                        label, value=default or "",
                        key=key, help=help_text
                    )
                
                # Validate required fields
                if required and (value is None or value == "" or (isinstance(value, list) and not value)):
                    st.error(f"{label} is required")
                    continue
                
                user_config[option_name] = value
                
            except Exception as e:
                self.logger.error(f"Error creating UI for option {option_name}: {str(e)}")
                st.error(f"Error configuring {label}: {str(e)}")
        
        return user_config
    
    def safe_render(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
                   config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Safely render the visualization with error handling and performance tracking.
        
        Args:
            data: Dataset used for training/testing
            model_results: List of model result dictionaries
            config: Configuration dictionary for the visualization
            **kwargs: Additional keyword arguments
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate data first
            is_valid, errors = self.validate_data(data, model_results)
            if not is_valid:
                st.error("❌ **Validation Error**")
                for error in errors:
                    st.error(f"• {error}")
                return False
            
            # Call the actual render method
            success = self.render(data, model_results, config, **kwargs)
            
            # Update performance tracking
            self._render_count += 1
            self._last_render_time = time.time() - start_time
            
            if success:
                self.logger.info(f"Successfully rendered {self.name} in {self._last_render_time:.2f}s")
            else:
                self.logger.warning(f"Render method returned False for {self.name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rendering {self.name}: {str(e)}")
            st.error(f"❌ **Error rendering {self.name}**")
            st.error(f"Error details: {str(e)}")
            
            # Show debug info in development
            if st.session_state.get('debug_mode', False):
                st.exception(e)
            
            return False
    
    def export_visualization(self, format: str = "png") -> Optional[bytes]:
        """
        Export the current visualization in the specified format.
        
        Args:
            format: Export format (png, pdf, html, etc.)
            
        Returns:
            Bytes data of the exported visualization or None if not supported
        """
        if format not in self.export_formats:
            raise VisualizationError(f"Export format '{format}' not supported. Supported formats: {self.export_formats}")
        
        # This is a base implementation - subclasses should override for specific export logic
        self.logger.warning(f"Export functionality not implemented for {self.name}")
        return None
    
    def get_sample_data(self) -> Optional[pd.DataFrame]:
        """
        Get sample data that can be used to demonstrate this visualization.
        
        Returns:
            Sample DataFrame or None if no sample data available
        """
        # Override in subclasses to provide sample data
        return None
    
    def get_documentation(self) -> Dict[str, str]:
        """
        Get documentation for this visualization plugin.
        
        Returns:
            Dictionary containing documentation sections
        """
        return {
            'overview': self.description,
            'usage': f"Use {self.name} to visualize {self.category.value} data.",
            'requirements': f"Requires: {', '.join(self.required_columns) if self.required_columns else 'No specific requirements'}",
            'supported_data_types': f"Supports: {', '.join([dt.value for dt in self.supported_data_types])}",
            'configuration': "Use the configuration panel to customize the visualization.",
            'examples': "No examples available."
        }
    
    def __str__(self) -> str:
        """String representation of the plugin"""
        return f"{self.name} v{self.version} ({self.category.value})"
    
    def __repr__(self) -> str:
        """Developer representation of the plugin"""
        return f"<{self.__class__.__name__}: {self.name}>"