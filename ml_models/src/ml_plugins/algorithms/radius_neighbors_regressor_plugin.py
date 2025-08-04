"""
Radius Neighbors Regressor Plugin - Main Integration
=================================================

This is the main plugin file that integrates all components of the Radius Neighbors Regressor
algorithm for the ML framework. It provides a unified interface for training, analysis,
and visualization.

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import json
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

# Import the three core components
try:
    from src.ml_plugins.algorithms.radius_neighbors_regressor.radius_neighbors_core import RadiusNeighborsCore
    from src.ml_plugins.algorithms.radius_neighbors_regressor.radius_neighbors_analysis import RadiusNeighborsAnalysis
    from src.ml_plugins.algorithms.radius_neighbors_regressor.radius_neighbors_display import RadiusNeighborsDisplay
except ImportError as e:
    # Fallback for different import structures
    try:
        from radius_neighbors_regressor.radius_neighbors_core import RadiusNeighborsCore
        from radius_neighbors_regressor.radius_neighbors_analysis import RadiusNeighborsAnalysis
        from radius_neighbors_regressor.radius_neighbors_display import RadiusNeighborsDisplay
    except ImportError:
        raise ImportError(f"Could not import Radius Neighbors components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadiusNeighborsRegressorPlugin(MLPlugin):
    """
    Unified Radius Neighbors Regressor Plugin.
    
    This class provides a complete interface for the Radius Neighbors Regressor algorithm,
    integrating training, analysis, and visualization capabilities.
    """
    
    def __init__(self, 
                 radius: float = 1.0,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 metric: str = 'euclidean',
                 p: int = 2,
                 auto_scale: bool = True,
                 adaptive_radius: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize the Radius Neighbors Regressor Plugin.
        
        Parameters:
        -----------
        radius : float, default=1.0
            The radius of neighborhood
        weights : str, default='uniform'
            Weight function for predictions ('uniform' or 'distance')
        algorithm : str, default='auto'
            Algorithm for neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute')
        metric : str, default='euclidean'
            Distance metric ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
        p : int, default=2
            Parameter for Minkowski metric
        auto_scale : bool, default=True
            Whether to automatically scale features
        adaptive_radius : bool, default=False
            Whether to use adaptive radius based on data density
        random_state : int, optional
            Random state for reproducibility
        """
        
        # Initialize base MLPlugin
        super().__init__()
        
        # Initialize core component
        self.core = RadiusNeighborsCore(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p,
            auto_scale=auto_scale,
            adaptive_radius=adaptive_radius,
            random_state=random_state
        )
        
        # Analysis and display components (initialized after training)
        self.analysis = None
        self.display = None
        
        # Plugin metadata for MLPlugin interface
        self._plugin_info = {
            'name': 'radius_neighbors_regressor',
            'display_name': 'Radius Neighbors Regressor',
            'version': '1.0.0',
            'algorithm_type': 'Instance-based Learning',
            'task_type': 'Regression',
            'author': 'Bachelor Thesis Project',
            'description': 'Advanced radius-based nearest neighbors regression with comprehensive analysis'
        }
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 3
        
        # Training history
        self.training_history = []
        self.is_trained = False
        
        logger.info("✅ Radius Neighbors Regressor Plugin initialized successfully")
    
    # ==================== REQUIRED ABSTRACT METHODS ====================
    
    def get_name(self) -> str:
        """Get the algorithm name (Required by MLPlugin)."""
        return "Radius Neighbors Regressor"
    
    def get_description(self) -> str:
        """Get the algorithm description (Required by MLPlugin)."""
        return ("A regression algorithm that makes predictions based on neighbors "
                "within a specified radius, using local density information for "
                "adaptive and robust predictions.")
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]) -> 'RadiusNeighborsRegressorPlugin':
        """Create model instance with given hyperparameters (Required by MLPlugin)."""
        return RadiusNeighborsRegressorPlugin(
            radius=hyperparameters.get("radius", self.core.radius),
            weights=hyperparameters.get("weights", self.core.weights),
            algorithm=hyperparameters.get("algorithm", self.core.algorithm),
            metric=hyperparameters.get("metric", self.core.metric),
            p=hyperparameters.get("p", self.core.p),
            auto_scale=hyperparameters.get("auto_scale", self.core.auto_scale),
            adaptive_radius=hyperparameters.get("adaptive_radius", self.core.adaptive_radius),
            random_state=hyperparameters.get("random_state", self.core.random_state)
        )
    
    def get_hyperparameter_config(self, key_prefix: str = "") -> Dict[str, Any]:
        """Get hyperparameter configuration for UI (Required by MLPlugin)."""
        try:
            import streamlit as st
            
            st.sidebar.subheader(f"{self.get_name()} Configuration")
            
            # Create tabs for different configuration sections
            basic_tab, analysis_tab, viz_tab, report_tab, advanced_tab = st.sidebar.tabs([
                "📊 Basic", "🔍 Analysis", "📈 Visualization", "📋 Reports", "⚙️ Advanced"
            ])
            
            config = {}
            
            # ==================== BASIC PARAMETERS TAB ====================
            with basic_tab:
                st.markdown("**Core Algorithm Parameters**")
                
                # Radius parameter
                config["radius"] = st.number_input(
                    "Radius:", 
                    value=float(self.core.radius), 
                    min_value=0.1, 
                    max_value=10.0, 
                    step=0.1,
                    help="Range of parameter space to use for neighbor search",
                    key=f"{key_prefix}_radius"
                )
                
                # Weights parameter
                config["weights"] = st.selectbox(
                    "Weights:",
                    options=['uniform', 'distance'],
                    index=0 if self.core.weights == 'uniform' else 1,
                    help="Weight function for predictions",
                    key=f"{key_prefix}_weights"
                )
                
                # Algorithm parameter
                config["algorithm"] = st.selectbox(
                    "Algorithm:",
                    options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                    index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(self.core.algorithm),
                    help="Algorithm for neighbor search",
                    key=f"{key_prefix}_algorithm"
                )
                
                # Metric parameter
                config["metric"] = st.selectbox(
                    "Distance Metric:",
                    options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                    index=['euclidean', 'manhattan', 'chebyshev', 'minkowski'].index(self.core.metric),
                    help="Distance metric for neighbor search",
                    key=f"{key_prefix}_metric"
                )
                
                # P parameter (only for minkowski)
                if config["metric"] == 'minkowski':
                    config["p"] = st.number_input(
                        "Minkowski P:",
                        value=self.core.p,
                        min_value=1,
                        max_value=10,
                        step=1,
                        help="Parameter for Minkowski metric",
                        key=f"{key_prefix}_p"
                    )
                else:
                    config["p"] = 2
                
                # Auto scale
                config["auto_scale"] = st.checkbox(
                    "Auto Scale Features",
                    value=self.core.auto_scale,
                    help="Automatically scale features",
                    key=f"{key_prefix}_auto_scale"
                )
                
                # Adaptive radius
                config["adaptive_radius"] = st.checkbox(
                    "Adaptive Radius",
                    value=self.core.adaptive_radius,
                    help="Use adaptive radius based on data density",
                    key=f"{key_prefix}_adaptive_radius"
                )
            
            # ==================== ANALYSIS SETTINGS TAB ====================
            with analysis_tab:
                st.markdown("**Analysis Configuration**")
                
                # Comprehensive analysis toggle
                config["enable_comprehensive_analysis"] = st.checkbox(
                    "Enable Comprehensive Analysis",
                    value=True,
                    help="Perform full analysis including feature importance and comparisons",
                    key=f"{key_prefix}_comprehensive"
                )
                
                # Cross-validation settings
                st.markdown("**Cross-Validation Settings**")
                config["cv_folds"] = st.slider(
                    "CV Folds:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Number of cross-validation folds",
                    key=f"{key_prefix}_cv_folds"
                )
                
                config["cv_scoring"] = st.selectbox(
                    "CV Scoring:",
                    options=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                    index=0,
                    help="Scoring metric for cross-validation",
                    key=f"{key_prefix}_cv_scoring"
                )
                
                # Feature importance settings
                st.markdown("**Feature Importance Settings**")
                config["feature_importance_method"] = st.selectbox(
                    "Importance Method:",
                    options=['permutation', 'drop_column'],
                    index=0,
                    help="Method for calculating feature importance",
                    key=f"{key_prefix}_fi_method"
                )
                
                config["fi_n_repeats"] = st.slider(
                    "Permutation Repeats:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Number of permutation repeats for feature importance",
                    key=f"{key_prefix}_fi_repeats"
                )
                
                # Algorithm comparison settings
                st.markdown("**Algorithm Comparison**")
                config["compare_with_knn"] = st.checkbox(
                    "Compare with KNN",
                    value=True,
                    help="Compare performance with K-Nearest Neighbors",
                    key=f"{key_prefix}_compare_knn"
                )
                
                config["compare_with_global"] = st.checkbox(
                    "Compare with Global Methods",
                    value=True,
                    help="Compare with linear/tree-based methods",
                    key=f"{key_prefix}_compare_global"
                )
                
                config["knn_neighbors_range"] = st.slider(
                    "KNN K Range:",
                    min_value=3,
                    max_value=20,
                    value=(3, 15),
                    help="Range of K values for KNN comparison",
                    key=f"{key_prefix}_knn_range"
                )
            
            # ==================== VISUALIZATION TAB ====================
            with viz_tab:
                st.markdown("**Visualization Settings**")
                
                # Plot settings
                config["enable_interactive_plots"] = st.checkbox(
                    "Interactive Plots",
                    value=True,
                    help="Generate interactive Plotly visualizations",
                    key=f"{key_prefix}_interactive"
                )
                
                config["plot_theme"] = st.selectbox(
                    "Plot Theme:",
                    options=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
                    index=0,
                    help="Theme for visualizations",
                    key=f"{key_prefix}_theme"
                )
                
                config["plot_width"] = st.slider(
                    "Plot Width:",
                    min_value=400,
                    max_value=1200,
                    value=800,
                    step=50,
                    help="Width of plots in pixels",
                    key=f"{key_prefix}_width"
                )
                
                config["plot_height"] = st.slider(
                    "Plot Height:",
                    min_value=300,
                    max_value=800,
                    value=500,
                    step=50,
                    help="Height of plots in pixels",
                    key=f"{key_prefix}_height"
                )
                
                # Dashboard settings
                st.markdown("**Dashboard Configuration**")
                config["dashboard_layout"] = st.selectbox(
                    "Dashboard Layout:",
                    options=['compact', 'detailed', 'presentation'],
                    index=1,
                    help="Layout style for performance dashboard",
                    key=f"{key_prefix}_layout"
                )
                
                config["show_confidence_intervals"] = st.checkbox(
                    "Show Confidence Intervals",
                    value=True,
                    help="Display confidence intervals in plots",
                    key=f"{key_prefix}_confidence"
                )
                
                # Plot selection
                st.markdown("**Plots to Generate**")
                config["generate_radius_coverage"] = st.checkbox(
                    "Radius Coverage Analysis",
                    value=True,
                    key=f"{key_prefix}_plot_radius"
                )
                
                config["generate_performance_comparison"] = st.checkbox(
                    "Performance Comparison",
                    value=True,
                    key=f"{key_prefix}_plot_performance"
                )
                
                config["generate_feature_importance"] = st.checkbox(
                    "Feature Importance Plot",
                    value=True,
                    key=f"{key_prefix}_plot_features"
                )
                
                config["generate_interactive_explorer"] = st.checkbox(
                    "Interactive Model Explorer",
                    value=False,
                    help="Generate interactive exploration interface",
                    key=f"{key_prefix}_explorer"
                )
            
            # ==================== REPORTING TAB ====================
            with report_tab:
                st.markdown("**Report Generation Settings**")
                
                # Report format
                config["report_format"] = st.selectbox(
                    "Report Format:",
                    options=['html', 'pdf', 'json', 'markdown'],
                    index=0,
                    help="Output format for analysis report",
                    key=f"{key_prefix}_report_format"
                )
                
                config["report_detail_level"] = st.selectbox(
                    "Detail Level:",
                    options=['summary', 'detailed', 'comprehensive'],
                    index=1,
                    help="Level of detail in generated reports",
                    key=f"{key_prefix}_detail_level"
                )
                
                # Report sections
                st.markdown("**Report Sections to Include**")
                config["include_model_summary"] = st.checkbox(
                    "Model Summary",
                    value=True,
                    key=f"{key_prefix}_inc_summary"
                )
                
                config["include_performance_metrics"] = st.checkbox(
                    "Performance Metrics",
                    value=True,
                    key=f"{key_prefix}_inc_metrics"
                )
                
                config["include_feature_analysis"] = st.checkbox(
                    "Feature Analysis",
                    value=True,
                    key=f"{key_prefix}_inc_features"
                )
                
                config["include_comparisons"] = st.checkbox(
                    "Algorithm Comparisons",
                    value=True,
                    key=f"{key_prefix}_inc_comparisons"
                )
                
                config["include_visualizations"] = st.checkbox(
                    "Visualizations",
                    value=True,
                    key=f"{key_prefix}_inc_viz"
                )
                
                config["include_recommendations"] = st.checkbox(
                    "Parameter Recommendations",
                    value=True,
                    key=f"{key_prefix}_inc_recommendations"
                )
                
                # Export settings
                st.markdown("**Export Options**")
                config["auto_save_plots"] = st.checkbox(
                    "Auto-Save Plots",
                    value=False,
                    help="Automatically save plots to files",
                    key=f"{key_prefix}_auto_save"
                )
                
                if config["auto_save_plots"]:
                    config["plot_save_format"] = st.selectbox(
                        "Plot Save Format:",
                        options=['png', 'pdf', 'svg', 'html'],
                        index=0,
                        key=f"{key_prefix}_plot_format"
                    )
            
            # ==================== ADVANCED OPTIONS TAB ====================
            with advanced_tab:
                st.markdown("**Advanced Configuration**")
                
                # Random state
                config["random_state"] = st.number_input(
                    "Random State:",
                    value=self.core.random_state if self.core.random_state is not None else 42,
                    min_value=0,
                    max_value=9999,
                    step=1,
                    help="Random state for reproducibility",
                    key=f"{key_prefix}_random_state"
                )
                
                # Performance optimization
                st.markdown("**Performance Optimization**")
                config["enable_parallel_processing"] = st.checkbox(
                    "Enable Parallel Processing",
                    value=True,
                    help="Use multiple cores for analysis",
                    key=f"{key_prefix}_parallel"
                )
                
                config["n_jobs"] = st.slider(
                    "Number of Jobs:",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of parallel jobs (-1 for all cores)",
                    key=f"{key_prefix}_n_jobs"
                )
                
                config["memory_limit_gb"] = st.slider(
                    "Memory Limit (GB):",
                    min_value=1,
                    max_value=16,
                    value=8,
                    help="Maximum memory usage for analysis",
                    key=f"{key_prefix}_memory"
                )
                
                # Analysis depth
                st.markdown("**Analysis Depth**")
                config["radius_optimization_steps"] = st.slider(
                    "Radius Optimization Steps:",
                    min_value=10,
                    max_value=100,
                    value=20,
                    help="Number of radius values to test",
                    key=f"{key_prefix}_radius_steps"
                )
                
                config["metric_comparison_depth"] = st.selectbox(
                    "Metric Comparison Depth:",
                    options=['basic', 'extended', 'comprehensive'],
                    index=1,
                    help="Depth of distance metric comparison",
                    key=f"{key_prefix}_metric_depth"
                )
                
                # Debugging options
                st.markdown("**Debugging & Logging**")
                config["enable_verbose_logging"] = st.checkbox(
                    "Verbose Logging",
                    value=False,
                    help="Enable detailed logging for debugging",
                    key=f"{key_prefix}_verbose"
                )
                
                config["save_intermediate_results"] = st.checkbox(
                    "Save Intermediate Results",
                    value=False,
                    help="Save intermediate analysis results",
                    key=f"{key_prefix}_intermediate"
                )
                
                # Expert options
                st.markdown("**Expert Options**")
                config["custom_radius_range"] = st.text_input(
                    "Custom Radius Range:",
                    value="",
                    help="Custom radius range (e.g., '0.1,0.5,1.0,2.0')",
                    key=f"{key_prefix}_custom_radius"
                )
                
                config["custom_metrics"] = st.text_input(
                    "Custom Metrics:",
                    value="",
                    help="Additional metrics to test (comma-separated)",
                    key=f"{key_prefix}_custom_metrics"
                )
            
            # Convert random_state to None if using default
            if config["random_state"] == 42:
                config["random_state"] = None
            
            return config
            
        except ImportError:
            # Fallback when streamlit is not available
            return {
                "radius": self.core.radius,
                "weights": self.core.weights,
                "algorithm": self.core.algorithm,
                "metric": self.core.metric,
                "p": self.core.p,
                "auto_scale": self.core.auto_scale,
                "adaptive_radius": self.core.adaptive_radius,
                "random_state": self.core.random_state,
                # Basic defaults for advanced options
                "enable_comprehensive_analysis": True,
                "cv_folds": 5,
                "enable_interactive_plots": True,
                "report_format": "html"
            }

    # ==================== OPTIONAL MLPLUGIN METHODS ====================
    
    def get_category(self) -> str:
        """Get the algorithm category."""
        return "Instance-based Learning"
    
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
        """Optional data preprocessing."""
        # The core component handles preprocessing internally
        return X, y
    
    def is_compatible_with_data(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """Check if algorithm is compatible with the data."""
        try:
            if target_column not in df.columns:
                return False, f"Target column '{target_column}' not found in data"
            
            # Get features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Check if target is numeric (regression)
            if not pd.api.types.is_numeric_dtype(y):
                return False, "Target must be numeric for regression"
            
            # Check minimum samples
            if len(df) < self._min_samples_required:
                return False, f"Requires at least {self._min_samples_required} samples, got {len(df)}"
            
            # Check for missing values in target
            if y.isnull().any():
                return False, "Target column contains missing values"
            
            # Check features
            if X.shape[1] == 0:
                return False, "No feature columns found"
            
            # Check for too many missing values in features
            missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
            if missing_ratio > 0.5:
                return False, f"Too many missing values in features ({missing_ratio:.1%})"
            
            return True, f"Compatible with {len(df)} samples and {X.shape[1]} features"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    # ==================== EXISTING MLPLUGIN INTERFACE ====================
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information for the ML framework (Required by MLPlugin)."""
        return {
            'name': 'radius_neighbors_regressor',
            'display_name': 'Radius Neighbors Regressor',
            'version': '1.0.0',
            'algorithm_type': 'Instance-based Learning',
            'task_type': 'Regression',
            'description': 'Advanced radius-based nearest neighbors regression with comprehensive analysis',
            'parameters': {
                'radius': {'type': 'float', 'default': 1.0, 'description': 'Neighborhood radius'},
                'weights': {'type': 'str', 'default': 'uniform', 'choices': ['uniform', 'distance']},
                'metric': {'type': 'str', 'default': 'euclidean', 'choices': ['euclidean', 'manhattan', 'chebyshev']},
                'auto_scale': {'type': 'bool', 'default': True, 'description': 'Automatic feature scaling'},
                'adaptive_radius': {'type': 'bool', 'default': False, 'description': 'Adaptive radius based on density'}
            },
            'capabilities': {
                'training': True,
                'prediction': True,
                'analysis': True,
                'visualization': True,
                'reporting': True,
                'parameter_optimization': True,
                'comparison': True
            },
            'requirements': {
                'min_samples': 3,
                'handles_missing_values': False,
                'requires_scaling': False,  # Optional but recommended
                'memory_intensive': True
            }
        }
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """Train the model (Required by MLPlugin)."""
        return self.fit(X, y)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Evaluate the model (Required by MLPlugin)."""
        try:
            if not self.is_trained:
                return {'error': 'Model must be trained before evaluation'}
            
            # Get predictions
            predictions = self.predict(X)
            if isinstance(predictions, dict) and 'error' in predictions:
                return predictions
            
            # Calculate metrics
            y_array = self._convert_to_array(y).flatten()
            score = self.score(X, y)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mse = mean_squared_error(y_array, predictions)
            mae = mean_absolute_error(y_array, predictions)
            
            return {
                'r2_score': float(score),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse)),
                'n_samples': len(y_array)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary (Required by MLPlugin)."""
        return self._create_model_summary()
    
    # ==================== MAIN PLUGIN INTERFACE ====================
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Train the Radius Neighbors Regressor model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metadata
        """
        try:
            logger.info("🚀 Starting Radius Neighbors Regressor training...")
            
            # Convert inputs if necessary
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y).flatten()
            
            # Validate inputs
            validation_result = self._validate_inputs(X_array, y_array)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Train core model
            training_result = self.core.train_model(X_array, y_array)
            
            if training_result.get('model_fitted', False):
                # Initialize analysis and display components
                self.analysis = RadiusNeighborsAnalysis(self.core)
                self.display = RadiusNeighborsDisplay(self.core, self.analysis)
                
                self.is_trained = True
                
                # Store training history
                training_record = {
                    'timestamp': pd.Timestamp.now(),
                    'data_shape': X_array.shape,
                    'training_result': training_result,
                    'parameters': self.get_params()
                }
                self.training_history.append(training_record)
                
                logger.info("✅ Training completed successfully")
                
                return {
                    'success': True,
                    'training_result': training_result,
                    'model_summary': self._create_model_summary(),
                    'quick_analysis': self._get_quick_analysis()
                }
            else:
                return {
                    'success': False,
                    'error': training_result.get('error', 'Training failed')
                }
                
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        np.ndarray or Dict[str, Any]
            Predictions or error information
        """
        try:
            if not self.is_trained:
                return {'error': 'Model must be trained before making predictions'}
            
            X_array = self._convert_to_array(X)
            return self.core.predict(X_array)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            return {'error': str(e)}
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate R² score on test data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test features
        y : array-like of shape (n_samples,)
            Test targets
            
        Returns:
        --------
        float
            R² score
        """
        try:
            if not self.is_trained:
                logger.warning("⚠️ Model not trained, returning 0 score")
                return 0.0
            
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y).flatten()
            
            return self.core.score(X_array, y_array)
            
        except Exception as e:
            logger.error(f"❌ Scoring failed: {str(e)}")
            return 0.0
    
    # ==================== ANALYSIS INTERFACE ====================
    
    def analyze_performance(self, comprehensive: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.
        
        Parameters:
        -----------
        comprehensive : bool, default=True
            Whether to perform full comprehensive analysis
            
        Returns:
        --------
        Dict[str, Any]
            Complete analysis results
        """
        if not self.is_trained:
            return {'error': 'Model must be trained before analysis'}
        
        try:
            logger.info("🔍 Starting performance analysis...")
            
            analysis_results = {}
            
            if comprehensive:
                # Full comprehensive analysis
                analysis_results.update({
                    'radius_behavior': self.analysis.analyze_radius_behavior(),
                    'cross_validation': self.analysis.analyze_cross_validation(),
                    'feature_importance': self.analysis.analyze_feature_importance(),
                    'performance_profile': self.analysis.profile_performance(),
                    'knn_comparison': self.analysis.compare_with_knn(),
                    'global_comparison': self.analysis.compare_with_global_methods(),
                    'metric_comparison': self.analysis.analyze_metric_comparison()
                })
            else:
                # Quick analysis
                analysis_results.update({
                    'cross_validation': self.analysis.analyze_cross_validation(),
                    'performance_profile': self.analysis.profile_performance()
                })
            
            logger.info("✅ Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance analysis."""
        if not self.is_trained:
            return {'error': 'Model must be trained before feature importance analysis'}
        
        try:
            return self.analysis.analyze_feature_importance()
        except Exception as e:
            return {'error': str(e)}
    
    def compare_algorithms(self) -> Dict[str, Any]:
        """Compare with other algorithms."""
        if not self.is_trained:
            return {'error': 'Model must be trained before comparison'}
        
        try:
            return {
                'knn_comparison': self.analysis.compare_with_knn(),
                'global_comparison': self.analysis.compare_with_global_methods(),
                'metric_comparison': self.analysis.analyze_metric_comparison()
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== VISUALIZATION INTERFACE ====================
    
    def create_dashboard(self) -> Dict[str, Any]:
        """Create performance dashboard."""
        if not self.is_trained:
            return {'error': 'Model must be trained before creating dashboard'}
        
        try:
            logger.info("📊 Creating performance dashboard...")
            dashboard = self.display.create_performance_dashboard()
            logger.info("✅ Dashboard created successfully")
            return dashboard
        except Exception as e:
            logger.error(f"❌ Dashboard creation failed: {str(e)}")
            return {'error': str(e)}
    
    def plot_radius_coverage(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot radius coverage analysis."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_radius_coverage(interactive=interactive, save_path=save_path)
        except Exception as e:
            return {'error': str(e)}
    
    def plot_performance_comparison(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot performance comparison."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_performance_comparison(interactive=interactive, save_path=save_path)
        except Exception as e:
            return {'error': str(e)}
    
    def plot_feature_importance(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot feature importance."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_feature_importance(interactive=interactive, save_path=save_path)
        except Exception as e:
            return {'error': str(e)}
    
    def create_interactive_explorer(self) -> Dict[str, Any]:
        """Create interactive model explorer."""
        if not self.is_trained:
            return {'error': 'Model must be trained before creating explorer'}
        
        try:
            logger.info("🎯 Creating interactive explorer...")
            explorer = self.display.create_interactive_explorer()
            logger.info("✅ Interactive explorer created successfully")
            return explorer
        except Exception as e:
            logger.error(f"❌ Explorer creation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== REPORTING INTERFACE ====================
    
    def generate_report(self, 
                       format_type: str = 'html',
                       comprehensive: bool = True,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Parameters:
        -----------
        format_type : str, default='html'
            Report format ('html', 'json', 'text')
        comprehensive : bool, default=True
            Whether to include comprehensive analysis
        save_path : str, optional
            Path to save the report
            
        Returns:
        --------
        Dict[str, Any]
            Report data and metadata
        """
        if not self.is_trained:
            return {'error': 'Model must be trained before generating report'}
        
        try:
            logger.info(f"📋 Generating {format_type.upper()} report...")
            
            # Perform analysis if not already done
            if comprehensive:
                analyses = self.analyze_performance(comprehensive=True)
            else:
                analyses = self.analyze_performance(comprehensive=False)
            
            # Generate report
            report = self.display.generate_comprehensive_report(
                format_type=format_type,
                save_path=save_path,
                analyses=analyses
            )
            
            logger.info("✅ Report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def export_model_data(self, save_path: str) -> Dict[str, Any]:
        """Export model data and results."""
        if not self.is_trained:
            return {'error': 'Model must be trained before export'}
        
        try:
            export_data = {
                'plugin_info': self._plugin_info,
                'model_parameters': self.get_params(),
                'training_history': self._serialize_training_history(),
                'model_summary': self._create_model_summary(),
                'timestamp': str(pd.Timestamp.now())
            }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"✅ Model data exported to {save_path}")
            return {'success': True, 'export_path': save_path}
            
        except Exception as e:
            logger.error(f"❌ Export failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== PARAMETER MANAGEMENT ====================
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'radius': self.core.radius,
            'weights': self.core.weights,
            'algorithm': self.core.algorithm,
            'metric': self.core.metric,
            'p': self.core.p,
            'auto_scale': self.core.auto_scale,
            'adaptive_radius': self.core.adaptive_radius,
            'effective_radius': getattr(self.core, 'effective_radius_', None)
        }
    
    def set_params(self, **params) -> 'RadiusNeighborsRegressorPlugin':
        """Set model parameters."""
        for param, value in params.items():
            if hasattr(self.core, param):
                setattr(self.core, param, value)
            else:
                logger.warning(f"⚠️ Unknown parameter: {param}")
        
        # If model was trained, it needs retraining with new parameters
        if self.is_trained:
            logger.info("🔄 Parameters changed - model needs retraining")
            self.is_trained = False
            self.analysis = None
            self.display = None
        
        return self
    
    def get_parameter_recommendations(self) -> Dict[str, Any]:
        """Get parameter optimization recommendations."""
        if not self.is_trained:
            return {'error': 'Model must be trained before parameter recommendations'}
        
        try:
            # Get recommendations from analysis
            analyses = self.analyze_performance(comprehensive=False)
            
            recommendations = []
            
            # Radius recommendations
            radius_analysis = analyses.get('radius_behavior', {})
            if 'optimal_radius' in radius_analysis:
                optimal = radius_analysis['optimal_radius']
                current = optimal.get('current_radius', self.core.effective_radius_)
                optimal_value = optimal.get('optimal_radius', current)
                improvement = optimal.get('improvement_potential', 0)
                
                if improvement > 0.05:  # 5% improvement threshold
                    recommendations.append({
                        'parameter': 'radius',
                        'current_value': current,
                        'recommended_value': optimal_value,
                        'improvement_potential': improvement,
                        'reason': 'Better radius value found through optimization'
                    })
            
            # Metric recommendations
            metric_analysis = analyses.get('metric_comparison', {})
            if 'best_metric_analysis' in metric_analysis:
                best_metric = metric_analysis['best_metric_analysis']
                if best_metric.get('improvement_over_current', 0) > 0.02:  # 2% improvement
                    recommendations.append({
                        'parameter': 'metric',
                        'current_value': self.core.metric,
                        'recommended_value': best_metric.get('best_metric', self.core.metric),
                        'improvement_potential': best_metric.get('improvement_over_current', 0),
                        'reason': 'Better distance metric identified'
                    })
            
            return {
                'recommendations': recommendations,
                'analysis_source': 'comprehensive_analysis'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== MODEL INFORMATION ====================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'plugin_info': self._plugin_info,
            'is_trained': self.is_trained,
            'parameters': self.get_params(),
            'training_history_count': len(self.training_history)
        }
        
        if self.is_trained:
            info.update({
                'model_summary': self._create_model_summary(),
                'data_info': {
                    'n_samples': self.core.n_samples_in_,
                    'n_features': self.core.n_features_in_,
                    'feature_names': list(self.core.feature_names_) if hasattr(self.core, 'feature_names_') else None
                }
            })
        
        return info
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self._serialize_training_history()
    
    # ==================== UTILITY METHODS ====================
    
    def _convert_to_array(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert various data formats to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate training inputs."""
        try:
            # Check shapes
            if X.ndim != 2:
                return {'valid': False, 'error': 'X must be 2-dimensional'}
            
            if y.ndim != 1:
                return {'valid': False, 'error': 'y must be 1-dimensional'}
            
            if X.shape[0] != y.shape[0]:
                return {'valid': False, 'error': 'X and y must have same number of samples'}
            
            # Check for missing values
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                return {'valid': False, 'error': 'Input data contains NaN values'}
            
            # Check for infinite values
            if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                return {'valid': False, 'error': 'Input data contains infinite values'}
            
            # Check minimum sample size
            if X.shape[0] < 3:
                return {'valid': False, 'error': 'Need at least 3 samples for training'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _create_model_summary(self) -> Dict[str, Any]:
        """Create model summary."""
        if not self.is_trained:
            return {}
        
        return {
            'algorithm': 'Radius Neighbors Regressor',
            'n_samples': self.core.n_samples_in_,
            'n_features': self.core.n_features_in_,
            'effective_radius': self.core.effective_radius_,
            'scaling_applied': self.core.scaler_ is not None,
            'adaptive_radius_enabled': self.core.adaptive_radius,
            'distance_metric': self.core.metric,
            'weight_function': self.core.weights
        }
    
    def _get_quick_analysis(self) -> Dict[str, Any]:
        """Get quick analysis after training."""
        try:
            if not self.analysis:
                return {}
            
            # Quick cross-validation
            cv_result = self.analysis.analyze_cross_validation(cv_folds=3)
            
            return {
                'quick_cv_score': cv_result.get('mean_cv_score', 0),
                'cv_stability': cv_result.get('stability_metrics', {}).get('consistency_score', 0),
                'training_successful': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _serialize_training_history(self) -> List[Dict[str, Any]]:
        """Serialize training history for JSON export."""
        serialized = []
        for record in self.training_history:
            serialized_record = {
                'timestamp': str(record['timestamp']),
                'data_shape': record['data_shape'],
                'parameters': record['parameters']
            }
            # Add training result if serializable
            if 'training_result' in record:
                try:
                    # Attempt to serialize, ensure it's basic types
                    json.dumps(record['training_result']) 
                    serialized_record['training_result'] = record['training_result']
                except TypeError:
                    serialized_record['training_result'] = 'Complex object - Not directly serializable'
            
            serialized.append(serialized_record)
        
        return serialized

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Radius Neighbors Regressor model.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics but kept for API consistency.
        y_pred : np.ndarray, optional
            Predicted target values. Used to calculate percentage of predictions with no neighbors if applicable.
        y_proba : np.ndarray, optional
            Not applicable for regressors.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_trained:
            return {"error": "Model not fitted. Cannot retrieve Radius Neighbors Regressor specific metrics."}

        metrics = {}
        prefix = "radneighreg_"

        # Core model parameters
        if hasattr(self.core, 'effective_radius_') and self.core.effective_radius_ is not None:
            metrics[f"{prefix}effective_radius"] = self.core.effective_radius_
        else:
            metrics[f"{prefix}configured_radius"] = self.core.radius
        
        metrics[f"{prefix}adaptive_radius_enabled"] = self.core.adaptive_radius
        metrics[f"{prefix}scaling_applied"] = self.core.scaler_ is not None
        metrics[f"{prefix}distance_metric_used"] = self.core.metric
        metrics[f"{prefix}weight_function_used"] = self.core.weights

        # Metrics from self.analysis component
        if self.analysis:
            try:
                # Assuming analyze_radius_behavior returns a dictionary with relevant stats
                # This might need to be called or its results retrieved if cached
                radius_behavior_stats = self.analysis.analyze_radius_behavior() 
                
                if isinstance(radius_behavior_stats, dict):
                    metrics[f"{prefix}mean_neighbors_analysis"] = radius_behavior_stats.get('mean_neighbors_found')
                    metrics[f"{prefix}empty_neighborhoods_analysis_pct"] = radius_behavior_stats.get('percentage_empty_neighborhoods')
                    
                    std_dev_neighbors = radius_behavior_stats.get('std_dev_neighbors_found')
                    mean_neighbors = radius_behavior_stats.get('mean_neighbors_found')
                    if mean_neighbors is not None and std_dev_neighbors is not None and mean_neighbors > 1e-9:
                        cv_neighbors = std_dev_neighbors / mean_neighbors
                        metrics[f"{prefix}neighborhood_size_consistency_analysis"] = 1.0 / (1.0 + cv_neighbors)
                    elif mean_neighbors == 0 :
                         metrics[f"{prefix}neighborhood_size_consistency_analysis"] = 0.0


            except Exception as e:
                logger.warning(f"Could not retrieve radius behavior analysis metrics: {e}")
                metrics[f"{prefix}radius_behavior_analysis_error"] = str(e)

        # Metrics from predictions (if y_pred is provided)
        # This assumes that self.core.predict might return np.nan for points with no neighbors.
        # If it raises an error instead, this part won't be effective.
        if y_pred is not None:
            try:
                y_pred_arr = self._convert_to_array(y_pred).flatten()
                nan_predictions_count = np.sum(np.isnan(y_pred_arr))
                if len(y_pred_arr) > 0:
                    metrics[f"{prefix}predictions_with_no_neighbors_pct"] = (nan_predictions_count / len(y_pred_arr)) * 100
                    metrics[f"{prefix}predictions_with_no_neighbors_count"] = int(nan_predictions_count)
                else:
                    metrics[f"{prefix}predictions_with_no_neighbors_pct"] = 0.0
                    metrics[f"{prefix}predictions_with_no_neighbors_count"] = 0
            except Exception as e:
                logger.warning(f"Could not calculate no-neighbor prediction stats from y_pred: {e}")


        if not metrics:
            metrics['info'] = "No specific Radius Neighbors Regressor metrics were available or calculated."
            
        return metrics    
    
    # ==================== SPECIAL METHODS ====================
    
    def __repr__(self) -> str:
        """String representation of the plugin."""
        status = "Trained" if self.is_trained else "Not Trained"
        params = f"radius={self.core.radius}, metric={self.core.metric}"
        return f"RadiusNeighborsRegressorPlugin({params}) - {status}"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Radius Neighbors Regressor Plugin (Status: {'Trained' if self.is_trained else 'Not Trained'})"


# ==================== PLUGIN FACTORY FUNCTIONS ====================

def create_radius_neighbors_plugin(**kwargs) -> RadiusNeighborsRegressorPlugin:
    """
    Factory function to create RadiusNeighborsRegressorPlugin.
    
    Parameters:
    -----------
    **kwargs : dict
        Plugin parameters
        
    Returns:
    --------
    RadiusNeighborsRegressorPlugin
        Configured plugin instance
    """
    return RadiusNeighborsRegressorPlugin(**kwargs)


def get_plugin_info() -> Dict[str, Any]:
    """
    Get plugin information for the ML framework.
    
    Returns:
    --------
    Dict[str, Any]
        Plugin metadata
    """
    return {
        'name': 'radius_neighbors_regressor',
        'display_name': 'Radius Neighbors Regressor',
        'version': '1.0.0',
        'algorithm_type': 'Instance-based Learning',
        'task_type': 'Regression',
        'description': 'Advanced radius-based nearest neighbors regression with comprehensive analysis',
        'parameters': {
            'radius': {'type': 'float', 'default': 1.0, 'description': 'Neighborhood radius'},
            'weights': {'type': 'str', 'default': 'uniform', 'choices': ['uniform', 'distance']},
            'metric': {'type': 'str', 'default': 'euclidean', 'choices': ['euclidean', 'manhattan', 'chebyshev']},
            'auto_scale': {'type': 'bool', 'default': True, 'description': 'Automatic feature scaling'},
            'adaptive_radius': {'type': 'bool', 'default': False, 'description': 'Adaptive radius based on density'}
        },
        'capabilities': {
            'training': True,
            'prediction': True,
            'analysis': True,
            'visualization': True,
            'reporting': True,
            'parameter_optimization': True,
            'comparison': True
        },
        'requirements': {
            'min_samples': 3,
            'handles_missing_values': False,
            'requires_scaling': False,  # Optional but recommended
            'memory_intensive': True
        }
    }

def get_plugin() -> RadiusNeighborsRegressorPlugin:
    """Factory function to get plugin instance (Required by framework)."""
    return RadiusNeighborsRegressorPlugin()

# ==================== TESTING FUNCTIONS ====================

def test_plugin_integration():
    """Test the complete plugin integration."""
    print("🧪 Testing Radius Neighbors Regressor Plugin Integration...")
    
    try:
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 100, 4
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)
        
        # Test 1: Plugin creation
        print("✅ Test 1: Plugin creation")
        plugin = RadiusNeighborsRegressorPlugin(radius=1.5, auto_scale=True)
        assert isinstance(plugin, RadiusNeighborsRegressorPlugin)
        assert isinstance(plugin, MLPlugin)  # Check inheritance
        
        # Test 2: Plugin info
        print("✅ Test 2: Plugin info")
        plugin_info = plugin.get_plugin_info()
        assert plugin_info['name'] == 'radius_neighbors_regressor'
        
        # Test 3: Hyperparameters
        print("✅ Test 3: Plugin hyperparameters")
        hyperparams = plugin.get_hyperparameters()
        assert 'radius' in hyperparams
        
        # Test 4: Model training
        print("✅ Test 4: Model training")
        training_result = plugin.train(X, y)  # Use train method from MLPlugin
        assert training_result['success'] == True
        assert plugin.is_trained == True
        
        # Test 5: Predictions
        print("✅ Test 5: Predictions")
        predictions = plugin.predict(X[:10])
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        
        # Test 6: Evaluation
        print("✅ Test 6: Evaluation")
        evaluation = plugin.evaluate(X, y)
        assert 'r2_score' in evaluation
        assert isinstance(evaluation['r2_score'], float)
        
        # Test 7: Model summary
        print("✅ Test 7: Model summary")
        summary = plugin.get_model_summary()
        assert 'algorithm' in summary
        
        # Test 8: Scoring
        print("✅ Test 8: Scoring")
        score = plugin.score(X, y)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # Test 9: Analysis
        print("✅ Test 9: Analysis")
        analysis_result = plugin.analyze_performance(comprehensive=False)
        assert 'cross_validation' in analysis_result
        
        # Test 10: Parameter management
        print("✅ Test 10: Parameter management")
        params = plugin.get_params()
        assert 'radius' in params
        
        plugin.set_params(radius=2.0)
        new_params = plugin.get_params()
        assert new_params['radius'] == 2.0
        
        print("🎉 All plugin integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Plugin integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_edge_cases():
    """Test plugin edge cases."""
    print("🔍 Testing Plugin Edge Cases...")
    
    try:
        # Test 1: Operations before training
        print("✅ Test 1: Operations before training")
        plugin = RadiusNeighborsRegressorPlugin()
        
        prediction_result = plugin.predict(np.random.randn(5, 3))
        assert 'error' in prediction_result
        
        evaluation_result = plugin.evaluate(np.random.randn(5, 3), np.random.randn(5))
        assert 'error' in evaluation_result
        
        score = plugin.score(np.random.randn(5, 3), np.random.randn(5))
        assert score == 0.0
        
        analysis_result = plugin.analyze_performance()
        assert 'error' in analysis_result
        
        # Test 2: Invalid input validation
        print("✅ Test 2: Invalid input validation")
        X_invalid = np.array([[1, 2], [3, np.nan]])
        y_invalid = np.array([1, 2])
        
        training_result = plugin.train(X_invalid, y_invalid)
        assert training_result['success'] == False
        
        # Test 3: Mismatched dimensions
        print("✅ Test 3: Mismatched dimensions")
        X_mismatch = np.random.randn(10, 3)
        y_mismatch = np.random.randn(5)
        
        training_result = plugin.train(X_mismatch, y_mismatch)
        assert training_result['success'] == False
        
        print("🎉 All edge case tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run comprehensive tests
    print("🚀 Running Radius Neighbors Regressor Plugin Tests...")
    
    # Test main functionality
    main_test = test_plugin_integration()
    
    # Test edge cases
    edge_test = test_plugin_edge_cases()
    
    if main_test and edge_test:
        print("\n🎉 ALL TESTS PASSED! Plugin is ready for deployment!")
        print("\n📋 Plugin Summary:")
        print("=" * 50)
        plugin_info = get_plugin_info()
        print(f"Name: {plugin_info['display_name']}")
        print(f"Version: {plugin_info['version']}")
        print(f"Type: {plugin_info['algorithm_type']}")
        print(f"Task: {plugin_info['task_type']}")
        print(f"Capabilities: {', '.join(plugin_info['capabilities'].keys())}")
        print("=" * 50)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")