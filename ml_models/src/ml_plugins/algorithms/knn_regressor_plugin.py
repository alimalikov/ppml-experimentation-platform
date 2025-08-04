"""
K-Nearest Neighbors Regressor Plugin - Main Integration
======================================================

This is the main plugin file that integrates all components of the K-Nearest Neighbors
Regressor algorithm for the ML framework. It provides a unified interface for training,
analysis, and visualization.

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
    from src.ml_plugins.algorithms.knn_regressor.knn_core import KNNCore
    from src.ml_plugins.algorithms.knn_regressor.knn_analysis import KNNAnalysis
    from src.ml_plugins.algorithms.knn_regressor.knn_display import KNNDisplay
except ImportError as e:
    # Fallback for different import structures
    try:
        from knn_regressor.knn_core import KNNCore
        from knn_regressor.knn_analysis import KNNAnalysis
        from knn_regressor.knn_display import KNNDisplay
    except ImportError:
        raise ImportError(f"Could not import KNN components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNRegressorPlugin(MLPlugin):
    """
    Unified K-Nearest Neighbors Regressor Plugin.
    
    This class provides a complete interface for the K-Nearest Neighbors Regressor algorithm,
    integrating training, analysis, and visualization capabilities.
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 metric: str = 'euclidean',
                 p: int = 2,
                 auto_scale: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize the K-Nearest Neighbors Regressor Plugin.
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use for prediction
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
        random_state : int, optional
            Random state for reproducibility
        """
        
        # Initialize base MLPlugin
        super().__init__()
        
        # Initialize core component
        self.core = KNNCore(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p,
            auto_scale=auto_scale,
            random_state=random_state
        )
        
        # Analysis and display components (initialized after training)
        self.analysis = None
        self.display = None
        
        # Plugin metadata for MLPlugin interface
        self._plugin_info = {
            'name': 'knn_regressor',
            'display_name': 'K-Nearest Neighbors Regressor',
            'version': '1.0.0',
            'algorithm_type': 'Instance-based Learning',
            'task_type': 'Regression',
            'author': 'Bachelor Thesis Project',
            'description': 'Advanced K-Nearest Neighbors regression with comprehensive analysis and optimization'
        }
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 2
        
        # Training history
        self.training_history = []
        self.is_trained = False
        
        logger.info("✅ K-Nearest Neighbors Regressor Plugin initialized successfully")
    
    # ==================== REQUIRED ABSTRACT METHODS ====================
    
    def get_name(self) -> str:
        """Get the algorithm name (Required by MLPlugin)."""
        return "K-Nearest Neighbors Regressor"
    
    def get_description(self) -> str:
        """Get the algorithm description (Required by MLPlugin)."""
        return ("A regression algorithm that makes predictions by averaging the target "
                "values of the K nearest neighbors in the feature space, using various "
                "distance metrics and weighting strategies for optimal performance.")
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]) -> 'KNNRegressorPlugin':
        """Create model instance with given hyperparameters (Required by MLPlugin)."""
        return KNNRegressorPlugin(
            n_neighbors=hyperparameters.get("n_neighbors", self.core.n_neighbors),
            weights=hyperparameters.get("weights", self.core.weights),
            algorithm=hyperparameters.get("algorithm", self.core.algorithm),
            metric=hyperparameters.get("metric", self.core.metric),
            p=hyperparameters.get("p", self.core.p),
            auto_scale=hyperparameters.get("auto_scale", self.core.auto_scale),
            random_state=hyperparameters.get("random_state", self.core.random_state)
        )
    
    def get_hyperparameter_config(self, key_prefix: str = "") -> Dict[str, Any]:
        """Get hyperparameter configuration for UI (Required by MLPlugin)."""
        try:
            import streamlit as st
            
            st.sidebar.subheader(f"{self.get_name()} Configuration")
            
            # Create tabs for different configuration sections - ADDING STRATEGY TAB
            basic_tab, strategy_tab, analysis_tab, viz_tab, report_tab, advanced_tab = st.sidebar.tabs([
                "📊 Basic", "🎯 Strategy", "🔍 Analysis", "📈 Visualization", "📋 Reports", "⚙️ Advanced"
            ])
            
            config = {}
            
            # ==================== BASIC PARAMETERS TAB ====================
            with basic_tab:
                st.markdown("**Core Algorithm Parameters**")
                
                # Number of neighbors
                config["n_neighbors"] = st.number_input(
                    "Number of Neighbors (K):", 
                    value=int(self.core.n_neighbors), 
                    min_value=1, 
                    max_value=50, 
                    step=1,
                    help="Number of neighbors to use for prediction",
                    key=f"{key_prefix}_n_neighbors"
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
                
                # Distance metric parameter
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
                    help="Automatically scale features for better distance calculations",
                    key=f"{key_prefix}_auto_scale"
                )
            
            # ==================== NEW STRATEGY TAB ====================
            with strategy_tab:
                st.markdown("**🎯 ML Strategy Configuration**")
                
                # Strategy overview
                st.info("Configure comprehensive ML strategies for optimal KNN performance")
                
                # Model Selection Strategy
                st.markdown("**🏆 Model Selection Strategy**")
                config["strategy_model_selection"] = st.selectbox(
                    "Model Selection Approach:",
                    options=[
                        'automatic_optimization',
                        'manual_tuning', 
                        'grid_search_comprehensive',
                        'bayesian_optimization',
                        'evolutionary_search'
                    ],
                    index=0,
                    help="Primary strategy for model parameter selection",
                    key=f"{key_prefix}_strategy_model"
                )
                
                if config["strategy_model_selection"] == 'automatic_optimization':
                    st.success("✅ Automatic K and metric optimization with intelligent defaults")
                    config["auto_k_optimization"] = True
                    config["auto_metric_selection"] = True
                    config["auto_algorithm_selection"] = True
                    
                elif config["strategy_model_selection"] == 'grid_search_comprehensive':
                    st.info("🔍 Comprehensive grid search across all parameters")
                    config["grid_search_k_range"] = st.slider(
                        "K Search Range:",
                        min_value=1, max_value=50, value=(1, 20), step=1,
                        help="Range of K values for grid search",
                        key=f"{key_prefix}_grid_k_range"
                    )
                    config["grid_search_metrics"] = st.multiselect(
                        "Metrics to Test:",
                        options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                        default=['euclidean', 'manhattan'],
                        help="Distance metrics to test in grid search",
                        key=f"{key_prefix}_grid_metrics"
                    )
                
                # Performance Strategy
                st.markdown("**⚡ Performance Strategy**")
                config["strategy_performance"] = st.selectbox(
                    "Performance Priority:",
                    options=[
                        'balanced_performance',
                        'accuracy_first',
                        'speed_first',
                        'memory_efficient',
                        'scalability_focused'
                    ],
                    index=0,
                    help="Primary performance optimization strategy",
                    key=f"{key_prefix}_strategy_performance"
                )
                
                # Strategy-based automatic configurations
                if config["strategy_performance"] == 'accuracy_first':
                    st.success("🎯 Prioritizing prediction accuracy")
                    config["strategy_use_distance_weights"] = True
                    config["strategy_comprehensive_cv"] = True
                    config["strategy_metric_comparison"] = True
                    
                elif config["strategy_performance"] == 'speed_first':
                    st.success("⚡ Prioritizing prediction speed")
                    config["strategy_fast_algorithm"] = True
                    config["strategy_reduced_analysis"] = True
                    config["strategy_parallel_processing"] = True
                    
                elif config["strategy_performance"] == 'memory_efficient':
                    st.success("💾 Optimizing memory usage")
                    config["strategy_brute_force_avoid"] = True
                    config["strategy_batch_processing"] = True
                    
                # Validation Strategy
                st.markdown("**🔬 Validation Strategy**")
                config["strategy_validation"] = st.selectbox(
                    "Validation Approach:",
                    options=[
                        'cross_validation_standard',
                        'cross_validation_stratified',
                        'time_series_validation',
                        'holdout_validation',
                        'bootstrap_validation'
                    ],
                    index=0,
                    help="Strategy for model validation and performance assessment",
                    key=f"{key_prefix}_strategy_validation"
                )
                
                config["validation_folds"] = st.slider(
                    "Validation Folds:",
                    min_value=3, max_value=10, value=5,
                    help="Number of folds for cross-validation",
                    key=f"{key_prefix}_val_folds"
                )
                
                # Analysis Strategy
                st.markdown("**📊 Analysis Strategy**")
                config["strategy_analysis_depth"] = st.selectbox(
                    "Analysis Depth:",
                    options=[
                        'quick_insights',
                        'standard_analysis',
                        'comprehensive_analysis',
                        'research_grade',
                        'production_focused'
                    ],
                    index=1,
                    help="Depth and scope of performance analysis",
                    key=f"{key_prefix}_strategy_analysis"
                )
                
                # Strategy-specific analysis configurations
                if config["strategy_analysis_depth"] == 'quick_insights':
                    st.info("⚡ Fast analysis for quick decisions")
                    config["enable_k_optimization"] = True
                    config["enable_basic_cv"] = True
                    config["enable_metric_comparison"] = False
                    
                elif config["strategy_analysis_depth"] == 'comprehensive_analysis':
                    st.info("🔍 Thorough analysis for optimal performance")
                    config["enable_k_optimization"] = True
                    config["enable_metric_comparison"] = True
                    config["enable_neighbor_analysis"] = True
                    config["enable_bias_variance_analysis"] = True
                    config["enable_local_density_analysis"] = True
                    
                elif config["strategy_analysis_depth"] == 'research_grade':
                    st.info("🔬 Research-level analysis with all features")
                    config["enable_all_analyses"] = True
                    config["detailed_neighbor_patterns"] = True
                    config["algorithm_performance_profiling"] = True
                    config["feature_impact_analysis"] = True
                    
                # Recommendation Strategy
                st.markdown("**💡 Recommendation Strategy**")
                config["strategy_recommendations"] = st.selectbox(
                    "Recommendation Level:",
                    options=[
                        'conservative_suggestions',
                        'balanced_recommendations',
                        'aggressive_optimization',
                        'domain_specific',
                        'automated_implementation'
                    ],
                    index=1,
                    help="How aggressive to be with parameter recommendations",
                    key=f"{key_prefix}_strategy_recommendations"
                )
                
                config["recommendation_confidence_threshold"] = st.slider(
                    "Recommendation Confidence Threshold:",
                    min_value=0.1, max_value=0.9, value=0.05, step=0.01,
                    help="Minimum improvement required for recommendations",
                    key=f"{key_prefix}_rec_threshold"
                )
                
                # Deployment Strategy
                st.markdown("**🚀 Deployment Strategy**")
                config["strategy_deployment"] = st.selectbox(
                    "Deployment Focus:",
                    options=[
                        'development_mode',
                        'staging_validation',
                        'production_ready',
                        'real_time_inference',
                        'batch_processing'
                    ],
                    index=0,
                    help="Optimization strategy for deployment environment",
                    key=f"{key_prefix}_strategy_deployment"
                )
                
                if config["strategy_deployment"] == 'production_ready':
                    st.success("🏭 Production optimization enabled")
                    config["enable_model_validation"] = True
                    config["enable_performance_monitoring"] = True
                    config["enable_drift_detection"] = True
                    
                elif config["strategy_deployment"] == 'real_time_inference':
                    st.success("⚡ Real-time optimization enabled")
                    config["optimize_prediction_speed"] = True
                    config["precompute_distances"] = True
                    config["memory_efficient_storage"] = True
                
                # Risk Management Strategy
                st.markdown("**⚠️ Risk Management**")
                config["strategy_risk_management"] = st.checkbox(
                    "Enable Risk Assessment",
                    value=True,
                    help="Assess and mitigate algorithm-specific risks",
                    key=f"{key_prefix}_risk_mgmt"
                )
                
                if config["strategy_risk_management"]:
                    config["assess_curse_dimensionality"] = st.checkbox(
                        "Curse of Dimensionality Check",
                        value=True,
                        help="Assess impact of high-dimensional data",
                        key=f"{key_prefix}_curse_check"
                    )
                    
                    config["assess_data_density"] = st.checkbox(
                        "Data Density Analysis",
                        value=True,
                        help="Analyze local data density effects",
                        key=f"{key_prefix}_density_check"
                    )
                    
                    config["assess_outlier_sensitivity"] = st.checkbox(
                        "Outlier Sensitivity Analysis",
                        value=True,
                        help="Assess model sensitivity to outliers",
                        key=f"{key_prefix}_outlier_check"
                    )
                
                # Business Context Strategy
                st.markdown("**💼 Business Context**")
                config["strategy_business_context"] = st.selectbox(
                    "Business Priority:",
                    options=[
                        'general_purpose',
                        'cost_optimization',
                        'quality_assurance',
                        'regulatory_compliance',
                        'competitive_advantage'
                    ],
                    index=0,
                    help="Primary business objective",
                    key=f"{key_prefix}_business_context"
                )
                
                if config["strategy_business_context"] == 'regulatory_compliance':
                    st.info("📋 Compliance features enabled")
                    config["enable_model_explainability"] = True
                    config["enable_audit_trail"] = True
                    config["enable_bias_detection"] = True
                    
                elif config["strategy_business_context"] == 'competitive_advantage':
                    st.info("🏆 Advanced features enabled")
                    config["enable_advanced_optimization"] = True
                    config["enable_ensemble_methods"] = True
                    config["enable_custom_metrics"] = True
                
                # Strategy Summary
                st.markdown("**📋 Strategy Summary**")
                if st.button("Generate Strategy Report", key=f"{key_prefix}_strategy_report"):
                    strategy_summary = self._generate_strategy_summary(config)
                    st.json(strategy_summary)
            
            # ==================== ANALYSIS SETTINGS TAB (Updated based on strategy) ====================
            with analysis_tab:
                st.markdown("**Analysis Configuration**")
                
                # Strategy-aware analysis settings
                if config.get("strategy_analysis_depth") == 'quick_insights':
                    st.info("⚡ Quick analysis mode (configured by strategy)")
                    analysis_enabled = False
                elif config.get("strategy_analysis_depth") == 'research_grade':
                    st.info("🔬 Research-grade analysis (configured by strategy)")
                    analysis_enabled = True
                else:
                    analysis_enabled = True
                
                # Comprehensive analysis toggle
                config["enable_comprehensive_analysis"] = st.checkbox(
                    "Enable Comprehensive Analysis",
                    value=analysis_enabled,
                    help="Perform full analysis including K optimization and comparisons",
                    key=f"{key_prefix}_comprehensive"
                )
                
                # K optimization settings
                st.markdown("**K Optimization Settings**")
                
                # Use strategy-based defaults
                if config.get("strategy_model_selection") == 'automatic_optimization':
                    k_min, k_max = 1, min(30, config.get("n_neighbors", 5) * 3)
                else:
                    k_min, k_max = 1, 20
                    
                config["k_range_min"] = st.number_input(
                    "K Range Min:",
                    value=k_min,
                    min_value=1,
                    max_value=20,
                    step=1,
                    help="Minimum K value to test",
                    key=f"{key_prefix}_k_min"
                )
                
                config["k_range_max"] = st.number_input(
                    "K Range Max:",
                    value=k_max,
                    min_value=config["k_range_min"],
                    max_value=50,
                    step=1,
                    help="Maximum K value to test",
                    key=f"{key_prefix}_k_max"
                )
                
                # Cross-validation settings (strategy-aware)
                st.markdown("**Cross-Validation Settings**")
                
                cv_folds_default = config.get("validation_folds", 5)
                config["cv_folds"] = st.slider(
                    "CV Folds:",
                    min_value=3,
                    max_value=10,
                    value=cv_folds_default,
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
                
                # Distance metric analysis
                st.markdown("**Distance Metric Analysis**")
                
                metric_test_default = config.get("enable_metric_comparison", True)
                config["test_all_metrics"] = st.checkbox(
                    "Test All Distance Metrics",
                    value=metric_test_default,
                    help="Compare performance across all available distance metrics",
                    key=f"{key_prefix}_test_metrics"
                )
                
                algo_comparison_default = config.get("strategy_performance") != 'speed_first'
                config["algorithm_comparison"] = st.checkbox(
                    "Algorithm Performance Comparison",
                    value=algo_comparison_default,
                    help="Compare performance of different neighbor search algorithms",
                    key=f"{key_prefix}_algo_comparison"
                )
                
                # Neighbor analysis settings
                st.markdown("**Neighbor Analysis**")
                
                neighbor_samples_default = 50 if config.get("strategy_analysis_depth") != 'quick_insights' else 20
                config["neighbor_analysis_samples"] = st.slider(
                    "Analysis Sample Size:",
                    min_value=10,
                    max_value=100,
                    value=neighbor_samples_default,
                    help="Number of samples to analyze for neighbor patterns",
                    key=f"{key_prefix}_neighbor_samples"
                )
                
                density_analysis_default = config.get("enable_local_density_analysis", True)
                config["analyze_local_density"] = st.checkbox(
                    "Analyze Local Density Effects",
                    value=density_analysis_default,
                    help="Analyze how local data density affects predictions",
                    key=f"{key_prefix}_local_density"
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
                    index=1,  # Default to plotly_white
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
                config["generate_k_optimization"] = st.checkbox(
                    "K Optimization Curve",
                    value=True,
                    key=f"{key_prefix}_plot_k_opt"
                )
                
                config["generate_distance_comparison"] = st.checkbox(
                    "Distance Metrics Comparison",
                    value=True,
                    key=f"{key_prefix}_plot_distance"
                )
                
                config["generate_neighbor_analysis"] = st.checkbox(
                    "Neighbor Analysis Plot",
                    value=True,
                    key=f"{key_prefix}_plot_neighbors"
                )
                
                config["generate_bias_variance"] = st.checkbox(
                    "Bias-Variance Analysis",
                    value=True,
                    key=f"{key_prefix}_plot_bias_var"
                )
                
                config["generate_performance_dashboard"] = st.checkbox(
                    "Performance Dashboard",
                    value=True,
                    help="Generate comprehensive performance dashboard",
                    key=f"{key_prefix}_dashboard"
                )
            
            # ==================== REPORTING TAB ====================
            with report_tab:
                st.markdown("**Report Generation Settings**")
                
                # Report format
                config["report_format"] = st.selectbox(
                    "Report Format:",
                    options=['html', 'json', 'markdown'],
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
                
                config["include_k_optimization"] = st.checkbox(
                    "K Optimization Analysis",
                    value=True,
                    key=f"{key_prefix}_inc_k_opt"
                )
                
                config["include_metric_analysis"] = st.checkbox(
                    "Distance Metric Analysis",
                    value=True,
                    key=f"{key_prefix}_inc_metrics"
                )
                
                config["include_performance_analysis"] = st.checkbox(
                    "Performance Analysis",
                    value=True,
                    key=f"{key_prefix}_inc_performance"
                )
                
                config["include_neighbor_analysis"] = st.checkbox(
                    "Neighbor Pattern Analysis",
                    value=True,
                    key=f"{key_prefix}_inc_neighbors"
                )
                
                config["include_recommendations"] = st.checkbox(
                    "Parameter Recommendations",
                    value=True,
                    key=f"{key_prefix}_inc_recommendations"
                )
                
                config["include_visualizations"] = st.checkbox(
                    "Visualizations",
                    value=True,
                    key=f"{key_prefix}_inc_viz"
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
                
                config["export_analysis_data"] = st.checkbox(
                    "Export Analysis Data",
                    value=False,
                    help="Export raw analysis data as JSON",
                    key=f"{key_prefix}_export_data"
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
                config["k_optimization_steps"] = st.slider(
                    "K Optimization Steps:",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Number of K values to test in optimization",
                    key=f"{key_prefix}_k_steps"
                )
                
                config["metric_comparison_depth"] = st.selectbox(
                    "Metric Comparison Depth:",
                    options=['basic', 'extended', 'comprehensive'],
                    index=1,
                    help="Depth of distance metric comparison",
                    key=f"{key_prefix}_metric_depth"
                )
                
                config["neighbor_analysis_depth"] = st.selectbox(
                    "Neighbor Analysis Depth:",
                    options=['basic', 'detailed', 'comprehensive'],
                    index=1,
                    help="Depth of neighbor pattern analysis",
                    key=f"{key_prefix}_neighbor_depth"
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
                
                config["enable_profiling"] = st.checkbox(
                    "Enable Performance Profiling",
                    value=False,
                    help="Profile performance of different components",
                    key=f"{key_prefix}_profiling"
                )
                
                # Expert options
                st.markdown("**Expert Options**")
                config["custom_k_values"] = st.text_input(
                    "Custom K Values:",
                    value="",
                    help="Custom K values to test (e.g., '1,3,5,7,10')",
                    key=f"{key_prefix}_custom_k"
                )
                
                config["custom_metrics"] = st.text_input(
                    "Custom Metrics:",
                    value="",
                    help="Additional metrics to test (comma-separated)",
                    key=f"{key_prefix}_custom_metrics"
                )
                
                config["weight_function_analysis"] = st.checkbox(
                    "Weight Function Analysis",
                    value=False,
                    help="Analyze different weighting strategies",
                    key=f"{key_prefix}_weight_analysis"
                )
            
            # Process k_range into tuple
            config["k_range"] = (config["k_range_min"], config["k_range_max"])
            
            config = self._process_strategy_config(config)

            # Convert random_state to None if using default
            if config["random_state"] == 42:
                config["random_state"] = None
            
            return config
            
        except ImportError:
            # Fallback when streamlit is not available
            return {
                "n_neighbors": self.core.n_neighbors,
                "weights": self.core.weights,
                "algorithm": self.core.algorithm,
                "metric": self.core.metric,
                "p": self.core.p,
                "auto_scale": self.core.auto_scale,
                "random_state": self.core.random_state,
                # Strategy defaults
                "strategy_model_selection": "automatic_optimization",
                "strategy_performance": "balanced_performance",
                "strategy_analysis_depth": "standard_analysis",
                "strategy_recommendations": "balanced_recommendations",
                "strategy_deployment": "development_mode",
                "enable_comprehensive_analysis": True,
                "cv_folds": 5,
                "k_range": (1, 20),
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
            
            # Check if K is reasonable for data size
            max_reasonable_k = len(df) // 2
            if self.core.n_neighbors > max_reasonable_k:
                return False, f"K={self.core.n_neighbors} too large for {len(df)} samples"
            
            return True, f"Compatible with {len(df)} samples and {X.shape[1]} features"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    # ==================== EXISTING MLPLUGIN INTERFACE ====================
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information for the ML framework (Required by MLPlugin)."""
        return {
            'name': 'knn_regressor',
            'display_name': 'K-Nearest Neighbors Regressor',
            'version': '1.0.0',
            'algorithm_type': 'Instance-based Learning',
            'task_type': 'Regression',
            'description': 'Advanced K-Nearest Neighbors regression with comprehensive K optimization and analysis',
            'parameters': {
                'n_neighbors': {'type': 'int', 'default': 5, 'description': 'Number of neighbors'},
                'weights': {'type': 'str', 'default': 'uniform', 'choices': ['uniform', 'distance']},
                'algorithm': {'type': 'str', 'default': 'auto', 'choices': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                'metric': {'type': 'str', 'default': 'euclidean', 'choices': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']},
                'auto_scale': {'type': 'bool', 'default': True, 'description': 'Automatic feature scaling'}
            },
            'capabilities': {
                'training': True,
                'prediction': True,
                'analysis': True,
                'visualization': True,
                'reporting': True,
                'parameter_optimization': True,
                'k_optimization': True,
                'metric_comparison': True,
                'neighbor_analysis': True
            },
            'requirements': {
                'min_samples': 2,
                'handles_missing_values': False,
                'requires_scaling': False,  # Optional but recommended
                'memory_intensive': True,
                'curse_of_dimensionality': True
            }
        }
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """Train the model (Required by MLPlugin)."""
        return self.fit(X, y, **kwargs)
    
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
                'n_samples': len(y_array),
                'effective_k': min(self.core.n_neighbors, len(self.core.X_train_)) if hasattr(self.core, 'X_train_') and self.core.X_train_ is not None else self.core.n_neighbors
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary (Required by MLPlugin)."""
        return self._create_model_summary()
    
    def _generate_strategy_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive strategy summary based on configuration."""
        try:
            strategy_summary = {
                "strategy_overview": {
                    "model_selection": config.get("strategy_model_selection", "automatic_optimization"),
                    "performance_priority": config.get("strategy_performance", "balanced_performance"),
                    "analysis_depth": config.get("strategy_analysis_depth", "standard_analysis"),
                    "validation_approach": config.get("strategy_validation", "cross_validation_standard"),
                    "deployment_focus": config.get("strategy_deployment", "development_mode"),
                    "business_context": config.get("strategy_business_context", "general_purpose")
                },
                "optimization_plan": {
                    "k_optimization": config.get("auto_k_optimization", True),
                    "metric_selection": config.get("auto_metric_selection", True),
                    "algorithm_selection": config.get("auto_algorithm_selection", True),
                    "performance_tuning": config.get("strategy_performance") != "speed_first"
                },
                "analysis_plan": {
                    "comprehensive_analysis": config.get("enable_comprehensive_analysis", True),
                    "k_range": config.get("k_range", (1, 20)),
                    "cv_folds": config.get("cv_folds", 5),
                    "metric_comparison": config.get("test_all_metrics", True),
                    "neighbor_analysis": config.get("analyze_local_density", True)
                },
                "risk_mitigation": {
                    "curse_dimensionality_check": config.get("assess_curse_dimensionality", False),
                    "data_density_analysis": config.get("assess_data_density", False),
                    "outlier_sensitivity": config.get("assess_outlier_sensitivity", False)
                },
                "deployment_readiness": {
                    "production_optimized": config.get("strategy_deployment") == "production_ready",
                    "real_time_capable": config.get("strategy_deployment") == "real_time_inference",
                    "monitoring_enabled": config.get("enable_performance_monitoring", False),
                    "validation_robust": config.get("enable_model_validation", False)
                },
                "expected_outcomes": self._predict_strategy_outcomes(config),
                "recommendations": self._get_strategy_recommendations(config)
            }
            
            return strategy_summary
            
        except Exception as e:
            logger.error(f"❌ Strategy summary generation failed: {str(e)}")
            return {'error': str(e)}

    def _process_strategy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process and apply strategy-based configurations."""
        try:
            # Apply model selection strategy
            if config.get("strategy_model_selection") == "automatic_optimization":
                config["enable_k_optimization"] = True
                config["enable_metric_comparison"] = True
                config["enable_algorithm_comparison"] = True
                
            elif config.get("strategy_model_selection") == "grid_search_comprehensive":
                config["enable_grid_search"] = True
                config["comprehensive_parameter_search"] = True
                
            # Apply performance strategy
            if config.get("strategy_performance") == "speed_first":
                config["algorithm"] = "ball_tree"  # Generally fastest for most cases
                config["enable_parallel_processing"] = True
                config["reduced_analysis"] = True
                
            elif config.get("strategy_performance") == "accuracy_first":
                config["weights"] = "distance"
                config["comprehensive_cv"] = True
                config["enable_bias_variance_analysis"] = True
                
            elif config.get("strategy_performance") == "memory_efficient":
                if config.get("algorithm") == "brute":
                    config["algorithm"] = "ball_tree"
                config["batch_processing"] = True
                
            # Apply analysis strategy
            if config.get("strategy_analysis_depth") == "quick_insights":
                config["k_range"] = (max(1, config.get("n_neighbors", 5) - 2), 
                                min(15, config.get("n_neighbors", 5) + 5))
                config["cv_folds"] = 3
                config["neighbor_analysis_samples"] = 20
                
            elif config.get("strategy_analysis_depth") == "research_grade":
                config["k_range"] = (1, 30)
                config["cv_folds"] = 10
                config["neighbor_analysis_samples"] = 100
                config["enable_all_analyses"] = True
                
            # Apply deployment strategy
            if config.get("strategy_deployment") == "production_ready":
                config["enable_model_validation"] = True
                config["enable_performance_monitoring"] = True
                config["robust_error_handling"] = True
                
            elif config.get("strategy_deployment") == "real_time_inference":
                config["optimize_prediction_speed"] = True
                config["algorithm"] = "ball_tree"
                config["precompute_neighbor_cache"] = True
                
            # Apply business context
            if config.get("strategy_business_context") == "regulatory_compliance":
                config["enable_audit_trail"] = True
                config["enable_explainability"] = True
                config["detailed_validation"] = True
                
            return config
            
        except Exception as e:
            logger.error(f"❌ Strategy config processing failed: {str(e)}")
            return config

    def _predict_strategy_outcomes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Predict expected outcomes based on strategy configuration."""
        try:
            outcomes = {}
            
            # Predict performance outcomes
            if config.get("strategy_performance") == "accuracy_first":
                outcomes["expected_accuracy"] = "High (optimized for prediction quality)"
                outcomes["expected_speed"] = "Medium (comprehensive analysis may slow training)"
                outcomes["expected_memory"] = "Medium to High"
                
            elif config.get("strategy_performance") == "speed_first":
                outcomes["expected_accuracy"] = "Medium (balanced for speed)"
                outcomes["expected_speed"] = "High (optimized algorithms and reduced analysis)"
                outcomes["expected_memory"] = "Low to Medium"
                
            # Predict analysis outcomes
            analysis_depth = config.get("strategy_analysis_depth", "standard_analysis")
            if analysis_depth == "comprehensive_analysis":
                outcomes["analysis_completeness"] = "Very High"
                outcomes["insights_quality"] = "Excellent"
                outcomes["analysis_time"] = "Long"
                
            elif analysis_depth == "quick_insights":
                outcomes["analysis_completeness"] = "Basic"
                outcomes["insights_quality"] = "Good"
                outcomes["analysis_time"] = "Short"
                
            # Predict deployment readiness
            if config.get("strategy_deployment") == "production_ready":
                outcomes["deployment_readiness"] = "High"
                outcomes["monitoring_capability"] = "Comprehensive"
                outcomes["maintenance_effort"] = "Low"
                
            return outcomes
            
        except Exception as e:
            return {"error": str(e)}

    def _get_strategy_recommendations(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get strategic recommendations based on configuration."""
        try:
            recommendations = []
            
            # Model selection recommendations
            if config.get("strategy_model_selection") == "manual_tuning":
                recommendations.append({
                    "category": "Model Selection",
                    "priority": "Medium",
                    "recommendation": "Consider enabling automatic optimization for better parameter discovery",
                    "impact": "May improve model performance with less manual effort"
                })
                
            # Performance recommendations
            if config.get("strategy_performance") == "balanced_performance":
                recommendations.append({
                    "category": "Performance",
                    "priority": "Low",
                    "recommendation": "Current balanced approach is good for most use cases",
                    "impact": "Maintains good trade-off between accuracy and speed"
                })
                
            # Analysis recommendations
            if config.get("strategy_analysis_depth") == "quick_insights":
                recommendations.append({
                    "category": "Analysis",
                    "priority": "Medium",
                    "recommendation": "Consider upgrading to standard analysis for production models",
                    "impact": "Better parameter optimization and risk assessment"
                })
                
            # Deployment recommendations
            if config.get("strategy_deployment") == "development_mode":
                recommendations.append({
                    "category": "Deployment",
                    "priority": "High",
                    "recommendation": "Upgrade to production_ready before deploying to production",
                    "impact": "Essential for reliable production performance"
                })
                
            return recommendations
            
        except Exception as e:
            return [{"error": str(e)}]
        
    # ==================== MAIN PLUGIN INTERFACE ====================
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Train the K-Nearest Neighbors Regressor model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        **kwargs : dict
            Advanced configuration options from hyperparameter config
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metadata
        """
        try:
            logger.info("🚀 Starting K-Nearest Neighbors Regressor training...")
            
            # Extract advanced config options
            config = kwargs.get('config', {})
            verbose_logging = config.get('enable_verbose_logging', False)
            
            if verbose_logging:
                logger.setLevel(logging.DEBUG)
                logger.debug(f"Advanced configuration: {config}")
            
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
                self.analysis = KNNAnalysis(self.core)
                self.display = KNNDisplay(self.core, self.analysis)
                
                # Apply advanced configuration to components
                self._apply_advanced_config(config)
                
                self.is_trained = True
                
                # Store training history
                training_record = {
                    'timestamp': pd.Timestamp.now(),
                    'data_shape': X_array.shape,
                    'training_result': training_result,
                    'parameters': self.get_params(),
                    'advanced_config': config
                }
                self.training_history.append(training_record)
                
                # Perform initial analysis if enabled
                quick_analysis = {}
                if config.get('enable_comprehensive_analysis', True):
                    try:
                        quick_analysis = self._get_quick_analysis(config)
                    except Exception as e:
                        logger.warning(f"Quick analysis failed: {e}")
                        quick_analysis = {'error': str(e)}
                
                logger.info("✅ Training completed successfully")
                
                return {
                    'success': True,
                    'training_result': training_result,
                    'model_summary': self._create_model_summary(),
                    'quick_analysis': quick_analysis,
                    'advanced_config_applied': True
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
                    'k_optimization': self.analysis.analyze_k_optimization(),
                    'distance_metrics': self.analysis.analyze_distance_metrics(),
                    'algorithm_performance': self.analysis.analyze_algorithm_performance(),
                    'cross_validation': self.analysis.analyze_cross_validation(),
                    'neighbor_distributions': self.analysis.analyze_neighbor_distributions(),
                    'local_density_analysis': self.analysis.analyze_local_density_effects(),
                    'bias_variance_analysis': self.analysis.analyze_k_bias_variance()
                })
            else:
                # Quick analysis
                analysis_results.update({
                    'k_optimization': self.analysis.analyze_k_optimization(),
                    'cross_validation': self.analysis.analyze_cross_validation()
                })
            
            logger.info("✅ Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_k_optimization(self) -> Dict[str, Any]:
        """Get K value optimization analysis."""
        if not self.is_trained:
            return {'error': 'Model must be trained before K optimization analysis'}
        
        try:
            return self.analysis.analyze_k_optimization()
        except Exception as e:
            return {'error': str(e)}
    
    def get_neighbor_analysis(self) -> Dict[str, Any]:
        """Get neighbor pattern analysis."""
        if not self.is_trained:
            return {'error': 'Model must be trained before neighbor analysis'}
        
        try:
            return self.analysis.analyze_neighbor_distributions()
        except Exception as e:
            return {'error': str(e)}
    
    def compare_distance_metrics(self) -> Dict[str, Any]:
        """Compare different distance metrics."""
        if not self.is_trained:
            return {'error': 'Model must be trained before distance metric comparison'}
        
        try:
            return self.analysis.analyze_distance_metrics()
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
    
    def plot_k_optimization_curve(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot K optimization curve."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_k_optimization_curve(interactive=interactive, save_path=save_path)
        except Exception as e:
            return {'error': str(e)}
    
    def plot_distance_metric_comparison(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot distance metric comparison."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_distance_metric_comparison(interactive=interactive, save_path=save_path)
        except Exception as e:
            return {'error': str(e)}
    
    def plot_neighbor_analysis(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot neighbor analysis."""
        if not self.is_trained:
            return {'error': 'Model must be trained before plotting'}
        
        try:
            return self.display.plot_neighbor_analysis(interactive=interactive, save_path=save_path)
        except Exception as e:
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
            Report format ('html', 'json', 'markdown')
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
            'n_neighbors': self.core.n_neighbors,
            'weights': self.core.weights,
            'algorithm': self.core.algorithm,
            'metric': self.core.metric,
            'p': self.core.p,
            'auto_scale': self.core.auto_scale,
            'random_state': self.core.random_state
        }
    
    def set_params(self, **params) -> 'KNNRegressorPlugin':
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
            
            # K recommendations
            k_analysis = analyses.get('k_optimization', {})
            if 'optimal_k' in k_analysis:
                optimal_k = k_analysis['optimal_k']
                current_k = self.core.n_neighbors
                improvement = k_analysis.get('improvement_potential', 0)
                
                if improvement > 0.05:  # 5% improvement threshold
                    recommendations.append({
                        'parameter': 'n_neighbors',
                        'current_value': current_k,
                        'recommended_value': optimal_k,
                        'improvement_potential': improvement,
                        'reason': f'K={optimal_k} shows significant improvement in cross-validation'
                    })
            
            # Metric recommendations
            metric_analysis = analyses.get('distance_metrics', {})
            if 'best_metric' in metric_analysis:
                best_metric = metric_analysis['best_metric']
                improvement = metric_analysis.get('improvement_potential', 0)
                if improvement > 0.02:  # 2% improvement
                    recommendations.append({
                        'parameter': 'metric',
                        'current_value': self.core.metric,
                        'recommended_value': best_metric,
                        'improvement_potential': improvement,
                        'reason': f'Distance metric "{best_metric}" performs better'
                    })
            
            return {
                'recommendations': recommendations,
                'analysis_source': 'comprehensive_analysis'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== UTILITY METHODS (COMPLETE IMPLEMENTATION) ====================
    
    def _convert_to_array(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert various data types to numpy array."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.values.astype(np.float64)
            elif isinstance(data, pd.Series):
                return data.values.astype(np.float64)
            elif isinstance(data, np.ndarray):
                return data.astype(np.float64)
            elif isinstance(data, (list, tuple)):
                return np.array(data, dtype=np.float64)
            else:
                return np.array(data, dtype=np.float64)
        except Exception as e:
            logger.error(f"❌ Data conversion failed: {str(e)}")
            raise ValueError(f"Could not convert data to array: {str(e)}")
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate input data."""
        try:
            # Check shapes
            if X.ndim != 2:
                return {'valid': False, 'error': f'X must be 2D, got {X.ndim}D'}
            
            if y.ndim != 1:
                return {'valid': False, 'error': f'y must be 1D, got {y.ndim}D'}
            
            if X.shape[0] != y.shape[0]:
                return {'valid': False, 'error': f'X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}'}
            
            # Check for sufficient samples
            if X.shape[0] < self.core.n_neighbors:
                return {'valid': False, 'error': f'Need at least {self.core.n_neighbors} samples for K={self.core.n_neighbors}, got {X.shape[0]}'}
            
            # Check for NaN/inf values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                return {'valid': False, 'error': 'X contains NaN or infinite values'}
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                return {'valid': False, 'error': 'y contains NaN or infinite values'}
            
            # Check for constant features
            X_var = np.var(X, axis=0)
            constant_features = np.sum(X_var == 0)
            if constant_features == X.shape[1]:
                return {'valid': False, 'error': 'All features are constant'}
            
            return {
                'valid': True,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'constant_features': constant_features,
                'effective_k': min(self.core.n_neighbors, X.shape[0])
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {str(e)}'}
    
    def _create_model_summary(self) -> Dict[str, Any]:
        """Create comprehensive model summary."""
        try:
            summary = {
                'algorithm': 'K-Nearest Neighbors Regressor',
                'version': self._plugin_info['version'],
                'status': 'trained' if self.is_trained else 'not_trained',
                'parameters': self.get_params(),
                'training_info': {},
                'performance_info': {}
            }
            
            if self.is_trained and hasattr(self.core, 'X_train_'):
                summary['training_info'].update({
                    'training_samples': len(self.core.X_train_),
                    'n_features': self.core.X_train_.shape[1] if self.core.X_train_ is not None else 0,
                    'effective_k': min(self.core.n_neighbors, len(self.core.X_train_)),
                    'training_time': getattr(self.core, 'training_time_', None),
                    'scaling_applied': self.core.auto_scale
                })
                
                # Add performance metrics if available
                if hasattr(self.core, 'X_train_scaled_') and self.core.X_train_scaled_ is not None:
                    try:
                        train_score = self.core.score(self.core.X_train_scaled_, self.core.y_train_)
                        summary['performance_info']['training_r2'] = train_score
                    except:
                        pass
            
            summary['capabilities'] = {
                'supports_regression': True,
                'supports_classification': False,
                'handles_missing_values': False,
                'automatic_scaling': True,
                'k_optimization': True,
                'metric_comparison': True
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Model summary creation failed: {str(e)}")
            return {'error': str(e)}
    
    def _apply_advanced_config(self, config: Dict[str, Any]) -> None:
        """Apply advanced configuration to analysis and display components."""
        try:
            # Configure analysis component
            if hasattr(self.analysis, 'configure'):
                analysis_config = {
                    'cv_folds': config.get('cv_folds', 5),
                    'cv_scoring': config.get('cv_scoring', 'r2'),
                    'k_range': config.get('k_range', (1, 20)),
                    'test_all_metrics': config.get('test_all_metrics', True),
                    'algorithm_comparison': config.get('algorithm_comparison', True),
                    'neighbor_analysis_samples': config.get('neighbor_analysis_samples', 50),
                    'analyze_local_density': config.get('analyze_local_density', True),
                    'n_jobs': config.get('n_jobs', 4) if config.get('enable_parallel_processing', True) else 1,
                    'metric_comparison_depth': config.get('metric_comparison_depth', 'extended'),
                    'neighbor_analysis_depth': config.get('neighbor_analysis_depth', 'detailed')
                }
                self.analysis.configure(analysis_config)
            
            # Configure display component
            if hasattr(self.display, 'configure'):
                display_config = {
                    'interactive_plots': config.get('enable_interactive_plots', True),
                    'plot_theme': config.get('plot_theme', 'plotly_white'),
                    'plot_width': config.get('plot_width', 800),
                    'plot_height': config.get('plot_height', 500),
                    'dashboard_layout': config.get('dashboard_layout', 'detailed'),
                    'show_confidence_intervals': config.get('show_confidence_intervals', True),
                    'auto_save_plots': config.get('auto_save_plots', False),
                    'plot_save_format': config.get('plot_save_format', 'png')
                }
                self.display.configure(display_config)
                
        except Exception as e:
            logger.warning(f"Failed to apply advanced config: {e}")
    
    def _get_quick_analysis(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get quick analysis after training with configuration."""
        try:
            if not self.analysis:
                return {}
            
            config = config or {}
            
            # Quick K optimization check
            k_analysis = self.analysis.analyze_k_optimization()
            
            # Quick cross-validation
            cv_result = self.analysis.analyze_cross_validation()
            
            result = {
                'current_k': self.core.n_neighbors,
                'optimal_k': k_analysis.get('optimal_k', self.core.n_neighbors),
                'k_improvement_potential': k_analysis.get('improvement_potential', 0),
                'cv_score': cv_result.get('basic_cv', {}).get('mean_score', 0),
                'cv_stability': cv_result.get('stability_analysis', {}).get('consistency_score', 0),
                'training_successful': True,
                'config_applied': True
            }
            
            # Add distance metric quick check if enabled
            if config.get('test_all_metrics', False):
                try:
                    metric_analysis = self.analysis.analyze_distance_metrics()
                    if 'error' not in metric_analysis:
                        result['best_metric'] = metric_analysis.get('best_metric', self.core.metric)
                        result['metric_improvement'] = metric_analysis.get('improvement_potential', 0)
                except Exception as e:
                    logger.debug(f"Metric analysis in quick analysis failed: {e}")
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _serialize_training_history(self) -> List[Dict[str, Any]]:
        """Serialize training history for export."""
        try:
            serialized = []
            for record in self.training_history:
                serialized_record = record.copy()
                # Convert pandas Timestamp to string
                if 'timestamp' in serialized_record:
                    serialized_record['timestamp'] = str(serialized_record['timestamp'])
                # Convert numpy arrays to lists
                if 'data_shape' in serialized_record:
                    serialized_record['data_shape'] = list(serialized_record['data_shape'])
                serialized.append(serialized_record)
            return serialized
        except Exception as e:
            logger.error(f"❌ Training history serialization failed: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'plugin_info': self._plugin_info,
            'is_trained': self.is_trained,
            'parameters': self.get_params(),
            'training_history_count': len(self.training_history),
            'capabilities': {
                'k_optimization': True,
                'distance_metric_comparison': True,
                'neighbor_analysis': True,
                'bias_variance_analysis': True,
                'interactive_visualization': True,
                'comprehensive_reporting': True
            }
        }
        
        if self.is_trained:
            info.update({
                'model_summary': self._create_model_summary(),
                'performance_summary': self.core.get_performance_summary() if hasattr(self.core, 'get_performance_summary') else {}
            })
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the KNN Plugin."""
        status = "trained" if self.is_trained else "not trained"
        return (f"KNNRegressorPlugin(n_neighbors={self.core.n_neighbors}, "
                f"metric='{self.core.metric}', algorithm='{self.core.algorithm}', "
                f"status={status})")
        
# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return KNNRegressorPlugin()