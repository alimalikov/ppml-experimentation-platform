# filepath: c:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\src\ml_plugins\visualizations\advanced_analysis_plugin.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import logging

# Fix the import - use absolute import instead of relative
try:
    import sys
    from pathlib import Path
    
    # Add the ml_plugins directory to path
    plugin_dir = Path(__file__).parent.parent
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
    
    from base_visualization_plugin import (
        BaseVisualizationPlugin, 
        VisualizationCategory, 
        DataType, 
        VisualizationError
    )
except ImportError as e:
    print(f"Import error in advanced_analysis_plugin: {e}")
    # Fallback - try alternative import
    try:
        from src.ml_plugins.base_visualization_plugin import (
            BaseVisualizationPlugin, 
            VisualizationCategory, 
            DataType, 
            VisualizationError
        )
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        raise

class AdvancedMLAnalysisPlugin(BaseVisualizationPlugin):
    """
    Advanced ML Analysis Dashboard - Unified Performance and PPML Analysis
    
    Tabbed interface combining performance comparison charts with privacy-preserving ML analysis
    in a comprehensive dashboard with flexible chart types and unified configuration.
    """
    
    def __init__(self):
        """Initialize the Advanced ML Analysis Plugin"""
        super().__init__()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Plugin metadata
        self.name = "Advanced ML Analysis Dashboard"
        self.description = "Unified performance and PPML analysis with flexible chart types and comprehensive insights"
        self.category = VisualizationCategory.COMPARISON
        
        # Supported configurations
        self.supported_data_types = [
            DataType.CLASSIFICATION,
            DataType.REGRESSION,
            DataType.BINARY,
            DataType.MULTICLASS
        ]
        
        # Requirements
        self.min_samples = 1
        self.requires_trained_model = True
        self.requires_predictions = False
        self.requires_probabilities = False
        self.interactive = True
        self.export_formats = ["png", "pdf", "html", "svg"]
        
        # Make this a direct visualization (no config step)
        self.requires_configuration = False # Ensures the system knows no pre-config is needed
        self.show_config_first = False    # Reinforces direct rendering
                
        # Available chart types for all analyses
        self.chart_types = {
            "bar": "Bar Chart",
            "horizontal_bar": "Horizontal Bar Chart", 
            "line": "Line Chart",
            "scatter": "Scatter Plot",
            "radar": "Radar Chart",
            "heatmap": "Heatmap",
            "box": "Box Plot",
            "violin": "Violin Plot",
            "bubble": "Bubble Chart",
            "area": "Area Chart"
        }
        
        # Common performance metrics
        self.common_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "mse", "rmse", "mae", "r2_score"
        ]
        
        # Color palettes
        self.color_palettes = {
            "viridis": px.colors.sequential.Viridis,
            "plasma": px.colors.sequential.Plasma,
            "blues": px.colors.sequential.Blues,
            "greens": px.colors.sequential.Greens,
            "reds": px.colors.sequential.Reds,
            "categorical": px.colors.qualitative.Set1,
            "set2": px.colors.qualitative.Set2,
            "turbo": px.colors.sequential.Turbo
        }
    
    def can_visualize(self, data_type: Union[str, DataType], model_results: List[Dict[str, Any]], 
                     data: Optional[pd.DataFrame] = None) -> bool:
        """Check if this plugin can handle the given data and model results."""
        try:
            # Convert string to enum if needed
            if isinstance(data_type, str):
                try:
                    data_type = DataType(data_type.lower())
                except ValueError:
                    return False
            
            # Check if data type is supported
            if data_type not in self.supported_data_types:
                return False
            
            # Need at least 1 model result
            if not model_results or len(model_results) < 1:
                return False
            
            # Check if model results have required structure
            valid_results = 0
            for result in model_results:
                if not isinstance(result, dict):
                    continue
                
                # Check for common performance metrics
                has_metrics = any(metric in result for metric in self.common_metrics)
                if has_metrics and "error" not in result:
                    valid_results += 1
            
            # Need at least 1 valid result
            return valid_results >= 1
            
        except Exception as e:
            self.logger.error(f"Error in can_visualize: {str(e)}")
            return False
    
    def get_config_options(self) -> Dict[str, Dict[str, Any]]:
        """
        Return an empty dictionary to signal to the VisualizationManager
        that this plugin does not require a separate configuration step.
        Configuration is handled internally via the '⚙️ Configuration' tab.
        """
        return {}
    
    def render(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
               config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """Render the unified advanced ML analysis dashboard with tabbed interface."""
        try:
            # Initialize session state key for this plugin's configuration
            config_key = "advanced_ml_analysis_config"
            
            # Initialize config in session state if not exists
            if config_key not in st.session_state:
                default_config = {
                    'chart_type': 'bar',
                    'color_palette': 'viridis', 
                    'chart_height': 600,
                    'show_values': True,
                    'ppml_primary_metric': 'f1_score',
                    'show_privacy_budget': True,
                    'include_data_retention': True,
                    'privacy_risk_threshold': 0.1,
                    'performance_metrics': ['accuracy', 'f1_score'],
                    'normalize_metrics': False
                }
                st.session_state[config_key] = default_config
            
            # Use session state config (this ensures persistence across tab switches)
            active_config = st.session_state[config_key]
            
            # Merge with any provided config (but session state takes precedence for persistence)
            if config:
                for key, value in config.items():
                    if key not in active_config:
                        active_config[key] = value
            
            # Create the main tabbed interface header
            st.header("📊 Advanced ML Analysis Dashboard")
            st.markdown("*Unified Performance and Privacy-Preserving ML Analysis*")
            
            # Create tabs for different analysis types
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Performance Analysis", 
                "🔒 PPML Analysis", 
                "📊 Comprehensive Dashboard",
                "⚙️ Configuration"
            ])
            
            with tab1:
                # Performance Analysis Tab - uses session state config
                success1 = self._render_performance_analysis(data, model_results, active_config)
            
            with tab2:
                # PPML Analysis Tab - uses session state config
                success2 = self._render_ppml_analysis(data, model_results, active_config)
            
            with tab3:
                # Comprehensive Dashboard Tab - uses session state config
                success3 = self._render_comprehensive_dashboard(data, model_results, active_config)
            
            with tab4:
                # Configuration tab - updates session state directly
                config_changed = self._render_unified_configuration(config_key)
                
                # If configuration changed, trigger a rerun to update all tabs
                if config_changed:
                    st.rerun()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rendering advanced ML analysis: {str(e)}")
            st.error(f"❌ Error creating advanced analysis: {str(e)}")
            return False

    def _render_unified_configuration(self, config_key: str) -> bool:
        """
        Render unified configuration interface that updates session state in real-time.
        Returns True if any configuration changed.
        """
        st.subheader("⚙️ Unified Configuration")
        
        # Get current config from session state
        current_config = st.session_state[config_key].copy()
        config_changed = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Chart Preferences**")
            
            # Chart Type Selection
            current_chart_type = current_config.get('chart_type', 'bar')
            if current_chart_type not in self.chart_types.keys():
                current_chart_type = 'bar'
            chart_type_idx = list(self.chart_types.keys()).index(current_chart_type)
            
            chart_type = st.selectbox(
                "Chart Type:",
                options=list(self.chart_types.keys()),
                format_func=lambda x: self.chart_types[x],
                index=chart_type_idx,
                key=f"{config_key}_chart_type",
                help="Select default chart type for all analyses"
            )
            
            # Color Palette Selection
            current_color_palette = current_config.get('color_palette', 'viridis')
            if current_color_palette not in self.color_palettes.keys():
                current_color_palette = 'viridis'
            color_palette_idx = list(self.color_palettes.keys()).index(current_color_palette)
            
            color_palette = st.selectbox(
                "Color Palette:",
                options=list(self.color_palettes.keys()),
                index=color_palette_idx,
                key=f"{config_key}_color_palette",
                help="Choose color scheme for all visualizations"
            )
            
            # Chart Height Slider
            chart_height = st.slider(
                "Chart Height:",
                min_value=400,
                max_value=1000,
                value=current_config.get('chart_height', 600),
                step=50,
                key=f"{config_key}_chart_height",
                help="Adjust height for all charts"
            )
            
            # Show Values Checkbox
            show_values = st.checkbox(
                "Show Values on Charts",
                value=current_config.get('show_values', True),
                key=f"{config_key}_show_values",
                help="Display metric values on all charts"
            )
        
        with col2:
            st.markdown("**🔒 PPML-Specific Options**")
            
            # PPML Primary Metric
            ppml_metrics_options = ['accuracy', 'f1_score', 'precision', 'recall']
            current_ppml_metric = current_config.get('ppml_primary_metric', 'f1_score')
            if current_ppml_metric not in ppml_metrics_options:
                current_ppml_metric = 'f1_score'
            ppml_primary_metric_idx = ppml_metrics_options.index(current_ppml_metric)
            
            ppml_primary_metric = st.selectbox(
                "PPML Primary Metric:",
                options=ppml_metrics_options,
                index=ppml_primary_metric_idx,
                key=f"{config_key}_ppml_primary_metric",
                help="Main metric for PPML utility evaluation"
            )
            
            # Privacy Budget Analysis
            show_privacy_budget = st.checkbox(
                "Show Privacy Budget Analysis",
                value=current_config.get('show_privacy_budget', True),
                key=f"{config_key}_show_privacy_budget",
                help="Include privacy budget analysis"
            )
            
            # Data Retention Analysis
            include_data_retention = st.checkbox(
                "Include Data Retention Analysis",
                value=current_config.get('include_data_retention', True),
                key=f"{config_key}_include_data_retention",
                help="Analyze data retention after anonymization"
            )
            
            # Privacy Risk Threshold
            privacy_risk_threshold = st.slider(
                "Utility Risk Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get('privacy_risk_threshold', 0.1),
                step=0.05,
                key=f"{config_key}_privacy_risk_threshold",
                help="Threshold for acceptable utility loss"
            )
        
        # Create new config with updated values
        new_config = {
            'chart_type': chart_type,
            'color_palette': color_palette,
            'chart_height': chart_height,
            'show_values': show_values,
            'ppml_primary_metric': ppml_primary_metric,
            'show_privacy_budget': show_privacy_budget,
            'include_data_retention': include_data_retention,
            'privacy_risk_threshold': privacy_risk_threshold
        }
        
        # Check if any configuration changed
        for key, value in new_config.items():
            if current_config.get(key) != value:
                config_changed = True
                break
        
        # Update session state with new configuration
        st.session_state[config_key].update(new_config)
        
        st.markdown("---")
        
        # Real-time update controls
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Apply Changes", type="primary", help="Apply configuration changes to all tabs"):
                st.rerun()
        
        with col_btn2:
            if st.button("↩️ Reset to Defaults", help="Reset all settings to default values"):
                default_config = {
                    'chart_type': 'bar',
                    'color_palette': 'viridis', 
                    'chart_height': 600,
                    'show_values': True,
                    'ppml_primary_metric': 'f1_score',
                    'show_privacy_budget': True,
                    'include_data_retention': True,
                    'privacy_risk_threshold': 0.1,
                    'performance_metrics': ['accuracy', 'f1_score'],
                    'normalize_metrics': False
                }
                st.session_state[config_key] = default_config
                st.success("✅ Configuration reset to defaults!")
                st.rerun()
        
        with col_btn3:
            if st.button("📋 Export Config", help="Export current configuration"):
                st.json(st.session_state[config_key])
        
        # Show current configuration status
        if config_changed:
            st.success("✅ Configuration updated! Changes will apply to all tabs.")
            st.info("💡 **Tip**: Switch to other tabs to see the updated visualizations with new settings.")
        else:
            st.info("💡 **Tip**: Make changes above and they will instantly apply to all analysis tabs.")
        
        return config_changed

    # --- Add STUB IMPLEMENTATIONS for missing methods ---
    def _organize_ppml_results(self, model_results: List[Dict[str, Any]]) -> Dict[str, Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]]:
        """Organize model results by dataset and anonymization method for PPML analysis."""
        self.logger.info("AdvancedMLAnalysisPlugin._organize_ppml_results called")
        
        ppml_data: Dict[
            str, 
            Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]
        ] = {
            'original': [],
            'anonymized': {}  # Stores anonymized results keyed by method
        }
        
        anonymized_results_by_method: Dict[str, List[Dict[str, Any]]] = {}

        for result in model_results:
            if not isinstance(result, dict):
                self.logger.warning(f"Skipping non-dict result: {type(result)}")
                continue
                
            if "error" in result:
                self.logger.info(f"Skipping result with error: {result.get('model_name', 'Unknown Model')}")
                continue
                
            dataset_type = result.get('dataset_type', 'unknown').lower()
            dataset_name = result.get('dataset_name', 'Unknown Dataset') # Default if not present
            
            if dataset_type == 'original':
                (ppml_data['original']).append(result)
            elif dataset_type == 'anonymized':
                # Use 'anonymization_method' if available, otherwise fall back to 'dataset_name'
                # This aligns with how PPMLComparisonVisualization handles it.
                anon_method = result.get('anonymization_method', dataset_name)
                if anon_method == 'Unknown Dataset' and 'anonymized' in dataset_name.lower(): # Basic fallback
                    anon_method = dataset_name 

                if anon_method not in anonymized_results_by_method:
                    anonymized_results_by_method[anon_method] = []
                anonymized_results_by_method[anon_method].append(result)
            else:
                self.logger.warning(f"Unknown dataset_type '{dataset_type}' for model {result.get('model_name', 'N/A')}")

        ppml_data['anonymized'] = anonymized_results_by_method
        
        if not ppml_data['original'] and not ppml_data['anonymized']:
            self.logger.warning("No original or anonymized data identified in _organize_ppml_results.")
        elif not ppml_data['anonymized']:
            self.logger.info("Original data found, but no anonymized data identified.")
        elif not ppml_data['original']:
            self.logger.info("Anonymized data found, but no original data identified.")
        else:
            self.logger.info(f"Organized PPML data: {len(ppml_data['original'])} original, {len(ppml_data['anonymized'])} anonymized methods.")
            
        return ppml_data
    def _render_privacy_utility_analysis(self, ppml_results: Dict[str, Any], config: Dict[str, Any], chart_type: str) -> bool:
        """Render professional privacy-utility trade-off analysis."""
        
        # Extract and prepare data
        scatter_data = self._prepare_privacy_utility_data(ppml_results, config)
        
        if scatter_data.empty:
            st.warning("⚠️ Insufficient data for privacy-utility analysis.")
            return False
        
        # Configuration controls (minimal, essential only)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 🔒 Privacy-Utility Trade-off Analysis")
        
        with col2:
            show_pareto = st.checkbox(
                "Show Pareto Frontier", 
                value=True,
                key="privacy_utility_show_pareto",  # Add unique key
                help="Highlight optimal privacy-utility combinations"
            )
        
        # Create visualization
        if chart_type == 'scatter':
            fig = self._create_privacy_utility_scatter(scatter_data, config, show_pareto)
        elif chart_type == 'bubble':
            fig = self._create_privacy_utility_bubble(scatter_data, config, show_pareto)
        else:
            # Fallback to scatter for unsupported types
            fig = self._create_privacy_utility_scatter(scatter_data, config, show_pareto)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights (concise summary)
        self._display_privacy_utility_insights(scatter_data)
        
        return True

    def _prepare_privacy_utility_data(self, ppml_results: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare clean data for privacy-utility analysis."""
        
        primary_metric = config.get('ppml_primary_metric', 'f1_score')
        scatter_rows = []
        
        # Process original data as baseline
        for result in ppml_results.get('original', []):
            if primary_metric in result:
                scatter_rows.append({
                    'method': 'Original',
                    'utility_score': float(result[primary_metric]),
                    'privacy_level': 0.0,  # No privacy protection
                    'data_retention': 100.0,  # Full data retention
                    'dataset_type': 'Original',
                    'model_name': result.get('model_name', 'Unknown')
                })
        
        # Process anonymized data
        baseline_utility = 0.0
        if scatter_rows:
            baseline_utility = max(row['utility_score'] for row in scatter_rows)
        
        for method, results in ppml_results.get('anonymized', {}).items():
            for result in results:
                if primary_metric in result:
                    utility = float(result[primary_metric])
                    
                    # Calculate privacy level (simplified heuristic)
                    privacy_level = self._estimate_privacy_level(method, result, baseline_utility, utility)
                    
                    # Calculate data retention
                    data_retention = self._estimate_data_retention(result)
                    
                    scatter_rows.append({
                        'method': method,
                        'utility_score': utility,
                        'privacy_level': privacy_level,
                        'data_retention': data_retention,
                        'dataset_type': 'Anonymized',
                        'model_name': result.get('model_name', 'Unknown')
                    })
        
        return pd.DataFrame(scatter_rows)

    def _create_privacy_utility_scatter(self, data: pd.DataFrame, config: Dict[str, Any], show_pareto: bool) -> go.Figure:
        """Create professional privacy-utility scatter plot."""
        
        fig = go.Figure()
        
        # Color mapping for methods
        colors = self.color_palettes.get(config.get('color_palette', 'viridis'))
        unique_methods = data['method'].unique()
        color_map = {method: colors[i % len(colors)] for i, method in enumerate(unique_methods)}
        
        # Add scatter points for each method
        for method in unique_methods:
            method_data = data[data['method'] == method]
            
            # Special styling for original data
            if method == 'Original':
                marker_symbol = 'diamond'
                marker_size = 12
                marker_line = dict(width=2, color='black')
            else:
                marker_symbol = 'circle'
                marker_size = 10
                marker_line = dict(width=1, color='white')
            
            fig.add_trace(go.Scatter(
                x=method_data['privacy_level'],
                y=method_data['utility_score'],
                mode='markers',
                name=method,
                marker=dict(
                    color=color_map[method],
                    size=marker_size,
                    symbol=marker_symbol,
                    line=marker_line,
                    opacity=0.8
                ),
                hovertemplate=
                    f"<b>{method}</b><br>" +
                    "Privacy Level: %{x:.3f}<br>" +
                    "Utility Score: %{y:.4f}<br>" +
                    "Data Retention: %{customdata:.1f}%<extra></extra>",
                customdata=method_data['data_retention']
            ))
        
        # Add Pareto frontier if requested
        if show_pareto and len(data) > 1:
            pareto_points = self._calculate_pareto_frontier(data)
            if len(pareto_points) > 1:
                fig.add_trace(go.Scatter(
                    x=pareto_points['privacy_level'],
                    y=pareto_points['utility_score'],
                    mode='lines',
                    name='Pareto Frontier',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        # Professional layout
        fig.update_layout(
            title="Privacy-Utility Trade-off Analysis",
            xaxis_title="Privacy Protection Level",
            yaxis_title=f"Utility Score ({config.get('ppml_primary_metric', 'f1_score').upper()})",
            height=config.get('chart_height', 600),
            showlegend=True,
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig

    def _create_privacy_utility_bubble(self, data: pd.DataFrame, config: Dict[str, Any], show_pareto: bool) -> go.Figure:
        """Create bubble chart with data retention as bubble size."""
        
        fig = go.Figure()
        
        colors = self.color_palettes.get(config.get('color_palette', 'viridis'))
        unique_methods = data['method'].unique()
        color_map = {method: colors[i % len(colors)] for i, method in enumerate(unique_methods)}
        
        for method in unique_methods:
            method_data = data[data['method'] == method]
            
            # Scale bubble size based on data retention
            bubble_sizes = method_data['data_retention'] * 0.5  # Scale for visibility
            
            fig.add_trace(go.Scatter(
                x=method_data['privacy_level'],
                y=method_data['utility_score'],
                mode='markers',
                name=method,
                marker=dict(
                    color=color_map[method],
                    size=bubble_sizes,
                    sizemode='diameter',
                    sizemin=8,
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                hovertemplate=
                    f"<b>{method}</b><br>" +
                    "Privacy Level: %{x:.3f}<br>" +
                    "Utility Score: %{y:.4f}<br>" +
                    "Data Retention: %{customdata:.1f}%<extra></extra>",
                customdata=method_data['data_retention']
            ))
        
        # Add Pareto frontier if requested
        if show_pareto:
            pareto_points = self._calculate_pareto_frontier(data)
            if len(pareto_points) > 1:
                fig.add_trace(go.Scatter(
                    x=pareto_points['privacy_level'],
                    y=pareto_points['utility_score'],
                    mode='lines',
                    name='Pareto Frontier',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title="Privacy-Utility Trade-off (Bubble size = Data Retention)",
            xaxis_title="Privacy Protection Level",
            yaxis_title=f"Utility Score ({config.get('ppml_primary_metric', 'f1_score').upper()})",
            height=config.get('chart_height', 600),
            showlegend=True,
            template="plotly_white",
            font=dict(size=12)
        )
        
        return fig

    def _calculate_pareto_frontier(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pareto frontier for privacy-utility trade-off."""
        
        # Sort by privacy level
        sorted_data = data.sort_values('privacy_level')
        pareto_points = []
        
        max_utility = -float('inf')
        
        for _, row in sorted_data.iterrows():
            if row['utility_score'] > max_utility:
                max_utility = row['utility_score']
                pareto_points.append(row)
        
        return pd.DataFrame(pareto_points)

    def _estimate_privacy_level(self, method: str, result: Dict[str, Any], baseline_utility: float, current_utility: float) -> float:
        """Estimate privacy level using method-specific heuristics."""
        
        # Simple utility-based privacy estimation
        if baseline_utility > 0:
            utility_loss = max(0, (baseline_utility - current_utility) / baseline_utility)
            base_privacy = min(utility_loss * 2, 1.0)  # Scale utility loss to privacy level
        else:
            base_privacy = 0.5  # Default moderate privacy
        
        # Method-specific adjustments
        method_lower = method.lower()
        
        if 'differential' in method_lower:
            # Extract epsilon if available
            epsilon = result.get('epsilon', result.get('privacy_budget', 1.0))
            return min(1.0 / max(epsilon, 0.1), 1.0)
        
        elif 'k-anonymity' in method_lower or 'k_anonymity' in method_lower:
            k_value = result.get('k', 5)
            return min(k_value / 20.0, 1.0)  # Normalize k-value
        
        elif 'l-diversity' in method_lower:
            l_value = result.get('l', 2)
            return min((l_value + base_privacy) / 3.0, 1.0)
        
        else:
            return base_privacy

    def _estimate_data_retention(self, result: Dict[str, Any]) -> float:
        """Estimate data retention percentage."""
        
        # Check if explicitly provided
        if 'data_retention' in result:
            return float(result['data_retention'])
        
        # Estimate based on available information
        sample_size = result.get('sample_size', result.get('n_samples', 1000))
        original_size = result.get('original_size', sample_size)
        
        if original_size > 0:
            return (sample_size / original_size) * 100
        
        return 95.0  # Default assumption

    def _display_privacy_utility_insights(self, data: pd.DataFrame) -> None:
        """Display concise key insights."""
        
        if data.empty:
            return
        
        # Calculate key metrics
        anonymized_data = data[data['dataset_type'] == 'Anonymized']
        
        if anonymized_data.empty:
            st.info("📊 Only baseline data available - upload anonymized datasets for comparison.")
            return
        
        # Find optimal trade-offs
        best_utility = anonymized_data.loc[anonymized_data['utility_score'].idxmax()]
        best_privacy = anonymized_data.loc[anonymized_data['privacy_level'].idxmax()]
        
        # Balanced trade-off (closest to top-right corner)
        anonymized_data_norm = anonymized_data.copy()
        anonymized_data_norm['composite_score'] = (
            anonymized_data_norm['utility_score'] / anonymized_data_norm['utility_score'].max() +
            anonymized_data_norm['privacy_level'] / anonymized_data_norm['privacy_level'].max()
        ) / 2
        
        balanced = anonymized_data_norm.loc[anonymized_data_norm['composite_score'].idxmax()]
        
        # Display insights in clean format
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎯 Best Utility",
                f"{best_utility['utility_score']:.3f}",
                f"{best_utility['method']}"
            )
        
        with col2:
            st.metric(
                "🔒 Best Privacy",
                f"{best_privacy['privacy_level']:.3f}",
                f"{best_privacy['method']}"
            )
        
        with col3:
            st.metric(
                "⚖️ Best Balance",
                f"{balanced['composite_score']:.3f}",
                f"{balanced['method']}"
            )
        
        # Single key recommendation
        if balanced['method'] == best_utility['method']:
            st.success(f"💡 **Recommendation**: {balanced['method']} offers excellent utility with reasonable privacy protection.")
        else:
            st.info(f"💡 **Recommendation**: Consider {balanced['method']} for balanced privacy-utility trade-off.")

    def _render_utility_degradation_analysis_dashboard(self, ppml_results: Dict[str, Any], config: Dict[str, Any], chart_type: str) -> bool:
        """Render utility degradation analysis for dashboard (with unique keys)."""
        
        # Prepare degradation data
        degradation_data = self._prepare_utility_degradation_data(ppml_results, config)
        
        if degradation_data.empty:
            st.warning("⚠️ Insufficient data for utility degradation analysis.")
            return False
        
        # Configuration controls with unique keys for dashboard
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_baseline = st.checkbox(
                "Show Baseline", 
                value=True,
                key="dashboard_utility_degradation_show_baseline",  # Unique key for dashboard
                help="Display original dataset performance as reference"
            )
        
        # Create visualization
        if chart_type == 'bar':
            fig = self._create_degradation_bar_chart(degradation_data, config, show_baseline)
        elif chart_type == 'line':
            fig = self._create_degradation_line_chart(degradation_data, config, show_baseline)
        elif chart_type == 'horizontal_bar':
            fig = self._create_degradation_horizontal_bar(degradation_data, config, show_baseline)
        else:
            # Fallback to bar chart
            fig = self._create_degradation_bar_chart(degradation_data, config, show_baseline)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        self._display_degradation_insights(degradation_data)
        
        return True

    def _prepare_utility_degradation_data(self, ppml_results: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare utility degradation data for analysis."""
        
        primary_metric = config.get('ppml_primary_metric', 'f1_score')
        degradation_rows = []
        
        # Calculate baseline performance from original data
        baseline_performance = 0.0
        original_results = ppml_results.get('original', [])
        
        if original_results:
            baseline_scores = [result.get(primary_metric, 0) for result in original_results if primary_metric in result]
            if baseline_scores:
                baseline_performance = np.mean(baseline_scores)
        
        # Process anonymized results
        for method, results in ppml_results.get('anonymized', {}).items():
            for result in results:
                if primary_metric in result:
                    current_performance = float(result[primary_metric])
                    
                    # Calculate utility loss
                    if baseline_performance > 0:
                        utility_loss = baseline_performance - current_performance
                        utility_loss_percent = (utility_loss / baseline_performance) * 100
                    else:
                        utility_loss = 0
                        utility_loss_percent = 0
                    
                    degradation_rows.append({
                        'method': method,
                        'model_name': result.get('model_name', 'Unknown'),
                        'baseline_performance': baseline_performance,
                        'current_performance': current_performance,
                        'utility_loss': utility_loss,
                        'utility_loss_percent': utility_loss_percent,
                        'retained_utility': (current_performance / baseline_performance * 100) if baseline_performance > 0 else 100
                    })
        
        return pd.DataFrame(degradation_rows)

    def _create_degradation_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any], show_baseline: bool) -> go.Figure:
        """Create professional degradation bar chart."""
        
        fig = go.Figure()
        
        # Group by method and calculate averages
        method_data = data.groupby('method').agg({
            'baseline_performance': 'mean',
            'current_performance': 'mean',
            'utility_loss_percent': 'mean',
            'retained_utility': 'mean'
        }).reset_index()
        
        colors = self.color_palettes.get(config.get('color_palette', 'viridis'))
        
        # Show baseline if requested
        if show_baseline and not method_data.empty:
            baseline_avg = method_data['baseline_performance'].iloc[0]
            fig.add_hline(
                y=baseline_avg, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Baseline: {baseline_avg:.3f}",
                annotation_position="top right"
            )
        
        # Add current performance bars
        fig.add_trace(go.Bar(
            name='Current Performance',
            x=method_data['method'],
            y=method_data['current_performance'],
            marker_color=colors[0],
            text=[f"{val:.3f}" for val in method_data['current_performance']],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Performance: %{y:.4f}<br>Loss: %{customdata:.1f}%<extra></extra>",
            customdata=method_data['utility_loss_percent']
        ))
        
        # Update layout
        fig.update_layout(
            title="Utility Degradation by Method",
            xaxis_title="Anonymization Method",
            yaxis_title=f"Performance ({config.get('ppml_primary_metric', 'f1_score').upper()})",
            height=config.get('chart_height', 600),
            template="plotly_white",
            font=dict(size=12),
            showlegend=True
        )
        
        return fig

    def _create_degradation_line_chart(self, data: pd.DataFrame, config: Dict[str, Any], show_baseline: bool) -> go.Figure:
        """Create degradation line chart."""
        
        fig = go.Figure()
        
        # Group data by method
        colors = self.color_palettes.get(config.get('color_palette', 'viridis'))
        
        for i, method in enumerate(data['method'].unique()):
            method_data = data[data['method'] == method]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(method_data))),
                y=method_data['current_performance'],
                mode='lines+markers',
                name=method,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>{method}</b><br>Performance: %{{y:.4f}}<br>Model: %{{customdata}}<extra></extra>",
                customdata=method_data['model_name']
            ))
        
        # Add baseline if requested
        if show_baseline and not data.empty:
            baseline_avg = data['baseline_performance'].iloc[0]
            fig.add_hline(
                y=baseline_avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Baseline: {baseline_avg:.3f}"
            )
        
        fig.update_layout(
            title="Performance Degradation Over Methods",
            xaxis_title="Method Index",
            yaxis_title=f"Performance ({config.get('ppml_primary_metric', 'f1_score').upper()})",
            height=config.get('chart_height', 600),
            template="plotly_white"
        )
        
        return fig

    def _create_degradation_horizontal_bar(self, data: pd.DataFrame, config: Dict[str, Any], show_baseline: bool) -> go.Figure:
        """Create horizontal degradation bar chart."""
        
        fig = go.Figure()
        
        # Group by method
        method_data = data.groupby('method').agg({
            'utility_loss_percent': 'mean',
            'current_performance': 'mean'
        }).reset_index().sort_values('utility_loss_percent')
        
        colors = self.color_palettes.get(config.get('color_palette', 'viridis'))
        
        # Color code by degradation level
        bar_colors = []
        for loss in method_data['utility_loss_percent']:
            if loss < 5:
                bar_colors.append('#2E8B57')  # Good (green)
            elif loss < 15:
                bar_colors.append('#FF8C00')  # Moderate (orange)
            else:
                bar_colors.append('#DC143C')  # High (red)
        
        fig.add_trace(go.Bar(
            x=method_data['utility_loss_percent'],
            y=method_data['method'],
            orientation='h',
            marker_color=bar_colors,
            text=[f"{val:.1f}%" for val in method_data['utility_loss_percent']],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Utility Loss: %{x:.2f}%<br>Current Performance: %{customdata:.4f}<extra></extra>",
            customdata=method_data['current_performance']
        ))
        
        fig.update_layout(
            title="Utility Loss by Method (Lower is Better)",
            xaxis_title="Utility Loss (%)",
            yaxis_title="Anonymization Method",
            height=config.get('chart_height', 600),
            template="plotly_white",
            font=dict(size=12)
        )
        
        return fig

    def _display_degradation_insights(self, data: pd.DataFrame) -> None:
        """Display concise degradation insights."""
        
        if data.empty:
            return
        
        # Calculate key metrics
        method_stats = data.groupby('method').agg({
            'utility_loss_percent': 'mean',
            'retained_utility': 'mean',
            'current_performance': 'mean'
        }).round(3)
        
        # Find best and worst methods
        best_method = method_stats['utility_loss_percent'].idxmin()
        worst_method = method_stats['utility_loss_percent'].idxmax()
        
        best_loss = method_stats.loc[best_method, 'utility_loss_percent']
        worst_loss = method_stats.loc[worst_method, 'utility_loss_percent']
        
        # Display insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏆 Best Method",
                best_method,
                f"-{best_loss:.1f}% loss",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "⚠️ Highest Loss",
                worst_method,
                f"-{worst_loss:.1f}% loss",
                delta_color="normal"
            )
        
        with col3:
            avg_loss = data['utility_loss_percent'].mean()
            st.metric(
                "📊 Average Loss",
                f"{avg_loss:.1f}%",
                f"Across {len(method_stats)} methods"
            )
        
        # Single key recommendation
        if best_loss < 10:
            st.success(f"💡 **Recommendation**: {best_method} maintains excellent utility with minimal loss ({best_loss:.1f}%)")
        elif best_loss < 20:
            st.info(f"💡 **Recommendation**: {best_method} offers reasonable utility preservation ({best_loss:.1f}% loss)")
        else:
            st.warning(f"⚠️ **Note**: All methods show significant utility loss. Consider parameter tuning or alternative approaches.")

    def _render_method_comparison_analysis(self, ppml_results: Dict[str, List[Any]], config: Dict[str, Any], chart_type: str) -> bool:
        st.info("🏆 Method Comparison Analysis: Content not yet implemented.")
        self.logger.info("AdvancedMLAnalysisPlugin._render_method_comparison_analysis called (stub implementation)")
        return True

    def _render_robustness_analysis(self, ppml_results: Dict[str, List[Any]], config: Dict[str, Any], chart_type: str) -> bool:
        st.info("🛡️ Model Robustness Analysis: Content not yet implemented.")
        self.logger.info("AdvancedMLAnalysisPlugin._render_robustness_analysis called (stub implementation)")
        return True

    def _render_alternative_model_comparison(self, model_results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        st.info("📊 Alternative Model Comparison (for non-PPML data): Content not yet implemented.")
        self.logger.info("AdvancedMLAnalysisPlugin._render_alternative_model_comparison called (stub implementation)")
        return True
    # --- End of STUB IMPLEMENTATIONS ---

    def _render_performance_analysis(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Render performance analysis with flexible chart types."""
        st.subheader("🎯 Performance Analysis")
        
        # Get configuration from session state for reactive updates
        chart_type = config.get('chart_type', 'bar')
        
        # Performance analysis configuration
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            # Use session state key for persistence
            selected_metrics = st.multiselect(
                "📈 Metrics to Compare:",
                options=self.common_metrics,
                default=config.get('performance_metrics', ['accuracy', 'f1_score']),
                key="perf_analysis_metrics",
                help="Select metrics for performance comparison"
            )
            
            # Show current chart type from config
            st.info(f"📊 **Active Chart Type**: {self.chart_types.get(chart_type, 'Bar Chart')}")
            
        with perf_col2:
            normalize_metrics = st.checkbox(
                "⚖️ Normalize Metrics",
                value=config.get('normalize_metrics', False),
                key="perf_analysis_normalize",
                help="Scale metrics to 0-1 range"
            )
            
            highlight_best = st.checkbox(
                "🏆 Highlight Best Model",
                value=True,
                key="perf_analysis_highlight",
                help="Highlight top performer"
            )
        
        if not selected_metrics:
            st.warning("⚠️ Please select at least one metric for analysis.")
            return False
        
        # Update session state with current selections
        config_key = "advanced_ml_analysis_config"
        if config_key in st.session_state:
            st.session_state[config_key]['performance_metrics'] = selected_metrics
            st.session_state[config_key]['normalize_metrics'] = normalize_metrics
        
        # Prepare performance data
        viz_data = self._prepare_performance_data(model_results, selected_metrics, normalize_metrics)
        
        if viz_data.empty:
            st.warning("⚠️ No valid performance data found.")
            return False
        
        # Create performance visualization with current config
        success = self._create_flexible_chart(
            viz_data, chart_type, selected_metrics, config, 
            title_prefix="Performance Comparison",
            highlight_best=highlight_best
        )
        
        if success:
            # Professional summary table
            self._display_performance_summary_table(viz_data, selected_metrics, highlight_best)
        
        return success
    
    def _render_ppml_analysis(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Render PPML analysis with flexible chart types."""
        st.subheader("🔒 PPML Analysis")
        
        # Check if we have PPML data
        ppml_results = self._organize_ppml_results(model_results)
        
        if not ppml_results or not ppml_results['anonymized']:
            st.warning("⚠️ No PPML data found. Need results from anonymized datasets.")
            st.info("💡 **Tip**: Upload anonymized datasets and train models to enable PPML analysis.")
            return False
        
        # PPML analysis options
        ppml_col1, ppml_col2 = st.columns(2)
        
        with ppml_col1:
            ppml_analysis_type = st.selectbox(
                "🔒 PPML Analysis Type:",
                options=[
                    "Privacy-Utility Scatter",
                    "Utility Degradation", 
                    "Method Comparison",
                    "Model Robustness"
                ],
                help="Select PPML analysis focus"
            )
            
        with ppml_col2:
            chart_type = config.get('chart_type', 'scatter')
            
            # Override chart type for specific PPML analyses
            if ppml_analysis_type == "Privacy-Utility Scatter":
                chart_type = st.selectbox("Chart Type:", ['scatter', 'bubble'], index=0)
            elif ppml_analysis_type == "Utility Degradation":
                chart_type = st.selectbox("Chart Type:", ['bar', 'horizontal_bar', 'line'], index=0)
            elif ppml_analysis_type == "Method Comparison":
                chart_type = st.selectbox("Chart Type:", ['radar', 'heatmap', 'bar'], index=0)
            elif ppml_analysis_type == "Model Robustness":
                chart_type = st.selectbox("Chart Type:", ['scatter', 'box', 'violin'], index=0)
        
        # Render selected PPML analysis
        if ppml_analysis_type == "Privacy-Utility Scatter":
            return self._render_privacy_utility_analysis(ppml_results, config, chart_type)
        elif ppml_analysis_type == "Utility Degradation":
            return self._render_utility_degradation_analysis(ppml_results, config, chart_type)
        elif ppml_analysis_type == "Method Comparison":
            return self._render_method_comparison_analysis(ppml_results, config, chart_type)
        elif ppml_analysis_type == "Model Robustness":
            return self._render_robustness_analysis(ppml_results, config, chart_type)
        
        return False
    
    def _render_comprehensive_dashboard(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """Render comprehensive dashboard with all analyses."""
        st.subheader("📊 Comprehensive Dashboard")
        
        # Check what type of data we have
        has_performance_data = len(model_results) >= 1
        ppml_results = self._organize_ppml_results(model_results)
        has_ppml_data = ppml_results and ppml_results['anonymized']
        
        if not has_performance_data:
            st.error("❌ No performance data available for comprehensive analysis.")
            return False
        
        # Create sub-tabs for comprehensive analysis
        if has_ppml_data:
            subtab1, subtab2, subtab3, subtab4 = st.tabs([
                "📈 Performance Overview",
                "🔒 Privacy-Utility Trade-off", 
                "📉 Utility Analysis",
                "🏆 Method Rankings"
            ])
        else:
            subtab1, subtab2 = st.tabs([
                "📈 Performance Overview",
                "📊 Model Comparison"
            ])
        
        with subtab1:
            st.markdown("#### 📈 Performance Overview")
            selected_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            available_metrics = [m for m in selected_metrics if any(m in result for result in model_results)]
            
            if available_metrics:
                viz_data = self._prepare_performance_data(model_results, available_metrics, False)
                if not viz_data.empty:
                    self._create_flexible_chart(
                        viz_data, config.get('chart_type', 'bar'), available_metrics, config,
                        title_prefix="Overall Performance"
                    )
                    self._display_performance_summary_table(viz_data, available_metrics)
        
        with subtab2:
            if has_ppml_data:
                st.markdown("#### 🔒 Privacy-Utility Trade-off")
                # Call with unique context to avoid checkbox ID conflicts
                self._render_privacy_utility_analysis_dashboard(ppml_results, config, 'scatter')
            else:
                st.markdown("#### 📊 Model Comparison")
                # Alternative model comparison for non-PPML data
                self._render_alternative_model_comparison(model_results, config)
        
        if has_ppml_data:
            with subtab3:
                st.markdown("#### 📉 Utility Analysis")
                # Call with unique context to avoid checkbox ID conflicts
                self._render_utility_degradation_analysis_dashboard(ppml_results, config, 'bar')
            
            with subtab4:
                st.markdown("#### 🏆 Method Rankings")
                self._render_method_comparison_analysis(ppml_results, config, 'radar')
        
        return True

    
    def _prepare_performance_data(self, model_results: List[Dict[str, Any]], 
                                selected_metrics: List[str], normalize_metrics: bool) -> pd.DataFrame:
        """Prepare data for performance visualization."""
        viz_rows = []
        
        for idx, result in enumerate(model_results):
            if "error" in result:
                continue
            
            # Create model identifier
            model_name = result.get('model_name', f'Model {idx + 1}')
            target = result.get('target_column', 'Unknown')
            dataset_name = result.get('dataset_name', 'Unknown')
            
            # Calculate run number for this model
            run_number = sum(1 for i, r in enumerate(model_results[:idx+1]) 
                           if r.get('model_name') == result.get('model_name') 
                           and r.get('target_column') == result.get('target_column')
                           and "error" not in r)
            
            # Create comprehensive model identifier
            if dataset_name and dataset_name != 'Unknown':
                model_id = f"{model_name} ({dataset_name}) - Run #{run_number}"
            else:
                model_id = f"{model_name} - Run #{run_number}"
            
            # Extract metrics
            row_data = {
                'Model': model_id, 
                'Model_Name': model_name, 
                'Dataset': dataset_name,
                'Run': run_number
            }
            
            for metric in selected_metrics:
                value = result.get(metric)
                if value is not None:
                    row_data[metric] = float(value)
                else:
                    # Check in custom_metrics
                    custom_metrics = result.get('custom_metrics', {})
                    if metric in custom_metrics:
                        row_data[metric] = float(custom_metrics[metric])
            
            viz_rows.append(row_data)
        
        if not viz_rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(viz_rows)
        
        # Normalize metrics if requested
        if normalize_metrics:
            for metric in selected_metrics:
                if metric in df.columns:
                    min_val = df[metric].min()
                    max_val = df[metric].max()
                    if max_val > min_val:
                        df[f'{metric}_normalized'] = (df[metric] - min_val) / (max_val - min_val)
                    else:
                        df[f'{metric}_normalized'] = 1.0
        
        return df
    
    def _create_flexible_chart(self, viz_data: pd.DataFrame, chart_type: str, 
                             selected_metrics: List[str], config: Dict[str, Any],
                             title_prefix: str = "Analysis", highlight_best: bool = False) -> bool:
        """Create flexible charts that work for both performance and PPML analysis."""
        try:
            colors = self.color_palettes.get(config.get('color_palette', 'viridis'), px.colors.qualitative.Set1)
            chart_height = config.get('chart_height', 600)
            show_values = config.get('show_values', True)
            
            # Calculate best performers for highlighting if requested
            best_performers = {}
            if highlight_best:
                best_performers = self._identify_best_performers(viz_data, selected_metrics)
            
            if chart_type == 'bar':
                fig = self._create_unified_bar_chart(viz_data, selected_metrics, colors, show_values, best_performers)
            elif chart_type == 'horizontal_bar':
                fig = self._create_unified_horizontal_bar_chart(viz_data, selected_metrics, colors, show_values, best_performers)
            elif chart_type == 'line':
                fig = self._create_unified_line_chart(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'scatter':
                fig = self._create_unified_scatter_chart(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'radar':
                fig = self._create_unified_radar_chart(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'heatmap':
                fig = self._create_unified_heatmap(viz_data, selected_metrics, best_performers)
            elif chart_type == 'box':
                fig = self._create_unified_box_plot(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'violin':
                fig = self._create_unified_violin_plot(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'bubble':
                fig = self._create_unified_bubble_chart(viz_data, selected_metrics, colors, best_performers)
            elif chart_type == 'area':
                fig = self._create_unified_area_chart(viz_data, selected_metrics, colors, best_performers)
            else:
                st.error(f"❌ Unsupported chart type: {chart_type}")
                return False
            
            # Update layout
            fig.update_layout(
                height=chart_height,
                title=f"{title_prefix} - {self.chart_types[chart_type]}",
                title_x=0.5,
                showlegend=True,
                template="plotly_white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating {chart_type} chart: {str(e)}")
            st.error(f"❌ Error creating {chart_type} chart: {str(e)}")
            return False
    
    def _create_unified_bar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], 
                                colors: List[str], show_values: bool, best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified bar chart for any analysis type."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            # Prepare colors for highlighting
            if best_performers and metric in best_performers:
                # Create color array for highlighting best performer
                bar_colors = []
                best_idx = best_performers[metric]['index']
                base_color = colors[i % len(colors)]
                
                for idx in viz_data.index:
                    if idx == best_idx:
                        # Gold color for best performer
                        bar_colors.append('#FFD700')
                    else:
                        # Regular color with reduced opacity
                        bar_colors.append(base_color)
                        
                # Add border to best performer
                marker_line = dict(
                    color=['#B8860B' if idx == best_idx else 'rgba(0,0,0,0)' for idx in viz_data.index],
                    width=[3 if idx == best_idx else 0 for idx in viz_data.index]
                )
            else:
                bar_colors = colors[i % len(colors)]
                marker_line = None
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=viz_data.get('Model', viz_data.index),
                y=viz_data[metric],
                text=viz_data[metric].round(4) if show_values else None,
                textposition='auto',
                marker=dict(
                    color=bar_colors,
                    line=marker_line
                ) if marker_line else dict(color=bar_colors),
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            xaxis_title="Models/Methods",
            yaxis_title="Performance Score",
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_unified_horizontal_bar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], 
                                           colors: List[str], show_values: bool, best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified horizontal bar chart."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            # Prepare colors for highlighting
            if best_performers and metric in best_performers:
                bar_colors = []
                best_idx = best_performers[metric]['index']
                base_color = colors[i % len(colors)]
                
                for idx in viz_data.index:
                    if idx == best_idx:
                        bar_colors.append('#FFD700')
                    else:
                        bar_colors.append(base_color)
                        
                marker_line = dict(
                    color=['#B8860B' if idx == best_idx else 'rgba(0,0,0,0)' for idx in viz_data.index],
                    width=[3 if idx == best_idx else 0 for idx in viz_data.index]
                )
            else:
                bar_colors = colors[i % len(colors)]
                marker_line = None
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                y=viz_data.get('Model', viz_data.index),
                x=viz_data[metric],
                text=viz_data[metric].round(4) if show_values else None,
                textposition='auto',
                marker=dict(
                    color=bar_colors,
                    line=marker_line
                ) if marker_line else dict(color=bar_colors),
                orientation='h',
                hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            yaxis_title="Models/Methods",
            xaxis_title="Performance Score",
            barmode='group'
        )
        
        return fig
    
    def _create_unified_line_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified line chart."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            # Highlight best performer with larger marker
            marker_sizes = [12 if best_performers and metric in best_performers and idx == best_performers[metric]['index'] 
                          else 8 for idx in viz_data.index]
            marker_colors = ['#FFD700' if best_performers and metric in best_performers and idx == best_performers[metric]['index'] 
                           else color for idx in viz_data.index]
            
            fig.add_trace(go.Scatter(
                name=metric.replace('_', ' ').title(),
                x=viz_data.get('Model', viz_data.index),
                y=viz_data[metric],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=marker_sizes, color=marker_colors),
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            xaxis_title="Models/Methods",
            yaxis_title="Performance Score",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_unified_radar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified radar chart."""
        fig = go.Figure()
        
        for i, row in viz_data.iterrows():
            values = []
            labels = []
            
            for metric in selected_metrics:
                if metric in viz_data.columns:
                    values.append(row[metric])
                    labels.append(metric.replace('_', ' ').title())
            
            # Close the radar chart
            if values:
                values.append(values[0])
                labels.append(labels[0])
                
                color = colors[i % len(colors)]
                model_name = row.get('Model', f'Entry {i+1}')
                
                # Highlight best performer with thicker line
                line_width = 4 if best_performers and any(
                    metric in best_performers and i == best_performers[metric]['index'] 
                    for metric in selected_metrics
                ) else 2
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=model_name,
                    line=dict(color=color, width=line_width),
                    opacity=0.7
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            )
        )
        
        return fig

    def _create_unified_heatmap(self, viz_data: pd.DataFrame, selected_metrics: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified heatmap."""
        # Prepare data for heatmap
        model_col = 'Model' if 'Model' in viz_data.columns else viz_data.index
        heatmap_data = viz_data.set_index(model_col)[selected_metrics].T
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            text=np.round(heatmap_data.values, 4),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Value: %{z:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis_title="Models/Methods",
            yaxis_title="Metrics"
        )
        
        return fig
    
    def _create_unified_box_plot(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified box plot."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Box(
                y=viz_data[metric],
                name=metric.replace('_', ' ').title(),
                marker_color=color
            ))
        
        fig.update_layout(
            yaxis_title="Performance Score"
        )
        
        return fig
    
    def _create_unified_violin_plot(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified violin plot."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Violin(
                y=viz_data[metric],
                name=metric.replace('_', ' ').title(),
                marker_color=color,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            yaxis_title="Performance Score"
        )
        
        return fig
    
    def _create_unified_bubble_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified bubble chart."""
        fig = go.Figure()
        
        if len(selected_metrics) >= 3:
            # Use first metric for x, second for y, third for size
            marker_sizes = viz_data[selected_metrics[2]] * 50  # Scale for visibility
            marker_colors = [colors[0]] * len(viz_data)
            
            # Highlight best performers
            if best_performers:
                for i, idx in enumerate(viz_data.index):
                    if any(metric in best_performers and idx == best_performers[metric]['index'] 
                           for metric in selected_metrics[:3]):
                        marker_colors[i] = '#FFD700'
            
            fig.add_trace(go.Scatter(
                x=viz_data[selected_metrics[0]],
                y=viz_data[selected_metrics[1]],
                mode='markers',
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    opacity=0.7
                ),
                text=viz_data.get('Model', viz_data.index),
                hovertemplate="<b>%{text}</b><br>" + 
                            f"{selected_metrics[0]}: %{{x:.4f}}<br>" +
                            f"{selected_metrics[1]}: %{{y:.4f}}<br>" +
                            f"{selected_metrics[2]}: %{{marker.size:.4f}}<extra></extra>"
            ))
            
            fig.update_layout(
                xaxis_title=selected_metrics[0].replace('_', ' ').title(),
                yaxis_title=selected_metrics[1].replace('_', ' ').title()
            )
        else:
            # Fallback to regular scatter
            return self._create_unified_scatter_chart(viz_data, selected_metrics, colors, best_performers)
        
        return fig
    
    def _create_unified_area_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str], best_performers: Dict[str, Any] = None) -> go.Figure:
        """Create unified area chart."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                name=metric.replace('_', ' ').title(),
                x=viz_data.get('Model', viz_data.index),
                y=viz_data[metric],
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=color, width=2),
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            xaxis_title="Models/Methods",
            yaxis_title="Performance Score",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _display_performance_summary_table(self, viz_data: pd.DataFrame, selected_metrics: List[str], highlight_best: bool = False) -> None:
        """Display professional summary table with optional highlighting."""
        st.markdown("### 📊 **Performance Summary**")
        
        # Calculate statistics for each metric
        summary_data = []
        for metric in selected_metrics:
            if metric in viz_data.columns:
                values = viz_data[metric]
                
                # Find best and worst performers
                best_idx = values.idxmax()
                worst_idx = values.idxmin()
                best_model = viz_data.loc[best_idx, 'Model'] if 'Model' in viz_data.columns else f"Entry {best_idx}"
                worst_model = viz_data.loc[worst_idx, 'Model'] if 'Model' in viz_data.columns else f"Entry {worst_idx}"
                
                # Add highlighting emoji if enabled
                best_performer_text = f"🏆 {best_model}" if highlight_best else best_model
                
                summary_data.append({
                    '📊 Metric': metric.replace('_', ' ').title(),
                    '🏆 Best Score': f"{values.max():.4f}",
                    '🥇 Best Performer': best_performer_text,
                    '📉 Worst Score': f"{values.min():.4f}",
                    '📊 Worst Performer': worst_model,
                    '📈 Average': f"{values.mean():.4f}",
                    '📋 Std Dev': f"{values.std():.4f}",
                    '📏 Range': f"{values.max() - values.min():.4f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Show highlighting legend if enabled
            if highlight_best:
                st.info("🏆 **Gold highlights indicate best performers for each metric**")
            
            # Additional insights
            if len(summary_data) > 1:
                best_overall_metric = max(summary_data, key=lambda x: float(x['🏆 Best Score'].replace(',', '')))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🏆 **Strongest Metric**: {best_overall_metric['📊 Metric']} (Best: {best_overall_metric['🏆 Best Score']})")
                with col2:
                    st.info(f"📊 **Most Variable Metric**: {max(summary_data, key=lambda x: float(x['📏 Range']))['📊 Metric']}")

    def _display_ppml_summary_table(self, scatter_df: pd.DataFrame, primary_metric: str) -> None:
        """Display PPML summary table."""
        st.markdown("### 🔒 **PPML Analysis Summary**")
        
        # Separate original and anonymized data
        original_df = scatter_df[scatter_df['dataset_type'] == 'Original']
        anonymized_df = scatter_df[scatter_df['dataset_type'] == 'Anonymized']
        
        summary_data = []
        
        if not original_df.empty:
            baseline_utility = original_df['utility_score'].mean()
            summary_data.append({
                '📊 Category': 'Original (Baseline)',
                '🎯 Avg Utility': f"{baseline_utility:.4f}",
                '🔒 Privacy Level': '0.0000',
                '📈 Data Retention': '100.00%',
                '🏆 Best Score': f"{original_df['utility_score'].max():.4f}",
                '📉 Worst Score': f"{original_df['utility_score'].min():.4f}"
            })
        
        # Group by anonymization method
        for method in anonymized_df['method'].unique():
            method_data = anonymized_df[anonymized_df['method'] == method]
            
            avg_utility = method_data['utility_score'].mean()
            avg_privacy = method_data['privacy_level'].mean()
            avg_retention = method_data['data_retention'].mean()
            
            summary_data.append({
                '📊 Category': method,
                '🎯 Avg Utility': f"{avg_utility:.4f}",
                '🔒 Privacy Level': f"{avg_privacy:.4f}",
                '📈 Data Retention': f"{avg_retention:.2f}%",
                '🏆 Best Score': f"{method_data['utility_score'].max():.4f}",
                '📉 Worst Score': f"{method_data['utility_score'].min():.4f}"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Privacy-utility insights
            if len(summary_data) > 1:
                best_privacy_method = max([s for s in summary_data if s['📊 Category'] != 'Original (Baseline)'], 
                                        key=lambda x: float(x['🔒 Privacy Level']), default=None)
                best_utility_method = max([s for s in summary_data if s['📊 Category'] != 'Original (Baseline)'], 
                                        key=lambda x: float(x['🎯 Avg Utility']), default=None)
                
                col1, col2 = st.columns(2)
                if best_privacy_method:
                    with col1:
                        st.info(f"🔒 **Best Privacy**: {best_privacy_method['📊 Category']} (Level: {best_privacy_method['🔒 Privacy Level']})")
                if best_utility_method:
                    with col2:
                        st.info(f"🎯 **Best Utility**: {best_utility_method['📊 Category']} (Score: {best_utility_method['🎯 Avg Utility']})")

    def _display_degradation_summary_table(self, degradation_df: pd.DataFrame) -> None:
        """Display degradation summary table."""
        st.markdown("### 📉 **Utility Degradation Summary**")
        
        # Calculate summary statistics
        summary_stats = degradation_df.groupby('method').agg({
            'utility_loss_percent': ['mean', 'min', 'max', 'std'],
            'baseline_performance': 'mean',
            'current_performance': 'mean'
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['Avg Loss %', 'Min Loss %', 'Max Loss %', 'Std Loss %', 'Avg Baseline', 'Avg Current']
        summary_stats = summary_stats.reset_index()
        
        # Add emoji columns
        summary_stats.insert(0, '🔧 Method', summary_stats['method'])
        summary_stats = summary_stats.drop('method', axis=1)
        
        st.dataframe(summary_stats, use_container_width=True, hide_index=True)
        
        # Key insights
        best_method = summary_stats.loc[summary_stats['Avg Loss %'].idxmin()]
        worst_method = summary_stats.loc[summary_stats['Avg Loss %'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Best Method**: {best_method['🔧 Method']} (Avg Loss: {best_method['Avg Loss %']:.2f}%)")
        with col2:
            st.warning(f"⚠️ **Highest Loss**: {worst_method['🔧 Method']} (Avg Loss: {worst_method['Avg Loss %']:.2f}%)")

    def _display_method_ranking_table(self, comparison_df: pd.DataFrame) -> None:
        """Display method ranking table."""
        st.markdown("### 🏆 **Method Rankings**")
        
        # Calculate composite ranking
        metrics = ['avg_accuracy', 'avg_f1_score', 'avg_precision', 'avg_recall', 'privacy_score']
        
        # Normalize all metrics to 0-1 scale for fair comparison
        normalized_df = comparison_df.copy()
        for metric in metrics:
            if metric in normalized_df.columns:
                min_val = normalized_df[metric].min()
                max_val = normalized_df[metric].max()
                if max_val > min_val:
                    normalized_df[f'{metric}_norm'] = (normalized_df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[f'{metric}_norm'] = 1.0
        
        # Calculate composite score
        norm_metrics = [f'{m}_norm' for m in metrics if f'{m}_norm' in normalized_df.columns]
        normalized_df['composite_score'] = normalized_df[norm_metrics].mean(axis=1)
        
        # Sort by composite score
        ranking_df = normalized_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        # Create display dataframe
        display_df = pd.DataFrame({
            '🏅 Rank': range(1, len(ranking_df) + 1),
            '🔧 Method': ranking_df['method'],
            '📊 Composite Score': ranking_df['composite_score'].round(4),
            '🎯 Accuracy': ranking_df['avg_accuracy'].round(4),
            '🏆 F1-Score': ranking_df['avg_f1_score'].round(4),
            '⚖️ Precision': ranking_df['avg_precision'].round(4),
            '📈 Recall': ranking_df['avg_recall'].round(4),
            '🔒 Privacy Score': ranking_df['privacy_score'].round(4)
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Top performer insight
        top_method = ranking_df.iloc[0]
        st.success(f"🥇 **Top Performer**: {top_method['method']} (Composite Score: {top_method['composite_score']:.4f})")

    def _display_robustness_ranking_table(self, robustness_df: pd.DataFrame) -> None:
        """Display robustness ranking table."""
        st.markdown("### 🛡️ **Model Robustness Rankings**")
        
        # Sort by robustness score
        ranking_df = robustness_df.sort_values('robustness_score', ascending=False).reset_index(drop=True)
        
        # Create display dataframe
        display_df = pd.DataFrame({
            '🏅 Rank': range(1, len(ranking_df) + 1),
            '🤖 Model': ranking_df['model'],
            '🛡️ Robustness Score': ranking_df['robustness_score'].round(4),
            '📊 Avg Performance': ranking_df['avg_performance'].round(4),
            '📋 Std Deviation': ranking_df['std_performance'].round(4),
            '📏 Performance Range': ranking_df['performance_range'].round(4)
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Insights
        most_robust = ranking_df.iloc[0]
        least_robust = ranking_df.iloc[-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🛡️ **Most Robust**: {most_robust['model']} (Score: {most_robust['robustness_score']:.4f})")
        with col2:
            st.warning(f"⚠️ **Least Robust**: {least_robust['model']} (Score: {least_robust['robustness_score']:.4f})")

    def _identify_best_performers(self, viz_data: pd.DataFrame, selected_metrics: List[str]) -> Dict[str, Any]:
        """Identify best performing models for each metric."""
        best_performers = {}
        
        for metric in selected_metrics:
            if metric in viz_data.columns:
                # Find the index of the best performer for this metric
                best_idx = viz_data[metric].idxmax()
                best_model = viz_data.loc[best_idx, 'Model'] if 'Model' in viz_data.columns else f"Entry {best_idx}"
                best_value = viz_data.loc[best_idx, metric]
                
                best_performers[metric] = {
                    'index': best_idx,
                    'model': best_model,
                    'value': best_value
                }
        
        return best_performers

    def _get_reactive_config(self, config_key: str = "advanced_ml_analysis_config") -> Dict[str, Any]:
        """Get current configuration from session state for reactive updates."""
        if config_key in st.session_state:
            return st.session_state[config_key]
        else:
            # Return default config if session state not initialized
            return {
                'chart_type': 'bar',
                'color_palette': 'viridis', 
                'chart_height': 600,
                'show_values': True,
                'ppml_primary_metric': 'f1_score',
                'show_privacy_budget': True,
                'include_data_retention': True,
                'privacy_risk_threshold': 0.1,
                'performance_metrics': ['accuracy', 'f1_score'],
                'normalize_metrics': False
            }

def get_plugin():
    """Factory function to create and return plugin instance."""
    return AdvancedMLAnalysisPlugin()