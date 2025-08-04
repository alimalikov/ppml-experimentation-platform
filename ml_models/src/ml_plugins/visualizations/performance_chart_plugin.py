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
    print(f"Import error in performance_chart_plugin: {e}")
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

class PerformanceChartPlugin(BaseVisualizationPlugin):
    """
    Advanced performance comparison visualization plugin.
    
    Creates interactive charts comparing model performance metrics
    across multiple models and runs with various visualization types.
    """
    
    def __init__(self):
        """Initialize the Performance Chart Plugin"""
        super().__init__()
        
        # Plugin metadata
        self.name = "Performance Comparison Chart"
        self.description = "Interactive charts comparing model performance metrics across multiple models"
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
        
        # Available chart types
        self.chart_types = {
            "bar": "Bar Chart",
            "horizontal_bar": "Horizontal Bar Chart", 
            "line": "Line Chart",
            "scatter": "Scatter Plot",
            "radar": "Radar Chart",
            "heatmap": "Heatmap",
            "box": "Box Plot",
            "violin": "Violin Plot"
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
            "categorical": px.colors.qualitative.Set1
        }
    
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
            
            # Need at least 2 model results for comparison
            if not model_results or len(model_results) < 2:
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
            
            # Need at least 2 valid results for comparison
            return valid_results >= 2
            
        except Exception as e:
            self.logger.error(f"Error in can_visualize: {str(e)}")
            return False
    
    def get_config_options(self) -> Dict[str, Dict[str, Any]]:
        """
        Return configuration options for this visualization.
        
        Returns:
            Dict containing configuration options with their properties
        """
        return {
            'chart_type': {
                'type': 'select',
                'label': '📊 Chart Type',
                'default': 'bar',
                'options': list(self.chart_types.keys()),
                'help': 'Select the type of chart to display'
            },
            'metrics': {
                'type': 'multiselect',
                'label': '📈 Metrics to Compare',
                'default': ['accuracy', 'f1_score'],
                'options': self.common_metrics,
                'help': 'Select which metrics to include in the comparison',
                'required': True
            },
            'color_palette': {
                'type': 'select',
                'label': '🎨 Color Palette',
                'default': 'viridis',
                'options': list(self.color_palettes.keys()),
                'help': 'Choose color scheme for the chart'
            },
            'show_values': {
                'type': 'checkbox',
                'label': '🔢 Show Values on Chart',
                'default': True,
                'help': 'Display metric values on the chart'
            },
            'normalize_metrics': {
                'type': 'checkbox',
                'label': '⚖️ Normalize Metrics',
                'default': False,
                'help': 'Scale all metrics to 0-1 range for better comparison'
            },
            'sort_by_metric': {
                'type': 'select',
                'label': '📊 Sort Models By',
                'default': 'none',
                'options': ['none'] + self.common_metrics,
                'help': 'Sort models by a specific metric'
            },
            'show_best_model': {
                'type': 'checkbox',
                'label': '🏆 Highlight Best Model',
                'default': True,
                'help': 'Highlight the best performing model'
            },
            'chart_height': {
                'type': 'slider',
                'label': '📏 Chart Height',
                'default': 500,
                'min': 300,
                'max': 1000,
                'help': 'Adjust the height of the chart'
            },
            'show_error_bars': {
                'type': 'checkbox',
                'label': '📊 Show Error Bars',
                'default': False,
                'help': 'Show standard deviation as error bars (if available)'
            }
        }
    
    def render(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
               config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Render the performance comparison visualization.
        
        Args:
            data: Dataset used for training/testing
            model_results: List of model result dictionaries
            config: Configuration dictionary for the visualization
            **kwargs: Additional keyword arguments
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        try:
            # Get configuration with defaults
            config = config or {}
            chart_type = config.get('chart_type', 'bar')
            selected_metrics = config.get('metrics', ['accuracy', 'f1_score'])
            color_palette = config.get('color_palette', 'viridis')
            show_values = config.get('show_values', True)
            normalize_metrics = config.get('normalize_metrics', False)
            sort_by_metric = config.get('sort_by_metric', 'none')
            show_best_model = config.get('show_best_model', True)
            chart_height = config.get('chart_height', 500)
            show_error_bars = config.get('show_error_bars', False)
            
            # Prepare data for visualization
            viz_data = self._prepare_visualization_data(
                model_results, selected_metrics, normalize_metrics, sort_by_metric
            )
            
            if viz_data.empty:
                st.warning("⚠️ No data available for the selected metrics")
                return False
            
            # Create the visualization based on chart type
            success = self._create_chart(
                viz_data, chart_type, selected_metrics, color_palette,
                show_values, show_best_model, chart_height, show_error_bars
            )
            
            if success:
                # Add summary statistics
                self._display_summary_statistics(viz_data, selected_metrics)
                
                # Add insights
                self._display_insights(viz_data, selected_metrics)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rendering performance chart: {str(e)}")
            st.error(f"❌ Error creating performance chart: {str(e)}")
            return False
    
    def _prepare_visualization_data(self, model_results: List[Dict[str, Any]], 
                                  selected_metrics: List[str], normalize_metrics: bool,
                                  sort_by_metric: str) -> pd.DataFrame:
        """
        Prepare data for visualization.
        
        Args:
            model_results: List of model result dictionaries
            selected_metrics: List of metrics to include
            normalize_metrics: Whether to normalize metrics to 0-1 range
            sort_by_metric: Metric to sort by
            
        Returns:
            DataFrame prepared for visualization
        """
        viz_rows = []
        
        for idx, result in enumerate(model_results):
            if "error" in result:
                continue
            
            # Create model identifier
            model_name = result.get('model_name', f'Model {idx + 1}')
            target = result.get('target_column', 'Unknown')
            
            # Calculate run number for this model
            run_number = sum(1 for i, r in enumerate(model_results[:idx+1]) 
                           if r.get('model_name') == result.get('model_name') 
                           and r.get('target_column') == result.get('target_column')
                           and "error" not in r)
            
            model_id = f"{model_name} - Run #{run_number}"
            
            # Extract metrics
            row_data = {'Model': model_id, 'Model_Name': model_name, 'Run': run_number}
            
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
                    df[f'{metric}_normalized'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        # Sort by metric if requested
        if sort_by_metric != 'none' and sort_by_metric in df.columns:
            df = df.sort_values(sort_by_metric, ascending=False)
        
        return df
    
    def _create_chart(self, viz_data: pd.DataFrame, chart_type: str, 
                     selected_metrics: List[str], color_palette: str,
                     show_values: bool, show_best_model: bool, 
                     chart_height: int, show_error_bars: bool) -> bool:
        """
        Create the actual chart based on configuration.
        
        Args:
            viz_data: Prepared data for visualization
            chart_type: Type of chart to create
            selected_metrics: Metrics to display
            color_palette: Color palette to use
            show_values: Whether to show values on chart
            show_best_model: Whether to highlight best model
            chart_height: Height of the chart
            show_error_bars: Whether to show error bars
            
        Returns:
            bool: True if chart creation was successful
        """
        try:
            colors = self.color_palettes.get(color_palette, px.colors.qualitative.Set1)
            
            if chart_type == 'bar':
                fig = self._create_bar_chart(viz_data, selected_metrics, colors, show_values, show_best_model)
            elif chart_type == 'horizontal_bar':
                fig = self._create_horizontal_bar_chart(viz_data, selected_metrics, colors, show_values, show_best_model)
            elif chart_type == 'line':
                fig = self._create_line_chart(viz_data, selected_metrics, colors)
            elif chart_type == 'scatter':
                fig = self._create_scatter_chart(viz_data, selected_metrics, colors)
            elif chart_type == 'radar':
                fig = self._create_radar_chart(viz_data, selected_metrics, colors)
            elif chart_type == 'heatmap':
                fig = self._create_heatmap(viz_data, selected_metrics, colors)
            elif chart_type == 'box':
                fig = self._create_box_plot(viz_data, selected_metrics, colors)
            elif chart_type == 'violin':
                fig = self._create_violin_plot(viz_data, selected_metrics, colors)
            else:
                st.error(f"❌ Unsupported chart type: {chart_type}")
                return False
            
            # Update layout
            fig.update_layout(
                height=chart_height,
                title=f"Model Performance Comparison - {self.chart_types[chart_type]}",
                title_x=0.5,
                showlegend=True,
                template="plotly_white",
                font=dict(size=12)
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating {chart_type} chart: {str(e)}")
            st.error(f"❌ Error creating {chart_type} chart: {str(e)}")
            return False
    
    def _create_bar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], 
                         colors: List[str], show_values: bool, show_best_model: bool) -> go.Figure:
        """Create a bar chart for performance comparison."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            # Highlight best model if requested
            marker_colors = [color] * len(viz_data)
            if show_best_model and len(viz_data) > 0:
                best_idx = viz_data[metric].idxmax()
                marker_colors[best_idx] = '#FFD700'  # Gold color
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=viz_data['Model'],
                y=viz_data[metric],
                text=viz_data[metric].round(4) if show_values else None,
                textposition='auto',
                marker_color=marker_colors if show_best_model else color,
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title="Performance Score",
            barmode='group'
        )
        
        return fig
    
    def _create_horizontal_bar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], 
                                   colors: List[str], show_values: bool, show_best_model: bool) -> go.Figure:
        """Create a horizontal bar chart for performance comparison."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                y=viz_data['Model'],
                x=viz_data[metric],
                text=viz_data[metric].round(4) if show_values else None,
                textposition='auto',
                marker_color=color,
                orientation='h',
                hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            yaxis_title="Models",
            xaxis_title="Performance Score",
            barmode='group'
        )
        
        return fig
    
    def _create_line_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a line chart for performance comparison."""
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            if metric not in viz_data.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                name=metric.replace('_', ' ').title(),
                x=viz_data['Model'],
                y=viz_data[metric],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
            ))
        
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title="Performance Score"
        )
        
        return fig
    
    def _create_radar_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a radar chart for performance comparison."""
        fig = go.Figure()
        
        for i, row in viz_data.iterrows():
            values = []
            labels = []
            
            for metric in selected_metrics:
                if metric in viz_data.columns:
                    values.append(row[metric])
                    labels.append(metric.replace('_', ' ').title())
            
            # Close the radar chart
            values.append(values[0])
            labels.append(labels[0])
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=row['Model'],
                line_color=color
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
    
    def _create_heatmap(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a heatmap for performance comparison."""
        # Prepare data for heatmap
        heatmap_data = viz_data.set_index('Model')[selected_metrics].T
        
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
            xaxis_title="Models",
            yaxis_title="Metrics"
        )
        
        return fig
    
    def _create_scatter_chart(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a scatter plot for performance comparison."""
        if len(selected_metrics) < 2:
            # If only one metric, create scatter vs index
            fig = go.Figure()
            metric = selected_metrics[0]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(viz_data))),
                y=viz_data[metric],
                mode='markers',
                marker=dict(size=12, color=colors[0]),
                text=viz_data['Model'],
                hovertemplate="<b>%{text}</b><br>" + f"{metric}: %{{y:.4f}}<extra></extra>"
            ))
            
            fig.update_layout(
                xaxis_title="Model Index",
                yaxis_title=metric.replace('_', ' ').title()
            )
        else:
            # Scatter plot of first two metrics
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=viz_data[selected_metrics[0]],
                y=viz_data[selected_metrics[1]],
                mode='markers',
                marker=dict(size=12, color=colors[0]),
                text=viz_data['Model'],
                hovertemplate="<b>%{text}</b><br>" + 
                            f"{selected_metrics[0]}: %{{x:.4f}}<br>" +
                            f"{selected_metrics[1]}: %{{y:.4f}}<extra></extra>"
            ))
            
            fig.update_layout(
                xaxis_title=selected_metrics[0].replace('_', ' ').title(),
                yaxis_title=selected_metrics[1].replace('_', ' ').title()
            )
        
        return fig
    
    def _create_box_plot(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a box plot for performance comparison."""
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
    
    def _create_violin_plot(self, viz_data: pd.DataFrame, selected_metrics: List[str], colors: List[str]) -> go.Figure:
        """Create a violin plot for performance comparison."""
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
    
    def _display_summary_statistics(self, viz_data: pd.DataFrame, selected_metrics: List[str]) -> None:
        """Display summary statistics for the performance metrics."""
        st.markdown("### 📊 **Performance Summary**")
        
        # Calculate statistics
        stats_data = []
        for metric in selected_metrics:
            if metric in viz_data.columns:
                values = viz_data[metric]
                best_model = viz_data.loc[values.idxmax(), 'Model']
                worst_model = viz_data.loc[values.idxmin(), 'Model']
                
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Best Score': f"{values.max():.4f}",
                    'Best Model': best_model,
                    'Worst Score': f"{values.min():.4f}",
                    'Worst Model': worst_model,
                    'Average': f"{values.mean():.4f}",
                    'Std Dev': f"{values.std():.4f}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def _display_insights(self, viz_data: pd.DataFrame, selected_metrics: List[str]) -> None:
        """Display insights about model performance."""
        st.markdown("### 💡 **Performance Insights**")
        
        insights = []
        
        # Find overall best model
        if len(selected_metrics) > 1:
            # Calculate average rank across metrics
            ranks = pd.DataFrame()
            for metric in selected_metrics:
                if metric in viz_data.columns:
                    ranks[metric] = viz_data[metric].rank(ascending=False)
            
            if not ranks.empty:
                avg_rank = ranks.mean(axis=1)
                best_overall_idx = avg_rank.idxmin()
                best_overall_model = viz_data.loc[best_overall_idx, 'Model']
                insights.append(f"🏆 **Best Overall Model**: {best_overall_model} (average rank: {avg_rank.iloc[best_overall_idx]:.1f})")
        
        # Find consistent performers
        if len(selected_metrics) > 1:
            consistent_threshold = 0.1  # Within 10% of best
            consistent_models = []
            
            for _, row in viz_data.iterrows():
                is_consistent = True
                for metric in selected_metrics:
                    if metric in viz_data.columns:
                        best_score = viz_data[metric].max()
                        model_score = row[metric]
                        if (best_score - model_score) / best_score > consistent_threshold:
                            is_consistent = False
                            break
                
                if is_consistent:
                    consistent_models.append(row['Model'])
            
            if consistent_models:
                insights.append(f"⚖️ **Consistent Performers**: {', '.join(consistent_models)}")
        
        # Performance spread analysis
        for metric in selected_metrics:
            if metric in viz_data.columns:
                values = viz_data[metric]
                spread = values.max() - values.min()
                if spread < 0.05:
                    insights.append(f"📊 **{metric.title()}**: Models show similar performance (spread: {spread:.4f})")
                elif spread > 0.2:
                    insights.append(f"📊 **{metric.title()}**: Significant performance differences (spread: {spread:.4f})")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("No specific insights available for this dataset.")
    
    def get_sample_data(self) -> Optional[pd.DataFrame]:
        """Get sample data for demonstration."""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
    
    def get_documentation(self) -> Dict[str, str]:
        """Get documentation for this plugin."""
        return {
            'overview': "Interactive performance comparison charts for machine learning models",
            'usage': "Select multiple trained models to compare their performance metrics across different chart types",
            'requirements': "At least 2 trained models with performance metrics",
            'supported_data_types': "Classification, Regression, Binary, Multiclass",
            'configuration': "Chart type, metrics selection, color palette, normalization options",
            'examples': "Use bar charts for simple comparisons, radar charts for multi-metric analysis, heatmaps for pattern recognition"
        }

def get_plugin():
    return PerformanceChartPlugin()  # Use whatever your actual class name is