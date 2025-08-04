"""
K-Nearest Neighbors Regressor Display Component
===============================================

This module provides comprehensive visualization and reporting capabilities
for the K-Nearest Neighbors Regressor algorithm analysis results.

Features:
- K-value optimization visualizations
- Distance metric and algorithm comparison plots
- Interactive performance dashboards
- Neighbor analysis visualizations
- Comprehensive reporting in multiple formats
- Interactive model exploration tools

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, List, Tuple, Union, Optional
import logging
from pathlib import Path

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Some visualizations will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be disabled.")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNDisplay:
    """
    Display component for K-Nearest Neighbors Regressor analysis.
    
    Provides comprehensive visualization and reporting capabilities for
    KNN analysis results, including interactive dashboards and detailed reports.
    """
    
    def __init__(self, core, analysis):
        """
        Initialize the KNN Display component.
        
        Parameters:
        -----------
        core : KNNCore
            The trained KNN core component
        analysis : KNNAnalysis
            The KNN analysis component
        """
        self.core = core
        self.analysis = analysis
        
        # Display configuration
        self.config = {
            'interactive_plots': PLOTLY_AVAILABLE,
            'plot_theme': 'plotly_white',
            'plot_width': 800,
            'plot_height': 500,
            'dashboard_layout': 'detailed',
            'show_confidence_intervals': True,
            'auto_save_plots': False,
            'plot_save_format': 'png',
            'color_palette': 'viridis'
        }
        
        # Plotting themes
        self.themes = {
            'plotly': 'plotly',
            'plotly_white': 'plotly_white',
            'plotly_dark': 'plotly_dark',
            'ggplot2': 'ggplot2',
            'seaborn': 'seaborn'
        }
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'accent': '#8c564b'
        }
        
        logger.info("✅ KNN Display component initialized")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the display component.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Display configuration dictionary
        """
        try:
            self.config.update(config)
            logger.info("✅ KNN Display configuration updated")
        except Exception as e:
            logger.error(f"❌ Display configuration failed: {str(e)}")
    
    def plot_k_optimization_curve(self, interactive: bool = None, 
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot K optimization curve showing bias-variance tradeoff.
        
        Parameters:
        -----------
        interactive : bool, optional
            Whether to create interactive plot. Uses config default if None
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot information and data
        """
        try:
            # Get K optimization analysis
            k_analysis = self.analysis.analyze_k_optimization()
            if 'error' in k_analysis:
                return k_analysis
            
            interactive = interactive if interactive is not None else self.config['interactive_plots']
            
            if interactive and PLOTLY_AVAILABLE:
                return self._plot_k_optimization_plotly(k_analysis, save_path)
            elif MATPLOTLIB_AVAILABLE:
                return self._plot_k_optimization_matplotlib(k_analysis, save_path)
            else:
                return {'error': 'No plotting library available'}
                
        except Exception as e:
            logger.error(f"❌ K optimization plot failed: {str(e)}")
            return {'error': str(e)}
    
    def _plot_k_optimization_plotly(self, k_analysis: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create interactive K optimization plot with Plotly."""
        try:
            k_values = k_analysis['k_values']
            train_scores = k_analysis['train_scores']
            cv_scores = k_analysis['cv_scores_mean']
            cv_stds = k_analysis['cv_scores_std']
            optimal_k = k_analysis['optimal_k']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'K Optimization Curve',
                    'Bias-Variance Tradeoff',
                    'Score Distribution',
                    'K Recommendation'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Main K optimization curve
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=train_scores,
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color=self.colors['primary'], width=2),
                    hovertemplate='K=%{x}<br>Training R²=%{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, 
                    y=np.array(cv_scores) + np.array(cv_stds),
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=np.array(cv_scores) - np.array(cv_stds),
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=cv_scores,
                    mode='lines+markers',
                    name='CV Score',
                    line=dict(color=self.colors['secondary'], width=2),
                    hovertemplate='K=%{x}<br>CV R²=%{y:.3f}±%{customdata:.3f}<extra></extra>',
                    customdata=cv_stds
                ),
                row=1, col=1
            )
            
            # Highlight optimal K
            optimal_idx = k_values.index(optimal_k)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k], 
                    y=[cv_scores[optimal_idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.colors['success'],
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name=f'Optimal K={optimal_k}',
                    hovertemplate=f'Optimal K={optimal_k}<br>CV R²={cv_scores[optimal_idx]:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Bias-variance analysis
            bias_proxy = [1 - score for score in train_scores]  # Training error as bias proxy
            variance_proxy = cv_stds  # CV std as variance proxy
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=bias_proxy,
                    mode='lines+markers',
                    name='Bias (proxy)',
                    line=dict(color=self.colors['warning'], width=2),
                    hovertemplate='K=%{x}<br>Bias=%{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=variance_proxy,
                    mode='lines+markers',
                    name='Variance (proxy)',
                    line=dict(color=self.colors['info'], width=2),
                    hovertemplate='K=%{x}<br>Variance=%{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Score distribution box plot
            score_data = []
            for i, k in enumerate(k_values[::2]):  # Sample every other K for clarity
                scores = k_analysis.get('detailed_cv_scores', {}).get(str(k), [cv_scores[i*2]])
                for score in scores:
                    score_data.append({'K': k, 'Score': score})
            
            if score_data:
                score_df = pd.DataFrame(score_data)
                for k in score_df['K'].unique():
                    k_scores = score_df[score_df['K'] == k]['Score']
                    fig.add_trace(
                        go.Box(
                            y=k_scores,
                            name=f'K={k}',
                            boxpoints='outliers',
                            hovertemplate=f'K={k}<br>Score=%{{y:.3f}}<extra></extra>'
                        ),
                        row=2, col=1
                    )
            
            # K recommendation visualization
            recommendation = k_analysis.get('recommendation', {})
            confidence = recommendation.get('confidence', 'medium')
            
            confidence_colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=optimal_k,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Recommended K<br><span style='font-size:0.8em;color:gray'>Confidence: {confidence}</span>"},
                    delta={'reference': self.core.n_neighbors, 'valueformat': '.0f'},
                    gauge={
                        'axis': {'range': [None, max(k_values)]},
                        'bar': {'color': confidence_colors.get(confidence, 'gray')},
                        'steps': [
                            {'range': [0, max(k_values) * 0.3], 'color': "lightgray"},
                            {'range': [max(k_values) * 0.3, max(k_values) * 0.7], 'color': "gray"},
                            {'range': [max(k_values) * 0.7, max(k_values)], 'color': "lightgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': self.core.n_neighbors
                        }
                    }
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'K-Nearest Neighbors Optimization Analysis',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                template=self.config['plot_theme'],
                width=self.config['plot_width'] * 1.5,
                height=self.config['plot_height'] * 1.2,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Number of Neighbors (K)", row=1, col=1)
            fig.update_yaxes(title_text="R² Score", row=1, col=1)
            fig.update_xaxes(title_text="Number of Neighbors (K)", row=1, col=2)
            fig.update_yaxes(title_text="Error Proxy", row=1, col=2)
            fig.update_xaxes(title_text="K Values", row=2, col=1)
            fig.update_yaxes(title_text="CV Score Distribution", row=2, col=1)
            
            # Save if requested
            if save_path:
                self._save_plotly_figure(fig, save_path)
            
            plot_data = {
                'figure': fig,
                'plot_type': 'k_optimization',
                'interactive': True,
                'data': {
                    'k_values': k_values,
                    'train_scores': train_scores,
                    'cv_scores': cv_scores,
                    'cv_stds': cv_stds,
                    'optimal_k': optimal_k,
                    'recommendation': recommendation
                }
            }
            
            logger.info("✅ Interactive K optimization plot created")
            return plot_data
            
        except Exception as e:
            logger.error(f"❌ Plotly K optimization plot failed: {str(e)}")
            return {'error': str(e)}
    
    def _plot_k_optimization_matplotlib(self, k_analysis: Dict[str, Any], 
                                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create static K optimization plot with Matplotlib."""
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            k_values = k_analysis['k_values']
            train_scores = k_analysis['train_scores']
            cv_scores = k_analysis['cv_scores_mean']
            cv_stds = k_analysis['cv_scores_std']
            optimal_k = k_analysis['optimal_k']
            
            # Main optimization curve
            ax1.plot(k_values, train_scores, 'o-', label='Training Score', 
                    color=self.colors['primary'], linewidth=2, markersize=6)
            ax1.fill_between(k_values, 
                           np.array(cv_scores) - np.array(cv_stds),
                           np.array(cv_scores) + np.array(cv_stds),
                           alpha=0.2, color=self.colors['secondary'])
            ax1.plot(k_values, cv_scores, 's-', label='CV Score', 
                    color=self.colors['secondary'], linewidth=2, markersize=6)
            ax1.axvline(optimal_k, color=self.colors['success'], linestyle='--', 
                       linewidth=2, label=f'Optimal K={optimal_k}')
            ax1.set_xlabel('Number of Neighbors (K)')
            ax1.set_ylabel('R² Score')
            ax1.set_title('K Optimization Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bias-variance tradeoff
            bias_proxy = [1 - score for score in train_scores]
            variance_proxy = cv_stds
            
            ax2.plot(k_values, bias_proxy, 'o-', label='Bias (proxy)', 
                    color=self.colors['warning'], linewidth=2)
            ax2.plot(k_values, variance_proxy, 's-', label='Variance (proxy)', 
                    color=self.colors['info'], linewidth=2)
            ax2.set_xlabel('Number of Neighbors (K)')
            ax2.set_ylabel('Error Proxy')
            ax2.set_title('Bias-Variance Tradeoff')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Score distribution
            positions = range(len(k_values[::3]))  # Sample for clarity
            sampled_k = k_values[::3]
            sampled_scores = [cv_scores[i*3] for i in range(len(sampled_k))]
            sampled_stds = [cv_stds[i*3] for i in range(len(sampled_k))]
            
            box_data = []
            for i, (score, std) in enumerate(zip(sampled_scores, sampled_stds)):
                # Generate synthetic distribution
                synthetic_scores = np.random.normal(score, std, 20)
                box_data.append(synthetic_scores)
            
            if box_data:
                ax3.boxplot(box_data, positions=positions)
                ax3.set_xticklabels([f'K={k}' for k in sampled_k])
                ax3.set_ylabel('CV Score Distribution')
                ax3.set_title('Score Variability Across K Values')
                ax3.grid(True, alpha=0.3)
            
            # Recommendation summary
            recommendation = k_analysis.get('recommendation', {})
            confidence = recommendation.get('confidence', 'medium')
            improvement = k_analysis.get('improvement_potential', 0)
            
            ax4.text(0.5, 0.7, f'Recommended K: {optimal_k}', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    transform=ax4.transAxes)
            ax4.text(0.5, 0.5, f'Confidence: {confidence.upper()}', 
                    ha='center', va='center', fontsize=12,
                    transform=ax4.transAxes)
            ax4.text(0.5, 0.3, f'Improvement: {improvement:.3f}', 
                    ha='center', va='center', fontsize=12,
                    transform=ax4.transAxes)
            ax4.text(0.5, 0.1, f'Current K: {self.core.n_neighbors}', 
                    ha='center', va='center', fontsize=10, style='italic',
                    transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('K Recommendation Summary')
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plot_data = {
                'figure': fig,
                'plot_type': 'k_optimization',
                'interactive': False,
                'data': k_analysis,
                'save_path': save_path
            }
            
            logger.info("✅ Static K optimization plot created")
            return plot_data
            
        except Exception as e:
            logger.error(f"❌ Matplotlib K optimization plot failed: {str(e)}")
            return {'error': str(e)}
    
    def plot_distance_metric_comparison(self, interactive: bool = None, 
                                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot comparison of different distance metrics.
        
        Parameters:
        -----------
        interactive : bool, optional
            Whether to create interactive plot
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot information and data
        """
        try:
            # Get distance metric analysis
            metric_analysis = self.analysis.analyze_distance_metrics()
            if 'error' in metric_analysis:
                return metric_analysis
            
            interactive = interactive if interactive is not None else self.config['interactive_plots']
            
            if interactive and PLOTLY_AVAILABLE:
                return self._plot_distance_metrics_plotly(metric_analysis, save_path)
            elif MATPLOTLIB_AVAILABLE:
                return self._plot_distance_metrics_matplotlib(metric_analysis, save_path)
            else:
                return {'error': 'No plotting library available'}
                
        except Exception as e:
            logger.error(f"❌ Distance metric plot failed: {str(e)}")
            return {'error': str(e)}
    
    def _plot_distance_metrics_plotly(self, metric_analysis: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create interactive distance metrics comparison with Plotly."""
        try:
            metric_results = metric_analysis['metric_results']
            
            # Prepare data
            metrics = []
            cv_scores = []
            cv_stds = []
            train_scores = []
            
            for metric, results in metric_results.items():
                if 'error' not in results:
                    metrics.append(metric)
                    cv_scores.append(results['cv_score_mean'])
                    cv_stds.append(results['cv_score_std'])
                    train_scores.append(results['train_score'])
            
            if not metrics:
                return {'error': 'No valid metric results to plot'}
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Cross-Validation Performance',
                    'Training vs CV Performance',
                    'Performance Stability',
                    'Metric Recommendations'
                ]
            )
            
            # CV performance bar chart
            colors = [self.colors['success'] if m == metric_analysis.get('best_metric') 
                     else self.colors['primary'] for m in metrics]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=cv_scores,
                    error_y=dict(type='data', array=cv_stds),
                    marker_color=colors,
                    name='CV Score',
                    hovertemplate='%{x}<br>CV R²: %{y:.3f}±%{error_y.array:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Training vs CV scatter
            fig.add_trace(
                go.Scatter(
                    x=train_scores,
                    y=cv_scores,
                    mode='markers+text',
                    text=metrics,
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=cv_stds,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="CV Std", x=0.45)
                    ),
                    name='Metrics',
                    hovertemplate='%{text}<br>Train R²: %{x:.3f}<br>CV R²: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add diagonal line for reference
            min_score = min(min(train_scores), min(cv_scores))
            max_score = max(max(train_scores), max(cv_scores))
            fig.add_trace(
                go.Scatter(
                    x=[min_score, max_score],
                    y=[min_score, max_score],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Perfect Agreement',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Stability analysis (CV std)
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=cv_stds,
                    marker_color=self.colors['warning'],
                    name='CV Std',
                    hovertemplate='%{x}<br>CV Std: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Recommendation radar chart
            best_metric = metric_analysis.get('best_metric', metrics[0])
            improvement = metric_analysis.get('improvement_potential', 0)
            
            recommendation_text = f"""
            <b>Best Metric:</b> {best_metric}<br>
            <b>Improvement:</b> {improvement:.3f}<br>
            <b>Current:</b> {metric_analysis.get('current_metric', 'N/A')}<br>
            <b>Confidence:</b> {metric_analysis.get('recommendation', {}).get('confidence', 'medium')}
            """
            
            fig.add_trace(
                go.Scatter(
                    x=[0.5], y=[0.5],
                    mode='text',
                    text=[recommendation_text],
                    textfont=dict(size=12),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Distance Metrics Comparison Analysis',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                template=self.config['plot_theme'],
                width=self.config['plot_width'] * 1.5,
                height=self.config['plot_height'] * 1.2,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Distance Metrics", row=1, col=1)
            fig.update_yaxes(title_text="CV R² Score", row=1, col=1)
            fig.update_xaxes(title_text="Training R² Score", row=1, col=2)
            fig.update_yaxes(title_text="CV R² Score", row=1, col=2)
            fig.update_xaxes(title_text="Distance Metrics", row=2, col=1)
            fig.update_yaxes(title_text="CV Standard Deviation", row=2, col=1)
            
            # Save if requested
            if save_path:
                self._save_plotly_figure(fig, save_path)
            
            plot_data = {
                'figure': fig,
                'plot_type': 'distance_metrics',
                'interactive': True,
                'data': metric_analysis
            }
            
            logger.info("✅ Interactive distance metrics plot created")
            return plot_data
            
        except Exception as e:
            logger.error(f"❌ Plotly distance metrics plot failed: {str(e)}")
            return {'error': str(e)}
    
    def plot_neighbor_analysis(self, sample_indices: Optional[List[int]] = None,
                              interactive: bool = None, 
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot neighbor analysis for specific samples.
        
        Parameters:
        -----------
        sample_indices : List[int], optional
            Indices of samples to analyze. If None, uses random samples
        interactive : bool, optional
            Whether to create interactive plot
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot information and data
        """
        try:
            # Get neighbor analysis
            neighbor_analysis = self.analysis.analyze_neighbor_distributions()
            if 'error' in neighbor_analysis:
                return neighbor_analysis
            
            interactive = interactive if interactive is not None else self.config['interactive_plots']
            
            if interactive and PLOTLY_AVAILABLE:
                return self._plot_neighbor_analysis_plotly(neighbor_analysis, sample_indices, save_path)
            elif MATPLOTLIB_AVAILABLE:
                return self._plot_neighbor_analysis_matplotlib(neighbor_analysis, sample_indices, save_path)
            else:
                return {'error': 'No plotting library available'}
                
        except Exception as e:
            logger.error(f"❌ Neighbor analysis plot failed: {str(e)}")
            return {'error': str(e)}
    
    def create_performance_dashboard(self) -> Dict[str, Any]:
        """
        Create comprehensive performance dashboard.
        
        Returns:
        --------
        Dict[str, Any]
            Dashboard information and components
        """
        try:
            logger.info("📊 Creating KNN performance dashboard...")
            
            dashboard_components = {}
            
            # Get all analysis results
            analyses = {
                'k_optimization': self.analysis.analyze_k_optimization(),
                'distance_metrics': self.analysis.analyze_distance_metrics(),
                'algorithm_performance': self.analysis.analyze_algorithm_performance(),
                'cross_validation': self.analysis.analyze_cross_validation(),
                'neighbor_distributions': self.analysis.analyze_neighbor_distributions()
            }
            
            # Create individual plots
            dashboard_components['k_optimization_plot'] = self.plot_k_optimization_curve()
            dashboard_components['distance_metrics_plot'] = self.plot_distance_metric_comparison()
            dashboard_components['neighbor_analysis_plot'] = self.plot_neighbor_analysis()
            
            # Create summary metrics
            dashboard_components['summary_metrics'] = self._create_summary_metrics(analyses)
            
            # Create recommendations
            dashboard_components['recommendations'] = self._create_recommendations(analyses)
            
            # Layout configuration
            dashboard_components['layout'] = {
                'title': 'K-Nearest Neighbors Performance Dashboard',
                'sections': [
                    'summary_metrics',
                    'k_optimization_plot',
                    'distance_metrics_plot',
                    'neighbor_analysis_plot',
                    'recommendations'
                ],
                'style': self.config['dashboard_layout']
            }
            
            logger.info("✅ Performance dashboard created successfully")
            
            return {
                'dashboard': dashboard_components,
                'timestamp': pd.Timestamp.now(),
                'config': self.config.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Dashboard creation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_summary_metrics(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary metrics for dashboard."""
        try:
            summary = {
                'model_info': {
                    'current_k': self.core.n_neighbors,
                    'metric': self.core.metric,
                    'algorithm': self.core.algorithm,
                    'weights': self.core.weights
                },
                'performance': {},
                'optimization': {},
                'stability': {}
            }
            
            # Performance metrics
            if 'cross_validation' in analyses and 'error' not in analyses['cross_validation']:
                cv_analysis = analyses['cross_validation']
                summary['performance'].update({
                    'cv_score': cv_analysis.get('basic_cv', {}).get('mean_score', 0),
                    'cv_std': cv_analysis.get('basic_cv', {}).get('std_score', 0),
                    'stability_score': cv_analysis.get('stability_analysis', {}).get('consistency_score', 0)
                })
            
            # Optimization potential
            if 'k_optimization' in analyses and 'error' not in analyses['k_optimization']:
                k_opt = analyses['k_optimization']
                summary['optimization'].update({
                    'optimal_k': k_opt.get('optimal_k', self.core.n_neighbors),
                    'improvement_potential': k_opt.get('improvement_potential', 0),
                    'k_confidence': k_opt.get('recommendation', {}).get('confidence', 'medium')
                })
            
            if 'distance_metrics' in analyses and 'error' not in analyses['distance_metrics']:
                metric_analysis = analyses['distance_metrics']
                summary['optimization'].update({
                    'best_metric': metric_analysis.get('best_metric', self.core.metric),
                    'metric_improvement': metric_analysis.get('improvement_potential', 0),
                    'metric_confidence': metric_analysis.get('recommendation', {}).get('confidence', 'medium')
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary metrics creation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_recommendations(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization recommendations."""
        try:
            recommendations = {
                'parameter_changes': [],
                'performance_insights': [],
                'next_steps': []
            }
            
            # K optimization recommendations
            if 'k_optimization' in analyses and 'error' not in analyses['k_optimization']:
                k_analysis = analyses['k_optimization']
                improvement = k_analysis.get('improvement_potential', 0)
                optimal_k = k_analysis.get('optimal_k', self.core.n_neighbors)
                
                if improvement > 0.05:
                    recommendations['parameter_changes'].append({
                        'parameter': 'n_neighbors',
                        'current': self.core.n_neighbors,
                        'recommended': optimal_k,
                        'improvement': improvement,
                        'priority': 'high',
                        'reason': f'K={optimal_k} shows {improvement:.3f} improvement in CV score'
                    })
                elif improvement > 0.02:
                    recommendations['parameter_changes'].append({
                        'parameter': 'n_neighbors',
                        'current': self.core.n_neighbors,
                        'recommended': optimal_k,
                        'improvement': improvement,
                        'priority': 'medium',
                        'reason': f'Moderate improvement possible with K={optimal_k}'
                    })
            
            # Distance metric recommendations
            if 'distance_metrics' in analyses and 'error' not in analyses['distance_metrics']:
                metric_analysis = analyses['distance_metrics']
                improvement = metric_analysis.get('improvement_potential', 0)
                best_metric = metric_analysis.get('best_metric', self.core.metric)
                
                if improvement > 0.03:
                    recommendations['parameter_changes'].append({
                        'parameter': 'metric',
                        'current': self.core.metric,
                        'recommended': best_metric,
                        'improvement': improvement,
                        'priority': 'high',
                        'reason': f'Distance metric "{best_metric}" shows significant improvement'
                    })
            
            # Performance insights
            if 'cross_validation' in analyses and 'error' not in analyses['cross_validation']:
                cv_analysis = analyses['cross_validation']
                stability = cv_analysis.get('stability_analysis', {}).get('consistency_score', 0)
                
                if stability < 0.8:
                    recommendations['performance_insights'].append({
                        'type': 'stability',
                        'message': 'Model predictions show high variability across folds',
                        'suggestion': 'Consider increasing dataset size or using ensemble methods'
                    })
                
                if stability > 0.95:
                    recommendations['performance_insights'].append({
                        'type': 'stability',
                        'message': 'Model shows excellent prediction stability',
                        'suggestion': 'Current configuration is well-suited for your data'
                    })
            
            # Next steps
            if recommendations['parameter_changes']:
                recommendations['next_steps'].append(
                    "Apply the recommended parameter changes and retrain the model"
                )
            
            recommendations['next_steps'].extend([
                "Consider feature engineering to improve local neighborhood structure",
                "Evaluate ensemble methods for improved robustness",
                "Monitor performance on new data to ensure generalization"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendations creation failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, format_type: str = 'html', 
                                    save_path: Optional[str] = None,
                                    analyses: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Parameters:
        -----------
        format_type : str, default='html'
            Report format ('html', 'json', 'markdown')
        save_path : str, optional
            Path to save the report
        analyses : Dict[str, Any], optional
            Pre-computed analyses. If None, will compute them
            
        Returns:
        --------
        Dict[str, Any]
            Report data and metadata
        """
        try:
            logger.info(f"📋 Generating {format_type.upper()} report...")
            
            # Get analyses if not provided
            if analyses is None:
                analyses = {
                    'k_optimization': self.analysis.analyze_k_optimization(),
                    'distance_metrics': self.analysis.analyze_distance_metrics(),
                    'algorithm_performance': self.analysis.analyze_algorithm_performance(),
                    'cross_validation': self.analysis.analyze_cross_validation(),
                    'neighbor_distributions': self.analysis.analyze_neighbor_distributions()
                }
            
            if format_type.lower() == 'html':
                return self._generate_html_report(analyses, save_path)
            elif format_type.lower() == 'json':
                return self._generate_json_report(analyses, save_path)
            elif format_type.lower() == 'markdown':
                return self._generate_markdown_report(analyses, save_path)
            else:
                return {'error': f'Unsupported report format: {format_type}'}
                
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_html_report(self, analyses: Dict[str, Any], 
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate HTML report."""
        try:
            # Create HTML content
            html_content = self._create_html_template()
            
            # Add model summary
            model_summary = self._create_model_summary_html()
            html_content = html_content.replace('{{MODEL_SUMMARY}}', model_summary)
            
            # Add analysis sections
            analysis_html = self._create_analysis_sections_html(analyses)
            html_content = html_content.replace('{{ANALYSIS_SECTIONS}}', analysis_html)
            
            # Add recommendations
            recommendations = self._create_recommendations(analyses)
            recommendations_html = self._create_recommendations_html(recommendations)
            html_content = html_content.replace('{{RECOMMENDATIONS}}', recommendations_html)
            
            # Save if path provided
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            return {
                'format': 'html',
                'content': html_content,
                'save_path': save_path,
                'timestamp': pd.Timestamp.now(),
                'analyses_included': list(analyses.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ HTML report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_html_template(self) -> str:
        """Create base HTML template for reports."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>KNN Regressor Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: #e9ecef; border-radius: 5px; }
                .recommendation { background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; }
                .warning { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }
                .error { background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 10px 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .chart-placeholder { background: #f8f9fa; padding: 40px; text-align: center; border: 2px dashed #dee2e6; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>K-Nearest Neighbors Regressor Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            {{MODEL_SUMMARY}}
            {{ANALYSIS_SECTIONS}}
            {{RECOMMENDATIONS}}
            
        </body>
        </html>
        """.format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    def _save_plotly_figure(self, fig, save_path: str) -> None:
        """Save Plotly figure to file."""
        try:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            elif save_path.endswith('.png'):
                fig.write_image(save_path)
            elif save_path.endswith('.pdf'):
                fig.write_image(save_path)
            elif save_path.endswith('.svg'):
                fig.write_image(save_path)
            else:
                # Default to HTML
                fig.write_html(save_path + '.html')
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save figure: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the KNN Display."""
        return (f"KNNDisplay(interactive={self.config['interactive_plots']}, "
                f"theme='{self.config['plot_theme']}', "
                f"core_fitted={self.core.is_fitted})")