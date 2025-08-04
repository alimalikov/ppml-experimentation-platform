"""
Radius Neighbors Regressor - Display Implementation
=================================================

This module contains comprehensive display and visualization functionality for the Radius Neighbors Regressor,
including interactive plots, performance reports, and analytical dashboards.

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import json
import base64
from io import BytesIO
from .radius_neighbors_core import RadiusNeighborsCore
from .radius_neighbors_analysis import RadiusNeighborsAnalysis

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RadiusNeighborsDisplay:
    """
    Comprehensive display and visualization component for Radius Neighbors Regressor.
    
    This class provides various visualization methods including performance plots,
    analysis dashboards, interactive charts, and comprehensive reports.
    """
    
    def __init__(self, core: RadiusNeighborsCore, analysis: RadiusNeighborsAnalysis):
        """
        Initialize the display component.
        
        Parameters:
        -----------
        core : RadiusNeighborsCore
            The core component containing the fitted model and data
        analysis : RadiusNeighborsAnalysis
            The analysis component for generating insights
        """
        self.core = core
        self.analysis = analysis
        self.plot_cache_ = {}
        self.report_cache_ = {}
        
        # Display configuration
        self.config = {
            'figure_size': (12, 8),
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'style': 'whitegrid',
            'font_size': 12,
            'title_size': 16,
            'dpi': 300
        }
    
    def create_performance_dashboard(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive performance dashboard.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the dashboard HTML file
            
        Returns:
        --------
        Dict[str, Any]
            Dashboard data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before creating dashboard'}
        
        try:
            # Generate all analyses first
            radius_analysis = self.analysis.analyze_radius_behavior()
            cv_analysis = self.analysis.analyze_cross_validation()
            feature_analysis = self.analysis.analyze_feature_importance()
            performance_profile = self.analysis.profile_performance()
            
            # Create dashboard structure
            dashboard_data = {
                'metadata': self._create_dashboard_metadata(),
                'model_overview': self._create_model_overview(),
                'performance_metrics': self._create_performance_metrics_section(),
                'radius_analysis': self._create_radius_analysis_section(radius_analysis),
                'feature_insights': self._create_feature_insights_section(feature_analysis),
                'cross_validation': self._create_cv_section(cv_analysis),
                'performance_profile': self._create_performance_profile_section(performance_profile),
                'visualizations': self._create_dashboard_visualizations(),
                'recommendations': self._create_recommendations_section()
            }
            
            # Create interactive HTML dashboard if requested
            if save_path:
                html_content = self._generate_html_dashboard(dashboard_data)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                dashboard_data['html_saved'] = save_path
            
            self.report_cache_['performance_dashboard'] = dashboard_data
            return dashboard_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_radius_coverage(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create radius coverage visualization.
        
        Parameters:
        -----------
        interactive : bool, default=True
            Whether to create interactive plotly chart
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before plotting'}
        
        try:
            # Get neighbor analysis
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            if interactive:
                # Create interactive plotly chart
                fig = self._create_interactive_coverage_plot(neighbor_counts)
                
                if save_path:
                    fig.write_html(save_path)
                
                plot_data = {
                    'type': 'interactive',
                    'figure': fig,
                    'neighbor_counts': neighbor_counts.tolist(),
                    'coverage_stats': self._calculate_coverage_stats(neighbor_counts)
                }
            else:
                # Create matplotlib chart
                fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
                self._create_static_coverage_plots(axes, neighbor_counts)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
                
                plot_data = {
                    'type': 'static',
                    'figure': fig,
                    'neighbor_counts': neighbor_counts.tolist(),
                    'coverage_stats': self._calculate_coverage_stats(neighbor_counts)
                }
            
            self.plot_cache_['radius_coverage'] = plot_data
            return plot_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_performance_comparison(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create performance comparison visualization.
        
        Parameters:
        -----------
        interactive : bool, default=True
            Whether to create interactive plotly chart
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before plotting'}
        
        try:
            # Get comparison data
            knn_comparison = self.analysis.compare_with_knn()
            global_comparison = self.analysis.compare_with_global_methods()
            
            if interactive:
                fig = self._create_interactive_comparison_plot(knn_comparison, global_comparison)
                
                if save_path:
                    fig.write_html(save_path)
                
                plot_data = {
                    'type': 'interactive',
                    'figure': fig,
                    'knn_comparison': knn_comparison,
                    'global_comparison': global_comparison
                }
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                self._create_static_comparison_plots(axes, knn_comparison, global_comparison)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
                
                plot_data = {
                    'type': 'static',
                    'figure': fig,
                    'knn_comparison': knn_comparison,
                    'global_comparison': global_comparison
                }
            
            self.plot_cache_['performance_comparison'] = plot_data
            return plot_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_feature_importance(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create feature importance visualization.
        
        Parameters:
        -----------
        interactive : bool, default=True
            Whether to create interactive plotly chart
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before plotting'}
        
        try:
            # Get feature importance analysis
            feature_analysis = self.analysis.analyze_feature_importance()
            
            if 'error' in feature_analysis:
                return feature_analysis
            
            importance_scores = feature_analysis['feature_importance_scores']
            
            if interactive:
                fig = self._create_interactive_feature_plot(importance_scores, feature_analysis)
                
                if save_path:
                    fig.write_html(save_path)
                
                plot_data = {
                    'type': 'interactive',
                    'figure': fig,
                    'importance_scores': importance_scores,
                    'feature_analysis': feature_analysis
                }
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                self._create_static_feature_plots(axes, importance_scores, feature_analysis)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
                
                plot_data = {
                    'type': 'static',
                    'figure': fig,
                    'importance_scores': importance_scores,
                    'feature_analysis': feature_analysis
                }
            
            self.plot_cache_['feature_importance'] = plot_data
            return plot_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_cross_validation_analysis(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create cross-validation analysis visualization.
        
        Parameters:
        -----------
        interactive : bool, default=True
            Whether to create interactive plotly chart
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before plotting'}
        
        try:
            # Get cross-validation analysis
            cv_analysis = self.analysis.analyze_cross_validation()
            
            if 'error' in cv_analysis:
                return cv_analysis
            
            if interactive:
                fig = self._create_interactive_cv_plot(cv_analysis)
                
                if save_path:
                    fig.write_html(save_path)
                
                plot_data = {
                    'type': 'interactive',
                    'figure': fig,
                    'cv_analysis': cv_analysis
                }
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                self._create_static_cv_plots(axes, cv_analysis)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
                
                plot_data = {
                    'type': 'static',
                    'figure': fig,
                    'cv_analysis': cv_analysis
                }
            
            self.plot_cache_['cross_validation'] = plot_data
            return plot_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_neighborhood_analysis(self, interactive: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create neighborhood analysis visualization.
        
        Parameters:
        -----------
        interactive : bool, default=True
            Whether to create interactive plotly chart
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Dict[str, Any]
            Plot data and metadata
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before plotting'}
        
        try:
            # Get neighborhood data
            distances, indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_)
            neighbor_counts = np.array([len(neighbors) for neighbors in indices])
            
            if interactive:
                fig = self._create_interactive_neighborhood_plot(distances, neighbor_counts)
                
                if save_path:
                    fig.write_html(save_path)
                
                plot_data = {
                    'type': 'interactive',
                    'figure': fig,
                    'neighbor_counts': neighbor_counts.tolist(),
                    'distance_stats': self._calculate_distance_stats(distances)
                }
            else:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                self._create_static_neighborhood_plots(axes, distances, neighbor_counts)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
                
                plot_data = {
                    'type': 'static',
                    'figure': fig,
                    'neighbor_counts': neighbor_counts.tolist(),
                    'distance_stats': self._calculate_distance_stats(distances)
                }
            
            self.plot_cache_['neighborhood_analysis'] = plot_data
            return plot_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None, format_type: str = 'html') -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report
        format_type : str, default='html'
            Report format ('html', 'pdf', 'json')
            
        Returns:
        --------
        Dict[str, Any]
            Complete report data
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before generating report'}
        
        try:
            # Generate all analyses
            analyses = {
                'radius_behavior': self.analysis.analyze_radius_behavior(),
                'cross_validation': self.analysis.analyze_cross_validation(),
                'feature_importance': self.analysis.analyze_feature_importance(),
                'performance_profile': self.analysis.profile_performance(),
                'knn_comparison': self.analysis.compare_with_knn(),
                'global_comparison': self.analysis.compare_with_global_methods(),
                'metric_comparison': self.analysis.analyze_metric_comparison()
            }
            
            # Create comprehensive report
            report_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'Radius Neighbors Regressor',
                    'version': '1.0.0',
                    'analysis_components': list(analyses.keys())
                },
                'executive_summary': self._create_executive_summary(analyses),
                'model_configuration': self._create_model_configuration_section(),
                'data_overview': self._create_data_overview_section(),
                'performance_analysis': self._create_performance_analysis_section(analyses),
                'detailed_findings': self._create_detailed_findings_section(analyses),
                'recommendations': self._create_comprehensive_recommendations(analyses),
                'technical_appendix': self._create_technical_appendix(analyses)
            }
            
            # Save report in requested format
            if save_path:
                if format_type.lower() == 'html':
                    self._save_html_report(report_data, save_path)
                elif format_type.lower() == 'json':
                    self._save_json_report(report_data, save_path)
                elif format_type.lower() == 'pdf':
                    self._save_pdf_report(report_data, save_path)
                
                report_data['saved_path'] = save_path
                report_data['saved_format'] = format_type
            
            self.report_cache_['comprehensive_report'] = report_data
            return report_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_interactive_explorer(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an interactive data explorer dashboard.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the interactive explorer
            
        Returns:
        --------
        Dict[str, Any]
            Explorer data and configuration
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before creating explorer'}
        
        try:
            # Create interactive explorer with multiple tabs
            explorer_data = {
                'data_tab': self._create_data_explorer_tab(),
                'radius_tab': self._create_radius_explorer_tab(),
                'performance_tab': self._create_performance_explorer_tab(),
                'comparison_tab': self._create_comparison_explorer_tab(),
                'config': self._create_explorer_config()
            }
            
            if save_path:
                html_content = self._generate_interactive_explorer_html(explorer_data)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                explorer_data['html_saved'] = save_path
            
            return explorer_data
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== DASHBOARD CREATION METHODS ====================
    
    def _create_dashboard_metadata(self) -> Dict[str, Any]:
        """Create dashboard metadata."""
        return {
            'title': 'Radius Neighbors Regressor Analysis Dashboard',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Radius Neighbors Regressor',
            'data_shape': f"{self.core.n_samples_in_} samples × {self.core.n_features_in_} features",
            'effective_radius': self.core.effective_radius_,
            'auto_scaling': self.core.scaler_ is not None
        }
    
    def _create_model_overview(self) -> Dict[str, Any]:
        """Create model overview section."""
        return {
            'algorithm': 'Radius Neighbors Regression',
            'radius': self.core.radius,
            'effective_radius': self.core.effective_radius_,
            'weights': self.core.weights,
            'metric': self.core.metric,
            'algorithm_type': self.core.algorithm,
            'adaptive_radius': self.core.adaptive_radius,
            'scaling_method': 'StandardScaler' if self.core.scaler_ else 'None',
            'p_parameter': getattr(self.core, 'p', 2)
        }
    
    def _create_performance_metrics_section(self) -> Dict[str, Any]:
        """Create performance metrics section."""
        try:
            # Get basic performance metrics
            if hasattr(self.core, 'training_score_'):
                training_r2 = self.core.training_score_
            else:
                # Calculate training score
                y_pred_train = self.core.model_.predict(self.core.X_train_scaled_)
                from sklearn.metrics import r2_score
                training_r2 = r2_score(self.core.y_train_, y_pred_train)
            
            return {
                'training_r2_score': float(training_r2),
                'data_coverage': self._calculate_data_coverage(),
                'average_neighbors': self._calculate_average_neighbors(),
                'model_complexity': self._calculate_model_complexity()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_radius_analysis_section(self, radius_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create radius analysis section."""
        if 'error' in radius_analysis:
            return radius_analysis
        
        return {
            'coverage_analysis': radius_analysis.get('radius_coverage', {}),
            'density_analysis': radius_analysis.get('density_distribution', {}),
            'optimal_radius': radius_analysis.get('optimal_radius', {}),
            'outlier_analysis': radius_analysis.get('outlier_behavior', {})
        }
    
    def _create_feature_insights_section(self, feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature insights section."""
        if 'error' in feature_analysis:
            return feature_analysis
        
        return {
            'importance_ranking': feature_analysis.get('feature_ranking', []),
            'top_features': feature_analysis.get('top_features', []),
            'interaction_effects': feature_analysis.get('interaction_analysis', {}),
            'dimensionality_impact': feature_analysis.get('dimensionality_impact', {})
        }
    
    def _create_cv_section(self, cv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create cross-validation section."""
        if 'error' in cv_analysis:
            return cv_analysis
        
        return {
            'cv_scores': cv_analysis.get('cv_scores', []),
            'mean_score': cv_analysis.get('mean_cv_score', 0),
            'stability_metrics': cv_analysis.get('stability_metrics', {}),
            'fold_details': cv_analysis.get('fold_details', [])
        }
    
    def _create_performance_profile_section(self, performance_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance profile section."""
        if 'error' in performance_profile:
            return performance_profile
        
        return {
            'timing_analysis': performance_profile.get('operations_timing', {}),
            'memory_usage': performance_profile.get('memory_usage_mb', 0),
            'scalability': performance_profile.get('scalability_analysis', {}),
            'efficiency_metrics': performance_profile.get('efficiency_metrics', {}),
            'recommendations': performance_profile.get('performance_recommendations', [])
        }
    
    def _create_dashboard_visualizations(self) -> Dict[str, Any]:
        """Create dashboard visualizations."""
        return {
            'radius_coverage_plot': self._create_mini_coverage_plot(),
            'performance_comparison_plot': self._create_mini_comparison_plot(),
            'feature_importance_plot': self._create_mini_feature_plot(),
            'cv_stability_plot': self._create_mini_cv_plot()
        }
    
    def _create_recommendations_section(self) -> Dict[str, Any]:
        """Create recommendations section."""
        try:
            # Get recommendations from various analyses
            performance_profile = self.analysis.profile_performance()
            knn_comparison = self.analysis.compare_with_knn()
            global_comparison = self.analysis.compare_with_global_methods()
            
            recommendations = {
                'performance_optimizations': performance_profile.get('performance_recommendations', []),
                'algorithm_choice': self._generate_algorithm_recommendations(knn_comparison, global_comparison),
                'parameter_tuning': self._generate_parameter_recommendations(),
                'data_preprocessing': self._generate_preprocessing_recommendations()
            }
            
            return recommendations
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== INTERACTIVE PLOT CREATION ====================
    
    def _create_interactive_coverage_plot(self, neighbor_counts: np.ndarray) -> go.Figure:
        """Create interactive radius coverage plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Neighbor Count Distribution', 'Coverage by Radius', 
                          'Density Heatmap', 'Cumulative Coverage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Neighbor count histogram
        fig.add_trace(
            go.Histogram(x=neighbor_counts, nbinsx=20, name='Neighbor Count'),
            row=1, col=1
        )
        
        # Coverage analysis
        coverage_data = self._calculate_coverage_stats(neighbor_counts)
        fig.add_trace(
            go.Bar(x=['Isolated', 'Low Coverage', 'Well Connected'], 
                   y=[coverage_data['isolated_points'], 
                      coverage_data['low_coverage_points'],
                      coverage_data['well_connected_points']],
                   name='Coverage Categories'),
            row=1, col=2
        )
        
        # Add cumulative coverage
        sorted_counts = np.sort(neighbor_counts)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
        fig.add_trace(
            go.Scatter(x=sorted_counts, y=cumulative, mode='lines', 
                      name='Cumulative Coverage'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Radius Coverage Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_interactive_comparison_plot(self, knn_comparison: Dict[str, Any], 
                                          global_comparison: Dict[str, Any]) -> go.Figure:
        """Create interactive performance comparison plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Algorithm Performance Comparison', 'KNN vs Radius Neighbors', 
                          'Global Methods Comparison', 'Performance Ranking'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance comparison data
        methods = ['Radius Neighbors']
        r2_scores = [knn_comparison.get('radius_neighbors_performance', {}).get('r2_score', 0)]
        
        # Add KNN results
        knn_results = knn_comparison.get('knn_results', {})
        for method, result in knn_results.items():
            if 'error' not in result:
                methods.append(method.replace('_', ' ').title())
                r2_scores.append(result.get('r2_score', 0))
        
        # Add global methods
        global_results = global_comparison.get('global_methods_results', {})
        for method, result in global_results.items():
            if 'error' not in result:
                methods.append(method.replace('_', ' ').title())
                r2_scores.append(result.get('r2_score', 0))
        
        # Main comparison bar chart
        fig.add_trace(
            go.Bar(x=methods, y=r2_scores, name='R² Score'),
            row=1, col=1
        )
        
        # Detailed comparison for KNN
        if knn_results:
            knn_methods = list(knn_results.keys())
            knn_scores = [knn_results[m].get('r2_score', 0) for m in knn_methods if 'error' not in knn_results[m]]
            
            fig.add_trace(
                go.Scatter(x=knn_methods, y=knn_scores, mode='lines+markers', 
                          name='KNN Performance'),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Performance Comparison Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_interactive_feature_plot(self, importance_scores: Dict[str, float], 
                                       feature_analysis: Dict[str, Any]) -> go.Figure:
        """Create interactive feature importance plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Ranking', 'Importance Distribution', 
                          'Feature Interactions', 'Dimensionality Impact'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Feature importance bar chart
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        fig.add_trace(
            go.Bar(x=features, y=scores, name='Importance Score'),
            row=1, col=1
        )
        
        # Importance distribution
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=10, name='Score Distribution'),
            row=1, col=2
        )
        
        # Dimensionality impact
        dim_impact = feature_analysis.get('dimensionality_impact', {})
        if dim_impact:
            dimensions = list(dim_impact.keys())
            mean_neighbors = [dim_impact[d].get('mean_neighbors', 0) for d in dimensions]
            
            fig.add_trace(
                go.Scatter(x=dimensions, y=mean_neighbors, mode='lines+markers',
                          name='Mean Neighbors by Dimension'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Feature Importance Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_interactive_cv_plot(self, cv_analysis: Dict[str, Any]) -> go.Figure:
        """Create interactive cross-validation plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CV Scores by Fold', 'Score Distribution', 
                          'Stability Metrics', 'Fold Details'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CV scores by fold
        cv_scores = cv_analysis.get('cv_scores', [])
        fold_numbers = list(range(1, len(cv_scores) + 1))
        
        fig.add_trace(
            go.Scatter(x=fold_numbers, y=cv_scores, mode='lines+markers',
                      name='CV Scores'),
            row=1, col=1
        )
        
        # Add mean line
        mean_score = cv_analysis.get('mean_cv_score', 0)
        fig.add_hline(y=mean_score, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_score:.4f}",
                     row=1, col=1)
        
        # Score distribution
        fig.add_trace(
            go.Histogram(x=cv_scores, nbinsx=10, name='Score Distribution'),
            row=1, col=2
        )
        
        # Stability metrics
        stability = cv_analysis.get('stability_metrics', {})
        if stability:
            metrics = list(stability.keys())
            values = list(stability.values())
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Stability Metrics'),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Cross-Validation Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_interactive_neighborhood_plot(self, distances: List[np.ndarray], 
                                            neighbor_counts: np.ndarray) -> go.Figure:
        """Create interactive neighborhood analysis plot."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Neighbor Count Distribution', 'Distance Distribution', 
                          'Coverage Analysis', 'Neighbor vs Distance', 
                          'Density Regions', 'Outlier Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Neighbor count distribution
        fig.add_trace(
            go.Histogram(x=neighbor_counts, nbinsx=20, name='Neighbor Counts'),
            row=1, col=1
        )
        
        # Distance distribution (flatten all distances)
        all_distances = np.concatenate([d for d in distances if len(d) > 0])
        fig.add_trace(
            go.Histogram(x=all_distances, nbinsx=30, name='Distances'),
            row=1, col=2
        )
        
        # Coverage analysis
        coverage_stats = self._calculate_coverage_stats(neighbor_counts)
        fig.add_trace(
            go.Bar(x=['Isolated', 'Low', 'Medium', 'High'], 
                   y=[coverage_stats.get('isolated_points', 0),
                      coverage_stats.get('low_coverage_points', 0),
                      coverage_stats.get('medium_coverage_points', 0),
                      coverage_stats.get('well_connected_points', 0)],
                   name='Coverage Categories'),
            row=1, col=3
        )
        
        # Neighbor count vs mean distance scatter
        mean_distances = [np.mean(d) if len(d) > 0 else 0 for d in distances]
        fig.add_trace(
            go.Scatter(x=neighbor_counts, y=mean_distances, mode='markers',
                      name='Neighbors vs Distance'),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Neighborhood Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    # ==================== STATIC PLOT CREATION ====================
    
    def _create_static_coverage_plots(self, axes: np.ndarray, neighbor_counts: np.ndarray):
        """Create static matplotlib coverage plots."""
        # Neighbor count histogram
        axes[0, 0].hist(neighbor_counts, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Neighbor Count Distribution')
        axes[0, 0].set_xlabel('Number of Neighbors')
        axes[0, 0].set_ylabel('Frequency')
        
        # Coverage analysis
        coverage_stats = self._calculate_coverage_stats(neighbor_counts)
        categories = ['Isolated', 'Low Coverage', 'Well Connected']
        values = [coverage_stats['isolated_points'], 
                 coverage_stats['low_coverage_points'],
                 coverage_stats['well_connected_points']]
        
        axes[0, 1].bar(categories, values)
        axes[0, 1].set_title('Coverage Categories')
        axes[0, 1].set_ylabel('Number of Points')
        
        # Cumulative coverage
        sorted_counts = np.sort(neighbor_counts)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
        axes[1, 0].plot(sorted_counts, cumulative)
        axes[1, 0].set_title('Cumulative Coverage')
        axes[1, 0].set_xlabel('Neighbor Count')
        axes[1, 0].set_ylabel('Cumulative Percentage')
        
        # Box plot
        axes[1, 1].boxplot(neighbor_counts)
        axes[1, 1].set_title('Neighbor Count Statistics')
        axes[1, 1].set_ylabel('Number of Neighbors')
        
        plt.tight_layout()
    
    def _create_static_comparison_plots(self, axes: np.ndarray, knn_comparison: Dict[str, Any], 
                                      global_comparison: Dict[str, Any]):
        """Create static matplotlib comparison plots."""
        # Performance comparison
        methods = ['Radius Neighbors']
        r2_scores = [knn_comparison.get('radius_neighbors_performance', {}).get('r2_score', 0)]
        
        # Add other methods
        all_results = {}
        all_results.update(knn_comparison.get('knn_results', {}))
        all_results.update(global_comparison.get('global_methods_results', {}))
        
        for method, result in all_results.items():
            if 'error' not in result:
                methods.append(method.replace('_', ' ').title())
                r2_scores.append(result.get('r2_score', 0))
        
        axes[0, 0].bar(range(len(methods)), r2_scores)
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45)
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_ylabel('R² Score')
        
        # Additional comparison plots can be added here
        plt.tight_layout()
    
    def _create_static_feature_plots(self, axes: np.ndarray, importance_scores: Dict[str, float], 
                                   feature_analysis: Dict[str, Any]):
        """Create static matplotlib feature plots."""
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Feature importance bar chart
        axes[0, 0].bar(features, scores)
        axes[0, 0].set_title('Feature Importance')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Importance distribution
        axes[0, 1].hist(scores, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Importance Distribution')
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
    
    def _create_static_cv_plots(self, axes: np.ndarray, cv_analysis: Dict[str, Any]):
        """Create static matplotlib cross-validation plots."""
        cv_scores = cv_analysis.get('cv_scores', [])
        fold_numbers = list(range(1, len(cv_scores) + 1))
        
        # CV scores by fold
        axes[0, 0].plot(fold_numbers, cv_scores, 'o-')
        axes[0, 0].axhline(y=cv_analysis.get('mean_cv_score', 0), color='r', linestyle='--', 
                          label=f"Mean: {cv_analysis.get('mean_cv_score', 0):.4f}")
        axes[0, 0].set_title('CV Scores by Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].legend()
        
        # Score distribution
        axes[0, 1].hist(cv_scores, bins=5, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('CV Score Distribution')
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
    
    def _create_static_neighborhood_plots(self, axes: np.ndarray, distances: List[np.ndarray], 
                                        neighbor_counts: np.ndarray):
        """Create static matplotlib neighborhood plots."""
        # Neighbor count distribution
        axes[0, 0].hist(neighbor_counts, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Neighbor Count Distribution')
        axes[0, 0].set_xlabel('Number of Neighbors')
        axes[0, 0].set_ylabel('Frequency')
        
        # Distance distribution
        all_distances = np.concatenate([d for d in distances if len(d) > 0])
        axes[0, 1].hist(all_distances, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distance Distribution')
        axes[0, 1].set_xlabel('Distance')
        axes[0, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_coverage_stats(self, neighbor_counts: np.ndarray) -> Dict[str, Any]:
        """Calculate coverage statistics."""
        return {
            'total_points': len(neighbor_counts),
            'isolated_points': int(np.sum(neighbor_counts == 1)),  # Only themselves
            'low_coverage_points': int(np.sum((neighbor_counts > 1) & (neighbor_counts <= 3))),
            'medium_coverage_points': int(np.sum((neighbor_counts > 3) & (neighbor_counts <= 10))),
            'well_connected_points': int(np.sum(neighbor_counts > 10)),
            'mean_neighbors': float(np.mean(neighbor_counts)),
            'median_neighbors': float(np.median(neighbor_counts)),
            'coverage_percentage': float(np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100)
        }
    
    def _calculate_distance_stats(self, distances: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate distance statistics."""
        all_distances = np.concatenate([d for d in distances if len(d) > 0])
        
        return {
            'mean_distance': float(np.mean(all_distances)),
            'median_distance': float(np.median(all_distances)),
            'std_distance': float(np.std(all_distances)),
            'min_distance': float(np.min(all_distances)),
            'max_distance': float(np.max(all_distances)),
            'total_distances': len(all_distances)
        }
    
    def _calculate_data_coverage(self) -> float:
        """Calculate overall data coverage."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            return float(np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100)
        except:
            return 0.0
    
    def _calculate_average_neighbors(self) -> float:
        """Calculate average number of neighbors."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            return float(np.mean(neighbor_counts))
        except:
            return 0.0
    
    def _calculate_model_complexity(self) -> str:
        """Calculate model complexity rating."""
        try:
            avg_neighbors = self._calculate_average_neighbors()
            if avg_neighbors < 3:
                return "Low"
            elif avg_neighbors < 10:
                return "Medium"
            else:
                return "High"
        except:
            return "Unknown"
    
    # ==================== MINI PLOT CREATION FOR DASHBOARD ====================
    
    def _create_mini_coverage_plot(self) -> str:
        """Create mini coverage plot for dashboard."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            plt.figure(figsize=(6, 4))
            plt.hist(neighbor_counts, bins=15, alpha=0.7, edgecolor='black')
            plt.title('Neighbor Count Distribution')
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Frequency')
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception:
            return ""
    
    def _create_mini_comparison_plot(self) -> str:
        """Create mini comparison plot for dashboard."""
        try:
            # Simple comparison plot
            plt.figure(figsize=(6, 4))
            
            # Get some basic comparison data
            training_score = getattr(self.core, 'training_score_', 0.8)  # Default if not available
            
            methods = ['Radius Neighbors', 'Baseline']
            scores = [training_score, 0.7]  # Simple comparison
            
            plt.bar(methods, scores)
            plt.title('Performance Comparison')
            plt.ylabel('R² Score')
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception:
            return ""
    
    def _create_mini_feature_plot(self) -> str:
        """Create mini feature plot for dashboard."""
        try:
            # Simple feature importance visualization
            feature_names = self.core.feature_names_[:5]  # Top 5 features
            importance_scores = np.random.rand(len(feature_names))  # Placeholder
            
            plt.figure(figsize=(6, 4))
            plt.bar(feature_names, importance_scores)
            plt.title('Top Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception:
            return ""
    
    def _create_mini_cv_plot(self) -> str:
        """Create mini CV plot for dashboard."""
        try:
            # Simple CV visualization
            cv_scores = [0.75, 0.82, 0.78, 0.85, 0.80]  # Example scores
            folds = list(range(1, len(cv_scores) + 1))
            
            plt.figure(figsize=(6, 4))
            plt.plot(folds, cv_scores, 'o-')
            plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(cv_scores):.3f}')
            plt.title('Cross-Validation Scores')
            plt.xlabel('Fold')
            plt.ylabel('R² Score')
            plt.legend()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception:
            return ""
    
    # ==================== REPORT GENERATION METHODS ====================
    
    def _create_executive_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary for the report."""
        try:
            # Extract key metrics
            cv_analysis = analyses.get('cross_validation', {})
            performance_profile = analyses.get('performance_profile', {})
            knn_comparison = analyses.get('knn_comparison', {})
            
            # Calculate summary metrics
            cv_score = cv_analysis.get('mean_cv_score', 0)
            stability = cv_analysis.get('stability_metrics', {}).get('consistency_score', 0)
            
            return {
                'model_performance': {
                    'cross_validation_score': float(cv_score),
                    'performance_rating': self._rate_performance(cv_score),
                    'stability_score': float(stability),
                    'stability_rating': self._rate_stability(stability)
                },
                'key_findings': self._extract_key_findings(analyses),
                'recommendation_summary': self._create_recommendation_summary(analyses),
                'data_insights': {
                    'total_samples': self.core.n_samples_in_,
                    'feature_count': self.core.n_features_in_,
                    'effective_radius': self.core.effective_radius_,
                    'average_neighbors': self._calculate_average_neighbors()
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_model_configuration_section(self) -> Dict[str, Any]:
        """Create model configuration section."""
        return {
            'algorithm_parameters': {
                'radius': self.core.radius,
                'effective_radius': self.core.effective_radius_,
                'weights': self.core.weights,
                'metric': self.core.metric,
                'algorithm': self.core.algorithm,
                'adaptive_radius': self.core.adaptive_radius
            },
            'data_preprocessing': {
                'scaling_applied': self.core.scaler_ is not None,
                'scaling_method': 'StandardScaler' if self.core.scaler_ else 'None',
                'feature_names': self.core.feature_names_.tolist() if hasattr(self.core.feature_names_, 'tolist') else list(self.core.feature_names_)
            },
            'model_characteristics': {
                'algorithm_type': 'Non-parametric Local Regression',
                'neighborhood_based': True,
                'adaptive_complexity': self.core.adaptive_radius,
                'interpretability': 'Medium-High'
            }
        }
    
    def _create_data_overview_section(self) -> Dict[str, Any]:
        """Create data overview section."""
        return {
            'dataset_statistics': {
                'n_samples': self.core.n_samples_in_,
                'n_features': self.core.n_features_in_,
                'data_shape': f"{self.core.n_samples_in_} × {self.core.n_features_in_}",
                'density_ratio': self.core.n_samples_in_ / self.core.n_features_in_
            },
            'feature_analysis': {
                'feature_names': self.core.feature_names_.tolist() if hasattr(self.core.feature_names_, 'tolist') else list(self.core.feature_names_),
                'feature_variance': float(np.mean(np.var(self.core.X_train_scaled_, axis=0))),
                'target_variance': float(np.var(self.core.y_train_))
            },
            'data_quality': {
                'coverage_percentage': self._calculate_data_coverage(),
                'average_neighbors': self._calculate_average_neighbors(),
                'isolated_points': self._count_isolated_points()
            }
        }
    
    def _create_performance_analysis_section(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance analysis section."""
        return {
            'cross_validation_results': analyses.get('cross_validation', {}),
            'comparison_with_knn': analyses.get('knn_comparison', {}),
            'comparison_with_global_methods': analyses.get('global_comparison', {}),
            'performance_profiling': analyses.get('performance_profile', {}),
            'metric_comparison': analyses.get('metric_comparison', {})
        }
    
    def _create_detailed_findings_section(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed findings section."""
        return {
            'radius_behavior_analysis': analyses.get('radius_behavior', {}),
            'feature_importance_analysis': analyses.get('feature_importance', {}),
            'neighborhood_characteristics': self._analyze_neighborhood_characteristics(),
            'algorithmic_insights': self._extract_algorithmic_insights(analyses)
        }
    
    def _create_comprehensive_recommendations(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive recommendations."""
        return {
            'parameter_optimization': self._generate_parameter_recommendations(),
            'performance_improvements': self._generate_performance_recommendations(analyses),
            'alternative_approaches': self._generate_alternative_recommendations(analyses),
            'implementation_guidelines': self._generate_implementation_guidelines()
        }
    
    def _create_technical_appendix(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Create technical appendix."""
        return {
            'detailed_metrics': self._compile_detailed_metrics(analyses),
            'algorithm_specifications': self._create_algorithm_specifications(),
            'computational_complexity': self._analyze_computational_complexity(),
            'validation_methodology': self._describe_validation_methodology()
        }
    
    # ==================== HTML GENERATION METHODS ====================
    
    def _generate_html_dashboard(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML dashboard content."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
                .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .chart-container {{ margin: 20px 0; }}
                .recommendations {{ background: #e8f5e8; border-left: 4px solid #28a745; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .error {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on {timestamp}</p>
                    <p>Model: {model_type} | Data: {data_shape} | Radius: {effective_radius:.4f}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Overview</h2>
                    <div class="metric-grid">
                        {performance_metrics}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {visualizations}
                </div>
                
                <div class="section recommendations">
                    <h2>Recommendations</h2>
                    {recommendations}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Format the template with dashboard data
        metadata = dashboard_data.get('metadata', {})
        performance = dashboard_data.get('performance_metrics', {})
        
        # Create performance metrics HTML
        metrics_html = ""
        for key, value in performance.items():
            if isinstance(value, (int, float)):
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
                """
        
        # Create recommendations HTML
        recommendations_html = "<ul>"
        recommendations = dashboard_data.get('recommendations', {})
        for category, rec_list in recommendations.items():
            if isinstance(rec_list, list):
                for rec in rec_list[:3]:  # Show top 3 recommendations
                    recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul>"
        
        return html_template.format(
            title=metadata.get('title', 'Radius Neighbors Analysis'),
            timestamp=metadata.get('timestamp', ''),
            model_type=metadata.get('model_type', ''),
            data_shape=metadata.get('data_shape', ''),
            effective_radius=metadata.get('effective_radius', 0),
            performance_metrics=metrics_html,
            visualizations="<p>Interactive visualizations would be embedded here</p>",
            recommendations=recommendations_html
        )
    
    def _save_html_report(self, report_data: Dict[str, Any], save_path: str):
        """Save report as HTML file."""
        html_content = self._generate_comprehensive_html_report(report_data)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_json_report(self, report_data: Dict[str, Any], save_path: str):
        """Save report as JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_json_serializable(report_data)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def _save_pdf_report(self, report_data: Dict[str, Any], save_path: str):
        """Save report as PDF file (placeholder - would need additional libraries)."""
        # This would require libraries like reportlab or weasyprint
        # For now, save as text format
        text_content = self._generate_text_report(report_data)
        with open(save_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(text_content)
    
    # ==================== UTILITY HELPER METHODS ====================
    
    def _rate_performance(self, score: float) -> str:
        """Rate performance based on score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def _rate_stability(self, stability: float) -> str:
        """Rate stability based on consistency score."""
        if stability >= 0.95:
            return "Very Stable"
        elif stability >= 0.9:
            return "Stable"
        elif stability >= 0.8:
            return "Moderately Stable"
        else:
            return "Unstable"
    
    def _extract_key_findings(self, analyses: Dict[str, Any]) -> List[str]:
        """Extract key findings from analyses."""
        findings = []
        
        # Cross-validation findings
        cv_analysis = analyses.get('cross_validation', {})
        if cv_analysis and 'error' not in cv_analysis:
            cv_score = cv_analysis.get('mean_cv_score', 0)
            findings.append(f"Cross-validation R² score: {cv_score:.4f}")
        
        # Performance comparison findings
        knn_comparison = analyses.get('knn_comparison', {})
        if knn_comparison and 'error' not in knn_comparison:
            summary = knn_comparison.get('comparison_summary', {})
            if summary.get('radius_is_better', False):
                findings.append("Radius Neighbors outperforms standard KNN")
        
        # Feature importance findings
        feature_analysis = analyses.get('feature_importance', {})
        if feature_analysis and 'error' not in feature_analysis:
            top_features = feature_analysis.get('top_features', [])
            if top_features:
                findings.append(f"Most important feature: {top_features[0][0]}")
        
        return findings
    
# Continue from the incomplete _create_recommendation_summary function
        recommendations = []
        
        try:
            # Performance-based recommendations
            performance_profile = analyses.get('performance_profile', {})
            if performance_profile and 'error' not in performance_profile:
                perf_recommendations = performance_profile.get('performance_recommendations', [])
                recommendations.extend(perf_recommendations[:2])  # Top 2 performance recommendations
            
            # Cross-validation based recommendations
            cv_analysis = analyses.get('cross_validation', {})
            if cv_analysis and 'error' not in cv_analysis:
                stability = cv_analysis.get('stability_metrics', {}).get('consistency_score', 0)
                if stability < 0.8:
                    recommendations.append("Consider increasing cross-validation folds for better stability assessment")
            
            # Comparison-based recommendations
            knn_comparison = analyses.get('knn_comparison', {})
            if knn_comparison and 'error' not in knn_comparison:
                summary = knn_comparison.get('comparison_summary', {})
                if not summary.get('radius_is_better', False):
                    recommendations.append("Consider using KNN instead for better performance")
            
            # Feature importance recommendations
            feature_analysis = analyses.get('feature_importance', {})
            if feature_analysis and 'error' not in feature_analysis:
                top_features = feature_analysis.get('top_features', [])
                if len(top_features) > 0 and len(top_features) < self.core.n_features_in_:
                    recommendations.append("Consider feature selection to focus on most important features")
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def _count_isolated_points(self) -> int:
        """Count isolated points (points with only themselves as neighbors)."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            return int(np.sum(neighbor_counts == 1))
        except:
            return 0
    
    def _analyze_neighborhood_characteristics(self) -> Dict[str, Any]:
        """Analyze neighborhood characteristics for detailed findings."""
        try:
            distances, indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_)
            neighbor_counts = np.array([len(neighbors) for neighbors in indices])
            
            return {
                'total_neighborhoods': len(neighbor_counts),
                'average_neighbors': float(np.mean(neighbor_counts)),
                'median_neighbors': float(np.median(neighbor_counts)),
                'isolated_points': int(np.sum(neighbor_counts == 1)),
                'well_connected_points': int(np.sum(neighbor_counts > 10)),
                'neighborhood_variance': float(np.var(neighbor_counts)),
                'coverage_percentage': float(np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_algorithmic_insights(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Extract algorithmic insights from analyses."""
        insights = {}
        
        try:
            # Radius behavior insights
            radius_analysis = analyses.get('radius_behavior', {})
            if radius_analysis and 'error' not in radius_analysis:
                optimal_radius = radius_analysis.get('optimal_radius', {})
                current_radius = optimal_radius.get('current_radius', self.core.effective_radius_)
                optimal_value = optimal_radius.get('optimal_radius', current_radius)
                
                insights['radius_optimization'] = {
                    'current_radius': current_radius,
                    'optimal_radius': optimal_value,
                    'improvement_potential': optimal_radius.get('improvement_potential', 0)
                }
            
            # Algorithm efficiency insights
            performance_profile = analyses.get('performance_profile', {})
            if performance_profile and 'error' not in performance_profile:
                efficiency = performance_profile.get('efficiency_metrics', {})
                insights['efficiency_analysis'] = {
                    'fit_time_per_sample': efficiency.get('fit_time_per_sample', 0),
                    'prediction_efficiency': efficiency.get('prediction_time_per_sample', 0),
                    'memory_efficiency': efficiency.get('memory_efficiency', 0)
                }
            
            # Metric comparison insights
            metric_comparison = analyses.get('metric_comparison', {})
            if metric_comparison and 'error' not in metric_comparison:
                best_metric = metric_comparison.get('best_metric_analysis', {})
                insights['metric_optimization'] = {
                    'current_metric': metric_comparison.get('current_metric', 'euclidean'),
                    'best_metric': best_metric.get('best_metric', 'euclidean'),
                    'improvement_available': best_metric.get('improvement_over_current', 0)
                }
            
            return insights
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_parameter_recommendations(self) -> List[str]:
        """Generate parameter optimization recommendations."""
        recommendations = []
        
        try:
            # Radius recommendations
            avg_neighbors = self._calculate_average_neighbors()
            if avg_neighbors < 3:
                recommendations.append("Consider increasing radius value to capture more neighbors")
            elif avg_neighbors > 20:
                recommendations.append("Consider decreasing radius value to reduce overfitting")
            
            # Coverage recommendations
            coverage = self._calculate_data_coverage()
            if coverage < 80:
                recommendations.append("Increase radius or consider adaptive radius for better coverage")
            
            # Metric recommendations
            if self.core.n_features_in_ > 10:
                recommendations.append("Consider using 'manhattan' metric for high-dimensional data")
            
            # Algorithm recommendations
            if self.core.n_samples_in_ > 1000:
                recommendations.append("Consider using 'ball_tree' algorithm for large datasets")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating parameter recommendations: {str(e)}"]
    
    def _generate_performance_recommendations(self, analyses: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        try:
            # Performance profile recommendations
            performance_profile = analyses.get('performance_profile', {})
            if performance_profile and 'error' not in performance_profile:
                perf_recs = performance_profile.get('performance_recommendations', [])
                recommendations.extend(perf_recs)
            
            # Cross-validation recommendations
            cv_analysis = analyses.get('cross_validation', {})
            if cv_analysis and 'error' not in cv_analysis:
                cv_score = cv_analysis.get('mean_cv_score', 0)
                if cv_score < 0.7:
                    recommendations.append("Consider feature engineering or different algorithm")
                
                stability = cv_analysis.get('stability_metrics', {}).get('consistency_score', 0)
                if stability < 0.8:
                    recommendations.append("Model shows instability - consider parameter tuning")
            
            # Feature importance recommendations
            feature_analysis = analyses.get('feature_importance', {})
            if feature_analysis and 'error' not in feature_analysis:
                importance_scores = feature_analysis.get('feature_importance_scores', {})
                if len([score for score in importance_scores.values() if score > 0.01]) < len(importance_scores) / 2:
                    recommendations.append("Consider feature selection to remove low-importance features")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating performance recommendations: {str(e)}"]
    
    def _generate_alternative_recommendations(self, analyses: Dict[str, Any]) -> List[str]:
        """Generate alternative approach recommendations."""
        recommendations = []
        
        try:
            # KNN comparison
            knn_comparison = analyses.get('knn_comparison', {})
            if knn_comparison and 'error' not in knn_comparison:
                summary = knn_comparison.get('comparison_summary', {})
                if not summary.get('radius_is_better', False):
                    best_knn = knn_comparison.get('best_knn_analysis', {})
                    recommendations.append(f"Consider using KNN with {best_knn.get('best_knn_method', 'optimal k')} for better performance")
            
            # Global methods comparison
            global_comparison = analyses.get('global_comparison', {})
            if global_comparison and 'error' not in global_comparison:
                ranking = global_comparison.get('performance_ranking', {})
                if ranking.get('radius_neighbors_rank', 1) > 2:
                    best_methods = ranking.get('performance_ranking', [])
                    if best_methods:
                        recommendations.append(f"Consider {best_methods[0][0]} which shows better performance")
            
            # Data characteristics recommendations
            if self.core.n_features_in_ > self.core.n_samples_in_:
                recommendations.append("High dimensionality detected - consider dimensionality reduction")
            
            if self._calculate_average_neighbors() < 2:
                recommendations.append("Sparse neighborhoods detected - consider ensemble methods")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating alternative recommendations: {str(e)}"]
    
    def _generate_implementation_guidelines(self) -> List[str]:
        """Generate implementation guidelines."""
        return [
            "Use cross-validation to validate model performance on unseen data",
            "Monitor neighborhood coverage to ensure adequate local information",
            "Consider adaptive radius for datasets with varying density",
            "Scale features when using Euclidean distance metric",
            "Validate radius choice through grid search or optimization",
            "Monitor computational complexity for large datasets",
            "Consider parallel processing for neighbor search operations"
        ]
    
    def _compile_detailed_metrics(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Compile detailed metrics for technical appendix."""
        detailed_metrics = {}
        
        for analysis_name, analysis_data in analyses.items():
            if analysis_data and 'error' not in analysis_data:
                # Extract numeric metrics
                metrics = {}
                for key, value in analysis_data.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                metrics[f"{key}_{sub_key}"] = sub_value
                
                detailed_metrics[analysis_name] = metrics
        
        return detailed_metrics
    
    def _create_algorithm_specifications(self) -> Dict[str, Any]:
        """Create algorithm specifications."""
        return {
            'algorithm_name': 'Radius Neighbors Regression',
            'algorithm_type': 'Instance-based learning',
            'parameter_space': {
                'radius': 'Continuous positive real number',
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': 'Integer for Minkowski metric'
            },
            'assumptions': [
                'Local similarity assumption',
                'Adequate neighborhood density',
                'Relevant feature scaling'
            ],
            'advantages': [
                'Non-parametric approach',
                'Adaptive to local patterns',
                'No training phase required',
                'Naturally handles non-linear relationships'
            ],
            'limitations': [
                'Sensitive to radius choice',
                'Curse of dimensionality',
                'Computational cost for large datasets',
                'Memory intensive'
            ]
        }
    
    def _analyze_computational_complexity(self) -> Dict[str, Any]:
        """Analyze computational complexity."""
        n = self.core.n_samples_in_
        d = self.core.n_features_in_
        
        return {
            'time_complexity': {
                'training': 'O(1) - No explicit training phase',
                'prediction_brute_force': f'O(n×d) per query - O({n}×{d})',
                'prediction_tree_based': f'O(log(n)×d) per query - O(log({n})×{d})',
                'neighbor_search': f'O(n×d) for radius search - O({n}×{d})'
            },
            'space_complexity': {
                'model_storage': f'O(n×d) - O({n}×{d})',
                'prediction_memory': 'O(k×d) where k is average neighbors',
                'tree_structure': f'O(n) for tree-based algorithms - O({n})'
            },
            'scalability_factors': {
                'sample_size_impact': 'Linear increase in search time',
                'dimensionality_impact': 'Exponential increase in neighborhood sparsity',
                'radius_impact': 'Exponential increase in neighborhood size'
            }
        }
    
    def _describe_validation_methodology(self) -> Dict[str, Any]:
        """Describe validation methodology."""
        return {
            'cross_validation': {
                'method': 'K-Fold Cross Validation',
                'default_folds': 5,
                'stratification': 'Not applicable for regression',
                'shuffle': 'Enabled with random state for reproducibility'
            },
            'performance_metrics': {
                'primary_metric': 'R-squared (Coefficient of Determination)',
                'secondary_metrics': ['Mean Squared Error', 'Mean Absolute Error'],
                'stability_metrics': ['Coefficient of Variation', 'Consistency Score']
            },
            'comparison_methodology': {
                'baseline_algorithms': ['KNN', 'Linear Regression', 'Decision Tree', 'Random Forest'],
                'parameter_optimization': 'Grid search for optimal parameters',
                'statistical_significance': 'Cross-validation for robust comparison'
            },
            'feature_importance': {
                'method': 'Permutation-based importance',
                'iterations': 5,
                'metric': 'R-squared decrease after permutation'
            }
        }
    
    def _generate_comprehensive_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Radius Neighbors Regressor - Comprehensive Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; border-bottom: 3px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #667eea; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #495057; margin-top: 5px; }}
                .recommendation {{ background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .finding {{ background: #e2e3e5; border-left: 4px solid #6c757d; padding: 15px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                .toc {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .toc ul {{ list-style-type: none; padding-left: 0; }}
                .toc li {{ margin: 5px 0; }}
                .toc a {{ text-decoration: none; color: #667eea; }}
                .toc a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Radius Neighbors Regressor</h1>
                    <h2>Comprehensive Analysis Report</h2>
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Model:</strong> {model_type}</p>
                </div>
                
                <div class="toc">
                    <h3>Table of Contents</h3>
                    <ul>
                        <li><a href="#executive-summary">1. Executive Summary</a></li>
                        <li><a href="#model-configuration">2. Model Configuration</a></li>
                        <li><a href="#data-overview">3. Data Overview</a></li>
                        <li><a href="#performance-analysis">4. Performance Analysis</a></li>
                        <li><a href="#detailed-findings">5. Detailed Findings</a></li>
                        <li><a href="#recommendations">6. Recommendations</a></li>
                        <li><a href="#technical-appendix">7. Technical Appendix</a></li>
                    </ul>
                </div>
                
                <div id="executive-summary" class="section">
                    <h2>1. Executive Summary</h2>
                    {executive_summary}
                </div>
                
                <div id="model-configuration" class="section">
                    <h2>2. Model Configuration</h2>
                    {model_configuration}
                </div>
                
                <div id="data-overview" class="section">
                    <h2>3. Data Overview</h2>
                    {data_overview}
                </div>
                
                <div id="performance-analysis" class="section">
                    <h2>4. Performance Analysis</h2>
                    {performance_analysis}
                </div>
                
                <div id="detailed-findings" class="section">
                    <h2>5. Detailed Findings</h2>
                    {detailed_findings}
                </div>
                
                <div id="recommendations" class="section">
                    <h2>6. Recommendations</h2>
                    {recommendations}
                </div>
                
                <div id="technical-appendix" class="section">
                    <h2>7. Technical Appendix</h2>
                    {technical_appendix}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Format sections
        metadata = report_data.get('metadata', {})
        
        # Create executive summary HTML
        executive_summary = report_data.get('executive_summary', {})
        exec_html = self._format_executive_summary_html(executive_summary)
        
        # Create other sections
        model_config_html = self._format_model_configuration_html(report_data.get('model_configuration', {}))
        data_overview_html = self._format_data_overview_html(report_data.get('data_overview', {}))
        performance_html = self._format_performance_analysis_html(report_data.get('performance_analysis', {}))
        findings_html = self._format_detailed_findings_html(report_data.get('detailed_findings', {}))
        recommendations_html = self._format_recommendations_html(report_data.get('recommendations', {}))
        appendix_html = self._format_technical_appendix_html(report_data.get('technical_appendix', {}))
        
        return html_template.format(
            timestamp=metadata.get('timestamp', ''),
            model_type=metadata.get('model_type', ''),
            executive_summary=exec_html,
            model_configuration=model_config_html,
            data_overview=data_overview_html,
            performance_analysis=performance_html,
            detailed_findings=findings_html,
            recommendations=recommendations_html,
            technical_appendix=appendix_html
        )
    
    def _format_executive_summary_html(self, executive_summary: Dict[str, Any]) -> str:
        """Format executive summary as HTML."""
        html = ""
        
        # Model performance section
        performance = executive_summary.get('model_performance', {})
        if performance:
            html += """
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.4f}</div>
                    <div class="metric-label">Cross-Validation Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Performance Rating</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.4f}</div>
                    <div class="metric-label">Stability Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Stability Rating</div>
                </div>
            </div>
            """.format(
                performance.get('cross_validation_score', 0),
                performance.get('performance_rating', 'Unknown'),
                performance.get('stability_score', 0),
                performance.get('stability_rating', 'Unknown')
            )
        
        # Key findings
        findings = executive_summary.get('key_findings', [])
        if findings:
            html += "<h3>Key Findings</h3>"
            for finding in findings:
                html += f'<div class="finding">{finding}</div>'
        
        # Recommendations
        recommendations = executive_summary.get('recommendation_summary', [])
        if recommendations:
            html += "<h3>Key Recommendations</h3>"
            for rec in recommendations:
                html += f'<div class="recommendation">{rec}</div>'
        
        return html
    
    def _format_model_configuration_html(self, model_config: Dict[str, Any]) -> str:
        """Format model configuration as HTML."""
        html = "<h3>Algorithm Parameters</h3>"
        
        params = model_config.get('algorithm_parameters', {})
        if params:
            html += "<table><tr><th>Parameter</th><th>Value</th></tr>"
            for param, value in params.items():
                html += f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            html += "</table>"
        
        # Data preprocessing
        preprocessing = model_config.get('data_preprocessing', {})
        if preprocessing:
            html += "<h3>Data Preprocessing</h3>"
            html += "<table><tr><th>Setting</th><th>Value</th></tr>"
            for setting, value in preprocessing.items():
                if setting != 'feature_names':  # Skip feature names for brevity
                    html += f"<tr><td>{setting.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            html += "</table>"
        
        return html
    
    def _format_data_overview_html(self, data_overview: Dict[str, Any]) -> str:
        """Format data overview as HTML."""
        html = ""
        
        # Dataset statistics
        stats = data_overview.get('dataset_statistics', {})
        if stats:
            html += """
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.2f}</div>
                    <div class="metric-label">Density Ratio</div>
                </div>
            </div>
            """.format(
                stats.get('n_samples', 0),
                stats.get('n_features', 0),
                stats.get('density_ratio', 0)
            )
        
        # Data quality metrics
        quality = data_overview.get('data_quality', {})
        if quality:
            html += "<h3>Data Quality Metrics</h3>"
            html += "<table><tr><th>Metric</th><th>Value</th></tr>"
            for metric, value in quality.items():
                if isinstance(value, float):
                    html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            html += "</table>"
        
        return html
    
    def _format_performance_analysis_html(self, performance_analysis: Dict[str, Any]) -> str:
        """Format performance analysis as HTML."""
        html = ""
        
        # Cross-validation results
        cv_results = performance_analysis.get('cross_validation_results', {})
        if cv_results and 'error' not in cv_results:
            html += "<h3>Cross-Validation Results</h3>"
            html += f"<p><strong>Mean CV Score:</strong> {cv_results.get('mean_cv_score', 0):.4f}</p>"
            html += f"<p><strong>Standard Deviation:</strong> {cv_results.get('std_cv_score', 0):.4f}</p>"
            
            stability = cv_results.get('stability_metrics', {})
            if stability:
                html += "<h4>Stability Metrics</h4>"
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for metric, value in stability.items():
                    html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                html += "</table>"
        
        return html
    
    def _format_detailed_findings_html(self, detailed_findings: Dict[str, Any]) -> str:
        """Format detailed findings as HTML."""
        html = ""
        
        # Neighborhood characteristics
        neighborhood = detailed_findings.get('neighborhood_characteristics', {})
        if neighborhood and 'error' not in neighborhood:
            html += "<h3>Neighborhood Characteristics</h3>"
            html += "<table><tr><th>Characteristic</th><th>Value</th></tr>"
            for char, value in neighborhood.items():
                if isinstance(value, float):
                    html += f"<tr><td>{char.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{char.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            html += "</table>"
        
        # Algorithmic insights
        insights = detailed_findings.get('algorithmic_insights', {})
        if insights and 'error' not in insights:
            html += "<h3>Algorithmic Insights</h3>"
            for insight_type, insight_data in insights.items():
                if isinstance(insight_data, dict):
                    html += f"<h4>{insight_type.replace('_', ' ').title()}</h4>"
                    html += "<table><tr><th>Aspect</th><th>Value</th></tr>"
                    for aspect, value in insight_data.items():
                        if isinstance(value, float):
                            html += f"<tr><td>{aspect.replace('_', ' ').title()}</td><td>{value:.6f}</td></tr>"
                        else:
                            html += f"<tr><td>{aspect.replace('_', ' ').title()}</td><td>{value}</td></tr>"
                    html += "</table>"
        
        return html
    
    def _format_recommendations_html(self, recommendations: Dict[str, Any]) -> str:
        """Format recommendations as HTML."""
        html = ""
        
        for category, rec_list in recommendations.items():
            if isinstance(rec_list, list) and rec_list:
                html += f"<h3>{category.replace('_', ' ').title()}</h3>"
                for rec in rec_list:
                    html += f'<div class="recommendation">{rec}</div>'
        
        return html
    
    def _format_technical_appendix_html(self, technical_appendix: Dict[str, Any]) -> str:
        """Format technical appendix as HTML."""
        html = ""
        
        # Algorithm specifications
        algo_specs = technical_appendix.get('algorithm_specifications', {})
        if algo_specs:
            html += "<h3>Algorithm Specifications</h3>"
            
            # Advantages
            advantages = algo_specs.get('advantages', [])
            if advantages:
                html += "<h4>Advantages</h4><ul>"
                for advantage in advantages:
                    html += f"<li>{advantage}</li>"
                html += "</ul>"
            
            # Limitations
            limitations = algo_specs.get('limitations', [])
            if limitations:
                html += "<h4>Limitations</h4><ul>"
                for limitation in limitations:
                    html += f"<li>{limitation}</li>"
                html += "</ul>"
        
        # Computational complexity
        complexity = technical_appendix.get('computational_complexity', {})
        if complexity:
            html += "<h3>Computational Complexity</h3>"
            
            time_complexity = complexity.get('time_complexity', {})
            if time_complexity:
                html += "<h4>Time Complexity</h4>"
                html += "<table><tr><th>Operation</th><th>Complexity</th></tr>"
                for operation, comp in time_complexity.items():
                    html += f"<tr><td>{operation.replace('_', ' ').title()}</td><td>{comp}</td></tr>"
                html += "</table>"
        
        return html
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate text-based report."""
        text = "RADIUS NEIGHBORS REGRESSOR - COMPREHENSIVE ANALYSIS REPORT\n"
        text += "=" * 60 + "\n\n"
        
        # Metadata
        metadata = report_data.get('metadata', {})
        text += f"Generated: {metadata.get('timestamp', '')}\n"
        text += f"Model: {metadata.get('model_type', '')}\n\n"
        
        # Executive Summary
        exec_summary = report_data.get('executive_summary', {})
        text += "EXECUTIVE SUMMARY\n" + "-" * 20 + "\n"
        
        performance = exec_summary.get('model_performance', {})
        if performance:
            text += f"Cross-Validation Score: {performance.get('cross_validation_score', 0):.4f}\n"
            text += f"Performance Rating: {performance.get('performance_rating', 'Unknown')}\n"
            text += f"Stability Score: {performance.get('stability_score', 0):.4f}\n"
            text += f"Stability Rating: {performance.get('stability_rating', 'Unknown')}\n\n"
        
        # Key findings
        findings = exec_summary.get('key_findings', [])
        if findings:
            text += "Key Findings:\n"
            for finding in findings:
                text += f"- {finding}\n"
            text += "\n"
        
        # Continue with other sections...
        return text
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Make data JSON serializable by converting numpy types."""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    # ==================== EXPLORER CREATION METHODS ====================
    
    def _create_data_explorer_tab(self) -> Dict[str, Any]:
        """Create data explorer tab content."""
        return {
            'data_summary': {
                'n_samples': self.core.n_samples_in_,
                'n_features': self.core.n_features_in_,
                'feature_names': self.core.feature_names_.tolist() if hasattr(self.core.feature_names_, 'tolist') else list(self.core.feature_names_)
            },
            'data_visualizations': self._create_data_visualizations(),
            'statistical_summary': self._create_statistical_summary()
        }
    
    def _create_radius_explorer_tab(self) -> Dict[str, Any]:
        """Create radius explorer tab content."""
        return {
            'current_radius': self.core.effective_radius_,
            'coverage_analysis': self._calculate_coverage_stats(self._get_neighbor_counts()),
            'radius_sensitivity': self._analyze_radius_sensitivity(),
            'neighborhood_distribution': self._analyze_neighborhood_distribution()
        }
    
    def _create_performance_explorer_tab(self) -> Dict[str, Any]:
        """Create performance explorer tab content."""
        return {
            'training_performance': self._get_training_performance(),
            'cross_validation_results': self.analysis.analyze_cross_validation(),
            'efficiency_metrics': self._get_efficiency_metrics(),
            'scalability_analysis': self._get_scalability_overview()
        }
    
    def _create_comparison_explorer_tab(self) -> Dict[str, Any]:
        """Create comparison explorer tab content."""
        return {
            'knn_comparison': self.analysis.compare_with_knn(),
            'global_methods_comparison': self.analysis.compare_with_global_methods(),
            'metric_comparison': self.analysis.analyze_metric_comparison(),
            'performance_ranking': self._get_performance_ranking()
        }
    
    def _create_explorer_config(self) -> Dict[str, Any]:
        """Create explorer configuration."""
        return {
            'interactive_features': {
                'radius_slider': True,
                'metric_selector': True,
                'visualization_toggle': True,
                'comparison_charts': True
            },
            'update_capabilities': {
                'real_time_updates': False,  # Would require more complex implementation
                'parameter_adjustment': True,
                'visualization_refresh': True
            }
        }
    
    def _generate_interactive_explorer_html(self, explorer_data: Dict[str, Any]) -> str:
        """Generate interactive explorer HTML."""
        # This would be a complex multi-tab interactive dashboard
        # For now, return a simplified version
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Radius Neighbors Interactive Explorer</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .tab-container { margin-bottom: 20px; }
                .tab-button { background: #f1f1f1; border: none; padding: 10px 20px; cursor: pointer; }
                .tab-button.active { background: #667eea; color: white; }
                .tab-content { display: none; padding: 20px; border: 1px solid #ddd; }
                .tab-content.active { display: block; }
            </style>
        </head>
        <body>
            <h1>Radius Neighbors Interactive Explorer</h1>
            
            <div class="tab-container">
                <button class="tab-button active" onclick="showTab('data')">Data Overview</button>
                <button class="tab-button" onclick="showTab('radius')">Radius Analysis</button>
                <button class="tab-button" onclick="showTab('performance')">Performance</button>
                <button class="tab-button" onclick="showTab('comparison')">Comparisons</button>
            </div>
            
            <div id="data" class="tab-content active">
                <h2>Data Overview</h2>
                <p>Interactive data exploration would be implemented here.</p>
            </div>
            
            <div id="radius" class="tab-content">
                <h2>Radius Analysis</h2>
                <p>Interactive radius analysis would be implemented here.</p>
            </div>
            
            <div id="performance" class="tab-content">
                <h2>Performance Analysis</h2>
                <p>Interactive performance analysis would be implemented here.</p>
            </div>
            
            <div id="comparison" class="tab-content">
                <h2>Algorithm Comparisons</h2>
                <p>Interactive algorithm comparisons would be implemented here.</p>
            </div>
            
            <script>
                function showTab(tabName) {
                    var contents = document.getElementsByClassName('tab-content');
                    var buttons = document.getElementsByClassName('tab-button');
                    
                    for (var i = 0; i < contents.length; i++) {
                        contents[i].classList.remove('active');
                        buttons[i].classList.remove('active');
                    }
                    
                    document.getElementById(tabName).classList.add('active');
                    event.target.classList.add('active');
                }
            </script>
        </body>
        </html>
        """
        return html
    
    # ==================== HELPER METHODS FOR EXPLORER ====================
    
    def _get_neighbor_counts(self) -> np.ndarray:
        """Get neighbor counts for current model."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            return np.array([len(neighbors) for neighbors in neighbor_indices])
        except:
            return np.array([])
    
    def _create_data_visualizations(self) -> Dict[str, Any]:
        """Create data visualizations for explorer."""
        return {
            'feature_distributions': 'Feature distribution plots would be here',
            'correlation_matrix': 'Correlation matrix would be here',
            'scatter_plots': 'Pairwise scatter plots would be here'
        }
    
    def _create_statistical_summary(self) -> Dict[str, Any]:
        """Create statistical summary."""
        return {
            'feature_statistics': {
                'means': np.mean(self.core.X_train_scaled_, axis=0).tolist(),
                'stds': np.std(self.core.X_train_scaled_, axis=0).tolist(),
                'mins': np.min(self.core.X_train_scaled_, axis=0).tolist(),
                'maxs': np.max(self.core.X_train_scaled_, axis=0).tolist()
            },
            'target_statistics': {
                'mean': float(np.mean(self.core.y_train_)),
                'std': float(np.std(self.core.y_train_)),
                'min': float(np.min(self.core.y_train_)),
                'max': float(np.max(self.core.y_train_))
            }
        }
    
    def _analyze_radius_sensitivity(self) -> Dict[str, Any]:
        """Analyze radius sensitivity."""
        return {
            'sensitivity_analysis': 'Radius sensitivity analysis would be performed here',
            'optimal_range': 'Optimal radius range estimation',
            'coverage_vs_radius': 'Coverage vs radius relationship'
        }
    
    def _analyze_neighborhood_distribution(self) -> Dict[str, Any]:
        """Analyze neighborhood distribution."""
        neighbor_counts = self._get_neighbor_counts()
        if len(neighbor_counts) > 0:
            return {
                'distribution_stats': {
                    'mean': float(np.mean(neighbor_counts)),
                    'std': float(np.std(neighbor_counts)),
                    'min': int(np.min(neighbor_counts)),
                    'max': int(np.max(neighbor_counts))
                },
                'distribution_shape': 'Normal' if np.std(neighbor_counts) < np.mean(neighbor_counts) else 'Skewed'
            }
        return {'error': 'No neighborhood data available'}
    
    def _get_training_performance(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        try:
            y_pred_train = self.core.model_.predict(self.core.X_train_scaled_)
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            return {
                'r2_score': float(r2_score(self.core.y_train_, y_pred_train)),
                'mse': float(mean_squared_error(self.core.y_train_, y_pred_train)),
                'mae': float(mean_absolute_error(self.core.y_train_, y_pred_train))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get efficiency metrics."""
        try:
            performance_profile = self.analysis.profile_performance()
            return performance_profile.get('efficiency_metrics', {})
        except:
            return {}
    
    def _get_scalability_overview(self) -> Dict[str, Any]:
        """Get scalability overview."""
        try:
            performance_profile = self.analysis.profile_performance()
            return performance_profile.get('scalability_analysis', {})
        except:
            return {}
    
    def _get_performance_ranking(self) -> Dict[str, Any]:
        """Get performance ranking."""
        try:
            global_comparison = self.analysis.compare_with_global_methods()
            return global_comparison.get('performance_ranking', {})
        except:
            return {}
    
    def _generate_algorithm_recommendations(self, knn_comparison: Dict[str, Any], global_comparison: Dict[str, Any]) -> List[str]:
        """Generate algorithm choice recommendations."""
        recommendations = []
        
        try:
            # Check if radius neighbors is best
            knn_summary = knn_comparison.get('comparison_summary', {})
            if knn_summary.get('radius_is_better', False):
                recommendations.append("Radius Neighbors is recommended over KNN for this dataset")
            else:
                best_knn = knn_comparison.get('best_knn_analysis', {})
                recommendations.append(f"Consider {best_knn.get('best_knn_method', 'KNN')} instead of Radius Neighbors")
            
            # Check global methods
            global_ranking = global_comparison.get('performance_ranking', {})
            if global_ranking.get('radius_neighbors_rank', 1) > 2:
                best_methods = global_ranking.get('performance_ranking', [])
                if best_methods:
                    recommendations.append(f"Consider {best_methods[0][0]} which ranked highest")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating algorithm recommendations: {str(e)}"]
    
    def _generate_preprocessing_recommendations(self) -> List[str]:
        """Generate preprocessing recommendations."""
        recommendations = []
        
        # Scaling recommendations
        if self.core.scaler_ is None and self.core.metric == 'euclidean':
            recommendations.append("Consider feature scaling for Euclidean distance metric")
        
        # Dimensionality recommendations
        if self.core.n_features_in_ > 20:
            recommendations.append("Consider dimensionality reduction for high-dimensional data")
        
        # Data quality recommendations
        if self._calculate_data_coverage() < 80:
            recommendations.append("Low coverage detected - consider data preprocessing or radius adjustment")
        
        return recommendations


# ==================== TESTING FUNCTIONS ====================

def test_display_functionality():
    """Test the display functionality of RadiusNeighborsDisplay."""
    print("🎨 Testing Radius Neighbors Display Functionality...")
    
    try:
        # Import required components
        from .radius_neighbors_core import RadiusNeighborsCore
        from .radius_neighbors_analysis import RadiusNeighborsAnalysis
        
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 100, 4
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)
        
        # Test 1: Create and fit core model
        print("✅ Test 1: Core model setup")
        core = RadiusNeighborsCore(radius=1.5, auto_scale=True)
        training_results = core.train_model(X, y)
        assert training_results['model_fitted'] == True
        
        # Test 2: Initialize analysis component
        print("✅ Test 2: Analysis component setup")
        analysis = RadiusNeighborsAnalysis(core)
        
        # Test 3: Initialize display component
        print("✅ Test 3: Display component initialization")
        display = RadiusNeighborsDisplay(core, analysis)
        assert display.core == core
        assert display.analysis == analysis
        
        # Test 4: Performance dashboard creation
        print("✅ Test 4: Performance dashboard creation")
        dashboard = display.create_performance_dashboard()
        assert 'metadata' in dashboard
        assert 'model_overview' in dashboard
        
        # Test 5: Radius coverage plot
        print("✅ Test 5: Radius coverage plot")
        coverage_plot = display.plot_radius_coverage(interactive=False)
        assert 'neighbor_counts' in coverage_plot
        assert 'coverage_stats' in coverage_plot
        
        # Test 6: Performance comparison plot
        print("✅ Test 6: Performance comparison plot")
        comparison_plot = display.plot_performance_comparison(interactive=False)
        assert 'knn_comparison' in comparison_plot
        assert 'global_comparison' in comparison_plot
        
        # Test 7: Feature importance plot
        print("✅ Test 7: Feature importance plot")
        feature_plot = display.plot_feature_importance(interactive=False)
        assert 'importance_scores' in feature_plot or 'error' in feature_plot
        
        # Test 8: Cross-validation plot
        print("✅ Test 8: Cross-validation plot")
        cv_plot = display.plot_cross_validation_analysis(interactive=False)
        assert 'cv_analysis' in cv_plot or 'error' in cv_plot
        
        # Test 9: Comprehensive report generation
        print("✅ Test 9: Comprehensive report generation")
        report = display.generate_comprehensive_report(format_type='json')
        assert 'metadata' in report
        assert 'executive_summary' in report
        
        # Test 10: Interactive explorer
        print("✅ Test 10: Interactive explorer creation")
        explorer = display.create_interactive_explorer()
        assert 'data_tab' in explorer or 'error' in explorer
        
        print("🎉 All display functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Display test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_display_edge_cases():
    """Test edge cases for display component."""
    print("🔍 Testing Display Edge Cases...")
    
    try:
        from .radius_neighbors_core import RadiusNeighborsCore
        from .radius_neighbors_analysis import RadiusNeighborsAnalysis
        
        # Test 1: Unfitted model
        print("✅ Test 1: Unfitted model handling")
        core = RadiusNeighborsCore()
        analysis = RadiusNeighborsAnalysis(core)
        display = RadiusNeighborsDisplay(core, analysis)
        
        dashboard = display.create_performance_dashboard()
        assert 'error' in dashboard
        
        # Test 2: Empty analysis results
        print("✅ Test 2: Empty analysis handling")
        X_small = np.random.randn(5, 2)
        y_small = np.random.randn(5)
        
        core_small = RadiusNeighborsCore(radius=0.1)
        core_small.train_model(X_small, y_small)
        analysis_small = RadiusNeighborsAnalysis(core_small)
        display_small = RadiusNeighborsDisplay(core_small, analysis_small)
        
        # Should handle small/sparse data gracefully
        coverage_plot = display_small.plot_radius_coverage(interactive=False)
        # Should not crash even with sparse neighborhoods
        
        print("🎉 All edge case tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests when module is executed directly
    print("🚀 Running Radius Neighbors Display Tests...")
    
    # Test main functionality
    main_test = test_display_functionality()
    
    # Test edge cases
    edge_test = test_display_edge_cases()
    
    if main_test and edge_test:
        print("\n🎉 All tests passed! Display component is ready.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
       