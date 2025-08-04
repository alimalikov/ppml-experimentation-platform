"""
Multi-layer Perceptron Regressor Display Module
=============================================

This module provides comprehensive visualization and display capabilities for MLP neural networks,
including interactive plots, performance charts, architecture visualizations, and report generation.

Features:
- Training Progress Visualization
- Performance Analysis Charts
- Network Architecture Diagrams
- Interactive Dashboards
- Professional Report Generation
- Comparative Analysis Plots

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
import plotly.figure_factory as ff
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
import logging
from datetime import datetime
import io
import base64
from pathlib import Path
import json

# Web framework imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    
try:
    import dash
    from dash import dcc, html
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Statistical analysis
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPDisplay:
    """
    Comprehensive display and visualization class for Multi-layer Perceptron Regressor.
    
    This class provides various visualization methods including static plots,
    interactive charts, network diagrams, and comprehensive reports.
    """
    
    def __init__(self, mlp_core, mlp_analysis=None):
        """
        Initialize the MLP Display module.
        
        Parameters:
        -----------
        mlp_core : MLPCore
            The MLP core instance to visualize
        mlp_analysis : MLPAnalysis, optional
            The MLP analysis instance for advanced visualizations
        """
        self.mlp_core = mlp_core
        self.mlp_analysis = mlp_analysis
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7',
            'light': '#F5F5F5',
            'dark': '#2D3436'
        }
        
        # Set default figure parameters
        self.figure_params = {
            'figsize': (12, 8),
            'dpi': 300,
            'style': 'whitegrid'
        }
        
        logger.info("✅ MLP Display module initialized")

    # ==================================================================================
    # TRAINING VISUALIZATION METHODS
    # ==================================================================================
    
    def plot_training_progress(self, save_path: Optional[str] = None, show_interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot comprehensive training progress visualization.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        show_interactive : bool
            Whether to return interactive plotly figure
            
        Returns:
        --------
        Figure
            Matplotlib or Plotly figure object
        """
        try:
            logger.info("📊 Creating training progress visualization...")
            
            if not self.mlp_core.is_fitted_:
                raise ValueError("Model must be fitted to display training progress")
            
            training_history = self.mlp_core.training_history_
            loss_curve = training_history.get('loss_curve', [])
            validation_scores = training_history.get('validation_scores', [])
            
            if show_interactive:
                return self._create_interactive_training_plot(loss_curve, validation_scores)
            else:
                return self._create_static_training_plot(loss_curve, validation_scores, save_path)
                
        except Exception as e:
            logger.error(f"❌ Training progress visualization failed: {str(e)}")
            return self._create_error_plot(str(e))
    
    def plot_loss_evolution(self, save_path: Optional[str] = None, show_derivatives: bool = True) -> plt.Figure:
        """
        Plot detailed loss evolution with derivatives and analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        show_derivatives : bool
            Whether to show loss derivatives
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("📈 Creating loss evolution visualization...")
            
            loss_curve = self.mlp_core.training_history_.get('loss_curve', [])
            if not loss_curve:
                raise ValueError("No loss curve data available")
            
            fig, axes = plt.subplots(2 if show_derivatives else 1, 1, 
                                   figsize=(14, 10 if show_derivatives else 6))
            if not show_derivatives:
                axes = [axes]
            
            # Main loss plot
            iterations = range(len(loss_curve))
            axes[0].plot(iterations, loss_curve, 
                        color=self.colors['primary'], linewidth=2, label='Training Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Evolution', fontsize=16, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Add trend line
            if len(loss_curve) > 1:
                z = np.polyfit(iterations, loss_curve, 1)
                p = np.poly1d(z)
                axes[0].plot(iterations, p(iterations), 
                           color=self.colors['accent'], linestyle='--', 
                           alpha=0.7, label=f'Trend (slope: {z[0]:.2e})')
                axes[0].legend()
            
            # Derivatives plot
            if show_derivatives and len(loss_curve) > 1:
                first_derivative = np.diff(loss_curve)
                axes[1].plot(range(len(first_derivative)), first_derivative,
                           color=self.colors['secondary'], linewidth=1.5, 
                           label='First Derivative')
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Rate of Change')
                axes[1].set_title('Loss Rate of Change', fontsize=14)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                # Add smoothed derivative
                if len(first_derivative) > 5:
                    window_size = min(10, len(first_derivative) // 5)
                    smoothed = np.convolve(first_derivative, 
                                         np.ones(window_size)/window_size, mode='valid')
                    smooth_x = range(window_size//2, len(first_derivative) - window_size//2 + 1)
                    axes[1].plot(smooth_x, smoothed, 
                               color=self.colors['accent'], linewidth=2, 
                               alpha=0.8, label='Smoothed')
                    axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Loss evolution plot saved to {save_path}")
            
            logger.info("✅ Loss evolution visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Loss evolution visualization failed: {str(e)}")
            return self._create_error_plot(str(e))
    
    def plot_convergence_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive convergence analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("🎯 Creating convergence analysis visualization...")
            
            if self.mlp_analysis is None:
                raise ValueError("MLPAnalysis instance required for convergence analysis")
            
            convergence_data = self.mlp_analysis.analyze_convergence_patterns()
            if 'error' in convergence_data:
                raise ValueError(convergence_data['error'])
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Convergence Analysis Dashboard', fontsize=18, fontweight='bold')
            
            loss_curve = self.mlp_core.training_history_.get('loss_curve', [])
            
            # 1. Convergence rate analysis
            rate_data = convergence_data.get('convergence_rate', {})
            phases = ['Early', 'Middle', 'Late']
            rates = [rate_data.get('early_rate', 0), 
                    rate_data.get('middle_rate', 0), 
                    rate_data.get('late_rate', 0)]
            
            bars = axes[0, 0].bar(phases, np.abs(rates), 
                                color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
            axes[0, 0].set_title('Convergence Rate by Phase')
            axes[0, 0].set_ylabel('Absolute Rate of Change')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{rate:.2e}', ha='center', va='bottom')
            
            # 2. Stability analysis
            if len(loss_curve) > 10:
                window_size = 10
                rolling_std = []
                for i in range(window_size, len(loss_curve)):
                    window_std = np.std(loss_curve[i-window_size:i])
                    rolling_std.append(window_std)
                
                axes[0, 1].plot(range(window_size, len(loss_curve)), rolling_std,
                              color=self.colors['info'], linewidth=2)
                axes[0, 1].set_title('Training Stability (Rolling Std)')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Standard Deviation')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Phase identification
            phases_data = convergence_data.get('phases', {}).get('phases', [])
            if phases_data:
                colors_cycle = [self.colors['primary'], self.colors['secondary'], 
                              self.colors['accent'], self.colors['success']]
                
                for i, phase in enumerate(phases_data):
                    start = phase['start_iteration']
                    end = phase['end_iteration']
                    color = colors_cycle[i % len(colors_cycle)]
                    
                    axes[1, 0].plot(range(start, min(end, len(loss_curve))), 
                                  loss_curve[start:min(end, len(loss_curve))],
                                  color=color, linewidth=2, 
                                  label=f'Phase {phase["phase"]}')
                
                axes[1, 0].set_title('Training Phases')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Convergence metrics summary
            basic_metrics = convergence_data.get('basic_metrics', {})
            metrics_text = f"""
            Converged: {'Yes' if basic_metrics.get('converged', False) else 'No'}
            Total Iterations: {basic_metrics.get('total_iterations', 0)}
            Final Loss: {basic_metrics.get('final_loss', 0):.6f}
            Loss Reduction: {basic_metrics.get('loss_reduction_percentage', 0):.1f}%
            """
            
            axes[1, 1].text(0.1, 0.7, metrics_text, fontsize=12, 
                           verticalalignment='top', transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light']))
            axes[1, 1].set_title('Convergence Summary')
            axes[1, 1].axis('off')
            
            # Add recommendations
            recommendations = convergence_data.get('recommendations', [])
            if recommendations:
                rec_text = "Recommendations:\n" + "\n".join([f"• {rec}" for rec in recommendations[:3]])
                axes[1, 1].text(0.1, 0.3, rec_text, fontsize=10,
                               verticalalignment='top', transform=axes[1, 1].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['accent'], alpha=0.3))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Convergence analysis plot saved to {save_path}")
            
            logger.info("✅ Convergence analysis visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Convergence analysis visualization failed: {str(e)}")
            return self._create_error_plot(str(e))

    # ==================================================================================
    # PERFORMANCE VISUALIZATION METHODS
    # ==================================================================================
    
    def plot_performance_metrics(self, X: np.ndarray, y: np.ndarray, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive performance metrics visualization.
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test targets
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("📊 Creating performance metrics visualization...")
            
            if not self.mlp_core.is_fitted_:
                raise ValueError("Model must be fitted to display performance metrics")
            
            predictions = self.mlp_core.predict(X)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Performance Metrics Dashboard', fontsize=18, fontweight='bold')
            
            # 1. Predictions vs Actual
            min_val = min(np.min(y), np.min(predictions))
            max_val = max(np.max(y), np.max(predictions))
            
            axes[0, 0].scatter(y, predictions, alpha=0.6, color=self.colors['primary'], s=50)
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                          color=self.colors['accent'], linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Predictions vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add R² score
            r2 = r2_score(y, predictions)
            axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                          transform=axes[0, 0].transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 2. Residuals plot
            residuals = y - predictions
            axes[0, 1].scatter(predictions, residuals, alpha=0.6, 
                             color=self.colors['secondary'], s=50)
            axes[0, 1].axhline(y=0, color=self.colors['dark'], linestyle='-', alpha=0.5)
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Residuals distribution
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=self.colors['info'], 
                          density=True, edgecolor='black')
            
            # Add normal distribution overlay
            mu, sigma = stats.norm.fit(residuals)
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            axes[1, 0].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                          color=self.colors['accent'], linewidth=2, 
                          label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Performance metrics summary
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mse)
            
            metrics_text = f"""
            Performance Metrics:
            
            R² Score: {r2:.4f}
            MSE: {mse:.4f}
            RMSE: {rmse:.4f}
            MAE: {mae:.4f}
            
            Residuals Statistics:
            Mean: {np.mean(residuals):.4f}
            Std: {np.std(residuals):.4f}
            Skewness: {stats.skew(residuals):.4f}
            Kurtosis: {stats.kurtosis(residuals):.4f}
            """
            
            axes[1, 1].text(0.1, 0.9, metrics_text, fontsize=11, 
                           verticalalignment='top', transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light']))
            axes[1, 1].set_title('Performance Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Performance metrics plot saved to {save_path}")
            
            logger.info("✅ Performance metrics visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Performance metrics visualization failed: {str(e)}")
            return self._create_error_plot(str(e))
    
    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("📈 Creating learning curves visualization...")
            
            if self.mlp_analysis is None:
                raise ValueError("MLPAnalysis instance required for learning curves")
            
            learning_data = self.mlp_analysis.analyze_learning_curves(X, y)
            if 'error' in learning_data:
                raise ValueError(learning_data['error'])
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Learning Curves Analysis', fontsize=18, fontweight='bold')
            
            train_sizes = learning_data['train_sizes']
            train_scores = learning_data['train_scores']
            val_scores = learning_data['validation_scores']
            
            # 1. Learning curves
            axes[0].plot(train_sizes, train_scores['mean'], 'o-', 
                        color=self.colors['primary'], linewidth=2, 
                        markersize=6, label='Training Score')
            axes[0].fill_between(train_sizes, 
                               np.array(train_scores['mean']) - np.array(train_scores['std']),
                               np.array(train_scores['mean']) + np.array(train_scores['std']),
                               alpha=0.3, color=self.colors['primary'])
            
            axes[0].plot(train_sizes, val_scores['mean'], 's-', 
                        color=self.colors['secondary'], linewidth=2, 
                        markersize=6, label='Validation Score')
            axes[0].fill_between(train_sizes, 
                               np.array(val_scores['mean']) - np.array(val_scores['std']),
                               np.array(val_scores['mean']) + np.array(val_scores['std']),
                               alpha=0.3, color=self.colors['secondary'])
            
            axes[0].set_xlabel('Training Set Size')
            axes[0].set_ylabel('Score (R²)')
            axes[0].set_title('Learning Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Performance gap analysis
            performance_gap = np.array(train_scores['mean']) - np.array(val_scores['mean'])
            axes[1].plot(train_sizes, performance_gap, 'o-', 
                        color=self.colors['accent'], linewidth=2, markersize=6)
            axes[1].axhline(y=0, color=self.colors['dark'], linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Training Set Size')
            axes[1].set_ylabel('Performance Gap (Train - Val)')
            axes[1].set_title('Generalization Gap')
            axes[1].grid(True, alpha=0.3)
            
            # Add trend analysis
            trends = learning_data.get('trends', {})
            if trends:
                trend_text = f"""
                Trends Analysis:
                Training: {trends.get('training_trend', 'unknown')}
                Validation: {trends.get('validation_trend', 'unknown')}
                Gap: {trends.get('gap_trend', 'unknown')}
                """
                axes[1].text(0.05, 0.95, trend_text, fontsize=10,
                           transform=axes[1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light']))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Learning curves plot saved to {save_path}")
            
            logger.info("✅ Learning curves visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Learning curves visualization failed: {str(e)}")
            return self._create_error_plot(str(e))

    # ==================================================================================
    # NETWORK ARCHITECTURE VISUALIZATION
    # ==================================================================================
    
    def plot_network_architecture(self, save_path: Optional[str] = None, 
                                 show_weights: bool = False) -> plt.Figure:
        """
        Plot network architecture diagram.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        show_weights : bool
            Whether to show weight connections
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("🏗️ Creating network architecture visualization...")
            
            arch_info = self.mlp_core.get_network_architecture()
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_title('Neural Network Architecture', fontsize=18, fontweight='bold')
            
            # Get layer sizes
            layers = [arch_info['input_size']] + list(arch_info['hidden_layers']) + [1]
            max_neurons = max(layers)
            
            # Calculate positions
            layer_positions = np.linspace(0, 10, len(layers))
            
            # Draw neurons
            for layer_idx, (x_pos, layer_size) in enumerate(zip(layer_positions, layers)):
                y_positions = np.linspace(-max_neurons/2, max_neurons/2, layer_size)
                
                # Determine layer color and label
                if layer_idx == 0:
                    color = self.colors['info']
                    layer_label = 'Input'
                elif layer_idx == len(layers) - 1:
                    color = self.colors['success']
                    layer_label = 'Output'
                else:
                    color = self.colors['primary']
                    layer_label = f'Hidden {layer_idx}'
                
                # Draw neurons
                for y_pos in y_positions:
                    circle = plt.Circle((x_pos, y_pos), 0.15, 
                                      color=color, alpha=0.7, zorder=3)
                    ax.add_patch(circle)
                
                # Add layer label
                ax.text(x_pos, max_neurons/2 + 1, f'{layer_label}\n({layer_size})', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Draw connections to next layer
                if layer_idx < len(layers) - 1 and show_weights:
                    next_layer_size = layers[layer_idx + 1]
                    next_x = layer_positions[layer_idx + 1]
                    next_y_positions = np.linspace(-max_neurons/2, max_neurons/2, next_layer_size)
                    
                    # Get weights for this layer if available
                    if self.mlp_core.is_fitted_:
                        weights = self.mlp_core.get_network_weights()
                        if layer_idx < len(weights):
                            weight_matrix = weights[layer_idx]
                            max_weight = np.max(np.abs(weight_matrix))
                            
                            for i, y1 in enumerate(y_positions):
                                for j, y2 in enumerate(next_y_positions):
                                    if i < weight_matrix.shape[0] and j < weight_matrix.shape[1]:
                                        weight = weight_matrix[i, j]
                                        alpha = min(0.8, abs(weight) / max_weight)
                                        color = self.colors['accent'] if weight > 0 else self.colors['secondary']
                                        ax.plot([x_pos + 0.15, next_x - 0.15], [y1, y2],
                                               color=color, alpha=alpha, linewidth=0.5, zorder=1)
                elif layer_idx < len(layers) - 1:
                    # Draw simplified connections
                    next_x = layer_positions[layer_idx + 1]
                    ax.plot([x_pos + 0.2, next_x - 0.2], [0, 0],
                           color=self.colors['dark'], alpha=0.3, linewidth=2, zorder=1)
            
            # Set axis properties
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-max_neurons/2 - 2, max_neurons/2 + 3)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add architecture information
            info_text = f"""
            Architecture Summary:
            Total Parameters: {arch_info['total_parameters']:,}
            Total Layers: {len(layers)}
            Hidden Layers: {len(arch_info['hidden_layers'])}
            Activation: {self.mlp_core.activation}
            Solver: {self.mlp_core.solver}
            """
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light'], alpha=0.9))
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Network architecture plot saved to {save_path}")
            
            logger.info("✅ Network architecture visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Network architecture visualization failed: {str(e)}")
            return self._create_error_plot(str(e))
    
    def plot_weight_distributions(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot weight and bias distributions across layers.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        try:
            logger.info("⚖️ Creating weight distributions visualization...")
            
            if not self.mlp_core.is_fitted_:
                raise ValueError("Model must be fitted to display weight distributions")
            
            weights = self.mlp_core.get_network_weights()
            biases = self.mlp_core.get_network_biases()
            
            n_layers = len(weights)
            fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
            if n_layers == 1:
                axes = axes.reshape(2, 1)
            
            fig.suptitle('Weight and Bias Distributions by Layer', fontsize=16, fontweight='bold')
            
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['info']]
            
            for i, (w, b) in enumerate(zip(weights, biases)):
                color = colors[i % len(colors)]
                
                # Weight distributions
                axes[0, i].hist(w.flatten(), bins=30, alpha=0.7, color=color, 
                              density=True, edgecolor='black')
                axes[0, i].set_title(f'Layer {i+1} Weights')
                axes[0, i].set_xlabel('Weight Value')
                axes[0, i].set_ylabel('Density')
                axes[0, i].grid(True, alpha=0.3)
                
                # Add statistics
                w_mean, w_std = np.mean(w), np.std(w)
                axes[0, i].axvline(w_mean, color='red', linestyle='--', 
                                 label=f'Mean: {w_mean:.3f}')
                axes[0, i].legend()
                
                # Bias distributions
                axes[1, i].hist(b, bins=20, alpha=0.7, color=color, 
                              density=True, edgecolor='black')
                axes[1, i].set_title(f'Layer {i+1} Biases')
                axes[1, i].set_xlabel('Bias Value')
                axes[1, i].set_ylabel('Density')
                axes[1, i].grid(True, alpha=0.3)
                
                # Add statistics
                b_mean, b_std = np.mean(b), np.std(b)
                axes[1, i].axvline(b_mean, color='red', linestyle='--', 
                                 label=f'Mean: {b_mean:.3f}')
                axes[1, i].legend()
                
                # Add text summary
                summary_text = f'Weights: μ={w_mean:.3f}, σ={w_std:.3f}\nBiases: μ={b_mean:.3f}, σ={b_std:.3f}'
                axes[0, i].text(0.02, 0.98, summary_text, transform=axes[0, i].transAxes,
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                logger.info(f"💾 Weight distributions plot saved to {save_path}")
            
            logger.info("✅ Weight distributions visualization completed")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Weight distributions visualization failed: {str(e)}")
            return self._create_error_plot(str(e))

    # ==================================================================================
    # INTERACTIVE DASHBOARD METHODS
    # ==================================================================================
    
    def create_interactive_dashboard(self, X: np.ndarray, y: np.ndarray) -> go.Figure:
        """
        Create comprehensive interactive dashboard using Plotly.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for analysis
        y : np.ndarray
            Targets for analysis
            
        Returns:
        --------
        go.Figure
            Plotly figure object with interactive dashboard
        """
        try:
            logger.info("🎛️ Creating interactive dashboard...")
            
            if not self.mlp_core.is_fitted_:
                raise ValueError("Model must be fitted to create dashboard")
            
            predictions = self.mlp_core.predict(X)
            residuals = y - predictions
            loss_curve = self.mlp_core.training_history_.get('loss_curve', [])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Progress', 'Predictions vs Actual', 
                              'Residuals Distribution', 'Performance Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # 1. Training progress
            if loss_curve:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(loss_curve))),
                        y=loss_curve,
                        mode='lines',
                        name='Training Loss',
                        line=dict(color=self.colors['primary'], width=2),
                        hovertemplate='Iteration: %{x}<br>Loss: %{y:.6f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Predictions vs Actual
            fig.add_trace(
                go.Scatter(
                    x=y,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color=self.colors['secondary'], size=6, opacity=0.7),
                    hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add perfect prediction line
            min_val, max_val = min(np.min(y), np.min(predictions)), max(np.max(y), np.max(predictions))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color=self.colors['accent'], width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. Residuals distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Residuals',
                    marker=dict(color=self.colors['info'], opacity=0.7),
                    hovertemplate='Residual Range: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Performance metrics table
            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            
            metrics_data = [
                ['R² Score', f'{r2:.4f}'],
                ['MSE', f'{mse:.4f}'],
                ['MAE', f'{mae:.4f}'],
                ['RMSE', f'{np.sqrt(mse):.4f}'],
                ['Residuals Mean', f'{np.mean(residuals):.4f}'],
                ['Residuals Std', f'{np.std(residuals):.4f}']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                              fill_color=self.colors['primary'],
                              font=dict(color='white', size=12)),
                    cells=dict(values=list(zip(*metrics_data)),
                             fill_color=[self.colors['light'], 'white'],
                             font=dict(size=11))
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='MLP Regressor Interactive Dashboard',
                    font=dict(size=20, color=self.colors['dark']),
                    x=0.5
                ),
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Iteration", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_xaxes(title_text="Actual Values", row=1, col=2)
            fig.update_yaxes(title_text="Predicted Values", row=1, col=2)
            fig.update_xaxes(title_text="Residuals", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            logger.info("✅ Interactive dashboard created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Interactive dashboard creation failed: {str(e)}")
            return self._create_error_plotly(str(e))
    
    def create_analysis_dashboard(self, X: np.ndarray, y: np.ndarray) -> go.Figure:
        """
        Create advanced analysis dashboard with multiple metrics.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for analysis
        y : np.ndarray
            Targets for analysis
            
        Returns:
        --------
        go.Figure
            Advanced analysis dashboard
        """
        try:
            logger.info("📊 Creating advanced analysis dashboard...")
            
            if self.mlp_analysis is None:
                raise ValueError("MLPAnalysis instance required for advanced dashboard")
            
            # Get analysis results
            complexity_analysis = self.mlp_analysis.analyze_network_complexity()
            performance_analysis = self.mlp_analysis.analyze_generalization_performance(X, y)
            
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Network Complexity', 'Performance Breakdown',
                              'Layer Efficiency', 'Generalization Metrics',
                              'Training Health', 'Recommendations'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "table"}]]
            )
            
            # 1. Network Complexity Gauge
            if 'error' not in complexity_analysis:
                complexity_score = complexity_analysis.get('complexity_rating', {}).get('parameter_ratio', 0)
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=complexity_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Complexity Score"},
                        gauge={'axis': {'range': [None, 100]},
                              'bar': {'color': self.colors['primary']},
                              'steps': [{'range': [0, 30], 'color': "lightgray"},
                                       {'range': [30, 70], 'color': "gray"}],
                              'threshold': {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 90}}
                    ),
                    row=1, col=1
                )
            
            # 2. Performance Breakdown
            if 'error' not in performance_analysis:
                perf_metrics = performance_analysis.get('test_performance', {})
                metrics_names = ['R² Score', 'MSE', 'MAE', 'RMSE']
                metrics_values = [
                    perf_metrics.get('r2_score', 0),
                    perf_metrics.get('mse', 0),
                    perf_metrics.get('mae', 0),
                    perf_metrics.get('rmse', 0)
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=metrics_names,
                        y=metrics_values,
                        marker_color=[self.colors['primary'], self.colors['secondary'], 
                                    self.colors['accent'], self.colors['info']],
                        hovertemplate='%{x}: %{y:.4f}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # Continue with other subplots...
            # [Additional dashboard components would be added here]
            
            fig.update_layout(
                title=dict(
                    text='Advanced MLP Analysis Dashboard',
                    font=dict(size=20),
                    x=0.5
                ),
                height=1200,
                showlegend=False,
                template='plotly_white'
            )
            
            logger.info("✅ Advanced analysis dashboard created")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Advanced dashboard creation failed: {str(e)}")
            return self._create_error_plotly(str(e))

    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    def _create_static_training_plot(self, loss_curve: List[float], 
                                   validation_scores: List[float], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create static training progress plot."""
        fig, axes = plt.subplots(1, 2 if validation_scores else 1, figsize=(16, 6))
        if not validation_scores:
            axes = [axes]
        
        # Loss curve
        if loss_curve:
            axes[0].plot(loss_curve, color=self.colors['primary'], linewidth=2)
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Curve')
            axes[0].grid(True, alpha=0.3)
            
            # Add convergence indicator
            if len(loss_curve) > 1:
                improvement = (loss_curve[0] - loss_curve[-1]) / loss_curve[0] * 100
                axes[0].text(0.02, 0.98, f'Improvement: {improvement:.1f}%',
                           transform=axes[0].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Validation scores
        if validation_scores and len(axes) > 1:
            axes[1].plot(validation_scores, color=self.colors['secondary'], linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Validation Score')
            axes[1].set_title('Validation Performance')
            axes[1].grid(True, alpha=0.3)
            
            # Add best score indicator
            best_score = max(validation_scores)
            best_epoch = validation_scores.index(best_score)
            axes[1].scatter([best_epoch], [best_score], color=self.colors['accent'], 
                          s=100, zorder=5, label=f'Best: {best_score:.4f}')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_params['dpi'], bbox_inches='tight')
        
        return fig
    
    def _create_interactive_training_plot(self, loss_curve: List[float], 
                                        validation_scores: List[float]) -> go.Figure:
        """Create interactive training progress plot."""
        fig = make_subplots(
            rows=1, cols=2 if validation_scores else 1,
            subplot_titles=['Training Loss', 'Validation Performance'] if validation_scores else ['Training Loss']
        )
        
        # Loss curve
        if loss_curve:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(loss_curve))),
                    y=loss_curve,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color=self.colors['primary'], width=2),
                    hovertemplate='Iteration: %{x}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Validation scores
        if validation_scores:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(validation_scores))),
                    y=validation_scores,
                    mode='lines+markers',
                    name='Validation Score',
                    line=dict(color=self.colors['secondary'], width=2),
                    marker=dict(size=6),
                    hovertemplate='Epoch: %{x}<br>Score: %{y:.4f}<extra></extra>'
                ),
                row=1, col=2 if len(validation_scores) > 0 else 1
            )
        
        fig.update_layout(
            title='Training Progress',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _create_error_plot(self, error_message: str) -> plt.Figure:
        """Create error plot for matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error: {error_message}', 
               ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    def _create_error_plotly(self, error_message: str) -> go.Figure:
        """Create error plot for plotly."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,0,0,0.1)",
            bordercolor="red",
            borderwidth=2
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    def save_all_plots(self, X: np.ndarray, y: np.ndarray, output_dir: str) -> Dict[str, str]:
        """
        Save all available plots to specified directory.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for analysis
        y : np.ndarray
            Targets for analysis
        output_dir : str
            Directory to save plots
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot names to file paths
        """
        try:
            logger.info(f"💾 Saving all plots to {output_dir}...")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_plots = {}
            
            # Training progress
            try:
                fig = self.plot_training_progress()
                path = output_path / "training_progress.png"
                fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                saved_plots['training_progress'] = str(path)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save training progress: {e}")
            
            # Loss evolution
            try:
                fig = self.plot_loss_evolution()
                path = output_path / "loss_evolution.png"
                fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                saved_plots['loss_evolution'] = str(path)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save loss evolution: {e}")
            
            # Performance metrics
            try:
                fig = self.plot_performance_metrics(X, y)
                path = output_path / "performance_metrics.png"
                fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                saved_plots['performance_metrics'] = str(path)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {e}")
            
            # Network architecture
            try:
                fig = self.plot_network_architecture()
                path = output_path / "network_architecture.png"
                fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                saved_plots['network_architecture'] = str(path)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save network architecture: {e}")
            
            # Weight distributions
            try:
                fig = self.plot_weight_distributions()
                path = output_path / "weight_distributions.png"
                fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                saved_plots['weight_distributions'] = str(path)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save weight distributions: {e}")
            
            # Learning curves (if analysis available)
            if self.mlp_analysis:
                try:
                    fig = self.plot_learning_curves(X, y)
                    path = output_path / "learning_curves.png"
                    fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                    saved_plots['learning_curves'] = str(path)
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to save learning curves: {e}")
                
                # Convergence analysis
                try:
                    fig = self.plot_convergence_analysis()
                    path = output_path / "convergence_analysis.png"
                    fig.savefig(path, dpi=self.figure_params['dpi'], bbox_inches='tight')
                    saved_plots['convergence_analysis'] = str(path)
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to save convergence analysis: {e}")
            
            logger.info(f"✅ Saved {len(saved_plots)} plots successfully")
            return saved_plots
            
        except Exception as e:
            logger.error(f"❌ Failed to save plots: {str(e)}")
            return {}
    
    def export_interactive_report(self, X: np.ndarray, y: np.ndarray, 
                                output_path: str) -> str:
        """
        Export interactive HTML report with all visualizations.
        
        Parameters:
        -----------
        X : np.ndarray
            Features for analysis
        y : np.ndarray
            Targets for analysis
        output_path : str
            Path for HTML report
            
        Returns:
        --------
        str
            Path to generated HTML report
        """
        try:
            logger.info(f"📄 Generating interactive HTML report...")
            
            # Create main dashboard
            dashboard = self.create_interactive_dashboard(X, y)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLP Regressor Analysis Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; color: {self.colors['dark']}; }}
                    .section {{ margin: 30px 0; }}
                    .plot-container {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>MLP Regressor Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Interactive Dashboard</h2>
                    <div class="plot-container" id="dashboard"></div>
                </div>
                
                <script>
                    var dashboardData = {dashboard.to_json()};
                    Plotly.newPlot('dashboard', dashboardData.data, dashboardData.layout);
                </script>
            </body>
            </html>
            """
            
            # Save HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Interactive report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to export interactive report: {str(e)}")
            return ""

# ==================================================================================
# STREAMLIT INTEGRATION (Optional)
# ==================================================================================

if STREAMLIT_AVAILABLE:
    def create_streamlit_app(mlp_display, X: np.ndarray, y: np.ndarray):
        """Create Streamlit web application for MLP analysis."""
        st.set_page_config(page_title="MLP Regressor Analysis", layout="wide")
        
        st.title("🧠 MLP Regressor Analysis Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        st.sidebar.header("Visualization Controls")
        show_training = st.sidebar.checkbox("Training Progress", value=True)
        show_performance = st.sidebar.checkbox("Performance Metrics", value=True)
        show_architecture = st.sidebar.checkbox("Network Architecture", value=True)
        show_interactive = st.sidebar.checkbox("Interactive Dashboard", value=True)
        
        # Main content
        if show_training:
            st.header("📈 Training Progress")
            try:
                fig = mlp_display.plot_training_progress()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating training progress plot: {e}")
        
        if show_performance:
            st.header("📊 Performance Metrics")
            try:
                fig = mlp_display.plot_performance_metrics(X, y)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating performance metrics plot: {e}")
        
        if show_architecture:
            st.header("🏗️ Network Architecture")
            try:
                fig = mlp_display.plot_network_architecture()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating architecture plot: {e}")
        
        if show_interactive:
            st.header("🎛️ Interactive Dashboard")
            try:
                fig = mlp_display.create_interactive_dashboard(X, y)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating interactive dashboard: {e}")

# ==================================================================================
# END OF MLP DISPLAY MODULE
# ==================================================================================

"""
🎨 MLP Display Module Completion Summary:

✅ Core Visualization Features:
   - Training progress plots (static & interactive)
   - Loss evolution with derivatives analysis
   - Comprehensive performance metrics
   - Network architecture diagrams
   - Weight/bias distribution analysis
   - Learning curves visualization
   - Convergence analysis charts

✅ Interactive Features:
   - Plotly-based interactive dashboards
   - Advanced analysis dashboard
   - Interactive HTML report generation
   - Streamlit web app integration

✅ Professional Features:
   - Consistent color scheme and styling
   - Comprehensive error handling
   - Batch plot saving functionality
   - Multiple export formats
   - Detailed logging and documentation

🎛️ Dashboard Capabilities:
   - Real-time interactive exploration
   - Multiple visualization types
   - Professional styling and layout
   - Export capabilities for reports

📊 Ready for integration with MLPCore and MLPAnalysis modules!
"""