import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings

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
    print(f"Import error in roc_curve_plugin: {e}")
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

class ROCCurvePlugin(BaseVisualizationPlugin):
    """
    Advanced ROC Curve Analysis Plugin.
    
    Creates comprehensive ROC curve visualizations for binary and multiclass
    classification models with detailed AUC analysis and comparison features.
    """
    
    def __init__(self):
        """Initialize the ROC Curve Plugin"""
        super().__init__()
        
        # Plugin metadata
        self.name = "ROC Curve Analysis"
        self.description = "Comprehensive ROC curves and AUC analysis for classification models"
        self.version = "1.3.0"
        self.author = "ML App Team"
        self.category = VisualizationCategory.PERFORMANCE
        
        # Supported configurations
        self.supported_data_types = [
            DataType.BINARY,
            DataType.CLASSIFICATION,
            DataType.MULTICLASS
        ]
        
        # Requirements
        self.min_samples = 10
        self.requires_trained_model = True
        self.requires_predictions = False
        self.requires_probabilities = True
        self.interactive = True
        self.export_formats = ["png", "pdf", "html", "svg"]
        
        # ROC curve color palettes
        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
            "professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592A3C", "#1F2041"],
            "colorblind": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]
        }
        
        # Analysis options
        self.curve_styles = {
            "smooth": "Smooth curves",
            "stepped": "Stepped curves", 
            "markers": "Curves with markers"
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
            
            # Need at least 1 model result for ROC curves
            if not model_results:
                return False
            
            # Check if model results have probabilities
            valid_results = 0
            for result in model_results:
                if not isinstance(result, dict) or "error" in result:
                    continue
                
                # Check for probabilities or prediction scores
                has_probabilities = (
                    'probabilities' in result or 
                    'prediction_probabilities' in result or
                    'predict_proba' in result or
                    'decision_function' in result
                )
                
                # Also check for y_true and y_pred for ROC calculation
                has_predictions = (
                    'y_true' in result and 'y_pred' in result
                ) or (
                    'actual' in result and 'predicted' in result
                )
                
                if has_probabilities or has_predictions:
                    valid_results += 1
            
            return valid_results >= 1
            
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
            'curve_style': {
                'type': 'select',
                'label': '📈 Curve Style',
                'default': 'smooth',
                'options': list(self.curve_styles.keys()),
                'help': 'Choose the style for ROC curves'
            },
            'color_palette': {
                'type': 'select',
                'label': '🎨 Color Palette',
                'default': 'default',
                'options': list(self.color_palettes.keys()),
                'help': 'Select color scheme for multiple curves'
            },
            'show_diagonal': {
                'type': 'checkbox',
                'label': '📐 Show Random Classifier Line',
                'default': True,
                'help': 'Display the diagonal line representing random classification'
            },
            'show_auc_values': {
                'type': 'checkbox',
                'label': '📊 Show AUC Values',
                'default': True,
                'help': 'Display AUC values in the legend'
            },
            'show_confidence_interval': {
                'type': 'checkbox',
                'label': '📊 Show Confidence Intervals',
                'default': False,
                'help': 'Display confidence intervals for ROC curves (if bootstrap data available)'
            },
            'compare_models': {
                'type': 'checkbox',
                'label': '⚖️ Compare Multiple Models',
                'default': True,
                'help': 'Show all models on the same plot for comparison'
            },
            'show_optimal_threshold': {
                'type': 'checkbox',
                'label': '🎯 Show Optimal Threshold',
                'default': True,
                'help': 'Mark the optimal threshold point on ROC curves'
            },
            'include_pr_curve': {
                'type': 'checkbox',
                'label': '📈 Include Precision-Recall Curve',
                'default': False,
                'help': 'Also display Precision-Recall curves alongside ROC'
            },
            'chart_height': {
                'type': 'slider',
                'label': '📏 Chart Height',
                'default': 600,
                'min': 400,
                'max': 1000,
                'help': 'Adjust the height of the chart'
            },
            'line_width': {
                'type': 'slider',
                'label': '📏 Line Width',
                'default': 3,
                'min': 1,
                'max': 6,
                'help': 'Thickness of the ROC curves'
            }
        }
    
    def render(self, data: pd.DataFrame, model_results: List[Dict[str, Any]], 
               config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Render the ROC curve visualization.
        
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
            curve_style = config.get('curve_style', 'smooth')
            color_palette = config.get('color_palette', 'default')
            show_diagonal = config.get('show_diagonal', True)
            show_auc_values = config.get('show_auc_values', True)
            show_confidence_interval = config.get('show_confidence_interval', False)
            compare_models = config.get('compare_models', True)
            show_optimal_threshold = config.get('show_optimal_threshold', True)
            include_pr_curve = config.get('include_pr_curve', False)
            chart_height = config.get('chart_height', 600)
            line_width = config.get('line_width', 3)
            
            # Process model results to extract ROC data
            roc_data = self._extract_roc_data(model_results)
            
            if not roc_data:
                st.warning("⚠️ No valid data found for ROC curve analysis")
                return False
            
            # Create visualization based on configuration
            if include_pr_curve:
                success = self._create_combined_curves(
                    roc_data, curve_style, color_palette, show_diagonal,
                    show_auc_values, show_optimal_threshold, chart_height, line_width
                )
            else:
                success = self._create_roc_curves(
                    roc_data, curve_style, color_palette, show_diagonal,
                    show_auc_values, show_optimal_threshold, compare_models,
                    chart_height, line_width
                )
            
            if success:
                # Display detailed analysis
                self._display_auc_analysis(roc_data)
                
                # Display optimal thresholds
                if show_optimal_threshold:
                    self._display_optimal_thresholds(roc_data)
                
                # Display model comparison insights
                if len(roc_data) > 1:
                    self._display_comparison_insights(roc_data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rendering ROC curves: {str(e)}")
            st.error(f"❌ Error creating ROC curves: {str(e)}")
            return False
    
    def _extract_roc_data(self, model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract ROC curve data from model results.
        
        Args:
            model_results: List of model result dictionaries
            
        Returns:
            List of dictionaries containing ROC curve data
        """
        roc_data = []
        
        for idx, result in enumerate(model_results):
            if "error" in result:
                continue
            
            try:
                # Create model identifier
                model_name = result.get('model_name', f'Model {idx + 1}')
                run_number = sum(1 for i, r in enumerate(model_results[:idx+1]) 
                               if r.get('model_name') == result.get('model_name') 
                               and "error" not in r)
                model_id = f"{model_name} - Run #{run_number}"
                
                # Extract true labels and predictions
                y_true, y_scores = self._extract_labels_and_scores(result)
                
                if y_true is None or y_scores is None:
                    continue
                
                # Handle multiclass case
                if len(np.unique(y_true)) > 2:
                    roc_curves = self._compute_multiclass_roc(y_true, y_scores, model_id)
                    roc_data.extend(roc_curves)
                else:
                    # Binary classification
                    roc_curve_data = self._compute_binary_roc(y_true, y_scores, model_id)
                    if roc_curve_data:
                        roc_data.append(roc_curve_data)
                
            except Exception as e:
                self.logger.warning(f"Error processing model {idx}: {str(e)}")
                continue
        
        return roc_data
    
    def _extract_labels_and_scores(self, result: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract true labels and prediction scores from model result.
        
        Args:
            result: Model result dictionary
            
        Returns:
            Tuple of (y_true, y_scores) or (None, None) if extraction fails
        """
        y_true = None
        y_scores = None
        
        # Try different keys for true labels
        for key in ['y_true', 'actual', 'true_labels']:
            if key in result:
                y_true = np.array(result[key])
                break
        
        # Try different keys for prediction scores
        for key in ['probabilities', 'prediction_probabilities', 'predict_proba', 'y_scores']:
            if key in result:
                scores = result[key]
                if isinstance(scores, (list, np.ndarray)):
                    y_scores = np.array(scores)
                    # For binary classification, take positive class probabilities
                    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                        y_scores = y_scores[:, 1]
                    break
        
        # If no probabilities found, try decision function or predicted values
        if y_scores is None:
            for key in ['decision_function', 'y_pred', 'predicted']:
                if key in result:
                    y_scores = np.array(result[key])
                    break
        
        # Validate extracted data
        if y_true is not None and y_scores is not None:
            if len(y_true) != len(y_scores):
                self.logger.warning("Mismatch in lengths of y_true and y_scores")
                return None, None
            
            return y_true, y_scores
        
        return None, None
    
    def _compute_binary_roc(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Compute ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            model_name: Name of the model
            
        Returns:
            Dictionary containing ROC curve data
        """
        try:
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold (Youden's index)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            optimal_point = (fpr[optimal_idx], tpr[optimal_idx])
            
            # Compute precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            return {
                'model_name': model_name,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'optimal_threshold': optimal_threshold,
                'optimal_point': optimal_point,
                'precision': precision,
                'recall': recall,
                'pr_thresholds': pr_thresholds,
                'pr_auc': pr_auc,
                'is_multiclass': False,
                'class_name': 'Binary'
            }
            
        except Exception as e:
            self.logger.error(f"Error computing binary ROC for {model_name}: {str(e)}")
            return None
    
    def _compute_multiclass_roc(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str) -> List[Dict[str, Any]]:
        """
        Compute ROC curves for multiclass classification (one-vs-rest).
        
        Args:
            y_true: True class labels
            y_scores: Prediction scores for each class
            model_name: Name of the model
            
        Returns:
            List of dictionaries containing ROC curve data for each class
        """
        roc_curves = []
        
        try:
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=classes)
            if n_classes == 2:
                y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
            
            # Ensure y_scores has the right shape
            if y_scores.ndim == 1:
                # For binary case, create complementary probabilities
                y_scores = np.column_stack([1 - y_scores, y_scores])
            
            # Compute ROC curve for each class
            for i, class_label in enumerate(classes):
                try:
                    if i < y_scores.shape[1]:
                        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_scores[:, i])
                        roc_auc = auc(fpr, tpr)
                        
                        # Find optimal threshold
                        optimal_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[optimal_idx]
                        optimal_point = (fpr[optimal_idx], tpr[optimal_idx])
                        
                        # Compute precision-recall curve
                        precision, recall, pr_thresholds = precision_recall_curve(
                            y_true_bin[:, i], y_scores[:, i]
                        )
                        pr_auc = auc(recall, precision)
                        
                        roc_curves.append({
                            'model_name': f"{model_name} (Class {class_label})",
                            'fpr': fpr,
                            'tpr': tpr,
                            'thresholds': thresholds,
                            'auc': roc_auc,
                            'optimal_threshold': optimal_threshold,
                            'optimal_point': optimal_point,
                            'precision': precision,
                            'recall': recall,
                            'pr_thresholds': pr_thresholds,
                            'pr_auc': pr_auc,
                            'is_multiclass': True,
                            'class_name': str(class_label)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error computing ROC for class {class_label}: {str(e)}")
                    continue
            
            # Compute micro-average ROC curve
            if len(roc_curves) > 1:
                try:
                    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
                    auc_micro = auc(fpr_micro, tpr_micro)
                    
                    roc_curves.append({
                        'model_name': f"{model_name} (Micro-average)",
                        'fpr': fpr_micro,
                        'tpr': tpr_micro,
                        'thresholds': None,
                        'auc': auc_micro,
                        'optimal_threshold': None,
                        'optimal_point': None,
                        'precision': None,
                        'recall': None,
                        'pr_thresholds': None,
                        'pr_auc': None,
                        'is_multiclass': True,
                        'class_name': 'Micro-average'
                    })
                except Exception as e:
                    self.logger.warning(f"Error computing micro-average ROC: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error computing multiclass ROC for {model_name}: {str(e)}")
        
        return roc_curves
    
    def _create_roc_curves(self, roc_data: List[Dict[str, Any]], curve_style: str,
                          color_palette: str, show_diagonal: bool, show_auc_values: bool,
                          show_optimal_threshold: bool, compare_models: bool,
                          chart_height: int, line_width: int) -> bool:
        """
        Create ROC curve visualization.
        
        Args:
            roc_data: List of ROC curve data dictionaries
            curve_style: Style of the curves
            color_palette: Color palette to use
            show_diagonal: Whether to show diagonal line
            show_auc_values: Whether to show AUC values
            show_optimal_threshold: Whether to mark optimal thresholds
            compare_models: Whether to show all models on same plot
            chart_height: Height of the chart
            line_width: Width of the curves
            
        Returns:
            bool: True if successful
        """
        try:
            colors = self.color_palettes[color_palette]
            
            # Create main ROC curve plot
            fig = go.Figure()
            
            # Add ROC curves
            for i, curve_data in enumerate(roc_data):
                color = colors[i % len(colors)]
                
                # Prepare curve name
                curve_name = curve_data['model_name']
                if show_auc_values:
                    curve_name += f" (AUC = {curve_data['auc']:.3f})"
                
                # Add ROC curve
                if curve_style == 'stepped':
                    line_shape = 'hv'
                elif curve_style == 'markers':
                    mode = 'lines+markers'
                    line_shape = 'linear'
                else:
                    mode = 'lines'
                    line_shape = 'linear'
                
                fig.add_trace(go.Scatter(
                    x=curve_data['fpr'],
                    y=curve_data['tpr'],
                    mode=mode if curve_style == 'markers' else 'lines',
                    line=dict(color=color, width=line_width, shape=line_shape),
                    marker=dict(size=6) if curve_style == 'markers' else None,
                    name=curve_name,
                    hovertemplate=f"<b>{curve_data['model_name']}</b><br>" +
                                f"False Positive Rate: %{{x:.3f}}<br>" +
                                f"True Positive Rate: %{{y:.3f}}<br>" +
                                f"AUC: {curve_data['auc']:.3f}<extra></extra>"
                ))
                
                # Add optimal threshold point
                if show_optimal_threshold and curve_data['optimal_point']:
                    fig.add_trace(go.Scatter(
                        x=[curve_data['optimal_point'][0]],
                        y=[curve_data['optimal_point'][1]],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=12,
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        name=f"Optimal Threshold ({curve_data['optimal_threshold']:.3f})",
                        hovertemplate=f"<b>Optimal Point</b><br>" +
                                    f"FPR: %{{x:.3f}}<br>" +
                                    f"TPR: %{{y:.3f}}<br>" +
                                    f"Threshold: {curve_data['optimal_threshold']:.3f}<extra></extra>",
                        showlegend=False
                    ))
            
            # Add diagonal line (random classifier)
            if show_diagonal:
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Random Classifier',
                    hovertemplate="Random Classifier<br>AUC = 0.500<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title="ROC Curve Analysis",
                title_x=0.5,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=None,
                height=chart_height,
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                ),
                hovermode='closest'
            )
            
            # Set equal aspect ratio
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ROC curves: {str(e)}")
            return False
    
    def _create_combined_curves(self, roc_data: List[Dict[str, Any]], curve_style: str,
                               color_palette: str, show_diagonal: bool, show_auc_values: bool,
                               show_optimal_threshold: bool, chart_height: int, line_width: int) -> bool:
        """
        Create combined ROC and Precision-Recall curves.
        
        Args:
            roc_data: List of ROC curve data dictionaries
            curve_style: Style of the curves
            color_palette: Color palette to use
            show_diagonal: Whether to show diagonal line
            show_auc_values: Whether to show AUC values
            show_optimal_threshold: Whether to mark optimal thresholds
            chart_height: Height of the chart
            line_width: Width of the curves
            
        Returns:
            bool: True if successful
        """
        try:
            colors = self.color_palettes[color_palette]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('ROC Curves', 'Precision-Recall Curves'),
                horizontal_spacing=0.1
            )
            
            # Add ROC curves
            for i, curve_data in enumerate(roc_data):
                color = colors[i % len(colors)]
                
                # ROC curve name
                roc_name = curve_data['model_name']
                if show_auc_values:
                    roc_name += f" (AUC = {curve_data['auc']:.3f})"
                
                # Add ROC curve
                fig.add_trace(
                    go.Scatter(
                        x=curve_data['fpr'],
                        y=curve_data['tpr'],
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        name=roc_name,
                        legendgroup=f"group{i}",
                        hovertemplate=f"<b>{curve_data['model_name']}</b><br>" +
                                    f"FPR: %{{x:.3f}}<br>" +
                                    f"TPR: %{{y:.3f}}<extra></extra>"
                    ),
                    row=1, col=1
                )
                
                # Add PR curve if available
                if curve_data['precision'] is not None and curve_data['recall'] is not None:
                    pr_name = curve_data['model_name']
                    if show_auc_values and curve_data['pr_auc'] is not None:
                        pr_name += f" (PR-AUC = {curve_data['pr_auc']:.3f})"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=curve_data['recall'],
                            y=curve_data['precision'],
                            mode='lines',
                            line=dict(color=color, width=line_width),
                            name=pr_name,
                            legendgroup=f"group{i}",
                            showlegend=False,
                            hovertemplate=f"<b>{curve_data['model_name']}</b><br>" +
                                        f"Recall: %{{x:.3f}}<br>" +
                                        f"Precision: %{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=2
                    )
            
            # Add diagonal line for ROC
            if show_diagonal:
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Random Classifier',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="ROC and Precision-Recall Curve Analysis",
                title_x=0.5,
                height=chart_height,
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)
            fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
            fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating combined curves: {str(e)}")
            return False
    
    def _display_auc_analysis(self, roc_data: List[Dict[str, Any]]) -> None:
        """Display detailed AUC analysis."""
        st.markdown("### 📊 **AUC Analysis**")
        
        # Create AUC summary table
        auc_data = []
        for curve_data in roc_data:
            auc_data.append({
                'Model': curve_data['model_name'],
                'ROC AUC': f"{curve_data['auc']:.4f}",
                'PR AUC': f"{curve_data['pr_auc']:.4f}" if curve_data['pr_auc'] is not None else "N/A",
                'Performance': self._interpret_auc(curve_data['auc'])
            })
        
        if auc_data:
            auc_df = pd.DataFrame(auc_data)
            st.dataframe(auc_df, use_container_width=True, hide_index=True)
        
        # AUC interpretation guide
        with st.expander("📚 AUC Interpretation Guide"):
            st.markdown("""
            **ROC AUC Score Interpretation:**
            - **0.90 - 1.00**: Excellent performance
            - **0.80 - 0.90**: Good performance  
            - **0.70 - 0.80**: Fair performance
            - **0.60 - 0.70**: Poor performance
            - **0.50 - 0.60**: Very poor performance (barely better than random)
            - **< 0.50**: Worse than random (model may be inverted)
            
            **Precision-Recall AUC** is especially useful for imbalanced datasets.
            """)
    
    def _display_optimal_thresholds(self, roc_data: List[Dict[str, Any]]) -> None:
        """Display optimal threshold analysis."""
        st.markdown("### 🎯 **Optimal Threshold Analysis**")
        
        threshold_data = []
        for curve_data in roc_data:
            if curve_data['optimal_threshold'] is not None:
                fpr, tpr = curve_data['optimal_point']
                threshold_data.append({
                    'Model': curve_data['model_name'],
                    'Optimal Threshold': f"{curve_data['optimal_threshold']:.4f}",
                    'True Positive Rate': f"{tpr:.4f}",
                    'False Positive Rate': f"{fpr:.4f}",
                    'Youden Index': f"{tpr - fpr:.4f}"
                })
        
        if threshold_data:
            threshold_df = pd.DataFrame(threshold_data)
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
            
            st.info("💡 Optimal thresholds are calculated using Youden's Index (TPR - FPR)")
    
    def _display_comparison_insights(self, roc_data: List[Dict[str, Any]]) -> None:
        """Display model comparison insights."""
        st.markdown("### 💡 **Model Comparison Insights**")
        
        # Find best and worst models
        auc_scores = [(curve_data['model_name'], curve_data['auc']) for curve_data in roc_data]
        auc_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_auc = auc_scores[0]
        worst_model, worst_auc = auc_scores[-1]
        
        insights = []
        insights.append(f"🏆 **Best Model**: {best_model} (AUC = {best_auc:.4f})")
        insights.append(f"📉 **Worst Model**: {worst_model} (AUC = {worst_auc:.4f})")
        
        # Performance spread
        auc_values = [score[1] for score in auc_scores]
        auc_spread = max(auc_values) - min(auc_values)
        insights.append(f"📊 **Performance Spread**: {auc_spread:.4f}")
        
        # Performance categories
        excellent = sum(1 for _, auc in auc_scores if auc >= 0.9)
        good = sum(1 for _, auc in auc_scores if 0.8 <= auc < 0.9)
        fair = sum(1 for _, auc in auc_scores if 0.7 <= auc < 0.8)
        poor = sum(1 for _, auc in auc_scores if auc < 0.7)
        
        insights.append(f"📈 **Performance Distribution**: {excellent} excellent, {good} good, {fair} fair, {poor} poor")
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    def _interpret_auc(self, auc_score: float) -> str:
        """Interpret AUC score."""
        if auc_score >= 0.9:
            return "Excellent"
        elif auc_score >= 0.8:
            return "Good"
        elif auc_score >= 0.7:
            return "Fair"
        elif auc_score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_sample_data(self) -> Optional[pd.DataFrame]:
        """Get sample data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample binary classification data
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        return pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })
    
    def get_documentation(self) -> Dict[str, str]:
        """Get documentation for this plugin."""
        return {
            'overview': "Comprehensive ROC curve analysis for classification models with AUC calculation and optimal threshold detection",
            'usage': "Select classification models with probability predictions to generate ROC curves and analyze performance",
            'requirements': "Classification models with prediction probabilities or scores",
            'supported_data_types': "Binary Classification, Multiclass Classification",
            'configuration': "Curve style, color palette, confidence intervals, threshold optimization",
            'examples': "Compare multiple models, analyze optimal thresholds, evaluate classification performance across different operating points"
        }

def get_plugin():
    return ROCCurvePlugin()  # Use whatever your actual class name is