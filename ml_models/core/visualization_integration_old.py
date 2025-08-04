"""
Professional Visualization Integration Module
==========================================

This module provides seamless integration of the visualization_summary.py
functionality into the main ML application, enabling both in-app display
and PNG export capabilities for thesis-quality visualizations.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import zipfile
import re

# Set matplotlib backend to avoid DLASCLS errors
plt.switch_backend('Agg')

# Add the parent directory to path to import visualization_summary
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Anonymization level mapping
ANONYMIZATION_LEVELS = {
    'original': 'Original',
    'laplace_minimal': 'Minimal',
    'laplace_medium': 'Medium', 
    'laplace_high': 'High',
    'gaussian_minimal': 'Minimal',
    'gaussian_medium': 'Medium',
    'gaussian_high': 'High',
    'micro_aggregation_minimal': 'Minimal',
    'micro_aggregation_medium': 'Medium',
    'micro_aggregation_high': 'High',
    'differential_privacy_minimal': 'Minimal',
    'differential_privacy_medium': 'Medium',
    'differential_privacy_high': 'High',
    'randomized_response_minimal': 'Minimal',
    'randomized_response_medium': 'Medium',
    'randomized_response_high': 'High'
}

# Professional color palette (same as visualization_summary.py)
PROFESSIONAL_COLORS = ['#007ACC', '#FF6B35', '#28A745']
MODEL_COLORS = {
    'Logistic Regression': '#007ACC',
    'Random Forest': '#FF6B35', 
    'XGBoost': '#28A745'
}

TECHNIQUE_COLORS = {
    'Original': '#2D3748',
    'Minimal': '#4299E1',
    'Medium': '#3182CE', 
    'High': '#2B6CB0'
}

try:
    # Import the visualization functions (but we'll create our own dynamic versions)
    from visualization_summary import (
        plot_summary_bar,
        plot_anonymization_impact_heatmap,
        plot_performance_degradation,
        plot_anonymization_comparison,
        plot_percentage_degradation
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import visualization_summary: {e}")
    VISUALIZATION_AVAILABLE = False

class VisualizationIntegrator:
    """
    Professional class for integrating visualization capabilities into the ML app
    """
    
    def __init__(self):
        """Initialize the visualization integrator"""
        self.temp_dir = tempfile.mkdtemp(prefix="ml_viz_")
        self.generated_files = []
        
    def transform_summary_to_visualization_data(self, summary_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform summary DataFrame to the nested dictionary format expected by visualization functions
        
        Args:
            summary_df: DataFrame with experiment results
            
        Returns:
            Dict: Nested structure for visualization functions
        """
        try:
            # Initialize the results structure
            results = {}
            
            # Extract unique actual datasets (not the anonymized versions)
            dataset_col = None
            for col in summary_df.columns:
                if any(term in col.lower() for term in ['dataset', 'data']) and 'type' not in col.lower():
                    dataset_col = col
                    break
            
            if dataset_col and dataset_col in summary_df.columns:
                unique_datasets = summary_df[dataset_col].unique()
            else:
                # If no dataset column found, use a default
                unique_datasets = ['Main_Dataset']
            
            # Process each actual dataset
            for dataset_name in unique_datasets:
                if dataset_name not in results:
                    results[dataset_name] = {}
                
                # Filter data for this dataset
                if dataset_col:
                    dataset_data = summary_df[summary_df[dataset_col] == dataset_name].copy()
                else:
                    dataset_data = summary_df.copy()
                
                if len(dataset_data) == 0:
                    continue
                
                # Get privacy method column (this contains the anonymization info)
                privacy_col = None
                for col in dataset_data.columns:
                    if any(term in col.lower() for term in ['privacy', 'method', 'type']) and 'dataset' not in col.lower():
                        privacy_col = col
                        break
                
                # Get model column
                model_col = None
                for col in dataset_data.columns:
                    if any(term in col.lower() for term in ['model', 'algorithm']):
                        model_col = col
                        break
                
                # Get metric columns (exclude change/diff columns)
                metric_cols = []
                for col in dataset_data.columns:
                    if any(metric in col.lower() for metric in ['accuracy', 'precision', 'recall', 'f1', 'r²', 'rmse', 'mae']):
                        if not any(exclude in col.lower() for exclude in ['change', 'diff', '%', '△', '▲', '▼']):
                            metric_cols.append(col)
                
                # Map privacy methods to anonymization levels
                if privacy_col:
                    unique_privacy_methods = dataset_data[privacy_col].unique()
                else:
                    unique_privacy_methods = ['Unknown']
                
                for privacy_method in unique_privacy_methods:
                    # Map to anonymization level
                    privacy_method_clean = str(privacy_method).lower().replace(' ', '_')
                    anonymization_level = ANONYMIZATION_LEVELS.get(privacy_method_clean, str(privacy_method))
                    
                    if anonymization_level not in results[dataset_name]:
                        results[dataset_name][anonymization_level] = {}
                    
                    # Filter data for this privacy method
                    if privacy_col:
                        method_data = dataset_data[dataset_data[privacy_col] == privacy_method]
                    else:
                        method_data = dataset_data.copy()
                    
                    # Get unique models
                    if model_col:
                        unique_models = method_data[model_col].unique()
                    else:
                        unique_models = [f'Model_{i}' for i in range(len(method_data))]
                    
                    # Process each model
                    for model in unique_models:
                        if model not in results[dataset_name][anonymization_level]:
                            results[dataset_name][anonymization_level][model] = {}
                        
                        # Filter data for this model
                        if model_col:
                            model_data = method_data[method_data[model_col] == model]
                        else:
                            model_data = method_data.iloc[[0]] if len(method_data) > 0 else method_data
                        
                        if len(model_data) == 0:
                            continue
                        
                        # Extract metrics for this model
                        model_row = model_data.iloc[0]
                        
                        for metric_col in metric_cols:
                            if metric_col in model_row.index and pd.notna(model_row[metric_col]):
                                try:
                                    # Clean metric name (remove emojis)
                                    clean_metric = re.sub(r'[^\w\s-]', '', metric_col).strip()
                                    clean_metric = clean_metric.replace('  ', ' ').strip()
                                    if not clean_metric:
                                        clean_metric = metric_col
                                    
                                    # Convert to float
                                    value = float(model_row[metric_col])
                                    results[dataset_name][anonymization_level][model][clean_metric] = value
                                    
                                except (ValueError, TypeError):
                                    continue
            
            return results
            
        except Exception as e:
            st.error(f"Error transforming data for visualization: {str(e)}")
            return {}
    
    def create_dynamic_bar_chart(self, results_data: Dict, dataset: str, metric: str) -> plt.Figure:
        """Create a dynamic bar chart for performance comparison"""
        try:
            # Set up the plot with professional styling
            plt.rcParams['font.family'] = 'DejaVu Sans'  # Use safe font
            plt.rcParams['figure.dpi'] = 100  # Lower DPI for display
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Prepare data
            methods = list(results_data[dataset].keys())
            models = []
            values = []
            colors = []
            
            # Get all models for this dataset
            all_models = set()
            for method in methods:
                all_models.update(results_data[dataset][method].keys())
            all_models = sorted(list(all_models))
            
            # Prepare data for plotting
            for method in methods:
                for model in all_models:
                    if model in results_data[dataset][method] and metric in results_data[dataset][method][model]:
                        models.append(f"{model}\n({method})")
                        values.append(results_data[dataset][method][model][metric])
                        colors.append(TECHNIQUE_COLORS.get(method, '#007ACC'))
            
            if not values:
                raise ValueError(f"No data found for metric '{metric}' in dataset '{dataset}'")
            
            # Create bar chart
            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_title(f'{metric} Comparison - {dataset}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Model (Anonymization Level)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def create_dynamic_heatmap(self, results_data: Dict, dataset: str) -> plt.Figure:
        """Create a dynamic heatmap for anonymization impact"""
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['figure.dpi'] = 100
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Prepare data for heatmap
            methods = list(results_data[dataset].keys())
            all_models = set()
            all_metrics = set()
            
            for method in methods:
                for model in results_data[dataset][method].keys():
                    all_models.add(model)
                    all_metrics.update(results_data[dataset][method][model].keys())
            
            all_models = sorted(list(all_models))
            all_metrics = sorted(list(all_metrics))
            
            # Create matrix
            matrix_data = []
            row_labels = []
            
            for method in methods:
                for model in all_models:
                    row_label = f"{model} ({method})"
                    row_labels.append(row_label)
                    row_data = []
                    
                    for metric in all_metrics:
                        if (model in results_data[dataset][method] and 
                            metric in results_data[dataset][method][model]):
                            value = results_data[dataset][method][model][metric]
                        else:
                            value = 0
                        row_data.append(value)
                    
                    matrix_data.append(row_data)
            
            if not matrix_data:
                raise ValueError(f"No data available for heatmap in dataset '{dataset}'")
            
            # Create heatmap
            matrix_data = np.array(matrix_data)
            im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
            
            # Customize the plot
            ax.set_title(f'Performance Heatmap - {dataset}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(range(len(all_metrics)))
            ax.set_xticklabels(all_metrics, rotation=45, ha='right')
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Score', fontsize=12, fontweight='bold')
            
            # Add text annotations
            for i in range(len(row_labels)):
                for j in range(len(all_metrics)):
                    text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def generate_visualizations_in_app(self, summary_df: pd.DataFrame) -> bool:
        """
        Generate and display visualizations within the Streamlit app
        
        Args:
            summary_df: Summary DataFrame from experiments
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Transform data
            results_data = self.transform_summary_to_visualization_data(summary_df)
            
            if not results_data:
                st.warning("⚠️ No data available for visualization")
                return False
            
            # Display debugging info
            st.info(f"📊 **Data Summary:** Found {len(results_data)} datasets with {sum(len(results_data[d]) for d in results_data)} total anonymization levels")
            
            # Display visualizations
            datasets = list(results_data.keys())
            
            st.markdown("### 📊 **Professional Visualization Dashboard**")
            st.markdown("---")
            
            # Create tabs for different visualization types
            viz_tabs = st.tabs([
                "📊 Performance Comparison", 
                "🔥 Impact Heatmap"
            ])
            
            with viz_tabs[0]:  # Performance Comparison
                st.markdown("#### 📊 **Performance Comparison by Metric**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        # Get available metrics
                        available_metrics = set()
                        for method in results_data[dataset].values():
                            for model in method.values():
                                available_metrics.update(model.keys())
                        
                        available_metrics = sorted(list(available_metrics))
                        
                        if not available_metrics:
                            st.warning("No metrics available for visualization")
                            continue
                        
                        # Create metric selection
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            selected_metric = st.selectbox(
                                "Select Metric:",
                                options=available_metrics,
                                key=f"metric_selector_{dataset}"
                            )
                        
                        # Generate bar chart
                        try:
                            fig = self.create_dynamic_bar_chart(results_data, dataset, selected_metric)
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.error("Failed to create bar chart")
                        except Exception as e:
                            st.error(f"Error generating bar chart: {str(e)}")
            
            with viz_tabs[1]:  # Impact Heatmap
                st.markdown("#### 🔥 **Anonymization Impact Heatmap**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        try:
                            fig = self.create_dynamic_heatmap(results_data, dataset)
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.error("Failed to create heatmap")
                        except Exception as e:
                            st.error(f"Error generating heatmap: {str(e)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error generating in-app visualizations: {str(e)}")
            return False
        """
        Generate and display visualizations within the Streamlit app
        
        Args:
            summary_df: Summary DataFrame from experiments
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not VISUALIZATION_AVAILABLE:
            st.error("❌ Visualization module not available. Please check visualization_summary.py")
            return False
        
        try:
            # Transform data
            results_data = self.transform_summary_to_visualization_data(summary_df)
            
            if not results_data:
                st.warning("⚠️ No data available for visualization")
                return False
            
            # Display visualizations
            datasets = list(results_data.keys())
            
            st.markdown("### 📊 **Professional Visualization Dashboard**")
            st.markdown("---")
            
            # Create tabs for different visualization types
            viz_tabs = st.tabs([
                "📊 Performance Comparison", 
                "🔥 Impact Heatmap", 
                "📈 Degradation Analysis", 
                "⚖️ Privacy-Utility Trade-off",
                "📉 Percentage Degradation"
            ])
            
            with viz_tabs[0]:  # Performance Comparison
                st.markdown("#### 📊 **Performance Comparison by Metric**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        # Get available metrics
                        sample_model = list(results_data[dataset].values())[0]
                        sample_metrics = list(list(sample_model.values())[0].keys())
                        
                        # Create metric selection
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            selected_metric = st.selectbox(
                                "Select Metric:",
                                options=sample_metrics,
                                key=f"metric_selector_{dataset}"
                            )
                        
                        # Generate bar chart
                        try:
                            import matplotlib.pyplot as plt
                            import io
                            
                            # Create the plot
                            fig = plt.figure(figsize=(14, 8))
                            plot_summary_bar(results_data, dataset, selected_metric, save_path=None)
                            
                            # Display in Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating bar chart: {str(e)}")
            
            with viz_tabs[1]:  # Impact Heatmap
                st.markdown("#### 🔥 **Anonymization Impact Heatmap**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        try:
                            import matplotlib.pyplot as plt
                            
                            # Generate heatmap
                            fig = plt.figure(figsize=(12, 10))
                            plot_anonymization_impact_heatmap(results_data, dataset, save_path=None)
                            
                            # Display in Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating heatmap: {str(e)}")
            
            with viz_tabs[2]:  # Degradation Analysis
                st.markdown("#### 📈 **Performance Degradation Analysis**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        try:
                            import matplotlib.pyplot as plt
                            
                            # Generate degradation analysis
                            fig = plt.figure(figsize=(18, 14))
                            plot_performance_degradation(results_data, dataset, save_path=None)
                            
                            # Display in Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating degradation analysis: {str(e)}")
            
            with viz_tabs[3]:  # Privacy-Utility Trade-off
                st.markdown("#### ⚖️ **Privacy vs. Utility Trade-off**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        try:
                            import matplotlib.pyplot as plt
                            
                            # Generate trade-off analysis
                            fig = plt.figure(figsize=(14, 10))
                            plot_anonymization_comparison(results_data, dataset, save_path=None)
                            
                            # Display in Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating trade-off analysis: {str(e)}")
            
            with viz_tabs[4]:  # Percentage Degradation
                st.markdown("#### 📉 **Percentage Performance Degradation**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        try:
                            import matplotlib.pyplot as plt
                            
                            # Generate percentage degradation
                            fig = plt.figure(figsize=(16, 12))
                            plot_percentage_degradation(results_data, dataset, save_path=None)
                            
                            # Display in Streamlit
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating percentage degradation: {str(e)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error generating in-app visualizations: {str(e)}")
            return False
    
    def generate_png_exports(self, summary_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Generate PNG exports of all visualizations
        
        Args:
            summary_df: Summary DataFrame from experiments
            
        Returns:
            Tuple[bool, Optional[str]]: (Success status, zip file path if successful)
        """
        if not VISUALIZATION_AVAILABLE:
            return False, None
        
        try:
            # Transform data
            results_data = self.transform_summary_to_visualization_data(summary_df)
            
            if not results_data:
                return False, None
            
            # Create export directory
            export_dir = os.path.join(self.temp_dir, f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(export_dir, exist_ok=True)
            
            datasets = list(results_data.keys())
            
            # Generate all visualizations for each dataset
            for dataset in datasets:
                dataset_dir = os.path.join(export_dir, f"{dataset.lower().replace(' ', '_')}_visualizations")
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Get available metrics for bar charts
                sample_model = list(results_data[dataset].values())[0]
                sample_metrics = list(list(sample_model.values())[0].keys())
                
                # Generate bar charts for each metric
                for metric in sample_metrics:
                    save_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_{metric}_comparison.png")
                    plot_summary_bar(results_data, dataset, metric, save_path=save_path)
                    self.generated_files.append(save_path)
                
                # Generate heatmap
                heatmap_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_performance_heatmap.png")
                plot_anonymization_impact_heatmap(results_data, dataset, save_path=heatmap_path)
                self.generated_files.append(heatmap_path)
                
                # Generate degradation analysis
                degradation_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_performance_degradation.png")
                plot_performance_degradation(results_data, dataset, save_path=degradation_path)
                self.generated_files.append(degradation_path)
                
                # Generate privacy-utility trade-off
                tradeoff_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_privacy_utility_tradeoff.png")
                plot_anonymization_comparison(results_data, dataset, save_path=tradeoff_path)
                self.generated_files.append(tradeoff_path)
                
                # Generate percentage degradation
                percentage_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_percentage_degradation.png")
                plot_percentage_degradation(results_data, dataset, save_path=percentage_path)
                self.generated_files.append(percentage_path)
            
            # Create zip file
            zip_path = os.path.join(self.temp_dir, f"thesis_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.generated_files:
                    if os.path.exists(file_path):
                        # Create relative path for zip
                        rel_path = os.path.relpath(file_path, export_dir)
                        zipf.write(file_path, rel_path)
            
            return True, zip_path
            
        except Exception as e:
            st.error(f"Error generating PNG exports: {str(e)}")
            return False, None
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass

# Global instance for session state
def get_visualization_integrator():
    """Get or create visualization integrator instance"""
    if 'viz_integrator' not in st.session_state:
        st.session_state.viz_integrator = VisualizationIntegrator()
    return st.session_state.viz_integrator
