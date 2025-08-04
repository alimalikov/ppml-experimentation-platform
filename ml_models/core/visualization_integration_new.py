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
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import zipfile
import re

# Professional styling setup
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

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

# Professional color palette
TECHNIQUE_COLORS = {
    'Original': '#2D3748',
    'Minimal': '#4299E1',
    'Medium': '#3182CE', 
    'High': '#2B6CB0'
}

MODEL_COLORS = {
    'Logistic Regression': '#007ACC',
    'Random Forest': '#FF6B35', 
    'XGBoost': '#28A745'
}

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
            results = {}
            
            # Find dataset column
            dataset_col = None
            for col in summary_df.columns:
                if any(term in col.lower() for term in ['dataset', 'data']) and 'type' not in col.lower():
                    dataset_col = col
                    break
            
            # Get unique datasets
            if dataset_col and dataset_col in summary_df.columns:
                unique_datasets = summary_df[dataset_col].unique()
            else:
                unique_datasets = ['Main_Dataset']
            
            # Process each dataset
            for dataset_name in unique_datasets:
                results[dataset_name] = {}
                
                # Filter data for this dataset
                if dataset_col:
                    dataset_data = summary_df[summary_df[dataset_col] == dataset_name].copy()
                else:
                    dataset_data = summary_df.copy()
                
                if len(dataset_data) == 0:
                    continue
                
                # Find privacy/method column
                privacy_col = None
                for col in dataset_data.columns:
                    if any(term in col.lower() for term in ['privacy', 'method', 'type']) and 'dataset' not in col.lower():
                        privacy_col = col
                        break
                
                # Find model column
                model_col = None
                for col in dataset_data.columns:
                    if any(term in col.lower() for term in ['model', 'algorithm']):
                        model_col = col
                        break
                
                # Find metric columns
                metric_cols = []
                for col in dataset_data.columns:
                    if any(metric in col.lower() for metric in ['accuracy', 'precision', 'recall', 'f1', 'r²', 'rmse', 'mae']):
                        if not any(exclude in col.lower() for exclude in ['change', 'diff', '%', '△', '▲', '▼']):
                            metric_cols.append(col)
                
                # Process each privacy method
                if privacy_col:
                    unique_privacy_methods = dataset_data[privacy_col].unique()
                else:
                    unique_privacy_methods = ['Unknown']
                
                for privacy_method in unique_privacy_methods:
                    # Map to anonymization level
                    privacy_clean = str(privacy_method).lower().replace(' ', '_')
                    anon_level = ANONYMIZATION_LEVELS.get(privacy_clean, str(privacy_method))
                    
                    results[dataset_name][anon_level] = {}
                    
                    # Filter data for this privacy method
                    if privacy_col:
                        method_data = dataset_data[dataset_data[privacy_col] == privacy_method]
                    else:
                        method_data = dataset_data.copy()
                    
                    # Get models
                    if model_col:
                        unique_models = method_data[model_col].unique()
                    else:
                        unique_models = [f'Model_{i}' for i in range(len(method_data))]
                    
                    # Process each model
                    for model in unique_models:
                        results[dataset_name][anon_level][model] = {}
                        
                        # Filter data for this model
                        if model_col:
                            model_data = method_data[method_data[model_col] == model]
                        else:
                            model_data = method_data.iloc[[0]] if len(method_data) > 0 else method_data
                        
                        if len(model_data) == 0:
                            continue
                        
                        # Extract metrics
                        model_row = model_data.iloc[0]
                        for metric_col in metric_cols:
                            if metric_col in model_row.index and pd.notna(model_row[metric_col]):
                                try:
                                    # Clean metric name
                                    clean_metric = re.sub(r'[^\w\s-]', '', metric_col).strip()
                                    clean_metric = clean_metric.replace('  ', ' ').strip()
                                    if not clean_metric:
                                        clean_metric = metric_col
                                    
                                    value = float(model_row[metric_col])
                                    results[dataset_name][anon_level][model][clean_metric] = value
                                except (ValueError, TypeError):
                                    continue
            
            return results
            
        except Exception as e:
            st.error(f"Error transforming data: {str(e)}")
            return {}
    
    def create_bar_chart(self, results_data: Dict, dataset: str, metric: str) -> plt.Figure:
        """Create bar chart for performance comparison"""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Prepare data
            methods = list(results_data[dataset].keys())
            models = set()
            for method in methods:
                models.update(results_data[dataset][method].keys())
            models = sorted(list(models))
            
            # Collect data
            plot_data = []
            labels = []
            colors = []
            
            for method in methods:
                for model in models:
                    if (model in results_data[dataset][method] and 
                        metric in results_data[dataset][method][model]):
                        value = results_data[dataset][method][model][metric]
                        plot_data.append(value)
                        labels.append(f"{model}\n({method})")
                        colors.append(TECHNIQUE_COLORS.get(method, '#007ACC'))
            
            if not plot_data:
                raise ValueError(f"No data for metric '{metric}'")
            
            # Create bars
            bars = ax.bar(range(len(plot_data)), plot_data, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize
            ax.set_title(f'{metric} Comparison - {dataset}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Model (Anonymization Level)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, plot_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(plot_data)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def create_heatmap(self, results_data: Dict, dataset: str) -> plt.Figure:
        """Create heatmap for performance overview"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get all methods, models, and metrics
            methods = list(results_data[dataset].keys())
            all_models = set()
            all_metrics = set()
            
            for method in methods:
                for model in results_data[dataset][method].keys():
                    all_models.add(model)
                    all_metrics.update(results_data[dataset][method][model].keys())
            
            all_models = sorted(list(all_models))
            all_metrics = sorted(list(all_metrics))
            
            # Create data matrix
            matrix_data = []
            row_labels = []
            
            for method in methods:
                for model in all_models:
                    row_labels.append(f"{model} ({method})")
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
                raise ValueError("No data for heatmap")
            
            # Create heatmap
            matrix_data = np.array(matrix_data)
            im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
            
            # Customize
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
                    ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def generate_visualizations_in_app(self, summary_df: pd.DataFrame) -> bool:
        """Generate and display visualizations in the app"""
        try:
            # Transform data
            results_data = self.transform_summary_to_visualization_data(summary_df)
            
            if not results_data:
                st.warning("⚠️ No data available for visualization")
                return False
            
            datasets = list(results_data.keys())
            st.info(f"📊 **Data Ready:** {len(datasets)} datasets found")
            
            # Create tabs
            viz_tabs = st.tabs(["📊 Performance Comparison", "🔥 Impact Heatmap"])
            
            with viz_tabs[0]:
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
                            st.warning("No metrics available")
                            continue
                        
                        # Metric selection
                        selected_metric = st.selectbox(
                            "Select Metric:",
                            options=available_metrics,
                            key=f"metric_{dataset}"
                        )
                        
                        # Generate chart
                        fig = self.create_bar_chart(results_data, dataset, selected_metric)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
            
            with viz_tabs[1]:
                st.markdown("#### 🔥 **Performance Heatmap**")
                
                for dataset in datasets:
                    with st.expander(f"📂 **{dataset} Dataset**", expanded=True):
                        fig = self.create_heatmap(results_data, dataset)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
            
            return True
            
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
            return False
    
    def generate_png_exports(self, summary_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Generate PNG exports of visualizations"""
        try:
            results_data = self.transform_summary_to_visualization_data(summary_df)
            
            if not results_data:
                return False, None
            
            # Create export directory
            export_dir = os.path.join(self.temp_dir, f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(export_dir, exist_ok=True)
            
            datasets = list(results_data.keys())
            
            # Generate charts for each dataset
            for dataset in datasets:
                dataset_dir = os.path.join(export_dir, f"{dataset.lower().replace(' ', '_')}_charts")
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Get available metrics
                available_metrics = set()
                for method in results_data[dataset].values():
                    for model in method.values():
                        available_metrics.update(model.keys())
                
                # Generate bar charts for each metric
                for metric in available_metrics:
                    fig = self.create_bar_chart(results_data, dataset, metric)
                    if fig:
                        save_path = os.path.join(dataset_dir, f"{metric.lower().replace(' ', '_')}_comparison.png")
                        fig.savefig(save_path, dpi=300, bbox_inches='tight')
                        self.generated_files.append(save_path)
                        plt.close(fig)
                
                # Generate heatmap
                fig = self.create_heatmap(results_data, dataset)
                if fig:
                    save_path = os.path.join(dataset_dir, f"{dataset.lower().replace(' ', '_')}_heatmap.png")
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.generated_files.append(save_path)
                    plt.close(fig)
            
            # Create ZIP file
            zip_path = os.path.join(self.temp_dir, f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.generated_files:
                    if os.path.exists(file_path):
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
