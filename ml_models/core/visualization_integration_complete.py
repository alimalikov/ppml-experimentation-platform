"""
Complete Visualization Integration - Professional Tab-Based System
================================================================

Extracts data from ML app tables and provides ALL visualization functions
from visualization_summary.py with professional tab-based navigation.

Features:
- Tab-based UI navigation
- Complete visualization suite (all functions from visualization_summary.py)
- Porsche Next TT font for visualizations
- Dual sizing: half-size for display, full-size for download
- Proper label mapping with anonymization level configuration
- Horizontal labels

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
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
import io
import base64

# EXACT SAME STYLING FROM ORIGINAL - WITH PORSCHE NEXT TT
plt.style.use('default')
sns.set_palette("husl")

from matplotlib import rcParams

# Try to use Porsche Next TT, fallback to professional alternatives
try:
    rcParams['font.family'] = 'Porsche Next TT'
except:
    try:
        rcParams['font.family'] = 'Arial'
    except:
        rcParams['font.family'] = 'DejaVu Sans'

rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 12
rcParams['legend.title_fontsize'] = 13
rcParams['figure.titlesize'] = 18
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['grid.linewidth'] = 0.8
rcParams['lines.linewidth'] = 2.5
rcParams['lines.markersize'] = 8

# EXACT SAME COLORS FROM ORIGINAL
PROFESSIONAL_COLORS = ['#007ACC', '#FF6B35', '#28A745']  # Bright Blue, Orange, Green
MODEL_COLORS = {
    'Logistic Regression': '#007ACC',  # Bright blue
    'Random Forest': '#FF6B35',       # Bright orange
    'XGBoost': '#28A745'              # Bright green
}

TECHNIQUE_COLORS = {
    'Original': '#2D3748',
    'Micro Aggregation (High)': '#38A169',
    'Micro Aggregation (Medium)': '#48BB78',
    'Micro Aggregation (Minimal)': '#68D391',
    'Differential Privacy (Minimal)': '#4299E1',
    'Differential Privacy (Medium)': '#3182CE',
    'Differential Privacy (High)': '#2B6CB0',
    'Randomized Response (Minimal)': '#ED8936',
    'Randomized Response (Medium)': '#DD6B20',
    'Randomized Response (High)': '#C05621'
}

class CompleteVisualizationSystem:
    """Complete visualization system with tab-based navigation and all original functions."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visualization_files = []
        
    def extract_table_data_with_proper_labels(self, df: pd.DataFrame, selected_metrics: List[str], 
                                            anonymization_configs: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract table data and create proper labels using privacy method + anonymization level.
        
        Args:
            df: DataFrame with table data
            selected_metrics: List of selected metric columns
            anonymization_configs: Dict mapping privacy methods to levels (from ⚙️ configuration)
        
        Returns:
            Dictionary in exact format expected by visualization_summary.py functions
        """
        try:
            results = {}
            
            # Get dataset name (assuming single dataset for now, can be extended)
            dataset_name = "Current Dataset"  # Can be extracted from context or made configurable
            
            results[dataset_name] = {}
            
            # Process each row to create proper anonymization technique labels
            for _, row in df.iterrows():
                try:
                    # Extract privacy method from table
                    privacy_method = ""
                    if '📊 Privacy Method' in df.columns:
                        privacy_method = str(row['📊 Privacy Method']).strip()
                    
                    # Get model name
                    model_name = ""
                    if '🤖 Model' in df.columns:
                        model_name = str(row['🤖 Model']).strip()
                    elif 'Model' in df.columns:
                        model_name = str(row['Model']).strip()
                    
                    if not model_name or not privacy_method:
                        continue
                    
                    # Create proper anonymization technique label
                    # Map privacy method to proper format and get user's selected level
                    anonymization_label = self._create_anonymization_label(privacy_method, anonymization_configs)
                    
                    # Initialize nested structure
                    if anonymization_label not in results[dataset_name]:
                        results[dataset_name][anonymization_label] = {}
                    if model_name not in results[dataset_name][anonymization_label]:
                        results[dataset_name][anonymization_label][model_name] = {}
                    
                    # Extract metric values
                    for metric in selected_metrics:
                        if metric in df.columns:
                            value = row[metric]
                            # Convert to float, handle various formats
                            if pd.isna(value):
                                value = 0.0
                            else:
                                # Remove percentage signs, convert to decimal
                                if isinstance(value, str):
                                    value = value.replace('%', '').replace(',', '')
                                    if value.endswith('K') or value.endswith('k'):
                                        value = float(value[:-1]) * 1000
                                    elif value.endswith('M') or value.endswith('m'):
                                        value = float(value[:-1]) * 1000000
                                    else:
                                        value = float(value)
                                    
                                    # If it was a percentage, convert to decimal
                                    if '%' in str(row[metric]):
                                        value = value / 100.0
                                else:
                                    value = float(value)
                            
                            # Map metric name to standard format
                            standard_metric = self._standardize_metric_name(metric)
                            results[dataset_name][anonymization_label][model_name][standard_metric] = value
                
                except Exception as e:
                    st.warning(f"⚠️ Error processing row: {e}")
                    continue
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error extracting table data: {e}")
            return {}
    
    def _create_anonymization_label(self, privacy_method: str, anonymization_configs: Dict[str, str]) -> str:
        """Create proper anonymization label from privacy method and user configuration."""
        try:
            # Clean privacy method name
            method_clean = privacy_method.lower().replace(' ', '').replace('-', '').replace('_', '')
            
            # Map to standard format
            method_mapping = {
                'original': 'Original',
                'microaggregation': 'Micro Aggregation',
                'differentialprivacy': 'Differential Privacy',
                'randomizedresponse': 'Randomized Response'
            }
            
            standard_method = None
            for key, value in method_mapping.items():
                if key in method_clean:
                    standard_method = value
                    break
            
            if not standard_method:
                standard_method = privacy_method  # Fallback to original
            
            # Get user's selected level from configuration
            selected_level = anonymization_configs.get(privacy_method, 'Medium')  # Default to Medium
            
            # Handle Original case
            if 'original' in method_clean or standard_method == 'Original':
                return 'Original'
            
            # Create final label
            return f"{standard_method} ({selected_level})"
            
        except Exception as e:
            return privacy_method  # Fallback
    
    def _standardize_metric_name(self, metric: str) -> str:
        """Standardize metric names to match visualization_summary.py format."""
        metric_mapping = {
            'accuracy': 'accuracy',
            'precision': 'precision', 
            'recall': 'recall',
            'f1_score': 'f1_score',
            'f1-score': 'f1_score',
            'f1 score': 'f1_score',
            'balanced_accuracy': 'balanced_accuracy',
            'balanced accuracy': 'balanced_accuracy',
            'matthews_correlation': 'matthews_correlation',
            'matthews correlation': 'matthews_correlation',
            'mcc': 'matthews_correlation',
            'roc_auc': 'roc_auc',
            'roc auc': 'roc_auc',
            'auc': 'roc_auc'
        }
        
        metric_clean = metric.lower().replace(' ', '_').replace('-', '_')
        return metric_mapping.get(metric_clean, metric_clean)

    # ========== ALL VISUALIZATION FUNCTIONS FROM ORIGINAL ==========
    
    def plot_summary_bar(self, results, dataset, metric, save_path=None, display_size=True):
        """Create grouped bar chart showing anonymization effects on model performance."""
        data = []
        for anonymization, models in results.get(dataset, {}).items():
            for model, metrics in models.items():
                value = metrics.get(metric, None)
                if value is not None:
                    data.append({
                        "Anonymization": anonymization,
                        "Model": model,
                        metric: value
                    })
        
        if not data:
            st.warning(f"No data found for dataset '{dataset}' and metric '{metric}'.")
            return None
        
        df = pd.DataFrame(data)
        
        # Set figure size based on display vs download
        figsize = (8, 5) if display_size else (16, 10)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Professional bar plot
        bar_plot = sns.barplot(
            data=df,
            x="Anonymization",
            y=metric,
            hue="Model",
            palette=PROFESSIONAL_COLORS,
            ax=ax,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.9
        )
        
        # Professional styling
        ax.set_title(f'{dataset} Dataset - {metric.replace("_", " ").title()} Performance', 
                    fontsize=14 if display_size else 18, fontweight='bold', pad=20)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12 if display_size else 14, fontweight='semibold')
        ax.set_xlabel('Anonymization Technique', fontsize=12 if display_size else 14, fontweight='semibold')
        
        # HORIZONTAL labels as requested
        plt.xticks(rotation=0, ha='center', fontsize=10 if display_size else 12)
        plt.yticks(fontsize=11 if display_size else 13)
        
        # Professional legend
        legend = ax.legend(title='ML Algorithm', fontsize=10 if display_size else 12, 
                          title_fontsize=11 if display_size else 13, 
                          loc='upper right', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('lightgray')
        
        # Add value labels on tallest bars only
        x_positions = {}
        for i, container in enumerate(bar_plot.containers):
            for j, bar in enumerate(container):
                x_pos = bar.get_x() + bar.get_width() / 2
                height = bar.get_height()
                if x_pos not in x_positions or height > x_positions[x_pos]['height']:
                    x_positions[x_pos] = {'height': height, 'container_idx': i, 'bar_idx': j}
        
        for container_idx, container in enumerate(bar_plot.containers):
            for bar_idx, bar in enumerate(container):
                x_pos = bar.get_x() + bar.get_width() / 2
                height = bar.get_height()
                if x_positions[x_pos]['container_idx'] == container_idx and x_positions[x_pos]['bar_idx'] == bar_idx:
                    ax.text(x_pos, height + 0.01, f'{height:.3f}', 
                           ha='center', va='bottom', fontsize=9 if display_size else 11, fontweight='semibold')
        
        ax.set_ylim(0, max(df[metric]) * 1.1)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        
        return fig

    def plot_anonymization_impact_heatmap(self, results, dataset, save_path=None, display_size=True):
        """Create heatmap showing impact of anonymization techniques across models and metrics."""
        anonymization_techniques = list(results[dataset].keys())
        models = list(results[dataset][anonymization_techniques[0]].keys())
        metrics = list(results[dataset][anonymization_techniques[0]][models[0]].keys())
        
        heatmap_data = []
        labels = []
        
        for anon in anonymization_techniques:
            for model in models:
                row = []
                for metric in metrics:
                    value = results[dataset][anon][model].get(metric, 0)
                    row.append(value)
                heatmap_data.append(row)
                labels.append(f"{anon}\n{model}")
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=metrics, index=labels)
        
        figsize = (5, 6) if display_size else (10, 12)
        plt.figure(figsize=figsize)
        ax = sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                         cbar_kws={'label': 'Performance Score'}, 
                         linewidths=0.5, linecolor='white')
        ax.set_title(f"{dataset} Dataset - Performance Heatmap\nAnonymization Impact Across Models and Metrics", 
                     fontsize=12 if display_size else 16, fontweight='bold', pad=20)
        ax.set_xlabel("Metrics", fontsize=11 if display_size else 14)
        ax.set_ylabel("Anonymization Technique & Model", fontsize=11 if display_size else 14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        
        return plt.gcf()

    def plot_performance_degradation(self, results, dataset, save_path=None, display_size=True):
        """Create line plot showing performance degradation from original to anonymized versions."""
        original_data = results[dataset]["Original"]
        
        plot_data = []
        anonymization_order = ["Original", 
                              "Micro Aggregation (Minimal)", "Micro Aggregation (Medium)", "Micro Aggregation (High)",
                              "Differential Privacy (Minimal)", "Differential Privacy (Medium)", "Differential Privacy (High)",
                              "Randomized Response (Minimal)", "Randomized Response (Medium)", "Randomized Response (High)"]
        
        for anon in anonymization_order:
            if anon in results[dataset]:
                for model, metrics in results[dataset][anon].items():
                    for metric, value in metrics.items():
                        plot_data.append({
                            "Anonymization": anon,
                            "Model": model,
                            "Metric": metric,
                            "Value": value
                        })
        
        df = pd.DataFrame(plot_data)
        available_metrics = list(original_data[list(original_data.keys())[0]].keys())
        
        # Determine subplot layout
        if len(available_metrics) == 4:
            fig, axes = plt.subplots(2, 2, figsize=(9, 7) if display_size else (18, 14))
            axes = axes.flatten()
        elif len(available_metrics) == 7:
            fig, axes = plt.subplots(3, 3, figsize=(12, 9) if display_size else (24, 18))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(3, 3, figsize=(12, 9) if display_size else (24, 18))
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                metric_data = df[df['Metric'] == metric]
                
                for model in metric_data['Model'].unique():
                    model_data = metric_data[metric_data['Model'] == model]
                    axes[i].plot(range(len(model_data)), model_data['Value'], 
                               marker='o', linewidth=2, markersize=6, 
                               label=model, color=MODEL_COLORS.get(model, '#007ACC'))
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                                fontsize=11 if display_size else 14, fontweight='bold')
                axes[i].set_xlabel('Anonymization Level', fontsize=10 if display_size else 12)
                axes[i].set_ylabel('Performance', fontsize=10 if display_size else 12)
                axes[i].legend(fontsize=8 if display_size else 10)
                axes[i].grid(True, alpha=0.3)
                
                # HORIZONTAL labels
                axes[i].set_xticks(range(len(anonymization_order)))
                axes[i].set_xticklabels([a.split('(')[0].strip() for a in anonymization_order], 
                                       rotation=0, ha='center', fontsize=8 if display_size else 10)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{dataset} - Performance Degradation Analysis', 
                     fontsize=14 if display_size else 18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        
        return fig

    def plot_percentage_degradation(self, results, dataset, save_path=None, display_size=True):
        """Create percentage degradation analysis."""
        original_data = results[dataset]["Original"]
        
        degradation_data = []
        for anon_technique, models in results[dataset].items():
            if anon_technique == "Original":
                continue
                
            for model, metrics in models.items():
                for metric, value in metrics.items():
                    original_value = original_data[model][metric]
                    if original_value > 0:
                        degradation_pct = ((original_value - value) / original_value) * 100
                        degradation_data.append({
                            "Technique": anon_technique,
                            "Model": model,
                            "Metric": metric,
                            "Degradation_Percent": degradation_pct
                        })
        
        df = pd.DataFrame(degradation_data)
        
        figsize = (10, 6) if display_size else (20, 12)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grouped bar plot
        sns.barplot(data=df, x="Technique", y="Degradation_Percent", hue="Model", 
                   palette=PROFESSIONAL_COLORS, ax=ax)
        
        ax.set_title(f'{dataset} - Performance Degradation (%)', 
                    fontsize=14 if display_size else 18, fontweight='bold')
        ax.set_ylabel('Performance Degradation (%)', fontsize=12 if display_size else 14)
        ax.set_xlabel('Anonymization Technique', fontsize=12 if display_size else 14)
        
        # HORIZONTAL labels
        plt.xticks(rotation=0, ha='center', fontsize=10 if display_size else 12)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        
        return fig

    def plot_model_impact_analysis(self, results, dataset, save_path=None, display_size=True):
        """Analyze how different models are affected by anonymization."""
        data = []
        for anon_technique, models in results[dataset].items():
            for model, metrics in models.items():
                avg_performance = np.mean(list(metrics.values()))
                data.append({
                    "Technique": anon_technique,
                    "Model": model,
                    "Average_Performance": avg_performance
                })
        
        df = pd.DataFrame(data)
        
        figsize = (10, 6) if display_size else (20, 12)
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.boxplot(data=df, x="Model", y="Average_Performance", ax=ax, palette=PROFESSIONAL_COLORS)
        
        ax.set_title(f'{dataset} - Model Robustness Analysis\nAverage Performance Across All Anonymization Techniques', 
                    fontsize=14 if display_size else 18, fontweight='bold')
        ax.set_ylabel('Average Performance', fontsize=12 if display_size else 14)
        ax.set_xlabel('ML Model', fontsize=12 if display_size else 14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        
        return fig

    def plot_anonymization_technique_ranking(self, results, dataset, save_path=None, display_size=True):
        """Rank anonymization techniques by average performance preservation."""
        technique_scores = {}
        
        for anon_technique, models in results[dataset].items():
            if anon_technique == "Original":
                continue
                
            scores = []
            for model, metrics in models.items():
                scores.extend(list(metrics.values()))
            
            technique_scores[anon_technique] = np.mean(scores)
        
        # Sort by score
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        
        techniques, scores = zip(*sorted_techniques)
        
        figsize = (10, 6) if display_size else (20, 12)
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(techniques, scores, color=PROFESSIONAL_COLORS[0], alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{dataset} - Anonymization Technique Ranking\n(By Average Performance Preservation)', 
                    fontsize=14 if display_size else 18, fontweight='bold')
        ax.set_ylabel('Average Performance Score', fontsize=12 if display_size else 14)
        ax.set_xlabel('Anonymization Technique', fontsize=12 if display_size else 14)
        
        # HORIZONTAL labels
        plt.xticks(rotation=0, ha='center', fontsize=10 if display_size else 12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        
        return fig

    def plot_privacy_utility_tradeoff(self, results, dataset, save_path=None, display_size=True):
        """Create privacy vs utility trade-off analysis."""
        # This is a conceptual plot - in reality, privacy levels would need to be quantified
        privacy_levels = {
            "Original": 0,
            "Micro Aggregation (Minimal)": 0.3,
            "Micro Aggregation (Medium)": 0.5,
            "Micro Aggregation (High)": 0.7,
            "Differential Privacy (Minimal)": 0.4,
            "Differential Privacy (Medium)": 0.6,
            "Differential Privacy (High)": 0.8,
            "Randomized Response (Minimal)": 0.3,
            "Randomized Response (Medium)": 0.5,
            "Randomized Response (High)": 0.7
        }
        
        tradeoff_data = []
        for anon_technique, models in results[dataset].items():
            privacy_level = privacy_levels.get(anon_technique, 0.5)
            
            for model, metrics in models.items():
                avg_utility = np.mean(list(metrics.values()))
                tradeoff_data.append({
                    "Technique": anon_technique,
                    "Model": model,
                    "Privacy_Level": privacy_level,
                    "Utility": avg_utility
                })
        
        df = pd.DataFrame(tradeoff_data)
        
        figsize = (8, 6) if display_size else (16, 12)
        fig, ax = plt.subplots(figsize=figsize)
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            ax.scatter(model_data['Privacy_Level'], model_data['Utility'], 
                      label=model, s=100 if display_size else 200, alpha=0.7,
                      color=MODEL_COLORS.get(model, '#007ACC'))
        
        ax.set_title(f'{dataset} - Privacy vs Utility Trade-off Analysis', 
                    fontsize=14 if display_size else 18, fontweight='bold')
        ax.set_xlabel('Privacy Level (Higher = More Private)', fontsize=12 if display_size else 14)
        ax.set_ylabel('Utility (Performance)', fontsize=12 if display_size else 14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        
        return fig

    def create_comprehensive_dashboard(self, results: Dict, dataset: str, selected_metrics: List[str]) -> Dict[str, Any]:
        """Create all visualizations and return organized results."""
        visualization_results = {}
        
        try:
            # Individual metric performance charts
            visualization_results['metric_charts'] = {}
            for metric in selected_metrics:
                try:
                    # Display version (half size)
                    fig_display = self.plot_summary_bar(results, dataset, metric, display_size=True)
                    
                    # Download version (full size)  
                    download_path = os.path.join(self.temp_dir, f"{metric}_performance_chart.png")
                    fig_download = self.plot_summary_bar(results, dataset, metric, save_path=download_path, display_size=False)
                    
                    visualization_results['metric_charts'][metric] = {
                        'display_fig': fig_display,
                        'download_path': download_path
                    }
                    
                    if fig_display:
                        plt.close(fig_display)
                    if fig_download:
                        plt.close(fig_download)
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not create chart for metric {metric}: {e}")
            
            # Anonymization Impact Heatmap
            try:
                fig_display = self.plot_anonymization_impact_heatmap(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "anonymization_impact_heatmap.png")
                fig_download = self.plot_anonymization_impact_heatmap(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['heatmap'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create heatmap: {e}")
            
            # Performance Degradation Analysis
            try:
                fig_display = self.plot_performance_degradation(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "performance_degradation.png")
                fig_download = self.plot_performance_degradation(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['degradation'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create degradation analysis: {e}")
            
            # Percentage Degradation
            try:
                fig_display = self.plot_percentage_degradation(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "percentage_degradation.png")
                fig_download = self.plot_percentage_degradation(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['percentage_degradation'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create percentage degradation: {e}")
            
            # Model Impact Analysis
            try:
                fig_display = self.plot_model_impact_analysis(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "model_impact_analysis.png")
                fig_download = self.plot_model_impact_analysis(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['model_impact'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create model impact analysis: {e}")
            
            # Anonymization Technique Ranking
            try:
                fig_display = self.plot_anonymization_technique_ranking(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "technique_ranking.png")
                fig_download = self.plot_anonymization_technique_ranking(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['technique_ranking'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create technique ranking: {e}")
            
            # Privacy vs Utility Trade-off
            try:
                fig_display = self.plot_privacy_utility_tradeoff(results, dataset, display_size=True)
                download_path = os.path.join(self.temp_dir, "privacy_utility_tradeoff.png")
                fig_download = self.plot_privacy_utility_tradeoff(results, dataset, save_path=download_path, display_size=False)
                
                visualization_results['privacy_utility'] = {
                    'display_fig': fig_display,
                    'download_path': download_path
                }
                
                if fig_display:
                    plt.close(fig_display)
                if fig_download:
                    plt.close(fig_download)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not create privacy-utility analysis: {e}")
                
        except Exception as e:
            st.error(f"❌ Error creating comprehensive dashboard: {e}")
        
        return visualization_results

    def create_download_package(self, visualization_results: Dict) -> str:
        """Create ZIP package with all high-resolution visualizations."""
        try:
            zip_path = os.path.join(self.temp_dir, f"visualization_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for category, content in visualization_results.items():
                    if isinstance(content, dict):
                        if 'download_path' in content and os.path.exists(content['download_path']):
                            zipf.write(content['download_path'], os.path.basename(content['download_path']))
                        elif isinstance(content, dict):
                            # Handle nested dictionaries (like metric_charts)
                            for subcategory, subcontent in content.items():
                                if isinstance(subcontent, dict) and 'download_path' in subcontent:
                                    if os.path.exists(subcontent['download_path']):
                                        zipf.write(subcontent['download_path'], os.path.basename(subcontent['download_path']))
            
            return zip_path
            
        except Exception as e:
            st.error(f"❌ Error creating download package: {e}")
            return None

    def display_tab_based_visualizations(self, visualization_results: Dict):
        """Display all visualizations in professional tab-based interface."""
        
        # Create tabs for different visualization categories
        tab_names = [
            "📊 Metric Performance", 
            "🔥 Impact Heatmap", 
            "📉 Performance Degradation",
            "📈 Percentage Analysis",
            "🤖 Model Impact", 
            "🏆 Technique Ranking", 
            "⚖️ Privacy vs Utility"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Tab 1: Individual Metric Performance Charts
        with tabs[0]:
            st.markdown("### 📊 Individual Metric Performance Analysis")
            st.markdown("Performance comparison across anonymization techniques for each metric.")
            
            if 'metric_charts' in visualization_results:
                for metric, chart_data in visualization_results['metric_charts'].items():
                    if 'display_fig' in chart_data and chart_data['display_fig']:
                        st.markdown(f"#### {metric.replace('_', ' ').title()}")
                        st.pyplot(chart_data['display_fig'])
                        
                        # Download button for individual chart
                        if 'download_path' in chart_data and os.path.exists(chart_data['download_path']):
                            with open(chart_data['download_path'], 'rb') as f:
                                st.download_button(
                                    f"📥 Download {metric} Chart (High-Res)",
                                    f.read(),
                                    file_name=f"{metric}_performance_chart.png",
                                    mime="image/png"
                                )
            else:
                st.info("No metric charts available.")
        
        # Tab 2: Impact Heatmap
        with tabs[1]:
            st.markdown("### 🔥 Anonymization Impact Heatmap")
            st.markdown("Comprehensive view of how anonymization affects all models and metrics.")
            
            if 'heatmap' in visualization_results and 'display_fig' in visualization_results['heatmap']:
                st.pyplot(visualization_results['heatmap']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['heatmap'] and os.path.exists(visualization_results['heatmap']['download_path']):
                    with open(visualization_results['heatmap']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Heatmap (High-Res)",
                            f.read(),
                            file_name="anonymization_impact_heatmap.png",
                            mime="image/png"
                        )
            else:
                st.info("Heatmap not available.")
        
        # Tab 3: Performance Degradation
        with tabs[2]:
            st.markdown("### 📉 Performance Degradation Analysis")
            st.markdown("Track how performance changes from original to anonymized data across techniques.")
            
            if 'degradation' in visualization_results and 'display_fig' in visualization_results['degradation']:
                st.pyplot(visualization_results['degradation']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['degradation'] and os.path.exists(visualization_results['degradation']['download_path']):
                    with open(visualization_results['degradation']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Degradation Analysis (High-Res)",
                            f.read(),
                            file_name="performance_degradation.png",
                            mime="image/png"
                        )
            else:
                st.info("Degradation analysis not available.")
        
        # Tab 4: Percentage Analysis  
        with tabs[3]:
            st.markdown("### 📈 Percentage Degradation Analysis")
            st.markdown("Quantify performance loss as percentages relative to original performance.")
            
            if 'percentage_degradation' in visualization_results and 'display_fig' in visualization_results['percentage_degradation']:
                st.pyplot(visualization_results['percentage_degradation']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['percentage_degradation'] and os.path.exists(visualization_results['percentage_degradation']['download_path']):
                    with open(visualization_results['percentage_degradation']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Percentage Analysis (High-Res)",
                            f.read(),
                            file_name="percentage_degradation.png",
                            mime="image/png"
                        )
            else:
                st.info("Percentage degradation analysis not available.")
        
        # Tab 5: Model Impact
        with tabs[4]:
            st.markdown("### 🤖 Model Robustness Analysis")
            st.markdown("Compare how different ML models handle anonymization across all techniques.")
            
            if 'model_impact' in visualization_results and 'display_fig' in visualization_results['model_impact']:
                st.pyplot(visualization_results['model_impact']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['model_impact'] and os.path.exists(visualization_results['model_impact']['download_path']):
                    with open(visualization_results['model_impact']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Model Impact Analysis (High-Res)",
                            f.read(),
                            file_name="model_impact_analysis.png",
                            mime="image/png"
                        )
            else:
                st.info("Model impact analysis not available.")
        
        # Tab 6: Technique Ranking
        with tabs[5]:
            st.markdown("### 🏆 Anonymization Technique Ranking")
            st.markdown("Rank techniques by average performance preservation across all models and metrics.")
            
            if 'technique_ranking' in visualization_results and 'display_fig' in visualization_results['technique_ranking']:
                st.pyplot(visualization_results['technique_ranking']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['technique_ranking'] and os.path.exists(visualization_results['technique_ranking']['download_path']):
                    with open(visualization_results['technique_ranking']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Technique Ranking (High-Res)",
                            f.read(),
                            file_name="technique_ranking.png",
                            mime="image/png"
                        )
            else:
                st.info("Technique ranking not available.")
        
        # Tab 7: Privacy vs Utility
        with tabs[6]:
            st.markdown("### ⚖️ Privacy vs Utility Trade-off Analysis")
            st.markdown("Visualize the fundamental trade-off between privacy protection and model utility.")
            
            if 'privacy_utility' in visualization_results and 'display_fig' in visualization_results['privacy_utility']:
                st.pyplot(visualization_results['privacy_utility']['display_fig'])
                
                # Download button
                if 'download_path' in visualization_results['privacy_utility'] and os.path.exists(visualization_results['privacy_utility']['download_path']):
                    with open(visualization_results['privacy_utility']['download_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Privacy-Utility Analysis (High-Res)",
                            f.read(),
                            file_name="privacy_utility_tradeoff.png",
                            mime="image/png"
                        )
            else:
                st.info("Privacy-utility analysis not available.")

# Global instance for easy access
complete_viz_system = CompleteVisualizationSystem()
