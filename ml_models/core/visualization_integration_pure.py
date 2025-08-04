"""
Pure Visualization_Summary Integration - EXACT Original Functions
================================================================

This module uses the EXACT same functions from visualization_summary.py
with only input management changed to work with table data.

NO logic changes, just data feeding to original functions.
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import zipfile
import re

# ===== COPY EXACT STYLING FROM ORIGINAL =====
plt.style.use('default')
sns.set_palette("husl")

from matplotlib import rcParams

# Try Porsche Next TT first, then fallbacks
try:
    rcParams['font.family'] = 'Porsche Next TT'
except:
    try:
        rcParams['font.family'] = 'Arial'
    except:
        rcParams['font.family'] = 'DejaVu Sans'

# EXACT same settings from visualization_summary.py
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

# EXACT same colors from original
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

# ===== EXACT COPY of ALL FUNCTIONS from visualization_summary.py =====

def plot_summary_bar(results, dataset, metric, save_path=None):
    """
    Create a grouped bar chart showing how anonymization affects model performance for a given dataset and metric.
    """
    # Prepare data for plotting
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
        print(f"No data found for dataset '{dataset}' and metric '{metric}'.")
        return
    
    df = pd.DataFrame(data)
    
    # Create professional figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Professional bar plot with improved styling matching ml_performance_tester
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
    
    # Professional styling matching ml_performance_tester standards
    ax.set_title(f'{dataset} Dataset - {metric.replace("_", " ").title()} Performance', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='semibold')
    ax.set_xlabel('Anonymization Technique', fontsize=12, fontweight='semibold')
    
    # Rotate x-axis labels for better readability - HORIZONTAL as requested
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.yticks(fontsize=11)
    
    # Professional legend styling
    legend = ax.legend(title='ML Algorithm', fontsize=10, title_fontsize=11, 
                      loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Add value labels ONLY on the tallest bars to avoid clutter
    # Get the maximum value for each x position
    x_positions = {}
    for i, container in enumerate(bar_plot.containers):
        for j, bar in enumerate(container):
            x_pos = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            if x_pos not in x_positions or height > x_positions[x_pos]['height']:
                x_positions[x_pos] = {'height': height, 'container_idx': i, 'bar_idx': j}
    
    # Only add labels to the tallest bars
    for container_idx, container in enumerate(bar_plot.containers):
        for bar_idx, bar in enumerate(container):
            x_pos = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            if x_positions[x_pos]['container_idx'] == container_idx and x_positions[x_pos]['bar_idx'] == bar_idx:
                ax.text(x_pos, height + 0.01, f'{height:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='semibold')
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, max(df[metric]) * 1.1)
    
    # Professional grid styling matching ml_performance_tester
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved professional chart to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_comprehensive_metrics_overview(results, dataset, metrics_list, save_path=None):
    """
    Create a comprehensive view showing all metrics in one chart with different colors/patterns.
    """
    # Prepare data for comprehensive plotting
    comprehensive_data = []
    
    for anonymization, models in results.get(dataset, {}).items():
        for model, metrics in models.items():
            for metric in metrics_list:
                value = metrics.get(metric, None)
                if value is not None:
                    comprehensive_data.append({
                        "Anonymization": anonymization,
                        "Model": model,
                        "Metric": metric,
                        "Value": value
                    })
    
    if not comprehensive_data:
        print(f"No comprehensive data found for dataset '{dataset}'.")
        return
    
    df = pd.DataFrame(comprehensive_data)
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create a comprehensive bar plot with metrics as hue
    bar_plot = sns.barplot(
        data=df,
        x="Anonymization",
        y="Value",
        hue="Metric",
        ax=ax,
        palette="Set2",  # Use a palette that distinguishes metrics well
        edgecolor='white',
        linewidth=1.0,
        alpha=0.8
    )
    
    # Professional styling
    ax.set_title(f'{dataset} Dataset - Comprehensive Metrics Overview\nAll Metrics Performance Across Anonymization Techniques', 
                fontsize=18, fontweight='bold', pad=30)
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='semibold')
    ax.set_xlabel('Anonymization Technique', fontsize=14, fontweight='semibold')
    
    # ANGLED x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Professional legend styling
    legend = ax.legend(title='Performance Metrics', fontsize=11, title_fontsize=13, 
                      loc='upper right', frameon=True, fancybox=True, shadow=True,
                      bbox_to_anchor=(1.02, 1))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Professional grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved comprehensive metrics overview to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_impact_heatmap(results, dataset, save_path=None):
    """
    Create a heatmap showing the impact of different anonymization techniques across all models and metrics.
    """
    # Prepare data for heatmap
    anonymization_techniques = list(results[dataset].keys())
    models = list(results[dataset][anonymization_techniques[0]].keys())
    metrics = list(results[dataset][anonymization_techniques[0]][models[0]].keys())
    
    # Create matrix for heatmap
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
    
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                     cbar_kws={'label': 'Performance Score'}, 
                     linewidths=0.5, linecolor='white')
    ax.set_title(f"{dataset} Dataset - Performance Heatmap\nAnonymization Impact Across Models and Metrics", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Metrics", fontsize=14)
    ax.set_ylabel("Anonymization Technique & Model", fontsize=14)
    
    # Make y-axis labels horizontal for better readability
    plt.yticks(rotation=0, ha='right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_performance_degradation(results, dataset, save_path=None):
    """
    Create a professional line plot showing performance degradation from original to anonymized versions.
    """
    # Get original performance as baseline
    original_data = results[dataset]["Original"]
    
    # Prepare data for line plot
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
    
    # Get available metrics (handle both 4-metric and 7-metric datasets)
    available_metrics = list(original_data[list(original_data.keys())[0]].keys())
    
    # Determine subplot layout based on number of metrics
    if len(available_metrics) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
    elif len(available_metrics) == 7:
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
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
                            fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Anonymization Level', fontsize=12)
            axes[i].set_ylabel('Performance', fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # ANGLED labels for better readability (no more crossing)
            axes[i].set_xticks(range(len(anonymization_order)))
            axes[i].set_xticklabels([a.split('(')[0].strip() for a in anonymization_order], 
                                   rotation=45, ha='right', fontsize=10)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{dataset} - Performance Degradation Analysis', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved degradation analysis to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_comparison(results, dataset, save_path=None):
    """
    Create a comprehensive comparison of anonymization techniques across all metrics.
    """
    # Calculate average performance for each anonymization technique
    technique_performance = {}
    
    for anon_technique, models in results[dataset].items():
        all_scores = []
        for model, metrics in models.items():
            all_scores.extend(list(metrics.values()))
        technique_performance[anon_technique] = np.mean(all_scores)
    
    # Sort by performance
    sorted_techniques = sorted(technique_performance.items(), key=lambda x: x[1], reverse=True)
    
    techniques, scores = zip(*sorted_techniques)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(techniques, scores, color=[TECHNIQUE_COLORS.get(t, '#007ACC') for t in techniques], alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'{dataset} - Anonymization Technique Comparison\n(Average Performance Across All Models & Metrics)', 
              fontsize=16, fontweight='bold')
    plt.ylabel('Average Performance Score', fontsize=14)
    plt.xlabel('Anonymization Technique', fontsize=14)
    
    # HORIZONTAL labels
    plt.xticks(rotation=0, ha='center', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_percentage_degradation(results, dataset, save_path=None):
    """
    Create a detailed percentage degradation analysis.
    """
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
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create grouped bar plot
    sns.barplot(data=df, x="Technique", y="Degradation_Percent", hue="Model", 
               palette=PROFESSIONAL_COLORS, ax=ax)
    
    ax.set_title(f'{dataset} - Performance Degradation (%)', 
                fontsize=18, fontweight='bold')
    ax.set_ylabel('Performance Degradation (%)', fontsize=14)
    ax.set_xlabel('Anonymization Technique', fontsize=14)
    
    # HORIZONTAL labels
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    # Add horizontal line at 0%
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved percentage degradation to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_model_impact_analysis(results, dataset, save_path=None):
    """
    Analyze how different models are affected by anonymization.
    """
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
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    sns.boxplot(data=df, x="Model", y="Average_Performance", ax=ax, palette=PROFESSIONAL_COLORS)
    
    ax.set_title(f'{dataset} - Model Robustness Analysis\nAverage Performance Across All Anonymization Techniques', 
                fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Performance', fontsize=14)
    ax.set_xlabel('ML Model', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved model impact analysis to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_technique_ranking(results, dataset, save_path=None):
    """
    Rank anonymization techniques by average performance preservation.
    """
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
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    bars = ax.bar(techniques, scores, color=PROFESSIONAL_COLORS[0], alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(f'{dataset} - Anonymization Technique Ranking\n(By Average Performance Preservation)', 
                fontsize=18, fontweight='bold')
    ax.set_ylabel('Average Performance Score', fontsize=14)
    ax.set_xlabel('Anonymization Technique', fontsize=14)
    
    # HORIZONTAL labels
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved technique ranking to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_dataset_performance(save_path=None):
    """
    Create combined performance analysis across all datasets.
    This function is from original but would need results data passed in.
    """
    # This function would need to be adapted when results data is available
    pass

def plot_combined_technique_effectiveness(save_path=None):
    """
    Create combined technique effectiveness analysis.
    This function is from original but would need results data passed in.
    """
    # This function would need to be adapted when results data is available
    pass

def plot_combined_model_robustness(save_path=None):
    """
    Create combined model robustness analysis.
    This function is from original but would need results data passed in.
    """
    # This function would need to be adapted when results data is available
    pass

# ===== INPUT MANAGEMENT CLASS =====

class PureVisualizationIntegrator:
    """Uses EXACT original visualization_summary.py functions with only input management."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def clean_metric_name(self, metric_name: str) -> str:
        """Remove icons and clean metric names for display."""
        # Remove ALL possible icons, emojis, and symbols
        cleaned = re.sub(r'[📊🎯📈📉💯🔥⚡️✨🌟💪🏆🎭🎨🎪🎯📏📐📊📈📉📋📌📍📎📝💼💻🖥️⌨️🖱️🖨️📱📞📠📧📨📩📪📫📬📭📮🗳️✉️📪📬🗂️📂📁🗄️🗃️📋📊📈📉📊🎯💯]', '', metric_name)
        
        # Remove additional symbols that might appear
        cleaned = re.sub(r'[★☆✓✗✕✖✔⚡⭐🌟💫⭐️🌠🎈🎆🎇✨🎉🎊🎁🎀🎗️🏷️🎫🎟️]', '', cleaned)
        
        # Remove any remaining non-alphanumeric characters except spaces, hyphens, underscores
        cleaned = re.sub(r'[^\w\s\-_]', '', cleaned)
        
        # Clean up extra spaces and make it neat
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # If the name becomes empty or very short, try to extract meaningful parts
        if len(cleaned) < 3:
            # Try to extract meaningful words from original
            words = re.findall(r'[a-zA-Z]+', metric_name)
            if words:
                cleaned = ' '.join(words)
        
        # Final fallback - use original if cleaning failed
        return cleaned if cleaned else metric_name
    
    def standardize_metric_name(self, metric: str) -> str:
        """Standardize metric names to match visualization_summary.py format."""
        # First clean the name thoroughly
        cleaned = self.clean_metric_name(metric)
        
        # Standard mapping with more comprehensive coverage
        metric_mapping = {
            'accuracy': 'accuracy',
            'precision': 'precision', 
            'recall': 'recall',
            'f1_score': 'f1_score',
            'f1-score': 'f1_score',
            'f1 score': 'f1_score',
            'f1score': 'f1_score',
            'balanced_accuracy': 'balanced_accuracy',
            'balanced accuracy': 'balanced_accuracy',
            'balancedaccuracy': 'balanced_accuracy',
            'matthews_correlation': 'matthews_correlation',
            'matthews correlation': 'matthews_correlation',
            'matthewscorrelation': 'matthews_correlation',
            'mcc': 'matthews_correlation',
            'roc_auc': 'roc_auc',
            'roc auc': 'roc_auc',
            'rocauc': 'roc_auc',
            'auc': 'roc_auc'
        }
        
        # Clean and normalize
        metric_clean = cleaned.lower().replace(' ', '_').replace('-', '_').replace('__', '_')
        
        # Try exact match first
        if metric_clean in metric_mapping:
            return metric_mapping[metric_clean]
        
        # Try partial matches
        for key, value in metric_mapping.items():
            if key in metric_clean or metric_clean in key:
                return value
        
        # Return cleaned version if no mapping found
        return metric_clean
    
    def create_anonymization_label(self, privacy_method: str, anonymization_configs: Dict[str, str]) -> str:
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
    
    def extract_table_data_to_results_format(self, df: pd.DataFrame, selected_metrics: List[str], 
                                            anonymization_configs: Dict[str, str], dataset_name: str) -> Dict[str, Any]:
        """
        Extract table data and create results format for EXACT original functions.
        
        Args:
            df: DataFrame with table data
            selected_metrics: List of selected metric columns
            anonymization_configs: Dict mapping privacy methods to levels
            dataset_name: User-provided dataset name
        
        Returns:
            Dictionary in EXACT format expected by visualization_summary.py functions
        """
        try:
            results = {}
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
                    anonymization_label = self.create_anonymization_label(privacy_method, anonymization_configs)
                    
                    # Initialize nested structure
                    if anonymization_label not in results[dataset_name]:
                        results[dataset_name][anonymization_label] = {}
                    if model_name not in results[dataset_name][anonymization_label]:
                        results[dataset_name][anonymization_label][model_name] = {}
                    
                    # Extract metric values with cleaned names
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
                            
                            # Use cleaned and standardized metric name
                            standard_metric = self.standardize_metric_name(metric)
                            results[dataset_name][anonymization_label][model_name][standard_metric] = value
                
                except Exception as e:
                    st.warning(f"⚠️ Error processing row: {e}")
                    continue
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error extracting table data: {e}")
            return {}
    
    def generate_all_visualizations(self, results: Dict, dataset_name: str, selected_metrics: List[str]) -> Dict[str, str]:
        """Generate all visualizations using EXACT original functions."""
        
        visualization_files = {}
        
        try:
            # Individual metric charts using EXACT original function
            for metric in selected_metrics:
                cleaned_metric = self.standardize_metric_name(metric)
                save_path = os.path.join(self.temp_dir, f"{cleaned_metric}_performance_chart.png")
                
                try:
                    plot_summary_bar(results, dataset_name, cleaned_metric, save_path)
                    if os.path.exists(save_path):
                        visualization_files[f"metric_{cleaned_metric}"] = save_path
                except Exception as e:
                    st.warning(f"⚠️ Could not create chart for {metric}: {e}")
            
            # Comprehensive Metrics Overview using new function
            try:
                cleaned_metrics = [self.standardize_metric_name(m) for m in selected_metrics]
                save_path = os.path.join(self.temp_dir, "comprehensive_metrics_overview.png")
                plot_comprehensive_metrics_overview(results, dataset_name, cleaned_metrics, save_path)
                if os.path.exists(save_path):
                    visualization_files["comprehensive_overview"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create comprehensive overview: {e}")
            
            # Anonymization Impact Heatmap using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "anonymization_impact_heatmap.png")
                plot_anonymization_impact_heatmap(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["heatmap"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create heatmap: {e}")
            
            # Performance Degradation using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "performance_degradation.png")
                plot_performance_degradation(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["degradation"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create degradation analysis: {e}")
            
            # Anonymization Comparison using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "anonymization_comparison.png")
                plot_anonymization_comparison(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["comparison"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create comparison: {e}")
            
            # Percentage Degradation using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "percentage_degradation.png")
                plot_percentage_degradation(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["percentage_degradation"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create percentage degradation: {e}")
            
            # Model Impact Analysis using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "model_impact_analysis.png")
                plot_model_impact_analysis(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["model_impact"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create model impact analysis: {e}")
            
            # Technique Ranking using EXACT original function
            try:
                save_path = os.path.join(self.temp_dir, "technique_ranking.png")
                plot_anonymization_technique_ranking(results, dataset_name, save_path)
                if os.path.exists(save_path):
                    visualization_files["technique_ranking"] = save_path
            except Exception as e:
                st.warning(f"⚠️ Could not create technique ranking: {e}")
                
        except Exception as e:
            st.error(f"❌ Error generating visualizations: {e}")
        
        return visualization_files
    
    def create_download_package(self, visualization_files: Dict[str, str], dataset_name: str) -> str:
        """Create ZIP package with all visualizations."""
        try:
            zip_path = os.path.join(self.temp_dir, f"{dataset_name}_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for viz_type, file_path in visualization_files.items():
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return zip_path
            
        except Exception as e:
            st.error(f"❌ Error creating download package: {e}")
            return None

    def display_visualizations_with_tabs(self, visualization_files: Dict[str, str], selected_metrics: List[str]):
        """Display visualizations in tab-based interface."""
        
        # Create tabs
        tab_names = [
            "📊 Individual Metrics", 
            "🔍 Comprehensive Overview",
            "🔥 Impact Heatmap", 
            "📉 Degradation Analysis",
            "📈 Comparison Analysis",
            "📊 Percentage Analysis",
            "🤖 Model Impact", 
            "🏆 Technique Ranking"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Tab 1: Individual Metric Charts
        with tabs[0]:
            st.markdown("### 📊 Individual Metric Performance Analysis")
            st.markdown("Performance comparison across anonymization techniques for each metric separately.")
            
            for metric in selected_metrics:
                cleaned_metric = self.standardize_metric_name(metric)
                file_key = f"metric_{cleaned_metric}"
                
                if file_key in visualization_files and os.path.exists(visualization_files[file_key]):
                    st.markdown(f"#### {self.clean_metric_name(metric)}")
                    
                    # Display image
                    st.image(visualization_files[file_key], use_container_width=True)
                    
                    # Download button
                    with open(visualization_files[file_key], 'rb') as f:
                        st.download_button(
                            f"📥 Download {self.clean_metric_name(metric)} Chart",
                            f.read(),
                            file_name=f"{cleaned_metric}_performance_chart.png",
                            mime="image/png"
                        )
        
        # Tab 2: Comprehensive Overview
        with tabs[1]:
            st.markdown("### 🔍 Comprehensive Metrics Overview")
            st.markdown("All metrics combined in one comprehensive view with color-coded legend for easy comparison.")
            
            if "comprehensive_overview" in visualization_files and os.path.exists(visualization_files["comprehensive_overview"]):
                st.image(visualization_files["comprehensive_overview"], use_container_width=True)
                
                with open(visualization_files["comprehensive_overview"], 'rb') as f:
                    st.download_button(
                        "📥 Download Comprehensive Overview",
                        f.read(),
                        file_name="comprehensive_metrics_overview.png",
                        mime="image/png"
                    )
            else:
                st.info("Comprehensive overview not available.")
        
        # Tab 3: Impact Heatmap
        with tabs[2]:
            st.markdown("### 🔥 Anonymization Impact Heatmap")
            st.markdown("Comprehensive view of how anonymization affects all models and metrics.")
            
            if "heatmap" in visualization_files and os.path.exists(visualization_files["heatmap"]):
                st.image(visualization_files["heatmap"], use_container_width=True)
                
                with open(visualization_files["heatmap"], 'rb') as f:
                    st.download_button(
                        "📥 Download Heatmap",
                        f.read(),
                        file_name="anonymization_impact_heatmap.png",
                        mime="image/png"
                    )
            else:
                st.info("Heatmap not available.")
        
        # Tab 4: Degradation Analysis
        with tabs[3]:
            st.markdown("### 📉 Performance Degradation Analysis")
            st.markdown("Track how performance changes from original to anonymized data.")
            
            if "degradation" in visualization_files and os.path.exists(visualization_files["degradation"]):
                st.image(visualization_files["degradation"], use_container_width=True)
                
                with open(visualization_files["degradation"], 'rb') as f:
                    st.download_button(
                        "📥 Download Degradation Analysis",
                        f.read(),
                        file_name="performance_degradation.png",
                        mime="image/png"
                    )
            else:
                st.info("Degradation analysis not available.")
        
        # Tab 5: Comparison Analysis
        with tabs[4]:
            st.markdown("### 📈 Anonymization Technique Comparison")
            st.markdown("Compare effectiveness of different anonymization techniques.")
            
            if "comparison" in visualization_files and os.path.exists(visualization_files["comparison"]):
                st.image(visualization_files["comparison"], use_container_width=True)
                
                with open(visualization_files["comparison"], 'rb') as f:
                    st.download_button(
                        "📥 Download Comparison Analysis",
                        f.read(),
                        file_name="anonymization_comparison.png",
                        mime="image/png"
                    )
            else:
                st.info("Comparison analysis not available.")
        
        # Tab 6: Percentage Analysis
        with tabs[5]:
            st.markdown("### 📊 Percentage Degradation Analysis")
            st.markdown("Quantify performance loss as percentages.")
            
            if "percentage_degradation" in visualization_files and os.path.exists(visualization_files["percentage_degradation"]):
                st.image(visualization_files["percentage_degradation"], use_container_width=True)
                
                with open(visualization_files["percentage_degradation"], 'rb') as f:
                    st.download_button(
                        "📥 Download Percentage Analysis",
                        f.read(),
                        file_name="percentage_degradation.png",
                        mime="image/png"
                    )
            else:
                st.info("Percentage analysis not available.")
        
        # Tab 7: Model Impact
        with tabs[6]:
            st.markdown("### 🤖 Model Robustness Analysis")
            st.markdown("Compare how different ML models handle anonymization.")
            
            if "model_impact" in visualization_files and os.path.exists(visualization_files["model_impact"]):
                st.image(visualization_files["model_impact"], use_container_width=True)
                
                with open(visualization_files["model_impact"], 'rb') as f:
                    st.download_button(
                        "📥 Download Model Impact Analysis",
                        f.read(),
                        file_name="model_impact_analysis.png",
                        mime="image/png"
                    )
            else:
                st.info("Model impact analysis not available.")
        
        # Tab 8: Technique Ranking
        with tabs[7]:
            st.markdown("### 🏆 Anonymization Technique Ranking")
            st.markdown("Rank techniques by performance preservation.")
            
            if "technique_ranking" in visualization_files and os.path.exists(visualization_files["technique_ranking"]):
                st.image(visualization_files["technique_ranking"], use_container_width=True)
                
                with open(visualization_files["technique_ranking"], 'rb') as f:
                    st.download_button(
                        "📥 Download Technique Ranking",
                        f.read(),
                        file_name="technique_ranking.png",
                        mime="image/png"
                    )
            else:
                st.info("Technique ranking not available.")

# Global instance
pure_viz_integrator = PureVisualizationIntegrator()
