"""
Script to generate degradation plots for thesis analysis.
Runs grouped degradation analysis without full application interface.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Configure matplotlib for publication-quality output results
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_grouped_plot(data, metric_list, title, output_dir):
    """Create subplot for specified metrics group"""
    available_metrics = [m for m in metric_list if m in data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]]]
    
    if not available_metrics:
        print(f"⚠️ No metrics available for {title}")
        return
    
    # Calculate subplot dimensions
    n_metrics = len(available_metrics)
    cols = min(2, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7*rows))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Extract metric data
        techniques = list(data.keys())
        x_labels = []
        y_values = {alg: [] for alg in data[techniques[0]].keys()}
        
        for technique in techniques:
            x_labels.append(technique.replace('_', ' ').title())
            for algorithm, metrics in data[technique].items():
                y_values[algorithm].append(metrics.get(metric, 0))
        
        # Generate bar plot
        x_pos = np.arange(len(x_labels))
        width = 0.25
        
        for i, (algorithm, values) in enumerate(y_values.items()):
            ax.bar(x_pos + i * width, values, width, 
                  label=algorithm, color=colors[i], alpha=0.8)
        
        # Format subplot
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=20, pad=15)
        ax.set_xlabel('Privacy Technique', fontsize=18)
        ax.set_ylabel('Score', fontsize=18)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=len(labels), fontsize=16, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.lower().replace(' ', '_').replace('-', '_')
    filename = f"degradation_analysis_{safe_title}_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved {title} plot to {save_path}")
    plt.close()

def run_degradation_analysis():
    """Execute degradation analysis with grouped plots"""
    
    # Sample data for demonstration
    sample_data = {
        "Original": {
            "Logistic Regression": {"accuracy": 0.9767, "precision": 0.9769, "recall": 0.9767, "f1_score": 0.9767, "balanced_accuracy": 0.9767, "matthews_correlation": 0.9651, "roc_auc": 0.9992},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Laplace_Noise": {
            "Logistic Regression": {"accuracy": 0.2867, "precision": 0.2803, "recall": 0.2867, "f1_score": 0.2801, "balanced_accuracy": 0.2867, "matthews_correlation": -0.0709, "roc_auc": 0.484},
            "Random Forest": {"accuracy": 0.3467, "precision": 0.3472, "recall": 0.3467, "f1_score": 0.3468, "balanced_accuracy": 0.3467, "matthews_correlation": 0.02, "roc_auc": 0.5146},
            "XGBoost": {"accuracy": 0.35, "precision": 0.3502, "recall": 0.35, "f1_score": 0.35, "balanced_accuracy": 0.35, "matthews_correlation": 0.025, "roc_auc": 0.5103}
        },
        "Gaussian_Noise": {
            "Logistic Regression": {"accuracy": 0.3867, "precision": 0.379, "recall": 0.3867, "f1_score": 0.3806, "balanced_accuracy": 0.3867, "matthews_correlation": 0.0805, "roc_auc": 0.5912},
            "Random Forest": {"accuracy": 0.4167, "precision": 0.4158, "recall": 0.4167, "f1_score": 0.4161, "balanced_accuracy": 0.4167, "matthews_correlation": 0.125, "roc_auc": 0.5988},
            "XGBoost": {"accuracy": 0.4367, "precision": 0.4397, "recall": 0.4367, "f1_score": 0.4373, "balanced_accuracy": 0.4367, "matthews_correlation": 0.1553, "roc_auc": 0.6066}
        },
        "K_Anonymity": {
            "Logistic Regression": {"accuracy": 0.92, "precision": 0.9201, "recall": 0.92, "f1_score": 0.92, "balanced_accuracy": 0.92, "matthews_correlation": 0.8801, "roc_auc": 0.9851},
            "Random Forest": {"accuracy": 0.94, "precision": 0.9404, "recall": 0.94, "f1_score": 0.94, "balanced_accuracy": 0.94, "matthews_correlation": 0.9102, "roc_auc": 0.9893},
            "XGBoost": {"accuracy": 0.9467, "precision": 0.9471, "recall": 0.9467, "f1_score": 0.9466, "balanced_accuracy": 0.9467, "matthews_correlation": 0.9202, "roc_auc": 0.9933}
        },
        "L_Diversity": {
            "Logistic Regression": {"accuracy": 0.4867, "precision": 0.4607, "recall": 0.5599, "f1_score": 0.4454, "balanced_accuracy": 0.5599, "matthews_correlation": 0.2135, "roc_auc": 0.6807},
            "Random Forest": {"accuracy": 0.4833, "precision": 0.4401, "recall": 0.5421, "f1_score": 0.4456, "balanced_accuracy": 0.5421, "matthews_correlation": 0.1674, "roc_auc": 0.6722},
            "XGBoost": {"accuracy": 0.5100, "precision": 0.3847, "recall": 0.3772, "f1_score": 0.3773, "balanced_accuracy": 0.3772, "matthews_correlation": 0.0952, "roc_auc": 0.6665}
        },
        "T_Closeness": {
            "Logistic Regression": {"accuracy": 0.7767, "precision": 0.7773, "recall": 0.7987, "f1_score": 0.7782, "balanced_accuracy": 0.7987, "matthews_correlation": 0.6711, "roc_auc": 0.8595},
            "Random Forest": {"accuracy": 0.7900, "precision": 0.7874, "recall": 0.8105, "f1_score": 0.7917, "balanced_accuracy": 0.8105, "matthews_correlation": 0.6884, "roc_auc": 0.8590},
            "XGBoost": {"accuracy": 0.7900, "precision": 0.7874, "recall": 0.8105, "f1_score": 0.7917, "balanced_accuracy": 0.8105, "matthews_correlation": 0.6884, "roc_auc": 0.8604}
        },
        "SMOTE": {
            "Logistic Regression": {"accuracy": 0.9333, "precision": 0.9334, "recall": 0.9365, "f1_score": 0.9336, "balanced_accuracy": 0.9365, "matthews_correlation": 0.9012, "roc_auc": 0.9583},
            "Random Forest": {"accuracy": 0.9433, "precision": 0.9435, "recall": 0.9464, "f1_score": 0.9439, "balanced_accuracy": 0.9464, "matthews_correlation": 0.916, "roc_auc": 0.9605},
            "XGBoost": {"accuracy": 0.9433, "precision": 0.9435, "recall": 0.9464, "f1_score": 0.9439, "balanced_accuracy": 0.9464, "matthews_correlation": 0.916, "roc_auc": 0.9661}
        }
    }
    
    # Create output directory
    output_dir = "degradation_plots_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metric groups
    core_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    extended_metrics = ['balanced_accuracy', 'matthews_correlation', 'roc_auc']
    
    print("🎯 Generating degradation analysis plots...")
    print("=" * 60)
    
    # Generate grouped plots
    create_grouped_plot(sample_data, core_metrics, "Core Metrics Analysis", output_dir)
    create_grouped_plot(sample_data, extended_metrics, "Extended Metrics Analysis", output_dir)
    
    # Generate comprehensive plot
    all_metrics = core_metrics + extended_metrics
    create_grouped_plot(sample_data, all_metrics, "Comprehensive Metrics Analysis", output_dir)
    
    print("=" * 60)
    print(f"✅ All degradation plots saved to: {os.path.abspath(output_dir)}")
    print("🎓 Plots are optimized for thesis paper with enhanced font sizes")

if __name__ == "__main__":
    print("🚀 Starting Degradation Analysis Plot Generation")
    print("📊 Configured for thesis-quality output with enhanced readability")
    run_degradation_analysis()
    print("✨ Degradation analysis complete!")
