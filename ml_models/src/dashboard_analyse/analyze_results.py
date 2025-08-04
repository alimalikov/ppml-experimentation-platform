# File: src/analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For percentage formatting
import seaborn as sns
import numpy as np
import os
import re
import warnings

# --- Configuration ---
RESULTS_DIR = "../results"
MASTER_SUMMARY_FILE = os.path.join(RESULTS_DIR, "final_master_summary_consolidated.csv")
PLOTS_DIR = "../results/plots_enhanced" # New folder for enhanced plots
os.makedirs(PLOTS_DIR, exist_ok=True)

KNOWN_SYNTHESIZER = "CTGAN" # Adjust if needed

# Metrics to analyze (absolute and relative)
METRICS_TO_ANALYZE = ['F1_>50K', 'ROC_AUC', 'Balanced_Accuracy', 'Accuracy']

# Plotting Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (11, 7) # Default figure size
plt.rcParams['figure.dpi'] = 100 # Adjust DPI for better resolution if needed

# Suppress specific FutureWarnings from Seaborn/Matplotlib if they become noisy
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Load and Prepare Data ---
print(f"--- Loading Consolidated Results from: {MASTER_SUMMARY_FILE} ---")
try:
    df = pd.read_csv(MASTER_SUMMARY_FILE)
    print(f"✅ Loaded data with shape: {df.shape}")
    # Basic check for essential columns
    required_cols = ['Model', 'Privacy_Type', 'Privacy_Params_Final'] + METRICS_TO_ANALYZE
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing essential columns in master summary: {missing}")
except FileNotFoundError:
    print(f"❌ Error: Master summary file not found at {MASTER_SUMMARY_FILE}")
    exit()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print("\n--- Preparing Data for Plotting ---")

# Function to extract numeric k value
def extract_k(param_str):
    if isinstance(param_str, str) and param_str.startswith('k='):
        try: return int(re.search(r'k=(\d+)', param_str).group(1))
        except: return None
    return None

# Function to extract numeric epsilon value
def extract_epsilon(param_str):
    if isinstance(param_str, str) and ('ε≈' in param_str or 'eps=' in param_str):
        try:
            match = re.search(r'[ε|eps]≈?([\d\.]+)', param_str)
            if match: return float(match.group(1))
            else: return None
        except: return None
    return None

# Apply extraction functions
df['k_value'] = df['Privacy_Params_Final'].apply(extract_k)
df['epsilon_value'] = df['Privacy_Params_Final'].apply(extract_epsilon)
# Create inverse epsilon (higher = stricter privacy) for potential plotting
df['inv_epsilon'] = 1 / (df['epsilon_value'] + 1e-9) # Add small value to avoid infinity

# --- Calculate Relative Metrics ---
print("Calculating relative performance metrics...")
df_baseline = df[df['Privacy_Type'] == 'Baseline'].copy()
if df_baseline.empty:
    print("❌ Error: No baseline data found. Cannot calculate relative metrics.")
    exit()

for metric in METRICS_TO_ANALYZE:
    baseline_map = df_baseline.set_index('Model')[metric].to_dict()
    df[f'{metric}_Baseline'] = df['Model'].map(baseline_map)
    # Calculate relative, handle division by zero or NaN baseline
    df[f'{metric}_Relative'] = np.where(
        (df[f'{metric}_Baseline'].notna()) & (df[f'{metric}_Baseline'] != 0),
        (df[metric] / df[f'{metric}_Baseline']) * 100,
        0 # Assign 0 if baseline is NaN or 0
    )
    # Ensure baseline itself is 100%
    df.loc[df['Privacy_Type'] == 'Baseline', f'{metric}_Relative'] = 100.0
    df[f'{metric}_Relative'].fillna(0, inplace=True) # Fill any remaining NaNs


# Identify unique models and privacy types
models = sorted(df['Model'].unique())
privacy_types = sorted(df['Privacy_Type'].unique())

print(f"Unique Models Found: {models}")
print(f"Unique Privacy Types Found: {privacy_types}")
print("✅ Data preparation complete.")

# --- Plotting Functions ---

def sanitize_filename(filename):
    """Removes or replaces characters invalid for Windows filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    sanitized = sanitized.replace('(', '_').replace(')', '').replace(' ', '_').replace('%','pct')
    if len(sanitized) > 180: name, ext = os.path.splitext(sanitized); sanitized = name[:180] + ext
    return sanitized

def save_plot(fig, filename):
    """Helper to save plot to configured directory after sanitizing filename."""
    sanitized_filename = sanitize_filename(filename)
    filepath = os.path.join(PLOTS_DIR, sanitized_filename)
    try:
        fig.savefig(filepath, bbox_inches='tight', dpi=150) # Save the passed figure object
        print(f"   💾 Plot saved: {sanitized_filename}")
    except Exception as e:
        print(f"   ❌ Error saving plot {sanitized_filename}: {e}")
    plt.close(fig) # Close the figure

# --- Updated/New Plotting Functions ---

def plot_baseline_comparison_enhanced(data, metrics, title='Baseline Model Performance Comparison'):
    """Bar chart comparing baseline performance across multiple metrics."""
    print(f"\n📊 Generating Plot: {title}")
    baseline_data = data[data['Privacy_Type'] == 'Baseline'].copy()
    if baseline_data.empty: print("   ⚠️ No baseline data found."); return

    melted_data = baseline_data.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_data, palette='viridis')
    plt.title(title)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(bottom=0)
    save_plot(plt.gcf(), f"enhanced_01_baseline_comparison.png") # Pass current figure

def plot_relative_put_faceted(data, privacy_param_col, privacy_type_filter, relative_metric='F1_>50K_Relative', title_prefix='Relative Privacy-Utility Trade-off', xlabel='Privacy Parameter', higher_is_stricter=True):
    """Faceted line plot showing relative utility vs. privacy parameter per model."""
    print(f"\n📊 Generating Plot: {title_prefix} ({relative_metric} vs {xlabel}) - Faceted")
    plot_data = data[data['Privacy_Type'].isin(privacy_type_filter) & data[privacy_param_col].notna()].copy()
    if plot_data.empty: print(f"   ⚠️ No data for {privacy_type_filter}. Skipping plot."); return

    # Add baseline reference point (100%)
    baseline_ref = data[data['Privacy_Type'] == 'Baseline'][['Model', relative_metric]].copy()
    baseline_ref[privacy_param_col] = 0 # Define baseline position
    plot_data = pd.concat([baseline_ref, plot_data], ignore_index=True)

    plot_data = plot_data.sort_values(by=[privacy_param_col])

    g = sns.relplot(
        data=plot_data,
        x=privacy_param_col,
        y=relative_metric,
        hue='Privacy_Type',
        style='Privacy_Type',
        col='Model',
        col_wrap=3, # Adjust number of columns per row
        kind='line',
        markers=True,
        markersize=7,
        dashes=False,
        palette='tab10',
        height=4, aspect=1.2,
        legend='full'
    )

    g.fig.suptitle(f'{title_prefix}\n({relative_metric.replace("_Relative","")} vs {xlabel})', y=1.03)
    g.set_axis_labels(xlabel + (' (Higher = Stricter Privacy)' if higher_is_stricter else ' (Lower = Stricter Privacy)'), "Relative Performance (%)")
    g.set_titles("Model: {col_name}")
    g.set(ylim=(0, 105)) # Y-axis from 0 to 105%
    if not higher_is_stricter: # Epsilon plots
         for ax in g.axes.flat:
             ax.set_xscale('log') # Use log scale for epsilon
    g.fig.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout

    save_plot(g.fig, f"enhanced_02_relative_put_{relative_metric}_vs_{privacy_param_col}_faceted.png")


def plot_relative_technique_comparison(data, relative_metric='F1_>50K_Relative', param_filters=None, title='Relative Technique Comparison'):
     """Bar chart comparing relative performance of techniques at specific points."""
     print(f"\n📊 Generating Plot: {title} ({relative_metric})")
     if not param_filters: print("   ⚠️ param_filters needed for this plot."); return

     # Include Baseline (which has 100% relative performance)
     plot_data = data[data['Privacy_Params_Final'].isin(param_filters) | (data['Privacy_Type'] == 'Baseline')].copy()
     plot_data['Plot_Label'] = np.where(plot_data['Privacy_Type'] == 'Baseline',
                                        'Baseline (100%)',
                                        plot_data['Privacy_Type'] + ' (' + plot_data['Privacy_Params_Final'] + ')')
     plot_data = plot_data.sort_values(by=['Model', 'Plot_Label'])

     if plot_data.empty: print("   ⚠️ No data found for specified filters."); return

     plt.figure(figsize=(14, 8))
     ax = sns.barplot(data=plot_data, x='Model', y=relative_metric, hue='Plot_Label', palette='muted', dodge=True)
     ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # Format y-axis as percentage
     plt.title(title + f'\n({relative_metric.replace("_Relative","")} Retention vs Baseline)')
     plt.ylabel("Relative Performance (%)")
     plt.xlabel("Model")
     plt.xticks(rotation=45, ha='right')
     plt.legend(title='Privacy Technique & Parameter', bbox_to_anchor=(1.05, 1), loc='upper left')
     plt.ylim(bottom=0, top=110) # Allow showing 100% baseline clearly
     plt.axhline(100, color='grey', linestyle='--', alpha=0.7) # Line at 100%
     save_plot(plt.gcf(), f"enhanced_03_relative_technique_comparison_{relative_metric}.png")


def plot_pareto_scatter(data, privacy_axis_col, y_metric='F1_>50K_Relative', x_label='Privacy Parameter', higher_x_is_stricter=True, title_suffix=''):
    """Scatter plot showing utility vs privacy parameter."""
    print(f"\n📊 Generating Plot: Pareto Scatter {y_metric} vs {x_label}")
    plot_data = data[data[privacy_axis_col].notna()].copy()
    if plot_data.empty: print(f"   ⚠️ No data for privacy axis {privacy_axis_col}. Skipping."); return

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_data, x=privacy_axis_col, y=y_metric, hue='Model', style='Privacy_Type', s=100, palette='deep', alpha=0.8)
    plt.title(f'Utility vs. Privacy Trade-off{title_suffix}\n({y_metric.replace("_Relative","")} vs {x_label})')
    plt.ylabel(f"{y_metric.replace('_Relative','')} (% Relative)" if 'Relative' in y_metric else y_metric)
    plt.xlabel(x_label + (' (Higher = Stricter Privacy)' if higher_x_is_stricter else ' (Lower = Stricter Privacy)'))
    plt.legend(title='Model / Privacy Type', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.ylim(bottom=0)
    if not higher_x_is_stricter: plt.xscale('log') # Log scale for epsilon
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    save_plot(plt.gcf(), f"enhanced_04_pareto_scatter_{y_metric}_vs_{privacy_axis_col}.png")


def plot_performance_heatmap(data, relative_metric='F1_>50K_Relative', title='Relative Performance Heatmap'):
    """Heatmap showing relative performance across models and techniques."""
    print(f"\n📊 Generating Plot: {title} ({relative_metric})")
    # Select representative points for clarity on heatmap columns
    representative_params = ['Baseline', 'k=5', 'k=10', 'k=20', 'Microagg k=5', 'Microagg k=10', 'Microagg k=20',
                             'ε≈1.0, δ=1e-05', 'ε≈5.0, δ=1e-05', 'ε≈10.0, δ=1e-05',
                             'LDP (RR) (ε≈1.0)', 'LDP (RR) (ε≈5.0)', 'LDP (RR) (ε≈10.0)', # Example naming
                             'Simple Rules', 'Medium Rules',
                             f'{KNOWN_SYNTHESIZER}-UnscaledIn', f'{KNOWN_SYNTHESIZER}-ScaledIn']

    # Create combined identifier, adapting for privacy type
    def create_heatmap_label(row):
        ptype = row['Privacy_Type']
        params = row['Privacy_Params_Final']
        if ptype == 'Baseline': return 'Baseline'
        if ptype == 'k-Anon (ARX)' or ptype == 'Microagg': return f"{ptype.split(' ')[0]} {params}" # e.g., k-Anon k=5
        if ptype == 'DP-SGD' or ptype == 'LDP (RR)': return f"{ptype} ({params})" # e.g. DP-SGD (ε≈1.0...)
        if ptype == 'Manual Gen/Supp': return f"Manual ({params})"
        if ptype == 'Synthetic': return f"Synth ({params})"
        return params # Fallback

    df_heatmap = data.copy()
    df_heatmap['Heatmap_Label'] = df_heatmap.apply(create_heatmap_label, axis=1)

    # Filter for representative labels if desired, or use all found
    # plot_labels = [lbl for lbl in representative_params if lbl in df_heatmap['Heatmap_Label'].unique()]
    # df_heatmap = df_heatmap[df_heatmap['Heatmap_Label'].isin(plot_labels)]

    if df_heatmap.empty: print("   ⚠️ No data for heatmap."); return

    try:
        pivot_table = df_heatmap.pivot_table(index='Model', columns='Heatmap_Label', values=relative_metric)
        # Sort columns logically if possible (this is hard automatically)
        # pivot_table = pivot_table.reindex(columns=plot_labels, axis=1) # Reindex based on defined order

        plt.figure(figsize=(18, 8)) # Wider figure for heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, cbar_kws={'label': f'{relative_metric.replace("_Relative","")} (% of Baseline)'})
        plt.title(title + f' ({relative_metric.replace("_Relative","")})')
        plt.xlabel("Privacy Technique & Parameter")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        save_plot(plt.gcf(), f"enhanced_05_heatmap_{relative_metric}.png")
    except Exception as e:
        print(f"   ❌ Error creating heatmap: {e}")


# --- Generate Enhanced Plots ---
if df is not None and not df.empty:
    print("\n" + "="*10 + " Generating Enhanced Visualizations " + "="*10)

    # 1. Enhanced Baseline Comparison (Multiple Metrics)
    plot_baseline_comparison_enhanced(df, metrics=METRICS_TO_ANALYZE)

    # 2. Relative PUT Curves (Faceted per Model)
    # k-based methods vs k
    plot_relative_put_faceted(df, 'k_value', ['k-Anon (ARX)', 'Microagg'], relative_metric='F1_>50K_Relative', title_prefix='k-Based Relative PUT', xlabel='k value', higher_is_stricter=True)
    plot_relative_put_faceted(df, 'k_value', ['k-Anon (ARX)', 'Microagg'], relative_metric='ROC_AUC_Relative', title_prefix='k-Based Relative PUT', xlabel='k value', higher_is_stricter=True)
    # Epsilon-based methods vs epsilon
    plot_relative_put_faceted(df, 'epsilon_value', ['DP-SGD', 'LDP (RR)'], relative_metric='F1_>50K_Relative', title_prefix='Epsilon-Based Relative PUT', xlabel='Epsilon (ε)', higher_is_stricter=False)
    plot_relative_put_faceted(df, 'epsilon_value', ['DP-SGD', 'LDP (RR)'], relative_metric='ROC_AUC_Relative', title_prefix='Epsilon-Based Relative PUT', xlabel='Epsilon (ε)', higher_is_stricter=False)

    # 3. Relative Technique Comparison (Specific Params)
    # Define representative params based on YOUR data (check unique Privacy_Params_Final)
    rep_params = [
        'k=5', 'Microagg k=5',
        'ε≈1.0, δ=1e-05', 'LDP (RR) (ε≈1.0)', # Make sure LDP param matches CSV exactly
        'Simple Rules',
        f'{KNOWN_SYNTHESIZER}-UnscaledIn'
    ]
    present_params = df['Privacy_Params_Final'].unique()
    rep_params_present = [p for p in rep_params if p in present_params]
    if rep_params_present:
         plot_relative_technique_comparison(df, relative_metric='F1_>50K_Relative', param_filters=rep_params_present, title='Relative Technique Comparison at Representative Parameters')
         plot_relative_technique_comparison(df, relative_metric='ROC_AUC_Relative', param_filters=rep_params_present, title='Relative Technique Comparison at Representative Parameters')
    else:
        print("\n   ⚠️ Could not find data for representative parameter comparison points. Skipping plots.")

    # 4. Pareto-Style Scatter Plots
    # Utility vs k (higher k = stricter)
    plot_pareto_scatter(df[df['k_value'].notna()], privacy_axis_col='k_value', y_metric='F1_>50K_Relative', x_label='k value', higher_x_is_stricter=True, title_suffix=' (k-based Methods)')
    # Utility vs Epsilon (lower epsilon = stricter)
    plot_pareto_scatter(df[df['epsilon_value'].notna()], privacy_axis_col='epsilon_value', y_metric='F1_>50K_Relative', x_label='Epsilon (ε)', higher_x_is_stricter=False, title_suffix=' (ε-based Methods)')
    # Optional: Utility vs Inverse Epsilon (higher inv_eps = stricter)
    # plot_pareto_scatter(df[df['inv_epsilon'].notna()], privacy_axis_col='inv_epsilon', y_metric='F1_>50K_Relative', x_label='1 / ε (Higher=Stricter)', higher_x_is_stricter=True, title_suffix=' (ε-based Methods vs 1/ε)')

    # 5. Heatmap of Relative Performance
    plot_performance_heatmap(df, relative_metric='F1_>50K_Relative', title='Relative F1 Score (>50K) Heatmap')
    plot_performance_heatmap(df, relative_metric='ROC_AUC_Relative', title='Relative ROC AUC Heatmap')

    # 6. Keep Training Time Comparison (already good)
    def plot_training_time(data):
        """Bar chart comparing training time across models."""
        print("\n📊 Generating Plot: Training Time Comparison")
        if 'Training_Time' not in data.columns:
            print("   ⚠️ 'Training_Time' column not found in data. Skipping plot.")
            return
    
        plt.figure(figsize=(12, 7))
        sns.barplot(data=data, x='Model', y='Training_Time', hue='Privacy_Type', palette='coolwarm')
        plt.title('Training Time Comparison')
        plt.ylabel("Training Time (seconds)")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Privacy Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(bottom=0)
        save_plot(plt.gcf(), "enhanced_training_time_comparison.png")
    
    plot_training_time(df)

    # 7. NN Complexity Comparison (Example using Relative F1)
    nn_comp_data = df[df['Model'].isin(['SimpleNN', 'MediumNN'])].copy()
    # Plot baseline vs privacy type (relative)
    if not nn_comp_data.empty:
         g = sns.catplot(data=nn_comp_data, x='Privacy_Type', y='F1_>50K_Relative', col='Model', kind='bar', palette='viridis', height=5, aspect=1.3, order=sorted(nn_comp_data['Privacy_Type'].unique()))
         g.fig.suptitle('NN Complexity: Relative F1 Performance by Privacy Type', y=1.03)
         g.set_axis_labels("Privacy Type", "Relative F1 Score (%)")
         g.set_titles("{col_name}")
         g.set_xticklabels(rotation=45, ha='right')
         g.set(ylim=(0, 110))
         for ax in g.axes.flat: ax.axhline(100, color='grey', linestyle='--', alpha=0.7)
         save_plot(g.fig, "enhanced_08_nn_complexity_comparison_relative_f1.png")

    # 8. Synthetic Source Comparison (Absolute and Relative)
    def plot_synthetic_comparison(data, metric='F1_>50K'):
        """Bar chart comparing synthetic data performance."""
        print(f"\n📊 Generating Plot: Synthetic Data Comparison ({metric})")
        synthetic_data = data[data['Privacy_Type'] == 'Synthetic'].copy()
        if synthetic_data.empty:
            print("   ⚠️ No synthetic data found. Skipping plot.")
            return
    
        plt.figure(figsize=(12, 7))
        sns.barplot(data=synthetic_data, x='Model', y=metric, hue='Privacy_Params_Final', palette='coolwarm')
        plt.title(f'Synthetic Data Comparison ({metric})')
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Synthetic Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(bottom=0)
        save_plot(plt.gcf(), f"enhanced_synthetic_comparison_{metric}.png")
    
        plot_synthetic_comparison(df, metric='F1_>50K') # Absolute
        plot_synthetic_comparison(df, metric='F1_>50K_Relative') # Relative

    print("\n--- All Enhanced Plotting Complete ---")
    print(f"Plots saved in: {os.path.abspath(PLOTS_DIR)}")
else:
    print("\n--- No data loaded, skipping plotting ---")