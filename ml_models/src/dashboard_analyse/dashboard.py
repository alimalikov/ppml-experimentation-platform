# File: src/dashboard.py (or place in project root)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import re

# --- Page Config ---
st.set_page_config(layout="wide", page_title="PPML Benchmark Dashboard")

# --- Configuration ---
RESULTS_DIR = "../results" # Adjust path if dashboard.py is not in src
MASTER_SUMMARY_FILE = os.path.join(RESULTS_DIR, "final_master_summary_consolidated.csv")

# --- Caching Data Loading ---
@st.cache_data # Cache the data loading to speed up reruns
def load_and_prep_data(file_path):
    print("--- Loading and Preparing Data ---") # Shows in console when cache misses
    try:
        df = pd.read_csv(file_path)
        # --- Data Prep (copied from analyze_results.py) ---
        def extract_k(param_str):
            if isinstance(param_str, str) and param_str.startswith('k='):
                try: return int(re.search(r'k=(\d+)', param_str).group(1))
                except: return None
            return None
        def extract_epsilon(param_str):
             if isinstance(param_str, str) and ('ε≈' in param_str or 'eps=' in param_str):
                 try:
                     match = re.search(r'[ε|eps]≈?([\d\.]+)', param_str)
                     if match: return float(match.group(1))
                     else: return None
                 except: return None
             return None
        df['k_value'] = df['Privacy_Params_Final'].apply(extract_k)
        df['epsilon_value'] = df['Privacy_Params_Final'].apply(extract_epsilon)

        # Calculate Relative Metrics
        metrics_to_analyze = ['F1_>50K', 'ROC_AUC', 'Balanced_Accuracy', 'Accuracy']
        df_baseline = df[df['Privacy_Type'] == 'Baseline'].copy()
        if not df_baseline.empty:
            for metric in metrics_to_analyze:
                if metric in df.columns: # Check if metric exists
                    baseline_map = df_baseline.set_index('Model')[metric].to_dict()
                    df[f'{metric}_Baseline'] = df['Model'].map(baseline_map)
                    df[f'{metric}_Relative'] = np.where(
                        (df[f'{metric}_Baseline'].notna()) & (df[f'{metric}_Baseline'] != 0) & (df[metric].notna()),
                        (df[metric] / df[f'{metric}_Baseline']) * 100, 0 )
                    df.loc[df['Privacy_Type'] == 'Baseline', f'{metric}_Relative'] = 100.0
                    df[f'{metric}_Relative'].fillna(0, inplace=True)
                else:
                    df[f'{metric}_Relative'] = np.nan # Add column even if metric missing

        # Add a combined label for easier selection/display if needed
        df['Display_Label'] = df['Privacy_Type'] + ': ' + df['Privacy_Params_Final']
        df.loc[df['Privacy_Type'] == 'Baseline', 'Display_Label'] = 'Baseline'

        return df.sort_values(by=['Model', 'Privacy_Type']).reset_index(drop=True)

    except FileNotFoundError:
        st.error(f"Error: Master summary file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading or preparing data: {e}")
        return None

# --- Load Data ---
df_full = load_and_prep_data(MASTER_SUMMARY_FILE)

if df_full is None:
    st.stop() # Stop execution if data loading failed

# --- Sidebar Filters ---
st.sidebar.header("📊 Filters")

# Dataset (Placeholder for future)
# selected_dataset = st.sidebar.selectbox("Dataset", ["Adult"]) # Only one for now

# Models
available_models = sorted(df_full['Model'].unique())
selected_models = st.sidebar.multiselect("Models", available_models, default=available_models)

# Privacy Types
available_types = sorted(df_full['Privacy_Type'].unique())
selected_types = st.sidebar.multiselect("Privacy Types", available_types, default=available_types)

# Utility Metric
relative_metrics = [col for col in df_full.columns if '_Relative' in col]
absolute_metrics = ['F1_>50K', 'ROC_AUC', 'Balanced_Accuracy', 'Accuracy']
metric_options = ["Relative F1 (>50K)", "Relative ROC AUC", "Absolute F1 (>50K)", "Absolute ROC AUC"] # User-friendly names
metric_map = {
    "Relative F1 (>50K)": "F1_>50K_Relative", "Relative ROC AUC": "ROC_AUC_Relative",
    "Absolute F1 (>50K)": "F1_>50K", "Absolute ROC AUC": "ROC_AUC"
}
selected_metric_display = st.sidebar.selectbox("Utility Metric", metric_options)
selected_metric = metric_map[selected_metric_display]

# Filter Data based on selections
df_filtered = df_full[
    (df_full['Model'].isin(selected_models)) &
    (df_full['Privacy_Type'].isin(selected_types))
].copy()


# --- Main Dashboard Area ---
st.title("PPML Benchmark Dashboard")
st.markdown(f"Comparing **{', '.join(selected_models)}** models across **{len(selected_types)}** privacy types.")
st.markdown(f"Displaying Metric: **{selected_metric_display}**")

if df_filtered.empty:
    st.warning("No data matches the current filter settings.")
    st.stop()

# --- Display Data Table (Optional) ---
with st.expander("View Filtered Data"):
    st.dataframe(df_filtered[[col for col in df_full.columns if '_Baseline' not in col and col != 'Display_Label']].round(4), use_container_width=True) # Show rounded data


# --- Visualizations ---
st.header("Visualizations")

# Plot 1: Technique Comparison (Selected Metric)
st.subheader(f"Technique Comparison ({selected_metric_display})")
try:
    # Group by model and privacy type, find max/median/specific value?
    # Simplest: Bar chart comparing all selected points
    df_plot1 = df_filtered.dropna(subset=[selected_metric])
    fig1 = px.bar(df_plot1, x='Model', y=selected_metric, color='Display_Label',
                 barmode='group', title=f"{selected_metric_display} by Model and Privacy Setting",
                 labels={'Display_Label':'Privacy Setting', selected_metric:selected_metric_display},
                 category_orders={"Model": sorted(selected_models)}) # Ensure consistent model order
    fig1.update_layout(xaxis_tickangle=-45)
    if 'Relative' in selected_metric: fig1.update_yaxes(range=[0, 110], ticksuffix="%")
    st.plotly_chart(fig1, use_container_width=True)
except Exception as e:
    st.error(f"Could not generate Technique Comparison plot: {e}")


# Plot 2: PUT Curves (k-based)
st.subheader("Privacy-Utility Trade-off (k-based)")
df_plot2_k = df_filtered[df_filtered['k_value'].notna()].dropna(subset=[selected_metric])
if not df_plot2_k.empty:
    try:
        fig2 = px.line(df_plot2_k, x='k_value', y=selected_metric, color='Model',
                     line_dash='Privacy_Type', markers=True, symbol='Privacy_Type',
                     title=f"{selected_metric_display} vs. K Value",
                     labels={'k_value':'K Value (Higher=Stricter)', selected_metric:selected_metric_display},
                     category_orders={"Model": sorted(selected_models)})
        if 'Relative' in selected_metric: fig2.update_yaxes(range=[0, 110], ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate k-based PUT plot: {e}")
else:
    st.info("No k-based data selected/available for PUT plot.")

# Plot 3: PUT Curves (Epsilon-based)
st.subheader("Privacy-Utility Trade-off (Epsilon-based)")
df_plot3_eps = df_filtered[df_filtered['epsilon_value'].notna()].dropna(subset=[selected_metric])
if not df_plot3_eps.empty:
    try:
        fig3 = px.line(df_plot3_eps, x='epsilon_value', y=selected_metric, color='Model',
                     line_dash='Privacy_Type', markers=True, symbol='Privacy_Type',
                     title=f"{selected_metric_display} vs. Epsilon (Log Scale)",
                     labels={'epsilon_value':'Epsilon (Lower=Stricter)', selected_metric:selected_metric_display},
                     log_x=True, # Use log scale for epsilon
                     category_orders={"Model": sorted(selected_models)})
        if 'Relative' in selected_metric: fig3.update_yaxes(range=[0, 110], ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate epsilon-based PUT plot: {e}")
else:
    st.info("No epsilon-based data selected/available for PUT plot.")


# Plot 4: Bubble Chart (Selected Metric vs Privacy Type vs Time)
st.subheader(f"Utility vs. Training Time ({selected_metric_display})")
df_plot4 = df_filtered[(df_filtered['Training Time (s)'].notna()) & (df_filtered['Training Time (s)'] > 0)].dropna(subset=[selected_metric])
if not df_plot4.empty:
     try:
         min_size, max_size = 5, 50 # Adjust bubble size range
         fig4 = px.scatter(df_plot4, x='Display_Label', y=selected_metric, size='Training Time (s)', color='Model',
                         hover_name='Display_Label', hover_data=['Training Time (s)', selected_metric],
                         size_max=max_size, # Control max bubble size
                         title=f"{selected_metric_display} vs Privacy Setting (Size = Training Time)",
                         labels={'Display_Label': 'Privacy Setting', selected_metric:selected_metric_display},
                         category_orders={"Model": sorted(selected_models)})
         fig4.update_layout(xaxis_tickangle=-60, height=600)
         if 'Relative' in selected_metric: fig4.update_yaxes(range=[0, 110], ticksuffix="%")
         st.plotly_chart(fig4, use_container_width=True)
     except Exception as e:
         st.error(f"Could not generate bubble chart: {e}")
else:
     st.info("No data with training time selected/available for bubble chart.")

# Add more plots here (e.g., direct comparisons like Microagg vs k-Anon)
# Add text summaries or insights based on filtered data

# --- Analysis & Insights Section ---
st.header("📊 Analysis & Insights")

# Define PRIMARY_METRIC for analysis
PRIMARY_METRIC = "F1_>50K"  # Default metric for analysis ranking
RELATIVE_PRIMARY_METRIC = f"{PRIMARY_METRIC}_Relative"

# Select focus for analysis
analysis_focus = st.selectbox("Analyze by:", ["Overall Best Performers", "Select Specific Model", "Select Specific Privacy Technique"])

# Prepare baseline data for comparison
baseline_metrics = df_full[df_full['Privacy_Type'] == 'Baseline'].set_index('Model')[[PRIMARY_METRIC, 'ROC_AUC']].add_suffix('_Baseline')

# Perform analysis based on focus
if analysis_focus == "Overall Best Performers":
    st.subheader(f"Top Performing Techniques (based on {PRIMARY_METRIC})")
    # Exclude baseline from comparison of techniques
    df_analysis = df_filtered[df_filtered['Privacy_Type'] != 'Baseline'].copy()

    if not df_analysis.empty:
        # Find best result for each Model/Privacy Type combo
        best_per_group = df_analysis.loc[df_analysis.groupby(['Model', 'Privacy_Type'])[PRIMARY_METRIC].idxmax()]

        # Merge baseline for comparison
        best_per_group = pd.merge(best_per_group, baseline_metrics, left_on='Model', right_index=True, how='left')

        # Conditionally calculate Utility Drop and Retention
        if f"{PRIMARY_METRIC}_Baseline" in best_per_group.columns:
             best_per_group['Utility_Drop'] = best_per_group[f"{PRIMARY_METRIC}_Baseline"] - best_per_group[PRIMARY_METRIC]
             # Check if the relative metric column exists before assigning
             if f"{PRIMARY_METRIC}_Relative" in best_per_group.columns:
                 best_per_group['Utility_Retention (%)'] = best_per_group[f"{PRIMARY_METRIC}_Relative"]
             else:
                 best_per_group['Utility_Retention (%)'] = np.nan # Assign NaN if relative metric missing
        else:
             # Ensure columns exist even if baseline is missing, fill with NaN
             best_per_group['Utility_Drop'] = np.nan
             best_per_group['Utility_Retention (%)'] = np.nan


        st.markdown(f"Showing best result achieved for each selected Model/Privacy combination, ranked by **{PRIMARY_METRIC}**.")

        # --- Dynamically create the list of columns to display ---
        cols_to_display_df = ['Model', 'Privacy_Type', 'Privacy_Params_Final', PRIMARY_METRIC]
        if 'Utility_Retention (%)' in best_per_group.columns:
             cols_to_display_df.append('Utility_Retention (%)')
        if 'Training Time (s)' in best_per_group.columns:
             cols_to_display_df.append('Training Time (s)')
        # Add other desired columns similarly, checking if they exist

        st.dataframe(best_per_group[cols_to_display_df]\
                     .sort_values(PRIMARY_METRIC, ascending=False).round(4), use_container_width=True)

        # Highlight best overall utility preservation (check if column exists)
        if 'Utility_Retention (%)' in best_per_group.columns and best_per_group['Utility_Retention (%)'].notna().any():
             best_utility_preservation = best_per_group.loc[best_per_group['Utility_Retention (%)'].idxmax()]
             st.success(f"**Highest Utility Retention:** {best_utility_preservation['Model']} with {best_utility_preservation['Privacy_Type']} ({best_utility_preservation['Privacy_Params_Final']}) retained **{best_utility_preservation['Utility_Retention (%)']:.1f}%** of its baseline {PRIMARY_METRIC}.")
        elif PRIMARY_METRIC in best_per_group.columns:
             # Fallback if retention couldn't be calculated but primary metric exists
             best_primary_metric = best_per_group.loc[best_per_group[PRIMARY_METRIC].idxmax()]
             st.info(f"**Highest {PRIMARY_METRIC}:** {best_primary_metric['Model']} with {best_primary_metric['Privacy_Type']} ({best_primary_metric['Privacy_Params_Final']}) achieved **{best_primary_metric[PRIMARY_METRIC]:.4f}**.")


    else:
        st.warning("No non-baseline data selected to analyze.")


elif analysis_focus == "Select Specific Model":
    selected_model_analyze = st.selectbox("Select Model to Analyze:", selected_models)
    st.subheader(f"Analysis for Model: {selected_model_analyze}")

    df_model = df_filtered[(df_filtered['Model'] == selected_model_analyze) & (df_filtered['Privacy_Type'] != 'Baseline')].copy()
    df_model_baseline = df_full[(df_full['Model'] == selected_model_analyze) & (df_full['Privacy_Type'] == 'Baseline')].iloc[0] if not df_full[(df_full['Model'] == selected_model_analyze) & (df_full['Privacy_Type'] == 'Baseline')].empty else None

    if not df_model.empty and df_model_baseline is not None:
        st.markdown(f"Comparing privacy techniques for **{selected_model_analyze}**. Baseline {PRIMARY_METRIC}: **{df_model_baseline[PRIMARY_METRIC]:.4f}**")
        # Sort by chosen metric (relative or absolute)
        df_model_sorted = df_model.sort_values(selected_metric, ascending=False)

        # Display key stats
        cols_to_show = ['Privacy_Type', 'Privacy_Params_Final', selected_metric]
        selected_metric_relative = f"{selected_metric}_Relative"  # Define the relative metric column
        if selected_metric_relative in df_model_sorted.columns:
            cols_to_show.append(selected_metric_relative)
        cols_to_show.extend(['Precision_>50K', 'Recall_>50K', 'Training Time (s)'])
        st.dataframe(df_model_sorted[[col for col in cols_to_show if col in df_model_sorted.columns]].round(4), use_container_width=True)

        # Add summary text
        best_tech = df_model_sorted.iloc[0]
        worst_tech = df_model_sorted.iloc[-1]

        # Calculate precision-recall trade-off comments
        pr_balance = ""
        if 'Precision_>50K' in best_tech and 'Recall_>50K' in best_tech:
            precision_best = best_tech['Precision_>50K']
            recall_best = best_tech['Recall_>50K']
            if precision_best > recall_best*1.2:
                pr_balance = "This technique favors precision over recall."
            elif recall_best > precision_best*1.2:
                pr_balance = "This technique favors recall over precision."
            else:
                pr_balance = "This technique maintains a good balance between precision and recall."

        st.markdown(f"""
        **Summary for {selected_model_analyze}:**
        *   **Best Utility ({selected_metric_display}):** Achieved with **{best_tech['Privacy_Type']} ({best_tech['Privacy_Params_Final']})**, reaching **{best_tech[selected_metric]:.4f}**.
            *   This represents **{best_tech.get(selected_metric_relative, 'N/A'):.1f}%** of the baseline performance.
            *   Training Time: {best_tech['Training Time (s)']:.2f}s.
            *   {pr_balance}
        *   **Lowest Utility ({selected_metric_display}):** Occurred with **{worst_tech['Privacy_Type']} ({worst_tech['Privacy_Params_Final']})**, reaching **{worst_tech[selected_metric]:.4f}**.
            *   This represents **{worst_tech.get(selected_metric_relative, 'N/A'):.1f}%** of the baseline performance.
            *   Training Time: {worst_tech['Training Time (s)']:.2f}s.
        """)
        
        # Generate observations based on data
        # Check if some techniques performed significantly better
        privacy_types_present = df_model['Privacy_Type'].unique()
        if len(privacy_types_present) > 1:
            avg_by_type = df_model.groupby('Privacy_Type')[selected_metric].mean().sort_values(ascending=False)
            best_type = avg_by_type.index[0]
            worst_type = avg_by_type.index[-1]
            st.markdown(f"""
            *   **Observations:** On average, **{best_type}** performs best for this model, while **{worst_type}** shows the largest utility drops.
                *   {selected_model_analyze} appears {"more robust" if avg_by_type.iloc[0] > 0.7*df_model_baseline[selected_metric] else "sensitive"} 
                   to privacy-preserving transformations.
            """)
    else:
        st.warning(f"No privacy results or baseline found for model: {selected_model_analyze}")


elif analysis_focus == "Select Specific Privacy Technique":
    # Exclude Baseline from selection list
    privacy_options = [pt for pt in selected_types if pt != 'Baseline']
    if not privacy_options:
        st.warning("Please select at least one non-Baseline privacy type in the sidebar.")
        st.stop()
    selected_privacy_analyze = st.selectbox("Select Privacy Technique to Analyze:", privacy_options)
    st.subheader(f"Analysis for Technique: {selected_privacy_analyze}")

    df_privacy = df_filtered[df_filtered['Privacy_Type'] == selected_privacy_analyze].copy()

    if not df_privacy.empty:
        # Add baseline comparison columns
        df_privacy = pd.merge(df_privacy, baseline_metrics, left_on='Model', right_index=True, how='left')

        st.markdown(f"Comparing model performance for **{selected_privacy_analyze}**.")
        # Sort by chosen metric
        df_privacy_sorted = df_privacy.sort_values([PRIMARY_METRIC], ascending=False) # Rank by primary metric

        cols_to_show = ['Model', 'Privacy_Params_Final', selected_metric]
        selected_metric_relative = f"{selected_metric}_Relative"  # Define the relative metric column
        if selected_metric_relative in df_privacy_sorted.columns:
            cols_to_show.append(selected_metric_relative)
        cols_to_show.extend(['Precision_>50K', 'Recall_>50K', 'Training Time (s)'])
        st.dataframe(df_privacy_sorted[[col for col in cols_to_show if col in df_privacy_sorted.columns]].round(4), use_container_width=True)

        # Add summary text
        best_model = df_privacy_sorted.iloc[0]
        worst_model = df_privacy_sorted.iloc[-1]

        # Calculate average retention across models
        avg_retention = df_privacy[f"{PRIMARY_METRIC}_Relative"].mean() if f"{PRIMARY_METRIC}_Relative" in df_privacy.columns else "N/A"
        
        # Detect if technique has a model type preference
        model_groups = {}
        for model in df_privacy['Model'].unique():
            if "NN" in model or "Neural" in model:
                model_groups.setdefault("Neural Networks", []).append(model)
            elif "RF" in model or "Random" in model:
                model_groups.setdefault("Tree-based", []).append(model)
            elif "LR" in model or "Log" in model:
                model_groups.setdefault("Linear", []).append(model)
                
        group_performance = {}
        for group, models in model_groups.items():
            group_performance[group] = df_privacy[df_privacy['Model'].isin(models)][f"{PRIMARY_METRIC}_Relative"].mean() \
                if f"{PRIMARY_METRIC}_Relative" in df_privacy.columns else 0
                
        technique_preference = ""
        if group_performance:
            best_group = max(group_performance.items(), key=lambda x: x[1])
            worst_group = min(group_performance.items(), key=lambda x: x[1])
            if best_group[1] > worst_group[1]*1.3:  # At least 30% better
                technique_preference = f"This technique works particularly well with {best_group[0]} models compared to {worst_group[0]} models."

        st.markdown(f"""
        **Summary for {selected_privacy_analyze}:**
        *   **Best Performing Model ({selected_metric_display}):** **{best_model['Model']}** ({best_model['Privacy_Params_Final']}), achieving **{best_model[selected_metric]:.4f}**.
            *   This is **{best_model.get(selected_metric_relative, 'N/A'):.1f}%** of that model's baseline performance.
        *   **Lowest Performing Model ({selected_metric_display}):** **{worst_model['Model']}** ({worst_model['Privacy_Params_Final']}), achieving **{worst_model[selected_metric]:.4f}**.
            *   This is **{worst_model.get(selected_metric_relative, 'N/A'):.1f}%** of that model's baseline performance.
        *   **Average Utility Retention:** {avg_retention:.1f}%
        *   **Observations:** {technique_preference}
        """)
    else:
        st.warning(f"No results found for privacy technique: {selected_privacy_analyze}")

# Add a section for privacy-utility recommendations
with st.expander("Privacy-Utility Trade-off Recommendations"):
    # Generate dynamic recommendations based on filtered data
    st.markdown("### Dynamic Recommendations Based on Current Data")
    
    # Calculate metrics for recommendations
    if not df_filtered.empty:
        # Find best privacy settings for each privacy type
        best_settings = {}
        rec_metrics = ["F1_>50K_Relative", "ROC_AUC_Relative"]
        
        for privacy_type in df_filtered['Privacy_Type'].unique():
            if privacy_type == 'Baseline':
                continue
                
            df_type = df_filtered[df_filtered['Privacy_Type'] == privacy_type]
            if not df_type.empty and all(metric in df_type.columns for metric in rec_metrics):
                # Get best average relative performance
                df_type['avg_rel_performance'] = df_type[rec_metrics].mean(axis=1)
                best_row = df_type.loc[df_type['avg_rel_performance'].idxmax()]
                best_settings[privacy_type] = {
                    'params': best_row['Privacy_Params_Final'],
                    'avg_performance': best_row['avg_rel_performance'],
                    'models': best_row['Model']
                }
        
        # Display dynamic recommendations
        if best_settings:
            st.markdown("**Based on your current selection, consider these options:**")
            
            cols = st.columns(len(best_settings))
            for i, (privacy_type, info) in enumerate(best_settings.items()):
                with cols[i]:
                    st.metric(
                        label=f"{privacy_type}",
                        value=f"{info['params']}",
                        delta=f"{info['avg_performance']:.1f}% retained"
                    )
    
    # Privacy vs utility slider
    st.markdown("### Find recommendations based on your requirements")
    priority_slider = st.slider(
        "Privacy-Utility Balance", 
        0, 100, 50, 
        help="0 = Maximum Privacy, 100 = Maximum Utility"
    )
    
    # Generate targeted recommendations based on slider
    if priority_slider <= 33:
        st.info("⚠️ **High Privacy Priority Detected** - Showing strong privacy recommendations")
    elif priority_slider <= 66:
        st.info("⚖️ **Balanced Approach Detected** - Showing balanced recommendations")
    else:
        st.info("📈 **High Utility Priority Detected** - Showing utility-preserving recommendations")
    
    # Display static recommendations
    st.markdown("""
    ### Recommendations Based on Analysis

    **For High Privacy Requirements:**
    - If strong formal guarantees are needed, consider DP-SGD with ε≈1.0, ideally with simpler models.
    - For dataset publishing, k-Anonymity with k≥10 can provide good privacy, though with fewer formal guarantees.

    **For Balanced Privacy-Utility:**
    - DP-SGD with ε values between 3-5 often provides a good balance of formal privacy and utility.
    - Microaggregation with k=5 typically preserves more utility than k-Anonymity at equivalent k values.

    **For Minimal Utility Impact:**
    - If formal DP guarantees are required, use ε≥8 (though this provides weaker privacy).
    - Consider manual suppression/generalization if domain expertise allows targeting specific attributes.
    - Synthetic data generation can preserve patterns while obscuring individual records.

    **Model-Specific Considerations:**
    - Tree-based models (RF, XGBoost) tend to be more sensitive to privacy mechanisms affecting feature distributions.
    - Neural networks can sometimes adapt better to noise added by privacy mechanisms.
    - Logistic Regression shows moderate resilience and predictable degradation across privacy techniques.
    """)