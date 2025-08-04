"""
Simple Visualization Integration - Extract Table Data and Use Original Logic
=========================================================================

This module extracts data from ML app tables and feeds it to the EXACT same
visualization functions from visualization_summary.py.

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

# EXACT SAME STYLING FROM ORIGINAL - Copy from visualization_summary.py
plt.style.use('default')
sns.set_palette("husl")

from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'  # Fallback font since Porsche Next TT may not be available
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
    'Logistic Regression': '#007ACC',
    'Random Forest': '#FF6B35',
    'XGBoost': '#28A745'
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

class SimpleVisualizationIntegrator:
    """
    Simple approach: Extract table data and use original visualization functions
    """
    
    def __init__(self):
        """Initialize the integrator"""
        self.temp_dir = tempfile.mkdtemp(prefix="ml_viz_")
        self.generated_files = []
        
        # For ML app compatibility
        self.ANONYMIZATION_LEVELS = {
            'original': 'Original',
            'differential_privacy': 'Differential Privacy',
            'laplace_mechanism': 'Differential Privacy',
            'micro_aggregation': 'Micro Aggregation',
            'randomized_response': 'Randomized Response'
        }
    
    def extract_table_data_to_results_format(self, df: pd.DataFrame, selected_metrics: List[str]) -> Dict[str, Any]:
        """
        Extract data from ML app table and convert to the EXACT same format as visualization_summary.py
        
        Expected table columns:
        - Dataset, Type, Model, Privacy Method, Accuracy, Precision, Recall, F1-Score, [Custom Metrics...]
        
        Returns:
        results[dataset][anonymization][model][metric] = value
        """
        try:
            # Find required columns
            dataset_col = None
            for col in df.columns:
                if 'dataset' in col.lower():
                    dataset_col = col
                    break
            
            type_col = None  
            for col in df.columns:
                if 'type' in col.lower():
                    type_col = col
                    break
                    
            model_col = None
            for col in df.columns:
                if 'model' in col.lower():
                    model_col = col
                    break
                    
            privacy_col = None
            for col in df.columns:
                if 'privacy' in col.lower() or 'method' in col.lower():
                    privacy_col = col
                    break
            
            if not all([dataset_col, type_col, model_col, privacy_col]):
                st.error(f"❌ Required columns not found: Dataset={dataset_col}, Type={type_col}, Model={model_col}, Privacy={privacy_col}")
                return {}
            
            # Initialize results structure
            results = {}
            
            # Process each row
            for _, row in df.iterrows():
                dataset = str(row[dataset_col]).strip()
                data_type = str(row[type_col]).strip()
                model = str(row[model_col]).strip()
                privacy_method = str(row[privacy_col]).strip()
                
                # Map privacy method to anonymization technique
                anonymization = self.map_privacy_to_anonymization(privacy_method, data_type)
                
                # Initialize nested structure
                if dataset not in results:
                    results[dataset] = {}
                if anonymization not in results[dataset]:
                    results[dataset][anonymization] = {}
                if model not in results[dataset][anonymization]:
                    results[dataset][anonymization][model] = {}
                
                # Extract selected metrics
                for metric_col in selected_metrics:
                    if metric_col in df.columns:
                        try:
                            # Clean the value (remove percentage signs, arrows, etc.)
                            raw_value = str(row[metric_col])
                            
                            # Extract numeric value from strings like "0.3433", "▼ -65.7%", etc.
                            if raw_value == "—" or raw_value == "nan":
                                continue
                                
                            # Extract just the number part
                            cleaned_value = re.sub(r'[^\d\.-]', '', raw_value.split(' ')[0])
                            if cleaned_value:
                                value = float(cleaned_value)
                                
                                # Convert percentage to decimal if value > 1 (except for some metrics)
                                metric_lower = metric_col.lower()
                                if value > 1.0 and not any(term in metric_lower for term in ['count', 'depth', 'estimators', 'leaves']):
                                    value = value / 100.0
                                
                                # Map column name to standard metric name
                                metric_name = self.map_column_to_metric(metric_col)
                                results[dataset][anonymization][model][metric_name] = value
                                
                        except (ValueError, TypeError) as e:
                            st.warning(f"⚠️ Could not parse value '{row[metric_col]}' for metric '{metric_col}': {e}")
                            continue
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error extracting table data: {str(e)}")
            return {}
    
    def map_privacy_to_anonymization(self, privacy_method: str, data_type: str) -> str:
        """Map privacy method and type to anonymization technique name"""
        privacy_lower = privacy_method.lower()
        type_lower = data_type.lower()
        
        if 'original' in privacy_lower or 'original' in type_lower:
            return 'Original'
        elif 'differential' in privacy_lower or 'laplace' in privacy_lower:
            if 'high' in privacy_lower:
                return 'Differential Privacy (High)'
            elif 'medium' in privacy_lower:
                return 'Differential Privacy (Medium)'
            else:
                return 'Differential Privacy (Minimal)'
        elif 'micro' in privacy_lower and 'aggregation' in privacy_lower:
            if 'high' in privacy_lower:
                return 'Micro Aggregation (High)'
            elif 'medium' in privacy_lower:
                return 'Micro Aggregation (Medium)'
            else:
                return 'Micro Aggregation (Minimal)'
        elif 'randomized' in privacy_lower and 'response' in privacy_lower:
            if 'high' in privacy_lower:
                return 'Randomized Response (High)'
            elif 'medium' in privacy_lower:
                return 'Randomized Response (Medium)'
            else:
                return 'Randomized Response (Minimal)'
        else:
            # Default mapping based on data type
            if 'anonymized' in type_lower:
                return 'Differential Privacy (High)'  # Default for anonymized data
            else:
                return 'Original'
    
    def map_column_to_metric(self, column_name: str) -> str:
        """Map table column names to standard metric names"""
        col_lower = column_name.lower()
        
        # Standard mappings
        if 'accuracy' in col_lower:
            return 'accuracy'
        elif 'precision' in col_lower:
            return 'precision'
        elif 'recall' in col_lower:
            return 'recall'
        elif 'f1' in col_lower or 'f1-score' in col_lower:
            return 'f1_score'
        elif 'r²' in col_lower or 'r2' in col_lower:
            return 'r2_score'
        elif 'rmse' in col_lower:
            return 'rmse'
        elif 'mae' in col_lower:
            return 'mae'
        elif 'mape' in col_lower:
            return 'mape'
        elif 'balanced' in col_lower and 'accuracy' in col_lower:
            return 'balanced_accuracy'
        elif 'matthews' in col_lower or 'correlation' in col_lower:
            return 'matthews_correlation'
        elif 'roc' in col_lower or 'auc' in col_lower:
            return 'roc_auc'
        else:
            # For custom metrics, clean the name
            return col_lower.replace(' ', '_').replace('-', '_').replace('📊', '').replace('🎯', '').replace('⚡', '').strip()

    # ============================================================================
    # COPIED VISUALIZATION FUNCTIONS FROM ORIGINAL visualization_summary.py
    # ============================================================================
    
    def plot_combined_dataset_performance(self, results: Dict, save_path: str = None) -> bool:
        """
        EXACT COPY of the comprehensive cross-dataset analysis from visualization_summary.py
        """
        try:
            if not results:
                st.error("❌ No data available for visualization")
                return False

            # Prepare data for cross-dataset comparison - EXACT SAME LOGIC AS ORIGINAL
            combined_data = []
            
            for dataset, anon_data in results.items():
                for anon, models in anon_data.items():
                    for model, metrics in models.items():
                        if len(metrics) > 0:
                            avg_performance = sum(metrics.values()) / len(metrics)
                        else:
                            avg_performance = 0.0
                        
                        # Calculate degradation from original if not original
                        degradation_pct = 0.0
                        if anon != "Original" and "Original" in anon_data and model in anon_data["Original"]:
                            original_metrics = anon_data["Original"][model]
                            if len(original_metrics) > 0:
                                original_avg = sum(original_metrics.values()) / len(original_metrics)
                                if original_avg > 0:
                                    degradation_pct = ((original_avg - avg_performance) / original_avg) * 100
                        
                        combined_data.append({
                            "Dataset": dataset,
                            "Anonymization": anon,
                            "Model": model,
                            "Average Performance": avg_performance,
                            "Degradation (%)": degradation_pct
                        })
            
            if not combined_data:
                st.error("❌ No valid data found for visualization")
                return False
            
            df = pd.DataFrame(combined_data)
            
            # Create subplots for different analyses - EXACT SAME AS ORIGINAL
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 16))
            
            # 1. Performance heatmap across datasets - EXACT SAME AS ORIGINAL
            performance_pivot = df.pivot_table(
                index=["Dataset", "Anonymization"], 
                columns="Model", 
                values="Average Performance"
            )
            
            if not performance_pivot.empty:
                sns.heatmap(performance_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, 
                           cbar_kws={'label': 'Performance Score'})
                ax1.set_title("Performance Across All Datasets\n(After Anonymization)", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Model", fontsize=12)
                ax1.set_ylabel("Dataset - Anonymization", fontsize=12)
            
            # 2. Degradation comparison - EXACT SAME AS ORIGINAL
            degradation_pivot = df[df["Anonymization"] != "Original"].pivot_table(
                index=["Dataset", "Anonymization"],
                columns="Model",
                values="Degradation (%)"
            )
            
            if not degradation_pivot.empty:
                sns.heatmap(degradation_pivot, annot=True, fmt='.1f', cmap='Reds', ax=ax2,
                           cbar_kws={'label': 'Degradation %'})
                ax2.set_title("Performance Degradation %\n(Across All Datasets)", fontsize=14, fontweight='bold')
                ax2.set_xlabel("Model", fontsize=12)
                ax2.set_ylabel("Dataset - Anonymization", fontsize=12)
            
            plt.suptitle("Comprehensive Cross-Dataset Analysis", fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
                st.success(f"✅ Saved comprehensive analysis to {save_path}")
            else:
                st.pyplot(fig, use_container_width=True)
            
            plt.close()
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating comprehensive analysis: {str(e)}")
            return False

    def generate_visualizations_in_app(self, df: pd.DataFrame, selected_metrics: List[str]) -> bool:
        """
        Generate visualizations using the original logic with extracted table data
        """
        try:
            st.markdown("### 🔍 **Data Extraction Debug**")
            with st.expander("📊 **Debug: Data Processing**", expanded=False):
                st.text(f"Input DataFrame: {len(df)} rows, {len(df.columns)} columns")
                st.text(f"Selected Metrics: {selected_metrics}")
                
                st.markdown("**Available Columns:**")
                for col in df.columns:
                    st.text(f"- {col}")
            
            # Extract data from table to results format
            results = self.extract_table_data_to_results_format(df, selected_metrics)
            
            if not results:
                st.error("❌ **Failed to extract data from table.** Please check your table format.")
                return False
            
            with st.expander("🔍 **Debug: Extracted Results Structure**", expanded=False):
                st.text(f"✅ Extracted Data: {len(results)} datasets found")
                
                for dataset, anon_data in results.items():
                    st.text(f"📊 **{dataset}:** {len(anon_data)} anonymization methods")
                    for anon, models in anon_data.items():
                        st.text(f"  - {anon}: {len(models)} models")
                        for model, metrics in models.items():
                            st.text(f"    - {model}: {list(metrics.keys())}")
            
            # Generate the main comprehensive visualization using ORIGINAL function
            st.markdown("### 📊 **Professional Visualizations**")
            st.markdown("*Using the exact same logic as visualization_summary.py*")
            
            success = self.plot_combined_dataset_performance(results)
            
            if success:
                st.success("✅ **Visualizations generated successfully!**")
                return True
            else:
                st.error("❌ **Failed to generate visualizations.**")
                return False
                
        except Exception as e:
            st.error(f"❌ **Visualization Error:** {str(e)}")
            return False

    def generate_png_exports(self, df: pd.DataFrame, selected_metrics: List[str]) -> Tuple[bool, str]:
        """
        Generate PNG exports using the original logic
        """
        try:
            # Extract data from table
            results = self.extract_table_data_to_results_format(df, selected_metrics)
            
            if not results:
                return False, ""
            
            # Create export directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(self.temp_dir, f"thesis_visualizations_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate comprehensive analysis
            comprehensive_path = os.path.join(export_dir, "comprehensive_cross_dataset_analysis.png")
            success = self.plot_combined_dataset_performance(results, save_path=comprehensive_path)
            
            if success and os.path.exists(comprehensive_path):
                self.generated_files.append(comprehensive_path)
            
            # Create ZIP file
            zip_path = os.path.join(self.temp_dir, f"thesis_visualizations_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in self.generated_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return len(self.generated_files) > 0, zip_path
            
        except Exception as e:
            st.error(f"❌ Export Error: {str(e)}")
            return False, ""

# Create global instance
viz_integrator = SimpleVisualizationIntegrator()

# Function for ML app compatibility
def get_visualization_integrator():
    """
    Return the global visualization integrator instance
    Required for ML app compatibility
    """
    return viz_integrator
