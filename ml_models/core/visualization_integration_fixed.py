"""
Professional Visualization Integration Module - EXACT REPLICATION
===============================================================

This module replicates the EXACT functionality of visualization_summary.py
but with dynamic input from the ML app table data.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
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

# Set global style for publication-quality plots - EXACT SAME AS ORIGINAL
plt.style.use('default')
sns.set_palette("husl")

# Professional publication settings with fallback font - EXACT SAME AS ORIGINAL
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'  # Fallback font
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

# Clean, bright professional color palette - EXACT SAME AS ORIGINAL
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

class VisualizationIntegrator:
    """
    Professional class for replicating visualization_summary.py functionality
    """
    
    def __init__(self):
        """Initialize the visualization integrator"""
        self.temp_dir = tempfile.mkdtemp(prefix="ml_viz_")
        self.generated_files = []
        
    def transform_table_to_results_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform the ML app table data to the exact same nested dictionary format
        as used in visualization_summary.py
        
        Returns:
            Dict: results[dataset][anonymization][model][metric] = value
        """
        try:
            # Find relevant columns
            dataset_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['dataset', 'data']) and 'type' not in col.lower():
                    dataset_col = col
                    break
            
            privacy_col = None
            for col in df.columns:
                if 'privacy' in col.lower() or 'method' in col.lower():
                    privacy_col = col
                    break
                    
            model_col = None
            for col in df.columns:
                if 'model' in col.lower():
                    model_col = col
                    break
            
            if not all([dataset_col, privacy_col, model_col]):
                st.error(f"❌ Required columns not found. Found: dataset={dataset_col}, privacy={privacy_col}, model={model_col}")
                return {}
            
            # Initialize results structure
            results = {}
            
            # Get metric columns (numeric columns that aren't identifiers)
            metric_columns = []
            for col in df.columns:
                if col not in [dataset_col, privacy_col, model_col] and df[col].dtype in ['float64', 'int64']:
                    # Clean column name for metric
                    clean_metric = col.lower().replace(' ', '_').replace('%', '').replace('📊', '').replace('🎯', '').replace('⚡', '').strip()
                    if clean_metric and clean_metric not in ['unnamed', 'index']:
                        metric_columns.append((col, clean_metric))
            
            # Process each row
            for _, row in df.iterrows():
                dataset = str(row[dataset_col]).strip()
                privacy_method = str(row[privacy_col]).strip()
                model = str(row[model_col]).strip()
                
                # Map privacy method to anonymization technique
                anonymization = self.map_privacy_method_to_anonymization(privacy_method)
                
                # Initialize nested structure
                if dataset not in results:
                    results[dataset] = {}
                if anonymization not in results[dataset]:
                    results[dataset][anonymization] = {}
                if model not in results[dataset][anonymization]:
                    results[dataset][anonymization][model] = {}
                
                # Extract metrics
                for original_col, clean_metric in metric_columns:
                    try:
                        value = float(row[original_col])
                        # Convert percentage to decimal if needed
                        if '%' in original_col or value > 1.0:
                            value = value / 100.0 if value > 1.0 else value
                        results[dataset][anonymization][model][clean_metric] = value
                    except (ValueError, TypeError):
                        continue
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error transforming data: {str(e)}")
            return {}
    
    def map_privacy_method_to_anonymization(self, privacy_method: str) -> str:
        """Map privacy method from table to anonymization technique name"""
        method_lower = privacy_method.lower()
        
        if 'original' in method_lower or 'baseline' in method_lower:
            return 'Original'
        elif 'micro' in method_lower and 'aggregation' in method_lower:
            if 'high' in method_lower:
                return 'Micro Aggregation (High)'
            elif 'medium' in method_lower:
                return 'Micro Aggregation (Medium)'
            else:
                return 'Micro Aggregation (Minimal)'
        elif 'differential' in method_lower and 'privacy' in method_lower:
            if 'high' in method_lower:
                return 'Differential Privacy (High)'
            elif 'medium' in method_lower:
                return 'Differential Privacy (Medium)'
            else:
                return 'Differential Privacy (Minimal)'
        elif 'randomized' in method_lower and 'response' in method_lower:
            if 'high' in method_lower:
                return 'Randomized Response (High)'
            elif 'medium' in method_lower:
                return 'Randomized Response (Medium)'
            else:
                return 'Randomized Response (Minimal)'
        elif 'laplace' in method_lower:
            if 'high' in method_lower:
                return 'Differential Privacy (High)'
            elif 'medium' in method_lower:
                return 'Differential Privacy (Medium)'
            else:
                return 'Differential Privacy (Minimal)'
        else:
            return privacy_method  # Use as-is if no mapping found

    def plot_combined_cross_dataset_analysis(self, results: Dict, save_path: str = None) -> bool:
        """
        Create the EXACT same comprehensive cross-dataset heatmap as in visualization_summary.py
        This is the main visualization you see in the attached image.
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
                        avg_performance = sum(metrics.values()) / len(metrics)
                        
                        # Calculate degradation from original if not original
                        if anon != "Original" and "Original" in anon_data:
                            original_avg = sum(anon_data["Original"][model].values()) / len(anon_data["Original"][model])
                            degradation_pct = ((original_avg - avg_performance) / original_avg) * 100 if original_avg > 0 else 0
                        else:
                            degradation_pct = 0
                        
                        combined_data.append({
                            "Dataset": dataset,
                            "Anonymization": anon,
                            "Model": model,
                            "Average Performance": avg_performance,
                            "Degradation (%)": degradation_pct
                        })
            
            df = pd.DataFrame(combined_data)
            
            # Create subplots for different analyses - EXACT SAME AS ORIGINAL
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 12))
            
            # 1. Performance heatmap across datasets - EXACT SAME AS ORIGINAL
            performance_pivot = df.pivot_table(
                index=["Dataset", "Anonymization"], 
                columns="Model", 
                values="Average Performance"
            )
            
            sns.heatmap(performance_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, 
                       cbar_kws={'label': 'Performance Score'})
            ax1.set_title("Performance Across All Datasets\n(After Anonymization)", fontsize=16, fontweight='bold')
            ax1.set_xlabel("Model", fontsize=14)
            ax1.set_ylabel("Dataset - Anonymization", fontsize=14)
            
            # 2. Degradation comparison - EXACT SAME AS ORIGINAL
            degradation_pivot = df[df["Anonymization"] != "Original"].pivot_table(
                index=["Dataset", "Anonymization"],
                columns="Model",
                values="Degradation (%)"
            )
            
            sns.heatmap(degradation_pivot, annot=True, fmt='.1f', cmap='Reds', ax=ax2,
                       cbar_kws={'label': 'Degradation %'})
            ax2.set_title("Performance Degradation %\n(Across All Datasets)", fontsize=16, fontweight='bold')
            ax2.set_xlabel("Model", fontsize=14)
            ax2.set_ylabel("Dataset - Anonymization", fontsize=14)
            
            plt.suptitle("Comprehensive Cross-Dataset Analysis", fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300, facecolor='white')
                st.success(f"✅ Saved comprehensive analysis to {save_path}")
            else:
                st.pyplot(fig, use_container_width=True)
            
            plt.close()
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating comprehensive analysis: {str(e)}")
            return False

    def generate_visualizations_in_app(self, df: pd.DataFrame) -> bool:
        """
        Generate and display the EXACT same visualizations as visualization_summary.py
        but with dynamic data from the table
        """
        try:
            # Transform data to results structure
            with st.expander("🔍 **Debug: Data Structure**", expanded=False):
                st.markdown("**Input Data:**")
                st.text(f"📊 Input Data: {len(df)} rows, {len(df.columns)} columns")
                
                st.markdown("**DataFrame Columns:**")
                for col in df.columns:
                    st.text(f"- {col}: {df[col].dtype}")
                
            results = self.transform_table_to_results_dict(df)
            
            if not results:
                st.error("❌ **Failed to transform data.** Please check your table format.")
                return False
            
            with st.expander("🔍 **Debug: Transformed Data Structure**", expanded=False):
                st.markdown("**Transformed Results Structure:**")
                st.text(f"✅ Data Ready: {len(results)} datasets found")
                
                for dataset, anon_data in results.items():
                    st.text(f"📊 **{dataset}:** {len(anon_data)} anonymization methods")
                    for anon, models in anon_data.items():
                        st.text(f"  - {anon}: {len(models)} models")
            
            # Generate the main comprehensive visualization - EXACT SAME AS ORIGINAL
            st.markdown("### 📊 **Performance Comparison**")
            st.markdown("*Cross-dataset performance and degradation analysis*")
            
            success = self.plot_combined_cross_dataset_analysis(results)
            
            if success:
                st.success("✅ **Visualizations generated successfully!**")
                return True
            else:
                st.error("❌ **Failed to generate visualizations.**")
                return False
                
        except Exception as e:
            st.error(f"❌ **Visualization Error:** {str(e)}")
            return False

    def generate_png_exports(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Generate PNG exports with the EXACT same structure as visualization_summary.py
        """
        try:
            results = self.transform_table_to_results_dict(df)
            
            if not results:
                return False, ""
            
            # Create export directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(self.temp_dir, f"thesis_visualizations_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate comprehensive cross-dataset analysis
            comprehensive_path = os.path.join(export_dir, "comprehensive_cross_dataset_analysis.png")
            success = self.plot_combined_cross_dataset_analysis(results, save_path=comprehensive_path)
            
            if success:
                self.generated_files.append(comprehensive_path)
            
            # Create ZIP file
            zip_path = os.path.join(self.temp_dir, f"thesis_visualizations_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in self.generated_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return True, zip_path
            
        except Exception as e:
            st.error(f"❌ Export Error: {str(e)}")
            return False, ""

# Create global instance
viz_integrator = VisualizationIntegrator()
