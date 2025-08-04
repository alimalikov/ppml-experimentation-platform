"""
Global Recoding Anonymization Plugin

This plugin implements global recoding anonymization, which replaces specific
values with more general categories across the entire dataset. This is commonly
used for categorical data to reduce granularity and increase k-anonymity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base_anonymizer import Anonymizer


class GlobalRecodingAnonymizer(Anonymizer):
    """
    Global Recoding anonymization implementation.
    
    Global recoding replaces specific values with more general categories
    consistently across the entire dataset. This is useful for reducing
    the granularity of categorical or numerical data.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Global Recoding"
        self.category = "Data Transformation"
        self.selected_columns = []
        self.recoding_rules = {}
        self.auto_generate_rules = True
        self.numerical_binning = True
        self.num_bins = 5
        
    def get_name(self) -> str:
        return self.name
        
    def get_category(self) -> str:
        return self.category
        
    def get_sidebar_ui(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Streamlit sidebar UI for global recoding configuration."""
        st.sidebar.markdown("### Global Recoding Configuration")
        
        # Column selection
        available_columns = data.columns.tolist()
        if not available_columns:
            st.sidebar.warning("No columns available for global recoding")
            return {}
            
        selected_columns = st.sidebar.multiselect(
            "Select columns for recoding:",
            options=available_columns,
            default=available_columns[:min(3, len(available_columns))],
            help="Choose which columns to apply global recoding to"
        )
        
        if not selected_columns:
            return {}
        
        # Auto-generation option
        auto_generate_rules = st.sidebar.checkbox(
            "Auto-generate recoding rules",
            value=True,
            help="Automatically create recoding rules based on data patterns"
        )
        
        recoding_rules = {}
        
        if auto_generate_rules:
            # Options for auto-generation
            st.sidebar.markdown("#### Auto-generation Settings")
            
            # Numerical binning
            numerical_binning = st.sidebar.checkbox(
                "Enable numerical binning",
                value=True,
                help="Bin numerical columns into ranges"
            )
            
            num_bins = 5
            if numerical_binning:
                num_bins = st.sidebar.slider(
                    "Number of bins for numerical columns:",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Number of bins to create for numerical columns"
                )
            
            # Preview auto-generated rules
            if st.sidebar.button("Preview Auto-generated Rules"):
                preview_rules = self._generate_auto_rules(data, selected_columns, numerical_binning, num_bins)
                st.sidebar.markdown("**Preview of rules:**")
                for col, rules in preview_rules.items():
                    st.sidebar.text(f"{col}: {len(rules)} mappings")
        else:
            # Manual rule definition
            st.sidebar.markdown("#### Manual Recoding Rules")
            st.sidebar.info("Define custom recoding rules in the main interface")
            
        # Privacy-utility estimation
        privacy_level = "High" if len(selected_columns) > 2 else "Medium"
        utility_level = "Medium" if auto_generate_rules else "High"
        
        st.sidebar.info(
            f"🔒 **Privacy Level**: {privacy_level}\n\n"
            f"📊 **Utility Level**: {utility_level}\n\n"
            f"Global recoding will be applied to {len(selected_columns)} column(s)."
        )
        
        return {
            'selected_columns': selected_columns,
            'auto_generate_rules': auto_generate_rules,
            'numerical_binning': numerical_binning,
            'num_bins': num_bins,
            'recoding_rules': recoding_rules
        }
        
    def anonymize(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply global recoding anonymization to the dataset."""
        if not config:
            return data.copy(), {"error": "No configuration provided"}
            
        selected_columns = config.get('selected_columns', [])
        if not selected_columns:
            return data.copy(), {"error": "No columns selected for global recoding"}
            
        # Store configuration
        self.selected_columns = selected_columns
        self.auto_generate_rules = config.get('auto_generate_rules', True)
        self.numerical_binning = config.get('numerical_binning', True)
        self.num_bins = config.get('num_bins', 5)
        
        anonymized_data = data.copy()
        metrics = {
            "columns_processed": [],
            "total_recodings": 0,
            "recoding_statistics": {},
            "generated_rules": {}
        }
        
        try:
            # Generate or use provided recoding rules
            if self.auto_generate_rules:
                self.recoding_rules = self._generate_auto_rules(
                    data, selected_columns, self.numerical_binning, self.num_bins
                )
            else:
                self.recoding_rules = config.get('recoding_rules', {})
            
            # Apply recoding to each column
            for column in selected_columns:
                if column not in data.columns:
                    continue
                    
                if column in self.recoding_rules:
                    original_values = data[column].copy()
                    recoded_values, column_metrics = self._apply_recoding(
                        original_values, self.recoding_rules[column]
                    )
                    
                    anonymized_data[column] = recoded_values
                    metrics["columns_processed"].append(column)
                    metrics["total_recodings"] += column_metrics["recodings_applied"]
                    metrics["recoding_statistics"][column] = column_metrics
                    metrics["generated_rules"][column] = self.recoding_rules[column]
            
            # Calculate overall metrics
            metrics.update(self._calculate_privacy_utility_metrics(data, anonymized_data, selected_columns))
            
            return anonymized_data, metrics
            
        except Exception as e:
            return data.copy(), {"error": f"Global recoding failed: {str(e)}"}
    
    def _generate_auto_rules(self, data: pd.DataFrame, columns: List[str], 
                           numerical_binning: bool, num_bins: int) -> Dict[str, Dict[Any, str]]:
        """Automatically generate recoding rules based on data characteristics."""
        rules = {}
        
        for column in columns:
            if column not in data.columns:
                continue
                
            col_data = data[column].dropna()
            if len(col_data) == 0:
                continue
            
            column_rules = {}
            
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(col_data) and numerical_binning:
                # Create bins for numerical data
                try:
                    bins = pd.cut(col_data, bins=num_bins, duplicates='drop', include_lowest=True)
                    bin_labels = bins.cat.categories
                    
                    for value in col_data.unique():
                        try:
                            bin_assignment = pd.cut([value], bins=bin_labels, include_lowest=True)[0]
                            if pd.notna(bin_assignment):
                                column_rules[value] = f"Range_{bin_assignment}"
                        except:
                            column_rules[value] = f"Range_Other"
                            
                except Exception:
                    # Fallback to quantile-based binning
                    quantiles = np.linspace(0, 1, num_bins + 1)
                    thresholds = col_data.quantile(quantiles).unique()
                    
                    for value in col_data.unique():
                        bin_idx = np.digitize([value], thresholds)[0] - 1
                        bin_idx = max(0, min(bin_idx, num_bins - 1))
                        column_rules[value] = f"Range_{bin_idx + 1}"
            
            else:
                # Handle categorical or non-binned numerical data
                unique_values = col_data.unique()
                
                if len(unique_values) <= 10:
                    # Few unique values - group by frequency
                    value_counts = col_data.value_counts()
                    
                    if len(value_counts) <= 5:
                        # Very few values - keep as is but generalize names
                        for i, value in enumerate(value_counts.index):
                            column_rules[value] = f"Category_{i + 1}"
                    else:
                        # Group by frequency quartiles
                        freq_quartiles = value_counts.quantile([0.25, 0.5, 0.75]).values
                        
                        for value in unique_values:
                            freq = value_counts[value]
                            if freq >= freq_quartiles[2]:
                                column_rules[value] = "High_Frequency"
                            elif freq >= freq_quartiles[1]:
                                column_rules[value] = "Medium_Frequency"
                            elif freq >= freq_quartiles[0]:
                                column_rules[value] = "Low_Frequency"
                            else:
                                column_rules[value] = "Very_Low_Frequency"
                
                else:
                    # Many unique values - use more aggressive grouping
                    if pd.api.types.is_string_dtype(col_data):
                        # String data - group by length or first character
                        for value in unique_values:
                            if isinstance(value, str):
                                if len(value) <= 5:
                                    column_rules[value] = "Short_String"
                                elif len(value) <= 15:
                                    column_rules[value] = "Medium_String"
                                else:
                                    column_rules[value] = "Long_String"
                            else:
                                column_rules[value] = "Other"
                    else:
                        # Other data types - create general categories
                        sorted_values = sorted(unique_values)
                        chunk_size = len(sorted_values) // 5
                        
                        for i, value in enumerate(sorted_values):
                            group_idx = min(i // max(1, chunk_size), 4)
                            column_rules[value] = f"Group_{group_idx + 1}"
            
            rules[column] = column_rules
        
        return rules
    
    def _apply_recoding(self, values: pd.Series, rules: Dict[Any, str]) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply recoding rules to a single column."""
        recoded_values = values.copy()
        recodings_applied = 0
        
        # Apply recoding rules
        for original_value, new_value in rules.items():
            mask = values == original_value
            if mask.any():
                recoded_values[mask] = new_value
                recodings_applied += mask.sum()
        
        # Handle values not in rules
        unmapped_mask = ~values.isin(rules.keys()) & values.notna()
        if unmapped_mask.any():
            recoded_values[unmapped_mask] = "Other"
            recodings_applied += unmapped_mask.sum()
        
        metrics = {
            "recodings_applied": int(recodings_applied),
            "total_values": len(values),
            "recoding_percentage": (recodings_applied / len(values)) * 100,
            "unique_original": len(values.unique()),
            "unique_recoded": len(recoded_values.unique()),
            "reduction_ratio": len(recoded_values.unique()) / len(values.unique()) if len(values.unique()) > 0 else 1
        }
        
        return recoded_values, metrics
    
    def _calculate_privacy_utility_metrics(self, original: pd.DataFrame, anonymized: pd.DataFrame, 
                                         columns: List[str]) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for global recoding."""
        metrics = {}
        
        try:
            # Privacy metrics (based on information reduction)
            total_reduction = 0
            total_columns = 0
            
            for col in columns:
                if col in original.columns and col in anonymized.columns:
                    orig_unique = len(original[col].unique())
                    anon_unique = len(anonymized[col].unique())
                    
                    if orig_unique > 0:
                        reduction = 1 - (anon_unique / orig_unique)
                        total_reduction += reduction
                        total_columns += 1
            
            privacy_score = (total_reduction / total_columns) * 100 if total_columns > 0 else 0
            
            # Utility metrics (distribution preservation)
            utility_scores = []
            for col in columns:
                if col in original.columns and col in anonymized.columns:
                    # For categorical data, compare distribution similarity
                    orig_dist = original[col].value_counts(normalize=True)
                    anon_dist = anonymized[col].value_counts(normalize=True)
                    
                    # Calculate distribution overlap
                    common_values = set(orig_dist.index) & set(anon_dist.index)
                    if common_values:
                        overlap_score = sum(min(orig_dist.get(val, 0), anon_dist.get(val, 0)) 
                                          for val in common_values) * 100
                        utility_scores.append(overlap_score)
            
            utility_score = np.mean(utility_scores) if utility_scores else 0
            
            metrics.update({
                "privacy_score": round(privacy_score, 2),
                "utility_score": round(utility_score, 2),
                "information_reduction": round(privacy_score, 2),
                "distribution_preservation": round(utility_score, 2)
            })
            
        except Exception as e:
            metrics.update({
                "privacy_score": 0,
                "utility_score": 0,
                "error": f"Metrics calculation failed: {str(e)}"
            })
        
        return metrics
    
    def build_config_export(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            "plugin_name": self.get_name(),
            "plugin_category": self.get_category(),
            "selected_columns": self.selected_columns,
            "auto_generate_rules": self.auto_generate_rules,
            "numerical_binning": self.numerical_binning,
            "num_bins": self.num_bins,
            "recoding_rules": self.recoding_rules
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import and apply configuration."""
        self.selected_columns = config.get("selected_columns", [])
        self.auto_generate_rules = config.get("auto_generate_rules", True)
        self.numerical_binning = config.get("numerical_binning", True)
        self.num_bins = config.get("num_bins", 5)
        self.recoding_rules = config.get("recoding_rules", {})


def get_plugin():
    """Factory function to create plugin instance."""
    return GlobalRecodingAnonymizer()
