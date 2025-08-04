"""
Top-Bottom Coding Anonymization Plugin

This plugin implements top-bottom coding anonymization, which replaces extreme
values (top and bottom percentiles) with threshold values to reduce the risk
of outlier-based identification while preserving the general distribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..base_anonymizer import Anonymizer


class TopBottomCodingAnonymizer(Anonymizer):
    """
    Top-Bottom Coding anonymization implementation.
    
    This technique replaces values above a top percentile threshold with the
    threshold value, and values below a bottom percentile threshold with that
    threshold value, effectively "censoring" extreme values.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Top-Bottom Coding"
        self.category = "Data Transformation"
        self.top_percentile = 95
        self.bottom_percentile = 5
        self.selected_columns = []
        self.apply_per_group = False
        self.group_column = None
        
    def get_name(self) -> str:
        return self.name
        
    def get_category(self) -> str:
        return self.category
        
    def get_sidebar_ui(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Streamlit sidebar UI for top-bottom coding configuration."""
        st.sidebar.markdown("### Top-Bottom Coding Configuration")
        
        # Column selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            st.sidebar.warning("No numeric columns available for top-bottom coding")
            return {}
            
        selected_columns = st.sidebar.multiselect(
            "Select numeric columns for coding:",
            options=numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            help="Choose which numeric columns to apply top-bottom coding to"
        )
        
        # Percentile thresholds
        col1, col2 = st.sidebar.columns(2)
        with col1:
            bottom_percentile = st.number_input(
                "Bottom percentile:",
                min_value=0.0,
                max_value=49.0,
                value=5.0,
                step=1.0,
                help="Values below this percentile will be replaced"
            )
        
        with col2:
            top_percentile = st.number_input(
                "Top percentile:",
                min_value=51.0,
                max_value=100.0,
                value=95.0,
                step=1.0,
                help="Values above this percentile will be replaced"
            )
        
        # Validation
        if bottom_percentile >= top_percentile:
            st.sidebar.error("Bottom percentile must be less than top percentile")
            return {}
        
        # Group-based coding option
        apply_per_group = st.sidebar.checkbox(
            "Apply per group",
            value=False,
            help="Apply coding separately for each group instead of globally"
        )
        
        group_column = None
        if apply_per_group:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_columns:
                group_column = st.sidebar.selectbox(
                    "Group by column:",
                    options=categorical_columns,
                    help="Column to group by for separate coding"
                )
            else:
                st.sidebar.warning("No categorical columns available for grouping")
                apply_per_group = False
        
        # Privacy-utility info
        privacy_level = "High" if (top_percentile - bottom_percentile) < 80 else "Medium" if (top_percentile - bottom_percentile) < 90 else "Low"
        utility_level = "Low" if (top_percentile - bottom_percentile) < 80 else "Medium" if (top_percentile - bottom_percentile) < 90 else "High"
        
        st.sidebar.info(
            f"🔒 **Privacy Level**: {privacy_level}\n\n"
            f"📊 **Utility Level**: {utility_level}\n\n"
            f"Coding range: {bottom_percentile}% - {top_percentile}%\n"
            f"Values outside this range will be replaced with threshold values."
        )
        
        return {
            'selected_columns': selected_columns,
            'top_percentile': top_percentile,
            'bottom_percentile': bottom_percentile,
            'apply_per_group': apply_per_group,
            'group_column': group_column
        }
        
    def anonymize(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply top-bottom coding anonymization to the dataset."""
        if not config:
            return data.copy(), {"error": "No configuration provided"}
            
        selected_columns = config.get('selected_columns', [])
        if not selected_columns:
            return data.copy(), {"error": "No columns selected for top-bottom coding"}
            
        # Store configuration
        self.selected_columns = selected_columns
        self.top_percentile = config.get('top_percentile', 95)
        self.bottom_percentile = config.get('bottom_percentile', 5)
        self.apply_per_group = config.get('apply_per_group', False)
        self.group_column = config.get('group_column')
        
        anonymized_data = data.copy()
        metrics = {
            "columns_processed": [],
            "total_values_coded": 0,
            "coding_statistics": {},
            "thresholds": {}
        }
        
        try:
            if self.apply_per_group and self.group_column and self.group_column in data.columns:
                # Apply coding per group
                for group_value in data[self.group_column].unique():
                    if pd.isna(group_value):
                        continue
                        
                    group_mask = data[self.group_column] == group_value
                    group_data = data[group_mask]
                    
                    for column in selected_columns:
                        if column not in data.columns:
                            continue
                            
                        coded_values, column_metrics = self._apply_top_bottom_coding(
                            group_data[column], self.bottom_percentile, self.top_percentile
                        )
                        
                        anonymized_data.loc[group_mask, column] = coded_values
                        
                        # Update metrics
                        key = f"{column}_{group_value}"
                        metrics["coding_statistics"][key] = column_metrics
                        metrics["total_values_coded"] += column_metrics["values_coded"]
                        
                        if column not in metrics["thresholds"]:
                            metrics["thresholds"][column] = {}
                        metrics["thresholds"][column][str(group_value)] = {
                            "bottom_threshold": column_metrics["bottom_threshold"],
                            "top_threshold": column_metrics["top_threshold"]
                        }
            else:
                # Apply coding globally
                for column in selected_columns:
                    if column not in data.columns:
                        continue
                        
                    original_values = data[column].copy()
                    coded_values, column_metrics = self._apply_top_bottom_coding(
                        original_values, self.bottom_percentile, self.top_percentile
                    )
                    
                    anonymized_data[column] = coded_values
                    metrics["columns_processed"].append(column)
                    metrics["total_values_coded"] += column_metrics["values_coded"]
                    metrics["coding_statistics"][column] = column_metrics
                    metrics["thresholds"][column] = {
                        "bottom_threshold": column_metrics["bottom_threshold"],
                        "top_threshold": column_metrics["top_threshold"]
                    }
            
            # Calculate overall metrics
            metrics.update(self._calculate_privacy_utility_metrics(data, anonymized_data, selected_columns))
            
            return anonymized_data, metrics
            
        except Exception as e:
            return data.copy(), {"error": f"Top-bottom coding failed: {str(e)}"}
    
    def _apply_top_bottom_coding(self, values: pd.Series, bottom_percentile: float, 
                                top_percentile: float) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply top-bottom coding to a single column."""
        # Handle missing values
        mask = values.notna()
        if not mask.any():
            return values, {
                "values_coded": 0,
                "bottom_threshold": None,
                "top_threshold": None,
                "bottom_coded": 0,
                "top_coded": 0
            }
        
        valid_values = values[mask]
        
        # Calculate thresholds
        bottom_threshold = np.percentile(valid_values, bottom_percentile)
        top_threshold = np.percentile(valid_values, top_percentile)
        
        # Apply coding
        coded_values = values.copy()
        
        # Count values to be coded
        bottom_coded = (valid_values < bottom_threshold).sum()
        top_coded = (valid_values > top_threshold).sum()
        
        # Apply thresholds
        coded_values[coded_values < bottom_threshold] = bottom_threshold
        coded_values[coded_values > top_threshold] = top_threshold
        
        metrics = {
            "values_coded": bottom_coded + top_coded,
            "bottom_threshold": float(bottom_threshold),
            "top_threshold": float(top_threshold),
            "bottom_coded": int(bottom_coded),
            "top_coded": int(top_coded),
            "total_values": len(valid_values),
            "coding_percentage": ((bottom_coded + top_coded) / len(valid_values)) * 100
        }
        
        return coded_values, metrics
    
    def _calculate_privacy_utility_metrics(self, original: pd.DataFrame, anonymized: pd.DataFrame, 
                                         columns: List[str]) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for top-bottom coding."""
        metrics = {}
        
        try:
            # Privacy metrics (based on value changes)
            total_changes = 0
            total_records = len(original)
            
            for col in columns:
                changes = (original[col] != anonymized[col]).sum()
                total_changes += changes
            
            privacy_score = min(100, (total_changes / (total_records * len(columns))) * 100)
            
            # Utility metrics (distribution preservation)
            utility_scores = []
            for col in columns:
                if col in original.columns and col in anonymized.columns:
                    orig_std = original[col].std()
                    anon_std = anonymized[col].std()
                    
                    if orig_std > 0:
                        # Standard deviation preservation
                        std_preservation = min(100, (anon_std / orig_std) * 100)
                        utility_scores.append(std_preservation)
            
            utility_score = np.mean(utility_scores) if utility_scores else 100
            
            # Information loss calculation
            information_loss = 100 - utility_score
            
            metrics.update({
                "privacy_score": round(privacy_score, 2),
                "utility_score": round(utility_score, 2),
                "information_loss": round(information_loss, 2),
                "records_modified": total_changes,
                "modification_rate": round((total_changes / (total_records * len(columns))) * 100, 2)
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
            "top_percentile": self.top_percentile,
            "bottom_percentile": self.bottom_percentile,
            "apply_per_group": self.apply_per_group,
            "group_column": self.group_column
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import and apply configuration."""
        self.selected_columns = config.get("selected_columns", [])
        self.top_percentile = config.get("top_percentile", 95)
        self.bottom_percentile = config.get("bottom_percentile", 5)
        self.apply_per_group = config.get("apply_per_group", False)
        self.group_column = config.get("group_column")


def get_plugin():
    """Factory function to create plugin instance."""
    return TopBottomCodingAnonymizer()
