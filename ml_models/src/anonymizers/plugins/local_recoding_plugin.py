"""
Local Recoding Anonymization Plugin

This plugin implements local recoding anonymization, which applies different
recoding rules to different subsets or records based on local characteristics,
providing more flexible and context-aware anonymization than global recoding.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from ..base_anonymizer import Anonymizer


class LocalRecodingAnonymizer(Anonymizer):
    """
    Local Recoding anonymization implementation.
    
    Local recoding applies different recoding strategies to different subsets
    of data based on local patterns, characteristics, or grouping criteria.
    This provides more nuanced anonymization than global approaches.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Local Recoding"
        self.category = "Data Transformation"
        self.selected_columns = []
        self.grouping_strategy = "automatic"
        self.group_column = None
        self.recoding_rules = {}
        self.adaptive_recoding = True
        self.min_group_size = 3
        
    def get_name(self) -> str:
        return self.name
        
    def get_category(self) -> str:
        return self.category
        
    def get_sidebar_ui(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Streamlit sidebar UI for local recoding configuration."""
        st.sidebar.markdown("### Local Recoding Configuration")
        
        # Column selection
        available_columns = data.columns.tolist()
        if not available_columns:
            st.sidebar.warning("No columns available for local recoding")
            return {}
            
        selected_columns = st.sidebar.multiselect(
            "Select columns for recoding:",
            options=available_columns,
            default=available_columns[:min(3, len(available_columns))],
            help="Choose which columns to apply local recoding to"
        )
        
        if not selected_columns:
            return {}
        
        # Grouping strategy
        grouping_strategy = st.sidebar.selectbox(
            "Grouping strategy:",
            options=["automatic", "column_based", "cluster_based"],
            index=0,
            help="How to create local groups for different recoding rules"
        )
        
        group_column = None
        if grouping_strategy == "column_based":
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_columns:
                group_column = st.sidebar.selectbox(
                    "Group by column:",
                    options=categorical_columns,
                    help="Column to use for creating local groups"
                )
            else:
                st.sidebar.warning("No categorical columns available for grouping")
                grouping_strategy = "automatic"
        
        # Additional options
        adaptive_recoding = st.sidebar.checkbox(
            "Adaptive recoding",
            value=True,
            help="Adapt recoding rules based on local data characteristics"
        )
        
        min_group_size = st.sidebar.slider(
            "Minimum group size:",
            min_value=2,
            max_value=min(20, len(data) // 5) if len(data) > 10 else 5,
            value=3,
            help="Minimum number of records per local group"
        )
        
        # Advanced options
        with st.sidebar.expander("Advanced Options"):
            cluster_features = []
            if grouping_strategy == "cluster_based":
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    cluster_features = st.multiselect(
                        "Features for clustering:",
                        options=numeric_columns,
                        default=numeric_columns[:min(3, len(numeric_columns))],
                        help="Numeric columns to use for automatic clustering"
                    )
        
        # Privacy-utility estimation
        privacy_level = "High" if grouping_strategy == "cluster_based" else "Medium"
        utility_level = "High" if adaptive_recoding else "Medium"
        
        st.sidebar.info(
            f"🔒 **Privacy Level**: {privacy_level}\n\n"
            f"📊 **Utility Level**: {utility_level}\n\n"
            f"Strategy: {grouping_strategy.replace('_', ' ').title()}\n"
            f"Columns: {len(selected_columns)}"
        )
        
        return {
            'selected_columns': selected_columns,
            'grouping_strategy': grouping_strategy,
            'group_column': group_column,
            'adaptive_recoding': adaptive_recoding,
            'min_group_size': min_group_size,
            'cluster_features': cluster_features if grouping_strategy == "cluster_based" else []
        }
        
    def anonymize(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply local recoding anonymization to the dataset."""
        if not config:
            return data.copy(), {"error": "No configuration provided"}
            
        selected_columns = config.get('selected_columns', [])
        if not selected_columns:
            return data.copy(), {"error": "No columns selected for local recoding"}
            
        # Store configuration
        self.selected_columns = selected_columns
        self.grouping_strategy = config.get('grouping_strategy', 'automatic')
        self.group_column = config.get('group_column')
        self.adaptive_recoding = config.get('adaptive_recoding', True)
        self.min_group_size = config.get('min_group_size', 3)
        cluster_features = config.get('cluster_features', [])
        
        anonymized_data = data.copy()
        metrics = {
            "columns_processed": [],
            "total_groups": 0,
            "total_recodings": 0,
            "group_statistics": {},
            "recoding_statistics": {}
        }
        
        try:
            # Create local groups
            groups = self._create_local_groups(data, self.grouping_strategy, 
                                             self.group_column, cluster_features)
            
            metrics["total_groups"] = len(groups)
            
            # Apply local recoding to each group
            for group_id, group_indices in groups.items():
                if len(group_indices) < self.min_group_size:
                    continue  # Skip small groups
                
                group_data = data.iloc[group_indices]
                group_metrics = {"group_size": len(group_indices), "columns_recoded": {}}
                
                # Generate local recoding rules for this group
                local_rules = self._generate_local_rules(group_data, selected_columns)
                
                # Apply recoding to each column in the group
                for column in selected_columns:
                    if column not in data.columns or column not in local_rules:
                        continue
                    
                    original_values = group_data[column].copy()
                    recoded_values, column_metrics = self._apply_local_recoding(
                        original_values, local_rules[column]
                    )
                    
                    # Update the anonymized data
                    anonymized_data.iloc[group_indices, anonymized_data.columns.get_loc(column)] = recoded_values
                    
                    # Update metrics
                    group_metrics["columns_recoded"][column] = column_metrics
                    metrics["total_recodings"] += column_metrics["recodings_applied"]
                
                metrics["group_statistics"][str(group_id)] = group_metrics
                
                if column in selected_columns:
                    if column not in metrics["recoding_statistics"]:
                        metrics["recoding_statistics"][column] = []
                    metrics["recoding_statistics"][column].append(group_metrics["columns_recoded"].get(column, {}))
            
            metrics["columns_processed"] = selected_columns
            
            # Calculate overall metrics
            metrics.update(self._calculate_privacy_utility_metrics(data, anonymized_data, selected_columns))
            
            return anonymized_data, metrics
            
        except Exception as e:
            return data.copy(), {"error": f"Local recoding failed: {str(e)}"}
    
    def _create_local_groups(self, data: pd.DataFrame, strategy: str, 
                           group_column: Optional[str], cluster_features: List[str]) -> Dict[str, List[int]]:
        """Create local groups based on the specified strategy."""
        groups = {}
        
        if strategy == "column_based" and group_column and group_column in data.columns:
            # Group by specified column
            for group_value in data[group_column].unique():
                if pd.isna(group_value):
                    continue
                group_indices = data[data[group_column] == group_value].index.tolist()
                groups[f"group_{group_value}"] = group_indices
                
        elif strategy == "cluster_based" and cluster_features:
            # Group by clustering
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Prepare clustering data
                cluster_data = data[cluster_features].copy()
                cluster_data = cluster_data.fillna(cluster_data.mean())
                
                # Determine number of clusters
                n_clusters = min(max(2, len(data) // (self.min_group_size * 2)), 10)
                
                # Scale and cluster
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Create groups from clusters
                for cluster_id in np.unique(cluster_labels):
                    group_indices = data[cluster_labels == cluster_id].index.tolist()
                    groups[f"cluster_{cluster_id}"] = group_indices
                    
            except ImportError:
                # Fallback to automatic grouping if sklearn not available
                strategy = "automatic"
            except Exception:
                # Fallback to automatic grouping on any error
                strategy = "automatic"
        
        if strategy == "automatic" or not groups:
            # Automatic grouping based on data characteristics
            group_size = max(self.min_group_size, len(data) // 10)
            
            # Simple sequential grouping
            indices = data.index.tolist()
            for i in range(0, len(indices), group_size):
                group_indices = indices[i:i + group_size]
                if len(group_indices) >= self.min_group_size:
                    groups[f"auto_group_{i // group_size}"] = group_indices
        
        return groups
    
    def _generate_local_rules(self, group_data: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[Any, str]]:
        """Generate recoding rules specific to a local group."""
        rules = {}
        
        for column in columns:
            if column not in group_data.columns:
                continue
                
            col_data = group_data[column].dropna()
            if len(col_data) == 0:
                continue
            
            column_rules = {}
            
            # Analyze local characteristics
            unique_values = col_data.unique()
            n_unique = len(unique_values)
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Numerical data - create local bins
                if n_unique > 5:
                    try:
                        # Use local quartiles for binning
                        quartiles = col_data.quantile([0, 0.25, 0.5, 0.75, 1.0])
                        
                        for value in unique_values:
                            if value <= quartiles[0.25]:
                                column_rules[value] = "Local_Low"
                            elif value <= quartiles[0.5]:
                                column_rules[value] = "Local_Medium_Low"
                            elif value <= quartiles[0.75]:
                                column_rules[value] = "Local_Medium_High"
                            else:
                                column_rules[value] = "Local_High"
                    except:
                        # Simple binning fallback
                        sorted_values = sorted(unique_values)
                        bin_size = len(sorted_values) // 3
                        for i, value in enumerate(sorted_values):
                            bin_idx = min(i // max(1, bin_size), 2)
                            column_rules[value] = f"Local_Bin_{bin_idx + 1}"
                else:
                    # Few unique values - keep but generalize
                    for i, value in enumerate(unique_values):
                        column_rules[value] = f"Local_Value_{i + 1}"
            
            else:
                # Categorical data
                if n_unique <= 3:
                    # Very few categories - keep structure
                    for i, value in enumerate(unique_values):
                        column_rules[value] = f"Local_Cat_{i + 1}"
                elif n_unique <= 8:
                    # Moderate categories - group by frequency
                    value_counts = col_data.value_counts()
                    median_freq = value_counts.median()
                    
                    for value in unique_values:
                        freq = value_counts[value]
                        if freq >= median_freq:
                            column_rules[value] = "Local_Common"
                        else:
                            column_rules[value] = "Local_Rare"
                else:
                    # Many categories - aggressive grouping
                    value_counts = col_data.value_counts()
                    top_values = value_counts.head(3).index.tolist()
                    
                    for value in unique_values:
                        if value in top_values:
                            column_rules[value] = f"Local_Top_{top_values.index(value) + 1}"
                        else:
                            column_rules[value] = "Local_Other"
            
            rules[column] = column_rules
        
        return rules
    
    def _apply_local_recoding(self, values: pd.Series, rules: Dict[Any, str]) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply local recoding rules to values."""
        recoded_values = values.copy()
        recodings_applied = 0
        
        # Apply recoding rules
        for original_value, new_value in rules.items():
            mask = values == original_value
            if mask.any():
                recoded_values[mask] = new_value
                recodings_applied += mask.sum()
        
        # Handle unmapped values
        unmapped_mask = ~values.isin(rules.keys()) & values.notna()
        if unmapped_mask.any():
            recoded_values[unmapped_mask] = "Local_Unmapped"
            recodings_applied += unmapped_mask.sum()
        
        metrics = {
            "recodings_applied": int(recodings_applied),
            "total_values": len(values),
            "recoding_percentage": (recodings_applied / len(values)) * 100,
            "unique_original": len(values.unique()),
            "unique_recoded": len(recoded_values.unique()),
            "rules_used": len(rules)
        }
        
        return recoded_values, metrics
    
    def _calculate_privacy_utility_metrics(self, original: pd.DataFrame, anonymized: pd.DataFrame, 
                                         columns: List[str]) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for local recoding."""
        metrics = {}
        
        try:
            # Privacy metrics (information diversity)
            diversity_scores = []
            for col in columns:
                if col in original.columns and col in anonymized.columns:
                    orig_unique = len(original[col].unique())
                    anon_unique = len(anonymized[col].unique())
                    
                    if orig_unique > 0:
                        diversity_reduction = 1 - (anon_unique / orig_unique)
                        diversity_scores.append(diversity_reduction)
            
            privacy_score = np.mean(diversity_scores) * 100 if diversity_scores else 0
            
            # Utility metrics (local pattern preservation)
            utility_scores = []
            for col in columns:
                if col in original.columns and col in anonymized.columns:
                    # Calculate local correlation preservation
                    try:
                        # Use numeric columns for correlation if available
                        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) > 1 and col in numeric_cols:
                            orig_corr = original[numeric_cols].corr()[col].drop(col).abs().mean()
                            anon_corr = anonymized[numeric_cols].corr()[col].drop(col).abs().mean()
                            
                            if not np.isnan(orig_corr) and not np.isnan(anon_corr) and orig_corr > 0:
                                preservation = (anon_corr / orig_corr) * 100
                                utility_scores.append(min(100, preservation))
                        else:
                            # For categorical data, measure distribution similarity
                            orig_dist = original[col].value_counts(normalize=True)
                            anon_dist = anonymized[col].value_counts(normalize=True)
                            
                            # Calculate Jensen-Shannon divergence
                            from scipy.spatial.distance import jensenshannon
                            
                            # Align distributions
                            all_values = set(orig_dist.index) | set(anon_dist.index)
                            orig_aligned = [orig_dist.get(v, 0) for v in all_values]
                            anon_aligned = [anon_dist.get(v, 0) for v in all_values]
                            
                            if sum(orig_aligned) > 0 and sum(anon_aligned) > 0:
                                js_distance = jensenshannon(orig_aligned, anon_aligned)
                                similarity = (1 - js_distance) * 100
                                utility_scores.append(max(0, similarity))
                    except:
                        # Simple fallback
                        orig_entropy = len(original[col].unique())
                        anon_entropy = len(anonymized[col].unique())
                        if orig_entropy > 0:
                            utility_scores.append((anon_entropy / orig_entropy) * 100)
            
            utility_score = np.mean(utility_scores) if utility_scores else 100
            
            metrics.update({
                "privacy_score": round(privacy_score, 2),
                "utility_score": round(utility_score, 2),
                "local_diversity": round(privacy_score, 2),
                "pattern_preservation": round(utility_score, 2)
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
            "grouping_strategy": self.grouping_strategy,
            "group_column": self.group_column,
            "adaptive_recoding": self.adaptive_recoding,
            "min_group_size": self.min_group_size,
            "recoding_rules": self.recoding_rules
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import and apply configuration."""
        self.selected_columns = config.get("selected_columns", [])
        self.grouping_strategy = config.get("grouping_strategy", "automatic")
        self.group_column = config.get("group_column")
        self.adaptive_recoding = config.get("adaptive_recoding", True)
        self.min_group_size = config.get("min_group_size", 3)
        self.recoding_rules = config.get("recoding_rules", {})


def get_plugin():
    """Factory function to create plugin instance."""
    return LocalRecodingAnonymizer()
