"""
Mondrian Anonymization Plugin for Data Anonymization

This plugin implements the Mondrian algorithm for k-anonymity, which recursively
partitions data into groups (buckets) such that each group contains at least k records.
The algorithm uses a top-down approach, selecting the best attribute and split point
at each step to minimize information loss.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import LabelEncoder
import warnings

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class MondrianAnonymizationPlugin(Anonymizer):
    """
    Mondrian k-anonymity plugin that implements the Mondrian partitioning algorithm.
    
    The Mondrian algorithm works by:
    1. Starting with the entire dataset
    2. Finding the best dimension and split point to partition the data
    3. Recursively applying the same process to each partition
    4. Stopping when partitions contain fewer than k records
    5. Generalizing values within each final partition    """
    
    def __init__(self):
        super().__init__()
        self.name = "Advanced Mondrian k-Anonymity"
        self.description = "Advanced recursive partitioning algorithm for k-anonymity with minimal information loss"
        
        # k-anonymity parameters
        self.k_value = 3
        self.allow_suppression = True
        self.suppression_threshold = 0.05
        
        # Mondrian algorithm parameters
        self.max_depth = 10
        self.min_leaf_size = None  # Will be set to k_value
        self.split_strategy = "range"  # range, median, entropy
        self.dimension_selection = "normalized_width"  # normalized_width, entropy, random
        
        # Generalization parameters
        self.categorical_hierarchy = {}
        self.numerical_granularity = {}
        self.preserve_format = True
        self.generalization_strategy = "range"  # range, mean, mode
        
        # Quality metrics
        self.information_loss_metric = "normalized_certainty_penalty"
        self.calculate_precision = True
        self.track_generalization_height = True
        
        # Internal state
        self.partitions = []
        self.generalization_hierarchies = {}
        self.label_encoders = {}
        self.anonymization_tree = None
        self.information_loss = 0.0
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Clustering & Grouping"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Mondrian k-anonymity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"📊 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Mondrian k-Anonymity"):
            st.markdown("""
            **Mondrian k-Anonymity** uses a recursive partitioning algorithm to create
            groups of at least k similar records. It minimizes information loss by
            carefully selecting split points and dimensions.
            
            **Key Features:**
            - Top-down recursive partitioning
            - Optimal dimension and split selection
            - Minimal information loss
            - Handles both numerical and categorical data
            """)
        
        # Basic k-anonymity parameters
        k_value = st.sidebar.slider(
            "k-anonymity value",
            min_value=2, max_value=20, value=3,
            key=f"{unique_key_prefix}_k_value",
            help="Minimum number of records in each equivalence class"
        )
        
        # Algorithm parameters
        st.sidebar.subheader("🔧 Algorithm Parameters")
        
        max_depth = st.sidebar.slider(
            "Maximum Tree Depth",
            min_value=3, max_value=20, value=10,
            key=f"{unique_key_prefix}_max_depth",
            help="Maximum depth of the Mondrian partitioning tree"
        )
        
        split_strategy = st.sidebar.selectbox(
            "Split Strategy",
            options=["range", "median", "entropy"],
            key=f"{unique_key_prefix}_split_strategy",
            help="Method for determining split points"
        )
        
        dimension_selection = st.sidebar.selectbox(
            "Dimension Selection",
            options=["normalized_width", "entropy", "random"],
            key=f"{unique_key_prefix}_dimension_selection",
            help="Method for selecting which dimension to split on"
        )
        
        # Generalization parameters
        with st.sidebar.expander("📈 Generalization Settings"):
            generalization_strategy = st.sidebar.selectbox(
                "Generalization Strategy",
                options=["range", "mean", "mode"],
                key=f"{unique_key_prefix}_generalization_strategy",
                help="How to generalize values within partitions"
            )
            
            preserve_format = st.sidebar.checkbox(
                "Preserve Data Format",
                value=True,
                key=f"{unique_key_prefix}_preserve_format",
                help="Try to maintain original data format"
            )
        
        # Quality settings
        with st.sidebar.expander("📊 Quality Settings"):
            allow_suppression = st.sidebar.checkbox(
                "Allow Record Suppression",
                value=True,
                key=f"{unique_key_prefix}_allow_suppression",
                help="Allow suppressing records that can't be k-anonymized"
            )
            
            if allow_suppression:
                suppression_threshold = st.sidebar.slider(
                    "Suppression Threshold",
                    min_value=0.0, max_value=0.2, value=0.05, step=0.01,
                    key=f"{unique_key_prefix}_suppression_threshold",
                    help="Maximum fraction of records that can be suppressed"
                )
            else:
                suppression_threshold = 0.0
        
        return {
            "k_value": k_value,
            "max_depth": max_depth,
            "split_strategy": split_strategy,
            "dimension_selection": dimension_selection,
            "generalization_strategy": generalization_strategy,
            "preserve_format": preserve_format,
            "allow_suppression": allow_suppression,
            "suppression_threshold": suppression_threshold
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Mondrian k-anonymity algorithm to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.k_value = parameters.get("k_value", 3)
            self.max_depth = parameters.get("max_depth", 10)
            self.split_strategy = parameters.get("split_strategy", "range")
            self.dimension_selection = parameters.get("dimension_selection", "normalized_width")
            self.generalization_strategy = parameters.get("generalization_strategy", "range")
            self.preserve_format = parameters.get("preserve_format", True)
            self.allow_suppression = parameters.get("allow_suppression", True)
            self.suppression_threshold = parameters.get("suppression_threshold", 0.05)
            
            self.min_leaf_size = self.k_value
            
            logger.info(f"Starting Mondrian k-anonymity with k={self.k_value}")
            
            # Prepare data for Mondrian algorithm
            prepared_data = self._prepare_data(df_input, sa_col)
            
            # Build Mondrian partitioning tree
            self.anonymization_tree = self._build_mondrian_tree(prepared_data)
            
            # Extract partitions from tree
            self.partitions = self._extract_partitions(self.anonymization_tree, prepared_data)
            
            # Apply generalization to each partition
            anonymized_data = self._apply_generalization(prepared_data, self.partitions)
            
            # Handle suppression if needed
            if self.allow_suppression:
                anonymized_data = self._apply_suppression(anonymized_data)
            
            # Calculate information loss
            self.information_loss = self._calculate_information_loss(prepared_data, anonymized_data)
            
            # Convert back to original format
            result = self._convert_to_original_format(anonymized_data, df_input)
            
            processing_time = time.time() - start_time
            logger.info(f"Mondrian anonymization completed in {processing_time:.2f}s, IL: {self.information_loss:.4f}")
            
            # Add metadata
            result.attrs['mondrian_k'] = self.k_value
            result.attrs['mondrian_partitions'] = len(self.partitions)
            result.attrs['information_loss'] = self.information_loss
            result.attrs['processing_time'] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Mondrian anonymization: {str(e)}")
            st.error(f"Mondrian anonymization failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "k_value": st.session_state.get(f"{unique_key_prefix}_k_value", 3),
            "max_depth": st.session_state.get(f"{unique_key_prefix}_max_depth", 10),
            "split_strategy": st.session_state.get(f"{unique_key_prefix}_split_strategy", "range"),
            "dimension_selection": st.session_state.get(f"{unique_key_prefix}_dimension_selection", "normalized_width"),
            "generalization_strategy": st.session_state.get(f"{unique_key_prefix}_generalization_strategy", "range"),
            "preserve_format": st.session_state.get(f"{unique_key_prefix}_preserve_format", True),
            "allow_suppression": st.session_state.get(f"{unique_key_prefix}_allow_suppression", True),
            "suppression_threshold": st.session_state.get(f"{unique_key_prefix}_suppression_threshold", 0.05)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        if "k_value" in config_params:
            st.session_state[f"{unique_key_prefix}_k_value"] = config_params["k_value"]
        if "max_depth" in config_params:
            st.session_state[f"{unique_key_prefix}_max_depth"] = config_params["max_depth"]
        if "split_strategy" in config_params:
            st.session_state[f"{unique_key_prefix}_split_strategy"] = config_params["split_strategy"]
        if "dimension_selection" in config_params:
            st.session_state[f"{unique_key_prefix}_dimension_selection"] = config_params["dimension_selection"]
        if "generalization_strategy" in config_params:
            st.session_state[f"{unique_key_prefix}_generalization_strategy"] = config_params["generalization_strategy"]
        if "preserve_format" in config_params:
            st.session_state[f"{unique_key_prefix}_preserve_format"] = config_params["preserve_format"]
        if "allow_suppression" in config_params:
            st.session_state[f"{unique_key_prefix}_allow_suppression"] = config_params["allow_suppression"]
        if "suppression_threshold" in config_params:
            st.session_state[f"{unique_key_prefix}_suppression_threshold"] = config_params["suppression_threshold"]

    def _prepare_data(self, df: pd.DataFrame, sa_col: str | None) -> pd.DataFrame:
        """Prepare data for Mondrian algorithm."""
        data = df.copy()
        
        # Encode categorical variables
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if col != sa_col:  # Don't encode sensitive attribute
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        # Normalize numerical data for better partitioning
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != sa_col:
                min_val, max_val = data[col].min(), data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)
                self.numerical_granularity[col] = (min_val, max_val)
        
        return data
    
    def _build_mondrian_tree(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build the Mondrian partitioning tree."""
        def build_node(subset_indices: List[int], depth: int) -> Dict[str, Any]:
            if len(subset_indices) < self.min_leaf_size or depth >= self.max_depth:
                return {
                    'type': 'leaf',
                    'indices': subset_indices,
                    'size': len(subset_indices)
                }
            
            # Find best dimension and split point
            best_dim, best_split = self._find_best_split(data.iloc[subset_indices])
            
            if best_dim is None:
                return {
                    'type': 'leaf',
                    'indices': subset_indices,
                    'size': len(subset_indices)
                }
            
            # Split the data
            left_indices = []
            right_indices = []
            
            for idx in subset_indices:
                if data.iloc[idx][best_dim] <= best_split:
                    left_indices.append(idx)
                else:
                    right_indices.append(idx)
            
            # Ensure both partitions have minimum size
            if len(left_indices) < self.min_leaf_size or len(right_indices) < self.min_leaf_size:
                return {
                    'type': 'leaf',
                    'indices': subset_indices,
                    'size': len(subset_indices)
                }
            
            return {
                'type': 'internal',
                'dimension': best_dim,
                'split_value': best_split,
                'left': build_node(left_indices, depth + 1),
                'right': build_node(right_indices, depth + 1)
            }
        
        return build_node(list(range(len(data))), 0)
    
    def _find_best_split(self, subset: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Find the best dimension and split point for a data subset."""
        best_dim = None
        best_split = None
        best_score = float('-inf')
        
        for col in subset.columns:
            if self.dimension_selection == "normalized_width":
                score = self._calculate_normalized_width(subset[col])
            elif self.dimension_selection == "entropy":
                score = self._calculate_entropy_gain(subset[col])
            else:  # random
                score = np.random.random()
            
            if score > best_score:
                best_score = score
                best_dim = col
                best_split = self._find_split_point(subset[col])
        
        return best_dim, best_split
    
    def _calculate_normalized_width(self, column: pd.Series) -> float:
        """Calculate normalized width for a column."""
        if len(column.unique()) == 1:
            return 0.0
        
        if column.dtype in ['object', 'category']:
            return len(column.unique()) / len(column)
        else:
            return (column.max() - column.min()) / (column.std() + 1e-8)
    
    def _calculate_entropy_gain(self, column: pd.Series) -> float:
        """Calculate entropy gain for a column."""
        unique_values = column.unique()
        if len(unique_values) <= 1:
            return 0.0
        
        entropy = 0.0
        for value in unique_values:
            p = (column == value).sum() / len(column)
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _find_split_point(self, column: pd.Series) -> float:
        """Find the best split point for a column."""
        if self.split_strategy == "median":
            return column.median()
        elif self.split_strategy == "entropy":
            # Find split that maximizes entropy gain
            unique_vals = sorted(column.unique())
            if len(unique_vals) <= 1:
                return unique_vals[0] if unique_vals else 0.0
            
            best_split = unique_vals[len(unique_vals) // 2]
            return best_split
        else:  # range
            return (column.min() + column.max()) / 2
    
    def _extract_partitions(self, tree: Dict[str, Any], data: pd.DataFrame) -> List[List[int]]:
        """Extract leaf partitions from the Mondrian tree."""
        partitions = []
        
        def extract_leaves(node):
            if node['type'] == 'leaf':
                partitions.append(node['indices'])
            else:
                extract_leaves(node['left'])
                extract_leaves(node['right'])
        
        extract_leaves(tree)
        return partitions
    
    def _apply_generalization(self, data: pd.DataFrame, partitions: List[List[int]]) -> pd.DataFrame:
        """Apply generalization to each partition."""
        result = data.copy()
        
        for partition_indices in partitions:
            partition_data = data.iloc[partition_indices]
            
            # Generalize each column within the partition
            for col in data.columns:
                if self.generalization_strategy == "range" and data[col].dtype in ['int64', 'float64']:
                    min_val = partition_data[col].min()
                    max_val = partition_data[col].max()
                    generalized_value = f"[{min_val:.3f}, {max_val:.3f}]"
                    result.loc[partition_indices, col] = generalized_value
                elif self.generalization_strategy == "mean" and data[col].dtype in ['int64', 'float64']:
                    mean_val = partition_data[col].mean()
                    result.loc[partition_indices, col] = mean_val
                else:  # mode for categorical or fallback
                    mode_val = partition_data[col].mode().iloc[0] if len(partition_data[col].mode()) > 0 else partition_data[col].iloc[0]
                    result.loc[partition_indices, col] = mode_val
        
        return result
    
    def _apply_suppression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply record suppression if needed."""
        # Find partitions that are too small
        partition_sizes = []
        for partition in self.partitions:
            partition_sizes.append(len(partition))
        
        # Calculate suppression ratio
        total_records = len(data)
        suppressed_records = sum(1 for size in partition_sizes if size < self.k_value)
        suppression_ratio = suppressed_records / total_records
        
        if suppression_ratio > self.suppression_threshold:
            logger.warning(f"Suppression ratio ({suppression_ratio:.3f}) exceeds threshold ({self.suppression_threshold})")
        
        # Mark suppressed records
        result = data.copy()
        for i, partition in enumerate(self.partitions):
            if len(partition) < self.k_value:
                result.loc[partition, :] = "*SUPPRESSED*"
        
        return result
    
    def _calculate_information_loss(self, original: pd.DataFrame, anonymized: pd.DataFrame) -> float:
        """Calculate information loss using normalized certainty penalty."""
        total_loss = 0.0
        
        for col in original.columns:
            if original[col].dtype in ['int64', 'float64']:
                # Numerical column
                original_range = original[col].max() - original[col].min()
                if original_range > 0:
                    # Calculate average generalization width
                    avg_loss = 0.0
                    for partition in self.partitions:
                        partition_data = original.iloc[partition]
                        partition_range = partition_data[col].max() - partition_data[col].min()
                        avg_loss += (partition_range / original_range) * len(partition)
                    avg_loss /= len(original)
                    total_loss += avg_loss
            else:
                # Categorical column
                unique_original = len(original[col].unique())
                if unique_original > 1:
                    avg_loss = 0.0
                    for partition in self.partitions:
                        partition_data = original.iloc[partition]
                        unique_partition = len(partition_data[col].unique())
                        loss = 1 - (unique_partition / unique_original)
                        avg_loss += loss * len(partition)
                    avg_loss /= len(original)
                    total_loss += avg_loss
        
        return total_loss / len(original.columns)
    
    def _convert_to_original_format(self, data: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Convert anonymized data back to original format."""
        result = data.copy()
        
        # Decode categorical variables if preserve_format is True
        if self.preserve_format:
            for col, encoder in self.label_encoders.items():
                if col in result.columns:
                    try:
                        # Handle generalized values
                        if result[col].dtype == 'object':
                            continue  # Already generalized as strings
                        
                        result[col] = encoder.inverse_transform(result[col].astype(int))
                    except Exception:
                        pass  # Keep as is if transformation fails
        
        # Denormalize numerical data
        for col, (min_val, max_val) in self.numerical_granularity.items():
            if col in result.columns and result[col].dtype != 'object':
                try:
                    result[col] = result[col] * (max_val - min_val) + min_val
                except Exception:
                    pass  # Keep as is if transformation fails
        
        return result

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return MondrianAnonymizationPlugin()
