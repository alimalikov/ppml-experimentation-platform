"""
Slicing Anonymization Plugin for Data Anonymization

This plugin implements the Slicing algorithm for privacy protection, which
partitions attributes into columns and tuples into buckets, then randomly
associates column values within buckets. This approach provides better utility
than traditional generalization while maintaining privacy protection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import itertools
from collections import defaultdict
import secrets

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class SlicingAnonymizationPlugin(Anonymizer):
    """
    Slicing anonymization plugin that implements the Slicing algorithm.
    
    The Slicing algorithm works by:
    1. Partitioning attributes into columns (attribute slices)
    2. Partitioning tuples into buckets of size at least l
    3. Randomly associating attribute values within each bucket
    4. Ensuring l-diversity within each bucket-column combination
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Slicing Anonymization"
        self.description = "Partitions attributes and tuples for privacy with high utility preservation"
        
        # Slicing parameters
        self.l_value = 3
        self.bucket_size = 5
        self.min_bucket_size = 3
        
        # Attribute partitioning parameters
        self.num_columns = 3
        self.column_strategy = "correlation_based"  # correlation_based, random, manual, sensitivity_based
        self.preserve_correlations = True
        self.correlation_threshold = 0.7
        
        # Tuple partitioning parameters
        self.bucket_formation = "random"  # random, clustering, sorted, stratified
        self.ensure_diversity = True
        self.diversity_attribute = None  # Will be set to sensitive attribute
        
        # Association parameters
        self.association_method = "random"  # random, similarity_preserving, utility_maximizing
        self.preserve_patterns = True
        self.maintain_frequencies = True
        
        # Privacy parameters
        self.add_noise = False  # Slicing typically doesn't need noise
        self.use_suppression = True
        self.suppression_threshold = 0.05
        
        # Quality parameters
        self.optimize_utility = True
        self.measure_information_loss = True
        self.preserve_marginals = True
        
        # Internal state
        self.attribute_columns = []
        self.tuple_buckets = []
        self.column_associations = {}
        self.slicing_statistics = {}
        self.original_correlations = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Clustering & Grouping"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Slicing anonymization specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔪 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Slicing Anonymization"):
            st.markdown("""
            **Slicing Anonymization** partitions attributes into columns and tuples into
            buckets, then randomly associates attribute values within buckets. This
            preserves more utility than generalization while providing privacy protection.
            
            **Key Features:**
            - Attribute and tuple partitioning
            - Random value association within buckets
            - High utility preservation
            - Flexible privacy-utility trade-offs
            """)
        
        # Basic slicing parameters
        l_value = st.sidebar.slider(
            "l-diversity value",
            min_value=2, max_value=10, value=3,
            key=f"{unique_key_prefix}_l_value",
            help="Minimum number of distinct sensitive values per bucket"
        )
        
        bucket_size = st.sidebar.slider(
            "Bucket Size",
            min_value=3, max_value=20, value=5,
            key=f"{unique_key_prefix}_bucket_size",
            help="Target size for each bucket"
        )
        
        num_columns = st.sidebar.slider(
            "Number of Columns",
            min_value=2, max_value=min(10, len(all_cols)), value=3,
            key=f"{unique_key_prefix}_num_columns",
            help="Number of attribute columns to create"
        )
        
        # Algorithm parameters
        st.sidebar.subheader("🔧 Algorithm Parameters")
        
        column_strategy = st.sidebar.selectbox(
            "Column Strategy",
            options=["correlation_based", "random", "sensitivity_based"],
            key=f"{unique_key_prefix}_column_strategy",
            help="Strategy for partitioning attributes into columns"
        )
        
        bucket_formation = st.sidebar.selectbox(
            "Bucket Formation",
            options=["random", "clustering", "sorted", "stratified"],
            key=f"{unique_key_prefix}_bucket_formation",
            help="Method for forming tuple buckets"
        )
        
        association_method = st.sidebar.selectbox(
            "Association Method",
            options=["random", "similarity_preserving", "utility_maximizing"],
            key=f"{unique_key_prefix}_association_method",
            help="How to associate values within buckets"
        )
        
        # Quality settings
        with st.sidebar.expander("📊 Quality Settings"):
            preserve_correlations = st.sidebar.checkbox(
                "Preserve Correlations",
                value=True,
                key=f"{unique_key_prefix}_preserve_correlations",
                help="Try to preserve important correlations"
            )
            
            if preserve_correlations:
                correlation_threshold = st.sidebar.slider(
                    "Correlation Threshold",
                    min_value=0.1, max_value=0.9, value=0.7, step=0.1,
                    key=f"{unique_key_prefix}_correlation_threshold",
                    help="Threshold for considering correlations significant"
                )
            else:
                correlation_threshold = 0.7
            
            preserve_patterns = st.sidebar.checkbox(
                "Preserve Patterns",
                value=True,
                key=f"{unique_key_prefix}_preserve_patterns",
                help="Try to preserve data patterns"
            )
            
            maintain_frequencies = st.sidebar.checkbox(
                "Maintain Frequencies",
                value=True,
                key=f"{unique_key_prefix}_maintain_frequencies",
                help="Maintain attribute value frequencies"
            )
        
        # Privacy settings
        with st.sidebar.expander("🔒 Privacy Settings"):
            ensure_diversity = st.sidebar.checkbox(
                "Ensure Diversity",
                value=True,
                key=f"{unique_key_prefix}_ensure_diversity",
                help="Ensure l-diversity in each bucket"
            )
            
            use_suppression = st.sidebar.checkbox(
                "Use Suppression",
                value=True,
                key=f"{unique_key_prefix}_use_suppression",
                help="Allow suppression of problematic records"
            )
            
            if use_suppression:
                suppression_threshold = st.sidebar.slider(
                    "Suppression Threshold",
                    min_value=0.0, max_value=0.2, value=0.05, step=0.01,
                    key=f"{unique_key_prefix}_suppression_threshold",
                    help="Maximum fraction of records to suppress"
                )
            else:
                suppression_threshold = 0.0
        
        return {
            "l_value": l_value,
            "bucket_size": bucket_size,
            "num_columns": num_columns,
            "column_strategy": column_strategy,
            "bucket_formation": bucket_formation,
            "association_method": association_method,
            "preserve_correlations": preserve_correlations,
            "correlation_threshold": correlation_threshold,
            "preserve_patterns": preserve_patterns,
            "maintain_frequencies": maintain_frequencies,
            "ensure_diversity": ensure_diversity,
            "use_suppression": use_suppression,
            "suppression_threshold": suppression_threshold
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Slicing anonymization algorithm to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.l_value = parameters.get("l_value", 3)
            self.bucket_size = parameters.get("bucket_size", 5)
            self.num_columns = parameters.get("num_columns", 3)
            self.column_strategy = parameters.get("column_strategy", "correlation_based")
            self.bucket_formation = parameters.get("bucket_formation", "random")
            self.association_method = parameters.get("association_method", "random")
            self.preserve_correlations = parameters.get("preserve_correlations", True)
            self.correlation_threshold = parameters.get("correlation_threshold", 0.7)
            self.preserve_patterns = parameters.get("preserve_patterns", True)
            self.maintain_frequencies = parameters.get("maintain_frequencies", True)
            self.ensure_diversity = parameters.get("ensure_diversity", True)
            self.use_suppression = parameters.get("use_suppression", True)
            self.suppression_threshold = parameters.get("suppression_threshold", 0.05)
            
            self.diversity_attribute = sa_col
            self.min_bucket_size = max(self.l_value, 3)
            
            logger.info(f"Starting Slicing anonymization with l={self.l_value}, bucket_size={self.bucket_size}")
            
            # Store original correlations
            if self.preserve_correlations:
                self._store_original_correlations(df_input)
            
            # Step 1: Partition attributes into columns
            self.attribute_columns = self._partition_attributes(df_input)
            
            # Step 2: Partition tuples into buckets
            self.tuple_buckets = self._partition_tuples(df_input)
            
            # Step 3: Associate values within buckets
            sliced_data = self._associate_values_in_buckets(df_input)
            
            # Step 4: Apply suppression if needed
            if self.use_suppression:
                sliced_data = self._apply_suppression(sliced_data)
            
            # Calculate slicing statistics
            self._calculate_slicing_statistics(df_input, sliced_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Slicing anonymization completed in {processing_time:.2f}s")
            
            # Add metadata
            sliced_data.attrs['slicing_l'] = self.l_value
            sliced_data.attrs['num_buckets'] = len(self.tuple_buckets)
            sliced_data.attrs['num_columns'] = len(self.attribute_columns)
            sliced_data.attrs['slicing_statistics'] = self.slicing_statistics
            sliced_data.attrs['processing_time'] = processing_time
            
            return sliced_data
            
        except Exception as e:
            logger.error(f"Error in Slicing anonymization: {str(e)}")
            st.error(f"Slicing anonymization failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "l_value": st.session_state.get(f"{unique_key_prefix}_l_value", 3),
            "bucket_size": st.session_state.get(f"{unique_key_prefix}_bucket_size", 5),
            "num_columns": st.session_state.get(f"{unique_key_prefix}_num_columns", 3),
            "column_strategy": st.session_state.get(f"{unique_key_prefix}_column_strategy", "correlation_based"),
            "bucket_formation": st.session_state.get(f"{unique_key_prefix}_bucket_formation", "random"),
            "association_method": st.session_state.get(f"{unique_key_prefix}_association_method", "random"),
            "preserve_correlations": st.session_state.get(f"{unique_key_prefix}_preserve_correlations", True),
            "correlation_threshold": st.session_state.get(f"{unique_key_prefix}_correlation_threshold", 0.7),
            "preserve_patterns": st.session_state.get(f"{unique_key_prefix}_preserve_patterns", True),
            "maintain_frequencies": st.session_state.get(f"{unique_key_prefix}_maintain_frequencies", True),
            "ensure_diversity": st.session_state.get(f"{unique_key_prefix}_ensure_diversity", True),
            "use_suppression": st.session_state.get(f"{unique_key_prefix}_use_suppression", True),
            "suppression_threshold": st.session_state.get(f"{unique_key_prefix}_suppression_threshold", 0.05)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _store_original_correlations(self, data: pd.DataFrame):
        """Store original correlations for preservation."""
        numerical_data = data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 1:
            self.original_correlations = numerical_data.corr()
    
    def _partition_attributes(self, data: pd.DataFrame) -> List[List[str]]:
        """Partition attributes into columns based on the specified strategy."""
        all_attributes = data.columns.tolist()
        columns = []
        
        if self.column_strategy == "correlation_based":
            columns = self._partition_by_correlation(data, all_attributes)
        elif self.column_strategy == "sensitivity_based":
            columns = self._partition_by_sensitivity(data, all_attributes)
        else:  # random
            columns = self._partition_randomly(all_attributes)
        
        logger.info(f"Created {len(columns)} attribute columns")
        return columns
    
    def _partition_by_correlation(self, data: pd.DataFrame, attributes: List[str]) -> List[List[str]]:
        """Partition attributes based on correlations."""
        # Calculate correlations for numerical attributes
        numerical_attrs = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_attrs = [col for col in attributes if col not in numerical_attrs]
        
        columns = []
        used_attrs = set()
        
        if len(numerical_attrs) > 1:
            corr_matrix = data[numerical_attrs].corr().abs()
            
            # Group highly correlated attributes together
            for attr in numerical_attrs:
                if attr in used_attrs:
                    continue
                
                # Find attributes highly correlated with this one
                correlated = []
                for other_attr in numerical_attrs:
                    if (other_attr != attr and other_attr not in used_attrs and 
                        corr_matrix.loc[attr, other_attr] > self.correlation_threshold):
                        correlated.append(other_attr)
                
                # Create column with this attribute and its correlated ones
                column = [attr] + correlated
                columns.append(column)
                used_attrs.update(column)
        
        # Add remaining numerical attributes
        remaining_numerical = [attr for attr in numerical_attrs if attr not in used_attrs]
        if remaining_numerical:
            columns.append(remaining_numerical)
        
        # Add categorical attributes
        if categorical_attrs:
            # Distribute categorical attributes among columns
            if columns:
                for i, attr in enumerate(categorical_attrs):
                    columns[i % len(columns)].append(attr)
            else:
                columns.append(categorical_attrs)
        
        # Ensure we have the desired number of columns
        while len(columns) < self.num_columns and any(len(col) > 1 for col in columns):
            # Split the largest column
            largest_col_idx = max(range(len(columns)), key=lambda i: len(columns[i]))
            largest_col = columns[largest_col_idx]
            
            if len(largest_col) > 1:
                mid = len(largest_col) // 2
                columns[largest_col_idx] = largest_col[:mid]
                columns.append(largest_col[mid:])
        
        return columns
    
    def _partition_by_sensitivity(self, data: pd.DataFrame, attributes: List[str]) -> List[List[str]]:
        """Partition attributes based on sensitivity (sensitive attribute separate)."""
        columns = []
        
        # Put sensitive attribute in its own column if specified
        if self.diversity_attribute and self.diversity_attribute in attributes:
            columns.append([self.diversity_attribute])
            remaining_attrs = [attr for attr in attributes if attr != self.diversity_attribute]
        else:
            remaining_attrs = attributes
        
        # Distribute remaining attributes
        attrs_per_column = max(1, len(remaining_attrs) // (self.num_columns - len(columns)))
        
        for i in range(0, len(remaining_attrs), attrs_per_column):
            column = remaining_attrs[i:i + attrs_per_column]
            if column:
                columns.append(column)
        
        return columns
    
    def _partition_randomly(self, attributes: List[str]) -> List[List[str]]:
        """Randomly partition attributes into columns."""
        shuffled_attrs = attributes.copy()
        np.random.shuffle(shuffled_attrs)
        
        attrs_per_column = max(1, len(attributes) // self.num_columns)
        columns = []
        
        for i in range(0, len(shuffled_attrs), attrs_per_column):
            column = shuffled_attrs[i:i + attrs_per_column]
            if column:
                columns.append(column)
        
        return columns
    
    def _partition_tuples(self, data: pd.DataFrame) -> List[List[int]]:
        """Partition tuples into buckets."""
        if self.bucket_formation == "random":
            return self._partition_tuples_randomly(data)
        elif self.bucket_formation == "clustering":
            return self._partition_tuples_by_clustering(data)
        elif self.bucket_formation == "sorted":
            return self._partition_tuples_sorted(data)
        else:  # stratified
            return self._partition_tuples_stratified(data)
    
    def _partition_tuples_randomly(self, data: pd.DataFrame) -> List[List[int]]:
        """Randomly partition tuples into buckets."""
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        
        buckets = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            if len(bucket) >= self.min_bucket_size:
                buckets.append(bucket)
            elif buckets:
                # Merge with last bucket
                buckets[-1].extend(bucket)
            else:
                # First bucket, keep even if small
                buckets.append(bucket)
        
        return buckets
    
    def _partition_tuples_by_clustering(self, data: pd.DataFrame) -> List[List[int]]:
        """Partition tuples using clustering."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for clustering
            numerical_data = data.select_dtypes(include=[np.number])
            categorical_data = data.select_dtypes(include=['object', 'category'])
            
            features = []
            if not numerical_data.empty:
                scaler = StandardScaler()
                scaled_numerical = scaler.fit_transform(numerical_data)
                features.extend(scaled_numerical.T)
            
            if not categorical_data.empty:
                # Encode categorical variables
                for col in categorical_data.columns:
                    encoded = pd.Categorical(categorical_data[col]).codes
                    features.append(encoded)
            
            if not features:
                # Fall back to random partitioning
                return self._partition_tuples_randomly(data)
            
            features = np.column_stack(features)
            
            # Determine number of clusters
            n_clusters = max(2, len(data) // self.bucket_size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Group indices by cluster
            buckets = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                buckets[label].append(idx)
            
            # Convert to list and ensure minimum bucket size
            result_buckets = []
            small_buckets = []
            
            for bucket in buckets.values():
                if len(bucket) >= self.min_bucket_size:
                    result_buckets.append(bucket)
                else:
                    small_buckets.extend(bucket)
            
            # Merge small buckets
            if small_buckets:
                if result_buckets:
                    result_buckets[0].extend(small_buckets)
                else:
                    result_buckets.append(small_buckets)
            
            return result_buckets
            
        except ImportError:
            logger.warning("sklearn not available, falling back to random partitioning")
            return self._partition_tuples_randomly(data)
    
    def _partition_tuples_sorted(self, data: pd.DataFrame) -> List[List[int]]:
        """Partition tuples after sorting by a key attribute."""
        # Sort by the first numerical attribute or sensitive attribute
        sort_col = None
        if self.diversity_attribute and self.diversity_attribute in data.columns:
            sort_col = self.diversity_attribute
        else:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                sort_col = numerical_cols[0]
            else:
                # Fall back to random
                return self._partition_tuples_randomly(data)
        
        sorted_indices = data.sort_values(by=sort_col).index.tolist()
        
        buckets = []
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size]
            if len(bucket) >= self.min_bucket_size:
                buckets.append(bucket)
            elif buckets:
                buckets[-1].extend(bucket)
            else:
                buckets.append(bucket)
        
        return buckets
    
    def _partition_tuples_stratified(self, data: pd.DataFrame) -> List[List[int]]:
        """Partition tuples ensuring diversity in each bucket."""
        if not self.diversity_attribute or self.diversity_attribute not in data.columns:
            return self._partition_tuples_randomly(data)
        
        # Group by sensitive attribute values
        sa_groups = defaultdict(list)
        for idx, value in enumerate(data[self.diversity_attribute]):
            sa_groups[value].append(idx)
        
        # Ensure each bucket has at least l different sensitive values
        unique_sa_values = list(sa_groups.keys())
        if len(unique_sa_values) < self.l_value:
            logger.warning(f"Not enough distinct sensitive values ({len(unique_sa_values)}) for l={self.l_value}")
            return self._partition_tuples_randomly(data)
        
        buckets = []
        current_bucket = []
        used_values_in_bucket = set()
        
        # Round-robin through sensitive values
        value_iterators = {val: iter(indices) for val, indices in sa_groups.items()}
        
        while any(sa_groups.values()):
            # Try to add one record from each sensitive value
            for sa_value in unique_sa_values[:self.l_value]:
                if sa_value in value_iterators:
                    try:
                        idx = next(value_iterators[sa_value])
                        current_bucket.append(idx)
                        used_values_in_bucket.add(sa_value)
                        
                        # Remove from the group
                        sa_groups[sa_value].remove(idx)
                        if not sa_groups[sa_value]:
                            del value_iterators[sa_value]
                        
                    except StopIteration:
                        continue
            
            # Check if bucket is ready
            if (len(current_bucket) >= self.bucket_size or 
                (len(current_bucket) >= self.min_bucket_size and len(used_values_in_bucket) >= self.l_value)):
                buckets.append(current_bucket)
                current_bucket = []
                used_values_in_bucket = set()
        
        # Handle remaining records
        if current_bucket:
            if buckets:
                buckets[-1].extend(current_bucket)
            else:
                buckets.append(current_bucket)
        
        return buckets
    
    def _associate_values_in_buckets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Associate attribute values within buckets."""
        result = data.copy()
        
        for bucket_idx, bucket_indices in enumerate(self.tuple_buckets):
            bucket_data = data.iloc[bucket_indices]
            
            for column_idx, column_attrs in enumerate(self.attribute_columns):
                if self.association_method == "random":
                    # Randomly permute values within the bucket for this column
                    for attr in column_attrs:
                        if attr in bucket_data.columns:
                            values = bucket_data[attr].values.copy()
                            np.random.shuffle(values)
                            result.loc[bucket_indices, attr] = values
                
                elif self.association_method == "similarity_preserving":
                    # Try to maintain similarity patterns
                    self._associate_similarity_preserving(result, bucket_indices, column_attrs, bucket_data)
                
                else:  # utility_maximizing
                    # Maximize utility while ensuring privacy
                    self._associate_utility_maximizing(result, bucket_indices, column_attrs, bucket_data)
        
        return result
    
    def _associate_similarity_preserving(self, result: pd.DataFrame, bucket_indices: List[int], 
                                       column_attrs: List[str], bucket_data: pd.DataFrame):
        """Associate values while preserving similarity patterns."""
        if not column_attrs:
            return
        
        # For each attribute in the column, try to preserve relative ordering
        for attr in column_attrs:
            if attr not in bucket_data.columns:
                continue
            
            values = bucket_data[attr].values
            if bucket_data[attr].dtype in ['object', 'category']:
                # For categorical, random shuffle
                np.random.shuffle(values)
            else:
                # For numerical, preserve relative ordering with some randomness
                sorted_indices = np.argsort(values)
                # Add some randomness while preserving general order
                noise = np.random.normal(0, 0.1, len(sorted_indices))
                noisy_indices = sorted_indices + noise
                reorder_indices = np.argsort(noisy_indices)
                values = values[reorder_indices]
            
            result.loc[bucket_indices, attr] = values
    
    def _associate_utility_maximizing(self, result: pd.DataFrame, bucket_indices: List[int], 
                                    column_attrs: List[str], bucket_data: pd.DataFrame):
        """Associate values to maximize utility."""
        if not column_attrs or len(column_attrs) == 1:
            # If only one attribute, random shuffle
            for attr in column_attrs:
                if attr in bucket_data.columns:
                    values = bucket_data[attr].values.copy()
                    np.random.shuffle(values)
                    result.loc[bucket_indices, attr] = values
            return
        
        # For multiple attributes, try to preserve some correlations
        numerical_attrs = [attr for attr in column_attrs 
                         if attr in bucket_data.columns and bucket_data[attr].dtype in ['int64', 'float64']]
        
        if len(numerical_attrs) > 1 and self.preserve_correlations:
            # Try to maintain correlation structure
            corr_matrix = bucket_data[numerical_attrs].corr()
            
            # Find the most correlated pair
            max_corr = 0
            corr_pair = None
            for i, attr1 in enumerate(numerical_attrs):
                for j, attr2 in enumerate(numerical_attrs[i+1:], i+1):
                    if abs(corr_matrix.loc[attr1, attr2]) > max_corr:
                        max_corr = abs(corr_matrix.loc[attr1, attr2])
                        corr_pair = (attr1, attr2)
            
            if corr_pair and max_corr > 0.3:
                # Preserve the ordering for the most correlated pair
                attr1, attr2 = corr_pair
                combined_data = list(zip(bucket_data[attr1], bucket_data[attr2]))
                
                # Sort by first attribute to preserve correlation direction
                if corr_matrix.loc[attr1, attr2] > 0:
                    combined_data.sort(key=lambda x: x[0])
                else:
                    combined_data.sort(key=lambda x: x[0], reverse=True)
                
                # Add some randomness
                n_swaps = len(combined_data) // 4
                for _ in range(n_swaps):
                    i, j = np.random.choice(len(combined_data), 2, replace=False)
                    combined_data[i], combined_data[j] = combined_data[j], combined_data[i]
                
                # Assign back
                values1, values2 = zip(*combined_data)
                result.loc[bucket_indices, attr1] = values1
                result.loc[bucket_indices, attr2] = values2
                
                # Random shuffle remaining attributes
                remaining_attrs = [attr for attr in column_attrs if attr not in corr_pair]
            else:
                remaining_attrs = column_attrs
        else:
            remaining_attrs = column_attrs
        
        # Random shuffle remaining attributes
        for attr in remaining_attrs:
            if attr in bucket_data.columns:
                values = bucket_data[attr].values.copy()
                np.random.shuffle(values)
                result.loc[bucket_indices, attr] = values
    
    def _apply_suppression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply suppression to problematic records."""
        # Identify buckets that don't meet l-diversity requirement
        problematic_records = []
        
        if self.ensure_diversity and self.diversity_attribute:
            for bucket_indices in self.tuple_buckets:
                bucket_data = data.iloc[bucket_indices]
                unique_sa_values = bucket_data[self.diversity_attribute].nunique()
                
                if unique_sa_values < self.l_value:
                    # Mark some records for suppression
                    records_to_suppress = max(1, len(bucket_indices) - self.l_value)
                    suppress_indices = np.random.choice(bucket_indices, records_to_suppress, replace=False)
                    problematic_records.extend(suppress_indices)
        
        # Apply suppression
        suppression_rate = len(problematic_records) / len(data)
        if suppression_rate > self.suppression_threshold:
            logger.warning(f"Suppression rate ({suppression_rate:.3f}) exceeds threshold ({self.suppression_threshold})")
            # Suppress only up to threshold
            max_suppress = int(len(data) * self.suppression_threshold)
            problematic_records = problematic_records[:max_suppress]
        
        result = data.copy()
        if problematic_records:
            result.loc[problematic_records, :] = "*SUPPRESSED*"
            logger.info(f"Suppressed {len(problematic_records)} records")
        
        return result
    
    def _calculate_slicing_statistics(self, original: pd.DataFrame, sliced: pd.DataFrame):
        """Calculate statistics about the slicing process."""
        self.slicing_statistics = {
            'num_buckets': len(self.tuple_buckets),
            'avg_bucket_size': np.mean([len(bucket) for bucket in self.tuple_buckets]),
            'num_attribute_columns': len(self.attribute_columns),
            'diversity_preserved': self._check_diversity_preservation(sliced),
            'correlation_preservation': self._measure_correlation_preservation(original, sliced)
        }
    
    def _check_diversity_preservation(self, data: pd.DataFrame) -> bool:
        """Check if l-diversity is preserved in all buckets."""
        if not self.diversity_attribute or self.diversity_attribute not in data.columns:
            return True
        
        for bucket_indices in self.tuple_buckets:
            bucket_data = data.iloc[bucket_indices]
            # Filter out suppressed records
            valid_data = bucket_data[bucket_data[self.diversity_attribute] != "*SUPPRESSED*"]
            
            if len(valid_data) > 0:
                unique_sa_values = valid_data[self.diversity_attribute].nunique()
                if unique_sa_values < self.l_value:
                    return False
        
        return True
    
    def _measure_correlation_preservation(self, original: pd.DataFrame, sliced: pd.DataFrame) -> float:
        """Measure how well correlations are preserved."""
        if not self.preserve_correlations or self.original_correlations.empty:
            return 1.0
        
        try:
            numerical_cols = original.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2:
                return 1.0
            
            # Calculate correlations in sliced data
            sliced_numerical = sliced[numerical_cols].select_dtypes(include=[np.number])
            if sliced_numerical.empty:
                return 0.0
            
            sliced_corr = sliced_numerical.corr()
            
            # Compare correlation matrices
            correlation_diff = 0.0
            count = 0
            
            for col1 in numerical_cols:
                for col2 in numerical_cols:
                    if col1 != col2 and col1 in sliced_corr.columns and col2 in sliced_corr.columns:
                        orig_corr = self.original_correlations.loc[col1, col2]
                        sliced_corr_val = sliced_corr.loc[col1, col2]
                        
                        if not pd.isna(orig_corr) and not pd.isna(sliced_corr_val):
                            correlation_diff += abs(orig_corr - sliced_corr_val)
                            count += 1
            
            if count > 0:
                avg_diff = correlation_diff / count
                return max(0.0, 1.0 - avg_diff)
            
        except Exception as e:
            logger.warning(f"Error calculating correlation preservation: {e}")
        
        return 1.0

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return SlicingAnonymizationPlugin()
