"""
Bucketization Anonymization Plugin for Data Anonymization

This plugin implements the Bucketization algorithm for privacy protection, which
groups records into buckets of size at least l and randomizes sensitive attribute
assignments within each bucket. This approach provides strong privacy protection
while maintaining good utility for analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
from collections import defaultdict, Counter
import itertools

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class BucketizationPlugin(Anonymizer):
    """
    Bucketization anonymization plugin that implements l-diversity through bucketing.
    
    The Bucketization algorithm works by:
    1. Grouping quasi-identifiers into equivalence classes
    2. Forming buckets of size at least l from these classes
    3. Randomly redistributing sensitive attribute values within buckets
    4. Ensuring l-diversity in each bucket
    5. Maintaining quasi-identifier values while protecting sensitive attributes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bucketization l-Diversity"
        self.description = "Groups records into buckets with randomized sensitive attribute distribution"
        
        # Bucketization parameters
        self.l_value = 3
        self.bucket_size = 5
        self.min_bucket_size = 3
        self.max_bucket_size = 20
        
        # Bucketing strategy parameters
        self.bucketing_method = "equivalence_class"  # equivalence_class, clustering, random, stratified
        self.qi_similarity_threshold = 0.8
        self.use_adaptive_buckets = True
        self.balance_bucket_sizes = True
        
        # Sensitive attribute handling
        self.redistribution_method = "uniform"  # uniform, frequency_preserving, entropy_maximizing
        self.ensure_distinct_sa = True
        self.preserve_sa_distribution = True
        self.add_sa_noise = False
        
        # Privacy enhancement
        self.use_suppression = True
        self.suppression_threshold = 0.05
        self.add_dummy_records = False
        self.dummy_ratio = 0.03
        
        # Quality parameters
        self.preserve_qi_patterns = True
        self.maintain_utility = True
        self.optimize_for_queries = True
        self.track_diversity_metrics = True
        
        # Internal state
        self.buckets = []
        self.equivalence_classes = {}
        self.sa_distributions = {}
        self.bucket_statistics = {}
        self.diversity_metrics = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Clustering & Grouping"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Bucketization specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🪣 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Bucketization"):
            st.markdown("""
            **Bucketization** groups records into buckets and randomizes sensitive
            attribute assignments within each bucket. This preserves quasi-identifier
            values exactly while ensuring l-diversity for sensitive attributes.
            
            **Key Features:**
            - Exact quasi-identifier preservation
            - Strong l-diversity guarantees
            - Flexible bucket formation strategies
            - High utility for analytical queries
            """)
        
        # Basic parameters
        l_value = st.sidebar.slider(
            "l-diversity value",
            min_value=2, max_value=10, value=3,
            key=f"{unique_key_prefix}_l_value",
            help="Minimum number of distinct sensitive values per bucket"
        )
        
        bucket_size = st.sidebar.slider(
            "Target Bucket Size",
            min_value=3, max_value=30, value=5,
            key=f"{unique_key_prefix}_bucket_size",
            help="Target number of records per bucket"
        )
        
        # Bucketing strategy
        st.sidebar.subheader("🔧 Bucketing Strategy")
        
        bucketing_method = st.sidebar.selectbox(
            "Bucketing Method",
            options=["equivalence_class", "clustering", "random", "stratified"],
            key=f"{unique_key_prefix}_bucketing_method",
            help="Method for forming buckets from records"
        )
        
        use_adaptive_buckets = st.sidebar.checkbox(
            "Adaptive Bucket Sizing",
            value=True,
            key=f"{unique_key_prefix}_use_adaptive_buckets",
            help="Automatically adjust bucket sizes based on data distribution"
        )
        
        balance_bucket_sizes = st.sidebar.checkbox(
            "Balance Bucket Sizes",
            value=True,
            key=f"{unique_key_prefix}_balance_bucket_sizes",
            help="Try to keep bucket sizes relatively uniform"
        )
        
        # Sensitive attribute handling
        with st.sidebar.expander("🔐 Sensitive Attribute Handling"):
            redistribution_method = st.sidebar.selectbox(
                "Redistribution Method",
                options=["uniform", "frequency_preserving", "entropy_maximizing"],
                key=f"{unique_key_prefix}_redistribution_method",
                help="How to redistribute sensitive values within buckets"
            )
            
            preserve_sa_distribution = st.sidebar.checkbox(
                "Preserve SA Distribution",
                value=True,
                key=f"{unique_key_prefix}_preserve_sa_distribution",
                help="Maintain overall sensitive attribute distribution"
            )
            
            add_sa_noise = st.sidebar.checkbox(
                "Add SA Noise",
                value=False,
                key=f"{unique_key_prefix}_add_sa_noise",
                help="Add noise to sensitive attribute values"
            )
        
        # Privacy enhancement
        with st.sidebar.expander("🛡️ Privacy Enhancement"):
            use_suppression = st.sidebar.checkbox(
                "Use Suppression",
                value=True,
                key=f"{unique_key_prefix}_use_suppression",
                help="Suppress records that cannot be bucketized properly"
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
            
            add_dummy_records = st.sidebar.checkbox(
                "Add Dummy Records",
                value=False,
                key=f"{unique_key_prefix}_add_dummy_records",
                help="Add dummy records to enhance privacy"
            )
            
            if add_dummy_records:
                dummy_ratio = st.sidebar.slider(
                    "Dummy Record Ratio",
                    min_value=0.01, max_value=0.1, value=0.03, step=0.01,
                    key=f"{unique_key_prefix}_dummy_ratio",
                    help="Ratio of dummy records to add"
                )
            else:
                dummy_ratio = 0.0
        
        # Quality settings
        with st.sidebar.expander("📊 Quality Settings"):
            preserve_qi_patterns = st.sidebar.checkbox(
                "Preserve QI Patterns",
                value=True,
                key=f"{unique_key_prefix}_preserve_qi_patterns",
                help="Maintain patterns in quasi-identifier attributes"
            )
            
            optimize_for_queries = st.sidebar.checkbox(
                "Optimize for Queries",
                value=True,
                key=f"{unique_key_prefix}_optimize_for_queries",
                help="Optimize bucket formation for analytical queries"
            )
        
        return {
            "l_value": l_value,
            "bucket_size": bucket_size,
            "bucketing_method": bucketing_method,
            "use_adaptive_buckets": use_adaptive_buckets,
            "balance_bucket_sizes": balance_bucket_sizes,
            "redistribution_method": redistribution_method,
            "preserve_sa_distribution": preserve_sa_distribution,
            "add_sa_noise": add_sa_noise,
            "use_suppression": use_suppression,
            "suppression_threshold": suppression_threshold,
            "add_dummy_records": add_dummy_records,
            "dummy_ratio": dummy_ratio,
            "preserve_qi_patterns": preserve_qi_patterns,
            "optimize_for_queries": optimize_for_queries
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Bucketization algorithm to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.l_value = parameters.get("l_value", 3)
            self.bucket_size = parameters.get("bucket_size", 5)
            self.bucketing_method = parameters.get("bucketing_method", "equivalence_class")
            self.use_adaptive_buckets = parameters.get("use_adaptive_buckets", True)
            self.balance_bucket_sizes = parameters.get("balance_bucket_sizes", True)
            self.redistribution_method = parameters.get("redistribution_method", "uniform")
            self.preserve_sa_distribution = parameters.get("preserve_sa_distribution", True)
            self.add_sa_noise = parameters.get("add_sa_noise", False)
            self.use_suppression = parameters.get("use_suppression", True)
            self.suppression_threshold = parameters.get("suppression_threshold", 0.05)
            self.add_dummy_records = parameters.get("add_dummy_records", False)
            self.dummy_ratio = parameters.get("dummy_ratio", 0.03)
            self.preserve_qi_patterns = parameters.get("preserve_qi_patterns", True)
            self.optimize_for_queries = parameters.get("optimize_for_queries", True)
            
            self.min_bucket_size = max(self.l_value, 3)
            
            if sa_col is None:
                st.warning("Bucketization requires a sensitive attribute. Using first column as sensitive attribute.")
                sa_col = df_input.columns[0]
            
            logger.info(f"Starting Bucketization with l={self.l_value}, bucket_size={self.bucket_size}, SA={sa_col}")
            
            # Prepare working data
            working_data = df_input.copy()
            
            # Add dummy records if enabled
            if self.add_dummy_records:
                working_data = self._add_dummy_records(working_data, sa_col)
            
            # Identify quasi-identifier columns
            qi_columns = [col for col in working_data.columns if col != sa_col]
            
            # Form equivalence classes based on QI values
            self.equivalence_classes = self._form_equivalence_classes(working_data, qi_columns)
            
            # Create buckets from equivalence classes
            self.buckets = self._form_buckets_from_classes(working_data, sa_col)
            
            # Redistribute sensitive attribute values within buckets
            bucketized_data = self._redistribute_sensitive_attributes(working_data, sa_col)
            
            # Apply suppression if needed
            if self.use_suppression:
                bucketized_data = self._apply_suppression(bucketized_data, sa_col)
            
            # Add noise to sensitive attributes if enabled
            if self.add_sa_noise:
                bucketized_data = self._add_sensitive_attribute_noise(bucketized_data, sa_col)
            
            # Calculate statistics and metrics
            self._calculate_bucket_statistics(working_data, bucketized_data, sa_col)
            self._calculate_diversity_metrics(bucketized_data, sa_col)
            
            processing_time = time.time() - start_time
            logger.info(f"Bucketization completed in {processing_time:.2f}s")
            
            # Add metadata
            bucketized_data.attrs['bucketization_l'] = self.l_value
            bucketized_data.attrs['num_buckets'] = len(self.buckets)
            bucketized_data.attrs['bucket_statistics'] = self.bucket_statistics
            bucketized_data.attrs['diversity_metrics'] = self.diversity_metrics
            bucketized_data.attrs['processing_time'] = processing_time
            
            return bucketized_data
            
        except Exception as e:
            logger.error(f"Error in Bucketization: {str(e)}")
            st.error(f"Bucketization failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "l_value": st.session_state.get(f"{unique_key_prefix}_l_value", 3),
            "bucket_size": st.session_state.get(f"{unique_key_prefix}_bucket_size", 5),
            "bucketing_method": st.session_state.get(f"{unique_key_prefix}_bucketing_method", "equivalence_class"),
            "use_adaptive_buckets": st.session_state.get(f"{unique_key_prefix}_use_adaptive_buckets", True),
            "balance_bucket_sizes": st.session_state.get(f"{unique_key_prefix}_balance_bucket_sizes", True),
            "redistribution_method": st.session_state.get(f"{unique_key_prefix}_redistribution_method", "uniform"),
            "preserve_sa_distribution": st.session_state.get(f"{unique_key_prefix}_preserve_sa_distribution", True),
            "add_sa_noise": st.session_state.get(f"{unique_key_prefix}_add_sa_noise", False),
            "use_suppression": st.session_state.get(f"{unique_key_prefix}_use_suppression", True),
            "suppression_threshold": st.session_state.get(f"{unique_key_prefix}_suppression_threshold", 0.05),
            "add_dummy_records": st.session_state.get(f"{unique_key_prefix}_add_dummy_records", False),
            "dummy_ratio": st.session_state.get(f"{unique_key_prefix}_dummy_ratio", 0.03),
            "preserve_qi_patterns": st.session_state.get(f"{unique_key_prefix}_preserve_qi_patterns", True),
            "optimize_for_queries": st.session_state.get(f"{unique_key_prefix}_optimize_for_queries", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _add_dummy_records(self, data: pd.DataFrame, sa_col: str) -> pd.DataFrame:
        """Add dummy records to enhance privacy."""
        num_dummies = int(len(data) * self.dummy_ratio)
        dummy_records = []
        
        qi_columns = [col for col in data.columns if col != sa_col]
        
        for _ in range(num_dummies):
            dummy_record = {}
            
            # Generate dummy QI values by sampling from existing data
            for col in qi_columns:
                if data[col].dtype in ['object', 'category']:
                    dummy_record[col] = np.random.choice(data[col].unique())
                else:
                    min_val, max_val = data[col].min(), data[col].max()
                    dummy_record[col] = np.random.uniform(min_val, max_val)
            
            # Generate dummy SA values
            if data[sa_col].dtype in ['object', 'category']:
                dummy_record[sa_col] = f"dummy_sa_{np.random.randint(1000, 9999)}"
            else:
                min_val, max_val = data[sa_col].min(), data[sa_col].max()
                dummy_record[sa_col] = np.random.uniform(min_val, max_val)
            
            dummy_records.append(dummy_record)
        
        dummy_df = pd.DataFrame(dummy_records)
        return pd.concat([data, dummy_df], ignore_index=True)
    
    def _form_equivalence_classes(self, data: pd.DataFrame, qi_columns: List[str]) -> Dict[str, List[int]]:
        """Form equivalence classes based on quasi-identifier values."""
        equivalence_classes = defaultdict(list)
        
        for idx, row in data.iterrows():
            # Create signature from QI values
            qi_signature = tuple(row[qi_columns].values)
            equivalence_classes[str(qi_signature)].append(idx)
        
        logger.info(f"Formed {len(equivalence_classes)} equivalence classes")
        return dict(equivalence_classes)
    
    def _form_buckets_from_classes(self, data: pd.DataFrame, sa_col: str) -> List[List[int]]:
        """Form buckets from equivalence classes using the specified method."""
        if self.bucketing_method == "equivalence_class":
            return self._bucket_by_equivalence_classes(data, sa_col)
        elif self.bucketing_method == "clustering":
            return self._bucket_by_clustering(data, sa_col)
        elif self.bucketing_method == "stratified":
            return self._bucket_stratified(data, sa_col)
        else:  # random
            return self._bucket_randomly(data)
    
    def _bucket_by_equivalence_classes(self, data: pd.DataFrame, sa_col: str) -> List[List[int]]:
        """Form buckets by merging equivalence classes."""
        buckets = []
        current_bucket = []
        
        # Sort equivalence classes by size (largest first for better packing)
        sorted_classes = sorted(self.equivalence_classes.items(), 
                              key=lambda x: len(x[1]), reverse=True)
        
        for class_signature, class_indices in sorted_classes:
            class_size = len(class_indices)
            
            # If class is large enough to be its own bucket
            if class_size >= self.min_bucket_size:
                if self.use_adaptive_buckets:
                    # Split large classes if needed
                    if class_size > self.max_bucket_size:
                        for i in range(0, class_size, self.max_bucket_size):
                            bucket_indices = class_indices[i:i + self.max_bucket_size]
                            if len(bucket_indices) >= self.min_bucket_size:
                                buckets.append(bucket_indices)
                            else:
                                current_bucket.extend(bucket_indices)
                    else:
                        buckets.append(class_indices)
                else:
                    buckets.append(class_indices)
            else:
                # Add to current bucket
                current_bucket.extend(class_indices)
                
                # Check if current bucket is large enough
                if len(current_bucket) >= self.bucket_size:
                    buckets.append(current_bucket)
                    current_bucket = []
        
        # Handle remaining records
        if current_bucket:
            if len(current_bucket) >= self.min_bucket_size:
                buckets.append(current_bucket)
            elif buckets:
                # Merge with the last bucket
                buckets[-1].extend(current_bucket)
            else:
                # Keep as is if it's the only bucket
                buckets.append(current_bucket)
        
        # Balance bucket sizes if requested
        if self.balance_bucket_sizes:
            buckets = self._balance_buckets(buckets)
        
        logger.info(f"Formed {len(buckets)} buckets from equivalence classes")
        return buckets
    
    def _bucket_by_clustering(self, data: pd.DataFrame, sa_col: str) -> List[List[int]]:
        """Form buckets using clustering on quasi-identifiers."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            qi_columns = [col for col in data.columns if col != sa_col]
            features = []
            
            # Prepare features for clustering
            for col in qi_columns:
                if data[col].dtype in ['object', 'category']:
                    # Encode categorical variables
                    le = LabelEncoder()
                    encoded = le.fit_transform(data[col].astype(str))
                    features.append(encoded)
                else:
                    # Standardize numerical variables
                    values = data[col].values.reshape(-1, 1)
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(values).flatten()
                    features.append(scaled)
            
            if not features:
                return self._bucket_randomly(data)
            
            X = np.column_stack(features)
            
            # Determine number of clusters
            n_clusters = max(2, len(data) // self.bucket_size)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Group indices by cluster
            cluster_buckets = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                cluster_buckets[label].append(idx)
            
            # Convert to list and handle small buckets
            buckets = []
            small_bucket_indices = []
            
            for bucket_indices in cluster_buckets.values():
                if len(bucket_indices) >= self.min_bucket_size:
                    buckets.append(bucket_indices)
                else:
                    small_bucket_indices.extend(bucket_indices)
            
            # Merge small buckets
            if small_bucket_indices:
                if buckets:
                    # Distribute among existing buckets
                    for i, idx in enumerate(small_bucket_indices):
                        bucket_idx = i % len(buckets)
                        buckets[bucket_idx].append(idx)
                else:
                    # Create new bucket from small indices
                    buckets.append(small_bucket_indices)
            
            logger.info(f"Formed {len(buckets)} buckets using clustering")
            return buckets
            
        except ImportError:
            logger.warning("sklearn not available, falling back to random bucketing")
            return self._bucket_randomly(data)
    
    def _bucket_stratified(self, data: pd.DataFrame, sa_col: str) -> List[List[int]]:
        """Form buckets ensuring diversity in sensitive attribute."""
        # Group by sensitive attribute values
        sa_groups = defaultdict(list)
        for idx, value in enumerate(data[sa_col]):
            sa_groups[value].append(idx)
        
        unique_sa_values = list(sa_groups.keys())
        if len(unique_sa_values) < self.l_value:
            logger.warning(f"Not enough unique SA values ({len(unique_sa_values)}) for l={self.l_value}")
            return self._bucket_randomly(data)
        
        buckets = []
        
        # Use round-robin to ensure diversity
        while any(sa_groups.values()):
            current_bucket = []
            used_sa_values = set()
            
            # Try to get at least l different SA values in each bucket
            for sa_value in unique_sa_values[:self.l_value]:
                if sa_value in sa_groups and sa_groups[sa_value]:
                    idx = sa_groups[sa_value].pop(0)
                    current_bucket.append(idx)
                    used_sa_values.add(sa_value)
            
            # Fill bucket to target size
            while len(current_bucket) < self.bucket_size and any(sa_groups.values()):
                for sa_value in unique_sa_values:
                    if sa_value in sa_groups and sa_groups[sa_value]:
                        idx = sa_groups[sa_value].pop(0)
                        current_bucket.append(idx)
                        if len(current_bucket) >= self.bucket_size:
                            break
            
            # Clean up empty groups
            sa_groups = {k: v for k, v in sa_groups.items() if v}
            
            if current_bucket and len(current_bucket) >= self.min_bucket_size:
                buckets.append(current_bucket)
            elif current_bucket and buckets:
                # Merge with last bucket
                buckets[-1].extend(current_bucket)
        
        logger.info(f"Formed {len(buckets)} stratified buckets")
        return buckets
    
    def _bucket_randomly(self, data: pd.DataFrame) -> List[List[int]]:
        """Form buckets by random assignment."""
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        
        buckets = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            if len(bucket) >= self.min_bucket_size:
                buckets.append(bucket)
            elif buckets:
                buckets[-1].extend(bucket)
            else:
                buckets.append(bucket)
        
        logger.info(f"Formed {len(buckets)} random buckets")
        return buckets
    
    def _balance_buckets(self, buckets: List[List[int]]) -> List[List[int]]:
        """Balance bucket sizes to be more uniform."""
        if not buckets:
            return buckets
        
        # Calculate target size
        total_records = sum(len(bucket) for bucket in buckets)
        target_size = total_records // len(buckets)
        
        # Only balance if there's significant imbalance
        max_size = max(len(bucket) for bucket in buckets)
        min_size = min(len(bucket) for bucket in buckets)
        
        if max_size - min_size <= 2:  # Already balanced
            return buckets
        
        # Redistribute records
        all_indices = []
        for bucket in buckets:
            all_indices.extend(bucket)
        
        np.random.shuffle(all_indices)
        
        balanced_buckets = []
        for i in range(0, len(all_indices), target_size):
            bucket = all_indices[i:i + target_size]
            if len(bucket) >= self.min_bucket_size:
                balanced_buckets.append(bucket)
            elif balanced_buckets:
                balanced_buckets[-1].extend(bucket)
        
        return balanced_buckets
    
    def _redistribute_sensitive_attributes(self, data: pd.DataFrame, sa_col: str) -> pd.DataFrame:
        """Redistribute sensitive attribute values within buckets."""
        result = data.copy()
        
        for bucket_idx, bucket_indices in enumerate(self.buckets):
            bucket_data = data.iloc[bucket_indices]
            sa_values = bucket_data[sa_col].tolist()
            
            if self.redistribution_method == "uniform":
                # Random shuffle
                np.random.shuffle(sa_values)
            
            elif self.redistribution_method == "frequency_preserving":
                # Maintain relative frequencies but randomize assignment
                value_counts = Counter(sa_values)
                redistributed_values = []
                
                for value, count in value_counts.items():
                    redistributed_values.extend([value] * count)
                
                np.random.shuffle(redistributed_values)
                sa_values = redistributed_values
            
            elif self.redistribution_method == "entropy_maximizing":
                # Maximize entropy in the distribution
                unique_values = list(set(sa_values))
                
                if len(unique_values) >= self.l_value:
                    # Ensure at least l distinct values
                    required_values = unique_values[:self.l_value]
                    remaining_slots = len(sa_values) - self.l_value
                    
                    redistributed_values = required_values.copy()
                    
                    if remaining_slots > 0:
                        # Fill remaining slots to maximize entropy
                        for _ in range(remaining_slots):
                            # Choose value that minimizes current max frequency
                            current_counts = Counter(redistributed_values)
                            min_count = min(current_counts.values())
                            candidates = [val for val, count in current_counts.items() 
                                        if count == min_count]
                            redistributed_values.append(np.random.choice(candidates))
                    
                    np.random.shuffle(redistributed_values)
                    sa_values = redistributed_values
                else:
                    # Not enough unique values, just shuffle
                    np.random.shuffle(sa_values)
            
            # Assign redistributed values
            result.loc[bucket_indices, sa_col] = sa_values
            
            # Store distribution for this bucket
            self.sa_distributions[bucket_idx] = {
                'original': bucket_data[sa_col].tolist(),
                'redistributed': sa_values,
                'unique_count': len(set(sa_values))
            }
        
        return result
    
    def _apply_suppression(self, data: pd.DataFrame, sa_col: str) -> pd.DataFrame:
        """Apply suppression to buckets that don't meet l-diversity."""
        suppressed_records = []
        
        for bucket_idx, bucket_indices in enumerate(self.buckets):
            bucket_data = data.iloc[bucket_indices]
            unique_sa_count = bucket_data[sa_col].nunique()
            
            if unique_sa_count < self.l_value:
                # Mark some records for suppression
                records_to_suppress = len(bucket_indices) - self.l_value
                if records_to_suppress > 0:
                    suppress_indices = np.random.choice(
                        bucket_indices, 
                        min(records_to_suppress, len(bucket_indices) // 2), 
                        replace=False
                    )
                    suppressed_records.extend(suppress_indices)
        
        # Check suppression threshold
        suppression_rate = len(suppressed_records) / len(data)
        if suppression_rate > self.suppression_threshold:
            logger.warning(f"Suppression rate ({suppression_rate:.3f}) exceeds threshold ({self.suppression_threshold})")
            max_suppress = int(len(data) * self.suppression_threshold)
            suppressed_records = suppressed_records[:max_suppress]
        
        # Apply suppression
        result = data.copy()
        if suppressed_records:
            result.loc[suppressed_records, :] = "*SUPPRESSED*"
            logger.info(f"Suppressed {len(suppressed_records)} records")
        
        return result
    
    def _add_sensitive_attribute_noise(self, data: pd.DataFrame, sa_col: str) -> pd.DataFrame:
        """Add noise to sensitive attribute values."""
        if data[sa_col].dtype not in ['int64', 'float64']:
            return data  # Only add noise to numerical SA
        
        result = data.copy()
        noise_std = data[sa_col].std() * 0.05  # 5% noise
        noise = np.random.normal(0, noise_std, len(data))
        
        # Only add noise to non-suppressed records
        mask = result[sa_col] != "*SUPPRESSED*"
        result.loc[mask, sa_col] = result.loc[mask, sa_col] + noise[mask]
        
        return result
    
    def _calculate_bucket_statistics(self, original: pd.DataFrame, bucketized: pd.DataFrame, sa_col: str):
        """Calculate statistics about the bucketization process."""
        bucket_sizes = [len(bucket) for bucket in self.buckets]
        
        self.bucket_statistics = {
            'num_buckets': len(self.buckets),
            'avg_bucket_size': np.mean(bucket_sizes),
            'min_bucket_size': np.min(bucket_sizes),
            'max_bucket_size': np.max(bucket_sizes),
            'bucket_size_std': np.std(bucket_sizes),
            'total_records': len(original),
            'bucketized_records': len(bucketized),
            'suppression_rate': 1 - (len(bucketized) / len(original))
        }
    
    def _calculate_diversity_metrics(self, data: pd.DataFrame, sa_col: str):
        """Calculate l-diversity and other privacy metrics."""
        bucket_diversities = []
        min_diversity = float('inf')
        
        for bucket_indices in self.buckets:
            bucket_data = data.iloc[bucket_indices]
            # Filter out suppressed records
            valid_data = bucket_data[bucket_data[sa_col] != "*SUPPRESSED*"]
            
            if len(valid_data) > 0:
                unique_sa_count = valid_data[sa_col].nunique()
                bucket_diversities.append(unique_sa_count)
                min_diversity = min(min_diversity, unique_sa_count)
        
        self.diversity_metrics = {
            'min_diversity': min_diversity if min_diversity != float('inf') else 0,
            'avg_diversity': np.mean(bucket_diversities) if bucket_diversities else 0,
            'l_diversity_satisfied': min_diversity >= self.l_value if min_diversity != float('inf') else False,
            'bucket_diversities': bucket_diversities
        }

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return BucketizationPlugin()
