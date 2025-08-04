"""
Anatomy Anonymization Plugin for Data Anonymization

This plugin implements the Anatomy algorithm for l-diversity, which separates
the quasi-identifiers from sensitive attributes and creates a mapping between
them. This approach preserves more utility than traditional generalization
while providing strong privacy protection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import secrets
from collections import defaultdict, Counter
import uuid

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class AnatomyAnonymizationPlugin(Anonymizer):
    """
    Anatomy anonymization plugin that implements the Anatomy algorithm for l-diversity.
    
    The Anatomy algorithm works by:
    1. Separating quasi-identifiers (QI) from sensitive attributes (SA)
    2. Creating groups of records with identical QI values
    3. Randomly distributing sensitive values among groups
    4. Creating separate QI-table and SA-table with group identifiers
    5. Ensuring l-diversity in each group
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Anatomy l-Diversity"
        self.description = "Separates quasi-identifiers from sensitive attributes for l-diversity"
        
        # l-diversity parameters
        self.l_value = 3
        self.diversity_type = "distinct"  # distinct, entropy, recursive
        self.min_group_size = 5
        
        # Anatomy algorithm parameters
        self.separation_strategy = "complete"  # complete, partial, adaptive
        self.group_formation = "equivalence"  # equivalence, clustering, random
        self.distribution_method = "uniform"  # uniform, weighted, entropy_maximizing
        
        # Privacy parameters
        self.add_noise = True
        self.noise_level = 0.1
        self.use_dummy_records = True
        self.dummy_ratio = 0.05
        
        # Table generation parameters
        self.qi_table_name = "QI_Table"
        self.sa_table_name = "SA_Table"
        self.group_id_column = "GroupID"
        self.record_id_column = "RecordID"
        
        # Quality parameters
        self.preserve_correlations = True
        self.maintain_distributions = True
        self.optimize_utility = True
        
        # Internal state
        self.qi_groups = {}
        self.sa_distributions = {}
        self.group_mappings = {}
        self.qi_table = None
        self.sa_table = None
        self.anatomy_statistics = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Clustering & Grouping"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Anatomy l-diversity specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🧬 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Anatomy l-Diversity"):
            st.markdown("""
            **Anatomy l-Diversity** separates quasi-identifiers from sensitive attributes
            and creates separate tables with group identifiers. This preserves more
            utility than traditional generalization while ensuring l-diversity.
            
            **Key Features:**
            - Separation of QI and SA attributes
            - Preserves exact QI values
            - Flexible sensitive attribute distribution
            - Strong privacy guarantees
            """)
        
        # Basic l-diversity parameters
        l_value = st.sidebar.slider(
            "l-diversity value",
            min_value=2, max_value=10, value=3,
            key=f"{unique_key_prefix}_l_value",
            help="Minimum number of distinct sensitive values per group"
        )
        
        diversity_type = st.sidebar.selectbox(
            "Diversity Type",
            options=["distinct", "entropy", "recursive"],
            key=f"{unique_key_prefix}_diversity_type",
            help="Type of l-diversity to ensure"
        )
        
        min_group_size = st.sidebar.slider(
            "Minimum Group Size",
            min_value=3, max_value=20, value=5,
            key=f"{unique_key_prefix}_min_group_size",
            help="Minimum number of records per group"
        )
        
        # Algorithm parameters
        st.sidebar.subheader("🔧 Algorithm Parameters")
        
        separation_strategy = st.sidebar.selectbox(
            "Separation Strategy",
            options=["complete", "partial", "adaptive"],
            key=f"{unique_key_prefix}_separation_strategy",
            help="How to separate QI from SA attributes"
        )
        
        group_formation = st.sidebar.selectbox(
            "Group Formation",
            options=["equivalence", "clustering", "random"],
            key=f"{unique_key_prefix}_group_formation",
            help="Method for forming equivalence groups"
        )
        
        distribution_method = st.sidebar.selectbox(
            "Distribution Method",
            options=["uniform", "weighted", "entropy_maximizing"],
            key=f"{unique_key_prefix}_distribution_method",
            help="How to distribute sensitive values"
        )
        
        # Privacy settings
        with st.sidebar.expander("🔒 Privacy Settings"):
            add_noise = st.sidebar.checkbox(
                "Add Noise",
                value=True,
                key=f"{unique_key_prefix}_add_noise",
                help="Add noise to numerical attributes"
            )
            
            if add_noise:
                noise_level = st.sidebar.slider(
                    "Noise Level",
                    min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                    key=f"{unique_key_prefix}_noise_level",
                    help="Level of noise to add"
                )
            else:
                noise_level = 0.0
            
            use_dummy_records = st.sidebar.checkbox(
                "Use Dummy Records",
                value=True,
                key=f"{unique_key_prefix}_use_dummy_records",
                help="Add dummy records to enhance privacy"
            )
            
            if use_dummy_records:
                dummy_ratio = st.sidebar.slider(
                    "Dummy Record Ratio",
                    min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                    key=f"{unique_key_prefix}_dummy_ratio",
                    help="Ratio of dummy records to add"
                )
            else:
                dummy_ratio = 0.0
        
        # Quality settings
        with st.sidebar.expander("📊 Quality Settings"):
            preserve_correlations = st.sidebar.checkbox(
                "Preserve Correlations",
                value=True,
                key=f"{unique_key_prefix}_preserve_correlations",
                help="Try to preserve correlations between attributes"
            )
            
            maintain_distributions = st.sidebar.checkbox(
                "Maintain Distributions",
                value=True,
                key=f"{unique_key_prefix}_maintain_distributions",
                help="Maintain original attribute distributions"
            )
        
        return {
            "l_value": l_value,
            "diversity_type": diversity_type,
            "min_group_size": min_group_size,
            "separation_strategy": separation_strategy,
            "group_formation": group_formation,
            "distribution_method": distribution_method,
            "add_noise": add_noise,
            "noise_level": noise_level,
            "use_dummy_records": use_dummy_records,
            "dummy_ratio": dummy_ratio,
            "preserve_correlations": preserve_correlations,
            "maintain_distributions": maintain_distributions
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Anatomy l-diversity algorithm to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.l_value = parameters.get("l_value", 3)
            self.diversity_type = parameters.get("diversity_type", "distinct")
            self.min_group_size = parameters.get("min_group_size", 5)
            self.separation_strategy = parameters.get("separation_strategy", "complete")
            self.group_formation = parameters.get("group_formation", "equivalence")
            self.distribution_method = parameters.get("distribution_method", "uniform")
            self.add_noise = parameters.get("add_noise", True)
            self.noise_level = parameters.get("noise_level", 0.1)
            self.use_dummy_records = parameters.get("use_dummy_records", True)
            self.dummy_ratio = parameters.get("dummy_ratio", 0.05)
            self.preserve_correlations = parameters.get("preserve_correlations", True)
            self.maintain_distributions = parameters.get("maintain_distributions", True)
            
            if sa_col is None:
                st.warning("Anatomy algorithm requires a sensitive attribute. Using first column as sensitive attribute.")
                sa_col = df_input.columns[0]
            
            logger.info(f"Starting Anatomy l-diversity with l={self.l_value}, SA={sa_col}")
            
            # Separate quasi-identifiers from sensitive attributes
            qi_columns, sa_columns = self._separate_attributes(df_input, sa_col)
            
            # Add dummy records if enabled
            working_data = df_input.copy()
            if self.use_dummy_records:
                working_data = self._add_dummy_records(working_data, qi_columns, sa_columns)
            
            # Form equivalence groups based on QI values
            self.qi_groups = self._form_equivalence_groups(working_data, qi_columns)
            
            # Ensure groups meet minimum size requirement
            self.qi_groups = self._merge_small_groups(self.qi_groups)
            
            # Distribute sensitive values to ensure l-diversity
            self.sa_distributions = self._distribute_sensitive_values(working_data, sa_columns)
            
            # Create QI and SA tables
            self.qi_table, self.sa_table = self._create_anatomy_tables(working_data, qi_columns, sa_columns)
            
            # Create final anonymized dataset
            anonymized_data = self._create_anonymized_dataset()
            
            # Add noise if enabled
            if self.add_noise:
                anonymized_data = self._add_noise_to_data(anonymized_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Anatomy anonymization completed in {processing_time:.2f}s")
            
            # Store statistics
            self.anatomy_statistics = {
                'num_groups': len(self.qi_groups),
                'avg_group_size': np.mean([len(group) for group in self.qi_groups.values()]),
                'l_diversity_achieved': self._verify_l_diversity(),
                'processing_time': processing_time
            }
            
            # Add metadata
            anonymized_data.attrs['anatomy_l'] = self.l_value
            anonymized_data.attrs['num_groups'] = len(self.qi_groups)
            anonymized_data.attrs['anatomy_statistics'] = self.anatomy_statistics
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Error in Anatomy anonymization: {str(e)}")
            st.error(f"Anatomy anonymization failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "l_value": st.session_state.get(f"{unique_key_prefix}_l_value", 3),
            "diversity_type": st.session_state.get(f"{unique_key_prefix}_diversity_type", "distinct"),
            "min_group_size": st.session_state.get(f"{unique_key_prefix}_min_group_size", 5),
            "separation_strategy": st.session_state.get(f"{unique_key_prefix}_separation_strategy", "complete"),
            "group_formation": st.session_state.get(f"{unique_key_prefix}_group_formation", "equivalence"),
            "distribution_method": st.session_state.get(f"{unique_key_prefix}_distribution_method", "uniform"),
            "add_noise": st.session_state.get(f"{unique_key_prefix}_add_noise", True),
            "noise_level": st.session_state.get(f"{unique_key_prefix}_noise_level", 0.1),
            "use_dummy_records": st.session_state.get(f"{unique_key_prefix}_use_dummy_records", True),
            "dummy_ratio": st.session_state.get(f"{unique_key_prefix}_dummy_ratio", 0.05),
            "preserve_correlations": st.session_state.get(f"{unique_key_prefix}_preserve_correlations", True),
            "maintain_distributions": st.session_state.get(f"{unique_key_prefix}_maintain_distributions", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _separate_attributes(self, data: pd.DataFrame, sa_col: str) -> Tuple[List[str], List[str]]:
        """Separate quasi-identifiers from sensitive attributes."""
        if self.separation_strategy == "complete":
            # All non-SA columns are QI
            qi_columns = [col for col in data.columns if col != sa_col]
            sa_columns = [sa_col]
        elif self.separation_strategy == "partial":
            # User-defined separation (simplified: use heuristics)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Keep some categorical as QI, some numerical as SA
            qi_columns = categorical_cols[:len(categorical_cols)//2] + numerical_cols[:len(numerical_cols)//2]
            sa_columns = [col for col in data.columns if col not in qi_columns]
        else:  # adaptive
            # Adaptively determine based on data characteristics
            qi_columns = []
            sa_columns = [sa_col]
            
            for col in data.columns:
                if col != sa_col:
                    # Use cardinality as heuristic
                    cardinality = data[col].nunique()
                    if cardinality < len(data) * 0.1:  # Low cardinality -> likely QI
                        qi_columns.append(col)
                    else:  # High cardinality -> likely sensitive
                        sa_columns.append(col)
        
        return qi_columns, sa_columns
    
    def _add_dummy_records(self, data: pd.DataFrame, qi_columns: List[str], sa_columns: List[str]) -> pd.DataFrame:
        """Add dummy records to enhance privacy."""
        num_dummies = int(len(data) * self.dummy_ratio)
        dummy_records = []
        
        for _ in range(num_dummies):
            dummy_record = {}
            
            # Generate dummy QI values
            for col in qi_columns:
                if data[col].dtype in ['object', 'category']:
                    # Sample from existing values
                    dummy_record[col] = np.random.choice(data[col].unique())
                else:
                    # Generate value within range
                    min_val, max_val = data[col].min(), data[col].max()
                    dummy_record[col] = np.random.uniform(min_val, max_val)
            
            # Generate dummy SA values
            for col in sa_columns:
                if data[col].dtype in ['object', 'category']:
                    dummy_record[col] = f"dummy_{secrets.token_hex(4)}"
                else:
                    min_val, max_val = data[col].min(), data[col].max()
                    dummy_record[col] = np.random.uniform(min_val, max_val)
            
            dummy_records.append(dummy_record)
        
        dummy_df = pd.DataFrame(dummy_records)
        return pd.concat([data, dummy_df], ignore_index=True)
    
    def _form_equivalence_groups(self, data: pd.DataFrame, qi_columns: List[str]) -> Dict[str, List[int]]:
        """Form equivalence groups based on QI values."""
        groups = defaultdict(list)
        
        if self.group_formation == "equivalence":
            # Group by exact QI values
            for idx, row in data.iterrows():
                qi_signature = tuple(row[qi_columns].values)
                groups[str(qi_signature)].append(idx)
        
        elif self.group_formation == "clustering":
            # Use clustering for numerical attributes
            from sklearn.cluster import KMeans
            
            # Encode categorical variables
            encoded_data = data[qi_columns].copy()
            for col in qi_columns:
                if encoded_data[col].dtype in ['object', 'category']:
                    encoded_data[col] = pd.Categorical(encoded_data[col]).codes
            
            # Determine number of clusters
            n_clusters = max(2, len(data) // self.min_group_size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(encoded_data)
            
            for idx, label in enumerate(cluster_labels):
                groups[f"cluster_{label}"].append(idx)
        
        else:  # random
            # Random grouping
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            
            group_size = max(self.min_group_size, len(data) // (len(data) // self.min_group_size))
            for i in range(0, len(indices), group_size):
                group_indices = indices[i:i + group_size]
                if group_indices:
                    groups[f"random_{i//group_size}"] = group_indices
        
        return dict(groups)
    
    def _merge_small_groups(self, groups: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Merge groups that are too small."""
        merged_groups = {}
        small_groups = []
        
        # Separate large and small groups
        for group_id, indices in groups.items():
            if len(indices) >= self.min_group_size:
                merged_groups[group_id] = indices
            else:
                small_groups.extend(indices)
        
        # Merge small groups
        if small_groups:
            # Try to merge with existing groups first
            for group_id in list(merged_groups.keys()):
                if len(small_groups) == 0:
                    break
                
                group_size = len(merged_groups[group_id])
                can_add = min(len(small_groups), self.min_group_size - (group_size % self.min_group_size))
                
                if can_add > 0:
                    merged_groups[group_id].extend(small_groups[:can_add])
                    small_groups = small_groups[can_add:]
            
            # Create new groups for remaining small groups
            group_counter = len(merged_groups)
            while small_groups:
                group_size = min(len(small_groups), self.min_group_size)
                merged_groups[f"merged_{group_counter}"] = small_groups[:group_size]
                small_groups = small_groups[group_size:]
                group_counter += 1
        
        return merged_groups
    
    def _distribute_sensitive_values(self, data: pd.DataFrame, sa_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """Distribute sensitive values among groups to ensure l-diversity."""
        distributions = {}
        
        for group_id, group_indices in self.qi_groups.items():
            group_data = data.iloc[group_indices]
            group_distribution = {}
            
            for sa_col in sa_columns:
                sa_values = group_data[sa_col].tolist()
                
                if self.distribution_method == "uniform":
                    # Uniform distribution
                    unique_values = data[sa_col].unique()
                    if len(unique_values) >= self.l_value:
                        selected_values = np.random.choice(unique_values, 
                                                         size=min(len(sa_values), self.l_value), 
                                                         replace=False)
                        # Distribute uniformly
                        distributed_values = []
                        for i, _ in enumerate(sa_values):
                            distributed_values.append(selected_values[i % len(selected_values)])
                        group_distribution[sa_col] = distributed_values
                    else:
                        group_distribution[sa_col] = sa_values
                
                elif self.distribution_method == "weighted":
                    # Weighted by original distribution
                    value_counts = data[sa_col].value_counts()
                    weights = value_counts / value_counts.sum()
                    
                    distributed_values = np.random.choice(
                        value_counts.index, 
                        size=len(sa_values),
                        p=weights,
                        replace=True
                    ).tolist()
                    group_distribution[sa_col] = distributed_values
                
                else:  # entropy_maximizing
                    # Maximize entropy in distribution
                    unique_values = data[sa_col].unique()
                    if len(unique_values) >= self.l_value:
                        # Ensure at least l distinct values
                        required_values = unique_values[:self.l_value]
                        remaining_slots = len(sa_values) - self.l_value
                        
                        distributed_values = list(required_values)
                        if remaining_slots > 0:
                            additional_values = np.random.choice(
                                unique_values, 
                                size=remaining_slots,
                                replace=True
                            ).tolist()
                            distributed_values.extend(additional_values)
                        
                        np.random.shuffle(distributed_values)
                        group_distribution[sa_col] = distributed_values
                    else:
                        group_distribution[sa_col] = sa_values
            
            distributions[group_id] = group_distribution
        
        return distributions
    
    def _create_anatomy_tables(self, data: pd.DataFrame, qi_columns: List[str], 
                             sa_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create separate QI and SA tables."""
        # Create QI table
        qi_records = []
        sa_records = []
        
        for group_id, group_indices in self.qi_groups.items():
            group_data = data.iloc[group_indices]
            
            # QI table entries
            for idx, (_, row) in enumerate(group_data.iterrows()):
                qi_record = {self.group_id_column: group_id, self.record_id_column: f"{group_id}_{idx}"}
                qi_record.update(row[qi_columns].to_dict())
                qi_records.append(qi_record)
            
            # SA table entries
            group_sa_distribution = self.sa_distributions[group_id]
            for idx, (_, row) in enumerate(group_data.iterrows()):
                sa_record = {self.group_id_column: group_id, self.record_id_column: f"{group_id}_{idx}"}
                
                # Use distributed sensitive values
                for sa_col in sa_columns:
                    if sa_col in group_sa_distribution:
                        sa_record[sa_col] = group_sa_distribution[sa_col][idx]
                    else:
                        sa_record[sa_col] = row[sa_col]
                
                sa_records.append(sa_record)
        
        qi_table = pd.DataFrame(qi_records)
        sa_table = pd.DataFrame(sa_records)
        
        return qi_table, sa_table
    
    def _create_anonymized_dataset(self) -> pd.DataFrame:
        """Create the final anonymized dataset by joining QI and SA tables."""
        # Join QI and SA tables on group and record IDs
        anonymized = pd.merge(
            self.qi_table, 
            self.sa_table, 
            on=[self.group_id_column, self.record_id_column],
            how='inner'
        )
        
        # Remove temporary ID columns
        result_columns = [col for col in anonymized.columns 
                         if col not in [self.group_id_column, self.record_id_column]]
        
        return anonymized[result_columns]
    
    def _add_noise_to_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add noise to numerical attributes."""
        result = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Add Gaussian noise
            noise_std = data[col].std() * self.noise_level
            noise = np.random.normal(0, noise_std, len(data))
            result[col] = data[col] + noise
        
        return result
    
    def _verify_l_diversity(self) -> bool:
        """Verify that l-diversity is achieved in all groups."""
        for group_id, distribution in self.sa_distributions.items():
            for sa_col, values in distribution.items():
                unique_count = len(set(values))
                if unique_count < self.l_value:
                    return False
        return True

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return AnatomyAnonymizationPlugin()
