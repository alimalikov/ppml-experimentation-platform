"""
Micro-aggregation Plugin for Data Anonymization

This plugin implements micro-aggregation, a technique that groups records into
small clusters (microdata) and replaces individual values with cluster representatives
(typically averages). This provides k-anonymity while minimizing information loss.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class MicroAggregationPlugin(Anonymizer):
    """
    Micro-aggregation plugin that groups records and replaces values with cluster statistics.
    
    Micro-aggregation works by:
    1. Grouping records into clusters of size k (minimum)
    2. Computing cluster representatives (mean, median, mode)
    3. Replacing individual values with cluster representatives
    4. Ensuring all clusters have at least k records
    5. Optimizing for minimal information loss
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Micro-aggregation"
        self.description = "Groups records into clusters and replaces values with cluster representatives"
        
        # Basic parameters
        self.k_value = 3
        self.cluster_size = 5  # Target cluster size
        self.min_cluster_size = None  # Will be set to k_value
        
        # Clustering parameters
        self.clustering_method = "kmeans"  # kmeans, hierarchical, distance_based
        self.distance_metric = "euclidean"  # euclidean, manhattan, cosine
        self.clustering_attributes = []  # Attributes used for clustering
        self.use_all_attributes = True
        
        # Aggregation parameters
        self.aggregation_method = "mean"  # mean, median, mode, centroid
        self.handle_categorical = "mode"  # mode, most_frequent, random
        self.preserve_data_types = True
        
        # Quality parameters
        self.optimize_clusters = True
        self.balance_cluster_sizes = True
        self.minimize_sse = True  # Sum of Squared Errors
        self.use_weighted_aggregation = False
        
        # Privacy parameters
        self.add_noise = False
        self.noise_level = 0.01
        self.suppress_small_clusters = True
        self.merge_small_clusters = True
        
        # Internal state
        self.clusters = {}
        self.cluster_representatives = {}
        self.cluster_assignments = {}
        self.aggregation_statistics = {}
        self.information_loss = 0.0
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Data Transformation"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Micro-aggregation specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"📊 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Micro-aggregation"):
            st.markdown("""
            **Micro-aggregation** groups similar records into small clusters and replaces
            individual values with cluster representatives (like averages). This provides
            k-anonymity while minimizing information loss.
            
            **Key Features:**
            - Groups records into k-anonymous clusters
            - Minimal information loss
            - Preserves statistical properties
            - Flexible aggregation methods
            """)
        
        # Basic parameters
        k_value = st.sidebar.slider(
            "k-anonymity value",
            min_value=2, max_value=20, value=3,
            key=f"{unique_key_prefix}_k_value",
            help="Minimum number of records per cluster"
        )
        
        cluster_size = st.sidebar.slider(
            "Target Cluster Size",
            min_value=k_value, max_value=50, value=max(5, k_value),
            key=f"{unique_key_prefix}_cluster_size",
            help="Target number of records per cluster"
        )
        
        # Clustering method
        st.sidebar.subheader("🔧 Clustering Method")
        
        clustering_method = st.sidebar.selectbox(
            "Clustering Algorithm",
            options=["kmeans", "hierarchical", "distance_based"],
            key=f"{unique_key_prefix}_clustering_method",
            help="Algorithm used for clustering records"
        )
        
        distance_metric = st.sidebar.selectbox(
            "Distance Metric",
            options=["euclidean", "manhattan", "cosine"],
            key=f"{unique_key_prefix}_distance_metric",
            help="Distance metric for clustering"
        )
        
        # Attribute selection for clustering
        use_all_attributes = st.sidebar.checkbox(
            "Use All Attributes for Clustering",
            value=True,
            key=f"{unique_key_prefix}_use_all_attributes",
            help="Use all attributes or select specific ones for clustering"
        )
        
        if not use_all_attributes:
            # Filter out sensitive attribute
            available_cols = [col for col in all_cols if col != sa_col_to_pass]
            clustering_attributes = st.sidebar.multiselect(
                "Clustering Attributes",
                options=available_cols,
                default=available_cols[:min(5, len(available_cols))],
                key=f"{unique_key_prefix}_clustering_attributes",
                help="Attributes to use for clustering"
            )
        else:
            clustering_attributes = all_cols
        
        # Aggregation method
        st.sidebar.subheader("📈 Aggregation Method")
        
        aggregation_method = st.sidebar.selectbox(
            "Aggregation Method",
            options=["mean", "median", "mode", "centroid"],
            key=f"{unique_key_prefix}_aggregation_method",
            help="Method for computing cluster representatives"
        )
        
        handle_categorical = st.sidebar.selectbox(
            "Categorical Handling",
            options=["mode", "most_frequent", "random"],
            key=f"{unique_key_prefix}_handle_categorical",
            help="How to handle categorical attributes in aggregation"
        )
        
        # Quality settings
        with st.sidebar.expander("📊 Quality Settings"):
            optimize_clusters = st.sidebar.checkbox(
                "Optimize Clusters",
                value=True,
                key=f"{unique_key_prefix}_optimize_clusters",
                help="Optimize cluster formation for minimal information loss"
            )
            
            balance_cluster_sizes = st.sidebar.checkbox(
                "Balance Cluster Sizes",
                value=True,
                key=f"{unique_key_prefix}_balance_cluster_sizes",
                help="Try to create clusters of similar sizes"
            )
            
            preserve_data_types = st.sidebar.checkbox(
                "Preserve Data Types",
                value=True,
                key=f"{unique_key_prefix}_preserve_data_types",
                help="Maintain original data types in output"
            )
        
        # Privacy settings
        with st.sidebar.expander("🔒 Privacy Settings"):
            add_noise = st.sidebar.checkbox(
                "Add Noise",
                value=False,
                key=f"{unique_key_prefix}_add_noise",
                help="Add noise to cluster representatives"
            )
            
            if add_noise:
                noise_level = st.sidebar.slider(
                    "Noise Level",
                    min_value=0.001, max_value=0.1, value=0.01, step=0.001,
                    key=f"{unique_key_prefix}_noise_level",
                    help="Level of noise to add"
                )
            else:
                noise_level = 0.0
            
            merge_small_clusters = st.sidebar.checkbox(
                "Merge Small Clusters",
                value=True,
                key=f"{unique_key_prefix}_merge_small_clusters",
                help="Merge clusters smaller than k"
            )
        
        return {
            "k_value": k_value,
            "cluster_size": cluster_size,
            "clustering_method": clustering_method,
            "distance_metric": distance_metric,
            "use_all_attributes": use_all_attributes,
            "clustering_attributes": clustering_attributes,
            "aggregation_method": aggregation_method,
            "handle_categorical": handle_categorical,
            "optimize_clusters": optimize_clusters,
            "balance_cluster_sizes": balance_cluster_sizes,
            "preserve_data_types": preserve_data_types,
            "add_noise": add_noise,
            "noise_level": noise_level,
            "merge_small_clusters": merge_small_clusters
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply micro-aggregation to the input DataFrame.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.k_value = parameters.get("k_value", 3)
            self.cluster_size = parameters.get("cluster_size", 5)
            self.clustering_method = parameters.get("clustering_method", "kmeans")
            self.distance_metric = parameters.get("distance_metric", "euclidean")
            self.use_all_attributes = parameters.get("use_all_attributes", True)
            self.clustering_attributes = parameters.get("clustering_attributes", [])
            self.aggregation_method = parameters.get("aggregation_method", "mean")
            self.handle_categorical = parameters.get("handle_categorical", "mode")
            self.optimize_clusters = parameters.get("optimize_clusters", True)
            self.balance_cluster_sizes = parameters.get("balance_cluster_sizes", True)
            self.preserve_data_types = parameters.get("preserve_data_types", True)
            self.add_noise = parameters.get("add_noise", False)
            self.noise_level = parameters.get("noise_level", 0.01)
            self.merge_small_clusters = parameters.get("merge_small_clusters", True)
            
            self.min_cluster_size = self.k_value
            
            if self.use_all_attributes:
                self.clustering_attributes = [col for col in df_input.columns if col != sa_col]
            
            if not self.clustering_attributes:
                logger.warning("No clustering attributes specified")
                return df_input
            
            logger.info(f"Starting micro-aggregation with k={self.k_value}, cluster_size={self.cluster_size}")
            
            # Prepare data for clustering
            prepared_data, encoders = self._prepare_data_for_clustering(df_input)
            
            # Perform clustering
            cluster_labels = self._perform_clustering(prepared_data)
            
            # Create clusters and ensure minimum size
            self.clusters = self._create_clusters(df_input, cluster_labels)
            
            # Merge or split clusters to meet size requirements
            if self.merge_small_clusters:
                self.clusters = self._merge_small_clusters(df_input, self.clusters)
            
            # Compute cluster representatives
            self.cluster_representatives = self._compute_cluster_representatives(df_input)
            
            # Apply micro-aggregation
            aggregated_data = self._apply_micro_aggregation(df_input)
            
            # Add noise if enabled
            if self.add_noise:
                aggregated_data = self._add_noise_to_representatives(aggregated_data)
            
            # Calculate information loss
            self.information_loss = self._calculate_information_loss(df_input, aggregated_data)
            
            # Calculate aggregation statistics
            self._calculate_aggregation_statistics(df_input, aggregated_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Micro-aggregation completed in {processing_time:.2f}s, IL: {self.information_loss:.4f}")
            
            # Add metadata
            aggregated_data.attrs['microagg_k'] = self.k_value
            aggregated_data.attrs['num_clusters'] = len(self.clusters)
            aggregated_data.attrs['information_loss'] = self.information_loss
            aggregated_data.attrs['aggregation_statistics'] = self.aggregation_statistics
            aggregated_data.attrs['processing_time'] = processing_time
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error in micro-aggregation: {str(e)}")
            st.error(f"Micro-aggregation failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "k_value": st.session_state.get(f"{unique_key_prefix}_k_value", 3),
            "cluster_size": st.session_state.get(f"{unique_key_prefix}_cluster_size", 5),
            "clustering_method": st.session_state.get(f"{unique_key_prefix}_clustering_method", "kmeans"),
            "distance_metric": st.session_state.get(f"{unique_key_prefix}_distance_metric", "euclidean"),
            "use_all_attributes": st.session_state.get(f"{unique_key_prefix}_use_all_attributes", True),
            "clustering_attributes": st.session_state.get(f"{unique_key_prefix}_clustering_attributes", []),
            "aggregation_method": st.session_state.get(f"{unique_key_prefix}_aggregation_method", "mean"),
            "handle_categorical": st.session_state.get(f"{unique_key_prefix}_handle_categorical", "mode"),
            "optimize_clusters": st.session_state.get(f"{unique_key_prefix}_optimize_clusters", True),
            "balance_cluster_sizes": st.session_state.get(f"{unique_key_prefix}_balance_cluster_sizes", True),
            "preserve_data_types": st.session_state.get(f"{unique_key_prefix}_preserve_data_types", True),
            "add_noise": st.session_state.get(f"{unique_key_prefix}_add_noise", False),
            "noise_level": st.session_state.get(f"{unique_key_prefix}_noise_level", 0.01),
            "merge_small_clusters": st.session_state.get(f"{unique_key_prefix}_merge_small_clusters", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        for key, value in config_params.items():
            st.session_state[f"{unique_key_prefix}_{key}"] = value

    def _prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare data for clustering by encoding and scaling."""
        clustering_data = data[self.clustering_attributes].copy()
        encoders = {}
        
        # Encode categorical variables
        for col in clustering_data.columns:
            if clustering_data[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                clustering_data[col] = le.fit_transform(clustering_data[col].astype(str))
                encoders[col] = le
        
        # Scale numerical variables
        numerical_cols = clustering_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            clustering_data[numerical_cols] = scaler.fit_transform(clustering_data[numerical_cols])
            encoders['scaler'] = scaler
        
        return clustering_data, encoders
    
    def _perform_clustering(self, data: pd.DataFrame) -> np.ndarray:
        """Perform clustering on the prepared data."""
        n_clusters = max(2, len(data) // self.cluster_size)
        
        try:
            if self.clustering_method == "kmeans":
                clusterer = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42,
                    n_init=10
                )
            elif self.clustering_method == "hierarchical":
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric=self.distance_metric,
                    linkage='ward' if self.distance_metric == 'euclidean' else 'complete'
                )
            else:  # distance_based
                # Simple distance-based clustering
                return self._distance_based_clustering(data, n_clusters)
            
            cluster_labels = clusterer.fit_predict(data)
            return cluster_labels
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using simple partitioning")
            return self._simple_partitioning(data, n_clusters)
    
    def _distance_based_clustering(self, data: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """Simple distance-based clustering."""
        # Initialize cluster centers randomly
        centers = data.sample(n=n_clusters, random_state=42)
        labels = np.zeros(len(data))
        
        # Assign each point to nearest center
        for i, (_, point) in enumerate(data.iterrows()):
            distances = []
            for _, center in centers.iterrows():
                if self.distance_metric == "euclidean":
                    dist = np.sqrt(np.sum((point - center) ** 2))
                elif self.distance_metric == "manhattan":
                    dist = np.sum(np.abs(point - center))
                else:  # cosine
                    dist = 1 - np.dot(point, center) / (np.linalg.norm(point) * np.linalg.norm(center) + 1e-8)
                distances.append(dist)
            
            labels[i] = np.argmin(distances)
        
        return labels.astype(int)
    
    def _simple_partitioning(self, data: pd.DataFrame, n_clusters: int) -> np.ndarray:
        """Simple partitioning when clustering fails."""
        labels = np.arange(len(data)) % n_clusters
        np.random.shuffle(labels)
        return labels
    
    def _create_clusters(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, List[int]]:
        """Create clusters from cluster labels."""
        clusters = defaultdict(list)
        
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
        
        # Convert to regular dict
        return dict(clusters)
    
    def _merge_small_clusters(self, data: pd.DataFrame, clusters: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Merge clusters that are smaller than the minimum size."""
        merged_clusters = {}
        small_clusters = []
        
        # Separate large and small clusters
        for cluster_id, indices in clusters.items():
            if len(indices) >= self.min_cluster_size:
                merged_clusters[cluster_id] = indices
            else:
                small_clusters.extend(indices)
        
        # Merge small clusters with existing ones or create new ones
        if small_clusters:
            if merged_clusters:
                # Distribute small cluster members among existing clusters
                cluster_ids = list(merged_clusters.keys())
                for i, idx in enumerate(small_clusters):
                    target_cluster = cluster_ids[i % len(cluster_ids)]
                    merged_clusters[target_cluster].append(idx)
            else:
                # Create new clusters from small cluster members
                cluster_id = 0
                for i in range(0, len(small_clusters), self.min_cluster_size):
                    cluster_indices = small_clusters[i:i + self.min_cluster_size]
                    if len(cluster_indices) >= self.min_cluster_size:
                        merged_clusters[cluster_id] = cluster_indices
                        cluster_id += 1
                    elif merged_clusters:
                        # Add remaining to last cluster
                        last_cluster = max(merged_clusters.keys())
                        merged_clusters[last_cluster].extend(cluster_indices)
        
        logger.info(f"Merged clusters: {len(clusters)} -> {len(merged_clusters)}")
        return merged_clusters
    
    def _compute_cluster_representatives(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Compute representative values for each cluster."""
        representatives = {}
        
        for cluster_id, indices in self.clusters.items():
            cluster_data = data.iloc[indices]
            cluster_rep = {}
            
            for col in data.columns:
                if data[col].dtype in ['object', 'category']:
                    # Categorical attribute
                    if self.handle_categorical == "mode":
                        cluster_rep[col] = cluster_data[col].mode().iloc[0] if len(cluster_data[col].mode()) > 0 else cluster_data[col].iloc[0]
                    elif self.handle_categorical == "most_frequent":
                        cluster_rep[col] = cluster_data[col].value_counts().index[0]
                    else:  # random
                        cluster_rep[col] = cluster_data[col].sample(1).iloc[0]
                else:
                    # Numerical attribute
                    if self.aggregation_method == "mean":
                        cluster_rep[col] = cluster_data[col].mean()
                    elif self.aggregation_method == "median":
                        cluster_rep[col] = cluster_data[col].median()
                    elif self.aggregation_method == "mode":
                        mode_values = cluster_data[col].mode()
                        cluster_rep[col] = mode_values.iloc[0] if len(mode_values) > 0 else cluster_data[col].mean()
                    else:  # centroid
                        cluster_rep[col] = cluster_data[col].mean()  # Same as mean for single attribute
            
            representatives[cluster_id] = cluster_rep
        
        return representatives
    
    def _apply_micro_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply micro-aggregation by replacing values with cluster representatives."""
        result = data.copy()
        
        for cluster_id, indices in self.clusters.items():
            representative = self.cluster_representatives[cluster_id]
            
            for col in data.columns:
                if self.preserve_data_types:
                    # Try to preserve original data type
                    try:
                        if data[col].dtype in ['int64', 'int32']:
                            result.loc[indices, col] = int(round(representative[col]))
                        elif data[col].dtype in ['float64', 'float32']:
                            result.loc[indices, col] = float(representative[col])
                        else:
                            result.loc[indices, col] = representative[col]
                    except (ValueError, TypeError):
                        result.loc[indices, col] = representative[col]
                else:
                    result.loc[indices, col] = representative[col]
        
        return result
    
    def _add_noise_to_representatives(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add noise to cluster representatives for additional privacy."""
        result = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            std_dev = data[col].std()
            noise = np.random.normal(0, std_dev * self.noise_level, len(data))
            result[col] = result[col] + noise
        
        return result
    
    def _calculate_information_loss(self, original: pd.DataFrame, aggregated: pd.DataFrame) -> float:
        """Calculate information loss due to micro-aggregation."""
        total_loss = 0.0
        count = 0
        
        for col in original.columns:
            if original[col].dtype in ['int64', 'float64']:
                # Sum of Squared Errors for numerical attributes
                sse = np.sum((original[col] - aggregated[col]) ** 2)
                total_variance = np.sum((original[col] - original[col].mean()) ** 2)
                
                if total_variance > 0:
                    normalized_loss = sse / total_variance
                    total_loss += normalized_loss
                    count += 1
            else:
                # Misclassification rate for categorical attributes
                mismatches = (original[col] != aggregated[col]).sum()
                normalized_loss = mismatches / len(original)
                total_loss += normalized_loss
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _calculate_aggregation_statistics(self, original: pd.DataFrame, aggregated: pd.DataFrame):
        """Calculate statistics about the aggregation process."""
        cluster_sizes = [len(indices) for indices in self.clusters.values()]
        
        self.aggregation_statistics = {
            'num_clusters': len(self.clusters),
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'cluster_size_std': np.std(cluster_sizes),
            'information_loss': self.information_loss,
            'k_anonymity_satisfied': np.min(cluster_sizes) >= self.k_value,
            'clustering_method': self.clustering_method,
            'aggregation_method': self.aggregation_method
        }

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return MicroAggregationPlugin()
