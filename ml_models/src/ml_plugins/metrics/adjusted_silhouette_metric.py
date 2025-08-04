import streamlit as st
import numpy as np
from typing import Any, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_metric_plugin import MetricPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_metric_plugin import MetricPlugin

class AdjustedSilhouetteMetric(MetricPlugin):
    def __init__(self):
        super().__init__()
        self._name = "Adjusted Silhouette Score"
        self._description = "Enhanced silhouette score with penalty for imbalanced clusters and noise handling"
        self._category = "Clustering"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (-1.0, 1.0)
        
    def get_name(self) -> str:
        return self._name
        
    def get_description(self) -> str:
        return self._description
        
    def get_category(self) -> str:
        return self._category
        
    def supports_classification(self) -> bool:
        return self._supports_classification
        
    def supports_regression(self) -> bool:
        return self._supports_regression
        
    def requires_probabilities(self) -> bool:
        return self._requires_probabilities
        
    def is_higher_better(self) -> bool:
        return self._higher_is_better
        
    def get_value_range(self) -> tuple:
        return self._range
        
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        """Calculate the adjusted silhouette score with cluster balance penalty"""
        try:
            # Check if we have enough unique clusters
            unique_clusters = np.unique(y_pred)
            n_clusters = len(unique_clusters)
            
            if n_clusters < 2:
                return -1.0  # Worst possible score for single cluster
            
            if n_clusters >= len(y_pred):
                return -1.0  # Worst possible score for too many clusters
            
            # Calculate basic silhouette score
            # Note: For this metric, we treat y_true as the feature data (X)
            # and y_pred as the cluster labels, which is common in clustering evaluation
            if len(y_true.shape) == 1:
                # If y_true is 1D, create a simple 2D representation
                X = np.column_stack([y_true, y_pred])
            else:
                X = y_true
            
            try:
                base_silhouette = silhouette_score(X, y_pred)
            except Exception:
                # Fallback: use a simple distance-based calculation
                base_silhouette = self._calculate_simple_silhouette(X, y_pred)
            
            # Calculate cluster balance penalty
            cluster_sizes = np.bincount(y_pred)
            cluster_sizes = cluster_sizes[cluster_sizes > 0]  # Remove empty clusters
            
            # Calculate balance ratio (how evenly distributed are the clusters)
            if len(cluster_sizes) > 1:
                balance_ratio = np.min(cluster_sizes) / np.max(cluster_sizes)
                balance_penalty = 1.0 - (1.0 - balance_ratio) * 0.3  # Max 30% penalty
            else:
                balance_penalty = 1.0
            
            # Calculate noise penalty (clusters that are too small)
            total_points = len(y_pred)
            min_cluster_size = max(2, total_points * 0.05)  # At least 5% of data or 2 points
            noise_clusters = np.sum(cluster_sizes < min_cluster_size)
            noise_penalty = 1.0 - (noise_clusters / n_clusters) * 0.2  # Max 20% penalty
            
            # Calculate compactness bonus using within-cluster variance
            compactness_bonus = self._calculate_compactness_bonus(X, y_pred)
            
            # Combine all factors
            adjusted_score = base_silhouette * balance_penalty * noise_penalty * compactness_bonus
            
            return float(np.clip(adjusted_score, -1.0, 1.0))
            
        except Exception as e:
            raise ValueError(f"Error calculating adjusted silhouette score: {str(e)}")
    
    def _calculate_simple_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Simple fallback silhouette calculation"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return -1.0
            
            silhouette_scores = []
            
            for i, point in enumerate(X):
                # Calculate mean distance to points in same cluster
                same_cluster_mask = labels == labels[i]
                same_cluster_points = X[same_cluster_mask]
                
                if len(same_cluster_points) > 1:
                    a_i = np.mean([np.linalg.norm(point - other) 
                                  for other in same_cluster_points if not np.array_equal(point, other)])
                else:
                    a_i = 0
                
                # Calculate mean distance to nearest different cluster
                min_b_i = float('inf')
                for label in unique_labels:
                    if label != labels[i]:
                        other_cluster_points = X[labels == label]
                        if len(other_cluster_points) > 0:
                            b_i = np.mean([np.linalg.norm(point - other) for other in other_cluster_points])
                            min_b_i = min(min_b_i, b_i)
                
                if min_b_i == float('inf'):
                    silhouette_scores.append(0)
                else:
                    if max(a_i, min_b_i) > 0:
                        silhouette_scores.append((min_b_i - a_i) / max(a_i, min_b_i))
                    else:
                        silhouette_scores.append(0)
            
            return np.mean(silhouette_scores)
            
        except Exception:
            return 0.0
    
    def _calculate_compactness_bonus(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate bonus for compact, well-separated clusters"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 1.0
            
            # Calculate within-cluster sum of squares (WCSS)
            wcss = 0
            total_points = 0
            
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    cluster_wcss = np.sum([(np.linalg.norm(point - centroid) ** 2) for point in cluster_points])
                    wcss += cluster_wcss
                    total_points += len(cluster_points)
            
            if total_points == 0:
                return 1.0
            
            # Normalize WCSS and convert to bonus (lower WCSS = higher bonus)
            avg_wcss = wcss / total_points
            # Create a bonus between 0.9 and 1.1 based on compactness
            compactness_bonus = 1.0 + 0.1 * np.exp(-avg_wcss)
            
            return min(compactness_bonus, 1.1)
            
        except Exception:
            return 1.0
        
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of the metric value"""
        if value >= 0.7:
            return "Excellent - Very well-defined, balanced clusters"
        elif value >= 0.5:
            return "Good - Clear cluster structure with good separation"
        elif value >= 0.3:
            return "Fair - Moderate clustering with some overlap"
        elif value >= 0.1:
            return "Poor - Weak cluster structure"
        elif value >= -0.2:
            return "Very Poor - Clusters barely distinguishable"
        else:
            return "Invalid - No meaningful cluster structure detected"

def get_metric_plugin() -> MetricPlugin:
    return AdjustedSilhouetteMetric()