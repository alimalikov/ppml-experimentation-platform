import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Try to import Radius Neighbors with graceful fallback
try:
    from sklearn.neighbors import RadiusNeighborsClassifier, NearestNeighbors
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RadiusNeighborsClassifier = None
    NearestNeighbors = None

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

class RadiusNeighborsClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Radius Neighbors Classifier Plugin - Distance-Based Classification
    
    Radius Neighbors is a variant of nearest neighbors that uses a fixed radius
    to define the neighborhood instead of a fixed number of neighbors. It classifies
    instances based on the majority vote of all neighbors within a specified radius.
    This approach is particularly useful for density-based classification and 
    handling variable-density regions in the data.
    """
    
    def __init__(self,
                 radius=1.0,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 outlier_label=None,
                 n_jobs=None,
                 # Advanced parameters
                 auto_scaling=True,
                 scaling_method='standard',
                 auto_radius_estimation=False,
                 radius_estimation_method='percentile',
                 outlier_detection=True,
                 min_neighbors_threshold=1,
                 adaptive_radius=False,
                 density_analysis=True):
        """
        Initialize Radius Neighbors Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        radius : float, default=1.0
            Range of parameter space to use for nearest neighbors queries
        weights : str or callable, default='uniform'
            Weight function used in prediction ('uniform', 'distance', or callable)
        algorithm : str, default='auto'
            Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        leaf_size : int, default=30
            Leaf size passed to BallTree or cKDTree
        p : int, default=2
            Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
        metric : str or callable, default='minkowski'
            Distance metric to use
        metric_params : dict, optional
            Additional keyword arguments for the metric function
        outlier_label : int or None, default=None
            Label for outliers (points with no neighbors within radius)
        n_jobs : int, optional
            Number of parallel jobs to run for neighbors search
        auto_scaling : bool, default=True
            Whether to automatically scale features
        scaling_method : str, default='standard'
            Scaling method ('standard', 'minmax', 'robust')
        auto_radius_estimation : bool, default=False
            Automatically estimate optimal radius from data
        radius_estimation_method : str, default='percentile'
            Method for radius estimation ('percentile', 'knn', 'density')
        outlier_detection : bool, default=True
            Enable outlier detection and handling
        min_neighbors_threshold : int, default=1
            Minimum neighbors required for classification
        adaptive_radius : bool, default=False
            Use adaptive radius based on local density
        density_analysis : bool, default=True
            Perform density analysis of the feature space
        """
        super().__init__()
        
        # Core Radius Neighbors parameters
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.outlier_label = outlier_label
        self.n_jobs = n_jobs
        
        # Advanced parameters
        self.auto_scaling = auto_scaling
        self.scaling_method = scaling_method
        self.auto_radius_estimation = auto_radius_estimation
        self.radius_estimation_method = radius_estimation_method
        self.outlier_detection = outlier_detection
        self.min_neighbors_threshold = min_neighbors_threshold
        self.adaptive_radius = adaptive_radius
        self.density_analysis = density_analysis
        
        # Plugin metadata
        self._name = "Radius Neighbors"
        self._description = "Distance-based classification using fixed radius neighborhoods instead of k nearest neighbors."
        self._category = "Instance-Based"
        self._algorithm_type = "Distance-Based Classifier"
        self._paper_reference = "Dasarathy, B. V. (1991). Nearest neighbor (NN) norms: NN pattern classification techniques."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 15
        self._handles_missing_values = False
        self._requires_scaling = True  # CRITICAL for radius-based methods!
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = False
        self._provides_probabilities = True
        self._handles_categorical = False
        self._ensemble_method = False
        self._lazy_learning = True
        self._non_parametric = True
        self._distance_based = True
        self._radius_based = True
        self._density_sensitive = True
        self._outlier_robust = True
        self._interpretable = True
        self._memory_intensive = True
        self._scalable = False
        
        # Internal attributes
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.training_data_ = None
        self.training_labels_ = None
        self.optimal_radius_ = None
        self.radius_analysis_ = None
        self.density_analysis_ = None
        self.outlier_analysis_ = None
        
    def get_name(self) -> str:
        """Return the algorithm name"""
        return self._name
        
    def get_description(self) -> str:
        """Return detailed algorithm description"""
        return self._description
        
    def get_category(self) -> str:
        """Return algorithm category"""
        return self._category
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Return comprehensive algorithm information"""
        return {
            "name": self._name,
            "category": self._category,
            "type": self._algorithm_type,
            "description": self._description,
            "paper_reference": self._paper_reference,
            "framework": "scikit-learn",
            "year_introduced": 1970,
            "key_characteristics": {
                "radius_based": "Uses fixed radius instead of fixed k",
                "density_sensitive": "Naturally adapts to varying data density",
                "outlier_robust": "Can identify outliers as points with no neighbors",
                "distance_dependent": "Classification based on distance threshold",
                "lazy_learning": "No explicit training phase",
                "local_learning": "Decisions based on local radius neighborhoods"
            },
            "algorithm_mechanics": {
                "training_phase": "Store all training instances",
                "prediction_process": [
                    "For each query point:",
                    "Find all training points within radius",
                    "If no neighbors found, classify as outlier",
                    "Apply weighting scheme to neighbors",
                    "Take majority vote for classification",
                    "Return class probabilities based on neighbor distribution"
                ],
                "radius_definition": "All points within distance ≤ radius are neighbors",
                "outlier_handling": "Points with no neighbors can be flagged or assigned default label",
                "density_adaptation": "Dense regions have many neighbors, sparse regions have few"
            },
            "advantages_over_knn": [
                "Naturally handles variable density regions",
                "Can identify outliers and anomalies",
                "More intuitive radius parameter than k",
                "Better for density-based classification",
                "Adapts neighborhood size to local density",
                "Robust to isolated noise points",
                "Natural uncertainty quantification",
                "Better for imbalanced datasets in some cases"
            ],
            "strengths": [
                "Density-aware classification",
                "Natural outlier detection",
                "Intuitive radius parameter",
                "Handles variable density well",
                "No assumption about neighborhood size",
                "Robust to isolated noise",
                "Good for anomaly detection",
                "Interpretable decision boundaries",
                "Confidence estimation via neighbor count",
                "Suitable for non-uniform data distributions",
                "Can handle empty neighborhoods gracefully"
            ],
            "weaknesses": [
                "Radius parameter can be difficult to set",
                "Very sensitive to feature scaling",
                "Performance depends heavily on radius choice",
                "Can have many empty neighborhoods with poor radius",
                "Computationally expensive for large datasets",
                "Memory intensive (stores all training data)",
                "Curse of dimensionality affects radius interpretation",
                "May have inconsistent neighborhood sizes",
                "Requires domain knowledge for radius setting",
                "Slow prediction time for large datasets"
            ],
            "ideal_use_cases": [
                "Datasets with variable density regions",
                "Anomaly and outlier detection",
                "Geographic or spatial data analysis",
                "Time series classification with varying patterns",
                "Medical diagnosis with rare conditions",
                "Quality control and defect detection",
                "Fraud detection in financial data",
                "Network intrusion detection",
                "Environmental monitoring",
                "Clustering validation",
                "Density-based pattern recognition",
                "Scientific data analysis with natural distance thresholds"
            ],
            "radius_setting_strategies": {
                "domain_knowledge": {
                    "description": "Use domain expertise to set meaningful radius",
                    "example": "Geographic data: radius in kilometers",
                    "advantages": ["Interpretable", "Meaningful results"],
                    "when_to_use": "When you understand the problem domain"
                },
                "data_driven": {
                    "description": "Estimate radius from data characteristics",
                    "methods": ["Percentile of pairwise distances", "k-nearest neighbor distances", "Density estimation"],
                    "advantages": ["Automatic", "Data-adaptive"],
                    "when_to_use": "When domain knowledge is limited"
                },
                "cross_validation": {
                    "description": "Use CV to find optimal radius",
                    "approach": "Grid search over radius values",
                    "advantages": ["Performance-optimized", "Systematic"],
                    "when_to_use": "When prediction accuracy is primary goal"
                },
                "density_based": {
                    "description": "Set radius based on local density",
                    "approach": "Adaptive radius per region",
                    "advantages": ["Handles variable density", "Sophisticated"],
                    "when_to_use": "For complex, multi-density datasets"
                }
            },
            "outlier_handling_strategies": {
                "outlier_label": {
                    "description": "Assign specific label to outliers",
                    "use_case": "Anomaly detection",
                    "implementation": "Set outlier_label parameter"
                },
                "default_class": {
                    "description": "Assign most common class to outliers",
                    "use_case": "Conservative classification",
                    "implementation": "Use majority class as fallback"
                },
                "reject_option": {
                    "description": "Flag outliers for manual review",
                    "use_case": "Critical applications",
                    "implementation": "Return uncertainty indicator"
                },
                "nearest_assignment": {
                    "description": "Assign to class of nearest neighbor",
                    "use_case": "Smooth classification",
                    "implementation": "Fall back to 1-NN for outliers"
                }
            },
            "scaling_criticality": {
                "why_critical": "Radius is absolute distance threshold",
                "example": "Radius=1.0 meaningless if features have different scales",
                "effect_of_no_scaling": "Features with large ranges dominate distance",
                "recommendation": "ALWAYS scale features for radius neighbors",
                "methods": {
                    "standard_scaling": "For normal distributions",
                    "minmax_scaling": "For bounded features",
                    "robust_scaling": "For data with outliers"
                }
            },
            "density_analysis_benefits": {
                "understanding_data": "Reveals data distribution characteristics",
                "radius_selection": "Helps choose appropriate radius",
                "outlier_identification": "Identifies low-density regions",
                "performance_prediction": "Predicts algorithm behavior",
                "parameter_tuning": "Guides hyperparameter selection"
            },
            "comparison_with_knn": {
                "neighborhood_definition": {
                    "radius_neighbors": "Fixed distance threshold",
                    "knn": "Fixed number of neighbors",
                    "implication": "RN adapts to density, KNN uses fixed count"
                },
                "outlier_handling": {
                    "radius_neighbors": "Natural outlier detection",
                    "knn": "Always finds k neighbors",
                    "advantage": "RN better for anomaly detection"
                },
                "parameter_interpretation": {
                    "radius_neighbors": "Distance threshold (domain-meaningful)",
                    "knn": "Number of neighbors (algorithm-specific)",
                    "advantage": "Radius often more interpretable"
                },
                "computational_complexity": {
                    "radius_neighbors": "Variable (depends on density)",
                    "knn": "Fixed O(k) per query",
                    "trade_off": "RN can be faster in sparse regions"
                },
                "decision_boundaries": {
                    "radius_neighbors": "Density-adapted boundaries",
                    "knn": "Uniform neighborhood boundaries",
                    "advantage": "RN better for variable-density data"
                }
            },
            "practical_considerations": {
                "memory_usage": "O(n*d) - stores entire training set",
                "prediction_time": "O(n*d) worst case per prediction",
                "training_time": "O(1) - just stores data",
                "scalability": "Poor for large datasets",
                "parameter_sensitivity": "Very sensitive to radius choice",
                "feature_engineering": "Benefits from good distance metrics"
            }
        }
    
    def fit(self, X, y, 
            estimate_radius=None,
            store_training_data=True,
            analyze_density=None):
        """
        Fit the Radius Neighbors Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        estimate_radius : bool, optional
            Whether to estimate optimal radius from data
        store_training_data : bool, default=True
            Whether to store training data for analysis
        analyze_density : bool, optional
            Whether to perform density analysis
            
        Returns:
        --------
        self : object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=True)
        
        # Store training info
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Encode labels if they're not numeric
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        
        # Store original data for analysis
        if store_training_data:
            self.training_data_ = X.copy() if hasattr(X, 'copy') else np.array(X, copy=True)
            self.training_labels_ = y.copy() if hasattr(y, 'copy') else np.array(y, copy=True)
        
        # Feature scaling (CRITICAL for radius-based methods!)
        if self.auto_scaling:
            if self.scaling_method == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler_ = MinMaxScaler()
            elif self.scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler_ = RobustScaler()
            else:
                self.scaler_ = StandardScaler()
            
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
            warnings.warn("Feature scaling is CRITICAL for Radius Neighbors! Consider enabling auto_scaling.")
        
        # Estimate optimal radius if requested
        if estimate_radius is None:
            estimate_radius = self.auto_radius_estimation
        
        if estimate_radius:
            self.optimal_radius_ = self._estimate_optimal_radius(X_scaled, y_encoded)
            radius = self.optimal_radius_
        else:
            radius = self.radius
        
        # Perform density analysis
        if analyze_density is None:
            analyze_density = self.density_analysis
        
        if analyze_density:
            self._analyze_data_density(X_scaled, y_encoded)
        
        # Create Radius Neighbors model
        self.model_ = RadiusNeighborsClassifier(
            radius=radius,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            outlier_label=self.outlier_label,
            n_jobs=self.n_jobs
        )
        
        # Fit the model
        self.model_.fit(X_scaled, y_encoded)
        
        # Analyze radius effectiveness
        self._analyze_radius_effectiveness(X_scaled, y_encoded, radius)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get predictions
        y_pred_encoded = self.model_.predict(X_scaled)
        
        # Convert back to original labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get probability predictions
        probabilities = self.model_.predict_proba(X_scaled)
        
        return probabilities
    
    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        """
        Find neighbors within a given radius of point(s)
        
        Parameters:
        -----------
        X : array-like, optional
            Query points (if None, use training data)
        radius : float, optional
            Limiting distance of neighbors to return (if None, use fitted value)
        return_distance : bool, default=True
            Whether to return distances
            
        Returns:
        --------
        distances : array (if return_distance=True)
            Distances to neighbors
        indices : array
            Indices of neighbors
        """
        check_is_fitted(self, 'is_fitted_')
        
        if X is not None:
            X = check_array(X, accept_sparse=True)
            if self.scaler_ is not None:
                X = self.scaler_.transform(X)
        
        return self.model_.radius_neighbors(X, radius, return_distance)
    
    def _estimate_optimal_radius(self, X, y):
        """
        Estimate optimal radius using data-driven approaches
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
            
        Returns:
        --------
        optimal_radius : float
            Estimated optimal radius
        """
        n_samples = len(X)
        
        if self.radius_estimation_method == 'percentile':
            # Use percentile of pairwise distances
            if n_samples > 1000:
                # Sample for efficiency
                indices = np.random.choice(n_samples, 1000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Calculate pairwise distances
            distances = pairwise_distances(X_sample, metric=self.metric)
            # Get upper triangle (excluding diagonal)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            
            # Use 10th percentile as radius (captures local neighborhoods)
            optimal_radius = np.percentile(upper_tri, 10)
            
        elif self.radius_estimation_method == 'knn':
            # Use average distance to k-th nearest neighbor
            k = min(5, n_samples - 1)  # Use 5-NN or fewer if dataset is small
            
            nn = NearestNeighbors(
                n_neighbors=k + 1,  # +1 to exclude self
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs
            )
            nn.fit(X)
            
            distances, _ = nn.kneighbors(X)
            # Take mean distance to k-th neighbor (excluding self)
            optimal_radius = np.mean(distances[:, -1])
            
        elif self.radius_estimation_method == 'density':
            # Use density-based estimation
            # Calculate local density and use median density radius
            nn = NearestNeighbors(
                n_neighbors=min(10, n_samples - 1),
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs
            )
            nn.fit(X)
            
            distances, _ = nn.kneighbors(X)
            # Use median of 5th neighbor distances
            optimal_radius = np.median(distances[:, min(5, distances.shape[1] - 1)])
            
        else:
            # Fallback to default
            optimal_radius = self.radius
        
        return optimal_radius
    
    def _analyze_data_density(self, X, y):
        """
        Analyze the density characteristics of the training data
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
        """
        n_samples = len(X)
        
        # Sample for analysis if dataset is large
        if n_samples > 1000:
            indices = np.random.choice(n_samples, 1000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Calculate pairwise distances
        distances = pairwise_distances(X_sample, metric=self.metric)
        
        # Density metrics
        # For each point, calculate distance to nearest neighbor
        np.fill_diagonal(distances, np.inf)  # Exclude self-distances
        nearest_distances = np.min(distances, axis=1)
        
        # Calculate local density (inverse of average distance to k neighbors)
        k = min(5, len(X_sample) - 1)
        knn_distances = np.sort(distances, axis=1)[:, :k]
        local_density = 1.0 / (np.mean(knn_distances, axis=1) + 1e-10)
        
        # Overall density statistics
        self.density_analysis_ = {
            'nearest_neighbor_distances': nearest_distances,
            'local_densities': local_density,
            'distance_statistics': {
                'mean_nearest_distance': np.mean(nearest_distances),
                'std_nearest_distance': np.std(nearest_distances),
                'median_nearest_distance': np.median(nearest_distances),
                'min_nearest_distance': np.min(nearest_distances),
                'max_nearest_distance': np.max(nearest_distances)
            },
            'density_statistics': {
                'mean_density': np.mean(local_density),
                'std_density': np.std(local_density),
                'density_variation': np.std(local_density) / np.mean(local_density),
                'density_range': np.max(local_density) - np.min(local_density)
            },
            'radius_recommendations': {
                'conservative': np.percentile(nearest_distances, 25),
                'moderate': np.percentile(nearest_distances, 50),
                'liberal': np.percentile(nearest_distances, 75)
            }
        }
    
    def _analyze_radius_effectiveness(self, X, y, radius):
        """
        Analyze how effective the chosen radius is
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
        radius : float
            Radius being analyzed
        """
        n_samples = len(X)
        
        # Create NearestNeighbors for analysis
        nn = NearestNeighbors(
            radius=radius,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        nn.fit(X)
        
        # Find neighbors for each point
        neighbor_counts = []
        empty_neighborhoods = 0
        
        for i in range(min(1000, n_samples)):  # Sample for efficiency
            distances, indices = nn.radius_neighbors([X[i]], radius=radius)
            # Count neighbors (excluding self)
            n_neighbors = len(indices[0]) - 1 if len(indices[0]) > 0 else 0
            neighbor_counts.append(n_neighbors)
            
            if n_neighbors == 0:
                empty_neighborhoods += 1
        
        neighbor_counts = np.array(neighbor_counts)
        
        self.radius_analysis_ = {
            'radius_used': radius,
            'neighbor_count_stats': {
                'mean_neighbors': np.mean(neighbor_counts),
                'std_neighbors': np.std(neighbor_counts),
                'median_neighbors': np.median(neighbor_counts),
                'min_neighbors': np.min(neighbor_counts),
                'max_neighbors': np.max(neighbor_counts),
                'empty_neighborhoods_pct': (empty_neighborhoods / len(neighbor_counts)) * 100
            },
            'effectiveness_assessment': self._assess_radius_effectiveness(neighbor_counts, empty_neighborhoods)
        }
    
    def _assess_radius_effectiveness(self, neighbor_counts, empty_neighborhoods):
        """Assess the effectiveness of the chosen radius"""
        mean_neighbors = np.mean(neighbor_counts)
        empty_pct = (empty_neighborhoods / len(neighbor_counts)) * 100
        
        if empty_pct > 20:
            return "Poor - Too many empty neighborhoods. Consider increasing radius."
        elif empty_pct > 10:
            return "Moderate - Some empty neighborhoods. Monitor performance."
        elif mean_neighbors < 2:
            return "Sparse - Very few neighbors on average. Consider increasing radius."
        elif mean_neighbors > 50:
            return "Dense - Many neighbors. Consider decreasing radius for efficiency."
        else:
            return "Good - Reasonable neighborhood sizes."
    
    def get_radius_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of radius effectiveness and data density
        
        Returns:
        --------
        analysis_info : dict
            Information about radius effectiveness and density characteristics
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "algorithm_config": {
                "radius": self.model_.radius,
                "weights": self.weights,
                "algorithm": self.algorithm,
                "metric": self.metric,
                "outlier_label": self.outlier_label
            },
            "data_characteristics": {
                "n_training_samples": len(self.training_data_) if self.training_data_ is not None else "Not stored",
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "feature_scaling_applied": self.scaler_ is not None,
                "scaling_method": self.scaling_method if self.scaler_ is not None else None
            }
        }
        
        # Add optimal radius information
        if self.optimal_radius_ is not None:
            analysis["radius_estimation"] = {
                "optimal_radius": self.optimal_radius_,
                "default_radius": self.radius,
                "estimation_method": self.radius_estimation_method,
                "radius_changed": abs(self.optimal_radius_ - self.radius) > 1e-6
            }
        
        # Add density analysis
        if self.density_analysis_:
            density_stats = self.density_analysis_['density_statistics']
            distance_stats = self.density_analysis_['distance_statistics']
            
            analysis["density_analysis"] = {
                "density_variation": density_stats['density_variation'],
                "density_interpretation": self._interpret_density_variation(density_stats['density_variation']),
                "mean_nearest_distance": distance_stats['mean_nearest_distance'],
                "distance_uniformity": 1.0 / (1.0 + distance_stats['std_nearest_distance']),
                "radius_recommendations": self.density_analysis_['radius_recommendations']
            }
        
        # Add radius effectiveness analysis
        if self.radius_analysis_:
            neighbor_stats = self.radius_analysis_['neighbor_count_stats']
            
            analysis["radius_effectiveness"] = {
                "mean_neighbors": neighbor_stats['mean_neighbors'],
                "empty_neighborhoods_pct": neighbor_stats['empty_neighborhoods_pct'],
                "effectiveness_assessment": self.radius_analysis_['effectiveness_assessment'],
                "neighborhood_consistency": self._calculate_neighborhood_consistency(neighbor_stats)
            }
        
        # Add outlier analysis
        if self.outlier_detection:
            analysis["outlier_potential"] = {
                "outlier_detection_enabled": True,
                "outlier_label": self.outlier_label,
                "expected_outlier_rate": self._estimate_outlier_rate()
            }
        
        return analysis
    
    def _interpret_density_variation(self, variation):
        """Interpret density variation score"""
        if variation < 0.3:
            return "Low variation - Uniform density throughout data"
        elif variation < 0.7:
            return "Moderate variation - Some density differences"
        elif variation < 1.2:
            return "High variation - Significant density differences"
        else:
            return "Very high variation - Highly non-uniform density"
    
    def _calculate_neighborhood_consistency(self, neighbor_stats):
        """Calculate how consistent neighborhood sizes are"""
        if neighbor_stats['mean_neighbors'] == 0:
            return 0.0
        
        cv = neighbor_stats['std_neighbors'] / neighbor_stats['mean_neighbors']
        consistency = 1.0 / (1.0 + cv)
        return consistency
    
    def _estimate_outlier_rate(self):
        """Estimate expected outlier rate based on radius analysis"""
        if self.radius_analysis_:
            return self.radius_analysis_['neighbor_count_stats']['empty_neighborhoods_pct']
        return "Unknown - radius analysis not performed"
    
    def plot_radius_analysis(self, figsize=(15, 12)):
        """
        Create comprehensive radius analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 12)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Radius analysis visualization
        """
        if not (self.density_analysis_ and self.radius_analysis_):
            return None
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Nearest Neighbor Distance Distribution
        nearest_distances = self.density_analysis_['nearest_neighbor_distances']
        ax1.hist(nearest_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(nearest_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(nearest_distances):.3f}')
        ax1.axvline(self.model_.radius, color='green', linestyle='-', linewidth=2,
                   label=f'Radius: {self.model_.radius:.3f}')
        ax1.set_xlabel('Distance to Nearest Neighbor')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Nearest Neighbor Distances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Local Density Distribution
        local_densities = self.density_analysis_['local_densities']
        ax2.hist(local_densities, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(np.mean(local_densities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(local_densities):.3f}')
        ax2.set_xlabel('Local Density')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Local Data Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Neighbor Count Distribution
        neighbor_stats = self.radius_analysis_['neighbor_count_stats']
        if 'neighbor_counts' in self.radius_analysis_:
            neighbor_counts = self.radius_analysis_['neighbor_counts']
        else:
            # Generate sample neighbor counts for visualization
            mean_neighbors = neighbor_stats['mean_neighbors']
            std_neighbors = neighbor_stats['std_neighbors']
            neighbor_counts = np.random.normal(mean_neighbors, std_neighbors, 1000).astype(int)
            neighbor_counts = np.clip(neighbor_counts, 0, None)
        
        ax3.hist(neighbor_counts, bins=range(int(np.max(neighbor_counts)) + 2), 
                alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(neighbor_stats['mean_neighbors'], color='red', linestyle='--',
                   label=f'Mean: {neighbor_stats["mean_neighbors"]:.1f}')
        ax3.set_xlabel('Number of Neighbors in Radius')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Neighbor Count Distribution (Radius={self.model_.radius:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Radius Effectiveness Metrics
        effectiveness_metrics = {
            'Mean Neighbors': neighbor_stats['mean_neighbors'],
            'Empty Neighborhoods %': neighbor_stats['empty_neighborhoods_pct'],
            'Neighborhood Consistency': self._calculate_neighborhood_consistency(neighbor_stats),
            'Density Variation': self.density_analysis_['density_statistics']['density_variation']
        }
        
        metric_names = list(effectiveness_metrics.keys())
        metric_values = list(effectiveness_metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, color=['cyan', 'pink', 'lightcoral', 'gold'])
        ax4.set_ylabel('Value')
        ax4.set_title('Radius Effectiveness Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(metric_values) * 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Radius Recommendations
        recommendations = self.density_analysis_['radius_recommendations']
        rec_names = list(recommendations.keys())
        rec_values = list(recommendations.values())
        
        bars = ax5.bar(rec_names, rec_values, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax5.axhline(y=self.model_.radius, color='red', linestyle='-', linewidth=2,
                   label=f'Current Radius: {self.model_.radius:.3f}')
        ax5.set_ylabel('Recommended Radius')
        ax5.set_title('Radius Recommendations Based on Data')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, rec_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(rec_values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Distance vs Density Scatter Plot
        # Sample points for visualization
        if len(nearest_distances) > 1000:
            indices = np.random.choice(len(nearest_distances), 1000, replace=False)
            sample_distances = nearest_distances[indices]
            sample_densities = local_densities[indices]
        else:
            sample_distances = nearest_distances
            sample_densities = local_densities
        
        scatter = ax6.scatter(sample_distances, sample_densities, alpha=0.6, color='purple')
        ax6.set_xlabel('Distance to Nearest Neighbor')
        ax6.set_ylabel('Local Density')
        ax6.set_title('Distance vs Density Relationship')
        ax6.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(sample_distances, sample_densities)[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_radius_sensitivity(self, X, y, radius_range=None, figsize=(12, 8)):
        """
        Plot sensitivity to radius parameter
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Target data
        radius_range : array-like, optional
            Range of radius values to test
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Radius sensitivity plot
        """
        if not self.is_fitted_:
            return None
        
        # Apply scaling if used
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Encode targets
        y_encoded = self.label_encoder_.transform(y)
        
        # Define radius range
        if radius_range is None:
            current_radius = self.model_.radius
            radius_range = np.logspace(
                np.log10(current_radius * 0.1), 
                np.log10(current_radius * 5), 
                20
            )
        
        # Test different radius values
        mean_neighbors = []
        empty_percentages = []
        
        nn = NearestNeighbors(
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        nn.fit(X_scaled)
        
        for radius in radius_range:
            neighbor_counts = []
            empty_count = 0
            
            # Sample points for efficiency
            n_samples = min(500, len(X_scaled))
            indices = np.random.choice(len(X_scaled), n_samples, replace=False)
            
            for i in indices:
                distances, neighbor_indices = nn.radius_neighbors([X_scaled[i]], radius=radius)
                n_neighbors = len(neighbor_indices[0]) - 1  # Exclude self
                neighbor_counts.append(n_neighbors)
                if n_neighbors == 0:
                    empty_count += 1
            
            mean_neighbors.append(np.mean(neighbor_counts))
            empty_percentages.append((empty_count / len(neighbor_counts)) * 100)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Mean neighbors vs radius
        ax1.semilogx(radius_range, mean_neighbors, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axvline(self.model_.radius, color='red', linestyle='--', 
                   label=f'Current Radius: {self.model_.radius:.3f}')
        ax1.set_xlabel('Radius')
        ax1.set_ylabel('Mean Number of Neighbors')
        ax1.set_title('Mean Neighbors vs Radius')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Empty neighborhoods vs radius
        ax2.semilogx(radius_range, empty_percentages, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.axvline(self.model_.radius, color='red', linestyle='--',
                   label=f'Current Radius: {self.model_.radius:.3f}')
        ax2.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='10% threshold')
        ax2.set_xlabel('Radius')
        ax2.set_ylabel('Empty Neighborhoods (%)')
        ax2.set_title('Empty Neighborhoods vs Radius')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 📍 Radius Neighbors Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Distance", "Outliers", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Radius Parameters**")
            
            # Radius
            radius = st.number_input(
                "Radius:",
                value=float(self.radius),
                min_value=0.01,
                max_value=10.0,
                step=0.1,
                help="Range of parameter space to use for nearest neighbors queries",
                key=f"{key_prefix}_radius"
            )
            
            # Auto radius estimation
            auto_radius_estimation = st.checkbox(
                "Auto Radius Estimation",
                value=self.auto_radius_estimation,
                help="Automatically estimate optimal radius from data characteristics",
                key=f"{key_prefix}_auto_radius_estimation"
            )
            
            if auto_radius_estimation:
                radius_estimation_method = st.selectbox(
                    "Estimation Method:",
                    options=['percentile', 'knn', 'density'],
                    index=['percentile', 'knn', 'density'].index(self.radius_estimation_method),
                    help="Method for automatic radius estimation",
                    key=f"{key_prefix}_radius_estimation_method"
                )
            else:
                radius_estimation_method = self.radius_estimation_method
            
            # Weights
            weights = st.selectbox(
                "Neighbor Weighting:",
                options=['uniform', 'distance'],
                index=['uniform', 'distance'].index(self.weights),
                help="uniform: all neighbors equal weight, distance: closer neighbors have more influence",
                key=f"{key_prefix}_weights"
            )
            
            # Leaf size
            leaf_size = st.slider(
                "Leaf Size:",
                min_value=10,
                max_value=100,
                value=int(self.leaf_size),
                step=5,
                help="Affects speed of construction and query, as well as memory usage",
                key=f"{key_prefix}_leaf_size"
            )
        
        with tab2:
            st.markdown("**Distance Metric Configuration**")
            
            # Distance metric
            metric = st.selectbox(
                "Distance Metric:",
                options=['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine'],
                index=['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine'].index(self.metric),
                help="Distance function to use for nearest neighbor search",
                key=f"{key_prefix}_metric"
            )
            
            # Minkowski parameter
            if metric == 'minkowski':
                p = st.slider(
                    "Minkowski Parameter (p):",
                    min_value=1,
                    max_value=5,
                    value=int(self.p),
                    help="p=1: Manhattan distance, p=2: Euclidean distance",
                    key=f"{key_prefix}_p"
                )
            else:
                p = 2
            
            # Algorithm
            algorithm = st.selectbox(
                "Algorithm:",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(self.algorithm),
                help="Algorithm used to compute nearest neighbors",
                key=f"{key_prefix}_algorithm"
            )
            
            # Feature scaling
            auto_scaling = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scaling,
                help="Automatically scale features (CRITICAL for radius-based methods!)",
                key=f"{key_prefix}_auto_scaling"
            )
            
            if auto_scaling:
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    options=['standard', 'minmax', 'robust'],
                    index=['standard', 'minmax', 'robust'].index(self.scaling_method),
                    help="Method for feature scaling",
                    key=f"{key_prefix}_scaling_method"
                )
            else:
                scaling_method = self.scaling_method
                st.error("⚠️ Feature scaling is CRITICAL for Radius Neighbors!")
        
        with tab3:
            st.markdown("**Outlier Handling**")
            
            # Outlier detection
            outlier_detection = st.checkbox(
                "Enable Outlier Detection",
                value=self.outlier_detection,
                help="Detect and handle points with no neighbors",
                key=f"{key_prefix}_outlier_detection"
            )
            
            # Outlier label
            outlier_label_option = st.selectbox(
                "Outlier Handling:",
                options=['None', 'Most Frequent', 'Custom Label'],
                index=0 if self.outlier_label is None else 1,
                help="How to handle points with no neighbors in radius",
                key=f"{key_prefix}_outlier_label_option"
            )
            
            if outlier_label_option == 'Custom Label':
                outlier_label = st.number_input(
                    "Custom Outlier Label:",
                    value=0,
                    help="Custom label for outlier points",
                    key=f"{key_prefix}_custom_outlier_label"
                )
            elif outlier_label_option == 'Most Frequent':
                outlier_label = 'most_frequent'
            else:
                outlier_label = None
            
            # Minimum neighbors threshold
            min_neighbors_threshold = st.slider(
                "Min Neighbors Threshold:",
                min_value=1,
                max_value=10,
                value=int(self.min_neighbors_threshold),
                help="Minimum neighbors required for confident classification",
                key=f"{key_prefix}_min_neighbors_threshold"
            )
        
        with tab4:
            st.markdown("**Advanced Settings**")
            
            # Number of jobs
            n_jobs_option = st.selectbox(
                "Parallel Processing:",
                options=['Auto', '1', '2', '4', '8'],
                index=0,
                help="Number of parallel jobs (-1 for all processors)",
                key=f"{key_prefix}_n_jobs_option"
            )
            
            if n_jobs_option == 'Auto':
                n_jobs = -1
            else:
                n_jobs = int(n_jobs_option)
            
            # Adaptive radius
            adaptive_radius = st.checkbox(
                "Adaptive Radius",
                value=self.adaptive_radius,
                help="Use adaptive radius based on local density (experimental)",
                key=f"{key_prefix}_adaptive_radius"
            )
            
            # Density analysis
            density_analysis = st.checkbox(
                "Density Analysis",
                value=self.density_analysis,
                help="Perform comprehensive density analysis of data",
                key=f"{key_prefix}_density_analysis"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **Radius Neighbors** - Distance-Based Classification:
            • 📍 Fixed radius instead of fixed k neighbors
            • 🎯 Density-aware classification
            • 🔍 Natural outlier detection
            • 📏 Interpretable distance threshold
            • 🌊 Adapts to variable density regions
            • 💫 Handles empty neighborhoods gracefully
            
            **Key Advantages:**
            • More intuitive than k-NN for some domains
            • Natural density-based classification
            • Built-in anomaly detection
            • Handles non-uniform data distributions
            """)
            
            # Critical scaling warning
            st.error("""
            **⚠️ CRITICAL: Feature Scaling Required!**
            
            Radius Neighbors is extremely sensitive to feature scales.
            A radius of 1.0 means different things for:
            • Age (0-100) vs Income (0-100,000)
            • ALWAYS enable feature scaling!
            """)
            
            # Radius setting guide
            if st.button("📍 How to Choose Radius?", key=f"{key_prefix}_radius_guide"):
                st.markdown("""
                **Radius Selection Strategies:**
                
                **Domain Knowledge Approach:**
                - Use meaningful distance thresholds
                - Example: Geographic data - radius in km
                - Medical data - clinical significance thresholds
                
                **Data-Driven Approach:**
                - Use percentiles of pairwise distances
                - 10th percentile: tight neighborhoods
                - 25th percentile: moderate neighborhoods
                - 50th percentile: loose neighborhoods
                
                **k-NN Based Approach:**
                - Average distance to k-th nearest neighbor
                - Start with k=5, use mean distance as radius
                
                **Cross-Validation Approach:**
                - Grid search over radius values
                - Optimize for your performance metric
                """)
            
            # Outlier handling guide
            if st.button("🎯 Outlier Handling Guide", key=f"{key_prefix}_outlier_guide"):
                st.markdown("""
                **Outlier Handling Strategies:**
                
                **None (Default):**
                - Points with no neighbors raise error
                - Use when outliers should not occur
                
                **Most Frequent:**
                - Assign most common class to outliers
                - Conservative approach
                
                **Custom Label:**
                - Assign specific label (e.g., -1 for anomaly)
                - Use for anomaly detection
                
                **Recommendation:**
                - Enable outlier detection for real-world data
                - Monitor outlier percentage in results
                """)
            
            # Comparison with k-NN
            if st.button("🆚 Radius vs k-NN", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Radius Neighbors vs k-NN:**
                
                **Radius Neighbors:**
                ✅ Adapts to variable density
                ✅ Natural outlier detection
                ✅ Interpretable radius parameter
                ✅ Density-based classification
                ❌ Harder to set radius parameter
                ❌ Variable neighborhood sizes
                
                **k-Nearest Neighbors:**
                ✅ Consistent neighborhood sizes
                ✅ Easier parameter tuning
                ✅ Always finds neighbors
                ❌ Fixed k may not suit all regions
                ❌ No natural outlier detection
                ❌ Poor for variable density data
                
                **Use Radius Neighbors when:**
                - Data has variable density regions
                - Outlier detection is important
                - You have domain knowledge about distances
                - Density-based decisions make sense
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "radius": radius,
            "weights": weights,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "p": p,
            "metric": metric,
            "outlier_label": outlier_label,
            "n_jobs": n_jobs,
            "auto_scaling": auto_scaling,
            "scaling_method": scaling_method,
            "auto_radius_estimation": auto_radius_estimation,
            "radius_estimation_method": radius_estimation_method,
            "outlier_detection": outlier_detection,
            "min_neighbors_threshold": min_neighbors_threshold,
            "adaptive_radius": adaptive_radius,
            "density_analysis": density_analysis,
            "_ui_options": {
                "show_radius_analysis": density_analysis,
                "show_radius_sensitivity": True,
                "show_outlier_analysis": outlier_detection
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RadiusNeighborsClassifierPlugin(
            radius=hyperparameters.get("radius", self.radius),
            weights=hyperparameters.get("weights", self.weights),
            algorithm=hyperparameters.get("algorithm", self.algorithm),
            leaf_size=hyperparameters.get("leaf_size", self.leaf_size),
            p=hyperparameters.get("p", self.p),
            metric=hyperparameters.get("metric", self.metric),
            outlier_label=hyperparameters.get("outlier_label", self.outlier_label),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            auto_scaling=hyperparameters.get("auto_scaling", self.auto_scaling),
            scaling_method=hyperparameters.get("scaling_method", self.scaling_method),
            auto_radius_estimation=hyperparameters.get("auto_radius_estimation", self.auto_radius_estimation),
            radius_estimation_method=hyperparameters.get("radius_estimation_method", self.radius_estimation_method),
            outlier_detection=hyperparameters.get("outlier_detection", self.outlier_detection),
            min_neighbors_threshold=hyperparameters.get("min_neighbors_threshold", self.min_neighbors_threshold),
            adaptive_radius=hyperparameters.get("adaptive_radius", self.adaptive_radius),
            density_analysis=hyperparameters.get("density_analysis", self.density_analysis)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Radius Neighbors
        
        Radius Neighbors requires careful preprocessing, especially feature scaling.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Radius Neighbors doesn't handle missing values well
        if np.any(pd.isna(X_processed)):
            warnings.warn("Radius Neighbors doesn't handle missing values. Consider imputation before training.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Radius Neighbors is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        if not SKLEARN_AVAILABLE:
            return False, "scikit-learn is not installed. Install with: pip install scikit-learn"
        
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Radius Neighbors requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "Radius Neighbors doesn't handle missing values well. Please impute missing values first."
        
        # Check dimensionality
        if X.shape[1] > 50:
            return True, "Warning: High dimensionality detected. Radius interpretation becomes difficult. Consider dimensionality reduction."
        elif X.shape[1] > 20:
            return True, "Moderate dimensionality. Monitor radius effectiveness carefully."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
        
        return True, "Radius Neighbors is compatible with this data. Ensure proper feature scaling and radius selection!"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "radius": self.model_.radius,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "outlier_label": self.outlier_label,
            "optimal_radius": self.optimal_radius_,
            "scaling_applied": self.scaler_ is not None,
# Continue from line 1531 where the code breaks:
            "scaling_method": self.scaling_method if self.scaler_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Radius Neighbors",
            "training_completed": True,
            "radius_characteristics": {
                "radius_based": True,
                "density_sensitive": True,
                "outlier_robust": True,
                "distance_dependent": True,
                "lazy_learning": True,
                "non_parametric": True
            },
            "model_configuration": {
                "radius": self.model_.radius,
                "weights": self.weights,
                "algorithm": self.algorithm,
                "metric": self.metric,
                "outlier_handling": self.outlier_label is not None,
                "feature_scaling": self.scaler_ is not None
            },
            "radius_analysis": self.get_radius_analysis(),
            "performance_considerations": {
                "memory_usage": "Stores entire training dataset",
                "prediction_time": "Variable - depends on local density",
                "training_time": "O(1) - just stores data",
                "scalability": "Poor for large datasets",
                "radius_sensitivity": "Very sensitive to radius choice",
                "scaling_dependency": "Critical - requires feature scaling"
            },
            "density_insights": {
                "variable_neighborhoods": "Neighborhood size adapts to local density",
                "outlier_detection": "Natural identification of isolated points",
                "density_boundaries": "Decision boundaries follow density patterns",
                "empty_neighborhoods": "Handles sparse regions gracefully"
            }
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Radius Neighbors Classifier model.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics but kept for API consistency.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics but kept for API consistency.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not directly used for these specific metrics but kept for API consistency.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve Radius Neighbors specific metrics."}

        metrics = {}
        prefix = "radiusneigh_"

        # Radius related metrics
        if hasattr(self.model_, 'radius'):
            metrics[f"{prefix}radius_used"] = self.model_.radius
        if self.optimal_radius_ is not None:
            metrics[f"{prefix}optimal_radius_estimated"] = self.optimal_radius_
            if hasattr(self.model_, 'radius'):
                 metrics[f"{prefix}radius_estimation_diff"] = abs(self.model_.radius - self.optimal_radius_)


        # Metrics from radius_analysis
        if self.radius_analysis_ and 'neighbor_count_stats' in self.radius_analysis_:
            neighbor_stats = self.radius_analysis_['neighbor_count_stats']
            metrics[f"{prefix}mean_neighbors_in_radius"] = neighbor_stats.get('mean_neighbors')
            metrics[f"{prefix}median_neighbors_in_radius"] = neighbor_stats.get('median_neighbors')
            metrics[f"{prefix}empty_neighborhoods_pct"] = neighbor_stats.get('empty_neighborhoods_pct')
            
            # Calculate neighborhood consistency if possible
            if neighbor_stats.get('mean_neighbors') is not None and neighbor_stats.get('std_neighbors') is not None:
                if neighbor_stats['mean_neighbors'] > 1e-9: # Avoid division by zero
                    cv_neighbors = neighbor_stats['std_neighbors'] / neighbor_stats['mean_neighbors']
                    metrics[f"{prefix}neighborhood_size_consistency"] = 1.0 / (1.0 + cv_neighbors)
                else:
                    metrics[f"{prefix}neighborhood_size_consistency"] = 0.0


        # Metrics from density_analysis
        if self.density_analysis_:
            if 'density_statistics' in self.density_analysis_:
                density_stats = self.density_analysis_['density_statistics']
                metrics[f"{prefix}data_density_variation_coeff"] = density_stats.get('density_variation')
            if 'distance_statistics' in self.density_analysis_:
                distance_stats = self.density_analysis_['distance_statistics']
                metrics[f"{prefix}mean_nearest_neighbor_dist"] = distance_stats.get('mean_nearest_distance')
                metrics[f"{prefix}median_nearest_neighbor_dist"] = distance_stats.get('median_nearest_distance')

        # Outlier related metrics
        if self.outlier_detection:
            metrics[f"{prefix}outlier_detection_enabled"] = True
            metrics[f"{prefix}outlier_label_configured"] = self.outlier_label
            if self.radius_analysis_ and 'neighbor_count_stats' in self.radius_analysis_:
                 metrics[f"{prefix}estimated_outlier_rate_pct"] = self.radius_analysis_['neighbor_count_stats'].get('empty_neighborhoods_pct')
        else:
            metrics[f"{prefix}outlier_detection_enabled"] = False
            
        # Number of samples that might be classified as outliers by the model
        # This requires y_pred if the outlier_label is one of the actual class labels.
        # If outlier_label is a special value (e.g., -1 and not in y_true), we can count directly from y_pred.
        if y_pred is not None and self.outlier_label is not None:
            # Check if outlier_label is a special label not present in original classes
            # This logic assumes outlier_label is distinct. If it can be an actual class, this count is ambiguous.
            # For a more robust count, one might need to know if self.outlier_label was used by predict.
            # The sklearn RadiusNeighborsClassifier itself can assign outlier_label if set.
            try:
                # Attempt to encode the outlier label to see if it's a known class
                encoded_outlier_label = self.label_encoder_.transform([self.outlier_label])[0]
                # If it's a known class, counting y_pred == self.outlier_label might be ambiguous
                # For now, we'll count occurrences of the raw outlier_label in y_pred
                num_outliers_predicted = np.sum(y_pred == self.outlier_label)
                metrics[f"{prefix}outliers_predicted_count"] = int(num_outliers_predicted)
                metrics[f"{prefix}outliers_predicted_pct"] = (num_outliers_predicted / len(y_pred)) * 100 if len(y_pred) > 0 else 0
            except ValueError: 
                # If outlier_label is not in known classes, it's a distinct outlier marker
                num_outliers_predicted = np.sum(y_pred == self.outlier_label)
                metrics[f"{prefix}outliers_predicted_count"] = int(num_outliers_predicted)
                metrics[f"{prefix}outliers_predicted_pct"] = (num_outliers_predicted / len(y_pred)) * 100 if len(y_pred) > 0 else 0
            except Exception:
                metrics[f"{prefix}outliers_predicted_info"] = "Could not determine predicted outlier count."


        if not metrics:
            metrics['info'] = "No specific Radius Neighbors metrics were available or calculated."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return RadiusNeighborsClassifierPlugin()