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

# Try to import KNN with graceful fallback
try:
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KNeighborsClassifier = None
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

class KNNClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    K-Nearest Neighbors Classifier Plugin - Simple Non-Parametric Learning
    
    KNN is a lazy learning algorithm that classifies instances based on the majority
    vote of their k nearest neighbors in the feature space. It makes no assumptions
    about the underlying data distribution and can capture complex decision boundaries.
    """
    
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 # Advanced parameters
                 auto_scaling=True,
                 scaling_method='standard',
                 distance_threshold=None,
                 outlier_detection=False,
                 cross_validation_k=True,
                 cv_folds=5):
        """
        Initialize KNN Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use for classification
        weights : str or callable, default='uniform'
            Weight function used in prediction ('uniform', 'distance', or callable)
        algorithm : str, default='auto'
            Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        leaf_size : int, default=30
            Leaf size passed to BallTree or cKDTree
        p : int, default=2
            Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
        metric : str or callable, default='minkowski'
            Distance metric to use ('minkowski', 'euclidean', 'manhattan', 'chebyshev', etc.)
        metric_params : dict, optional
            Additional keyword arguments for the metric function
        n_jobs : int, optional
            Number of parallel jobs to run for neighbors search
        auto_scaling : bool, default=True
            Whether to automatically scale features
        scaling_method : str, default='standard'
            Scaling method ('standard', 'minmax', 'robust')
        distance_threshold : float, optional
            Maximum distance for considering neighbors
        outlier_detection : bool, default=False
            Enable outlier detection based on average neighbor distance
        cross_validation_k : bool, default=True
            Use cross-validation to suggest optimal k
        cv_folds : int, default=5
            Number of folds for cross-validation
        """
        super().__init__()
        
        # Core KNN parameters
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
        # Advanced parameters
        self.auto_scaling = auto_scaling
        self.scaling_method = scaling_method
        self.distance_threshold = distance_threshold
        self.outlier_detection = outlier_detection
        self.cross_validation_k = cross_validation_k
        self.cv_folds = cv_folds
        
        # Plugin metadata
        self._name = "K-Nearest Neighbors"
        self._description = "Simple non-parametric algorithm that classifies based on majority vote of k nearest neighbors."
        self._category = "Instance-Based"
        self._algorithm_type = "Non-Parametric Classifier"
        self._paper_reference = "Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10
        self._handles_missing_values = False
        self._requires_scaling = True  # HIGHLY RECOMMENDED!
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = False
        self._provides_probabilities = True
        self._handles_categorical = False
        self._ensemble_method = False
        self._lazy_learning = True
        self._non_parametric = True
        self._distance_based = True
        self._interpretable = True
        self._memory_intensive = True
        self._scalable = False  # Poor scaling with large datasets
        
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
        self.optimal_k_ = None
        self.cv_scores_ = None
        self.neighbor_analysis_ = None
        
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
            "year_introduced": 1951,
            "key_characteristics": {
                "non_parametric": "Makes no assumptions about data distribution",
                "lazy_learning": "No explicit training phase - stores all data",
                "instance_based": "Classification based on similarity to stored instances",
                "distance_dependent": "Relies on distance metrics in feature space",
                "local_learning": "Decisions based on local neighborhood only",
                "memory_based": "Requires storing entire training dataset"
            },
            "algorithm_mechanics": {
                "training_phase": "Simply store all training instances",
                "prediction_process": [
                    "Calculate distances to all training instances",
                    "Find k nearest neighbors",
                    "Apply weighting scheme (uniform or distance-based)",
                    "Take majority vote for classification",
                    "Return class probabilities based on neighbor distribution"
                ],
                "distance_metrics": [
                    "Euclidean (L2): sqrt(Σ(x_i - y_i)²)",
                    "Manhattan (L1): Σ|x_i - y_i|",
                    "Minkowski (Lp): (Σ|x_i - y_i|^p)^(1/p)",
                    "Chebyshev: max|x_i - y_i|",
                    "Cosine: 1 - (x·y)/(||x|| ||y||)"
                ],
                "weighting_schemes": {
                    "uniform": "All neighbors weighted equally",
                    "distance": "Closer neighbors weighted more heavily",
                    "custom": "User-defined weighting function"
                }
            },
            "strengths": [
                "Simple to understand and implement",
                "No assumptions about data distribution",
                "Naturally handles multi-class problems",
                "Can capture complex decision boundaries",
                "Works well with small datasets",
                "Adapts to local patterns in data",
                "No parameters to tune during training",
                "Can provide confidence estimates",
                "Effective for recommendation systems",
                "Good baseline algorithm",
                "Handles non-linear relationships naturally",
                "Robust to noisy training data (with appropriate k)"
            ],
            "weaknesses": [
                "Computationally expensive for large datasets",
                "Sensitive to irrelevant features (curse of dimensionality)",
                "Requires feature scaling for meaningful distances",
                "Memory intensive (stores entire training set)",
                "Slow prediction time (O(n) for each prediction)",
                "Sensitive to local structure of data",
                "Performance degrades in high dimensions",
                "Sensitive to unbalanced datasets",
                "No interpretable model parameters",
                "Vulnerable to noise in training data"
            ],
            "ideal_use_cases": [
                "Small to medium-sized datasets",
                "Problems with complex decision boundaries",
                "Recommendation systems",
                "Pattern recognition",
                "Image classification (with appropriate features)",
                "Text classification (with good representations)",
                "Anomaly detection",
                "Multi-class classification problems",
                "Datasets with local patterns",
                "Prototype-based learning",
                "Missing value imputation",
                "Baseline model comparison"
            ],
            "distance_metrics_guide": {
                "euclidean": {
                    "use_case": "Continuous features, natural geometry",
                    "formula": "sqrt(Σ(x_i - y_i)²)",
                    "characteristics": "Sensitive to scale, good for spatial data"
                },
                "manhattan": {
                    "use_case": "Grid-like movements, robust to outliers",
                    "formula": "Σ|x_i - y_i|",
                    "characteristics": "Less sensitive to outliers than Euclidean"
                },
                "cosine": {
                    "use_case": "Text data, high-dimensional sparse data",
                    "formula": "1 - cosine_similarity(x, y)",
                    "characteristics": "Angle-based, ignores magnitude"
                },
                "hamming": {
                    "use_case": "Categorical or binary features",
                    "formula": "Proportion of differing components",
                    "characteristics": "Good for discrete features"
                }
            },
            "hyperparameter_guide": {
                "n_neighbors": {
                    "range": "3-20 (odd numbers for binary classification)",
                    "effect": "Low k: more flexible, high variance; High k: smoother, high bias",
                    "tuning": "Use cross-validation to find optimal value"
                },
                "weights": {
                    "uniform": "All neighbors contribute equally",
                    "distance": "Closer neighbors have more influence",
                    "recommendation": "Use 'distance' for continuous problems"
                },
                "algorithm": {
                    "auto": "Automatically chooses best algorithm",
                    "ball_tree": "Good for high dimensions",
                    "kd_tree": "Good for low dimensions",
                    "brute": "For small datasets or custom metrics"
                },
                "metric": {
                    "recommendation": "Euclidean for continuous, Manhattan for mixed types",
                    "scaling_dependency": "Most metrics require feature scaling"
                }
            },
            "scaling_importance": {
                "why_critical": "Distance metrics treat all features equally",
                "example": "Age (0-100) vs Income (0-100000) - income dominates",
                "methods": {
                    "standard_scaler": "Mean=0, Std=1, preserves outliers",
                    "minmax_scaler": "Range [0,1], affected by outliers",
                    "robust_scaler": "Uses median and IQR, robust to outliers"
                },
                "recommendation": "Always scale features unless using distance metrics designed for mixed types"
            },
            "curse_of_dimensionality": {
                "problem": "In high dimensions, all points become equidistant",
                "effects": [
                    "Distance measures become meaningless",
                    "Nearest neighbors are not actually 'near'",
                    "Performance degrades significantly"
                ],
                "solutions": [
                    "Dimensionality reduction (PCA, t-SNE)",
                    "Feature selection",
                    "Use appropriate distance metrics",
                    "Consider other algorithms for high-d data"
                ],
                "threshold": "Performance typically degrades beyond 10-20 features"
            },
            "practical_considerations": {
                "memory_usage": "O(n*d) where n=samples, d=features",
                "prediction_time": "O(n*d) for each prediction",
                "training_time": "O(1) - just stores data",
                "scalability": "Poor for large datasets (>100k samples)",
                "parallel_processing": "Prediction can be parallelized",
                "online_learning": "Not suitable - requires full dataset"
            },
            "comparison_with_other_methods": {
                "vs_naive_bayes": {
                    "assumptions": "KNN: none, NB: feature independence",
                    "decision_boundary": "KNN: complex, NB: linear/quadratic",
                    "training_speed": "KNN: instant, NB: fast",
                    "prediction_speed": "KNN: slow, NB: fast"
                },
                "vs_decision_trees": {
                    "interpretability": "KNN: local similarity, DT: explicit rules",
                    "overfitting": "KNN: controlled by k, DT: controlled by pruning",
                    "feature_selection": "KNN: sensitive to irrelevant features, DT: automatic selection",
                    "missing_values": "KNN: requires imputation, DT: handles natively"
                },
                "vs_svm": {
                    "complexity": "KNN: simple concept, SVM: mathematical sophistication",
                    "kernel_trick": "Both can handle non-linear boundaries",
                    "sparsity": "KNN: stores all data, SVM: only support vectors",
                    "high_dimensions": "KNN: poor performance, SVM: designed for high-d"
                }
            }
        }
    
    def fit(self, X, y, 
            find_optimal_k=None,
            store_training_data=True):
        """
        Fit the KNN Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        find_optimal_k : bool, optional
            Whether to find optimal k using cross-validation
        store_training_data : bool, default=True
            Whether to store training data for analysis
            
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
        
        # Feature scaling
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
        
        # Find optimal k using cross-validation
        if find_optimal_k is None:
            find_optimal_k = self.cross_validation_k
        
        if find_optimal_k:
            self.optimal_k_, self.cv_scores_ = self._find_optimal_k(X_scaled, y_encoded)
            n_neighbors = self.optimal_k_
        else:
            n_neighbors = self.n_neighbors
        
        # Create and fit KNN model
        self.model_ = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        
        self.model_.fit(X_scaled, y_encoded)
        
        # Analyze neighbor structure
        self._analyze_neighbor_structure(X_scaled, y_encoded)
        
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
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find the k-neighbors of a point
        
        Parameters:
        -----------
        X : array-like, optional
            Query points (if None, use training data)
        n_neighbors : int, optional
            Number of neighbors (if None, use fitted value)
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
        
        return self.model_.kneighbors(X, n_neighbors, return_distance)
    
    def _find_optimal_k(self, X, y):
        """
        Find optimal k using cross-validation
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
            
        Returns:
        --------
        optimal_k : int
            Optimal number of neighbors
        cv_scores : dict
            Cross-validation scores for each k
        """
        from sklearn.model_selection import cross_val_score
        
        # Test range of k values
        max_k = min(30, len(X) // 2)  # Don't exceed half the dataset size
        k_range = range(1, max_k + 1, 2)  # Use odd numbers
        
        cv_scores = {}
        best_score = 0
        optimal_k = self.n_neighbors
        
        for k in k_range:
            # Create temporary KNN model
            temp_knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs
            )
            
            # Perform cross-validation
            scores = cross_val_score(temp_knn, X, y, cv=self.cv_folds, scoring='accuracy')
            mean_score = scores.mean()
            cv_scores[k] = {
                'mean': mean_score,
                'std': scores.std(),
                'scores': scores
            }
            
            # Update best k
            if mean_score > best_score:
                best_score = mean_score
                optimal_k = k
        
        return optimal_k, cv_scores
    
    def _analyze_neighbor_structure(self, X, y):
        """
        Analyze the neighbor structure of the training data
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
        """
        # Sample subset for analysis if dataset is large
        if len(X) > 1000:
            indices = np.random.choice(len(X), 1000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Create NearestNeighbors for analysis
        nn = NearestNeighbors(
            n_neighbors=min(self.model_.n_neighbors + 1, len(X_sample)),
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        nn.fit(X_sample)
        
        # Find neighbors for each point
        distances, indices = nn.kneighbors(X_sample)
        
        # Analyze neighbor class consistency
        neighbor_consistency = []
        avg_distances = []
        
        for i in range(len(X_sample)):
            # Exclude self (first neighbor)
            neighbor_indices = indices[i][1:]
            neighbor_distances = distances[i][1:]
            neighbor_labels = y_sample[neighbor_indices]
            
            # Calculate consistency (fraction of neighbors with same class)
            consistency = np.mean(neighbor_labels == y_sample[i])
            neighbor_consistency.append(consistency)
            avg_distances.append(np.mean(neighbor_distances))
        
        self.neighbor_analysis_ = {
            'neighbor_consistency': np.array(neighbor_consistency),
            'average_distances': np.array(avg_distances),
            'mean_consistency': np.mean(neighbor_consistency),
            'std_consistency': np.std(neighbor_consistency),
            'mean_distance': np.mean(avg_distances),
            'std_distance': np.std(avg_distances)
        }
    
    def get_neighbor_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of neighbor structure
        
        Returns:
        --------
        analysis_info : dict
            Information about neighbor structure and data characteristics
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "algorithm_config": {
                "n_neighbors": self.model_.n_neighbors,
                "weights": self.weights,
                "algorithm": self.algorithm,
                "metric": self.metric,
                "p_parameter": self.p
            },
            "data_characteristics": {
                "n_training_samples": len(self.training_data_) if self.training_data_ is not None else "Not stored",
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "feature_scaling_applied": self.scaler_ is not None,
                "scaling_method": self.scaling_method if self.scaler_ is not None else None
            }
        }
        
        # Add optimal k information
        if self.optimal_k_ is not None:
            analysis["optimal_k_analysis"] = {
                "optimal_k": self.optimal_k_,
                "default_k": self.n_neighbors,
                "k_changed": self.optimal_k_ != self.n_neighbors,
                "cv_best_score": max([scores['mean'] for scores in self.cv_scores_.values()]) if self.cv_scores_ else None
            }
        
        # Add neighbor structure analysis
        if self.neighbor_analysis_:
            analysis["neighbor_structure"] = {
                "mean_neighbor_consistency": self.neighbor_analysis_['mean_consistency'],
                "consistency_interpretation": self._interpret_consistency(self.neighbor_analysis_['mean_consistency']),
                "mean_neighbor_distance": self.neighbor_analysis_['mean_distance'],
                "distance_std": self.neighbor_analysis_['std_distance'],
                "data_density": "High" if self.neighbor_analysis_['mean_distance'] < 1.0 else "Medium" if self.neighbor_analysis_['mean_distance'] < 2.0 else "Low"
            }
        
        # Add dimensionality analysis
        analysis["dimensionality_analysis"] = {
            "n_features": self.n_features_in_,
            "curse_of_dimensionality_risk": "High" if self.n_features_in_ > 20 else "Medium" if self.n_features_in_ > 10 else "Low",
            "recommendation": self._get_dimensionality_recommendation()
        }
        
        return analysis
    
    def _interpret_consistency(self, consistency):
        """Interpret neighbor consistency score"""
        if consistency > 0.8:
            return "Excellent - Data has clear local structure"
        elif consistency > 0.6:
            return "Good - Reasonable local class separation"
        elif consistency > 0.4:
            return "Moderate - Some class mixing in neighborhoods"
        else:
            return "Poor - High class mixing, consider other algorithms"
    
    def _get_dimensionality_recommendation(self):
        """Get recommendation based on dimensionality"""
        if self.n_features_in_ > 20:
            return "Consider dimensionality reduction (PCA, feature selection) before using KNN"
        elif self.n_features_in_ > 10:
            return "Monitor performance carefully; consider feature selection"
        else:
            return "Dimensionality is appropriate for KNN"
    
    def plot_k_validation_curve(self, figsize=(10, 6)):
        """
        Plot cross-validation scores for different k values
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            K validation curve plot
        """
        if not self.cv_scores_:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        k_values = list(self.cv_scores_.keys())
        mean_scores = [self.cv_scores_[k]['mean'] for k in k_values]
        std_scores = [self.cv_scores_[k]['std'] for k in k_values]
        
        # Plot mean scores with error bars
        ax.errorbar(k_values, mean_scores, yerr=std_scores, 
                   marker='o', linewidth=2, capsize=5, capthick=2)
        
        # Highlight optimal k
        if self.optimal_k_ in k_values:
            optimal_idx = k_values.index(self.optimal_k_)
            ax.plot(self.optimal_k_, mean_scores[optimal_idx], 
                   'ro', markersize=10, label=f'Optimal k={self.optimal_k_}')
        
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.set_title('KNN Cross-Validation Scores vs k')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotation for optimal k
        if self.optimal_k_ in k_values:
            ax.annotate(f'Optimal k={self.optimal_k_}\nAccuracy={mean_scores[optimal_idx]:.3f}',
                       xy=(self.optimal_k_, mean_scores[optimal_idx]),
                       xytext=(self.optimal_k_ + 2, mean_scores[optimal_idx] + 0.01),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, ha='left')
        
        plt.tight_layout()
        return fig
    
    def plot_neighbor_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive neighbor analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Neighbor analysis visualization
        """
        if not self.neighbor_analysis_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Neighbor Consistency Distribution
        consistency = self.neighbor_analysis_['neighbor_consistency']
        ax1.hist(consistency, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(consistency), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(consistency):.3f}')
        ax1.set_xlabel('Neighbor Class Consistency')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Neighbor Class Consistency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distance Distribution
        distances = self.neighbor_analysis_['average_distances']
        ax2.hist(distances, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(np.mean(distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(distances):.3f}')
        ax2.set_xlabel('Average Distance to Neighbors')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Average Neighbor Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Consistency vs Distance Scatter
        ax3.scatter(distances, consistency, alpha=0.6, color='purple')
        ax3.set_xlabel('Average Distance to Neighbors')
        ax3.set_ylabel('Neighbor Class Consistency')
        ax3.set_title('Consistency vs Distance Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(distances, consistency)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Algorithm Performance Metrics
        metrics = {
            'Consistency': np.mean(consistency),
            'Distance Uniformity': 1 / (1 + np.std(distances)),
            'Data Density': 1 / (1 + np.mean(distances)),
            'Class Separation': np.mean(consistency)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, color=['orange', 'cyan', 'pink', 'lightcoral'])
        ax4.set_ylabel('Score')
        ax4.set_title('KNN Performance Indicators')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary_2d(self, X, y, feature_indices=(0, 1), figsize=(10, 8), resolution=100):
        """
        Plot 2D decision boundary for visualization (works only with 2D data or 2 selected features)
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Target data
        feature_indices : tuple, default=(0, 1)
            Indices of features to plot
        figsize : tuple, default=(10, 8)
            Figure size
        resolution : int, default=100
            Grid resolution for decision boundary
            
        Returns:
        --------
        fig : matplotlib figure
            Decision boundary plot
        """
        if not self.is_fitted_:
            return None
        
        if X.shape[1] < 2:
            return None
        
        # Select two features
        X_2d = X[:, feature_indices]
        
        # Apply scaling if used
        if self.scaler_ is not None:
            # Create a temporary scaler for 2D data
            temp_scaler = type(self.scaler_)()
            X_2d_scaled = temp_scaler.fit_transform(X_2d)
        else:
            X_2d_scaled = X_2d
        
        # Create a temporary 2D KNN model
        temp_knn = KNeighborsClassifier(
            n_neighbors=self.model_.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p
        )
        
        # Encode labels
        y_encoded = self.label_encoder_.transform(y)
        temp_knn.fit(X_2d_scaled, y_encoded)
        
        # Create meshgrid
        x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
        y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on meshgrid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = temp_knn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
        
        # Plot data points
        scatter = ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], 
                           c=y_encoded, cmap=plt.cm.Set1, edgecolors='black')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
        
        ax.set_xlabel(f'Feature {feature_indices[0]}' + 
                     (f' ({self.feature_names_[feature_indices[0]]})' if self.feature_names_ else ''))
        ax.set_ylabel(f'Feature {feature_indices[1]}' + 
                     (f' ({self.feature_names_[feature_indices[1]]})' if self.feature_names_ else ''))
        ax.set_title(f'KNN Decision Boundary (k={self.model_.n_neighbors})')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🎯 K-Nearest Neighbors Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Distance", "Algorithm", "Scaling", "Info"])
        
        with tab1:
            st.markdown("**Core KNN Parameters**")
            
            # Number of neighbors
            n_neighbors = st.slider(
                "Number of Neighbors (k):",
                min_value=1,
                max_value=50,
                value=int(self.n_neighbors),
                step=1,
                help="Number of nearest neighbors to consider. Odd numbers recommended for binary classification.",
                key=f"{key_prefix}_n_neighbors"
            )
            
            # Auto-find optimal k
            cross_validation_k = st.checkbox(
                "Auto-find Optimal k",
                value=self.cross_validation_k,
                help="Use cross-validation to find the best k value",
                key=f"{key_prefix}_cross_validation_k"
            )
            
            if cross_validation_k:
                cv_folds = st.slider(
                    "CV Folds:",
                    min_value=3,
                    max_value=10,
                    value=int(self.cv_folds),
                    help="Number of cross-validation folds",
                    key=f"{key_prefix}_cv_folds"
                )
            else:
                cv_folds = self.cv_folds
            
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
                options=['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'hamming'],
                index=['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'hamming'].index(self.metric),
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
            
            # Distance threshold (advanced)
            enable_distance_threshold = st.checkbox(
                "Enable Distance Threshold",
                value=self.distance_threshold is not None,
                help="Set maximum distance for considering neighbors",
                key=f"{key_prefix}_enable_distance_threshold"
            )
            
            if enable_distance_threshold:
                distance_threshold = st.number_input(
                    "Distance Threshold:",
                    value=1.0 if self.distance_threshold is None else float(self.distance_threshold),
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Maximum distance for neighbors to be considered",
                    key=f"{key_prefix}_distance_threshold"
                )
            else:
                distance_threshold = None
        
        with tab3:
            st.markdown("**Algorithm Optimization**")
            
            # Algorithm
            algorithm = st.selectbox(
                "Algorithm:",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(self.algorithm),
                help="Algorithm used to compute nearest neighbors",
                key=f"{key_prefix}_algorithm"
            )
            
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
            
            # Outlier detection
            outlier_detection = st.checkbox(
                "Enable Outlier Detection",
                value=self.outlier_detection,
                help="Detect outliers based on average neighbor distance",
                key=f"{key_prefix}_outlier_detection"
            )
        
        with tab4:
            st.markdown("**Feature Scaling**")
            
            # Auto scaling
            auto_scaling = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scaling,
                help="Automatically scale features (HIGHLY RECOMMENDED for KNN)",
                key=f"{key_prefix}_auto_scaling"
            )
            
            if auto_scaling:
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    options=['standard', 'minmax', 'robust'],
                    index=['standard', 'minmax', 'robust'].index(self.scaling_method),
                    help="standard: mean=0, std=1; minmax: range [0,1]; robust: uses median and IQR",
                    key=f"{key_prefix}_scaling_method"
                )
            else:
                scaling_method = self.scaling_method
                st.warning("⚠️ Feature scaling is highly recommended for KNN!")
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **K-Nearest Neighbors** - Simple Non-Parametric Learning:
            • 🎯 No assumptions about data distribution
            • 🏃‍♂️ Lazy learning - no explicit training phase
            • 🔍 Instance-based classification
            • 📏 Distance-dependent predictions
            • 🌍 Local learning approach
            • 💾 Memory-based algorithm
            
            **Key Characteristics:**
            • Simple to understand and implement
            • Naturally handles multi-class problems
            • Can capture complex decision boundaries
            • Sensitive to feature scaling
            """)
            
            # Feature scaling guide
            if st.button("📏 Why Feature Scaling?", key=f"{key_prefix}_scaling_guide"):
                st.markdown("""
                **Feature Scaling for KNN:**
                
                **Problem:** Distance metrics treat all features equally
                - Age: 0-100 vs Income: 0-100,000
                - Income dominates distance calculation
                - KNN becomes biased toward high-magnitude features
                
                **Solution:** Scale all features to similar ranges
                - **Standard Scaling:** Mean=0, Std=1
                - **Min-Max Scaling:** Range [0,1]
                - **Robust Scaling:** Uses median and IQR
                
                **Recommendation:** Always scale unless you have a specific reason not to!
                """)
            
            # Distance metrics guide
            if st.button("📐 Distance Metrics Guide", key=f"{key_prefix}_distance_guide"):
                st.markdown("""
                **Choosing Distance Metrics:**
                
                **Euclidean (L2):** √(Σ(x_i - y_i)²)
                - Best for: Continuous features, natural geometry
                - Most common choice
                
                **Manhattan (L1):** Σ|x_i - y_i|
                - Best for: Grid-like data, robust to outliers
                - Good for mixed data types
                
                **Cosine:** 1 - cos(θ)
                - Best for: Text data, high-dimensional sparse data
                - Focuses on direction, not magnitude
                
                **Minkowski:** (Σ|x_i - y_i|^p)^(1/p)
                - Generalizes Manhattan (p=1) and Euclidean (p=2)
                """)
            
            # Hyperparameter tuning guide
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_strategy"):
                st.markdown("""
                **KNN Tuning Strategy:**
                
                **Step 1: Data Preparation**
                - Scale features (critical!)
                - Handle missing values
                - Consider dimensionality reduction if >20 features
                
                **Step 2: Choose k**
                - Start with k=5 or k=7
                - Use cross-validation to find optimal k
                - Odd numbers for binary classification
                - Rule of thumb: k = √n_samples
                
                **Step 3: Select Distance Metric**
                - Euclidean for continuous features
                - Manhattan for mixed types
                - Cosine for text/sparse data
                
                **Step 4: Algorithm Optimization**
                - 'auto' for automatic selection
                - 'ball_tree' for high dimensions
                - 'kd_tree' for low dimensions (<20)
                """)
            
            # Curse of dimensionality warning
            if st.button("⚠️ Curse of Dimensionality", key=f"{key_prefix}_curse_info"):
                st.markdown("""
                **High-Dimensional Data Warning:**
                
                **Problem:** In high dimensions (>20 features):
                - All points become equidistant
                - "Nearest" neighbors aren't actually near
                - Performance degrades significantly
                
                **Solutions:**
                - Dimensionality reduction (PCA, t-SNE)
                - Feature selection
                - Consider other algorithms (SVM, Random Forest)
                
                **Rule:** KNN works best with <20 features
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "p": p,
            "metric": metric,
            "n_jobs": n_jobs,
            "auto_scaling": auto_scaling,
            "scaling_method": scaling_method,
            "distance_threshold": distance_threshold,
            "outlier_detection": outlier_detection,
            "cross_validation_k": cross_validation_k,
            "cv_folds": cv_folds,
            "_ui_options": {
                "show_k_validation": cross_validation_k,
                "show_neighbor_analysis": True,
                "show_decision_boundary": True
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return KNNClassifierPlugin(
            n_neighbors=hyperparameters.get("n_neighbors", self.n_neighbors),
            weights=hyperparameters.get("weights", self.weights),
            algorithm=hyperparameters.get("algorithm", self.algorithm),
            leaf_size=hyperparameters.get("leaf_size", self.leaf_size),
            p=hyperparameters.get("p", self.p),
            metric=hyperparameters.get("metric", self.metric),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            auto_scaling=hyperparameters.get("auto_scaling", self.auto_scaling),
            scaling_method=hyperparameters.get("scaling_method", self.scaling_method),
            distance_threshold=hyperparameters.get("distance_threshold", self.distance_threshold),
            outlier_detection=hyperparameters.get("outlier_detection", self.outlier_detection),
            cross_validation_k=hyperparameters.get("cross_validation_k", self.cross_validation_k),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for KNN
        
        KNN requires careful preprocessing, especially feature scaling.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # KNN doesn't handle missing values well
        if np.any(pd.isna(X_processed)):
            warnings.warn("KNN doesn't handle missing values. Consider imputation before training.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if KNN is compatible with the given data
        
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
            return False, f"KNN requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "KNN doesn't handle missing values well. Please impute missing values first."
        
        # Check dimensionality
        if X.shape[1] > 50:
            return True, "Warning: High dimensionality detected. KNN may perform poorly. Consider dimensionality reduction."
        elif X.shape[1] > 20:
            return True, "Moderate dimensionality. Monitor performance and consider feature selection."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
        
        return True, "KNN is compatible with this data. Remember to scale features!"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "n_neighbors": self.model_.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "optimal_k": self.optimal_k_,
            "scaling_applied": self.scaler_ is not None,
            "scaling_method": self.scaling_method if self.scaler_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "K-Nearest Neighbors",
            "training_completed": True,
            "knn_characteristics": {
                "lazy_learning": True,
                "non_parametric": True,
                "instance_based": True,
                "distance_dependent": True,
                "local_learning": True,
                "memory_based": True
            },
            "model_configuration": {
                "n_neighbors": self.model_.n_neighbors,
                "weights": self.weights,
                "algorithm": self.algorithm,
                "metric": self.metric,
                "feature_scaling": self.scaler_ is not None
            },
            "neighbor_analysis": self.get_neighbor_analysis(),
            "performance_considerations": {
                "memory_usage": "Stores entire training dataset",
                "prediction_time": "O(n*d) per prediction",
                "training_time": "O(1) - just stores data",
                "scalability": "Poor for large datasets"
            }
        }
        
        return info


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return KNNClassifierPlugin()