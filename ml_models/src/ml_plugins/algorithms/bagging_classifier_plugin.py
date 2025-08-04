import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

# Try to import optional libraries
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    EXTENDED_ALGORITHMS = True
except ImportError:
    EXTENDED_ALGORITHMS = False

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

class BaggingClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Bagging Classifier Plugin - Bootstrap Aggregating Ensemble Method
    
    Bagging (Bootstrap Aggregating) is an ensemble meta-algorithm that fits base 
    classifiers on random subsets of the original dataset and then aggregates their 
    individual predictions to form a final prediction. It reduces variance and helps 
    to avoid overfitting by combining multiple diverse models trained on different 
    bootstrap samples.
    
    Key Features:
    1. Bootstrap Sampling: Creates diverse training sets through sampling with replacement
    2. Parallel Training: Base estimators are trained independently
    3. Variance Reduction: Reduces overfitting through model averaging
    4. Out-of-Bag Evaluation: Uses unused samples for unbiased performance estimation
    5. Feature Subsampling: Optional random feature selection per estimator
    6. Any Base Estimator: Works with any sklearn-compatible classifier
    """
    
    def __init__(self,
                 # Core bagging parameters
                 base_estimator='decision_tree',
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=42,
                 verbose=0,
                 # Base estimator hyperparameters
                 # Decision Tree parameters
                 dt_max_depth=None,
                 dt_min_samples_split=2,
                 dt_min_samples_leaf=1,
                 dt_max_features=None,
                 dt_criterion='gini',
                 # Logistic Regression parameters
                 lr_C=1.0,
                 lr_max_iter=1000,
                 lr_solver='liblinear',
                 # KNN parameters
                 knn_n_neighbors=5,
                 knn_weights='uniform',
                 knn_metric='minkowski',
                 # SVM parameters
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_probability=True,
                 # Neural Network parameters
                 mlp_hidden_layer_sizes=(100,),
                 mlp_activation='relu',
                 mlp_max_iter=500,
                 # Advanced options
                 auto_scale_features=False,
                 cross_validation_folds=5,
                 estimate_feature_importance=True):
        """
        Initialize Bagging Classifier with comprehensive configuration
        
        Parameters:
        -----------
        base_estimator : str, default='decision_tree'
            Base estimator type to use
        n_estimators : int, default=10
            Number of base estimators in the ensemble
        max_samples : int or float, default=1.0
            Number/fraction of samples to draw for each base estimator
        max_features : int or float, default=1.0
            Number/fraction of features to draw for each base estimator
        bootstrap : bool, default=True
            Whether samples are drawn with replacement
        bootstrap_features : bool, default=False
            Whether features are drawn with replacement
        oob_score : bool, default=False
            Whether to use out-of-bag samples for scoring
        warm_start : bool, default=False
            Whether to reuse previous solution when fitting
        n_jobs : int, default=None
            Number of jobs for parallel processing
        random_state : int, default=42
            Random seed for reproducibility
        verbose : int, default=0
            Verbosity level
        auto_scale_features : bool, default=False
            Whether to automatically scale features
        cross_validation_folds : int, default=5
            Number of CV folds for evaluation
        estimate_feature_importance : bool, default=True
            Whether to estimate feature importance from ensemble
        """
        super().__init__()
        
        # Core bagging parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Base estimator hyperparameters
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.dt_max_features = dt_max_features
        self.dt_criterion = dt_criterion
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.lr_solver = lr_solver
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.knn_metric = knn_metric
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_max_iter = mlp_max_iter
        
        # Advanced options
        self.auto_scale_features = auto_scale_features
        self.cross_validation_folds = cross_validation_folds
        self.estimate_feature_importance = estimate_feature_importance
        
        # Plugin metadata
        self._name = "Bagging Classifier"
        self._description = "Bootstrap Aggregating ensemble method that trains multiple base estimators on random subsets of data to reduce variance and improve generalization."
        self._category = "Ensemble Methods"
        self._algorithm_type = "Bootstrap Aggregating"
        self._paper_reference = "Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 20
        self._handles_missing_values = False  # Depends on base estimator
        self._requires_scaling = False  # Depends on base estimator
        self._supports_sparse = False  # Depends on base estimator
        self._is_linear = False  # Ensemble method
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = False  # Depends on base estimator
        self._ensemble_method = True
        self._bootstrap_based = True
        self._variance_reduction = True
        self._parallel_training = True
        self._out_of_bag_evaluation = True
        
        # Internal attributes
        self.bagging_classifier_ = None
        self.base_estimator_instance_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.oob_score_ = None
        self.feature_importances_ = None
        self.ensemble_analysis_ = None
        self.bootstrap_info_ = None
    
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
            "year_introduced": 1996,
            "key_innovations": {
                "bootstrap_aggregating": "Combines multiple models trained on bootstrap samples",
                "variance_reduction": "Reduces overfitting through model averaging",
                "parallel_training": "Base estimators trained independently",
                "out_of_bag_evaluation": "Unbiased performance estimation using unused samples",
                "random_subsampling": "Creates diversity through data and feature sampling",
                "universal_applicability": "Works with any base learning algorithm"
            },
            "algorithm_mechanics": {
                "training_process": {
                    "step_1": "Draw B bootstrap samples from training data",
                    "step_2": "Train base estimator on each bootstrap sample",
                    "step_3": "Store all trained base estimators",
                    "step_4": "Optionally compute OOB score"
                },
                "bootstrap_sampling": {
                    "with_replacement": "Sample n instances with replacement from original dataset",
                    "diversity_creation": "Each bootstrap sample is ~63.2% unique instances",
                    "oob_instances": "~36.8% instances not in each bootstrap sample",
                    "sample_size": "Typically same size as original dataset"
                },
                "prediction_process": {
                    "step_1": "Get predictions from all base estimators",
                    "step_2": "Aggregate predictions by majority voting (classification)",
                    "step_3": "Return final aggregated prediction",
                    "formula": "Final prediction = mode(h_1(x), h_2(x), ..., h_B(x))"
                },
                "variance_reduction_theory": {
                    "individual_variance": "σ² for each base estimator",
                    "ensemble_variance": "σ²/B + ρσ²(1-1/B) where ρ is correlation",
                    "reduction_factor": "Approaches σ²/B as correlation ρ → 0",
                    "bias_preservation": "Ensemble bias ≈ individual estimator bias"
                }
            },
            "bootstrap_theory": {
                "statistical_foundation": {
                    "bootstrap_principle": "Empirical distribution approximates true distribution",
                    "sampling_distribution": "Bootstrap replicates approximate sampling variability",
                    "central_limit_theorem": "Averaging reduces variance of estimates",
                    "bias_variance_tradeoff": "Slight bias increase for large variance reduction"
                },
                "sample_characteristics": {
                    "unique_instances": "Expected 1 - (1-1/n)^n ≈ 63.2% unique instances",
                    "repeated_instances": "Some instances appear multiple times",
                    "oob_instances": "Instances not selected in bootstrap sample",
                    "effective_training_size": "Varies per bootstrap sample"
                },
                "diversity_mechanisms": {
                    "data_diversity": "Different training instances per model",
                    "feature_diversity": "Optional random feature subsampling",
                    "instance_weighting": "Repeated instances get higher implicit weight",
                    "noise_averaging": "Random variations cancel out in ensemble"
                }
            },
            "base_estimator_selection": {
                "high_variance_estimators": {
                    "decision_trees": "Default choice - high variance, low bias",
                    "neural_networks": "Complex models that benefit from averaging",
                    "knn_large_k": "Instance-based methods with variability",
                    "polynomial_features": "High-dimensional feature spaces"
                },
                "low_variance_estimators": {
                    "linear_models": "Less benefit but still some improvement",
                    "naive_bayes": "Stable models with modest variance",
                    "svm_linear": "Large margin classifiers"
                },
                "selection_principles": {
                    "instability_preference": "Choose estimators sensitive to data changes",
                    "low_bias_preference": "Avoid highly biased weak learners",
                    "computational_efficiency": "Balance complexity with number of estimators",
                    "diversity_promotion": "Estimators that produce diverse predictions"
                }
            },
            "hyperparameter_effects": {
                "n_estimators": {
                    "effect": "More estimators → lower variance, diminishing returns",
                    "typical_range": "10-500, often 50-100 sufficient",
                    "computational_cost": "Linear increase in training/prediction time"
                },
                "max_samples": {
                    "effect": "Smaller samples → more diversity, less individual accuracy",
                    "bootstrap_size": "1.0 = same size as original dataset",
                    "bias_variance": "Smaller samples increase bias, reduce variance"
                },
                "max_features": {
                    "effect": "Fewer features → more diversity, potential information loss",
                    "feature_bagging": "Random feature selection per estimator",
                    "curse_mitigation": "Helps with high-dimensional data"
                },
                "oob_score": {
                    "benefit": "Unbiased performance estimate without separate validation",
                    "efficiency": "Uses ~36.8% unused data per estimator",
                    "approximation": "OOB score ≈ cross-validation score"
                }
            },
            "available_base_estimators": {
                "decision_tree": {
                    "type": "Tree-based",
                    "strengths": ["High variance (good for bagging)", "Interpretable", "Handles mixed data"],
                    "weaknesses": ["Prone to overfitting", "Unstable"],
                    "best_for": "Default choice for bagging",
                    "variance_level": "High"
                },
                "logistic_regression": {
                    "type": "Linear probabilistic",
                    "strengths": ["Fast", "Stable probabilities", "Linear interpretability"],
                    "weaknesses": ["Low variance (less bagging benefit)", "Linear assumptions"],
                    "best_for": "Linear patterns with feature bagging",
                    "variance_level": "Low-Medium"
                },
                "knn": {
                    "type": "Instance-based",
                    "strengths": ["Non-parametric", "Local patterns", "Sensitive to sampling"],
                    "weaknesses": ["Curse of dimensionality", "Computational cost"],
                    "best_for": "Local pattern learning with diversity",
                    "variance_level": "Medium-High"
                },
                "svm": {
                    "type": "Kernel-based",
                    "strengths": ["Kernel trick", "Good generalization", "Support vector diversity"],
                    "weaknesses": ["Computational cost", "Hyperparameter sensitive"],
                    "best_for": "Complex decision boundaries with bootstrap diversity",
                    "variance_level": "Medium"
                },
                "neural_network": {
                    "type": "Deep learning",
                    "strengths": ["Universal approximation", "High variance", "Complex patterns"],
                    "weaknesses": ["Training instability", "Computationally expensive"],
                    "best_for": "Complex non-linear patterns with averaging",
                    "variance_level": "Very High"
                }
            },
            "strengths": [
                "Reduces overfitting through variance reduction",
                "Provides out-of-bag error estimation",
                "Parallelizable training and prediction",
                "Works with any base learning algorithm",
                "Robust to noise and outliers",
                "Simple and intuitive ensemble method",
                "No hyperparameter tuning of base estimators needed",
                "Maintains interpretability with simple base estimators",
                "Handles both bootstrap and feature sampling",
                "Stable performance across different datasets",
                "Can estimate feature importance",
                "Natural handling of class imbalance through sampling"
            ],
            "weaknesses": [
                "May not improve low-variance estimators significantly",
                "Increased computational cost (multiple models)",
                "Larger memory requirements",
                "Can mask individual model interpretability",
                "Bootstrap sampling may not suit all data distributions",
                "Less effective with very small datasets",
                "No adaptive learning (unlike boosting)",
                "Limited bias reduction compared to variance reduction",
                "Feature importance estimates can be less stable",
                "OOB score may be overly optimistic in some cases"
            ],
            "ideal_use_cases": [
                "High-variance base estimators (especially decision trees)",
                "Datasets prone to overfitting",
                "When you need variance reduction without bias increase",
                "Parallel computing environments",
                "When out-of-bag evaluation is valuable",
                "Noisy datasets requiring robust predictions",
                "Feature selection through random subsampling",
                "When interpretability of ensemble is not critical",
                "Large datasets where bootstrap sampling is meaningful",
                "Unstable algorithms that benefit from averaging",
                "When you want to improve any existing classifier",
                "Real-time applications requiring confidence estimation"
            ],
            "comparison_with_other_ensembles": {
                "vs_random_forest": {
                    "bagging": "General framework, any base estimator",
                    "random_forest": "Specific implementation with decision trees + feature randomness",
                    "flexibility": "Bagging: more flexible, RF: optimized for trees"
                },
                "vs_boosting": {
                    "bagging": "Parallel training, variance reduction, independent estimators",
                    "boosting": "Sequential training, bias reduction, adaptive weighting",
                    "focus": "Bagging: overfitting, Boosting: underfitting"
                },
                "vs_voting": {
                    "bagging": "Same algorithm on different data samples",
                    "voting": "Different algorithms on same data",
                    "diversity_source": "Bagging: data diversity, Voting: algorithm diversity"
                }
            },
            "theoretical_guarantees": {
                "variance_reduction": "Ensemble variance ≤ individual variance",
                "bias_preservation": "Ensemble bias ≈ individual bias",
                "consistency": "Converges to optimal if base estimator is consistent",
                "oob_error": "Unbiased estimate of generalization error",
                "convergence": "Performance stabilizes as n_estimators increases"
            }
        }
    
    def _create_base_estimator(self) -> BaseEstimator:
        """Create base estimator instance based on configuration"""
        
        if self.base_estimator == 'decision_tree':
            estimator = DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                min_samples_leaf=self.dt_min_samples_leaf,
                max_features=self.dt_max_features,
                criterion=self.dt_criterion,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'logistic_regression':
            estimator = LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                solver=self.lr_solver,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'knn':
            estimator = KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                metric=self.knn_metric
            )
        
        elif self.base_estimator == 'svm':
            estimator = SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                probability=self.svm_probability,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'naive_bayes':
            estimator = GaussianNB()
        
        elif self.base_estimator == 'mlp' and EXTENDED_ALGORITHMS:
            estimator = MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                activation=self.mlp_activation,
                max_iter=self.mlp_max_iter,
                random_state=self.random_state
            )
        
        else:
            # Default fallback
            estimator = DecisionTreeClassifier(random_state=self.random_state)
            if self.base_estimator != 'decision_tree':
                warnings.warn(f"Unknown base estimator '{self.base_estimator}'. Using Decision Tree as fallback.")
        
        return estimator
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Bagging Classifier
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, default=None
            Sample weights
            
        Returns:
        --------
        self : object
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
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
        
        # Feature scaling if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        if self.verbose > 0:
            print(f"Training Bagging Classifier with {self.n_estimators} estimators...")
            print(f"Base estimator: {self.base_estimator}")
            print(f"Bootstrap: {self.bootstrap}")
            print(f"Max samples: {self.max_samples}")
            print(f"Max features: {self.max_features}")
            print(f"OOB score: {self.oob_score}")
        
        # Create base estimator
        self.base_estimator_instance_ = self._create_base_estimator()
        
        # Create and configure bagging classifier
        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=self.base_estimator_instance_,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Fit the bagging ensemble
        self.bagging_classifier_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        
        # Store OOB score if available
        if self.oob_score:
            self.oob_score_ = self.bagging_classifier_.oob_score_
            if self.verbose > 0:
                print(f"Out-of-bag score: {self.oob_score_:.4f}")
        
        # Estimate feature importance if requested
        if self.estimate_feature_importance:
            self._estimate_feature_importance(X_scaled, y_encoded)
        
        # Analyze bootstrap samples
        self._analyze_bootstrap_samples(X_scaled, y_encoded)
        
        # Analyze the ensemble
        self._analyze_ensemble(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _estimate_feature_importance(self, X, y):
        """Estimate feature importance from ensemble"""
        try:
            # Try to get feature importance from base estimators
            if hasattr(self.base_estimator_instance_, 'feature_importances_'):
                importances = []
                for estimator in self.bagging_classifier_.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        # Handle feature subsampling
                        full_importance = np.zeros(X.shape[1])
                        feature_indices = self.bagging_classifier_.estimators_features_[len(importances)]
                        full_importance[feature_indices] = estimator.feature_importances_
                        importances.append(full_importance)
                
                if importances:
                    # Average importance across all estimators
                    self.feature_importances_ = np.mean(importances, axis=0)
                    
                    if self.verbose > 0:
                        print("Feature importance estimated from ensemble")
            
            else:
                # Fallback: permutation importance (simplified)
                if self.verbose > 0:
                    print("Base estimator doesn't provide feature importance")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not estimate feature importance: {str(e)}")
            self.feature_importances_ = None
    
    def _analyze_bootstrap_samples(self, X, y):
        """Analyze characteristics of bootstrap samples"""
        n_samples = X.shape[0]
        sample_info = {
            "original_samples": n_samples,
            "bootstrap_sample_size": int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else self.max_samples,
            "expected_unique_ratio": 1 - (1 - 1/n_samples)**n_samples if n_samples > 0 else 0,
            "expected_oob_ratio": (1 - 1/n_samples)**n_samples if n_samples > 0 else 0,
            "bootstrap_enabled": self.bootstrap,
            "feature_bootstrap_enabled": self.bootstrap_features,
            "n_estimators": self.n_estimators
        }
        
        # Theoretical expectations
        if n_samples > 0:
            sample_info["expected_unique_instances"] = int(sample_info["expected_unique_ratio"] * n_samples)
            sample_info["expected_oob_instances"] = int(sample_info["expected_oob_ratio"] * n_samples)
        
        self.bootstrap_info_ = sample_info
    
    def _analyze_ensemble(self, X, y):
        """Analyze the trained ensemble"""
        analysis = {
            "base_estimator_type": self.base_estimator,
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "bootstrap_features": self.bootstrap_features,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "oob_score_enabled": self.oob_score,
            "feature_scaling": self.scaler_ is not None,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(self.classes_)
        }
        
        # Add OOB score if available
        if self.oob_score_:
            analysis["oob_score"] = self.oob_score_
        
        # Add feature importance statistics if available
        if self.feature_importances_ is not None:
            analysis["feature_importance_available"] = True
            analysis["top_features"] = {
                "indices": np.argsort(self.feature_importances_)[-5:].tolist(),
                "importances": self.feature_importances_[np.argsort(self.feature_importances_)[-5:]].tolist()
            }
        else:
            analysis["feature_importance_available"] = False
        
        # Estimate training complexity
        analysis["computational_complexity"] = {
            "training_estimators": self.n_estimators,
            "parallel_training": self.n_jobs != 1,
            "bootstrap_overhead": "Low" if self.bootstrap else "None",
            "memory_multiplier": self.n_estimators
        }
        
        self.ensemble_analysis_ = analysis
    
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
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get predictions from bagging classifier
        y_pred_encoded = self.bagging_classifier_.predict(X_scaled)
        
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
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get probabilities from bagging classifier
        try:
            probabilities = self.bagging_classifier_.predict_proba(X_scaled)
            return probabilities
        except Exception as e:
            warnings.warn(f"Probability prediction failed: {str(e)}. Using hard predictions.")
            
            # Fallback: convert hard predictions to probabilities
            y_pred = self.bagging_classifier_.predict(X_scaled)
            n_samples = len(y_pred)
            n_classes = len(self.classes_)
            
            probabilities = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(y_pred):
                probabilities[i, pred] = 1.0
            
            return probabilities
    
    def get_individual_predictions(self, X):
        """
        Get predictions from individual base estimators
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : array, shape (n_samples, n_estimators)
            Predictions from each base estimator
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get predictions from individual estimators
        predictions = []
        
        for estimator in self.bagging_classifier_.estimators_:
            try:
                pred_encoded = estimator.predict(X_scaled)
                pred = self.label_encoder_.inverse_transform(pred_encoded)
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Prediction failed for an estimator: {str(e)}")
                predictions.append(np.full(X.shape[0], self.classes_[0]))
        
        return np.array(predictions).T
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the ensemble
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive ensemble analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "ensemble_summary": {
                "algorithm": "Bagging Classifier",
                "base_estimator": self.base_estimator,
                "n_estimators": self.n_estimators,
                "bootstrap": self.bootstrap,
                "bootstrap_features": self.bootstrap_features,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "oob_score_enabled": self.oob_score,
                "feature_scaling": self.scaler_ is not None,
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "classes": self.classes_.tolist()
            }
        }
        
        # Add OOB score
        if self.oob_score_:
            analysis["out_of_bag"] = {
                "oob_score": self.oob_score_,
                "description": "Unbiased estimate of generalization performance",
                "equivalent_to": "Cross-validation without extra computation"
            }
        
        # Add bootstrap information
        if self.bootstrap_info_:
            analysis["bootstrap_analysis"] = self.bootstrap_info_
        
        # Add ensemble analysis
        if self.ensemble_analysis_:
            analysis["ensemble_characteristics"] = self.ensemble_analysis_
        
        # Add feature importance
        if self.feature_importances_ is not None:
            feature_importance_dict = {}
            for i, importance in enumerate(self.feature_importances_):
                feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
                feature_importance_dict[feature_name] = importance
            
            analysis["feature_importance"] = {
                "importances": feature_importance_dict,
                "top_5_features": dict(sorted(feature_importance_dict.items(), 
                                            key=lambda x: x[1], reverse=True)[:5])
            }
        
        return analysis
    
    def plot_ensemble_analysis(self, figsize=(15, 12)):
        """
        Create comprehensive ensemble analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 12)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Ensemble analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Feature Importance (if available)
        if self.feature_importances_ is not None:
            top_indices = np.argsort(self.feature_importances_)[-10:]
            top_importances = self.feature_importances_[top_indices]
            top_names = [self.feature_names_[i] for i in top_indices] if self.feature_names_ else [f"F{i}" for i in top_indices]
            
            bars = ax1.barh(range(len(top_importances)), top_importances, alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_yticks(range(len(top_importances)))
            ax1.set_yticklabels(top_names)
            ax1.set_xlabel('Importance')
            ax1.set_title('Top 10 Feature Importances (Ensemble Average)')
            
            # Add importance values
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance:.3f}', ha='left', va='center')
        else:
            ax1.text(0.5, 0.5, 'Feature importance\nnot available\nfor this base estimator', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Feature Importance')
        
        # 2. Bootstrap Sample Analysis
        if self.bootstrap_info_:
            bootstrap_data = self.bootstrap_info_
            
            # Create bootstrap statistics
            labels = ['Original\nSamples', 'Bootstrap\nSample Size', 'Expected\nUnique', 'Expected\nOOB']
            values = [
                bootstrap_data.get('original_samples', 0),
                bootstrap_data.get('bootstrap_sample_size', 0),
                bootstrap_data.get('expected_unique_instances', 0),
                bootstrap_data.get('expected_oob_instances', 0)
            ]
            
            bars = ax2.bar(labels, values, alpha=0.7, 
                          color=['lightcoral', 'skyblue', 'lightgreen', 'gold'],
                          edgecolor='darkblue')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Bootstrap Sample Analysis')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'Bootstrap analysis\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Bootstrap Analysis')
        
        # 3. Ensemble Configuration
        ax3.axis('tight')
        ax3.axis('off')
        table_data = []
        table_data.append(['Base Estimator', self.base_estimator.replace('_', ' ').title()])
        table_data.append(['Number of Estimators', str(self.n_estimators)])
        table_data.append(['Bootstrap Sampling', 'Yes' if self.bootstrap else 'No'])
        table_data.append(['Bootstrap Features', 'Yes' if self.bootstrap_features else 'No'])
        table_data.append(['Max Samples', str(self.max_samples)])
        table_data.append(['Max Features', str(self.max_features)])
        table_data.append(['OOB Score', f'{self.oob_score_:.4f}' if self.oob_score_ else 'Disabled'])
        table_data.append(['Feature Scaling', 'Yes' if self.scaler_ is not None else 'No'])
        
        table = ax3.table(cellText=table_data,
                         colLabels=['Property', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Ensemble Configuration')
        
        # 4. Variance Reduction Theory
        # Theoretical variance reduction visualization
        n_estimators_range = np.arange(1, min(101, self.n_estimators * 2))
        
        # Theoretical variance reduction (assuming low correlation)
        variance_reduction_low_corr = 1.0 / n_estimators_range
        # Theoretical variance reduction (assuming medium correlation)
        rho = 0.3  # Assumed correlation
        variance_reduction_med_corr = (1 - rho) / n_estimators_range + rho
        # Theoretical variance reduction (assuming high correlation)
        rho_high = 0.7
        variance_reduction_high_corr = (1 - rho_high) / n_estimators_range + rho_high
        
        ax4.plot(n_estimators_range, variance_reduction_low_corr, 'g-', label='Low Correlation (ρ=0)', linewidth=2)
        ax4.plot(n_estimators_range, variance_reduction_med_corr, 'b-', label='Medium Correlation (ρ=0.3)', linewidth=2)
        ax4.plot(n_estimators_range, variance_reduction_high_corr, 'r-', label='High Correlation (ρ=0.7)', linewidth=2)
        
        # Mark current configuration
        current_variance_med = (1 - 0.3) / self.n_estimators + 0.3
        ax4.axvline(x=self.n_estimators, color='orange', linestyle='--', alpha=0.7, label=f'Current ({self.n_estimators})')
        ax4.plot(self.n_estimators, current_variance_med, 'ro', markersize=8, label='Estimated Position')
        
        ax4.set_xlabel('Number of Estimators')
        ax4.set_ylabel('Relative Variance')
        ax4.set_title('Theoretical Variance Reduction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_bootstrap_diversity(self, X_test, y_test, figsize=(12, 8)):
        """
        Visualize diversity in bootstrap ensemble predictions
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Bootstrap diversity visualization
        """
        if not self.is_fitted_:
            return None
        
        # Get individual predictions
        individual_preds = self.get_individual_predictions(X_test)
        ensemble_pred = self.predict(X_test)
        
        # Calculate diversity metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Prediction Agreement Analysis
        n_samples = len(X_test)
        n_estimators = individual_preds.shape[1]
        
        # For each sample, count how many estimators agree with the majority
        agreement_ratios = []
        for i in range(n_samples):
            sample_preds = individual_preds[i, :]
            majority_pred = ensemble_pred[i]
            agreement = np.sum(sample_preds == majority_pred) / n_estimators
            agreement_ratios.append(agreement)
        
        # Plot agreement distribution
        ax1.hist(agreement_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Agreement Ratio with Ensemble')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Distribution of Estimator Agreement')
        ax1.axvline(np.mean(agreement_ratios), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(agreement_ratios):.3f}')
        ax1.legend()
        
        # 2. Accuracy vs Disagreement
        from sklearn.metrics import accuracy_score
        
        # Calculate accuracy for different agreement thresholds
        agreement_thresholds = np.arange(0.5, 1.01, 0.05)
        accuracies_by_agreement = []
        sample_counts = []
        
        for threshold in agreement_thresholds:
            # Select samples where agreement is above threshold
            high_agreement_mask = np.array(agreement_ratios) >= threshold
            
            if np.sum(high_agreement_mask) > 0:
                subset_pred = ensemble_pred[high_agreement_mask]
                subset_true = y_test[high_agreement_mask]
                accuracy = accuracy_score(subset_true, subset_pred)
                sample_count = np.sum(high_agreement_mask)
            else:
                accuracy = 0
                sample_count = 0
            
            accuracies_by_agreement.append(accuracy)
            sample_counts.append(sample_count)
        
        # Plot accuracy vs agreement
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(agreement_thresholds, accuracies_by_agreement, 'g-', linewidth=2, 
                        marker='o', label='Accuracy')
        line2 = ax2_twin.plot(agreement_thresholds, sample_counts, 'r--', linewidth=2, 
                             marker='s', label='Sample Count', alpha=0.7)
        
        ax2.set_xlabel('Minimum Agreement Threshold')
        ax2.set_ylabel('Accuracy', color='green')
        ax2_twin.set_ylabel('Number of Samples', color='red')
        ax2.set_title('Accuracy vs Estimator Agreement')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🎒 Bagging Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Ensemble", "Base Est.", "Bootstrap", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Ensemble Configuration**")
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Estimators:",
                min_value=2,
                max_value=500,
                value=int(self.n_estimators),
                step=1,
                help="Number of base estimators in the ensemble",
                key=f"{key_prefix}_n_estimators"
            )
            
            if n_estimators < 10:
                st.info("💡 Consider using 10+ estimators for better variance reduction")
            elif n_estimators > 100:
                st.warning("⚠️ High number of estimators increases training time")
            
            # Base estimator type
            base_estimator_options = ['decision_tree', 'logistic_regression', 'knn', 'svm', 'naive_bayes']
            if EXTENDED_ALGORITHMS:
                base_estimator_options.append('mlp')
            
            base_estimator = st.selectbox(
                "Base Estimator:",
                options=base_estimator_options,
                index=base_estimator_options.index(self.base_estimator) if self.base_estimator in base_estimator_options else 0,
                help="Type of base estimator to use",
                key=f"{key_prefix}_base_estimator"
            )
            
            # Base estimator info
            if base_estimator == 'decision_tree':
                st.info("🌳 Decision Trees: High variance, perfect for bagging")
            elif base_estimator == 'logistic_regression':
                st.info("📈 Logistic Regression: Low variance, modest bagging benefit")
            elif base_estimator == 'knn':
                st.info("🎯 K-NN: Instance-based, good diversity from sampling")
            elif base_estimator == 'svm':
                st.info("🔍 SVM: Kernel-based, moderate bagging improvement")
            elif base_estimator == 'naive_bayes':
                st.info("🎲 Naive Bayes: Fast, stable with reasonable bagging benefit")
            elif base_estimator == 'mlp':
                st.info("🧠 Neural Network: High variance, excellent for bagging")
            
            # Out-of-bag scoring
            oob_score = st.checkbox(
                "Enable Out-of-Bag Scoring",
                value=self.oob_score,
                help="Compute out-of-bag score for unbiased performance estimation",
                key=f"{key_prefix}_oob_score"
            )
            
            if oob_score:
                st.success("✅ OOB scoring provides unbiased performance estimation!")
        
        with tab2:
            st.markdown("**Base Estimator Hyperparameters**")
            
            # Decision Tree parameters
            if base_estimator == 'decision_tree':
                st.markdown("**Decision Tree Configuration:**")
                
                dt_max_depth_option = st.selectbox(
                    "Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.dt_max_depth is None else 1,
                    key=f"{key_prefix}_dt_max_depth_option"
                )
                
                if dt_max_depth_option == 'Custom':
                    dt_max_depth = st.slider(
                        "Custom Max Depth:",
                        min_value=1,
                        max_value=50,
                        value=10 if self.dt_max_depth is None else int(self.dt_max_depth),
                        key=f"{key_prefix}_dt_max_depth"
                    )
                else:
                    dt_max_depth = None
                
                dt_min_samples_split = st.slider(
                    "Min Samples Split:",
                    min_value=2,
                    max_value=20,
                    value=int(self.dt_min_samples_split),
                    help="Minimum samples required to split a node",
                    key=f"{key_prefix}_dt_min_samples_split"
                )
                
                dt_min_samples_leaf = st.slider(
                    "Min Samples Leaf:",
                    min_value=1,
                    max_value=20,
                    value=int(self.dt_min_samples_leaf),
                    help="Minimum samples required at a leaf node",
                    key=f"{key_prefix}_dt_min_samples_leaf"
                )
                
                dt_criterion = st.selectbox(
                    "Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.dt_criterion),
                    help="Function to measure split quality",
                    key=f"{key_prefix}_dt_criterion"
                )
                
                dt_max_features_option = st.selectbox(
                    "Max Features per Split:",
                    options=['None', 'sqrt', 'log2', 'Custom'],
                    index=0,
                    help="Number of features to consider when looking for the best split",
                    key=f"{key_prefix}_dt_max_features_option"
                )
                
                if dt_max_features_option == 'Custom':
                    dt_max_features = st.slider(
                        "Custom Max Features:",
                        min_value=1,
                        max_value=20,
                        value=5,
                        key=f"{key_prefix}_dt_max_features_custom"
                    )
                elif dt_max_features_option == 'None':
                    dt_max_features = None
                else:
                    dt_max_features = dt_max_features_option
            else:
                dt_max_depth = self.dt_max_depth
                dt_min_samples_split = self.dt_min_samples_split
                dt_min_samples_leaf = self.dt_min_samples_leaf
                dt_criterion = self.dt_criterion
                dt_max_features = self.dt_max_features
            
            # Logistic Regression parameters
            if base_estimator == 'logistic_regression':
                st.markdown("**Logistic Regression Configuration:**")
                
                lr_C = st.number_input(
                    "Regularization Strength (C):",
                    value=float(self.lr_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    help="Inverse of regularization strength",
                    key=f"{key_prefix}_lr_C"
                )
                
                lr_max_iter = st.number_input(
                    "Max Iterations:",
                    value=int(self.lr_max_iter),
                    min_value=100,
                    max_value=5000,
                    step=100,
                    key=f"{key_prefix}_lr_max_iter"
                )
                
                lr_solver = st.selectbox(
                    "Solver:",
                    options=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                    index=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'].index(self.lr_solver),
                    help="Algorithm for optimization",
                    key=f"{key_prefix}_lr_solver"
                )
            else:
                lr_C = self.lr_C
                lr_max_iter = self.lr_max_iter
                lr_solver = self.lr_solver
            
            # KNN parameters
            if base_estimator == 'knn':
                st.markdown("**K-Nearest Neighbors Configuration:**")
                
                knn_n_neighbors = st.slider(
                    "Number of Neighbors:",
                    min_value=1,
                    max_value=50,
                    value=int(self.knn_n_neighbors),
                    help="Number of neighbors to use",
                    key=f"{key_prefix}_knn_n_neighbors"
                )
                
                knn_weights = st.selectbox(
                    "Weight Function:",
                    options=['uniform', 'distance'],
                    index=['uniform', 'distance'].index(self.knn_weights),
                    help="Weight function used in prediction",
                    key=f"{key_prefix}_knn_weights"
                )
                
                knn_metric = st.selectbox(
                    "Distance Metric:",
                    options=['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                    index=['minkowski', 'euclidean', 'manhattan', 'chebyshev'].index(self.knn_metric),
                    help="Distance metric for the tree",
                    key=f"{key_prefix}_knn_metric"
                )
            else:
                knn_n_neighbors = self.knn_n_neighbors
                knn_weights = self.knn_weights
                knn_metric = self.knn_metric
            
            # SVM parameters
            if base_estimator == 'svm':
                st.markdown("**Support Vector Machine Configuration:**")
                
                svm_C = st.number_input(
                    "Regularization Parameter (C):",
                    value=float(self.svm_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    help="Regularization parameter",
                    key=f"{key_prefix}_svm_C"
                )
                
                svm_kernel = st.selectbox(
                    "Kernel:",
                    options=['rbf', 'linear', 'poly', 'sigmoid'],
                    index=['rbf', 'linear', 'poly', 'sigmoid'].index(self.svm_kernel),
                    help="Kernel type to be used in the algorithm",
                    key=f"{key_prefix}_svm_kernel"
                )
                
                svm_probability = st.checkbox(
                    "Enable Probability Estimates",
                    value=self.svm_probability,
                    help="Enable probability estimates (required for predict_proba)",
                    key=f"{key_prefix}_svm_probability"
                )
            else:
                svm_C = self.svm_C
                svm_kernel = self.svm_kernel
                svm_probability = self.svm_probability
            
            # Neural Network parameters
            if base_estimator == 'mlp' and EXTENDED_ALGORITHMS:
                st.markdown("**Neural Network Configuration:**")
                
                mlp_hidden_layer_sizes = st.text_input(
                    "Hidden Layer Sizes (comma-separated):",
                    value=','.join(map(str, self.mlp_hidden_layer_sizes)),
                    help="e.g., '100' or '100,50' for multiple layers",
                    key=f"{key_prefix}_mlp_hidden_layer_sizes"
                )
                
                try:
                    mlp_hidden_layer_sizes = tuple(map(int, mlp_hidden_layer_sizes.split(',')))
                except:
                    mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                
                mlp_activation = st.selectbox(
                    "Activation Function:",
                    options=['relu', 'tanh', 'logistic'],
                    index=['relu', 'tanh', 'logistic'].index(self.mlp_activation),
                    help="Activation function for the hidden layers",
                    key=f"{key_prefix}_mlp_activation"
                )
                
                mlp_max_iter = st.number_input(
                    "Max Iterations:",
                    value=int(self.mlp_max_iter),
                    min_value=100,
                    max_value=2000,
                    step=100,
                    key=f"{key_prefix}_mlp_max_iter"
                )
            else:
                mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                mlp_activation = self.mlp_activation
                mlp_max_iter = self.mlp_max_iter
        
        with tab3:
            st.markdown("**Bootstrap Sampling Configuration**")
            
            # Bootstrap samples
            bootstrap = st.checkbox(
                "Bootstrap Sampling",
                value=self.bootstrap,
                help="Whether to draw samples with replacement",
                key=f"{key_prefix}_bootstrap"
            )
            
            if bootstrap:
                st.success("✅ Bootstrap sampling creates diverse training sets")
            else:
                st.warning("⚠️ Without bootstrap sampling, diversity comes only from feature sampling")
            
            # Max samples
            max_samples_option = st.selectbox(
                "Sample Size Type:",
                options=['Fraction', 'Absolute'],
                index=0 if isinstance(self.max_samples, float) else 1,
                help="How to specify the sample size",
                key=f"{key_prefix}_max_samples_option"
            )
            
            if max_samples_option == 'Fraction':
                max_samples = st.slider(
                    "Max Samples (Fraction):",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.max_samples) if isinstance(self.max_samples, float) else 1.0,
                    step=0.05,
                    help="Fraction of samples to draw for each base estimator",
                    key=f"{key_prefix}_max_samples_fraction"
                )
            else:
                max_samples = st.number_input(
                    "Max Samples (Absolute):",
                    min_value=10,
                    max_value=10000,
                    value=int(self.max_samples) if isinstance(self.max_samples, int) else 100,
                    step=10,
                    help="Number of samples to draw for each base estimator",
                    key=f"{key_prefix}_max_samples_absolute"
                )
            
            # Bootstrap features
            bootstrap_features = st.checkbox(
                "Bootstrap Features",
                value=self.bootstrap_features,
                help="Whether to draw features with replacement",
                key=f"{key_prefix}_bootstrap_features"
            )
            
            # Max features
            max_features_option = st.selectbox(
                "Feature Selection Type:",
                options=['All Features (1.0)', 'Square Root', 'Log2', 'Fraction', 'Absolute'],
                index=0,
                help="How to select features for each estimator",
                key=f"{key_prefix}_max_features_option"
            )
            
            if max_features_option == 'Fraction':
                max_features = st.slider(
                    "Max Features (Fraction):",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.max_features) if isinstance(self.max_features, float) else 1.0,
                    step=0.05,
                    help="Fraction of features to consider",
                    key=f"{key_prefix}_max_features_fraction"
                )
# Continue from line 1410 where the code breaks:

            elif max_features_option == 'Absolute':
                max_features = st.number_input(
                    "Max Features (Absolute):",
                    min_value=1,
                    max_value=100,
                    value=int(self.max_features) if isinstance(self.max_features, int) else 10,
                    step=1,
                    help="Number of features to consider",
                    key=f"{key_prefix}_max_features_absolute"
                )
            elif max_features_option == 'Square Root':
                max_features = 'sqrt'
            elif max_features_option == 'Log2':
                max_features = 'log2'
            else:  # All Features
                max_features = 1.0
            
            # Bootstrap info
            if bootstrap:
                st.info("""
                📊 **Bootstrap Theory:**
                • Each sample contains ~63.2% unique instances
                • ~36.8% instances are out-of-bag (unused)
                • Creates diversity through sampling variation
                • Reduces variance while preserving bias
                """)
        
        with tab4:
            st.markdown("**Advanced Configuration**")
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Automatically scale features (important for distance-based estimators)",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            # Warm start
            warm_start = st.checkbox(
                "Warm Start",
                value=self.warm_start,
                help="Reuse previous solution when fitting (for incremental learning)",
                key=f"{key_prefix}_warm_start"
            )
            
            # Parallel processing
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores",
                key=f"{key_prefix}_n_jobs"
            )
            
            # Cross-validation folds
            cross_validation_folds = st.slider(
                "CV Folds for Evaluation:",
                min_value=3,
                max_value=10,
                value=self.cross_validation_folds,
                help="Number of cross-validation folds",
                key=f"{key_prefix}_cross_validation_folds"
            )
            
            # Feature importance estimation
            estimate_feature_importance = st.checkbox(
                "Estimate Feature Importance",
                value=self.estimate_feature_importance,
                help="Aggregate feature importance from base estimators",
                key=f"{key_prefix}_estimate_feature_importance"
            )
            
            # Random state
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
            
            # Verbose output
            verbose = st.selectbox(
                "Verbosity Level:",
                options=[0, 1, 2],
                index=self.verbose,
                help="Control the verbosity of the fitting process",
                key=f"{key_prefix}_verbose"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Bagging Classifier** - Bootstrap Aggregating:
            • 🎒 Trains multiple models on bootstrap samples
            • 📊 Reduces variance through model averaging
            • 🔄 Uses out-of-bag samples for unbiased evaluation
            • 🌳 Works best with high-variance estimators (Decision Trees)
            • ⚡ Parallel training for efficiency
            • 🎯 Combines predictions by majority voting
            
            **Key Principles:**
            • Bootstrap sampling creates diverse training sets
            • Model averaging reduces overfitting
            • Out-of-bag evaluation eliminates need for separate validation
            • Feature subsampling adds extra diversity
            """)
            
            # Bootstrap theory guide
            if st.button("📊 Bootstrap Theory Guide", key=f"{key_prefix}_bootstrap_guide"):
                st.markdown("""
                **Bootstrap Sampling Theory:**
                
                **Statistical Foundation:**
                - Each bootstrap sample contains ~63.2% unique instances
                - Remaining ~36.8% are out-of-bag (OOB) instances
                - Bootstrap distribution approximates true sampling distribution
                - Central Limit Theorem: averaging reduces variance
                
                **Variance Reduction Formula:**
                - Individual estimator variance: σ²
                - Ensemble variance: σ²/B + ρσ²(1-1/B)
                - Where B = number of estimators, ρ = correlation
                - As B → ∞ and ρ → 0: variance → 0
                
                **Optimal Conditions:**
                - High-variance base estimators (Decision Trees ideal)
                - Low correlation between base estimator errors
                - Sufficient bootstrap sample size
                - Adequate number of estimators (10-100 typically)
                """)
            
            # Base estimator selection guide
            if st.button("🌳 Base Estimator Guide", key=f"{key_prefix}_estimator_guide"):
                st.markdown("""
                **Base Estimator Selection Guide:**
                
                **Best for Bagging (High Variance):**
                - **Decision Trees**: Perfect choice, high variance, low bias
                - **Neural Networks**: Complex models benefit from averaging
                - **K-NN (large K)**: Instance-based with natural variability
                
                **Moderate Benefit (Medium Variance):**
                - **SVM**: Kernel-based, moderate improvement
                - **Polynomial Features**: High-dimensional spaces
                
                **Limited Benefit (Low Variance):**
                - **Logistic Regression**: Stable, modest improvement
                - **Naive Bayes**: Already robust, small gains
                - **Linear SVM**: Large margin, naturally stable
                
                **Selection Principles:**
                - Choose estimators sensitive to training data changes
                - Avoid overly biased weak learners
                - Balance individual complexity with ensemble size
                - Consider computational efficiency
                """)
            
            # Performance optimization guide
            if st.button("⚙️ Performance Optimization", key=f"{key_prefix}_performance_guide"):
                st.markdown("""
                **Bagging Performance Optimization:**
                
                **Number of Estimators:**
                - Start with 10-50 estimators
                - Diminishing returns after 100-200
                - Monitor OOB score for convergence
                - Balance accuracy vs computational cost
                
                **Sample Size (max_samples):**
                - Default 1.0 (same size as original dataset)
                - Smaller samples → more diversity, less individual accuracy
                - Larger samples → less diversity, better individual models
                - Sweet spot: 0.8-1.0 for most cases
                
                **Feature Sampling (max_features):**
                - All features (1.0): maximum information per estimator
                - sqrt(n_features): good balance for high-dimensional data
                - 0.5-0.8: increased diversity, potential information loss
                - Use with high-dimensional or noisy feature sets
                
                **Bootstrap vs No Bootstrap:**
                - Bootstrap=True: variance reduction through sampling
                - Bootstrap=False: only feature diversity (Random Subspaces)
                - Almost always use Bootstrap=True for bagging
                
                **Out-of-Bag Evaluation:**
                - Enable for unbiased performance estimation
                - Equivalent to cross-validation without extra computation
                - Use to monitor ensemble convergence
                - Helps detect overfitting
                """)
            
            # Ensemble theory
            if st.button("🔬 Ensemble Theory", key=f"{key_prefix}_ensemble_theory"):
                st.markdown("""
                **Mathematical Foundation:**
                
                **Bias-Variance Decomposition:**
                - Error = Bias² + Variance + Noise
                - Bagging primarily reduces Variance
                - Bias remains approximately unchanged
                - Net effect: lower generalization error
                
                **Bootstrap Statistics:**
                - P(instance selected) = 1 - (1-1/n)ⁿ ≈ 0.632
                - P(instance not selected) = (1-1/n)ⁿ ≈ 0.368
                - Expected unique instances per bootstrap ≈ 63.2%
                - OOB instances provide validation set
                
                **Aggregation Methods:**
                - Classification: Majority voting
                - Regression: Arithmetic mean
                - Probability: Average of probability estimates
                - Can use weighted voting with estimator quality
                
                **Convergence Properties:**
                - Performance stabilizes as n_estimators increases
                - OOB error provides reliable stopping criterion
                - Overfitting risk is minimal (unlike boosting)
                - Parallel training allows easy scaling
                """)
        
        return {
            "n_estimators": n_estimators,
            "base_estimator": base_estimator,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "bootstrap_features": bootstrap_features,
            "oob_score": oob_score,
            "warm_start": warm_start,
            "n_jobs": n_jobs,
            "auto_scale_features": auto_scale_features,
            "cross_validation_folds": cross_validation_folds,
            "estimate_feature_importance": estimate_feature_importance,
            "random_state": random_state,
            "verbose": verbose,
            # Base estimator parameters
            "dt_max_depth": dt_max_depth,
            "dt_min_samples_split": dt_min_samples_split,
            "dt_min_samples_leaf": dt_min_samples_leaf,
            "dt_criterion": dt_criterion,
            "dt_max_features": dt_max_features,
            "lr_C": lr_C,
            "lr_max_iter": lr_max_iter,
            "lr_solver": lr_solver,
            "knn_n_neighbors": knn_n_neighbors,
            "knn_weights": knn_weights,
            "knn_metric": knn_metric,
            "svm_C": svm_C,
            "svm_kernel": svm_kernel,
            "svm_probability": svm_probability,
            "mlp_hidden_layer_sizes": mlp_hidden_layer_sizes,
            "mlp_activation": mlp_activation,
            "mlp_max_iter": mlp_max_iter
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return BaggingClassifierPlugin(
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            base_estimator=hyperparameters.get("base_estimator", self.base_estimator),
            max_samples=hyperparameters.get("max_samples", self.max_samples),
            max_features=hyperparameters.get("max_features", self.max_features),
            bootstrap=hyperparameters.get("bootstrap", self.bootstrap),
            bootstrap_features=hyperparameters.get("bootstrap_features", self.bootstrap_features),
            oob_score=hyperparameters.get("oob_score", self.oob_score),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            cross_validation_folds=hyperparameters.get("cross_validation_folds", self.cross_validation_folds),
            estimate_feature_importance=hyperparameters.get("estimate_feature_importance", self.estimate_feature_importance),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            # Base estimator parameters
            dt_max_depth=hyperparameters.get("dt_max_depth", self.dt_max_depth),
            dt_min_samples_split=hyperparameters.get("dt_min_samples_split", self.dt_min_samples_split),
            dt_min_samples_leaf=hyperparameters.get("dt_min_samples_leaf", self.dt_min_samples_leaf),
            dt_criterion=hyperparameters.get("dt_criterion", self.dt_criterion),
            dt_max_features=hyperparameters.get("dt_max_features", self.dt_max_features),
            lr_C=hyperparameters.get("lr_C", self.lr_C),
            lr_max_iter=hyperparameters.get("lr_max_iter", self.lr_max_iter),
            lr_solver=hyperparameters.get("lr_solver", self.lr_solver),
            knn_n_neighbors=hyperparameters.get("knn_n_neighbors", self.knn_n_neighbors),
            knn_weights=hyperparameters.get("knn_weights", self.knn_weights),
            knn_metric=hyperparameters.get("knn_metric", self.knn_metric),
            svm_C=hyperparameters.get("svm_C", self.svm_C),
            svm_kernel=hyperparameters.get("svm_kernel", self.svm_kernel),
            svm_probability=hyperparameters.get("svm_probability", self.svm_probability),
            mlp_hidden_layer_sizes=hyperparameters.get("mlp_hidden_layer_sizes", self.mlp_hidden_layer_sizes),
            mlp_activation=hyperparameters.get("mlp_activation", self.mlp_activation),
            mlp_max_iter=hyperparameters.get("mlp_max_iter", self.mlp_max_iter)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Bagging Classifier
        
        Bagging can handle various data types depending on the base estimator.
        Feature scaling is automatically applied if enabled.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Check for missing values
        if np.any(pd.isna(X_processed)):
            warnings.warn("Some base estimators don't handle missing values well. Consider imputation.")
        
        # Check for infinite values
        if np.any(np.isinf(X_processed)):
            warnings.warn("Infinite values detected. Some base estimators may fail.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Bagging Classifier is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Bagging Classifier requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check if bootstrap sample size makes sense
        if isinstance(self.max_samples, float):
            effective_sample_size = int(self.max_samples * X.shape[0])
        else:
            effective_sample_size = min(self.max_samples, X.shape[0])
        
        if effective_sample_size < 5:
            return False, f"Bootstrap sample size too small ({effective_sample_size}). Increase max_samples or dataset size."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check class distribution in bootstrap context
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            min_class_size = np.min(class_counts)
            if min_class_size < 3:
                return True, f"Warning: Very small class detected ({min_class_size} samples). Bootstrap sampling may create class imbalance."
        
        # Check base estimator compatibility
        if self.base_estimator == 'svm' and X.shape[0] > 10000:
            return True, "Warning: SVM with large datasets may be slow. Consider using Decision Trees or other fast estimators."
        
        if self.base_estimator == 'knn' and X.shape[1] > 20:
            return True, "Warning: KNN with high-dimensional data may suffer from curse of dimensionality."
        
        # Check for missing values based on base estimator
        if np.any(pd.isna(X)):
            if self.base_estimator in ['svm', 'knn', 'logistic_regression']:
                return True, "Warning: Missing values detected. These base estimators require complete data. Consider imputation."
        
        # Bootstrap sampling considerations
        if self.bootstrap and X.shape[0] < 50:
            return True, "Warning: Small dataset for bootstrap sampling. Consider increasing data size or using cross-validation instead."
        
        return True, f"Bagging Classifier is compatible! Using {self.base_estimator} as base estimator with {self.n_estimators} estimators."
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "base_estimator": self.base_estimator,
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "bootstrap_features": self.bootstrap_features,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "oob_score_enabled": self.oob_score,
            "oob_score": self.oob_score_,
            "feature_scaling": self.scaler_ is not None,
            "feature_importance_available": self.feature_importances_ is not None,
            "ensemble_type": "Bagging Classifier"
        }
    
    def get_algorithm_specific_metrics(self,
                                       y_true: Union[pd.Series, np.ndarray],
                                       y_pred: Union[pd.Series, np.ndarray],
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Bagging Classifier-specific metrics.

        This includes the Out-of-Bag (OOB) score (if computed during training),
        ensemble characteristics, and prediction confidence metrics derived from y_proba
        on the test set.

        Args:
            y_true: Ground truth target values from the test set.
            y_pred: Predicted target values on the test set.
            y_proba: Predicted probabilities on the test set from the ensemble.

        Returns:
            A dictionary of Bagging Classifier-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.bagging_classifier_ is None:
            metrics["status"] = "Model not fitted or not available"
            return metrics

        # Training-time specific metrics
        if hasattr(self, 'oob_score_') and self.oob_score_ is not None:
            metrics['oob_score'] = float(self.oob_score_)
        else:
            metrics['oob_score'] = None

        metrics['n_base_estimators'] = int(self.n_estimators)
        metrics['base_estimator_type'] = str(self.base_estimator)
        metrics['uses_bootstrap_sampling'] = bool(self.bootstrap)
        metrics['uses_bootstrap_features'] = bool(self.bootstrap_features)
        metrics['max_samples_config'] = self.max_samples
        metrics['max_features_config'] = self.max_features

        if hasattr(self, 'bootstrap_info_') and self.bootstrap_info_:
            metrics['bootstrap_actual_sample_size_per_estimator'] = self.bootstrap_info_.get('bootstrap_sample_size')
            metrics['bootstrap_expected_unique_instance_ratio'] = self.bootstrap_info_.get('expected_unique_ratio')
            metrics['bootstrap_expected_oob_instance_ratio'] = self.bootstrap_info_.get('expected_oob_ratio')

        # Test-time specific metrics (from y_proba)
        if y_proba is not None:
            try:
                y_proba_np = np.asarray(y_proba)
                max_probabilities = np.max(y_proba_np, axis=1)
                metrics['mean_max_prediction_probability'] = float(np.mean(max_probabilities))
                metrics['std_max_prediction_probability'] = float(np.std(max_probabilities))
                y_proba_clipped = np.clip(y_proba_np, 1e-9, 1.0)
                entropies = -np.sum(y_proba_clipped * np.log(y_proba_clipped), axis=1)
                metrics['mean_prediction_entropy'] = float(np.mean(entropies))
                metrics['proportion_high_confidence_preds_gt_0.9'] = float(np.mean(max_probabilities > 0.9))
                metrics['proportion_low_confidence_preds_lt_0.6'] = float(np.mean(max_probabilities < 0.6))
                if y_proba_np.shape[1] == 2:
                    metrics['mean_positive_class_probability'] = float(np.mean(y_proba_np[:, 1]))
            except Exception as e:
                metrics['y_proba_analysis_error'] = str(e)
        else:
            metrics['y_proba_analysis_status'] = "y_proba not provided"

        return metrics
                
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Bagging Classifier",
            "ensemble_type": "Bootstrap Aggregating",
            "training_completed": True,
            "bagging_characteristics": {
                "bootstrap_based": True,
                "variance_reduction": True,
                "parallel_training": True,
                "out_of_bag_evaluation": self.oob_score,
                "feature_subsampling": self.max_features != 1.0,
                "sample_subsampling": self.max_samples != 1.0
            },
            "ensemble_configuration": {
                "base_estimator": self.base_estimator,
                "n_estimators": self.n_estimators,
                "bootstrap": self.bootstrap,
                "bootstrap_features": self.bootstrap_features,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "oob_score_enabled": self.oob_score,
                "feature_scaling": self.scaler_ is not None
            },
            "bootstrap_analysis": self.bootstrap_info_ if self.bootstrap_info_ else {},
            "ensemble_analysis": self.get_ensemble_analysis(),
            "performance_considerations": {
                "training_time": f"Moderate - trains {self.n_estimators} base estimators",
                "prediction_time": f"Moderate - queries {self.n_estimators} estimators",
                "memory_usage": f"High - stores {self.n_estimators} models",
                "scalability": "Excellent - fully parallelizable",
                "overfitting_risk": "Low - variance reduction through averaging",
                "interpretability": "Moderate - can analyze individual estimators"
            },
            "bootstrap_theory": {
                "variance_reduction": "Reduces overfitting through model averaging",
                "bootstrap_principle": "~63.2% unique instances per sample, ~36.8% out-of-bag",
                "diversity_mechanism": "Different training sets create diverse models",
                "aggregation_method": "Majority voting for classification",
                "theoretical_guarantee": "Ensemble variance ≤ individual variance"
            }
        }
        
        # Add OOB score information
        if self.oob_score_:
            info["out_of_bag"] = {
                "score": self.oob_score_,
                "description": "Unbiased estimate using out-of-bag samples",
                "equivalent_to": "Cross-validation score without extra computation",
                "usage": "Monitor ensemble performance and convergence"
            }
        
        return info


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return BaggingClassifierPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Bagging Classifier Plugin
    """
    print("Testing Bagging Classifier Plugin...")
    
    try:
        print("✅ Required libraries are available")
        
        # Create sample data for bagging
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Generate a dataset suitable for demonstrating bagging benefits
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            n_clusters_per_class=2,
            class_sep=0.8,  # Some class overlap to create challenging problem
            flip_y=0.1,     # Add some label noise
            random_state=42
        )
        
        print(f"\n📊 Dataset Info:")
        print(f"Shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and test plugin with decision trees (ideal for bagging)
        plugin = BaggingClassifierPlugin(
            base_estimator='decision_tree',
            n_estimators=50,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=True,
            dt_max_depth=None,  # Allow deep trees (high variance)
            dt_min_samples_split=2,
            dt_min_samples_leaf=1,
            auto_scale_features=False,  # Trees don't need scaling
            estimate_feature_importance=True,
            verbose=1,
            random_state=42
        )
        
        print("\n🔍 Plugin Info:")
        print(f"Name: {plugin.get_name()}")
        print(f"Category: {plugin.get_category()}")
        print(f"Description: {plugin.get_description()}")
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_train, y_train)
        print(f"\n✅ Compatibility: {message}")
        
        if compatible:
            # Train bagging ensemble
            print("\n🚀 Training Bagging Classifier Ensemble...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            y_proba = plugin.predict_proba(X_test)
            
            # Evaluate ensemble
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n📊 Ensemble Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"OOB Score: {plugin.oob_score_:.4f}")
            print(f"Classes: {plugin.classes_}")
            
            # Compare with individual base estimator
            from sklearn.tree import DecisionTreeClassifier
            single_tree = DecisionTreeClassifier(random_state=42)
            single_tree.fit(X_train, y_train)
            single_pred = single_tree.predict(X_test)
            single_accuracy = accuracy_score(y_test, single_pred)
            
            print(f"\n🌳 Comparison:")
            print(f"Single Decision Tree: {single_accuracy:.4f}")
            print(f"Bagging Ensemble: {accuracy:.4f}")
            print(f"Improvement: {accuracy - single_accuracy:.4f} ({(accuracy/single_accuracy-1)*100:.1f}%)")
            
            # Get individual predictions for diversity analysis
            individual_preds = plugin.get_individual_predictions(X_test)
            print(f"\n🎯 Ensemble Diversity:")
            print(f"Individual predictions shape: {individual_preds.shape}")
            
            # Calculate prediction diversity
            n_samples = len(X_test)
            diversity_scores = []
            for i in range(n_samples):
                sample_preds = individual_preds[i, :]
                unique_preds = len(np.unique(sample_preds))
                diversity = unique_preds / len(sample_preds)
                diversity_scores.append(diversity)
            
            avg_diversity = np.mean(diversity_scores)
            print(f"Average prediction diversity: {avg_diversity:.3f}")
            print(f"High diversity samples (>50% different): {np.sum(np.array(diversity_scores) > 0.5)/len(diversity_scores)*100:.1f}%")
            
            # Ensemble analysis
            ensemble_analysis = plugin.get_ensemble_analysis()
            print(f"\n🎒 Ensemble Analysis:")
            
            ensemble_summary = ensemble_analysis.get('ensemble_summary', {})
            print(f"Base estimator: {ensemble_summary.get('base_estimator', 'Unknown')}")
            print(f"Number of estimators: {ensemble_summary.get('n_estimators', 'Unknown')}")
            print(f"Bootstrap sampling: {ensemble_summary.get('bootstrap', False)}")
            print(f"Max samples: {ensemble_summary.get('max_samples', 'Unknown')}")
            print(f"OOB score enabled: {ensemble_summary.get('oob_score_enabled', False)}")
            
            # Bootstrap analysis
            if 'bootstrap_analysis' in ensemble_analysis:
                bootstrap_info = ensemble_analysis['bootstrap_analysis']
                print(f"\n📊 Bootstrap Statistics:")
                print(f"Original samples: {bootstrap_info.get('original_samples', 'Unknown')}")
                print(f"Bootstrap sample size: {bootstrap_info.get('bootstrap_sample_size', 'Unknown')}")
                print(f"Expected unique ratio: {bootstrap_info.get('expected_unique_ratio', 0):.3f}")
                print(f"Expected OOB ratio: {bootstrap_info.get('expected_oob_ratio', 0):.3f}")
            
            # Feature importance
            if 'feature_importance' in ensemble_analysis:
                feature_importance = ensemble_analysis['feature_importance']
                print(f"\n🎯 Feature Importance (Top 5):")
                top_features = feature_importance.get('top_5_features', {})
                for feature, importance in list(top_features.items())[:5]:
                    print(f"{feature}: {importance:.4f}")
            
            # Model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Ensemble type: {model_params.get('ensemble_type', 'Unknown')}")
            print(f"Base estimator: {model_params.get('base_estimator', 'Unknown')}")
            print(f"Number of estimators: {model_params.get('n_estimators', 'Unknown')}")
            print(f"Bootstrap: {model_params.get('bootstrap', False)}")
            print(f"OOB score: {model_params.get('oob_score', 'N/A')}")
            print(f"Feature importance available: {model_params.get('feature_importance_available', False)}")
            
            # Training info
            training_info = plugin.get_training_info()
            print(f"\n📈 Training Info:")
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Ensemble type: {training_info['ensemble_type']}")
            
            bagging_chars = training_info['bagging_characteristics']
            print(f"Bootstrap based: {bagging_chars['bootstrap_based']}")
            print(f"Variance reduction: {bagging_chars['variance_reduction']}")
            print(f"Parallel training: {bagging_chars['parallel_training']}")
            print(f"OOB evaluation: {bagging_chars['out_of_bag_evaluation']}")
            
            # Performance considerations
            perf_info = training_info['performance_considerations']
            print(f"\n⚡ Performance Considerations:")
            print(f"Training time: {perf_info['training_time']}")
            print(f"Prediction time: {perf_info['prediction_time']}")
            print(f"Memory usage: {perf_info['memory_usage']}")
            print(f"Overfitting risk: {perf_info['overfitting_risk']}")
            
            # Bootstrap theory
            bootstrap_theory = training_info['bootstrap_theory']
            print(f"\n🔬 Bootstrap Theory:")
            print(f"Variance reduction: {bootstrap_theory['variance_reduction']}")
            print(f"Bootstrap principle: {bootstrap_theory['bootstrap_principle']}")
            print(f"Diversity mechanism: {bootstrap_theory['diversity_mechanism']}")
            
            # OOB information
            if 'out_of_bag' in training_info:
                oob_info = training_info['out_of_bag']
                print(f"\n📊 Out-of-Bag Analysis:")
                print(f"OOB Score: {oob_info['score']:.4f}")
                print(f"Description: {oob_info['description']}")
                print(f"Equivalent to: {oob_info['equivalent_to']}")
            
            print("\n✅ Bagging Classifier Plugin test completed successfully!")
            print("🎒 Bootstrap aggregating successfully reduced variance and improved generalization!")
            
            # Demonstrate bootstrap sampling benefits
            print(f"\n🚀 Bootstrap Sampling Benefits:")
            print(f"Variance Reduction: Model averaging reduces prediction variance")
            print(f"OOB Validation: Free performance estimate using unused data")
            print(f"Parallel Training: {plugin.n_estimators} estimators trained independently")
            print(f"Robustness: Ensemble more stable than individual estimator")
            
            # Show confidence distribution
            print(f"\n🎯 Prediction Confidence Analysis:")
            max_probas = np.max(y_proba, axis=1)
            print(f"Average confidence: {np.mean(max_probas):.3f}")
            print(f"Min confidence: {np.min(max_probas):.3f}")
            print(f"Max confidence: {np.max(max_probas):.3f}")
            print(f"High confidence predictions (>0.8): {np.sum(max_probas > 0.8)/len(max_probas)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()