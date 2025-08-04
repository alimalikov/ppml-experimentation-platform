import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

# Try to import optional libraries
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
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

class OneVsRestClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    One-vs-Rest (OvR) Classifier Plugin - Binary Classifier Wrapper for Multi-class
    
    One-vs-Rest, also known as One-vs-All (OvA), is a strategy for multi-class classification
    that trains one binary classifier per class. Each classifier distinguishes one class from
    all other classes combined. This approach transforms any binary classifier into a
    multi-class classifier while maintaining interpretability and allowing per-class analysis.
    
    Key Features:
    1. Universal Multi-class Wrapper: Converts any binary classifier to multi-class
    2. Independent Binary Problems: Each class gets its own dedicated binary classifier
    3. Parallel Training: All binary classifiers can be trained independently
    4. Class-specific Analysis: Provides individual classifier performance per class
    5. Probability Calibration: Normalizes binary probabilities to valid multi-class probabilities
    6. Interpretable Results: Each binary classifier can be analyzed separately
    7. Handles Class Imbalance: Can apply different strategies per binary problem
    """
    
    def __init__(self,
                 # Base estimator configuration
                 base_estimator='logistic_regression',
                 
                 # OvR-specific parameters
                 n_jobs=None,
                 verbose=0,
                 random_state=42,
                 
                 # Base estimator hyperparameters
                 # Logistic Regression
                 lr_C=1.0,
                 lr_max_iter=1000,
                 lr_solver='liblinear',
                 lr_penalty='l2',
                 
                 # Random Forest
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 rf_criterion='gini',
                 
                 # SVM
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_probability=True,
                 svm_gamma='scale',
                 
                 # Decision Tree
                 dt_max_depth=None,
                 dt_min_samples_split=2,
                 dt_min_samples_leaf=1,
                 dt_criterion='gini',
                 
                 # KNN
                 knn_n_neighbors=5,
                 knn_weights='uniform',
                 knn_metric='minkowski',
                 
                 # Gradient Boosting
                 gb_n_estimators=100,
                 gb_learning_rate=0.1,
                 gb_max_depth=3,
                 gb_subsample=1.0,
                 
                 # Neural Network
                 mlp_hidden_layer_sizes=(100,),
                 mlp_activation='relu',
                 mlp_max_iter=500,
                 mlp_alpha=0.0001,
                 
                 # XGBoost
                 xgb_n_estimators=100,
                 xgb_learning_rate=0.1,
                 xgb_max_depth=6,
                 xgb_subsample=1.0,
                 
                 # Advanced options
                 auto_scale_features=True,
                 estimate_class_performance=True,
                 class_weight_strategy='auto',
                 probability_calibration=True):
        """
        Initialize One-vs-Rest Classifier with comprehensive configuration
        
        Parameters:
        -----------
        base_estimator : str, default='logistic_regression'
            Base binary classifier type
        n_jobs : int, default=None
            Number of parallel jobs for training classifiers
        verbose : int, default=0
            Verbosity level
        random_state : int, default=42
            Random seed for reproducibility
        auto_scale_features : bool, default=True
            Whether to automatically scale features
        estimate_class_performance : bool, default=True
            Whether to evaluate per-class performance
        class_weight_strategy : str, default='auto'
            Strategy for handling class weights in binary problems
        probability_calibration : bool, default=True
            Whether to calibrate probabilities across classes
        """
        super().__init__()
        
        # Base estimator configuration
        self.base_estimator = base_estimator
        
        # OvR parameters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # Base estimator hyperparameters
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.lr_solver = lr_solver
        self.lr_penalty = lr_penalty
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_criterion = rf_criterion
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
        self.svm_gamma = svm_gamma
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.dt_criterion = dt_criterion
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.knn_metric = knn_metric
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_depth = gb_max_depth
        self.gb_subsample = gb_subsample
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_max_iter = mlp_max_iter
        self.mlp_alpha = mlp_alpha
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_subsample = xgb_subsample
        
        # Advanced options
        self.auto_scale_features = auto_scale_features
        self.estimate_class_performance = estimate_class_performance
        self.class_weight_strategy = class_weight_strategy
        self.probability_calibration = probability_calibration
        
        # Plugin metadata
        self._name = "One-vs-Rest Classifier"
        self._description = "Multi-class wrapper that trains one binary classifier per class, transforming any binary algorithm into a multi-class classifier with excellent interpretability."
        self._category = "Multi-class Strategies"
        self._algorithm_type = "Binary Classifier Wrapper"
        self._paper_reference = "Rifkin, R., & Klautau, A. (2004). In defense of one-vs-all classification. Journal of machine learning research, 5, 101-141."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True  # Can handle binary, but designed for multi-class
        self._supports_multiclass = True
        self._min_samples_required = 20  # Need samples for each binary problem
        self._handles_missing_values = False  # Depends on base estimator
        self._requires_scaling = False  # Depends on base estimator
        self._supports_sparse = False  # Depends on base estimator
        self._is_linear = False  # Depends on base estimator
        self._provides_feature_importance = True  # If base estimator supports it
        self._provides_probabilities = True
        self._handles_categorical = False  # Depends on base estimator
        self._multi_class_strategy = True
        self._binary_wrapper = True
        self._per_class_analysis = True
        self._parallel_training = True
        self._class_decomposition = True
        
        # Internal attributes
        self.ovr_classifier_ = None
        self.base_estimator_instance_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.is_fitted_ = False
        self.class_performance_ = None
        self.binary_classifiers_ = None
        self.ovr_analysis_ = None
        self.class_distributions_ = None
    
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
            "year_introduced": 1995,
            "key_innovations": {
                "binary_decomposition": "Decomposes multi-class into multiple binary problems",
                "universal_wrapper": "Works with any binary classifier",
                "independent_training": "Each binary classifier trained independently",
                "per_class_analysis": "Enables detailed analysis of each class separately",
                "parallel_scalability": "Binary classifiers can be trained in parallel",
                "interpretability_preservation": "Maintains interpretability of base classifier"
            },
            "algorithm_mechanics": {
                "problem_decomposition": {
                    "approach": "One binary classifier per class (Class i vs. All other classes)",
                    "n_classifiers": "K binary classifiers for K classes",
                    "training_data": "Each classifier sees all training data with modified labels",
                    "label_transformation": "Class i → Positive, All others → Negative"
                },
                "training_process": {
                    "step_1": "For each class i, create binary training set",
                    "step_2": "Label class i samples as positive (1)",
                    "step_3": "Label all other class samples as negative (0)",
                    "step_4": "Train binary classifier on this transformed dataset",
                    "step_5": "Repeat for all K classes",
                    "parallelization": "All K binary problems can be solved independently"
                },
                "prediction_process": {
                    "step_1": "Get decision scores from all K binary classifiers",
                    "step_2": "For classification: argmax(decision_scores)",
                    "step_3": "For probabilities: normalize scores to sum to 1",
                    "tie_breaking": "Highest decision score wins",
                    "confidence": "Margin between top two scores"
                },
                "probability_estimation": {
                    "individual_probs": "Each binary classifier outputs P(class_i | x)",
                    "normalization": "Probabilities normalized to sum to 1",
                    "calibration": "Optional calibration for better probability estimates",
                    "interpretation": "P(class_i | x) reflects confidence for class i vs. all others"
                }
            },
            "mathematical_foundation": {
                "decision_function": "f_i(x) = decision score for class i vs. all others",
                "classification_rule": "ŷ = argmax_i f_i(x)",
                "probability_estimation": "P(y=i|x) = exp(f_i(x)) / Σ_j exp(f_j(x))",
                "confidence_measure": "confidence = max_i f_i(x) - max_{j≠i} f_j(x)",
                "theoretical_foundation": "Reduces multi-class to well-studied binary problems"
            },
            "ovr_strategy_analysis": {
                "advantages_over_other_strategies": {
                    "vs_one_vs_one": {
                        "efficiency": "O(K) classifiers vs O(K²) classifiers",
                        "training_data": "Uses all training data vs. pairwise subsets",
                        "interpretability": "Clear class vs. rest interpretation",
                        "scalability": "Better for large number of classes"
                    },
                    "vs_multinomial": {
                        "compatibility": "Works with any binary classifier",
                        "simplicity": "Reduces to known binary problem",
                        "debugging": "Each class can be analyzed separately",
                        "flexibility": "Can use different algorithms per class"
                    }
                },
                "class_imbalance_handling": {
                    "natural_imbalance": "Each binary problem inherently imbalanced",
                    "mitigation_strategies": ["Class weights", "Resampling", "Threshold tuning"],
                    "per_class_optimization": "Can optimize each binary classifier separately",
                    "adaptive_strategies": "Different strategies per class based on class size"
                },
                "decision_boundary_analysis": {
                    "geometric_interpretation": "K hyperplanes, each separating one class from others",
                    "region_formation": "Decision regions formed by intersection of half-spaces",
                    "ambiguous_regions": "Regions where multiple classifiers predict positive",
                    "void_regions": "Regions where all classifiers predict negative",
                    "confidence_landscape": "Confidence varies based on distance from boundaries"
                }
            },
            "base_estimator_effects": {
                "linear_classifiers": {
                    "logistic_regression": {
                        "advantages": ["Fast training", "Probabilistic output", "Interpretable"],
                        "considerations": ["Linear boundaries", "Feature scaling needed"],
                        "ovr_benefits": "Clear coefficient interpretation per class"
                    },
                    "linear_svm": {
                        "advantages": ["Good generalization", "Sparse solutions", "Kernel trick"],
                        "considerations": ["No direct probabilities", "Hyperparameter sensitive"],
                        "ovr_benefits": "Maximum margin principle applied per class"
                    }
                },
                "nonlinear_classifiers": {
                    "random_forest": {
                        "advantages": ["Robust", "Feature importance", "No scaling needed"],
                        "considerations": ["Can overfit", "Less interpretable"],
                        "ovr_benefits": "Feature importance analysis per class"
                    },
                    "svm_rbf": {
                        "advantages": ["Complex boundaries", "Kernel flexibility"],
                        "considerations": ["Slow on large data", "Many hyperparameters"],
                        "ovr_benefits": "Flexible decision boundaries per class"
                    },
                    "neural_networks": {
                        "advantages": ["Universal approximation", "Feature learning"],
                        "considerations": ["Black box", "Requires tuning"],
                        "ovr_benefits": "Rich representation learning per class"
                    }
                }
            },
            "performance_characteristics": {
                "computational_complexity": {
                    "training": "O(K × C_base) where C_base is base classifier complexity",
                    "prediction": "O(K × P_base) where P_base is base classifier prediction time",
                    "memory": "O(K × M_base) where M_base is base classifier memory",
                    "parallelization": "Perfect parallelization possible (K independent problems)"
                },
                "accuracy_considerations": {
                    "optimal_conditions": "When classes are well-separated and balanced",
                    "challenging_scenarios": "Highly imbalanced data, overlapping classes",
                    "performance_factors": ["Base classifier quality", "Class distribution", "Feature quality"],
                    "improvement_strategies": ["Ensemble of OvR", "Calibration", "Feature engineering"]
                },
                "scalability_analysis": {
                    "number_of_classes": "Linear scaling O(K)",
                    "dataset_size": "Depends on base classifier scaling",
                    "feature_dimensionality": "Depends on base classifier",
                    "parallel_potential": "Excellent - embarrassingly parallel"
                }
            },
            "strengths": [
                "Universal: works with any binary classifier",
                "Interpretable: each class vs. rest is clear",
                "Parallel: independent binary problems",
                "Efficient: only K classifiers needed",
                "Flexible: can use different algorithms per class",
                "Debuggable: can analyze each binary problem separately",
                "Well-established: theoretical foundation solid",
                "Per-class analysis: detailed insights per class",
                "Probability estimates: can provide class probabilities",
                "Handles class imbalance: can apply strategies per binary problem",
                "Feature importance: if base supports it, available per class",
                "Calibration friendly: probabilities can be calibrated"
            ],
            "weaknesses": [
                "Class imbalance: each binary problem naturally imbalanced",
                "Inconsistent predictions: possible in ambiguous regions",
                "Probability calibration: may need adjustment for valid probabilities",
                "Feature scaling: may need scaling for distance-based base classifiers",
                "Overfitting risk: K times the risk of base classifier overfitting",
                "Void regions: possible regions with no positive predictions",
                "Computational overhead: K times base classifier cost",
                "Memory usage: stores K complete models",
                "Decision confidence: harder to interpret than single classifier",
                "Class correlation: ignores relationships between classes",
                "Threshold sensitivity: performance depends on decision thresholds",
                "Base classifier limitations: inherits all base classifier weaknesses"
            ],
            "ideal_use_cases": [
                "Multi-class problems with clear class separation",
                "When interpretability per class is important",
                "Large number of classes (scales linearly)",
                "When using well-tuned binary classifiers",
                "Parallel computing environments available",
                "Per-class performance analysis needed",
                "When binary classifier expertise exists",
                "Text classification with multiple categories",
                "Image classification with distinct object types",
                "Medical diagnosis with multiple conditions",
                "Fraud detection with multiple fraud types",
                "Natural language processing with multiple intents"
            ],
            "comparison_with_other_strategies": {
                "vs_one_vs_one": {
                    "ovr": "K binary classifiers, uses all data",
                    "ovo": "K(K-1)/2 classifiers, uses pairwise data",
                    "winner": "OvR for large K, OvO for few classes with high overlap"
                },
                "vs_multinomial": {
                    "ovr": "Multiple binary problems, any base classifier",
                    "multinomial": "Single multi-class problem, specific algorithms only",
                    "winner": "OvR for flexibility, Multinomial for unified approach"
                },
                "vs_error_correcting": {
                    "ovr": "Simple, interpretable, efficient",
                    "error_correcting": "Robust to individual classifier errors",
                    "winner": "OvR for simplicity, ECOC for robustness"
                }
            },
            "hyperparameter_effects": {
                "base_estimator_choice": {
                    "impact": "Determines fundamental behavior of each binary classifier",
                    "considerations": ["Problem complexity", "Interpretability needs", "Computational budget"],
                    "recommendations": "Start with Logistic Regression for baseline"
                },
                "class_weight_strategy": {
                    "balanced": "Automatically balance each binary problem",
                    "manual": "Specify weights for positive class in each binary problem",
                    "none": "No balancing (may favor negative class)",
                    "recommendation": "Use balanced for imbalanced datasets"
                },
                "probability_calibration": {
                    "enabled": "Better calibrated multi-class probabilities",
                    "disabled": "Raw binary classifier outputs",
                    "methods": ["Platt scaling", "Isotonic regression"],
                    "recommendation": "Enable for probability-sensitive applications"
                },
                "parallel_jobs": {
                    "effect": "Number of binary classifiers trained simultaneously",
                    "optimal": "Number of CPU cores or classes, whichever is smaller",
                    "trade_off": "Speed vs. memory usage"
                }
            },
            "available_base_estimators": {
                "logistic_regression": {
                    "type": "Linear probabilistic",
                    "strengths": ["Fast", "Interpretable", "Good probabilities"],
                    "weaknesses": ["Linear assumptions"],
                    "best_for": "Baseline, interpretable results, well-separated classes",
                    "ovr_synergy": "High - coefficients clearly interpretable per class"
                },
                "random_forest": {
                    "type": "Tree ensemble",
                    "strengths": ["Robust", "Feature importance", "Nonlinear"],
                    "weaknesses": ["Can overfit", "Less interpretable"],
                    "best_for": "Complex patterns, robust predictions, feature analysis",
                    "ovr_synergy": "High - feature importance analysis per class"
                },
                "svm": {
                    "type": "Maximum margin",
                    "strengths": ["Good generalization", "Kernel flexibility"],
                    "weaknesses": ["Slow on large data", "Many hyperparameters"],
                    "best_for": "Complex decision boundaries, high-dimensional data",
                    "ovr_synergy": "Medium - powerful but requires careful tuning"
                },
                "gradient_boosting": {
                    "type": "Sequential ensemble",
                    "strengths": ["High accuracy", "Handles complex patterns"],
                    "weaknesses": ["Can overfit", "Slower training"],
                    "best_for": "High accuracy requirements, complex datasets",
                    "ovr_synergy": "High - excellent performance per binary problem"
                },
                "neural_network": {
                    "type": "Multi-layer perceptron",
                    "strengths": ["Universal approximation", "Feature learning"],
                    "weaknesses": ["Black box", "Requires tuning"],
                    "best_for": "Complex patterns, large datasets, representation learning",
                    "ovr_synergy": "Medium - powerful but less interpretable"
                },
                "decision_tree": {
                    "type": "Tree-based",
                    "strengths": ["Highly interpretable", "No scaling needed"],
                    "weaknesses": ["Can overfit", "Unstable"],
                    "best_for": "Interpretable rules, categorical features",
                    "ovr_synergy": "High - clear decision rules per class"
                }
            }
        }
    
    def _create_base_estimator(self) -> BaseEstimator:
        """Create base estimator instance based on type"""
        
        if self.base_estimator == 'logistic_regression':
            return LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                solver=self.lr_solver,
                penalty=self.lr_penalty,
                random_state=self.random_state,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        elif self.base_estimator == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                criterion=self.rf_criterion,
                random_state=self.random_state,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        elif self.base_estimator == 'svm':
            return SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                probability=self.svm_probability,
                gamma=self.svm_gamma,
                random_state=self.random_state,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        elif self.base_estimator == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                min_samples_leaf=self.dt_min_samples_leaf,
                criterion=self.dt_criterion,
                random_state=self.random_state,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        elif self.base_estimator == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                metric=self.knn_metric
            )
        
        elif self.base_estimator == 'naive_bayes':
            return GaussianNB()
        
        elif self.base_estimator == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.gb_n_estimators,
                learning_rate=self.gb_learning_rate,
                max_depth=self.gb_max_depth,
                subsample=self.gb_subsample,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'extra_trees' and EXTENDED_ALGORITHMS:
            return ExtraTreesClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        elif self.base_estimator == 'adaboost' and EXTENDED_ALGORITHMS:
            return AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'mlp' and EXTENDED_ALGORITHMS:
            return MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                activation=self.mlp_activation,
                max_iter=self.mlp_max_iter,
                alpha=self.mlp_alpha,
                random_state=self.random_state
            )
        
        elif self.base_estimator == 'xgboost' and EXTENDED_ALGORITHMS:
            return XGBClassifier(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                subsample=self.xgb_subsample,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            )
        
        elif self.base_estimator == 'lightgbm' and EXTENDED_ALGORITHMS:
            return LGBMClassifier(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                subsample=self.xgb_subsample,
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced' if self.class_weight_strategy == 'balanced' else None
            )
        
        else:
            # Default fallback
            warnings.warn(f"Unknown base estimator '{self.base_estimator}'. Using Logistic Regression as fallback.")
            return LogisticRegression(random_state=self.random_state)
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the One-vs-Rest Classifier
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, default=None
            Sample weights (not supported by all base estimators)
            
        Returns:
        --------
        self : object
        """
        # 🎯 STORE FEATURE NAMES BEFORE VALIDATION!
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Encode labels if they're not numeric
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Check if we actually need OvR (more than 2 classes)
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes for classification")
        elif self.n_classes_ == 2:
            warnings.warn("Only 2 classes detected. OvR will work but consider using the base classifier directly.")
        
        # Feature scaling if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        if self.verbose > 0:
            print(f"Training One-vs-Rest Classifier with {self.n_classes_} classes...")
            print(f"Base estimator: {self.base_estimator}")
            print(f"Classes: {self.classes_}")
            print(f"Class distribution: {dict(zip(self.classes_, np.bincount(y_encoded)))}")
        
        # Store class distributions for analysis
        self.class_distributions_ = {
            "class_counts": dict(zip(self.classes_, np.bincount(y_encoded))),
            "class_proportions": dict(zip(self.classes_, np.bincount(y_encoded) / len(y_encoded))),
            "total_samples": len(y_encoded)
        }
        
        # Create base estimator instance
        self.base_estimator_instance_ = self._create_base_estimator()
        
        # Create and configure One-vs-Rest classifier
        self.ovr_classifier_ = OneVsRestClassifier(
            estimator=self.base_estimator_instance_,
            n_jobs=self.n_jobs
        )
        
        # Train the OvR ensemble
        with warnings.catch_warnings():
            if self.verbose == 0:
                warnings.filterwarnings("ignore", category=UserWarning)
            
            if sample_weight is not None:
                # Check if base estimator supports sample weights
                if hasattr(self.base_estimator_instance_, 'fit') and 'sample_weight' in self.base_estimator_instance_.fit.__code__.co_varnames:
                    self.ovr_classifier_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
                else:
                    warnings.warn("Base estimator doesn't support sample weights. Ignoring sample_weight parameter.")
                    self.ovr_classifier_.fit(X_scaled, y_encoded)
            else:
                self.ovr_classifier_.fit(X_scaled, y_encoded)
        
        # Store binary classifiers for analysis
        self.binary_classifiers_ = {
            f"class_{self.classes_[i]}_vs_rest": self.ovr_classifier_.estimators_[i]
            for i in range(self.n_classes_)
        }
        
        # Evaluate per-class performance if requested
        if self.estimate_class_performance:
            self._evaluate_class_performance(X_scaled, y_encoded)
        
        # Analyze the OvR strategy
        self._analyze_ovr_strategy(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _evaluate_class_performance(self, X, y):
        """Evaluate per-class performance using cross-validation"""
        try:
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            class_scores = {}
            
            for class_idx, class_name in enumerate(self.classes_):
                # Create binary target for this class
                y_binary = (y == class_idx).astype(int)
                
                # Create base estimator for this class
                estimator = self._create_base_estimator()
                
                try:
                    # Cross-validation scores for this binary problem
                    cv_scores = cross_val_score(estimator, X, y_binary, cv=cv_strategy, scoring='f1', n_jobs=self.n_jobs)
                    
                    class_scores[str(class_name)] = {
                        'mean_f1_score': float(np.mean(cv_scores)),
                        'std_f1_score': float(np.std(cv_scores)),
                        'cv_scores': cv_scores.tolist(),
                        'class_size': int(np.sum(y_binary)),
                        'class_proportion': float(np.mean(y_binary))
                    }
                    
                    if self.verbose > 0:
                        print(f"Class {class_name} F1 Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Could not evaluate class {class_name}: {str(e)}")
                    class_scores[str(class_name)] = {
                        'error': str(e),
                        'class_size': int(np.sum(y_binary)),
                        'class_proportion': float(np.mean(y_binary))
                    }
            
            self.class_performance_ = class_scores
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not evaluate class performance: {str(e)}")
            self.class_performance_ = None
    
    def _analyze_ovr_strategy(self, X, y):
        """Analyze the One-vs-Rest strategy characteristics"""
        analysis = {
            "strategy_type": "One-vs-Rest (OvR)",
            "base_estimator": self.base_estimator,
            "n_classes": self.n_classes_,
            "n_binary_classifiers": self.n_classes_,
            "classes": self.classes_.tolist(),
            "feature_scaling": self.scaler_ is not None,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "class_weight_strategy": self.class_weight_strategy,
            "probability_calibration": self.probability_calibration
        }
        
        # Add class distribution analysis
        analysis["class_distribution"] = self.class_distributions_
        
        # Add per-class performance if available
        if self.class_performance_:
            analysis["class_performance"] = self.class_performance_
        
        # Analyze class imbalance in binary problems
        binary_imbalance = {}
        for class_idx, class_name in enumerate(self.classes_):
            positive_samples = np.sum(y == class_idx)
            negative_samples = len(y) - positive_samples
            imbalance_ratio = negative_samples / positive_samples if positive_samples > 0 else float('inf')
            
            binary_imbalance[str(class_name)] = {
                "positive_samples": int(positive_samples),
                "negative_samples": int(negative_samples),
                "imbalance_ratio": float(imbalance_ratio),
                "positive_proportion": float(positive_samples / len(y))
            }
        
        analysis["binary_imbalance"] = binary_imbalance
        
        # Estimate computational complexity
        analysis["computational_complexity"] = {
            "training_complexity": f"O({self.n_classes_} × Base_Training_Complexity)",
            "prediction_complexity": f"O({self.n_classes_} × Base_Prediction_Complexity)",
            "memory_complexity": f"O({self.n_classes_} × Base_Memory_Complexity)",
            "parallel_potential": "Perfect - all binary problems independent",
            "scalability": f"Linear in number of classes ({self.n_classes_})"
        }
        
        # Analyze decision regions
        analysis["decision_analysis"] = {
            "decision_rule": "argmax over all binary classifier outputs",
            "tie_breaking": "Highest decision score wins",
            "ambiguous_regions": "Regions where multiple classifiers predict positive",
            "void_regions": "Regions where all classifiers predict negative",
            "confidence_interpretation": "Margin between highest and second-highest scores"
        }
        
        # Feature importance aggregation strategy
        if hasattr(self.base_estimator_instance_, 'feature_importances_') or hasattr(self.base_estimator_instance_, 'coef_'):
            analysis["feature_importance_strategy"] = {
                "per_class": "Each binary classifier provides class-specific importance",
                "aggregation_options": ["Mean", "Max", "Weighted by class size"],
                "interpretation": "Feature importance for distinguishing each class from others"
            }
        
        self.ovr_analysis_ = analysis
    
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
        
        # Get predictions from OvR classifier
        y_pred_encoded = self.ovr_classifier_.predict(X_scaled)
        
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
        
        # Get probabilities from OvR classifier
        try:
            probabilities = self.ovr_classifier_.predict_proba(X_scaled)
            
            # Apply calibration if requested
            if self.probability_calibration:
                # Normalize probabilities to ensure they sum to 1
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            
            return probabilities
            
        except Exception as e:
            warnings.warn(f"Probability prediction failed: {str(e)}. Using decision function.")
            
            # Fallback: use decision function and convert to probabilities
            try:
                decision_scores = self.ovr_classifier_.decision_function(X_scaled)
                # Convert decision scores to probabilities using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                return probabilities
            except:
                # Final fallback: hard predictions to probabilities
                y_pred = self.ovr_classifier_.predict(X_scaled)
                n_samples = len(y_pred)
                probabilities = np.zeros((n_samples, self.n_classes_))
                for i, pred in enumerate(y_pred):
                    probabilities[i, pred] = 1.0
                return probabilities
    
    def decision_function(self, X):
        """
        Compute decision function for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        decision_scores : array, shape (n_samples, n_classes)
            Decision function scores per class
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        return self.ovr_classifier_.decision_function(X_scaled)
    
    def get_class_predictions(self, X):
        """
        Get predictions from individual binary classifiers
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : dict
            Dictionary with binary classifier predictions for each class
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        predictions = {}
        
        try:
            # Get predictions from each binary classifier
            for class_idx, class_name in enumerate(self.classes_):
                binary_classifier = self.ovr_classifier_.estimators_[class_idx]
                
                # Get binary predictions (class vs. rest)
                binary_pred = binary_classifier.predict(X_scaled)
                predictions[f"class_{class_name}_vs_rest"] = binary_pred.astype(bool)
                
                # Get probability/decision scores if available
                if hasattr(binary_classifier, 'predict_proba'):
                    try:
                        binary_proba = binary_classifier.predict_proba(X_scaled)
                        # Take probability of positive class (class vs. rest)
                        predictions[f"class_{class_name}_probability"] = binary_proba[:, 1]
                    except:
                        pass
                
                elif hasattr(binary_classifier, 'decision_function'):
                    try:
                        binary_scores = binary_classifier.decision_function(X_scaled)
                        predictions[f"class_{class_name}_decision_score"] = binary_scores
                    except:
                        pass
            
            return predictions
            
        except Exception as e:
            warnings.warn(f"Could not get class predictions: {str(e)}")
            return {}
    
    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance from binary classifiers
        
        Returns:
        --------
        importance : dict or None
            Dictionary with feature importance per class and aggregated
        """
        if not self.is_fitted_:
            return None
        
        try:
            class_importance = {}
            all_importances = []
            
            for class_idx, class_name in enumerate(self.classes_):
                binary_classifier = self.ovr_classifier_.estimators_[class_idx]
                
                # Get feature importance from binary classifier
                importance = None
                if hasattr(binary_classifier, 'feature_importances_'):
                    importance = binary_classifier.feature_importances_
                elif hasattr(binary_classifier, 'coef_'):
                    importance = np.abs(binary_classifier.coef_[0])
                
                if importance is not None:
                    class_importance[f"class_{class_name}_vs_rest"] = importance.copy()
                    all_importances.append(importance)
            
            if all_importances:
                # Aggregate feature importance across classes
                all_importances = np.array(all_importances)
                
                aggregated = {
                    "mean_importance": np.mean(all_importances, axis=0),
                    "max_importance": np.max(all_importances, axis=0),
                    "std_importance": np.std(all_importances, axis=0),
                    "sum_importance": np.sum(all_importances, axis=0)
                }
                
                return {
                    "per_class": class_importance,
                    "aggregated": aggregated,
                    "feature_names": self.feature_names_
                }
            
            return None
            
        except Exception as e:
            warnings.warn(f"Could not extract feature importance: {str(e)}")
            return None
    
    def get_ovr_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the One-vs-Rest strategy
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive OvR analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "strategy_summary": {
                "algorithm": "One-vs-Rest Classifier",
                "base_estimator": self.base_estimator,
                "n_classes": self.n_classes_,
                "n_binary_classifiers": self.n_classes_,
                "classes": self.classes_.tolist(),
                "feature_scaling": self.scaler_ is not None,
                "n_features": self.n_features_in_,
                "class_weight_strategy": self.class_weight_strategy
            }
        }
        
        # Add OvR analysis
        if self.ovr_analysis_:
            analysis["ovr_characteristics"] = self.ovr_analysis_
        
        # Add class performance
        if self.class_performance_:
            analysis["class_performance"] = self.class_performance_
        
        # Add class distributions
        if self.class_distributions_:
            analysis["class_distributions"] = self.class_distributions_
        
        return analysis
    
    def plot_ovr_analysis(self, figsize=(16, 12)):
        """
        Create comprehensive One-vs-Rest analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(16, 12)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            OvR analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Class Distribution and Imbalance
        if self.class_distributions_:
            class_counts = list(self.class_distributions_['class_counts'].values())
            class_names = [str(name) for name in self.class_distributions_['class_counts'].keys()]
            
            bars = ax1.bar(range(len(class_names)), class_counts, alpha=0.7, 
                          color='skyblue', edgecolor='navy')
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45)
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Class Distribution in Original Dataset')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, class_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                        str(count), ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'Class distribution\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Class Distribution')
        
        # 2. Binary Problem Imbalance
        if self.ovr_analysis_ and 'binary_imbalance' in self.ovr_analysis_:
            imbalance_data = self.ovr_analysis_['binary_imbalance']
            classes = list(imbalance_data.keys())
            imbalance_ratios = [imbalance_data[cls]['imbalance_ratio'] for cls in classes]
            
            # Cap extreme ratios for visualization
            capped_ratios = [min(ratio, 50) for ratio in imbalance_ratios]
            
            bars = ax2.barh(range(len(classes)), capped_ratios, alpha=0.7, 
                           color='lightcoral', edgecolor='darkred')
            ax2.set_yticks(range(len(classes)))
            ax2.set_yticklabels([f"Class {cls}" for cls in classes])
            ax2.set_xlabel('Imbalance Ratio (Negative:Positive)')
            ax2.set_title('Binary Problem Imbalance (Class vs. Rest)')
            ax2.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Balanced')
            ax2.legend()
            
            # Add ratio labels
            for i, (bar, ratio) in enumerate(zip(bars, imbalance_ratios)):
                label = f'{ratio:.1f}' if ratio < 100 else f'{ratio:.0f}'
                ax2.text(bar.get_width() + max(capped_ratios)*0.01, bar.get_y() + bar.get_height()/2,
                        label, ha='left', va='center')
        else:
            ax2.text(0.5, 0.5, 'Binary imbalance\nanalysis\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Binary Problem Imbalance')
        
        # 3. Per-Class Performance
        if self.class_performance_:
            classes = []
            f1_scores = []
            f1_stds = []
            
            for class_name, perf in self.class_performance_.items():
                if 'mean_f1_score' in perf:
                    classes.append(str(class_name))
                    f1_scores.append(perf['mean_f1_score'])
                    f1_stds.append(perf['std_f1_score'])
            
            if classes:
                bars = ax3.bar(range(len(classes)), f1_scores, 
                              yerr=f1_stds, capsize=5, alpha=0.7, 
                              color='lightgreen', edgecolor='darkgreen')
                ax3.set_xticks(range(len(classes)))
                ax3.set_xticklabels([f"Class {cls}" for cls in classes], rotation=45)
                ax3.set_ylabel('F1 Score')
                ax3.set_title('Per-Class Binary Classification Performance')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 1)
                
                # Add score labels
                for i, (bar, score) in enumerate(zip(bars, f1_scores)):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{score:.3f}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No valid\nperformance data', 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Class performance\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Per-Class Performance')
        
        # 4. OvR Configuration Summary
        ax4.axis('tight')
        ax4.axis('off')
        table_data = []
        table_data.append(['Strategy', 'One-vs-Rest'])
        table_data.append(['Base Estimator', str(self.base_estimator).replace('_', ' ').title()])
        table_data.append(['Number of Classes', str(self.n_classes_)])
        table_data.append(['Binary Classifiers', str(self.n_classes_)])
        table_data.append(['Feature Scaling', 'Yes' if self.scaler_ is not None else 'No'])
        table_data.append(['Class Weighting', str(self.class_weight_strategy)])
        table_data.append(['Parallel Training', 'Yes' if self.n_jobs != 1 else 'No'])
        
        if self.ovr_analysis_:
            table_data.append(['Total Samples', str(self.ovr_analysis_['n_samples'])])
            table_data.append(['Features', str(self.ovr_analysis_['n_features'])])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Property', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('One-vs-Rest Configuration')
        
        plt.tight_layout()
        return fig
    
    def plot_class_predictions_heatmap(self, X_test, y_test, figsize=(12, 8)):
        """
        Visualize binary classifier predictions as a heatmap
        
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
            Class predictions heatmap
        """
        if not self.is_fitted_:
            return None
        
        # Get binary classifier predictions
        class_predictions = self.get_class_predictions(X_test)
        
        if not class_predictions:
            return None
        
        # Filter for binary predictions only
        binary_preds = {k: v for k, v in class_predictions.items() 
                       if k.endswith('_vs_rest')}
        
        if not binary_preds:
            return None
        
        # Create prediction matrix
        class_names = [k.replace('_vs_rest', '').replace('class_', '') for k in binary_preds.keys()]
        pred_matrix = np.array([binary_preds[k].astype(int) for k in binary_preds.keys()])
        
        # Limit to first 100 samples for visualization
        n_samples_show = min(100, X_test.shape[0])
        pred_matrix_show = pred_matrix[:, :n_samples_show]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Binary predictions heatmap
        sns.heatmap(pred_matrix_show, 
                   yticklabels=class_names,
                   xticklabels=False,
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Prediction (0: Negative, 1: Positive)'},
                   ax=ax1)
        ax1.set_title(f'Binary Classifier Predictions (First {n_samples_show} samples)')
        ax1.set_ylabel('Class vs. Rest')
        
        # 2. Prediction summary statistics
        pred_summary = {
            'Class': class_names,
            'Positive Predictions': [np.sum(pred_matrix[i]) for i in range(len(class_names))],
            'Positive Rate': [np.mean(pred_matrix[i]) for i in range(len(class_names))]
        }
        
        summary_df = pd.DataFrame(pred_summary)
        
        bars = ax2.bar(range(len(class_names)), summary_df['Positive Rate'], 
                      alpha=0.7, color='steelblue', edgecolor='navy')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.set_ylabel('Positive Prediction Rate')
        ax2.set_title('Positive Prediction Rate per Binary Classifier')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, summary_df['Positive Rate']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🎯 One-vs-Rest Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4 = st.sidebar.tabs(["Base Est.", "Advanced", "Strategy", "Info"])
        
        with tab1:
            st.markdown("**Base Estimator Selection**")
            
            # Base estimator selection
            available_estimators = ['logistic_regression', 'random_forest', 'svm', 'decision_tree', 
                                  'knn', 'naive_bayes', 'gradient_boosting']
            
            if EXTENDED_ALGORITHMS:
                available_estimators.extend(['extra_trees', 'adaboost', 'mlp', 'xgboost', 'lightgbm'])
            
            base_estimator = st.selectbox(
                "Base Binary Classifier:",
                options=available_estimators,
                index=available_estimators.index(self.base_estimator) if self.base_estimator in available_estimators else 0,
                help="Binary classifier used for each 'class vs. rest' problem",
                key=f"{key_prefix}_base_estimator"
            )
            
            # Base estimator info
            if base_estimator == 'logistic_regression':
                st.info("📈 Logistic Regression: Fast, interpretable, good probabilities")
            elif base_estimator == 'random_forest':
                st.info("🌳 Random Forest: Robust, handles nonlinearity, feature importance")
            elif base_estimator == 'svm':
                st.info("🔍 SVM: Good generalization, flexible kernels")
            elif base_estimator == 'decision_tree':
                st.info("🌿 Decision Tree: Highly interpretable, no scaling needed")
            elif base_estimator == 'gradient_boosting':
                st.info("🚀 Gradient Boosting: High performance, complex patterns")
            
            st.markdown("**Base Estimator Hyperparameters**")
            
            # Show relevant hyperparameters based on selected estimator
            if base_estimator == 'logistic_regression':
                lr_C = st.number_input(
                    "LR - Regularization (C):",
                    value=float(self.lr_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_lr_C"
                )
                
                lr_max_iter = st.number_input(
                    "LR - Max Iterations:",
                    value=int(self.lr_max_iter),
                    min_value=100,
                    max_value=5000,
                    step=100,
# Continue from line 1366 where the code breaks:

                    key=f"{key_prefix}_lr_max_iter"
                )
                
                lr_solver = st.selectbox(
                    "LR - Solver:",
                    options=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                    index=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'].index(self.lr_solver),
                    key=f"{key_prefix}_lr_solver"
                )
                
                lr_penalty = st.selectbox(
                    "LR - Penalty:",
                    options=['l1', 'l2', 'elasticnet', 'none'],
                    index=['l1', 'l2', 'elasticnet', 'none'].index(self.lr_penalty),
                    key=f"{key_prefix}_lr_penalty"
                )
            else:
                lr_C = self.lr_C
                lr_max_iter = self.lr_max_iter
                lr_solver = self.lr_solver
                lr_penalty = self.lr_penalty
            
            # Random Forest parameters
            if base_estimator == 'random_forest':
                rf_n_estimators = st.slider(
                    "RF - Number of Trees:",
                    min_value=10,
                    max_value=500,
                    value=int(self.rf_n_estimators),
                    step=10,
                    key=f"{key_prefix}_rf_n_estimators"
                )
                
                rf_max_depth_option = st.selectbox(
                    "RF - Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.rf_max_depth is None else 1,
                    key=f"{key_prefix}_rf_max_depth_option"
                )
                
                if rf_max_depth_option == 'Custom':
                    rf_max_depth = st.slider(
                        "RF - Custom Max Depth:",
                        min_value=1,
                        max_value=50,
                        value=10 if self.rf_max_depth is None else int(self.rf_max_depth),
                        key=f"{key_prefix}_rf_max_depth"
                    )
                else:
                    rf_max_depth = None
                
                rf_min_samples_split = st.slider(
                    "RF - Min Samples Split:",
                    min_value=2,
                    max_value=20,
                    value=int(self.rf_min_samples_split),
                    key=f"{key_prefix}_rf_min_samples_split"
                )
                
                rf_min_samples_leaf = st.slider(
                    "RF - Min Samples Leaf:",
                    min_value=1,
                    max_value=10,
                    value=int(self.rf_min_samples_leaf),
                    key=f"{key_prefix}_rf_min_samples_leaf"
                )
                
                rf_criterion = st.selectbox(
                    "RF - Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.rf_criterion),
                    key=f"{key_prefix}_rf_criterion"
                )
            else:
                rf_n_estimators = self.rf_n_estimators
                rf_max_depth = self.rf_max_depth
                rf_min_samples_split = self.rf_min_samples_split
                rf_min_samples_leaf = self.rf_min_samples_leaf
                rf_criterion = self.rf_criterion
            
            # SVM parameters
            if base_estimator == 'svm':
                svm_C = st.number_input(
                    "SVM - C Parameter:",
                    value=float(self.svm_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_svm_C"
                )
                
                svm_kernel = st.selectbox(
                    "SVM - Kernel:",
                    options=['rbf', 'linear', 'poly', 'sigmoid'],
                    index=['rbf', 'linear', 'poly', 'sigmoid'].index(self.svm_kernel),
                    key=f"{key_prefix}_svm_kernel"
                )
                
                svm_probability = st.checkbox(
                    "SVM - Enable Probabilities",
                    value=self.svm_probability,
                    help="Required for probability estimation",
                    key=f"{key_prefix}_svm_probability"
                )
                
                svm_gamma = st.selectbox(
                    "SVM - Gamma:",
                    options=['scale', 'auto', 'custom'],
                    index=['scale', 'auto'].index(self.svm_gamma) if self.svm_gamma in ['scale', 'auto'] else 2,
                    key=f"{key_prefix}_svm_gamma_option"
                )
            else:
                svm_C = self.svm_C
                svm_kernel = self.svm_kernel
                svm_probability = self.svm_probability
                svm_gamma = self.svm_gamma
            
            # Decision Tree parameters
            if base_estimator == 'decision_tree':
                dt_max_depth_option = st.selectbox(
                    "DT - Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.dt_max_depth is None else 1,
                    key=f"{key_prefix}_dt_max_depth_option"
                )
                
                if dt_max_depth_option == 'Custom':
                    dt_max_depth = st.slider(
                        "DT - Custom Max Depth:",
                        min_value=1,
                        max_value=30,
                        value=10 if self.dt_max_depth is None else int(self.dt_max_depth),
                        key=f"{key_prefix}_dt_max_depth"
                    )
                else:
                    dt_max_depth = None
                
                dt_min_samples_split = st.slider(
                    "DT - Min Samples Split:",
                    min_value=2,
                    max_value=20,
                    value=int(self.dt_min_samples_split),
                    key=f"{key_prefix}_dt_min_samples_split"
                )
                
                dt_min_samples_leaf = st.slider(
                    "DT - Min Samples Leaf:",
                    min_value=1,
                    max_value=10,
                    value=int(self.dt_min_samples_leaf),
                    key=f"{key_prefix}_dt_min_samples_leaf"
                )
                
                dt_criterion = st.selectbox(
                    "DT - Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.dt_criterion),
                    key=f"{key_prefix}_dt_criterion"
                )
            else:
                dt_max_depth = self.dt_max_depth
                dt_min_samples_split = self.dt_min_samples_split
                dt_min_samples_leaf = self.dt_min_samples_leaf
                dt_criterion = self.dt_criterion
            
            # KNN parameters
            if base_estimator == 'knn':
                knn_n_neighbors = st.slider(
                    "KNN - Number of Neighbors:",
                    min_value=1,
                    max_value=50,
                    value=int(self.knn_n_neighbors),
                    key=f"{key_prefix}_knn_n_neighbors"
                )
                
                knn_weights = st.selectbox(
                    "KNN - Weight Function:",
                    options=['uniform', 'distance'],
                    index=['uniform', 'distance'].index(self.knn_weights),
                    key=f"{key_prefix}_knn_weights"
                )
                
                knn_metric = st.selectbox(
                    "KNN - Distance Metric:",
                    options=['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                    index=['minkowski', 'euclidean', 'manhattan', 'chebyshev'].index(self.knn_metric),
                    key=f"{key_prefix}_knn_metric"
                )
            else:
                knn_n_neighbors = self.knn_n_neighbors
                knn_weights = self.knn_weights
                knn_metric = self.knn_metric
            
            # Gradient Boosting parameters
            if base_estimator == 'gradient_boosting':
                gb_n_estimators = st.slider(
                    "GB - Number of Estimators:",
                    min_value=10,
                    max_value=500,
                    value=int(self.gb_n_estimators),
                    step=10,
                    key=f"{key_prefix}_gb_n_estimators"
                )
                
                gb_learning_rate = st.number_input(
                    "GB - Learning Rate:",
                    value=float(self.gb_learning_rate),
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_gb_learning_rate"
                )
                
                gb_max_depth = st.slider(
                    "GB - Max Depth:",
                    min_value=1,
                    max_value=10,
                    value=int(self.gb_max_depth),
                    key=f"{key_prefix}_gb_max_depth"
                )
                
                gb_subsample = st.slider(
                    "GB - Subsample Ratio:",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.gb_subsample),
                    step=0.1,
                    key=f"{key_prefix}_gb_subsample"
                )
            else:
                gb_n_estimators = self.gb_n_estimators
                gb_learning_rate = self.gb_learning_rate
                gb_max_depth = self.gb_max_depth
                gb_subsample = self.gb_subsample
            
            # Neural Network parameters
            if base_estimator == 'mlp' and EXTENDED_ALGORITHMS:
                mlp_hidden_sizes = st.text_input(
                    "MLP - Hidden Layer Sizes:",
                    value=','.join(map(str, self.mlp_hidden_layer_sizes)),
                    help="Comma-separated layer sizes, e.g., '100,50'",
                    key=f"{key_prefix}_mlp_hidden_sizes"
                )
                
                try:
                    mlp_hidden_layer_sizes = tuple(map(int, mlp_hidden_sizes.split(',')))
                except:
                    mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                
                mlp_activation = st.selectbox(
                    "MLP - Activation:",
                    options=['relu', 'tanh', 'logistic'],
                    index=['relu', 'tanh', 'logistic'].index(self.mlp_activation),
                    key=f"{key_prefix}_mlp_activation"
                )
                
                mlp_max_iter = st.slider(
                    "MLP - Max Iterations:",
                    min_value=100,
                    max_value=2000,
                    value=int(self.mlp_max_iter),
                    step=100,
                    key=f"{key_prefix}_mlp_max_iter"
                )
                
                mlp_alpha = st.number_input(
                    "MLP - Alpha (L2 penalty):",
                    value=float(self.mlp_alpha),
                    min_value=1e-6,
                    max_value=1e-2,
                    step=1e-5,
                    format="%.2e",
                    key=f"{key_prefix}_mlp_alpha"
                )
            else:
                mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                mlp_activation = self.mlp_activation
                mlp_max_iter = self.mlp_max_iter
                mlp_alpha = self.mlp_alpha
            
            # XGBoost parameters
            if base_estimator == 'xgboost' and EXTENDED_ALGORITHMS:
                xgb_n_estimators = st.slider(
                    "XGB - Number of Estimators:",
                    min_value=10,
                    max_value=500,
                    value=int(self.xgb_n_estimators),
                    step=10,
                    key=f"{key_prefix}_xgb_n_estimators"
                )
                
                xgb_learning_rate = st.number_input(
                    "XGB - Learning Rate:",
                    value=float(self.xgb_learning_rate),
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_xgb_learning_rate"
                )
                
                xgb_max_depth = st.slider(
                    "XGB - Max Depth:",
                    min_value=1,
                    max_value=15,
                    value=int(self.xgb_max_depth),
                    key=f"{key_prefix}_xgb_max_depth"
                )
                
                xgb_subsample = st.slider(
                    "XGB - Subsample Ratio:",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.xgb_subsample),
                    step=0.1,
                    key=f"{key_prefix}_xgb_subsample"
                )
            else:
                xgb_n_estimators = self.xgb_n_estimators
                xgb_learning_rate = self.xgb_learning_rate
                xgb_max_depth = self.xgb_max_depth
                xgb_subsample = self.xgb_subsample
        
        with tab2:
            st.markdown("**Advanced Options**")
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Scale features for distance-based estimators (SVM, KNN, Neural Networks)",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            # Class weight strategy
            class_weight_strategy = st.selectbox(
                "Class Weight Strategy:",
                options=['auto', 'balanced', 'none'],
                index=['auto', 'balanced', 'none'].index(self.class_weight_strategy),
                help="Strategy for handling class imbalance in binary problems",
                key=f"{key_prefix}_class_weight_strategy"
            )
            
            if class_weight_strategy == 'balanced':
                st.info("✅ Each binary classifier will automatically balance positive/negative classes")
            elif class_weight_strategy == 'auto':
                st.info("🤖 Strategy chosen based on base estimator and data characteristics")
            else:
                st.warning("⚠️ No class balancing - may favor negative class in binary problems")
            
            # Probability calibration
            probability_calibration = st.checkbox(
                "Probability Calibration",
                value=self.probability_calibration,
                help="Calibrate probabilities to ensure they sum to 1",
                key=f"{key_prefix}_probability_calibration"
            )
            
            # Performance estimation
            estimate_class_performance = st.checkbox(
                "Estimate Class Performance",
                value=self.estimate_class_performance,
                help="Evaluate individual binary classifier performance using CV",
                key=f"{key_prefix}_estimate_class_performance"
            )
            
            # Parallel processing
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores for parallel training",
                key=f"{key_prefix}_n_jobs"
            )
            
            if n_jobs == -1:
                st.success("🚀 Perfect parallelization: all binary classifiers train independently")
            elif n_jobs and n_jobs > 1:
                st.info(f"🔄 Using {n_jobs} cores for parallel binary classifier training")
            else:
                st.info("🔄 Sequential training of binary classifiers")
            
            # Verbosity
            verbose = st.selectbox(
                "Verbosity Level:",
                options=[0, 1],
                index=self.verbose,
                help="Control training output verbosity",
                key=f"{key_prefix}_verbose"
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
        
        with tab3:
            st.markdown("**One-vs-Rest Strategy Analysis**")
            
            st.info("""
            **One-vs-Rest Approach:**
            • Creates K binary classifiers for K classes
            • Each classifier: "Class i vs. All Other Classes"
            • Independent training enables perfect parallelization
            • Final prediction: argmax(binary_classifier_scores)
            
            **Key Advantages:**
            • Universal: works with any binary classifier
            • Interpretable: each binary problem is clear
            • Parallel: embarrassingly parallel training
            • Efficient: only K classifiers (vs. K(K-1)/2 for One-vs-One)
            """)
            
            # Strategy comparison
            if st.button("📊 OvR vs Other Strategies", key=f"{key_prefix}_strategy_comparison"):
                st.markdown("""
                **One-vs-Rest vs One-vs-One:**
                - OvR: K binary classifiers, uses all training data
                - OvO: K(K-1)/2 classifiers, uses pairwise subsets
                - Winner: OvR for large K, OvO for few overlapping classes
                
                **One-vs-Rest vs Multinomial:**
                - OvR: Multiple independent binary problems
                - Multinomial: Single unified multi-class problem
                - Winner: OvR for flexibility, Multinomial for coherence
                
                **When to Use OvR:**
                - Large number of classes (K > 10)
                - Need per-class interpretability
                - Parallel computing available
                - Proven binary classifier available
                """)
            
            # Binary problem analysis
            if st.button("🔍 Binary Problem Analysis", key=f"{key_prefix}_binary_analysis"):
                st.markdown("""
                **Binary Problem Characteristics:**
                
                **Natural Class Imbalance:**
                • Each binary problem: 1 positive class vs. (K-1) negative classes
                • Imbalance ratio: (K-1):1 for balanced original dataset
                • Mitigation: Use class_weight='balanced'
                
                **Decision Regions:**
                • K hyperplanes, each separating one class from rest
                • Regions where multiple classifiers predict positive (ambiguous)
                • Regions where no classifier predicts positive (void regions)
                
                **Probability Interpretation:**
                • Each classifier outputs P(class_i | x)
                • Need normalization for valid multi-class probabilities
                • Calibration improves probability quality
                """)
            
            # Per-class optimization
            if st.button("⚙️ Per-Class Optimization", key=f"{key_prefix}_per_class_optimization"):
                st.markdown("""
                **Per-Class Binary Optimization:**
                
                **Independent Tuning:**
                • Each binary classifier can be tuned separately
                • Different hyperparameters per class possible
                • Class-specific feature engineering possible
                
                **Class-Specific Strategies:**
                • Rare classes: Use different resampling strategies
                • Large classes: Focus on precision/recall trade-offs
                • Noisy classes: Increase regularization
                
                **Performance Analysis:**
                • Monitor F1 score per binary problem
                • Identify problematic classes early
                • Apply targeted improvements per class
                """)
            
            # Computational analysis
            if st.button("💻 Computational Analysis", key=f"{key_prefix}_computational_analysis"):
                st.markdown("""
                **Computational Characteristics:**
                
                **Training Complexity:** O(K × Base_Training_Complexity)
                **Prediction Complexity:** O(K × Base_Prediction_Complexity)
                **Memory Usage:** O(K × Base_Memory_Usage)
                
                **Parallelization:**
                • Perfect embarrassingly parallel training
                • Each binary classifier is independent
                • Scale linearly with number of cores (up to K cores)
                
                **Scalability:**
                • Linear scaling with number of classes
                • Better than One-vs-One for large K
                • Memory scales linearly with K
                """)
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **One-vs-Rest Classifier** - Universal Multi-class Wrapper:
            • 🎯 Transforms any binary classifier into multi-class
            • 🔄 K independent binary problems for K classes
            • ⚡ Perfect parallelization potential
            • 📊 Clear interpretability per class
            • 🔍 Per-class analysis and optimization
            • 🌐 Works with any binary classifier
            
            **Mathematical Foundation:**
            • Binary Problem i: Class i vs. All Other Classes
            • Decision Rule: ŷ = argmax_i f_i(x)
            • Probability: P(y=i|x) = softmax(f_i(x))
            """)
            
            # Base estimator guide
            if st.button("🏗️ Base Estimator Guide", key=f"{key_prefix}_base_estimator_guide"):
                st.markdown("""
                **Choosing the Right Base Estimator:**
                
                **Logistic Regression:**
                - Best for: Interpretability, speed, probability estimation
                - OvR Benefit: Clear coefficient interpretation per class
                
                **Random Forest:**
                - Best for: Robustness, feature importance, nonlinear patterns
                - OvR Benefit: Feature importance analysis per class
                
                **SVM:**
                - Best for: High-dimensional data, complex boundaries
                - OvR Benefit: Maximum margin principle per class
                
                **Gradient Boosting:**
                - Best for: High accuracy, complex patterns
                - OvR Benefit: Superior performance per binary problem
                
                **Decision Tree:**
                - Best for: Rule extraction, categorical features
                - OvR Benefit: Clear decision rules per class
                """)
            
            # Implementation details
            if st.button("🔧 Implementation Details", key=f"{key_prefix}_implementation_details"):
                st.markdown("""
                **One-vs-Rest Implementation:**
                
                **Training Process:**
                1. For each class i (i = 1, 2, ..., K):
                   - Create binary dataset: class i → positive, others → negative
                   - Train binary classifier on this transformed dataset
                2. Store all K trained binary classifiers
                
                **Prediction Process:**
                1. For each sample x:
                   - Get decision scores from all K binary classifiers
                   - Predicted class = argmax(decision_scores)
                   - Probabilities = softmax(decision_scores)
                
                **Key Technical Details:**
                - Each binary classifier sees all training data
                - Natural class imbalance in each binary problem
                - Independent classifiers enable perfect parallelization
                - Memory usage scales linearly with number of classes
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **OvR Best Practices:**
                
                **Base Estimator Selection:**
                1. Choose estimator suitable for binary classification
                2. Consider interpretability requirements
                3. Account for computational constraints
                4. Ensure probability support if needed
                
                **Handling Class Imbalance:**
                1. Use class_weight='balanced' for each binary problem
                2. Consider resampling techniques per binary problem
                3. Monitor per-class performance metrics
                4. Adjust decision thresholds if needed
                
                **Performance Optimization:**
                1. Use parallel training (n_jobs=-1)
                2. Feature scaling for distance-based estimators
                3. Cross-validation for hyperparameter tuning
                4. Per-class performance analysis
                
                **Probability Calibration:**
                1. Enable calibration for probability-sensitive applications
                2. Consider Platt scaling or isotonic regression
                3. Validate probability quality on held-out data
                """)
        
        # Return all selected hyperparameters
        return {
            "base_estimator": base_estimator,
            "auto_scale_features": auto_scale_features,
            "class_weight_strategy": class_weight_strategy,
            "probability_calibration": probability_calibration,
            "estimate_class_performance": estimate_class_performance,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "random_state": random_state,
            
            # Base estimator hyperparameters
            "lr_C": lr_C,
            "lr_max_iter": lr_max_iter,
            "lr_solver": lr_solver,
            "lr_penalty": lr_penalty,
            "rf_n_estimators": rf_n_estimators,
            "rf_max_depth": rf_max_depth,
            "rf_min_samples_split": rf_min_samples_split,
            "rf_min_samples_leaf": rf_min_samples_leaf,
            "rf_criterion": rf_criterion,
            "svm_C": svm_C,
            "svm_kernel": svm_kernel,
            "svm_probability": svm_probability,
            "svm_gamma": svm_gamma,
            "dt_max_depth": dt_max_depth,
            "dt_min_samples_split": dt_min_samples_split,
            "dt_min_samples_leaf": dt_min_samples_leaf,
            "dt_criterion": dt_criterion,
            "knn_n_neighbors": knn_n_neighbors,
            "knn_weights": knn_weights,
            "knn_metric": knn_metric,
            "gb_n_estimators": gb_n_estimators,
            "gb_learning_rate": gb_learning_rate,
            "gb_max_depth": gb_max_depth,
            "gb_subsample": gb_subsample,
            "mlp_hidden_layer_sizes": mlp_hidden_layer_sizes,
            "mlp_activation": mlp_activation,
            "mlp_max_iter": mlp_max_iter,
            "mlp_alpha": mlp_alpha,
            "xgb_n_estimators": xgb_n_estimators,
            "xgb_learning_rate": xgb_learning_rate,
            "xgb_max_depth": xgb_max_depth,
            "xgb_subsample": xgb_subsample
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return OneVsRestClassifierPlugin(
            base_estimator=hyperparameters.get("base_estimator", self.base_estimator),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            class_weight_strategy=hyperparameters.get("class_weight_strategy", self.class_weight_strategy),
            probability_calibration=hyperparameters.get("probability_calibration", self.probability_calibration),
            estimate_class_performance=hyperparameters.get("estimate_class_performance", self.estimate_class_performance),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            verbose=hyperparameters.get("verbose", self.verbose),
            random_state=hyperparameters.get("random_state", self.random_state),
            
            # Base estimator hyperparameters
            lr_C=hyperparameters.get("lr_C", self.lr_C),
            lr_max_iter=hyperparameters.get("lr_max_iter", self.lr_max_iter),
            lr_solver=hyperparameters.get("lr_solver", self.lr_solver),
            lr_penalty=hyperparameters.get("lr_penalty", self.lr_penalty),
            rf_n_estimators=hyperparameters.get("rf_n_estimators", self.rf_n_estimators),
            rf_max_depth=hyperparameters.get("rf_max_depth", self.rf_max_depth),
            rf_min_samples_split=hyperparameters.get("rf_min_samples_split", self.rf_min_samples_split),
            rf_min_samples_leaf=hyperparameters.get("rf_min_samples_leaf", self.rf_min_samples_leaf),
            rf_criterion=hyperparameters.get("rf_criterion", self.rf_criterion),
            svm_C=hyperparameters.get("svm_C", self.svm_C),
            svm_kernel=hyperparameters.get("svm_kernel", self.svm_kernel),
            svm_probability=hyperparameters.get("svm_probability", self.svm_probability),
            svm_gamma=hyperparameters.get("svm_gamma", self.svm_gamma),
            dt_max_depth=hyperparameters.get("dt_max_depth", self.dt_max_depth),
            dt_min_samples_split=hyperparameters.get("dt_min_samples_split", self.dt_min_samples_split),
            dt_min_samples_leaf=hyperparameters.get("dt_min_samples_leaf", self.dt_min_samples_leaf),
            dt_criterion=hyperparameters.get("dt_criterion", self.dt_criterion),
            knn_n_neighbors=hyperparameters.get("knn_n_neighbors", self.knn_n_neighbors),
            knn_weights=hyperparameters.get("knn_weights", self.knn_weights),
            knn_metric=hyperparameters.get("knn_metric", self.knn_metric),
            gb_n_estimators=hyperparameters.get("gb_n_estimators", self.gb_n_estimators),
            gb_learning_rate=hyperparameters.get("gb_learning_rate", self.gb_learning_rate),
            gb_max_depth=hyperparameters.get("gb_max_depth", self.gb_max_depth),
            gb_subsample=hyperparameters.get("gb_subsample", self.gb_subsample),
            mlp_hidden_layer_sizes=hyperparameters.get("mlp_hidden_layer_sizes", self.mlp_hidden_layer_sizes),
            mlp_activation=hyperparameters.get("mlp_activation", self.mlp_activation),
            mlp_max_iter=hyperparameters.get("mlp_max_iter", self.mlp_max_iter),
            mlp_alpha=hyperparameters.get("mlp_alpha", self.mlp_alpha),
            xgb_n_estimators=hyperparameters.get("xgb_n_estimators", self.xgb_n_estimators),
            xgb_learning_rate=hyperparameters.get("xgb_learning_rate", self.xgb_learning_rate),
            xgb_max_depth=hyperparameters.get("xgb_max_depth", self.xgb_max_depth),
            xgb_subsample=hyperparameters.get("xgb_subsample", self.xgb_subsample)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for One-vs-Rest Classifier"""
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """Check if One-vs-Rest Classifier is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"One-vs-Rest requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            n_classes = len(unique_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Consider reducing the number of classes."
            
            # Check class distribution
            min_class_samples = min(np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes))
            if min_class_samples < 2:
                return False, "Each class needs at least 2 samples"
            
            # OvR specific advantages
            advantages = []
            if n_classes > 10:
                advantages.append(f"Large number of classes ({n_classes}) - OvR scales linearly")
            elif n_classes > 2:
                advantages.append(f"Multi-class problem ({n_classes} classes) - perfect for OvR")
            
            if X.shape[0] >= 1000:
                advantages.append("Large dataset - benefits from parallel binary classifier training")
            
            # Binary imbalance analysis
            max_imbalance = (n_classes - 1) / 1  # worst case imbalance in binary problems
            if max_imbalance > 5:
                considerations = f"High class imbalance in binary problems (up to {max_imbalance:.1f}:1) - use balanced class weights"
            else:
                considerations = "Moderate imbalance in binary problems - should train well"
            
            message = f"✅ Compatible with {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes"
            if advantages:
                message += f" | 🎯 OvR advantages: {'; '.join(advantages)}"
            message += f" | 💡 {considerations}"
            
            return True, message
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist(),
            "feature_names": self.feature_names_,
            "base_estimator": self.base_estimator,
            "n_binary_classifiers": self.n_classes_,
            "feature_scaling": self.scaler_ is not None,
            "class_weight_strategy": self.class_weight_strategy,
            "probability_calibration": self.probability_calibration,
            "parallel_training": self.n_jobs != 1,
            "strategy_type": "One-vs-Rest"
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "One-vs-Rest Classifier",
            "strategy_type": "Binary Wrapper Multi-class",
            "training_completed": True,
            "ovr_characteristics": {
                "binary_decomposition": True,
                "independent_training": True,
                "parallel_potential": True,
                "per_class_analysis": True,
                "universal_wrapper": True,
                "class_imbalance_handling": True
            },
            "strategy_configuration": {
                "base_estimator": self.base_estimator,
                "n_classes": self.n_classes_,
                "n_binary_classifiers": self.n_classes_,
                "feature_scaling": self.scaler_ is not None,
                "class_weight_strategy": self.class_weight_strategy,
                "probability_calibration": self.probability_calibration,
                "parallel_training": self.n_jobs != 1
            },
            "ovr_analysis": self.get_ovr_analysis(),
            "performance_considerations": {
                "training_time": f"O({self.n_classes_} × Base_Training_Time)",
                "prediction_time": f"O({self.n_classes_} × Base_Prediction_Time)",
                "memory_usage": f"Stores {self.n_classes_} complete binary classifiers",
                "parallelization": "Perfect - all binary problems independent",
                "scalability": f"Linear scaling with classes ({self.n_classes_})",
                "interpretability": "High - each binary problem separately interpretable"
            },
            "ovr_theory": {
                "approach": "Decomposes K-class problem into K binary problems",
                "binary_problems": "Each: Class i vs. All Other Classes",
                "decision_rule": "argmax over all binary classifier outputs",
                "probability_estimation": "Softmax normalization of binary outputs",
                "advantages": "Universal, interpretable, parallel, efficient"
            }
        }
        
        # Add class performance if available
        if self.class_performance_:
            info["class_performance"] = self.class_performance_
        
        # Add class distributions
        if self.class_distributions_:
            info["class_distributions"] = self.class_distributions_
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the One-vs-Rest Classifier.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Required for metrics like McFadden's R-squared.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Required for metrics like McFadden's R-squared.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve One-vs-Rest specific metrics."}

        metrics = {}
        prefix = "ovr_" # One-vs-Rest

        # Structural metrics
        if self.n_classes_ is not None:
            metrics[f"{prefix}num_original_classes"] = self.n_classes_
            metrics[f"{prefix}num_binary_classifiers"] = self.n_classes_ # OvR has K binary classifiers for K classes
        
        metrics[f"{prefix}base_estimator_type"] = str(self.base_estimator)
        metrics[f"{prefix}class_weight_strategy"] = str(self.class_weight_strategy)
        metrics[f"{prefix}probability_calibration_enabled"] = self.probability_calibration
        metrics[f"{prefix}auto_feature_scaling_enabled"] = self.auto_scale_features

        # Per-class binary performance summary (if estimated)
        if self.class_performance_ and self.estimate_class_performance:
            f1_scores = [perf['mean_f1_score'] for perf in self.class_performance_.values() if isinstance(perf, dict) and 'mean_f1_score' in perf and perf['mean_f1_score'] is not None]
            if f1_scores:
                metrics[f"{prefix}mean_binary_f1_score"] = float(np.mean(f1_scores))
                metrics[f"{prefix}std_binary_f1_score"] = float(np.std(f1_scores))
                metrics[f"{prefix}min_binary_f1_score"] = float(np.min(f1_scores))
                metrics[f"{prefix}max_binary_f1_score"] = float(np.max(f1_scores))
                metrics[f"{prefix}num_evaluated_binary_classifiers"] = len(f1_scores)
            else:
                metrics[f"{prefix}binary_f1_info"] = "No valid F1 scores found in class_performance."
        elif self.estimate_class_performance:
            metrics[f"{prefix}binary_performance_info"] = "Per-class performance estimation was enabled but no results found."
        else:
            metrics[f"{prefix}binary_performance_info"] = "Per-class performance estimation was not enabled."

        # Binary problem imbalance summary (if analyzed)
        if self.ovr_analysis_ and 'binary_imbalance' in self.ovr_analysis_:
            imbalance_ratios = [
                data['imbalance_ratio'] 
                for data in self.ovr_analysis_['binary_imbalance'].values() 
                if isinstance(data, dict) and 'imbalance_ratio' in data and np.isfinite(data['imbalance_ratio'])
            ]
            if imbalance_ratios:
                metrics[f"{prefix}mean_binary_imbalance_ratio"] = float(np.mean(imbalance_ratios))
                metrics[f"{prefix}min_binary_imbalance_ratio"] = float(np.min(imbalance_ratios))
                metrics[f"{prefix}max_binary_imbalance_ratio"] = float(np.max(imbalance_ratios))
            else:
                metrics[f"{prefix}binary_imbalance_info"] = "No valid imbalance ratios found."

        # McFadden's Pseudo R-squared for the overall OvR model
        if y_true is not None and y_proba is not None and self.label_encoder_ is not None and self.classes_ is not None:
            try:
                y_true_encoded = self.label_encoder_.transform(y_true)
                n_samples = len(y_true_encoded)
                n_classes_model = len(self.classes_)

                if np.any(y_true_encoded >= y_proba.shape[1]) or np.any(y_true_encoded < 0):
                    raise ValueError("y_true_encoded contains out-of-bounds class indices for y_proba.")

                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                ll_model = np.sum(log_likelihoods_model)

                class_counts = np.bincount(y_true_encoded, minlength=n_classes_model)
                if len(class_counts) < n_classes_model: 
                    class_counts = np.pad(class_counts, (0, n_classes_model - len(class_counts)), 'constant')
                
                class_probas_null = class_counts / n_samples
                
                ll_null = 0
                for k_idx in range(n_classes_model):
                    if class_counts[k_idx] > 0 and class_probas_null[k_idx] > 0:
                         ll_null += class_counts[k_idx] * np.log(np.clip(class_probas_null[k_idx], 1e-15, 1))
                
                if ll_null == 0: # Avoid division by zero; if null model has 0 likelihood, any improvement is infinite.
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = 1.0 if ll_model == 0 else 0.0 # Or handle as undefined/error
                elif ll_model > ll_null: # Model is worse than null model
                     metrics[f"{prefix}mcfaddens_pseudo_r2"] = 0.0
                else:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics[f"{prefix}mcfaddens_pseudo_r2_error"] = str(e)
        
        if not metrics:
            metrics['info'] = "No specific One-vs-Rest metrics were available."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return OneVsRestClassifierPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of One-vs-Rest Classifier Plugin
    """
    print("Testing One-vs-Rest Classifier Plugin...")
    
    try:
        print("✅ Required libraries are available")
        
        # Create sample multi-class data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Generate multi-class dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=5,  # 5 classes for demonstrating OvR
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=0.02,
            random_state=42
        )
        
        print(f"\n📊 Multi-class Dataset Info:")
        print(f"Shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and test OvR plugin
        plugin = OneVsRestClassifierPlugin(
            base_estimator='logistic_regression',
            auto_scale_features=True,
            class_weight_strategy='balanced',
            estimate_class_performance=True,
            n_jobs=-1,  # Parallel training
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
            # Train One-vs-Rest classifier
            print("\n🚀 Training One-vs-Rest Classifier...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            y_proba = plugin.predict_proba(X_test)
            
            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n📊 One-vs-Rest Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Classes: {plugin.classes_}")
            
            # Get individual binary classifier predictions
            class_predictions = plugin.get_class_predictions(X_test[:10])  # First 10 samples
            print(f"\n🎯 Binary Classifier Analysis (first 10 samples):")
            for key, values in class_predictions.items():
                if key.endswith('_vs_rest'):
                    positive_count = np.sum(values)
                    print(f"{key}: {positive_count}/10 positive predictions")
            
            # Get OvR analysis
            ovr_analysis = plugin.get_ovr_analysis()
            print(f"\n🏗️ One-vs-Rest Strategy Analysis:")
            
            strategy_summary = ovr_analysis.get('strategy_summary', {})
            print(f"Base estimator: {strategy_summary.get('base_estimator', 'Unknown')}")
            print(f"Number of classes: {strategy_summary.get('n_classes', 'Unknown')}")
            print(f"Binary classifiers: {strategy_summary.get('n_binary_classifiers', 'Unknown')}")
            print(f"Feature scaling: {strategy_summary.get('feature_scaling', False)}")
            
            # Class distribution analysis
            if 'class_distributions' in ovr_analysis:
                class_dist = ovr_analysis['class_distributions']
                print(f"\n📈 Class Distribution Analysis:")
                print(f"Total samples: {class_dist['total_samples']}")
                for class_name, count in class_dist['class_counts'].items():
                    proportion = class_dist['class_proportions'][class_name]
                    print(f"Class {class_name}: {count} samples ({proportion:.1%})")
            
            # Binary imbalance analysis
            if 'ovr_characteristics' in ovr_analysis and 'binary_imbalance' in ovr_analysis['ovr_characteristics']:
                binary_imbalance = ovr_analysis['ovr_characteristics']['binary_imbalance']
                print(f"\n⚖️ Binary Problem Imbalance:")
                for class_name, imbalance_info in binary_imbalance.items():
                    ratio = imbalance_info['imbalance_ratio']
                    pos_prop = imbalance_info['positive_proportion']
                    print(f"Class {class_name} vs. Rest: {ratio:.1f}:1 imbalance ({pos_prop:.1%} positive)")
            
            # Class performance analysis
            if 'class_performance' in ovr_analysis:
                class_perf = ovr_analysis['class_performance']
                print(f"\n🎯 Per-Class Binary Performance:")
                for class_name, perf in class_perf.items():
                    if 'mean_f1_score' in perf:
                        f1 = perf['mean_f1_score']
                        std = perf['std_f1_score']
                        size = perf['class_size']
                        print(f"Class {class_name}: F1={f1:.3f}±{std:.3f} (size={size})")
            
            # Feature importance if available
            feature_importance = plugin.get_feature_importance()
            if feature_importance:
                print(f"\n🔍 Feature Importance Analysis:")
                aggregated = feature_importance['aggregated']
                mean_importance = aggregated['mean_importance']
                
                # Top 5 features
                top_features_idx = np.argsort(mean_importance)[-5:][::-1]
                print("Top 5 Most Discriminative Features (averaged across classes):")
                for i, idx in enumerate(top_features_idx):
                    feature_name = feature_importance['feature_names'][idx]
                    importance = mean_importance[idx]
                    print(f"{i+1}. {feature_name}: {importance:.4f}")
            
            # Model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Strategy: {model_params.get('strategy_type', 'Unknown')}")
            print(f"Base estimator: {model_params.get('base_estimator', 'Unknown')}")
            print(f"Binary classifiers: {model_params.get('n_binary_classifiers', 'Unknown')}")
            print(f"Feature scaling: {model_params.get('feature_scaling', False)}")
            print(f"Class weighting: {model_params.get('class_weight_strategy', 'Unknown')}")
            
            # Training info
            training_info = plugin.get_training_info()
            print(f"\n📈 Training Info:")
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Strategy type: {training_info['strategy_type']}")
            
            ovr_chars = training_info['ovr_characteristics']
            print(f"Binary decomposition: {ovr_chars['binary_decomposition']}")
            print(f"Independent training: {ovr_chars['independent_training']}")
            print(f"Parallel potential: {ovr_chars['parallel_potential']}")
            
            # Performance considerations
            perf_info = training_info['performance_considerations']
            print(f"\n⚡ Performance Characteristics:")
            print(f"Training time: {perf_info['training_time']}")
            print(f"Prediction time: {perf_info['prediction_time']}")
            print(f"Memory usage: {perf_info['memory_usage']}")
            print(f"Parallelization: {perf_info['parallelization']}")
            
            # OvR theory
            ovr_theory = training_info['ovr_theory']
            print(f"\n🧠 One-vs-Rest Theory:")
            print(f"Approach: {ovr_theory['approach']}")
            print(f"Binary problems: {ovr_theory['binary_problems']}")
            print(f"Decision rule: {ovr_theory['decision_rule']}")
            
            print("\n✅ One-vs-Rest Classifier Plugin test completed successfully!")
            print("🎯 Successfully decomposed multi-class problem into binary classifiers!")
            
            # Demonstrate OvR benefits
            print(f"\n🚀 One-vs-Rest Benefits:")
            print(f"Universal Wrapper: Works with any binary classifier")
            print(f"Perfect Parallelization: All {plugin.n_classes_} classifiers independent")
            print(f"Linear Scaling: O(K) complexity vs O(K²) for One-vs-One")
            print(f"Clear Interpretation: Each binary problem separately understandable")
            
            # Show confidence analysis
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