import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

# Try to import optional libraries
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

class StackingClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Stacking Classifier Plugin - Meta-Learning Ensemble Method
    
    Stacking (Stacked Generalization) is an ensemble meta-algorithm that uses a meta-learner 
    to learn how to best combine the predictions of multiple base learners. Unlike simple 
    voting, stacking trains a meta-model to optimally combine base predictions, potentially 
    achieving better performance than any individual base estimator.
    
    Key Features:
    1. Multi-Level Learning: Base level (diverse estimators) + Meta level (meta-learner)
    2. Cross-Validation Blending: Uses CV predictions to train meta-learner
    3. Bias Reduction: Meta-learner can correct systematic errors of base estimators
    4. Flexible Architecture: Any estimators as base learners, any estimator as meta-learner
    5. Feature Enhancement: Can use original features alongside base predictions
    6. Advanced Ensemble: More sophisticated than voting or simple averaging
    """
    
    def __init__(self,
                 # Base estimators configuration
                 base_estimators=['logistic_regression', 'random_forest', 'svm'],
                 meta_learner='logistic_regression',
                 
                 # Stacking parameters
                 cv_folds=5,
                 use_features_in_secondary=False,
                 stack_method='auto',
                 passthrough=False,
                 
                 # Cross-validation and training
                 n_jobs=None,
                 random_state=42,
                 verbose=0,
                 
                 # Base estimator hyperparameters
                 # Logistic Regression
                 lr_C=1.0,
                 lr_max_iter=1000,
                 lr_solver='liblinear',
                 
                 # Random Forest
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 
                 # SVM
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_probability=True,
                 
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
                 
                 # Neural Network
                 mlp_hidden_layer_sizes=(100,),
                 mlp_activation='relu',
                 mlp_max_iter=500,
                 
                 # XGBoost
                 xgb_n_estimators=100,
                 xgb_learning_rate=0.1,
                 xgb_max_depth=6,
                 
                 # Advanced options
                 auto_scale_features=True,
                 estimate_base_importance=True,
                 cross_validation_folds=5):
        """
        Initialize Stacking Classifier with comprehensive configuration
        
        Parameters:
        -----------
        base_estimators : list, default=['logistic_regression', 'random_forest', 'svm']
            List of base estimator types
        meta_learner : str, default='logistic_regression'
            Meta-learner type for combining base predictions
        cv_folds : int, default=5
            Number of folds for cross-validation in stacking
        use_features_in_secondary : bool, default=False
            Whether to use original features in meta-learner
        stack_method : str, default='auto'
            Method for obtaining base estimator predictions
        passthrough : bool, default=False
            Whether to concatenate original features with base predictions
        auto_scale_features : bool, default=True
            Whether to automatically scale features
        estimate_base_importance : bool, default=True
            Whether to estimate importance of base estimators
        """
        super().__init__()
        
        # Base estimators configuration
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        
        # Stacking parameters
        self.cv_folds = cv_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.stack_method = stack_method
        self.passthrough = passthrough
        
        # Training parameters
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Base estimator hyperparameters
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.lr_solver = lr_solver
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
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
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_max_iter = mlp_max_iter
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        
        # Advanced options
        self.auto_scale_features = auto_scale_features
        self.estimate_base_importance = estimate_base_importance
        self.cross_validation_folds = cross_validation_folds
        
        # Plugin metadata
        self._name = "Stacking Classifier"
        self._description = "Meta-learning ensemble that uses a meta-learner to optimally combine predictions from multiple base estimators through cross-validation blending."
        self._category = "Ensemble Methods"
        self._algorithm_type = "Stacked Generalization"
        self._paper_reference = "Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = False  # Depends on base estimators
        self._requires_scaling = False  # Can be configured
        self._supports_sparse = False  # Depends on base estimators
        self._is_linear = False  # Meta-learning ensemble
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = False  # Depends on base estimators
        self._ensemble_method = True
        self._meta_learning = True
        self._cross_validation_based = True
        self._bias_reduction = True
        self._adaptive_combination = True
        
        # Internal attributes
        self.stacking_classifier_ = None
        self.base_estimator_instances_ = []
        self.meta_learner_instance_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.base_estimator_scores_ = None
        self.meta_learner_features_ = None
        self.stacking_analysis_ = None
        self.base_importance_ = None
    
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
            "year_introduced": 1992,
            "key_innovations": {
                "meta_learning": "Uses a meta-learner to combine base estimator predictions",
                "cross_validation_blending": "Prevents overfitting in meta-learner training",
                "bias_reduction": "Meta-learner can correct systematic errors of base estimators",
                "adaptive_combination": "Learns optimal weights rather than using fixed rules",
                "feature_enhancement": "Can incorporate original features alongside predictions",
                "universal_base_estimators": "Works with any combination of base learners"
            },
            "algorithm_mechanics": {
                "training_process": {
                    "step_1": "Split training data into K folds for cross-validation",
                    "step_2": "For each fold, train base estimators on K-1 folds",
                    "step_3": "Predict on held-out fold to create meta-features",
                    "step_4": "Combine all meta-features to form meta-training set",
                    "step_5": "Train meta-learner on meta-features and original targets",
                    "step_6": "Retrain all base estimators on full training data"
                },
                "prediction_process": {
                    "step_1": "Get predictions from all trained base estimators",
                    "step_2": "Optionally concatenate with original features",
                    "step_3": "Feed combined features to meta-learner",
                    "step_4": "Return meta-learner's final prediction"
                },
                "meta_feature_creation": {
                    "cross_validation": "K-fold CV ensures meta-features are out-of-sample",
                    "prediction_types": "Can use class predictions, probabilities, or decision functions",
                    "feature_combination": "Meta-features can be combined with original features",
                    "dimensionality": "Meta-features dimension = n_base_estimators × n_classes"
                },
                "bias_variance_analysis": {
                    "base_level": "Diverse base estimators reduce variance through diversity",
                    "meta_level": "Meta-learner reduces bias by correcting systematic errors",
                    "overall_effect": "Combines variance reduction (diversity) with bias reduction (adaptation)",
                    "theoretical_guarantee": "Can achieve better performance than best base estimator"
                }
            },
            "stacking_theory": {
                "mathematical_foundation": {
                    "level_0": "Base estimators: h₁(x), h₂(x), ..., hₘ(x)",
                    "level_1": "Meta-learner: g(h₁(x), h₂(x), ..., hₘ(x), [x])",
                    "final_prediction": "ŷ = g(h₁(x), h₂(x), ..., hₘ(x))",
                    "cv_training": "Meta-features created using out-of-fold predictions"
                },
                "cross_validation_mechanics": {
                    "fold_splitting": "Stratified K-fold to maintain class distribution",
                    "out_of_sample": "Each instance used exactly once for meta-feature creation",
                    "generalization": "Prevents meta-learner overfitting to base predictions",
                    "efficiency": "Full training data used for both levels"
                },
                "meta_learner_selection": {
                    "simple_learners": "Logistic regression, linear models work well",
                    "complex_learners": "Can use any classifier as meta-learner",
                    "regularization": "Often beneficial to prevent overfitting",
                    "interpretability": "Simple meta-learners provide insight into base combination"
                },
                "feature_engineering": {
                    "prediction_only": "Use only base estimator predictions",
                    "feature_passthrough": "Include original features alongside predictions",
                    "hybrid_approach": "Meta-learner learns both feature and prediction patterns",
                    "dimensionality_considerations": "Balance information with overfitting risk"
                }
            },
            "base_estimator_selection": {
                "diversity_principle": {
                    "algorithm_diversity": "Different learning algorithms (tree, linear, instance-based)",
                    "hyperparameter_diversity": "Same algorithm with different parameters",
                    "data_diversity": "Different feature subsets or transformations",
                    "bias_diversity": "Estimators with different inductive biases"
                },
                "performance_requirements": {
                    "individual_quality": "Base estimators should be better than random",
                    "complementary_errors": "Different error patterns enable meta-learning",
                    "stability": "Stable estimators provide reliable meta-features",
                    "efficiency": "Balance complexity with training time"
                },
                "optimal_combinations": {
                    "classic_trio": "Logistic Regression + Random Forest + SVM",
                    "tree_ensemble": "Different tree-based methods with varied parameters",
                    "linear_nonlinear": "Linear models + nonlinear models for complementarity",
                    "fast_slow": "Fast models + complex models for speed-accuracy balance"
                }
            },
            "hyperparameter_effects": {
                "cv_folds": {
                    "effect": "More folds → better meta-feature quality, longer training",
                    "typical_range": "3-10 folds, 5 is standard",
                    "bias_variance": "More folds reduce bias but increase variance"
                },
                "stack_method": {
                    "predict": "Use class predictions (discrete)",
                    "predict_proba": "Use probability estimates (continuous)",
                    "decision_function": "Use decision scores (if available)",
                    "auto": "Automatically select best method"
                },
                "passthrough": {
                    "enabled": "Meta-learner sees original features + predictions",
                    "disabled": "Meta-learner sees only base predictions",
                    "trade_off": "More information vs. higher dimensionality"
                },
                "meta_learner_choice": {
                    "simple": "Logistic regression, linear SVM for interpretability",
                    "complex": "Random Forest, Gradient Boosting for flexibility",
                    "regularized": "Ridge, Lasso for high-dimensional meta-features"
                }
            },
            "available_base_estimators": {
                "logistic_regression": {
                    "type": "Linear probabilistic",
                    "strengths": ["Fast", "Interpretable", "Good probabilities", "Regularization"],
                    "weaknesses": ["Linear assumptions", "May underfit complex patterns"],
                    "best_for": "Linear patterns, baseline model, probability calibration",
                    "stacking_value": "High - provides linear perspective and good probabilities"
                },
                "random_forest": {
                    "type": "Tree ensemble",
                    "strengths": ["Robust", "Handles nonlinearity", "Feature importance", "Stable"],
                    "weaknesses": ["Can overfit", "Less interpretable"],
                    "best_for": "Nonlinear patterns, feature interactions, robust predictions",
                    "stacking_value": "High - excellent diversity from bagging and tree randomness"
                },
                "svm": {
                    "type": "Kernel-based",
                    "strengths": ["Kernel trick", "Good generalization", "Memory efficient"],
                    "weaknesses": ["Slow on large data", "Hyperparameter sensitive"],
                    "best_for": "Complex decision boundaries, medium-sized datasets",
                    "stacking_value": "High - different mathematical approach provides unique perspective"
                },
                "gradient_boosting": {
                    "type": "Boosted trees",
                    "strengths": ["High accuracy", "Handles complex patterns", "Feature importance"],
                    "weaknesses": ["Can overfit", "Sensitive to noise", "Slower training"],
                    "best_for": "Complex nonlinear patterns, high accuracy requirements",
                    "stacking_value": "Very High - sequential learning provides sophisticated predictions"
                },
                "neural_network": {
                    "type": "Multi-layer perceptron",
                    "strengths": ["Universal approximation", "Learns representations", "Flexible"],
                    "weaknesses": ["Black box", "Requires tuning", "Can overfit"],
                    "best_for": "Complex patterns, large datasets, representation learning",
                    "stacking_value": "High - nonlinear transformations provide unique features"
                },
                "knn": {
                    "type": "Instance-based",
                    "strengths": ["Non-parametric", "Local patterns", "Simple"],
                    "weaknesses": ["Curse of dimensionality", "Sensitive to noise"],
                    "best_for": "Local patterns, irregular decision boundaries",
                    "stacking_value": "Medium - provides local perspective but may be noisy"
                }
            },
            "meta_learner_options": {
                "logistic_regression": {
                    "advantages": ["Fast", "Interpretable weights", "Good probabilities", "Regularization"],
                    "disadvantages": ["Linear assumptions"],
                    "best_for": "Most stacking scenarios, provides interpretable combination weights"
                },
                "random_forest": {
                    "advantages": ["Handles nonlinear combinations", "Robust", "Feature importance"],
                    "disadvantages": ["Less interpretable", "May overfit"],
                    "best_for": "Complex base prediction interactions"
                },
                "linear_regression": {
                    "advantages": ["Simple", "Fast", "Interpretable"],
                    "disadvantages": ["Very restrictive", "No probability output"],
                    "best_for": "Simple combination scenarios, regression tasks"
                },
                "neural_network": {
                    "advantages": ["Very flexible", "Learns complex combinations"],
                    "disadvantages": ["Can overfit easily", "Black box"],
                    "best_for": "Large datasets with complex base prediction patterns"
                }
            },
            "strengths": [
                "Often achieves better performance than any individual base estimator",
                "Reduces both bias and variance through meta-learning",
                "Flexible architecture accommodates any base estimators",
                "Cross-validation prevents overfitting in meta-learner",
                "Can learn complex combination patterns",
                "Maintains diversity while optimizing combination",
                "Provides interpretable combination weights (with simple meta-learners)",
                "Handles heterogeneous base estimator outputs",
                "Scalable to many base estimators",
                "Can incorporate original features alongside predictions",
                "Theoretically grounded in ensemble learning",
                "Robust to poor individual base estimators"
            ],
            "weaknesses": [
                "Increased computational complexity (multiple training rounds)",
                "Higher memory requirements (stores all base estimators)",
                "Risk of overfitting with too many base estimators",
                "Performance depends heavily on base estimator diversity",
                "Can be sensitive to cross-validation strategy",
                "May not improve over simpler ensembles on some datasets",
                "Difficult to interpret final model behavior",
                "Requires careful hyperparameter tuning",
                "Longer training time due to cross-validation",
                "Meta-learner can overfit to base prediction patterns",
                "Performance plateau with too many similar base estimators",
                "Complexity may not be justified for simple problems"
            ],
            "ideal_use_cases": [
                "Diverse set of well-performing base estimators available",
                "Performance improvement over voting/averaging is critical",
                "Sufficient training data for both levels of learning",
                "Base estimators make different types of errors",
                "Complex decision boundaries requiring adaptive combination",
                "Competition or high-stakes prediction scenarios",
                "When interpretability of combination is important",
                "Multiple algorithms perform well but none dominates",
                "Large datasets that can support meta-learning",
                "Scenarios where bias reduction is as important as variance reduction",
                "Problems where simple averaging fails to capture optimal combination",
                "Applications requiring probability calibration across different methods"
            ],
            "comparison_with_other_ensembles": {
                "vs_voting": {
                    "stacking": "Learns optimal combination weights",
                    "voting": "Uses fixed combination rules (majority/average)",
                    "advantage": "Stacking can achieve better performance through adaptive weighting"
                },
                "vs_bagging": {
                    "stacking": "Combines different algorithms with meta-learning",
                    "bagging": "Combines same algorithm on different data samples",
                    "focus": "Stacking: algorithm diversity, Bagging: data diversity"
                },
                "vs_boosting": {
                    "stacking": "Parallel training of diverse estimators + meta-learning",
                    "boosting": "Sequential training with adaptive sample weighting",
                    "complexity": "Stacking: more complex architecture, Boosting: sequential dependency"
                },
                "vs_blending": {
                    "stacking": "Uses cross-validation for meta-feature creation",
                    "blending": "Uses holdout set for meta-learner training",
                    "robustness": "Stacking: more robust, uses full data more efficiently"
                }
            },
            "theoretical_guarantees": {
                "performance": "Can achieve better performance than best base estimator",
                "generalization": "Cross-validation ensures good generalization of meta-learner",
                "consistency": "Consistent if base estimators and meta-learner are consistent",
                "optimal_combination": "Meta-learner learns optimal combination under given constraints",
                "bias_variance": "Reduces both bias (through adaptation) and variance (through diversity)"
            }
        }
    
    def _create_base_estimator(self, estimator_type: str) -> BaseEstimator:
        """Create base estimator instance based on type"""
        
        if estimator_type == 'logistic_regression':
            return LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                solver=self.lr_solver,
                random_state=self.random_state
            )
        
        elif estimator_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state
            )
        
        elif estimator_type == 'svm':
            return SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                probability=self.svm_probability,
                random_state=self.random_state
            )
        
        elif estimator_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                min_samples_leaf=self.dt_min_samples_leaf,
                criterion=self.dt_criterion,
                random_state=self.random_state
            )
        
        elif estimator_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                metric=self.knn_metric
            )
        
        elif estimator_type == 'naive_bayes':
            return GaussianNB()
        
        elif estimator_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.gb_n_estimators,
                learning_rate=self.gb_learning_rate,
                max_depth=self.gb_max_depth,
                random_state=self.random_state
            )
        
        elif estimator_type == 'extra_trees' and EXTENDED_ALGORITHMS:
            return ExtraTreesClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state
            )
        
        elif estimator_type == 'adaboost' and EXTENDED_ALGORITHMS:
            return AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=self.random_state
            )
        
        elif estimator_type == 'mlp' and EXTENDED_ALGORITHMS:
            return MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                activation=self.mlp_activation,
                max_iter=self.mlp_max_iter,
                random_state=self.random_state
            )
        
        elif estimator_type == 'xgboost' and EXTENDED_ALGORITHMS:
            return XGBClassifier(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        elif estimator_type == 'lightgbm' and EXTENDED_ALGORITHMS:
            return LGBMClassifier(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                random_state=self.random_state,
                verbose=-1
            )
        
        else:
            # Default fallback
            warnings.warn(f"Unknown base estimator '{estimator_type}'. Using Logistic Regression as fallback.")
            return LogisticRegression(random_state=self.random_state)
    
    def _create_meta_learner(self) -> BaseEstimator:
        """Create meta-learner instance"""
        
        if self.meta_learner == 'logistic_regression':
            return LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                solver=self.lr_solver,
                random_state=self.random_state
            )
        
        elif self.meta_learner == 'random_forest':
            return RandomForestClassifier(
                n_estimators=50,  # Smaller for meta-learner
                max_depth=3,      # Shallower to prevent overfitting
                random_state=self.random_state
            )
        
        elif self.meta_learner == 'svm':
            return SVC(
                C=1.0,
                kernel='linear',  # Linear for interpretability
                probability=True,
                random_state=self.random_state
            )
        
        elif self.meta_learner == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=3,  # Shallow tree to prevent overfitting
                random_state=self.random_state
            )
        
        elif self.meta_learner == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=2,
                random_state=self.random_state
            )
        
        elif self.meta_learner == 'mlp' and EXTENDED_ALGORITHMS:
            return MLPClassifier(
                hidden_layer_sizes=(50,),  # Smaller network
                activation='relu',
                max_iter=500,
                random_state=self.random_state
            )
        
        else:
            # Default fallback
            return LogisticRegression(random_state=self.random_state)
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Stacking Classifier
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, default=None
            Sample weights (not supported by StackingClassifier)
            
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
        
        # Feature scaling if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        if self.verbose > 0:
            print(f"Training Stacking Classifier with {len(self.base_estimators)} base estimators...")
            print(f"Base estimators: {self.base_estimators}")
            print(f"Meta-learner: {self.meta_learner}")
            print(f"CV folds: {self.cv_folds}")
            print(f"Passthrough: {self.passthrough}")
        
        # Create base estimator instances
        self.base_estimator_instances_ = []
        estimators_list = []
        
        for i, estimator_type in enumerate(self.base_estimators):
            estimator = self._create_base_estimator(estimator_type)
            self.base_estimator_instances_.append(estimator)
            estimators_list.append((f"{estimator_type}_{i}", estimator))
        
        # Create meta-learner instance
        self.meta_learner_instance_ = self._create_meta_learner()
        
        # Create and configure stacking classifier
        self.stacking_classifier_ = StackingClassifier(
            estimators=estimators_list,
            final_estimator=self.meta_learner_instance_,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            stack_method=self.stack_method,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Fit the stacking ensemble
        self.stacking_classifier_.fit(X_scaled, y_encoded)
        
        # Evaluate base estimators using cross-validation
        if self.estimate_base_importance:
            self._evaluate_base_estimators(X_scaled, y_encoded)
        
        # Analyze the meta-features
        self._analyze_meta_features(X_scaled, y_encoded)
        
        # Analyze the stacking ensemble
        self._analyze_stacking(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _evaluate_base_estimators(self, X, y):
        """Evaluate base estimators using cross-validation"""
        try:
            cv_strategy = StratifiedKFold(n_splits=self.cross_validation_folds, shuffle=True, random_state=self.random_state)
            
            scores = {}
            for i, estimator in enumerate(self.base_estimator_instances_):
                estimator_name = self.base_estimators[i]
                cv_scores = cross_val_score(estimator, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=self.n_jobs)
                scores[estimator_name] = {
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'cv_scores': cv_scores.tolist()
                }
                
                if self.verbose > 0:
                    print(f"{estimator_name} CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
            self.base_estimator_scores_ = scores
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not evaluate base estimators: {str(e)}")
            self.base_estimator_scores_ = None
    
    def _analyze_meta_features(self, X, y):
        """Analyze the meta-features created by base estimators"""
        try:
            # Get meta-features using cross-validation
            cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            meta_features = []
            for train_idx, val_idx in cv_strategy.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                fold_meta_features = []
                for estimator in self.base_estimator_instances_:
                    # Fit estimator on fold training data
                    estimator_copy = estimator.__class__(**estimator.get_params())
                    estimator_copy.fit(X_train_fold, y_train_fold)
                    
                    # Get predictions on validation fold
                    if hasattr(estimator_copy, 'predict_proba'):
                        pred = estimator_copy.predict_proba(X_val_fold)
                    else:
                        pred = estimator_copy.predict(X_val_fold).reshape(-1, 1)
                    
                    fold_meta_features.append(pred)
                
                # Combine meta-features for this fold
                if fold_meta_features:
                    fold_combined = np.concatenate(fold_meta_features, axis=1)
                    meta_features.append(fold_combined)
            
            if meta_features:
                # Combine all folds
                all_meta_features = np.vstack(meta_features)
                
                self.meta_learner_features_ = {
                    'shape': all_meta_features.shape,
                    'mean': np.mean(all_meta_features, axis=0).tolist(),
                    'std': np.std(all_meta_features, axis=0).tolist(),
                    'min': np.min(all_meta_features, axis=0).tolist(),
                    'max': np.max(all_meta_features, axis=0).tolist()
                }
                
                if self.verbose > 0:
                    print(f"Meta-features shape: {all_meta_features.shape}")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not analyze meta-features: {str(e)}")
            self.meta_learner_features_ = None
    
    def _analyze_stacking(self, X, y):
        """Analyze the stacking ensemble"""
        analysis = {
            "base_estimators": self.base_estimators,
            "meta_learner": self.meta_learner,
            "n_base_estimators": len(self.base_estimators),
            "cv_folds": self.cv_folds,
            "stack_method": self.stack_method,
            "passthrough": self.passthrough,
            "feature_scaling": self.scaler_ is not None,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(self.classes_)
        }
        
        # Add base estimator performance
        if self.base_estimator_scores_:
            analysis["base_estimator_performance"] = self.base_estimator_scores_
        
        # Add meta-feature analysis
        if self.meta_learner_features_:
            analysis["meta_features"] = self.meta_learner_features_
        
        # Estimate computational complexity
        analysis["computational_complexity"] = {
            "training_phases": 2,
            "cv_rounds": self.cv_folds,
            "total_base_training": len(self.base_estimators) * (self.cv_folds + 1),
            "meta_training": 1,
            "parallel_potential": "High for base estimators, sequential for meta-learner"
        }
        
        # Analyze base estimator importance (if meta-learner supports it)
        if hasattr(self.stacking_classifier_.final_estimator_, 'coef_'):
            try:
                importance = np.abs(self.stacking_classifier_.final_estimator_.coef_[0])
                if len(importance) >= len(self.base_estimators):
                    base_importance = importance[:len(self.base_estimators)]
                    analysis["base_estimator_importance"] = {
                        estimator: float(imp) for estimator, imp in zip(self.base_estimators, base_importance)
                    }
                    self.base_importance_ = base_importance
            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not extract base estimator importance: {str(e)}")
        
        self.stacking_analysis_ = analysis
    
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
        
        # Get predictions from stacking classifier
        y_pred_encoded = self.stacking_classifier_.predict(X_scaled)
        
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
        
        # Get probabilities from stacking classifier
        try:
            probabilities = self.stacking_classifier_.predict_proba(X_scaled)
            return probabilities
        except Exception as e:
            warnings.warn(f"Probability prediction failed: {str(e)}. Using hard predictions.")
            
            # Fallback: convert hard predictions to probabilities
            y_pred = self.stacking_classifier_.predict(X_scaled)
            n_samples = len(y_pred)
            n_classes = len(self.classes_)
            
            probabilities = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(y_pred):
                probabilities[i, pred] = 1.0
            
            return probabilities
    
    def get_base_predictions(self, X):
        """
        Get predictions from individual base estimators
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : dict
            Dictionary with base estimator predictions
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
            # Get predictions from base estimators
            for estimator_name, estimator in self.stacking_classifier_.estimators_:
                pred_encoded = estimator.predict(X_scaled)
                pred = self.label_encoder_.inverse_transform(pred_encoded)
                predictions[estimator_name] = pred
            
            return predictions
            
        except Exception as e:
            warnings.warn(f"Could not get base predictions: {str(e)}")
            return {}
    
    def get_meta_features(self, X):
        """
        Get meta-features (base estimator predictions) for samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        meta_features : array, shape (n_samples, n_meta_features)
            Meta-features used by meta-learner
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        try:
            # Get meta-features from stacking classifier
            # This requires access to the internal transform method
            meta_features = []
            
            for estimator_name, estimator in self.stacking_classifier_.estimators_:
                if hasattr(estimator, 'predict_proba') and self.stack_method in ['predict_proba', 'auto']:
                    pred = estimator.predict_proba(X_scaled)
                elif hasattr(estimator, 'decision_function') and self.stack_method == 'decision_function':
                    pred = estimator.decision_function(X_scaled)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                else:
                    pred = estimator.predict(X_scaled)
                    pred = pred.reshape(-1, 1)
                
                meta_features.append(pred)
            
            if meta_features:
                combined_meta_features = np.concatenate(meta_features, axis=1)
                
                # Add original features if passthrough is enabled
                if self.passthrough:
                    combined_meta_features = np.concatenate([combined_meta_features, X_scaled], axis=1)
                
                return combined_meta_features
            else:
                return np.array([])
                
        except Exception as e:
            warnings.warn(f"Could not get meta-features: {str(e)}")
            return np.array([])
    
    def get_stacking_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the stacking ensemble
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive stacking analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "ensemble_summary": {
                "algorithm": "Stacking Classifier",
                "base_estimators": self.base_estimators,
                "meta_learner": self.meta_learner,
                "n_base_estimators": len(self.base_estimators),
                "cv_folds": self.cv_folds,
                "stack_method": self.stack_method,
                "passthrough": self.passthrough,
                "feature_scaling": self.scaler_ is not None,
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "classes": self.classes_.tolist()
            }
        }
        
        # Add base estimator performance
        if self.base_estimator_scores_:
            analysis["base_estimator_performance"] = self.base_estimator_scores_
        
        # Add stacking analysis
        if self.stacking_analysis_:
            analysis["stacking_characteristics"] = self.stacking_analysis_
        
        # Add meta-feature analysis
        if self.meta_learner_features_:
            analysis["meta_features_analysis"] = self.meta_learner_features_
        
        # Add base estimator importance
        if self.base_importance_ is not None:
            analysis["base_estimator_importance"] = {
                "importances": {estimator: float(imp) for estimator, imp in zip(self.base_estimators, self.base_importance_)},
                "top_3_estimators": dict(sorted(
                    {estimator: float(imp) for estimator, imp in zip(self.base_estimators, self.base_importance_)}.items(),
                    key=lambda x: x[1], reverse=True
                )[:3])
            }
        
        return analysis
    
    def plot_stacking_analysis(self, figsize=(16, 12)):
        """
        Create comprehensive stacking analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(16, 12)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Stacking analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Base Estimator Performance
        if self.base_estimator_scores_:
            estimators = list(self.base_estimator_scores_.keys())
            mean_scores = [self.base_estimator_scores_[est]['mean_cv_score'] for est in estimators]
            std_scores = [self.base_estimator_scores_[est]['std_cv_score'] for est in estimators]
            
            bars = ax1.bar(range(len(estimators)), mean_scores, 
                          yerr=std_scores, capsize=5, alpha=0.7, 
                          color='skyblue', edgecolor='navy')
            ax1.set_xticks(range(len(estimators)))
            ax1.set_xticklabels([est.replace('_', ' ').title() for est in estimators], rotation=45)
            ax1.set_ylabel('Cross-Validation Accuracy')
            ax1.set_title('Base Estimator Performance')
            ax1.grid(True, alpha=0.3)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, mean_scores)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'Base estimator\nperformance\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Base Estimator Performance')
        
        # 2. Base Estimator Importance (if available)
        if self.base_importance_ is not None:
            importance_dict = {estimator: float(imp) for estimator, imp in zip(self.base_estimators, self.base_importance_)}
            estimators = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            bars = ax2.barh(range(len(estimators)), importances, alpha=0.7, 
                           color='lightgreen', edgecolor='darkgreen')
            ax2.set_yticks(range(len(estimators)))
            ax2.set_yticklabels([est.replace('_', ' ').title() for est in estimators])
            ax2.set_xlabel('Meta-Learner Importance')
            ax2.set_title('Base Estimator Importance in Meta-Learner')
            
            # Add importance values
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{importance:.3f}', ha='left', va='center')
        else:
            ax2.text(0.5, 0.5, 'Base estimator\nimportance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Base Estimator Importance')
        
        # 3. Meta-Features Analysis
        if self.meta_learner_features_:
            meta_info = self.meta_learner_features_
            
            # Plot meta-feature statistics
            n_features = len(meta_info['mean'])
            x_pos = np.arange(min(n_features, 20))  # Show first 20 features
            
            if n_features > 0:
                means = meta_info['mean'][:20]
                stds = meta_info['std'][:20]
                
                ax3.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7, 
                       color='orange', edgecolor='darkorange')
                ax3.set_xlabel('Meta-Feature Index')
                ax3.set_ylabel('Mean Value ± Std')
                ax3.set_title(f'Meta-Features Statistics (showing first 20/{n_features})')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No meta-features\navailable', 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Meta-features\nanalysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Meta-Features Analysis')
        
        # 4. Stacking Configuration
        ax4.axis('tight')
        ax4.axis('off')
        table_data = []
        table_data.append(['Base Estimators', str(len(self.base_estimators))])
        table_data.append(['Estimator Types', ', '.join([est.replace('_', ' ').title() for est in self.base_estimators[:3]]) + 
                          ('...' if len(self.base_estimators) > 3 else '')])
        table_data.append(['Meta-Learner', self.meta_learner.replace('_', ' ').title()])
        table_data.append(['CV Folds', str(self.cv_folds)])
        table_data.append(['Stack Method', self.stack_method])
        table_data.append(['Passthrough', 'Yes' if self.passthrough else 'No'])
        table_data.append(['Feature Scaling', 'Yes' if self.scaler_ is not None else 'No'])
        table_data.append(['Classes', str(len(self.classes_))])
        
        if self.meta_learner_features_:
            table_data.append(['Meta-Features', str(self.meta_learner_features_['shape'][1])])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Property', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Stacking Configuration')
        
        plt.tight_layout()
        return fig
    
    def plot_base_predictions_comparison(self, X_test, y_test, figsize=(14, 10)):
        """
        Visualize comparison of base estimator predictions
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        figsize : tuple, default=(14, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Base predictions comparison visualization
        """
        if not self.is_fitted_:
            return None
        
        # Get predictions from all base estimators and the ensemble
        base_predictions = self.get_base_predictions(X_test)
        ensemble_pred = self.predict(X_test)
        
        if not base_predictions:
            return None
        
        from sklearn.metrics import accuracy_score, classification_report
        
        # Calculate accuracies
        accuracies = {}
        for estimator_name, pred in base_predictions.items():
            accuracies[estimator_name] = accuracy_score(y_test, pred)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        accuracies['Stacking Ensemble'] = ensemble_accuracy
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Accuracy Comparison
        estimators = list(accuracies.keys())
        accs = list(accuracies.values())
        colors = ['skyblue'] * (len(estimators) - 1) + ['gold']  # Highlight ensemble
        
        bars = ax1.bar(range(len(estimators)), accs, color=colors, alpha=0.7, edgecolor='navy')
        ax1.set_xticks(range(len(estimators)))
        ax1.set_xticklabels([est.replace('_', ' ').title() for est in estimators], rotation=45)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison: Base Estimators vs Stacking')
        ax1.grid(True, alpha=0.3)
        
        # Add accuracy labels
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Prediction Agreement Matrix
        # Calculate pairwise agreement between estimators
        pred_matrix = []
        estimator_names = []
        
        for name, pred in base_predictions.items():
            pred_matrix.append(pred)
            estimator_names.append(name.replace('_', ' ').title())
        
        pred_matrix.append(ensemble_pred)
        estimator_names.append('Stacking')
        
        n_estimators = len(pred_matrix)
        agreement_matrix = np.zeros((n_estimators, n_estimators))
        
        for i in range(n_estimators):
            for j in range(n_estimators):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = np.mean(pred_matrix[i] == pred_matrix[j])
                    agreement_matrix[i, j] = agreement
        
        im = ax2.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(n_estimators))
        ax2.set_yticks(range(n_estimators))
        ax2.set_xticklabels(estimator_names, rotation=45)
        ax2.set_yticklabels(estimator_names)
        ax2.set_title('Prediction Agreement Matrix')
        
        # Add agreement values
        for i in range(n_estimators):
            for j in range(n_estimators):
                ax2.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                        ha='center', va='center',
                        color='white' if agreement_matrix[i, j] > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax2, label='Agreement Rate')
        
        # 3. Error Analysis
        # Count errors made by each estimator
        error_counts = {}
        for name, pred in base_predictions.items():
            error_counts[name] = np.sum(pred != y_test)
        error_counts['Stacking Ensemble'] = np.sum(ensemble_pred != y_test)
        
        estimators = list(error_counts.keys())
        errors = list(error_counts.values())
        
        bars = ax3.bar(range(len(estimators)), errors, alpha=0.7, 
                      color='lightcoral', edgecolor='darkred')
        ax3.set_xticks(range(len(estimators)))
        ax3.set_xticklabels([est.replace('_', ' ').title() for est in estimators], rotation=45)
        ax3.set_ylabel('Number of Errors')
        ax3.set_title('Error Count Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add error count labels
        for bar, err in zip(bars, errors):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{err}', ha='center', va='bottom')
        
        # 4. Ensemble Improvement Analysis
        # Show how often ensemble prediction differs from base predictions
        base_names = list(base_predictions.keys())
        ensemble_improvements = []
        
        for name in base_names:
            base_pred = base_predictions[name]
            
            # Cases where base estimator was wrong but ensemble was right
            base_wrong = (base_pred != y_test)
            ensemble_right = (ensemble_pred == y_test)
            improvements = np.sum(base_wrong & ensemble_right)
            ensemble_improvements.append(improvements)
        
        bars = ax4.bar(range(len(base_names)), ensemble_improvements, alpha=0.7, 
                      color='lightgreen', edgecolor='darkgreen')
        ax4.set_xticks(range(len(base_names)))
        ax4.set_xticklabels([name.replace('_', ' ').title() for name in base_names], rotation=45)
        ax4.set_ylabel('Number of Corrections')
        ax4.set_title('Ensemble Corrections per Base Estimator')
        ax4.grid(True, alpha=0.3)
        
        # Add correction count labels
        for bar, imp in zip(bars, ensemble_improvements):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{imp}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🏗️ Stacking Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Ensemble", "Base Est.", "Meta-Learn", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Ensemble Architecture**")
            
            # Base estimators selection
            available_estimators = ['logistic_regression', 'random_forest', 'svm', 'decision_tree', 
                                  'knn', 'naive_bayes', 'gradient_boosting']
            
            if EXTENDED_ALGORITHMS:
                available_estimators.extend(['extra_trees', 'adaboost', 'mlp', 'xgboost', 'lightgbm'])
            
            selected_estimators = st.multiselect(
                "Base Estimators:",
                options=available_estimators,
                default=self.base_estimators,
                help="Select base estimators for the ensemble",
                key=f"{key_prefix}_base_estimators"
            )
            
            if len(selected_estimators) < 2:
                st.error("⚠️ Select at least 2 base estimators for stacking")
            elif len(selected_estimators) > 8:
                st.warning("⚠️ Too many estimators may cause overfitting")
            else:
                st.success(f"✅ Selected {len(selected_estimators)} diverse base estimators")
            
            # Meta-learner selection
            meta_learner_options = ['logistic_regression', 'random_forest', 'svm', 'decision_tree', 'gradient_boosting']
            if EXTENDED_ALGORITHMS:
                meta_learner_options.append('mlp')
            
            meta_learner = st.selectbox(
                "Meta-Learner:",
                options=meta_learner_options,
                index=meta_learner_options.index(self.meta_learner) if self.meta_learner in meta_learner_options else 0,
                help="Algorithm to combine base estimator predictions",
                key=f"{key_prefix}_meta_learner"
            )
            
            # Meta-learner info
            if meta_learner == 'logistic_regression':
                st.info("📈 Logistic Regression: Interpretable linear combination with coefficients")
            elif meta_learner == 'random_forest':
                st.info("🌳 Random Forest: Learns complex nonlinear combinations")
            elif meta_learner == 'svm':
                st.info("🔍 SVM: Finds optimal hyperplane in meta-feature space")
            elif meta_learner == 'decision_tree':
                st.info("🌿 Decision Tree: Interpretable rules for combining predictions")
            elif meta_learner == 'gradient_boosting':
                st.info("🚀 Gradient Boosting: Powerful meta-learner for complex patterns")
            
            # Cross-validation folds
            cv_folds = st.slider(
                "Cross-Validation Folds:",
                min_value=3,
                max_value=10,
                value=self.cv_folds,
                help="Number of folds for creating meta-features",
                key=f"{key_prefix}_cv_folds"
            )
            
            if cv_folds < 5:
                st.warning("⚠️ Few folds may lead to noisy meta-features")
            
            # Stacking method
            stack_method = st.selectbox(
                "Stack Method:",
                options=['auto', 'predict_proba', 'decision_function', 'predict'],
                index=['auto', 'predict_proba', 'decision_function', 'predict'].index(self.stack_method),
                help="Method for obtaining base estimator predictions",
                key=f"{key_prefix}_stack_method"
            )
            
            if stack_method == 'predict_proba':
                st.info("🎯 Using probability estimates (recommended for most cases)")
            elif stack_method == 'decision_function':
                st.info("📊 Using decision function scores (may not be available for all estimators)")
            elif stack_method == 'predict':
                st.info("🔢 Using hard predictions (least informative)")
            else:
                st.info("🤖 Auto: automatically selects best available method")
            
            # Passthrough
            passthrough = st.checkbox(
                "Feature Passthrough",
                value=self.passthrough,
                help="Include original features alongside base predictions in meta-learner",
                key=f"{key_prefix}_passthrough"
            )
            
            if passthrough:
                st.success("✅ Meta-learner will see both predictions and original features")
            else:
                st.info("📊 Meta-learner will see only base estimator predictions")
        
        with tab2:
            st.markdown("**Base Estimator Hyperparameters**")
            
            # Show hyperparameters for selected estimators
            selected_estimators = selected_estimators if 'selected_estimators' in locals() else self.base_estimators
            
            # Logistic Regression parameters
            if 'logistic_regression' in selected_estimators:
                st.markdown("**Logistic Regression:**")
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
                    key=f"{key_prefix}_lr_max_iter"
                )
                
                lr_solver = st.selectbox(
                    "LR - Solver:",
                    options=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                    index=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'].index(self.lr_solver),
                    key=f"{key_prefix}_lr_solver"
                )
            else:
                lr_C = self.lr_C
                lr_max_iter = self.lr_max_iter
                lr_solver = self.lr_solver
            
            # Random Forest parameters
            if 'random_forest' in selected_estimators:
                st.markdown("**Random Forest:**")
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
            else:
                rf_n_estimators = self.rf_n_estimators
                rf_max_depth = self.rf_max_depth
                rf_min_samples_split = self.rf_min_samples_split
            
            # SVM parameters
            if 'svm' in selected_estimators:
                st.markdown("**Support Vector Machine:**")
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
                    help="Required for predict_proba method",
                    key=f"{key_prefix}_svm_probability"
                )
            else:
                svm_C = self.svm_C
                svm_kernel = self.svm_kernel
                svm_probability = self.svm_probability
            
            # Decision Tree parameters
            if 'decision_tree' in selected_estimators:
                st.markdown("**Decision Tree:**")
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
                
                dt_criterion = st.selectbox(
                    "DT - Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.dt_criterion),
                    key=f"{key_prefix}_dt_criterion"
                )
            else:
                dt_max_depth = self.dt_max_depth
                dt_criterion = self.dt_criterion
            
            # KNN parameters
            if 'knn' in selected_estimators:
                st.markdown("**K-Nearest Neighbors:**")
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
            else:
                knn_n_neighbors = self.knn_n_neighbors
                knn_weights = self.knn_weights
            
            # Gradient Boosting parameters
            if 'gradient_boosting' in selected_estimators:
                st.markdown("**Gradient Boosting:**")
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
            else:
                gb_n_estimators = self.gb_n_estimators
                gb_learning_rate = self.gb_learning_rate
                gb_max_depth = self.gb_max_depth
            
            # Neural Network parameters
            if 'mlp' in selected_estimators and EXTENDED_ALGORITHMS:
                st.markdown("**Neural Network:**")
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
            else:
                mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                mlp_activation = self.mlp_activation
            
            # XGBoost parameters
            if 'xgboost' in selected_estimators and EXTENDED_ALGORITHMS:
                st.markdown("**XGBoost:**")
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
            else:
                xgb_n_estimators = self.xgb_n_estimators
                xgb_learning_rate = self.xgb_learning_rate
                xgb_max_depth = self.xgb_max_depth
        
        with tab3:
            st.markdown("**Meta-Learner Configuration**")
            
            st.info("""
            **Meta-Learner Role:**
            • Learns to optimally combine base estimator predictions
            • Trained on out-of-fold predictions (prevents overfitting)
            • Can learn complex combination patterns
            • Should be simpler than base estimators to avoid overfitting
            """)
            
            # Meta-learner specific parameters based on selection
            meta_learner = meta_learner if 'meta_learner' in locals() else self.meta_learner
            
            if meta_learner == 'logistic_regression':
                st.markdown("**Meta-Learner: Logistic Regression**")
                st.success("✅ Excellent choice: provides interpretable combination weights")
                
                meta_lr_C = st.number_input(
                    "Meta-LR - Regularization (C):",
                    value=1.0,
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    help="Higher C = less regularization",
                    key=f"{key_prefix}_meta_lr_C"
                )
                
                if meta_lr_C < 0.1:
                    st.info("🔒 Strong regularization - prevents overfitting to base predictions")
                elif meta_lr_C > 10:
                    st.warning("⚠️ Weak regularization - may overfit to meta-features")
            
            elif meta_learner == 'random_forest':
                st.markdown("**Meta-Learner: Random Forest**")
                st.info("🌳 Good for learning complex base prediction interactions")
                
                meta_rf_n_estimators = st.slider(
                    "Meta-RF - Number of Trees:",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="Fewer trees for meta-learner to prevent overfitting",
                    key=f"{key_prefix}_meta_rf_n_estimators"
                )
                
                meta_rf_max_depth = st.slider(
                    "Meta-RF - Max Depth:",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Shallow trees prevent overfitting to meta-features",
                    key=f"{key_prefix}_meta_rf_max_depth"
                )
            
            elif meta_learner == 'svm':
                st.markdown("**Meta-Learner: SVM**")
                st.info("🔍 Linear SVM recommended for interpretable combination")
                
                meta_svm_C = st.number_input(
                    "Meta-SVM - C Parameter:",
                    value=1.0,
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_meta_svm_C"
                )
                
                meta_svm_kernel = st.selectbox(
                    "Meta-SVM - Kernel:",
                    options=['linear', 'rbf', 'poly'],
                    index=0,  # Default to linear
                    help="Linear kernel recommended for interpretability",
                    key=f"{key_prefix}_meta_svm_kernel"
                )
                
                if meta_svm_kernel == 'linear':
                    st.success("✅ Linear kernel provides interpretable combination")
                else:
                    st.warning("⚠️ Nonlinear kernel may overfit to meta-features")
            
            # Cross-validation strategy info
            cv_folds = cv_folds if 'cv_folds' in locals() else self.cv_folds
            st.markdown("**Cross-Validation Strategy:**")
            st.info(f"""
            **Meta-Feature Creation Process:**
            • Data split into {cv_folds} folds
            • For each fold: train base estimators on other {cv_folds-1} folds
            • Predict on held-out fold to create meta-features
            • Combine all out-of-fold predictions for meta-learner training
            • This prevents meta-learner overfitting to base predictions
            """)
            
            # Feature engineering options
            st.markdown("**Feature Engineering:**")
            
            use_features_in_secondary = st.checkbox(
                "Use Original Features in Meta-Learner",
                value=self.use_features_in_secondary,
                help="Include original features alongside base predictions",
                key=f"{key_prefix}_use_features_in_secondary"
            )
            
            if use_features_in_secondary or (passthrough if 'passthrough' in locals() else self.passthrough):
                st.success("✅ Meta-learner will see both predictions and original features")
                st.info("• Provides more information to meta-learner\n• May improve performance\n• Increases risk of overfitting")
            else:
                st.info("📊 Meta-learner will see only base estimator predictions")
                st.info("• Simpler meta-learning problem\n• Reduces overfitting risk\n• May miss feature-prediction interactions")
        
        with tab4:
            st.markdown("**Advanced Configuration**")
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Scale features for distance-based estimators (SVM, KNN, Neural Networks)",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            if auto_scale_features:
                st.success("✅ Features will be standardized (important for SVM, KNN, Neural Networks)")
            else:
                st.warning("⚠️ No scaling - may hurt performance of distance-based estimators")
            
            # Parallel processing
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores",
                key=f"{key_prefix}_n_jobs"
            )
            
            if n_jobs == -1:
                st.success("🚀 Using all CPU cores for parallel training")
            elif n_jobs and n_jobs > 1:
                st.info(f"🔄 Using {n_jobs} CPU cores")
            else:
                st.info("🔄 Sequential training (single core)")
            
            # Base estimator evaluation
            estimate_base_importance = st.checkbox(
                "Estimate Base Estimator Importance",
                value=self.estimate_base_importance,
                help="Evaluate individual base estimator performance",
                key=f"{key_prefix}_estimate_base_importance"
            )
            
            # Cross-validation for evaluation
            cross_validation_folds = st.slider(
                "CV Folds for Evaluation:",
                min_value=3,
                max_value=10,
                value=self.cross_validation_folds,
                help="Separate CV for base estimator evaluation",
                key=f"{key_prefix}_cross_validation_folds"
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
                help="Control training output verbosity",
                key=f"{key_prefix}_verbose"
            )
            
            # Advanced stacking options
            st.markdown("**Advanced Stacking Options:**")
            
            if st.button("🔬 Show Stacking Theory", key=f"{key_prefix}_stacking_theory"):
                st.markdown("""
                **Stacking Theory & Best Practices:**
                
                **Meta-Learning Formula:**
                - Level 0: h₁(x), h₂(x), ..., hₘ(x) (base estimators)
                - Level 1: g(h₁(x), h₂(x), ..., hₘ(x)) (meta-learner)
                - Final: ŷ = g(h₁(x), h₂(x), ..., hₘ(x))
                
                **Key Principles:**
                - Base estimators should be diverse and reasonably accurate
                - Meta-learner should be simpler to avoid overfitting
                - Cross-validation prevents meta-learner overfitting
                - More base estimators ≠ always better (diminishing returns)
                
                **Optimal Configuration:**
                - 3-7 diverse base estimators
                - Simple meta-learner (Logistic Regression recommended)
                - 5-fold cross-validation for meta-feature creation
                - Regularization in meta-learner
                """)
            
            if st.button("⚙️ Performance Optimization", key=f"{key_prefix}_performance_optimization"):
                st.markdown("""
                **Stacking Performance Tips:**
                
                **Base Estimator Selection:**
                - Choose complementary algorithms (different biases)
                - Include at least one linear and one nonlinear estimator
                - Avoid too many similar estimators
                - Ensure each estimator performs better than random
                
                **Meta-Learner Selection:**
                - Start with Logistic Regression (interpretable, robust)
                - Use regularization to prevent overfitting
                - Consider ensemble meta-learners for complex patterns
                - Avoid overly complex meta-learners
                
                **Cross-Validation Strategy:**
                - 5-fold CV is usually sufficient
                - More folds = better meta-features, longer training
                - Stratified CV maintains class distribution
                - Ensure sufficient samples per fold
                
                **Feature Engineering:**
                - Try both with and without feature passthrough
                - Scale features if using distance-based estimators
                - Consider feature selection for meta-learner
                - Monitor meta-feature dimensionality
                """)
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Stacking Classifier** - Meta-Learning Ensemble:
            • 🏗️ Two-level architecture: base estimators + meta-learner
            • 🔄 Cross-validation prevents meta-learner overfitting
            • 🧠 Meta-learner learns optimal combination of base predictions
            • 📊 Can achieve better performance than voting or averaging
            • 🎯 Reduces both bias (adaptation) and variance (diversity)
            • 🔍 Provides interpretable combination weights (with linear meta-learner)
            
            **Key Advantages:**
            • Adaptive combination of base estimators
            • Learns from base estimator error patterns
            • Flexible architecture with any algorithms
            • Theoretically grounded ensemble method
            """)
            
            # Algorithm comparison
            if st.button("📊 Ensemble Method Comparison", key=f"{key_prefix}_ensemble_comparison"):
                st.markdown("""
                **Stacking vs Other Ensemble Methods:**
                
                **Stacking vs Voting:**
                - Stacking: Learns optimal combination weights
                - Voting: Uses fixed rules (majority/average)
                - Advantage: Adaptive weighting often performs better
                
                **Stacking vs Bagging:**
                - Stacking: Combines different algorithms
                - Bagging: Combines same algorithm on different data
                - Focus: Algorithm diversity vs Data diversity
                
                **Stacking vs Boosting:**
                - Stacking: Parallel training + meta-learning
                - Boosting: Sequential adaptive training
                - Complexity: More complex architecture vs sequential dependency
                
                **When to Use Stacking:**
                - Multiple well-performing but diverse base estimators
                - Performance improvement over simple ensembles is critical
                - Sufficient data for both learning levels
                - Interpretability of combination is valuable
                """)
            
            # Best practices guide
            if st.button("🎯 Stacking Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Stacking Best Practices:**
                
                **Base Estimator Selection:**
                1. Choose 3-7 diverse, well-performing estimators
                2. Include different algorithm types (linear, tree, instance-based)
                3. Ensure each estimator > random performance
                4. Consider computational cost vs benefit
                
                **Meta-Learner Guidelines:**
                1. Start with Logistic Regression (simple, interpretable)
                2. Use regularization to prevent overfitting
                3. Keep simpler than base estimators
                4. Consider linear models for interpretability
                
                **Cross-Validation Strategy:**
                1. Use 5-fold stratified CV for meta-features
                2. Ensure adequate samples per fold
                3. Consider leave-one-out for small datasets
                4. Use same CV strategy for all base estimators
                
                **Hyperparameter Tuning:**
                1. Tune base estimators first (individually)
                2. Then tune meta-learner hyperparameters
                3. Consider grid search for meta-learner
                4. Monitor overfitting at meta-level
                
                **Validation & Evaluation:**
                1. Use separate test set for final evaluation
                2. Compare against individual base estimators
                3. Analyze meta-learner coefficients (if linear)
                4. Check for base estimator redundancy
                """)
            
            # Implementation details
            if st.button("🔧 Implementation Details", key=f"{key_prefix}_implementation_details"):
                st.markdown("""
                **Stacking Implementation Details:**
                
                **Training Process:**
                1. Split training data into K folds
                2. For each fold i:
                   - Train base estimators on folds ≠ i
                   - Predict on fold i to create meta-features
                3. Combine all meta-features into meta-training set
                4. Train meta-learner on meta-features and original targets
                5. Retrain all base estimators on full training data
                
                **Prediction Process:**
                1. Get predictions from all trained base estimators
                2. Optionally concatenate with original features
                3. Feed combined features to trained meta-learner
                4. Return meta-learner's final prediction
                
                **Key Technical Considerations:**
                - Meta-features must be out-of-sample to prevent overfitting
                - Final base estimators trained on full data for best performance
                - Meta-learner sees only aggregated base predictions during inference
                - Memory usage scales with number of base estimators
                - Training time = base training × (K+1) + meta training
                """)
        
        # Return selected hyperparameters
        return {
            "base_estimators": selected_estimators if 'selected_estimators' in locals() else self.base_estimators,
            "meta_learner": meta_learner if 'meta_learner' in locals() else self.meta_learner,
            "cv_folds": cv_folds if 'cv_folds' in locals() else self.cv_folds,
            "stack_method": stack_method if 'stack_method' in locals() else self.stack_method,
            "passthrough": passthrough if 'passthrough' in locals() else self.passthrough,
            "use_features_in_secondary": use_features_in_secondary if 'use_features_in_secondary' in locals() else self.use_features_in_secondary,
            "auto_scale_features": auto_scale_features if 'auto_scale_features' in locals() else self.auto_scale_features,
            "n_jobs": n_jobs if 'n_jobs' in locals() else self.n_jobs,
            "estimate_base_importance": estimate_base_importance if 'estimate_base_importance' in locals() else self.estimate_base_importance,
            "cross_validation_folds": cross_validation_folds if 'cross_validation_folds' in locals() else self.cross_validation_folds,
            "random_state": random_state if 'random_state' in locals() else self.random_state,
            "verbose": verbose if 'verbose' in locals() else self.verbose,
            # Base estimator parameters
            "lr_C": lr_C if 'lr_C' in locals() else self.lr_C,
            "lr_max_iter": lr_max_iter if 'lr_max_iter' in locals() else self.lr_max_iter,
            "lr_solver": lr_solver if 'lr_solver' in locals() else self.lr_solver,
            "rf_n_estimators": rf_n_estimators if 'rf_n_estimators' in locals() else self.rf_n_estimators,
            "rf_max_depth": rf_max_depth if 'rf_max_depth' in locals() else self.rf_max_depth,
            "rf_min_samples_split": rf_min_samples_split if 'rf_min_samples_split' in locals() else self.rf_min_samples_split,
            "svm_C": svm_C if 'svm_C' in locals() else self.svm_C,
            "svm_kernel": svm_kernel if 'svm_kernel' in locals() else self.svm_kernel,
            "svm_probability": svm_probability if 'svm_probability' in locals() else self.svm_probability,
            "dt_max_depth": dt_max_depth if 'dt_max_depth' in locals() else self.dt_max_depth,
            "dt_criterion": dt_criterion if 'dt_criterion' in locals() else self.dt_criterion,
            "knn_n_neighbors": knn_n_neighbors if 'knn_n_neighbors' in locals() else self.knn_n_neighbors,
            "knn_weights": knn_weights if 'knn_weights' in locals() else self.knn_weights,
            "gb_n_estimators": gb_n_estimators if 'gb_n_estimators' in locals() else self.gb_n_estimators,
            "gb_learning_rate": gb_learning_rate if 'gb_learning_rate' in locals() else self.gb_learning_rate,
            "gb_max_depth": gb_max_depth if 'gb_max_depth' in locals() else self.gb_max_depth,
            "mlp_hidden_layer_sizes": mlp_hidden_layer_sizes if 'mlp_hidden_layer_sizes' in locals() else self.mlp_hidden_layer_sizes,
            "mlp_activation": mlp_activation if 'mlp_activation' in locals() else self.mlp_activation,
            "xgb_n_estimators": xgb_n_estimators if 'xgb_n_estimators' in locals() else self.xgb_n_estimators,
            "xgb_learning_rate": xgb_learning_rate if 'xgb_learning_rate' in locals() else self.xgb_learning_rate,
            "xgb_max_depth": xgb_max_depth if 'xgb_max_depth' in locals() else self.xgb_max_depth
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return StackingClassifierPlugin(
            base_estimators=hyperparameters.get("base_estimators", self.base_estimators),
            meta_learner=hyperparameters.get("meta_learner", self.meta_learner),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            use_features_in_secondary=hyperparameters.get("use_features_in_secondary", self.use_features_in_secondary),
            stack_method=hyperparameters.get("stack_method", self.stack_method),
            passthrough=hyperparameters.get("passthrough", self.passthrough),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            estimate_base_importance=hyperparameters.get("estimate_base_importance", self.estimate_base_importance),
            cross_validation_folds=hyperparameters.get("cross_validation_folds", self.cross_validation_folds),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            # Base estimator parameters
            lr_C=hyperparameters.get("lr_C", self.lr_C),
            lr_max_iter=hyperparameters.get("lr_max_iter", self.lr_max_iter),
            lr_solver=hyperparameters.get("lr_solver", self.lr_solver),
            rf_n_estimators=hyperparameters.get("rf_n_estimators", self.rf_n_estimators),
            rf_max_depth=hyperparameters.get("rf_max_depth", self.rf_max_depth),
            rf_min_samples_split=hyperparameters.get("rf_min_samples_split", self.rf_min_samples_split),
            svm_C=hyperparameters.get("svm_C", self.svm_C),
            svm_kernel=hyperparameters.get("svm_kernel", self.svm_kernel),
            svm_probability=hyperparameters.get("svm_probability", self.svm_probability),
            dt_max_depth=hyperparameters.get("dt_max_depth", self.dt_max_depth),
            dt_criterion=hyperparameters.get("dt_criterion", self.dt_criterion),
            knn_n_neighbors=hyperparameters.get("knn_n_neighbors", self.knn_n_neighbors),
            knn_weights=hyperparameters.get("knn_weights", self.knn_weights),
            gb_n_estimators=hyperparameters.get("gb_n_estimators", self.gb_n_estimators),
            gb_learning_rate=hyperparameters.get("gb_learning_rate", self.gb_learning_rate),
            gb_max_depth=hyperparameters.get("gb_max_depth", self.gb_max_depth),
            mlp_hidden_layer_sizes=hyperparameters.get("mlp_hidden_layer_sizes", self.mlp_hidden_layer_sizes),
            mlp_activation=hyperparameters.get("mlp_activation", self.mlp_activation),
            xgb_n_estimators=hyperparameters.get("xgb_n_estimators", self.xgb_n_estimators),
            xgb_learning_rate=hyperparameters.get("xgb_learning_rate", self.xgb_learning_rate),
            xgb_max_depth=hyperparameters.get("xgb_max_depth", self.xgb_max_depth)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Stacking Classifier
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Check for missing values
        if np.any(pd.isna(X_processed)):
            warnings.warn("Some base estimators don't handle missing values well. Consider imputation.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Stacking Classifier is compatible with the given data
        """
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Stacking Classifier requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for sufficient samples per CV fold
        min_samples_per_fold = X.shape[0] // self.cv_folds
        if min_samples_per_fold < 5:
            return False, f"Too few samples per CV fold ({min_samples_per_fold}). Reduce cv_folds or increase dataset size."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check class distribution
            min_class_samples = min(np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes))
            samples_per_fold_per_class = min_class_samples // self.cv_folds
            if samples_per_fold_per_class < 1:
                return True, f"Warning: Small classes may cause issues in CV folds. Consider reducing cv_folds."
        
        # Check base estimator count
        if len(self.base_estimators) < 2:
            return False, "Need at least 2 base estimators for stacking"
        elif len(self.base_estimators) > 10:
            return True, "Warning: Many base estimators may cause overfitting. Consider reducing the number."
        
        # Check computational feasibility
        if len(self.base_estimators) * self.cv_folds > 50:
            return True, "Warning: High computational cost due to many estimators and CV folds."
        
        return True, f"Stacking Classifier is compatible! Using {len(self.base_estimators)} base estimators with {self.meta_learner} meta-learner."
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "base_estimators": self.base_estimators,
            "meta_learner": self.meta_learner,
            "n_base_estimators": len(self.base_estimators),
            "cv_folds": self.cv_folds,
            "stack_method": self.stack_method,
            "passthrough": self.passthrough,
            "feature_scaling": self.scaler_ is not None,
            "base_importance_available": self.base_importance_ is not None,
            "ensemble_type": "Stacking Classifier"
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Stacking Classifier",
            "ensemble_type": "Meta-Learning",
            "training_completed": True,
            "stacking_characteristics": {
                "meta_learning": True,
                "cross_validation_based": True,
                "bias_reduction": True,
                "adaptive_combination": True,
                "two_level_architecture": True,
                "interpretable_combination": self.meta_learner == 'logistic_regression'
            },
            "ensemble_configuration": {
                "base_estimators": self.base_estimators,
                "meta_learner": self.meta_learner,
                "n_base_estimators": len(self.base_estimators),
                "cv_folds": self.cv_folds,
                "stack_method": self.stack_method,
                "passthrough": self.passthrough,
                "feature_scaling": self.scaler_ is not None
            },
            "stacking_analysis": self.get_stacking_analysis(),
            "performance_considerations": {
                "training_time": f"High - trains {len(self.base_estimators)} × {self.cv_folds + 1} + meta-learner",
                "prediction_time": f"Moderate - queries {len(self.base_estimators)} estimators + meta-learner",
                "memory_usage": f"High - stores {len(self.base_estimators)} base models + meta-learner",
                "scalability": "Good - base training parallelizable",
                "overfitting_risk": "Medium - controlled by CV and meta-learner regularization",
                "interpretability": "High with linear meta-learner, Medium otherwise"
            },
            "meta_learning_theory": {
                "level_0": "Base estimators learn diverse patterns from data",
                "level_1": "Meta-learner learns optimal combination of base predictions",
                "cv_protection": "Cross-validation prevents meta-learner overfitting",
                "adaptive_weighting": "Learns context-dependent combination weights",
                "bias_variance": "Reduces both bias (adaptation) and variance (diversity)"
            }
        }
        
        # Add base estimator performance if available
        if self.base_estimator_scores_:
            info["base_estimator_performance"] = self.base_estimator_scores_
        
        # Add base importance if available
        if self.base_importance_ is not None:
            info["base_estimator_importance"] = {
                estimator: float(imp) for estimator, imp in zip(self.base_estimators, self.base_importance_)
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Stacking Classifier model.

        These metrics are derived from the model's learned parameters, base estimator
        evaluations, and meta-learner characteristics.
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from the fitted model's internal attributes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values.
        y_pred : np.ndarray, optional
            Predicted target values.
        y_proba : np.ndarray, optional
            Predicted probabilities.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self, 'stacking_classifier_') or self.stacking_classifier_ is None:
            return {"error": "Model not fitted. Cannot retrieve Stacking Classifier specific metrics."}

        metrics = {}
        prefix = "stacking_" # Prefix for Stacking Classifier specific metrics

        # Ensemble Configuration Metrics
        metrics[f"{prefix}num_base_estimators"] = len(self.base_estimators)
        metrics[f"{prefix}cv_folds_for_stacking"] = self.cv_folds
        metrics[f"{prefix}passthrough_original_features"] = 1 if self.passthrough else 0
        metrics[f"{prefix}stack_method_used"] = self.stack_method 
        # Note: self.stack_method is a string, convert to numerical if needed for some logging systems, or keep as string.
        # For now, keeping as string as it's descriptive.

        final_estimator = self.stacking_classifier_.final_estimator_
        if hasattr(final_estimator, 'n_features_in_'):
            metrics[f"{prefix}num_meta_features_to_final_estimator"] = int(final_estimator.n_features_in_)

        # Base Estimator Importance (from meta-learner's perspective)
        if hasattr(self, 'base_importance_') and self.base_importance_ is not None and len(self.base_importance_) > 0:
            metrics[f"{prefix}mean_base_estimator_importance_in_meta"] = float(np.mean(self.base_importance_))
            metrics[f"{prefix}max_base_estimator_importance_in_meta"] = float(np.max(self.base_importance_))
            metrics[f"{prefix}std_base_estimator_importance_in_meta"] = float(np.std(self.base_importance_))
            metrics[f"{prefix}num_base_estimators_with_importance"] = len(self.base_importance_)

        # Individual Base Estimator Performance (CV accuracy)
        if hasattr(self, 'base_estimator_scores_') and self.base_estimator_scores_:
            all_mean_cv_scores = [
                data['mean_cv_score'] for data in self.base_estimator_scores_.values() if 'mean_cv_score' in data
            ]
            if all_mean_cv_scores:
                metrics[f"{prefix}mean_cv_accuracy_all_base_estimators"] = float(np.mean(all_mean_cv_scores))
                metrics[f"{prefix}std_cv_accuracy_all_base_estimators"] = float(np.std(all_mean_cv_scores))
                metrics[f"{prefix}min_cv_accuracy_among_base_estimators"] = float(np.min(all_mean_cv_scores))
                metrics[f"{prefix}max_cv_accuracy_among_base_estimators"] = float(np.max(all_mean_cv_scores))

        # Meta-Learner (Final Estimator) Parameters
        if hasattr(final_estimator, 'coef_'):
            coef = final_estimator.coef_
            metrics[f"{prefix}meta_learner_coef_l2_norm"] = float(np.linalg.norm(coef))
            metrics[f"{prefix}meta_learner_mean_abs_coef"] = float(np.mean(np.abs(coef)))
        
        if hasattr(final_estimator, 'intercept_'):
            intercept = final_estimator.intercept_
            # Intercept can be an array for multi-class. Report mean if array, else the value.
            if isinstance(intercept, (np.ndarray, list)) and len(intercept) > 1:
                 metrics[f"{prefix}meta_learner_mean_abs_intercept"] = float(np.mean(np.abs(intercept)))
            elif isinstance(intercept, (np.ndarray, list)): # Single element array
                 metrics[f"{prefix}meta_learner_intercept"] = float(intercept[0])
            else: # Scalar
                 metrics[f"{prefix}meta_learner_intercept"] = float(intercept)

        if hasattr(final_estimator, 'n_iter_'):
            n_iter_ = final_estimator.n_iter_
            # n_iter_ can be an array for multi-class sag/saga. Report sum or mean.
            if isinstance(n_iter_, (np.ndarray, list)):
                metrics[f"{prefix}meta_learner_n_iter_sum"] = int(np.sum(n_iter_))
            else:
                metrics[f"{prefix}meta_learner_n_iter"] = int(n_iter_)
        
        if not metrics:
            metrics['info'] = "No specific Stacking Classifier metrics were available from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return StackingClassifierPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Stacking Classifier Plugin
    """
    print("Testing Stacking Classifier Plugin...")
    
    try:
        print("✅ Required libraries are available")
        
        # Create sample data for stacking
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Generate a complex dataset suitable for demonstrating stacking benefits
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=3,
            n_clusters_per_class=2,
            class_sep=0.7,
            flip_y=0.05,
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
        
        # Create and test stacking plugin with diverse base estimators
        plugin = StackingClassifierPlugin(
            base_estimators=['logistic_regression', 'random_forest', 'svm', 'gradient_boosting'],
            meta_learner='logistic_regression',
            cv_folds=5,
            stack_method='auto',
            passthrough=False,
            auto_scale_features=True,
            estimate_base_importance=True,
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
            # Train stacking ensemble
            print("\n🚀 Training Stacking Classifier...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            y_proba = plugin.predict_proba(X_test)
            
            # Evaluate ensemble
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n📊 Stacking Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Classes: {plugin.classes_}")
            
            # Compare with individual base estimators
            base_predictions = plugin.get_base_predictions(X_test)
            base_accuracies = {}
            
            for estimator_name, pred in base_predictions.items():
                base_acc = accuracy_score(y_test, pred)
                base_accuracies[estimator_name] = base_acc
                print(f"{estimator_name}: {base_acc:.4f}")
            
            print(f"\n🏗️ Ensemble vs Base Estimators:")
            best_base_accuracy = max(base_accuracies.values())
            print(f"Best base estimator: {best_base_accuracy:.4f}")
            print(f"Stacking ensemble: {accuracy:.4f}")
            print(f"Improvement: {accuracy - best_base_accuracy:.4f} ({(accuracy/best_base_accuracy-1)*100:.1f}%)")
            
            # Analyze meta-features
            meta_features = plugin.get_meta_features(X_test)
            print(f"\n🧠 Meta-Learning Analysis:")
            print(f"Meta-features shape: {meta_features.shape}")
            
            # Get stacking analysis
            stacking_analysis = plugin.get_stacking_analysis()
            print(f"\n🏗️ Stacking Analysis:")
            
            ensemble_summary = stacking_analysis.get('ensemble_summary', {})
            print(f"Base estimators: {ensemble_summary.get('base_estimators', [])}")
            print(f"Meta-learner: {ensemble_summary.get('meta_learner', 'Unknown')}")
            print(f"CV folds: {ensemble_summary.get('cv_folds', 'Unknown')}")
            print(f"Stack method: {ensemble_summary.get('stack_method', 'Unknown')}")
            
            # Base estimator importance
            if 'base_estimator_importance' in stacking_analysis:
                importance_info = stacking_analysis['base_estimator_importance']
                print(f"\n🎯 Base Estimator Importance:")
                for estimator, importance in importance_info['importances'].items():
                    print(f"{estimator}: {importance:.4f}")
                
                print(f"\nTop 3 estimators: {list(importance_info['top_3_estimators'].keys())}")
            
            # Base estimator performance
            if 'base_estimator_performance' in stacking_analysis:
                perf_info = stacking_analysis['base_estimator_performance']
                print(f"\n📈 Base Estimator CV Performance:")
                for estimator, scores in perf_info.items():
                    print(f"{estimator}: {scores['mean_cv_score']:.4f} ± {scores['std_cv_score']:.4f}")
            
            # Model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Ensemble type: {model_params.get('ensemble_type', 'Unknown')}")
            print(f"Base estimators: {model_params.get('n_base_estimators', 'Unknown')}")
            print(f"Meta-learner: {model_params.get('meta_learner', 'Unknown')}")
            print(f"CV folds: {model_params.get('cv_folds', 'Unknown')}")
            print(f"Feature scaling: {model_params.get('feature_scaling', False)}")
            
            # Training info
            training_info = plugin.get_training_info()
            print(f"\n📈 Training Info:")
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Ensemble type: {training_info['ensemble_type']}")
            
            stacking_chars = training_info['stacking_characteristics']
            print(f"Meta-learning: {stacking_chars['meta_learning']}")
            print(f"Cross-validation based: {stacking_chars['cross_validation_based']}")
            print(f"Bias reduction: {stacking_chars['bias_reduction']}")
            print(f"Adaptive combination: {stacking_chars['adaptive_combination']}")
            
            # Performance considerations
            perf_info = training_info['performance_considerations']
            print(f"\n⚡ Performance Considerations:")
            print(f"Training time: {perf_info['training_time']}")
            print(f"Prediction time: {perf_info['prediction_time']}")
            print(f"Memory usage: {perf_info['memory_usage']}")
            print(f"Overfitting risk: {perf_info['overfitting_risk']}")
            
            # Meta-learning theory
            meta_theory = training_info['meta_learning_theory']
            print(f"\n🧠 Meta-Learning Theory:")
            print(f"Level 0: {meta_theory['level_0']}")
            print(f"Level 1: {meta_theory['level_1']}")
            print(f"CV protection: {meta_theory['cv_protection']}")
            
            print("\n✅ Stacking Classifier Plugin test completed successfully!")
            print("🏗️ Meta-learning successfully combined diverse base estimators!")
            
            # Demonstrate stacking benefits
            print(f"\n🚀 Stacking Benefits:")
            print(f"Adaptive Combination: Meta-learner learns optimal weights")
            print(f"Bias Reduction: Corrects systematic errors of base estimators")
            print(f"CV Protection: Prevents meta-learner overfitting")
            print(f"Flexibility: Works with any base estimator combination")
            
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