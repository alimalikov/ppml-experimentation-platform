import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Try to import LightGBM with graceful fallback
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

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

class LightGBMClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    LightGBM Classifier Plugin - Fast and Memory Efficient Gradient Boosting
    
    LightGBM is a gradient boosting framework that uses tree-based learning 
    algorithms. It's designed to be distributed and efficient with faster 
    training speed and higher efficiency than other boosting frameworks.
    """
    
    def __init__(self, 
                 boosting_type='gbdt',
                 objective='multiclass',
                 num_class=None,
                 metric='multi_logloss',
                 num_leaves=31,
                 max_depth=-1,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample_for_bin=200000,
                 min_split_gain=0.0,
                 min_child_weight=1e-3,
                 min_child_samples=20,
                 subsample=1.0,
                 subsample_freq=0,
                 colsample_bytree=1.0,
                 reg_alpha=0.0,
                 reg_lambda=0.0,
                 random_state=42,
                 n_jobs=-1,
                 silent='warn',
                 importance_type='split',
                 # LightGBM specific parameters
                 max_bin=255,
                 min_data_in_bin=3,
                 feature_fraction=1.0,
                 feature_fraction_bynode=1.0,
                 bagging_fraction=1.0,
                 bagging_freq=0,
                 lambda_l1=0.0,
                 lambda_l2=0.0,
                 min_gain_to_split=0.0,
                 drop_rate=0.1,
                 max_drop=50,
                 skip_drop=0.5,
                 xgboost_dart_mode=False,
                 uniform_drop=False,
                 top_rate=0.2,
                 other_rate=0.1,
                 min_data_per_group=100,
                 max_cat_threshold=32,
                 cat_l2=10.0,
                 cat_smooth=10.0,
                 max_cat_to_onehot=4,
                 cegb_tradeoff=1.0,
                 cegb_penalty_split=0.0,
                 path_smooth=0.0,
                 # Training control
                 early_stopping_rounds=None,
                 feature_pre_filter=True,
                 linear_tree=False,
                 monotone_constraints=None,
                 monotone_constraints_method='basic',
                 monotone_penalty=0.0,
                 interaction_constraints=None,
                 verbosity=-1,
                 seed=42,
                 deterministic=False):
        """
        Initialize LightGBM Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        boosting_type : str, default='gbdt'
            Boosting type: 'gbdt', 'dart', 'goss', 'rf'
        objective : str, default='multiclass'
            Objective function: 'binary', 'multiclass', 'multiclassova'
        num_class : int, default=None
            Number of classes (automatically determined)
        metric : str or list, default='multi_logloss'
            Evaluation metric(s)
        num_leaves : int, default=31
            Maximum number of leaves in one tree
        max_depth : int, default=-1
            Maximum depth of tree (-1 means no limit)
        learning_rate : float, default=0.1
            Boosting learning rate
        n_estimators : int, default=100
            Number of boosting iterations
        subsample_for_bin : int, default=200000
            Number of samples for constructing bins
        min_split_gain : float, default=0.0
            Minimum gain to make a split
        min_child_weight : float, default=1e-3
            Minimum sum of instance weight (hessian) needed in a child
        min_child_samples : int, default=20
            Minimum number of data points in a leaf
        subsample : float, default=1.0
            Subsample ratio of the training instance
        subsample_freq : int, default=0
            Frequency for bagging
        colsample_bytree : float, default=1.0
            Subsample ratio of columns when constructing each tree
        reg_alpha : float, default=0.0
            L1 regularization term
        reg_lambda : float, default=0.0
            L2 regularization term
        random_state : int, default=42
            Random seed
        n_jobs : int, default=-1
            Number of parallel threads
        silent : str, default='warn'
            Whether to print messages while running boosting
        importance_type : str, default='split'
            Feature importance type: 'split' or 'gain'
        """
        super().__init__()
        
        # Core parameters
        self.boosting_type = boosting_type
        self.objective = objective
        self.num_class = num_class
        self.metric = metric
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        
        # LightGBM specific parameters
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.drop_rate = drop_rate
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.uniform_drop = uniform_drop
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.min_data_per_group = min_data_per_group
        self.max_cat_threshold = max_cat_threshold
        self.cat_l2 = cat_l2
        self.cat_smooth = cat_smooth
        self.max_cat_to_onehot = max_cat_to_onehot
        self.cegb_tradeoff = cegb_tradeoff
        self.cegb_penalty_split = cegb_penalty_split
        self.path_smooth = path_smooth
        
        # Training control
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_pre_filter = feature_pre_filter
        self.linear_tree = linear_tree
        self.monotone_constraints = monotone_constraints
        self.monotone_constraints_method = monotone_constraints_method
        self.monotone_penalty = monotone_penalty
        self.interaction_constraints = interaction_constraints
        self.verbosity = verbosity
        self.seed = seed
        self.deterministic = deterministic
        
        # Plugin metadata
        self._name = "LightGBM"
        self._description = "Fast and memory efficient gradient boosting framework with optimized tree learning algorithms."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Gradient Boosting Classifier"
        self._paper_reference = "Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NIPS."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = True
        self._requires_scaling = False
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True
        self._ensemble_method = True
        self._supports_early_stopping = True
        self._supports_gpu = True
        self._memory_efficient = True
        self._fast_training = True
        self._industry_grade = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.categorical_features_ = None
        self.eval_results_ = None
        
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
            "framework": "Microsoft LightGBM",
            "key_innovations": {
                "exclusive_feature_bundling": "Bundles sparse features for efficiency",
                "gradient_based_one_side_sampling": "Samples data points based on gradients",
                "leaf_wise_tree_growth": "Grows trees leaf-wise instead of level-wise",
                "optimized_memory_usage": "Reduced memory consumption vs other GBMs",
                "native_categorical_support": "Handles categorical features without encoding",
                "network_communication_optimization": "Efficient distributed training"
            },
            "strengths": [
                "Extremely fast training speed",
                "Low memory usage",
                "Better accuracy than XGBoost in many cases",
                "Native categorical feature handling",
                "Built-in missing value handling",
                "GPU acceleration support",
                "Efficient sparse feature handling",
                "Early stopping with multiple metrics",
                "Advanced sampling techniques (GOSS)",
                "Feature bundling (EFB) for high-dimensional data",
                "Distributed training capabilities",
                "Multiple boosting strategies (GBDT, DART, GOSS, RF)"
            ],
            "weaknesses": [
                "Can overfit with small datasets",
                "Sensitive to hyperparameters",
                "Newer framework (less mature than XGBoost)",
                "May require careful tuning for optimal performance",
                "Can be sensitive to outliers",
                "Documentation less extensive than XGBoost"
            ],
            "use_cases": [
                "Large-scale machine learning",
                "Time-sensitive training scenarios",
                "High-dimensional datasets",
                "Categorical-heavy datasets",
                "Memory-constrained environments",
                "Real-time model training",
                "Kaggle competitions (increasingly popular)",
                "Industrial applications requiring speed",
                "Financial modeling and risk assessment",
                "Recommendation systems",
                "Click-through rate prediction",
                "Feature-rich tabular data problems"
            ],
            "algorithmic_details": {
                "tree_growth": "Leaf-wise (vs. level-wise in XGBoost)",
                "feature_selection": "Exclusive Feature Bundling (EFB)",
                "data_sampling": "Gradient-based One-Side Sampling (GOSS)",
                "memory_optimization": "Histogram-based algorithms",
                "categorical_handling": "Native support without pre-processing",
                "missing_values": "Built-in optimal split finding",
                "parallelization": "Feature parallelization and data parallelization"
            },
            "performance_characteristics": {
                "training_speed": "Fastest among major GBM frameworks",
                "memory_usage": "Lowest among major GBM frameworks",
                "accuracy": "Competitive with or better than XGBoost",
                "gpu_acceleration": "Excellent GPU support",
                "distributed_training": "Efficient network communication"
            },
            "comparison_with_competitors": {
                "vs_xgboost": {
                    "speed": "2-10x faster training",
                    "memory": "Lower memory usage",
                    "accuracy": "Often better, especially on categorical data",
                    "ecosystem": "Smaller but growing"
                },
                "vs_catboost": {
                    "categorical_handling": "Good but CatBoost specializes in this",
                    "speed": "Generally faster",
                    "overfitting": "Both resistant but CatBoost more so"
                },
                "vs_sklearn_gbm": {
                    "speed": "Much faster",
                    "features": "More advanced features",
                    "scalability": "Much better for large datasets"
                }
            },
            "hyperparameter_guide": {
                "num_leaves": "31-300, most important parameter for controlling overfitting",
                "learning_rate": "0.01-0.3, lower for better generalization",
                "n_estimators": "100-10000, use early stopping to find optimal",
                "max_depth": "-1 (no limit) or 6-15 for controlling overfitting",
                "min_child_samples": "20-100, higher values prevent overfitting",
                "subsample": "0.8-1.0, lower values for regularization",
                "colsample_bytree": "0.8-1.0, feature subsampling",
                "reg_alpha": "0-10, L1 regularization",
                "reg_lambda": "0-10, L2 regularization"
            },
            "boosting_types": {
                "gbdt": "Traditional gradient boosting (default)",
                "dart": "Dropouts meet Multiple Additive Regression Trees",
                "goss": "Gradient-based One-Side Sampling",
                "rf": "Random Forest mode"
            }
        }
    
    def fit(self, X, y, 
            eval_set=None, 
            eval_names=None, 
            eval_metric=None,
            sample_weight=None,
            init_score=None,
            feature_name='auto',
            categorical_feature='auto',
            callbacks=None,
            init_model=None):
        """
        Fit the LightGBM Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        eval_set : list of tuples, optional
            A list of (X, y) tuple pairs to use as validation sets
        eval_names : list of strings, optional
            Names of eval_set
        eval_metric : str or callable, optional
            Evaluation metric
        sample_weight : array-like, optional
            Sample weights
        init_score : array-like, optional
            Initial score
        feature_name : list of strings or 'auto', default='auto'
            Feature names
        categorical_feature : list of strings or int, or 'auto', default='auto'
            Categorical features
        callbacks : list of callables, optional
            List of callback functions
        init_model : str or lgb.Booster, optional
            Filename of init model or lgb.Booster instance
            
        Returns:
        --------
        self : object
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install with: pip install lightgbm")
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        
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
        
        # Set up objective and num_class based on problem type
        if n_classes == 2:
            objective = 'binary'
            num_class = None
            if self.metric == 'multi_logloss':
                metric = 'binary_logloss'
            else:
                metric = self.metric
        else:
            objective = 'multiclass'
            num_class = n_classes
            metric = self.metric
        
        # Build parameters dictionary
        params = {
            'boosting_type': self.boosting_type,
            'objective': objective,
            'metric': metric,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample_for_bin': self.subsample_for_bin,
            'min_split_gain': self.min_split_gain,
            'min_child_weight': self.min_child_weight,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': self.verbosity,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'importance_type': self.importance_type,
            
            # LightGBM specific
            'max_bin': self.max_bin,
            'min_data_in_bin': self.min_data_in_bin,
            'feature_fraction': self.feature_fraction,
            'feature_fraction_bynode': self.feature_fraction_bynode,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'min_gain_to_split': self.min_gain_to_split,
            'max_cat_threshold': self.max_cat_threshold,
            'cat_l2': self.cat_l2,
            'cat_smooth': self.cat_smooth,
            'max_cat_to_onehot': self.max_cat_to_onehot,
            'feature_pre_filter': self.feature_pre_filter,
            'linear_tree': self.linear_tree
        }
        
        # Add num_class for multiclass
        if num_class is not None:
            params['num_class'] = num_class
        
        # Add DART specific parameters
        if self.boosting_type == 'dart':
            params.update({
                'drop_rate': self.drop_rate,
                'max_drop': self.max_drop,
                'skip_drop': self.skip_drop,
                'xgboost_dart_mode': self.xgboost_dart_mode,
                'uniform_drop': self.uniform_drop
            })
        
        # Add GOSS specific parameters
        if self.boosting_type == 'goss':
            params.update({
                'top_rate': self.top_rate,
                'other_rate': self.other_rate
            })
        
        # Add monotone constraints if specified
        if self.monotone_constraints is not None:
            params['monotone_constraints'] = self.monotone_constraints
            params['monotone_constraints_method'] = self.monotone_constraints_method
            params['monotone_penalty'] = self.monotone_penalty
        
        # Add interaction constraints if specified
        if self.interaction_constraints is not None:
            params['interaction_constraints'] = self.interaction_constraints
        
        # Create LightGBM dataset
        if isinstance(categorical_feature, str) and categorical_feature == 'auto':
            # Auto-detect categorical features
            categorical_feature = self._detect_categorical_features(X)
        
        self.categorical_features_ = categorical_feature
        
        train_data = lgb.Dataset(
            X, 
            label=y_encoded,
            weight=sample_weight,
            init_score=init_score,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            params=params,
            free_raw_data=False
        )
        
        # Prepare validation sets
        valid_sets = [train_data]
        valid_names = ['training']
        
        if eval_set is not None:
            for i, (X_val, y_val) in enumerate(eval_set):
                # Encode validation labels
                y_val_encoded = self.label_encoder_.transform(y_val)
                
                valid_data = lgb.Dataset(
                    X_val,
                    label=y_val_encoded,
                    reference=train_data,
                    categorical_feature=categorical_feature
                )
                valid_sets.append(valid_data)
                
                if eval_names is not None and i < len(eval_names):
                    valid_names.append(eval_names[i])
                else:
                    valid_names.append(f'validation_{i}')
        
        # Set up callbacks
        callback_list = []
        if callbacks is not None:
            callback_list.extend(callbacks)
        
        # Early stopping
        if self.early_stopping_rounds is not None and len(valid_sets) > 1:
            callback_list.append(lgb.early_stopping(self.early_stopping_rounds))
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            evals_result = {}
            self.model_ = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callback_list,
                evals_result=evals_result,
                init_model=init_model
            )
            
            self.eval_results_ = evals_result
        
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install with: pip install lightgbm")
        
        # Get probabilities and convert to class predictions
        probabilities = self.model_.predict(X, num_iteration=self.model_.best_iteration)
        
        if len(self.classes_) == 2:
            # Binary classification
            y_pred_encoded = (probabilities > 0.5).astype(int)
        else:
            # Multi-class classification
            y_pred_encoded = np.argmax(probabilities, axis=1)
        
        # Decode labels back to original format
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install with: pip install lightgbm")
        
        probabilities = self.model_.predict(X, num_iteration=self.model_.best_iteration)
        
        if len(self.classes_) == 2:
            # Binary classification - convert to 2D array
            prob_positive = probabilities
            prob_negative = 1 - prob_positive
            probabilities = np.column_stack([prob_negative, prob_positive])
        
        return probabilities
    
    def _detect_categorical_features(self, X):
        """
        Auto-detect categorical features in the dataset
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        categorical_features : list
            List of categorical feature indices or names
        """
        if hasattr(X, 'dtypes'):
            # DataFrame with dtype information
            categorical_features = []
            for i, dtype in enumerate(X.dtypes):
                if dtype == 'object' or dtype.name == 'category':
                    categorical_features.append(i)
            return categorical_features
        else:
            # NumPy array - no automatic detection
            return 'auto'
    
    def get_feature_importance(self, importance_type=None):
        """
        Get feature importance
        
        Parameters:
        -----------
        importance_type : str, optional
            Type of importance: 'split' or 'gain'
            
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
        
        if importance_type is None:
            importance_type = self.importance_type
            
        return self.model_.feature_importance(importance_type=importance_type)
    
    def get_evaluation_results(self):
        """
        Get evaluation results from training
        
        Returns:
        --------
        eval_results : dict
            Evaluation results for each dataset and metric
        """
        if not self.is_fitted_:
            return None
            
        return self.eval_results_
    
    def get_best_iteration(self):
        """
        Get the best iteration number if early stopping was used
        
        Returns:
        --------
        best_iteration : int
            Best iteration number
        """
        if not self.is_fitted_:
            return None
            
        return getattr(self.model_, 'best_iteration', self.model_.num_trees())
    
    def get_lgb_analysis(self) -> Dict[str, Any]:
        """
        Analyze LightGBM specific features and optimizations
        
        Returns:
        --------
        analysis_info : dict
            Information about LightGBM optimizations and performance
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "lightgbm_version": lgb.__version__ if LIGHTGBM_AVAILABLE else "Not available",
            "boosting_type": self.boosting_type,
            "tree_learner": "leaf-wise" if self.boosting_type != 'rf' else "random forest",
            "num_trees": self.model_.num_trees(),
            "best_iteration": self.get_best_iteration(),
            "early_stopping_used": self.early_stopping_rounds is not None,
            "categorical_features_used": self.categorical_features_ is not None,
            "memory_optimizations": {
                "max_bin": self.max_bin,
                "subsample_for_bin": self.subsample_for_bin,
                "feature_bundling": "Enabled (EFB)",
                "sparse_optimization": "Enabled",
                "histogram_based": "Enabled"
            },
            "sampling_techniques": {
                "gradient_based_sampling": self.boosting_type == 'goss',
                "random_sampling": self.subsample < 1.0,
                "feature_sampling": self.colsample_bytree < 1.0
            },
            "regularization": {
                "l1_regularization": self.reg_alpha > 0,
                "l2_regularization": self.reg_lambda > 0,
                "min_child_samples": self.min_child_samples,
                "min_split_gain": self.min_split_gain
            },
            "performance_features": {
                "parallel_training": self.n_jobs != 1,
                "gpu_training": False,  # Would need GPU-specific parameters
                "categorical_native": self.categorical_features_ is not None,
                "missing_value_handling": "Native support"
            }
        }
        
        # Add evaluation results if available
        if self.eval_results_:
            analysis["evaluation_history"] = self.eval_results_
        
        # Add DART specific info
        if self.boosting_type == 'dart':
            analysis["dart_parameters"] = {
                "drop_rate": self.drop_rate,
                "max_drop": self.max_drop,
                "skip_drop": self.skip_drop,
                "uniform_drop": self.uniform_drop
            }
        
        # Add GOSS specific info
        if self.boosting_type == 'goss':
            analysis["goss_parameters"] = {
                "top_rate": self.top_rate,
                "other_rate": self.other_rate,
                "description": "Gradient-based One-Side Sampling for efficiency"
            }
        
        return analysis
    
    def plot_feature_importance(self, max_features=20, importance_type=None, figsize=(10, 8)):
        """
        Create a feature importance plot
        
        Parameters:
        -----------
        max_features : int, default=20
            Maximum number of features to display
        importance_type : str, optional
            Type of importance: 'split' or 'gain'
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Feature importance plot
        """
        if not self.is_fitted_:
            return None
        
        importance = self.get_feature_importance(importance_type)
        if importance is None:
            return None
        
        # Get top features
        indices = np.argsort(importance)[::-1][:max_features]
        top_features = [self.feature_names_[i] for i in indices]
        top_importance = importance[indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(top_features)), top_importance, 
                      color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel(f'Feature Importance ({importance_type or self.importance_type})')
        ax.set_title(f'Top {len(top_features)} Feature Importances - LightGBM')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(top_importance) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.0f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, figsize=(12, 8)):
        """
        Plot learning curves from evaluation results
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Learning curves plot
        """
        if not self.is_fitted_ or not self.eval_results_:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Plot metrics for each dataset
        colors = ['blue', 'red', 'green', 'orange']
        datasets = list(self.eval_results_.keys())
        
        for i, (dataset, metrics) in enumerate(self.eval_results_.items()):
            color = colors[i % len(colors)]
            
            for j, (metric, values) in enumerate(metrics.items()):
                ax = axes[min(j, 3)]  # Use first 4 subplots
                
                iterations = range(1, len(values) + 1)
                ax.plot(iterations, values, label=f'{dataset}', 
                       color=color, alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Iterations')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Learning Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Mark best iteration if available
                best_iter = self.get_best_iteration()
                if best_iter and best_iter <= len(values):
                    ax.axvline(x=best_iter, color='green', linestyle='--', 
                             alpha=0.7, label=f'Best Iteration ({best_iter})')
        
        plt.tight_layout()
        return fig
    
    def plot_lgb_analysis(self, figsize=(12, 8)):
        """
        Create LightGBM-specific analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            LightGBM analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        analysis = self.get_lgb_analysis()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Boosting type comparison
        boosting_types = ['GBDT', 'DART', 'GOSS', 'RF']
        performance_scores = [8, 7, 9, 6]  # Example relative scores
        memory_scores = [7, 6, 9, 8]
        speed_scores = [8, 6, 10, 9]
        
        x = np.arange(len(boosting_types))
        width = 0.25
        
        ax1.bar(x - width, performance_scores, width, label='Performance', alpha=0.8)
        ax1.bar(x, memory_scores, width, label='Memory Efficiency', alpha=0.8)
        ax1.bar(x + width, speed_scores, width, label='Speed', alpha=0.8)
        
        ax1.set_xlabel('Boosting Type')
        ax1.set_ylabel('Score (1-10)')
        ax1.set_title('LightGBM Boosting Types Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(boosting_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Current model configuration
        config_features = ['Native Categorical', 'Early Stopping', 'Regularization', 'Feature Sampling']
        config_values = [
            1 if analysis.get('categorical_features_used') else 0,
            1 if analysis.get('early_stopping_used') else 0,
            1 if analysis['regularization']['l1_regularization'] or analysis['regularization']['l2_regularization'] else 0,
            1 if analysis['sampling_techniques']['feature_sampling'] else 0
        ]
        
        colors = ['green' if val else 'lightcoral' for val in config_values]
        ax2.bar(config_features, config_values, color=colors, alpha=0.7)
        ax2.set_ylabel('Enabled (1) / Disabled (0)')
        ax2.set_title('Current Model Configuration')
        ax2.set_ylim(0, 1.2)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Memory optimizations
        memory_opts = ['Max Bin', 'Feature Bundling', 'Sparse Opt', 'Histogram']
        memory_impact = [8, 9, 7, 9]  # Relative impact scores
        
        bars = ax3.barh(memory_opts, memory_impact, color='lightblue', alpha=0.8)
        ax3.set_xlabel('Memory Optimization Impact')
        ax3.set_title('LightGBM Memory Optimizations')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width}/10', ha='left', va='center')
        
        # Training progress (if evaluation results available)
        if analysis.get('evaluation_history'):
            # Plot the first metric from the first dataset
            eval_data = analysis['evaluation_history']
            if eval_data:
                first_dataset = list(eval_data.keys())[0]
                first_metric = list(eval_data[first_dataset].keys())[0]
                values = eval_data[first_dataset][first_metric]
                
                iterations = range(1, len(values) + 1)
                ax4.plot(iterations, values, 'b-', linewidth=2, alpha=0.8)
                ax4.set_xlabel('Iterations')
                ax4.set_ylabel(first_metric.replace('_', ' ').title())
                ax4.set_title(f'Training Progress - {first_metric}')
                ax4.grid(True, alpha=0.3)
                
                # Mark best iteration
                best_iter = analysis.get('best_iteration')
                if best_iter:
                    ax4.axvline(x=best_iter, color='red', linestyle='--', 
                               alpha=0.7, label=f'Best: {best_iter}')
                    ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No evaluation\nhistory available', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, alpha=0.6)
            ax4.set_title('Training Progress')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### ⚡ LightGBM Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Tree", "Sampling", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Boosting type
            boosting_type = st.selectbox(
                "Boosting Type:",
                options=['gbdt', 'dart', 'goss', 'rf'],
                index=['gbdt', 'dart', 'goss', 'rf'].index(self.boosting_type),
                help="gbdt: Traditional, dart: Dropout, goss: Gradient sampling, rf: Random forest",
                key=f"{key_prefix}_boosting_type"
            )
            
            # Learning rate
            learning_rate = st.number_input(
                "Learning Rate:",
                value=float(self.learning_rate),
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                help="Shrinkage rate for preventing overfitting",
                key=f"{key_prefix}_learning_rate"
            )
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Estimators:",
                min_value=10,
                max_value=2000,
                value=int(self.n_estimators),
                step=10,
                help="Number of boosting iterations",
                key=f"{key_prefix}_n_estimators"
            )
            
            # Number of leaves
            num_leaves = st.slider(
                "Number of Leaves:",
                min_value=10,
                max_value=500,
                value=int(self.num_leaves),
                help="Maximum number of leaves in one tree (most important parameter)",
                key=f"{key_prefix}_num_leaves"
            )
            
            # Max depth
            max_depth_enabled = st.checkbox(
                "Limit Tree Depth",
                value=self.max_depth > 0,
                help="Limit tree depth (-1 means no limit)",
                key=f"{key_prefix}_max_depth_enabled"
            )
            
            if max_depth_enabled:
                max_depth = st.slider(
                    "Max Depth:",
                    min_value=1,
                    max_value=20,
                    value=int(self.max_depth) if self.max_depth > 0 else 6,
                    help="Maximum depth of trees",
                    key=f"{key_prefix}_max_depth"
                )
            else:
                max_depth = -1
        
        with tab2:
            st.markdown("**Tree Parameters**")
            
            # Min child samples
            min_child_samples = st.slider(
                "Min Child Samples:",
                min_value=1,
                max_value=100,
                value=int(self.min_child_samples),
                help="Minimum number of data points in a leaf",
                key=f"{key_prefix}_min_child_samples"
            )
            
            # Min split gain
            min_split_gain = st.number_input(
                "Min Split Gain:",
                value=float(self.min_split_gain),
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                format="%.4f",
                help="Minimum gain to make a split",
                key=f"{key_prefix}_min_split_gain"
            )
            
            # Min child weight
            min_child_weight = st.number_input(
                "Min Child Weight:",
                value=float(self.min_child_weight),
                min_value=1e-6,
                max_value=1.0,
                step=1e-4,
                format="%.4f",
                help="Minimum sum of instance weight needed in a child",
                key=f"{key_prefix}_min_child_weight"
            )
            
            # Max bin
            max_bin = st.slider(
                "Max Bin:",
                min_value=10,
                max_value=1000,
                value=int(self.max_bin),
                help="Maximum number of bins for feature values",
                key=f"{key_prefix}_max_bin"
            )
            
            # Categorical features
            max_cat_to_onehot = st.slider(
                "Max Cat to One-Hot:",
                min_value=1,
                max_value=20,
                value=int(self.max_cat_to_onehot),
                help="Maximum cardinality for one-hot encoding of categorical features",
                key=f"{key_prefix}_max_cat_to_onehot"
            )
        
        with tab3:
            st.markdown("**Sampling Parameters**")
            
            # Subsample
            subsample = st.slider(
                "Subsample:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.subsample),
                step=0.05,
                help="Subsample ratio of the training instance",
                key=f"{key_prefix}_subsample"
            )
            
            # Subsample frequency
            subsample_freq = st.slider(
                "Subsample Frequency:",
                min_value=0,
                max_value=10,
                value=int(self.subsample_freq),
                help="Frequency for bagging (0 means disable bagging)",
                key=f"{key_prefix}_subsample_freq"
            )
            
            # Column sampling
            colsample_bytree = st.slider(
                "Feature Fraction:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.colsample_bytree),
                step=0.05,
                help="Subsample ratio of columns when constructing each tree",
                key=f"{key_prefix}_colsample_bytree"
            )
            
            # GOSS parameters (only if GOSS is selected)
            if boosting_type == 'goss':
                st.markdown("**GOSS Parameters**")
                
                top_rate = st.slider(
                    "Top Rate:",
                    min_value=0.1,
                    max_value=0.5,
                    value=float(self.top_rate),
                    step=0.05,
                    help="Retain ratio of large gradient data",
                    key=f"{key_prefix}_top_rate"
                )
                
                other_rate = st.slider(
                    "Other Rate:",
                    min_value=0.05,
                    max_value=0.3,
                    value=float(self.other_rate),
                    step=0.05,
                    help="Retain ratio of small gradient data",
                    key=f"{key_prefix}_other_rate"
                )
            else:
                top_rate = self.top_rate
                other_rate = self.other_rate
            
            # DART parameters (only if DART is selected)
            if boosting_type == 'dart':
                st.markdown("**DART Parameters**")
                
                drop_rate = st.slider(
                    "Drop Rate:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.drop_rate),
                    step=0.05,
                    help="Dropout rate for DART",
                    key=f"{key_prefix}_drop_rate"
                )
                
                max_drop = st.slider(
                    "Max Drop:",
                    min_value=1,
                    max_value=100,
                    value=int(self.max_drop),
                    help="Maximum number of dropped trees during one boosting iteration",
                    key=f"{key_prefix}_max_drop"
                )
                
                skip_drop = st.slider(
                    "Skip Drop:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.skip_drop),
                    step=0.05,
                    help="Probability of skipping the dropout procedure",
                    key=f"{key_prefix}_skip_drop"
                )
            else:
                drop_rate = self.drop_rate
                max_drop = self.max_drop
                skip_drop = self.skip_drop
        
        with tab4:
            st.markdown("**Advanced Parameters**")
            
            # Regularization
            reg_alpha = st.number_input(
                "L1 Regularization (Alpha):",
                value=float(self.reg_alpha),
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                help="L1 regularization term",
                key=f"{key_prefix}_reg_alpha"
            )
            
            reg_lambda = st.number_input(
                "L2 Regularization (Lambda):",
                value=float(self.reg_lambda),
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                help="L2 regularization term",
                key=f"{key_prefix}_reg_lambda"
            )
            
            # Early stopping
            early_stopping_enabled = st.checkbox(
                "Enable Early Stopping",
                value=self.early_stopping_rounds is not None,
                help="Stop when validation score stops improving",
                key=f"{key_prefix}_early_stopping_enabled"
            )
            
            if early_stopping_enabled:
                early_stopping_rounds = st.slider(
                    "Early Stopping Rounds:",
                    min_value=5,
                    max_value=100,
                    value=int(self.early_stopping_rounds) if self.early_stopping_rounds else 10,
                    help="Number of rounds with no improvement before stopping",
                    key=f"{key_prefix}_early_stopping_rounds"
                )
            else:
                early_stopping_rounds = None
            
            # Feature importance type
            importance_type = st.selectbox(
                "Feature Importance Type:",
                options=['split', 'gain'],
                index=['split', 'gain'].index(self.importance_type),
                help="split: Number of times feature is used, gain: Total gain of splits",
                key=f"{key_prefix}_importance_type"
            )
            
            # Advanced options
            linear_tree = st.checkbox(
                "Linear Tree",
                value=self.linear_tree,
                help="Use linear tree (combines linear model at leaves)",
                key=f"{key_prefix}_linear_tree"
            )
            
            feature_pre_filter = st.checkbox(
                "Feature Pre-filter",
                value=self.feature_pre_filter,
                help="Pre-filter features before training",
                key=f"{key_prefix}_feature_pre_filter"
            )
            
            # Random state
            random_state = st.number_input(
                "Random State:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if LIGHTGBM_AVAILABLE:
                st.success(f"✅ LightGBM {lgb.__version__} is available")
            else:
                st.error("❌ LightGBM not installed. Run: pip install lightgbm")
            
            st.info("""
            **LightGBM** - Microsoft's Fast GBM:
            • ⚡ Fastest training among major GBM frameworks
            • 💾 Most memory efficient gradient boosting
            • 🎯 Often better accuracy than XGBoost
            • 🔧 Native categorical feature support
            • 📊 Advanced sampling techniques (GOSS)
            • 🌿 Leaf-wise tree growth
            
            **Key Innovations:**
            • Exclusive Feature Bundling (EFB)
            • Gradient-based One-Side Sampling (GOSS)
            • Histogram-based algorithms
            • Optimized network communication
            """)
            
            # Performance comparison
            if st.button("⚡ Speed Comparison", key=f"{key_prefix}_speed_comparison"):
                st.markdown("""
                **Training Speed Comparison:**
                - LightGBM: ⭐⭐⭐⭐⭐ (Fastest)
                - XGBoost: ⭐⭐⭐⭐ (Fast)
                - CatBoost: ⭐⭐⭐ (Moderate)
                - Sklearn GBM: ⭐⭐ (Slower)
                
                **Memory Usage:**
                - LightGBM: ⭐⭐⭐⭐⭐ (Most efficient)
                - XGBoost: ⭐⭐⭐⭐ (Efficient)
                - CatBoost: ⭐⭐⭐ (Moderate)
                - Sklearn GBM: ⭐⭐ (Higher usage)
                
                **Accuracy:**
                - All are competitive, LightGBM often wins on categorical data
                """)
            
            # Tuning strategy
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_strategy"):
                st.markdown("""
                **LightGBM Tuning Strategy:**
                
                **Step 1: Core Parameters**
                - Start: num_leaves=31, learning_rate=0.1, n_estimators=100
                - num_leaves is the most important parameter
                
                **Step 2: Overfitting Control**
                - Tune num_leaves and max_depth together
                - Increase min_child_samples if overfitting
                
                **Step 3: Sampling & Regularization**
                - Try subsample < 1.0 and colsample_bytree < 1.0
                - Add L1/L2 regularization if needed
                
                **Step 4: Advanced Features**
                - Try GOSS boosting for large datasets
                - Enable categorical features if applicable
                - Use early stopping for optimal iterations
                """)
            
            # Boosting types explanation
            if st.button("🚀 Boosting Types", key=f"{key_prefix}_boosting_types"):
                st.markdown("""
                **LightGBM Boosting Types:**
                
                **GBDT (Default):**
                - Traditional gradient boosting
                - Best for most cases
                
                **DART:**
                - Dropout meets Multiple Additive Regression Trees
                - Can improve generalization
                - Slower training
                
                **GOSS:**
                - Gradient-based One-Side Sampling
                - Excellent for large datasets
                - Maintains accuracy with less data
                
                **RF:**
                - Random Forest mode
                - Each iteration builds multiple trees
                - Good for parallel training
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "boosting_type": boosting_type,
            "objective": 'multiclass',  # Will be set automatically
            "num_class": None,  # Will be set automatically
            "metric": 'multi_logloss',  # Will be adjusted for binary
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample_for_bin": 200000,
            "min_split_gain": min_split_gain,
            "min_child_weight": min_child_weight,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": -1,
            "silent": 'warn',
            "importance_type": importance_type,
            "max_bin": max_bin,
            "min_data_in_bin": 3,
            "feature_fraction": colsample_bytree,
            "feature_fraction_bynode": 1.0,
            "bagging_fraction": subsample,
            "bagging_freq": subsample_freq,
            "lambda_l1": reg_alpha,
            "lambda_l2": reg_lambda,
            "min_gain_to_split": min_split_gain,
            "drop_rate": drop_rate,
            "max_drop": max_drop,
            "skip_drop": skip_drop,
            "xgboost_dart_mode": False,
            "uniform_drop": False,
            "top_rate": top_rate,
            "other_rate": other_rate,
            "min_data_per_group": 100,
            "max_cat_threshold": 32,
            "cat_l2": 10.0,
            "cat_smooth": 10.0,
            "max_cat_to_onehot": max_cat_to_onehot,
            "cegb_tradeoff": 1.0,
            "cegb_penalty_split": 0.0,
            "path_smooth": 0.0,
            "early_stopping_rounds": early_stopping_rounds,
            "feature_pre_filter": feature_pre_filter,
            "linear_tree": linear_tree,
            "monotone_constraints": None,
            "monotone_constraints_method": 'basic',
            "monotone_penalty": 0.0,
            "interaction_constraints": None,
            "verbosity": -1,
            "seed": random_state,
            "deterministic": False
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return LightGBMClassifierPlugin(
            boosting_type=hyperparameters.get("boosting_type", self.boosting_type),
            objective=hyperparameters.get("objective", self.objective),
            num_class=hyperparameters.get("num_class", self.num_class),
            metric=hyperparameters.get("metric", self.metric),
            num_leaves=hyperparameters.get("num_leaves", self.num_leaves),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            subsample_for_bin=hyperparameters.get("subsample_for_bin", self.subsample_for_bin),
            min_split_gain=hyperparameters.get("min_split_gain", self.min_split_gain),
            min_child_weight=hyperparameters.get("min_child_weight", self.min_child_weight),
            min_child_samples=hyperparameters.get("min_child_samples", self.min_child_samples),
            subsample=hyperparameters.get("subsample", self.subsample),
            subsample_freq=hyperparameters.get("subsample_freq", self.subsample_freq),
            colsample_bytree=hyperparameters.get("colsample_bytree", self.colsample_bytree),
            reg_alpha=hyperparameters.get("reg_alpha", self.reg_alpha),
            reg_lambda=hyperparameters.get("reg_lambda", self.reg_lambda),
            random_state=hyperparameters.get("random_state", self.random_state),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            silent=hyperparameters.get("silent", self.silent),
            importance_type=hyperparameters.get("importance_type", self.importance_type),
            # LightGBM specific parameters
            max_bin=hyperparameters.get("max_bin", self.max_bin),
            min_data_in_bin=hyperparameters.get("min_data_in_bin", self.min_data_in_bin),
            feature_fraction=hyperparameters.get("feature_fraction", self.feature_fraction),
            feature_fraction_bynode=hyperparameters.get("feature_fraction_bynode", self.feature_fraction_bynode),
            bagging_fraction=hyperparameters.get("bagging_fraction", self.bagging_fraction),
            bagging_freq=hyperparameters.get("bagging_freq", self.bagging_freq),
            lambda_l1=hyperparameters.get("lambda_l1", self.lambda_l1),
            lambda_l2=hyperparameters.get("lambda_l2", self.lambda_l2),
            min_gain_to_split=hyperparameters.get("min_gain_to_split", self.min_gain_to_split),
            drop_rate=hyperparameters.get("drop_rate", self.drop_rate),
            max_drop=hyperparameters.get("max_drop", self.max_drop),
            skip_drop=hyperparameters.get("skip_drop", self.skip_drop),
            xgboost_dart_mode=hyperparameters.get("xgboost_dart_mode", self.xgboost_dart_mode),
            uniform_drop=hyperparameters.get("uniform_drop", self.uniform_drop),
            top_rate=hyperparameters.get("top_rate", self.top_rate),
            other_rate=hyperparameters.get("other_rate", self.other_rate),
            min_data_per_group=hyperparameters.get("min_data_per_group", self.min_data_per_group),
            max_cat_threshold=hyperparameters.get("max_cat_threshold", self.max_cat_threshold),
            cat_l2=hyperparameters.get("cat_l2", self.cat_l2),
            cat_smooth=hyperparameters.get("cat_smooth", self.cat_smooth),
            max_cat_to_onehot=hyperparameters.get("max_cat_to_onehot", self.max_cat_to_onehot),
            cegb_tradeoff=hyperparameters.get("cegb_tradeoff", self.cegb_tradeoff),
            cegb_penalty_split=hyperparameters.get("cegb_penalty_split", self.cegb_penalty_split),
            path_smooth=hyperparameters.get("path_smooth", self.path_smooth),
            # Training control
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds", self.early_stopping_rounds),
            feature_pre_filter=hyperparameters.get("feature_pre_filter", self.feature_pre_filter),
            linear_tree=hyperparameters.get("linear_tree", self.linear_tree),
            monotone_constraints=hyperparameters.get("monotone_constraints", self.monotone_constraints),
            monotone_constraints_method=hyperparameters.get("monotone_constraints_method", self.monotone_constraints_method),
            monotone_penalty=hyperparameters.get("monotone_penalty", self.monotone_penalty),
            interaction_constraints=hyperparameters.get("interaction_constraints", self.interaction_constraints),
            verbosity=hyperparameters.get("verbosity", self.verbosity),
            seed=hyperparameters.get("seed", self.seed),
            deterministic=hyperparameters.get("deterministic", self.deterministic)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for LightGBM
        
        LightGBM handles missing values and categorical features natively,
        so minimal preprocessing is needed.
        """
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
        """
        Check if LightGBM is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        if not LIGHTGBM_AVAILABLE:
            return False, "LightGBM is not installed. Install with: pip install lightgbm"
        
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"LightGBM requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            if len(unique_values) > 1000:
                return False, "Too many classes (>1000). LightGBM may not be suitable for this problem."
        
        return True, "LightGBM is compatible with this data"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "best_iteration": self.get_best_iteration(),
            "categorical_features": self.categorical_features_
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "LightGBM",
            "training_completed": True,
            "boosting_type": self.boosting_type,
            "total_iterations": self.model_.num_trees(),
            "best_iteration": self.get_best_iteration(),
            "early_stopping_used": self.early_stopping_rounds is not None,
            "categorical_features_detected": self.categorical_features_ is not None,
            "memory_optimizations": {
                "histogram_based": True,
                "feature_bundling": True,
                "sparse_optimization": True
            }
        }
        
        # Add evaluation results if available
        if self.eval_results_:
            info["evaluation_results"] = self.eval_results_
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate LightGBM Classifier-specific metrics based on the fitted model's
        internal state, training process, and analyses.

        Note: Most metrics are derived from the model's properties and evaluation results
        captured during training (`self.eval_results_`). The y_true, y_pred, y_proba
        parameters (typically for test set evaluation) are not directly used for these
        internal model-specific metrics.

        Args:
            y_true: Ground truth target values from a test set.
            y_pred: Predicted target values on a test set.
            y_proba: Predicted probabilities on a test set.

        Returns:
            A dictionary of LightGBM Classifier-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # Helper to safely extract nested dictionary values
        def safe_get(data_dict, path, default=np.nan):
            keys = path.split('.')
            current = data_dict
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and isinstance(key, int) and -len(current) <= key < len(current):
                    current = current[key]
                else:
                    return default
            # Avoid returning large lists/arrays if a scalar is expected
            if isinstance(current, (list, np.ndarray)) and len(current) > 10 and not path.endswith("_scores"):
                 return default
            return current if pd.notna(current) else default

        lgb_analysis = self.get_lgb_analysis() # Contains many useful pre-computed details

        metrics['best_iteration'] = self.get_best_iteration()
        metrics['num_trees_built'] = safe_get(lgb_analysis, 'num_trees')
        metrics['l1_reg_used'] = self.reg_alpha # or self.lambda_l1
        metrics['l2_reg_used'] = self.reg_lambda # or self.lambda_l2
        metrics['num_leaves_setting'] = self.num_leaves
        metrics['learning_rate_setting'] = self.learning_rate
        metrics['boosting_type_used'] = self.boosting_type
        
        early_stopping_rounds_active = self.early_stopping_rounds is not None
        metrics['early_stopping_enabled'] = early_stopping_rounds_active
        if early_stopping_rounds_active and metrics['best_iteration'] is not None:
            metrics['early_stopping_triggered'] = metrics['best_iteration'] < self.n_estimators
        else:
            metrics['early_stopping_triggered'] = False
            
        metrics['native_categorical_handling_active'] = safe_get(lgb_analysis, 'categorical_features_used', False)
        
        # Metrics from evaluation results
        if self.eval_results_ and metrics['best_iteration'] is not None:
            best_iter_idx = metrics['best_iteration'] - 1 # 0-indexed

            # Determine the primary metric key used during training
            # The 'metric' param in 'fit' method's 'params' dict is the key.
            # It might be a list or string. We'll try to get the first one if it's a list.
            primary_metric_key = self.model_.params.get('metric')
            if isinstance(primary_metric_key, list):
                primary_metric_key = primary_metric_key[0]
            
            if primary_metric_key: # Ensure we have a metric key
                # Training metric
                train_metrics = self.eval_results_.get('training', {}).get(primary_metric_key)
                if train_metrics and len(train_metrics) > best_iter_idx:
                    metrics[f'final_train_{primary_metric_key}'] = train_metrics[best_iter_idx]

                # Validation metric (assuming first validation set if multiple)
                # Find first validation set key (e.g., 'validation_0', 'valid_1')
                validation_key = None
                for k in self.eval_results_.keys():
                    if k != 'training': # Any key other than 'training' is a validation set
                        validation_key = k
                        break
                
                if validation_key:
                    val_metrics = self.eval_results_.get(validation_key, {}).get(primary_metric_key)
                    if val_metrics and len(val_metrics) > best_iter_idx:
                        metrics[f'final_{validation_key}_{primary_metric_key}'] = val_metrics[best_iter_idx]

        # Feature importance statistics
        try:
            fi_split = self.model_.feature_importance(importance_type='split')
            if fi_split is not None and len(fi_split) > 0:
                metrics['mean_fi_split'] = np.mean(fi_split)
                metrics['std_fi_split'] = np.std(fi_split)
                metrics['sum_fi_split'] = np.sum(fi_split)
        except Exception: # pragma: no cover
            pass # Could fail if model is trivial or importance not computable

        try:
            fi_gain = self.model_.feature_importance(importance_type='gain')
            if fi_gain is not None and len(fi_gain) > 0:
                metrics['mean_fi_gain'] = np.mean(fi_gain)
                metrics['std_fi_gain'] = np.std(fi_gain)
                metrics['sum_fi_gain'] = np.sum(fi_gain)
        except Exception: # pragma: no cover
            pass

        # Boosting type specific parameters
        if self.boosting_type == 'goss':
            metrics['goss_top_rate'] = self.top_rate
            metrics['goss_other_rate'] = self.other_rate
        elif self.boosting_type == 'dart':
            metrics['dart_drop_rate'] = self.drop_rate
            metrics['dart_max_drop'] = self.max_drop
            metrics['dart_skip_drop'] = self.skip_drop

        # Remove NaN or None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if pd.notna(v) and not (isinstance(v, float) and np.isinf(v))}

        # Convert numpy types to native python types
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int_)):
                metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float_)):
                metrics[key] = float(value)
            elif isinstance(value, np.bool_):
                metrics[key] = bool(value)
            elif isinstance(value, list) and len(value) == 1: # Unpack single item lists if they are metric values
                 if isinstance(value[0], (np.integer, np.int_)):
                    metrics[key] = int(value[0])
                 elif isinstance(value[0], (np.floating, np.float_)):
                    metrics[key] = float(value[0])

        return metrics


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return LightGBMClassifierPlugin()