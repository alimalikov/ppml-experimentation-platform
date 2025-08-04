import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# LightGBM import with fallback
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin


class LightGBMRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    LightGBM Regressor Plugin - Fast Gradient Boosting Framework
    
    LightGBM (Light Gradient Boosting Machine) is Microsoft's fast, distributed,
    high-performance gradient boosting framework based on decision tree algorithms.
    It uses histogram-based algorithms and leaf-wise tree growth strategy for
    optimal performance and memory efficiency.
    
    Key Features:
    - ⚡ Ultra-fast training speed with histogram-based algorithms
    - 💾 Low memory usage and cache-friendly implementation
    - 🎯 High accuracy with leaf-wise tree growth
    - 📊 Native categorical feature support
    - 🔥 GPU acceleration support
    - 🌐 Distributed training capabilities
    - 🛡️ Built-in regularization and overfitting prevention
    - 📈 Advanced feature importance metrics
    - 🔧 Automatic missing value handling
    - 🎛️ Rich hyperparameter space for fine-tuning
    - 📉 Early stopping with multiple evaluation metrics
    - 🔍 Network communication optimization for distributed training
    - 💡 DART (Dropouts meet Multiple Additive Regression Trees) support
    - 🌳 Advanced tree structure with optimal split finding
    """
    
    def __init__(
        self,
        # Core boosting parameters
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        
        # Tree structure parameters
        min_child_samples=20,
        min_child_weight=1e-3,
        min_split_gain=0.0,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        
        # Regularization parameters
        reg_alpha=0.0,  # L1 regularization
        reg_lambda=0.0,  # L2 regularization
        min_gain_to_split=0.0,
        
        # Learning control
        boosting_type='gbdt',
        objective='regression',
        metric='rmse',
        
        # Performance parameters
        num_threads=-1,
        device_type='cpu',
        gpu_use_dp=False,
        
        # Advanced parameters
        max_bin=255,
        min_data_per_group=100,
        max_cat_threshold=32,
        cat_l2=10.0,
        cat_smooth=10.0,
        
        # Feature selection
        feature_fraction=1.0,
        feature_fraction_bynode=1.0,
        extra_trees=False,
        
        # Control parameters
        random_state=42,
        verbose=-1,
        
        # Early stopping and validation
        early_stopping_rounds=None,
        validation_fraction=0.1,
        eval_at=[1, 2, 3, 4, 5],
        
        # Advanced boosting
        dart_rate_drop=0.1,
        max_drop=50,
        skip_drop=0.5,
        uniform_drop=False,
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        lightgbm_analysis=True,
        early_stopping_analysis=True,
        hyperparameter_sensitivity_analysis=True,
        
        # Advanced analysis
        tree_analysis=True,
        prediction_uncertainty_analysis=True,
        convergence_analysis=True,
        regularization_analysis=True,
        feature_interaction_analysis=True,
        cross_validation_analysis=True,
        categorical_analysis=True,
        
        # Comparison analysis
        compare_with_xgboost=True,
        compare_with_sklearn_gbm=True,
        performance_profiling=True,
        
        # Performance monitoring
        cv_folds=5,
        monitor_training=True,
        
        # Visualization options
        plot_importance=True,
        plot_trees=False,
        max_trees_to_plot=3,
        plot_metric_evolution=True
    ):
        super().__init__()
        
        # Check LightGBM availability
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Please install it using:\n"
                "pip install lightgbm\n"
                "or\n"
                "conda install -c conda-forge lightgbm"
            )
        
        # Core boosting parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        
        # Tree structure parameters
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        
        # Regularization parameters
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        
        # Learning control
        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        
        # Performance parameters
        self.num_threads = num_threads
        self.device_type = device_type
        self.gpu_use_dp = gpu_use_dp
        
        # Advanced parameters
        self.max_bin = max_bin
        self.min_data_per_group = min_data_per_group
        self.max_cat_threshold = max_cat_threshold
        self.cat_l2 = cat_l2
        self.cat_smooth = cat_smooth
        
        # Feature selection
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.extra_trees = extra_trees
        
        # Control parameters
        self.random_state = random_state
        self.verbose = verbose
        
        # Early stopping and validation
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.eval_at = eval_at
        
        # Advanced boosting (DART)
        self.dart_rate_drop = dart_rate_drop
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.uniform_drop = uniform_drop
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.lightgbm_analysis = lightgbm_analysis
        self.early_stopping_analysis = early_stopping_analysis
        self.hyperparameter_sensitivity_analysis = hyperparameter_sensitivity_analysis
        
        # Advanced analysis
        self.tree_analysis = tree_analysis
        self.prediction_uncertainty_analysis = prediction_uncertainty_analysis
        self.convergence_analysis = convergence_analysis
        self.regularization_analysis = regularization_analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        self.cross_validation_analysis = cross_validation_analysis
        self.categorical_analysis = categorical_analysis
        
        # Comparison analysis
        self.compare_with_xgboost = compare_with_xgboost
        self.compare_with_sklearn_gbm = compare_with_sklearn_gbm
        self.performance_profiling = performance_profiling
        
        # Performance monitoring
        self.cv_folds = cv_folds
        self.monitor_training = monitor_training
        
        # Visualization options
        self.plot_importance = plot_importance
        self.plot_trees = plot_trees
        self.max_trees_to_plot = max_trees_to_plot
        self.plot_metric_evolution = plot_metric_evolution
        
        # Required plugin metadata
        self._name = "LightGBM Regressor"
        self._description = "Fast gradient boosting with optimal memory usage and categorical feature support"
        self._category = "Gradient Boosting"
        
        # Required capability flags - THESE ARE ESSENTIAL!
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 50
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.categorical_features_ = None
        self.training_history_ = {}
        self.validation_scores_ = {}
        
        # Analysis results storage
        self.feature_importance_analysis_ = {}
        self.lightgbm_analysis_ = {}
        self.early_stopping_analysis_ = {}
        self.convergence_analysis_ = {}
        self.regularization_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.tree_analysis_ = {}
        self.prediction_uncertainty_analysis_ = {}
        self.categorical_analysis_ = {}
        self.performance_profile_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None, categorical_features=None):
        """
        Fit the LightGBM Regressor with comprehensive analysis
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample
        categorical_features : list, optional
            List of categorical feature names or indices
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Store feature information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Detect categorical features
        self.categorical_features_ = self._detect_categorical_features(X, categorical_features)
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store original data for analysis
        self.X_original_ = X.copy()
        self.y_original_ = y.copy()
        
        # Prepare LightGBM parameters
        lgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_gain_to_split': self.min_gain_to_split,
            'boosting_type': self.boosting_type,
            'objective': self.objective,
            'metric': self.metric,
            'num_threads': self.num_threads,
            'device_type': self.device_type,
            'max_bin': self.max_bin,
            'feature_fraction': self.feature_fraction,
            'feature_fraction_bynode': self.feature_fraction_bynode,
            'extra_trees': self.extra_trees,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        # Add DART parameters if using DART boosting
        if self.boosting_type == 'dart':
            lgb_params.update({
                'drop_rate': self.dart_rate_drop,
                'max_drop': self.max_drop,
                'skip_drop': self.skip_drop,
                'uniform_drop': self.uniform_drop
            })
        
        # Add categorical features handling
        if self.categorical_features_:
            lgb_params.update({
                'max_cat_threshold': self.max_cat_threshold,
                'cat_l2': self.cat_l2,
                'cat_smooth': self.cat_smooth
            })
        
        # Handle early stopping
        eval_set = None
        eval_names = None
        callbacks = []
        
        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
            )
            eval_set = [(X_val, y_val)]
            eval_names = ['validation']
            X, y = X_train, y_train
            
            # Add early stopping callback
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))
            
            # Add evaluation callback for monitoring
            callbacks.append(lgb.record_evaluation(self.training_history_))
        
        # Create and configure LightGBM model
        self.model_ = lgb.LGBMRegressor(**lgb_params)
        
        # Fit the model with callbacks
        fit_params = {}
        if eval_set is not None:
            fit_params.update({
                'eval_set': eval_set,
                'eval_names': eval_names,
                'callbacks': callbacks
            })
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        if self.categorical_features_:
            fit_params['categorical_feature'] = self.categorical_features_
        
        # Train the model
        self.model_.fit(X, y, **fit_params)
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_lightgbm_specifics()
        self._analyze_early_stopping()
        self._analyze_convergence()
        self._analyze_regularization_effects()
        self._analyze_cross_validation()
        self._analyze_tree_structure()
        self._analyze_prediction_uncertainty()
        self._analyze_categorical_features()
        
        if self.hyperparameter_sensitivity_analysis:
            self._analyze_hyperparameter_sensitivity()
        
        if self.performance_profiling:
            self._profile_performance()
        
        if self.compare_with_xgboost:
            self._compare_with_xgboost()
        
        if self.compare_with_sklearn_gbm:
            self._compare_with_sklearn_gbm()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted LightGBM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        return self.model_.predict(X)
    
    def predict_with_uncertainty(self, X, n_iterations=None):
        """
        Make predictions with uncertainty estimates using boosting iterations
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        n_iterations : int, optional
            Number of boosting iterations to use for uncertainty
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions and uncertainty estimates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Get predictions at different iterations
        if n_iterations is None:
            n_iterations = min(50, self.model_.n_estimators)
        
        iteration_preds = []
        for i in range(1, min(n_iterations + 1, self.model_.n_estimators + 1)):
            pred = self.model_.predict(X, num_iteration=i)
            iteration_preds.append(pred)
        
        iteration_preds = np.array(iteration_preds)
        
        # Calculate uncertainty metrics
        final_predictions = iteration_preds[-1]
        prediction_std = np.std(iteration_preds, axis=0)
        prediction_range = np.max(iteration_preds, axis=0) - np.min(iteration_preds, axis=0)
        
        # Uncertainty score based on prediction variability
        uncertainty_score = prediction_std / (np.abs(final_predictions) + 1e-10)
        
        # Confidence intervals
        confidence_95_lower = final_predictions - 1.96 * prediction_std
        confidence_95_upper = final_predictions + 1.96 * prediction_std
        
        return {
            'predictions': final_predictions,
            'prediction_std': prediction_std,
            'prediction_range': prediction_range,
            'uncertainty_score': uncertainty_score,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'prediction_interval_width': confidence_95_upper - confidence_95_lower,
            'boosting_stability': 1.0 / (1.0 + uncertainty_score),
            'iterations_used': len(iteration_preds)
        }
    
    def _detect_categorical_features(self, X, categorical_features=None):
        """Detect categorical features in the dataset"""
        if categorical_features is not None:
            return categorical_features
        
        if hasattr(X, 'dtypes'):
            # For pandas DataFrame, detect categorical or object columns
            categorical_cols = []
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    categorical_cols.append(col)
                elif X[col].dtype in ['int64', 'int32'] and X[col].nunique() <= 20:
                    # Low cardinality integer features might be categorical
                    categorical_cols.append(col)
            return categorical_cols
        
        return None
    
    def _analyze_feature_importance(self):
        """Analyze feature importance with LightGBM specific metrics"""
        if not self.compute_feature_importance:
            return
        
        try:
            # Get different types of importance from LightGBM
            importance_types = ['split', 'gain']
            importance_scores = {}
            
            for imp_type in importance_types:
                try:
                    scores = self.model_.feature_importances_
                    if imp_type == 'gain':
                        # LightGBM's default is 'split', get gain if available
                        if hasattr(self.model_, 'booster_'):
                            scores = self.model_.booster_.feature_importance(importance_type='gain')
                    importance_scores[imp_type] = scores
                except:
                    importance_scores[imp_type] = np.zeros(len(self.feature_names_))
            
            # Permutation importance if requested
            permutation_imp = None
            permutation_imp_std = None
            if self.compute_permutation_importance:
                try:
                    perm_imp_result = permutation_importance(
                        self.model_, self.X_original_, self.y_original_,
                        n_repeats=10, random_state=self.random_state,
                        scoring='neg_mean_squared_error'
                    )
                    permutation_imp = perm_imp_result.importances_mean
                    permutation_imp_std = perm_imp_result.importances_std
                except:
                    permutation_imp = None
                    permutation_imp_std = None
            
            # Feature importance ranking for each type
            rankings = {}
            for imp_type, scores in importance_scores.items():
                rankings[imp_type] = np.argsort(scores)[::-1]
            
            # Calculate comprehensive importance statistics
            importance_stats = {}
            for imp_type, scores in importance_scores.items():
                importance_stats[imp_type] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'max': np.max(scores),
                    'min': np.min(scores),
                    'concentration': self._calculate_importance_concentration(scores),
                    'effective_features': np.sum(scores > np.mean(scores))
                }
            
            # Top features analysis
            top_features = []
            for i in range(min(15, len(self.feature_names_))):
                feature_info = {
                    'name': self.feature_names_[rankings['gain'][i]],
                    'gain_importance': importance_scores['gain'][rankings['gain'][i]],
                    'split_importance': importance_scores['split'][rankings['gain'][i]],
                    'gain_rank': i + 1,
                    'split_rank': np.where(rankings['split'] == rankings['gain'][i])[0][0] + 1,
                    'is_categorical': self.feature_names_[rankings['gain'][i]] in (self.categorical_features_ or [])
                }
                
                if permutation_imp is not None:
                    feature_info['permutation_importance'] = permutation_imp[rankings['gain'][i]]
                    feature_info['permutation_std'] = permutation_imp_std[rankings['gain'][i]]
                
                top_features.append(feature_info)
            
            self.feature_importance_analysis_ = {
                'importance_scores': importance_scores,
                'permutation_importance': permutation_imp,
                'permutation_importance_std': permutation_imp_std,
                'rankings': rankings,
                'importance_statistics': importance_stats,
                'top_features': top_features,
                'feature_names': self.feature_names_,
                'categorical_features': self.categorical_features_,
                'interpretation': {
                    'gain': 'Total gain of splits using the feature',
                    'split': 'Number of times feature appears in trees'
                }
            }
            
        except Exception as e:
            self.feature_importance_analysis_ = {
                'error': f'Could not analyze feature importance: {str(e)}'
            }
    
    def _calculate_importance_concentration(self, importance_values):
        """Calculate how concentrated the importance is in top features"""
        try:
            sorted_importance = np.sort(importance_values)[::-1]
            total_importance = np.sum(sorted_importance)
            
            if total_importance == 0:
                return {'top_5_concentration': 0, 'top_10_concentration': 0}
            
            return {
                'top_5_concentration': np.sum(sorted_importance[:5]) / total_importance,
                'top_10_concentration': np.sum(sorted_importance[:10]) / total_importance,
                'gini_coefficient': self._calculate_gini_coefficient(importance_values)
            }
        except:
            return {'error': 'Could not calculate concentration'}
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for importance distribution"""
        try:
            sorted_values = np.sort(values)
            n = len(values)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        except:
            return 0.0
    
    def _analyze_lightgbm_specifics(self):
        """Analyze LightGBM specific features and optimizations"""
        if not self.lightgbm_analysis:
            return
        
        try:
            analysis = {}
            
            # Model configuration analysis
            analysis['model_config'] = {
                'boosting_type': self.boosting_type,
                'objective_function': self.objective,
                'metric': self.metric,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'n_estimators': self.model_.n_estimators,
                'learning_rate': self.learning_rate,
                'device_type': self.device_type
            }
            
            # LightGBM optimizations
            analysis['lightgbm_optimizations'] = {
                'histogram_based_algorithm': True,
                'leaf_wise_growth': True,
                'native_categorical_support': len(self.categorical_features_ or []) > 0,
                'memory_optimization': True,
                'cache_friendly': True,
                'network_communication_optimized': True,
                'sparse_optimization': True,
                'gradient_one_side_sampling': self.boosting_type == 'goss',
                'exclusive_feature_bundling': True
            }
            
            # Tree growth strategy
            analysis['tree_growth'] = {
                'strategy': 'Leaf-wise (vs level-wise)',
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth if self.max_depth > 0 else 'Unlimited',
                'complexity_control': self._assess_tree_complexity(),
                'overfitting_risk': self._assess_overfitting_risk_lgb()
            }
            
            # Performance characteristics
            analysis['performance'] = {
                'training_speed': 'Very Fast (histogram-based)',
                'memory_usage': 'Low (optimized data structures)',
                'categorical_handling': 'Native (no preprocessing needed)',
                'missing_value_handling': 'Automatic',
                'parallel_training': self.num_threads != 1,
                'gpu_acceleration': self.device_type == 'gpu'
            }
            
            # Feature handling
            analysis['feature_handling'] = {
                'total_features': len(self.feature_names_),
                'categorical_features': len(self.categorical_features_ or []),
                'numerical_features': len(self.feature_names_) - len(self.categorical_features_ or []),
                'max_categories_per_feature': self.max_cat_threshold,
                'categorical_regularization': {
                    'l2_regularization': self.cat_l2,
                    'smoothing': self.cat_smooth
                }
            }
            
            # Regularization and sampling
            analysis['regularization'] = {
                'l1_alpha': self.reg_alpha,
                'l2_lambda': self.reg_lambda,
                'min_gain_to_split': self.min_gain_to_split,
                'min_child_samples': self.min_child_samples,
                'feature_fraction': self.feature_fraction,
                'row_sampling': self.subsample,
                'extra_trees': self.extra_trees
            }
            
            self.lightgbm_analysis_ = analysis
            
        except Exception as e:
            self.lightgbm_analysis_ = {
                'error': f'Could not analyze LightGBM specifics: {str(e)}'
            }
    
    def _assess_tree_complexity(self):
        """Assess tree complexity configuration"""
        if self.max_depth <= 0:
            depth_constraint = "No depth limit"
        elif self.max_depth <= 6:
            depth_constraint = "Conservative depth"
        elif self.max_depth <= 12:
            depth_constraint = "Moderate depth"
        else:
            depth_constraint = "Deep trees"
        
        if self.num_leaves <= 31:
            leaves_constraint = "Conservative leaf count"
        elif self.num_leaves <= 127:
            leaves_constraint = "Moderate leaf count"
        else:
            leaves_constraint = "High leaf count"
        
        return f"{depth_constraint}, {leaves_constraint}"
    
    def _assess_overfitting_risk_lgb(self):
        """Assess overfitting risk for LightGBM configuration"""
        risk_factors = []
        
        if self.num_leaves > 100:
            risk_factors.append("High num_leaves")
        
        if self.max_depth > 10 and self.max_depth > 0:
            risk_factors.append("Deep trees")
        
        if self.reg_alpha + self.reg_lambda < 0.1:
            risk_factors.append("Low regularization")
        
        if self.min_child_samples < 10:
            risk_factors.append("Low min_child_samples")
        
        if self.feature_fraction > 0.9 and self.subsample > 0.9:
            risk_factors.append("Minimal feature/row sampling")
        
        if len(risk_factors) == 0:
            return "Low - Good regularization"
        elif len(risk_factors) <= 2:
            return f"Moderate - {len(risk_factors)} risk factors: {', '.join(risk_factors)}"
        else:
            return f"High - {len(risk_factors)} risk factors: {', '.join(risk_factors)}"
    
    def _analyze_early_stopping(self):
        """Analyze early stopping behavior and training evolution"""
        if not self.early_stopping_analysis:
            return
        
        try:
            analysis = {}
            
            # Check if early stopping was used
            early_stopping_used = self.early_stopping_rounds is not None
            analysis['early_stopping_used'] = early_stopping_used
            
            if early_stopping_used and hasattr(self.model_, 'best_iteration'):
                analysis['best_iteration'] = self.model_.best_iteration
                analysis['total_iterations'] = self.model_.n_estimators
                analysis['early_stopping_triggered'] = self.model_.best_iteration < self.model_.n_estimators
                
                if analysis['early_stopping_triggered']:
                    analysis['iterations_saved'] = self.model_.n_estimators - self.model_.best_iteration
                    analysis['efficiency_gain'] = analysis['iterations_saved'] / self.model_.n_estimators
                else:
                    analysis['iterations_saved'] = 0
                    analysis['efficiency_gain'] = 0.0
            
            # Analyze training history if available
            if self.training_history_:
                analysis['training_history'] = self._analyze_training_progression_lgb()
            
            self.early_stopping_analysis_ = analysis
            
        except Exception as e:
            self.early_stopping_analysis_ = {
                'error': f'Could not analyze early stopping: {str(e)}'
            }
    
    def _analyze_training_progression_lgb(self):
        """Analyze LightGBM training progression"""
        try:
            progression = {}
            
            for dataset_name, metrics in self.training_history_.items():
                dataset_progression = {}
                
                for metric_name, values in metrics.items():
                    if len(values) > 1:
                        values_array = np.array(values)
                        
                        # Calculate improvement metrics
                        initial_score = values_array[0]
                        final_score = values_array[-1]
                        best_score = np.min(values_array) if 'rmse' in metric_name.lower() or 'mae' in metric_name.lower() else np.max(values_array)
                        
                        # Find best iteration
                        best_iteration = np.argmin(values_array) if 'rmse' in metric_name.lower() or 'mae' in metric_name.lower() else np.argmax(values_array)
                        
                        # Calculate convergence characteristics
                        improvements = np.diff(values_array)
                        if 'rmse' in metric_name.lower() or 'mae' in metric_name.lower():
                            improvements = -improvements  # For error metrics, improvement is decrease
                        
                        dataset_progression[metric_name] = {
                            'initial_score': float(initial_score),
                            'final_score': float(final_score),
                            'best_score': float(best_score),
                            'best_iteration': int(best_iteration),
                            'total_improvement': float(final_score - initial_score),
                            'improvement_to_best': float(best_score - initial_score),
                            'mean_improvement_per_iteration': float(np.mean(improvements)),
                            'improvement_stability': float(np.std(improvements)),
                            'convergence_rate': self._calculate_convergence_rate(values_array)
                        }
                
                progression[dataset_name] = dataset_progression
            
            return progression
            
        except Exception as e:
            return {'error': f'Could not analyze training progression: {str(e)}'}
    
    def _calculate_convergence_rate(self, values):
        """Calculate convergence rate of training metric"""
        try:
            if len(values) < 10:
                return 0.0
            
            # Check stability in last 20% of iterations
            recent_portion = max(10, len(values) // 5)
            recent_values = values[-recent_portion:]
            
            # Calculate coefficient of variation for recent values
            cv = np.std(recent_values) / (abs(np.mean(recent_values)) + 1e-10)
            
            # Convergence rate: higher values indicate better convergence
            convergence_rate = 1.0 / (1.0 + cv)
            return float(convergence_rate)
            
        except:
            return 0.0
    
    def _analyze_convergence(self):
        """Analyze overall model convergence characteristics"""
        if not self.convergence_analysis:
            return
        
        try:
            analysis = {}
            
            # Training convergence
            if hasattr(self.model_, 'best_iteration'):
                analysis['optimal_iterations'] = self.model_.best_iteration
                analysis['convergence_efficiency'] = self.model_.best_iteration / self.model_.n_estimators
            else:
                analysis['optimal_iterations'] = self.model_.n_estimators
                analysis['convergence_efficiency'] = 1.0
            
            # Learning rate analysis
            analysis['learning_rate_analysis'] = {
                'learning_rate': self.learning_rate,
                'category': self._categorize_learning_rate_lgb(),
                'recommended_adjustment': self._recommend_learning_rate_adjustment_lgb()
            }
            
            # Tree structure impact on convergence
            analysis['tree_structure_impact'] = {
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'complexity_assessment': self._assess_tree_complexity(),
                'growth_strategy': 'Leaf-wise (LightGBM default)',
                'convergence_characteristics': self._assess_convergence_characteristics()
            }
            
            self.convergence_analysis_ = analysis
            
        except Exception as e:
            self.convergence_analysis_ = {
                'error': f'Could not analyze convergence: {str(e)}'
            }
    
    def _categorize_learning_rate_lgb(self):
        """Categorize learning rate for LightGBM"""
        if self.learning_rate >= 0.3:
            return "High - Fast learning, may overshoot optimal"
        elif self.learning_rate >= 0.1:
            return "Standard - Good balance for most datasets"
        elif self.learning_rate >= 0.05:
            return "Conservative - Stable learning, needs more iterations"
        else:
            return "Very Conservative - Very stable, needs many iterations"
    
    def _recommend_learning_rate_adjustment_lgb(self):
        """Recommend learning rate adjustments for LightGBM"""
        if hasattr(self.model_, 'best_iteration'):
            efficiency = self.model_.best_iteration / self.model_.n_estimators
            
            if efficiency < 0.3:
                return "Consider increasing learning rate for faster convergence"
            elif efficiency > 0.9:
                return "Consider decreasing learning rate for better optimization"
            else:
                return "Learning rate appears well-tuned"
        else:
            return "No early stopping data available for recommendation"
    
    def _assess_convergence_characteristics(self):
        """Assess convergence characteristics based on tree structure"""
        complexity_score = 0
        
        # Leaf count impact
        if self.num_leaves >= 127:
            complexity_score += 3
        elif self.num_leaves >= 63:
            complexity_score += 2
        elif self.num_leaves >= 31:
            complexity_score += 1
        
        # Depth impact (if limited)
        if self.max_depth > 0:
            if self.max_depth >= 10:
                complexity_score += 2
            elif self.max_depth >= 6:
                complexity_score += 1
        
        if complexity_score >= 4:
            return "Complex trees - May converge slowly but capture intricate patterns"
        elif complexity_score >= 2:
            return "Moderate complexity - Balanced convergence speed and pattern capture"
        else:
            return "Simple trees - Fast convergence but may miss complex patterns"
    
    def _analyze_regularization_effects(self):
        """Analyze the effects of different regularization techniques in LightGBM"""
        if not self.regularization_analysis:
            return
        
        try:
            analysis = {}
            
            # L1 and L2 regularization
            analysis['l1_l2_regularization'] = {
                'l1_alpha': self.reg_alpha,
                'l2_lambda': self.reg_lambda,
                'l1_effect': 'Feature selection' if self.reg_alpha > 0 else 'None',
                'l2_effect': 'Weight smoothing' if self.reg_lambda > 0 else 'None',
                'combined_strength': self.reg_alpha + self.reg_lambda
            }
            
            # Structural regularization
            analysis['structural_regularization'] = {
                'min_gain_to_split': self.min_gain_to_split,
                'min_child_samples': self.min_child_samples,
                'min_child_weight': self.min_child_weight,
                'effect': 'Controls tree growth and prevents overfitting',
                'strength_assessment': self._assess_structural_regularization_lgb()
            }
            
            # Feature and sample regularization
            analysis['sampling_regularization'] = {
                'feature_fraction': self.feature_fraction,
                'feature_fraction_bynode': self.feature_fraction_bynode,
                'subsample': self.subsample,
                'subsample_freq': self.subsample_freq,
                'extra_trees': self.extra_trees,
                'effect': 'Reduces overfitting through stochastic sampling',
                'strength_assessment': self._assess_sampling_regularization_lgb()
            }
            
            # Categorical feature regularization (if applicable)
            if self.categorical_features_:
                analysis['categorical_regularization'] = {
                    'cat_l2': self.cat_l2,
                    'cat_smooth': self.cat_smooth,
                    'max_cat_threshold': self.max_cat_threshold,
                    'effect': 'Prevents overfitting on categorical features',
                    'categorical_features_count': len(self.categorical_features_)
                }
            
            # Overall assessment
            analysis['overall_assessment'] = {
                'total_regularization_strength': self._calculate_total_regularization_strength_lgb(),
                'regularization_balance': self._assess_regularization_balance_lgb(),
                'overfitting_protection': self._assess_overfitting_protection_lgb(),
                'recommendations': self._get_regularization_recommendations_lgb()
            }
            
            self.regularization_analysis_ = analysis
            
        except Exception as e:
            self.regularization_analysis_ = {
                'error': f'Could not analyze regularization effects: {str(e)}'
            }
    
    def _assess_structural_regularization_lgb(self):
        """Assess structural regularization strength for LightGBM"""
        score = 0
        
        if self.min_child_samples >= 50:
            score += 3
        elif self.min_child_samples >= 20:
            score += 2
        elif self.min_child_samples >= 10:
            score += 1
        
        if self.min_gain_to_split > 0.1:
            score += 2
        elif self.min_gain_to_split > 0.0:
            score += 1
        
        if score >= 4:
            return "Strong - Conservative tree growth"
        elif score >= 2:
            return "Moderate - Balanced tree complexity control"
        else:
            return "Weak - Minimal structural constraints"
    
    def _assess_sampling_regularization_lgb(self):
        """Assess sampling regularization strength for LightGBM"""
        feature_sampling = self.feature_fraction * self.feature_fraction_bynode
        
        sampling_strength = 0
        if feature_sampling < 0.8:
            sampling_strength += 2
        elif feature_sampling < 0.9:
            sampling_strength += 1
        
        if self.subsample < 0.8:
            sampling_strength += 2
        elif self.subsample < 0.9:
            sampling_strength += 1
        
        if self.extra_trees:
            sampling_strength += 1
        
        if sampling_strength >= 4:
            return "Strong - High variance reduction through sampling"
        elif sampling_strength >= 2:
            return "Moderate - Balanced sampling regularization"
        else:
            return "Weak - Minimal sampling regularization"
    
    def _calculate_total_regularization_strength_lgb(self):
        """Calculate overall regularization strength for LightGBM"""
        score = 0
        
        # L1/L2 contribution
        score += min(self.reg_alpha * 3, 5)
        score += min(self.reg_lambda * 2, 5)
        
        # Structural contribution
        score += min(self.min_child_samples / 10, 3)
        score += min(self.min_gain_to_split * 10, 3)
        
        # Sampling contribution
        score += (1 - self.feature_fraction) * 3
        score += (1 - self.subsample) * 3
        
        if self.extra_trees:
            score += 2
        
        return min(score, 20)
    
    def _assess_regularization_balance_lgb(self):
        """Assess balance between different regularization types in LightGBM"""
        l1_l2_strength = self.reg_alpha + self.reg_lambda
        structural_strength = self.min_child_samples / 50 + self.min_gain_to_split * 5
        sampling_strength = (1 - self.feature_fraction) + (1 - self.subsample)
        
        strengths = [l1_l2_strength, structural_strength, sampling_strength]
        max_strength = max(strengths)
        min_strength = min(strengths)
        
        if max_strength - min_strength < 0.5:
            return "Well-balanced across all regularization types"
        elif l1_l2_strength > max(structural_strength, sampling_strength) + 1:
            return "Heavily relies on L1/L2 regularization"
        elif structural_strength > max(l1_l2_strength, sampling_strength) + 0.5:
            return "Heavily relies on structural regularization"
        else:
            return "Balanced with emphasis on sampling regularization"
    
    def _assess_overfitting_protection_lgb(self):
        """Assess overall overfitting protection in LightGBM"""
        protection_score = self._calculate_total_regularization_strength_lgb()
        
        if protection_score >= 15:
            return "Excellent - Multiple strong regularization mechanisms"
        elif protection_score >= 10:
            return "Good - Adequate overfitting protection"
        elif protection_score >= 5:
            return "Moderate - Basic overfitting protection"
        else:
            return "Weak - Risk of overfitting, consider increasing regularization"
    
    def _get_regularization_recommendations_lgb(self):
        """Get LightGBM-specific regularization recommendations"""
        recommendations = []
        
        total_strength = self._calculate_total_regularization_strength_lgb()
        
        if total_strength < 5:
            recommendations.append("Increase regularization to prevent overfitting")
            if self.reg_lambda < 0.1:
                recommendations.append("Try reg_lambda >= 0.1 for L2 regularization")
            if self.min_child_samples < 20:
                recommendations.append("Increase min_child_samples to 20 or higher")
        
        if self.feature_fraction == 1.0 and len(self.feature_names_) > 50:
            recommendations.append("Consider feature_fraction < 1.0 for feature sampling")
        
        if self.subsample == 1.0 and self.subsample_freq == 0:
            recommendations.append("Consider subsample < 1.0 with subsample_freq > 0")
        
        if self.num_leaves > 100 and total_strength < 10:
            recommendations.append("High num_leaves requires stronger regularization")
        
        if len(self.categorical_features_ or []) > 0:
            if self.cat_l2 < 5.0:
                recommendations.append("Consider increasing cat_l2 for categorical regularization")
        
        if not recommendations:
            recommendations.append("Regularization appears well-configured")
        
        return recommendations
    
    def _analyze_cross_validation(self):
        """Perform cross-validation analysis for LightGBM"""
        if not self.cross_validation_analysis:
            return
        
        try:
            # Create a fresh model for CV
            cv_model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                min_child_weight=self.min_child_weight,
                min_split_gain=self.min_split_gain,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                num_threads=self.num_threads,
                verbose=-1
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                cv_model, self.X_original_, self.y_original_,
                cv=self.cv_folds, scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Convert to positive MSE
            cv_scores = -cv_scores
            
            # Calculate statistics
            cv_analysis = {
                'cv_scores': cv_scores,
                'mean_mse': np.mean(cv_scores),
                'std_mse': np.std(cv_scores),
                'mean_rmse': np.sqrt(np.mean(cv_scores)),
                'cv_coefficient_variation': np.std(cv_scores) / np.mean(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'score_range': np.max(cv_scores) - np.min(cv_scores),
                'stability_assessment': self._assess_cv_stability(cv_scores)
            }
            
            # R² cross-validation
            try:
                r2_scores = cross_val_score(
                    cv_model, self.X_original_, self.y_original_,
                    cv=self.cv_folds, scoring='r2',
                    n_jobs=-1
                )
                
                cv_analysis.update({
                    'r2_scores': r2_scores,
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores)
                })
            except:
                pass
            
            self.cross_validation_analysis_ = cv_analysis
            
        except Exception as e:
            self.cross_validation_analysis_ = {
                'error': f'Could not perform cross-validation analysis: {str(e)}'
            }
    
    def _assess_cv_stability(self, cv_scores):
        """Assess cross-validation stability"""
        cv = np.std(cv_scores) / np.mean(cv_scores)
        
        if cv < 0.1:
            return "Excellent - Very stable performance across folds"
        elif cv < 0.2:
            return "Good - Stable performance with minor variation"
        elif cv < 0.3:
            return "Moderate - Some variation across folds"
        else:
            return "Poor - High variation, check for data issues"
    
    def _analyze_tree_structure(self):
        """Analyze LightGBM tree structure and characteristics"""
        if not self.tree_analysis:
            return
        
        try:
            analysis = {}
            
            # Basic tree statistics
            analysis['ensemble_stats'] = {
                'total_trees': self.model_.n_estimators,
                'num_leaves': self.num_leaves,
                'max_depth_setting': self.max_depth,
                'boosting_type': self.boosting_type,
                'growth_strategy': 'Leaf-wise'
            }
            
            # Tree complexity analysis
            analysis['complexity_analysis'] = {
                'leaves_per_tree': self.num_leaves,
                'depth_constraint': 'None' if self.max_depth <= 0 else f'Max {self.max_depth}',
                'complexity_category': self._categorize_tree_complexity_lgb(),
                'overfitting_risk': self._assess_overfitting_risk_lgb()
            }
            
            # Leaf-wise vs level-wise comparison
            analysis['growth_strategy_analysis'] = {
                'strategy': 'Leaf-wise growth',
                'advantages': [
                    'Faster training than level-wise growth',
                    'Better accuracy on large datasets',
                    'More efficient memory usage',
                    'Can capture complex patterns with fewer leaves'
                ],
                'considerations': [
                    'May overfit on small datasets',
                    'Requires careful regularization',
                    'Less interpretable than level-wise trees'
                ]
            }
            
            # Feature usage in trees
            if self.feature_importance_analysis_:
                split_importance = self.feature_importance_analysis_.get('importance_scores', {}).get('split', np.array([]))
                if len(split_importance) > 0:
                    analysis['feature_usage'] = {
                        'features_used': np.sum(split_importance > 0),
                        'features_unused': np.sum(split_importance == 0),
                        'usage_efficiency': np.sum(split_importance > 0) / len(split_importance),
                        'most_used_features': self._get_most_used_features_lgb(split_importance)
                    }
            
            self.tree_analysis_ = analysis
            
        except Exception as e:
            self.tree_analysis_ = {
                'error': f'Could not analyze tree structure: {str(e)}'
            }
    
    def _categorize_tree_complexity_lgb(self):
        """Categorize LightGBM tree complexity"""
        complexity_score = 0
        
        # Leaves contribution
        if self.num_leaves >= 255:
            complexity_score += 3
        elif self.num_leaves >= 127:
            complexity_score += 2
        elif self.num_leaves >= 63:
            complexity_score += 1
        
        # Depth contribution (if constrained)
        if self.max_depth > 0:
            if self.max_depth >= 15:
                complexity_score += 3
            elif self.max_depth >= 10:
                complexity_score += 2
            elif self.max_depth >= 6:
                complexity_score += 1
        
        # Regularization penalty
        regularization_strength = self.reg_alpha + self.reg_lambda + (self.min_child_samples / 50)
        if regularization_strength < 0.1:
            complexity_score += 1
        elif regularization_strength > 2.0:
            complexity_score -= 1
        
        if complexity_score >= 5:
            return "High - Complex trees with overfitting risk"
        elif complexity_score >= 3:
            return "Moderate - Balanced complexity"
        else:
            return "Low - Simple trees, may underfit"
    
    def _get_most_used_features_lgb(self, split_importance):
        """Get the most frequently used features in LightGBM trees"""
        try:
            top_indices = np.argsort(split_importance)[::-1][:10]
            return [
                {
                    'feature': self.feature_names_[i],
                    'split_count': split_importance[i],
                    'split_percentage': split_importance[i] / np.sum(split_importance) * 100,
                    'is_categorical': self.feature_names_[i] in (self.categorical_features_ or [])
                }
                for i in top_indices if split_importance[i] > 0
            ]
        except:
            return []
    
    def _analyze_prediction_uncertainty(self):
        """Analyze prediction uncertainty characteristics for LightGBM"""
        if not self.prediction_uncertainty_analysis:
            return
        
        try:
            # Sample some data for uncertainty analysis
            sample_size = min(100, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            
            # Get uncertainty estimates
            uncertainty_results = self.predict_with_uncertainty(X_sample)
            
            # Analyze uncertainty patterns
            uncertainty_analysis = {
                'mean_uncertainty': np.mean(uncertainty_results['uncertainty_score']),
                'std_uncertainty': np.std(uncertainty_results['uncertainty_score']),
                'max_uncertainty': np.max(uncertainty_results['uncertainty_score']),
                'min_uncertainty': np.min(uncertainty_results['uncertainty_score']),
                'mean_prediction_std': np.mean(uncertainty_results['prediction_std']),
                'mean_interval_width': np.mean(uncertainty_results['prediction_interval_width']),
                'uncertainty_distribution': self._categorize_uncertainty_distribution_lgb(uncertainty_results['uncertainty_score'])
            }
            
            # Stability analysis
            uncertainty_analysis['stability'] = {
                'mean_boosting_stability': np.mean(uncertainty_results['boosting_stability']),
                'stable_predictions_ratio': np.mean(uncertainty_results['boosting_stability'] > 0.8),
                'unstable_predictions_ratio': np.mean(uncertainty_results['boosting_stability'] < 0.5)
            }
            
            # LightGBM-specific uncertainty factors
            uncertainty_analysis['lightgbm_factors'] = {
                'leaf_wise_growth_impact': 'May increase uncertainty due to asymmetric trees',
                'histogram_algorithm_impact': 'Slight uncertainty increase due to binning',
                'categorical_handling_impact': 'Native handling reduces uncertainty for categorical features'
            }
            
            self.prediction_uncertainty_analysis_ = uncertainty_analysis
            
        except Exception as e:
            self.prediction_uncertainty_analysis_ = {
                'error': f'Could not analyze prediction uncertainty: {str(e)}'
            }
    
    def _categorize_uncertainty_distribution_lgb(self, uncertainty_scores):
        """Categorize uncertainty distribution for LightGBM"""
        try:
            mean_uncertainty = np.mean(uncertainty_scores)
            std_uncertainty = np.std(uncertainty_scores)
            
            if mean_uncertainty < 0.05:
                uncertainty_level = "Very Low"
            elif mean_uncertainty < 0.15:
                uncertainty_level = "Low"
            elif mean_uncertainty < 0.3:
                uncertainty_level = "Moderate"
            else:
                uncertainty_level = "High"
            
            if std_uncertainty < 0.02:
                variability = "Highly Consistent"
            elif std_uncertainty < 0.05:
                variability = "Consistent"
            elif std_uncertainty < 0.15:
                variability = "Variable"
            else:
                variability = "Highly Variable"
            
            return f"{uncertainty_level} uncertainty with {variability} distribution"
        except:
            return "Unknown distribution"
    
    def _analyze_categorical_features(self):
        """Analyze categorical feature handling in LightGBM"""
        if not self.categorical_analysis or not self.categorical_features_:
            return
        
        try:
            analysis = {}
            
            # Basic categorical feature information
            analysis['categorical_summary'] = {
                'total_categorical_features': len(self.categorical_features_),
                'total_features': len(self.feature_names_),
                'categorical_ratio': len(self.categorical_features_) / len(self.feature_names_),
                'categorical_features': self.categorical_features_
            }
            
            # Categorical feature configuration
            analysis['categorical_config'] = {
                'max_cat_threshold': self.max_cat_threshold,
                'cat_l2_regularization': self.cat_l2,
                'cat_smoothing': self.cat_smooth,
                'native_handling': True,  # LightGBM always handles categoricals natively
                'optimal_split_finding': True
            }
            
            # Benefits of native categorical handling
            analysis['native_handling_benefits'] = [
                'No need for one-hot encoding',
                'Optimal split finding for categorical features',
                'Memory efficient representation',
                'Automatic handling of unseen categories',
                'Better performance than traditional encoding methods',
                'Reduced feature space dimensionality'
            ]
            
            # Categorical feature importance analysis
            if self.feature_importance_analysis_:
                top_features = self.feature_importance_analysis_.get('top_features', [])
                categorical_in_top = [f for f in top_features if f.get('is_categorical', False)]
                
                analysis['categorical_importance'] = {
                    'categorical_in_top_15': len(categorical_in_top),
                    'top_categorical_features': categorical_in_top[:5],
                    'categorical_dominance': len(categorical_in_top) / min(15, len(top_features)) if top_features else 0
                }
            
            self.categorical_analysis_ = analysis
            
        except Exception as e:
            self.categorical_analysis_ = {
                'error': f'Could not analyze categorical features: {str(e)}'
            }
# Replace lines 1503-1518 with:
    def _analyze_hyperparameter_sensitivity(self):
        """Analyze sensitivity to hyperparameter changes for LightGBM"""
        if not self.hyperparameter_sensitivity_analysis:
            return
        
        try:
            sensitivity_analysis = {
                'learning_rate_sensitivity': self._assess_learning_rate_sensitivity_lgb(),
                'num_leaves_sensitivity': self._assess_num_leaves_sensitivity(),
                'regularization_sensitivity': self._assess_regularization_sensitivity_lgb(),
                'sampling_sensitivity': self._assess_sampling_sensitivity_lgb()
            }
            
            # Overall sensitivity assessment
            sensitivities = [
                sensitivity_analysis['learning_rate_sensitivity']['sensitivity_score'],
                sensitivity_analysis['num_leaves_sensitivity']['sensitivity_score'],
                sensitivity_analysis['regularization_sensitivity']['sensitivity_score'],
                sensitivity_analysis['sampling_sensitivity']['sensitivity_score']
            ]
            
            mean_sensitivity = np.mean(sensitivities)
            
            if mean_sensitivity > 0.7:
                overall_assessment = "High sensitivity - Careful tuning required"
            elif mean_sensitivity > 0.4:
                overall_assessment = "Moderate sensitivity - Standard tuning beneficial"
            else:
                overall_assessment = "Low sensitivity - Robust to parameter changes"
            
            sensitivity_analysis['overall_assessment'] = overall_assessment
            sensitivity_analysis['mean_sensitivity_score'] = mean_sensitivity
            
            self.hyperparameter_sensitivity_analysis_ = sensitivity_analysis
            
        except Exception as e:
            self.hyperparameter_sensitivity_analysis_ = {
                'error': f'Could not analyze hyperparameter sensitivity: {str(e)}'
            }
    
    def _assess_learning_rate_sensitivity_lgb(self):
        """Assess learning rate sensitivity for LightGBM"""
        current_lr = self.learning_rate
        
        if current_lr >= 0.3:
            sensitivity = 0.9
            recommendation = "High learning rate - very sensitive to changes"
        elif current_lr >= 0.1:
            sensitivity = 0.6
            recommendation = "Standard learning rate - moderate sensitivity"
        elif current_lr >= 0.05:
            sensitivity = 0.4
            recommendation = "Conservative learning rate - good stability"
        else:
            sensitivity = 0.2
            recommendation = "Very conservative learning rate - minimal sensitivity"
        
        return {
            'current_value': current_lr,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _assess_num_leaves_sensitivity(self):
        """Assess sensitivity to num_leaves changes"""
        current_leaves = self.num_leaves
        
        if current_leaves <= 15:
            sensitivity = 0.8
            recommendation = "Few leaves - sensitive to increases"
        elif current_leaves <= 31:
            sensitivity = 0.6
            recommendation = "Standard leaf count - balanced sensitivity"
        elif current_leaves <= 127:
            sensitivity = 0.4
            recommendation = "High leaf count - moderate sensitivity"
        else:
            sensitivity = 0.3
            recommendation = "Very high leaf count - less sensitive to changes"
        
        return {
            'current_value': current_leaves,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _assess_regularization_sensitivity_lgb(self):
        """Assess sensitivity to regularization changes in LightGBM"""
        total_reg = self.reg_alpha + self.reg_lambda + (self.min_child_samples / 100)
        
        if total_reg < 0.1:
            sensitivity = 0.9
            recommendation = "Low regularization - highly sensitive to increases"
        elif total_reg < 0.5:
            sensitivity = 0.6
            recommendation = "Moderate regularization - balanced sensitivity"
        elif total_reg < 1.0:
            sensitivity = 0.3
            recommendation = "High regularization - less sensitive to changes"
        else:
            sensitivity = 0.1
            recommendation = "Very high regularization - minimal sensitivity"
        
        return {
            'current_value': total_reg,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _assess_sampling_sensitivity_lgb(self):
        """Assess sensitivity to sampling parameter changes in LightGBM"""
        total_sampling = self.feature_fraction * self.subsample
        
        if total_sampling >= 0.9:
            sensitivity = 0.7
            recommendation = "High sampling ratio - sensitive to decreases"
        elif total_sampling >= 0.7:
            sensitivity = 0.5
            recommendation = "Moderate sampling - balanced sensitivity"
        elif total_sampling >= 0.5:
            sensitivity = 0.3
            recommendation = "Low sampling - less sensitive to changes"
        else:
            sensitivity = 0.2
            recommendation = "Very low sampling - minimal sensitivity"
        
        return {
            'current_value': total_sampling,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _profile_performance(self):
        """Profile LightGBM performance characteristics"""
        if not self.performance_profiling:
            return
        
        try:
            import time
            
            # Performance profiling
            profile = {
                'algorithm': 'LightGBM',
                'training_algorithm': 'Histogram-based gradient boosting',
                'tree_growth': 'Leaf-wise',
                'memory_optimization': 'Advanced (histogram + sparse)',
                'categorical_handling': 'Native optimal splits',
                'parallel_efficiency': 'Excellent',
                'cache_friendliness': 'Optimized'
            }
            
            # Memory usage estimation
            n_samples, n_features = self.X_original_.shape
            estimated_memory_mb = self._estimate_lightgbm_memory_usage(n_samples, n_features)
            
            profile['memory_usage'] = {
                'estimated_peak_memory_mb': estimated_memory_mb,
                'memory_efficiency': 'High - histogram algorithm',
                'sparse_optimization': True,
                'categorical_memory_savings': len(self.categorical_features_ or []) > 0
            }
            
            # Training speed characteristics
            profile['speed_characteristics'] = {
                'histogram_binning': 'Very fast feature discretization',
                'leaf_wise_growth': 'Faster than level-wise',
                'optimal_split_finding': 'O(#features × #bins) complexity',
                'parallel_training': self.num_threads != 1,
                'gpu_acceleration': self.device_type == 'gpu'
            }
            
            # Scalability assessment
            profile['scalability'] = {
                'dataset_size_efficiency': self._assess_dataset_size_efficiency(),
                'feature_count_efficiency': self._assess_feature_count_efficiency(),
                'distributed_capable': True,
                'streaming_capable': False
            }
            
            self.performance_profile_ = profile
            
        except Exception as e:
            self.performance_profile_ = {
                'error': f'Could not profile performance: {str(e)}'
            }
    
    def _estimate_lightgbm_memory_usage(self, n_samples, n_features):
        """Estimate LightGBM memory usage"""
        try:
            # Base memory for data
            data_memory = n_samples * n_features * 4 / (1024 * 1024)  # 4 bytes per float, convert to MB
            
            # Histogram memory
            histogram_memory = n_features * self.max_bin * 8 / (1024 * 1024)  # 8 bytes per bin
            
            # Tree memory
            trees_memory = self.n_estimators * self.num_leaves * 32 / (1024 * 1024)  # ~32 bytes per leaf
            
            # Additional overhead
            overhead = (data_memory + histogram_memory + trees_memory) * 0.3
            
            total_memory = data_memory + histogram_memory + trees_memory + overhead
            
            return total_memory
        except:
            return 0
    
    def _assess_dataset_size_efficiency(self):
        """Assess efficiency based on dataset size"""
        n_samples = self.X_original_.shape[0]
        
        if n_samples >= 100000:
            return "Excellent - LightGBM optimized for large datasets"
        elif n_samples >= 10000:
            return "Very Good - Good performance on medium datasets"
        elif n_samples >= 1000:
            return "Good - Suitable for small-medium datasets"
        else:
            return "Fair - Consider simpler algorithms for very small datasets"
    
    def _assess_feature_count_efficiency(self):
        """Assess efficiency based on feature count"""
        n_features = self.X_original_.shape[1]
        
        if n_features >= 1000:
            return "Excellent - Histogram algorithm handles high dimensionality well"
        elif n_features >= 100:
            return "Very Good - Efficient feature processing"
        elif n_features >= 20:
            return "Good - Standard feature handling"
        else:
            return "Good - Efficient even with few features"
    
    def _compare_with_xgboost(self):
        """Compare LightGBM performance with XGBoost"""
        if not self.compare_with_xgboost:
            return
        
        try:
            # Try to import and use XGBoost for comparison
            try:
                import xgboost as xgb
                
                # Create comparable XGBoost model
                xgb_model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=6,  # Approximate equivalent
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    n_jobs=self.num_threads,
                    verbosity=0
                )
                
                # Fit and evaluate XGBoost model
                xgb_model.fit(self.X_original_, self.y_original_)
                xgb_pred = xgb_model.predict(self.X_original_)
                
                # LightGBM predictions
                lgb_pred = self.model_.predict(self.X_original_)
                
                # Compare metrics
                from sklearn.metrics import mean_squared_error, r2_score
                
                xgb_mse = mean_squared_error(self.y_original_, xgb_pred)
                xgb_r2 = r2_score(self.y_original_, xgb_pred)
                
                lgb_mse = mean_squared_error(self.y_original_, lgb_pred)
                lgb_r2 = r2_score(self.y_original_, lgb_pred)
                
                comparison = {
                    'xgboost': {
                        'mse': xgb_mse,
                        'rmse': np.sqrt(xgb_mse),
                        'r2': xgb_r2
                    },
                    'lightgbm': {
                        'mse': lgb_mse,
                        'rmse': np.sqrt(lgb_mse),
                        'r2': lgb_r2
                    },
                    'performance_comparison': {
                        'mse_improvement': (xgb_mse - lgb_mse) / xgb_mse * 100,
                        'r2_improvement': (lgb_r2 - xgb_r2) / abs(xgb_r2) * 100 if xgb_r2 != 0 else 0,
                        'better_model': 'LightGBM' if lgb_mse < xgb_mse else 'XGBoost'
                    },
                    'advantages': {
                        'lightgbm': [
                            'Faster training speed',
                            'Lower memory usage',
                            'Native categorical feature support',
                            'Better performance on large datasets',
                            'More memory efficient'
                        ],
                        'xgboost': [
                            'More mature ecosystem',
                            'Better hyperparameter stability',
                            'More extensive documentation',
                            'Better performance on small datasets',
                            'More regularization options'
                        ]
                    }
                }
                
                self.xgboost_comparison_ = comparison
                
            except ImportError:
                self.xgboost_comparison_ = {
                    'error': 'XGBoost not available for comparison'
                }
                
        except Exception as e:
            self.xgboost_comparison_ = {
                'error': f'Could not compare with XGBoost: {str(e)}'
            }
    
    def _compare_with_sklearn_gbm(self):
        """Compare LightGBM performance with sklearn's GradientBoostingRegressor"""
        if not self.compare_with_sklearn_gbm:
            return
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Create comparable sklearn model
            sklearn_gbm = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=6,  # Approximate equivalent
                random_state=self.random_state,
                subsample=self.subsample
            )
            
            # Fit and evaluate sklearn model
            sklearn_gbm.fit(self.X_original_, self.y_original_)
            sklearn_pred = sklearn_gbm.predict(self.X_original_)
            
            # LightGBM predictions
            lgb_pred = self.model_.predict(self.X_original_)
            
            # Compare metrics
            from sklearn.metrics import mean_squared_error, r2_score
            
            sklearn_mse = mean_squared_error(self.y_original_, sklearn_pred)
            sklearn_r2 = r2_score(self.y_original_, sklearn_pred)
            
            lgb_mse = mean_squared_error(self.y_original_, lgb_pred)
            lgb_r2 = r2_score(self.y_original_, lgb_pred)
            
            comparison = {
                'sklearn_gbm': {
                    'mse': sklearn_mse,
                    'rmse': np.sqrt(sklearn_mse),
                    'r2': sklearn_r2
                },
                'lightgbm': {
                    'mse': lgb_mse,
                    'rmse': np.sqrt(lgb_mse),
                    'r2': lgb_r2
                },
                'performance_comparison': {
                    'mse_improvement': (sklearn_mse - lgb_mse) / sklearn_mse * 100,
                    'r2_improvement': (lgb_r2 - sklearn_r2) / abs(sklearn_r2) * 100 if sklearn_r2 != 0 else 0,
                    'better_model': 'LightGBM' if lgb_mse < sklearn_mse else 'Sklearn GBM'
                },
                'advantages': {
                    'lightgbm': [
                        'Much faster training',
                        'Lower memory usage',
                        'Native categorical handling',
                        'GPU acceleration support',
                        'Better performance on large datasets'
                    ],
                    'sklearn_gbm': [
                        'No additional dependencies',
                        'More interpretable',
                        'Better integration with sklearn',
                        'More stable on small datasets',
                        'Simpler parameter space'
                    ]
                }
            }
            
            self.sklearn_gbm_comparison_ = comparison
            
        except Exception as e:
            self.sklearn_gbm_comparison_ = {
                'error': f'Could not compare with sklearn GBM: {str(e)}'
            }
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different parameter categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚡ Core LightGBM", 
            "🌳 Tree Structure", 
            "🛡️ Regularization", 
            "📊 Analysis Options",
            "📚 Documentation"
        ])
        
        with tab1:
            st.markdown("**Core LightGBM Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input(
                    "Number of Estimators:",
                    value=self.n_estimators,
                    min_value=10,
                    max_value=2000,
                    step=10,
                    help="Number of boosting iterations. LightGBM is fast, so higher values are feasible",
                    key=f"{key_prefix}_n_estimators"
                )
                
                learning_rate = st.number_input(
                    "Learning Rate:",
                    value=self.learning_rate,
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    format="%.3f",
                    help="Step size shrinkage. LightGBM often works well with higher learning rates",
                    key=f"{key_prefix}_learning_rate"
                )
                
                num_leaves = st.number_input(
                    "Number of Leaves:",
                    value=self.num_leaves,
                    min_value=2,
                    max_value=1000,
                    step=1,
                    help="Max number of leaves in one tree. Key parameter for LightGBM",
                    key=f"{key_prefix}_num_leaves"
                )
                
                max_depth = st.number_input(
                    "Max Depth (-1=unlimited):",
                    value=self.max_depth,
                    min_value=-1,
                    max_value=50,
                    step=1,
                    help="Maximum tree depth. -1 means no limit (controlled by num_leaves)",
                    key=f"{key_prefix}_max_depth"
                )
            
            with col2:
                boosting_type = st.selectbox(
                    "Boosting Type:",
                    options=['gbdt', 'dart', 'goss', 'rf'],
                    index=['gbdt', 'dart', 'goss', 'rf'].index(self.boosting_type),
                    help="Boosting algorithm: gbdt (standard), dart (dropout), goss (sampling), rf (random forest)",
                    key=f"{key_prefix}_boosting_type"
                )
                
                objective = st.selectbox(
                    "Objective Function:",
                    options=['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie'],
                    index=['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie'].index(self.objective),
                    help="Loss function to optimize",
                    key=f"{key_prefix}_objective"
                )
                
                metric = st.selectbox(
                    "Evaluation Metric:",
                    options=['rmse', 'mae', 'mape', 'huber', 'fair', 'poisson', 'quantile', 'gamma'],
                    index=['rmse', 'mae', 'mape', 'huber', 'fair', 'poisson', 'quantile', 'gamma'].index(self.metric),
                    help="Metric for evaluation and early stopping",
                    key=f"{key_prefix}_metric"
                )
                
                device_type = st.selectbox(
                    "Device Type:",
                    options=['cpu', 'gpu'],
                    index=['cpu', 'gpu'].index(self.device_type),
                    help="Training device: CPU or GPU acceleration",
                    key=f"{key_prefix}_device_type"
                )
        
        with tab2:
            st.markdown("**Tree Structure & Sampling**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_child_samples = st.number_input(
                    "Min Child Samples:",
                    value=self.min_child_samples,
                    min_value=1,
                    max_value=1000,
                    step=1,
                    help="Minimum number of data needed in a child (leaf)",
                    key=f"{key_prefix}_min_child_samples"
                )
                
                min_child_weight = st.number_input(
                    "Min Child Weight:",
                    value=self.min_child_weight,
                    min_value=1e-6,
                    max_value=1000.0,
                    step=0.001,
                    format="%.3f",
                    help="Minimum sum of instance weight (hessian) needed in a child",
                    key=f"{key_prefix}_min_child_weight"
                )
                
                min_split_gain = st.number_input(
                    "Min Split Gain:",
                    value=self.min_split_gain,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="Minimum loss reduction required to make a further partition",
                    key=f"{key_prefix}_min_split_gain"
                )
                
                subsample = st.number_input(
                    "Row Sampling Ratio:",
                    value=self.subsample,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of the training instances",
                    key=f"{key_prefix}_subsample"
                )
                
                subsample_freq = st.number_input(
                    "Subsample Frequency:",
                    value=self.subsample_freq,
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="Frequency of subsample. 0 means disable",
                    key=f"{key_prefix}_subsample_freq"
                )
            
            with col2:
                colsample_bytree = st.number_input(
                    "Feature Sampling Ratio:",
                    value=self.colsample_bytree,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of columns when constructing each tree",
                    key=f"{key_prefix}_colsample_bytree"
                )
                
                feature_fraction = st.number_input(
                    "Feature Fraction:",
                    value=self.feature_fraction,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="LightGBM will randomly select subset of features on each iteration",
                    key=f"{key_prefix}_feature_fraction"
                )
                
                feature_fraction_bynode = st.number_input(
                    "Feature Fraction by Node:",
                    value=self.feature_fraction_bynode,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="LightGBM will randomly select subset of features on each tree node",
                    key=f"{key_prefix}_feature_fraction_bynode"
                )
                
                extra_trees = st.checkbox(
                    "Extra Trees:",
                    value=self.extra_trees,
                    help="Use extremely randomized trees",
                    key=f"{key_prefix}_extra_trees"
                )
                
                max_bin = st.number_input(
                    "Max Bins:",
                    value=self.max_bin,
                    min_value=16,
                    max_value=512,
                    step=16,
                    help="Max number of bins that feature values will be bucketed in",
                    key=f"{key_prefix}_max_bin"
                )
        
        with tab3:
            st.markdown("**Regularization Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                reg_alpha = st.number_input(
                    "L1 Regularization:",
                    value=self.reg_alpha,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="L1 regularization term on weights",
                    key=f"{key_prefix}_reg_alpha"
                )
                
                reg_lambda = st.number_input(
                    "L2 Regularization:",
                    value=self.reg_lambda,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="L2 regularization term on weights",
                    key=f"{key_prefix}_reg_lambda"
                )
                
                min_gain_to_split = st.number_input(
                    "Min Gain to Split:",
                    value=self.min_gain_to_split,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="Minimum gain to perform split",
                    key=f"{key_prefix}_min_gain_to_split"
                )
            
            with col2:
                # Categorical feature handling
                st.markdown("**Categorical Feature Parameters**")
                
                max_cat_threshold = st.number_input(
                    "Max Category Threshold:",
                    value=self.max_cat_threshold,
                    min_value=1,
                    max_value=1000,
                    step=1,
                    help="Limit number of categories in a categorical feature",
                    key=f"{key_prefix}_max_cat_threshold"
                )
                
                cat_l2 = st.number_input(
                    "Categorical L2:",
                    value=self.cat_l2,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    help="L2 regularization in categorical split",
                    key=f"{key_prefix}_cat_l2"
                )
                
                cat_smooth = st.number_input(
                    "Categorical Smoothing:",
                    value=self.cat_smooth,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    help="Smoothing for categorical split",
                    key=f"{key_prefix}_cat_smooth"
                )
        
        with tab4:
            st.markdown("**Early Stopping & Validation**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                early_stopping_rounds = st.selectbox(
                    "Early Stopping Rounds:",
                    options=[None, 10, 20, 50, 100],
                    index=0 if self.early_stopping_rounds is None else [None, 10, 20, 50, 100].index(self.early_stopping_rounds),
                    help="Stop training if no improvement for N rounds",
                    key=f"{key_prefix}_early_stopping_rounds"
                )
                
                if early_stopping_rounds is not None:
                    validation_fraction = st.number_input(
                        "Validation Fraction:",
                        value=self.validation_fraction,
                        min_value=0.1,
                        max_value=0.3,
                        step=0.05,
                        help="Fraction of data used for validation",
                        key=f"{key_prefix}_validation_fraction"
                    )
                else:
                    validation_fraction = self.validation_fraction
                
                random_state = st.number_input(
                    "Random Seed:",
                    value=int(self.random_state),
                    min_value=0,
                    max_value=1000,
                    help="For reproducible results",
                    key=f"{key_prefix}_random_state"
                )
                
                num_threads = st.number_input(
                    "Number of Threads:",
                    value=self.num_threads,
                    min_value=-1,
                    max_value=16,
                    help="-1 uses all available cores",
                    key=f"{key_prefix}_num_threads"
                )
            
            with col2:
                verbose = st.selectbox(
                    "Verbose Output:",
                    options=[-1, 0, 1, 2],
                    index=[-1, 0, 1, 2].index(self.verbose),
                    help="Control training output verbosity",
                    key=f"{key_prefix}_verbose"
                )
                
                # DART parameters (if DART boosting is selected)
                if boosting_type == 'dart':
                    st.markdown("**DART Parameters**")
                    dart_rate_drop = st.number_input(
                        "Drop Rate:",
                        value=self.dart_rate_drop,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        help="Dropout rate for DART",
                        key=f"{key_prefix}_dart_rate_drop"
                    )
                    
                    max_drop = st.number_input(
                        "Max Drop:",
                        value=self.max_drop,
                        min_value=1,
                        max_value=200,
                        step=1,
                        help="Max number of dropped trees in one iteration",
                        key=f"{key_prefix}_max_drop"
                    )
                    
                    skip_drop = st.number_input(
                        "Skip Drop:",
                        value=self.skip_drop,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        help="Probability of skipping drop",
                        key=f"{key_prefix}_skip_drop"
                    )
                    
                    uniform_drop = st.checkbox(
                        "Uniform Drop:",
                        value=self.uniform_drop,
                        help="Use uniform drop instead of proportional drop",
                        key=f"{key_prefix}_uniform_drop"
                    )
                else:
                    dart_rate_drop = self.dart_rate_drop
                    max_drop = self.max_drop
                    skip_drop = self.skip_drop
                    uniform_drop = self.uniform_drop
            
            # Analysis options
            st.markdown("**Analysis Options**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                compute_feature_importance = st.checkbox(
                    "Feature Importance Analysis",
                    value=self.compute_feature_importance,
                    help="Compute comprehensive feature importance analysis",
                    key=f"{key_prefix}_compute_feature_importance"
                )
                
                compute_permutation_importance = st.checkbox(
                    "Permutation Importance",
                    value=self.compute_permutation_importance,
                    help="Compute permutation-based feature importance",
                    key=f"{key_prefix}_compute_permutation_importance"
                )
                
                lightgbm_analysis = st.checkbox(
                    "LightGBM Specific Analysis",
                    value=self.lightgbm_analysis,
                    help="Analyze LightGBM-specific optimizations and features",
                    key=f"{key_prefix}_lightgbm_analysis"
                )
                
                early_stopping_analysis = st.checkbox(
                    "Early Stopping Analysis",
                    value=self.early_stopping_analysis,
                    help="Analyze early stopping effectiveness",
                    key=f"{key_prefix}_early_stopping_analysis"
                )
                
                convergence_analysis = st.checkbox(
                    "Convergence Analysis",
                    value=self.convergence_analysis,
                    help="Analyze model convergence characteristics",
                    key=f"{key_prefix}_convergence_analysis"
                )
                
                categorical_analysis = st.checkbox(
                    "Categorical Analysis",
                    value=self.categorical_analysis,
                    help="Analyze categorical feature handling",
                    key=f"{key_prefix}_categorical_analysis"
                )
            
            with col4:
                regularization_analysis = st.checkbox(
                    "Regularization Analysis",
                    value=self.regularization_analysis,
                    help="Analyze effects of different regularization techniques",
                    key=f"{key_prefix}_regularization_analysis"
                )
                
                cross_validation_analysis = st.checkbox(
                    "Cross-Validation Analysis",
                    value=self.cross_validation_analysis,
                    help="Perform cross-validation for robust evaluation",
                    key=f"{key_prefix}_cross_validation_analysis"
                )
                
                tree_analysis = st.checkbox(
                    "Tree Structure Analysis",
                    value=self.tree_analysis,
                    help="Analyze tree structures and complexity",
                    key=f"{key_prefix}_tree_analysis"
                )
                
                prediction_uncertainty_analysis = st.checkbox(
                    "Prediction Uncertainty",
                    value=self.prediction_uncertainty_analysis,
                    help="Estimate prediction uncertainty using boosting iterations",
                    key=f"{key_prefix}_prediction_uncertainty_analysis"
                )
                
                hyperparameter_sensitivity_analysis = st.checkbox(
                    "Hyperparameter Sensitivity",
                    value=self.hyperparameter_sensitivity_analysis,
                    help="Analyze sensitivity to hyperparameter changes",
                    key=f"{key_prefix}_hyperparameter_sensitivity_analysis"
                )
                
                performance_profiling = st.checkbox(
                    "Performance Profiling",
                    value=self.performance_profiling,
                    help="Profile LightGBM performance characteristics",
                    key=f"{key_prefix}_performance_profiling"
                )
            
            # Performance monitoring
            if cross_validation_analysis:
                st.markdown("**Cross-Validation Settings**")
                cv_folds = st.number_input(
                    "CV Folds:",
                    value=self.cv_folds,
                    min_value=3,
                    max_value=10,
                    step=1,
                    help="Number of cross-validation folds",
                    key=f"{key_prefix}_cv_folds"
                )
            else:
                cv_folds = self.cv_folds
            
            # Comparison options
            compare_with_xgboost = st.checkbox(
                "Compare with XGBoost",
                value=self.compare_with_xgboost,
                help="Compare performance with XGBoost",
                key=f"{key_prefix}_compare_with_xgboost"
            )
            
            compare_with_sklearn_gbm = st.checkbox(
                "Compare with Sklearn GBM",
                value=self.compare_with_sklearn_gbm,
                help="Compare performance with sklearn's GradientBoostingRegressor",
                key=f"{key_prefix}_compare_with_sklearn_gbm"
            )
            
            # Visualization options
            st.markdown("**Visualization Options**")
            plot_importance = st.checkbox(
                "Plot Feature Importance",
                value=self.plot_importance,
                help="Generate feature importance plots",
                key=f"{key_prefix}_plot_importance"
            )
            
            plot_trees = st.checkbox(
                "Plot Tree Structures",
                value=self.plot_trees,
                help="Visualize individual tree structures (for small models)",
                key=f"{key_prefix}_plot_trees"
            )
            
            if plot_trees:
                max_trees_to_plot = st.number_input(
                    "Max Trees to Plot:",
                    value=self.max_trees_to_plot,
                    min_value=1,
                    max_value=5,
                    step=1,
                    help="Maximum number of trees to visualize",
                    key=f"{key_prefix}_max_trees_to_plot"
                )
            else:
                max_trees_to_plot = self.max_trees_to_plot
            
            plot_metric_evolution = st.checkbox(
                "Plot Metric Evolution",
                value=self.plot_metric_evolution,
                help="Plot training and validation metric evolution",
                key=f"{key_prefix}_plot_metric_evolution"
            )
        
        with tab5:
            st.markdown("**LightGBM Regressor - Fast Gradient Boosting**")
            
            # Algorithm information
            if st.button("⚡ LightGBM Overview", key=f"{key_prefix}_overview"):
                st.markdown("""
                **LightGBM - Light Gradient Boosting Machine**
                
                LightGBM is Microsoft's fast, distributed, high-performance gradient boosting
                framework based on decision tree algorithms. It uses histogram-based algorithms
                and leaf-wise tree growth strategy for optimal performance.
                
                **Key Innovations:**
                • **Histogram-based Algorithm** - Fast feature binning and split finding
                • **Leaf-wise Tree Growth** - More accurate than level-wise growth
                • **Native Categorical Support** - Optimal splits for categorical features
                • **Memory Optimization** - Cache-friendly data structures
                • **Network Communication Optimization** - Efficient distributed training
                • **GPU Acceleration** - CUDA-based training support
                • **Gradient-based One-Side Sampling (GOSS)** - Smart instance sampling
                • **Exclusive Feature Bundling (EFB)** - Reduces feature dimensionality
                
                **Performance Advantages:**
                • ⚡ **Speed**: Fastest gradient boosting implementation
                • 💾 **Memory**: Lowest memory footprint among GBM frameworks
                • 🎯 **Accuracy**: Often outperforms XGBoost on large datasets
                • 📊 **Categorical**: Native handling without preprocessing
                • 🔥 **GPU**: Excellent GPU acceleration
                """)
            
            # When to use LightGBM
            if st.button("🎯 When to Use LightGBM", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases for LightGBM:**
                
                **Problem Characteristics:**
                • Large datasets where training speed is critical
                • High-dimensional datasets with many features
                • Datasets with categorical features (native support)
                • Memory-constrained environments
                • Projects requiring fast iteration and experimentation
                
                **Data Characteristics:**
                • Large datasets (10K+ samples for best performance)
                • Mixed data types with categorical features
                • High-dimensional feature spaces
                • Datasets where overfitting is a concern
                • Time-series and sequential data
                
                **Business Applications:**
                • Real-time recommendation systems
                • Large-scale web analytics
                • Financial fraud detection
                • Supply chain optimization
                • Customer behavior prediction
                • Advertising click-through rate prediction
                • E-commerce ranking systems
                • IoT sensor data analysis
                
                **Technical Requirements:**
                • Need for fastest possible training
                • Memory efficiency is important
                • Categorical features without preprocessing
                • GPU acceleration capabilities
                • Distributed training across multiple machines
                """)
            
            # LightGBM vs other methods
            if st.button("⚖️ LightGBM vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **LightGBM vs Other Gradient Boosting Methods:**
                
                **LightGBM vs XGBoost:**
                • **Speed**: LightGBM is significantly faster (2-10x)
                • **Memory**: LightGBM uses less memory
                • **Accuracy**: Comparable, LightGBM often better on large datasets
                • **Overfitting**: LightGBM more prone to overfitting on small datasets
                • **Categorical**: LightGBM has native categorical support
                
                **LightGBM vs Standard Gradient Boosting:**
                • **Speed**: 10-20x faster than sklearn GBM
                • **Memory**: Much more memory efficient
                • **Accuracy**: Significantly better accuracy
                • **Features**: More advanced features and optimizations
                • **Scalability**: Better scalability to large datasets
                
                **LightGBM vs CatBoost:**
                • **Speed**: LightGBM faster on most datasets
                • **Categorical**: CatBoost has more advanced categorical handling
                • **Overfitting**: CatBoost more robust to overfitting
                • **Memory**: LightGBM more memory efficient
                • **GPU**: LightGBM has better GPU support
                
                **LightGBM vs Random Forest:**
                • **Accuracy**: LightGBM typically higher accuracy
                • **Speed**: LightGBM faster on large datasets
                • **Overfitting**: RF more robust to overfitting
                • **Interpretability**: Both provide feature importance
                • **Hyperparameters**: LightGBM requires more tuning
                
                **LightGBM vs Neural Networks:**
                • **Tabular Data**: LightGBM superior for structured data
                • **Training Speed**: LightGBM much faster to train
                • **Interpretability**: LightGBM more interpretable
                • **Feature Engineering**: LightGBM needs less preprocessing
                • **Small Data**: LightGBM better on small-medium datasets
                """)
            
            # Best practices
            if st.button("💡 LightGBM Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **LightGBM Optimization Best Practices:**
                
                **Hyperparameter Tuning Strategy:**
                1. **Start with num_leaves** - Most important parameter (31 is good start)
                2. **Tune learning_rate and n_estimators** together
                3. **Control overfitting** with min_child_samples, reg_alpha, reg_lambda
                4. **Optimize sampling** for large datasets
                5. **Fine-tune categorical parameters** if applicable
                
                **Key Parameter Guidelines:**
                • **num_leaves**: Start with 31, increase for complex patterns
                • **learning_rate**: 0.1 for most cases, 0.05-0.3 range
                • **min_child_samples**: 20-100 to prevent overfitting
                • **max_depth**: Usually leave at -1 (unlimited)
                • **feature_fraction**: 0.8-0.9 for feature sampling
                
                **Overfitting Prevention:**
                1. **Regularization**: Use reg_alpha and reg_lambda
                2. **Sampling**: Set feature_fraction < 1.0 and subsample < 1.0
                3. **Early Stopping**: Always use with validation set
                4. **Tree Constraints**: Increase min_child_samples
                5. **Cross-Validation**: Monitor CV performance
                
                **Performance Optimization:**
                1. **Categorical Features**: Let LightGBM handle natively
                2. **Memory**: Use histogram algorithm (default)
                3. **GPU**: Enable for large datasets
                4. **Parallel**: Set num_threads=-1
                5. **Data Format**: Use LightGBM's native format for repeated training
                
                **Data Preparation:**
                1. **Categorical Encoding**: Use label encoding or keep as categorical
                2. **Missing Values**: LightGBM handles automatically
                3. **Feature Selection**: Use LightGBM's feature importance
                4. **Data Size**: LightGBM excels with 10K+ samples
                
                **Model Validation:**
                1. **Cross-Validation**: Use lgb.cv() for parameter tuning
                2. **Early Stopping**: Prevent overfitting automatically
                3. **Metric Monitoring**: Track multiple metrics
                4. **Feature Importance**: Use both gain and split importance
                5. **Prediction Intervals**: Estimate uncertainty
                """)
            
            # Advanced techniques
            if st.button("🔬 Advanced LightGBM Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced LightGBM Techniques:**
                
                **Boosting Variants:**
                • **DART (Dropouts)**: Use boosting_type='dart' for regularization
                • **GOSS**: Gradient-based One-Side Sampling for large datasets
                • **Random Forest Mode**: Use boosting_type='rf' for ensemble
                
                **Advanced Regularization:**
                • **Path Smoothing**: Use path_smooth parameter
                • **Linear Tree**: Combine linear models with trees
                • **Monotonic Constraints**: Enforce feature-target relationships
                • **Feature Interaction Constraints**: Control feature combinations
                
                **Categorical Feature Optimization:**
                • **Optimal Splits**: Let LightGBM find optimal categorical splits
                • **High Cardinality**: Use max_cat_threshold for control
                • **Regularization**: Tune cat_l2 and cat_smooth
                • **Memory Management**: Use max_cat_to_onehot for very high cardinality
                
                **Performance Optimization:**
                • **GPU Training**: Use device='gpu' for acceleration
                • **Distributed Training**: Multi-machine training
                • **Memory Mapping**: Use mmap for large datasets
                • **Feature Bundling**: Automatic via EFB algorithm
                
                **Custom Objectives and Metrics:**
                • **Quantile Regression**: Custom quantile objectives
                • **Ranking**: Learning-to-rank objectives
                • **Multi-class**: Multiclass classification extensions
                • **Custom Metrics**: Define domain-specific evaluation metrics
                
                **Model Interpretation:**
                • **SHAP Integration**: Built-in SHAP value computation
                • **Feature Importance**: Multiple importance types
                • **Tree Structure**: Analyze individual trees
                • **Partial Dependence**: Feature effect visualization
                
                **Production Deployment:**
                • **Model Serialization**: Save/load trained models
                • **Incremental Learning**: Continue training existing models
                • **A/B Testing**: Compare model versions
                • **Monitoring**: Track feature importance drift
                • **Optimization**: Model compression and quantization
                """)
        
        return {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "min_child_weight": min_child_weight,
            "min_split_gain": min_split_gain,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_gain_to_split": min_gain_to_split,
            "boosting_type": boosting_type,
            "objective": objective,
            "metric": metric,
            "num_threads": num_threads,
            "device_type": device_type,
            "max_bin": max_bin,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "extra_trees": extra_trees,
            "random_state": random_state,
            "verbose": verbose,
            "early_stopping_rounds": early_stopping_rounds,
            "validation_fraction": validation_fraction,
            "dart_rate_drop": dart_rate_drop,
            "max_drop": max_drop,
            "skip_drop": skip_drop,
            "uniform_drop": uniform_drop,
            "max_cat_threshold": max_cat_threshold,
            "cat_l2": cat_l2,
            "cat_smooth": cat_smooth,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "lightgbm_analysis": lightgbm_analysis,
            "early_stopping_analysis": early_stopping_analysis,
            "convergence_analysis": convergence_analysis,
            "regularization_analysis": regularization_analysis,
            "cross_validation_analysis": cross_validation_analysis,
            "tree_analysis": tree_analysis,
            "prediction_uncertainty_analysis": prediction_uncertainty_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "categorical_analysis": categorical_analysis,
            "performance_profiling": performance_profiling,
            "cv_folds": cv_folds,
            "compare_with_xgboost": compare_with_xgboost,
            "compare_with_sklearn_gbm": compare_with_sklearn_gbm,
            "plot_importance": plot_importance,
            "plot_trees": plot_trees,
            "max_trees_to_plot": max_trees_to_plot,
            "plot_metric_evolution": plot_metric_evolution,
            "monitor_training": True,
            "feature_interaction_analysis": True
        }

    def create_model_instance(self, **params):
        """
        Create a new LightGBM model instance with given parameters
        
        Parameters:
        -----------
        **params : dict
            Parameters to override default values
        
        Returns:
        --------
        LightGBMRegressorPlugin : object
            New instance with updated parameters
        """
        # Start with current parameters
        current_params = {
            # Core boosting parameters
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            
            # Tree structure parameters
            'min_child_samples': self.min_child_samples,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            
            # Regularization parameters
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_gain_to_split': self.min_gain_to_split,
            
            # Learning control
            'boosting_type': self.boosting_type,
            'objective': self.objective,
            'metric': self.metric,
            
            # Performance parameters
            'num_threads': self.num_threads,
            'device_type': self.device_type,
            'gpu_use_dp': self.gpu_use_dp,
            
            # Advanced parameters
            'max_bin': self.max_bin,
            'min_data_per_group': self.min_data_per_group,
            'max_cat_threshold': self.max_cat_threshold,
            'cat_l2': self.cat_l2,
            'cat_smooth': self.cat_smooth,
            
            # Feature selection
            'feature_fraction': self.feature_fraction,
            'feature_fraction_bynode': self.feature_fraction_bynode,
            'extra_trees': self.extra_trees,
            
            # Control parameters
            'random_state': self.random_state,
            'verbose': self.verbose,
            
            # Early stopping and validation
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction,
            'eval_at': self.eval_at,
            
            # Advanced boosting
            'dart_rate_drop': self.dart_rate_drop,
            'max_drop': self.max_drop,
            'skip_drop': self.skip_drop,
            'uniform_drop': self.uniform_drop,
            
            # Analysis options
            'compute_feature_importance': self.compute_feature_importance,
            'compute_permutation_importance': self.compute_permutation_importance,
            'lightgbm_analysis': self.lightgbm_analysis,
            'early_stopping_analysis': self.early_stopping_analysis,
            'hyperparameter_sensitivity_analysis': self.hyperparameter_sensitivity_analysis,
            
            # Advanced analysis
            'tree_analysis': self.tree_analysis,
            'prediction_uncertainty_analysis': self.prediction_uncertainty_analysis,
            'convergence_analysis': self.convergence_analysis,
            'regularization_analysis': self.regularization_analysis,
            'feature_interaction_analysis': self.feature_interaction_analysis,
            'cross_validation_analysis': self.cross_validation_analysis,
            'categorical_analysis': self.categorical_analysis,
            
            # Comparison analysis
            'compare_with_xgboost': self.compare_with_xgboost,
            'compare_with_sklearn_gbm': self.compare_with_sklearn_gbm,
            'performance_profiling': self.performance_profiling,
            
            # Performance monitoring
            'cv_folds': self.cv_folds,
            'monitor_training': self.monitor_training,
            
            # Visualization options
            'plot_importance': self.plot_importance,
            'plot_trees': self.plot_trees,
            'max_trees_to_plot': self.max_trees_to_plot,
            'plot_metric_evolution': self.plot_metric_evolution
        }
        
        # Update with provided parameters
        current_params.update(params)
        
        # Create and return new instance
        return LightGBMRegressorPlugin(**current_params)

    def is_compatible_with_data(self, X, y) -> Tuple[bool, str]:
        """
        Check if LightGBM is compatible with the given data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        tuple : (bool, str)
            (is_compatible, message)
        """
        try:
            # Check LightGBM availability
            if not LIGHTGBM_AVAILABLE:
                return False, "LightGBM is not installed. Install with: pip install lightgbm"
            
            # Check if target is numeric (regression)
            if hasattr(y, 'dtype'):
                if not np.issubdtype(y.dtype, np.number):
                    return False, "LightGBM Regressor requires numeric target values"
            
            # Check minimum samples
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            if n_samples < self._min_samples_required:
                return False, f"LightGBM requires at least {self._min_samples_required} samples, got {n_samples}"
            
            # Check for features
            n_features = X.shape[1] if hasattr(X, 'shape') else 1
            if n_features < 1:
                return False, "At least one feature is required"
            
            # Check for excessive missing values
            if hasattr(X, 'isnull'):
                missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
                if missing_ratio > 0.9:
                    return False, f"Too many missing values ({missing_ratio:.1%}). LightGBM handles missing values but >90% is excessive"
            
            # Check target values
            if hasattr(y, '__len__'):
                if len(np.unique(y)) == 1:
                    return False, "Target has only one unique value - no variance to predict"
            
            # All checks passed
            return True, f"✅ Compatible - {n_samples} samples, {n_features} features"
            
        except Exception as e:
            return False, f"Error checking compatibility: {str(e)}"

    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "LightGBM Regressor",
            "lightgbm_version": getattr(lgb, '__version__', 'Unknown'),
            "core_params": {
                "n_estimators": self.model_.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "boosting_type": self.boosting_type,
                "objective": self.objective
            },
            "tree_structure": {
                "min_child_samples": self.min_child_samples,
                "min_child_weight": self.min_child_weight,
                "min_split_gain": self.min_split_gain
            },
            "regularization": {
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "min_gain_to_split": self.min_gain_to_split
            },
            "sampling": {
                "subsample": self.subsample,
                "feature_fraction": self.feature_fraction,
                "colsample_bytree": self.colsample_bytree
            },
            "performance": {
                "device_type": self.device_type,
                "num_threads": self.num_threads,
                "max_bin": self.max_bin,
                "early_stopping_used": self.early_stopping_rounds is not None
            },
            "categorical": {
                "max_cat_threshold": self.max_cat_threshold,
                "cat_l2": self.cat_l2,
                "cat_smooth": self.cat_smooth,
                "categorical_features_count": len(self.categorical_features_ or [])
            }
        }

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "LightGBM Regressor",
            "type": "Fast gradient boosting with histogram-based algorithm",
            "training_completed": True,
            "optimization_features": {
                "histogram_based": True,
                "leaf_wise_growth": True,
                "native_categorical": len(self.categorical_features_ or []) > 0,
                "gpu_acceleration": self.device_type == 'gpu',
                "early_stopping": self.early_stopping_rounds is not None,
                "memory_optimized": True
            }
        }
        
        # Add analysis results if available
        if self.lightgbm_analysis_:
            info["lightgbm_analysis"] = self.lightgbm_analysis_
        
        if self.early_stopping_analysis_:
            info["early_stopping"] = self.early_stopping_analysis_
        
        if self.categorical_analysis_:
            info["categorical_analysis"] = self.categorical_analysis_
        
        return info

    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted - no analysis available"}
        
        results = {
            "algorithm": "LightGBM Regressor",
            "model_summary": {
                "n_estimators": self.model_.n_estimators,
                "best_iteration": getattr(self.model_, 'best_iteration', None),
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "boosting_type": self.boosting_type
            }
        }
        
        # Add all available analysis results
        if self.feature_importance_analysis_:
            results["feature_importance"] = self.feature_importance_analysis_
        
        if self.lightgbm_analysis_:
            results["lightgbm_specifics"] = self.lightgbm_analysis_
        
        if self.early_stopping_analysis_:
            results["early_stopping"] = self.early_stopping_analysis_
        
        if self.convergence_analysis_:
            results["convergence"] = self.convergence_analysis_
        
        if self.regularization_analysis_:
            results["regularization"] = self.regularization_analysis_
        
        if self.cross_validation_analysis_:
            results["cross_validation"] = self.cross_validation_analysis_
        
        if self.tree_analysis_:
            results["tree_analysis"] = self.tree_analysis_
        
        if self.prediction_uncertainty_analysis_:
            results["prediction_uncertainty"] = self.prediction_uncertainty_analysis_
        
        if self.categorical_analysis_:
            results["categorical_analysis"] = self.categorical_analysis_
        
        if hasattr(self, 'hyperparameter_sensitivity_analysis_') and self.hyperparameter_sensitivity_analysis_:
            results["hyperparameter_sensitivity"] = self.hyperparameter_sensitivity_analysis_
        
        if self.performance_profile_:
            results["performance_profile"] = self.performance_profile_
        
        if hasattr(self, 'xgboost_comparison_') and self.xgboost_comparison_:
            results["xgboost_comparison"] = self.xgboost_comparison_
        
        if hasattr(self, 'sklearn_gbm_comparison_') and self.sklearn_gbm_comparison_:
            results["sklearn_gbm_comparison"] = self.sklearn_gbm_comparison_
        
        return results

    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance analysis"""
        if not self.is_fitted_ or not self.feature_importance_analysis_:
            return None
        
        return self.feature_importance_analysis_

    def supports_feature_importance(self) -> bool:
        """Check if the model supports feature importance"""
        return True

    def supports_prediction_intervals(self) -> bool:
        """Check if the model supports prediction intervals"""
        return True

    def get_prediction_intervals(self, X, confidence_level=0.95):
        """Get prediction intervals using uncertainty estimation"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting prediction intervals")
        
        uncertainty_results = self.predict_with_uncertainty(X)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
        
        predictions = uncertainty_results['predictions']
        std = uncertainty_results['prediction_std']
        
        lower = predictions - z_score * std
        upper = predictions + z_score * std
        
        return {
            'predictions': predictions,
            'lower_bound': lower,
            'upper_bound': upper,
            'interval_width': upper - lower,
            'confidence_level': confidence_level,
            'uncertainty_score': uncertainty_results['uncertainty_score']
        }

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for LightGBM Regressor.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics but part of the standard interface.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics but part of the standard interface.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not applicable for regressors.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not self.model_:
            return {"error": "Model not fitted. Cannot retrieve LightGBM specific metrics."}

        metrics = {}

        # Early stopping metrics
        if hasattr(self.model_, 'best_iteration_') and self.model_.best_iteration_ is not None:
            metrics['lgbm_best_iteration'] = self.model_.best_iteration_
        
        if hasattr(self.model_, 'best_score_') and self.model_.best_score_:
            # Assuming 'validation' is the name of the eval set and self.metric is the primary metric
            # The structure is typically {'validation_set_name': {'metric_name': score_value}}
            # Example: self.model_.best_score_ = {'validation': {'rmse': 0.5}}
            if 'validation' in self.model_.best_score_ and self.metric in self.model_.best_score_['validation']:
                 metrics['lgbm_best_score_validation'] = self.model_.best_score_['validation'][self.metric]
            elif self.eval_names and self.eval_names[0] in self.model_.best_score_ and \
                 self.metric in self.model_.best_score_[self.eval_names[0]]:
                 metrics['lgbm_best_score_validation'] = self.model_.best_score_[self.eval_names[0]][self.metric]


        # Feature importance metrics
        if self.feature_importance_analysis_ and 'importance_scores' in self.feature_importance_analysis_:
            gain_importances = self.feature_importance_analysis_['importance_scores'].get('gain')
            split_importances = self.feature_importance_analysis_['importance_scores'].get('split')

            if gain_importances is not None and len(gain_importances) > 0:
                metrics['lgbm_feature_importance_gain_mean'] = float(np.mean(gain_importances))
                metrics['lgbm_feature_importance_gain_sum'] = float(np.sum(gain_importances))
                metrics['lgbm_num_active_features_gain'] = int(np.sum(gain_importances > 0))
            
            if split_importances is not None and len(split_importances) > 0:
                metrics['lgbm_feature_importance_split_mean'] = float(np.mean(split_importances))
                metrics['lgbm_feature_importance_split_sum'] = float(np.sum(split_importances))
                metrics['lgbm_num_active_features_split'] = int(np.sum(split_importances > 0))
        
        # Number of trees actually built (might differ from n_estimators if early stopping occurred)
        if hasattr(self.model_, 'n_estimators_') and self.model_.n_estimators_ is not None:
             metrics['lgbm_actual_n_estimators'] = self.model_.n_estimators_
        elif hasattr(self.model_, 'best_iteration_') and self.model_.best_iteration_ is not None:
             metrics['lgbm_actual_n_estimators'] = self.model_.best_iteration_


        # Number of leaves (parameter)
        metrics['lgbm_num_leaves_param'] = self.num_leaves
        
        # Objective function used
        metrics['lgbm_objective_used'] = self.objective
        
        # Boosting type used
        metrics['lgbm_boosting_type_used'] = self.boosting_type

        if not metrics:
            metrics['info'] = "No specific LightGBM metrics were available or calculated (e.g., early stopping not used or feature importance not computed)."
            
        return metrics
        
# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return LightGBMRegressorPlugin()

