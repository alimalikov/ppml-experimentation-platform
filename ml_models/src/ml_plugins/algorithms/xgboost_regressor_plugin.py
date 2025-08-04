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

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin


class XGBoostRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    XGBoost Regressor Plugin - Optimized Gradient Boosting
    
    XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting
    library designed to be highly efficient, flexible and portable. It implements
    machine learning algorithms under the Gradient Boosting framework with advanced
    optimizations and regularization techniques.
    
    Key Features:
    - 🚀 Optimized gradient boosting with advanced tree learning algorithm
    - 🛡️ Built-in regularization (L1 & L2) to prevent overfitting
    - ⚡ Parallel processing for faster training
    - 🔧 Handle missing values automatically
    - 📊 Early stopping with validation monitoring
    - 📈 Feature importance with gain, weight, and cover metrics
    - 🌳 Advanced tree pruning and structure optimization
    - 🎯 Support for various objective functions
    - 📉 Robust performance on structured/tabular data
    - 🔍 Extensive hyperparameter tuning capabilities
    - 💾 Memory efficient implementation
    - 🎛️ Advanced boosting parameters control
    """
    
    def __init__(
        self,
        # Core boosting parameters
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        
        # Regularization parameters
        reg_alpha=0.0,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        gamma=0.0,  # Minimum loss reduction for split
        
        # Tree construction parameters
        min_child_weight=1,
        max_delta_step=0,
        subsample=1.0,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        
        # Advanced parameters
        grow_policy='depthwise',
        max_leaves=0,
        tree_method='auto',
        sketch_eps=0.03,
        
        # Control parameters
        random_state=42,
        n_jobs=-1,
        verbose=0,
        
        # Early stopping and validation
        early_stopping_rounds=None,
        validation_fraction=0.1,
        eval_metric=None,
        
        # XGBoost specific
        booster='gbtree',
        objective='reg:squarederror',
        importance_type='gain',
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        xgboost_analysis=True,
        early_stopping_analysis=True,
        hyperparameter_sensitivity_analysis=True,
        
        # Advanced analysis
        tree_analysis=True,
        prediction_uncertainty_analysis=True,
        convergence_analysis=True,
        regularization_analysis=True,
        feature_interaction_analysis=True,
        cross_validation_analysis=True,
        
        # Comparison analysis
        compare_with_sklearn_gbm=True,
        compare_with_lightgbm=False,
        
        # Performance monitoring
        cv_folds=5,
        monitor_training=True,
        
        # Visualization options
        plot_importance=True,
        plot_trees=False,
        max_trees_to_plot=3
    ):
        super().__init__()
        
        # Check XGBoost availability
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it using:\n"
                "pip install xgboost\n"
                "or\n"
                "conda install -c conda-forge xgboost"
            )
        
        # Core boosting parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # Regularization parameters
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        
        # Tree construction parameters
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        
        # Advanced parameters
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.tree_method = tree_method
        self.sketch_eps = sketch_eps
        
        # Control parameters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Early stopping and validation
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.eval_metric = eval_metric
        
        # XGBoost specific
        self.booster = booster
        self.objective = objective
        self.importance_type = importance_type
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.xgboost_analysis = xgboost_analysis
        self.early_stopping_analysis = early_stopping_analysis
        self.hyperparameter_sensitivity_analysis = hyperparameter_sensitivity_analysis
        
        # Advanced analysis
        self.tree_analysis = tree_analysis
        self.prediction_uncertainty_analysis = prediction_uncertainty_analysis
        self.convergence_analysis = convergence_analysis
        self.regularization_analysis = regularization_analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        self.cross_validation_analysis = cross_validation_analysis
        
        # Comparison analysis
        self.compare_with_sklearn_gbm = compare_with_sklearn_gbm
        self.compare_with_lightgbm = compare_with_lightgbm
        
        # Performance monitoring
        self.cv_folds = cv_folds
        self.monitor_training = monitor_training
        
        # Visualization options
        self.plot_importance = plot_importance
        self.plot_trees = plot_trees
        self.max_trees_to_plot = max_trees_to_plot
        
        # Required plugin metadata
        self._name = "XGBoost Regressor"
        self._description = "Optimized gradient boosting with advanced regularization and performance"
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
        self.training_history_ = {}
        self.validation_scores_ = {}
        
        # Analysis results storage
        self.feature_importance_analysis_ = {}
        self.xgboost_analysis_ = {}
        self.early_stopping_analysis_ = {}
        self.convergence_analysis_ = {}
        self.regularization_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.tree_analysis_ = {}
        self.prediction_uncertainty_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the XGBoost Regressor with comprehensive analysis
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample
        
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
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store original data for analysis
        self.X_original_ = X.copy()
        self.y_original_ = y.copy()
        
        # Prepare XGBoost parameters
        xgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'max_delta_step': self.max_delta_step,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'colsample_bynode': self.colsample_bynode,
            'grow_policy': self.grow_policy,
            'max_leaves': self.max_leaves if self.max_leaves > 0 else None,
            'tree_method': self.tree_method,
            'sketch_eps': self.sketch_eps,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': self.verbose,
            'booster': self.booster,
            'objective': self.objective,
            'importance_type': self.importance_type
        }
        
        # Handle early stopping
        eval_set = None
        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
            )
            eval_set = [(X_val, y_val)]
            X, y = X_train, y_train
        
        # Create and configure XGBoost model
        self.model_ = xgb.XGBRegressor(**xgb_params)
        
        # Fit the model with optional early stopping
        fit_params = {}
        if eval_set is not None:
            fit_params.update({
                'eval_set': eval_set,
                'early_stopping_rounds': self.early_stopping_rounds,
                'verbose': False
            })
            if self.eval_metric:
                fit_params['eval_metric'] = self.eval_metric
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # Train the model
        self.model_.fit(X, y, **fit_params)
        
        # Store training history
        if hasattr(self.model_, 'evals_result_'):
            self.training_history_ = self.model_.evals_result_
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_xgboost_specifics()
        self._analyze_early_stopping()
        self._analyze_convergence()
        self._analyze_regularization_effects()
        self._analyze_cross_validation()
        self._analyze_tree_structure()
        self._analyze_prediction_uncertainty()
        
        if self.hyperparameter_sensitivity_analysis:
            self._analyze_hyperparameter_sensitivity()
        
        if self.compare_with_sklearn_gbm:
            self._compare_with_sklearn_gbm()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted XGBoost model
        
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
            pred = self.model_.predict(X, iteration_range=(0, i))
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
    
    def _analyze_feature_importance(self):
        """Analyze feature importance with XGBoost specific metrics"""
        if not self.compute_feature_importance:
            return
        
        try:
            # Get different types of importance from XGBoost
            importance_types = ['weight', 'gain', 'cover']
            importance_scores = {}
            
            for imp_type in importance_types:
                try:
                    scores = self.model_.get_booster().get_score(importance_type=imp_type)
                    # Convert to array format matching feature names
                    importance_array = np.zeros(len(self.feature_names_))
                    for i, name in enumerate(self.feature_names_):
                        feature_key = f'f{i}'  # XGBoost uses f0, f1, etc.
                        importance_array[i] = scores.get(feature_key, 0.0)
                    importance_scores[imp_type] = importance_array
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
                    'weight_importance': importance_scores['weight'][rankings['gain'][i]],
                    'cover_importance': importance_scores['cover'][rankings['gain'][i]],
                    'gain_rank': i + 1,
                    'weight_rank': np.where(rankings['weight'] == rankings['gain'][i])[0][0] + 1,
                    'cover_rank': np.where(rankings['cover'] == rankings['gain'][i])[0][0] + 1
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
                'interpretation': {
                    'gain': 'Average gain across all splits using the feature',
                    'weight': 'Number of times feature appears in trees',
                    'cover': 'Average coverage (samples affected) when feature is used'
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
    
    def _analyze_xgboost_specifics(self):
        """Analyze XGBoost specific features and optimizations"""
        if not self.xgboost_analysis:
            return
        
        try:
            analysis = {}
            
            # Model configuration analysis
            analysis['model_config'] = {
                'booster_type': self.booster,
                'objective_function': self.objective,
                'tree_method': self.tree_method,
                'grow_policy': self.grow_policy,
                'n_estimators': self.model_.n_estimators,
                'max_depth': self.model_.max_depth,
                'learning_rate': self.model_.learning_rate
            }
            
            # Regularization analysis
            analysis['regularization'] = {
                'l1_alpha': self.reg_alpha,
                'l2_lambda': self.reg_lambda,
                'gamma': self.gamma,
                'min_child_weight': self.min_child_weight,
                'regularization_strength': self.reg_alpha + self.reg_lambda
            }
            
            # Sampling analysis
            analysis['sampling'] = {
                'row_sampling': self.subsample,
                'column_sampling_tree': self.colsample_bytree,
                'column_sampling_level': self.colsample_bylevel,
                'column_sampling_node': self.colsample_bynode,
                'total_sampling_ratio': self.subsample * self.colsample_bytree
            }
            
            # Performance characteristics
            try:
                # Get model dump for analysis
                booster = self.model_.get_booster()
                analysis['performance_stats'] = {
                    'total_trees': len(booster.get_dump()),
                    'average_tree_depth': self._calculate_average_tree_depth(),
                    'model_size_estimate': self._estimate_model_size(),
                    'feature_usage_diversity': self._analyze_feature_usage_diversity()
                }
            except:
                analysis['performance_stats'] = {'error': 'Could not analyze performance stats'}
            
            # XGBoost optimizations
            analysis['optimizations'] = {
                'parallel_processing': self.n_jobs != 1,
                'tree_learning_algorithm': 'Histogram-based' if self.tree_method == 'hist' else 'Exact' if self.tree_method == 'exact' else 'Auto',
                'memory_optimization': self.tree_method in ['hist', 'gpu_hist'],
                'sparse_optimization': True,  # XGBoost always optimizes for sparse data
                'cache_optimization': True   # XGBoost uses internal caching
            }
            
            self.xgboost_analysis_ = analysis
            
        except Exception as e:
            self.xgboost_analysis_ = {
                'error': f'Could not analyze XGBoost specifics: {str(e)}'
            }
    
    def _calculate_average_tree_depth(self):
        """Calculate average depth of trees in the ensemble"""
        try:
            dumps = self.model_.get_booster().get_dump()
            depths = []
            for dump in dumps:
                # Count maximum depth by counting nested levels
                max_depth = dump.count('\t')
                depths.append(max_depth)
            return np.mean(depths) if depths else 0
        except:
            return 0
    
    def _estimate_model_size(self):
        """Estimate model size in memory"""
        try:
            # Rough estimation based on trees and parameters
            n_trees = self.model_.n_estimators
            max_nodes_per_tree = 2 ** (self.max_depth + 1) - 1
            # Assuming each node takes roughly 32 bytes (rough estimate)
            estimated_size_bytes = n_trees * max_nodes_per_tree * 32
            
            if estimated_size_bytes < 1024:
                return f"{estimated_size_bytes} bytes"
            elif estimated_size_bytes < 1024**2:
                return f"{estimated_size_bytes/1024:.1f} KB"
            elif estimated_size_bytes < 1024**3:
                return f"{estimated_size_bytes/(1024**2):.1f} MB"
            else:
                return f"{estimated_size_bytes/(1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def _analyze_feature_usage_diversity(self):
        """Analyze how diversely features are used across trees"""
        try:
            importance_weight = self.feature_importance_analysis_.get('importance_scores', {}).get('weight', np.array([]))
            if len(importance_weight) == 0:
                return 0.0
            
            # Calculate entropy of feature usage
            total_usage = np.sum(importance_weight)
            if total_usage == 0:
                return 0.0
            
            probabilities = importance_weight / total_usage
            probabilities = probabilities[probabilities > 0]  # Remove zeros for log
            
            entropy = -np.sum(probabilities * np.log2(probabilities))
            max_entropy = np.log2(len(importance_weight))
            
            return entropy / max_entropy if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def _analyze_early_stopping(self):
        """Analyze early stopping behavior and optimal iterations"""
        if not self.early_stopping_analysis:
            return
        
        try:
            analysis = {}
            
            # Check if early stopping was used
            early_stopping_used = self.early_stopping_rounds is not None
            analysis['early_stopping_used'] = early_stopping_used
            
            if early_stopping_used and hasattr(self.model_, 'best_iteration'):
                analysis['best_iteration'] = self.model_.best_iteration
                analysis['best_score'] = getattr(self.model_, 'best_score', None)
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
                analysis['training_history'] = self._analyze_training_progression()
            
            # Validation curve analysis
            if self.validation_fraction > 0:
                analysis['validation_analysis'] = self._analyze_validation_performance()
            
            self.early_stopping_analysis_ = analysis
            
        except Exception as e:
            self.early_stopping_analysis_ = {
                'error': f'Could not analyze early stopping: {str(e)}'
            }
    
    def _analyze_training_progression(self):
        """Analyze training progression from history"""
        try:
            history = self.training_history_
            progression = {}
            
            for dataset_name, metrics in history.items():
                dataset_progression = {}
                
                for metric_name, values in metrics.items():
                    if len(values) > 1:
                        # Calculate improvement metrics
                        initial_score = values[0]
                        final_score = values[-1]
                        best_score = min(values) if 'error' in metric_name.lower() or 'loss' in metric_name.lower() else max(values)
                        
                        # Find best iteration
                        best_iteration = np.argmin(values) if 'error' in metric_name.lower() or 'loss' in metric_name.lower() else np.argmax(values)
                        
                        # Calculate convergence characteristics
                        improvements = np.diff(values)
                        if 'error' in metric_name.lower() or 'loss' in metric_name.lower():
                            improvements = -improvements  # For error metrics, improvement is decrease
                        
                        dataset_progression[metric_name] = {
                            'initial_score': initial_score,
                            'final_score': final_score,
                            'best_score': best_score,
                            'best_iteration': best_iteration,
                            'total_improvement': final_score - initial_score,
                            'improvement_to_best': best_score - initial_score,
                            'mean_improvement_per_iteration': np.mean(improvements),
                            'improvement_stability': np.std(improvements),
                            'convergence_rate': self._calculate_convergence_rate(values)
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
            return convergence_rate
            
        except:
            return 0.0
    
    def _analyze_validation_performance(self):
        """Analyze validation performance characteristics"""
        try:
            # This would require access to validation data during training
            # For now, return placeholder structure
            return {
                'validation_fraction_used': self.validation_fraction,
                'early_stopping_rounds': self.early_stopping_rounds,
                'eval_metric': self.eval_metric,
                'note': 'Detailed validation analysis requires training history access'
            }
        except:
            return {'error': 'Could not analyze validation performance'}
    
    def _analyze_convergence(self):
        """Analyze overall model convergence characteristics"""
        if not self.convergence_analysis:
            return
        
        try:
            analysis = {}
            
            # Training convergence based on early stopping
            if hasattr(self.model_, 'best_iteration'):
                analysis['optimal_iterations'] = self.model_.best_iteration
                analysis['convergence_efficiency'] = self.model_.best_iteration / self.model_.n_estimators
            else:
                analysis['optimal_iterations'] = self.model_.n_estimators
                analysis['convergence_efficiency'] = 1.0
            
            # Analyze learning rate effectiveness
            analysis['learning_rate_analysis'] = {
                'learning_rate': self.learning_rate,
                'category': self._categorize_learning_rate(),
                'recommended_adjustment': self._recommend_learning_rate_adjustment()
            }
            
            # Convergence based on tree depth and complexity
            analysis['complexity_analysis'] = {
                'max_depth': self.max_depth,
                'average_depth': self._calculate_average_tree_depth(),
                'complexity_category': self._categorize_model_complexity(),
                'overfitting_risk': self._assess_overfitting_risk()
            }
            
            self.convergence_analysis_ = analysis
            
        except Exception as e:
            self.convergence_analysis_ = {
                'error': f'Could not analyze convergence: {str(e)}'
            }
    
    def _categorize_learning_rate(self):
        """Categorize the learning rate"""
        if self.learning_rate >= 0.3:
            return "High - Fast learning, may overshoot"
        elif self.learning_rate >= 0.1:
            return "Moderate - Balanced learning speed"
        elif self.learning_rate >= 0.03:
            return "Low - Stable learning, needs more iterations"
        else:
            return "Very Low - Very stable, needs many iterations"
    
    def _recommend_learning_rate_adjustment(self):
        """Recommend learning rate adjustments"""
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
    
    def _categorize_model_complexity(self):
        """Categorize model complexity based on hyperparameters"""
        complexity_score = 0
        
        # Max depth contribution
        if self.max_depth >= 10:
            complexity_score += 3
        elif self.max_depth >= 6:
            complexity_score += 2
        elif self.max_depth >= 3:
            complexity_score += 1
        
        # Number of estimators contribution
        if self.n_estimators >= 500:
            complexity_score += 3
        elif self.n_estimators >= 200:
            complexity_score += 2
        elif self.n_estimators >= 100:
            complexity_score += 1
        
        # Regularization penalty
        regularization_strength = self.reg_alpha + self.reg_lambda + self.gamma
        if regularization_strength < 0.1:
            complexity_score += 1
        elif regularization_strength > 2.0:
            complexity_score -= 1
        
        if complexity_score >= 5:
            return "High - Risk of overfitting"
        elif complexity_score >= 3:
            return "Moderate - Well-balanced"
        else:
            return "Low - May underfit complex patterns"
    
    def _assess_overfitting_risk(self):
        """Assess overfitting risk based on model configuration"""
        risk_factors = []
        
        if self.max_depth > 8:
            risk_factors.append("Deep trees (max_depth > 8)")
        
        if self.reg_alpha + self.reg_lambda < 0.1:
            risk_factors.append("Low regularization")
        
        if self.min_child_weight < 3:
            risk_factors.append("Low min_child_weight")
        
        if self.gamma < 0.1:
            risk_factors.append("Low gamma (minimum loss reduction)")
        
        if len(risk_factors) == 0:
            return "Low - Good regularization"
        elif len(risk_factors) <= 2:
            return f"Moderate - {len(risk_factors)} risk factors: {', '.join(risk_factors)}"
        else:
            return f"High - {len(risk_factors)} risk factors: {', '.join(risk_factors)}"
    
    def _analyze_regularization_effects(self):
        """Analyze the effects of different regularization techniques"""
        if not self.regularization_analysis:
            return
        
        try:
            analysis = {}
            
            # L1 and L2 regularization analysis
            analysis['l1_regularization'] = {
                'alpha': self.reg_alpha,
                'effect': 'Feature selection' if self.reg_alpha > 0 else 'None',
                'strength': 'High' if self.reg_alpha > 1.0 else 'Moderate' if self.reg_alpha > 0.1 else 'Low' if self.reg_alpha > 0 else 'None'
            }
            
            analysis['l2_regularization'] = {
                'lambda': self.reg_lambda,
                'effect': 'Weight smoothing' if self.reg_lambda > 0 else 'None',
                'strength': 'High' if self.reg_lambda > 2.0 else 'Moderate' if self.reg_lambda > 1.0 else 'Low' if self.reg_lambda > 0 else 'None'
            }
            
            # Gamma regularization (complexity control)
            analysis['gamma_regularization'] = {
                'gamma': self.gamma,
                'effect': 'Tree pruning' if self.gamma > 0 else 'None',
                'strength': 'High' if self.gamma > 1.0 else 'Moderate' if self.gamma > 0.1 else 'Low' if self.gamma > 0 else 'None'
            }
            
            # Structural regularization
            analysis['structural_regularization'] = {
                'max_depth': self.max_depth,
                'min_child_weight': self.min_child_weight,
                'effect': 'Controls tree complexity',
                'assessment': self._assess_structural_regularization()
            }
            
            # Stochastic regularization
            analysis['stochastic_regularization'] = {
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'colsample_bynode': self.colsample_bynode,
                'effect': 'Reduces overfitting through sampling',
                'total_sampling_ratio': self.subsample * self.colsample_bytree,
                'assessment': self._assess_stochastic_regularization()
            }
            
            # Overall regularization assessment
            analysis['overall_assessment'] = {
                'total_regularization_strength': self._calculate_total_regularization_strength(),
                'regularization_balance': self._assess_regularization_balance(),
                'recommendations': self._get_regularization_recommendations()
            }
            
            self.regularization_analysis_ = analysis
            
        except Exception as e:
            self.regularization_analysis_ = {
                'error': f'Could not analyze regularization effects: {str(e)}'
            }
    
    def _assess_structural_regularization(self):
        """Assess structural regularization strength"""
        if self.max_depth <= 3 and self.min_child_weight >= 5:
            return "Strong - Conservative tree structure"
        elif self.max_depth <= 6 and self.min_child_weight >= 3:
            return "Moderate - Balanced tree complexity"
        elif self.max_depth <= 10:
            return "Weak - Allows complex trees"
        else:
            return "Very Weak - Risk of overfitting"
    
    def _assess_stochastic_regularization(self):
        """Assess stochastic regularization strength"""
        total_sampling = self.subsample * self.colsample_bytree
        
        if total_sampling <= 0.5:
            return "Strong - High variance reduction"
        elif total_sampling <= 0.8:
            return "Moderate - Good balance"
        elif total_sampling < 1.0:
            return "Weak - Minimal stochastic effects"
        else:
            return "None - No stochastic regularization"
    
    def _calculate_total_regularization_strength(self):
        """Calculate overall regularization strength score"""
        score = 0
        
        # L1 and L2 contribution
        score += min(self.reg_alpha * 2, 5)  # Cap at 5
        score += min(self.reg_lambda, 5)     # Cap at 5
        score += min(self.gamma * 3, 5)      # Cap at 5
        
        # Structural contribution
        if self.max_depth <= 3:
            score += 3
        elif self.max_depth <= 6:
            score += 2
        elif self.max_depth <= 10:
            score += 1
        
        if self.min_child_weight >= 5:
            score += 2
        elif self.min_child_weight >= 3:
            score += 1
        
        # Stochastic contribution
        sampling_penalty = (1 - self.subsample) * 2 + (1 - self.colsample_bytree) * 2
        score += sampling_penalty
        
        return min(score, 20)  # Cap total score at 20
    
    def _assess_regularization_balance(self):
        """Assess balance between different regularization types"""
        l1_l2_strength = self.reg_alpha + self.reg_lambda
        structural_strength = (10 - self.max_depth) / 10 + self.min_child_weight / 10
        stochastic_strength = (1 - self.subsample) + (1 - self.colsample_bytree)
        
        strengths = [l1_l2_strength, structural_strength, stochastic_strength]
        
        if max(strengths) - min(strengths) < 0.5:
            return "Well-balanced across all types"
        elif l1_l2_strength > max(structural_strength, stochastic_strength) + 1:
            return "Heavily relies on L1/L2 regularization"
        elif structural_strength > max(l1_l2_strength, stochastic_strength) + 0.5:
            return "Heavily relies on structural regularization"
        elif stochastic_strength > max(l1_l2_strength, structural_strength) + 0.5:
            return "Heavily relies on stochastic regularization"
        else:
            return "Moderate balance with some emphasis"
    
    def _get_regularization_recommendations(self):
        """Get recommendations for regularization tuning"""
        recommendations = []
        
        total_strength = self._calculate_total_regularization_strength()
        
        if total_strength < 5:
            recommendations.append("Consider increasing regularization to prevent overfitting")
            if self.reg_lambda < 1.0:
                recommendations.append("Increase reg_lambda for L2 regularization")
            if self.gamma < 0.1:
                recommendations.append("Increase gamma for tree pruning")
        elif total_strength > 15:
            recommendations.append("Consider reducing regularization - may be too conservative")
            recommendations.append("Try reducing reg_lambda or increasing max_depth")
        
        if self.subsample == 1.0 and self.colsample_bytree == 1.0:
            recommendations.append("Consider adding stochastic regularization (subsample < 1.0)")
        
        if self.reg_alpha == 0 and len(self.feature_names_) > 50:
            recommendations.append("Consider L1 regularization (reg_alpha > 0) for feature selection")
        
        if not recommendations:
            recommendations.append("Regularization appears well-tuned")
        
        return recommendations
    
    def _analyze_cross_validation(self):
        """Perform cross-validation analysis"""
        if not self.cross_validation_analysis:
            return
        
        try:
            # Create a fresh model for CV (to avoid using fitted model)
            cv_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
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
            return "Poor - High variation, check for data leakage or small dataset"
    
    def _analyze_tree_structure(self):
        """Analyze the structure and characteristics of individual trees"""
        if not self.tree_analysis:
            return
        
        try:
            analysis = {}
            
            # Basic tree statistics
            analysis['ensemble_stats'] = {
                'total_trees': self.model_.n_estimators,
                'max_depth_setting': self.max_depth,
                'average_actual_depth': self._calculate_average_tree_depth(),
                'booster_type': self.booster
            }
            
            # Tree complexity analysis
            analysis['complexity_analysis'] = {
                'depth_utilization': self._calculate_average_tree_depth() / self.max_depth,
                'complexity_category': self._categorize_model_complexity(),
                'pruning_effectiveness': self._assess_pruning_effectiveness()
            }
            
            # Feature usage in trees
            if self.feature_importance_analysis_:
                weight_importance = self.feature_importance_analysis_.get('importance_scores', {}).get('weight', np.array([]))
                if len(weight_importance) > 0:
                    analysis['feature_usage'] = {
                        'features_used': np.sum(weight_importance > 0),
                        'features_unused': np.sum(weight_importance == 0),
                        'usage_diversity': self._analyze_feature_usage_diversity(),
                        'most_used_features': self._get_most_used_features(weight_importance)
                    }
            
            self.tree_analysis_ = analysis
            
        except Exception as e:
            self.tree_analysis_ = {
                'error': f'Could not analyze tree structure: {str(e)}'
            }
    
    def _assess_pruning_effectiveness(self):
        """Assess how effectively trees are being pruned"""
        try:
            avg_depth = self._calculate_average_tree_depth()
            max_possible_depth = self.max_depth
            
            if avg_depth < max_possible_depth * 0.5:
                return "Highly effective - Trees much shallower than max depth"
            elif avg_depth < max_possible_depth * 0.8:
                return "Moderately effective - Some pruning occurring"
            elif avg_depth < max_possible_depth * 0.95:
                return "Minimally effective - Trees near max depth"
            else:
                return "Ineffective - Trees at max depth, consider increasing max_depth"
        except:
            return "Unknown"
    
    def _get_most_used_features(self, weight_importance):
        """Get the most frequently used features across trees"""
        try:
            top_indices = np.argsort(weight_importance)[::-1][:10]
            return [
                {
                    'feature': self.feature_names_[i],
                    'usage_count': weight_importance[i],
                    'usage_percentage': weight_importance[i] / np.sum(weight_importance) * 100
                }
                for i in top_indices if weight_importance[i] > 0
            ]
        except:
            return []
    
    def _analyze_prediction_uncertainty(self):
        """Analyze prediction uncertainty characteristics"""
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
                'uncertainty_distribution': self._categorize_uncertainty_distribution(uncertainty_results['uncertainty_score'])
            }
            
            # Stability analysis
            uncertainty_analysis['stability'] = {
                'mean_boosting_stability': np.mean(uncertainty_results['boosting_stability']),
                'stable_predictions_ratio': np.mean(uncertainty_results['boosting_stability'] > 0.8),
                'unstable_predictions_ratio': np.mean(uncertainty_results['boosting_stability'] < 0.5)
            }
            
            self.prediction_uncertainty_analysis_ = uncertainty_analysis
            
        except Exception as e:
            self.prediction_uncertainty_analysis_ = {
                'error': f'Could not analyze prediction uncertainty: {str(e)}'
            }
    
    def _categorize_uncertainty_distribution(self, uncertainty_scores):
        """Categorize the distribution of uncertainty scores"""
        try:
            mean_uncertainty = np.mean(uncertainty_scores)
            std_uncertainty = np.std(uncertainty_scores)
            
            if mean_uncertainty < 0.1:
                uncertainty_level = "Low"
            elif mean_uncertainty < 0.3:
                uncertainty_level = "Moderate"
            else:
                uncertainty_level = "High"
            
            if std_uncertainty < 0.05:
                variability = "Consistent"
            elif std_uncertainty < 0.15:
                variability = "Variable"
            else:
                variability = "Highly Variable"
            
            return f"{uncertainty_level} uncertainty with {variability} distribution"
        except:
            return "Unknown distribution"
    
    def _analyze_hyperparameter_sensitivity(self):
        """Analyze sensitivity to hyperparameter changes"""
        if not self.hyperparameter_sensitivity_analysis:
            return
        
        try:
            # This is a simplified sensitivity analysis
            # In practice, you'd want to test multiple hyperparameter values
            
            sensitivity_analysis = {
                'learning_rate_sensitivity': self._assess_learning_rate_sensitivity(),
                'depth_sensitivity': self._assess_depth_sensitivity(),
                'regularization_sensitivity': self._assess_regularization_sensitivity(),
                'sampling_sensitivity': self._assess_sampling_sensitivity()
            }
            
            # Overall sensitivity assessment
            sensitivities = [
                sensitivity_analysis['learning_rate_sensitivity']['sensitivity_score'],
                sensitivity_analysis['depth_sensitivity']['sensitivity_score'],
                sensitivity_analysis['regularization_sensitivity']['sensitivity_score'],
                sensitivity_analysis['sampling_sensitivity']['sensitivity_score']
            ]
            
            mean_sensitivity = np.mean(sensitivities)
            
            if mean_sensitivity > 0.7:
                overall_assessment = "High sensitivity - Careful tuning required"
            elif mean_sensitivity > 0.4:
                overall_assessment = "Moderate sensitivity - Some tuning beneficial"
            else:
                overall_assessment = "Low sensitivity - Robust to parameter changes"
            
            sensitivity_analysis['overall_assessment'] = overall_assessment
            sensitivity_analysis['mean_sensitivity_score'] = mean_sensitivity
            
            self.hyperparameter_sensitivity_analysis_ = sensitivity_analysis
            
        except Exception as e:
            self.hyperparameter_sensitivity_analysis_ = {
                'error': f'Could not analyze hyperparameter sensitivity: {str(e)}'
            }
    
    def _assess_learning_rate_sensitivity(self):
        """Assess sensitivity to learning rate changes"""
        current_lr = self.learning_rate
        
        if current_lr >= 0.3:
            sensitivity = 0.9
            recommendation = "High learning rate - very sensitive to changes"
        elif current_lr >= 0.1:
            sensitivity = 0.6
            recommendation = "Moderate learning rate - balanced sensitivity"
        elif current_lr >= 0.03:
            sensitivity = 0.3
            recommendation = "Low learning rate - less sensitive, more stable"
        else:
            sensitivity = 0.1
            recommendation = "Very low learning rate - minimal sensitivity"
        
        return {
            'current_value': current_lr,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _assess_depth_sensitivity(self):
        """Assess sensitivity to max_depth changes"""
        current_depth = self.max_depth
        
        if current_depth <= 3:
            sensitivity = 0.8
            recommendation = "Shallow trees - sensitive to depth increases"
        elif current_depth <= 6:
            sensitivity = 0.5
            recommendation = "Moderate depth - balanced sensitivity"
        elif current_depth <= 10:
            sensitivity = 0.3
            recommendation = "Deep trees - less sensitive to depth changes"
        else:
            sensitivity = 0.2
            recommendation = "Very deep trees - minimal sensitivity to further increases"
        
        return {
            'current_value': current_depth,
            'sensitivity_score': sensitivity,
            'recommendation': recommendation
        }
    
    def _assess_regularization_sensitivity(self):
        """Assess sensitivity to regularization changes"""
        total_reg = self.reg_alpha + self.reg_lambda + self.gamma
        
        if total_reg < 0.1:
            sensitivity = 0.9
            recommendation = "Low regularization - highly sensitive to increases"
        elif total_reg < 1.0:
            sensitivity = 0.6
            recommendation = "Moderate regularization - balanced sensitivity"
        elif total_reg < 3.0:
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
    
    def _assess_sampling_sensitivity(self):
        """Assess sensitivity to sampling parameter changes"""
        total_sampling = self.subsample * self.colsample_bytree
        
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
    
    def _compare_with_sklearn_gbm(self):
        """Compare performance with sklearn's GradientBoostingRegressor"""
        if not self.compare_with_sklearn_gbm:
            return
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Create comparable sklearn model
            sklearn_gbm = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                subsample=self.subsample
            )
            
            # Fit and evaluate sklearn model
            sklearn_gbm.fit(self.X_original_, self.y_original_)
            sklearn_pred = sklearn_gbm.predict(self.X_original_)
            
            # XGBoost predictions
            xgb_pred = self.model_.predict(self.X_original_)
            
            # Compare metrics
            sklearn_mse = mean_squared_error(self.y_original_, sklearn_pred)
            sklearn_r2 = r2_score(self.y_original_, sklearn_pred)
            
            xgb_mse = mean_squared_error(self.y_original_, xgb_pred)
            xgb_r2 = r2_score(self.y_original_, xgb_pred)
            
            comparison = {
                'sklearn_gbm': {
                    'mse': sklearn_mse,
                    'rmse': np.sqrt(sklearn_mse),
                    'r2': sklearn_r2
                },
                'xgboost': {
                    'mse': xgb_mse,
                    'rmse': np.sqrt(xgb_mse),
                    'r2': xgb_r2
                },
                'performance_comparison': {
                    'mse_improvement': (sklearn_mse - xgb_mse) / sklearn_mse * 100,
                    'r2_improvement': (xgb_r2 - sklearn_r2) / abs(sklearn_r2) * 100 if sklearn_r2 != 0 else 0,
                    'better_model': 'XGBoost' if xgb_mse < sklearn_mse else 'Sklearn GBM'
                },
                'advantages': {
                    'xgboost': [
                        'Advanced regularization options',
                        'Better handling of missing values',
                        'More efficient memory usage',
                        'Parallel processing',
                        'Multiple importance metrics'
                    ],
                    'sklearn_gbm': [
                        'No additional dependencies',
                        'Simpler parameter space',
                        'Better integration with sklearn ecosystem',
                        'More interpretable default behavior'
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
            "🚀 Core Boosting", 
            "🛡️ Regularization", 
            "🌳 Tree Structure", 
            "📊 Analysis Options",
            "📚 Documentation"
        ])
        
        with tab1:
            st.markdown("**Core XGBoost Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input(
                    "Number of Estimators:",
                    value=self.n_estimators,
                    min_value=10,
                    max_value=2000,
                    step=10,
                    help="Number of gradient boosted trees. More trees = better performance but slower training",
                    key=f"{key_prefix}_n_estimators"
                )
                
                learning_rate = st.number_input(
                    "Learning Rate (eta):",
                    value=self.learning_rate,
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    format="%.3f",
                    help="Step size shrinkage to prevent overfitting. Lower values need more estimators",
                    key=f"{key_prefix}_learning_rate"
                )
                
                max_depth = st.number_input(
                    "Maximum Tree Depth:",
                    value=self.max_depth,
                    min_value=1,
                    max_value=20,
                    step=1,
                    help="Maximum depth of trees. Deeper trees model more complex patterns",
                    key=f"{key_prefix}_max_depth"
                )
                
                tree_method = st.selectbox(
                    "Tree Construction Method:",
                    options=['auto', 'exact', 'approx', 'hist'],
                    index=['auto', 'exact', 'approx', 'hist'].index(self.tree_method),
                    help="Algorithm for tree construction. 'hist' is fastest for large datasets",
                    key=f"{key_prefix}_tree_method"
                )
            
            with col2:
                booster = st.selectbox(
                    "Booster Type:",
                    options=['gbtree', 'gblinear', 'dart'],
                    index=['gbtree', 'gblinear', 'dart'].index(self.booster),
                    help="Type of booster: gbtree (trees), gblinear (linear), dart (dropout)",
                    key=f"{key_prefix}_booster"
                )
                
                objective = st.selectbox(
                    "Objective Function:",
                    options=['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror', 'reg:absoluteerror', 'reg:tweedie'],
                    index=['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror', 'reg:absoluteerror', 'reg:tweedie'].index(self.objective),
                    help="Loss function to optimize",
                    key=f"{key_prefix}_objective"
                )
                
                random_state = st.number_input(
                    "Random Seed:",
                    value=int(self.random_state),
                    min_value=0,
                    max_value=1000,
                    help="For reproducible results",
                    key=f"{key_prefix}_random_state"
                )
                
                n_jobs = st.number_input(
                    "Parallel Jobs:",
                    value=self.n_jobs,
                    min_value=-1,
                    max_value=16,
                    help="-1 uses all available cores",
                    key=f"{key_prefix}_n_jobs"
                )
        
        with tab2:
            st.markdown("**Regularization Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                reg_alpha = st.number_input(
                    "L1 Regularization (Alpha):",
                    value=self.reg_alpha,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="L1 regularization term on weights. Promotes sparsity",
                    key=f"{key_prefix}_reg_alpha"
                )
                
                reg_lambda = st.number_input(
                    "L2 Regularization (Lambda):",
                    value=self.reg_lambda,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="L2 regularization term on weights. Smooths weights",
                    key=f"{key_prefix}_reg_lambda"
                )
                
                gamma = st.number_input(
                    "Gamma (Min Split Loss):",
                    value=self.gamma,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    format="%.3f",
                    help="Minimum loss reduction required to make split. Controls tree pruning",
                    key=f"{key_prefix}_gamma"
                )
            
            with col2:
                min_child_weight = st.number_input(
                    "Min Child Weight:",
                    value=self.min_child_weight,
                    min_value=0,
                    max_value=20,
                    step=1,
                    help="Minimum sum of instance weight needed in a child",
                    key=f"{key_prefix}_min_child_weight"
                )
                
                max_delta_step = st.number_input(
                    "Max Delta Step:",
                    value=self.max_delta_step,
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="Maximum delta step allowed for each tree's weight estimation",
                    key=f"{key_prefix}_max_delta_step"
                )
                
                st.markdown("**Regularization Strategy:**")
                st.info("""
                🛡️ **L1 (Alpha)**: Feature selection, creates sparse models
                🔧 **L2 (Lambda)**: Weight smoothing, prevents large weights
                ✂️ **Gamma**: Tree pruning, prevents overfitting
                ⚖️ **Min Child Weight**: Prevents overfitting in leaf nodes
                """)
        
        with tab3:
            st.markdown("**Tree Structure & Sampling**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                grow_policy = st.selectbox(
                    "Tree Growing Policy:",
                    options=['depthwise', 'lossguide'],
                    index=['depthwise', 'lossguide'].index(self.grow_policy),
                    help="How to grow trees: depthwise (level by level) or lossguide (best loss reduction)",
                    key=f"{key_prefix}_grow_policy"
                )
                
                max_leaves = st.number_input(
                    "Max Leaves (0=unlimited):",
                    value=self.max_leaves,
                    min_value=0,
                    max_value=1000,
                    step=10,
                    help="Maximum number of leaves. Only used with lossguide policy",
                    key=f"{key_prefix}_max_leaves"
                )
                
                subsample = st.number_input(
                    "Row Sampling Ratio:",
                    value=self.subsample,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of training instances",
                    key=f"{key_prefix}_subsample"
                )
                
                colsample_bytree = st.number_input(
                    "Column Sampling (by tree):",
                    value=self.colsample_bytree,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of columns when constructing each tree",
                    key=f"{key_prefix}_colsample_bytree"
                )
            
            with col2:
                colsample_bylevel = st.number_input(
                    "Column Sampling (by level):",
                    value=self.colsample_bylevel,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of columns for each level",
                    key=f"{key_prefix}_colsample_bylevel"
                )
                
                colsample_bynode = st.number_input(
                    "Column Sampling (by node):",
                    value=self.colsample_bynode,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Subsample ratio of columns for each split",
                    key=f"{key_prefix}_colsample_bynode"
                )
                
                if tree_method == 'hist':
                    sketch_eps = st.number_input(
                        "Sketch Epsilon:",
                        value=self.sketch_eps,
                        min_value=0.01,
                        max_value=0.1,
                        step=0.01,
                        format="%.3f",
                        help="Approximation factor for histogram method",
                        key=f"{key_prefix}_sketch_eps"
                    )
                else:
                    sketch_eps = self.sketch_eps
                
                st.markdown("**Sampling Benefits:**")
                st.info("""
                🎲 **Row Sampling**: Reduces overfitting, faster training
                📊 **Column Sampling**: Feature bagging, improves generalization
                🌳 **Level/Node Sampling**: Fine-grained regularization
                """)
        
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
                    
                    eval_metric = st.selectbox(
                        "Evaluation Metric:",
                        options=[None, 'rmse', 'mae', 'mape', 'mphe'],
                        index=0 if self.eval_metric is None else [None, 'rmse', 'mae', 'mape', 'mphe'].index(self.eval_metric),
                        help="Metric for early stopping evaluation",
                        key=f"{key_prefix}_eval_metric"
                    )
                else:
                    validation_fraction = self.validation_fraction
                    eval_metric = self.eval_metric
                
                verbose = st.selectbox(
                    "Verbose Output:",
                    options=[0, 1, 2],
                    index=self.verbose,
                    help="Control training output verbosity",
                    key=f"{key_prefix}_verbose"
                )
            
            with col2:
                importance_type = st.selectbox(
                    "Feature Importance Type:",
                    options=['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
                    index=['gain', 'weight', 'cover', 'total_gain', 'total_cover'].index(self.importance_type),
                    help="Type of feature importance to compute",
                    key=f"{key_prefix}_importance_type"
                )
                
                st.markdown("**Importance Types:**")
                st.info("""
                📈 **Gain**: Average gain of splits using feature
                ⚖️ **Weight**: Number of times feature is used
                📊 **Cover**: Average coverage when feature is used
                📋 **Total**: Sum across all trees
                """)
            
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
                
                xgboost_analysis = st.checkbox(
                    "XGBoost Specific Analysis",
                    value=self.xgboost_analysis,
                    help="Analyze XGBoost-specific optimizations and features",
                    key=f"{key_prefix}_xgboost_analysis"
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
                    help="Analyze individual tree structures and complexity",
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
        
        with tab5:
            st.markdown("**XGBoost Regressor - Optimized Gradient Boosting**")
            
            # Algorithm information
            if st.button("🚀 XGBoost Overview", key=f"{key_prefix}_overview"):
                st.markdown("""
                **XGBoost - eXtreme Gradient Boosting**
                
                XGBoost is an optimized distributed gradient boosting library designed to be
                highly efficient, flexible and portable. It implements machine learning 
                algorithms under the Gradient Boosting framework with significant improvements
                over standard implementations.
                
                **Key Innovations:**
                • **Regularized Learning Objective** - Built-in L1 & L2 regularization
                • **Gradient Tree Boosting** - Second-order Taylor approximation
                • **Shrinkage and Column Subsampling** - Advanced overfitting prevention
                • **Handling Sparse Data** - Optimized sparse data structures
                • **Parallel Processing** - Multi-threaded tree construction
                • **Tree Pruning** - Max-depth-first approach with global perspective
                • **Built-in Cross-Validation** - Efficient validation during training
                • **Continue on Existing Model** - Incremental learning support
                
                **Performance Advantages:**
                • 🚀 **Speed**: 10x faster than traditional GBM implementations
                • 💾 **Memory**: Efficient memory usage and block-wise computation
                • 🎯 **Accuracy**: Consistently wins ML competitions
                • 🔧 **Flexibility**: Rich feature set and hyperparameter options
                • ⚡ **Scalability**: Distributed computing support
                """)
            
            # When to use XGBoost
            if st.button("🎯 When to Use XGBoost", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases for XGBoost:**
                
                **Problem Characteristics:**
                • High-performance requirements with accuracy priority
                • Structured/tabular data with mixed feature types
                • Competition-level machine learning problems
                • Large datasets where training speed matters
                • Problems requiring robust feature importance analysis
                
                **Data Characteristics:**
                • Medium to large datasets (1K+ samples)
                • Mixed data types (numerical, categorical, sparse)
                • Datasets with missing values
                • High-dimensional feature spaces
                • Non-linear relationships and feature interactions
                
                **Business Applications:**
                • Financial modeling and risk assessment
                • Marketing response prediction and customer analytics
                • Demand forecasting and inventory optimization
                • Quality control and anomaly detection
                • Ranking and recommendation systems
                • Medical diagnosis and treatment outcome prediction
                • Real estate and asset valuation
                • IoT sensor data analysis
                
                **Technical Requirements:**
                • Need for model interpretability through feature importance
                • Requirements for prediction uncertainty estimates
                • Memory and computational efficiency constraints
                • Need for incremental/online learning capabilities
                """)
            
            # XGBoost vs other methods
            if st.button("⚖️ XGBoost vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **XGBoost vs Other Gradient Boosting Methods:**
                
                **XGBoost vs Standard Gradient Boosting:**
                • **Accuracy**: Superior due to regularization and second-order optimization
                • **Speed**: 10x faster through parallel processing and optimizations
                • **Memory**: More efficient memory usage and cache optimization
                • **Features**: More hyperparameters and built-in functionality
                • **Robustness**: Better handling of overfitting and missing values
                
                **XGBoost vs LightGBM:**
                • **Speed**: LightGBM is faster on large datasets
                • **Memory**: LightGBM uses less memory
                • **Accuracy**: Comparable, sometimes XGBoost slightly better
                • **Stability**: XGBoost more stable, LightGBM more prone to overfitting
                • **Features**: XGBoost has more mature ecosystem
                
                **XGBoost vs Random Forest:**
                • **Accuracy**: XGBoost typically higher accuracy
                • **Overfitting**: RF more robust to overfitting
                • **Speed**: RF faster to train (parallel trees)
                • **Tuning**: XGBoost requires more hyperparameter tuning
                • **Interpretability**: Both provide feature importance
                
                **XGBoost vs Neural Networks:**
                • **Tabular Data**: XGBoost superior for structured data
                • **Unstructured Data**: Neural networks better for images/text
                • **Training Time**: XGBoost faster on small-medium datasets
                • **Interpretability**: XGBoost more interpretable
                • **Hyperparameters**: XGBoost easier to tune
                
                **XGBoost vs Linear Models:**
                • **Non-linearity**: XGBoost handles complex patterns automatically
                • **Feature Engineering**: XGBoost reduces need for manual engineering
                • **Interpretability**: Linear models more interpretable
                • **Extrapolation**: Linear models better for extrapolation
                • **Training Speed**: Linear models much faster
                """)
            
            # Best practices
            if st.button("💡 XGBoost Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **XGBoost Optimization Best Practices:**
                
                **Hyperparameter Tuning Strategy:**
                1. **Start with defaults** and establish baseline
                2. **Tune learning_rate and n_estimators** together
                3. **Optimize tree structure** (max_depth, min_child_weight)
                4. **Add regularization** (reg_alpha, reg_lambda, gamma)
                5. **Fine-tune sampling** (subsample, colsample_*)
                
                **Learning Rate Guidelines:**
                • **0.3**: Fast prototyping, small datasets
                • **0.1**: General purpose, good starting point
                • **0.03**: Large datasets, more stable learning
                • **0.01**: Maximum stability, very large datasets
                
                **Overfitting Prevention:**
                1. **Early Stopping**: Use validation set with early_stopping_rounds
                2. **Regularization**: Start with reg_lambda=1, tune reg_alpha for sparsity
                3. **Tree Constraints**: Limit max_depth (3-10), increase min_child_weight
                4. **Sampling**: Use subsample=0.8, colsample_bytree=0.8
                5. **Cross-Validation**: Monitor CV scores for stable performance
                
                **Performance Optimization:**
                1. **Tree Method**: Use 'hist' for large datasets (>10K samples)
                2. **Parallel Processing**: Set n_jobs=-1 for multi-core training
                3. **Memory Management**: Use appropriate tree_method for dataset size
                4. **Early Stopping**: Prevent unnecessary iterations
                
                **Feature Engineering:**
                1. **Minimal Required**: XGBoost handles raw features well
                2. **Missing Values**: Leave as-is, XGBoost handles automatically
                3. **Categorical Features**: Use label encoding or one-hot for high cardinality
                4. **Feature Selection**: Use XGBoost feature importance for selection
                
                **Model Validation:**
                1. **Cross-Validation**: Use xgb.cv() for robust evaluation
                2. **Stratified Sampling**: Ensure representative train/validation splits
                3. **Feature Importance**: Check importance stability across folds
                4. **Learning Curves**: Plot to detect overfitting
                5. **Prediction Analysis**: Examine residuals and prediction distributions
                
                **Production Deployment:**
                1. **Model Serialization**: Save using pickle or joblib
                2. **Version Control**: Track hyperparameters and model versions
                3. **Monitoring**: Track feature importance drift
                4. **Updates**: Regular retraining with new data
                5. **A/B Testing**: Compare against baseline models
                """)
            
            # Advanced techniques
            if st.button("🔬 Advanced XGBoost Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced XGBoost Techniques:**
                
                **Custom Objective Functions:**
                • **Business-Specific Losses**: Implement domain-specific objectives
                • **Quantile Regression**: Predict confidence intervals
                • **Multi-objective Optimization**: Balance multiple targets
                • **Robust Losses**: Handle outliers with custom loss functions
                
                **Advanced Regularization:**
                • **DART (Dropout)**: Use booster='dart' for dropout regularization
                • **Monotonic Constraints**: Enforce feature-target relationships
                • **Feature Interaction Constraints**: Control feature combinations
                • **Learning Rate Scheduling**: Adaptive learning rate decay
                
                **Hyperparameter Optimization:**
                • **Bayesian Optimization**: Use Optuna, Hyperopt for efficient search
                • **Multi-fidelity**: Use early stopping for faster hyperparameter search
                • **Population-based Training**: Evolve hyperparameters during training
                • **AutoML Integration**: Use auto-sklearn, TPOT with XGBoost
                
                **Ensemble Methods:**
                • **Stacking**: Use XGBoost as meta-learner
                • **Blending**: Combine multiple XGBoost models
                • **Multi-level Ensembles**: Hierarchical model combinations
                • **Dynamic Ensembles**: Adaptive model selection
                
                **Feature Engineering:**
                • **Automatic Feature Interaction**: Let XGBoost discover interactions
                • **Feature Importance-based Selection**: Iterative feature pruning
                • **Temporal Features**: Time-based feature engineering
                • **Target Encoding**: For high-cardinality categorical features
                
                **Model Interpretation:**
                • **SHAP Values**: Individual prediction explanations
                • **Partial Dependence Plots**: Feature effect visualization
                • **Feature Interaction Detection**: Find important feature pairs
                • **Tree Surrogate Models**: Extract interpretable rules
                
                **Distributed Computing:**
                • **Dask Integration**: Scale to larger-than-memory datasets
                • **Spark Integration**: Use with PySpark for big data
                • **Multi-GPU Training**: Accelerate with GPU clusters
                • **Federated Learning**: Train across distributed data sources
                """)
        
        return {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "max_delta_step": max_delta_step,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "grow_policy": grow_policy,
            "max_leaves": max_leaves,
            "tree_method": tree_method,
            "sketch_eps": sketch_eps,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "early_stopping_rounds": early_stopping_rounds,
            "validation_fraction": validation_fraction,
            "eval_metric": eval_metric,
            "booster": booster,
            "objective": objective,
            "importance_type": importance_type,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "xgboost_analysis": xgboost_analysis,
            "early_stopping_analysis": early_stopping_analysis,
            "convergence_analysis": convergence_analysis,
            "regularization_analysis": regularization_analysis,
            "cross_validation_analysis": cross_validation_analysis,
            "tree_analysis": tree_analysis,
            "prediction_uncertainty_analysis": prediction_uncertainty_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "cv_folds": cv_folds,
            "compare_with_sklearn_gbm": compare_with_sklearn_gbm,
            "plot_importance": plot_importance,
            "plot_trees": plot_trees,
            "max_trees_to_plot": max_trees_to_plot,
            "monitor_training": True,
            "feature_interaction_analysis": True,
            "compare_with_lightgbm": False
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return XGBoostRegressorPlugin(
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            reg_alpha=hyperparameters.get("reg_alpha", self.reg_alpha),
            reg_lambda=hyperparameters.get("reg_lambda", self.reg_lambda),
            gamma=hyperparameters.get("gamma", self.gamma),
            min_child_weight=hyperparameters.get("min_child_weight", self.min_child_weight),
            max_delta_step=hyperparameters.get("max_delta_step", self.max_delta_step),
            subsample=hyperparameters.get("subsample", self.subsample),
            colsample_bytree=hyperparameters.get("colsample_bytree", self.colsample_bytree),
            colsample_bylevel=hyperparameters.get("colsample_bylevel", self.colsample_bylevel),
            colsample_bynode=hyperparameters.get("colsample_bynode", self.colsample_bynode),
            grow_policy=hyperparameters.get("grow_policy", self.grow_policy),
            max_leaves=hyperparameters.get("max_leaves", self.max_leaves),
            tree_method=hyperparameters.get("tree_method", self.tree_method),
            sketch_eps=hyperparameters.get("sketch_eps", self.sketch_eps),
            random_state=hyperparameters.get("random_state", self.random_state),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            verbose=hyperparameters.get("verbose", self.verbose),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds", self.early_stopping_rounds),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            eval_metric=hyperparameters.get("eval_metric", self.eval_metric),
            booster=hyperparameters.get("booster", self.booster),
            objective=hyperparameters.get("objective", self.objective),
            importance_type=hyperparameters.get("importance_type", self.importance_type),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            xgboost_analysis=hyperparameters.get("xgboost_analysis", self.xgboost_analysis),
            early_stopping_analysis=hyperparameters.get("early_stopping_analysis", self.early_stopping_analysis),
            convergence_analysis=hyperparameters.get("convergence_analysis", self.convergence_analysis),
            regularization_analysis=hyperparameters.get("regularization_analysis", self.regularization_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            tree_analysis=hyperparameters.get("tree_analysis", self.tree_analysis),
            prediction_uncertainty_analysis=hyperparameters.get("prediction_uncertainty_analysis", self.prediction_uncertainty_analysis),
            hyperparameter_sensitivity_analysis=hyperparameters.get("hyperparameter_sensitivity_analysis", self.hyperparameter_sensitivity_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            compare_with_sklearn_gbm=hyperparameters.get("compare_with_sklearn_gbm", self.compare_with_sklearn_gbm),
            plot_importance=hyperparameters.get("plot_importance", self.plot_importance),
            plot_trees=hyperparameters.get("plot_trees", self.plot_trees),
            max_trees_to_plot=hyperparameters.get("max_trees_to_plot", self.max_trees_to_plot),
            monitor_training=hyperparameters.get("monitor_training", self.monitor_training),
            feature_interaction_analysis=hyperparameters.get("feature_interaction_analysis", self.feature_interaction_analysis),
            compare_with_lightgbm=hyperparameters.get("compare_with_lightgbm", self.compare_with_lightgbm)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for XGBoost (minimal preprocessing needed)"""
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
        """Check if XGBoost is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"XGBoost requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "XGBoost Regressor requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # XGBoost-specific advantages
            advantages.append("Optimized gradient boosting with advanced regularization")
            advantages.append("Excellent performance on structured/tabular data")
            advantages.append("Built-in handling of missing values")
            advantages.append("Multiple regularization mechanisms prevent overfitting")
            advantages.append("Parallel processing for faster training")
            advantages.append("Rich feature importance analysis")
            
            # Sample size assessment
            if n_samples >= 5000:
                advantages.append(f"Large dataset ({n_samples}) - ideal for XGBoost optimization")
            elif n_samples >= 1000:
                advantages.append(f"Good dataset size ({n_samples}) - suitable for XGBoost")
            elif n_samples >= 200:
                considerations.append(f"Moderate dataset size ({n_samples}) - use regularization")
            else:
                considerations.append(f"Small dataset ({n_samples}) - high regularization recommended")
            
            # Feature dimensionality
            if n_features >= 100:
                advantages.append(f"High dimensionality ({n_features}) - XGBoost excels with many features")
            elif n_features >= 20:
                advantages.append(f"Good feature count ({n_features}) - suitable for complex patterns")
            else:
                considerations.append(f"Few features ({n_features}) - may benefit from feature engineering")
            
            # Check feature-to-sample ratio
            feature_sample_ratio = n_features / n_samples
            if feature_sample_ratio > 0.5:
                considerations.append(f"High feature-to-sample ratio ({feature_sample_ratio:.2f}) - use strong regularization")
            elif feature_sample_ratio > 0.1:
                considerations.append(f"Moderate feature-to-sample ratio ({feature_sample_ratio:.2f}) - tune regularization")
            else:
                advantages.append(f"Excellent feature-to-sample ratio ({feature_sample_ratio:.2f})")
            
            # XGBoost specific benefits
            advantages.append("Advanced tree pruning prevents overfitting")
            advantages.append("L1/L2 regularization for feature selection")
            advantages.append("Early stopping with validation monitoring")
            
            # Build compatibility message
            if len(considerations) == 0:
                suitability = "Excellent"
            elif len(considerations) <= 1:
                suitability = "Very Good"
            elif len(considerations) <= 2:
                suitability = "Good"
            else:
                suitability = "Fair"
            
            message_parts = [
                f"✅ Compatible with {n_samples} samples, {n_features} features",
                f"🚀 Suitability for XGBoost: {suitability}"
            ]
            
            if advantages:
                message_parts.append("⚡ Advantages: " + "; ".join(advantages[:3]))  # Show top 3
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"

    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive feature importance analysis"""
        if not self.is_fitted_:
            return None
        
        if not self.feature_importance_analysis_:
            return None
        
        analysis = self.feature_importance_analysis_
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        return {
            'feature_importance': analysis,
            'xgboost_specific': {
                'gain_importance': analysis.get('importance_scores', {}).get('gain', []),
                'weight_importance': analysis.get('importance_scores', {}).get('weight', []),
                'cover_importance': analysis.get('importance_scores', {}).get('cover', []),
                'importance_interpretations': analysis.get('interpretation', {})
            },
            'top_features': analysis.get('top_features', []),
            'interpretation': 'XGBoost feature importance with gain, weight, and cover metrics'
        }

    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "XGBoost Regressor",
            "xgboost_version": getattr(xgb, '__version__', 'Unknown'),
            "core_params": {
                "n_estimators": self.model_.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "booster": self.booster,
                "objective": self.objective
            },
            "regularization": {
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "gamma": self.gamma,
                "min_child_weight": self.min_child_weight
            },
            "sampling": {
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "colsample_bylevel": self.colsample_bylevel,
                "colsample_bynode": self.colsample_bynode
            },
            "performance": {
                "tree_method": self.tree_method,
                "n_jobs": self.n_jobs,
                "early_stopping_used": self.early_stopping_rounds is not None
            }
        }

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "XGBoost Regressor",
            "type": "Optimized gradient boosting with advanced regularization",
            "training_completed": True,
            "optimization_features": {
                "parallel_processing": self.n_jobs != 1,
                "advanced_regularization": True,
                "automatic_missing_value_handling": True,
                "built_in_cross_validation": True,
                "early_stopping": self.early_stopping_rounds is not None
            }
        }
        
        # Add analysis results if available
        if self.xgboost_analysis_:
            info["xgboost_analysis"] = self.xgboost_analysis_
        
        if self.early_stopping_analysis_:
            info["early_stopping"] = self.early_stopping_analysis_
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the XGBoost Regressor model.

        These metrics are derived from the model's training process, internal analyses,
        and fitted attributes. Parameters y_true, y_pred are kept for API consistency
        but are not primarily used as metrics are sourced from the plugin's internal state.
        y_proba is not applicable for regressors.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing XGBoost Regressor-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self, 'model_') or self.model_ is None:
            return {"error": "Model not fitted. Cannot retrieve XGBoost Regressor specific metrics."}

        metrics = {}
        prefix = "xgbr_"  # Prefix for XGBoost Regressor specific metrics

        # Basic model info
        try:
            booster = self.model_.get_booster()
            metrics[f"{prefix}num_boosted_rounds_actual"] = booster.num_boosted_rounds()
        except Exception as e:
            metrics[f"{prefix}num_boosted_rounds_error"] = str(e)
        
        metrics[f"{prefix}n_features_in_model"] = self.model_.n_features_in_

        # Early Stopping Metrics (from model and self.early_stopping_analysis_)
        if hasattr(self.model_, 'best_iteration') and self.model_.best_iteration is not None:
            metrics[f"{prefix}best_iteration"] = self.model_.best_iteration
            if hasattr(self.model_, 'best_score') and self.model_.best_score is not None:
                metrics[f"{prefix}best_score_on_eval"] = float(self.model_.best_score)
            metrics[f"{prefix}early_stopping_triggered"] = 1 if self.model_.best_iteration < self.model_.n_estimators else 0
            if metrics[f"{prefix}early_stopping_triggered"] == 1 and hasattr(self.early_stopping_analysis_, 'get'):
                 metrics[f"{prefix}iterations_saved_by_early_stopping"] = self.early_stopping_analysis_.get('iterations_saved', 0)

        elif hasattr(self, 'early_stopping_analysis_') and self.early_stopping_analysis_:
            if 'best_iteration' in self.early_stopping_analysis_:
                metrics[f"{prefix}best_iteration"] = self.early_stopping_analysis_['best_iteration']
            if 'best_score' in self.early_stopping_analysis_:
                 metrics[f"{prefix}best_score_on_eval"] = float(self.early_stopping_analysis_['best_score'])
            metrics[f"{prefix}early_stopping_triggered"] = 1 if self.early_stopping_analysis_.get('early_stopping_triggered') else 0
            metrics[f"{prefix}iterations_saved_by_early_stopping"] = self.early_stopping_analysis_.get('iterations_saved', 0)


        # Training History Metrics (from self.training_history_)
        if hasattr(self, 'training_history_') and self.training_history_:
            for dataset_name, metric_dict in self.training_history_.items():
                for metric_name, values in metric_dict.items():
                    if values:
                        safe_dataset_name = dataset_name.replace('[', '').replace(']', '').replace('\'', '') # Sanitize name
                        metrics[f"{prefix}eval_{safe_dataset_name}_{metric_name}_final"] = float(values[-1])
                        if metrics.get(f"{prefix}best_iteration") is not None:
                            best_iter_idx = metrics[f"{prefix}best_iteration"]
                            # Ensure best_iter_idx is within bounds for the values list
                            if 0 <= best_iter_idx < len(values):
                                metrics[f"{prefix}eval_{safe_dataset_name}_{metric_name}_at_best_iter"] = float(values[best_iter_idx])
                            elif best_iter_idx == len(values): # If best_iteration is n_estimators
                                metrics[f"{prefix}eval_{safe_dataset_name}_{metric_name}_at_best_iter"] = float(values[-1])


        # Feature Importance Metrics (from self.feature_importance_analysis_)
        if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_ and 'importance_scores' in self.feature_importance_analysis_:
            for imp_type, scores_array in self.feature_importance_analysis_['importance_scores'].items():
                if scores_array is not None and len(scores_array) > 0:
                    metrics[f"{prefix}mean_importance_{imp_type}"] = float(np.mean(scores_array))
                    metrics[f"{prefix}max_importance_{imp_type}"] = float(np.max(scores_array))
                    metrics[f"{prefix}std_importance_{imp_type}"] = float(np.std(scores_array))
                    metrics[f"{prefix}num_important_features_{imp_type}"] = int(np.sum(scores_array > 0))
            if 'importance_statistics' in self.feature_importance_analysis_:
                 for imp_type, stats in self.feature_importance_analysis_['importance_statistics'].items():
                    if 'concentration' in stats and isinstance(stats['concentration'], dict):
                        metrics[f"{prefix}importance_concentration_top5_{imp_type}"] = stats['concentration'].get('top_5_concentration')
                        metrics[f"{prefix}importance_gini_coeff_{imp_type}"] = stats['concentration'].get('gini_coefficient')


        # XGBoost Specifics (from self.xgboost_analysis_)
        if hasattr(self, 'xgboost_analysis_') and self.xgboost_analysis_:
            if 'performance_stats' in self.xgboost_analysis_ and isinstance(self.xgboost_analysis_['performance_stats'], dict):
                metrics[f"{prefix}avg_tree_depth_actual"] = self.xgboost_analysis_['performance_stats'].get('average_tree_depth')
                metrics[f"{prefix}feature_usage_diversity"] = self.xgboost_analysis_['performance_stats'].get('feature_usage_diversity')

        # Cross-Validation Metrics (from self.cross_validation_analysis_)
        if hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_ and not self.cross_validation_analysis_.get('error'):
            metrics[f"{prefix}cv_mean_mse"] = self.cross_validation_analysis_.get('mean_mse')
            metrics[f"{prefix}cv_std_mse"] = self.cross_validation_analysis_.get('std_mse')
            metrics[f"{prefix}cv_mean_rmse"] = self.cross_validation_analysis_.get('mean_rmse')
            metrics[f"{prefix}cv_mean_r2"] = self.cross_validation_analysis_.get('mean_r2')
            metrics[f"{prefix}cv_std_r2"] = self.cross_validation_analysis_.get('std_r2')

        # Prediction Uncertainty Metrics (from self.prediction_uncertainty_analysis_)
        if hasattr(self, 'prediction_uncertainty_analysis_') and self.prediction_uncertainty_analysis_ and not self.prediction_uncertainty_analysis_.get('error'):
            metrics[f"{prefix}mean_prediction_uncertainty_score"] = self.prediction_uncertainty_analysis_.get('mean_uncertainty')
            metrics[f"{prefix}mean_prediction_std_from_boosting"] = self.prediction_uncertainty_analysis_.get('mean_prediction_std')
            if 'stability' in self.prediction_uncertainty_analysis_ and isinstance(self.prediction_uncertainty_analysis_['stability'], dict):
                metrics[f"{prefix}mean_boosting_stability"] = self.prediction_uncertainty_analysis_['stability'].get('mean_boosting_stability')

        # Regularization Analysis (from self.regularization_analysis_)
        if hasattr(self, 'regularization_analysis_') and self.regularization_analysis_ and 'overall_assessment' in self.regularization_analysis_:
            if isinstance(self.regularization_analysis_['overall_assessment'], dict):
                metrics[f"{prefix}total_regularization_strength_score"] = self.regularization_analysis_['overall_assessment'].get('total_regularization_strength')

        # Convergence Analysis (from self.convergence_analysis_)
        if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
            metrics[f"{prefix}convergence_efficiency"] = self.convergence_analysis_.get('convergence_efficiency')


        # Remove None values to keep metrics clean
        metrics = {k: v for k, v in metrics.items() if v is not None}

        if not metrics:
            metrics['info'] = "No specific XGBoost Regressor metrics were extracted from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return XGBoostRegressorPlugin()


# Example usage and testing
if __name__ == "__main__":
    """Example usage of XGBoost Regressor Plugin"""
    print("Testing XGBoost Regressor Plugin...")
    
    try:
        # Check XGBoost availability
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost is not installed. Please install with: pip install xgboost")
            exit(1)
        
        # Create sample data
        np.random.seed(42)
        
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
        
        # Add some missing values to test XGBoost's handling
        X_missing = X.copy()
        missing_mask = np.random.random(X.shape) < 0.05  # 5% missing values
        X_missing[missing_mask] = np.nan
        
        print(f"\n📊 Test Dataset:")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Missing values: {np.sum(missing_mask)}")
        print(f"Target variance: {np.var(y):.3f}")
        
        # Test XGBoost
        print(f"\n🚀 Testing XGBoost Regression...")
        
        # Create DataFrame for proper feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X_missing, columns=feature_names)
        
        plugin = XGBoostRegressorPlugin(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            early_stopping_rounds=10,
            validation_fraction=0.2,
            compute_feature_importance=True,
            compute_permutation_importance=True,
            xgboost_analysis=True,
            early_stopping_analysis=True,
            cross_validation_analysis=True,
            random_state=42
        )
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_df, y)
        print(f"✅ Compatibility: {message}")
        
        if compatible:
            # Train model
            plugin.fit(X_df, y)
            
            # Make predictions
            y_pred = plugin.predict(X_df)
            
            # Test uncertainty prediction
            uncertainty_results = plugin.predict_with_uncertainty(X_df[:100])
            
            # Evaluate
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            print(f"\n📊 XGBoost Results:")
            print(f"Training R²: {r2:.4f}")
            print(f"Training RMSE: {np.sqrt(mse):.4f}")
            print(f"Estimators used: {plugin.model_.n_estimators}")
            
            # Feature importance
            feature_imp = plugin.get_feature_importance()
            if feature_imp and 'xgboost_specific' in feature_imp:
                xgb_specific = feature_imp['xgboost_specific']
                print(f"\nTop 5 Features (by gain):")
                top_features = feature_imp['feature_importance']['top_features'][:5]
                for i, feat in enumerate(top_features):
                    print(f"{i+1}. {feat['name']}: gain={feat['gain_importance']:.4f}, weight={feat['weight_importance']:.0f}")
            
            # XGBoost analysis
            if plugin.xgboost_analysis_:
                xgb_analysis = plugin.xgboost_analysis_
                if 'regularization' in xgb_analysis:
                    reg = xgb_analysis['regularization']
                    print(f"\nRegularization: L1={reg['l1_alpha']}, L2={reg['l2_lambda']}, Gamma={reg['gamma']}")
            
            # Early stopping analysis
            if plugin.early_stopping_analysis_:
                early_stop = plugin.early_stopping_analysis_
                if 'best_iteration' in early_stop:
                    print(f"Early stopping: Best iteration {early_stop['best_iteration']}")
                    if early_stop.get('early_stopping_triggered'):
                        print(f"Saved {early_stop['iterations_saved']} iterations")
            
            # Uncertainty analysis
            print(f"\nPrediction Uncertainty (first 5 samples):")
            uncertainty = uncertainty_results
            for i in range(min(5, len(uncertainty['predictions']))):
                pred = uncertainty['predictions'][i]
                std = uncertainty['prediction_std'][i]
                stability = uncertainty['boosting_stability'][i]
                print(f"Sample {i+1}: {pred:.3f} ± {std:.3f} (stability: {stability:.3f})")
            
            print(f"\n✅ XGBoost plugin test completed successfully!")
            
        else:
            print(f"❌ Compatibility issue: {message}")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()