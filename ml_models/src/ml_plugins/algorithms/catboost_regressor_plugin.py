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

# CatBoost import with fallback
try:
    import catboost as cb
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None
    CatBoostRegressor = None
    Pool = None

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin


class CatBoostRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    CatBoost Regressor Plugin - Advanced Categorical Feature Handling
    
    CatBoost (Categorical Boosting) is a machine learning algorithm developed by Yandex
    that excels at handling categorical features without extensive preprocessing.
    It uses an innovative approach to process categorical features during training
    that reduces overfitting and improves generalization.
    
    Key Features:
    - 🏆 Superior categorical feature handling without preprocessing
    - 🎯 Automatic handling of categorical features with optimal encoding
    - 🛡️ Built-in protection against target leakage in categorical encoding
    - 📊 Ordered boosting to reduce overfitting
    - 🔥 High accuracy with minimal hyperparameter tuning
    - 💾 Memory efficient training and inference
    - 📈 Robust to overfitting even with default parameters
    - 🌐 GPU acceleration support
    - 📉 Built-in cross-validation and early stopping
    - 🎛️ Rich feature importance metrics
    - 🔍 Advanced regularization techniques
    - 💡 Symmetric trees for faster inference
    - 🌳 Oblivious decision trees structure
    - 📚 Comprehensive model analysis and interpretation
    """
    
    def __init__(
        self,
        # Core boosting parameters
        iterations=1000,
        learning_rate=None,  # Auto if None
        depth=6,
        l2_leaf_reg=3.0,
        
        # Categorical handling parameters
        cat_features=None,
        one_hot_max_size=2,
        max_ctr_complexity=4,
        simple_ctr_description=None,
        combinations_ctr_description=None,
        
        # Model structure parameters
        model_size_reg=0.5,
        rsm=1.0,  # Random subspace method
        border_count=254,
        feature_border_type='GreedyLogSum',
        
        # Regularization parameters
        bagging_temperature=1.0,
        random_strength=1.0,
        leaf_estimation_method='Newton',
        leaf_estimation_iterations=None,
        
        # Performance parameters
        thread_count=-1,
        used_ram_limit=None,
        gpu_ram_part=0.95,
        
        # Training control
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        
        # Loss function and metrics
        loss_function='RMSE',
        eval_metric='RMSE',
        custom_metric=None,
        
        # Overfitting detection
        od_type='IncToDec',
        od_wait=20,
        
        # Advanced features
        bootstrap_type='Bayesian',
        subsample=0.66,
        sampling_frequency='PerTreeLevel',
        leaf_estimation_backtracking='AnyImprovement',
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        catboost_analysis=True,
        categorical_analysis=True,
        overfitting_analysis=True,
        cross_validation_analysis=True,
        
        # Advanced analysis
        tree_analysis=True,
        prediction_uncertainty_analysis=True,
        convergence_analysis=True,
        regularization_analysis=True,
        categorical_encoding_analysis=True,
        feature_interaction_analysis=True,
        
        # Comparison analysis
        compare_with_lightgbm=True,
        compare_with_xgboost=True,
        compare_with_sklearn_gbm=True,
        performance_profiling=True,
        
        # Performance monitoring
        cv_folds=5,
        early_stopping_rounds=None,
        use_best_model=True,
        
        # Visualization options
        plot_importance=True,
        plot_trees=False,
        max_trees_to_plot=3,
        plot_learning_curve=True,
        plot_categorical_effects=True
    ):
        super().__init__()
        
        # Check CatBoost availability
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed. Please install it using:\n"
                "pip install catboost\n"
                "or\n"
                "conda install -c conda-forge catboost"
            )
        
        # Core boosting parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        
        # Categorical handling parameters
        self.cat_features = cat_features
        self.one_hot_max_size = one_hot_max_size
        self.max_ctr_complexity = max_ctr_complexity
        self.simple_ctr_description = simple_ctr_description
        self.combinations_ctr_description = combinations_ctr_description
        
        # Model structure parameters
        self.model_size_reg = model_size_reg
        self.rsm = rsm
        self.border_count = border_count
        self.feature_border_type = feature_border_type
        
        # Regularization parameters
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.leaf_estimation_method = leaf_estimation_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        
        # Performance parameters
        self.thread_count = thread_count
        self.used_ram_limit = used_ram_limit
        self.gpu_ram_part = gpu_ram_part
        
        # Training control
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        
        # Loss function and metrics
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.custom_metric = custom_metric
        
        # Overfitting detection
        self.od_type = od_type
        self.od_wait = od_wait
        
        # Advanced features
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.sampling_frequency = sampling_frequency
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.catboost_analysis = catboost_analysis
        self.categorical_analysis = categorical_analysis
        self.overfitting_analysis = overfitting_analysis
        self.cross_validation_analysis = cross_validation_analysis
        
        # Advanced analysis
        self.tree_analysis = tree_analysis
        self.prediction_uncertainty_analysis = prediction_uncertainty_analysis
        self.convergence_analysis = convergence_analysis
        self.regularization_analysis = regularization_analysis
        self.categorical_encoding_analysis = categorical_encoding_analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        
        # Comparison analysis
        self.compare_with_lightgbm = compare_with_lightgbm
        self.compare_with_xgboost = compare_with_xgboost
        self.compare_with_sklearn_gbm = compare_with_sklearn_gbm
        self.performance_profiling = performance_profiling
        
        # Performance monitoring
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.use_best_model = use_best_model
        
        # Visualization options
        self.plot_importance = plot_importance
        self.plot_trees = plot_trees
        self.max_trees_to_plot = max_trees_to_plot
        self.plot_learning_curve = plot_learning_curve
        self.plot_categorical_effects = plot_categorical_effects
        
        # Required plugin metadata
        self._name = "CatBoost Regressor"
        self._description = "Advanced gradient boosting with superior categorical feature handling"
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
        self.categorical_features_indices_ = None
        self.training_history_ = {}
        
        # Analysis results storage
        self.feature_importance_analysis_ = {}
        self.catboost_analysis_ = {}
        self.categorical_analysis_ = {}
        self.overfitting_analysis_ = {}
        self.convergence_analysis_ = {}
        self.regularization_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.tree_analysis_ = {}
        self.prediction_uncertainty_analysis_ = {}
        self.categorical_encoding_analysis_ = {}
        self.feature_interaction_analysis_ = {}
        self.performance_profile_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None, categorical_features=None):
        """
        Fit the CatBoost Regressor with comprehensive categorical analysis
        
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
        
        # Detect and store categorical features
        self.categorical_features_ = self._detect_categorical_features(X, categorical_features)
        self.categorical_features_indices_ = self._get_categorical_indices(X, self.categorical_features_)
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store original data for analysis
        self.X_original_ = X.copy()
        self.y_original_ = y.copy()
        
        # Prepare CatBoost parameters
        catboost_params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'model_size_reg': self.model_size_reg,
            'rsm': self.rsm,
            'border_count': self.border_count,
            'feature_border_type': self.feature_border_type,
            'bagging_temperature': self.bagging_temperature,
            'random_strength': self.random_strength,
            'leaf_estimation_method': self.leaf_estimation_method,
            'thread_count': self.thread_count,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'allow_writing_files': self.allow_writing_files,
            'loss_function': self.loss_function,
            'eval_metric': self.eval_metric,
            'od_type': self.od_type,
            'od_wait': self.od_wait,
            'bootstrap_type': self.bootstrap_type,
            'sampling_frequency': self.sampling_frequency,
            'leaf_estimation_backtracking': self.leaf_estimation_backtracking,
            'use_best_model': self.use_best_model
        }
        
        # Add learning rate if specified
        if self.learning_rate is not None:
            catboost_params['learning_rate'] = self.learning_rate
        
        # Add categorical feature parameters
        if self.categorical_features_indices_:
            catboost_params.update({
                'one_hot_max_size': self.one_hot_max_size,
                'max_ctr_complexity': self.max_ctr_complexity
            })
            
            if self.simple_ctr_description is not None:
                catboost_params['simple_ctr'] = self.simple_ctr_description
            
            if self.combinations_ctr_description is not None:
                catboost_params['combinations_ctr'] = self.combinations_ctr_description
        
        # Add bootstrap parameters based on type
        if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
            catboost_params['subsample'] = self.subsample
        
        # Add leaf estimation iterations if specified
        if self.leaf_estimation_iterations is not None:
            catboost_params['leaf_estimation_iterations'] = self.leaf_estimation_iterations
        
        # Add custom metrics if specified
        if self.custom_metric is not None:
            catboost_params['custom_metric'] = self.custom_metric
        
        # Add RAM limit if specified
        if self.used_ram_limit is not None:
            catboost_params['used_ram_limit'] = self.used_ram_limit
        
        # Handle early stopping and validation
        eval_set = None
        if self.early_stopping_rounds is not None:
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=self.random_seed
            )
            
            # Create validation pool
            if self.categorical_features_indices_:
                eval_set = Pool(X_val, y_val, cat_features=self.categorical_features_indices_)
            else:
                eval_set = Pool(X_val, y_val)
            
            X, y = X_train, y_train
            catboost_params['early_stopping_rounds'] = self.early_stopping_rounds
        
        # Create CatBoost model
        self.model_ = CatBoostRegressor(**catboost_params)
        
        # Create training pool
        if self.categorical_features_indices_:
            train_pool = Pool(X, y, cat_features=self.categorical_features_indices_, weight=sample_weight)
        else:
            train_pool = Pool(X, y, weight=sample_weight)
        
        # Train the model
        if eval_set is not None:
            self.model_.fit(train_pool, eval_set=eval_set, plot=False)
        else:
            self.model_.fit(train_pool, plot=False)
        
        # Store training history
        if hasattr(self.model_, 'get_evals_result'):
            self.training_history_ = self.model_.get_evals_result()
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_catboost_specifics()
        self._analyze_categorical_features()
        self._analyze_overfitting_behavior()
        self._analyze_convergence()
        self._analyze_regularization_effects()
        self._analyze_categorical_encoding()
        self._analyze_cross_validation()
        self._analyze_tree_structure()
        self._analyze_prediction_uncertainty()
        self._analyze_feature_interactions()
        
        if self.performance_profiling:
            self._profile_performance()
        
        if self.compare_with_lightgbm:
            self._compare_with_lightgbm()
        
        if self.compare_with_xgboost:
            self._compare_with_xgboost()
        
        if self.compare_with_sklearn_gbm:
            self._compare_with_sklearn_gbm()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted CatBoost model
        
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
        
        # Create prediction pool with categorical features if available
        if self.categorical_features_indices_:
            pred_pool = Pool(X, cat_features=self.categorical_features_indices_)
            return self.model_.predict(pred_pool)
        else:
            return self.model_.predict(X)
    
    def predict_with_uncertainty(self, X, prediction_type='RMSEWithUncertainty'):
        """
        Make predictions with uncertainty estimates using CatBoost's built-in uncertainty
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        prediction_type : str, optional
            Type of uncertainty prediction ('RMSEWithUncertainty' or 'Uncertainty')
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions and uncertainty estimates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        try:
            # Create prediction pool
            if self.categorical_features_indices_:
                pred_pool = Pool(X, cat_features=self.categorical_features_indices_)
            else:
                pred_pool = Pool(X)
            
            # Get predictions with uncertainty
            if prediction_type == 'RMSEWithUncertainty':
                predictions = self.model_.predict(pred_pool, prediction_type='RMSEWithUncertainty')
                pred_values = predictions[:, 0]
                uncertainty_values = predictions[:, 1]
            else:
                pred_values = self.model_.predict(pred_pool)
                # Fallback uncertainty using virtual ensembles
                uncertainty_values = self._estimate_uncertainty_virtual_ensembles(pred_pool)
            
            # Calculate confidence intervals
            confidence_95_lower = pred_values - 1.96 * uncertainty_values
            confidence_95_upper = pred_values + 1.96 * uncertainty_values
            
            # Uncertainty score (relative uncertainty)
            uncertainty_score = uncertainty_values / (np.abs(pred_values) + 1e-10)
            
            return {
                'predictions': pred_values,
                'uncertainty': uncertainty_values,
                'uncertainty_score': uncertainty_score,
                'confidence_95_lower': confidence_95_lower,
                'confidence_95_upper': confidence_95_upper,
                'prediction_interval_width': confidence_95_upper - confidence_95_lower,
                'prediction_stability': 1.0 / (1.0 + uncertainty_score),
                'uncertainty_type': prediction_type
            }
            
        except Exception as e:
            # Fallback to standard predictions if uncertainty is not available
            pred_values = self.predict(X)
            fallback_uncertainty = np.std(pred_values) * np.ones(len(pred_values)) * 0.1
            
            return {
                'predictions': pred_values,
                'uncertainty': fallback_uncertainty,
                'uncertainty_score': fallback_uncertainty / (np.abs(pred_values) + 1e-10),
                'confidence_95_lower': pred_values - 1.96 * fallback_uncertainty,
                'confidence_95_upper': pred_values + 1.96 * fallback_uncertainty,
                'prediction_interval_width': 3.92 * fallback_uncertainty,
                'prediction_stability': 0.9 * np.ones(len(pred_values)),
                'uncertainty_type': 'Fallback',
                'note': f'Uncertainty estimation failed: {str(e)}'
            }
    
    def _estimate_uncertainty_virtual_ensembles(self, pred_pool):
        """Estimate uncertainty using virtual ensembles approach"""
        try:
            # Use different tree ranges to create virtual ensemble
            n_iterations = min(self.model_.tree_count_, 100)
            step = max(1, n_iterations // 10)
            
            predictions = []
            for ntree_end in range(step, n_iterations + 1, step):
                pred = self.model_.predict(pred_pool, ntree_end=ntree_end)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            uncertainty = np.std(predictions, axis=0)
            
            return uncertainty
        except:
            # Final fallback
            return np.ones(pred_pool.num_row()) * 0.1
    
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
                elif X[col].dtype in ['int64', 'int32'] and X[col].nunique() <= 50:
                    # Low cardinality integer features might be categorical
                    categorical_cols.append(col)
            return categorical_cols
        
        return []
    
    def _get_categorical_indices(self, X, categorical_features):
        """Convert categorical feature names to indices"""
        if not categorical_features:
            return []
        
        if hasattr(X, 'columns'):
            # DataFrame case
            indices = []
            for cat_feature in categorical_features:
                if isinstance(cat_feature, str):
                    try:
                        idx = list(X.columns).index(cat_feature)
                        indices.append(idx)
                    except ValueError:
                        continue
                elif isinstance(cat_feature, int):
                    indices.append(cat_feature)
            return indices
        else:
            # Array case - assume categorical_features are already indices
            return [f for f in categorical_features if isinstance(f, int)]
    
    def _analyze_feature_importance(self):
        """Analyze feature importance with CatBoost specific metrics"""
        if not self.compute_feature_importance:
            return
        
        try:
            # Get different types of importance from CatBoost
            importance_types = ['PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance']
            importance_scores = {}
            
            for imp_type in importance_types:
                try:
                    if imp_type == 'FeatureImportance':
                        scores = self.model_.feature_importances_
                    else:
                        scores = self.model_.get_feature_importance(type=imp_type)
                    importance_scores[imp_type] = scores
                except:
                    importance_scores[imp_type] = np.zeros(len(self.feature_names_))
            
            # Permutation importance if requested
            permutation_imp = None
            permutation_imp_std = None
            if self.compute_permutation_importance:
                try:
                    # Create pool for permutation importance
                    if self.categorical_features_indices_:
                        eval_pool = Pool(self.X_original_, self.y_original_, 
                                       cat_features=self.categorical_features_indices_)
                    else:
                        eval_pool = Pool(self.X_original_, self.y_original_)
                    
                    perm_imp_result = permutation_importance(
                        self.model_, eval_pool, self.y_original_,
                        n_repeats=10, random_state=self.random_seed,
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
            
            # Top features analysis with categorical information
            top_features = []
            main_ranking = rankings.get('PredictionValuesChange', rankings['FeatureImportance'])
            
            for i in range(min(15, len(self.feature_names_))):
                feature_idx = main_ranking[i]
                feature_info = {
                    'name': self.feature_names_[feature_idx],
                    'prediction_change_importance': importance_scores.get('PredictionValuesChange', [0])[feature_idx],
                    'loss_change_importance': importance_scores.get('LossFunctionChange', [0])[feature_idx],
                    'feature_importance': importance_scores['FeatureImportance'][feature_idx],
                    'rank': i + 1,
                    'is_categorical': feature_idx in (self.categorical_features_indices_ or [])
                }
                
                if permutation_imp is not None:
                    feature_info['permutation_importance'] = permutation_imp[feature_idx]
                    feature_info['permutation_std'] = permutation_imp_std[feature_idx]
                
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
                'categorical_indices': self.categorical_features_indices_,
                'interpretation': {
                    'PredictionValuesChange': 'Change in prediction when feature is excluded',
                    'LossFunctionChange': 'Change in loss function when feature is excluded',
                    'FeatureImportance': 'Sum of split gains using the feature'
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
    
    def _analyze_catboost_specifics(self):
        """Analyze CatBoost specific features and optimizations"""
        if not self.catboost_analysis:
            return
        
        try:
            analysis = {}
            
            # Model configuration analysis
            analysis['model_config'] = {
                'algorithm': 'CatBoost (Categorical Boosting)',
                'tree_structure': 'Oblivious (symmetric) decision trees',
                'categorical_handling': 'Native with advanced encoding',
                'iterations': self.model_.tree_count_,
                'depth': self.depth,
                'learning_rate': self.model_.learning_rate_ if hasattr(self.model_, 'learning_rate_') else self.learning_rate,
                'loss_function': self.loss_function,
                'leaf_estimation_method': self.leaf_estimation_method
            }
            
            # CatBoost innovations
            analysis['catboost_innovations'] = {
                'ordered_boosting': 'Reduces overfitting through ordered target statistics',
                'symmetric_trees': 'Oblivious decision trees for faster inference',
                'categorical_encoding': 'Advanced categorical feature encoding without preprocessing',
                'target_leakage_protection': 'Built-in protection against target leakage',
                'optimal_splits': 'Optimal split finding for categorical features',
                'gpu_acceleration': 'Full GPU training support',
                'automatic_regularization': 'Self-tuning regularization parameters'
            }
            
            # Categorical feature advantages
            analysis['categorical_advantages'] = {
                'native_handling': len(self.categorical_features_indices_ or []) > 0,
                'no_preprocessing_needed': True,
                'optimal_encoding': 'Learns optimal categorical encodings during training',
                'high_cardinality_support': 'Handles high-cardinality categoricals efficiently',
                'missing_category_handling': 'Automatic handling of unseen categories',
                'target_statistics': 'Uses target statistics for categorical encoding',
                'combinations': 'Automatically finds useful categorical combinations'
            }
            
            # Performance characteristics
            analysis['performance'] = {
                'training_speed': 'Moderate (focus on accuracy over speed)',
                'inference_speed': 'Very Fast (symmetric trees)',
                'memory_usage': 'Moderate to Low',
                'overfitting_resistance': 'Excellent (ordered boosting)',
                'categorical_efficiency': 'Superior (native handling)',
                'gpu_utilization': 'Excellent',
                'default_params_quality': 'Excellent (minimal tuning needed)'
            }
            
            # Tree structure analysis
            analysis['tree_structure'] = {
                'type': 'Oblivious (Symmetric) Decision Trees',
                'depth': self.depth,
                'symmetry': 'All trees have the same structure at each level',
                'advantages': [
                    'Faster inference than asymmetric trees',
                    'Better cache locality',
                    'Reduced model size',
                    'More robust to overfitting'
                ],
                'leaf_count_per_tree': 2 ** self.depth,
                'total_leaves': (2 ** self.depth) * self.model_.tree_count_
            }
            
            # Regularization mechanisms
            analysis['regularization'] = {
                'l2_leaf_regularization': self.l2_leaf_reg,
                'model_size_regularization': self.model_size_reg,
                'random_strength': self.random_strength,
                'bagging_temperature': self.bagging_temperature,
                'ordered_boosting': 'Primary overfitting prevention mechanism',
                'bootstrap_type': self.bootstrap_type,
                'rsm': f'Random subspace method with {self.rsm} feature fraction'
            }
            
            self.catboost_analysis_ = analysis
            
        except Exception as e:
            self.catboost_analysis_ = {
                'error': f'Could not analyze CatBoost specifics: {str(e)}'
            }
    
    def _analyze_categorical_features(self):
        """Comprehensive analysis of categorical feature handling"""
        if not self.categorical_analysis or not self.categorical_features_indices_:
            return
        
        try:
            analysis = {}
            
            # Basic categorical feature information
            analysis['categorical_summary'] = {
                'total_categorical_features': len(self.categorical_features_indices_),
                'total_features': len(self.feature_names_),
                'categorical_ratio': len(self.categorical_features_indices_) / len(self.feature_names_),
                'categorical_feature_names': [self.feature_names_[i] for i in self.categorical_features_indices_]
            }
            
            # Categorical feature statistics (if original data is DataFrame)
            if hasattr(self, 'X_original_') and len(self.categorical_features_indices_) > 0:
                cat_stats = {}
                for idx in self.categorical_features_indices_:
                    if hasattr(self.X_original_, 'iloc'):
                        # DataFrame case
                        feature_data = self.X_original_.iloc[:, idx]
                    else:
                        # Array case
                        feature_data = self.X_original_[:, idx]
                    
                    cat_stats[self.feature_names_[idx]] = {
                        'unique_values': len(np.unique(feature_data)),
                        'most_frequent': self._get_most_frequent_category(feature_data),
                        'cardinality_level': self._assess_cardinality_level(feature_data),
                        'missing_values': np.sum(pd.isna(feature_data)) if hasattr(pd, 'isna') else 0
                    }
                
                analysis['categorical_statistics'] = cat_stats
            
            # CatBoost categorical configuration
            analysis['catboost_categorical_config'] = {
                'one_hot_max_size': self.one_hot_max_size,
                'max_ctr_complexity': self.max_ctr_complexity,
                'border_count': self.border_count,
                'feature_border_type': self.feature_border_type,
                'encoding_method': 'Target statistics with ordered boosting protection'
            }
            
            # Categorical feature advantages in CatBoost
            analysis['categorical_advantages'] = {
                'no_preprocessing': 'No need for manual encoding (one-hot, label, etc.)',
                'optimal_splits': 'Learns optimal categorical splits during training',
                'target_leakage_protection': 'Ordered boosting prevents target leakage',
                'high_cardinality': 'Efficiently handles high-cardinality categories',
                'combinations': 'Automatically discovers useful categorical combinations',
                'missing_handling': 'Automatic handling of missing categorical values',
                'new_categories': 'Graceful handling of unseen categories in prediction'
            }
            
            # Performance impact
            analysis['performance_impact'] = {
                'training_time': 'Minimal overhead for categorical processing',
                'memory_usage': 'Efficient categorical representation',
                'accuracy_improvement': 'Often significant improvement over manual encoding',
                'inference_speed': 'No preprocessing needed during prediction'
            }
            
            self.categorical_analysis_ = analysis
            
        except Exception as e:
            self.categorical_analysis_ = {
                'error': f'Could not analyze categorical features: {str(e)}'
            }
    
    def _get_most_frequent_category(self, feature_data):
        """Get the most frequent category in a feature"""
        try:
            unique, counts = np.unique(feature_data, return_counts=True)
            most_frequent_idx = np.argmax(counts)
            return {
                'value': unique[most_frequent_idx],
                'frequency': counts[most_frequent_idx],
                'percentage': counts[most_frequent_idx] / len(feature_data) * 100
            }
        except:
            return {'error': 'Could not determine most frequent category'}
    
    def _assess_cardinality_level(self, feature_data):
        """Assess the cardinality level of a categorical feature"""
        try:
            unique_count = len(np.unique(feature_data))
            total_count = len(feature_data)
            ratio = unique_count / total_count
            
            if unique_count <= 2:
                return f"Binary ({unique_count} categories)"
            elif unique_count <= 10:
                return f"Low cardinality ({unique_count} categories)"
            elif unique_count <= 50:
                return f"Medium cardinality ({unique_count} categories)"
            elif ratio < 0.5:
                return f"High cardinality ({unique_count} categories, {ratio:.1%} unique)"
            else:
                return f"Very high cardinality ({unique_count} categories, {ratio:.1%} unique)"
        except:
            return "Unknown cardinality"
    
    def _analyze_overfitting_behavior(self):
        """Analyze overfitting behavior and prevention mechanisms"""
        if not self.overfitting_analysis:
            return
        
        try:
            analysis = {}
            
            # Overfitting detection results
            if hasattr(self.model_, 'best_iteration_') and self.model_.best_iteration_ is not None:
                analysis['overfitting_detection'] = {
                    'best_iteration': self.model_.best_iteration_,
                    'total_iterations': self.model_.tree_count_,
                    'stopped_early': self.model_.best_iteration_ < self.iterations,
                    'iterations_saved': max(0, self.iterations - self.model_.best_iteration_),
                    'efficiency_gain': max(0, self.iterations - self.model_.best_iteration_) / self.iterations
                }
            else:
                analysis['overfitting_detection'] = {
                    'best_iteration': self.model_.tree_count_,
                    'total_iterations': self.model_.tree_count_,
                    'stopped_early': False,
                    'iterations_saved': 0,
                    'efficiency_gain': 0.0
                }
            
            # CatBoost's overfitting prevention mechanisms
            analysis['overfitting_prevention'] = {
                'ordered_boosting': 'Primary mechanism - prevents target leakage',
                'symmetric_trees': 'Reduces model complexity and overfitting risk',
                'regularization': {
                    'l2_leaf_reg': self.l2_leaf_reg,
                    'model_size_reg': self.model_size_reg,
                    'random_strength': self.random_strength
                },
                'sampling': {
                    'bootstrap_type': self.bootstrap_type,
                    'bagging_temperature': self.bagging_temperature,
                    'rsm': self.rsm
                },
                'automatic_features': 'Built-in overfitting detection and prevention'
            }
            
            # Risk assessment
            analysis['overfitting_risk_assessment'] = {
                'overall_risk': self._assess_overfitting_risk(),
                'protective_factors': self._identify_protective_factors(),
                'risk_factors': self._identify_risk_factors(),
                'recommendations': self._get_overfitting_recommendations()
            }
            
            self.overfitting_analysis_ = analysis
            
        except Exception as e:
            self.overfitting_analysis_ = {
                'error': f'Could not analyze overfitting behavior: {str(e)}'
            }
    
    def _assess_overfitting_risk(self):
        """Assess overall overfitting risk"""
        risk_score = 0
        
        # High iterations without early stopping
        if self.iterations > 1000 and self.early_stopping_rounds is None:
            risk_score += 2
        
        # Deep trees
        if self.depth > 8:
            risk_score += 2
        elif self.depth > 6:
            risk_score += 1
        
        # Low regularization
        if self.l2_leaf_reg < 1.0:
            risk_score += 1
        
        # Small dataset
        if hasattr(self, 'X_original_') and len(self.X_original_) < 1000:
            risk_score += 2
        
        # High feature to sample ratio
        if hasattr(self, 'X_original_'):
            n_samples, n_features = self.X_original_.shape
            if n_features / n_samples > 0.1:
                risk_score += 1
        
        # Protective factors reduce risk
        if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
            risk_score -= 1
        
        if self.model_size_reg > 0.5:
            risk_score -= 1
        
        if self.early_stopping_rounds is not None:
            risk_score -= 2
        
        risk_score = max(0, risk_score)
        
        if risk_score >= 5:
            return "High - Consider stronger regularization"
        elif risk_score >= 3:
            return "Moderate - Monitor validation performance"
        elif risk_score >= 1:
            return "Low - Good configuration"
        else:
            return "Very Low - Excellent overfitting protection"
    
    def _identify_protective_factors(self):
        """Identify factors that protect against overfitting"""
        factors = []
        
        factors.append("Ordered boosting (CatBoost's primary protection)")
        factors.append("Symmetric trees reduce complexity")
        
        if self.early_stopping_rounds is not None:
            factors.append(f"Early stopping with {self.early_stopping_rounds} rounds")
        
        if self.l2_leaf_reg >= 3.0:
            factors.append(f"Strong L2 regularization ({self.l2_leaf_reg})")
        
        if self.model_size_reg > 0.5:
            factors.append(f"Model size regularization ({self.model_size_reg})")
        
        if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
            factors.append(f"{self.bootstrap_type} bootstrap with sampling")
        
        if self.random_strength > 1.0:
            factors.append(f"Random strength regularization ({self.random_strength})")
        
        if self.rsm < 1.0:
            factors.append(f"Random subspace method ({self.rsm})")
        
        return factors
    
    def _identify_risk_factors(self):
        """Identify factors that increase overfitting risk"""
        factors = []
        
        if self.iterations > 1500:
            factors.append(f"High number of iterations ({self.iterations})")
        
        if self.depth > 8:
            factors.append(f"Deep trees (depth {self.depth})")
        
        if self.l2_leaf_reg < 1.0:
            factors.append(f"Low L2 regularization ({self.l2_leaf_reg})")
        
        if self.early_stopping_rounds is None:
            factors.append("No early stopping configured")
        
        if self.bootstrap_type == 'No':
            factors.append("No bootstrap sampling")
        
        if hasattr(self, 'X_original_'):
            n_samples, n_features = self.X_original_.shape
            if n_samples < 500:
                factors.append(f"Small dataset ({n_samples} samples)")
            
            if n_features / n_samples > 0.2:
                factors.append(f"High feature-to-sample ratio ({n_features}/{n_samples})")
        
        return factors if factors else ["No significant risk factors identified"]
    
    def _get_overfitting_recommendations(self):
        """Get recommendations to prevent overfitting"""
        recommendations = []
        
        risk_factors = self._identify_risk_factors()
        
        if "High number of iterations" in str(risk_factors):
            recommendations.append("Consider reducing iterations or enabling early stopping")
        
        if "Deep trees" in str(risk_factors):
            recommendations.append("Consider reducing tree depth to 6 or less")
        
        if "Low L2 regularization" in str(risk_factors):
            recommendations.append("Increase l2_leaf_reg to 3.0 or higher")
        
        if "No early stopping" in str(risk_factors):
            recommendations.append("Enable early stopping with validation set")
        
        if "Small dataset" in str(risk_factors):
            recommendations.append("Use stronger regularization and fewer iterations")
        
        if not recommendations:
            recommendations.append("Current configuration provides good overfitting protection")
        
        # Always recommend CatBoost best practices
        recommendations.append("Leverage CatBoost's ordered boosting for natural overfitting protection")
        
        return recommendations
    
    def _analyze_convergence(self):
        """Analyze model convergence characteristics"""
        if not self.convergence_analysis:
            return
        
        try:
            analysis = {}
            
            # Training convergence
            analysis['training_convergence'] = {
                'final_iterations': self.model_.tree_count_,
                'learning_rate': self.model_.learning_rate_ if hasattr(self.model_, 'learning_rate_') else self.learning_rate,
                'convergence_assessment': self._assess_convergence_quality()
            }
            
            # Learning rate analysis
            analysis['learning_rate_analysis'] = {
                'current_rate': self.learning_rate,
                'automatic_rate': self.learning_rate is None,
                'rate_category': self._categorize_learning_rate(),
                'convergence_characteristics': self._assess_lr_convergence_impact()
            }
            
            # Iteration efficiency
            if hasattr(self.model_, 'best_iteration_') and self.model_.best_iteration_ is not None:
                analysis['iteration_efficiency'] = {
                    'optimal_iterations': self.model_.best_iteration_,
                    'used_iterations': self.model_.tree_count_,
                    'efficiency_ratio': self.model_.best_iteration_ / self.iterations,
                    'convergence_speed': self._assess_convergence_speed()
                }
            
            # Tree depth impact on convergence
            analysis['depth_impact'] = {
                'tree_depth': self.depth,
                'complexity_per_tree': 2 ** self.depth,
                'convergence_characteristics': self._assess_depth_convergence_impact()
            }
            
            self.convergence_analysis_ = analysis
            
        except Exception as e:
            self.convergence_analysis_ = {
                'error': f'Could not analyze convergence: {str(e)}'
            }
    
    def _assess_convergence_quality(self):
        """Assess the quality of model convergence"""
        if hasattr(self.model_, 'best_iteration_') and self.model_.best_iteration_ is not None:
            efficiency = self.model_.best_iteration_ / self.iterations
            
            if efficiency < 0.3:
                return "Very fast convergence - may benefit from higher learning rate"
            elif efficiency < 0.6:
                return "Good convergence speed"
            elif efficiency < 0.9:
                return "Slow convergence - consider higher learning rate"
            else:
                return "Very slow convergence - needs optimization"
        else:
            return "No early stopping - convergence assessment limited"
    
    def _categorize_learning_rate(self):
        """Categorize the learning rate"""
        if self.learning_rate is None:
            return "Automatic (CatBoost optimized)"
        elif self.learning_rate >= 0.3:
            return "High - Fast learning but may overshoot"
        elif self.learning_rate >= 0.1:
            return "Standard - Good balance"
        elif self.learning_rate >= 0.03:
            return "Conservative - Stable learning"
        else:
            return "Very conservative - Very stable but slow"
    
    def _assess_lr_convergence_impact(self):
        """Assess learning rate impact on convergence"""
        if self.learning_rate is None:
            return "Automatic rate optimization for optimal convergence"
        elif self.learning_rate >= 0.3:
            return "Fast convergence but higher overfitting risk"
        elif self.learning_rate >= 0.1:
            return "Balanced convergence speed and stability"
        else:
            return "Slow but stable convergence"
    
    def _assess_convergence_speed(self):
        """Assess convergence speed based on iterations used"""
        if hasattr(self.model_, 'best_iteration_'):
            ratio = self.model_.best_iteration_ / self.iterations
            
            if ratio < 0.2:
                return "Very Fast"
            elif ratio < 0.5:
                return "Fast"
            elif ratio < 0.8:
                return "Moderate"
            else:
                return "Slow"
        else:
            return "Unknown"
    
    def _assess_depth_convergence_impact(self):
        """Assess tree depth impact on convergence"""
        if self.depth <= 4:
            return "Shallow trees - fast convergence, may need more iterations"
        elif self.depth <= 6:
            return "Medium depth - balanced convergence characteristics"
        elif self.depth <= 8:
            return "Deep trees - slower convergence but better pattern capture"
        else:
            return "Very deep trees - slow convergence, high capacity"
    
    def _analyze_regularization_effects(self):
        """Analyze the effects of different regularization techniques"""
        if not self.regularization_analysis:
            return
        
        try:
            analysis = {}
            
            # L2 leaf regularization
            analysis['l2_regularization'] = {
                'value': self.l2_leaf_reg,
                'effect': 'Penalizes large leaf values to prevent overfitting',
                'strength_assessment': self._assess_l2_strength()
            }
            
            # Model size regularization
            analysis['model_size_regularization'] = {
                'value': self.model_size_reg,
                'effect': 'Controls model complexity',
                'strength_assessment': self._assess_model_size_reg_strength()
            }
            
            # Random strength
            analysis['random_strength'] = {
                'value': self.random_strength,
                'effect': 'Adds randomness to tree structure',
                'strength_assessment': self._assess_random_strength()
            }
            
            # Bootstrap and sampling
            analysis['sampling_regularization'] = {
                'bootstrap_type': self.bootstrap_type,
                'bagging_temperature': self.bagging_temperature,
                'rsm': self.rsm,
                'subsample': self.subsample if self.bootstrap_type in ['Bayesian', 'Bernoulli'] else 1.0,
                'effect': 'Reduces overfitting through data sampling',
                'strength_assessment': self._assess_sampling_strength()
            }
            
            # Overall regularization assessment
            analysis['overall_assessment'] = {
                'total_regularization_strength': self._calculate_total_regularization_strength(),
                'regularization_balance': self._assess_regularization_balance(),
                'overfitting_protection': self._assess_regularization_effectiveness(),
                'recommendations': self._get_regularization_recommendations()
            }
            
            self.regularization_analysis_ = analysis
            
        except Exception as e:
            self.regularization_analysis_ = {
                'error': f'Could not analyze regularization effects: {str(e)}'
            }
    
    def _assess_l2_strength(self):
        """Assess L2 regularization strength"""
        if self.l2_leaf_reg >= 10.0:
            return "Very Strong - May cause underfitting"
        elif self.l2_leaf_reg >= 5.0:
            return "Strong - Good overfitting protection"
        elif self.l2_leaf_reg >= 1.0:
            return "Moderate - Balanced regularization"
        else:
            return "Weak - Minimal regularization effect"
    
    def _assess_model_size_reg_strength(self):
        """Assess model size regularization strength"""
        if self.model_size_reg >= 1.0:
            return "Strong - Significant complexity penalty"
        elif self.model_size_reg >= 0.5:
            return "Moderate - Balanced complexity control"
        elif self.model_size_reg >= 0.1:
            return "Weak - Minimal complexity penalty"
        else:
            return "Very Weak - Almost no complexity control"
    
    def _assess_random_strength(self):
        """Assess random strength regularization"""
        if self.random_strength >= 2.0:
            return "High - Strong randomization effect"
        elif self.random_strength >= 1.0:
            return "Standard - Moderate randomization"
        elif self.random_strength >= 0.5:
            return "Low - Minimal randomization"
        else:
            return "Very Low - Almost deterministic"
    
    def _assess_sampling_strength(self):
        """Assess sampling-based regularization strength"""
        strength_score = 0
        
        if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
            strength_score += 2
            if self.subsample < 0.8:
                strength_score += 2
            elif self.subsample < 0.9:
                strength_score += 1
        
        if self.rsm < 0.8:
            strength_score += 2
        elif self.rsm < 0.9:
            strength_score += 1
        
        if self.bagging_temperature > 1.0:
            strength_score += 1
        
        if strength_score >= 5:
            return "Very Strong - High variance reduction through sampling"
        elif strength_score >= 3:
            return "Strong - Good sampling regularization"
        elif strength_score >= 1:
            return "Moderate - Some sampling regularization"
        else:
            return "Weak - Minimal sampling regularization"
    
    def _calculate_total_regularization_strength(self):
        """Calculate overall regularization strength score"""
        score = 0
        
        # L2 contribution
        score += min(self.l2_leaf_reg / 2, 5)
        
        # Model size contribution
        score += self.model_size_reg * 3
        
        # Random strength contribution
        score += self.random_strength
        
        # Sampling contribution
        if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
            score += 2
            score += (1 - self.subsample) * 3
        
        score += (1 - self.rsm) * 2
        
        # Bagging temperature
        if self.bagging_temperature > 1.0:
            score += (self.bagging_temperature - 1.0)
        
        return min(score, 15)
    
    def _assess_regularization_balance(self):
        """Assess balance between different regularization types"""
        l2_strength = min(self.l2_leaf_reg / 5, 1.0)
        model_size_strength = self.model_size_reg
        sampling_strength = 0.5 if self.bootstrap_type != 'No' else 0
        
        strengths = [l2_strength, model_size_strength, sampling_strength]
        max_strength = max(strengths)
        min_strength = min(strengths)
        
        if max_strength - min_strength < 0.3:
            return "Well-balanced across all regularization types"
        elif l2_strength > max(model_size_strength, sampling_strength) + 0.3:
            return "Heavily relies on L2 regularization"
        elif model_size_strength > max(l2_strength, sampling_strength) + 0.3:
            return "Heavily relies on model size regularization"
        else:
            return "Balanced with emphasis on sampling regularization"
    
    def _assess_regularization_effectiveness(self):
        """Assess overall regularization effectiveness"""
        total_strength = self._calculate_total_regularization_strength()
        
        if total_strength >= 12:
            return "Excellent - Very strong overfitting protection"
        elif total_strength >= 8:
            return "Good - Adequate overfitting protection"
        elif total_strength >= 4:
            return "Moderate - Basic overfitting protection"
        else:
            return "Weak - May need stronger regularization"
    
    def _get_regularization_recommendations(self):
        """Get regularization recommendations"""
        recommendations = []
        total_strength = self._calculate_total_regularization_strength()
        
        if total_strength < 4:
            recommendations.append("Consider increasing regularization strength")
            if self.l2_leaf_reg < 3.0:
                recommendations.append("Try l2_leaf_reg >= 3.0")
            if self.model_size_reg < 0.5:
                recommendations.append("Consider model_size_reg >= 0.5")
        
        if self.bootstrap_type == 'No':
            recommendations.append("Consider Bayesian or Bernoulli bootstrap for better regularization")
        
        if self.early_stopping_rounds is None:
            recommendations.append("Enable early stopping for additional overfitting protection")
        
        if total_strength >= 12:
            recommendations.append("Regularization may be too strong - consider reducing if underfitting")
        
        if not recommendations:
            recommendations.append("Regularization appears well-configured")
        
        return recommendations
    
    def _analyze_categorical_encoding(self):
        """Analyze CatBoost's categorical encoding mechanisms"""
        if not self.categorical_encoding_analysis or not self.categorical_features_indices_:
            return
        
        try:
            analysis = {}
            
            # Encoding configuration
            analysis['encoding_config'] = {
                'one_hot_max_size': self.one_hot_max_size,
                'max_ctr_complexity': self.max_ctr_complexity,
                'border_count': self.border_count,
                'feature_border_type': self.feature_border_type
            }
            
            # CTR (Counter) features analysis
            analysis['ctr_features'] = {
                'description': 'Categorical-to-Real transformations using target statistics',
                'max_complexity': self.max_ctr_complexity,
                'automatic_generation': True,
                'target_leakage_protection': 'Ordered boosting prevents target leakage',
                'combinations': 'Automatically generates useful categorical combinations'
            }
            
            # Encoding strategies
            analysis['encoding_strategies'] = {
                'small_categories': f'One-hot encoding for ≤{self.one_hot_max_size} unique values',
                'large_categories': 'Target statistics with ordered boosting',
                'missing_values': 'Automatic handling as separate category',
                'new_categories': 'Safe handling of unseen categories during prediction',
                'optimal_splits': 'Learns optimal categorical splits during training'
            }
            
            # Advantages over manual encoding
            analysis['advantages_over_manual'] = {
                'no_preprocessing': 'No need for manual one-hot or label encoding',
                'optimal_encoding': 'Learns task-specific optimal encodings',
                'memory_efficiency': 'No explosion of feature space from one-hot encoding',
                'robustness': 'Handles high cardinality and missing values automatically',
                'target_aware': 'Uses target information for better encoding',
                'overfitting_safe': 'Ordered boosting prevents target leakage'
            }
            
            # Categorical feature impact on model
            if self.feature_importance_analysis_:
                top_features = self.feature_importance_analysis_.get('top_features', [])
                categorical_in_top = [f for f in top_features if f.get('is_categorical', False)]
                
                analysis['categorical_impact'] = {
                    'categorical_in_top_15': len(categorical_in_top),
                    'top_categorical_features': categorical_in_top[:5],
                    'categorical_importance_ratio': len(categorical_in_top) / min(15, len(top_features)) if top_features else 0,
                    'encoding_effectiveness': 'High' if len(categorical_in_top) > 0 else 'Moderate'
                }
            
            self.categorical_encoding_analysis_ = analysis
            
        except Exception as e:
            self.categorical_encoding_analysis_ = {
                'error': f'Could not analyze categorical encoding: {str(e)}'
            }
    
    def _analyze_cross_validation(self):
        """Perform cross-validation analysis"""
        if not self.cross_validation_analysis:
            return
        
        try:
            # Create a fresh model for CV (simplified version)
            cv_params = {
# Continue from line 1501 where the code broke off
                'learning_rate': self.learning_rate,
                'depth': min(6, self.depth),  # Reduced for CV
                'l2_leaf_reg': self.l2_leaf_reg,
                'verbose': False,
                'random_seed': self.random_seed,
                'thread_count': self.thread_count
            }
            
            cv_model = CatBoostRegressor(**cv_params)
            
            # Perform cross-validation
            if self.categorical_features_indices_:
                # Create Pool for cross-validation with categorical features
                cv_pool = Pool(self.X_original_, self.y_original_, 
                             cat_features=self.categorical_features_indices_)
                cv_scores = cross_val_score(
                    cv_model, cv_pool, self.y_original_,
                    cv=self.cv_folds, scoring='neg_mean_squared_error',
                    n_jobs=1  # CatBoost doesn't support parallel CV well
                )
            else:
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
                if self.categorical_features_indices_:
                    r2_scores = cross_val_score(
                        cv_model, cv_pool, self.y_original_,
                        cv=self.cv_folds, scoring='r2', n_jobs=1
                    )
                else:
                    r2_scores = cross_val_score(
                        cv_model, self.X_original_, self.y_original_,
                        cv=self.cv_folds, scoring='r2', n_jobs=-1
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
        """Analyze CatBoost tree structure and characteristics"""
        if not self.tree_analysis:
            return
        
        try:
            analysis = {}
            
            # Basic tree statistics
            analysis['ensemble_stats'] = {
                'total_trees': self.model_.tree_count_,
                'tree_depth': self.depth,
                'tree_type': 'Oblivious (Symmetric) Decision Trees',
                'leaves_per_tree': 2 ** self.depth,
                'total_leaves': (2 ** self.depth) * self.model_.tree_count_
            }
            
            # Oblivious tree advantages
            analysis['oblivious_trees'] = {
                'structure': 'All trees use the same features at each level',
                'advantages': [
                    'Faster inference than asymmetric trees',
                    'Better cache locality and memory efficiency',
                    'More robust to overfitting',
                    'Easier to interpret tree structure',
                    'Better parallelization'
                ],
                'trade_offs': [
                    'May need more trees than asymmetric approaches',
                    'Less flexible than leaf-wise growth'
                ]
            }
            
            # Tree complexity analysis
            analysis['complexity_analysis'] = {
                'depth_setting': self.depth,
                'complexity_category': self._categorize_tree_complexity(),
                'overfitting_risk': self._assess_tree_overfitting_risk(),
                'memory_usage': f"~{(2 ** self.depth) * self.model_.tree_count_ * 8} bytes for tree structure"
            }
            
            self.tree_analysis_ = analysis
            
        except Exception as e:
            self.tree_analysis_ = {
                'error': f'Could not analyze tree structure: {str(e)}'
            }
            
    def _categorize_tree_complexity(self):
        """Categorize tree complexity based on depth"""
        if self.depth <= 4:
            return "Low complexity - Fast training and inference"
        elif self.depth <= 6:
            return "Medium complexity - Good balance"
        elif self.depth <= 8:
            return "High complexity - Can capture complex patterns"
        else:
            return "Very high complexity - Risk of overfitting"
        
    def _assess_tree_overfitting_risk(self):
        """Assess overfitting risk from tree structure"""
        risk_score = 0
        
        if self.depth > 8:
            risk_score += 3
        elif self.depth > 6:
            risk_score += 1
        
        # CatBoost's ordered boosting provides protection
        risk_score -= 2
        
        # Regularization effects
        if self.l2_leaf_reg >= 3.0:
            risk_score -= 1
        
        risk_score = max(0, risk_score)
        
        if risk_score >= 3:
            return "High - Consider reducing depth or increasing regularization"
        elif risk_score >= 1:
            return "Moderate - Monitor validation performance"
        else:
            return "Low - Good protection from ordered boosting"
        
    def _analyze_prediction_uncertainty(self):
        """Analyze prediction uncertainty capabilities"""
        if not self.prediction_uncertainty_analysis:
            return
        
        try:
            analysis = {}
            
            # Uncertainty estimation capabilities
            analysis['uncertainty_capabilities'] = {
                'native_uncertainty': 'CatBoost supports RMSEWithUncertainty prediction type',
                'virtual_ensembles': 'Can estimate uncertainty using tree subsets',
                'confidence_intervals': 'Provides prediction confidence intervals',
                'uncertainty_types': ['RMSEWithUncertainty', 'Virtual Ensemble', 'Fallback']
            }
            
            # Test uncertainty estimation on training data sample
            try:
                sample_size = min(100, len(self.X_original_))
                sample_indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
                X_sample = self.X_original_[sample_indices]
                
                uncertainty_result = self.predict_with_uncertainty(X_sample)
                
                analysis['uncertainty_analysis'] = {
                    'mean_uncertainty': np.mean(uncertainty_result['uncertainty']),
                    'std_uncertainty': np.std(uncertainty_result['uncertainty']),
                    'mean_uncertainty_score': np.mean(uncertainty_result['uncertainty_score']),
                    'mean_prediction_stability': np.mean(uncertainty_result['prediction_stability']),
                    'uncertainty_distribution': {
                        'min': np.min(uncertainty_result['uncertainty']),
                        'max': np.max(uncertainty_result['uncertainty']),
                        'median': np.median(uncertainty_result['uncertainty']),
                        'q75': np.percentile(uncertainty_result['uncertainty'], 75),
                        'q25': np.percentile(uncertainty_result['uncertainty'], 25)
                    }
                }
                
                # Assess uncertainty quality
                analysis['uncertainty_quality'] = self._assess_uncertainty_quality(uncertainty_result)
                
            except Exception as e:
                analysis['uncertainty_analysis'] = {
                    'error': f'Could not analyze uncertainty on sample: {str(e)}'
                }
            
            # Uncertainty recommendations
            analysis['recommendations'] = {
                'use_cases': [
                    'Risk assessment in financial modeling',
                    'Quality control in manufacturing',
                    'Medical diagnosis confidence',
                    'Scientific measurement reliability'
                ],
                'best_practices': [
                    'Use RMSEWithUncertainty for most accurate estimates',
                    'Consider virtual ensembles for comparative analysis',
                    'Monitor prediction stability scores',
                    'Use confidence intervals for decision making'
                ]
            }
            
            self.prediction_uncertainty_analysis_ = analysis
            
        except Exception as e:
            self.prediction_uncertainty_analysis_ = {
                'error': f'Could not analyze prediction uncertainty: {str(e)}'
            }
    def _assess_uncertainty_quality(self, uncertainty_result):
        """Assess the quality of uncertainty estimates"""
        try:
            uncertainty_score = uncertainty_result['uncertainty_score']
            stability = uncertainty_result['prediction_stability']
            
            # Calculate quality metrics
            mean_uncertainty_score = np.mean(uncertainty_score)
            mean_stability = np.mean(stability)
            uncertainty_consistency = 1.0 - np.std(uncertainty_score) / (np.mean(uncertainty_score) + 1e-10)
            
            # Quality assessment
            if mean_uncertainty_score < 0.1 and mean_stability > 0.9:
                quality = "Excellent - Very reliable uncertainty estimates"
            elif mean_uncertainty_score < 0.2 and mean_stability > 0.8:
                quality = "Good - Reliable uncertainty estimates"
            elif mean_uncertainty_score < 0.3 and mean_stability > 0.7:
                quality = "Moderate - Reasonable uncertainty estimates"
            else:
                quality = "Poor - High uncertainty in estimates"
            
            return {
                'overall_quality': quality,
                'mean_uncertainty_score': mean_uncertainty_score,
                'mean_stability': mean_stability,
                'uncertainty_consistency': uncertainty_consistency,
                'reliability': 'High' if uncertainty_consistency > 0.8 else 'Moderate' if uncertainty_consistency > 0.6 else 'Low'
            }
        except:
            return {'error': 'Could not assess uncertainty quality'}
        
    def _analyze_feature_interactions(self):
        """Analyze feature interactions in CatBoost model"""
        if not self.feature_interaction_analysis:
            return
        
        try:
            analysis = {}
            
            # CatBoost's approach to feature interactions
            analysis['interaction_capabilities'] = {
                'automatic_interactions': 'CatBoost automatically discovers feature interactions',
                'categorical_combinations': 'Advanced categorical feature combinations',
                'tree_based_interactions': 'Interactions emerge naturally from tree structure',
                'ctr_combinations': 'Categorical-to-Real feature combinations'
            }
            
            # Analyze interactions through feature importance patterns
            if self.feature_importance_analysis_:
                importance_data = self.feature_importance_analysis_
                
                # Look for interaction patterns in importance scores
                top_features = importance_data.get('top_features', [])
                categorical_features = [f for f in top_features if f.get('is_categorical', False)]
                numerical_features = [f for f in top_features if not f.get('is_categorical', False)]
                
                analysis['interaction_patterns'] = {
                    'top_categorical_count': len(categorical_features),
                    'top_numerical_count': len(numerical_features),
                    'categorical_dominance': len(categorical_features) > len(numerical_features),
                    'mixed_importance': len(categorical_features) > 0 and len(numerical_features) > 0,
                    'interaction_potential': 'High' if len(categorical_features) >= 2 else 'Moderate'
                }
                
                # Feature interaction recommendations
                if len(categorical_features) >= 2:
                    analysis['interaction_recommendations'] = [
                        'High potential for categorical feature interactions',
                        'Consider increasing max_ctr_complexity for more combinations',
                        'Monitor categorical encoding analysis for interaction effects'
                    ]
                elif len(categorical_features) == 1 and len(numerical_features) >= 2:
                    analysis['interaction_recommendations'] = [
                        'Mixed categorical-numerical interactions likely',
                        'Good balance for CatBoost optimization',
                        'Tree structure naturally captures interactions'
                    ]
                else:
                    analysis['interaction_recommendations'] = [
                        'Primarily numerical feature interactions',
                        'Tree depth affects interaction complexity',
                        'Consider feature engineering for categorical variables'
                    ]
            
            # CatBoost-specific interaction advantages
            analysis['catboost_advantages'] = {
                'ordered_interactions': 'Interactions computed with overfitting protection',
                'efficient_categorical': 'Categorical interactions without feature explosion',
                'automatic_discovery': 'No manual feature engineering needed',
                'optimal_complexity': 'Automatically selects optimal interaction complexity'
            }
            
            self.feature_interaction_analysis_ = analysis
            
        except Exception as e:
            self.feature_interaction_analysis_ = {
                'error': f'Could not analyze feature interactions: {str(e)}'
            }
            
    def _profile_performance(self):
        """Profile CatBoost performance characteristics"""
        if not self.performance_profiling:
            return
        
        try:
            import time
            
            analysis = {}
            
            # Model characteristics
            analysis['model_characteristics'] = {
                'algorithm': 'CatBoost',
                'tree_count': self.model_.tree_count_,
                'tree_depth': self.depth,
                'categorical_features': len(self.categorical_features_indices_ or []),
                'total_features': len(self.feature_names_),
                'oblivious_trees': True
            }
            
            # Memory usage estimation
            analysis['memory_profile'] = {
                'tree_structure_bytes': (2 ** self.depth) * self.model_.tree_count_ * 8,
                'categorical_overhead': 'Minimal (efficient internal representation)',
                'model_size_category': self._assess_model_size(),
                'memory_efficiency': 'High (symmetric trees + efficient categorical handling)'
            }
            
            # Performance characteristics
            analysis['performance_characteristics'] = {
                'training_speed': 'Moderate (focus on accuracy)',
                'inference_speed': 'Very Fast (oblivious trees)',
                'categorical_efficiency': 'Excellent (native handling)',
                'overfitting_resistance': 'Excellent (ordered boosting)',
                'parallelization': 'Good (tree-level and feature-level)',
                'gpu_acceleration': 'Excellent' if self.thread_count == -1 else 'CPU-only'
            }
            
            # Scalability assessment
            analysis['scalability'] = {
                'sample_scalability': self._assess_sample_scalability(),
                'feature_scalability': self._assess_feature_scalability(),
                'categorical_scalability': 'Excellent (handles high cardinality efficiently)',
                'recommended_use_cases': self._get_recommended_use_cases()
            }
            
            # Performance recommendations
            analysis['optimization_recommendations'] = self._get_performance_recommendations()
            
            self.performance_profile_ = analysis
            
        except Exception as e:
            self.performance_profile_ = {
                'error': f'Could not profile performance: {str(e)}'
            }

    def _assess_model_size(self):
        """Assess model size category"""
        total_leaves = (2 ** self.depth) * self.model_.tree_count_
        
        if total_leaves < 1000:
            return "Small - Fast inference, low memory"
        elif total_leaves < 10000:
            return "Medium - Good balance"
        elif total_leaves < 100000:
            return "Large - High capacity, more memory"
        else:
            return "Very Large - High memory requirements"

    def _assess_sample_scalability(self):
        """Assess scalability with respect to sample size"""
        if hasattr(self, 'X_original_'):
            n_samples = len(self.X_original_)
            
            if n_samples < 1000:
                return "Small dataset - Excellent performance"
            elif n_samples < 10000:
                return "Medium dataset - Very good performance"
            elif n_samples < 100000:
                return "Large dataset - Good performance, consider GPU"
            else:
                return "Very large dataset - Consider distributed training"
        return "Unknown"
    
    def _assess_feature_scalability(self):
        """Assess scalability with respect to feature count"""
        if hasattr(self, 'X_original_'):
            n_features = self.X_original_.shape[1]
            
            if n_features < 50:
                return "Low dimensional - Excellent performance"
            elif n_features < 200:
                return "Medium dimensional - Very good performance"
            elif n_features < 1000:
                return "High dimensional - Good performance"
            else:
                return "Very high dimensional - Consider feature selection"
        return "Unknown"
    
    def _get_recommended_use_cases(self):
        """Get recommended use cases based on data characteristics"""
        use_cases = []
        
        if self.categorical_features_indices_:
            use_cases.extend([
                "Datasets with categorical features",
                "Mixed data types (categorical + numerical)",
                "High-cardinality categorical variables",
                "Customer segmentation and marketing",
                "Recommendation systems"
            ])
        
        use_cases.extend([
            "Tabular data regression problems",
            "Structured data with complex patterns",
            "Applications requiring high accuracy",
            "Scenarios with limited preprocessing time",
            "Production systems needing fast inference"
        ])
        
        return use_cases
    
    def _get_performance_recommendations(self):
        """Get performance optimization recommendations"""
        recommendations = []
        
        # Based on dataset size
        if hasattr(self, 'X_original_'):
            n_samples, n_features = self.X_original_.shape
            
            if n_samples > 100000:
                recommendations.append("Consider GPU acceleration for large datasets")
            
            if n_features > 500:
                recommendations.append("Consider feature selection for high-dimensional data")
        
        # Based on model configuration
        if self.iterations > 2000:
            recommendations.append("High iteration count - ensure early stopping is enabled")
        
        if self.depth > 8:
            recommendations.append("Deep trees - consider reducing depth for faster training")
        
        if len(self.categorical_features_indices_ or []) > 0:
            recommendations.append("Leverage CatBoost's categorical advantages - no preprocessing needed")
        
        # General recommendations
        recommendations.extend([
            "Use oblivious trees advantage for fast inference",
            "Monitor memory usage with very large models",
            "Consider model compression for production deployment"
        ])
        
        return recommendations
    
    def _get_performance_recommendations(self):
        """Get performance optimization recommendations"""
        recommendations = []
        
        # Based on dataset size
        if hasattr(self, 'X_original_'):
            n_samples, n_features = self.X_original_.shape
            
            if n_samples > 100000:
                recommendations.append("Consider GPU acceleration for large datasets")
            
            if n_features > 500:
                recommendations.append("Consider feature selection for high-dimensional data")
        
        # Based on model configuration
        if self.iterations > 2000:
            recommendations.append("High iteration count - ensure early stopping is enabled")
        
        if self.depth > 8:
            recommendations.append("Deep trees - consider reducing depth for faster training")
        
        if len(self.categorical_features_indices_ or []) > 0:
            recommendations.append("Leverage CatBoost's categorical advantages - no preprocessing needed")
        
        # General recommendations
        recommendations.extend([
            "Use oblivious trees advantage for fast inference",
            "Monitor memory usage with very large models",
            "Consider model compression for production deployment"
        ])
        
        return recommendations

    def _run_lightgbm_comparison(self):
        """Run a performance comparison with LightGBM"""
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        
        try:
            # Create LightGBM model with similar parameters
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': min(31, 2 ** min(6, self.depth)),
                'learning_rate': 0.1 if self.learning_rate is None else self.learning_rate,
                'feature_fraction': self.rsm,
                'bagging_fraction': self.subsample if self.bootstrap_type != 'No' else 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': self.l2_leaf_reg,
                'random_state': self.random_seed,
                'verbose': -1
            }
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=min(100, self.iterations // 10),  # Reduced for comparison
                **lgb_params
            )
            
            # Handle categorical features for LightGBM (basic support)
            X_lgb = self.X_original_.copy()
            if self.categorical_features_indices_:
                # LightGBM expects categorical features to be properly encoded
                # This is a simplified approach
                categorical_features = self.categorical_features_indices_
            else:
                categorical_features = 'auto'
            
            # Perform cross-validation comparison
            lgb_scores = cross_val_score(
                lgb_model, X_lgb, self.y_original_,
                cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # Get CatBoost scores for comparison
            catboost_scores = self.cross_validation_analysis_.get('cv_scores', [])
            if not isinstance(catboost_scores, list) or len(catboost_scores) == 0:
                # Fallback: run a quick CV for CatBoost
                from sklearn.model_selection import cross_val_score
                cb_model = CatBoostRegressor(
                    iterations=min(100, self.iterations // 10),
                    depth=min(6, self.depth),
                    verbose=False,
                    random_seed=self.random_seed
                )
                catboost_scores = cross_val_score(
                    cb_model, self.X_original_, self.y_original_,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=1
                )
                catboost_scores = -catboost_scores
            
            lgb_scores = -lgb_scores  # Convert to positive MSE
            
            comparison_result = {
                'catboost_mse': {
                    'mean': np.mean(catboost_scores),
                    'std': np.std(catboost_scores),
                    'scores': catboost_scores.tolist() if hasattr(catboost_scores, 'tolist') else catboost_scores
                },
                'lightgbm_mse': {
                    'mean': np.mean(lgb_scores),
                    'std': np.std(lgb_scores),
                    'scores': lgb_scores.tolist()
                },
                'winner': 'CatBoost' if np.mean(catboost_scores) < np.mean(lgb_scores) else 'LightGBM',
                'improvement': abs(np.mean(catboost_scores) - np.mean(lgb_scores)) / max(np.mean(catboost_scores), np.mean(lgb_scores)),
                'note': 'Limited comparison with reduced iterations for speed'
            }
            
            return comparison_result
            
        except Exception as e:
            return {'error': f'LightGBM comparison failed: {str(e)}'}

    def _compare_with_xgboost(self):
        """Compare CatBoost with XGBoost"""
        if not self.compare_with_xgboost:
            return
        
        try:
            # Try to import XGBoost
            try:
                import xgboost as xgb
                xgb_available = True
            except ImportError:
                xgb_available = False
            
            comparison = {}
            
            # Basic comparison
            comparison['algorithm_comparison'] = {
                'catboost_advantages': [
                    'Superior categorical feature handling',
                    'Better default parameters',
                    'Ordered boosting prevents overfitting',
                    'No need for extensive preprocessing',
                    'More robust to hyperparameter choices'
                ],
                'xgboost_advantages': [
                    'Very mature and stable',
                    'Extensive documentation and community',
                    'Highly optimized for performance',
                    'Wide range of objective functions',
                    'Better support for custom loss functions'
                ]
            }
            
            # Technical differences
            comparison['technical_differences'] = {
                'tree_structure': {
                    'catboost': 'Oblivious (symmetric) trees',
                    'xgboost': 'Level-wise asymmetric trees'
                },
                'categorical_handling': {
                    'catboost': 'Native support with optimal encoding',
                    'xgboost': 'Limited support, requires manual encoding'
                },
                'regularization': {
                    'catboost': 'Ordered boosting + L2 + model size + sampling',
                    'xgboost': 'L1 + L2 + early stopping'
                },
                'missing_values': {
                    'catboost': 'Automatic handling',
                    'xgboost': 'Automatic handling (different approach)'
                }
            }
            
            # Performance comparison (if XGBoost is available)
            if xgb_available and hasattr(self, 'X_original_'):
                try:
                    comparison['performance_comparison'] = self._run_xgboost_comparison()
                except Exception as e:
                    comparison['performance_comparison'] = {
                        'error': f'Could not run performance comparison: {str(e)}'
                    }
            else:
                comparison['performance_comparison'] = {
                    'note': 'XGBoost not available for performance comparison'
                }
            
            # Use case recommendations
            comparison['use_case_recommendations'] = {
                'prefer_catboost': [
                    'Datasets with categorical features',
                    'Need for minimal preprocessing',
                    'Risk-sensitive applications (ordered boosting)',
                    'Limited time for hyperparameter tuning',
                    'High-cardinality categorical variables'
                ],
                'prefer_xgboost': [
                    'Well-established production pipelines',
                    'Need for custom objective functions',
                    'Extensive hyperparameter tuning capability',
                    'Purely numerical datasets',
                    'Maximum performance optimization'
                ]
            }
            
            self.xgboost_comparison_ = comparison
            
        except Exception as e:
            self.xgboost_comparison_ = {
                'error': f'Could not compare with XGBoost: {str(e)}'
            }

    def _run_xgboost_comparison(self):
        """Run a performance comparison with XGBoost"""
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        try:
            # Create XGBoost model with similar parameters
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': min(6, self.depth),
                'learning_rate': 0.1 if self.learning_rate is None else self.learning_rate,
                'subsample': self.subsample if self.bootstrap_type != 'No' else 1.0,
                'colsample_bytree': self.rsm,
                'reg_alpha': 0.0,
                'reg_lambda': self.l2_leaf_reg,
                'random_state': self.random_seed,
                'verbosity': 0
            }
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=min(100, self.iterations // 10),  # Reduced for comparison
                **xgb_params
            )
            
            # XGBoost doesn't handle categorical features natively well
            # We need to preprocess them
            X_xgb = self.X_original_.copy()
            if self.categorical_features_indices_ and hasattr(self.X_original_, 'iloc'):
                # Simple label encoding for categorical features (not optimal, but functional)
                from sklearn.preprocessing import LabelEncoder
                
                for idx in self.categorical_features_indices_:
                    if hasattr(X_xgb, 'iloc'):
                        le = LabelEncoder()
                        X_xgb.iloc[:, idx] = le.fit_transform(X_xgb.iloc[:, idx].astype(str))
                    else:
                        le = LabelEncoder()
                        X_xgb[:, idx] = le.fit_transform(X_xgb[:, idx].astype(str))
            
            # Perform cross-validation comparison
            xgb_scores = cross_val_score(
                xgb_model, X_xgb, self.y_original_,
                cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # Get CatBoost scores for comparison
            catboost_scores = self.cross_validation_analysis_.get('cv_scores', [])
            if not isinstance(catboost_scores, list) or len(catboost_scores) == 0:
                # Fallback: run a quick CV for CatBoost
                cb_model = CatBoostRegressor(
                    iterations=min(100, self.iterations // 10),
                    depth=min(6, self.depth),
                    verbose=False,
                    random_seed=self.random_seed
                )
                catboost_scores = cross_val_score(
                    cb_model, self.X_original_, self.y_original_,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=1
                )
                catboost_scores = -catboost_scores
            
            xgb_scores = -xgb_scores  # Convert to positive MSE
            
            comparison_result = {
                'catboost_mse': {
                    'mean': np.mean(catboost_scores),
                    'std': np.std(catboost_scores),
                    'scores': catboost_scores.tolist() if hasattr(catboost_scores, 'tolist') else catboost_scores
                },
                'xgboost_mse': {
                    'mean': np.mean(xgb_scores),
                    'std': np.std(xgb_scores),
                    'scores': xgb_scores.tolist()
                },
                'winner': 'CatBoost' if np.mean(catboost_scores) < np.mean(xgb_scores) else 'XGBoost',
                'improvement': abs(np.mean(catboost_scores) - np.mean(xgb_scores)) / max(np.mean(catboost_scores), np.mean(xgb_scores)),
                'note': 'XGBoost used label encoding for categorical features (suboptimal)'
            }
            
            return comparison_result
            
        except Exception as e:
            return {'error': f'XGBoost comparison failed: {str(e)}'}

    def _compare_with_sklearn_gbm(self):
        """Compare CatBoost with Scikit-learn Gradient Boosting"""
        if not self.compare_with_sklearn_gbm:
            return
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            
            comparison = {}
            
            # Basic comparison
            comparison['algorithm_comparison'] = {
                'catboost_advantages': [
                    'Superior categorical feature handling',
                    'Better overfitting protection (ordered boosting)',
                    'Faster training and inference',
                    'Better default parameters',
                    'Modern algorithm with latest innovations'
                ],
                'sklearn_gbm_advantages': [
                    'Part of standard scikit-learn',
                    'No additional dependencies',
                    'Simple and well-understood',
                    'Good documentation and examples',
                    'Stable and reliable'
                ]
            }
            
            # Technical differences
            comparison['technical_differences'] = {
                'algorithm_age': {
                    'catboost': '2017 - Modern with latest research',
                    'sklearn_gbm': '2001 - Classical gradient boosting'
                },
                'tree_structure': {
                    'catboost': 'Oblivious (symmetric) trees',
                    'sklearn_gbm': 'Standard decision trees'
                },
                'categorical_handling': {
                    'catboost': 'Native with optimal encoding',
                    'sklearn_gbm': 'No native support, requires preprocessing'
                },
                'performance': {
                    'catboost': 'Optimized for speed and accuracy',
                    'sklearn_gbm': 'General purpose, slower'
                }
            }
            
            # Performance comparison
            if hasattr(self, 'X_original_'):
                try:
                    comparison['performance_comparison'] = self._run_sklearn_gbm_comparison()
                except Exception as e:
                    comparison['performance_comparison'] = {
                        'error': f'Could not run performance comparison: {str(e)}'
                    }
            
            # Use case recommendations
            comparison['use_case_recommendations'] = {
                'prefer_catboost': [
                    'Any serious machine learning project',
                    'Datasets with categorical features',
                    'Need for high performance',
                    'Production systems',
                    'Time-sensitive projects'
                ],
                'prefer_sklearn_gbm': [
                    'Educational purposes',
                    'Simple prototyping',
                    'Environments where dependencies matter',
                    'Legacy system compatibility',
                    'When simplicity is preferred over performance'
                ]
            }
            
            self.sklearn_gbm_comparison_ = comparison
            
        except Exception as e:
            self.sklearn_gbm_comparison_ = {
                'error': f'Could not compare with Scikit-learn GBM: {str(e)}'
            }

    def _run_sklearn_gbm_comparison(self):
        """Run a performance comparison with Scikit-learn Gradient Boosting"""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        try:
            # Create Scikit-learn GBM model
            sklearn_gbm = GradientBoostingRegressor(
                n_estimators=min(100, self.iterations // 10),  # Reduced for comparison
                max_depth=min(6, self.depth),
                learning_rate=0.1 if self.learning_rate is None else self.learning_rate,
                subsample=self.subsample if self.bootstrap_type != 'No' else 1.0,
                random_state=self.random_seed
            )
            
            # Preprocess categorical features for sklearn
            X_sklearn = self.X_original_.copy()
            if self.categorical_features_indices_ and hasattr(self.X_original_, 'iloc'):
                # Label encode categorical features
                for idx in self.categorical_features_indices_:
                    if hasattr(X_sklearn, 'iloc'):
                        le = LabelEncoder()
                        X_sklearn.iloc[:, idx] = le.fit_transform(X_sklearn.iloc[:, idx].astype(str))
                    else:
                        le = LabelEncoder()
                        X_sklearn[:, idx] = le.fit_transform(X_sklearn[:, idx].astype(str))
            
            # Perform cross-validation comparison
            sklearn_scores = cross_val_score(
                sklearn_gbm, X_sklearn, self.y_original_,
                cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # Get CatBoost scores for comparison
            catboost_scores = self.cross_validation_analysis_.get('cv_scores', [])
            if not isinstance(catboost_scores, list) or len(catboost_scores) == 0:
                # Fallback: run a quick CV for CatBoost
                cb_model = CatBoostRegressor(
                    iterations=min(100, self.iterations // 10),
                    depth=min(6, self.depth),
                    verbose=False,
                    random_seed=self.random_seed
                )
                catboost_scores = cross_val_score(
                    cb_model, self.X_original_, self.y_original_,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=1
                )
                catboost_scores = -catboost_scores
            
            sklearn_scores = -sklearn_scores  # Convert to positive MSE
            
            comparison_result = {
                'catboost_mse': {
                    'mean': np.mean(catboost_scores),
                    'std': np.std(catboost_scores),
                    'scores': catboost_scores.tolist() if hasattr(catboost_scores, 'tolist') else catboost_scores
                },
                'sklearn_gbm_mse': {
                    'mean': np.mean(sklearn_scores),
                    'std': np.std(sklearn_scores),
                    'scores': sklearn_scores.tolist()
                },
                'winner': 'CatBoost' if np.mean(catboost_scores) < np.mean(sklearn_scores) else 'Scikit-learn GBM',
                'improvement': abs(np.mean(catboost_scores) - np.mean(sklearn_scores)) / max(np.mean(catboost_scores), np.mean(sklearn_scores)),
                'note': 'Sklearn GBM used label encoding for categorical features'
            }
            
            return comparison_result
            
        except Exception as e:
            return {'error': f'Sklearn GBM comparison failed: {str(e)}'}

    def _compare_with_lightgbm(self):
        """Compare CatBoost with LightGBM"""
        if not self.compare_with_lightgbm:
            return
        
        try:
            # Try to import LightGBM
            try:
                import lightgbm as lgb
                lgb_available = True
            except ImportError:
                lgb_available = False
            
            comparison = {}
            
            # Basic comparison
            comparison['algorithm_comparison'] = {
                'catboost_advantages': [
                    'Superior categorical feature handling without preprocessing',
                    'Ordered boosting prevents overfitting',
                    'Built-in target leakage protection',
                    'Better default parameters',
                    'Symmetric trees for faster inference'
                ],
                'lightgbm_advantages': [
                    'Generally faster training speed',
                    'Lower memory usage',
                    'More flexible tree growth (leaf-wise)',
                    'Better performance on purely numerical data',
                    'More mature ecosystem'
                ]
            }
            
            # Technical differences
            comparison['technical_differences'] = {
                'tree_structure': {
                    'catboost': 'Oblivious (symmetric) trees',
                    'lightgbm': 'Leaf-wise asymmetric trees'
                },
                'categorical_handling': {
                    'catboost': 'Native with target statistics and ordered boosting',
                    'lightgbm': 'Basic categorical support, requires preprocessing for best results'
                },
                'overfitting_protection': {
                    'catboost': 'Ordered boosting (primary), regularization (secondary)',
                    'lightgbm': 'Regularization, early stopping, feature selection'
                },
                'default_performance': {
                    'catboost': 'Excellent out-of-the-box performance',
                    'lightgbm': 'Good, but requires more tuning'
                }
            }
            
            # Performance comparison (if LightGBM is available)
            if lgb_available and hasattr(self, 'X_original_'):
                try:
                    comparison['performance_comparison'] = self._run_lightgbm_comparison()
                except Exception as e:
                    comparison['performance_comparison'] = {
                        'error': f'Could not run performance comparison: {str(e)}'
                    }
            else:
                comparison['performance_comparison'] = {
                    'note': 'LightGBM not available for performance comparison'
                }
            
            # Use case recommendations
            comparison['use_case_recommendations'] = {
                'prefer_catboost': [
                    'Datasets with many categorical features',
                    'High-cardinality categorical variables',
                    'Need for robust default parameters',
                    'Risk of target leakage in categorical encoding',
                    'Production systems requiring stable performance'
                ],
                'prefer_lightgbm': [
                    'Large datasets where training speed is critical',
                    'Purely numerical datasets',
                    'Memory-constrained environments',
                    'Need for extensive hyperparameter tuning flexibility'
                ]
            }
            
            self.lightgbm_comparison_ = comparison
            
        except Exception as e:
            self.lightgbm_comparison_ = {
                'error': f'Could not compare with LightGBM: {str(e)}'
            }

    def supports_task(self, task_type: str) -> bool:
        """
        Check if the plugin supports a specific task type
        
        Parameters:
        -----------
        task_type : str
            The task type to check ('regression', 'classification', 'clustering', etc.)
        
        Returns:
        --------
        bool
            True if the task type is supported, False otherwise
        """
        task_type = task_type.lower()
        
        if task_type in ['regression', 'regressor']:
            return self._supports_regression
        elif task_type in ['classification', 'classifier']:
            return self._supports_classification
        else:
            return False

    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🚀 CatBoost Regressor Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Categorical", "Regularization", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Iterations
            iterations = st.slider(
                "Iterations:",
                min_value=100,
                max_value=5000,
                value=int(self.iterations),
                step=50,
                help="Number of boosting iterations",
                key=f"{key_prefix}_iterations"
            )
            
            # Learning rate
            auto_lr = st.checkbox(
                "Auto Learning Rate",
                value=self.learning_rate is None,
                help="Let CatBoost automatically tune learning rate",
                key=f"{key_prefix}_auto_lr"
            )
            
            if not auto_lr:
                learning_rate = st.number_input(
                    "Learning Rate:",
                    value=0.1 if self.learning_rate is None else float(self.learning_rate),
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    help="Shrinkage rate for preventing overfitting",
                    key=f"{key_prefix}_learning_rate"
                )
            else:
                learning_rate = None
            
            # Tree depth
            depth = st.slider(
                "Tree Depth:",
                min_value=1,
                max_value=16,
                value=int(self.depth),
                help="Depth of trees (4-10 recommended)",
                key=f"{key_prefix}_depth"
            )
            
            # Loss function for regression
            loss_function = st.selectbox(
                "Loss Function:",
                options=['RMSE', 'MAE', 'MAPE', 'Quantile', 'Poisson'],
                index=['RMSE', 'MAE', 'MAPE', 'Quantile', 'Poisson'].index(self.loss_function),
                help="Loss function for regression",
                key=f"{key_prefix}_loss_function"
            )
            
            # Bootstrap type
            bootstrap_type = st.selectbox(
                "Bootstrap Type:",
                options=['Bayesian', 'Bernoulli', 'No'],
                index=['Bayesian', 'Bernoulli', 'No'].index(self.bootstrap_type),
                help="Bayesian: default, Bernoulli: for large data, No: no sampling",
                key=f"{key_prefix}_bootstrap_type"
            )
        
        with tab2:
            st.markdown("**Categorical Features**")
            
            # Auto-detect categorical features
            auto_detect_cat = st.checkbox(
                "Auto-detect Categorical Features",
                value=True,
                help="Automatically detect categorical features based on data types",
                key=f"{key_prefix}_auto_detect_cat"
            )
            
            # One-hot encoding threshold
            one_hot_max_size = st.slider(
                "One-Hot Max Size:",
                min_value=1,
                max_value=20,
                value=int(self.one_hot_max_size),
                help="Maximum cardinality for one-hot encoding (vs target encoding)",
                key=f"{key_prefix}_one_hot_max_size"
            )
            
            # CTR complexity
            max_ctr_complexity = st.slider(
                "Max CTR Complexity:",
                min_value=1,
                max_value=8,
                value=int(self.max_ctr_complexity),
                help="Maximum complexity for categorical feature combinations",
                key=f"{key_prefix}_max_ctr_complexity"
            )
            
            # Border count
            border_count = st.slider(
                "Border Count:",
                min_value=32,
                max_value=512,
                value=int(self.border_count),
                help="Number of splits for numerical features",
                key=f"{key_prefix}_border_count"
            )
            
            # Feature border type
            feature_border_type = st.selectbox(
                "Feature Border Type:",
                options=['GreedyLogSum', 'Uniform', 'MinEntropy', 'MaxLogSum'],
                index=['GreedyLogSum', 'Uniform', 'MinEntropy', 'MaxLogSum'].index(self.feature_border_type),
                help="Algorithm for selecting borders",
                key=f"{key_prefix}_feature_border_type"
            )
        
        with tab3:
            st.markdown("**Regularization & Overfitting**")
            
            # L2 regularization
            l2_leaf_reg = st.number_input(
                "L2 Leaf Regularization:",
                value=float(self.l2_leaf_reg),
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                help="L2 regularization coefficient (3.0 is good default)",
                key=f"{key_prefix}_l2_leaf_reg"
            )
            
            # Model size regularization
            model_size_reg = st.slider(
                "Model Size Regularization:",
                min_value=0.0,
                max_value=2.0,
                value=float(self.model_size_reg),
                step=0.1,
                help="Model complexity penalty",
                key=f"{key_prefix}_model_size_reg"
            )
            
            # Random strength
            random_strength = st.slider(
                "Random Strength:",
                min_value=0.0,
                max_value=5.0,
                value=float(self.random_strength),
                step=0.1,
                help="Amount of randomness in tree structure",
                key=f"{key_prefix}_random_strength"
            )
            
            # Bagging temperature
            bagging_temperature = st.slider(
                "Bagging Temperature:",
                min_value=0.0,
                max_value=10.0,
                value=float(self.bagging_temperature),
                step=0.1,
                help="Controls intensity of Bayesian bagging",
                key=f"{key_prefix}_bagging_temperature"
            )
            
            # Overfitting detection
            od_type = st.selectbox(
                "Overfitting Detection:",
                options=['IncToDec', 'Iter'],
                index=['IncToDec', 'Iter'].index(self.od_type),
                help="Type of overfitting detection",
                key=f"{key_prefix}_od_type"
            )
            
            # Early stopping patience
            od_wait = st.slider(
                "Early Stopping Patience:",
                min_value=5,
                max_value=100,
                value=int(self.od_wait),
                help="Number of iterations to wait before stopping",
                key=f"{key_prefix}_od_wait"
            )
        
        with tab4:
            st.markdown("**Advanced Parameters**")
            
            # RSM (Random Subspace Method)
            rsm = st.slider(
                "Random Subspace (RSM):",
                min_value=0.1,
                max_value=1.0,
                value=float(self.rsm),
                step=0.05,
                help="Random subspace method - fraction of features to use",
                key=f"{key_prefix}_rsm"
            )
            
            # Bootstrap subsample (if applicable)
            if bootstrap_type in ['Bayesian', 'Bernoulli']:
                subsample = st.slider(
                    "Subsample Ratio:",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.subsample),
                    step=0.05,
                    help="Fraction of samples for training each tree",
                    key=f"{key_prefix}_subsample"
                )
            else:
                subsample = 1.0
            
            # Early stopping rounds
            enable_early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=self.early_stopping_rounds is not None,
                key=f"{key_prefix}_enable_early_stopping"
            )
            if enable_early_stopping:
                early_stopping_rounds = st.number_input(
                    "Early Stopping Rounds:",
                    min_value=1,
                    max_value=1000,
                    value=self.early_stopping_rounds or 20,
                    key=f"{key_prefix}_early_stopping_rounds"
                )
            else:
                early_stopping_rounds = None
            
            # Thread count
            thread_count = st.selectbox(
                "Thread Count:",
                options=[-1, 1, 2, 4, 8, 16],
                index=0,
                help="Number of threads (-1 for auto)",
                key=f"{key_prefix}_thread_count"
            )
            
            # Random seed
            random_seed = st.number_input(
                "Random Seed:",
                value=int(self.random_seed),
                min_value=0,
                max_value=99999,
                help="Random seed for reproducibility",
                key=f"{key_prefix}_random_seed"
            )
            
            # Verbose training
            verbose = st.checkbox(
                "Verbose Training",
                value=self.verbose,
                key=f"{key_prefix}_verbose"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if CATBOOST_AVAILABLE:
                st.success(f"✅ CatBoost is available")
            else:
                st.error("❌ CatBoost not installed. Run: pip install catboost")
            
            st.info("""
            **CatBoost Regressor** - Advanced Gradient Boosting:
            • 🚀 Superior categorical feature handling
            • 🎯 No preprocessing required
            • 🛡️ Built-in overfitting protection
            • ⚡ Symmetric trees for fast inference
            • 🔧 Minimal hyperparameter tuning
            • 📊 Advanced target encoding
            
            **Regression Advantages:**
            • Handles high-cardinality categories
            • Automatic feature combinations
            • Robust to missing categories
            • Ordered boosting prevents overfitting
            • Multiple loss functions available
            """)
            
            # Analysis options
            st.markdown("**Analysis Options**")
            col1, col2 = st.columns(2)
            
            with col1:
                compute_feature_importance = st.checkbox(
                    "Feature Importance",
                    value=self.compute_feature_importance,
                    key=f"{key_prefix}_compute_feature_importance"
                )
                catboost_analysis = st.checkbox(
                    "CatBoost Analysis",
                    value=self.catboost_analysis,
                    key=f"{key_prefix}_catboost_analysis"
                )
                categorical_analysis = st.checkbox(
                    "Categorical Analysis",
                    value=self.categorical_analysis,
                    key=f"{key_prefix}_categorical_analysis"
                )
            
            with col2:
                overfitting_analysis = st.checkbox(
                    "Overfitting Analysis",
                    value=self.overfitting_analysis,
                    key=f"{key_prefix}_overfitting_analysis"
                )
                convergence_analysis = st.checkbox(
                    "Convergence Analysis",
                    value=self.convergence_analysis,
                    key=f"{key_prefix}_convergence_analysis"
                )
                cross_validation_analysis = st.checkbox(
                    "Cross Validation",
                    value=self.cross_validation_analysis,
                    key=f"{key_prefix}_cross_validation_analysis"
                )
            
            # Regression guide
            if st.button("📈 Regression Guide", key=f"{key_prefix}_regression_guide"):
                st.markdown("""
                **CatBoost Regression Best Practices:**
                
                **Loss Functions:**
                - RMSE: Standard regression (default)
                - MAE: Robust to outliers
                - MAPE: Mean absolute percentage error
                - Quantile: Quantile regression
                - Poisson: Count data
                
                **Parameter Tuning:**
                - Start with defaults
                - Increase iterations (500-2000)
                - Tune depth (4-10)
                - Adjust regularization if overfitting
                """)
            
            # Performance comparison
            if st.button("⚖️ vs Other Regressors", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **CatBoost vs Competitors:**
                
                **vs XGBoost:**
                - ✅ Much better categorical handling
                - ✅ No preprocessing needed
                - ✅ Better overfitting resistance
                - ❌ Slightly slower training
                
                **vs LightGBM:**
                - ✅ Superior categorical encoding
                - ✅ Better default parameters
                - ✅ More robust to overfitting
                - ❌ Higher memory usage
                
                **vs Random Forest:**
                - ✅ Better accuracy typically
                - ✅ Better categorical handling
                - ✅ More sophisticated boosting
                - ❌ More complex hyperparameters
                """)
        
        return {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "model_size_reg": model_size_reg,
            "rsm": rsm,
            "loss_function": loss_function,
            "border_count": border_count,
            "feature_border_type": feature_border_type,
            "od_wait": od_wait,
            "od_type": od_type,
            "thread_count": thread_count,
            "random_seed": random_seed,
            "verbose": verbose,
            "allow_writing_files": False,
            "max_ctr_complexity": max_ctr_complexity,
            "one_hot_max_size": one_hot_max_size,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "bootstrap_type": bootstrap_type,
            "subsample": subsample,
            "early_stopping_rounds": early_stopping_rounds,
            "cat_features": None if auto_detect_cat else [],
            "compute_feature_importance": compute_feature_importance,
            "catboost_analysis": catboost_analysis,
            "categorical_analysis": categorical_analysis,
            "overfitting_analysis": overfitting_analysis,
            "convergence_analysis": convergence_analysis,
            "cross_validation_analysis": cross_validation_analysis
        }

    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return CatBoostRegressorPlugin(
            iterations=hyperparameters.get("iterations", self.iterations),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            depth=hyperparameters.get("depth", self.depth),
            l2_leaf_reg=hyperparameters.get("l2_leaf_reg", self.l2_leaf_reg),
            model_size_reg=hyperparameters.get("model_size_reg", self.model_size_reg),
            rsm=hyperparameters.get("rsm", self.rsm),
            loss_function=hyperparameters.get("loss_function", self.loss_function),
            border_count=hyperparameters.get("border_count", self.border_count),
            feature_border_type=hyperparameters.get("feature_border_type", self.feature_border_type),
            od_wait=hyperparameters.get("od_wait", self.od_wait),
            od_type=hyperparameters.get("od_type", self.od_type),
            thread_count=hyperparameters.get("thread_count", self.thread_count),
            random_seed=hyperparameters.get("random_seed", self.random_seed),
            verbose=hyperparameters.get("verbose", self.verbose),
            allow_writing_files=hyperparameters.get("allow_writing_files", False),
            max_ctr_complexity=hyperparameters.get("max_ctr_complexity", self.max_ctr_complexity),
            one_hot_max_size=hyperparameters.get("one_hot_max_size", self.one_hot_max_size),
            random_strength=hyperparameters.get("random_strength", self.random_strength),
            bagging_temperature=hyperparameters.get("bagging_temperature", self.bagging_temperature),
            bootstrap_type=hyperparameters.get("bootstrap_type", self.bootstrap_type),
            subsample=hyperparameters.get("subsample", self.subsample),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds", self.early_stopping_rounds),
            cat_features=hyperparameters.get("cat_features", self.cat_features),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            catboost_analysis=hyperparameters.get("catboost_analysis", self.catboost_analysis),
            categorical_analysis=hyperparameters.get("categorical_analysis", self.categorical_analysis),
            overfitting_analysis=hyperparameters.get("overfitting_analysis", self.overfitting_analysis),
            convergence_analysis=hyperparameters.get("convergence_analysis", self.convergence_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis)
        )

        def get_hyperparameters(self) -> Dict[str, Any]:
            """Get current hyperparameters"""
            return {
                # Core boosting parameters
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'l2_leaf_reg': self.l2_leaf_reg,
                
                # Categorical handling
                'one_hot_max_size': self.one_hot_max_size,
                'max_ctr_complexity': self.max_ctr_complexity,
                
                # Model structure
                'model_size_reg': self.model_size_reg,
                'rsm': self.rsm,
                'border_count': self.border_count,
                'feature_border_type': self.feature_border_type,
                
                # Regularization
                'bagging_temperature': self.bagging_temperature,
                'random_strength': self.random_strength,
                'bootstrap_type': self.bootstrap_type,
                'subsample': self.subsample,
                
                # Training control
                'random_seed': self.random_seed,
                'thread_count': self.thread_count,
                'early_stopping_rounds': self.early_stopping_rounds,
                
                # Loss function
                'loss_function': self.loss_function,
                'eval_metric': self.eval_metric
            }

        def get_feature_importance(self) -> Dict[str, float]:
            """Get feature importance as a dictionary"""
            if not self.is_fitted_ or not self.feature_importance_analysis_:
                return {}
            
            try:
                importance_scores = self.feature_importance_analysis_.get('importance_scores', {})
                feature_names = self.feature_importance_analysis_.get('feature_names', [])
                
                # Use PredictionValuesChange as primary importance if available
                if 'PredictionValuesChange' in importance_scores:
                    scores = importance_scores['PredictionValuesChange']
                elif 'FeatureImportance' in importance_scores:
                    scores = importance_scores['FeatureImportance']
                else:
                    return {}
                
                return dict(zip(feature_names, scores))
            except:
                return {}

        def get_model_info(self) -> Dict[str, Any]:
            """Get comprehensive model information"""
            if not self.is_fitted_:
                return {'error': 'Model not fitted yet'}
            
            info = {
                'model_type': 'CatBoost Regressor',
                'is_fitted': self.is_fitted_,
                'n_features': self.n_features_in_,
                'feature_names': self.feature_names_,
                'categorical_features': self.categorical_features_,
                'n_categorical_features': len(self.categorical_features_indices_ or []),
                'tree_count': self.model_.tree_count_ if self.model_ else 0,
                'tree_depth': self.depth,
                'learning_rate': self.learning_rate,
                'loss_function': self.loss_function,
                'random_seed': self.random_seed
            }
            
            # Add analysis summaries if available
            if self.catboost_analysis_:
                info['catboost_innovations'] = self.catboost_analysis_.get('catboost_innovations', {})
            
            if self.overfitting_analysis_:
                info['overfitting_risk'] = self.overfitting_analysis_.get('overfitting_risk_assessment', {}).get('overall_risk', 'Unknown')
            
            if self.cross_validation_analysis_:
                cv_analysis = self.cross_validation_analysis_
                if 'mean_rmse' in cv_analysis:
                    info['cv_rmse'] = cv_analysis['mean_rmse']
                    info['cv_stability'] = cv_analysis.get('stability_assessment', 'Unknown')
            
            return info

        def create_streamlit_ui(self) -> None:
            """Create Streamlit UI for hyperparameter configuration"""
            st.subheader("🚀 CatBoost Regressor Configuration")
            
            # Core Parameters
            with st.expander("🎯 Core Boosting Parameters", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    self.iterations = st.number_input(
                        "Iterations", 
                        min_value=10, max_value=10000, value=self.iterations,
                        help="Number of boosting iterations"
                    )
                    
                    self.depth = st.slider(
                        "Tree Depth", 
                        min_value=1, max_value=16, value=self.depth,
                        help="Depth of trees (CatBoost uses symmetric trees)"
                    )
                
                with col2:
                    learning_rate_auto = st.checkbox("Auto Learning Rate", value=self.learning_rate is None)
                    if not learning_rate_auto:
                        self.learning_rate = st.number_input(
                            "Learning Rate", 
                            min_value=0.01, max_value=1.0, value=self.learning_rate or 0.1, step=0.01,
                            help="Learning rate for boosting"
                        )
                    else:
                        self.learning_rate = None
                    
                    self.l2_leaf_reg = st.number_input(
                        "L2 Leaf Regularization", 
                        min_value=0.0, max_value=50.0, value=self.l2_leaf_reg, step=0.1,
                        help="L2 regularization coefficient"
                    )
            
            # Categorical Features
            with st.expander("🏷️ Categorical Feature Handling"):
                col1, col2 = st.columns(2)
                
                with col1:
                    self.one_hot_max_size = st.number_input(
                        "One-Hot Max Size", 
                        min_value=1, max_value=100, value=self.one_hot_max_size,
                        help="Maximum unique values for one-hot encoding"
                    )
                    
                    self.max_ctr_complexity = st.slider(
                        "Max CTR Complexity", 
                        min_value=1, max_value=10, value=self.max_ctr_complexity,
                        help="Maximum complexity of categorical combinations"
                    )
                
                with col2:
                    self.border_count = st.number_input(
                        "Border Count", 
                        min_value=32, max_value=512, value=self.border_count,
                        help="Number of splits for numerical features"
                    )
                    
                    self.feature_border_type = st.selectbox(
                        "Feature Border Type",
                        ['GreedyLogSum', 'Uniform', 'MinEntropy', 'MaxLogSum'],
                        index=0,
                        help="Algorithm for selecting borders"
                    )
            
            # Regularization
            with st.expander("🛡️ Regularization & Sampling"):
                col1, col2 = st.columns(2)
                
                with col1:
                    self.model_size_reg = st.slider(
                        "Model Size Regularization", 
                        min_value=0.0, max_value=2.0, value=self.model_size_reg, step=0.1,
                        help="Model complexity penalty"
                    )
                    
                    self.random_strength = st.slider(
                        "Random Strength", 
                        min_value=0.0, max_value=5.0, value=self.random_strength, step=0.1,
                        help="Amount of randomness in tree structure"
                    )
                    
                    self.bagging_temperature = st.slider(
                        "Bagging Temperature", 
                        min_value=0.0, max_value=10.0, value=self.bagging_temperature, step=0.1,
                        help="Controls intensity of Bayesian bagging"
                    )
                
                with col2:
                    self.bootstrap_type = st.selectbox(
                        "Bootstrap Type",
                        ['Bayesian', 'Bernoulli', 'No'],
                        index=0,
                        help="Type of bootstrap sampling"
                    )
                    
                    if self.bootstrap_type in ['Bayesian', 'Bernoulli']:
                        self.subsample = st.slider(
                            "Subsample Ratio", 
                            min_value=0.1, max_value=1.0, value=self.subsample, step=0.05,
                            help="Fraction of samples for training each tree"
                        )
                    
                    self.rsm = st.slider(
                        "Random Subspace Method", 
                        min_value=0.1, max_value=1.0, value=self.rsm, step=0.05,
                        help="Fraction of features for training each tree"
                    )
            
            # Advanced Settings
            with st.expander("⚙️ Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    enable_early_stopping = st.checkbox(
                        "Enable Early Stopping", 
                        value=self.early_stopping_rounds is not None
                    )
                    if enable_early_stopping:
                        self.early_stopping_rounds = st.number_input(
                            "Early Stopping Rounds", 
                            min_value=1, max_value=1000, value=self.early_stopping_rounds or 20
                        )
                    else:
                        self.early_stopping_rounds = None
                    
                    self.random_seed = st.number_input(
                        "Random Seed", 
                        min_value=0, max_value=99999, value=self.random_seed,
                        help="Random seed for reproducibility"
                    )
                
                with col2:
                    self.thread_count = st.selectbox(
                        "Thread Count",
                        [-1, 1, 2, 4, 8, 16],
                        index=0,
                        help="Number of threads (-1 for auto)"
                    )
                    
                    self.verbose = st.checkbox("Verbose Training", value=self.verbose)
            
            # Analysis Options
            with st.expander("📊 Analysis Options"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    self.compute_feature_importance = st.checkbox(
                        "Feature Importance", value=self.compute_feature_importance
                    )
                    self.catboost_analysis = st.checkbox(
                        "CatBoost Analysis", value=self.catboost_analysis
                    )
                    self.categorical_analysis = st.checkbox(
                        "Categorical Analysis", value=self.categorical_analysis
                    )
                
                with col2:
                    self.overfitting_analysis = st.checkbox(
                        "Overfitting Analysis", value=self.overfitting_analysis
                    )
                    self.convergence_analysis = st.checkbox(
                        "Convergence Analysis", value=self.convergence_analysis
                    )
                    self.cross_validation_analysis = st.checkbox(
                        "Cross Validation", value=self.cross_validation_analysis
                    )
                
                with col3:
                    self.compare_with_lightgbm = st.checkbox(
                        "Compare with LightGBM", value=self.compare_with_lightgbm
                    )
                    self.compare_with_xgboost = st.checkbox(
                        "Compare with XGBoost", value=self.compare_with_xgboost
                    )
                    self.performance_profiling = st.checkbox(
                        "Performance Profiling", value=self.performance_profiling
                    )

        def get_analysis_results(self) -> Dict[str, Any]:
            """Get all analysis results"""
            if not self.is_fitted_:
                return {'error': 'Model not fitted yet'}
            
            results = {
                'model_info': self.get_model_info(),
                'feature_importance': self.feature_importance_analysis_,
                'catboost_analysis': self.catboost_analysis_,
                'categorical_analysis': self.categorical_analysis_,
                'overfitting_analysis': self.overfitting_analysis_,
                'convergence_analysis': self.convergence_analysis_,
                'regularization_analysis': self.regularization_analysis_,
                'cross_validation_analysis': self.cross_validation_analysis_,
                'tree_analysis': self.tree_analysis_,
                'prediction_uncertainty_analysis': self.prediction_uncertainty_analysis_,
                'categorical_encoding_analysis': self.categorical_encoding_analysis_,
                'feature_interaction_analysis': self.feature_interaction_analysis_,
                'performance_profile': self.performance_profile_
            }
            
            # Add comparison results if available
            if hasattr(self, 'lightgbm_comparison_'):
                results['lightgbm_comparison'] = self.lightgbm_comparison_
            
            if hasattr(self, 'xgboost_comparison_'):
                results['xgboost_comparison'] = self.xgboost_comparison_
            
            if hasattr(self, 'sklearn_gbm_comparison_'):
                results['sklearn_gbm_comparison'] = self.sklearn_gbm_comparison_
            
            return results

        # ADD THE OVERRIDDEN METHOD HERE:
        def get_algorithm_specific_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_proba: Optional[np.ndarray] = None
                                        ) -> Dict[str, Any]:
            """
            Calculate CatBoost Regressor-specific metrics based on the fitted model
            and optionally on test set prediction uncertainties.

            Args:
                y_true: Ground truth target values from the test set.
                y_pred: Predicted target values on the test set.
                y_proba: Optional. For regression, this is interpreted as the
                        standard deviations of the predictions (y_pred_std) on the test set.

            Returns:
                A dictionary of CatBoost Regressor-specific metrics.
            """
            metrics = {}
            if not self.is_fitted_ or not CATBOOST_AVAILABLE or self.model_ is None:
                metrics["status"] = "Model not fitted or CatBoost not available"
                return metrics

            # --- Metrics from the fitted CatBoost model ---
            metrics['num_trees_built'] = int(self.model_.tree_count_)
            
            best_iter = self.model_.get_best_iteration()
            if best_iter is not None:
                metrics['best_iteration'] = int(best_iter)

            best_score_data = self.model_.get_best_score()
            if best_score_data:
                # best_score_data is like: {'learn': {'RMSE': 0.1}, 'validation': {'RMSE': 0.2}}
                if 'learn' in best_score_data and best_score_data['learn']:
                    primary_learn_metric_name = next(iter(best_score_data['learn']))
                    metrics[f'best_score_learn_{primary_learn_metric_name}'] = float(best_score_data['learn'][primary_learn_metric_name])
                
                val_set_keys = [k for k in best_score_data if k.startswith('validation')]
                if val_set_keys:
                    # Use the first validation set found (e.g., 'validation' or 'validation_0')
                    val_key = val_set_keys[0]
                    if best_score_data[val_key]:
                        primary_val_metric_name = next(iter(best_score_data[val_key]))
                        metrics[f'best_score_{val_key}_{primary_val_metric_name}'] = float(best_score_data[val_key][primary_val_metric_name])

            metrics['loss_function_used'] = self.model_.get_params().get('loss_function', self.loss_function)
            eval_metric_used = self.model_.get_params().get('eval_metric')
            if eval_metric_used:
                metrics['primary_eval_metric_configured'] = eval_metric_used
            
            if hasattr(self.model_, 'learning_rate_'):
                metrics['learned_learning_rate'] = float(self.model_.learning_rate_)


            # --- Metrics from categorical feature handling ---
            if self.categorical_features_indices_ is not None:
                metrics['num_categorical_features_handled'] = len(self.categorical_features_indices_)
            else:
                metrics['num_categorical_features_handled'] = 0
                
            # --- Metrics from training history (evals_result) at best_iteration ---
            if hasattr(self, 'training_history_') and self.training_history_ and best_iter is not None:
                for eval_set_name, metric_dict in self.training_history_.items():
                    for metric_name, values in metric_dict.items():
                        if best_iter < len(values):
                            metrics[f'{eval_set_name}_{metric_name}_at_best_iter'] = float(values[best_iter])

            # --- Metrics from internal analyses (if computed) ---
            if hasattr(self, 'prediction_uncertainty_analysis_') and self.prediction_uncertainty_analysis_:
                if 'uncertainty_analysis' in self.prediction_uncertainty_analysis_ and \
                'error' not in self.prediction_uncertainty_analysis_['uncertainty_analysis']:
                    ua = self.prediction_uncertainty_analysis_['uncertainty_analysis']
                    metrics['train_sample_mean_uncertainty_score'] = float(ua.get('mean_uncertainty_score', np.nan))
                    metrics['train_sample_mean_prediction_stability'] = float(ua.get('mean_prediction_stability', np.nan))
                if 'uncertainty_quality' in self.prediction_uncertainty_analysis_ and \
                'error' not in self.prediction_uncertainty_analysis_['uncertainty_quality']:
                    uq = self.prediction_uncertainty_analysis_['uncertainty_quality']
                    metrics['train_sample_uncertainty_reliability'] = uq.get('reliability')


            if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_:
                if 'importance_statistics' in self.feature_importance_analysis_ and \
                'error' not in self.feature_importance_analysis_:
                    stats = self.feature_importance_analysis_['importance_statistics']
                    # Example: Gini of PredictionValuesChange importances
                    pvc_stats = stats.get('PredictionValuesChange', {})
                    if 'concentration' in pvc_stats and isinstance(pvc_stats['concentration'], dict):
                        metrics['feature_importance_gini_pvc'] = float(pvc_stats['concentration'].get('gini_coefficient', np.nan))

            # --- Test-set specific uncertainty metrics (if y_proba is provided as y_pred_std) ---
            if y_proba is not None:
                y_pred_std_test = np.asarray(y_proba)
                if y_pred_std_test.ndim == 1 and len(y_pred_std_test) == len(y_true):
                    metrics['mean_test_prediction_std_dev'] = float(np.mean(y_pred_std_test))
                    metrics['median_test_prediction_std_dev'] = float(np.median(y_pred_std_test))
                    metrics['std_dev_of_test_prediction_std_devs'] = float(np.std(y_pred_std_test))

                    # Calculate empirical coverage for 95% confidence interval
                    z_score_95 = 1.96 
                    lower_bound = y_pred - z_score_95 * y_pred_std_test
                    upper_bound = y_pred + z_score_95 * y_pred_std_test
                    covered = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
                    metrics['empirical_coverage_95_test'] = float(covered / len(y_true))
                else:
                    metrics['y_proba_format_warning'] = "y_proba (for y_pred_std) was not in expected 1D array format matching y_true."
            else:
                metrics['test_prediction_std_dev_status'] = "y_pred_std (via y_proba) not provided for test set."
                
            return metrics


def get_plugin():
    """
    Factory function to create and return the plugin instance.
    This function is called by the plugin system to instantiate the plugin.
    
    Returns:
    --------
    CatBoostRegressorPlugin
        An instance of the CatBoost Regressor Plugin
    """
    return CatBoostRegressorPlugin()






